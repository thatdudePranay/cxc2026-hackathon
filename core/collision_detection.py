"""
Collision Detection Module: Advanced Trajectory Monitoring System
Tracks moving objects over time and predicts potential collisions using velocity,
acceleration, and trajectory analysis. This module provides early warning of dangerous
situations before they become critical.

Process:
1. Track object positions across multiple frames
2. Calculate velocity and acceleration vectors
3. Predict future positions based on trajectories
4. Detect potential collision paths
5. Calculate Time-To-Collision (TTC) for warnings
6. Assess collision risk levels and generate alerts

Key Features:
- Multi-object trajectory tracking with unique IDs
- Velocity and acceleration estimation
- Collision path prediction
- Time-to-collision calculation
- Moving vs stationary object classification
- Historical trajectory analysis
"""

import numpy as np
from collections import deque, defaultdict
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    DANGER_DISTANCE, WARNING_DISTANCE, CAMERA_WIDTH, CAMERA_HEIGHT,
    PRIORITY_CRITICAL, PRIORITY_WARNING, PRIORITY_INFO
)


class TrajectoryTracker:
    """
    Tracks individual object trajectories over time
    """
    
    def __init__(self, track_id, initial_position, initial_timestamp, object_info):
        """
        Initialize trajectory tracker for a single object
        
        Args:
            track_id: Unique identifier for this tracked object
            initial_position: (x, y, distance) tuple
            initial_timestamp: Time of first detection
            object_info: Dict with class_name, class_id, etc.
        """
        self.track_id = track_id
        self.object_info = object_info
        
        # Position history: [(x, y, distance, timestamp), ...]
        self.position_history = deque(maxlen=30)  # Keep last 30 positions (~1 sec at 30fps)
        self.position_history.append((*initial_position, initial_timestamp))
        
        # Velocity and acceleration
        self.velocity = np.array([0.0, 0.0, 0.0])  # (vx, vy, vz) in pixels/sec and m/s
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        # State tracking
        self.is_moving = False
        self.last_update = initial_timestamp
        self.frames_tracked = 1
        self.lost_frames = 0
        
        # Collision warnings
        self.collision_warnings = []
        
    def update(self, position, timestamp):
        """
        Update trajectory with new position
        
        Args:
            position: (x, y, distance) tuple
            timestamp: Current time
        """
        self.position_history.append((*position, timestamp))
        self.last_update = timestamp
        self.frames_tracked += 1
        self.lost_frames = 0
        
        # Calculate velocity and acceleration
        self._update_kinematics()
        
    def _update_kinematics(self):
        """Calculate velocity and acceleration from position history"""
        if len(self.position_history) < 2:
            return
        
        # Get last two positions
        x1, y1, d1, t1 = self.position_history[-2]
        x2, y2, d2, t2 = self.position_history[-1]
        
        dt = t2 - t1
        if dt < 0.001:  # Avoid division by very small numbers
            return
        
        # Calculate velocity
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        vz = (d2 - d1) / dt
        
        new_velocity = np.array([vx, vy, vz])
        
        # Calculate acceleration
        if len(self.position_history) >= 3:
            self.acceleration = (new_velocity - self.velocity) / dt
        
        self.velocity = new_velocity
        
        # Determine if object is moving (threshold: 5 pixels/sec or 0.1 m/s)
        speed_2d = np.sqrt(vx**2 + vy**2)
        speed_z = abs(vz)
        self.is_moving = speed_2d > 5.0 or speed_z > 0.1
        
    def predict_position(self, time_delta):
        """
        Predict future position based on current velocity and acceleration
        
        Args:
            time_delta: Time into future (seconds)
            
        Returns:
            (x, y, distance) predicted position
        """
        if len(self.position_history) == 0:
            return None
        
        x0, y0, d0, _ = self.position_history[-1]
        
        # Use kinematic equation: s = s0 + v*t + 0.5*a*t^2
        pred_x = x0 + self.velocity[0] * time_delta + 0.5 * self.acceleration[0] * (time_delta ** 2)
        pred_y = y0 + self.velocity[1] * time_delta + 0.5 * self.acceleration[1] * (time_delta ** 2)
        pred_d = d0 + self.velocity[2] * time_delta + 0.5 * self.acceleration[2] * (time_delta ** 2)
        
        # Ensure distance stays positive
        pred_d = max(0.1, pred_d)
        
        return (pred_x, pred_y, pred_d)
    
    def get_current_position(self):
        """Get most recent position"""
        if len(self.position_history) == 0:
            return None
        x, y, d, _ = self.position_history[-1]
        return (x, y, d)
    
    def get_trajectory_direction(self):
        """
        Determine general direction of movement
        
        Returns:
            String description: "approaching", "receding", "crossing_left", "crossing_right", "stationary"
        """
        if not self.is_moving:
            return "stationary"
        
        vx, vy, vz = self.velocity
        
        # Check if approaching or receding (based on z-velocity)
        if vz < -0.2:  # Moving closer
            if abs(vx) > 20:  # Also moving laterally
                return "approaching_left" if vx < 0 else "approaching_right"
            return "approaching"
        elif vz > 0.2:  # Moving away
            return "receding"
        else:  # Mostly lateral movement
            if abs(vx) > 10:
                return "crossing_left" if vx < 0 else "crossing_right"
            return "stationary"


class CollisionDetector:
    """
    Main collision detection system using trajectory monitoring
    """
    
    def __init__(self):
        """Initialize collision detection system"""
        print("🎯 Initializing Collision Detection System...")
        
        # Active trackers: {track_id: TrajectoryTracker}
        self.trackers = {}
        
        # ID assignment
        self.next_track_id = 0
        
        # Configuration
        self.max_lost_frames = 15  # Remove tracker after 15 frames without detection
        self.matching_threshold = 100  # Max pixel distance for matching detections
        
        # Collision detection parameters
        self.collision_lookahead_time = 3.0  # Predict 3 seconds ahead
        self.collision_distance_threshold = 2.0  # meters
        self.collision_lateral_threshold = 1.5  # meters (width threshold)
        
        print("✅ Collision Detection System Ready")
        
    def update(self, detections, timestamp=None):
        """
        Update all trackers with new detections
        
        Args:
            detections: List of detection dicts from VisionSystem
            timestamp: Current timestamp (uses time.time() if None)
            
        Returns:
            List of collision warnings
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Match detections to existing trackers
        matched_trackers = set()
        unmatched_detections = []
        
        for det in detections:
            center = det['center']
            distance = det['distance']
            position = (center[0], center[1], distance)
            
            # Find best matching tracker
            best_match = None
            best_distance = self.matching_threshold
            
            for track_id, tracker in self.trackers.items():
                if track_id in matched_trackers:
                    continue
                
                curr_pos = tracker.get_current_position()
                if curr_pos is None:
                    continue
                
                # Calculate distance between detection and predicted position
                match_dist = np.sqrt((curr_pos[0] - position[0])**2 + 
                                   (curr_pos[1] - position[1])**2)
                
                if match_dist < best_distance:
                    best_distance = match_dist
                    best_match = track_id
            
            if best_match is not None:
                # Update existing tracker
                self.trackers[best_match].update(position, timestamp)
                matched_trackers.add(best_match)
            else:
                # New detection - will create new tracker
                unmatched_detections.append((det, position))
        
        # Create new trackers for unmatched detections
        for det, position in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            object_info = {
                'class_name': det['class_name'],
                'class_id': det['class_id'],
                'is_critical': det['is_critical']
            }
            
            tracker = TrajectoryTracker(track_id, position, timestamp, object_info)
            self.trackers[track_id] = tracker
        
        # Update lost frame counter for unmatched trackers
        for track_id in list(self.trackers.keys()):
            if track_id not in matched_trackers:
                self.trackers[track_id].lost_frames += 1
                
                # Remove tracker if lost for too long
                if self.trackers[track_id].lost_frames > self.max_lost_frames:
                    del self.trackers[track_id]
        
        # Detect potential collisions
        collision_warnings = self._detect_collisions(timestamp)
        
        return collision_warnings
    
    def _detect_collisions(self, current_time):
        """
        Analyze trajectories and detect potential collisions
        ONLY warns when objects are approaching the user with collision risk
        
        Returns:
            List of collision warning dicts
        """
        warnings = []
        
        # Get all moving objects
        moving_trackers = [t for t in self.trackers.values() 
                          if t.is_moving and t.object_info['is_critical']]
        
        if len(moving_trackers) == 0:
            return warnings
        
        # Check each moving object
        for tracker in moving_trackers:
            # Get current state
            curr_pos = tracker.get_current_position()
            if curr_pos is None:
                continue
            
            x, y, distance = curr_pos
            direction = tracker.get_trajectory_direction()
            obj_name = tracker.object_info['class_name']
            
            # ONLY process if object is approaching (moving toward camera/user)
            if "approaching" not in direction:
                continue
            
            # Predict future position
            future_pos = tracker.predict_position(self.collision_lookahead_time)
            if future_pos is None:
                continue
            
            future_x, future_y, future_distance = future_pos
            
            # Calculate time to collision (TTC)
            if tracker.velocity[2] < -0.2:  # Must be moving closer (negative z velocity)
                ttc = -distance / tracker.velocity[2]  # Negative because approaching
            else:
                continue  # Not approaching fast enough, skip
            
            # Only warn if collision is imminent and within lookahead time
            if ttc <= 0 or ttc > self.collision_lookahead_time:
                continue
            
            # Check if object will actually collide (not just pass by)
            # Calculate lateral position at predicted collision point
            lateral_offset = abs(x - CAMERA_WIDTH / 2)
            
            # Predict lateral position when object reaches us
            predicted_x = x + tracker.velocity[0] * ttc
            predicted_lateral_offset = abs(predicted_x - CAMERA_WIDTH / 2)
            
            # Only warn if object is on collision course (within frame width)
            if predicted_lateral_offset > CAMERA_WIDTH / 3:
                continue  # Will pass by to the side, not a collision risk
            
            # Determine severity based on time-to-collision and distance
            if ttc < 1.5 and distance < DANGER_DISTANCE:
                priority = PRIORITY_CRITICAL
                message = f"⚠️ COLLISION ALERT! {obj_name.capitalize()} approaching rapidly in {ttc:.1f}s at {distance:.1f}m"
            elif ttc < 2.5 and distance < WARNING_DISTANCE:
                priority = PRIORITY_WARNING
                message = f"Warning: {obj_name.capitalize()} approaching in {ttc:.1f}s, currently {distance:.1f}m away"
            elif ttc < 3.0:
                priority = PRIORITY_INFO
                message = f"{obj_name.capitalize()} approaching, estimated arrival in {ttc:.1f}s"
            else:
                continue  # Too far away to warn
            
            # Determine direction for clarity
            if x < CAMERA_WIDTH / 3:
                dir_str = "from left"
            elif x > 2 * CAMERA_WIDTH / 3:
                dir_str = "from right"
            else:
                dir_str = "straight ahead"
            
            warnings.append({
                'priority': priority,
                'message': f"{message} ({dir_str})",
                'object': tracker.object_info,
                'direction': direction,
                'ttc': ttc,
                'current_distance': distance,
                'predicted_distance': future_distance,
                'track_id': tracker.track_id,
                'velocity': tracker.velocity,
                'position': curr_pos
            })
        
        # Sort warnings by priority and then by time-to-collision (most urgent first)
        warnings.sort(key=lambda x: (x['priority'], x['ttc']))
        
        return warnings
    
    def get_tracked_objects(self):
        """
        Get all currently tracked objects
        
        Returns:
            List of dicts with tracker info
        """
        tracked = []
        
        for track_id, tracker in self.trackers.items():
            pos = tracker.get_current_position()
            if pos is None:
                continue
            
            tracked.append({
                'track_id': track_id,
                'object_name': tracker.object_info['class_name'],
                'position': pos,
                'velocity': tracker.velocity,
                'acceleration': tracker.acceleration,
                'is_moving': tracker.is_moving,
                'direction': tracker.get_trajectory_direction(),
                'frames_tracked': tracker.frames_tracked
            })
        
        return tracked
    
    def get_collision_risk_for_detection(self, detection):
        """
        Get collision risk level for a specific detection
        Used for color-coding bounding boxes
        
        Args:
            detection: Detection dict with 'center' and 'distance'
            
        Returns:
            String: 'critical', 'warning', 'approaching', 'safe'
        """
        center = detection['center']
        distance = detection['distance']
        
        # Find matching tracker
        for tracker in self.trackers.values():
            curr_pos = tracker.get_current_position()
            if curr_pos is None:
                continue
            
            # Check if this detection matches this tracker
            match_dist = np.sqrt((curr_pos[0] - center[0])**2 + 
                               (curr_pos[1] - center[1])**2)
            
            if match_dist < self.matching_threshold:
                # Found matching tracker
                direction = tracker.get_trajectory_direction()
                
                # Check if approaching
                if "approaching" in direction and tracker.velocity[2] < -0.2:
                    # Calculate TTC
                    if tracker.velocity[2] != 0:
                        ttc = -distance / tracker.velocity[2]
                        
                        # Check if on collision course
                        predicted_x = center[0] + tracker.velocity[0] * ttc
                        predicted_lateral_offset = abs(predicted_x - CAMERA_WIDTH / 2)
                        
                        if predicted_lateral_offset < CAMERA_WIDTH / 3:
                            # On collision course
                            if ttc < 1.5 and distance < DANGER_DISTANCE:
                                return 'critical'
                            elif ttc < 2.5 and distance < WARNING_DISTANCE:
                                return 'warning'
                            elif ttc < 3.0:
                                return 'approaching'
                
                break
        
        return 'safe'
    
    def draw_trajectories(self, frame, show_predictions=True):
        """
        Draw trajectory paths and predictions on frame
        
        Args:
            frame: OpenCV image (BGR)
            show_predictions: Whether to show predicted future positions
            
        Returns:
            Annotated frame
        """
        import cv2
        
        for tracker in self.trackers.values():
            if not tracker.is_moving:
                continue
            
            # Draw trajectory history
            positions = [(int(x), int(y)) for x, y, d, t in tracker.position_history]
            
            if len(positions) > 1:
                # Draw path
                for i in range(len(positions) - 1):
                    color = (255, 0, 255)  # Magenta for trajectory
                    cv2.line(frame, positions[i], positions[i+1], color, 2)
                
                # Draw velocity arrow
                curr_pos = positions[-1]
                vx, vy = tracker.velocity[0], tracker.velocity[1]
                
                # Scale velocity for visualization (0.1 second ahead)
                arrow_end = (int(curr_pos[0] + vx * 0.1), 
                           int(curr_pos[1] + vy * 0.1))
                
                cv2.arrowedLine(frame, curr_pos, arrow_end, (0, 255, 255), 2, tipLength=0.3)
                
                # Draw predicted position if enabled
                if show_predictions:
                    future_pos = tracker.predict_position(1.0)  # 1 second ahead
                    if future_pos is not None:
                        pred_x, pred_y = int(future_pos[0]), int(future_pos[1])
                        cv2.circle(frame, (pred_x, pred_y), 8, (0, 255, 255), 2)
                        cv2.line(frame, curr_pos, (pred_x, pred_y), (0, 255, 255), 1, cv2.LINE_AA)
                
                # Draw track ID
                cv2.putText(frame, f"ID:{tracker.track_id}", 
                          (curr_pos[0] + 10, curr_pos[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        return frame
    
    def draw_detections_with_collision_risk(self, frame, detections, vision_system):
        """
        Draw detections with color-coded bounding boxes based on collision risk
        
        Args:
            frame: OpenCV image (BGR)
            detections: List of detection dicts
            vision_system: VisionSystem instance (for FPS counter)
            
        Returns:
            Annotated frame with color-coded boxes
        """
        import cv2
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            distance = det['distance']
            
            # Get collision risk level
            risk_level = self.get_collision_risk_for_detection(det)
            
            # Set color and thickness based on risk
            if risk_level == 'critical':
                color = (0, 0, 255)  # Red - CRITICAL
                thickness = 4
                label_bg_color = (0, 0, 255)
                alert_text = "⚠️ COLLISION RISK!"
            elif risk_level == 'warning':
                color = (0, 165, 255)  # Orange - WARNING
                thickness = 3
                label_bg_color = (0, 165, 255)
                alert_text = "⚠️ APPROACHING"
            elif risk_level == 'approaching':
                color = (0, 255, 255)  # Yellow - APPROACHING
                thickness = 3
                label_bg_color = (0, 255, 255)
                alert_text = "→ Approaching"
            else:
                color = (0, 255, 0)  # Green - SAFE
                thickness = 2
                label_bg_color = (0, 200, 0)
                alert_text = None
            
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Create label with object info
            label = f"{class_name} {confidence:.2f} ({distance:.1f}m)"
            
            # Get label size
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (x1, y1 - label_height - 10), 
                (x1 + label_width, y1),
                label_bg_color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Draw alert text if there's a collision risk
            if alert_text and risk_level in ['critical', 'warning', 'approaching']:
                (alert_width, alert_height), _ = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                # Draw alert above the label
                alert_y = y1 - label_height - 15
                cv2.rectangle(
                    frame,
                    (x1, alert_y - alert_height - 10),
                    (x1 + alert_width, alert_y),
                    color,
                    -1
                )
                
                cv2.putText(
                    frame, alert_text, (x1, alert_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
        
        # Draw FPS counter
        if hasattr(vision_system, 'fps'):
            cv2.putText(
                frame, f"FPS: {vision_system.fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        
        return frame


# =============================================================================
# STANDALONE TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COLLISION DETECTION SYSTEM TEST")
    print("=" * 60)
    print()
    
    # Import vision system for detections
    from vision import VisionSystem
    import cv2
    
    # Initialize systems
    vision = VisionSystem()
    collision_detector = CollisionDetector()
    
    # Open webcam
    print("📸 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        exit()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("✅ Camera opened successfully")
    print()
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 't' to toggle trajectory visualization")
    print("  - Press 's' to save screenshot")
    print()
    
    show_trajectories = True
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Detect objects
            detections = vision.detect_objects(frame)
            
            # Update collision detector
            collision_warnings = collision_detector.update(detections)
            
            # Print collision warnings
            if collision_warnings:
                print(f"\n{'='*70}")
                print(f"🚨 COLLISION WARNING - {len(collision_warnings)} OBJECT(S) APPROACHING")
                print('='*70)
                
                for idx, warning in enumerate(collision_warnings, 1):
                    # Priority labels with colors
                    if warning['priority'] == 0:
                        priority_label = "🔴 CRITICAL DANGER"
                        border = "█" * 70
                    elif warning['priority'] == 1:
                        priority_label = "🟠 WARNING"
                        border = "▓" * 70
                    else:
                        priority_label = "🟡 CAUTION"
                        border = "░" * 70
                    
                    print(f"\n{border}")
                    print(f"[Alert #{idx}] {priority_label}")
                    print(f"{border}")
                    print(f"  Object Type: {warning['object']['class_name'].upper()}")
                    print(f"  Current Distance: {warning['current_distance']:.1f} meters")
                    
                    if warning['ttc'] is not None:
                        print(f"  Time Until Collision: {warning['ttc']:.1f} seconds")
                    
                    print(f"  Direction: {warning['direction']}")
                    print(f"  Speed: {abs(warning['velocity'][2]):.2f} m/s toward you")
                    print(f"\n  ► {warning['message']}")
                    print(f"{border}\n")
            
            # Draw detections with collision risk colors
            annotated_frame = collision_detector.draw_detections_with_collision_risk(
                frame, detections, vision
            )
            
            # Draw trajectories if enabled
            if show_trajectories:
                annotated_frame = collision_detector.draw_trajectories(annotated_frame)
            
            # Display tracking info
            tracked_objects = collision_detector.get_tracked_objects()
            y_offset = 60
            for obj in tracked_objects:
                if obj['is_moving']:
                    info_text = f"ID {obj['track_id']}: {obj['object_name']} - {obj['direction']}"
                    cv2.putText(annotated_frame, info_text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 25
            
            # Show frame
            cv2.imshow("Collision Detection Test (Press 'q' to quit)", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('t'):
                show_trajectories = not show_trajectories
                print(f"Trajectory visualization: {'ON' if show_trajectories else 'OFF'}")
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"collision_test_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")
        print("=" * 60)
