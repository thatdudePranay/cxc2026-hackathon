print("🚀 VISION.PY STARTING (LLM-READY VERSION)...")
print("RUNNING FILE:", __file__)

'''
Enhanced Vision Module - LLM Integration Ready
===============================================

This module is a DATA PROVIDER for the main system.
It returns structured information about:
- All detected objects (for Gemini to process)
- Collision warnings (for immediate audio alerts)
- Wall/obstacle warnings (for navigation)

Architecture:
- YOLO detects all objects → returns full list
- MiDaS provides accurate distances
- Gemini (in main.py) processes object queries
- Audio system speaks alerts

NO PRINTING IN MAIN LOOP - ONLY DATA RETURNS
Camera uses Windows default (DroidCam or built-in)
'''

from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os
import time
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Try to import MiDaS
try:
    import torch
    from torchvision.transforms import Compose, Resize, Normalize
    MIDAS_AVAILABLE = True
    print("✅ MiDaS available - using accurate depth estimation")
except ImportError:
    MIDAS_AVAILABLE = False
    print("⚠️  MiDaS not available - using geometric depth estimation")


class EnhancedVisionSystem:
    """
    Vision system that returns structured data for LLM processing
    
    Usage:
        vision = EnhancedVisionSystem(use_midas=True)
        cap = vision.open_camera()
        
        while True:
            ret, frame = cap.read()
            data = vision.process_frame(frame)
            
            # data contains:
            # - detections: All objects for Gemini
            # - walls: Wall warnings
            # - collisions: Collision predictions
            # - alerts: Danger alerts
    """
    
    def __init__(self, use_midas=True):
        """
        Initialize vision system
        
        Args:
            use_midas: Whether to use MiDaS depth estimation (more accurate)
        """
        print("🔧 Initializing Enhanced Vision System...")
        
        # YOLO model
        self.model = YOLO(YOLO_MODEL)
        print(f"   ✅ YOLO loaded: {YOLO_MODEL}")
        
        # MiDaS depth model
        self.use_midas = use_midas and MIDAS_AVAILABLE
        if self.use_midas:
            self._init_midas()
        else:
            print("   ℹ️  Using geometric depth estimation")
        
        # Object tracking
        self.object_history = {}
        self.max_history = 30
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Collision params
        self.collision_time_horizon = 3.0
        self.min_collision_distance = 0.5
        
        print(f"✅ Vision System ready")
        print(f"   Device: {YOLO_DEVICE}")
        print(f"   Depth: {'MiDaS' if self.use_midas else 'Geometric'}")
    
    def _init_midas(self):
        """Initialize MiDaS depth estimation model"""
        try:
            model_type = "DPT_Small"  # Best balance for laptops
            
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Small":
                self.transform = self.midas_transforms.dpt_transform
            else:
                self.transform = self.midas_transforms.small_transform
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.midas.to(self.device)
            self.midas.eval()
            
            print(f"   ✅ MiDaS loaded: {model_type} on {self.device}")
        except Exception as e:
            print(f"   ⚠️  MiDaS init failed: {e}")
            self.use_midas = False
    
    def open_camera(self):
        """
        Open camera using CAMERA_INDEX from config
        Automatically uses Windows default camera (DroidCam or built-in)
        
        Returns:
            cv2.VideoCapture object or None if failed
        """
        print(f"📸 Opening camera {CAMERA_INDEX}...")
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {CAMERA_INDEX}")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        print("✅ Camera opened successfully")
        return cap
    
    def get_depth_map(self, frame):
        """
        Generate depth map using MiDaS
        
        Returns:
            Normalized depth map (0=far, 1=close) or None
        """
        if not self.use_midas:
            return None
        
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = 1.0 - depth_map  # Invert: closer = higher
            
            return depth_map
            
        except Exception as e:
            return None
        
    def calculate_horizontal_angle(self, center_x, distance):
        """
        Calculate horizontal angle to object from camera center
        
        Uses trigonometry: tan(θ) = horizontal_offset / distance
        
        Args:
            center_x: X coordinate of object center
            distance: Distance to object in meters
            
        Returns:
            Angle in degrees (negative = left, positive = right)
            0° = straight ahead
        """
        frame_width = CAMERA_WIDTH
        frame_center = frame_width / 2
        
        # Horizontal offset from center (in pixels)
        pixel_offset = center_x - frame_center
        
        # Convert pixel offset to meters (rough approximation)
        # Assuming camera FOV ~60 degrees horizontal
        # At distance D, frame width represents: 2 * D * tan(30°)
        horizontal_fov = 60  # degrees (typical webcam)
        meters_per_pixel = (2 * distance * np.tan(np.radians(horizontal_fov / 2))) / frame_width
        horizontal_offset_meters = pixel_offset * meters_per_pixel
        
        # Calculate angle using arctangent
        angle_radians = np.arctan2(horizontal_offset_meters, distance)
        angle_degrees = np.degrees(angle_radians)
        
        return round(angle_degrees, 1)
    
    def estimate_distance_from_depth(self, depth_map, bbox):
        """
        Extract distance from MiDaS depth map for bounding box
        
        Returns:
            (distance_meters, confidence)
        """
        if depth_map is None:
            return None, 0.0
        
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        roi = depth_map[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None, 0.0
        
        median_depth = np.median(roi)
        
        # Calibration: depth 1.0 → 0.5m, depth 0.0 → 20m
        max_distance = 20.0
        min_distance = 0.5
        distance = min_distance + (1.0 - median_depth) * (max_distance - min_distance)
        
        # Confidence based on depth variance
        depth_std = np.std(roi)
        confidence = 1.0 - min(depth_std * 5, 1.0)
        
        return distance, confidence
    
    def estimate_distance_geometric(self, bbox, object_name):
        """
        Fallback geometric distance estimation (pinhole camera model)
        
        Returns:
            (distance_meters, confidence)
        """
        real_size = OBJECT_SIZES.get(object_name, 1.0)
        bbox_height = bbox[3] - bbox[1]
        
        if bbox_height < 1:
            return 999.0, 0.1
        
        distance = (real_size * FOCAL_LENGTH) / bbox_height
        bbox_area = bbox_height * (bbox[2] - bbox[0])
        confidence = min(bbox_area / 10000, 1.0)
        
        return distance, confidence
    
    def get_precise_direction(self, center_x):
        """
        Get precise 7-point directional description
        
        Returns: far-left, left, center-left, ahead, center-right, right, far-right
        """
        frame_width = CAMERA_WIDTH
        position = center_x / frame_width
        
        if position < 0.15:
            return "far-left"
        elif position < 0.33:
            return "left"
        elif position < 0.45:
            return "center-left"
        elif position < 0.55:
            return "ahead"
        elif position < 0.67:
            return "center-right"
        elif position < 0.85:
            return "right"
        else:
            return "far-right"
    
    def get_height_description(self, center_y):
        """
        Get vertical position description
        
        Returns: "high up", "at eye level", or "low down"
        """
        frame_height = CAMERA_HEIGHT
        position = center_y / frame_height
        
        if position < 0.33:
            return "high up"
        elif position < 0.67:
            return "at eye level"
        else:
            return "low down"
    
    def detect_objects(self, frame, depth_map=None):
        """
        Detect ALL objects in frame using YOLO + MiDaS
        
        Returns:
            List of ALL detected objects with metadata
            (Gemini will process queries about specific objects)
        """
        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE,
            device=YOLO_DEVICE,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                class_name = results[0].names[class_id]
                
                # Distance estimation (MiDaS preferred, geometric fallback)
                if depth_map is not None:
                    distance, dist_confidence = self.estimate_distance_from_depth(
                        depth_map, bbox
                    )
                    if distance is None:
                        distance, dist_confidence = self.estimate_distance_geometric(
                            bbox, class_name
                        )
                else:
                    distance, dist_confidence = self.estimate_distance_geometric(
                        bbox, class_name
                    )
                
                center = self.get_bbox_center(bbox)
                
                # Calculate horizontal angle
                angle = self.calculate_horizontal_angle(center[0], distance)

                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': round(confidence, 2),
                    'bbox': bbox.tolist(),
                    'distance': round(distance, 2),
                    'distance_confidence': round(dist_confidence, 2),
                    'center': center,
                    'direction': self.get_precise_direction(center[0]),
                    'height': self.get_height_description(center[1]),
                    'angle': angle,  # NEW: Horizontal angle in degrees
                    'is_critical': class_id in CRITICAL_OBJECTS,
                    'timestamp': time.time()
                }
                
                detections.append(detection)
        
        self.update_fps()
        return detections
    
    def detect_walls_and_obstacles(self, depth_map, frame):
        """
        Detect walls and obstacles using BOTH depth map AND edge detection
        
        Two-stage approach:
        1. MiDaS depth (works on textured surfaces)
        2. Edge detection (catches white/featureless walls)
        
        Returns:
            List of wall/obstacle warnings
        """
        warnings = []
        h, w = frame.shape[:2]
        
        # ============================================================
        # STAGE 1: Depth-based detection (existing logic)
        # ============================================================
        if depth_map is not None:
            # 5 sections for precise wall location
            section_width = w // 5
            sections = {
                'far-left': depth_map[:, :section_width],
                'left': depth_map[:, section_width:2*section_width],
                'ahead': depth_map[:, 2*section_width:3*section_width],
                'right': depth_map[:, 3*section_width:4*section_width],
                'far-right': depth_map[:, 4*section_width:]
            }
            
            for direction, section in sections.items():
                close_pixels = np.sum(section > 0.7)
                total_pixels = section.size
                close_percentage = close_pixels / total_pixels
                
                if close_percentage > 0.3:
                    median_depth = np.median(section[section > 0.5])
                    
                    max_distance = 20.0
                    min_distance = 0.5
                    wall_distance = min_distance + (1.0 - median_depth) * (max_distance - min_distance)
                    
                    if wall_distance < 2.5:
                        warnings.append({
                            'type': 'wall',
                            'severity': 'WARNING' if wall_distance > 1.5 else 'CRITICAL',
                            'direction': direction,
                            'distance': round(wall_distance, 2),
                            'coverage': round(close_percentage, 2),
                            'detection_method': 'depth',
                            'message': f"Wall {direction}, {wall_distance:.1f}m"
                        })
            
            # Check path ahead (bottom-center)
            bottom_center = depth_map[int(h*0.6):, int(w*0.3):int(w*0.7)]
            if np.median(bottom_center) > 0.75:
                distance = 0.5 + (1.0 - np.median(bottom_center)) * 19.5
                if distance < 1.5:
                    warnings.append({
                        'type': 'path_blocked',
                        'severity': 'CRITICAL',
                        'direction': 'ahead',
                        'distance': round(distance, 2),
                        'detection_method': 'depth',
                        'message': f"Path blocked, obstacle {distance:.1f}m ahead"
                    })
        
        # ============================================================
        # STAGE 2: Edge-based detection (for white/featureless walls)
        # ============================================================
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Focus on lower half of frame (where walls/obstacles matter for walking)
        lower_half = edges[int(h*0.4):, :]
        
        # Divide into 5 sections (same as depth)
        section_width = w // 5
        edge_sections = {
            'far-left': lower_half[:, :section_width],
            'left': lower_half[:, section_width:2*section_width],
            'ahead': lower_half[:, 2*section_width:3*section_width],
            'right': lower_half[:, 3*section_width:4*section_width],
            'far-right': lower_half[:, 4*section_width:]
        }
        
        for direction, section in edge_sections.items():
            # Count vertical edges (walls typically have strong vertical edges)
            vertical_edges = cv2.Sobel(section, cv2.CV_64F, 1, 0, ksize=3)
            vertical_edge_strength = np.sum(np.abs(vertical_edges))
            
            # Normalize by section size
            edge_density = vertical_edge_strength / section.size
            
            # If high edge density in lower portion of frame = likely wall/obstacle
            if edge_density > 15:  # Threshold (tune if needed)
                # Check if we already detected this wall via depth
                already_detected = any(
                    w['direction'] == direction and w['detection_method'] == 'depth' 
                    for w in warnings
                )
                
                if not already_detected:
                    # Estimate distance based on edge position
                    # Lower in frame = closer
                    edge_rows = np.where(section > 0)[0]
                    if len(edge_rows) > 0:
                        avg_row = np.mean(edge_rows)
                        # Map row position to distance (rough estimate)
                        # Bottom of lower_half (row h*0.6) = ~0.5m
                        # Top of lower_half = ~3m
                        distance = 3.0 - (avg_row / section.shape[0]) * 2.5
                        distance = max(0.5, min(3.0, distance))  # Clamp
                        
                        warnings.append({
                            'type': 'wall',
                            'severity': 'WARNING' if distance > 1.5 else 'CRITICAL',
                            'direction': direction,
                            'distance': round(distance, 2),
                            'coverage': round(edge_density / 50, 2),  # Normalize
                            'detection_method': 'edge',
                            'message': f"Wall {direction}, ~{distance:.1f}m (edge detected)"
                        })
        
        # Remove duplicates (prefer depth-based over edge-based)
        seen_directions = {}
        filtered_warnings = []
        for warning in warnings:
            direction = warning['direction']
            if direction not in seen_directions:
                filtered_warnings.append(warning)
                seen_directions[direction] = warning
            elif warning['detection_method'] == 'depth' and seen_directions[direction]['detection_method'] == 'edge':
                # Replace edge detection with depth detection (more accurate)
                filtered_warnings = [w for w in filtered_warnings if w['direction'] != direction]
                filtered_warnings.append(warning)
                seen_directions[direction] = warning
        
        return filtered_warnings
    
    def track_objects(self, detections):
        """
        Track objects across frames for movement detection
        
        Returns:
            Dict of {obj_id: {'velocity', 'acceleration', 'approaching'}}
        """
        current_time = time.time()
        tracking_data = {}
        
        for i, det in enumerate(detections):
            obj_id = f"{det['class_name']}_{i}"
            
            if obj_id not in self.object_history:
                self.object_history[obj_id] = []
            
            self.object_history[obj_id].append({
                'time': current_time,
                'distance': det['distance'],
                'bbox': det['bbox'],
                'center': det['center']
            })
            
            self.object_history[obj_id] = self.object_history[obj_id][-self.max_history:]
            
            if len(self.object_history[obj_id]) >= 5:
                history = self.object_history[obj_id]
                recent = history[-5:]
                time_diff = recent[-1]['time'] - recent[0]['time']
                dist_diff = recent[0]['distance'] - recent[-1]['distance']
                
                velocity = dist_diff / time_diff if time_diff > 0 else 0
                
                if len(history) >= 10:
                    older = history[-10:-5]
                    old_time_diff = older[-1]['time'] - older[0]['time']
                    old_dist_diff = older[0]['distance'] - older[-1]['distance']
                    old_velocity = old_dist_diff / old_time_diff if old_time_diff > 0 else 0
                    acceleration = (velocity - old_velocity) / time_diff if time_diff > 0 else 0
                else:
                    acceleration = 0
                
                tracking_data[obj_id] = {
                    'velocity': round(velocity, 2),
                    'acceleration': round(acceleration, 2),
                    'approaching': velocity > 0.3,
                    'distance': det['distance']
                }
        
        return tracking_data
    
    def predict_collisions(self, detections, tracking_data):
        """
        Predict potential collisions
        
        Returns:
            List of collision warnings sorted by urgency
        """
        collisions = []
        
        for i, det in enumerate(detections):
            if not det['is_critical']:
                continue
            
            obj_id = f"{det['class_name']}_{i}"
            
            if obj_id not in tracking_data:
                continue
            
            track = tracking_data[obj_id]
            
            if not track['approaching']:
                continue
            
            distance = det['distance']
            velocity = track['velocity']
            acceleration = track['acceleration']
            
            if velocity > 0.1:
                time_to_collision = distance / velocity
                
                if acceleration > 0:
                    time_to_collision *= 0.8
                
                if time_to_collision < self.collision_time_horizon:
                    severity = 'CRITICAL' if time_to_collision < 1.5 else 'WARNING'
                    
                    collisions.append({
                        'type': 'collision',
                        'severity': severity,
                        'object': det['class_name'],
                        'distance': round(distance, 2),
                        'velocity': round(velocity, 2),
                        'time_to_collision': round(time_to_collision, 2),
                        'direction': det['direction'],
                        'message': f"{det['class_name']} {det['direction']}, {time_to_collision:.1f}s to impact!"
                    })
        
        collisions.sort(key=lambda x: x['time_to_collision'])
        return collisions
    
    def get_bbox_center(self, bbox):
        """Get bounding box center point"""
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        return (int(x_center), int(y_center))
    
    def assess_danger_level(self, detections, tracking_data=None):
        """
        Assess danger level for critical objects
        
        Returns:
            List of alerts with severity, message, and metadata
        """
        alerts = []
        
        for i, det in enumerate(detections):
            if not det['is_critical']:
                continue
            
            distance = det['distance']
            obj_name = det['class_name']
            direction = det['direction']
            
            is_approaching = False
            velocity = 0
            if tracking_data:
                obj_id = f"{obj_name}_{i}"
                if obj_id in tracking_data:
                    is_approaching = tracking_data[obj_id]['approaching']
                    velocity = tracking_data[obj_id]['velocity']
            
            # CRITICAL: Immediate danger
            if distance < DANGER_DISTANCE:
                if obj_name in ['car', 'truck', 'bus', 'motorcycle']:
                    urgency = "STOP! " if is_approaching else "DANGER! "
                    message = f"{urgency}{obj_name.capitalize()} {direction}, {distance:.1f}m"
                    if is_approaching:
                        message += f" approaching at {velocity:.1f}m/s"
                    severity = 'CRITICAL'
                elif obj_name == 'person':
                    # Only CRITICAL if person is approaching, otherwise just WARNING
                    if is_approaching:
                        message = f"Person approaching {direction}, {distance:.1f}m at {velocity:.1f}m/s"
                        severity = 'CRITICAL'
                    else:
                        message = f"Person nearby {direction}, {distance:.1f}m"
                        severity = 'WARNING'  # Still person, not critical if standing
                else:
                    message = f"{obj_name.capitalize()} {direction}, {distance:.1f}m"
                    severity = 'CRITICAL' if is_approaching else 'WARNING'
                
                alerts.append({
                    'type': 'danger',
                    'severity': severity,
                    'object': obj_name,
                    'distance': round(distance, 2),
                    'direction': direction,
                    'is_approaching': is_approaching,
                    'velocity': round(velocity, 2) if velocity else 0,
                    'message': message
                })
            
            # WARNING: Potential danger
            elif distance < WARNING_DISTANCE:
                if is_approaching and velocity > 0.5:
                    message = f"{obj_name.capitalize()} approaching {direction}, {distance:.1f}m, {velocity:.1f}m/s"
                    severity = 'WARNING'
                else:
                    message = f"{obj_name.capitalize()} {direction}, {distance:.1f}m"
                    severity = 'INFO'
                
                alerts.append({
                    'type': 'warning',
                    'severity': severity,
                    'object': obj_name,
                    'distance': round(distance, 2),
                    'direction': direction,
                    'is_approaching': is_approaching,
                    'velocity': round(velocity, 2) if velocity else 0,
                    'message': message
                })
        
        # Sort by severity, then distance
        alerts.sort(key=lambda x: (
            0 if x['severity'] == 'CRITICAL' else 1 if x['severity'] == 'WARNING' else 2,
            -x['distance']
        ))
        
        return alerts
    
    def process_frame(self, frame):
        '''
        MAIN PROCESSING PIPELINE
        
        Call this method for each frame to get all vision data
        
        Returns:
            {
                'detections': [...],   # All objects (for Gemini)
                'walls': [...],        # Wall warnings
                'collisions': [...],   # Collision predictions
                'alerts': [...],       # Danger alerts
                'tracking': {...},     # Movement data
                'fps': float,          # Current FPS
                'timestamp': float     # Unix timestamp
            }
        ''' 
        # Get depth map
        depth_map = self.get_depth_map(frame) if self.use_midas else None
        
        # Detect all objects
        detections = self.detect_objects(frame, depth_map)
        
        # Detect walls
        walls = self.detect_walls_and_obstacles(depth_map, frame)
        
        # Track movement
        tracking_data = self.track_objects(detections)
        
        # Predict collisions
        collisions = self.predict_collisions(detections, tracking_data)
        
        # Assess dangers
        alerts = self.assess_danger_level(detections, tracking_data)
        
        return {
            'detections': detections,
            'walls': walls,
            'collisions': collisions,
            'alerts': alerts,
            'tracking': tracking_data,
            'fps': round(self.fps, 1),
            'timestamp': time.time()
        }
        
    def draw_detections(self, frame, detections, depth_map=None, tracking_data=None):
        """
        Draw visualizations on frame (for testing/debugging)
        
        Optional - only needed for visual debugging
        """
        if depth_map is not None and SHOW_DETECTIONS:
            depth_colored = cv2.applyColorMap(
                (depth_map * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 0.7, depth_colored, 0.3, 0)
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            distance = det['distance']
            direction = det['direction']
            
            x1, y1, x2, y2 = [int(x) for x in bbox]
            
            is_approaching = False
            if tracking_data:
                obj_id = f"{class_name}_{i}"
                if obj_id in tracking_data:
                    is_approaching = tracking_data[obj_id]['approaching']
            
            if distance < DANGER_DISTANCE:
                color = (0, 0, 255) if is_approaching else (0, 100, 255)
            elif distance < WARNING_DISTANCE:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            
            thickness = 3 if is_approaching else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{class_name} ({distance:.1f}m) {direction}"
            
            if tracking_data and f"{class_name}_{i}" in tracking_data:
                vel = tracking_data[f"{class_name}_{i}"]['velocity']
                if abs(vel) > 0.3:
                    label += f" {vel:.1f}m/s"
            
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        mode = "MiDaS" if self.use_midas else "Geometric"
        cv2.putText(
            frame, f"FPS: {self.fps:.1f} | Depth: {mode}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

# =============================================================================
# TEST CODE - Prints AND Returns Data
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VISION SYSTEM TEST - LLM-READY VERSION")
    print("=" * 70)
    print()
    print("NOTE: Set DroidCam as default camera in Windows Settings")
    print("      Or disable built-in webcam to use DroidCam automatically")
    print()
    
    vision = EnhancedVisionSystem(use_midas=True)
    cap = vision.open_camera()
    
    if not cap:
        exit()
    
    print("\nControls:")
    print("  q - Quit")
    print("  s - Screenshot")
    print("  p - Toggle print mode (ON by default)")
    print("  d - Toggle depth overlay")
    print()
    print("Starting with PRINT MODE ON - you'll see all alerts!")
    print()
    
    screenshot_count = 0
    print_mode = True  # Print by default
    show_depth = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # MAIN PROCESSING - Returns all data
            data = vision.process_frame(frame)
            
            # PRINT MODE - Show alerts in console (for testing)
            if print_mode:
                # Only print if there's something to show
                has_alerts = (data['walls'] or data['collisions'] or 
                             data['alerts'] or len(data['detections']) > 0)
                
                if has_alerts:
                    print("\n" + "─" * 70)
                    print(f"⚡ FPS: {data['fps']}")
                    
                    # Print detections summary
                    if data['detections']:
                        print(f"\n  DETECTED ({len(data['detections'])} objects):")
                        # Group by type and show counts
                        object_counts = {}
                        for det in data['detections']:
                            obj_type = det['class_name']
                            if obj_type not in object_counts:
                                object_counts[obj_type] = []
                            object_counts[obj_type].append(det)
                        
                        for obj_type, objs in sorted(object_counts.items()):
                            if len(objs) == 1:
                                obj = objs[0]
                                angle_str = f"{obj['angle']:+.1f}°"  # +5.2° or -12.3°
                                print(f"   • {obj_type}: {obj['direction']}, {obj['distance']}m, {angle_str}, {obj['height']}")
                            else:
                                distances = [f"{o['distance']:.1f}m" for o in objs]
                                print(f"   • {obj_type} ({len(objs)}): {', '.join(distances)}")
                    
                    # Print CRITICAL items first
                    if data['collisions']:
                        print(f"\n⚠️  COLLISION WARNINGS ({len(data['collisions'])}):")
                        for col in data['collisions']:
                            severity_icon = "🔴" if col['severity'] == 'CRITICAL' else "🟠"
                            print(f"   {severity_icon} [{col['severity']}] {col['message']}")
                    
                    if data['walls']:
                        print(f"\n🧱 WALLS/OBSTACLES ({len(data['walls'])}):")
                        for wall in data['walls']:
                            severity_icon = "🔴" if wall['severity'] == 'CRITICAL' else "🟠"
                            print(f"   {severity_icon} [{wall['severity']}] {wall['message']}")
                    
                    if data['alerts']:
                        print(f"\n🚨 DANGER ALERTS ({len(data['alerts'])}):")
                        for alert in data['alerts']:
                            if alert['severity'] == 'CRITICAL':
                                print(f"   🔴 [CRITICAL] {alert['message']}")
                            elif alert['severity'] == 'WARNING':
                                print(f"   🟠 [WARNING] {alert['message']}")
                            else:
                                print(f"   🟢 [INFO] {alert['message']}")
            
            # Optional: Draw for visual feedback
            depth_map = vision.get_depth_map(frame) if vision.use_midas and show_depth else None
            annotated = vision.draw_detections(
                frame, 
                data['detections'], 
                depth_map, 
                data['tracking']
            )
            
            # Show print mode status
            status_text = f"Print: {'ON' if print_mode else 'OFF'} | Depth: {'ON' if show_depth else 'OFF'}"
            cv2.putText(
                annotated, status_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            
            cv2.imshow("Vision System (q=quit, p=toggle print, d=depth, s=screenshot)", annotated)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('p'):
                print_mode = not print_mode
                status = "ON ✅" if print_mode else "OFF ❌"
                print(f"\n🔄 Print mode: {status}")
            elif key == ord('d'):
                show_depth = not show_depth
                status = "ON" if show_depth else "OFF"
                print(f"\n🔄 Depth overlay: {status}")
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"\n📸 Saved {filename}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera released")
        print("=" * 70)

