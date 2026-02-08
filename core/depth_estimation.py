"""
Depth Estimation Module: MiDaS-based Environmental Awareness
Uses MiDaS depth estimation to detect static obstacles like walls, furniture, and 
provide general spatial awareness. This complements YOLO's object detection by 
catching things that aren't specific objects.

Process:
1. Generate depth map using MiDaS
2. Identify close obstacles in different zones (left, center, right)
3. Detect walls and large obstacles
4. Provide proximity warnings for navigation
5. Generate spatial awareness alerts

Key Features:
- Real-time depth map generation
- Zone-based obstacle detection (left/center/right)
- Wall and large obstacle detection
- Distance estimation for static obstacles
- Complements YOLO for complete environmental awareness
"""

import cv2
import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    DANGER_DISTANCE, WARNING_DISTANCE, CAMERA_WIDTH, CAMERA_HEIGHT,
    PRIORITY_CRITICAL, PRIORITY_WARNING, PRIORITY_INFO
)


class DepthEstimator:
    """
    MiDaS-based depth estimation for static obstacle detection
    """
    
    def __init__(self, model_type="DPT_Large"):
        """
        Initialize MiDaS depth estimation
        
        Args:
            model_type: "DPT_Large" (best quality), "DPT_Hybrid", or "MiDaS_small" (fastest)
        """
        print("🌊 Initializing MiDaS Depth Estimation System...")
        
        # Load MiDaS model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {self.device}")
        
        try:
            # Load model from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            self.model_type = model_type
            print(f"✅ MiDaS model loaded: {model_type}")
            
        except Exception as e:
            print(f"❌ Error loading MiDaS: {e}")
            print("   Make sure you have internet connection for first-time model download")
            raise
        
        # Zone configuration (divide frame into 3 zones)
        self.zone_left = (0, CAMERA_WIDTH // 3)
        self.zone_center = (CAMERA_WIDTH // 3, 2 * CAMERA_WIDTH // 3)
        self.zone_right = (2 * CAMERA_WIDTH // 3, CAMERA_WIDTH)
        
        # Detection thresholds (normalized depth values, 0=far, 1=close)
        self.critical_depth = 0.7  # Very close obstacle
        self.warning_depth = 0.5   # Moderately close obstacle
        self.info_depth = 0.3      # Far obstacle but worth noting
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ Depth Estimation System Ready")
    
    def estimate_depth(self, frame):
        """
        Generate depth map from input frame
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            depth_map: Normalized depth map (0=far, 1=close)
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-1 range (invert so 1 is close, 0 is far)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = 1 - depth_map  # Invert: now 1 = close, 0 = far
        
        # Update FPS
        self.update_fps()
        
        return depth_map
    
    def analyze_obstacles(self, depth_map):
        """
        Analyze depth map to detect obstacles in different zones
        
        Args:
            depth_map: Normalized depth map from estimate_depth()
            
        Returns:
            List of obstacle warnings sorted by priority
        """
        warnings = []
        
        # Analyze each zone
        zones = {
            'left': (self.zone_left, "LEFT"),
            'center': (self.zone_center, "AHEAD"),
            'right': (self.zone_right, "RIGHT")
        }
        
        for zone_name, ((x_start, x_end), direction) in zones.items():
            # Extract zone from depth map
            zone_depth = depth_map[:, x_start:x_end]
            
            # Calculate statistics for this zone
            mean_depth = np.mean(zone_depth)
            max_depth = np.max(zone_depth)
            
            # Count pixels in danger zones
            critical_pixels = np.sum(zone_depth > self.critical_depth)
            warning_pixels = np.sum(zone_depth > self.warning_depth)
            
            # Calculate percentage of zone that's close
            total_pixels = zone_depth.size
            critical_percent = (critical_pixels / total_pixels) * 100
            warning_percent = (warning_pixels / total_pixels) * 100
            
            # Generate warnings based on analysis
            if critical_percent > 20:  # More than 20% of zone is very close
                # Estimate approximate distance (rough conversion)
                estimated_distance = self._depth_to_distance(mean_depth)
                
                warnings.append({
                    'priority': PRIORITY_CRITICAL,
                    'message': f"⚠️ OBSTACLE {direction}! Very close (~{estimated_distance:.1f}m)",
                    'zone': zone_name,
                    'direction': direction,
                    'depth_percent': critical_percent,
                    'estimated_distance': estimated_distance,
                    'type': 'static_obstacle'
                })
                
            elif warning_percent > 30:  # More than 30% of zone has obstacles
                estimated_distance = self._depth_to_distance(mean_depth)
                
                warnings.append({
                    'priority': PRIORITY_WARNING,
                    'message': f"Obstacle {direction.lower()}, approximately {estimated_distance:.1f}m",
                    'zone': zone_name,
                    'direction': direction,
                    'depth_percent': warning_percent,
                    'estimated_distance': estimated_distance,
                    'type': 'static_obstacle'
                })
                
            elif max_depth > self.warning_depth:
                estimated_distance = self._depth_to_distance(max_depth)
                
                warnings.append({
                    'priority': PRIORITY_INFO,
                    'message': f"Object detected {direction.lower()}",
                    'zone': zone_name,
                    'direction': direction,
                    'depth_percent': warning_percent,
                    'estimated_distance': estimated_distance,
                    'type': 'static_obstacle'
                })
        
        # Sort by priority
        warnings.sort(key=lambda x: (x['priority'], -x['depth_percent']))
        
        return warnings
    
    def _depth_to_distance(self, normalized_depth):
        """
        Convert normalized depth value to approximate distance in meters
        This is a rough estimate - you may need to calibrate based on your setup
        
        Args:
            normalized_depth: Depth value from 0 (far) to 1 (close)
            
        Returns:
            Estimated distance in meters
        """
        # Rough exponential mapping: close objects get small distances
        # This is approximate and should be calibrated for accuracy
        if normalized_depth > 0.8:
            return 0.5 + (1.0 - normalized_depth) * 5  # 0.5-1.5m for very close
        elif normalized_depth > 0.5:
            return 1.5 + (0.8 - normalized_depth) * 10  # 1.5-4.5m for medium
        else:
            return 4.5 + (0.5 - normalized_depth) * 20  # 4.5+ for far
    
    def draw_depth_overlay(self, frame, depth_map, show_zones=True, alpha=0.4):
        """
        Draw depth map overlay on frame with color coding
        
        Args:
            frame: Original BGR frame
            depth_map: Depth map from estimate_depth()
            show_zones: Whether to show zone boundaries
            alpha: Transparency of overlay (0=transparent, 1=opaque)
            
        Returns:
            Frame with depth overlay
        """
        # Create colored depth map
        depth_colored = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 1 - alpha, depth_colored, alpha, 0)
        
        # Draw zone boundaries
        if show_zones:
            # Left zone boundary
            cv2.line(overlay, 
                    (self.zone_center[0], 0), 
                    (self.zone_center[0], CAMERA_HEIGHT),
                    (255, 255, 255), 2)
            
            # Right zone boundary
            cv2.line(overlay, 
                    (self.zone_center[1], 0), 
                    (self.zone_center[1], CAMERA_HEIGHT),
                    (255, 255, 255), 2)
            
            # Zone labels
            cv2.putText(overlay, "LEFT", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, "CENTER", (CAMERA_WIDTH // 2 - 40, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, "RIGHT", (CAMERA_WIDTH - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(overlay, f"Depth FPS: {self.fps:.1f}", (10, CAMERA_HEIGHT - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw color scale legend
        legend_height = 150
        legend_width = 30
        legend_x = CAMERA_WIDTH - legend_width - 20
        legend_y = CAMERA_HEIGHT // 2 - legend_height // 2
        
        # Create gradient
        gradient = np.linspace(0, 255, legend_height).reshape(-1, 1)
        gradient = np.repeat(gradient, legend_width, axis=1).astype(np.uint8)
        gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        
        overlay[legend_y:legend_y + legend_height, legend_x:legend_x + legend_width] = gradient_colored
        
        # Labels
        cv2.putText(overlay, "CLOSE", (legend_x - 60, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, "FAR", (legend_x - 50, legend_y + legend_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return overlay
    
    def draw_warnings(self, frame, warnings):
        """
        Draw warning indicators on frame
        
        Args:
            frame: OpenCV BGR frame
            warnings: List of warnings from analyze_obstacles()
            
        Returns:
            Frame with warning indicators
        """
        # Display most critical warning as banner
        if warnings:
            top_warning = warnings[0]
            
            if top_warning['priority'] == PRIORITY_CRITICAL:
                color = (0, 0, 255)  # Red
                banner_text = "⚠️ OBSTACLE AHEAD"
            elif top_warning['priority'] == PRIORITY_WARNING:
                color = (0, 165, 255)  # Orange
                banner_text = "⚠ WARNING"
            else:
                color = (0, 255, 255)  # Yellow
                banner_text = "INFO"
            
            # Draw banner
            cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 50), color, -1)
            cv2.putText(frame, banner_text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Draw direction arrows for critical warnings
            if top_warning['priority'] == PRIORITY_CRITICAL:
                direction = top_warning['direction']
                
                if direction == "LEFT":
                    arrow_pos = (100, CAMERA_HEIGHT // 2)
                    cv2.arrowedLine(frame, (arrow_pos[0] + 100, arrow_pos[1]),
                                  arrow_pos, (0, 0, 255), 8, tipLength=0.3)
                    cv2.putText(frame, "◄◄◄ LEFT", (50, CAMERA_HEIGHT // 2 - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                elif direction == "RIGHT":
                    arrow_pos = (CAMERA_WIDTH - 100, CAMERA_HEIGHT // 2)
                    cv2.arrowedLine(frame, (arrow_pos[0] - 100, arrow_pos[1]),
                                  arrow_pos, (0, 0, 255), 8, tipLength=0.3)
                    cv2.putText(frame, "RIGHT ►►►", (CAMERA_WIDTH - 250, CAMERA_HEIGHT // 2 - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                else:  # AHEAD
                    arrow_pos = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)
                    cv2.arrowedLine(frame, (arrow_pos[0], arrow_pos[1] - 100),
                                  arrow_pos, (0, 0, 255), 8, tipLength=0.3)
                    cv2.putText(frame, "▼ AHEAD ▼", (CAMERA_WIDTH // 2 - 100, CAMERA_HEIGHT // 2 - 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
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
# STANDALONE TEST CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MIDAS DEPTH ESTIMATION SYSTEM TEST")
    print("=" * 60)
    print()
    
    # Initialize depth estimator
    # Use "MiDaS_small" for faster performance, "DPT_Large" for best quality
    depth_estimator = DepthEstimator(model_type="MiDaS_small")
    
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
    print("  - Press 'd' to toggle depth overlay")
    print("  - Press 'z' to toggle zone boundaries")
    print("  - Press 's' to save screenshot")
    print()
    
    show_depth = True
    show_zones = True
    screenshot_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Generate depth map
            depth_map = depth_estimator.estimate_depth(frame)
            
            # Analyze for obstacles
            warnings = depth_estimator.analyze_obstacles(depth_map)
            
            # Print warnings
            if warnings:
                print(f"\n{'='*70}")
                print(f"⚠️ OBSTACLE DETECTED - {len(warnings)} WARNING(S)")
                print('='*70)
                
                for idx, warning in enumerate(warnings, 1):
                    if warning['priority'] == 0:
                        priority_label = "🔴 CRITICAL"
                    elif warning['priority'] == 1:
                        priority_label = "🟠 WARNING"
                    else:
                        priority_label = "🟡 INFO"
                    
                    print(f"[{idx}] {priority_label}: {warning['message']}")
                    print(f"    Zone: {warning['zone']} | Distance: ~{warning['estimated_distance']:.1f}m")
            
            # Create visualization
            if show_depth:
                display_frame = depth_estimator.draw_depth_overlay(
                    frame, depth_map, show_zones=show_zones
                )
            else:
                display_frame = frame.copy()
            
            # Draw warnings
            display_frame = depth_estimator.draw_warnings(display_frame, warnings)
            
            # Show frame
            cv2.imshow("MiDaS Depth Estimation Test (Press 'q' to quit)", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('d'):
                show_depth = not show_depth
                print(f"Depth overlay: {'ON' if show_depth else 'OFF'}")
            elif key == ord('z'):
                show_zones = not show_zones
                print(f"Zone boundaries: {'ON' if show_zones else 'OFF'}")
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"depth_test_{screenshot_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")
        print("=" * 60)
