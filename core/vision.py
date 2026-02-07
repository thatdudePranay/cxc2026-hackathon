print("🚀 VISION.PY STARTING...")
print("RUNNING FILE:", __file__)

'''
Vision Module: Our System for Object Detection and Depth Estimiation
Powered by YOLOv8 for real time detection and geometric depth estimation for accurate distance measurement. This module is the eyes of our system, providing critical information about the environment to ensure safety and awareness.

Process:

1. YOLO will detect objects in camara frame (aim for 30 fps)
2. We will estimate the distance using bbox size and known object sizes (from config)
3. Assessment of danger level based on distance and object type
4. Returns structured data for other modules to use (audio, logging, etc)

'''

from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

import time

class VisionSystem:
    """
    Here we will handle all computer vision tasks
    - object detection with YOLO
    - Distance estimation 
    - Danger assessment
    """

    def __init__(self):
        print("Vision System")

        self.model = YOLO(YOLO_MODEL)

        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        print(f" Loaded YOLO model: {YOLO_MODEL} on device: {YOLO_DEVICE}")
        print(f" Condfidence threshold: {YOLO_CONFIDENCE}  ")

    def detect_objects(self, frame):
        """
        Run YOLO detection on the frame

        Aruments:
        frame: open cv input image (BGR format)

        Returns:
        List of detected objects with format:
        [
            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
                'distance': float (estimated meters),
                'center': (x, y),
                'is_critical': bool
            },
            ...]
        """

        ##The object detection using YOLO
        results = self.model(
            frame, 
            conf = YOLO_CONFIDENCE,
            device = YOLO_DEVICE,
            verbose = False

        )

        detections = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()


                class_name = results[0].names[class_id]

                # Distance estimation
                distance = self.estimate_distance(bbox, class_name)

                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'distance': distance,
                    'center': self.get_bbox_center(bbox),
                    'is_critical': class_id in CRITICAL_OBJECTS
                }

                detections.append(detection)

        # so then we have to update the fps counter after each detection
        self.update_fps()

        return detections
    
    def estimate_distance(self, bbox, object_name):
        """
        Estimate distance to object using pinhole camera model
        
        SIMPLE GEOMETRY:
        distance = (real_height * focal_length) / pixel_height
        
        Example:
        - Car is 1.5m tall in real life
        - Appears as 300 pixels in image
        - Focal length = 800 pixels
        - Distance = (1.5 × 800) / 300 = 4 meters
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            object_name: Name of detected object
            
        Returns:
            Estimated distance in meters
        """

        real_size = OBJECT_SIZES.get(object_name, 1.0)

        # Calculate bbox height in pixels
        bbox_height = bbox[3] - bbox[1]

        # div by 0 edge case
        if bbox_height < 1:
            return 999.0
        
        distance = (real_size * FOCAL_LENGTH) / bbox_height

        return distance
    
    def get_bbox_center(self, bbox):
        """
        Get center point of bounding box
        
        Returns:
            (x, y) tuple of center coordinates
        """

        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        return (x_center, y_center)
    
    def assess_danger_level(self, detections):
        """
        Analyze detections and assign danger levels
        
        Returns:
            List of alerts sorted by priority:
            [{
                'priority': int (0=critical, 1=warning, 2=info),
                'message': str,
                'object': detection dict,
                'direction': str (left/center/right)
            }]
        """

        alerts = []
        frame_width = CAMERA_WIDTH

        for det in detections:
            if not det['is_critical']:
                continue

            distance = det['distance']
            obj_name = det['class_name']
            center_x = det['center'][0]

            if center_x < frame_width / 3:
                direction = "left"
            elif center_x > 2 * frame_width / 3:
                direction = "right"
            else:
                direction = "ahead"

            if distance < DANGER_DISTANCE:
                if obj_name in ['car', 'truck', 'bus', 'motorcycle']:
                    message = f"Danger {obj_name.capitalize()} {direction}, {distance:.1f} meters!"
                    priority = PRIORITY_CRITICAL
                elif obj_name == 'person':
                    message = f"Person {direction}, {distance:.1f} meters!"
                    priority = PRIORITY_WARNING
                
                else:
                    message = f"{obj_name.capitalize()} {direction}, {distance:.1f} meters"

                    priority = PRIORITY_WARNING

                alerts.append({
                    'priority': priority,
                    'message': message,
                    'object': det,
                    'direction': direction
                })

            elif distance < WARNING_DISTANCE:
                message = f"{obj_name.capitalize()} approaching {direction}"
                alerts.append({
                    'priority': PRIORITY_WARNING,
                    'message': message,
                    'object': det,
                    'direction': direction
                })

        # Sort alerts by priority
        alerts.sort(key=lambda x: x['priority'])

        return alerts
    
    def draw_detections(self, frame, detections):
        """
        Draw boudning boxes and labels on frame 

        Arguments:
        frame: OpenCV image (BGR)
        detections: list of detection dicts

        Returns:
        Annotated frame

        """

        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            distance = det['distance']

            x1, y1, x2, y2 = bbox.astype(int)

            #color on distance for now
            if distance < DANGER_DISTANCE:
                color = (0, 0, 255) # red
            elif distance < WARNING_DISTANCE:
                color = (0, 255, 255) # yellow
            else:
                color = (0, 255, 0) # green
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}{confidence:.2f}({distance:.1f}m)"
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
            
            # Draw text
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        # Draw FPS counter
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
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
# STANDALONE TEST CODE
# Run this file directly to test the vision system!
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISION SYSTEM TEST")
    print("=" * 60)
    print()
    
    # Initialize vision system
    vision = VisionSystem()
    
    # Open webcam
    print("📸 Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        exit()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("✅ Camera opened successfully")
    print()
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print()
    
    screenshot_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Detect objects
            detections = vision.detect_objects(frame)
            
            # Get danger alerts
            alerts = vision.assess_danger_level(detections)
            
            # Print alerts to console
            if alerts:
                print(f"\n🚨 ALERTS ({len(alerts)}):")
                for alert in alerts:
                    priority_label = ["🔴 CRITICAL", "🟠 WARNING", "🟢 INFO"][alert['priority']]
                    print(f"  {priority_label}: {alert['message']}")
            
            # Draw detections on frame
            annotated_frame = vision.draw_detections(frame, detections)
            
            # Show frame
            cv2.imshow("Vision System Test (Press 'q' to quit)", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")
        print("=" * 60)

