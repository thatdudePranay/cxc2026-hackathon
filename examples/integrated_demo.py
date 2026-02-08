"""
Example Integration: Vision System + Collision Detection
This demonstrates how to integrate the collision detection system with the existing vision module.
"""

import cv2
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.vision import VisionSystem
from core.collision_detection import CollisionDetector
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


def main():
    print("=" * 60)
    print("INTEGRATED VISION + COLLISION DETECTION DEMO")
    print("=" * 60)
    print()
    
    # Initialize both systems
    print("🔧 Initializing systems...")
    vision = VisionSystem()
    collision_detector = CollisionDetector()
    print("✅ Systems ready!")
    print()
    
    # Open camera
    print("📸 Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("✅ Camera opened successfully")
    print()
    print("=" * 60)
    print("CONTROLS:")
    print("  q - Quit")
    print("  t - Toggle trajectory visualization")
    print("  d - Toggle detection boxes")
    print("  s - Save screenshot")
    print("  i - Print tracking info")
    print("=" * 60)
    print()
    
    # State variables
    show_trajectories = True
    show_detections = True
    screenshot_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            # === STEP 1: Object Detection ===
            detections = vision.detect_objects(frame)
            
            # === STEP 2: Collision Detection ===
            collision_warnings = collision_detector.update(detections)
            
            # === STEP 3: Process Alerts ===
            if collision_warnings:
                print(f"\n{'='*70}")
                print(f"🚨 COLLISION WARNING - {len(collision_warnings)} OBJECT(S) APPROACHING")
                print('='*70)
                
                for idx, warning in enumerate(collision_warnings, 1):
                    # Priority labels with better formatting
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
            
            # === STEP 4: Visualization ===
            annotated_frame = frame.copy()
            
            # Draw detection boxes with collision risk colors
            if show_detections:
                annotated_frame = collision_detector.draw_detections_with_collision_risk(
                    annotated_frame, detections, vision
                )
            
            # Draw trajectories
            if show_trajectories:
                annotated_frame = collision_detector.draw_trajectories(
                    annotated_frame, 
                    show_predictions=True
                )
            
            # Draw collision warning overlay
            if collision_warnings:
                # Count warnings by priority
                critical_count = sum(1 for w in collision_warnings if w['priority'] == 0)
                warning_count = sum(1 for w in collision_warnings if w['priority'] == 1)
                
                if critical_count > 0:
                    # Red banner for critical warnings
                    cv2.rectangle(annotated_frame, (0, 0), (CAMERA_WIDTH, 40), (0, 0, 255), -1)
                    cv2.putText(annotated_frame, f"CRITICAL COLLISION WARNING ({critical_count})",
                              (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                elif warning_count > 0:
                    # Orange banner for regular warnings
                    cv2.rectangle(annotated_frame, (0, 0), (CAMERA_WIDTH, 40), (0, 165, 255), -1)
                    cv2.putText(annotated_frame, f"COLLISION WARNING ({warning_count})",
                              (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display tracking statistics
            tracked_objects = collision_detector.get_tracked_objects()
            moving_count = sum(1 for obj in tracked_objects if obj['is_moving'])
            
            stats_y = CAMERA_HEIGHT - 80
            cv2.putText(annotated_frame, f"Tracked: {len(tracked_objects)} | Moving: {moving_count}",
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show controls hint
            cv2.putText(annotated_frame, "Press 'q' to quit | 't' for trajectories | 'i' for info",
                       (10, CAMERA_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # === STEP 5: Display ===
            cv2.imshow("Vision + Collision Detection Demo", annotated_frame)
            
            # === STEP 6: Handle User Input ===
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            
            elif key == ord('t'):
                show_trajectories = not show_trajectories
                print(f"📊 Trajectory visualization: {'ON' if show_trajectories else 'OFF'}")
            
            elif key == ord('d'):
                show_detections = not show_detections
                print(f"📦 Detection boxes: {'ON' if show_detections else 'OFF'}")
            
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"integrated_demo_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
            
            elif key == ord('i'):
                # Print detailed tracking info
                print("\n" + "="*60)
                print("TRACKING INFORMATION")
                print("="*60)
                
                tracked = collision_detector.get_tracked_objects()
                
                if len(tracked) == 0:
                    print("No objects currently tracked")
                else:
                    for obj in tracked:
                        print(f"\nTrack ID: {obj['track_id']}")
                        print(f"  Object: {obj['object_name']}")
                        print(f"  Position: ({obj['position'][0]:.1f}, {obj['position'][1]:.1f}, {obj['position'][2]:.2f}m)")
                        print(f"  Velocity: {obj['velocity']}")
                        print(f"  Moving: {obj['is_moving']}")
                        print(f"  Direction: {obj['direction']}")
                        print(f"  Frames tracked: {obj['frames_tracked']}")
                
                print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Cleanup complete")
        print("=" * 60)


if __name__ == "__main__":
    main()
