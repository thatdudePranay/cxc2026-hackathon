"""
Main Integration Script - Vision + OCR + Audio + Gemini
========================================================

This script integrates:
- YOLO object detection (continuous monitoring)
- OCR text detection (on-demand)
- Audio input (Deepgram speech-to-text)
- Gemini LLM (context-aware guidance)
- Audio output (spatial TTS based on object positions)

Architecture:
- Single camera stream (read once per frame)
- Main thread: Camera loop + YOLO vision processing
- Audio input thread: Continuous listening for user queries
- Critical warnings: Immediate audio alerts with spatial panning

Controls:
- Press 'q' to quit
- Press SPACE to start listening (release to stop and process)
- Critical warnings (people approaching) are automatic
"""

import cv2
import numpy as np
import time
import threading
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import *
from core.vision import EnhancedVisionSystem
from utils.ocr_engine import OCREngine
from utils.audio_input import listen_and_transcribe, stop as stop_audio_input
from utils.audio_output import speak
from utils.interpret import find_and_guide


# =============================================================================
# GLOBAL STATE
# =============================================================================

# System state
running = True
listening_active = False  # Changed: only listen when button pressed
space_is_held = False  # Track if space is currently held down

# Shared frame data (thread-safe with locks)
current_frame = None
frame_lock = threading.Lock()

# Latest vision data
latest_vision_data = None
vision_lock = threading.Lock()

# Critical alert cooldown (prevent spam)
last_critical_alert_time = 0
CRITICAL_ALERT_COOLDOWN = 3.0  # seconds

# Audio output lock (prevent overlapping speech)
audio_output_lock = threading.Lock()

# Text display for on-screen output
user_input_text = ""
gemini_output_text = ""
critical_warning_text = ""  # For displaying critical warnings on screen
text_lock = threading.Lock()


# =============================================================================
# VISION THREAD - Continuous YOLO Monitoring
# =============================================================================

def vision_monitoring_loop(vision_system, ocr_engine):
    """
    Continuously process frames with YOLO for critical warnings.
    Runs in main thread.
    """
    global running, current_frame, latest_vision_data
    global last_critical_alert_time, listening_active, space_is_held
    global user_input_text, gemini_output_text
    
    print("🔍 Vision monitoring started")
    
    cap = vision_system.open_camera()
    if not cap:
        print("❌ Failed to open camera")
        running = False
        return
    
    # For push-to-talk
    audio_thread = None
    
    try:
        while running:
            # Read frame once
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Store frame for other threads
            with frame_lock:
                current_frame = frame.copy()
            
            # Process with YOLO
            vision_data = vision_system.process_frame(frame)
            
            # Store for Gemini queries
            with vision_lock:
                latest_vision_data = vision_data
            
            # Check for critical alerts
            check_critical_alerts(vision_data)
            
            # Display visualization
            if SHOW_DETECTIONS:
                annotated = vision_system.draw_detections(
                    frame,
                    vision_data['detections'],
                    None,
                    vision_data['tracking']
                )
            else:
                annotated = frame.copy()
            
            # Add text overlay panel on the right
            h, w = annotated.shape[:2]
            panel_width = 400
            panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)  # Dark gray background
            
            # Add FPS and status
            y_pos = 30
            cv2.putText(panel, f"FPS: {vision_data['fps']:.1f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30
            
            # Listening status
            if listening_active:
                status_text = "LISTENING..."
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "Press SPACE to talk"
                status_color = (150, 150, 150)  # Gray
            cv2.putText(panel, status_text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            y_pos += 40
            
            # CRITICAL WARNING banner at top (if active)
            with text_lock:
                if critical_warning_text:
                    # Draw red warning banner
                    banner_height = 80
                    banner = np.zeros((banner_height, panel_width, 3), dtype=np.uint8)
                    banner[:] = (0, 0, 200)  # Red background
                    
                    cv2.putText(banner, "!!! CRITICAL WARNING !!!", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Word wrap warning text
                    words = critical_warning_text.split()
                    line = ""
                    warn_y = 50
                    for word in words:
                        test_line = line + word + " "
                        if len(test_line) > 35:
                            cv2.putText(banner, line, (10, warn_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            warn_y += 20
                            line = word + " "
                        else:
                            line = test_line
                    if line:
                        cv2.putText(banner, line, (10, warn_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Insert banner at top of panel
                    panel = np.vstack([banner, panel[banner_height:, :]])
            
            # Draw separator line
            cv2.line(panel, (10, y_pos), (panel_width-10, y_pos), (100, 100, 100), 1)
            y_pos += 20
            
            # User input text
            with text_lock:
                if user_input_text:
                    cv2.putText(panel, "YOU:", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y_pos += 25
                    # Word wrap user input
                    words = user_input_text.split()
                    line = ""
                    for word in words:
                        test_line = line + word + " "
                        if len(test_line) > 40:
                            cv2.putText(panel, line, (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            y_pos += 20
                            line = word + " "
                        else:
                            line = test_line
                    if line:
                        cv2.putText(panel, line, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_pos += 30
                
                # Gemini output text
                if gemini_output_text:
                    cv2.putText(panel, "ASSISTANT:", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    y_pos += 25
                    # Word wrap gemini output
                    words = gemini_output_text.split()
                    line = ""
                    for word in words:
                        test_line = line + word + " "
                        if len(test_line) > 40:
                            cv2.putText(panel, line, (10, y_pos),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
                            y_pos += 20
                            line = word + " "
                        else:
                            line = test_line
                    if line:
                        cv2.putText(panel, line, (10, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
            
            # Combine camera feed and panel
            combined = np.hstack([annotated, panel])
            
            cv2.imshow("Vision System (SPACE=talk, Q=quit)", combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                running = False
                break
            elif key == ord(' '):  # Space bar pressed
                if not space_is_held and not listening_active:
                    space_is_held = True
                    listening_active = True
                    print("\n[SPACE pressed - Starting to listen...]")
                    # Start audio capture in thread
                    audio_thread = threading.Thread(
                        target=handle_push_to_talk,
                        args=(ocr_engine,),
                        daemon=True
                    )
                    audio_thread.start()
            elif key == 255:  # No key pressed
                # Space was released
                if space_is_held:
                    space_is_held = False
                    print("[SPACE released - Stopping recording...]")
                    stop_audio_input()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")


def check_critical_alerts(vision_data):
    """
    Check for critical warnings (people approaching quickly).
    Feed to Gemini for intelligent warnings and speak with spatial audio.
    """
    global last_critical_alert_time, critical_warning_text
    
    current_time = time.time()
    
    # Cooldown check
    if current_time - last_critical_alert_time < CRITICAL_ALERT_COOLDOWN:
        return
    
    # Check critical alerts from vision system
    critical_alerts = [
        alert for alert in vision_data['alerts']
        if alert['severity'] == 'CRITICAL'
    ]
    
    if not critical_alerts:
        # Clear warning text if no critical alerts
        with text_lock:
            critical_warning_text = ""
        return
    
    # Get highest priority alert
    alert = critical_alerts[0]
    
    # Find the corresponding detection for spatial audio
    pan_angle = 0
    for det in vision_data['detections']:
        if det['class_name'] == alert['object']:
            pan_angle = det['angle']
            break
    
    # Generate intelligent warning through Gemini
    from utils.interpret import generate_critical_warning
    gemini_warning = generate_critical_warning(alert)
    
    # Use Gemini's response if available, otherwise use raw message
    message = gemini_warning if gemini_warning else alert['message']
    
    # Update on-screen warning display
    with text_lock:
        critical_warning_text = message
    
    # Use threading to avoid blocking vision loop
    def speak_alert():
        with audio_output_lock:
            speak(message, pan=pan_angle)
    
    threading.Thread(target=speak_alert, daemon=True).start()
    
    last_critical_alert_time = current_time
    print(f"🚨 CRITICAL ALERT: {message} (pan={pan_angle}°)")


# =============================================================================
# AUDIO INPUT - Push to Talk
# =============================================================================

def handle_push_to_talk(ocr_engine):
    """
    Handle push-to-talk audio input when space is pressed.
    Runs in separate thread.
    """
    global running, listening_active, user_input_text
    
    try:
        # Listen for user speech
        print("🎤 Listening...")
        with text_lock:
            user_input_text = "Listening..."
        
        transcript = listen_and_transcribe()
        
        if not running:
            return
        
        # Stop listening
        listening_active = False
        
        # Check if we got valid speech
        if not transcript or len(transcript.strip()) < 3:
            print("No speech detected")
            with text_lock:
                user_input_text = ""
            return
        
        print(f"📝 User said: '{transcript}'")
        with text_lock:
            user_input_text = transcript
        
        # Process Gemini query
        process_gemini_query(transcript, ocr_engine)
    
    except Exception as e:
        print(f"⚠️  Audio input error: {e}")
        listening_active = False
        with text_lock:
            user_input_text = ""


def process_gemini_query(user_speech, ocr_engine):
    """
    Process user query with Gemini using YOLO + OCR context.
    """
    global current_frame, latest_vision_data, gemini_output_text
    
    # Update display
    with text_lock:
        gemini_output_text = "Processing..."
    
    # Get current frame for OCR
    with frame_lock:
        if current_frame is None:
            response = "No camera frame available."
            with text_lock:
                gemini_output_text = response
            speak(response)
            return
        frame = current_frame.copy()
    
    # Get latest vision data
    with vision_lock:
        if latest_vision_data is None:
            vision_data = {'detections': []}
        else:
            vision_data = latest_vision_data
    
    # Run OCR on current frame
    print("🔤 Running OCR...")
    ocr_result = ocr_engine.scan_frame(frame)
    ocr_texts = [d.text for d in ocr_result.detections]
    
    print(f"   Found {len(ocr_texts)} text regions: {ocr_texts}")
    
    # Get YOLO objects
    yolo_objects = [
        f"{det['class_name']} at {det['distance']:.1f}m {det['direction']}"
        for det in vision_data['detections']
    ]
    
    print(f"   YOLO objects: {len(yolo_objects)}")
    
    # Query Gemini
    print("🤖 Querying Gemini...")
    result = find_and_guide(user_speech, ocr_texts, yolo_objects)
    
    print(f"   Target: {result['target']}")
    print(f"   Found: {result['found']}")
    print(f"   Instructions: {result['instructions']}")
    
    # Update display and speak result
    with text_lock:
        gemini_output_text = result['instructions']
    
    with audio_output_lock:
        speak(result['instructions'])


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point - initialize systems and start threads.
    """
    global running
    
    print("=" * 70)
    print("MAIN2.PY - INTEGRATED ASSISTIVE SYSTEM")
    print("=" * 70)
    print()
    print("Initializing systems...")
    print()
    
    # Initialize vision system
    print("📸 Loading YOLO + MiDaS...")
    vision_system = EnhancedVisionSystem(use_midas=True)
    
    # Initialize OCR engine
    print("\n🔤 Loading OCR engine...")
    ocr_engine = OCREngine(confidence_threshold=0.6, use_gpu=False, scale_factor=0.5)
    
    print("\n" + "=" * 70)
    print("✅ ALL SYSTEMS READY")
    print("=" * 70)
    print()
    print("Controls:")
    print("  - Press and hold SPACE to start listening")
    print("  - Release SPACE or wait for transcription to complete")
    print("  - Press 'Q' to quit")
    print("  - Critical warnings are automatic")
    print()
    print("Starting...")
    print()
    
    # Run vision monitoring in main thread (handles everything now)
    try:
        vision_monitoring_loop(vision_system, ocr_engine)
    except KeyboardInterrupt:
        print("\n👋 Keyboard interrupt")
    finally:
        # Cleanup
        running = False
        stop_audio_input()
        
        print("\n" + "=" * 70)
        print("✅ SYSTEM SHUTDOWN COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
