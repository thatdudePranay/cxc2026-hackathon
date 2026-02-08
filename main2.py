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
ocr_detected_texts = []  # Store OCR detected texts
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
    global user_input_text, gemini_output_text, ocr_detected_texts
    
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
            
            # ========== MODERN TECH UI PANEL ==========
            h, w = annotated.shape[:2]
            panel_width = 420  # Reduced slightly to prevent cropping
            
            # Create panel with proper bounds checking
            panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
            
            # Sleek dark background with subtle gradient
            for i in range(h):
                gradient_factor = i / h
                r = int(15 + gradient_factor * 5)
                g = int(18 + gradient_factor * 7)
                b = int(28 + gradient_factor * 12)
                panel[i, :] = (b, g, r)
            
            y_pos = 0
            margin = 18  # Consistent margins
            
            # ========== FUTURISTIC HEADER ==========
            header_height = 85
            
            # Neon gradient header background
            for i in range(header_height):
                alpha = i / header_height
                # Cyan to dark blue gradient
                b = int(180 - alpha * 120)
                g = int(120 - alpha * 100)
                r = int(25 - alpha * 15)
                cv2.line(panel, (0, i), (panel_width, i), (b, g, r), 1)
            
            # Tech corner accents (top corners)
            corner_size = 20
            neon_cyan = (255, 255, 0)
            # Top-left corner
            cv2.line(panel, (0, 0), (corner_size, 0), neon_cyan, 2)
            cv2.line(panel, (0, 0), (0, corner_size), neon_cyan, 2)
            # Top-right corner
            cv2.line(panel, (panel_width - corner_size, 0), (panel_width, 0), neon_cyan, 2)
            cv2.line(panel, (panel_width - 1, 0), (panel_width - 1, corner_size), neon_cyan, 2)
            
            # System title with tech styling
            cv2.putText(panel, "A.V.S", (margin, 32),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)  # Cyan accent
            cv2.putText(panel, "ASSISTIVE VISION", (margin, 58),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            # Tech divider line under title
            cv2.line(panel, (margin, 66), (panel_width - margin, 66), (255, 255, 0), 1)
            
            # FPS display - tech style with box
            fps_text = f"{vision_data['fps']:.1f}"
            fps_box_x = panel_width - 85
            fps_box_y = 15
            fps_box_w = 70
            fps_box_h = 50
            
            # FPS container
            cv2.rectangle(panel, (fps_box_x, fps_box_y), 
                         (fps_box_x + fps_box_w, fps_box_y + fps_box_h),
                         (60, 60, 40), -1)
            cv2.rectangle(panel, (fps_box_x, fps_box_y), 
                         (fps_box_x + fps_box_w, fps_box_y + fps_box_h),
                         (255, 255, 0), 2)
            
            cv2.putText(panel, "FPS", (fps_box_x + 8, fps_box_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            cv2.putText(panel, fps_text, (fps_box_x + 10, fps_box_y + 42),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 150), 2)
            
            y_pos = header_height + 12
            
            # ========== STATUS INDICATOR - TECH STYLE ==========
            status_box_y = y_pos
            status_box_height = 48
            
            # Glass-morphism effect background
            cv2.rectangle(panel, (margin, status_box_y), 
                         (panel_width - margin, status_box_y + status_box_height),
                         (35, 40, 50), -1)
            
            # Animated border based on state
            if listening_active:
                border_color = (0, 255, 150)  # Neon green
                glow_intensity = int(150 + 105 * abs(np.sin(time.time() * 3)))
                cv2.rectangle(panel, (margin, status_box_y), 
                             (panel_width - margin, status_box_y + status_box_height),
                             (0, glow_intensity, 100), 2)
            else:
                border_color = (180, 140, 80)  # Muted tech orange
                cv2.rectangle(panel, (margin, status_box_y), 
                             (panel_width - margin, status_box_y + status_box_height),
                             border_color, 1)
            
            # Status indicator with tech icon
            icon_x = margin + 25
            icon_y = status_box_y + 24
            
            if listening_active:
                # Animated listening indicator - concentric circles
                pulse_size = int(8 + 4 * abs(np.sin(time.time() * 6)))
                cv2.circle(panel, (icon_x, icon_y), pulse_size, (0, 255, 150), -1)
                cv2.circle(panel, (icon_x, icon_y), pulse_size + 4, (0, 255, 150), 1)
                cv2.circle(panel, (icon_x, icon_y), pulse_size + 8, (0, 200, 100), 1)
                
                status_text = "LISTENING"
                status_color = (0, 255, 150)
            else:
                # Idle indicator - tech hexagon outline
                hex_size = 8
                for i in range(6):
                    angle1 = i * np.pi / 3
                    angle2 = (i + 1) * np.pi / 3
                    pt1 = (int(icon_x + hex_size * np.cos(angle1)), 
                           int(icon_y + hex_size * np.sin(angle1)))
                    pt2 = (int(icon_x + hex_size * np.cos(angle2)), 
                           int(icon_y + hex_size * np.sin(angle2)))
                    cv2.line(panel, pt1, pt2, (180, 140, 80), 2)
                
                status_text = "PRESS SPACE TO TALK"
                status_color = (180, 180, 200)
            
            cv2.putText(panel, status_text, (icon_x + 20, status_box_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            
            y_pos += status_box_height + 15
            
            # ========== CRITICAL WARNING - HOLOGRAPHIC STYLE ==========
            with text_lock:
                if critical_warning_text:
                    warning_box_height = 95
                    
                    # Holographic flashing effect
                    flash_intensity = abs(np.sin(time.time() * 10))
                    
                    # Multiple layered borders for depth
                    for offset in [3, 2, 1, 0]:
                        alpha = (4 - offset) / 4.0
                        color_intensity = int(255 * flash_intensity * alpha)
                        cv2.rectangle(panel, (margin - offset, y_pos - offset), 
                                     (panel_width - margin + offset, y_pos + warning_box_height + offset),
                                     (0, 0, color_intensity), 1)
                    
                    # Main warning box
                    cv2.rectangle(panel, (margin, y_pos), 
                                 (panel_width - margin, y_pos + warning_box_height),
                                 (20, 20, int(80 + 120 * flash_intensity)), -1)
                    
                    # Top accent line
                    cv2.line(panel, (margin, y_pos + 2), (panel_width - margin, y_pos + 2),
                            (0, 0, 255), 3)
                    
                    # Warning icon - triangle
                    tri_x = margin + 20
                    tri_y = y_pos + 30
                    tri_size = 12
                    triangle = np.array([
                        [tri_x, tri_y - tri_size],
                        [tri_x - tri_size, tri_y + tri_size],
                        [tri_x + tri_size, tri_y + tri_size]
                    ], np.int32)
                    cv2.fillPoly(panel, [triangle], (0, 0, 255))
                    cv2.polylines(panel, [triangle], True, (255, 255, 255), 2)
                    cv2.putText(panel, "!", (tri_x - 4, tri_y + 5),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.putText(panel, "CRITICAL ALERT", (tri_x + 20, y_pos + 28),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Word wrap warning text with better spacing
                    words = critical_warning_text.split()
                    line = ""
                    warn_y = y_pos + 55
                    max_width = 45
                    
                    for word in words:
                        test_line = line + word + " "
                        if len(test_line) > max_width:
                            if line:
                                cv2.putText(panel, line.strip(), (margin + 8, warn_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                                warn_y += 20
                            line = word + " "
                        else:
                            line = test_line
                    
                    if line:
                        cv2.putText(panel, line.strip(), (margin + 8, warn_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    y_pos += warning_box_height + 15
            
            # ========== OCR DETECTED TEXT - TECH DATA DISPLAY ==========
            with text_lock:
                if ocr_detected_texts and y_pos < h - 220:  # Bounds check
                    # Section header with tech styling
                    cv2.rectangle(panel, (margin, y_pos), (panel_width - margin, y_pos + 30),
                                 (40, 50, 80), -1)
                    cv2.line(panel, (margin, y_pos), (panel_width - margin, y_pos),
                            (255, 180, 0), 2)  # Orange accent line
                    
                    cv2.putText(panel, "OCR DATA", (margin + 8, y_pos + 20),
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 180, 0), 2)
                    
                    # Data count indicator
                    count_text = f"[{len(ocr_detected_texts)}]"
                    cv2.putText(panel, count_text, (panel_width - margin - 35, y_pos + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)
                    
                    y_pos += 35
                    
                    # Display detected texts with tech styling
                    max_items = min(4, len(ocr_detected_texts))  # Show max 4
                    for idx in range(max_items):
                        if y_pos > h - 180:  # Stop if too close to bottom
                            break
                            
                        text = ocr_detected_texts[idx]
                        box_height = 32
                        
                        # Alternating tech colors
                        if idx % 2 == 0:
                            bg_color = (45, 50, 65)
                            border_color = (100, 180, 255)  # Cyan
                            text_color = (200, 240, 255)
                        else:
                            bg_color = (50, 45, 60)
                            border_color = (255, 180, 100)  # Orange
                            text_color = (255, 240, 200)
                        
                        # Glass panel effect
                        cv2.rectangle(panel, (margin + 4, y_pos), 
                                     (panel_width - margin - 4, y_pos + box_height),
                                     bg_color, -1)
                        
                        # Tech border with corner accents
                        cv2.rectangle(panel, (margin + 4, y_pos), 
                                     (panel_width - margin - 4, y_pos + box_height),
                                     border_color, 1)
                        
                        # Corner tech details (small)
                        corner_len = 6
                        cv2.line(panel, (margin + 4, y_pos), (margin + 4 + corner_len, y_pos),
                                border_color, 2)
                        cv2.line(panel, (panel_width - margin - 4 - corner_len, y_pos),
                                (panel_width - margin - 4, y_pos), border_color, 2)
                        
                        # Index number
                        cv2.putText(panel, f"{idx + 1}", (margin + 12, y_pos + 21),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
                        
                        # Truncate and display text
                        display_text = text[:38] + "..." if len(text) > 38 else text
                        cv2.putText(panel, display_text, (margin + 30, y_pos + 21),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                        
                        y_pos += box_height + 6
                    
                    y_pos += 10
            
            # ========== CONVERSATION DISPLAY - HOLOGRAPHIC CHAT ==========
            # Tech divider line
            if y_pos < h - 200:  # Bounds check
                cv2.line(panel, (margin, y_pos), (panel_width - margin, y_pos), 
                        (255, 255, 0), 1)
                cv2.line(panel, (margin, y_pos + 2), (panel_width - margin, y_pos + 2), 
                        (255, 255, 0), 1)
                y_pos += 15
            
            with text_lock:
                # User input message
                if user_input_text and y_pos < h - 150:
                    # User label with tech icon
                    label_y = y_pos
                    cv2.putText(panel, ">", (margin, label_y + 18),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 220, 100), 2)
                    cv2.putText(panel, "USER", (margin + 20, label_y + 18),
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 220, 100), 2)
                    y_pos += 28
                    
                    # Calculate message box height
                    words = user_input_text.split()
                    line = ""
                    lines = []
                    max_width = 48
                    
                    for word in words:
                        test_line = line + word + " "
                        if len(test_line) > max_width:
                            if line:
                                lines.append(line.strip())
                            line = word + " "
                        else:
                            line = test_line
                    if line:
                        lines.append(line.strip())
                    
                    box_height = len(lines) * 20 + 12
                    
                    # Holographic message box
                    if y_pos + box_height < h - 120:
                        cv2.rectangle(panel, (margin + 4, y_pos - 4), 
                                     (panel_width - margin - 4, y_pos + box_height),
                                     (50, 60, 70), -1)
                        cv2.rectangle(panel, (margin + 4, y_pos - 4), 
                                     (panel_width - margin - 4, y_pos + box_height),
                                     (255, 220, 100), 1)
                        
                        # Render text lines
                        for i, line in enumerate(lines):
                            if y_pos + i * 20 < h - 130:
                                cv2.putText(panel, line, (margin + 10, y_pos + i * 20 + 14),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                        
                        y_pos += box_height + 18
                
                # Assistant response message
                if gemini_output_text and y_pos < h - 100:
                    # Assistant label with tech icon
                    cv2.putText(panel, "<", (margin, y_pos + 18),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 255, 200), 2)
                    cv2.putText(panel, "A.I.", (margin + 20, y_pos + 18),
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 255, 200), 2)
                    y_pos += 28
                    
                    # Calculate message box height
                    words = gemini_output_text.split()
                    line = ""
                    lines = []
                    max_width = 48
                    
                    for word in words:
                        test_line = line + word + " "
                        if len(test_line) > max_width:
                            if line:
                                lines.append(line.strip())
                            line = word + " "
                        else:
                            line = test_line
                    if line:
                        lines.append(line.strip())
                    
                    box_height = len(lines) * 20 + 12
                    
                    # Holographic message box
                    if y_pos + box_height < h - 80:
                        cv2.rectangle(panel, (margin + 4, y_pos - 4), 
                                     (panel_width - margin - 4, y_pos + box_height),
                                     (40, 60, 60), -1)
                        cv2.rectangle(panel, (margin + 4, y_pos - 4), 
                                     (panel_width - margin - 4, y_pos + box_height),
                                     (100, 255, 200), 1)
                        
                        # Render text lines
                        for i, line in enumerate(lines):
                            if y_pos + i * 20 < h - 90:
                                cv2.putText(panel, line, (margin + 10, y_pos + i * 20 + 14),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 230), 1)
            
            # ========== TECH FOOTER ==========
            footer_y = h - 35
            
            # Footer background
            cv2.rectangle(panel, (0, footer_y - 5), (panel_width, h),
                         (20, 25, 35), -1)
            
            # Top accent lines
            cv2.line(panel, (0, footer_y - 5), (panel_width, footer_y - 5),
                    (255, 255, 0), 1)
            cv2.line(panel, (0, footer_y - 3), (panel_width, footer_y - 3),
                    (255, 255, 0), 1)
            
            # Controls text
            cv2.putText(panel, "[SPACE] TALK", (margin, footer_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1)
            cv2.putText(panel, "[Q] QUIT", (panel_width - margin - 70, footer_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 200), 1)
            
            # Status indicator dot
            status_dot_color = (0, 255, 150) if listening_active else (100, 100, 120)
            cv2.circle(panel, (panel_width // 2, footer_y + 12), 4, status_dot_color, -1)
            
            # Bottom corner accents
            corner_size = 15
            cv2.line(panel, (0, h - corner_size), (0, h), neon_cyan, 2)
            cv2.line(panel, (0, h - 1), (corner_size, h - 1), neon_cyan, 2)
            cv2.line(panel, (panel_width - corner_size, h - 1), (panel_width, h - 1), neon_cyan, 2)
            cv2.line(panel, (panel_width - 1, h - corner_size), (panel_width - 1, h), neon_cyan, 2)
            
            # Combine camera feed and panel with proper sizing
            combined = np.hstack([annotated, panel])
            
            # Ensure window fits on screen
            cv2.namedWindow("Assistive Vision System", cv2.WINDOW_NORMAL)
            cv2.imshow("Assistive Vision System", combined)
            
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
    global current_frame, latest_vision_data, gemini_output_text, ocr_detected_texts
    
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
    
    # Update OCR detected texts for display
    with text_lock:
        ocr_detected_texts = ocr_texts.copy()
    
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
