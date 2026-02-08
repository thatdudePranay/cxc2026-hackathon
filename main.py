import cv2
import numpy as np
import time
import threading

from vision import VisionSystem
from ocr_engine import OCREngine
from audio_input import listen_and_transcribe
from audio_output import speak
from gemini_bridge import find_and_guide

# Globals
running = False
current_frame = None
latest_detections = []
latest_alerts = []
frame_lock = threading.Lock()

# Substuff
vision = VisionSystem()
ocr = OCREngine(confidence_threshold=0.6, scale_factor=0.5)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


"""
Continuos webcam vision capture
""" 
def vision_stream():
    global running, current_frame, latest_detections, latest_alerts

    frame_count = 0
    last_summary_time = time.time()
    
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # Run YOLO from vision
        detections = vision.detect_objects(frame)
        latest_detections = detections

        # Get alerts
        alerts = vision.assess_danger_level(detections)
        latest_alerts = alerts

        # Alert critical dangers
        for alert in alerts:
            if alert['priority'] == 0:
                print(f"\nCritical: {alert['message']}")
                threading.Thread(
                    target=speak,
                    args=(alert['message'],),
                    daemon=True
                ).start()
                break

        annotated = vision.draw_detections(frame, detections)

        with frame_lock:
            current_frame = annotated
        
        time.sleep(0.03)

"""
On demand navigation request with continuous OCR scanning
"""
def handle_navigation_request():
    global current_frame, latest_detections
    max_scan_duration = 5.0
    
    print("Audio input open.")
    transcript = listen_and_transcribe()
    
    if not transcript or len(transcript.strip()) < 3:
        speak("Sorry I didn't hear anything. Please try again.")
        print("No speech detected\n")
        return
        
    # Start scanning OCR
    scan_start_time = time.time()
    attempt = 0
    
    while (time.time() - scan_start_time) < max_scan_duration:
        attempt += 1
        
        # Get continuosly updating frame from camera
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            frame_copy = current_frame.copy()
        
        # Run OCR on the frame
        ocr_result = ocr.scan_frame(frame_copy)
        ocr_texts = [d.text for d in ocr_result.detections]
        
        # Get current YOLO detections
        yolo_objects = [d['class_name'] for d in latest_detections]
        
        result = find_and_guide(transcript, ocr_texts, yolo_objects)
        
        # Check if found
        if result['found']:
            elapsed = time.time() - scan_start_time
            print(f"Instructions: {result['instructions']}")
            speak(result['instructions'])
            return
        
        # Small delay to prevent hammering CPU/API
        time.sleep(0.3)
    
    # Time expired, stop checking
    elapsed = time.time() - scan_start_time
    print(f"\nNot found.)")
    speak(result['instructions'])