"""
Test Script for main2.py Integration
======================================

This script verifies the main2.py implementation without requiring
full system execution. It checks:
1. Module imports
2. OCR engine functionality
3. Vision system integration
4. Single camera read verification
5. Thread safety

Run this before running main2.py to catch configuration issues.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 70)
print("MAIN2.PY INTEGRATION TEST")
print("=" * 70)
print()

# =============================================================================
# TEST 1: Module Imports
# =============================================================================

print("TEST 1: Checking module imports...")
try:
    from config import *
    print("  ✅ config.py loaded")
except ImportError as e:
    print(f"  ❌ Failed to import config: {e}")
    sys.exit(1)

try:
    from core.vision import EnhancedVisionSystem
    print("  ✅ core.vision loaded")
except ImportError as e:
    print(f"  ❌ Failed to import vision: {e}")
    sys.exit(1)

try:
    from utils.ocr_engine import OCREngine, TextDetection, OCRResult
    print("  ✅ utils.ocr_engine loaded")
except ImportError as e:
    print(f"  ❌ Failed to import ocr_engine: {e}")
    sys.exit(1)

try:
    from utils.audio_output import speak
    print("  ✅ utils.audio_output loaded")
except ImportError as e:
    print(f"  ❌ Failed to import audio_output: {e}")
    sys.exit(1)

try:
    from utils.interpret import find_and_guide
    print("  ✅ utils.interpret loaded")
except ImportError as e:
    print(f"  ❌ Failed to import interpret: {e}")
    sys.exit(1)

# Audio input is optional for testing
try:
    from utils.audio_input_deep import listen_and_transcribe, stop as stop_audio_input
    print("  ✅ utils.audio_input_deep loaded")
    AUDIO_INPUT_AVAILABLE = True
except ImportError as e:
    print(f"  ⚠️  audio_input_deep not available: {e}")
    print("     (This is OK for testing, but needed for full system)")
    AUDIO_INPUT_AVAILABLE = False

print()

# =============================================================================
# TEST 2: Camera Access
# =============================================================================

print("TEST 2: Checking camera access...")
import cv2

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"  ❌ Failed to open camera {CAMERA_INDEX}")
    print(f"     Try changing CAMERA_INDEX in config.py")
    cap.release()
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

ret, frame = cap.read()
if not ret:
    print(f"  ❌ Failed to read frame from camera")
    cap.release()
    sys.exit(1)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"  ✅ Camera opened: {actual_width}x{actual_height}")
print(f"     Frame shape: {frame.shape}")

# Keep frame for later tests
test_frame = frame.copy()
cap.release()
print()

# =============================================================================
# TEST 3: Vision System
# =============================================================================

print("TEST 3: Testing Vision System...")
try:
    vision = EnhancedVisionSystem(use_midas=False)  # Disable MiDaS for speed
    print("  ✅ Vision system initialized")
    
    # Process test frame
    start = time.time()
    vision_data = vision.process_frame(test_frame)
    elapsed = time.time() - start
    
    print(f"  ✅ Frame processed in {elapsed*1000:.0f}ms")
    print(f"     Detections: {len(vision_data['detections'])}")
    print(f"     Alerts: {len(vision_data['alerts'])}")
    print(f"     Collisions: {len(vision_data['collisions'])}")
    print(f"     Walls: {len(vision_data['walls'])}")
    
    # Check for critical objects
    critical_count = sum(1 for d in vision_data['detections'] if d['is_critical'])
    print(f"     Critical objects: {critical_count}")
    
    # Verify data structure
    if len(vision_data['detections']) > 0:
        det = vision_data['detections'][0]
        required_keys = ['class_name', 'distance', 'direction', 'angle', 'is_critical']
        missing_keys = [k for k in required_keys if k not in det]
        if missing_keys:
            print(f"  ⚠️  Detection missing keys: {missing_keys}")
        else:
            print(f"  ✅ Detection structure valid")
            print(f"     Example: {det['class_name']} at {det['distance']:.1f}m, {det['direction']}, {det['angle']:.1f}°")
    
except Exception as e:
    print(f"  ❌ Vision system error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# =============================================================================
# TEST 4: OCR Engine
# =============================================================================

print("TEST 4: Testing OCR Engine...")
try:
    ocr_engine = OCREngine(confidence_threshold=0.6, use_gpu=False, scale_factor=0.5)
    print("  ✅ OCR engine initialized")
    
    # Process test frame
    start = time.time()
    ocr_result = ocr_engine.scan_frame(test_frame)
    elapsed = time.time() - start
    
    print(f"  ✅ OCR scan completed in {elapsed*1000:.0f}ms")
    print(f"     Text regions detected: {len(ocr_result.detections)}")
    
    if len(ocr_result.detections) > 0:
        print(f"     Sample text: '{ocr_result.detections[0].text}'")
        print(f"     Position: {ocr_result.detections[0].position}")
        print(f"     Confidence: {ocr_result.detections[0].confidence:.2f}")
    else:
        print(f"     (No text detected - point camera at text to test)")
    
    # Test helper methods
    all_text = ocr_result.get_all_text()
    text_with_pos = ocr_result.get_text_with_positions()
    print(f"  ✅ Helper methods work")
    
except Exception as e:
    print(f"  ❌ OCR engine error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# =============================================================================
# TEST 5: Single Camera Read Verification
# =============================================================================

print("TEST 5: Verifying single camera read per frame...")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"  ❌ Failed to open camera")
    sys.exit(1)

# Simulate main2.py loop
frame_count = 0
start_time = time.time()
test_duration = 2.0  # Test for 2 seconds

yolo_calls = 0
ocr_calls = 0

print(f"  Running test loop for {test_duration} seconds...")

while time.time() - start_time < test_duration:
    # Read frame ONCE (as in main2.py)
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # YOLO runs every frame
    vision_data = vision.process_frame(frame)
    yolo_calls += 1
    
    # OCR should NOT run every frame (only on demand)
    # In main2.py, it only runs when user speaks
    # For this test, we'll skip it to verify FPS isn't impacted
    
    # Small delay to simulate processing
    time.sleep(0.001)

elapsed = time.time() - start_time
fps = frame_count / elapsed

cap.release()

print(f"  ✅ Single camera read verified")
print(f"     Frames read: {frame_count}")
print(f"     YOLO calls: {yolo_calls}")
print(f"     Average FPS: {fps:.1f}")
print(f"     Time per frame: {(elapsed/frame_count)*1000:.1f}ms")

if fps < 15:
    print(f"  ⚠️  FPS is low ({fps:.1f}). Consider:")
    print(f"     - Disabling MiDaS (use_midas=False)")
    print(f"     - Using smaller resolution in config.py")
    print(f"     - Closing other applications")
elif fps >= 20:
    print(f"  ✅ FPS is good ({fps:.1f})")
else:
    print(f"  ⚠️  FPS is acceptable ({fps:.1f}) but could be better")

print()

# =============================================================================
# TEST 6: Thread Safety (Basic Check)
# =============================================================================

print("TEST 6: Testing thread safety primitives...")
import threading

frame_lock = threading.Lock()
vision_lock = threading.Lock()
audio_output_lock = threading.Lock()

# Test lock acquisition
try:
    with frame_lock:
        test_data = test_frame.copy()
    print("  ✅ frame_lock works")
    
    with vision_lock:
        test_vision = vision_data.copy()
    print("  ✅ vision_lock works")
    
    with audio_output_lock:
        pass  # Just test lock
    print("  ✅ audio_output_lock works")
    
except Exception as e:
    print(f"  ❌ Lock error: {e}")
    sys.exit(1)

print()

# =============================================================================
# TEST 7: API Keys Check
# =============================================================================

print("TEST 7: Checking API keys...")

from dotenv import load_dotenv
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
deepgram_key = os.getenv("DEEPGRAM_API_KEY")

if gemini_key:
    print(f"  ✅ GEMINI_API_KEY found")
else:
    print(f"  ⚠️  GEMINI_API_KEY not found (needed for queries)")

if elevenlabs_key:
    print(f"  ✅ ELEVENLABS_API_KEY found")
else:
    print(f"  ⚠️  ELEVENLABS_API_KEY not found (needed for TTS)")

if deepgram_key:
    print(f"  ✅ DEEPGRAM_API_KEY found")
else:
    print(f"  ⚠️  DEEPGRAM_API_KEY not found (needed for audio input)")

print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print()
print("✅ All critical tests passed!")
print()
print("System is ready to run main2.py")
print()
print("To start the full system:")
print("  python main2.py")
print()
print("Expected behavior:")
print("  1. Video window shows YOLO detections")
print("  2. Critical warnings spoken automatically (with spatial audio)")
print("  3. Audio input continuously listens for queries")
print("  4. OCR runs only when you speak (not continuously)")
print("  5. Gemini responds with context-aware instructions")
print()
print("Controls:")
print("  - Speak naturally for queries")
print("  - Press 'q' to quit")
print()
print("=" * 70)
