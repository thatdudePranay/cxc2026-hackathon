# Main2.py Integration - Implementation Complete

## Overview

`main2.py` successfully integrates all components:
- ✅ YOLO object detection (continuous, 30 FPS)
- ✅ OCR text detection (on-demand only)
- ✅ Audio input (Deepgram continuous listening)
- ✅ Gemini LLM (context-aware guidance)
- ✅ Audio output (spatial TTS with panning)

## Key Features

### 1. Single Camera Read Per Frame
- Camera opened once in `vision_monitoring_loop()`
- Frame read once per iteration: `ret, frame = cap.read()`
- Frame copied and shared via `frame_lock` for thread safety
- OCR only runs when user prompts Gemini (not every frame)

### 2. Critical Warning System
- Detects people approaching quickly (velocity > 0.5 m/s, distance < 3.0m)
- Immediate audio alerts with spatial panning based on object angle
- Cooldown system (3 seconds) prevents alert spam
- Non-blocking: runs in separate thread to avoid disrupting vision loop

### 3. Spatial Audio
- Uses object `angle` field from vision.py (-60° to +60°)
- Negative angles → left speaker
- Positive angles → right speaker
- Center objects → center audio

### 4. Thread-Safe Architecture
- **Main thread**: Camera loop + YOLO processing
- **Audio thread**: Continuous Deepgram listening
- **Shared state**: Protected by locks (`frame_lock`, `vision_lock`, `audio_output_lock`)

## Files Created

### 1. `utils/ocr_engine.py` (265 lines)
Extracted from notebooks/ocr.ipynb:
- `TextDetection` dataclass - individual text detection
- `OCRResult` dataclass - collection of detections with helper methods
- `OCREngine` class - main OCR processing
- Environment setup: `os.environ["FLAGS_use_mkldnn"] = "0"`
- PaddleOCR PP-OCRv5 mobile models for speed

### 2. `main2.py` (352 lines)
Complete integration script:
- Camera management (single read per frame)
- Vision monitoring loop (continuous YOLO)
- Critical alert detection and spatial audio
- Audio input thread (continuous listening)
- Gemini query handler (YOLO + OCR context)
- Thread-safe state management

## Usage

### Running the System

```bash
python main2.py
```

### Controls
- **Speak naturally** - audio input is always listening
- **Press 'q'** in video window to quit
- **Critical warnings** are automatic (no user action needed)

### Example Interactions

**Critical Warning:**
```
[System detects person approaching at 1.2 m/s, -15° angle]
🚨 Audio: "Person approaching from left" (heard from left speaker)
```

**User Query:**
```
User: "Where is the Rexall pharmacy?"
System: [Runs OCR on current frame]
System: [Gets YOLO detections: car, person, building]
System: [Queries Gemini with all context]
System: "The Rexall is on your left, walk forward"
```

## Testing Checklist

### Basic Functionality
- [ ] Camera opens successfully
- [ ] YOLO detections appear in video window
- [ ] FPS counter shows ~20+ FPS
- [ ] Press 'q' to quit works

### Vision Monitoring
- [ ] Objects detected with distance and direction
- [ ] Moving objects tracked across frames
- [ ] Critical objects (person, car, etc.) flagged correctly

### Critical Alerts
- [ ] Person approaching triggers audio warning
- [ ] Spatial audio pans correctly (left object → left speaker)
- [ ] Cooldown prevents spam (no alert within 3 seconds)
- [ ] Vision loop continues during alert

### Audio Input
- [ ] Deepgram transcribes speech accurately
- [ ] Multiple queries can be made sequentially
- [ ] No interference with critical alerts

### OCR Integration
- [ ] OCR only runs when user speaks (not continuously)
- [ ] Text detected from signs, labels, etc.
- [ ] Positioning reported correctly (left/center/right)

### Gemini Queries
- [ ] Context includes both YOLO and OCR data
- [ ] Gemini provides spatial instructions
- [ ] Results spoken via TTS
- [ ] Errors handled gracefully

### Performance
- [ ] Frame read only once per loop iteration
- [ ] No duplicate camera opens
- [ ] FPS remains stable (20+ FPS target)
- [ ] OCR doesn't block vision monitoring

## Configuration

From `config.py`:
```python
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
DANGER_DISTANCE = 2.0
WARNING_DISTANCE = 4.0
```

OCR settings (in `ocr_engine.py`):
```python
confidence_threshold = 0.6  # Filter low-confidence text
scale_factor = 0.5          # Resize for speed
```

## Troubleshooting

### Camera Not Opening
- Check `CAMERA_INDEX` in config.py (try 0, 1, or 2)
- Ensure no other app is using the camera
- On Windows, set DroidCam or built-in webcam as default

### OCR Not Detecting Text
- Point camera at clear, well-lit text
- Adjust `confidence_threshold` (lower = more detections)
- Check console for "Running OCR..." message

### Audio Input Not Working
- Verify Deepgram API key in `.env`
- Check microphone permissions
- Look for "Listening..." message in console

### Gemini Not Responding
- Verify Gemini API key in `.env`
- Check internet connection
- Look for error messages in console

### Low FPS
- Disable MiDaS: `EnhancedVisionSystem(use_midas=False)`
- Increase OCR `scale_factor` to 0.3 (less accurate but faster)
- Close video window: set `SHOW_DETECTIONS = False` in config.py

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                      MAIN2.PY                           │
│                                                         │
│  ┌──────────────┐                                      │
│  │ Camera Loop  │  (30 FPS, Main Thread)               │
│  │  cv2.read()  │────┬──────────────────────┐          │
│  └──────────────┘    │                      │          │
│         │             ▼                      ▼          │
│         │   ┌──────────────────┐   ┌──────────────┐   │
│         │   │  YOLO Detection  │   │ Frame Storage│   │
│         │   │  (vision.py)     │   │ (with lock)  │   │
│         │   └──────────────────┘   └──────────────┘   │
│         │             │                      │          │
│         │             ▼                      │          │
│         │   ┌──────────────────┐            │          │
│         │   │ Critical Alerts  │            │          │
│         │   │ Check            │            │          │
│         │   └──────────────────┘            │          │
│         │             │                      │          │
│         │             ▼                      │          │
│         │   ┌──────────────────┐            │          │
│         │   │ Spatial Audio    │            │          │
│         │   │ (speak with pan) │            │          │
│         │   └──────────────────┘            │          │
│         │                                    │          │
│  ┌──────────────────────┐                   │          │
│  │ Audio Input Thread   │                   │          │
│  │ (Deepgram listening) │                   │          │
│  └──────────────────────┘                   │          │
│         │                                    │          │
│         ▼                                    │          │
│  ┌─────────────────────────────────────────┘│          │
│  │ User Speech Detected                     │          │
│  └───┬──────────────────────────────────────┘          │
│      │                                                 │
│      ▼                                                 │
│  ┌──────────────────┐                                 │
│  │ Get Frame (lock) │                                 │
│  └────────┬──────────┘                                 │
│           │                                            │
│           ▼                                            │
│  ┌──────────────────┐                                 │
│  │ Run OCR          │                                 │
│  │ (ocr_engine.py)  │                                 │
│  └────────┬──────────┘                                 │
│           │                                            │
│           ▼                                            │
│  ┌──────────────────┐                                 │
│  │ Get YOLO Data    │                                 │
│  │ (from vision)    │                                 │
│  └────────┬──────────┘                                 │
│           │                                            │
│           ▼                                            │
│  ┌──────────────────┐                                 │
│  │ Query Gemini     │                                 │
│  │ (interpret.py)   │                                 │
│  └────────┬──────────┘                                 │
│           │                                            │
│           ▼                                            │
│  ┌──────────────────┐                                 │
│  │ Speak Result     │                                 │
│  │ (audio_output)   │                                 │
│  └──────────────────┘                                 │
└─────────────────────────────────────────────────────────┘
```

## Performance Metrics

**Expected Performance:**
- Vision FPS: 20-30 FPS (with YOLO + MiDaS)
- OCR scan time: 200-500ms per frame
- Critical alert latency: <100ms
- Audio input response: 1-3 seconds (Deepgram processing)
- Gemini query total: 2-5 seconds (OCR + API + TTS)

**Resource Usage:**
- CPU: 40-60% (one core, YOLO + MiDaS)
- RAM: 2-4 GB
- Network: Minimal (only during queries)

## Next Steps

1. **Test with real camera** - verify detection accuracy
2. **Test audio input** - verify Deepgram transcription
3. **Test Gemini queries** - verify context integration
4. **Test spatial audio** - verify panning works correctly
5. **Performance tuning** - optimize FPS if needed
6. **Error handling** - test API failures and recovery

## API Keys Required

Ensure these are set in `.env`:
```
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
```
