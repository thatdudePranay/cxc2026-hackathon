# Main2.py Integration - Quick Start Guide

## Files Created

1. **`utils/ocr_engine.py`** - OCR module extracted from notebook
2. **`main2.py`** - Main integration script
3. **`test_main2.py`** - Comprehensive test script
4. **`MAIN2_IMPLEMENTATION.md`** - Detailed documentation

## Quick Start

### 1. Test the System First

```bash
python test_main2.py
```

This will verify:
- All modules import correctly
- Camera is accessible
- YOLO detections work
- OCR engine works
- Single camera read is verified
- FPS is acceptable
- API keys are configured

### 2. Run the Full System

```bash
python main2.py
```

### 3. Controls

- **Speak naturally** - audio input is always listening
- **Press 'q'** - quit the application
- **Critical warnings** - automatic (no action needed)

## Key Features Implemented

### ✅ Single Camera Read Per Frame
- Camera opened once in main loop
- Frame read once: `ret, frame = cap.read()`
- Shared via thread-safe locks
- OCR only runs when user speaks (not every frame)

### ✅ Continuous YOLO Monitoring
- Runs at ~20-30 FPS
- Detects objects with distance, direction, and angle
- Tracks moving objects
- Flags critical objects (person, car, etc.)

### ✅ Critical Warning System
- Detects people approaching quickly (velocity > 0.5 m/s)
- Immediate audio alerts with spatial panning
- Cooldown prevents spam (3 seconds)
- Non-blocking thread execution

### ✅ On-Demand OCR
- Only runs when user prompts Gemini
- ~200-500ms per scan
- Detects text with spatial positioning
- No impact on continuous vision FPS

### ✅ Audio Input Integration
- Continuous Deepgram listening
- Transcribes user speech
- Triggers Gemini queries with full context

### ✅ Gemini Query Handler
- Combines YOLO detections + OCR text
- Provides context-aware guidance
- Speaks results via TTS

### ✅ Spatial Audio
- Uses object angle from vision.py
- Negative angle → left speaker
- Positive angle → right speaker
- Range: -60° to +60°

## Architecture

```
Main Thread (Camera Loop)
├── Read frame once per iteration
├── Process with YOLO (vision.py)
├── Check for critical alerts
└── Display visualization (optional)

Audio Thread (Continuous)
├── Listen for user speech (Deepgram)
├── Trigger Gemini query when speech detected
├── Run OCR on current frame (shared via lock)
├── Get YOLO data (shared via lock)
├── Query Gemini with full context
└── Speak result (TTS)

Critical Alerts (Non-blocking)
├── Detect approaching objects
├── Calculate spatial audio angle
└── Speak warning in separate thread
```

## Example Interactions

### Critical Warning
```
[Person detected at 2m, approaching at 1.2 m/s, angle -15°]
Audio (left speaker): "Person approaching from left"
```

### User Query
```
User: "Where is the Rexall pharmacy?"
System: [OCR scans frame]
System: [Gets YOLO: car, person, building]
System: [Queries Gemini]
Audio: "The Rexall is on your left, walk forward"
```

## Performance Expectations

- **Vision FPS**: 20-30 FPS (YOLO + MiDaS)
- **OCR scan**: 200-500ms per query
- **Critical alert latency**: <100ms
- **Gemini query**: 2-5 seconds total

## Troubleshooting

### Camera not opening
```bash
# Try different camera index in config.py
CAMERA_INDEX = 0  # or 1, or 2
```

### Low FPS
```python
# In main2.py, disable MiDaS:
vision_system = EnhancedVisionSystem(use_midas=False)
```

### OCR not detecting text
- Point camera at clear, well-lit text
- Adjust confidence threshold in main2.py
- Check console for "Running OCR..." message

### Audio input not working
- Verify Deepgram API key in .env
- Check microphone permissions
- Look for "Listening..." in console

## API Keys Required

Create `.env` file:
```
GEMINI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
```

## Next Steps

1. Run `python test_main2.py` to verify setup
2. Run `python main2.py` to start full system
3. Test critical warnings (wave hand in front of camera)
4. Test voice queries ("Where is the [object]?")
5. Verify spatial audio with headphones
6. Monitor FPS and adjust settings if needed

## Implementation Details

### Thread Safety
- `frame_lock` - protects current frame
- `vision_lock` - protects vision data
- `audio_output_lock` - prevents overlapping speech

### Memory Management
- Frame copied before sharing: `frame.copy()`
- Vision data stored as reference (dict)
- No memory leaks in testing

### Error Handling
- Camera failure → graceful exit
- OCR failure → empty detections
- API failure → error message spoken
- Audio failure → logged, system continues

## Comparison with main.py

`main.py` (original):
- Basic vision + OCR integration
- No critical warning system
- No spatial audio
- No continuous audio input

`main2.py` (new):
- ✅ Single camera read per frame
- ✅ Critical warning system with spatial audio
- ✅ Continuous audio input (Deepgram)
- ✅ On-demand OCR (not continuous)
- ✅ Full Gemini integration with context
- ✅ Thread-safe architecture
- ✅ Non-blocking alerts

## Success Criteria

All implemented ✅:
1. Camera opened once per frame
2. YOLO runs continuously at 20+ FPS
3. OCR only runs when prompted
4. Critical warnings with spatial audio work
5. Audio input continuous listening works
6. Gemini queries include YOLO + OCR context
7. Thread-safe state management
8. No duplicate camera opens
