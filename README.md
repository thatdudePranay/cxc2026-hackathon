# Foresight

Navigation assistant for visually impaired users. **Phone as webcam** (DroidCam) streams video to a **laptop** that runs detection, tracking, collision alerts, and cloud AI.

## Architecture

```
PHONE (DroidCam) → WiFi → LAPTOP
├─ YOLOv8 + ByteTrack (detection + tracking)
├─ 9-zone grid memory
├─ Collision detection (bbox size heuristics)
└─ Cloud: Gemini (scene understanding) + TTS
```

## Quick Start

### 1. Phone setup (5 min)

- Install **DroidCam** (Android) or **EpocCam** (iOS)
- Connect phone and laptop to same WiFi
- Start DroidCam on phone, note the URL (e.g. `http://192.168.1.100:4747/video`)

### 2. Laptop setup

```bash
pip install -r requirements.txt
```

Create `.env` (optional, for Gemini):

```
GEMINI_API_KEY=your_key_here
```

### 3. Run

```bash
# Default webcam
python -m foresight.main

# DroidCam over WiFi
python -m foresight.main --url http://192.168.1.100:4747/video

# Local only (no Gemini)
python -m foresight.main --no-gemini
```

Press **q** to quit.

## Features

| Feature | Description |
|---------|-------------|
| **Collision alerts** | TTS warns when objects are close and centered |
| **Object memory** | 9-zone grid tracks "chair in left-near" |
| **Scene understanding** | Gemini describes environment every 4 sec |
| **TTS** | pyttsx3 (default) or Google Cloud Neural2 |

## Config

- `GEMINI_API_KEY` – Gemini 2.0 Flash for scene understanding
- `FORESIGHT_CAMERA` – OpenCV camera index (default 0)
- `FORESIGHT_GOOGLE_TTS` – Set to `true` for Google TTS (requires credentials)

## Stack

- **Video**: OpenCV (DroidCam / IP Webcam / local webcam)
- **Detection**: YOLOv8n (ultralytics)
- **Tracking**: ByteTrack (built into ultralytics)
- **Depth**: Bbox size heuristic (no depth model)
- **AI**: Gemini 2.0 Flash
- **TTS**: pyttsx3 (local) or Google Cloud TTS
