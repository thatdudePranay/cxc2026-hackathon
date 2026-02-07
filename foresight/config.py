"""Foresight configuration - API keys and settings."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Video input (DroidCam / webcam)
CAMERA_INDEX = int(os.getenv("FORESIGHT_CAMERA", "0"))  # 0 = default, or DroidCam virtual device
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15

# Detection
YOLO_MODEL = "yolov8n.pt"  # nano = fastest; use yolov8s.pt for better accuracy
DETECT_EVERY_N_FRAMES = 2  # Run YOLO every 2nd frame for speed
CONFIDENCE_THRESHOLD = 0.5
COLLISION_CLASSES = None  # None = all; or [0, 56, 57] for person, chair, couch

# Zone memory (3x3 grid)
ZONE_GRID = (3, 3)
ZONE_DECAY_SECONDS = 10  # How long objects stay in memory

# Collision
COLLISION_DISTANCE_THRESHOLD = 0.3  # Normalized bbox size = "close"
WARNING_COOLDOWN_SECONDS = 2  # Don't spam alerts

# Cloud APIs (set in .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Scene understanding
GEMINI_SCENE_INTERVAL_SECONDS = 4
GEMINI_MODEL = "gemini-2.0-flash"

# TTS
USE_GOOGLE_TTS = os.getenv("FORESIGHT_GOOGLE_TTS", "false").lower() == "true"
TTS_LANGUAGE = "en-US"
