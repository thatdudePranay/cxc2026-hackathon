import os

# =============================================================================
# API KEYS - Only needed on whichever one we pick for demo machine
# =============================================================================

GEMINI_API_KEY = "DEMO_MACHINE_ONLY"
ELEVENLABS_API_KEY = "DEMO_MACHINE_ONLY"

# =============================================================================
# SYSTEM SETTINGS - this is for all of us
# =============================================================================

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Detection settings
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
YOLO_DEVICE = "cpu"  # Change to "cuda" if you have GPU

# Important object classes (COCO dataset indices)
CRITICAL_OBJECTS = {
    0: "person",
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
    13: "bench",
}

# Distance thresholds (in meters)
DANGER_DISTANCE = 2.0
WARNING_DISTANCE = 4.0

# Collision detection settings
COLLISION_LOOKAHEAD_TIME = 3.0  # Predict collisions 3 seconds ahead
COLLISION_DISTANCE_THRESHOLD = 2.0  # Collision warning distance in meters
COLLISION_LATERAL_THRESHOLD = 1.5  # Lateral distance threshold in meters
MAX_TRACKING_LOST_FRAMES = 15  # Remove tracker after N frames without detection
TRAJECTORY_MATCHING_THRESHOLD = 100  # Max pixel distance for matching objects
TRAJECTORY_HISTORY_LENGTH = 30  # Number of frames to keep in history (~1 sec at 30fps)
MIN_MOVEMENT_SPEED = 5.0  # Min pixels/sec to consider object as moving
MIN_DEPTH_SPEED = 0.1  # Min m/s depth change to consider approaching/receding

# OCR settings
OCR_INTERVAL = 2.0
OCR_MIN_CONFIDENCE = 0.7

# Audio settings
WHISPER_MODEL = "base"
ELEVENLABS_VOICE = "Adam"
ELEVENLABS_MODEL = "eleven_turbo_v2"

# Alert priorities
PRIORITY_CRITICAL = 0
PRIORITY_WARNING = 1
PRIORITY_INFO = 2

# Object sizes database (meters)
OBJECT_SIZES = {
    "person": 1.7,
    "car": 1.5,
    "motorcycle": 1.2,
    "bus": 3.0,
    "truck": 2.5,
    "bicycle": 1.5,
    "traffic light": 0.8,
    "stop sign": 0.6,
    "bench": 0.5,
    "chair": 0.5,
    "door": 2.0,
}

FOCAL_LENGTH = 800

# Audio feedback
BEEP_FREQUENCIES = {
    "danger": 880,
    "warning": 440,
    "info": 220,
}

BEEP_DURATION = 0.2

# Voice commands
WAKE_WORDS = ["hey assistant", "hey guide", "assistant"]
COMMAND_KEYWORDS = {
    "where": "navigation",
    "what": "description",
    "read": "ocr",
    "help": "help",
}

# Debug settings
DEBUG_MODE = True
SHOW_DETECTIONS = True
SAVE_LOGS = True
LOG_DIR = "logs/"

# Cost tracking
TRACK_API_COSTS = True
GEMINI_COST_PER_IMAGE = 0.00025
ELEVENLABS_COST_PER_1K_CHARS = 0.30