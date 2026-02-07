#!/usr/bin/env python3
"""
Foresight - Navigation assistant for visually impaired users.

Phone (DroidCam) -> WiFi -> Laptop (this app)
- YOLO + tracking for collision detection
- 9-zone object memory
- Gemini for scene understanding
- TTS for audio alerts

Usage:
  python -m foresight.main
  python -m foresight.main --url http://192.168.1.100:4747/video  # DroidCam
  python -m foresight.main --no-gemini  # Local only
"""

import argparse
from foresight.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser(description="Foresight - visually impaired navigation")
    ap.add_argument("--url", type=str, default=None, help="Camera URL (DroidCam, IP Webcam)")
    ap.add_argument("--no-gemini", action="store_true", help="Skip Gemini scene understanding")
    ap.add_argument("--no-tts", action="store_true", help="Skip text-to-speech")
    ap.add_argument("--no-window", action="store_true", help="Headless (no OpenCV window)")
    args = ap.parse_args()

    p = Pipeline(
        camera_url=args.url,
        use_gemini=not args.no_gemini,
        use_tts=not args.no_tts,
        show_window=not args.no_window,
    )
    print("Starting Foresight. Press 'q' to quit.")
    p.run()


if __name__ == "__main__":
    main()
