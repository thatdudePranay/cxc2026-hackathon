"""Main processing pipeline - real-time loop + cloud loop."""

import cv2
import time
import threading
from typing import Optional
from foresight.camera import open_camera, read_frame
from foresight.detector import Detector
from foresight.zones import ZoneMemory
from foresight.collision import CollisionDetector, CollisionWarning
from foresight.cloud.gemini_client import GeminiClient
from foresight.cloud.tts_client import TTSClient
from foresight.config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    GEMINI_SCENE_INTERVAL_SECONDS,
)


class Pipeline:
    """Real-time detection + collision + periodic Gemini scene + TTS."""

    def __init__(
        self,
        camera_url: Optional[str] = None,
        use_gemini: bool = True,
        use_tts: bool = True,
        show_window: bool = True,
    ):
        self.cap = open_camera(width=FRAME_WIDTH, height=FRAME_HEIGHT, url=camera_url)
        self.detector = Detector()
        self.zones = ZoneMemory()
        self.collision = CollisionDetector()
        self.gemini = GeminiClient() if use_gemini else None
        self.tts = TTSClient() if use_tts else None
        self.show_window = show_window

        self._last_gemini_time = 0.0
        self._running = False
        self._gemini_thread: Optional[threading.Thread] = None
        self._latest_frame_bytes: Optional[bytes] = None
        self._frame_lock = threading.Lock()

    def _run_realtime_loop(self):
        """Main loop: grab frame, detect, track, collision, display."""
        while self._running:
            ret, frame = read_frame(self.cap)
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            with self._frame_lock:
                self._latest_frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()

            dets = self.detector.process(frame, track=True)
            self.zones.update(dets, w, h)
            warn = self.collision.check(dets, w, h)

            if warn and self.tts:
                msg = f"Watch out, {warn.obj_name} {warn.direction}"
                self.tts.speak(msg)

            # Periodic Gemini (in background)
            now = time.time()
            if (
                self.gemini
                and self.gemini.is_available()
                and now - self._last_gemini_time > GEMINI_SCENE_INTERVAL_SECONDS
            ):
                self._last_gemini_time = now
                frame_copy = self._latest_frame_bytes
                zone_ctx = self.zones.describe_all()
                t = threading.Thread(
                    target=self._gemini_scene_async,
                    args=(frame_copy, zone_ctx),
                )
                t.daemon = True
                t.start()

            if self.show_window:
                self._draw(frame, dets)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False

        self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()

    def _gemini_scene_async(self, frame_bytes: bytes, zone_ctx: str):
        desc = self.gemini.describe_scene(frame_bytes, zone_ctx)
        if self.tts and desc and not desc.startswith("Error"):
            self.tts.speak(desc)

    def _draw(self, frame, dets):
        for d in dets:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['cls_name']} {d.get('id', '?')}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        zone_desc = self.zones.describe_all()
        if zone_desc:
            cv2.putText(
                frame, zone_desc[:80] + "..." if len(zone_desc) > 80 else zone_desc,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
        cv2.imshow("Foresight", frame)

    def run(self):
        """Start pipeline."""
        self._running = True
        self._run_realtime_loop()

    def stop(self):
        self._running = False
