"""YOLOv8 detection + built-in ByteTrack-style tracking."""

from typing import List, Optional, Tuple
import numpy as np
from foresight.config import YOLO_MODEL, CONFIDENCE_THRESHOLD, DETECT_EVERY_N_FRAMES


class Detector:
    """YOLOv8 with tracking. Uses ultralytics built-in tracker (ByteTrack/BoT-SORT)."""

    def __init__(
        self,
        model_path: str = YOLO_MODEL,
        conf: float = CONFIDENCE_THRESHOLD,
        every_n: int = DETECT_EVERY_N_FRAMES,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.every_n = every_n
        self._frame_count = 0

    def process(
        self,
        frame: np.ndarray,
        track: bool = True,
    ) -> List[dict]:
        """
        Run detection (and optionally tracking). Returns list of:
        {id, xyxy, conf, cls, cls_name, center, bbox_height_norm}
        """
        self._frame_count += 1
        run_track = track and (self._frame_count % self.every_n == 1 or self._frame_count == 1)

        if run_track:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                verbose=False,
                classes=None,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.conf,
                verbose=False,
                classes=None,
            )

        h, w = frame.shape[:2]
        dets = []

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                tid = None
                if boxes.id is not None:
                    tid = int(boxes.id[i].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bh = (y2 - y1) / h  # Normalized bbox height (larger = closer)
                cls_name = self.model.names.get(cls, "object")

                dets.append({
                    "id": tid,
                    "xyxy": xyxy,
                    "conf": conf,
                    "cls": cls,
                    "cls_name": cls_name,
                    "center": (cx, cy),
                    "bbox_height_norm": bh,
                })
        return dets
