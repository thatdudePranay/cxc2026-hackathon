"""Collision detection using bbox size + direction heuristics."""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from foresight.config import COLLISION_DISTANCE_THRESHOLD, WARNING_COOLDOWN_SECONDS
import time


@dataclass
class CollisionWarning:
    obj_name: str
    direction: str  # "left", "center", "right"
    severity: str   # "close", "very_close"


class CollisionDetector:
    """Check for objects in path using bbox size (larger = closer)."""

    def __init__(
        self,
        distance_threshold: float = COLLISION_DISTANCE_THRESHOLD,
        cooldown_sec: float = WARNING_COOLDOWN_SECONDS,
    ):
        self.distance_threshold = distance_threshold
        self.cooldown_sec = cooldown_sec
        self._last_warn_time = 0.0

    def _direction(self, cx: float, w: float) -> str:
        x_norm = cx / w
        if x_norm < 0.35:
            return "left"
        if x_norm > 0.65:
            return "right"
        return "center"

    def check(
        self,
        detections: List[dict],
        frame_width: int,
        frame_height: int,
    ) -> Optional[CollisionWarning]:
        """
        Return CollisionWarning if something is close and in center.
        Respects cooldown to avoid spam.
        """
        now = time.time()
        if now - self._last_warn_time < self.cooldown_sec:
            return None

        for d in detections:
            bh = d.get("bbox_height_norm", 0)
            if bh < self.distance_threshold:
                continue
            cx, _ = d["center"]
            direction = self._direction(cx, frame_width)
            severity = "very_close" if bh > 0.5 else "close"
            self._last_warn_time = now
            return CollisionWarning(
                obj_name=d["cls_name"],
                direction=direction,
                severity=severity,
            )
        return None
