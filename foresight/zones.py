"""9-zone grid memory - track objects by screen region."""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import time
from foresight.config import ZONE_GRID, ZONE_DECAY_SECONDS


@dataclass
class ZoneObject:
    cls_name: str
    zone: Tuple[int, int]
    distance_hint: str  # "near", "mid", "far" from bbox size
    last_seen: float = field(default_factory=time.time)

    def is_stale(self, max_age: float = ZONE_DECAY_SECONDS) -> bool:
        return time.time() - self.last_seen > max_age


class ZoneMemory:
    """
    3x3 grid over camera view. Zones:
    (0,0) (1,0) (2,0)   left    center   right
    (0,1) (1,1) (2,1)   -       -        -
    (0,2) (1,2) (2,2)   -       -        -
    Rows: top, middle, bottom (near = bottom in camera coords)
    """

    def __init__(self, grid_size: Tuple[int, int] = ZONE_GRID, decay_sec: float = ZONE_DECAY_SECONDS):
        self.nx, self.ny = grid_size
        self.decay_sec = decay_sec
        self._objects: Dict[Tuple[int, int], List[ZoneObject]] = {}
        self._track_ids: Dict[int, ZoneObject] = {}  # track_id -> latest ZoneObject

    def _xy_to_zone(self, cx: float, cy: float, w: int, h: int) -> Tuple[int, int]:
        """Normalized center (0-1) or pixel center -> zone (col, row)."""
        if cx <= 1 and cy <= 1:
            nx = int(cx * self.nx)
            ny = int(cy * self.ny)
        else:
            nx = int((cx / w) * self.nx)
            ny = int((cy / h) * self.ny)
        nx = max(0, min(self.nx - 1, nx))
        ny = max(0, min(self.ny - 1, ny))
        return (nx, ny)

    def _bbox_to_distance_hint(self, bbox_height_norm: float) -> str:
        if bbox_height_norm > 0.4:
            return "near"
        if bbox_height_norm > 0.15:
            return "mid"
        return "far"

    def _prune_stale(self):
        for zone in list(self._objects.keys()):
            self._objects[zone] = [o for o in self._objects[zone] if not o.is_stale(self.decay_sec)]
            if not self._objects[zone]:
                del self._objects[zone]
        stale_ids = [tid for tid, o in self._track_ids.items() if o.is_stale(self.decay_sec)]
        for tid in stale_ids:
            del self._track_ids[tid]

    def update(
        self,
        detections: List[dict],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Update zone memory from detections."""
        self._prune_stale()
        seen_ids = set()
        for d in detections:
            cx, cy = d["center"]
            zone = self._xy_to_zone(cx, cy, frame_width, frame_height)
            dist = self._bbox_to_distance_hint(d.get("bbox_height_norm", 0.2))
            obj = ZoneObject(
                cls_name=d["cls_name"],
                zone=zone,
                distance_hint=dist,
            )
            tid = d.get("id")
            if tid is not None:
                self._track_ids[tid] = obj
                seen_ids.add(tid)
            if zone not in self._objects:
                self._objects[zone] = []
            self._objects[zone].append(obj)

    def get_objects_in_zone(self, col: int, row: int) -> List[str]:
        """Return list of object descriptions in zone, e.g. ['chair near', 'person mid']."""
        self._prune_stale()
        zone = (col, row)
        if zone not in self._objects:
            return []
        # Dedupe by cls_name + distance
        seen = set()
        out = []
        for o in self._objects[zone]:
            key = (o.cls_name, o.distance_hint)
            if key not in seen:
                seen.add(key)
                out.append(f"{o.cls_name} {o.distance_hint}")
        return out

    def describe_all(self) -> str:
        """Human-readable description for Gemini context."""
        self._prune_stale()
        parts = []
        positions = ["left", "center", "right"]
        rows = ["top", "middle", "bottom"]
        for (col, row), objs in sorted(self._objects.items()):
            pos = positions[col] if col < 3 else str(col)
            rw = rows[row] if row < 3 else str(row)
            names = list({f"{o.cls_name} ({o.distance_hint})" for o in objs})
            if names:
                parts.append(f"{pos}-{rw}: {', '.join(names)}")
        return "; ".join(parts) if parts else "No objects in memory"
