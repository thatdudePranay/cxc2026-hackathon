"""Video capture - DroidCam, IP Webcam, or local webcam."""

import cv2
from typing import Optional, Tuple
from foresight.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


def open_camera(
    index: Optional[int] = None,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    url: Optional[str] = None,
) -> cv2.VideoCapture:
    """
    Open video source. Use url for IP Webcam / DroidCam WiFi stream.
    Examples:
      url="http://192.168.1.100:4747/video"  # DroidCam
      url="http://192.168.1.100:8080/video"   # IP Webcam
    """
    if url:
        cap = cv2.VideoCapture(url)
    else:
        cap = cv2.VideoCapture(index if index is not None else CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index={index or CAMERA_INDEX}, url={url})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional["cv2.Mat"]]:
    """Read next frame. Returns (success, frame)."""
    ret, frame = cap.read()
    return ret, frame if ret else None
