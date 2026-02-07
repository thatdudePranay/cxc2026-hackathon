"""CLIP / object search - use Gemini for 'where is X' queries (simpler)."""

from typing import Optional


class CLIPClient:
    """
    Stub for semantic object search. For hackathon, object search
    is handled by Gemini in pipeline. Add Replicate CLIP later if needed.
    """

    def __init__(self, api_token: Optional[str] = None):
        self._available = False

    def is_available(self) -> bool:
        return self._available

    def find_object(self, image_bytes: bytes, query: str) -> Optional[str]:
        return None
