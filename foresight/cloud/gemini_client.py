"""Gemini 2.0 Flash for scene understanding."""

import base64
from typing import Optional
from foresight.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiClient:
    """Scene description and Q&A via Gemini."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model
        self._client = None
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(model)

    def is_available(self) -> bool:
        return self._client is not None

    def describe_scene(
        self,
        image_bytes: bytes,
        zone_context: str = "",
    ) -> str:
        """Get scene description for ambient awareness."""
        if not self._client:
            return "Gemini API key not set."
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))
            prompt = (
                "Describe this indoor scene briefly for a visually impaired user. "
                "Mention obstacles, furniture, people, and open paths. "
                "Be concise (1-2 sentences)."
            )
            if zone_context:
                prompt += f"\nObject memory: {zone_context}"
            response = self._client.generate_content([prompt, img])
            return response.text if response.text else "No description."
        except Exception as e:
            return f"Error: {e}"

    def answer_query(self, query: str, image_bytes: Optional[bytes] = None) -> str:
        """Answer user question, optionally with current frame."""
        if not self._client:
            return "Gemini API key not set."
        try:
            parts = [f"User query: {query}"]
            if image_bytes:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_bytes))
                parts.append(img)
            response = self._client.generate_content(parts)
            return response.text if response.text else "I couldn't answer that."
        except Exception as e:
            return f"Error: {e}"
