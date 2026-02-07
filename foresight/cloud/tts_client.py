"""Text-to-speech - Google Cloud or pyttsx3 fallback."""

from typing import Optional
from foresight.config import USE_GOOGLE_TTS, TTS_LANGUAGE


class TTSClient:
    """Speak text via Google TTS or local pyttsx3."""

    def __init__(self, use_google: Optional[bool] = None):
        self.use_google = use_google if use_google is not None else USE_GOOGLE_TTS
        self._engine = None
        if not self.use_google:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except Exception:
                self._engine = None

    def speak(self, text: str) -> bool:
        """Speak text. Returns True if successful."""
        if self.use_google:
            return self._google_speak(text)
        return self._local_speak(text)

    def _local_speak(self, text: str) -> bool:
        if not self._engine:
            print(f"[TTS] (no engine): {text}")
            return False
        try:
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as e:
            print(f"[TTS] pyttsx3 error: {e} | {text}")
            return False

    def _google_speak(self, text: str) -> bool:
        try:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=TTS_LANGUAGE,
                name="en-US-Neural2-F",
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
            )
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            import pygame
            import io
            pygame.mixer.init()
            snd = pygame.mixer.Sound(io.BytesIO(response.audio_content))
            snd.play()
            import time
            while pygame.mixer.get_busy():
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"[TTS] Google error: {e}, falling back to print | {text}")
            print(f"[TTS] {text}")
            return False
