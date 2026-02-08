from elevenlabs import play
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from pydub.playback import play as pydub_play
import io
import os

from dotenv import load_dotenv

load_dotenv()

# Initialize ElevenLabs client with API key from environment
api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

client = ElevenLabs(api_key=api_key)

def apply_panning(audio_bytes, pan_value):
    """
    Apply stereo panning to audio.
    
    Args:
        audio_bytes: Audio data as bytes
        pan_value: Float from -60 (full left) to 60 (full right), 0 is center.
                   Values outside -60..60 are clamped, then mapped to -1..1.
    
    Returns:
        Modified audio as AudioSegment
    """
    # Convert bytes to AudioSegment
    audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
    
    # Ensure audio is stereo
    if audio_segment.channels == 1:
        audio_segment = audio_segment.set_channels(2)
    
    # Clamp pan_value to -60..60, then map to -1..1 for equal-power panning
    pan_value = max(-60.0, min(60.0, pan_value))
    pan_value = pan_value / 60.0  # Map -60..60 to -1..1
    
    # Calculate left and right channel gains
    # pan_value of -1.0 = full left (left: 0dB, right: -inf dB)
    # pan_value of 0.0 = center (left: 0dB, right: 0dB)
    # pan_value of 1.0 = full right (left: -inf dB, right: 0dB)
    
    # Calculate gain adjustments using equal-power panning
    import math
    angle = (pan_value + 1.0) * math.pi / 4  # Map -1..1 to 0..pi/2
    left_gain = math.cos(angle)
    right_gain = math.sin(angle)
    
    # Split stereo into left and right
    left_channel = audio_segment.split_to_mono()[0]
    right_channel = audio_segment.split_to_mono()[1]
    
    # Apply panning gains (convert to dB)
    left_db = 20 * math.log10(left_gain) if left_gain > 0 else -100
    right_db = 20 * math.log10(right_gain) if right_gain > 0 else -100
    
    left_channel = left_channel + left_db
    right_channel = right_channel + right_db
    
    # Merge back to stereo
    panned_audio = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
    
    return panned_audio

def speak(text, pan=0.0, save_path=None):
    """
    Generate and play text-to-speech with optional panning.
    
    Args:
        text: Text to speak
        pan: Float from -60 (full left) to 60 (full right), 0 is center (default)
        save_path: Optional path to save audio as MP4/M4A (e.g. "output.m4a")
    """
    if text:
        # Use the correct API method - text_to_speech.convert
        # Using "Rachel" - a free pre-made voice (voice_id: 21m00Tcm4TlvDq8ikWAM)
        # Other free voices: "Adam", "Antoni", "Arnold", "Bella", "Domi", "Elli", "Josh", "Sam"
        audio = client.text_to_speech.convert(
            voice_id="onwK4e9ZLuTAKqWW03F9",  # Rachel - free voice
            text=text,
            model_id="eleven_multilingual_v2"  # Using v1 model for free tier
        )
        
        # Convert generator to bytes if needed
        audio_bytes = b''.join(audio) if hasattr(audio, '__iter__') else audio
        
        # Apply panning if not center
        if pan != 0.0:
            panned_audio = apply_panning(audio_bytes, pan)
            pydub_play(panned_audio)
            audio_segment = panned_audio
        else:
            # Use default play for center audio
            play(audio_bytes)
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

        if save_path:
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            audio_segment.export(save_path, format="ipod")


if __name__ == "__main__":
    speak()