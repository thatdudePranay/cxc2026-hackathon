from elevenlabs import play
from elevenlabs.client import ElevenLabs

client = ElevenLabs()

def speak(text):
    if text:
        audio = client.generate(
            text=text,
            voice="FUfBrNit0NNZAwb58KWH",
            model="eleven_multilingual_v2"
        )
        play(audio)