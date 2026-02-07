import sounddevice as sd
import numpy as np
from deepgram import Deepgram
import asyncio
import time
from scipy.io.wavfile import write
import io

DEEPGRAM_API_KEY = "21fa262ef6bc0e3ddf5e25a9036994076424b1f2"

rate = 16000
chunk_dur = 2

print("Speak into your microphone. Press Ctrl+C to stop.\n")

audio_buffer = []
last_transcription_time = time.time()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.append(indata.copy())

async def transcribe_audio(audio_data):
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    
    # Create proper WAV file in memory
    wav_buffer = io.BytesIO()
    write(wav_buffer, rate, audio_data)
    wav_bytes = wav_buffer.getvalue()
    
    source = {
        'buffer': wav_bytes,
        'mimetype': 'audio/wav'
    }
    
    response = await deepgram.transcription.prerecorded(
        source,
        {
            'model': 'nova-2',
            'smart_format': True,
            'language': 'en'
        }
    )
    
    if response and 'results' in response:
        return response['results']['channels'][0]['alternatives'][0]['transcript']
    return ""

try:
    with sd.InputStream(samplerate=rate, channels=1, callback=audio_callback, dtype='int16'):
        while True:
            time.sleep(0.1)
            
            current_time = time.time()
            if current_time - last_transcription_time >= chunk_dur:
                
                if len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer, axis=0)
                    
                    text = asyncio.run(transcribe_audio(audio_data))
                    
                    if text.strip():
                        print(f"{text}")
                    
                    audio_buffer = []
                    last_transcription_time = current_time

except KeyboardInterrupt:
    print("\n\nStopped.")