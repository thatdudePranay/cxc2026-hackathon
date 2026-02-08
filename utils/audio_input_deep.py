import sounddevice as sd
import numpy as np
from deepgram import Deepgram
import asyncio
import time
from scipy.io.wavfile import write
import io
import os

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
rate = 16000
chunk_dur = 2

print("Speak into your microphone. Press Ctrl+C to stop.\n")

stop_listening = False
audio_buffer = []
last_transcription_time = time.time()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.append(indata.copy())

async def transcribe_audio(audio_data):
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    
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

def listen_and_transcribe():
    global audio_buffer, last_transcription_time, stop_listening
    audio_buffer = []
    last_transcription_time = time.time()
    stop_listening = False
    full_transcript = ""

    print("Now listening.")
    
    with sd.InputStream(samplerate=rate, channels=1, callback=audio_callback, dtype='int16'):
        while not stop_listening:  # Check if we are still running
            time.sleep(0.1)
            
            current_time = time.time()
            if current_time - last_transcription_time >= chunk_dur:
                
                if len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer, axis=0)
                    
                    text = asyncio.run(transcribe_audio(audio_data))
                    
                    if text.strip():
                        print(f"{text}")
                        full_transcript += " " + text
                    
                    audio_buffer = []
                    last_transcription_time = current_time
    
    print("Stopped listening.")
    return full_transcript.strip()

def stop():
    global stop_listening
    stop_listening = True