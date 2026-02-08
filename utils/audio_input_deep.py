import sounddevice as sd
import numpy as np
from deepgram import DeepgramClient, PrerecordedOptions
import asyncio
import time
from scipy.io.wavfile import write
import io
import os
from dotenv import load_dotenv

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables. Please check your .env file.")
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
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        wav_buffer = io.BytesIO()
        write(wav_buffer, rate, audio_data)
        wav_bytes = wav_buffer.getvalue()
        
        options = PrerecordedOptions(
            model='nova-2',
            smart_format=True,
            language='en'
        )
        
        response = await deepgram.listen.asyncprerecorded.v("1").transcribe_file(
            {"buffer": wav_bytes},
            options
        )
        
        if response and response.results and response.results.channels:
            return response.results.channels[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"⚠️  Audio input error: {e}")
        return ""

def listen_and_transcribe(max_duration=10):
    """
    Listen and transcribe audio with automatic timeout.
    
    Args:
        max_duration: Maximum seconds to listen (default 10)
    """
    global audio_buffer, last_transcription_time, stop_listening
    audio_buffer = []
    last_transcription_time = time.time()
    stop_listening = False
    full_transcript = ""
    start_time = time.time()

    print("Now listening.")
    
    with sd.InputStream(samplerate=rate, channels=1, callback=audio_callback, dtype='int16'):
        while not stop_listening:
            time.sleep(0.1)
            
            current_time = time.time()
            
            # Check for timeout
            if current_time - start_time >= max_duration:
                print(f"⏱️  Timeout after {max_duration}s")
                break
            
            # Transcribe every chunk_dur seconds
            if current_time - last_transcription_time >= chunk_dur:
                
                if len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer, axis=0)
                    
                    text = asyncio.run(transcribe_audio(audio_data))
                    
                    if text.strip():
                        print(f"{text}")
                        full_transcript += " " + text
                    
                    audio_buffer = []
                    last_transcription_time = current_time
        
        # Transcribe any remaining audio in buffer
        if len(audio_buffer) > 0:
            audio_data = np.concatenate(audio_buffer, axis=0)
            text = asyncio.run(transcribe_audio(audio_data))
            if text.strip():
                print(f"{text}")
                full_transcript += " " + text
    
    print("Stopped listening.")
    return full_transcript.strip()

def stop():
    global stop_listening
    stop_listening = True