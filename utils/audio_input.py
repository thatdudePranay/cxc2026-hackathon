import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import time

rate = 16000
chunk_dur = 3

model = whisper.load_model("base")

print("Speak into your microphone. Press Ctrl+C to stop.\n")

audio_buffer = []
last_transcription_time = time.time()

# Store audio chunks
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_buffer.append(indata.copy())

# Main logic
try:
    with sd.InputStream(samplerate=rate, channels=1, callback=audio_callback, dtype='int16'):
        
        while True:
            time.sleep(0.1)
            
            # Check if enough audio collected
            current_time = time.time()
            if current_time - last_transcription_time >= chunk_dur:
                
                if len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                    audio_float = audio_data.astype(np.float32) / 32768.0
                    
                    result = model.transcribe(
                        audio_float,
                        language='en',
                        fp16=False,
                        temperature=0.0,
                        condition_on_previous_text=False
                    )
                    
                    text = result['text'].strip()
                    if text:
                        print(f"{text}")
                    
                    audio_buffer = []
                    last_transcription_time = current_time

except KeyboardInterrupt:
    print("\n\nStopped.")
