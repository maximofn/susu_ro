import sys
import os
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.stt.whisper import Whisper

whisper = Whisper(model_size="large", device="cpu")

print(whisper.device)

language = "en"
audio_path = "../data/MicroMachines.mp3"

start_time = time.time()
transcription = whisper.transcribe(audio_path, language)
end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
