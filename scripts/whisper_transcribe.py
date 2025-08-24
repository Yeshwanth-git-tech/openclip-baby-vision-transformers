import whisper
import os

# Load Whisper model
model = whisper.load_model("base")  # or "medium", "large"

# Directory with audio files
audio_dir = "path/to/audio/files"

for fname in os.listdir(audio_dir):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(audio_dir, fname)
    print(f"\nTranscribing {fname}...")
    result = model.transcribe(path)
    print("→", result["text"])