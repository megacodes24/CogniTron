import sounddevice as sd
import whisper
import numpy as np
import wavio

# Record 5 seconds
duration = 5
rate = 16000
print("🎙️ Speak now...")
audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='int16')
sd.wait()
wavio.write("test.wav", audio, rate, sampwidth=2)
print("✅ Audio recorded as test.wav")

# Transcribe using Whisper
model = whisper.load_model("base")
result = model.transcribe("test.wav")
print("🗣️ You said:", result["text"])
