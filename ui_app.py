import streamlit as st
from audiorecorder import audiorecorder   # or from st_audiorec import st_audiorec
import whisper
import tempfile

st.title("Voice AI Interview Coach")
st.write("Click the record button and start speaking...")

# Record audio directly from mic
audio = audiorecorder("Click to record", "Recording...")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    st.success("Audio captured! Transcribing...")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        audio.export(temp_file.name, format="wav")
        temp_path = temp_file.name

    # Load Whisper model
    model = whisper.load_model("base")
    result = model.transcribe(temp_path)
    st.write("🗣️ You said:", result["text"])
