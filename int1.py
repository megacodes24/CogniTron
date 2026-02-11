import streamlit as st
from audiorecorder import audiorecorder
import whisper
import torch
import tempfile
import asyncio
import edge_tts
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# ---------------------------
# CONFIG
# ---------------------------
HF_TOKEN = "hf_"   # put your HF token here

st.set_page_config(page_title="Minimal Interview Coach", layout="centered")

# ---------------------------
# SESSION STATE INIT
# ---------------------------
def init_state(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

init_state("track", "System Design")
init_state("difficulty", "Intermediate")
init_state("question_num", 1)
init_state("current_question", None)
init_state("audio_buffer", None)
init_state("history", [])
init_state("last_result", None)
init_state("show_summary", False)

# ---------------------------
# LOAD MODELS ONCE
# ---------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base.en")

@st.cache_resource
def load_readability():
    model_path = r"/models/readability_regressor"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return tok, mdl, device

@st.cache_resource
def load_llm():
    return InferenceClient(api_key=HF_TOKEN)

# ---------------------------
# SPEAK (Edge TTS)
# ---------------------------
async def _speak(text):
    tts = edge_tts.Communicate(text, "en-US-AriaNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        await tts.save(f.name)
        return f.name

def speak(text):
    path = asyncio.run(_speak(text))
    st.audio(path, format="audio/mp3")

# ---------------------------
# SIMPLE QUESTION GENERATOR
# ---------------------------
def generate_question(track, difficulty):
    llm = load_llm()
    prompt = f"""
Generate ONE interview question.
Track: {track}
Difficulty: {difficulty}
Only output the question text.
"""
    out = llm.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.8,
    )
    return out.choices[0].message["content"].strip()

# ---------------------------
# METRIC ENGINE
# ---------------------------
FILLER = ["um","uh","like","basically","you know","sort of","kind of","actually","literally"]
TECH = ["time complexity","space complexity","runtime","memory","optimize","scalable","algorithm","data structure"]
STRUCT = ["first","second","then","next","after","finally","because","therefore"]

def count_words(t): return len(re.findall(r"\b\w+\b", t.lower()))
def count_fillers(t): return sum(t.lower().count(f) for f in FILLER)
def count_hits(t, lst): return sum(1 for x in lst if x in t.lower())

def compute_metrics(text, readability):
    w = max(count_words(text), 1)
    f = count_fillers(text)
    filler_ratio = f / w
    kws = count_hits(text, TECH)
    struct = count_hits(text, STRUCT)

    length_score = 10 if w < 20 else 25 if w < 40 else 35 if w < 80 else 40
    kw_score = min(kws * 5, 25)
    struct_score = min(struct * 5, 20)
    filler_pen = 10 if filler_ratio > 0.15 else 5 if filler_ratio > 0.07 else 0
    read_comp = 3 if readability > 70 else 8 if readability > 50 else 12

    depth = max(0, min(100, length_score + kw_score + struct_score + read_comp - filler_pen))

    return {
        "depth": depth,
        "readability": readability,
        "words": w,
        "fillers": f,
        "filler_ratio": round(filler_ratio, 3),
        "keywords": kws,
        "structure": struct
    }

# ---------------------------
# FEEDBACK GENERATOR
# ---------------------------
def generate_feedback(text, metrics):
    llm = load_llm()
    prompt = f"""
You are an interview coach.
Answer: {text}

Metrics:
Depth: {metrics['depth']}
Readability: {metrics['readability']}
Keywords: {metrics['keywords']}
Structure: {metrics['structure']}

Give a short, natural paragraph of feedback.
"""
    out = llm.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.4,
    )
    return out.choices[0].message["content"].strip()

# ---------------------------
# SUMMARY PAGE
# ---------------------------
def show_summary():
    st.header("Interview Summary")
    for i, item in enumerate(st.session_state.history, 1):
        st.write(f"**Q{i}:** {item['question']}")
        st.write(f"Your answer: {item['answer']}")
        st.write(f"Depth Score: {item['metrics']['depth']}")
        st.write(f"Feedback: {item['feedback']}")
        st.markdown("---")

    if st.button("Restart"):
        st.session_state.history = []
        st.session_state.question_num = 1
        st.session_state.current_question = None
        st.session_state.show_summary = False
        st.rerun()

# ---------------------------
# MAIN LOGIC
# ---------------------------
st.title("Minimal AI Interview Coach")

# Select track & difficulty
st.session_state.track = st.selectbox("Track", ["System Design", "DSA", "Behavioral"])
st.session_state.difficulty = st.radio("Difficulty", ["Starter", "Intermediate", "Deep Dive"])

if st.session_state.show_summary:
    show_summary()
    st.stop()

# NEW question creation
if st.session_state.current_question is None:
    st.write("Generating question...")
    q = generate_question(st.session_state.track, st.session_state.difficulty)
    st.session_state.current_question = q
    speak(f"Question {st.session_state.question_num}. {q}")

st.subheader(f"📌 Question {st.session_state.question_num}")
st.write(st.session_state.current_question)

# AUDIO
audio_raw = audiorecorder("Record", "Stop")

if audio_raw and len(audio_raw) > 0:
    st.session_state.audio_buffer = audio_raw

audio = st.session_state.audio_buffer

if not audio:
    st.info("Record your answer to continue.")
    st.stop()

# PROCESS AUDIO
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
    audio.export(f.name, format="wav")
    wav = f.name

# Whisper
wh = load_whisper()
text = wh.transcribe(wav)["text"].strip()

st.subheader("Your answer")
st.write(text)

# Readability
tok, mdl, dev = load_readability()
inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(dev)

with torch.no_grad():
    readability = mdl(**inputs).logits.squeeze().item()

# Metrics
metrics = compute_metrics(text, readability)

# Feedback
feedback = generate_feedback(text, metrics)
speak("Here is your feedback. " + feedback)

st.subheader("Feedback")
st.write(feedback)

# Save result
st.session_state.last_result = {
    "question": st.session_state.current_question,
    "answer": text,
    "metrics": metrics,
    "feedback": feedback,
}

# Buttons
if st.button("Next Question"):
    st.session_state.history.append(st.session_state.last_result)
    st.session_state.last_result = None
    st.session_state.current_question = None
    st.session_state.audio_buffer = None
    st.session_state.question_num += 1
    st.rerun()

if st.button("End Interview"):
    st.session_state.history.append(st.session_state.last_result)
    st.session_state.show_summary = True
    st.rerun()
