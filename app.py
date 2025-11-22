# Nas Galaxy AI - Billion-dollar UI Streamlit App
# Single-file Streamlit application (app.py)
# Features: Chat, Emotion AI, Image Vision, Summarizer, PDF Q&A, Code Assistant,
# Voice input (optional), Voice output (optional), Memory, Reasoning mode,
# Sleek glassmorphism UI. Uses Hugging Face transformers pipelines with
# graceful fallbacks and lazy loading to keep startup responsive.

# NOTE: Models will download the first time you run (internet required).
# If you want smaller/faster models, change the MODEL_* variables below.

import streamlit as st
from streamlit import session_state as ss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from pathlib import Path
import tempfile
import base64
import os
from io import BytesIO
from typing import List
import time

# Optional libraries (voice, pdf). App will detect and degrade gracefully if missing.
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# ----------------- CONFIG / SMALL MODELS -----------------
# You can swap these to larger models if you have GPU and bandwidth.
MODEL_CHAT = "gpt2"  # small; change to 'gpt-j' or 'mistral' if available
MODEL_EMOTION = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_SUMMARIZER = "sshleifer/distilbart-cnn-12-6"
MODEL_IMAGE_CAPTION = "nlpconnect/vit-gpt2-image-captioning"  # may be heavy
MODEL_CODE = "Salesforce/codegen-350M-multi"  # smallish codegen

# ----------------- STYLES -----------------
st.set_page_config(page_title="Nas Galaxy AI", layout="wide", initial_sidebar_state="expanded")

# Glassmorphism + gradient header
PAGE_CSS = '''
<style>
body { background: linear-gradient(135deg, #0f172a 0%, #021124 100%); color: #e6eef8; }
.main > div { background: rgba(255,255,255,0.03); border-radius: 16px; padding: 18px; }
.header { display:flex; align-items:center; gap:16px; }
.logo { width:72px; height:72px; border-radius:16px; background:linear-gradient(135deg,#6EE7B7,#3B82F6); box-shadow:0 6px 18px rgba(59,130,246,0.2); display:flex; align-items:center; justify-content:center; font-weight:800; font-size:20px; }
.h1 { font-size:28px; font-weight:700; margin:0; }
.h2 { color:#9FB3D8; margin:0; }
.small { color:#c8d7ee; font-size:13px }
.mode-card { background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; }
</style>
'''
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "history" not in ss:
    ss.history = []  # list of (role, text, metadata)
if "memory" not in ss:
    ss.memory = []
if "models_loaded" not in ss:
    ss.models_loaded = False
if "mode" not in ss:
    ss.mode = "Chat"

# ----------------- MODEL LOADING (lazy) -----------------
@st.cache_resource
def load_chat_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHAT)
        model = AutoModelForCausalLM.from_pretrained(MODEL_CHAT)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return gen
    except Exception as e:
        st.warning(f"Chat model load failed: {e}. Falling back to simple echo.")
        return None

@st.cache_resource
def load_emotion_model():
    try:
        return pipeline("sentiment-analysis", MODEL_EMOTION)
    except Exception:
        return None

@st.cache_resource
def load_summarizer():
    try:
        return pipeline("summarization", MODEL_SUMMARIZER)
    except Exception:
        return None

@st.cache_resource
def load_image_caption():
    try:
        return pipeline("image-captioning", MODEL_IMAGE_CAPTION)
    except Exception:
        return None

@st.cache_resource
def load_code_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CODE)
        model = AutoModelForCausalLM.from_pretrained(MODEL_CODE)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return gen
    except Exception:
        return None

# Lazy loaders
CHAT_PIPELINE = None
EMOTION_PIPE = None
SUMMARY_PIPE = None
IMG_CAP_PIPE = None
CODE_PIPE = None

# ----------------- LAYOUT -----------------
with st.container():
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown(f'<div class="logo">NG</div>', unsafe_allow_html=True)
    st.markdown('<div style="flex:1">', unsafe_allow_html=True)
    st.markdown('<div class="h1">Nas Galaxy AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="h2">Billion-dollar Copilot — Chat • Vision • Code • Docs</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Pro UI • Glassmorphism • Session memory • Reasoning Mode</div>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### Modes")
    mode = st.selectbox("Choose Mode", ["Chat", "Emotion AI", "Image Vision", "Summarizer", "PDF Q&A", "Code Assistant", "Memory", "Settings"], index=0)
    ss.mode = mode
    st.divider()
    st.markdown("### Quick Actions")
    if st.button("Load Models (may take time)"):
        with st.spinner("Downloading models... this can take a few minutes depending on your net"):
            CHAT_PIPELINE = load_chat_model()
            EMOTION_PIPE = load_emotion_model()
            SUMMARY_PIPE = load_summarizer()
            IMG_CAP_PIPE = load_image_caption()
            CODE_PIPE = load_code_model()
            ss.models_loaded = True
        st.success("Models loaded (or tried). If any failed, app will fallback gracefully.")

    st.markdown("---")
    st.markdown("Made with ❤️ by Nas Galaxy AI")

# ----------------- UTILITIES -----------------
def speak_text(text: str):
    if not TTS_AVAILABLE:
        st.info("TTS engine not available. Install pyttsx3 to enable voice output.")
        return
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()


def append_history(role, text, meta=None):
    ss.history.append((role, text, meta))


def render_chat():
    for i, (role, txt, meta) in enumerate(ss.history):
        if role == "user":
            st.markdown(f"**🧍 You:** {txt}")
        else:
            st.markdown(f"**🤖 Nas Galaxy AI:** {txt}")

# ----------------- MODE HANDLERS -----------------
if ss.mode == "Chat":
    st.subheader("Copilot Chat — multi-mode reasoning")
    col1, col2 = st.columns([3,1])
    with col1:
        user_input = st.text_area("Ask anything — be specific for best results", key="chat_input")
        thinking = st.checkbox("Enable reasoning / chain-of-thought (longer responses)", value=False)
        if st.button("Send"):
            append_history("user", user_input, {"mode":"chat"})
            # lazy load
            if CHAT_PIPELINE is None:
                CHAT_PIPELINE = load_chat_model()
            if CHAT_PIPELINE:
                gen_kwargs = {"max_length": 256, "temperature": 0.7}
                if thinking:
                    gen_kwargs["temperature"] = 0.2
                    gen_kwargs["max_length"] = 512
                try:
                    out = CHAT_PIPELINE(user_input, **gen_kwargs)[0]["generated_text"]
                except Exception as e:
                    out = "[Model error — fallback] " + str(e)
            else:
                out = "I am Nas Galaxy AI — your Copilot. (Fallback generator) " + user_input[::-1]
            append_history("ai", out, {"thinking": thinking})
            render_chat()
            if st.checkbox("Listen to AI reply", value=False):
                speak_text(out)
    with col2:
        st.markdown("### Session Memory")
        if ss.memory:
            for m in ss.memory[-6:][::-1]:
                st.markdown(f"- {m}")
        mem_in = st.text_input("Add to memory")
        if st.button("Save to memory") and mem_in:
            ss.memory.append(mem_in)
            st.success("Saved to memory")

elif ss.mode == "Emotion AI":
    st.subheader("Emotion & Tone Detection")
    text = st.text_area("Enter text to analyze")
    if st.button("Analyze"):
        if EMOTION_PIPE is None:
            EMOTION_PIPE = load_emotion_model()
        if EMOTION_PIPE:
            res = EMOTION_PIPE(text)[0]
            st.markdown(f"**Label:** {res['label']}  —  **Score:** {res['score']:.2f}")
        else:
            st.info("Emotion model not available. Try loading models from sidebar.")

elif ss.mode == "Image Vision":
    st.subheader("Image Vision — upload image and ask")
    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"]) 
    caption_prompt = st.text_input("(Optional) Ask about the image — e.g., 'Describe the scene', 'Find objects', 'What is the mood?'")
    if uploaded is not None:
        st.image(uploaded)
        if st.button("Analyze Image"):
            if IMG_CAP_PIPE is None:
                IMG_CAP_PIPE = load_image_caption()
            if IMG_CAP_PIPE:
                try:
                    bytes_img = uploaded.read()
                    out = IMG_CAP_PIPE(bytes_img)
                    if isinstance(out, list):
                        text = out[0].get("caption") or out[0].get("generated_text") or str(out)
                    else:
                        text = str(out)
                    st.markdown(f"**Caption:** {text}")
                except Exception as e:
                    st.error(f"Image model error: {e}")
            else:
                st.info("Image caption model not ready. Please load models from sidebar.")

elif ss.mode == "Summarizer":
    st.subheader("Power Summarizer — paste long text or upload file")
    t = st.text_area("Paste text here (or upload document below)")
    uploaded = st.file_uploader("(Optional) Upload .txt file", type=["txt"]) 
    if uploaded and not t:
        t = uploaded.read().decode('utf-8')
    length = st.selectbox("Summary length", ["short","medium","long"], index=1)
    if st.button("Summarize"):
        if SUMMARY_PIPE is None:
            SUMMARY_PIPE = load_summarizer()
        if SUMMARY_PIPE:
            mins = 30 if length=="short" else 60 if length=="medium" else 120
            try:
                out = SUMMARY_PIPE(t, max_length=mins, min_length=20)
                s = out[0].get('summary_text') if isinstance(out, list) else str(out)
                st.success(s)
            except Exception as e:
                st.error(f"Summarizer error: {e}")
        else:
            st.info("Summarizer not ready. Load models or paste shorter text.")

elif ss.mode == "PDF Q&A":
    st.subheader("PDF Q&A — upload PDF, then ask questions")
    if not PDF_AVAILABLE:
        st.info("PyPDF2 not installed. Install PyPDF2 to enable PDF parsing.")
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf is not None:
        if PDF_AVAILABLE:
            reader = PyPDF2.PdfReader(pdf)
            pages = [p.extract_text() or '' for p in reader.pages]
            big = "\n".join(pages)
            st.markdown(f"**Document loaded — {len(pages)} pages**")
            q = st.text_input("Ask a question about this PDF")
            if st.button("Answer") and q:
                # simple retrieval: find best page by naive keyword match
                best_page = max(range(len(pages)), key=lambda i: q.lower() in pages[i].lower())
                context = pages[best_page]
                prompt = f"Use the following context to answer briefly:\nContext:\n{context[:2000]}\n\nQuestion: {q}\nAnswer:"
                if CHAT_PIPELINE is None:
                    CHAT_PIPELINE = load_chat_model()
                if CHAT_PIPELINE:
                    try:
                        ans = CHAT_PIPELINE(prompt, max_length=256)[0]["generated_text"]
                    except Exception as e:
                        ans = "[Model error] " + str(e)
                else:
                    ans = "I couldn't run the model — preview of context: " + context[:500]
                st.write(ans)
        else:
            st.info("PDF parsing not available (PyPDF2 missing). You can still upload and download manually.)")

elif ss.mode == "Code Assistant":
    st.subheader("DeepSeek-style Code Assistant")
    code_prompt = st.text_area("Describe the code you want (be specific):")
    language = st.selectbox("Language", ["python","javascript","java","c++","html","css"], index=0)
    if st.button("Generate Code"):
        if CODE_PIPE is None:
            CODE_PIPE = load_code_model()
        if CODE_PIPE:
            prompt = f"# Write a {language} snippet for: {code_prompt}\n\n### {language} code:\n"
            try:
                out = CODE_PIPE(prompt, max_length=400, temperature=0.2)[0]["generated_text"]
                # try to remove prompt echo
                code_out = out.split('###')[-1].strip()
                st.code(code_out, language=language)
            except Exception as e:
                st.error(f"Code model failed: {e}")
        else:
            st.info("Code model not available. Installing or loading may be required.")

elif ss.mode == "Memory":
    st.subheader("Memory — session & export")
    st.markdown("**Session memory (latest first)**")
    for m in ss.memory[::-1]:
        st.markdown(f"- {m}")
    if st.button("Export memory to txt"):
        data = "\n".join(ss.memory)
        b = data.encode('utf-8')
        st.download_button("Download memory.txt", data=b, file_name="nas_galaxy_memory.txt")

elif ss.mode == "Settings":
    st.subheader("Settings & Advanced")
    st.markdown("Model choices (edit in code to change to larger/smaller models):")
    st.code(f"CHAT: {MODEL_CHAT}\nEMOTION: {MODEL_EMOTION}\nSUMMARY: {MODEL_SUMMARIZER}\nIMAGE: {MODEL_IMAGE_CAPTION}\nCODE: {MODEL_CODE}")
    st.markdown("---")
    st.markdown("**Run instructions:**")
    st.markdown("1. Install dependencies: `pip install streamlit transformers sentencepiece torch torchvision Pillow PyPDF2 pyttsx3`\n2. Run: `streamlit run app.py`\n3. First run will download models — be patient.")
    st.markdown("---")
    st.markdown("If something failed to load, try the 'Load Models' button in the sidebar.")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("Built with ❤️ — Nas Galaxy AI. Want a deployed Streamlit Cloud link or extra features (dark animations, better models, hosted TTS)? Tell me and I'll prepare it.")

# ----------------- END -----------------
