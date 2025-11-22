import streamlit as st
from transformers import pipeline
import PyPDF2
from PIL import Image
import base64
import io

# ---------------------------------------------------------
# ✨ Premium Billion-Dollar UI Theme
# ---------------------------------------------------------

st.set_page_config(
    page_title="Nas Galaxy AI",
    page_icon="✨",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0a0f24, #000000);
    color: white;
}
.sidebar .sidebar-content {
    background: rgba(255,255,255,0.05) !important;
}
.block-container {
    padding-top: 2rem;
}
.big-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff, #3a47d5);
    -webkit-background-clip: text;
    color: transparent;
}
.glass {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(12px);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ✨ Load AI Models
# ---------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_models():
    chat = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
    emo = pipeline("sentiment-analysis")
    summ = pipeline("summarization")
    code = pipeline("text-generation", model="deepseek-ai/deepseek-coder-1.3b")
    return chat, emo, summ, code

st.sidebar.title("⚙️ Nas Galaxy Settings")
if st.sidebar.button("Load AI Models"):
    chat_model, emotion_model, summary_model, code_model = load_models()
    st.sidebar.success("Models loaded successfully!")

# ---------------------------------------------------------
# ✨ Sidebar Mode Selection
# ---------------------------------------------------------

mode = st.sidebar.radio(
    "Select Mode",
    ["Chat", "Emotion AI", "Summarizer", "PDF Reader", "Code Assistant"]
)

st.markdown("<h1 class='big-title'>Nas Galaxy AI 🌌</h1>", unsafe_allow_html=True)
st.write("Your futuristic multi-AI Copilot — built with style and power.")

# ---------------------------------------------------------
# ✨ Chat Mode
# ---------------------------------------------------------

if mode == "Chat":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("💬 Chat with AI")

    user = st.text_area("You:", placeholder="Type your message...")

    if st.button("Send"):
        st.write("### 🤖 Nas Galaxy:")
        res = chat_model(user, max_length=350, temperature=0.7)[0]["generated_text"]
        st.write(res)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ✨ Emotion Detection
# ---------------------------------------------------------

elif mode == "Emotion AI":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("😊 Emotion AI")

    text = st.text_area("Enter text to analyze")

    if st.button("Analyze"):
        result = emotion_model(text)[0]
        st.success(f"Emotion: **{result['label']}** (score: {result['score']:.2f})")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ✨ Summarizer
# ---------------------------------------------------------

elif mode == "Summarizer":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📝 AI Summarizer")

    txt = st.text_area("Paste long text here")

    if st.button("Summarize"):
        sumr = summary_model(txt, max_length=150)[0]["summary_text"]
        st.success(sumr)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ✨ PDF Reader
# ---------------------------------------------------------

elif mode == "PDF Reader":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📄 PDF Text Extractor")

    pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf:
        reader = PyPDF2.PdfReader(pdf)
        out = ""
        for page in reader.pages:
            out += page.extract_text()

        st.text_area("Extracted Text", out, height=300)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ✨ Code Assistant (DeepSeek Style)
# ---------------------------------------------------------

elif mode == "Code Assistant":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("💻 DeepSeek Code Assistant")

    prompt = st.text_area("Ask the AI to write or fix code")

    if st.button("Generate Code"):
        result = code_model(prompt, max_length=300)[0]["generated_text"]
        st.code(result, language="python")

    st.markdown("</div>", unsafe_allow_html=True)
