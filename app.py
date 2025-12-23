import streamlit as st
import pandas as pd
import datetime
import time

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Life Revolution App", page_icon="🚀", layout="wide")

# ================= THEME ENGINE =================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

LIGHT = """
<style>
body {background: #f5f7fa !important;}
header {visibility: hidden;}
.block {background: rgba(255,255,255,0.85); border-radius: 20px; padding:20px; margin-bottom:20px;}
.bigtitle {font-size:40px; font-weight:bold; color:#000;}
.subtitle {font-size:18px; color:#333;}
</style>
"""

DARK = """
<style>
body {background: #121212 !important;}
header {visibility: hidden;}
.block {background: rgba(30,30,30,0.85); border-radius: 20px; padding:20px; margin-bottom:20px;}
.bigtitle {font-size:40px; font-weight:bold; color:#fff;}
.subtitle {font-size:18px; color:#ccc;}
</style>
"""

if st.session_state.theme == "light":
    st.markdown(LIGHT, unsafe_allow_html=True)
else:
    st.markdown(DARK, unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🚀 Life Revolution App")
    menu = st.radio("Navigate", [
        "🏠 Dashboard",
        "🎭 Mood Tracker",
        "📊 Mood Charts",
        "📝 Diary",
        "🤖 AI Chat (Coming Soon)",
        "🎤 Voice Assistant (Coming Soon)",
        "⚙️ Settings",
        "🧩 Other Features (Coming Soon)"
    ])

# ================= DASHBOARD =================
if menu == "🏠 Dashboard":
    st.markdown("<div class='bigtitle'>Welcome to Life Revolution App</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>All your life tracking & AI features in one place.</div>", unsafe_allow_html=True)
    st.markdown("<div class='block'>Explore the app using the sidebar!</div>", unsafe_allow_html=True)

# ================= MOOD TRACKER =================
elif menu == "🎭 Mood Tracker":
    st.markdown("<div class='bigtitle'>Mood Tracker</div>", unsafe_allow_html=True)
    st.markdown("<div class='block'>Rate your mood today (1 = worst, 10 = best):</div>", unsafe_allow_html=True)
    mood = st.slider("Your Mood:", 1, 10, 5)
    if st.button("Save Mood"):
        if "mood_data" not in st.session_state:
            st.session_state.mood_data = []
        st.session_state.mood_data.append({"date": datetime.date.today(), "mood": mood})
        st.success(f"Mood {mood} saved for today!")

# ================= MOOD CHARTS =================
elif menu == "📊 Mood Charts":
    st.markdown("<div class='bigtitle'>Mood Charts</div>", unsafe_allow_html=True)
    if "mood_data" in st.session_state and st.session_state.mood_data:
        df = pd.DataFrame(st.session_state.mood_data)
        df['date'] = pd.to_datetime(df['date'])
        st.line_chart(df.set_index('date')['mood'])
    else:
        st.info("No mood data yet. Go to Mood Tracker to add.")

# ================= DIARY =================
elif menu == "📝 Diary":
    st.markdown("<div class='bigtitle'>Diary</div>", unsafe_allow_html=True)
    entry = st.text_area("Write your diary entry:")
    if st.button("Save Entry"):
        if "diary_data" not in st.session_state:
            st.session_state.diary_data = []
        st.session_state.diary_data.append({"date": datetime.datetime.now(), "entry": entry})
        st.success("Diary entry saved!")
    if "diary_data" in st.session_state:
        for d in reversed(st.session_state.diary_data[-5:]):
            st.markdown(f"**{d['date'].strftime('%Y-%m-%d %H:%M')}**: {d['entry']}")

# ================= AI CHATPLACEHOLDER =================
elif menu == "🤖 AI Chat (Coming Soon)":
    st.markdown("<div class='bigtitle'>AI Chat (Coming Soon)</div>", unsafe_allow_html=True)
    st.info("AI features will be added soon!")

# ================= VOICE ASSISTANT =================
elif menu == "🎤 Voice Assistant (Coming Soon)":
    st.markdown("<div class='bigtitle'>Voice Assistant (Coming Soon)</div>", unsafe_allow_html=True)
    st.info("Voice Assistant will be added soon!")

# ================= SETTINGS =================
elif menu == "⚙️ Settings":
    st.markdown("<div class='bigtitle'>Settings</div>", unsafe_allow_html=True)
    st.button("Toggle Light/Dark Theme", on_click=toggle_theme)

# ================= OTHER FEATURES =================
elif menu == "🧩 Other Features (Coming Soon)":
    st.markdown("<div class='bigtitle'>Other Features (Coming Soon)</div>", unsafe_allow_html=True)
    st.info("More revolutionary features will be added soon!")

