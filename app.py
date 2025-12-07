import time
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(
    page_title="MoodScoop - AI Ice Cream Recommender",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: #fafbfc; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 800px; }
    .element-container { margin-bottom: 0.5rem; }
    .app-header { text-align: center; margin-bottom: 2rem; }
    .app-header h1 { font-size: 2rem; font-weight: 700; color: #1a202c; margin-bottom: 0.25rem; letter-spacing: -0.02em; }
    .app-header p { font-size: 1rem; color: #718096; font-weight: 400; }
    .step-indicator { display: flex; justify-content: center; align-items: center; gap: 0.75rem; margin: 1.5rem 0 2rem 0; }
    .step { display: flex; align-items: center; gap: 0.4rem; }
    .step-circle { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.85rem; transition: all 0.3s ease; }
    .step-circle.active { background: #4f46e5; color: white; }
    .step-circle.completed { background: #10b981; color: white; }
    .step-circle.inactive { background: #e5e7eb; color: #9ca3af; }
    .step-line { width: 40px; height: 2px; background: #e5e7eb; }
    .step-line.completed { background: #10b981; }
    .card { background: white; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: box-shadow 0.3s ease; }
    .card-title { font-size: 1.15rem; font-weight: 600; color: #1a202c; margin-bottom: 0.5rem; }
    .card-description { font-size: 0.95rem; color: #718096; margin-bottom: 1rem; line-height: 1.5; }
    .stButton > button { width: 100%; background: #4f46e5; color: white; border: none; border-radius: 12px; padding: 0.75rem 1.75rem; font-size: 0.95rem; font-weight: 500; }
    .stRadio > div > label { background: white !important; border: 2px solid #e5e7eb !important; border-radius: 12px !important; padding: 0.75rem 1rem !important; font-size: 0.95rem !important; }
    .stRadio > div > label[data-checked="true"] { border-color: #4f46e5 !important; background: #eff6ff !important; }
    .success-message { background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); border-radius: 12px; padding: 1.25rem; text-align: center; margin: 1rem 0; }
    .success-message h3 { color: #065f46; font-size: 1.15rem; font-weight: 600; margin-bottom: 0.4rem; }
    .success-message p { color: #047857; font-size: 0.95rem; margin: 0; }
    .result-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 2rem 1.5rem; text-align: center; margin: 1.5rem 0; color: white; }
    .result-emoji { font-size: 4rem; margin-bottom: 0.75rem; }
    .result-title { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.75rem; }
    .result-description { font-size: 1rem; opacity: 0.95; margin-bottom: 0.5rem; }
    .result-detail { font-size: 0.9rem; opacity: 0.85; }
    .result-profile { margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.3); font-size: 0.85rem; opacity: 0.9; }
    .info-box { background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.75rem 0; }
    .tips-container { background: #f9fafb; border-radius: 12px; padding: 1.25rem; }
    .divider { height: 1px; background: #e5e7eb; margin: 1.5rem 0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_emotion_model():
    processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
    model.eval()
    return processor, model

processor, model = load_emotion_model()
ID2LABEL = model.config.id2label

MODEL_TO_CATEGORY = {
    "happy": "happy_energetic",
    "sad": "sad",
    "angry": "mad_irritated",
    "fear": "worried_anxious",
    "surprise": "happy_energetic",
    "disgust": "mad_irritated",
    "neutral": "chill",
}

CATEGORY_DISPLAY = {
    "happy_energetic": "Happy & Energetic",
    "chill": "Chill & Relaxed",
    "sad": "Sad & Down",
    "worried_anxious": "Worried & Anxious",
    "mad_irritated": "Frustrated & Irritated",
}

ICE_CREAM_MAP = {
    "happy_energetic": {"name": "Strawberry Rainbow Sprinkle","description": "Sweet and colorful to boost your happy mood","detail": "Strawberry ice cream with rainbow sprinkles","emoji": "🍓"},
    "chill": {"name": "Vanilla Matcha Glaze","description": "Soft and calm vibes for relaxation","detail": "Vanilla ice cream with matcha glaze topping","emoji": "🍵"},
    "sad": {"name": "Red Velvet Chocolate","description": "Comfort dessert to lift your spirits","detail": "Chocolate ice cream with red velvet crumbs","emoji": "🍫"},
    "worried_anxious": {"name": "Cookies & Cream","description": "Classic comfort to ease your mind","detail": "Vanilla ice cream with cookie crumbles","emoji": "🍪"},
    "mad_irritated": {"name": "Dark Chocolate Tiramisu","description": "Bold and intense for when you need strength","detail": "Rich chocolate ice cream with tiramisu glaze","emoji": "☕"},
}

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'detected_cat' not in st.session_state:
    st.session_state.detected_cat = None
if 'detected_conf' not in st.session_state:
    st.session_state.detected_conf = None
if 'final_cat' not in st.session_state:
    st.session_state.final_cat = None
if 'flavor_pref' not in st.session_state:
    st.session_state.flavor_pref = None
if 'energy_level' not in st.session_state:
    st.session_state.energy_level = None

st.markdown("""
<div class='app-header'>
    <h1>🍦 MoodScoop</h1>
    <p>AI-powered ice cream recommendations based on your mood</p>
</div>
""", unsafe_allow_html=True)

step_html = """
<div class='step-indicator'>
    <div class='step'><div class='step-circle {}'>1</div></div>
    <div class='step-line {}'></div>
    <div class='step'><div class='step-circle {}'>2</div></div>
    <div class='step-line {}'></div>
    <div class='step'><div class='step-circle {}'>3</div></div>
    <div class='step-line {}'></div>
    <div class='step'><div class='step-circle {}'>4</div></div>
</div>
""".format(
    "completed" if st.session_state.step > 1 else "active",
    "completed" if st.session_state.step > 1 else "",
    "completed" if st.session_state.step > 2 else "active",
    "completed" if st.session_state.step > 2 else "",
    "completed" if st.session_state.step > 3 else "active",
    "completed" if st.session_state.step > 3 else "",
    "active" if st.session_state.step == 4 else "inactive"
)

st.markdown(step_html, unsafe_allow_html=True)

# =====================================================================
# ======================== STEP 1 (CAPTURE ONLY) =======================
# =====================================================================

if st.session_state.step == 1:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 1: Mood Detection</div>
        <div class='card-description'>
            We'll use your camera to analyze your facial expression and detect your current mood.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("""
        <div class='tips-container'>
            <div class='tips-title'>Tips for best results:</div>
            <ul class='tips-list'>
                <li>Face the camera directly</li>
                <li>Ensure good lighting</li>
                <li>Stay 50–100cm away</li>
                <li>Hold your expression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        img = st.camera_input("Take a photo")

        if img is not None:
            pil = Image.open(img)
            inputs = processor(images=pil, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            probs = outputs.logits.softmax(dim=1)[0]
            idx = int(torch.argmax(probs).item())
            raw = ID2LABEL[idx].lower()
            conf = float(probs[idx].item())
            cat = MODEL_TO_CATEGORY.get(raw, "chill")

            st.session_state.detected_cat = cat
            st.session_state.detected_conf = conf
            st.session_state.final_cat = cat
            st.session_state.step = 2

            st.markdown(f"""
            <div class='success-message'>
                <h3>✓ Scan Complete!</h3>
                <p>Detected mood: <strong>{CATEGORY_DISPLAY.get(cat, cat)}</strong> ({conf:.0%} confidence)</p>
            </div>
            """, unsafe_allow_html=True)

            st.balloons()
            time.sleep(1)
            st.rerun()

# =====================================================================
# ========================== STEP 2–4 SAMA =============================
# =====================================================================

elif st.session_state.step == 2:
    st.markdown(f"""
    <div class='card'>
        <div class='card-title'>Step 2: Confirm Your Mood</div>
        <div class='card-description'>
            We detected your mood as <span class='badge'>{CATEGORY_DISPLAY.get(st.session_state.detected_cat, '')}</span>
            <br><br>
            Please confirm or adjust to ensure the best recommendation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    options = list(CATEGORY_DISPLAY.values())
    default_idx = list(CATEGORY_DISPLAY.keys()).index(st.session_state.detected_cat)

    chosen = st.radio("Select your current mood:", options, index=default_idx, key="mood_radio")
    chosen_key = list(CATEGORY_DISPLAY.keys())[list(CATEGORY_DISPLAY.values()).index(chosen)]
    st.session_state.final_cat = chosen_key

    st.markdown("""
    <div class='info-box'>
        <p>💡 AI provides an initial suggestion, but you have the final say. Choose what feels right for you.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="back_2"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Continue →", key="next_2"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 3: Your Preferences</div>
        <div class='card-description'>
            Help us personalize your recommendation by sharing your current preferences.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Flavor preference:**")
        flavor = st.radio("flavor", ["Sweet", "Fresh", "Creamy", "Strong"], key="flavor_radio", label_visibility="collapsed")
        st.session_state.flavor_pref = flavor.lower()

    with col2:
        st.markdown("**Energy level:**")
        energy = st.radio("energy", ["Tired", "Normal", "Energetic"], index=1, key="energy_radio", label_visibility="collapsed")
        st.session_state.energy_level = energy.lower()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="back_3"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Get Recommendation →", key="next_3"):
            st.session_state.step = 4
            st.rerun()

elif st.session_state.step == 4:
    ice = ICE_CREAM_MAP.get(st.session_state.final_cat, ICE_CREAM_MAP["chill"])

    st.markdown(f"""
    <div class='result-card'>
        <div class='result-emoji'>{ice['emoji']}</div>
        <div class='result-title'>{ice['name']}</div>
        <div class='result-description'>{ice['description']}</div>
        <div class='result-detail'>{ice['detail']}</div>
        <div class='result-profile'>
            <strong>Your Profile:</strong><br>
            Mood: {CATEGORY_DISPLAY.get(st.session_state.final_cat, "")} • Flavor: {st.session_state.flavor_pref.title()} • Energy: {st.session_state.energy_level.title()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <p>✨ This recommendation is AI-powered but meant to be fun and interactive. Enjoy your treat!</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Over", key="reset"):
        st.session_state.step = 1
        st.session_state.detected_cat = None
        st.session_state.detected_conf = None
        st.session_state.final_cat = None
        st.session_state.flavor_pref = None
        st.session_state.energy_level = None
        st.rerun()

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#9ca3af;font-size:0.875rem;padding:1rem 0;'>MoodScoop v3.0</div>", unsafe_allow_html=True)
