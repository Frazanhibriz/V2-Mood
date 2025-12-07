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
* {font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;}
.stApp {background: #fafbfc;}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 800px;}
.app-header {text-align: center; margin-bottom: 2rem;}
.app-header h1 {font-size: 2rem; font-weight: 700; color: #1a202c; margin-bottom: 0.25rem;}
.app-header p {font-size: 1rem; color: #718096;}
.step-indicator {display: flex; justify-content: center; gap: 0.75rem; margin: 1.5rem 0 2rem;}
.step-circle {width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.85rem;}
.step-circle.active {background: #4f46e5; color: white;}
.step-circle.completed {background: #10b981; color: white;}
.step-circle.inactive {background: #e5e7eb; color: #9ca3af;}
.step-line {width: 40px; height: 2px; background: #e5e7eb;}
.step-line.completed {background: #10b981;}
.card {background: white; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
.card-title {font-size: 1.15rem; font-weight: 600; color: #1a202c;}
.card-description {font-size: 0.95rem; color: #718096; line-height: 1.5;}
.stButton > button {width: 100%; background: #4f46e5; color: white; border-radius: 12px; padding: 0.75rem 1.75rem; font-size: 0.95rem;}
.stRadio > div {gap: 0.5rem;}
.stRadio > div > label {background: white!important; border: 2px solid #e5e7eb!important; border-radius: 12px!important; padding: 0.75rem 1rem!important;}
.stRadio > div > label[data-checked="true"] {border-color: #4f46e5!important; background: #eff6ff!important;}
.success-message {background: linear-gradient(135deg,#d4fc79,#96e6a1); border-radius: 12px; padding: 1.25rem; text-align: center;}
.success-message h3 {color: #065f46; font-size: 1.15rem; font-weight: 600;}
.success-message p {color: #047857; font-size: 0.95rem;}
.result-card {background: linear-gradient(135deg,#667eea,#764ba2); border-radius: 20px; padding: 2rem 1.5rem; text-align: center; color: white;}
.result-emoji {font-size: 4rem;}
.result-title {font-size: 1.75rem; font-weight: 700;}
.result-description {font-size: 1rem; opacity: .95;}
.result-detail {font-size: .9rem; opacity: .85;}
.result-profile {margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.3);}
.info-box {background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 0.75rem 1rem;}
.info-box p {color: #1e40af; font-size: 0.9rem;}
.divider {height: 1px; background: #e5e7eb; margin: 1.5rem 0;}
#MainMenu, footer, .stDeployButton {visibility: hidden;}
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
    "happy_energetic": {"name": "Strawberry Rainbow Sprinkle", "description": "Sweet and colorful to boost your happy mood", "detail": "Strawberry ice cream with rainbow sprinkles", "emoji": "🍓"},
    "chill": {"name": "Vanilla Matcha Glaze", "description": "Soft and calm vibes for relaxation", "detail": "Vanilla ice cream with matcha glaze topping", "emoji": "🍵"},
    "sad": {"name": "Red Velvet Chocolate", "description": "Comfort dessert to lift your spirits", "detail": "Chocolate ice cream with red velvet crumbs", "emoji": "🍫"},
    "worried_anxious": {"name": "Cookies & Cream", "description": "Classic comfort to ease your mind", "detail": "Vanilla ice cream with cookie crumbles", "emoji": "🍪"},
    "mad_irritated": {"name": "Dark Chocolate Tiramisu", "description": "Bold and intense for when you need strength", "detail": "Rich chocolate ice cream with tiramisu glaze", "emoji": "☕"},
}

def predict_emotion_from_image(img):
    pil_img = Image.open(img)
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)[0]
    idx = int(torch.argmax(probs).item())
    raw_label = ID2LABEL[idx].lower()
    confidence = float(probs[idx].item())
    category = MODEL_TO_CATEGORY.get(raw_label, "chill")
    return raw_label, category, confidence

if "step" not in st.session_state:
    st.session_state.step = 1
if "final_cat" not in st.session_state:
    st.session_state.final_cat = None

st.markdown("""
<div class='app-header'>
    <h1>🍦 MoodScoop</h1>
    <p>AI-powered ice cream recommendations based on your mood</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 1: Mood Detection</div>
        <div class='card-description'>
            We'll analyze your facial expression using a single captured photo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    img = st.camera_input("Capture your mood")

    if img:
        raw, cat, conf = predict_emotion_from_image(img)
        st.session_state.final_cat = cat
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    detected = CATEGORY_DISPLAY.get(st.session_state.final_cat, "")
    st.markdown(f"""
    <div class='card'>
        <div class='card-title'>Step 2: Confirm Your Mood</div>
        <div class='card-description'>
            We detected your mood as <span class='badge'>{detected}</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    options = list(CATEGORY_DISPLAY.values())
    default_idx = list(CATEGORY_DISPLAY.keys()).index(st.session_state.final_cat)
    chosen = st.radio("Select your current mood:", options, index=default_idx)

    chosen_key = list(CATEGORY_DISPLAY.keys())[options.index(chosen)]
    st.session_state.final_cat = chosen_key

    if st.button("Continue →"):
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 3: Your Preferences</div>
    </div>
    """, unsafe_allow_html=True)

    flavor = st.radio("Flavor preference:", ["Sweet", "Fresh", "Creamy", "Strong"])
    energy = st.radio("Energy level:", ["Tired", "Normal", "Energetic"], index=1)

    st.session_state.flavor_pref = flavor
    st.session_state.energy_level = energy

    if st.button("Get Recommendation →"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    ice = ICE_CREAM_MAP.get(st.session_state.final_cat, ICE_CREAM_MAP["chill"])
    mood_text = CATEGORY_DISPLAY.get(st.session_state.final_cat, "")

    st.markdown(f"""
    <div class='result-card'>
        <div class='result-emoji'>{ice['emoji']}</div>
        <div class='result-title'>{ice['name']}</div>
        <div class='result-description'>{ice['description']}</div>
        <div class='result-detail'>{ice['detail']}</div>
        <div class='result-profile'>
            <strong>Your Profile:</strong><br>
            Mood: {mood_text} • Flavor: {st.session_state.flavor_pref} • Energy: {st.session_state.energy_level}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Over"):
        st.session_state.step = 1
        st.rerun()
