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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

* {
    font-family: 'DM Sans', sans-serif;
    margin: 0;
    padding: 0;
}

.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafb 100%);
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 680px;
}

/* Header */
.app-header {
    text-align: center;
    margin-bottom: 3rem;
}

.app-logo {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.08));
}

.app-title {
    font-size: 2.25rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.app-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    font-weight: 400;
}

/* Progress Steps */
.progress-container {
    margin: 2.5rem 0;
}

.progress-bar {
    width: 100%;
    height: 4px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #06b6d4 0%, #0891b2 100%);
    transition: width 0.4s ease;
}

.progress-text {
    text-align: center;
    font-size: 0.875rem;
    color: #64748b;
    font-weight: 500;
}

/* Main Content Card */
.content-card {
    background: white;
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 10px 40px rgba(0,0,0,0.03);
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}

.section-description {
    font-size: 1rem;
    color: #64748b;
    line-height: 1.6;
    margin-bottom: 2rem;
}

/* Camera Input */
.camera-container {
    margin: 1.5rem 0;
}

/* Radio Buttons */
.stRadio > div {
    gap: 0.75rem;
}

.stRadio > div > label {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.stRadio > div > label:hover {
    border-color: #cbd5e1 !important;
    background: #f1f5f9 !important;
}

.stRadio > div > label[data-checked="true"] {
    border-color: #06b6d4 !important;
    background: #ecfeff !important;
    font-weight: 500 !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.875rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(6, 182, 212, 0.2);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Detected Badge */
.mood-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%);
    color: #0e7490;
    padding: 0.5rem 1.25rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 1rem 0;
    border: 1px solid #a5f3fc;
}

/* Info Box */
.info-box {
    background: #fffbeb;
    border-left: 3px solid #fbbf24;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 1.5rem 0;
}

.info-box p {
    color: #92400e;
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0;
}

/* Result Card */
.result-container {
    text-align: center;
    padding: 1rem 0;
}

.result-emoji-container {
    width: 120px;
    height: 120px;
    margin: 0 auto 1.5rem;
    background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4rem;
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.15);
}

.result-title {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.75rem;
    letter-spacing: -0.02em;
}

.result-description {
    font-size: 1.125rem;
    color: #64748b;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}

.result-detail {
    font-size: 1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

.result-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    margin: 2rem 0;
}

.profile-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}

.profile-item {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.profile-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.profile-value {
    font-size: 1rem;
    color: #0f172a;
    font-weight: 600;
}

/* Secondary Button */
.secondary-button {
    margin-top: 1rem;
}

.secondary-button button {
    background: white !important;
    color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

.secondary-button button:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
}

/* Hide Streamlit Elements */
#MainMenu, footer, .stDeployButton {
    visibility: hidden;
}

/* Camera Input Styling */
.stCameraInput > label {
    display: none;
}

/* Responsive */
@media (max-width: 640px) {
    .content-card {
        padding: 1.75rem;
    }
    
    .profile-grid {
        grid-template-columns: 1fr;
        gap: 0.75rem;
    }
    
    .app-title {
        font-size: 1.875rem;
    }
    
    .result-title {
        font-size: 1.625rem;
    }
}
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
    "happy_energetic": {
        "name": "Strawberry Rainbow Sprinkle",
        "description": "Sweet and colorful to boost your happy mood",
        "detail": "Strawberry ice cream with rainbow sprinkles",
        "emoji": "🍓"
    },
    "chill": {
        "name": "Vanilla Matcha Glaze",
        "description": "Soft and calm vibes for relaxation",
        "detail": "Vanilla ice cream with matcha glaze topping",
        "emoji": "🍵"
    },
    "sad": {
        "name": "Red Velvet Chocolate",
        "description": "Comfort dessert to lift your spirits",
        "detail": "Chocolate ice cream with red velvet crumbs",
        "emoji": "🍫"
    },
    "worried_anxious": {
        "name": "Cookies & Cream",
        "description": "Classic comfort to ease your mind",
        "detail": "Vanilla ice cream with cookie crumbles",
        "emoji": "🍪"
    },
    "mad_irritated": {
        "name": "Dark Chocolate Tiramisu",
        "description": "Bold and intense for when you need strength",
        "detail": "Rich chocolate ice cream with tiramisu glaze",
        "emoji": "☕"
    },
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

# Header
st.markdown("""
<div class='app-header'>
    <div class='app-logo'>🍦</div>
    <div class='app-title'>MoodScoop</div>
    <div class='app-subtitle'>Your mood-based ice cream companion</div>
</div>
""", unsafe_allow_html=True)

# Progress Indicator
progress_percent = (st.session_state.step / 4) * 100
st.markdown(f"""
<div class='progress-container'>
    <div class='progress-bar'>
        <div class='progress-fill' style='width: {progress_percent}%'></div>
    </div>
    <div class='progress-text'>Step {st.session_state.step} of 4</div>
</div>
""", unsafe_allow_html=True)

# Step 1: Mood Detection
if st.session_state.step == 1:
    st.markdown("""
    <div class='content-card'>
        <div class='section-title'>Capture Your Mood</div>
        <div class='section-description'>
            Let's start by taking a quick photo. Our AI will analyze your facial expression to understand how you're feeling right now.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='camera-container'>", unsafe_allow_html=True)
    img = st.camera_input("Take a photo", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if img:
        raw, cat, conf = predict_emotion_from_image(img)
        st.session_state.final_cat = cat
        st.session_state.step = 2
        st.rerun()

# Step 2: Confirm Mood
elif st.session_state.step == 2:
    detected = CATEGORY_DISPLAY.get(st.session_state.final_cat, "")
    
    st.markdown(f"""
    <div class='content-card'>
        <div class='section-title'>Verify Your Mood</div>
        <div class='section-description'>
            We detected your mood as <span class='mood-badge'>{detected}</span>
        </div>
        <div class='section-description'>
            Does this feel right? You can adjust it below if needed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    options = list(CATEGORY_DISPLAY.values())
    default_idx = list(CATEGORY_DISPLAY.keys()).index(st.session_state.final_cat)
    chosen = st.radio("Your current mood:", options, index=default_idx, label_visibility="collapsed")

    chosen_key = list(CATEGORY_DISPLAY.keys())[options.index(chosen)]
    st.session_state.final_cat = chosen_key

    if st.button("Continue"):
        st.session_state.step = 3
        st.rerun()

# Step 3: Preferences
elif st.session_state.step == 3:
    st.markdown("""
    <div class='content-card'>
        <div class='section-title'>Tell Us More</div>
        <div class='section-description'>
            A few quick questions to personalize your recommendation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    flavor = st.radio("What flavor profile appeals to you right now?", 
                     ["Sweet", "Fresh", "Creamy", "Strong"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 2rem;'>", unsafe_allow_html=True)
    energy = st.radio("How's your energy level?", 
                     ["Tired", "Normal", "Energetic"], 
                     index=1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.flavor_pref = flavor
    st.session_state.energy_level = energy

    if st.button("Get My Recommendation"):
        st.session_state.step = 4
        st.rerun()

# Step 4: Results
elif st.session_state.step == 4:
    ice = ICE_CREAM_MAP.get(st.session_state.final_cat, ICE_CREAM_MAP["chill"])
    mood_text = CATEGORY_DISPLAY.get(st.session_state.final_cat, "")

    st.markdown(f"""
    <div class='content-card'>
        <div class='result-container'>
            <div class='result-emoji-container'>
                {ice['emoji']}
            </div>
            <div class='result-title'>{ice['name']}</div>
            <div class='result-description'>{ice['description']}</div>
            <div class='result-detail'>{ice['detail']}</div>
            
            <div class='result-divider'></div>
            
            <div class='profile-grid'>
                <div class='profile-item'>
                    <div class='profile-label'>Mood</div>
                    <div class='profile-value'>{mood_text}</div>
                </div>
                <div class='profile-item'>
                    <div class='profile-label'>Flavor</div>
                    <div class='profile-value'>{st.session_state.flavor_pref}</div>
                </div>
                <div class='profile-item'>
                    <div class='profile-label'>Energy</div>
                    <div class='profile-value'>{st.session_state.energy_level}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='info-box'><p>💡 <strong>Pro tip:</strong> Your mood affects taste perception! The perfect scoop matches both your emotional state and flavor preferences.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='secondary-button'>", unsafe_allow_html=True)
    if st.button("Try Again"):
        st.session_state.step = 1
        st.session_state.final_cat = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
