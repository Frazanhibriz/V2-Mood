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
    /* Import Clean Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --accent: #4f46e5;
        --accent-soft: #eef2ff;
        --accent-strong: #4338ca;
        --surface: #ffffff;
        --surface-muted: #f9fafb;
        --border-subtle: #e5e7eb;
        --text-main: #111827;
        --text-soft: #6b7280;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-sizing: border-box;
    }
    
    /* Main Container */
    .stApp {
        background: #fafbfc;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Reduce gap between elements */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .app-header p {
        font-size: 0.95rem;
        color: var(--text-soft);
        font-weight: 400;
    }
    
    /* Step Indicator */
    .step-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.75rem;
        margin: 1.5rem 0 2rem 0;
    }
    
    .step {
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .step-circle {
        width: 32px;
        height: 32px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid var(--border-subtle);
        background: var(--surface);
        color: var(--text-soft);
    }
    
    .step-circle.active {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: var(--accent);
    }
    
    .step-circle.completed {
        border-color: var(--accent);
        background: var(--accent);
        color: #ffffff;
    }
    
    .step-circle.inactive {
        opacity: 0.6;
    }
    
    .step-line {
        width: 40px;
        height: 2px;
        background: var(--border-subtle);
        border-radius: 999px;
    }
    
    .step-line.completed {
        background: var(--accent);
    }
    
    /* Card Component */
    .card {
        background: var(--surface);
        border-radius: 16px;
        padding: 1.5rem 1.5rem 1.25rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        border: 1px solid #e5e7eb;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-size: 0.95rem;
        color: var(--text-soft);
        line-height: 1.6;
        margin-bottom: 0.25rem;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.75rem 1.75rem;
        font-size: 0.95rem;
        font-weight: 500;
        transition: background 0.18s ease, box-shadow 0.18s ease, transform 0.12s ease;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.20);
    }
    
    .stButton > button:hover {
        background: var(--accent-strong);
        box-shadow: 0 10px 28px rgba(79, 70, 229, 0.28);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.18);
    }
    
    /* Radio Buttons - Minimal Pill Design */
    .stRadio > label {
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--text-main);
        margin-bottom: 0.75rem;
    }
    
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: var(--surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 999px !important;
        padding: 0.5rem 1rem !important;
        cursor: pointer !important;
        transition: border 0.15s ease, background 0.15s ease, box-shadow 0.15s ease !important;
        font-size: 0.9rem !important;
        color: #4b5563 !important;
        display: flex !important;
        align-items: center !important;
        white-space: nowrap !important;
    }
    
    .stRadio > div > label:hover {
        border-color: var(--accent) !important;
        background: var(--surface-muted) !important;
    }
    
    .stRadio > div > label > div {
        color: #4b5563 !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
        box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.08);
        color: var(--accent) !important;
    }
    
    /* Success Message (Subtle) */
    .success-message {
        background: #f5f7ff;
        border-left: 3px solid var(--accent);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 1rem 0;
        text-align: left;
    }
    
    .success-message h3 {
        color: var(--text-main);
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    
    .success-message p {
        color: var(--text-soft);
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Ice Cream Result Card */
    .result-card {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        border-radius: 20px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0 1.25rem 0;
        color: white;
        box-shadow: 0 18px 40px rgba(79, 70, 229, 0.45);
    }
    
    .result-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
        letter-spacing: -0.01em;
    }
    
    .result-description {
        font-size: 0.95rem;
        opacity: 0.96;
        margin-bottom: 0.3rem;
    }
    
    .result-detail {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .result-profile {
        margin-top: 1.2rem;
        padding-top: 1.1rem;
        border-top: 1px solid rgba(255,255,255,0.25);
        font-size: 0.8rem;
        opacity: 0.9;
    }
    
    /* Stats Display */
    .stats-container {
        background: var(--surface);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid var(--border-subtle);
    }
    
    .stats-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    
    .stats-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--accent);
    }
    
    .stats-subtext {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.15rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: var(--accent);
    }
    
    /* Info Box */
    .info-box {
        background: #f5f7ff;
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
    }
    
    .info-box p {
        color: #374151;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Tips Section */
    .tips-container {
        background: var(--surface-muted);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        border: 1px solid #e5e7eb;
    }
    
    .tips-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.4rem;
    }
    
    .tips-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tips-list li {
        color: #6b7280;
        font-size: 0.85rem;
        padding: 0.25rem 0;
        padding-left: 1.1rem;
        position: relative;
    }
    
    .tips-list li:before {
        content: "•";
        color: var(--accent);
        font-weight: bold;
        position: absolute;
        left: 0;
        top: 0.15rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent);
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 1.5rem 0 1.25rem 0;
    }

    /* Camera placeholder styling */
    .stCamera > div {
        border-radius: 16px !important;
        border: 1px dashed #d1d5db !important;
        padding: 0.75rem !important;
        background: #f9fafb !important;
    }

    .stCamera label {
        font-size: 0.9rem !important;
        color: #4b5563 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Hide Streamlit Elements */
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


@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
    return face_cascade if not face_cascade.empty() else None


with st.spinner("Loading AI model..."):
    processor, model = load_emotion_model()
    ID2LABEL = model.config.id2label
    face_detector = load_face_detector()


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

def predict_emotion_from_frame(frame_bgr):
    if face_detector is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_roi = frame_bgr[y:y + h, x:x + w]
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = outputs.logits.softmax(dim=1)[0]
    idx = int(torch.argmax(probs).item())
    raw_label = ID2LABEL[idx].lower()
    confidence = float(probs[idx].item())
    category = MODEL_TO_CATEGORY.get(raw_label, "chill")
    
    return raw_label, category, confidence


def realtime_scan(duration_sec=8, fps=10, buffer_size=15):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera. Please check permissions.")
        return None, None, None

    frame_placeholder = st.empty()
    progress_placeholder = st.empty()
    stats_placeholder = st.empty()

    start_time = time.time()
    interval = 1.0 / fps
    emo_buffer = deque(maxlen=buffer_size)
    conf_buffer = deque(maxlen=buffer_size)
    last_raw = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        raw_label, category, conf = predict_emotion_from_frame(frame)
        emo_buffer.append(category)
        conf_buffer.append(conf)
        last_raw = raw_label

        if emo_buffer:
            counts = Counter(emo_buffer)
            stable_cat = counts.most_common(1)[0][0]
            stable_conf_vals = [c for c, cat in zip(conf_buffer, emo_buffer) if cat == stable_cat]
            stable_conf = sum(stable_conf_vals) / len(stable_conf_vals) if stable_conf_vals else conf
        else:
            stable_cat = category
            stable_conf = conf

        display = frame.copy()
        
        if face_detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x + w, y + h), (79, 70, 229), 2)

        frame_placeholder.image(display, channels="BGR", use_container_width=True)

        elapsed = time.time() - start_time
        progress = min(elapsed / duration_sec, 1.0)
        progress_placeholder.progress(progress, text=f"Scanning... {elapsed:.1f}s / {duration_sec}s")

        stats_placeholder.markdown(
            f"""
            <div class='stats-container'>
                <div class='stats-label'>Detected Mood</div>
                <div class='stats-value'>{CATEGORY_DISPLAY.get(stable_cat, stable_cat)}</div>
                <div class='stats-subtext'>Confidence: {stable_conf:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if elapsed >= duration_sec:
            break

    cap.release()

    if not emo_buffer:
        return None, None, None

    final_counts = Counter(emo_buffer)
    final_cat = final_counts.most_common(1)[0][0]
    final_conf_vals = [c for c, cat in zip(conf_buffer, emo_buffer) if cat == final_cat]
    final_conf = sum(final_conf_vals) / len(final_conf_vals) if final_conf_vals else 0.0

    return last_raw, final_cat, final_conf

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
    <div class='step'>
        <div class='step-circle {}'>1</div>
    </div>
    <div class='step-line {}'></div>
    <div class='step'>
        <div class='step-circle {}'>2</div>
    </div>
    <div class='step-line {}'></div>
    <div class='step'>
        <div class='step-circle {}'>3</div>
    </div>
    <div class='step-line {}'></div>
    <div class='step'>
        <div class='step-circle {}'>4</div>
    </div>
</div>
""".format(
    "completed" if st.session_state.step > 1 else "active" if st.session_state.step == 1 else "inactive",
    "completed" if st.session_state.step > 1 else "",
    "completed" if st.session_state.step > 2 else "active" if st.session_state.step == 2 else "inactive",
    "completed" if st.session_state.step > 2 else "",
    "completed" if st.session_state.step > 3 else "active" if st.session_state.step == 3 else "inactive",
    "completed" if st.session_state.step > 3 else "",
    "active" if st.session_state.step == 4 else "inactive"
)

st.markdown(step_html, unsafe_allow_html=True)


if st.session_state.step == 1:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 1: Mood Detection</div>
        <div class='card-description'>
            We'll use your camera to analyze your facial expression and detect your current mood.
            The scan takes about 8 seconds and uses advanced AI technology.
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
                <li>Stay 50–100 cm away</li>
                <li>Hold your expression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        img = st.camera_input("Start Mood Scan", key="scan_cam")
        if img is not None:
            pil_img = Image.open(img)
            inputs = processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=1)[0]
            idx = int(torch.argmax(probs).item())
            raw_label = ID2LABEL[idx].lower()
            confidence = float(probs[idx].item())
            cat = MODEL_TO_CATEGORY.get(raw_label, "chill")

            st.session_state.detected_cat = cat
            st.session_state.detected_conf = confidence
            st.session_state.final_cat = cat
            st.session_state.step = 2
                
            st.markdown(f"""
            <div class='success-message'>
                <h3>Scan complete</h3>
                <p>Detected mood: <strong>{CATEGORY_DISPLAY.get(cat, cat)}</strong> ({confidence:.0%} confidence)</p>
            </div>
            """, unsafe_allow_html=True)
                
            st.balloons()
            time.sleep(1)
            st.rerun()


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
    
    col1, col2 = st.columns([4, 1])
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
        flavor = st.radio(
            "flavor",
            ["Sweet", "Fresh", "Creamy", "Strong"],
            label_visibility="collapsed",
            key="flavor_radio"
        )
        st.session_state.flavor_pref = flavor.lower()
    
    with col2:
        st.markdown("**Energy level:**")
        energy = st.radio(
            "energy",
            ["Tired", "Normal", "Energetic"],
            index=1,
            label_visibility="collapsed",
            key="energy_radio"
        )
        st.session_state.energy_level = energy.lower()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
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
    mood_text = CATEGORY_DISPLAY.get(st.session_state.final_cat, "")
    
    st.markdown(f"""
    <div class='result-card'>
        <div class='result-emoji'>{ice['emoji']}</div>
        <div class='result-title'>{ice['name']}</div>
        <div class='result-description'>{ice['description']}</div>
        <div class='result-detail'>{ice['detail']}</div>
        <div class='result-profile'>
            <strong>Your Profile:</strong><br>
            Mood: {mood_text} • Flavor: {st.session_state.flavor_pref.title()} • Energy: {st.session_state.energy_level.title()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <p>✨ This recommendation is AI-powered but meant to be fun and interactive. Enjoy your treat.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Over", key="reset"):
        for key in ['step', 'detected_cat', 'detected_conf', 'final_cat', 'flavor_pref', 'energy_level']:
            if key == 'step':
                st.session_state[key] = 1
            else:
                st.session_state[key] = None
        st.rerun()


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 0.8rem; padding: 0.5rem 0 0.25rem 0;'>
    <p>MoodScoop v3.0</p>
</div>
""", unsafe_allow_html=True)
