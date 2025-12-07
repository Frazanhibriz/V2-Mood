import time
from collections import Counter, deque
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="MoodScoop - AI Ice Cream Recommender", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #fafbfc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 800px;
}

.element-container {
    margin-bottom: 0.5rem;
}

.app-header {
    text-align: center;
    margin-bottom: 2rem;
}

.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #1a202c;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}

.app-header p {
    font-size: 1rem;
    color: #718096;
    font-weight: 400;
}

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
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.3s ease;
}

.step-circle.active {
    background: #4f46e5;
    color: white;
}

.step-circle.completed {
    background: #10b981;
    color: white;
}

.step-circle.inactive {
    background: #e5e7eb;
    color: #9ca3af;
}

.step-line {
    width: 40px;
    height: 2px;
    background: #e5e7eb;
}

.step-line.completed {
    background: #10b981;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    transition: box-shadow 0.3s ease;
}

.card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1a202c;
    margin-bottom: 0.5rem;
}

.card-description {
    font-size: 0.95rem;
    color: #718096;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.stButton > button {
    width: 100%;
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.75rem;
    font-size: 0.95rem;
    font-weight: 500;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

.stButton > button:hover {
    background: #4338ca;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

.stRadio > label {
    font-size: 0.95rem;
    font-weight: 500;
    color: #1a202c;
    margin-bottom: 0.75rem;
}

.tips-container {
    background: #f9fafb;
    border-radius: 12px;
    padding: 1.25rem;
}

.tips-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
}

.tips-list li {
    color: #6b7280;
    font-size: 0.85rem;
    padding: 0.3rem 0;
    padding-left: 1.25rem;
    position: relative;
}

.tips-list li:before {
    content: "•";
    color: #4f46e5;
    font-weight: bold;
    position: absolute;
    left: 0;
}

.success-message {
    background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    margin: 1rem 0;
}

.result-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 1.5rem 0;
    color: white;
}

.result-emoji {
    font-size: 4rem;
    margin-bottom: 0.75rem;
}

.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
}

.badge {
    background: #ede9fe;
    color: #5b21b6;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
}

.divider {
    height: 1px;
    background: #e5e7eb;
    margin: 1.5rem 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

div[data-testid="stRadio"] label {
    background: white !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
}

div[data-testid="stRadio"] label span {
    font-size: 0.95rem !important;
    color: #1a1a1a !important;
}

div[data-testid="stRadio"] label[data-checked="true"] {
    border-color: #4f46e5 !important;
    background: #eff6ff !important;
}

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
    "happy_energetic": {"name": "Strawberry Rainbow Sprinkle","description": "Sweet and colorful to boost your happy mood","detail": "Strawberry ice cream with rainbow sprinkles","emoji": "🍓"},
    "chill": {"name": "Vanilla Matcha Glaze","description": "Soft and calm vibes for relaxation","detail": "Vanilla ice cream with matcha glaze topping","emoji": "🍵"},
    "sad": {"name": "Red Velvet Chocolate","description": "Comfort dessert to lift your spirits","detail": "Chocolate ice cream with red velvet crumbs","emoji": "🍫"},
    "worried_anxious": {"name": "Cookies & Cream","description": "Classic comfort to ease your mind","detail": "Vanilla ice cream with cookie crumbles","emoji": "🍪"},
    "mad_irritated": {"name": "Dark Chocolate Tiramisu","description": "Bold and intense for when you need strength","detail": "Rich chocolate ice cream with tiramisu glaze","emoji": "☕"},
}

def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face = frame[y:y+h, x:x+w]
    else:
        face = frame
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(1)[0]
    idx = int(torch.argmax(probs))
    label = ID2LABEL[idx].lower()
    cat = MODEL_TO_CATEGORY.get(label, "chill")
    conf = float(probs[idx])
    return label, cat, conf, faces

def realtime_scan(sec=8, fps=10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera.")
        return None, None, None
    buf = deque(maxlen=15)
    confs = deque(maxlen=15)
    start = time.time()
    box_frame = st.empty()
    prog = st.empty()
    stat = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        raw, cat, conf, faces = predict_emotion(frame)
        buf.append(cat)
        confs.append(conf)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(79,70,229),2)
        box_frame.image(frame, channels="BGR", use_container_width=True)
        elapsed = time.time() - start
        prog.progress(min(elapsed/sec,1.0))
        if elapsed >= sec:
            break
        time.sleep(1/fps)
    cap.release()
    if not buf:
        return None, None, None
    final = Counter(buf).most_common(1)[0][0]
    final_conf = sum([c for c,cat in zip(confs,buf) if cat==final]) / len(buf)
    return raw, final, final_conf

if "step" not in st.session_state:
    st.session_state.step = 1

if "final_cat" not in st.session_state:
    st.session_state.final_cat = None

if "flavor_pref" not in st.session_state:
    st.session_state.flavor_pref = None

if "energy_level" not in st.session_state:
    st.session_state.energy_level = None

st.markdown("<div class='app-header'><h1>🍦 MoodScoop</h1><p>AI-powered ice cream recommendations based on your mood</p></div>", unsafe_allow_html=True)

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
    "completed" if st.session_state.step>1 else "active",
    "completed" if st.session_state.step>1 else "",
    "completed" if st.session_state.step>2 else "active" if st.session_state.step==2 else "inactive",
    "completed" if st.session_state.step>2 else "",
    "completed" if st.session_state.step>3 else "active" if st.session_state.step==3 else "inactive",
    "completed" if st.session_state.step>3 else "",
    "active" if st.session_state.step==4 else "inactive"
)

st.markdown(step_html, unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 1: Mood Detection</div>
        <div class='card-description'>We'll use your camera to analyze your facial expression and detect your current mood.</div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col2:
        st.markdown("""
        <div class='tips-container'>
            <div class='tips-title'>Tips for best results:</div>
            <ul class='tips-list'>
                <li>Face the camera directly</li>
                <li>Ensure good lighting</li>
                <li>Stay 50-100cm away</li>
                <li>Hold your expression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col1:
        if st.button("Start Mood Scan"):
            with st.spinner("Scanning..."):
                raw, cat, conf = realtime_scan()
            if cat is None:
                st.error("Scan failed. Try again.")
            else:
                st.session_state.final_cat = cat
                st.session_state.detect_conf = conf
                st.session_state.step = 2
                st.rerun()

elif st.session_state.step == 2:
    st.markdown(f"""
    <div class='card'>
        <div class='card-title'>Step 2: Confirm Your Mood</div>
        <div class='card-description'>We detected your mood as <span class='badge'>{CATEGORY_DISPLAY.get(st.session_state.final_cat)}</span></div>
    </div>
    """, unsafe_allow_html=True)
    options = list(CATEGORY_DISPLAY.values())
    keys = list(CATEGORY_DISPLAY.keys())
    idx = keys.index(st.session_state.final_cat)
    chosen = st.radio("Select your current mood:", options, index=idx, key="mood_radio")
    st.session_state.final_cat = keys[options.index(chosen)]
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Continue →"):
            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 3: Your Preferences</div>
        <div class='card-description'>Help us personalize your recommendation by sharing your current preferences.</div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        flavor = st.radio("Flavor", ["Sweet","Fresh","Creamy","Strong"], key="flavor_radio")
        st.session_state.flavor_pref = flavor.lower()
    with col2:
        energy = st.radio("Energy", ["Tired","Normal","Energetic"], key="energy_radio")
        st.session_state.energy_level = energy.lower()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Get Recommendation →"):
            st.session_state.step = 4
            st.rerun()

elif st.session_state.step == 4:
    ice = ICE_CREAM_MAP[st.session_state.final_cat]
    mood = CATEGORY_DISPLAY[st.session_state.final_cat]
    st.markdown(f"""
    <div class='result-card'>
        <div class='result-emoji'>{ice['emoji']}</div>
        <div class='result-title'>{ice['name']}</div>
        <div class='result-description'>{ice['description']}</div>
        <div class='result-detail'>{ice['detail']}</div>
        <div class='result-profile'>
            <strong>Your Profile:</strong><br>
            Mood: {mood} • Flavor: {st.session_state.flavor_pref.title()} • Energy: {st.session_state.energy_level.title()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Over"):
        st.session_state.step = 1
        st.session_state.final_cat = None
        st.session_state.energy_level = None
        st.session_state.flavor_pref = None
        st.rerun()

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#9ca3af;font-size:0.875rem;'>MoodScoop v3.0</div>", unsafe_allow_html=True)
