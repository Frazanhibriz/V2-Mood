import time
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageEnhance
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
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container */
    .stApp {
        background:
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
        margin-bottom: 2rem;
    }
    
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color:
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .app-header p {
        font-size: 1rem;
        color:
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
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s ease;
    }
    
    .step-circle.active {
        background:
        color: white;
    }
    
    .step-circle.completed {
        background:
        color: white;
    }
    
    .step-circle.inactive {
        background:
        color:
    }
    
    .step-line {
        width: 40px;
        height: 2px;
        background:
    }
    
    .step-line.completed {
        background:
    }
    
    /* Card Component */
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
        color:
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-size: 0.95rem;
        color:
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background:
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
        background:
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Radio Buttons - Minimal Design */
    .stRadio > label {
        font-size: 0.95rem;
        font-weight: 500;
        color:
        margin-bottom: 0.75rem;
    }
    
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: white !important;
        border: 2px solid
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
        color:
        display: flex !important;
        align-items: center !important;
    }
    
    .stRadio > div > label:hover {
        border-color:
        background:
    }
    
    .stRadio > div > label > div {
        color:
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color:
        background:
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(135deg,
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .success-message h3 {
        color:
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    
    .success-message p {
        color:
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Ice Cream Result Card */
    .result-card {
        background: linear-gradient(135deg,
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
    
    .result-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .result-description {
        font-size: 1rem;
        opacity: 0.95;
        margin-bottom: 0.5rem;
    }
    
    .result-detail {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    
    .result-profile {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.3);
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    /* Stats Display */
    .stats-container {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 2px solid
    }
    
    .stats-label {
        font-size: 0.8rem;
        color:
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    
    .stats-value {
        font-size: 1.5rem;
        font-weight: 700;
        color:
    }
    
    .stats-subtext {
        font-size: 0.85rem;
        color:
        margin-top: 0.25rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background:
    }
    
    /* Info Box */
    .info-box {
        background:
        border-left: 4px solid
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
    }
    
    .info-box p {
        color:
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Warning Box */
    .warning-box {
        background:
        border-left: 4px solid
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
    }
    
    .warning-box p {
        color:
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Tips Section */
    .tips-container {
        background:
        border-radius: 12px;
        padding: 1.25rem;
    }
    
    .tips-title {
        font-size: 0.9rem;
        font-weight: 600;
        color:
        margin-bottom: 0.5rem;
    }
    
    .tips-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tips-list li {
        color:
        font-size: 0.85rem;
        padding: 0.3rem 0;
        padding-left: 1.25rem;
        position: relative;
    }
    
    .tips-list li:before {
        content: "•";
        color:
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background:
        color:
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background:
        margin: 1.5rem 0;
    }
    
    /* Hide Streamlit Elements */
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Smooth Transitions */
    * {
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


def enhance_face_image(image_rgb):
    """
    Enhanced preprocessing untuk better emotion detection:
    - Normalize brightness dengan CLAHE
    - Enhance contrast
    - Reduce noise
    """
    pil_img = Image.fromarray(image_rgb)
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    img_array = np.array(pil_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    for i in range(3):
        img_array[:,:,i] = clahe.apply(img_array[:,:,i])
    
    return Image.fromarray(img_array)


def predict_emotion_robust(frame_bgr, processor, model, face_detector, ID2LABEL):
    """
    IMPROVED emotion prediction dengan:
    - Better face detection dengan margin
    - Enhanced preprocessing
    - Multi-angle testing (original + flipped)
    - Stricter confidence thresholds
    - Smart fallback logic
    """
    
    if face_detector is not None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            
            margin = int(0.2 * w)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame_bgr.shape[1], x + w + margin)
            y2 = min(frame_bgr.shape[0], y + h + margin)
            
            face_roi = frame_bgr[y1:y2, x1:x2]
            
            if face_roi.shape[0] < 120 or face_roi.shape[1] < 120:
                face_roi = cv2.resize(face_roi, (150, 150))
            
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        else:
            h, w = frame_bgr.shape[:2]
            center_x, center_y = w // 2, h // 2
            crop_size = min(w, h) // 2
            
            x1 = max(0, center_x - crop_size)
            y1 = max(0, center_y - crop_size)
            x2 = min(w, center_x + crop_size)
            y2 = min(h, center_y + crop_size)
            
            face_roi = frame_bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    pil_img = enhance_face_image(rgb)
    
    predictions = []
    
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)[0]
    predictions.append(probs)
    
    flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    inputs_flip = processor(images=flipped, return_tensors="pt")
    with torch.no_grad():
        outputs_flip = model(**inputs_flip)
    probs_flip = outputs_flip.logits.softmax(dim=1)[0]
    predictions.append(probs_flip)
    
    avg_probs = torch.stack(predictions).mean(dim=0)
    
    top_vals, top_idx = torch.topk(avg_probs, k=3)
    p1, p2, p3 = [float(v.item()) for v in top_vals]
    i1, i2, i3 = [int(idx.item()) for idx in top_idx]
    
    raw_label1 = ID2LABEL[i1].lower()
    raw_label2 = ID2LABEL[i2].lower()
    
    MIN_CONF = 0.65
    MIN_MARGIN = 0.20
    MIN_TOP3_GAP = 0.10
    
    is_reliable = True
    
    if p1 < MIN_CONF:
        is_reliable = False
    
    elif (p1 - p2) < MIN_MARGIN:
        is_reliable = False
    
    elif (p2 - p3) < MIN_TOP3_GAP and p2 > 0.25:
        is_reliable = False
    
    conflicting_pairs = [
        ("happy", "sad"), ("happy", "angry"),
        ("surprise", "sad"), ("fear", "happy")
    ]
    pair = tuple(sorted([raw_label1, raw_label2]))
    if pair in conflicting_pairs and (p1 - p2) < 0.25:
        is_reliable = False
    
    if not is_reliable:
        return "neutral", "chill", p1, False
    
    MODEL_TO_CATEGORY = {
        "happy": "happy_energetic",
        "surprise": "happy_energetic",
        "neutral": "chill",
        "disgust": "chill",
        "fear": "worried_anxious",
        "sad": "sad",
        "angry": "mad_irritated",
    }
    
    raw_label = raw_label1
    category = MODEL_TO_CATEGORY.get(raw_label, "chill")
    
    if category in ["mad_irritated", "sad", "worried_anxious"]:
        if p1 < 0.70:
            category = "chill"
            is_reliable = False
    
    return raw_label, category, p1, is_reliable



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



if 'step' not in st.session_state:
    st.session_state.step = 1
if 'detected_cat' not in st.session_state:
    st.session_state.detected_cat = None
if 'detected_conf' not in st.session_state:
    st.session_state.detected_conf = None
if 'is_reliable' not in st.session_state:
    st.session_state.is_reliable = None
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
            Our improved AI provides more accurate and reliable results.
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
                <li>Stay 50-100cm away</li>
                <li>Hold your expression</li>
                <li>Remove glasses if possible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        img = st.camera_input("Start Mood Scan", key="scan_cam")
        if img is not None:
            pil_img = Image.open(img)
            img_array = np.array(pil_img)
            
            frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            raw_label, category, confidence, is_reliable = predict_emotion_robust(
                frame_bgr, processor, model, face_detector, ID2LABEL
            )
            
            st.session_state.detected_cat = category
            st.session_state.detected_conf = confidence
            st.session_state.is_reliable = is_reliable
            st.session_state.final_cat = category
            
            st.markdown(f"""
            <div class='success-message'>
                <h3>✓ Mood Detected!</h3>
                <p>Detected mood: <strong>{CATEGORY_DISPLAY.get(category, category)}</strong> ({confidence:.0%} confidence)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Continue →", key="continue_step1", use_container_width=True):
                st.session_state.step = 2
                st.rerun()




elif st.session_state.step == 2:
    st.markdown(f"""
    <div class='card'>
        <div class='card-title'>Step 2: Confirm Your Mood</div>
        <div class='card-description'>
            AI detected: <span class='badge'>{CATEGORY_DISPLAY.get(st.session_state.detected_cat, '')}</span>
            <span style='color:
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
    
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("""
        <div style='font-size:1rem; font-weight:600; color:
            Saat ini kamu pengen rasa yang…
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        flavor = st.radio(
            "flavor",
            ["Sweet", "Fresh", "Creamy", "Strong"],
            label_visibility="collapsed",
            key="flavor_radio"
        )
        st.session_state.flavor_pref = flavor.lower()

    with col2:
        st.markdown("""
        <div style='font-size:1rem; font-weight:600; color:
            Energi kamu hari ini gimana?
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        energy = st.radio(
            "energy",
            ["Tired", "Normal", "Energetic"],
            index=1,
            label_visibility="collapsed",
            key="energy_radio"
        )
        st.session_state.energy_level = energy.lower()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
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
    
    if st.button("Start Over", key="reset"):
        for key in ['step', 'detected_cat', 'detected_conf', 'is_reliable', 'final_cat', 'flavor_pref', 'energy_level']:
            if key == 'step':
                st.session_state[key] = 1
            else:
                st.session_state[key] = None
        st.rerun()



st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color:
    <p>MoodScoop v3.0 - Improved AI Edition</p>
</div>
""", unsafe_allow_html=True)
