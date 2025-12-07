import time
from collections import Counter, deque
from skimage.transform import rotate
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
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
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
        color: #1a202c;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-size: 0.95rem;
        color: #718096;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
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
    
    /* Radio Buttons - Minimal Design */
    .stRadio > label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #1a202c;
        margin-bottom: 0.75rem;
    }
    
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio > div > label {
        background: white !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
        color: #4b5563 !important;
        display: flex !important;
        align-items: center !important;
    }
    
    .stRadio > div > label:hover {
        border-color: #4f46e5 !important;
        background: #f9fafb !important;
    }
    
    .stRadio > div > label > div {
        color: #4b5563 !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        border-color: #4f46e5 !important;
        background: #eff6ff !important;
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .success-message h3 {
        color: #065f46;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    
    .success-message p {
        color: #047857;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Ice Cream Result Card */
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
        border: 2px solid #e5e7eb;
    }
    
    .stats-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    
    .stats-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4f46e5;
    }
    
    .stats-subtext {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: #4f46e5;
    }
    
    /* Info Box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
    }
    
    .info-box p {
        color: #1e40af;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Tips Section */
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
    
    .tips-list {
        list-style: none;
        padding: 0;
        margin: 0;
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
    
    /* Badge */
    .badge {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 1.5rem 0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Smooth Transitions */
    * {
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

def mouth_curvature(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lower = gray[int(h*0.55):int(h*0.75), int(w*0.2):int(w*0.8)]
    edges = cv2.Canny(lower, 40, 110)
    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return 0.0
    z = np.polyfit(xs, ys, 2)[0]
    s = max(0, min(1, -z * 2500))
    return s


def eyebrow_tension(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    upper = gray[int(h*0.12):int(h*0.30), int(w*0.2):int(w*0.8)]
    edges = cv2.Canny(upper, 40, 120)
    t = min(1, edges.sum() / 35000)
    return t


def eye_aspect_ratio(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    eyes = gray[int(h*0.25):int(h*0.50), int(w*0.2):int(w*0.8)]
    closed = np.mean(eyes < 60)
    return closed


def enhance_image(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


def align_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    pts = np.column_stack(np.where(edges > 0))
    if len(pts) < 60:
        return img
    ang = cv2.minAreaRect(pts)[-1]
    if ang < -45:
        ang += 90
    return rotate(img, ang, reshape=False)


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
    "surprise": "happy_energetic",

    # emosi yang sering noise → turunin ke chill / neutral-ish
    "neutral": "chill",
    "disgust": "chill",
    "fear": "worried_anxious",   # boleh, tapi nanti kita cek confidence

    "sad": "sad",
    "angry": "mad_irritated",
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
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            crop = frame_bgr[y:y+h, x:x+w]
        else:
            crop = frame_bgr
    else:
        crop = frame_bgr

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = enhance_image(rgb)
    rgb = align_face(rgb)

    smile = mouth_curvature(crop)
    tension = eyebrow_tension(crop)
    tired = eye_aspect_ratio(crop)

    pil_img = Image.fromarray(np.uint8(rgb))
    inp = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inp).logits.softmax(dim=1)[0].numpy()

    base_idx = int(np.argmax(logits))
    base_label = ID2LABEL[base_idx].lower()
    base_conf = float(logits[base_idx])

    F = {}

    F["happy"] = base_conf * (1 + 0.7*smile - 0.2*tension)
    F["sad"] = base_conf * (1 + 0.4*tired)
    F["angry"] = base_conf * (1 + 0.5*tension - 0.3*smile)
    F["fear"] = base_conf * (1 + 0.2*tension)
    F["neutral"] = base_conf * (1 - 0.2*smile)

    final = max(F, key=F.get)
    prob = F[final]

    if final in ["angry", "fear", "sad"] and prob < 0.55:
        final = "neutral"

    cat = MODEL_TO_CATEGORY.get(final, "chill")
    return final, cat, prob


def realtime_scan(duration_sec=12, fps=15, buffer_size=32):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera.")
        return None, None, None

    frame_box = st.empty()
    prog = st.empty()
    stat = st.empty()

    start = time.time()
    interval = 1/fps

    buf = deque(maxlen=buffer_size)
    buf_conf = deque(maxlen=buffer_size)

    snap1 = snap2 = snap3 = None
    t1, t2, t3 = duration_sec*0.25, duration_sec*0.50, duration_sec*0.80

    raw_last = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        raw, cat, conf = predict_emotion_from_frame(frame)
        raw_last = raw
        buf.append(cat)
        buf_conf.append(conf)

        el = time.time() - start

        wts = np.linspace(0.4, 1.0, len(buf))
        score = {}

        for w, c in zip(wts, buf):
            score[c] = score.get(c, 0) + w

        stable = max(score, key=score.get)
        stable_conf = np.mean([c for c, cc in zip(buf_conf, buf) if cc == stable])

        if snap1 is None and el >= t1: snap1 = stable
        if snap2 is None and el >= t2: snap2 = stable
        if snap3 is None and el >= t3: snap3 = stable

        frame_box.image(frame, channels="BGR", use_container_width=True)
        prog.progress(min(el/duration_sec,1.0))
        stat.markdown(f"<div class='stats-container'><div class='stats-label'>Detected</div><div class='stats-value'>{CATEGORY_DISPLAY.get(stable, stable)}</div><div class='stats-subtext'>Confidence {stable_conf:.0%}</div></div>", unsafe_allow_html=True)

        if el >= duration_sec:
            break

        time.sleep(interval)

    cap.release()

    votes = [v for v in [snap1, snap2, snap3] if v]
    if not votes:
        return None, None, None

    final = Counter(votes).most_common(1)[0][0]
    final_conf = np.mean([c for c, cc in zip(buf_conf, buf) if cc == final])

    return raw_last, final, final_conf


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
        <div class='step-circle {}'">1</div>
    </div>
    <div class='step-line {}'></div>
    <div class='step'>
        <div class='step-circle {}'">2</div>
    </div>
    <div class='step-line {}'></div>
    <div class='step'>
        <div class='step-circle {}'">3</div>
    </div>
    <div class='step-line {}'></div>
    <div class='step'>
        <div class='step-circle {}'">4</div>
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
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    if st.button("Start Mood Scan"):
        raw, cat, conf = realtime_scan()

        if cat is None:
            st.error("Scan failed. Please try again.")
        else:
            st.session_state.detected_cat = cat
            st.session_state.detected_conf = conf
            st.session_state.final_cat = cat
            st.session_state.step = 2
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
    
    # Two columns, evenly spaced
    col1, col2 = st.columns([2, 1], gap="large")

    # -------- LEFT SIDE (FLAVOR) --------
    with col1:
        st.markdown("""
        <div style='font-size:1rem; font-weight:600; color:#111827; margin-bottom:0.6rem;'>
            Saat ini kamu pengen rasa yang…
        </div>
        """, unsafe_allow_html=True)

        # Spacer so both columns align vertically
        st.write("")

        flavor = st.radio(
            "flavor",
            ["Sweet", "Fresh", "Creamy", "Strong"],
            label_visibility="collapsed",
            key="flavor_radio"
        )
        st.session_state.flavor_pref = flavor.lower()

    # -------- RIGHT SIDE (ENERGY) --------
    with col2:
        st.markdown("""
        <div style='font-size:1rem; font-weight:600; color:#111827; margin-bottom:0.6rem;'>
            Energi kamu hari ini gimana?
        </div>
        """, unsafe_allow_html=True)

        # Same spacer for alignment
        st.write("")

        energy = st.radio(
            "energy",
            ["Tired", "Normal", "Energetic"],
            index=1,
            label_visibility="collapsed",
            key="energy_radio"
        )
        st.session_state.energy_level = energy.lower()

    # Navigation buttons
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
    
    st.markdown("""
    <div class='info-box'>
        <p>✨ This recommendation is AI-powered but meant to be fun and interactive. Enjoy your treat!</p>
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
<div style='text-align: center; color: #9ca3af; font-size: 0.875rem; padding: 1rem 0;'>
    <p>MoodScoop v3.0</p>
</div>
""", unsafe_allow_html=True)
