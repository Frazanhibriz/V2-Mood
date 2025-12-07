import time
from collections import Counter, deque
from typing import Tuple, List, Optional

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="MoodScoop Enhanced - AI Ice Cream Recommender",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS (sama seperti sebelumnya, tidak berubah)
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
    
    .stProgress > div > div > div {
        background: #4f46e5;
    }
    
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
    
    .badge {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 1.5rem 0;
    }
    
    .confidence-bar {
        background: #e5e7eb;
        border-radius: 8px;
        height: 8px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #10b981, #4f46e5);
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    * {
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ADAPTIVE THRESHOLDS ====================
EMOTION_THRESHOLDS = {
    "happy": 0.60,
    "surprise": 0.65,
    "neutral": 0.50,
    "disgust": 0.70,
    "fear": 0.70,
    "sad": 0.65,
    "angry": 0.65,
}

# ==================== IMPROVED FACE PREPROCESSING ====================
def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """
    Enhanced face preprocessing with CLAHE for better lighting normalization
    """
    # Resize to standard size
    face_resized = cv2.resize(face_img, (224, 224))
    
    # Convert to LAB color space for better lighting normalization
    lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    face_normalized = cv2.merge([l, a, b])
    face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_LAB2BGR)
    
    return face_normalized


# ==================== MEDIAPIPE FACE DETECTION ====================
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available, using Haar Cascade fallback")


@st.cache_resource
def load_face_detector():
    """
    Load face detector - MediaPipe if available, otherwise Haar Cascade
    """
    if MEDIAPIPE_AVAILABLE:
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(
            min_detection_confidence=0.7,
            model_selection=0  # 0 for close range (2m), 1 for far range (5m)
        )
        return detector, 'mediapipe'
    else:
        # Fallback to Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not face_cascade.empty():
            return face_cascade, 'haar'
        return None, None


def detect_face_mediapipe(image_bgr: np.ndarray, detector) -> Optional[np.ndarray]:
    """
    Detect face using MediaPipe
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    
    if results.detections:
        # Get the largest face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image_bgr.shape
        
        # Convert relative coordinates to absolute
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h)
        width = int(bboxC.width * w)
        height = int(bboxC.height * h)
        
        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        return image_bgr[y:y+height, x:x+width]
    
    return None


def detect_face_haar(image_bgr: np.ndarray, cascade) -> Optional[np.ndarray]:
    """
    Detect face using Haar Cascade
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(60, 60)
    )
    
    if len(faces) > 0:
        # Get the largest face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        return image_bgr[y:y+h, x:x+w]
    
    return None


# ==================== ENSEMBLE MODELS ====================
@st.cache_resource
def load_emotion_models():
    """
    Load primary model and optional ensemble models
    """
    models = []
    
    # Primary model (dima806 - 91% accuracy)
    try:
        processor1 = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        model1 = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        model1.eval()
        models.append({
            'processor': processor1,
            'model': model1,
            'name': 'dima806',
            'weight': 1.0,
            'id2label': model1.config.id2label
        })
    except Exception as e:
        st.error(f"Failed to load primary model: {e}")
        return None
    
    # Optional: Add second model for ensemble (commented out by default for speed)
    # Uncomment to enable ensemble prediction
    """
    try:
        processor2 = AutoImageProcessor.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large")
        model2 = AutoModelForImageClassification.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large")
        model2.eval()
        models.append({
            'processor': processor2,
            'model': model2,
            'name': 'beit',
            'weight': 0.7,
            'id2label': model2.config.id2label
        })
    except Exception as e:
        print(f"Secondary model not loaded: {e}")
    """
    
    return models


# ==================== ENHANCED PREDICTION ====================
def predict_emotion_ensemble(
    face_rgb: np.ndarray, 
    models: List[dict]
) -> Tuple[str, float, dict]:
    """
    Predict emotion using ensemble of models
    """
    pil_img = Image.fromarray(face_rgb)
    
    all_predictions = []
    emotion_scores = {}
    
    for model_info in models:
        processor = model_info['processor']
        model = model_info['model']
        weight = model_info['weight']
        id2label = model_info['id2label']
        
        # Get prediction
        inputs = processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = outputs.logits.softmax(dim=1)[0]
        
        # Weighted voting
        for idx, prob in enumerate(probs):
            emotion = id2label[idx].lower()
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0
            emotion_scores[emotion] += float(prob) * weight
    
    # Get top prediction
    top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    raw_label = top_emotion[0]
    confidence = top_emotion[1] / sum(model_info['weight'] for model_info in models)
    
    return raw_label, confidence, emotion_scores


def apply_adaptive_threshold(emotion: str, confidence: float) -> Tuple[str, float]:
    """
    Apply emotion-specific confidence thresholds
    """
    threshold = EMOTION_THRESHOLDS.get(emotion, 0.60)
    
    if confidence < threshold:
        return "neutral", confidence
    
    return emotion, confidence


def predict_emotion_from_frame(
    frame_bgr: np.ndarray,
    face_detector,
    detector_type: str,
    models: List[dict]
) -> Tuple[str, str, float]:
    """
    Complete emotion prediction pipeline with all enhancements
    """
    # Step 1: Detect face
    if detector_type == 'mediapipe':
        face_roi = detect_face_mediapipe(frame_bgr, face_detector)
    elif detector_type == 'haar':
        face_roi = detect_face_haar(frame_bgr, face_detector)
    else:
        face_roi = None
    
    # Use whole image if no face detected
    if face_roi is None or face_roi.size == 0:
        face_roi = frame_bgr
    
    # Step 2: Preprocess face
    face_preprocessed = preprocess_face(face_roi)
    
    # Step 3: Convert to RGB
    face_rgb = cv2.cvtColor(face_preprocessed, cv2.COLOR_BGR2RGB)
    
    # Step 4: Ensemble prediction
    raw_label, confidence, emotion_scores = predict_emotion_ensemble(face_rgb, models)
    
    # Step 5: Apply adaptive threshold
    raw_label, confidence = apply_adaptive_threshold(raw_label, confidence)
    
    # Step 6: Map to category
    MODEL_TO_CATEGORY = {
        "happy": "happy_energetic",
        "surprise": "happy_energetic",
        "neutral": "chill",
        "disgust": "chill",
        "fear": "worried_anxious",
        "sad": "sad",
        "angry": "mad_irritated",
    }
    
    category = MODEL_TO_CATEGORY.get(raw_label, "chill")
    
    # Additional filtering for negative emotions with low confidence
    if category in ["mad_irritated", "sad", "worried_anxious"] and confidence < 0.60:
        category = "chill"
    
    return raw_label, category, confidence


# ==================== MULTI-FRAME AVERAGING ====================
class EmotionAverager:
    """
    Averages emotion predictions across multiple frames for stability
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
    
    def add_prediction(self, category: str, confidence: float):
        self.predictions.append(category)
        self.confidences.append(confidence)
    
    def get_averaged_prediction(self) -> Tuple[str, float]:
        if not self.predictions:
            return "neutral", 0.5
        
        # Majority voting
        most_common = Counter(self.predictions).most_common(1)[0][0]
        avg_confidence = np.mean(list(self.confidences))
        
        return most_common, float(avg_confidence)
    
    def reset(self):
        self.predictions.clear()
        self.confidences.clear()


# ==================== LOAD RESOURCES ====================
with st.spinner("🚀 Loading enhanced AI models..."):
    models = load_emotion_models()
    if models is None:
        st.error("Failed to load emotion detection models!")
        st.stop()
    
    face_detector, detector_type = load_face_detector()
    if face_detector is None:
        st.warning("⚠️  No face detector available. Using full image for detection.")
        detector_type = None

# Display detection method
if detector_type == 'mediapipe':
    detection_method = "MediaPipe (Advanced)"
elif detector_type == 'haar':
    detection_method = "Haar Cascade (Standard)"
else:
    detection_method = "Full Image"

# ==================== MOOD CATEGORIES ====================
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

# ==================== SESSION STATE ====================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'detected_cat' not in st.session_state:
    st.session_state.detected_cat = None
if 'detected_conf' not in st.session_state:
    st.session_state.detected_conf = None
if 'detected_raw' not in st.session_state:
    st.session_state.detected_raw = None
if 'final_cat' not in st.session_state:
    st.session_state.final_cat = None
if 'flavor_pref' not in st.session_state:
    st.session_state.flavor_pref = None
if 'energy_level' not in st.session_state:
    st.session_state.energy_level = None
if 'emotion_averager' not in st.session_state:
    st.session_state.emotion_averager = EmotionAverager(window_size=5)

# ==================== HEADER ====================
st.markdown(f"""
<div class='app-header'>
    <h1>🍦 MoodScoop Enhanced</h1>
    <p>AI-powered ice cream recommendations with advanced emotion detection</p>
    <span class='badge'>Detection: {detection_method}</span>
</div>
""", unsafe_allow_html=True)

# ==================== STEP INDICATOR ====================
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

# ==================== STEP 1: MOOD DETECTION ====================
if st.session_state.step == 1:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 1: Enhanced Mood Detection</div>
        <div class='card-description'>
            Using advanced AI with face preprocessing, adaptive thresholds, and multi-model ensemble
            for the highest accuracy emotion detection.
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
        img = st.camera_input("Start Enhanced Mood Scan", key="scan_cam")
        
        if img is not None:
            # Convert to numpy array
            pil_img = Image.open(img)
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            status_text.text("🔍 Detecting face...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            status_text.text("🎨 Preprocessing image...")
            progress_bar.progress(40)
            time.sleep(0.3)
            
            status_text.text("🤖 Analyzing emotions...")
            progress_bar.progress(60)
            
            # Predict emotion
            raw_label, cat, confidence = predict_emotion_from_frame(
                frame_bgr, 
                face_detector, 
                detector_type, 
                models
            )
            
            progress_bar.progress(80)
            status_text.text("✨ Finalizing results...")
            time.sleep(0.3)
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            # Store results
            st.session_state.detected_cat = cat
            st.session_state.detected_conf = confidence
            st.session_state.detected_raw = raw_label
            st.session_state.final_cat = cat
            
            # Display results with confidence visualization
            confidence_level = "High" if confidence > 0.75 else "Medium" if confidence > 0.60 else "Moderate"
            confidence_color = "#10b981" if confidence > 0.75 else "#f59e0b" if confidence > 0.60 else "#6b7280"
            
            st.markdown(f"""
            <div class='success-message'>
                <h3>✓ Scan Complete!</h3>
                <p>Detected mood: <strong>{CATEGORY_DISPLAY.get(cat, cat)}</strong></p>
                <p>Raw emotion: <strong>{raw_label.title()}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence meter
            col_conf1, col_conf2 = st.columns([2, 1])
            with col_conf1:
                st.markdown(f"""
                <div class='stats-container'>
                    <div class='stats-label'>Confidence Score</div>
                    <div class='stats-value' style='color: {confidence_color};'>{confidence:.0%}</div>
                    <div class='stats-subtext'>{confidence_level} confidence</div>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {confidence*100}%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_conf2:
                st.markdown(f"""
                <div class='stats-container'>
                    <div class='stats-label'>Detection Method</div>
                    <div class='stats-value' style='font-size: 1rem;'>{detection_method}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Continue →", key="continue_step1", use_container_width=True):
                st.session_state.step = 2
                st.rerun()

# ==================== STEP 2: CONFIRM MOOD ====================
elif st.session_state.step == 2:
    st.markdown(f"""
    <div class='card'>
        <div class='card-title'>Step 2: Confirm Your Mood</div>
        <div class='card-description'>
            We detected your mood as <span class='badge'>{CATEGORY_DISPLAY.get(st.session_state.detected_cat, '')}</span>
            with <strong>{st.session_state.detected_conf:.0%} confidence</strong>
            <br><br>
            Please confirm or adjust to ensure the best recommendation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show detection details
    st.markdown(f"""
    <div class='info-box'>
        <p>🧠 Detected emotion: <strong>{st.session_state.detected_raw.title()}</strong> → Category: <strong>{CATEGORY_DISPLAY.get(st.session_state.detected_cat, '')}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    options = list(CATEGORY_DISPLAY.values())
    default_idx = list(CATEGORY_DISPLAY.keys()).index(st.session_state.detected_cat)
    
    chosen = st.radio("Select your current mood:", options, index=default_idx, key="mood_radio")
    
    chosen_key = list(CATEGORY_DISPLAY.keys())[list(CATEGORY_DISPLAY.values()).index(chosen)]
    st.session_state.final_cat = chosen_key
    
    st.markdown("""
    <div class='info-box'>
        <p>💡 AI provides an initial suggestion with confidence scoring, but you have the final say.</p>
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

# ==================== STEP 3: PREFERENCES ====================
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
        <div style='font-size:1rem; font-weight:600; color:#111827; margin-bottom:0.6rem;'>
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
        <div style='font-size:1rem; font-weight:600; color:#111827; margin-bottom:0.6rem;'>
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

# ==================== STEP 4: RESULTS ====================
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
            Mood: {mood_text} • Confidence: {st.session_state.detected_conf:.0%} • Flavor: {st.session_state.flavor_pref.title()} • Energy: {st.session_state.energy_level.title()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='info-box'>
        <p>✨ Enhanced with: {detection_method} face detection + CLAHE preprocessing + Adaptive thresholds</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Over", key="reset"):
        for key in ['step', 'detected_cat', 'detected_conf', 'detected_raw', 'final_cat', 'flavor_pref', 'energy_level']:
            if key == 'step':
                st.session_state[key] = 1
            else:
                st.session_state[key] = None
        st.session_state.emotion_averager.reset()
        st.rerun()

# ==================== FOOTER ====================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #9ca3af; font-size: 0.875rem; padding: 1rem 0;'>
    <p><strong>MoodScoop Enhanced v4.0</strong></p>
    <p style='font-size: 0.75rem;'>
        Features: {detection_method} • CLAHE Preprocessing • Adaptive Thresholds • Ensemble Ready
    </p>
</div>
""", unsafe_allow_html=True)
