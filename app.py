"""
MoodScoop - AI Ice Cream Recommender
An AI-powered web application that recommends ice cream flavors based on detected emotions
"""

import time
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Page Configuration
st.set_page_config(
    page_title="MoodScoop - AI Ice Cream Recommender",
    page_icon="🍦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS Styling
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
        font-size: 2.25rem;
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
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
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
        transition: all 0.3s ease;
    }
    
    .step-line.completed {
        background: #10b981;
    }
    
    /* Card Component */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        transition: box-shadow 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        font-size: 0.95rem;
        color: #718096;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 1.75rem;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: #4338ca;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Radio Buttons - Pill Style */
    .stRadio > label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #1a202c;
        margin-bottom: 0.75rem;
    }
    
    .stRadio > div {
        gap: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    
    .stRadio > div > label {
        background: white !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 0.875rem 1.25rem !important;
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
    
    .stRadio > div > label[data-checked="true"] {
        border-color: #4f46e5 !important;
        background: #eff6ff !important;
        color: #4f46e5 !important;
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .success-message h3 {
        color: #065f46;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
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
        padding: 2.5rem 1.75rem;
        text-align: center;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
        animation: fadeIn 0.6s ease;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .result-emoji {
        font-size: 4.5rem;
        margin-bottom: 0.75rem;
        animation: bounce 0.6s ease;
    }
    
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .result-title {
        font-size: 1.875rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .result-description {
        font-size: 1.05rem;
        opacity: 0.95;
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    .result-detail {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    
    .result-profile {
        margin-top: 1.75rem;
        padding-top: 1.75rem;
        border-top: 1px solid rgba(255,255,255,0.3);
        font-size: 0.9rem;
        opacity: 0.9;
        line-height: 1.6;
    }
    
    /* Stats Display */
    .stats-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #e5e7eb;
        margin: 1rem 0;
    }
    
    .stats-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .stats-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4f46e5;
        margin-bottom: 0.25rem;
    }
    
    .stats-subtext {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: #4f46e5;
        border-radius: 4px;
    }
    
    /* Info Box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    
    .info-box p {
        color: #1e40af;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }
    
    /* Warning Box */
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    
    .warning-box p {
        color: #92400e;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }
    
    /* Tips Section */
    .tips-container {
        background: #f9fafb;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #e5e7eb;
    }
    
    .tips-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.75rem;
    }
    
    .tips-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tips-list li {
        color: #6b7280;
        font-size: 0.875rem;
        padding: 0.4rem 0;
        padding-left: 1.5rem;
        position: relative;
        line-height: 1.5;
    }
    
    .tips-list li:before {
        content: "•";
        color: #4f46e5;
        font-weight: bold;
        font-size: 1.2rem;
        position: absolute;
        left: 0.25rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 2rem 0;
    }
    
    /* Camera Input Styling */
    [data-testid="stCameraInput"] {
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stCameraInput"] > div {
        border-radius: 12px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    
    /* Smooth Transitions */
    * {
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING & CONFIGURATION
# ============================================================================

@st.cache_resource
def load_emotion_model():
    """Load the pre-trained emotion detection model"""
    try:
        processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        model.eval()
        return processor, model, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def load_face_detector():
    """Load Haar Cascade face detector"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return face_cascade if not face_cascade.empty() else None
    except:
        return None


# Load models with error handling
with st.spinner("🔄 Loading AI model..."):
    processor, model, error = load_emotion_model()
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.stop()
    
    ID2LABEL = model.config.id2label
    face_detector = load_face_detector()

# ============================================================================
# EMOTION & ICE CREAM MAPPINGS
# ============================================================================

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
    "happy_energetic": "Happy & Energetic 😊",
    "chill": "Chill & Relaxed 😌",
    "sad": "Sad & Down 😔",
    "worried_anxious": "Worried & Anxious 😰",
    "mad_irritated": "Frustrated & Irritated 😤",
}

ICE_CREAM_MAP = {
    "happy_energetic": {
        "name": "Strawberry Rainbow Sprinkle",
        "description": "Sweet and colorful to boost your happy mood",
        "detail": "Strawberry ice cream with rainbow sprinkles and whipped cream",
        "emoji": "🍓"
    },
    "chill": {
        "name": "Vanilla Matcha Glaze",
        "description": "Soft and calm vibes for relaxation",
        "detail": "Vanilla ice cream with matcha glaze and mochi topping",
        "emoji": "🍵"
    },
    "sad": {
        "name": "Red Velvet Chocolate",
        "description": "Comfort dessert to lift your spirits",
        "detail": "Chocolate ice cream with red velvet crumbs and caramel drizzle",
        "emoji": "🍫"
    },
    "worried_anxious": {
        "name": "Cookies & Cream Delight",
        "description": "Classic comfort to ease your mind",
        "detail": "Vanilla ice cream with cookie crumbles and chocolate chips",
        "emoji": "🍪"
    },
    "mad_irritated": {
        "name": "Dark Chocolate Tiramisu",
        "description": "Bold and intense for when you need strength",
        "detail": "Rich chocolate ice cream with tiramisu glaze and espresso drizzle",
        "emoji": "☕"
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_emotion_from_image(pil_image):
    """Predict emotion from a PIL Image"""
    try:
        inputs = processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = outputs.logits.softmax(dim=1)[0]
        idx = int(torch.argmax(probs).item())
        raw_label = ID2LABEL[idx].lower()
        confidence = float(probs[idx].item())
        category = MODEL_TO_CATEGORY.get(raw_label, "chill")
        
        return raw_label, category, confidence
    except Exception as e:
        st.error(f"Error during emotion prediction: {e}")
        return None, None, None


def get_step_indicator_html(current_step):
    """Generate HTML for step indicator"""
    def get_circle_class(step_num):
        if current_step > step_num:
            return "completed"
        elif current_step == step_num:
            return "active"
        else:
            return "inactive"
    
    def get_line_class(step_num):
        return "completed" if current_step > step_num else ""
    
    return f"""
    <div class='step-indicator'>
        <div class='step'>
            <div class='step-circle {get_circle_class(1)}'>1</div>
        </div>
        <div class='step-line {get_line_class(1)}'></div>
        <div class='step'>
            <div class='step-circle {get_circle_class(2)}'>2</div>
        </div>
        <div class='step-line {get_line_class(2)}'></div>
        <div class='step'>
            <div class='step-circle {get_circle_class(3)}'>3</div>
        </div>
        <div class='step-line {get_line_class(3)}'></div>
        <div class='step'>
            <div class='step-circle {get_circle_class(4)}'>4</div>
        </div>
    </div>
    """

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

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

# ============================================================================
# MAIN APP UI
# ============================================================================

# Header
st.markdown("""
<div class='app-header'>
    <h1>🍦 MoodScoop</h1>
    <p>AI-powered ice cream recommendations based on your mood</p>
</div>
""", unsafe_allow_html=True)

# Step Indicator
st.markdown(get_step_indicator_html(st.session_state.step), unsafe_allow_html=True)

# ============================================================================
# STEP 1: MOOD DETECTION
# ============================================================================

if st.session_state.step == 1:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 1: Mood Detection 📸</div>
        <div class='card-description'>
            Capture your photo and let our AI analyze your facial expression to detect your current mood.
            The process is instant and uses advanced emotion recognition technology.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("""
        <div class='tips-container'>
            <div class='tips-title'>📋 Tips for best results:</div>
            <ul class='tips-list'>
                <li>Face the camera directly</li>
                <li>Ensure good lighting</li>
                <li>Stay 50-100cm away from camera</li>
                <li>Natural expression works best</li>
                <li>Remove glasses if possible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        img = st.camera_input("📷 Capture Your Photo", key="scan_cam", help="Click to take a photo")
        
        if img is not None:
            try:
                pil_img = Image.open(img)
                
                # Show processing indicator
                with st.spinner("🔍 Analyzing your mood..."):
                    raw_label, cat, confidence = predict_emotion_from_image(pil_img)
                
                if cat is not None:
                    st.session_state.detected_cat = cat
                    st.session_state.detected_conf = confidence
                    st.session_state.final_cat = cat
                    st.session_state.step = 2
                    
                    st.markdown(f"""
                    <div class='success-message'>
                        <h3>✓ Scan Complete!</h3>
                        <p>Detected mood: <strong>{CATEGORY_DISPLAY.get(cat, cat)}</strong> • Confidence: {confidence:.0%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Unable to detect emotion. Please try again.")
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")

# ============================================================================
# STEP 2: CONFIRM MOOD
# ============================================================================

elif st.session_state.step == 2:
    mood_name = CATEGORY_DISPLAY.get(st.session_state.detected_cat, '')
    confidence_pct = st.session_state.detected_conf * 100 if st.session_state.detected_conf else 0
    
    st.markdown(f"""
    <div class='card'>
        <div class='card-title'>Step 2: Confirm Your Mood ✅</div>
        <div class='card-description'>
            Our AI detected your mood as <span class='badge'>{mood_name}</span> with {confidence_pct:.0f}% confidence.
            <br><br>
            Please confirm or adjust to ensure the best ice cream recommendation for you.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    options = list(CATEGORY_DISPLAY.values())
    default_idx = list(CATEGORY_DISPLAY.keys()).index(st.session_state.detected_cat)
    
    st.markdown("**How are you feeling right now?**")
    chosen = st.radio(
        "mood_selection",
        options,
        index=default_idx,
        key="mood_radio",
        label_visibility="collapsed"
    )
    
    chosen_key = list(CATEGORY_DISPLAY.keys())[list(CATEGORY_DISPLAY.values()).index(chosen)]
    st.session_state.final_cat = chosen_key
    
    st.markdown("""
    <div class='info-box'>
        <p>💡 <strong>Your choice matters!</strong> AI provides a suggestion, but you know yourself best. 
        Choose what feels right for your current emotional state.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", key="back_2", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Continue →", key="next_2", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ============================================================================
# STEP 3: PREFERENCES
# ============================================================================

elif st.session_state.step == 3:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Step 3: Your Preferences 🎯</div>
        <div class='card-description'>
            Help us personalize your ice cream recommendation by sharing your current taste preferences and energy level.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🍨 Flavor preference:**")
        flavor = st.radio(
            "flavor",
            ["Sweet", "Fresh", "Creamy", "Strong"],
            label_visibility="collapsed",
            key="flavor_radio",
            help="What type of flavor are you craving?"
        )
        st.session_state.flavor_pref = flavor.lower()
    
    with col2:
        st.markdown("**⚡ Energy level:**")
        energy = st.radio(
            "energy",
            ["Tired", "Normal", "Energetic"],
            index=1,
            label_visibility="collapsed",
            key="energy_radio",
            help="How energetic do you feel right now?"
        )
        st.session_state.energy_level = energy.lower()
    
    st.markdown("""
    <div class='info-box'>
        <p>✨ These preferences help us fine-tune your recommendation to match not just your mood, 
        but also your taste and energy needs right now.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", key="back_3", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Get Recommendation →", key="next_3", use_container_width=True):
            st.session_state.step = 4
            st.balloons()
            st.rerun()

# ============================================================================
# STEP 4: RESULTS
# ============================================================================

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
            <strong>🎨 Your Personalized Profile</strong><br>
            Mood: {mood_text}<br>
            Flavor Preference: {st.session_state.flavor_pref.title()}<br>
            Energy Level: {st.session_state.energy_level.title()}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <p>✨ <strong>Perfect Match!</strong> This recommendation is AI-powered and personalized based on your 
        emotional state and preferences. Enjoy your treat and let your mood shine! 🌟</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Start Over", key="reset", use_container_width=True):
            for key in ['step', 'detected_cat', 'detected_conf', 'final_cat', 'flavor_pref', 'energy_level']:
                if key == 'step':
                    st.session_state[key] = 1
                else:
                    st.session_state[key] = None
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 0.875rem; padding: 1rem 0;'>
    <p><strong>MoodScoop</strong> v4.0 | Powered by AI Emotion Recognition</p>
    <p style='font-size: 0.8rem; margin-top: 0.25rem;'>Made with ❤️ for ice cream lovers</p>
</div>
""", unsafe_allow_html=True)
