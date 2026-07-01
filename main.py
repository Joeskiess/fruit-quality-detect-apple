import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import joblib
import gdown
import os
import requests
from io import BytesIO
from PIL import Image


# Page config

st.set_page_config(
    page_title="Fruit Quality Detector",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        min-height: 100vh;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }
    /* ---- Header ---- */
    .app-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2.5rem;
    }
    /* ---- Upload card ---- */
    .upload-card {
        background: rgba(255,255,255,0.06);
        border: 2px dashed rgba(255,255,255,0.18);
        border-radius: 18px;
        padding: 1.5rem 1.5rem 0.5rem 1.5rem;
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }
    .upload-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    /* ---- Sample row ---- */
    .sample-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
        margin-top: 0.5rem;
    }
    /* ---- Result card ---- */
    .result-card {
        background: rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-top: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .result-conf {
        text-align: center;
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .conf-badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.15rem;
        color: white;
    }
    .badge-good  { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .badge-bad   { background: linear-gradient(90deg, #ef4444, #dc2626); }
    .badge-mixed { background: linear-gradient(90deg, #f59e0b, #d97706); }
    /* ---- Confusion matrix section ---- */
    .section-title {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    .cm-container {
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    /* ---- Placeholder ---- */
    .placeholder-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 3rem 2rem;
        border: 2px dashed rgba(255,255,255,0.12);
        margin-top: 1.5rem;
        text-align: center;
    }
    .placeholder-emoji { font-size: 4rem; margin-bottom: 0.75rem; }
    .placeholder-text  { color: #64748b; font-size: 1rem; }
    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        height: 2.6rem;
        transition: all 0.2s ease;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    /* ---- Selectbox / file uploader labels ---- */
    .stSelectbox label, .stFileUploader label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* ---- Footer ---- */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES   = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Quality_Fruits']
DISPLAY_NAMES = ['Bad Quality',        'Good Quality',        'Mixed Quality']
EMOJIS        = ['🔴',                 '🟢',                  '🟡']
BADGE_CLASS   = ['badge-bad',          'badge-good',          'badge-mixed']
IMG_SIZE      = (224, 224)
SAMPLE_URLS = {
    "Good Fruit 🟢": "https://i.ibb.co/tT6vGVQx/20190809-115613.jpg",
    "Bad Fruit  🔴": "https://i.ibb.co/zTBj1L8N/IMG-8152.jpg",
    "Mixed Fruit 🟡": "https://i.ibb.co/rRkRxPG9/IMG20200728130108.jpg",
}



# Model loading
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    files = {
    "models/best_ef_model.keras": st.secrets["EF_MODEL_ID"],
    "models/best_resnet_model.keras": st.secrets["RESNET_MODEL_ID"],
    "models/rf_model.joblib": st.secrets["RF_MODEL_ID"],
    "models/feature_scaler.joblib": st.secrets["SCALER_ID"],
    }
    for path, file_id in files.items():
        if not os.path.exists(path):
            try:
                gdown.download(id=file_id, output=path, quiet=True)
            except Exception as e:
                st.error(f"Failed to download {path}: {e}")
                st.stop()
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                st.error(f"Download failed or empty file: {path}")
                st.stop()
    ef_model     = tf.keras.models.load_model("models/best_ef_model.keras")
    resnet_model = tf.keras.models.load_model("models/best_resnet_model.keras")
    rf_model     = joblib.load("models/rf_model.joblib")
    scaler       = joblib.load("models/feature_scaler.joblib")
    return ef_model, resnet_model, rf_model, scaler

# Prediction
def predict(image_source, ef_model, resnet_model, rf_model, scaler):
    if hasattr(image_source, 'seek'):
        image_source.seek(0)
    image = Image.open(image_source).convert("RGB")
    image_array = np.array(image.resize(IMG_SIZE), dtype=np.float32)
    image_tensor = tf.constant(np.expand_dims(image_array, axis=0))
    ef_proba     = ef_model.predict(image_tensor, verbose=0)
    resnet_proba = resnet_model.predict(image_tensor, verbose=0)
    ef_extractor     = models.Model(inputs=ef_model.input,     outputs=ef_model.layers[-2].output)
    resnet_extractor = models.Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
    combined_feat    = np.hstack([
        ef_extractor.predict(image_tensor, verbose=0),
        resnet_extractor.predict(image_tensor, verbose=0)
    ])
    rf_proba = rf_model.predict_proba(scaler.transform(combined_feat))
    final_proba     = (ef_proba + resnet_proba + rf_proba) / 3
    predicted_class = int(np.argmax(final_proba, axis=1)[0])
    confidence      = float(final_proba[0][predicted_class])
    return predicted_class, confidence, image
# ============================================
# Session state
# ============================================
for key, default in [("image_data", None), ("prediction", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# Header
st.markdown('<div class="app-title">🍎 Fruit Quality Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload a fruit image or pick a sample — the ensemble model will judge its quality.</div>', unsafe_allow_html=True)

# Upload / sample section
with st.container():
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a fruit image",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    # Sample image picker + action buttons on one row
    col_sel, col_load, col_reset = st.columns([3, 1.2, 1.2])
    with col_sel:
        sample_choice = st.selectbox(
            "Or try a sample",
            ["— pick a sample —"] + list(SAMPLE_URLS.keys()),
            label_visibility="visible"
        )
    with col_load:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)  # vertical align
        load_clicked = st.button("Load Sample", use_container_width=True)
    with col_reset:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        reset_clicked = st.button("🔄 Reset", use_container_width=True)
# Handle reset
if reset_clicked:
    st.session_state.image_data  = None
    st.session_state.prediction  = None
    st.rerun()
# Handle uploaded file
if uploaded_file is not None:
    if st.session_state.image_data != uploaded_file.name:
        st.session_state.image_data = uploaded_file.name
        st.session_state.prediction = None
    image_source = uploaded_file
# Handle sample load
elif load_clicked and sample_choice != "— pick a sample —":
    url = SAMPLE_URLS[sample_choice]
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        image_source = BytesIO(resp.content)
        st.session_state.image_data = url
        st.session_state.prediction = None
    except Exception as e:
        st.error(f"Couldn't load sample image: {e}")
        image_source = None
else:
    image_source = None


# Prediction & result display

if image_source is not None:
    # Load models (cached after first run)
    with st.spinner("Loading models… first run takes a minute."):
        ef_model, resnet_model, rf_model, scaler = load_models()
    # Run prediction
    if st.session_state.prediction is None:
        with st.spinner("Analysing image…"):
            predicted_class, confidence, pil_image = predict(
                image_source, ef_model, resnet_model, rf_model, scaler
            )
        st.session_state.prediction = (predicted_class, confidence, pil_image)
    else:
        predicted_class, confidence, pil_image = st.session_state.prediction
    #  Result card 
    badge_cls    = BADGE_CLASS[predicted_class]
    display_name = DISPLAY_NAMES[predicted_class]
    emoji        = EMOJIS[predicted_class]
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    img_col, info_col = st.columns([1, 1], gap="large")
    with img_col:
        st.image(pil_image, use_container_width=True, caption="Input image")
    with info_col:
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; justify-content:center; height:100%; gap:1rem; padding-top:1rem;">
            <div class="result-label">{emoji} {display_name}</div>
            <div class="result-conf">
                Confidence &nbsp;
                <span class="conf-badge {badge_cls}">{confidence:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Placeholder
    st.markdown("""
    <div class="placeholder-box">
        <div class="placeholder-emoji">🍏</div>
        <div class="placeholder-text">Upload a fruit image above or load a sample to get started.</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="footer">Ensemble model · EfficientNetB0 + ResNet50 + Random Forest · Built with Streamlit & TensorFlow</div>',
    unsafe_allow_html=True
)
