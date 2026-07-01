import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import joblib
import gdown
import os
from PIL import Image

# ============================================
# Download models from Google Drive (once)
# ============================================
@st.cache_resource
def load_models():
    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Download each file from Google Drive if not already present
    files = {
        "models/best_ef_model.keras": "1CM1bPY3dHb9sP-WuT2CIXS9c4o0HungL",
        "models/best_resnet_model.keras": "1Sn2BC-ST3NFjndlBmmyXjSw47ic6vzYb",
        "models/rf_model.joblib": "166uHbJdgv-C41dNYctE1WfbZBXTrffdU",
        "models/feature_scaler.joblib": "12RrfEo1Dwmxy0Fz7D2oV4lw_hLhLs12r",
    }

    for path, file_id in files.items():
        if not os.path.exists(path):
            st.info(f"Downloading {path}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)

    ef_model = tf.keras.models.load_model("models/best_ef_model.keras")
    resnet_model = tf.keras.models.load_model("models/best_resnet_model.keras")
    rf_model = joblib.load("models/rf_model.joblib")
    scaler = joblib.load("models/feature_scaler.joblib")

    return ef_model, resnet_model, rf_model, scaler

# ============================================
# Prediction function
# ============================================
CLASS_NAMES = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Quality_Fruits']
IMG_SIZE = (224, 224)

def predict(image_file, ef_model, resnet_model, rf_model, scaler):
    # Preprocess
    image = Image.open(image_file).convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    image_array = np.array(image_resized, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    image_tensor = tf.constant(image_batch)

    # Model predictions
    ef_proba = ef_model.predict(image_tensor, verbose=0)
    resnet_proba = resnet_model.predict(image_tensor, verbose=0)

    ef_extractor = models.Model(inputs=ef_model.input, outputs=ef_model.layers[-2].output)
    resnet_extractor = models.Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
    ef_feat = ef_extractor.predict(image_tensor, verbose=0)
    resnet_feat = resnet_extractor.predict(image_tensor, verbose=0)

    combined_feat = np.hstack([ef_feat, resnet_feat])
    combined_feat = scaler.transform(combined_feat)
    rf_proba = rf_model.predict_proba(combined_feat)

    final_proba = (ef_proba + resnet_proba + rf_proba) / 3
    predicted_class = np.argmax(final_proba, axis=1)[0]
    confidence = final_proba[0][predicted_class]

    return CLASS_NAMES[predicted_class], confidence, final_proba[0], image

# ============================================
# Streamlit UI
# ============================================
st.set_page_config(page_title="Fruit Quality Detector", page_icon="🍎", layout="centered")
st.title("🍎 Fruit Quality Detector")
st.write("Upload a fruit image to detect whether it is Good, Bad, or Mixed quality.")

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Loading models (first run may take a minute)..."):
        ef_model, resnet_model, rf_model, scaler = load_models()

    with st.spinner("Analyzing image..."):
        label, confidence, all_probs, image = predict(uploaded_file, ef_model, resnet_model, rf_model, scaler)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        # Color based on result
        color = {"Bad Quality_Fruits": "🔴", "Good Quality_Fruits": "🟢", "Mixed Quality_Fruits": "🟡"}
        st.markdown(f"### {color[label]} {label.replace('_', ' ')}")
        st.metric("Confidence", f"{confidence:.2%}")

        st.write("**All class probabilities:**")
        for i, name in enumerate(CLASS_NAMES):
            st.progress(float(all_probs[i]), text=f"{name.replace('_', ' ')}: {all_probs[i]:.2%}")