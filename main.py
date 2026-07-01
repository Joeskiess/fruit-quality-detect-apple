import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import joblib
import gdown
import os
from PIL import Image

# ============================================
# Download and load models from Google Drive
# ============================================
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)

    # Paste your actual file IDs here
    files = {
    "models/best_ef_model.keras": st.secrets["EF_MODEL_ID"],
    "models/best_resnet_model.keras": st.secrets["RESNET_MODEL_ID"],
    "models/rf_model.joblib": st.secrets["RF_MODEL_ID"],
    "models/feature_scaler.joblib": st.secrets["SCALER_ID"],
    }

    for path, file_id in files.items():
        if not os.path.exists(path):
            try:
                gdown.download(id=file_id, output=path, quiet=False)
            except Exception as e:
                st.error(f"Failed to download {path}: {e}")
                st.stop()
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                st.error(f"Download failed or empty file: {path}")
                st.stop()

    ef_model = tf.keras.models.load_model("models/best_ef_model.keras")
    resnet_model = tf.keras.models.load_model("models/best_resnet_model.keras")
    rf_model = joblib.load("models/rf_model.joblib")
    scaler = joblib.load("models/feature_scaler.joblib")

    return ef_model, resnet_model, rf_model, scaler

# ============================================
# Prediction
# ============================================
CLASS_NAMES = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed Quality_Fruits']
IMG_SIZE = (224, 224)

def predict(image_file, ef_model, resnet_model, rf_model, scaler):
    image = Image.open(image_file).convert("RGB")
    image_resized = image.resize(IMG_SIZE)
    image_array = np.array(image_resized, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    image_tensor = tf.constant(image_batch)

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
# UI
# ============================================
st.set_page_config(page_title="Fruit Quality Detector", page_icon="🍎", layout="centered")
st.title("🍎 Fruit Quality Detector")
st.write("Upload a fruit image to check whether it is Good, Bad, or Mixed quality.")

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Loading models — first run takes a minute..."):
        ef_model, resnet_model, rf_model, scaler = load_models()

    with st.spinner("Analyzing..."):
        label, confidence, all_probs, image = predict(
            uploaded_file, ef_model, resnet_model, rf_model, scaler
        )

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", width=300)
    with col2:
        color = {"Bad Quality_Fruits": "🔴", "Good Quality_Fruits": "🟢", "Mixed Quality_Fruits": "🟡"}
        st.markdown(f"### {color[label]} {label.replace('_', ' ')}")
        st.metric("Confidence", f"{confidence:.2%}")
        st.write("**All class probabilities:**")
        for i, name in enumerate(CLASS_NAMES):
            st.progress(float(all_probs[i]), text=f"{name.replace('_', ' ')}: {all_probs[i]:.2%}")
