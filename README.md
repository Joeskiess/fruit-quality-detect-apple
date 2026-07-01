# fruit-quality-detect-apple
The Innovations are limitless, the horizons are endless


Fruit Quality Detector

Streamlit app that classifies apple fruit images as Good, Bad, or Mixed quality using an ensemble of EfficientNet, ResNet, and a Random Forest on combined features.

Dataset Used
https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

## Live app
https://fruit-quality-detect-apple-bz2zoomogpgtywtzr9hj6e.streamlit.app/

## How it works
Three models vote: EfficientNet and ResNet each give a softmax prediction, and a Random Forest classifies pooled features from both networks after scaling. Final prediction is the average of all three probability vectors.

## Model files
Weights aren't in this repo (too big for GitHub). They're pulled from Google Drive at runtime via `gdown`, cached after first download.

Files needed:
- `best_ef_model.keras`
- `best_resnet_model.keras`
- `rf_model.joblib`
- `feature_scaler.joblib`

## Setup

### 1. Upload your model files to Google Drive
Set sharing to **Anyone with the link → Viewer** for each file. `gdown` needs view access only — nothing more.

Grab each file ID from the share link:
https://drive.google.com/file/d/FILE_ID_HERE/view


### 2. Add secrets

**Local dev** — create `.streamlit/secrets.toml` (gitignored, never commit this):
```toml
EF_MODEL_ID = "your_id"
RESNET_MODEL_ID = "your_id"
RF_MODEL_ID = "your_id"
SCALER_ID = "your_id"
```

**Streamlit Cloud** — Settings → Secrets → paste the same block. Saves and reboots the app automatically.

### 3. Install and run
```bash
pip install -r requirements.txt
streamlit run main.py
```

## Deploy
1. Push to GitHub (public repo is fine — no IDs or keys in the code, they live in Secrets)
2. [share.streamlit.io](https://share.streamlit.io) → New app → point at this repo
3. Add secrets before first run (see above)

## Notes
- First load is slow — models download once, then get cached via `@st.cache_resource`.
- Free tier on Streamlit Cloud caps at 1GB RAM. If your models are large, watch for OOM crashes.
- View-only Drive access means anyone with the file ID can download the weights. Fine for a public demo; not fine if the weights are meant to stay private — use a service account with the Drive API in that case.
