# %% [markdown]
# # 🎙️ Speech Emotion Recognition App
# This Streamlit app allows users to record their voice or upload an audio file and predicts the emotional state using a pre-trained model.
# The model is trained on RAVDESS dataset with 8 emotions.
# 
# Features: MFCCs, Zero Crossing Rate, RMS Energy
# Classifier: MLP or Keras neural network

# %% [markdown]
# 1. Libraries

# %%
import os
import numpy as np
import streamlit as st
import librosa
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# %% [markdown]
# 2. Configuration
# These settings should match the ones used during model training.

# %%
SAMPLE_RATE = 16000
DURATION_DEFAULT = 3.0  # default clip duration in seconds

N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

MODEL_PATH = "best_model.keras"
SCALER_PATH = "scaler.joblib"
CLASSES_PATH = "label_classes.npy"

# %% [markdown]
# 3. Feature Extraction Functions
# The features are extracted to match training:
# - Zero Crossing Rate (ZCR)
# - RMS Energy
# - MFCCs (mean & std)

# %%
def load_audio(path: str, sr=SAMPLE_RATE, duration=DURATION_DEFAULT):
    """Load an audio file and pad/truncate to the target duration"""
    y, _ = librosa.load(path, sr=sr)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def extract_features(y: np.ndarray, sr=SAMPLE_RATE) -> np.ndarray:
    """Extract audio features: ZCR, RMS, MFCCs"""
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    feat = []
    # ZCR statistics
    feat.extend([np.mean(zcr), np.std(zcr)])
    # RMS statistics
    feat.extend([np.mean(rms), np.std(rms)])
    # MFCC mean & std
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feat.extend(mfcc_mean.tolist())
    feat.extend(mfcc_std.tolist())

    return np.array(feat, dtype=np.float32)

# %% [markdown]
# 4. Load Model and Assets
# These are cached to avoid reloading on each interaction.

# %%
@st.cache_resource
def load_assets():
    """Load model, scaler, and class labels"""
    missing = [p for p in [MODEL_PATH, SCALER_PATH, CLASSES_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing files: " + ", ".join(missing) +
            "\nMake sure the app.py is next to best_model.keras, scaler.joblib, label_classes.npy"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    return model, scaler, classes

# %% [markdown]
# 5. Prediction Function

# %%
def predict_from_path(audio_path: str, dur: float, model, scaler, classes):
    """
    Predict emotion from an audio file.
    Returns:
        - probs: probabilities for all classes
        - pred_label: predicted class
        - confidence: confidence score
    """
    y = load_audio(audio_path, sr=SAMPLE_RATE, duration=dur)
    feat = extract_features(y, sr=SAMPLE_RATE).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    probs = model.predict(feat_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = str(classes[pred_idx])
    confidence = float(probs[pred_idx])
    return probs, pred_label, confidence

# %% [markdown]
# 6. Streamlit UI Setup

# %%
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.caption("Record your voice or upload an audio file. Predicts emotions (RAVDESS - 8 classes).")

# Load model and preprocessing assets
model, scaler, classes = load_assets()

# Sidebar: user settings
with st.sidebar:
    st.header("Settings")
    dur = st.slider("Clip duration (seconds)", 1.0, 5.0, float(DURATION_DEFAULT), 0.5)
    st.write("Keep this the same as training duration.")

# Main UI: choose input method
mode = st.radio("Input method", ["🎤 Record", "📁 Upload"], horizontal=True)
temp_path = None

# %% [markdown]
# 7. Handle Audio Input

# %%
if mode == "🎤 Record":
    st.subheader("🎤 Record your voice")
    audio_file = st.audio_input("Click to record")

    if audio_file:
        temp_path = "recorded_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_file.getvalue())
        st.audio(temp_path)
    else:
        st.info("Press Record and speak. Prediction will appear after recording.")

elif mode == "📁 Upload":
    st.subheader("📁 Upload audio file")
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a"])
    if uploaded:
        temp_path = "uploaded_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        st.audio(temp_path)
    else:
        st.info("Upload an audio file to see predictions.")

# %% [markdown]
# 8. Make Predictions and Display Results

# %%
if temp_path is not None:
    probs, pred_label, confidence = predict_from_path(temp_path, dur, model, scaler, classes)

    st.subheader("✅ Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Emotion", pred_label)
    with col2:
        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0))

    st.subheader("Top-3 Predictions")
    top3 = np.argsort(probs)[::-1][:3]
    for r, i in enumerate(top3, start=1):
        st.write(f"**{r}. {classes[i]}** — {probs[i]*100:.2f}%")

    st.subheader("All Class Probabilities")
    fig = plt.figure()
    plt.bar(classes, probs)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Probability")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Model: best_model.keras | Scaler: scaler.joblib | Labels: label_classes.npy")

    # Cleanup temporary file
    try:
        os.remove(temp_path)
    except Exception:
        pass
