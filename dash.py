import os
import numpy as np
import streamlit as st
import librosa
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# CONFIG (لا تغيّريها إلا لو غيرتيها في التدريب)
# =========================
SAMPLE_RATE = 16000
DURATION_DEFAULT = 3.0

N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

MODEL_PATH = "best_model.keras"
SCALER_PATH = "scaler.joblib"
CLASSES_PATH = "label_classes.npy"


# =========================
# Feature Extraction (نفس التدريب)
# =========================
def load_audio(path: str, sr=SAMPLE_RATE, duration=DURATION_DEFAULT):
    y, _ = librosa.load(path, sr=sr)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def extract_features(y: np.ndarray, sr=SAMPLE_RATE) -> np.ndarray:
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    feat = []
    feat.extend([np.mean(zcr), np.std(zcr)])
    feat.extend([np.mean(rms), np.std(rms)])

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feat.extend(mfcc_mean.tolist())
    feat.extend(mfcc_std.tolist())

    return np.array(feat, dtype=np.float32)


# =========================
# Load assets once
# =========================
@st.cache_resource
def load_assets():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, CLASSES_PATH] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing files in the app folder: " + ", ".join(missing) +
            "\nPut app.py next to: best_model.keras, scaler.joblib, label_classes.npy"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    return model, scaler, classes


def predict_from_path(audio_path: str, dur: float, model, scaler, classes):
    y = load_audio(audio_path, sr=SAMPLE_RATE, duration=dur)
    feat = extract_features(y, sr=SAMPLE_RATE).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    probs = model.predict(feat_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = str(classes[pred_idx])
    confidence = float(probs[pred_idx])
    return probs, pred_label, confidence


# =========================
# UI
# =========================
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

st.title("🎙️ Speech Emotion Recognition")
st.caption("Choose: Record your voice or Upload an audio file. (RAVDESS - 8 emotions)")

model, scaler, classes = load_assets()

with st.sidebar:
    st.header("Settings")
    dur = st.slider("Clip duration (seconds)", 1.0, 5.0, float(DURATION_DEFAULT), 0.5)
    st.write("Tip: keep this equal to training duration (default 3.0s).")

mode = st.radio("Input method", ["🎤 Record", "📁 Upload"], horizontal=True)

temp_path = None

if mode == "🎤 Record":
    st.subheader("🎤 Record your voice")
    audio_file = st.audio_input("Click to record")

    if audio_file is None:
        st.info("Record")
    else:
        temp_path = "recorded_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_file.getvalue())
        st.audio(temp_path)

elif mode == "📁 Upload":
    st.subheader("📁 Upload audio file")
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a"])
    if uploaded is None:
        st.info("upload")
    else:
        temp_path = "uploaded_audio"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        st.audio(temp_path)

# If we have an audio file, predict
if temp_path is not None:
    probs, pred_label, confidence = predict_from_path(temp_path, dur, model, scaler, classes)

    st.subheader("✅ Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Emotion", pred_label)
    with col2:
        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0))

    st.subheader("Top-3 predictions")
    top3 = np.argsort(probs)[::-1][:3]
    for r, i in enumerate(top3, start=1):
        st.write(f"**{r}. {classes[i]}** — {probs[i]*100:.2f}%")

    st.subheader("All class probabilities")
    fig = plt.figure()
    plt.bar(classes, probs)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Probability")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Model: best_model.keras | Preprocessing: scaler.joblib | Labels: label_classes.npy")

    # Cleanup temp
    try:
        os.remove(temp_path)
    except Exception:
        pass
