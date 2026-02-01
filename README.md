# Speech Emotion Recognition (SER) using RAVDESS Dataset 🎙️

This project implements a complete **Speech Emotion Recognition (SER)** pipeline using the **RAVDESS Emotional Speech Audio dataset**. The goal is to classify human emotions from speech signals using **hand-crafted audio features** and a **Multi-Layer Perceptron (MLP) classifier**.

---

## 📁 Dataset

- Dataset: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- Format: WAV audio files from multiple actors.
- Emotions used: **neutral, calm, happy, angry, disgust, surprise**  
  *(Note: 'fear' and 'sad' removed from this version)*

---

## Features Extracted

The following **audio features** are extracted from each WAV file:

1. **Zero Crossing Rate (ZCR)** – measures the rate at which the signal changes sign.
2. **Spectral Centroid** – indicates the "center of mass" of the spectrum.
3. **MFCCs (Mel-Frequency Cepstral Coefficients)** – 40 coefficients (mean & std per coefficient).
4. **Chroma Features** – captures harmonic content.
5. **Mel Spectrogram** – 128 Mel bands in log scale (mean & std per band).

**Tools Used:** `librosa` for audio processing, `numpy` for numerical operations.

---

## Model

- **Type:** Multi-Layer Perceptron (MLP) classifier
- **Library:** `scikit-learn`
- **Pipeline:**
  1. `StandardScaler` – feature normalization
  2. `MLPClassifier` – trained on audio features
- **Hyperparameter Tuning:** `GridSearchCV` for:
  - `hidden_layer_sizes` (e.g., (256,), (512,256))
  - `learning_rate_init` (0.001, 0.0005)
  - `alpha` (0.0001, 0.001, 0.01)
  - `activation` (relu, tanh)

- **Training/Test Split:** 80% training, 20% testing
- **Random State:** 42 for reproducibility
