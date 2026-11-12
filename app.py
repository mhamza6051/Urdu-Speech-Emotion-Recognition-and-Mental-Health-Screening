import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch

from transformers import Wav2Vec2Processor, Wav2Vec2Model


st.set_page_config(
    page_title="Urdu Speech Mental Health Detection",
    page_icon="🎙️",
    layout="wide"
)


st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f7fbff 0%, #eef4ff 100%);
}

.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #123c69;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 1rem;
    color: #4b5563;
    margin-bottom: 1.2rem;
}

.section-card {
    background: white;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}

.pred-card {
    background: linear-gradient(135deg, #edf6ff 0%, #f7fbff 100%);
    padding: 1rem;
    border-radius: 18px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

.compare-card {
    background: white;
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    border-top: 4px solid #3b82f6;
}

.note-box {
    background: #fff8e7;
    padding: 1rem;
    border-radius: 16px;
    border-left: 6px solid #f59e0b;
    color: #5b4a00;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.small-note {
    font-size: 0.9rem;
    color: #6b7280;
}

.metric-label {
    font-size: 0.95rem;
    color: #6b7280;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #111827;
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    font-weight: 700;
    border: none;
    padding: 0.75rem 1rem;
    box-shadow: 0 8px 18px rgba(37,99,235,0.25);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    color: white;
}
</style>
""", unsafe_allow_html=True)


SVM_MODEL_PATH = "models/final_svm_rbf_model.pkl"
SVM_LABEL_ENCODER_PATH = "models/label_encoder.pkl"
SVM_FEATURE_COLUMNS_PATH = "models/final_feature_columns.pkl"

W2V_MODEL_PATH = "models/best_w2v_model.pkl"
W2V_LABEL_ENCODER_PATH = "models/w2v_label_encoder.pkl"
W2V_FEATURE_COLUMNS_PATH = "models/w2v_feature_cols.pkl"

W2V_HF_NAME = "facebook/wav2vec2-base"

TARGET_SR_SVM = 44100
TARGET_DURATION = 3.0
TARGET_LENGTH = int(TARGET_SR_SVM * TARGET_DURATION)
TARGET_SR_W2V = 16000

LABEL_DISPLAY = {
    "anxiety": "Anxiety",
    "depression": "Depression",
    "stress": "Stress",
    "healthy": "Healthy"
}


@st.cache_resource
def load_svm_artifacts():
    model = joblib.load(SVM_MODEL_PATH)
    label_encoder = joblib.load(SVM_LABEL_ENCODER_PATH)
    feature_columns = joblib.load(SVM_FEATURE_COLUMNS_PATH)
    return model, label_encoder, feature_columns

@st.cache_resource
def load_w2v_artifacts():
    model = joblib.load(W2V_MODEL_PATH)
    label_encoder = joblib.load(W2V_LABEL_ENCODER_PATH)
    feature_columns = joblib.load(W2V_FEATURE_COLUMNS_PATH)
    processor = Wav2Vec2Processor.from_pretrained(W2V_HF_NAME)
    hf_model = Wav2Vec2Model.from_pretrained(W2V_HF_NAME)
    hf_model.eval()
    return model, label_encoder, feature_columns, processor, hf_model


def peak_normalize(y):
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y

def pad_or_truncate(y, target_length=TARGET_LENGTH):
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]
    return y

def preprocess_for_display(filepath, sr=TARGET_SR_SVM):
    y, sr = librosa.load(filepath, sr=sr)
    return y, sr

def extract_svm_features_from_trimmed_signal(y_original, sr, top_db=25):
    y_trimmed, _ = librosa.effects.trim(y_original, top_db=top_db)

    if len(y_trimmed) < int(0.3 * sr):
        y_trimmed = y_original.copy()

    y_trimmed = peak_normalize(y_trimmed)

    features = {}

    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

    rms = librosa.feature.rms(y=y_trimmed)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)

    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))
    features["bandwidth_mean"] = float(np.mean(bandwidth))
    features["bandwidth_std"] = float(np.std(bandwidth))
    features["rolloff_mean"] = float(np.mean(rolloff))
    features["rolloff_std"] = float(np.std(rolloff))

    for i in range(contrast.shape[0]):
        features[f"contrast_{i}_mean"] = float(np.mean(contrast[i]))
        features[f"contrast_{i}_std"] = float(np.std(contrast[i]))

    f0, voiced_flag, _ = librosa.pyin(
        y_trimmed,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(valid_f0) > 0:
        features["pitch_mean"] = float(np.mean(valid_f0))
        features["pitch_std"] = float(np.std(valid_f0))
        features["pitch_min"] = float(np.min(valid_f0))
        features["pitch_max"] = float(np.max(valid_f0))
    else:
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0

    if voiced_flag is not None:
        features["voiced_ratio"] = float(np.mean(voiced_flag.astype(float)))
    else:
        features["voiced_ratio"] = 0.0

    original_duration_sec = librosa.get_duration(y=y_original, sr=sr)
    trimmed_duration_sec = librosa.get_duration(y=y_trimmed, sr=sr)
    silence_removed_sec = original_duration_sec - trimmed_duration_sec
    silence_removed_ratio = silence_removed_sec / original_duration_sec if original_duration_sec > 0 else 0.0

    features["original_duration_sec"] = float(original_duration_sec)
    features["trimmed_duration_sec"] = float(trimmed_duration_sec)
    features["silence_removed_sec"] = float(silence_removed_sec)
    features["silence_removed_ratio"] = float(silence_removed_ratio)

    return features

def prepare_svm_input(filepath, feature_columns):
    y_original, sr = librosa.load(filepath, sr=TARGET_SR_SVM)
    feats = extract_svm_features_from_trimmed_signal(y_original, sr)
    df = pd.DataFrame([feats])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_columns]

def extract_wav2vec2_embedding(filepath, processor, hf_model, target_sr=TARGET_SR_W2V):
    y, _ = librosa.load(filepath, sr=target_sr)

    inputs = processor(
        y,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = hf_model(**inputs)
        hidden_states = outputs.last_hidden_state

    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def prepare_w2v_input(filepath, feature_columns, processor, hf_model):
    emb = extract_wav2vec2_embedding(filepath, processor, hf_model)
    emb_dict = {f"w2v_{i}": float(emb[i]) for i in range(len(emb))}
    df = pd.DataFrame([emb_dict])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_columns]

def predict_with_model(model, label_encoder, X_df):
    pred_encoded = model.predict(X_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_df)[0]
        class_names = label_encoder.inverse_transform(np.arange(len(probs)))
        prob_df = pd.DataFrame({
            "Label": [LABEL_DISPLAY.get(c, c) for c in class_names],
            "Probability": probs
        }).sort_values("Probability", ascending=False)
    else:
        prob_df = pd.DataFrame({
            "Label": [LABEL_DISPLAY.get(pred_label, pred_label)],
            "Probability": [1.0]
        })

    return pred_label, prob_df

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_mel_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

def plot_probability_chart(prob_df, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(prob_df["Label"], prob_df["Probability"])
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.invert_yaxis()
    st.pyplot(fig)

def label_to_display_text(label):
    return LABEL_DISPLAY.get(label, label)


st.markdown('<div class="main-title">🎙️ Urdu Speech Mental Health Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Analyze Urdu speech using handcrafted acoustic features and wav2vec2 embeddings.</div>',
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("⚙️ App Controls")

    model_choice = st.selectbox(
        "Select model mode",
        ["SVM (Main Model)", "wav2vec2 + LR", "Compare Both"]
    )

    uploaded_file = st.file_uploader(
        "Upload WAV audio",
        type=["wav"]
    )

    analyze_clicked = st.button("🔍 Analyze Audio")

    st.markdown("---")
    st.markdown("**Main deployed model:** SVM")
    st.markdown("**Advanced comparison model:** wav2vec2 + LR")


st.markdown("""
<div class="note-box">
<b>Disclaimer:</b> This system provides an early mental health indication from Urdu speech patterns. 
It is intended for research and educational use only and is not a medical diagnosis tool.
</div>
""", unsafe_allow_html=True)


if uploaded_file is None:
    st.info("Upload a WAV file from the sidebar.")
elif not analyze_clicked:
    st.info("Click the **Analyze Audio** button to run the model.")
else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    try:
        y_display, sr_display = preprocess_for_display(temp_audio_path, sr=TARGET_SR_SVM)
        duration_sec = librosa.get_duration(y=y_display, sr=sr_display)

        info_col, wave_col = st.columns([1, 1])

        with info_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📌 Audio Details")
            st.audio(temp_audio_path)
            st.markdown(f'<div class="metric-label">Filename</div><div class="metric-value" style="font-size:1rem;">{uploaded_file.name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Duration</div><div class="metric-value">{duration_sec:.2f} sec</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Sample Rate</div><div class="metric-value">{sr_display} Hz</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with wave_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📈 Waveform")
            plot_waveform(y_display, sr_display)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🎼 Spectrogram")
        plot_mel_spectrogram(y_display, sr_display)
        st.markdown('</div>', unsafe_allow_html=True)

        svm_pred_label = None
        svm_prob_df = None
        w2v_pred_label = None
        w2v_prob_df = None

        if model_choice in ["SVM (Main Model)", "Compare Both"]:
            svm_model, svm_le, svm_feature_cols = load_svm_artifacts()
            svm_input = prepare_svm_input(temp_audio_path, svm_feature_cols)
            svm_pred_label, svm_prob_df = predict_with_model(svm_model, svm_le, svm_input)

        if model_choice in ["wav2vec2 + LR", "Compare Both"]:
            with st.spinner("Loading wav2vec2 and extracting embeddings..."):
                w2v_model, w2v_le, w2v_feature_cols, processor, hf_model = load_w2v_artifacts()
                w2v_input = prepare_w2v_input(temp_audio_path, w2v_feature_cols, processor, hf_model)
                w2v_pred_label, w2v_prob_df = predict_with_model(w2v_model, w2v_le, w2v_input)

        if model_choice == "SVM (Main Model)":
            st.markdown('<div class="pred-card">', unsafe_allow_html=True)
            st.subheader("✅ SVM Prediction")
            st.write(f"**Predicted Class:** {label_to_display_text(svm_pred_label)}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📊 Confidence Scores")
            st.dataframe(svm_prob_df, use_container_width=True)
            plot_probability_chart(svm_prob_df, "SVM Confidence Scores")
            st.markdown('</div>', unsafe_allow_html=True)

        elif model_choice == "wav2vec2 + LR":
            st.markdown('<div class="pred-card">', unsafe_allow_html=True)
            st.subheader("✅ wav2vec2 + LR Prediction")
            st.write(f"**Predicted Class:** {label_to_display_text(w2v_pred_label)}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📊 Confidence Scores")
            st.dataframe(w2v_prob_df, use_container_width=True)
            plot_probability_chart(w2v_prob_df, "wav2vec2 Confidence Scores")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            left, right = st.columns(2)

            with left:
                st.markdown('<div class="compare-card">', unsafe_allow_html=True)
                st.subheader("✅ SVM Result")
                st.write(f"**Predicted Class:** {label_to_display_text(svm_pred_label)}")
                st.dataframe(svm_prob_df, use_container_width=True)
                plot_probability_chart(svm_prob_df, "SVM Confidence")
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                st.markdown('<div class="compare-card">', unsafe_allow_html=True)
                st.subheader("✅ wav2vec2 + LR Result")
                st.write(f"**Predicted Class:** {label_to_display_text(w2v_pred_label)}")
                st.dataframe(w2v_prob_df, use_container_width=True)
                plot_probability_chart(w2v_prob_df, "wav2vec2 Confidence")
                st.markdown('</div>', unsafe_allow_html=True)

            comparison_df = pd.DataFrame([
                {
                    "Model": "SVM",
                    "Predicted Class": label_to_display_text(svm_pred_label),
                    "Top Confidence": float(svm_prob_df.iloc[0]["Probability"])
                },
                {
                    "Model": "wav2vec2 + LR",
                    "Predicted Class": label_to_display_text(w2v_pred_label),
                    "Top Confidence": float(w2v_prob_df.iloc[0]["Probability"])
                }
            ])

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("🆚 Comparison Summary")
            st.dataframe(comparison_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="small-note">
            The SVM model is the main selected model based on overall performance. 
            The wav2vec2 model is included for advanced comparison.
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Error processing audio: {e}")

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)