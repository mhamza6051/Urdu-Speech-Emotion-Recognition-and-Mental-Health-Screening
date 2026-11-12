# Urdu Speech Emotion Recognition & Mental-Health Screening (Research Demo)

A speech-emotion system for **Urdu voice clips** that compares three modelling approaches head-to-head — handcrafted audio features, **wav2vec2** pretrained embeddings, and a spectrogram **CNN** — and reports all three honestly, including the one that failed completely. A Streamlit app ("Urdu Speech Mental Health Detection") serves the two working models.

> ⚠️ **Important:** this is a **research/educational demo, not a medical device**. The "clinical labels" are simple proxy mappings of emotion categories (Angry→anxiety, Happy→healthy, Neutral→stress, Sad→depression) used to explore screening-style presentation. They are **not diagnoses**, and this system must never be used to assess a real person's mental health.

## Results (400 clips: 100 each of Angry / Happy / Neutral / Sad, standardized to 3 s)

| Approach | Model | Test Accuracy | Weighted F1 |
|---|---|---|---|
| **Handcrafted features — FINAL** | **SVM (RBF)** | **0.9125** | **0.9134** |
| wav2vec2 embeddings | Logistic Regression | 0.875 | 0.8769 |
| wav2vec2 embeddings | SVM (RBF) | — | 0.8394 |
| wav2vec2 embeddings | Random Forest | — | 0.8266 |
| wav2vec2 embeddings | XGBoost | — | 0.7776 |
| Spectrogram CNN | CNN (two attempts) | **0.25 (chance)** | rejected ❌ |

> **The honest findings:** (1) classical features + SVM beat deep embeddings on this small dataset; (2) the spectrogram CNN collapsed to chance level (25% on 4 classes) in both attempts despite augmentation and class weights — severe overfitting on 400 clips, documented and rejected, not hidden; (3) a regularized/tuned SVM scored a higher CV F1 (0.9564) but a *lower* test F1 (0.9003) than the original SVM (0.9134), so the **original** SVM is the deployed model.

5-fold CV of the final SVM: **F1 0.9527 ± 0.0287**. Full numbers: [docs/RESULTS.md](docs/RESULTS.md)

## How it works

1. **Preprocessing** — silence trimming and duration standardization: every clip normalized to exactly 3.0 s (metadata for all 400 clips preserved in `metadata/`)
2. **Approach A — handcrafted features:** MFCC/prosody-style features → SVM-RBF and baselines (RF, XGBoost, LR)
3. **Approach B — spectrogram CNN:** mel-spectrogram images → compact CNN, then an improved CNN with augmentation + class weights (both failed — see Results)
4. **Approach C — wav2vec2 embeddings:** pretrained wav2vec2 representations (768-d, stored in `models/wav2vec2_embeddings.csv`) → shallow classifiers; LR best
5. **App** — upload/record a clip → emotion prediction from the SVM model or the wav2vec2 model, with the proxy screening label shown alongside a non-diagnostic framing

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

(The wav2vec2 mode downloads the pretrained `facebook/wav2vec2` processor/model from Hugging Face on first run.)

## Repository structure

```
app.py                         Streamlit app (dual-model: SVM + wav2vec2)
notebooks/                     Full 200-cell pipeline: preprocessing → 3 approaches → comparison
models/                        Final SVM, wav2vec2 LR model, encoders, embeddings, result tables
metadata/                      Per-clip metadata for all 400 clips (durations, labels, mapping)
data/README.md                 Why the audio files themselves are not in Git
docs/RESULTS.md                Complete verified metrics
```

## Author

**Muhammad Hamza** — Data Scientist · [GitHub @mhamza6051](https://github.com/mhamza6051)
