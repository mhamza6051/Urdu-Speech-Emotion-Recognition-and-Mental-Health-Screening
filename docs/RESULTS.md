# Full Results — Speech Emotion Recognition

All figures are taken verbatim from the project notebook outputs and the result CSVs in `models/`.

## Dataset

- 400 Urdu speech clips, balanced: Angry 100 · Happy 100 · Neutral 100 · Sad 100
- Original durations ~2–3 s at 44.1 kHz; after preprocessing every clip is exactly 3.0 s (132,300 samples)
- Test split: 80 clips (support 80 in all classification reports below)

## Approach A — handcrafted features (test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **SVM (RBF) — FINAL** | **0.9125** | 0.9154 | 0.9125 | **0.9134** |
| Random Forest | 0.8875 | 0.8904 | 0.8875 | 0.8884 |
| XGBoost | 0.8750 | 0.8787 | 0.8750 | 0.8761 |
| Logistic Regression | 0.8375 | 0.8369 | 0.8375 | 0.8364 |

- Final SVM detailed report: accuracy 0.91, macro P/R/F1 = 0.92 / 0.91 / 0.91
- 5-fold CV F1 (SVM): [0.9503, 0.9005, 0.9875, 0.9625, 0.9625] → **mean 0.9527 ± 0.0287**
- Regularization tuning (C=10, gamma='scale', best CV F1 0.9564): test weighted F1 **fell** to 0.9003 vs. original 0.9134 → original SVM kept as final (`models/svm_regularization_comparison.csv`)

## Approach B — spectrogram CNN (failed, documented)

| Attempt | Test accuracy | Outcome |
|---|---|---|
| Compact CNN | 0.25 | Chance level; validation accuracy stuck at 0.25 while training accuracy rose to ~0.65 and val loss climbed 1.36 → 6.20 (textbook overfitting) |
| Improved CNN (augmentation, class weights, better callbacks) | 0.25 | Same collapse |

Verdict recorded in the project: *"CNN — severe overfitting, rejected."*

## Approach C — wav2vec2 embeddings + shallow classifiers

| Model | Weighted F1 |
|---|---|
| **Logistic Regression — best wav2vec2 model** | **0.8769** (test accuracy 0.875–0.88, macro F1 0.88) |
| SVM (RBF) | 0.8394 |
| Random Forest | 0.8266 |
| XGBoost | 0.7776 |

- LR on wav2vec2 embeddings, 5-fold CV F1: [0.8368, 0.8736, 0.8639, 0.9132, 0.8485] → **mean 0.8672 ± 0.0262**

## Final comparison (from `models/final_model_comparison_updated.csv`)

| Model | F1 | Remark |
|---|---|---|
| SVM_RBF (handcrafted) — FINAL | 0.9134 | Best overall performance |
| Logistic Regression (wav2vec2) | 0.8769 | Strong pretrained-embedding model |
| SVM_RBF (wav2vec2) | 0.8394 | Good but weaker than LR |
| Random Forest (wav2vec2) | 0.8266 | Moderate |
| XGBoost (wav2vec2) | 0.7776 | Lowest of the wav2vec2 group |
| CNN (spectrogram) | — | Failed / rejected (chance-level test accuracy) |

## Clinical proxy mapping (demo only — NOT diagnostic)

Angry → anxiety · Happy → healthy · Neutral → stress · Sad → depression (`models/clinical_label_mapping.csv`)
