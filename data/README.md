# Audio data — not stored in Git

The project works on **400 preprocessed WAV clips (~103 MB total)**: 100 clips each of Angry / Happy / Neutral / Sad Urdu speech, standardized to exactly 3.0 s by the notebook's preprocessing pipeline.

The audio files themselves are **not stored in this repository** (repo hygiene for 400 binary files). What *is* included:

- `metadata/processed_metadata.csv` — the full per-clip record: filename, emotion label, clinical proxy label, sample rate, original/trimmed/final durations and sample counts for all 400 clips
- `models/wav2vec2_embeddings.csv` — the extracted wav2vec2 embeddings for all 400 clips
- `models/features_v2_trimmed.csv` — the handcrafted feature table

To reproduce the audio set: run the preprocessing section of the notebook in `notebooks/` on the source speech-emotion dataset (clips organized as `Angry/`, `Happy/`, `Neutral/`, `Sad/` folders); it regenerates the `preprocessed_audio/` tree exactly.
