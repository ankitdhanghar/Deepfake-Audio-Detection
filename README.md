# Deepfake Audio Detection

A machine learning pipeline to detect AI-generated (deepfake) audio from genuine human speech using MFCC feature extraction and a Random Forest classifier.

---
Demo video
https://drive.google.com/file/d/1i6zOZNrpXjZBURTo3TUj1OUnkzYt9tfq/view?usp=sharing

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

---

## Overview

With the rapid advancement of AI voice synthesis, distinguishing genuine human speech from AI-generated audio has become a critical challenge. This project builds a binary classification system that:

- Accepts a `.wav` audio file as input
- Extracts MFCC (Mel Frequency Cepstral Coefficients) features
- Predicts whether the audio is **Genuine (Real)** or **Deepfake (AI-Generated)**
- Returns a confidence score with the prediction

---

## Dataset

**The Fake-or-Real Dataset**
- Source: [Kaggle — mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- Folder used: `for-norm` (pre-normalized audio)
- Total size: ~17 GB
- Total files: ~170,000 audio files

### Data Split

| Split      | Real Files | Fake Files | Total  |
|------------|------------|------------|--------|
| Training   | 27,008     | 26,954     | 53,962 |
| Validation | 5,416      | 5,398      | 10,814 |
| Testing    | 2,264      | 2,370      | 4,634  |

### Labels

| Label | Class   |
|-------|---------|
| 0     | Real    |
| 1     | Fake    |

---

## Pipeline

```
Raw Audio Files (.wav)
        │
        ▼
  Load at 16,000 Hz
        │
        ▼
  Extract 40 MFCC Coefficients
        │
        ▼
  Compute Mean across Time Frames
        │
        ▼
  Feature Vector (40-dim)
        │
        ▼
  Random Forest Classifier
        │
        ▼
  Prediction + Confidence Score
```

---

## Preprocessing

All audio files in the `for-norm` folder are pre-normalized, meaning:

- Volume normalized across all samples
- Silence trimmed from start and end
- Converted to mono channel
- Resampled to 16,000 Hz

During feature extraction, each file is loaded at a fixed sample rate of **16,000 Hz** using `librosa.load()` to ensure consistency. No additional augmentation or noise injection was applied.

---

## Feature Extraction

**MFCC (Mel Frequency Cepstral Coefficients)** were used as the primary feature representation. MFCCs capture the spectral envelope of audio signals and are widely used in speech recognition and audio classification tasks.

### Extraction Steps

1. Load audio at 16,000 Hz
2. Compute 40 MFCC coefficients using `librosa.feature.mfcc()`
3. Take the mean of each coefficient across all time frames
4. Produce a fixed-length **40-dimensional feature vector** per file

```python
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean
```

### Why MFCC?

- Captures timbral and vocal tract characteristics
- Compact representation — 40 values per file regardless of duration
- Proven effectiveness in speech-based classification tasks
- Computationally efficient for large datasets

---

## Model Architecture

### Random Forest Classifier

A **Random Forest** ensemble model was selected as the classifier.

```python
RandomForestClassifier(
    n_estimators = 200,
    random_state = 42,
    n_jobs       = -1
)
```

### Why Random Forest?

- Ensemble of 200 decision trees — reduces variance via bagging
- Does not require feature scaling
- Handles high-dimensional input (40 features) effectively
- Provides class probability estimates via `predict_proba()`
- Robust to overfitting on large datasets
- Parallelizable across all CPU cores (`n_jobs=-1`)

### Training Strategy

- Training data split 80/20 using stratified `train_test_split()`
- Trained on full dataset — 53,962 audio files
- Evaluated separately on official validation and test splits



### Key Observations

- High validation accuracy (97.90%) reflects strong performance on normalized audio
- Test accuracy (88.51%) represents real-world generalization on truly unseen data
- Balanced performance across both classes — no bias toward Real or Fake
- The gap between validation and test accuracy is expected since the model was trained and validated on the same normalized distribution

---

## Project Structure

```
Deepfake-Audio-Detection/
│
├── notebook.ipynb          # Full training and evaluation notebook
├── model.pkl               # Trained Random Forest model
├── predict.py              # CLI inference script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Installation

```bash
git clone https://github.com/ankitdhanghar/Deepfake-Audio-Detection.git
cd Deepfake-Audio-Detectio

pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
librosa
scikit-learn
matplotlib
seaborn
streamlit
joblib
tqdm
soundfile
```

---

### Streamlit Web App

```bash
streamlit run app.py
```

Upload a `.wav`, `.mp3`, or `.flac` file through the browser interface to get an instant prediction.

---

## Evaluation Metrics Explained

| Metric    | Description |
|-----------|-------------|
| Accuracy  | Percentage of correct predictions out of total |
| Precision | Of all files predicted as Fake, how many were actually Fake |
| Recall    | Of all actual Fake files, how many were correctly detected |
| F1 Score  | Harmonic mean of Precision and Recall — balances both |
| EER       | Equal Error Rate — point where False Accept Rate equals False Reject Rate. Lower is better |

---

## Results

| Metric    | Real (0) | Fake (1) | Overall |
|-----------|----------|----------|---------|
| Precision | 1.00     | 1.00     | 0.89    |
| Recall    | 0.87     | 0.90     | 0.89    |
| F1 Score  | 0.93     | 0.95     | 0.89    |
| Accuracy  | 86.93%   |  90.04%  | 88.51%  |

| Metric    | Overall  | 
|-----------|----------|
|  Accuracy | 88.51%   | 
|    EER    | 11.53%   |

---

## Limitations

- Trained on pre-normalized audio — performance may vary on raw, noisy, or compressed audio
- MFCC mean aggregation loses temporal information within each file
- Model has not been tested on audio generated by latest voice synthesis models

---


