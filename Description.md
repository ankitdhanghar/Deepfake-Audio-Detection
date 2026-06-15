## Preprocessing

The dataset used is the **Fake-or-Real Dataset** (`for-norm` folder),
which contains pre-normalized audio files split into three subsets:

| Split      | Real   | Fake   |
|------------|--------|--------|
| Training   | 27,008 | 26,954 |
| Validation | 5,416  | 5,398  |
| Testing    | 2,264  | 2,370  |

All audio files are pre-normalized (volume normalized, silence
trimmed, resampled). During loading, each file is resampled to a
fixed sample rate of **16,000 Hz** using `librosa.load()` to ensure
consistency across all samples. No additional preprocessing such as
noise removal or augmentation was applied.

---

## Feature Extraction

Features are extracted using **MFCC (Mel Frequency Cepstral
Coefficients)**, a widely used representation in speech and audio
classification tasks.

**Steps:**

1. Each audio file is loaded at 16,000 Hz using `librosa.load()`
2. **40 MFCC coefficients** are computed using `librosa.feature.mfcc()`
3. The mean of each coefficient is taken across time frames using
   `np.mean(mfcc.T, axis=0)`
4. This produces a **fixed-length feature vector of 40 values**
   per audio file regardless of duration

```python
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean
```

The final feature matrix shape for training is **(N, 40)** where N
is the total number of audio files processed.

---

## Model Architecture

### Random Forest Classifier

A **Random Forest Classifier** was used as the primary model.
Random Forest is an ensemble of decision trees that aggregates
predictions via majority voting. It was chosen because:

- It handles high-dimensional feature spaces well
- It does not require feature scaling
- It provides probability estimates via `predict_proba()`
- It is robust to overfitting due to bagging

**Configuration:**

```python
RandomForestClassifier(
    n_estimators = 200,   # number of trees
    random_state = 42,
    n_jobs       = -1     # use all CPU cores
)
```

Training was performed on an 80/20 stratified split of the full
training data using `train_test_split()`.

---

## Evaluation Metrics

The model was evaluated on three levels:

- **Internal Test Set** — 20% holdout from training data
- **Validation Set** — full official validation split (10,798 samples)
- **Test Set** — full official test split (4,634 samples)

Metrics reported:

| Metric    | Description                              |
|-----------|------------------------------------------|
| Accuracy  | Overall correct predictions              |
| Precision | Correctness of positive predictions      |
| Recall    | Coverage of actual positive samples      |
| F1 Score  | Harmonic mean of Precision and Recall    |
| EER       | Equal Error Rate — threshold where False Acceptance Rate equals False Rejection Rate |
