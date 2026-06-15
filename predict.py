
import sys
import joblib
import librosa
import numpy as np

model = joblib.load("model.pkl")

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=40)
    feature = np.mean(mfcc.T,axis=0)
    return feature.reshape(1,-1)

audio_path = sys.argv[1]
features = extract_features(audio_path)
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0]
confidence = np.max(probability)*100
label = "Genuine" if prediction == 0 else "Deepfake"
print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}%")