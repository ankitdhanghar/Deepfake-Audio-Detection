
import streamlit as st
import joblib
import librosa
import numpy as np

model = joblib.load("model.pkl")

def extract_features(uploaded_file):

    audio, sr = librosa.load(uploaded_file,sr=16000)

    mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=40)
    feature = np.mean(mfcc.T,axis=0)
    return feature.reshape(1,-1)

st.title("Deepfake Audio Detection")
uploaded_file = st.file_uploader("Upload Audio",type=["wav","mp3","flac"])

if uploaded_file:
    features = extract_features(uploaded_file)
    prediction = model.predict(features )[0]

    probability = model.predict_proba(features)[0]
    confidence = np.max(probability)*100
    if prediction == 0:st.success(f"Genuine Audio ({confidence:.2f}%)")
    else:st.error(f"Deepfake Audio ({confidence:.2f}%)")