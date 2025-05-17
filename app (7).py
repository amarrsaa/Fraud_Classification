import streamlit as st
import os
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import speech_recognition as sr
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import streamlit as st

# -------------------- CONFIG --------------------

DATA_PATHS = {
    "hindi_sms": "hindi_sms_dataset.csv",
    "telugu_sms": "telugu_sms_dataset.csv",
    "hindi_call": "hindi_call_records_dataset.csv",
    "telugu_call": "telugu_call_dataset.csv",
}

MODEL_PATHS = {
    "hindi_sms": "hindi_fraud_classifier.pkl",
    "telugu_sms": "telugu_fraud_classifier.pkl",
    "hindi_call": {
        "model": "hindi_fraud_model.pkl",
        "vectorizer": "hindi_vectorizer.pkl"
    },
    "telugu_call": {
        "model": "telugu_call_classifier.h5",
        "tokenizer": "tokenizer.pkl"
    }
}

CALL_MODEL_PATH_EN = "call_model.h5"
MSG_MODEL_PATH_EN = "msg_model.h5"
CALL_TOKENIZER_PATH_EN = "call_tokenizer.pkl"
MSG_TOKENIZER_PATH_EN = "msg_tokenizer.pkl"

# Load English models/tokenizers once
call_model_en = load_model(CALL_MODEL_PATH_EN)
msg_model_en = load_model(MSG_MODEL_PATH_EN)

with open(CALL_TOKENIZER_PATH_EN, 'rb') as f:
    call_tokenizer_en = pickle.load(f)

with open(MSG_TOKENIZER_PATH_EN, 'rb') as f:
    msg_tokenizer_en = pickle.load(f)

# -------------------- FUNCTIONS --------------------

def classify_english_message(text):
    seq = msg_tokenizer_en.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200, padding="post")
    pred = msg_model_en.predict(padded)[0][0]
    return "Fraud Message" if pred >= 0.5 else "Normal Message"

def classify_english_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
        seq = call_tokenizer_en.texts_to_sequences([transcript])
        padded = pad_sequences(seq, maxlen=200, padding="post")
        pred = call_model_en.predict(padded)[0][0]
        return "Fraud Call" if pred >= 0.5 else "Normal Call", transcript
    except Exception as e:
        return f"Error: {e}", ""

def load_sms_dataset(language):
    key = f"{language}_sms"
    df = pd.read_csv(DATA_PATHS[key])
    assert 'Text' in df.columns and 'Label' in df.columns
    return df

def train_or_load_sms_model(df, language):
    filename = MODEL_PATHS[f"{language}_sms"]
    if os.path.exists(filename):
        return joblib.load(filename)
    X = df['Text']
    y = df['Label']
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ])
    model.fit(X, y)
    joblib.dump(model, filename)
    return model

def train_or_load_hindi_call_model():
    model_file = MODEL_PATHS['hindi_call']['model']
    vec_file = MODEL_PATHS['hindi_call']['vectorizer']

    if os.path.exists(model_file) and os.path.exists(vec_file):
        return joblib.load(model_file), joblib.load(vec_file)

    df = pd.read_csv(DATA_PATHS['hindi_call'])
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vec_file)
    return model, vectorizer

def predict_hindi_call_from_audio(audio_file):
    model, vectorizer = train_or_load_hindi_call_model()
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="hi-IN")
        features = vectorizer.transform([text])
        pred = model.predict(features)[0]
        return "Fraud" if pred == 1 else "Real", text
    except Exception as e:
        return f"Error: {e}", ""

def train_or_load_telugu_call_model():
    model_file = MODEL_PATHS['telugu_call']['model']
    tokenizer_file = MODEL_PATHS['telugu_call']['tokenizer']

    df = pd.read_csv(DATA_PATHS['telugu_call'])
    df['label'] = df['label'].map({'fraud': 1, 'real': 0})
    X = df['transcript'].astype(str).values
    y = df['label'].values

    if os.path.exists(tokenizer_file):
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)

    sequences = tokenizer.texts_to_sequences(X)
    max_len = max(len(seq) for seq in sequences)
    X_pad = pad_sequences(sequences, maxlen=max_len, padding='post')

    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = Sequential([
            Embedding(input_dim=10000, output_dim=64, input_length=max_len),
            LSTM(64),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_pad, y, epochs=5, batch_size=32, validation_split=0.1)
        model.save(model_file)

    return model, tokenizer, max_len

def predict_telugu_call(text):
    model, tokenizer, max_len = train_or_load_telugu_call_model()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    return "Fraud" if pred >= 0.5 else "Real", pred

# -------------------- STREAMLIT UI --------------------

st.title("Multilingual Fraud Detection System")

option = st.selectbox("Choose Task", [
    "Hindi SMS Classification",
    "Telugu SMS Classification",
    "Hindi Call Audio Classification",
    "Telugu Call Transcript Classification",
    "English SMS Classification",
    "English Call Audio Classification"
])

if option == "Hindi SMS Classification":
    st.write("Enter Hindi SMS text:")
    sms = st.text_area("")
    if st.button("Predict"):
        df = load_sms_dataset('hindi')
        model = train_or_load_sms_model(df, 'hindi')
        pred = model.predict([sms])[0]
        st.success(f"Prediction: {'Fraud' if pred == 1 or pred == 'fraud' else 'Real'}")

elif option == "Telugu SMS Classification":
    st.write("Enter Telugu SMS text:")
    sms = st.text_area("")
    if st.button("Predict"):
        df = load_sms_dataset('telugu')
        model = train_or_load_sms_model(df, 'telugu')
        pred = model.predict([sms])[0]
        st.success(f"Prediction: {'Fraud' if pred == 1 or pred == 'fraud' else 'Real'}")

elif option == "Hindi Call Audio Classification":
    audio_file = st.file_uploader("Upload Hindi call audio (.wav)", type=["wav"])
    if audio_file and st.button("Predict"):
        result, transcript = predict_hindi_call_from_audio(audio_file)
        st.success(f"Prediction: {result}")
        st.write(f"Transcript: {transcript}")

elif option == "Telugu Call Transcript Classification":
    text = st.text_area("Enter Telugu call transcript")
    if st.button("Predict"):
        label, conf = predict_telugu_call(text)
        st.success(f"Prediction: {label} (Confidence: {conf:.4f})")

elif option == "English SMS Classification":
    sms = st.text_area("Enter English SMS text")
    if st.button("Predict"):
        result = classify_english_message(sms)
        st.success(f"Prediction: {result}")

elif option == "English Call Audio Classification":
    audio_file = st.file_uploader("Upload English call audio (.wav)", type=["wav"])
    if audio_file and st.button("Predict"):
        result, transcript = classify_english_audio(audio_file)
        if "Error" in result:
            st.error(result)
        else:
            st.success(f"Prediction: {result}")
            st.write(f"Transcript: {transcript}")

