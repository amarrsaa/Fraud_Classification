# app.py

import streamlit as st
import os
import tempfile
import pickle
import tensorflow as tf

from utils import (
    classify_english_message,
    classify_english_audio,
    predict_hindi_call_from_audio,
    predict_telugu_call,
    load_sms_dataset,
    train_or_load_sms_model,
    load_english_models
)

# Cache the loading and training of SMS model to speed up UI
@st.cache_resource
def get_sms_model(language):
    try:
        df = load_sms_dataset(language)
        return train_or_load_sms_model(df, language)
    except FileNotFoundError as e:
        st.error(f"Dataset not found for language '{language}': {e}")
        return None

# Load English models once to reuse
english_msg_model, english_msg_tokenizer = load_english_models()[2:]  # msg_model, msg_tokenizer

st.title("Multilingual Fraud Detection System")
st.markdown("Classify SMS and Call inputs as Fraud or Normal in English, Hindi, and Telugu.")

tab1, tab2 = st.tabs(["SMS", "Call"])

with tab1:
    st.subheader("Classify SMS Message")
    language = st.selectbox("Choose language", ["english", "hindi", "telugu"])
    sms_input = st.text_area("Enter the SMS text")

    if st.button("Classify SMS"):
        if not sms_input.strip():
            st.warning("Please enter some text.")
        else:
            if language == "english":
                result = classify_english_message(sms_input, english_msg_model, english_msg_tokenizer)
                st.success(f"Prediction: {result}")
            else:
                model = get_sms_model(language)
                if model is None:
                    st.error(f"Cannot classify SMS because dataset/model for '{language}' is missing.")
                else:
                    pred = model.predict([sms_input])[0]
                    label = "Fraud Message" if pred == 1 or pred == "fraud" else "Normal Message"
                    st.success(f"Prediction: {label}")

with tab2:
    st.subheader("Classify Call (Audio or Transcript)")
    call_language = st.selectbox("Choose call language", ["english", "hindi", "telugu"])

    uploaded_file = st.file_uploader("Upload Call Audio (.wav)", type=["wav"])

    if st.button("Classify Call Audio") and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if call_language == "english":
            prediction = classify_english_audio(tmp_path, *load_english_models()[:2])  # call_model, call_tokenizer
        elif call_language == "hindi":
            prediction = predict_hindi_call_from_audio(tmp_path)
        else:  # Telugu
            st.warning("For Telugu call classification, please enter transcript text instead.")
            prediction = None

        if prediction:
            st.success(f"Prediction: {prediction}")
        os.remove(tmp_path)

    if call_language == "telugu":
        transcript = st.text_area("Enter Telugu call transcript")
        if st.button("Classify Transcript"):
            if transcript.strip():
                prediction = predict_telugu_call(transcript)
                st.success(f"Prediction: {prediction}")
            else:
                st.warning("Please enter transcript text.")
