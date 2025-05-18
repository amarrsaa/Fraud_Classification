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

# Load models once using caching
@st.cache_resource
def get_sms_model(language):
    df = load_sms_dataset(language)
    return train_or_load_sms_model(df, language)

@st.cache_resource
def load_english():
    return load_english_models()

# App title
st.title("Multilingual Fraud Detection System")
st.markdown("Classify SMS and Call inputs as Fraud or Normal in English, Hindi, and Telugu.")

# Tabs
tab1, tab2 = st.tabs(["SMS", "Call"])

# SMS Tab
with tab1:
    st.subheader("Classify SMS Message")
    language = st.selectbox("Choose language", ["english", "hindi", "telugu"])
    sms_input = st.text_area("Enter the SMS text")
    
    if st.button("Classify SMS"):
        if not sms_input.strip():
            st.warning("Please enter some text.")
        else:
            if language == "english":
                call_model, call_tokenizer, msg_model, msg_tokenizer = load_english()
                result = classify_english_message(
                    sms_input,
                    msg_model=msg_model,
                    msg_tokenizer=msg_tokenizer
                )
                st.success(f"Prediction: {result}")
            else:
                model = get_sms_model(language)
                pred = model.predict([sms_input])[0]
                label = "Fraud Message" if pred == 1 or pred == "fraud" else "Normal Message"
                st.success(f"Prediction: {label}")

# Call Tab
with tab2:
    st.subheader("Classify Call (Audio or Transcript)")
    call_language = st.selectbox("Choose call language", ["english", "hindi", "telugu"])

    if call_language in ["english", "hindi"]:
        uploaded_file = st.file_uploader("Upload Call Audio (.wav)", type=["wav"])
        if st.button("Classify Call Audio") and uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

                if call_language == "english":
                    call_model, call_tokenizer, _, _ = load_english()
                    prediction = classify_english_audio(tmp_path, call_model, call_tokenizer)
                else:
                    prediction = predict_hindi_call_from_audio(tmp_path)

                st.success(f"Prediction: {prediction}")
                os.remove(tmp_path)
    else:
        transcript = st.text_area("Enter Telugu call transcript")
        if st.button("Classify Transcript"):
            if transcript.strip():
                result = predict_telugu_call(transcript)
                st.success(result)
            else:
                st.warning("Please enter transcript text.")
