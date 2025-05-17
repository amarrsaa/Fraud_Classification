# app.py
import streamlit as st
import os
import tempfile
from utils import (
    classify_english_message,
    classify_english_audio,
    predict_hindi_call_from_audio,
    predict_telugu_call,
    load_sms_dataset,
    train_or_load_sms_model
)

# Load models
@st.cache_resource
def get_sms_model(language):
    df = load_sms_dataset(language)
    return train_or_load_sms_model(df, language)

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
                result = classify_english_message(sms_input)
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
                    prediction = classify_english_audio(tmp_path)
                else:
                    prediction = predict_hindi_call_from_audio(tmp_path)
                st.success(f"Prediction: {prediction}")
                os.remove(tmp_path)
    else:
        transcript = st.text_area("Enter Telugu call transcript")
        if st.button("Classify Transcript"):
            if transcript.strip():
                predict_telugu_call(transcript)
            else:
                st.warning("Please enter transcript text.")
