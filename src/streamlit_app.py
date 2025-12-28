# src/streamlit_app.py
import streamlit as st
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Folder model & tokenizer
# ==============================
BASE_DIR = os.path.dirname(__file__)  # folder src/
APP_FOLDER = os.path.join(BASE_DIR, "app_sentimen")  # app_sentimen sekarang di dalam src/

# Load model & tokenizer
model_path = os.path.join(APP_FOLDER, "bilstm_model.keras")
tokenizer_path = os.path.join(APP_FOLDER, "tokenizer.pkl")

model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100  # sesuai training

# ==============================
# Streamlit GUI
# ==============================
st.title("Prediksi Sentimen WhatsApp")

user_input = st.text_area("Masukkan teks pesan:")

if st.button("Prediksi"):
    if not user_input.strip():
        st.warning("Silakan masukkan teks dulu!")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        pred = model.predict(padded)[0][0]
        
        # Emoji
        if pred >= 0.5:
            label = "Positif 😄"
        else:
            label = "Negatif 😞"
        
        st.success(f"Hasil Prediksi: {label}")
