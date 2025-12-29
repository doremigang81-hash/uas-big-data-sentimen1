import streamlit as st
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
APP_FOLDER = os.path.join(BASE_DIR, "..", "app_sentimen")

model_path = os.path.join(APP_FOLDER, "bilstm_model.keras")
tokenizer_path = os.path.join(APP_FOLDER, "tokenizer.pkl")

# ================= LOAD MODEL =================
model = load_model(model_path, compile=False)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# ================= STREAMLIT APP =================
max_len = 100

st.title("📱 Prediksi Sentimen WhatsApp")

text = st.text_area("Masukkan teks")

if st.button("Prediksi"):
    if text.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        pred = model.predict(padded)[0][0]

        label = "Positif 😊" if pred >= 0.5 else "Negatif 😞"
        st.success(f"Hasil: {label}")
