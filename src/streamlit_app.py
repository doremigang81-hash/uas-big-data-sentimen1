import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("../app_sentimen/bilstm_model.keras")

with open("../app_sentimen/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

st.title("Prediksi Sentimen WhatsApp")

text = st.text_area("Masukkan teks")

if st.button("Prediksi"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    pred = model.predict(padded)[0][0]
    label = "Positif" if pred >= 0.5 else "Negatif"
    st.success(f"Hasil: {label}")
