
import streamlit as st
import pickle
import pandas as pd

# Load model
with open("model_cekfakta.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Cek Fakta Hoaks", layout="centered")
st.title("📰 Deteksi hoax dari berita kompas")
st.write("Masukkan teks berita/artikel di bawah ini, lalu klik prediksi:")

# Input teks
input_text = st.text_area("Teks Berita/Artikel", height=200)

# Tombol prediksi
if st.button("🔍 Prediksi"):
    if input_text.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        data = pd.Series([input_text])
        prediction = model.predict(data)[0]
        st.success(f"Hasil Prediksi: **{prediction.upper()}**")
        if prediction.lower() == "hoaks":
            st.error("⚠️ Berita ini terdeteksi sebagai HOAKS.")
        else: prediction.lower() == "fakta":
            st.info("✅ Berita ini terdeteksi sebagai FAKTA.")
