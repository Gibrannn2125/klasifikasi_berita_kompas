
import streamlit as st
import pickle

# Styling CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; padding: 2rem; border-radius: 10px; }
    .title { font-size: 28px; font-weight: bold; color: #2c3e50; }
    .hoax { background-color: #ffe6e6; color: #c0392b; padding: 1rem; border-left: 6px solid #e74c3c; border-radius: 5px; }
    .fakta { background-color: #e6f4ea; color: #27ae60; padding: 1rem; border-left: 6px solid #2ecc71; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">📢 Klasifikasi Berita Hoaks - Kompas</div>', unsafe_allow_html=True)
st.write("Masukkan teks berita atau klaim yang ingin dicek:")

# Input text
text_input = st.text_area("Teks Berita/Klaim")

# Load model
with open("model_berita.pkl", "rb") as f:
    model = pickle.load(f)

# Predict
if st.button("🔍 Cek Fakta"):
    if text_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        pred = model.predict([text_input])[0]
        if pred.lower() == "hoaks":
            st.markdown(f'<div class="hoax">🚫 Kategori Prediksi: <strong>{pred.upper()}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fakta">✅ Kategori Prediksi: <strong>{pred.upper()}</strong></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
