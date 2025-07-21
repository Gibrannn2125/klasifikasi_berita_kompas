import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# CSS styling
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; padding: 2rem; border-radius: 10px; }
    .title { font-size: 28px; font-weight: bold; color: #2c3e50; }
    .hoax { background-color: #ffe6e6; color: #c0392b; padding: 1rem; border-left: 6px solid #e74c3c; border-radius: 5px; margin-top: 1rem; }
    .fakta { background-color: #e6f4ea; color: #27ae60; padding: 1rem; border-left: 6px solid #2ecc71; border-radius: 5px; margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">📢 Klasifikasi Berita Hoaks - Kompas</div>', unsafe_allow_html=True)
st.write("Masukkan teks berita atau klaim yang ingin dicek:")

text_input = st.text_area("📝 Teks Berita/Klaim")

# Fungsi untuk melatih model
@st.cache_resource
def load_model():
    # Ganti URL ini dengan URL raw CSV-mu dari GitHub
    url = "https://raw.githubusercontent.com/Gibrannn2125/klasifikasi_berita_kompas/refs/heads/main/cekfakta_kompas.csv"
    df = pd.read_csv(url)
    df = df.dropna(subset=["text", "categories"])
    X = df["text"]
    y = df["categories"].str.lower()
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vec, y)
    return model, vectorizer

model, vectorizer = load_model()

if st.button("🔍 Cek Fakta"):
    if not text_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        input_vector = vectorizer.transform([text_input])
        pred = model.predict(input_vector)[0]
        if pred == "hoaks":
            st.markdown(f'<div class="hoax">🚫 Kategori Prediksi: <strong>{pred.upper()}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="fakta">✅ Kategori Prediksi: <strong>{pred.upper()}</strong></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
