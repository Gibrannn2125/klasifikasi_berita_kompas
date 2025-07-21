import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample

# Set judul halaman
st.set_page_config(page_title="Klasifikasi Berita Hoaks", layout="wide")

# Tampilan header
st.markdown("""
    <style>
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
    }
    .result-hoax {
        background-color: #ffe6e6;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 20px;
    }
    .result-fakta {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📢 Klasifikasi Berita Hoaks - Kompas</div>', unsafe_allow_html=True)

# Muat data dari file CSV lokal
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("cekfakta_kompas.csv")

    # Tampilkan kolom untuk debugging jika perlu
    # st.write("Kolom tersedia:", df.columns.tolist())

    # Ubah kolom sesuai isi dataset
    # Misal: klaim = 'klaim', label = 'klasifikasi'
    df = df.dropna(subset=['klaim', 'klasifikasi'])
    df['klasifikasi'] = df['klasifikasi'].str.strip().str.upper()

    # Seimbangkan jumlah hoaks dan fakta
    hoaks = df[df['klasifikasi'] == 'HOAKS']
    fakta = df[df['klasifikasi'] == 'FAKTA']
    min_len = min(len(hoaks), len(fakta))

    df_balanced = pd.concat([
        resample(hoaks, replace=False, n_samples=min_len, random_state=42),
        resample(fakta, replace=False, n_samples=min_len, random_state=42)
    ])

    X = df_balanced['klaim']
    y = df_balanced['klasifikasi']

    # Pipeline model
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    model.fit(X, y)

    return model

model = load_and_train_model()

# Input pengguna
st.subheader("Masukkan teks berita atau klaim yang ingin dicek:")
user_input = st.text_area("Teks Berita/Klaim", height=150)

if st.button("🔍 Cek Kategori"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        prediction = model.predict([user_input])[0]

        if prediction == 'HOAKS':
            st.markdown(f'<div class="result-hoax">🚨 Prediksi: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-fakta">✅ Prediksi: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
