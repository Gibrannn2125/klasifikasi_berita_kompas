import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# --- SETTINGS ---
DATA_URL = "https://raw.githubusercontent.com/Gibrannn2125/klasifikasi_berita_kompas/refs/heads/main/cekfakta_kompas.csv"  

# --- TITLE ---
st.title("📰 Klasifikasi Berita Kompas - HOAKS atau FAKTA")

# --- LOAD & CLEAN DATA ---
@st.cache_data
def load_and_train_model():
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=['isi', 'kategori_klarifikasi'])

    # Standarisasi label
    df['kategori_klarifikasi'] = df['kategori_klarifikasi'].str.strip().str.upper()

    # Tampilkan jumlah label
    label_counts = df['kategori_klarifikasi'].value_counts()
    st.sidebar.markdown("### Distribusi Label")
    st.sidebar.write(label_counts)

    # Penyeimbangan data
    if 'HOAKS' in df['kategori_klarifikasi'].unique() and 'FAKTA' in df['kategori_klarifikasi'].unique():
        hoaks = df[df['kategori_klarifikasi'] == 'HOAKS']
        fakta = df[df['kategori_klarifikasi'] == 'FAKTA']
        min_len = min(len(hoaks), len(fakta))
        df_balanced = pd.concat([
            resample(hoaks, replace=False, n_samples=min_len, random_state=42),
            resample(fakta, replace=False, n_samples=min_len, random_state=42)
        ])
    else:
        df_balanced = df.copy()

    X = df_balanced['isi']
    y = df_balanced['kategori_klarifikasi']

    # Pipeline model
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    model.fit(X, y)
    return model

model = load_and_train_model()

# --- FORM INPUT ---
st.markdown("### Masukkan teks berita atau klaim:")
user_input = st.text_area("Teks Berita/Klaim")

if st.button("🔍 Cek Kebenaran"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == "HOAKS":
            st.error("❌ Kategori Prediksi: HOAKS")
        elif prediction == "FAKTA":
            st.success("✅ Kategori Prediksi: FAKTA")
        else:
            st.info(f"Kategori tidak dikenal: {prediction}")
