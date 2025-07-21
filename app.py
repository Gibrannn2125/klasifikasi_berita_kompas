import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ------------------ Title and Description ------------------ #
st.set_page_config(page_title="Klasifikasi Berita Kompas", layout="centered")
st.markdown("<h2 style='color:#004d99;'>📰 Klasifikasi Berita Kompas: Fakta atau Hoaks?</h2>", unsafe_allow_html=True)
st.markdown("Masukkan teks berita atau klaim yang ingin dicek keasliannya.")

# ------------------ Load & Train Model ------------------ #
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("cekfakta_kompas.csv")

    # Pastikan kolom ada
    df = df.dropna(subset=["text", "categories"])

    X = df["text"]
    y = df["categories"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="indonesian")),
        ("clf", MultinomialNB())
    ])
    model.fit(X_train, y_train)

    return model

model = load_and_train_model()

# ------------------ User Input ------------------ #
user_input = st.text_area("📝 Teks Berita/Klaim", height=200)

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks berita untuk diprediksi.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction.lower() == "hoaks":
            st.error("🚫 Kategori Prediksi: HOAKS")
        else:
            st.success("✅ Kategori Prediksi: FAKTA")
