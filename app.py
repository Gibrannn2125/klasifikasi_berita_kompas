import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ------------------- PAGE CONFIG ------------------- #
st.set_page_config(page_title="Deteksi Berita Hoaks Kompas", layout="centered")

# ------------------- CSS STYLE ------------------- #
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #1a237e;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-hoax {
        background-color: #ffebee;
        color: #c62828;
        padding: 1.2rem;
        border-left: 6px solid #b71c1c;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .result-fact {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1.2rem;
        border-left: 6px solid #1b5e20;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">📰 Deteksi Berita Kompas</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan teks berita atau klaim, dan sistem akan mendeteksi apakah itu <strong>Fakta</strong> atau <strong>Hoaks</strong>.</div>', unsafe_allow_html=True)

# ------------------- Load Data & Train Model ------------------- #
@st.cache_data
def train_model():
    df = pd.read_csv("cekfakta_kompas.csv")
    df = df.dropna(subset=["text", "categories"])

    X = df["text"]
    y = df["categories"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="indonesian", max_df=0.9)),
        ("clf", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    return model

model = train_model()

# ------------------- User Input Form ------------------- #
user_text = st.text_area("✍️ Tulis teks berita/klaim di sini", height=180)

if st.button("🔍 Deteksi Sekarang"):
    if user_text.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        pred = model.predict([user_text])[0]
        if pred.lower() == "hoaks":
            st.markdown('<div class="result-hoax">🚫 Prediksi: Berita ini merupakan <strong>HOAKS</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-fact">✅ Prediksi: Berita ini <strong>FAKTA</strong>.</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
