import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Load dan Latih Model
# ----------------------------
@st.cache_data
def load_and_train_model():
    st.write("Kolom tersedia:", df.columns.tolist())
    df = pd.read_csv("cekfakta_kompas.csv")
    df = df.dropna(subset=["isi", "kategori_klarifikasi"])  

    X = df["isi"]
    y = df["kategori_klarifikasi"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    model = pipeline.fit(X_train, y_train)
    return model

model = load_and_train_model()

# ----------------------------
# Styling
# ----------------------------
st.markdown("""
    <style>
    body { background-color: #f4f4f4; }
    .stApp {
        background-color: #e9f5ff;
        padding: 2rem;
    }
    .title {
        font-size: 2.5rem;
        color: #0a66c2;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# UI
# ----------------------------
st.markdown('<div class="title">📰 Deteksi HOAKS Berita Kompas</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="box">', unsafe_allow_html=True)

    input_text = st.text_area("Masukkan isi berita atau klaim:", height=200)

    if st.button("🔍 Cek Fakta"):
        if input_text.strip() == "":
            st.warning("Harap masukkan teks terlebih dahulu.")
        else:
            prediction = model.predict([input_text])[0]
            if prediction.lower() == "hoaks":
                st.error("❌ Kategori Prediksi: HOAKS")
            else:
                st.success("✅ Kategori Prediksi: FAKTA")

    st.markdown('</div>', unsafe_allow_html=True)
