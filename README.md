# Klasifikasi Berita Hoaks - Kompas

Website ini menggunakan model machine learning untuk mengklasifikasikan apakah sebuah klaim/berita termasuk **HOAKS** atau **FAKTA**, berdasarkan dataset Kompas.

## Cara Menjalankan

1. Install dependency:

```
pip install -r requirements.txt
```

2. Jalankan aplikasi:

```
streamlit run app.py
```

## Struktur File

- `app.py`: Aplikasi Streamlit
- `model_berita.pkl`: Model klasifikasi berbasis Random Forest
- `requirements.txt`: Daftar dependency
- `README.md`: Dokumentasi proyek
