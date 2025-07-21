# Aplikasi Cek Fakta: Deteksi Hoaks atau Fakta

Aplikasi berbasis web menggunakan Streamlit untuk mendeteksi apakah suatu teks berita mengandung informasi **hoaks** atau **fakta**. Model dilatih dari dataset `cekfakta_kompas.csv`.

## 📦 Fitur
- Input teks artikel atau berita
- Klasifikasi otomatis: Hoaks atau Fakta
- Interface sederhana berbasis Streamlit

## 🚀 Cara Menjalankan di Lokal

1. Clone repository ini:
    ```bash
    git clone https://github.com/namauser/nama-repo.git
    cd nama-repo
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Jalankan aplikasi:
    ```bash
    streamlit run app.py
    ```

## 📁 Struktur File
```
.
├── app.py                # Web interface
├── model_cekfakta.pkl    # Model klasifikasi teks
├── requirements.txt      # Dependency Python
└── README.md             # Dokumentasi proyek
```

## 🧠 Model
Model menggunakan TF-IDF + Logistic Regression untuk memproses teks dan klasifikasi hoaks/fakta.

## 💡 Sumber Data
Dataset dari situs CekFakta Kompas: `cekfakta_kompas.csv`

---

By: [Nama Kamu]