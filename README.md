# Deteksi Stres Mahasiswa

Aplikasi Gradio untuk memprediksi tingkat stres mahasiswa berdasarkan 7 faktor utama, menggunakan kombinasi model supervised (Random Forest) untuk klasifikasi tingkat stres dan model unsupervised (K-Means) untuk pengelompokan profil.

## Struktur Folder

```
triverse-fsd-project/
├── src/
│   ├── app.py            # Aplikasi Gradio (UI + prediksi)
│   └── train_model.py    # Script training model
├── models/
│   ├── scaler.pkl
│   ├── supervised_model.pkl
│   ├── unsupervised_model.pkl
│   └── importances.txt   # Feature importance dari Random Forest
├── data/
│   └── StressLevelDataset.xlsx
└── requirements.txt
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Training Ulang Model

```bash
python src/train_model.py
```

Script ini membaca dataset dari `data/StressLevelDataset.xlsx`, melatih ulang scaler, Random Forest, dan K-Means, lalu menyimpan hasilnya ke folder `models/`.

## Menjalankan Aplikasi

```bash
python src/app.py
```

Aplikasi akan terbuka otomatis di browser. Geser slider untuk 7 faktor utama (tekanan darah, kualitas tidur, aktivitas ekstrakurikuler, bullying, kebutuhan dasar, hubungan dosen-mahasiswa, dan tingkat depresi), lalu klik **Analisis Data Mahasiswa** untuk melihat hasil prediksi.
