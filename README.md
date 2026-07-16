# TriVerse — Tugas Akhir Fundamen Sains Data

Tugas akhir mata kuliah Fundamental Sains Data, terdiri dari dua studi kasus dengan dataset berbeda: prediksi tingkat stres mahasiswa (supervised) dan profil kebiasaan hidup mahasiswa (unsupervised). Detail lengkap requirement ada di `TriVerse_Project_Brief.md`.

## Struktur Folder

```
triverse-fsd-project/
├── studi_kasus_1_supervised/     # Prediksi tingkat stres (Random Forest)
│   ├── data/
│   │   └── StressLevelDataset.csv
│   ├── models/
│   │   ├── scaler.pkl
│   │   ├── supervised_model.pkl
│   │   └── importances.txt       # Feature importance dari Random Forest
│   ├── notebooks/                # Notebook EDA & training (Colab)
│   └── src/
│       ├── app.py                # Aplikasi Gradio (UI + prediksi)
│       └── train_model.py        # Script training model
├── studi_kasus_2_unsupervised/   # Clustering kebiasaan hidup mahasiswa
│   ├── data/
│   │   └── student_habits_performance.csv
│   ├── models/                   # (belum ada — model clustering menyusul)
│   └── notebooks/                # (belum ada — notebook clustering menyusul)
└── requirements.txt
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Studi Kasus 1 — Prediksi Tingkat Stres

### Training Ulang Model

```bash
python studi_kasus_1_supervised/src/train_model.py
```

Script ini membaca dataset dari `studi_kasus_1_supervised/data/StressLevelDataset.csv`, melatih ulang scaler dan Random Forest, lalu menyimpan hasilnya ke `studi_kasus_1_supervised/models/`.

### Menjalankan Aplikasi

```bash
python studi_kasus_1_supervised/src/app.py
```

Aplikasi akan terbuka otomatis di browser. Geser slider untuk 7 faktor utama (tekanan darah, kualitas tidur, aktivitas ekstrakurikuler, bullying, kebutuhan dasar, hubungan dosen-mahasiswa, dan tingkat depresi), lalu klik **Analisis Data Mahasiswa** untuk melihat hasil prediksi.

## Studi Kasus 2 — Profil Kebiasaan Hidup Mahasiswa

Belum dikerjakan. Dataset (`student_habits_performance.csv`) sudah tersedia di `studi_kasus_2_unsupervised/data/`; notebook EDA, preprocessing, dan clustering (K-Means dengan Elbow Method + Silhouette Score) menyusul.
