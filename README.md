# TriVerse: Data Fundamentals Final Project

This is the final project for the Data Fundamentals course, consisting of two case studies built on different datasets: student stress level prediction (supervised learning) and student lifestyle habits profiling (unsupervised learning). Full requirements are documented in `TriVerse_Project_Brief.md`.

## Team

| Name | Student ID | Role |
| --- | --- | --- |
| Hidayat Nur Hijrah | 24523201 | Data Preparation, Feature Engineering, Documentation |
| Muhamad Rafi Raditya | 24523231 | Model Training and Evaluation |
| Rayyan Galih Indarto | 24523224 | Application Development and Deployment |

## Folder Structure

```
triverse-fsd-project/
├── studi_kasus_1_supervised/     # Student stress level prediction (Random Forest)
│   ├── data/
│   │   └── StressLevelDataset.csv
│   ├── models/
│   │   ├── scaler.pkl
│   │   ├── supervised_model.pkl
│   │   └── importances.txt       # Feature importance from Random Forest
│   ├── notebooks/                # EDA and training notebook (Colab)
│   └── src/
│       ├── app.py                # Gradio application (UI and prediction)
│       └── train_model.py        # Model training script
├── studi_kasus_2_unsupervised/   # Student lifestyle habits clustering
│   ├── data/
│   │   └── student_habits_performance.csv
│   ├── models/                   # Pending, clustering model to be added
│   └── notebooks/                # Pending, clustering notebook to be added
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Case Study 1: Student Stress Prediction

### Retraining the Model

```bash
python studi_kasus_1_supervised/src/train_model.py
```

This script reads the dataset from `studi_kasus_1_supervised/data/StressLevelDataset.csv`, retrains the scaler and the Random Forest model, and saves the results to `studi_kasus_1_supervised/models/`.

### Running the Application

```bash
python studi_kasus_1_supervised/src/app.py
```

The application opens automatically in the browser. Adjust the sliders for the seven main factors (blood pressure, sleep quality, extracurricular activities, bullying, basic needs, teacher student relationship, and depression level), then click **Analisis Data Mahasiswa** to view the prediction result.

## Case Study 2: Student Lifestyle Habits Profiling

This case study has not been implemented yet. The dataset (`student_habits_performance.csv`) is already available under `studi_kasus_2_unsupervised/data/`. The EDA, preprocessing, and clustering work (K-Means with Elbow Method and Silhouette Score) will follow.
