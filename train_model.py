import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import joblib
import os

# Ensure the working directory contains data folder
data_path = 'data/StressLevelDataset.xlsx'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cannot find dataset at {data_path}. Please check your current working directory.")

# Load dataset
df = pd.read_excel(data_path)

# Features and target
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# Supervised Learning: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_all = scaler.transform(X)

# 1. Supervised Learning Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 2. Unsupervised Learning Model (K-Means Clustering)
# Cluster into 3 groups since there are 3 stress levels in the original dataset
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X_scaled_all)

# Save the models and scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(rf_model, 'supervised_model.pkl')
joblib.dump(kmeans_model, 'unsupervised_model.pkl')

print("Models trained and saved successfully!")
