import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths are resolved relative to this file so the script works regardless of cwd
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(ROOT_DIR, 'data', 'StressLevelDataset.csv')
models_dir = os.path.join(ROOT_DIR, 'models')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cannot find dataset at {data_path}. Please check your current working directory.")

# Load dataset
df = pd.read_csv(data_path)

# Features and target
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# Supervised Learning: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Supervised Learning Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model and scaler
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(rf_model, os.path.join(models_dir, 'supervised_model.pkl'))

print("Models trained and saved successfully!")
