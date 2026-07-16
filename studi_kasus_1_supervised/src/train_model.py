import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Data quality check
n_missing = df.isnull().sum().sum()
n_duplicates = df.duplicated().sum()
print(f"Missing values: {n_missing}")
print(f"Duplicate rows: {n_duplicates}")
df = df.drop_duplicates()

# Features and target
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# Supervised Learning: Train-Test Split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train both algorithms
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate both algorithms
candidates = {
    'Logistic Regression': logreg_model,
    'Random Forest': rf_model,
}

accuracies = {}
for name, model in candidates.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\n=== Model comparison ===")
for name, acc in accuracies.items():
    print(f"{name}: accuracy = {acc:.4f}")

best_name = max(accuracies, key=accuracies.get)
best_model = candidates[best_name]
print(f"\nBest model: {best_name} (accuracy = {accuracies[best_name]:.4f})")

# Save the scaler and the best-performing model
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(best_model, os.path.join(models_dir, 'supervised_model.pkl'))

# Save Random Forest feature importances for interpretability, regardless of which model was deployed
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.head(7).to_csv(os.path.join(models_dir, 'importances.txt'))

print("\nModels trained and saved successfully!")
