import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Atur MLflow Tracking URI ke localhost sesuai ketentuan Dicoding
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Latihan Credit Scoring Superstore")

# 2. Aktifkan fitur Autolog untuk otomatis mencatat metrik & parameter
mlflow.sklearn.autolog()

# 3. Load dataset hasil preprocessing
print("Memuat dataset...")
df = pd.read_csv("namadataset_preprocessing.csv")

# 4. Pisahkan fitur (X) dan label target (y)
X = df.drop(columns=["profit_status"])
y = df["profit_status"]

# 5. Split data menjadi Training dan Testing set (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Mulai pelatihan model di dalam MLflow Run
print("Mulai melatih model RandomForest...")
with mlflow.start_run() as run:
  # Inisialisasi dan latih model
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  
  model.fit(X_train, y_train)

  # Evaluasi model
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)

  print(f"Pelatihan selesai! Akurasi Model pada Test Set: {acc:.4f}")
  print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))
  print(f"\nMLflow Run ID: {run.info.run_id}")