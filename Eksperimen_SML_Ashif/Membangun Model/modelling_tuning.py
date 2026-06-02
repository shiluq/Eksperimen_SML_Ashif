import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_saham_model():
    # 1. Set nama eksperimen di MLflow
    mlflow.set_experiment("Eksperimen_Saham_ANTM")
    
    # 2. Memuat data hasil preprocessing
    df = pd.read_csv("namadataset_preprocessing.csv")
    
    # Buat fitur sederhana: menggunakan harga 'close' kemarin untuk prediksi hari ini
    df['close_lag1'] = df['close'].shift(1)
    df = df.dropna().reset_index(drop=True)
    
    # Menentukan Fitur (X) dan Target (y)
    X = df[['close_lag1', 'open', 'high', 'low', 'volume']]
    y = df['close']
    
    # Split data menjadi Train dan Test (80% Train, 20% Test)
    # Karena data saham bertipe waktu, kita tidak mengacaknya (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. Hyperparameter Tuning Manual & Logging ke MLflow
    # Kita akan coba beberapa nilai kombinasi n_estimators
    parameters_to_try = [10, 50, 100]
    
    print("=== Memulai Training Model & MLflow Logging ===")
    
    for n_est in parameters_to_try:
        # Memulai sesi run MLflow secara manual
        with mlflow.start_run(run_name=f"RandomForest_n_{n_est}"):
            
            # Inisialisasi model dengan parameter saat ini
            model = RandomForestRegressor(n_estimators=n_est, random_state=42)
            model.fit(X_train, y_train)
            
            # Prediksi
            predictions = model.predict(X_test)
            
            # Hitung Metrik Evaluasi
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # --- MANUAL LOGGING (Syarat Skor Skilled) ---
            # Log Parameter
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("model_type", "RandomForestRegressor")
            
            # Log Metrik
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Log Model Arsitektur/Artefak
            mlflow.sklearn.log_model(model, "model_saham_antam")
            
            print(f"Run Sukses untuk n_estimators={n_est} -> R2 Score: {r2:.4f}")
            
    print("=== Semua Proses Training Selesai! ===")

if __name__ == '__main__':
    train_saham_model()