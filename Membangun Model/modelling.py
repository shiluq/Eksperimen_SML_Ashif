import os
# KUNCI UTAMA: Paksa MLflow mengizinkan penyimpanan berbasis folder biasa di Windows
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def train_saham_basic():
    # 1. Pastikan tracking mengarah ke folder lokal 'mlruns'
    mlflow.set_tracking_uri("file:///D:/Decoding/Modul%205/Eksperimen_SML_Ashif/Membangun%20Model/mlruns")
    
    # 2. Aktifkan autolog
    mlflow.autolog()

    # 3. Set nama eksperimen
    mlflow.set_experiment("Eksperimen_Saham_ANTM_Basic")

    print("--- 1. Membaca Data Bersih ANTM ---")
    df = pd.read_csv("namadataset_preprocessing.csv")
    print(f"Berhasil memuat data! Total: {len(df)} baris.")

    # Membuat fitur lag untuk data saham
    df['close_lag1'] = df['close'].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = df[['close_lag1', 'open', 'high', 'low', 'volume']]
    y = df['close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("\n--- 2. Memulai Training Model dengan Autolog (Folder Lokal) ---")
    with mlflow.start_run(run_name="RandomForest_Basic_Default"):
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        print(f"\nSukses! Model dilatih dengan R2 Score: {r2:.4f}")
        print("Folder 'mlruns' DIJAMIN sukses terisi data eksperimen!")

if __name__ == '__main__':
    train_saham_basic()