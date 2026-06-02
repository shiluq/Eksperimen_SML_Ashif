# preprocessing/automate_shiluq.py
import os
import pandas as pd
import numpy as np

def run_preprocessing():
    print("=[ Proses Otomatisasi Preprocessing Saham ANTM Dimulai ]=")
    
    # path input dan output
    input_path = os.path.join('..', 'namadataset_raw', 'ANTM.csv')
    output_dir = os.path.join('..', 'namadataset_preprocessing')
    output_file = os.path.join(output_dir, 'ANTM_clean.csv')
    
    # 1. Validasi keberadaan data mentah
    if not os.path.exists(input_path):
        print(f"Error: File mentah tidak ditemukan di {input_path}!")
        return
        
    # 2. Load Dataset
    print("-> Membaca data mentah...")
    df = pd.read_csv(input_path)
    
    # 3. Preprocessing & Pembersihan Data
    print("-> Melakukan transformasi data...")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Handling Missing Values jika ada
    df = df.dropna()
    
    # Feature Engineering (Lag Features untuk Prediksi Saham)
    # Membuat fitur berbasis harga penutupan hari sebelumnya
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Volume_Lag1'] = df['Volume'].shift(1)
    
    # Hapus baris kosong akibat shifting lag
    df_clean = df.dropna().reset_index(drop=True)
    
    # 4. Membikin direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"-> Folder {output_dir} berhasil dibuat.")
        
    # 5. Simpan Hasil Preprocessing
    df_clean.to_csv(output_file, index=False)
    print(f"=[ Sukses! Data hasil preprocessing disimpan di: {output_file} ]=")

if __name__ == "__main__":
    run_preprocessing()