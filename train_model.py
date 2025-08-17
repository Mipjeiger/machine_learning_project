import mlflow
import pandas as pd
import mlflow.sklearn
import dill
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Langkah 1: Memuat dan Membersihkan Data ---
try:
    df = pd.read_csv('fraud_data.csv')
except FileNotFoundError:
    print("Error: File 'fraud_data.csv' tidak ditemukan. Pastikan file ada di direktori kerja.")
    exit()

# Cek nama kolom yang sebenarnya, lalu ganti nama jika perlu
# Diasumsikan kolom target Anda adalah 'is_fraud' dan ingin dinamai ulang menjadi 'fraud'
# Jika kolom sudah bernama 'fraud' sejak awal, baris ini bisa dilewati.
# Atau jika kolom 'fraud' sudah ada dan 'is_fraud' tidak, maka:

try:
    df.rename(columns={'is_fraud': 'fraud'}, inplace=True) 
except KeyError:
    print("Error: Kolom 'is_fraud' tidak ditemukan di DataFrame. Periksa nama kolom target Anda.")
    print("Daftar kolom yang tersedia:")
    print(df.columns.tolist())
    exit()

# --- Perbaikan Logika Pembersihan Data ---
# Pertama, periksa nilai unik di kolom target Anda ('fraud')
# Ini akan membantu Anda menemukan nilai yang aneh
print("Nilai unik di kolom 'fraud' sebelum pembersihan:")
print(df['fraud'].unique())

# Anda perlu menghapus baris yang memiliki nilai aneh atau tidak valid.
# Berdasarkan kode Anda sebelumnya, ada nilai seperti '1"2020-12-24 16:56:24"'.
# Mari kita bersihkan itu.
# Anda bisa menggunakan regular expression untuk menghapus karakter non-numerik jika perlu.
# Cara yang lebih aman adalah dengan menghapus baris yang tidak valid.
df = df[df['fraud'].astype(str).str.isnumeric()]
df['fraud'] = df['fraud'].astype(int)

# Sekarang, periksa kembali nilai unik di kolom 'fraud'
print("\nNilai unik di kolom 'fraud' setelah pembersihan:")
print(df['fraud'].unique())
print("\nJumlah data per kelas:")
print(df['fraud'].value_counts())


# --- Langkah 2: Memisahkan Fitur (X) dan Target (y) ---
# Pastikan Anda memisahkan kolom target ('fraud') dari fitur.
X = df.drop(columns=['trans_date_trans_time', 'merchant', 'category', 'city', 'state', 'job', 'dob', 'trans_num', 'fraud', 'city_pop'])
y = df['fraud']

# --- Langkah 3: Periksa Jumlah Kelas Minoritas ---
# Ini adalah inti dari masalah Anda.
# Periksa apakah ada kelas dengan jumlah kurang dari 2.
min_class_count = y.value_counts().min()
if min_class_count < 2:
    print(f"\nError: Kelas terkecil hanya memiliki {min_class_count} anggota. Tidak bisa menggunakan stratify.")
    print("Pertimbangkan untuk menghapus baris ini, atau jangan gunakan 'stratify'.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    # Jika jumlahnya cukup, baru gunakan stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- Langkah 4: MLflow Tracking (jika data sudah siap) ---
with mlflow.start_run():
    # Latih model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    # Lakukan prediksi dan hitung metrik
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log parameter dan metrik
    mlflow.log_param("solver", 'liblinear')
    mlflow.log_metric("accuracy", accuracy)

    # Log model ke MLflow
    mlflow.sklearn.log_model(model, "fraud-detection-model")