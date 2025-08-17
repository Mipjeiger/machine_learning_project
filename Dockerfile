# gunakan base image python
FROM python:3.13.3

# atur direktori kerja di dalam kontainer
WORKDIR /app

# Menambahkan dependensi sistem yang umum dibutuhkan
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# salin file requirement.txt
COPY requirements.txt .

# install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# salin model dan file aplikasi
COPY ml_analysis_fraud.dill .
COPY app.py .

# menyalin kode aplikasi lainnya
COPY . .

# Ekspos port yang digunakan
EXPOSE 5000

# jalankan aplikasi saat kontainer dimulai
CMD [ "python", "app.py" ]