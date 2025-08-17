from flask import Flask, request, jsonify
import dill # Digunakan untuk memuat objek Python yang diserialisasi
import numpy as np # Digunakan untuk operasi array numerik
import warnings
warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)

app = Flask(__name__)


MODEL_PATH = 'ml_analysis_fraud.dill' # path model what we want to test 


# --- Variabel Global untuk Model dan Data yang Dimuat ---
loaded_vars = None # Akan menyimpan seluruh kamus 'important_vars' dari file dill
prediction_model = None # Akan menyimpan objek model yang akan digunakan untuk prediksi
df_axis = None # Akan menyimpan DataFrame df_axis jika ada dalam file dill
fraud_data = None # Akan menyimpan DataFrame fraud_data jika ada dalam file dill
nofraud_data = None # Akan menyimpan DataFrame nofraud_data jika ada dalam file dill


# --- Memuat Model dan Variabel Penting Saat Aplikasi Dimulai ---
try:
    with open(MODEL_PATH, 'rb') as f:
        loaded_vars = dill.load(f)
    print(f"Semua variabel berhasil dimuat dari '{MODEL_PATH}' menggunakan dill!")

    # Mengakses df_axis jika ada. Ini mungkin berguna untuk pra-pemrosesan data input
    # atau untuk memahami fitur yang diharapkan model.
    if 'df_axis' in loaded_vars and loaded_vars['df_axis'] is not None:
        df_axis = loaded_vars['df_axis']
        print("DataFrame 'df_axis' berhasil diakses.")

    else:
        print("PERINGATAN: 'df_axis' tidak ditemukan dalam file dill atau bernilai None.")

    # Mengakses fraud_data jika ada. Ini akan berguna dalam pra-pemrosesan data input

    if 'fraud_data' in loaded_vars and loaded_vars['fraud_data'] is not None:
        fraud_data = loaded_vars['fraud_data']
        print("DataFrame 'fraud_data' berhasil di akses")
    elif 'nofraud_data' in loaded_vars and loaded_vars['nofraud_data'] is not None:
        nofraud_data = loaded_vars['nofraud_data']
        print("DataFrame 'nofraud_data' berhasil di akses")
    else:
        print("PERINGATAN: 'fraud_data' dan 'nofraud_data' tidak ditemukan dalam file dill atau bernilai None.")

    # --- Memilih Model yang Akan Digunakan untuk Prediksi ---
    # Logika ini akan mencoba menemukan model yang paling sesuai dari kamus 'loaded_vars'.
    # Urutan prioritas dapat disesuaikan sesuai kebutuhan Anda.
    if 'model' in loaded_vars and loaded_vars['model'] is not None:
        prediction_model = loaded_vars['model']
        print("Menggunakan 'model' (Keras atau lainnya) dari variabel yang dimuat untuk prediksi.")
    elif 'log_reg' in loaded_vars and loaded_vars['log_reg'] is not None:
        prediction_model = loaded_vars['log_reg']
        print("Menggunakan 'log_reg' (Logistic Regression) dari variabel yang dimuat untuk prediksi.")
    elif 'classifiers' in loaded_vars and loaded_vars['classifiers'] is not None:
        prediction_model = loaded_vars['classifiers']
        print("Menggunakan 'classifiers' (n=4) dari variabel yang dimuat untuk prediksi")
    elif 'svc' in loaded_vars and loaded_vars['svc'] is not None:
        prediction_model = loaded_vars['svc']
        print("Menggunakan 'svc' (Support Vector Classifier) dari variabel yang dimuat untuk prediksi.")
    elif 'DTC_clf' in loaded_vars and loaded_vars['DTC_clf'] is not None:
        prediction_model = loaded_vars['DTC_clf']
        print("Menggunakan 'DTC_clf' (Decision Tree Classifier) dari variabel yang dimuat untuk prediksi.")
    elif 'knears_neighbors' in loaded_vars and loaded_vars['knears_neighbors'] is not None:
        prediction_model = loaded_vars['knears_neighbors']
        print("Menggunakan 'knears_neighbors' (K-Nearest Neighbors) dari variabel yang dimuat untuk prediksi.")
    elif 'oversample_model' in loaded_vars and loaded_vars['oversample_model'] is not None:
        prediction_model = loaded_vars['oversample_model']
        print("Menggunakan 'undersample_model' dari variabel yang dimuat untuk prediksi.")
    elif 'undersample_model' in loaded_vars and loaded_vars['undersample_model'] is not None:
        prediction_model = loaded_vars['undersample_model']
        print("Menggunakan 'undersample_model' dari variabel yang dimuat untuk prediksi.")
    elif 'sm' in loaded_vars and loaded_vars['sm'] is not None:
        prediction_model = loaded_vars['sm']
        print("Menggunakan 'undersample_model' dari variabel yang dimuat untuk prediksi.")
    else:
        print("PERINGATAN: Tidak ada model yang dikenali ('model', 'log_reg', 'svc', 'DTC_clf', 'knears_neighbors', 'undersample_model') dalam file dill.")
        print("Endpoint prediksi mungkin tidak berfungsi karena tidak ada model yang tersedia.")
        prediction_model = None # Pastikan model tetap None jika tidak ada yang ditemukan

except FileNotFoundError:
    print(f"ERROR: File model '{MODEL_PATH}' tidak ditemukan.")
    print("Pastikan jalur file model sudah benar dan file ada di lokasi tersebut.")
    loaded_vars = None
    prediction_model = None
except Exception as e:
    print(f"Error lain saat memuat model atau variabel dari '{MODEL_PATH}': {e}")
    loaded_vars = None
    prediction_model = None


# --- Endpoint Prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah model berhasil dimuat sebelum mencoba prediksi
    if prediction_model is None:
        return jsonify({"error": "Model belum dimuat. Periksa log server untuk detail kesalahan saat startup."}), 500

    try:
        # Mendapatkan data dari request JSON
        data = request.json

        # Validasi dasar payload JSON
        if data is None:
            return jsonify({"error": "Payload JSON tidak ditemukan. Pastikan di Headers Postman untuk key nya 'Content-Type' adalah 'application/json' dan body untuk raw tidak kosong."}), 400
        if 'data' not in data:
            return jsonify({"error": "Kunci 'data' tidak ditemukan dalam payload JSON. Pastikan format input adalah {'data': [...]}."}), 400


        input_features = np.array([data['data']])

        predictions = prediction_model.predict(input_features)

        if predictions.ndim == 1 and predictions.shape[0] == 0 and np.all(np.isin(predictions, [0, 1])):
            predictions = 1 - predictions # Membalikkan 0 menjadi 1, dan 1 menjadi 0
            print("Prediksi biner dibalikkan.")
        elif predictions.ndim == 2 and predictions.shape[1] == 1 and np.all(np.isin(predictions, [0, 1])):
            predictions = 1 - predictions # Membalikkan 0 menjadi 1, dan 1 menjadi 0 untuk output 2D (1 kolom)
            print("Prediksi biner (2D) dibalikkan.")

        return jsonify({'prediction': predictions.tolist()})

    except Exception as e:
        # Tangani error yang terjadi selama proses prediksi
        print(f"Error saat memproses request prediksi: {e}")
        return jsonify({"error": f"Terjadi kesalahan saat memproses request: {e}"}), 500

# --- Menjalankan Aplikasi Flask ---
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
