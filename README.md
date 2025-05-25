# 🎓 Prediksi Jurusan Kuliah

Proyek machine learning ini bertujuan untuk memprediksi jurusan kuliah siswa berdasarkan nilai akademik, minat, dan latar belakang.

## 🔧 Teknologi yang Digunakan
- Python, Pandas, Scikit-Learn
- Streamlit untuk aplikasi web
- GitHub untuk version control

## 📁 File Penting
- `app.py` → aplikasi web
- `Prediksi_Jurusan_Kuliah.ipynb` → training model
- `model_prediksi_jurusan.pkl` → model siap pakai
- `requirements.txt` → untuk setup

## 🚀 Coba Aplikasinya
👉 [Klik di sini untuk mencoba]((https://prediksi-jurusan-kuliah-vu6fabcfcuzd5nnqcam36g.streamlit.app/))

## 📊 Dataset
Dataset disimulasikan untuk latihan, berisi nilai rapor, minat, dan jurusan yang dipilih dari nilai 500 siswa.

## Note:
- Dataset yang digunakan bersifat buatan (sintetik), sehingga hasil prediksi tidak bisa dijadikan acuan dalam keputusan nyata.
- Menggunakan model Decision Tree tanpa tuning hyperparameter atau validasi silang, sehingga performa dan akurasi belum optimal.
- Tidak ada metrik performa seperti akurasi, precision, recall yang ditampilkan kepada pengguna.
- Tidak ada integrasi dengan data pendidikan asli atau sistem pendidikan nasional.
- Prediksi bisa berubah jika kondisi input ambigu karena ada elemen acak di model pelabelan data saat pelatihan.
