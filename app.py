import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model_prediksi_jurusan.pkl")
scaler = joblib.load("scaler.pkl")

# Label encoder untuk menampilkan hasilnya
label_dict = {0: "Bahasa", 1: "IPA", 2: "IPS"}

st.title("🎓 Prediksi Jurusan Kuliah")
st.write("Masukkan data siswa di bawah ini untuk memprediksi jurusan yang cocok.")

# Input nilai akademik
nilai_matematika = st.slider("Nilai Matematika", 0, 100, 75)
nilai_bahasa_inggris = st.slider("Nilai Bahasa Inggris", 0, 100, 75)
nilai_ipa = st.slider("Nilai IPA", 0, 100, 75)
nilai_ips = st.slider("Nilai IPS", 0, 100, 75)

# Input minat
minat_ipa = st.selectbox("Minat IPA", ["Low", "Medium", "High"])
minat_ips = st.selectbox("Minat IPS", ["Low", "Medium", "High"])
minat_bahasa = st.selectbox("Minat Bahasa", ["Low", "Medium", "High"])

# Input ekonomi dan sekolah
ekonomi_keluarga = st.selectbox("Ekonomi Keluarga", ["Low", "Mid", "High"])
tipe_sekolah = st.selectbox("Tipe Sekolah", ["SMA", "SMK"])

# Mapping input ke angka
map_dict = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "SMA": 1,
    "SMK": 0
}
input_data = np.array([
    nilai_matematika,
    nilai_bahasa_inggris,
    nilai_ipa,
    nilai_ips,
    map_dict[minat_ipa],
    map_dict[minat_ips],
    map_dict[minat_bahasa],
    map_dict[ekonomi_keluarga],
    map_dict[tipe_sekolah]
]).reshape(1, -1)

# Prediction
if st.button("Prediksi Jurusan"):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    st.success(f"✅ Jurusan yang direkomendasikan: **{label_dict[prediction]}**")
