import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Prediksi Jurusan Kuliah", layout="centered")

# Judul
st.title("🎓 Prediksi Jurusan Kuliah")
st.write("Masukkan data siswa di bawah ini untuk memprediksi jurusan yang cocok.")

# Input nilai akademik
math = st.slider("Nilai Matematika", 0, 100, 75)
english = st.slider("Nilai Bahasa Inggris", 0, 100, 75)
science = st.slider("Nilai IPA", 0, 100, 75)
social = st.slider("Nilai IPS", 0, 100, 75)

# Input minat
minat_ipa = st.selectbox("Minat IPA", ["Low", "Medium", "High"])
minat_ips = st.selectbox("Minat IPS", ["Low", "Medium", "High"])
minat_bahasa = st.selectbox("Minat Bahasa", ["Low", "Medium", "High"])

# Input latar belakang
ekonomi = st.selectbox("Ekonomi Keluarga", ["Low", "Medium", "High"])
tipe_sekolah = st.selectbox("Tipe Sekolah", ["SMA", "SMK", "MA"])

# Mapping semua input non-numerik ke angka
def map_input(val):
    return {"Low": 0, "Medium": 1, "High": 2}[val]

minat_ipa_num = map_input(minat_ipa)
minat_ips_num = map_input(minat_ips)
minat_bahasa_num = map_input(minat_bahasa)
ekonomi_num = map_input(ekonomi)
tipe_sekolah_map = {"SMA": 0, "SMK": 1, "MA": 2}
tipe_sekolah_num = tipe_sekolah_map[tipe_sekolah]

# Simulasi data training
@st.cache_data
def generate_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "math": np.random.randint(50, 100, 500),
        "english": np.random.randint(50, 100, 500),
        "science": np.random.randint(50, 100, 500),
        "social": np.random.randint(50, 100, 500),
        "minat_ipa": np.random.randint(0, 3, 500),
        "minat_ips": np.random.randint(0, 3, 500),
        "minat_bahasa": np.random.randint(0, 3, 500),
        "ekonomi": np.random.randint(0, 3, 500),
        "sekolah": np.random.randint(0, 3, 500)
    })

    def get_label(row):
        if row["minat_ipa"] == 2 and row["science"] > 75:
            return "IPA"
        elif row["minat_ips"] == 2 and row["social"] > 75:
            return "IPS"
        elif row["minat_bahasa"] == 2 and row["english"] > 75:
            return "Bahasa"
        else:
            return np.random.choice(["IPA", "IPS", "Bahasa"])

    df["jurusan"] = df.apply(get_label, axis=1)
    return df

df = generate_data()

# Fitur & label
X = df[["math", "english", "science", "social", "minat_ipa", "minat_ips", "minat_bahasa", "ekonomi", "sekolah"]]
y = df["jurusan"]

# Normalisasi dan training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = DecisionTreeClassifier()
model.fit(X_scaled, y)

# Tombol prediksi
if st.button("Prediksi Jurusan"):
    input_data = np.array([[math, english, science, social,
                            minat_ipa_num, minat_ips_num, minat_bahasa_num,
                            ekonomi_num, tipe_sekolah_num]])
    input_scaled = scaler.transform(input_data)
    prediksi = model.predict(input_scaled)
    st.success(f"✅ Jurusan yang direkomendasikan: **{prediksi[0]}**")

