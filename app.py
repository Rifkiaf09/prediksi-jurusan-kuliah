import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.title("🎓 Aplikasi Prediksi Jurusan Kuliah")
st.write("Masukkan data siswa di bawah untuk memprediksi jurusan yang sesuai.")

# Input fitur dari pengguna
math = st.slider("Nilai Matematika", 0, 100, 75)
english = st.slider("Nilai Bahasa Inggris", 0, 100, 75)
science = st.slider("Nilai IPA", 0, 100, 75)
social = st.slider("Nilai IPS", 0, 100, 75)
interest = st.selectbox("Minat Siswa", ["IPA", "IPS", "Bahasa"])

# Mapping minat ke numerik
interest_map = {"IPA": 0, "IPS": 1, "Bahasa": 2}
interest_num = interest_map[interest]

# Simulasi data training
@st.cache_data
def load_data():
    np.random.seed(42)
    data = {
        "math_score": np.random.randint(50, 100, 500),
        "english_score": np.random.randint(50, 100, 500),
        "science_score": np.random.randint(50, 100, 500),
        "social_score": np.random.randint(50, 100, 500),
        "interest": np.random.randint(0, 3, 500)
    }

    df = pd.DataFrame(data)
    
    def label_row(row):
        if row["interest"] == 0 and row["science_score"] > 75:
            return "IPA"
        elif row["interest"] == 1 and row["social_score"] > 75:
            return "IPS"
        elif row["interest"] == 2 and row["english_score"] > 75:
            return "Bahasa"
        else:
            return np.random.choice(["IPA", "IPS", "Bahasa"])

    df["jurusan"] = df.apply(label_row, axis=1)
    return df

df = load_data()

# Split fitur dan label
X = df[["math_score", "english_score", "science_score", "social_score", "interest"]]
y = df["jurusan"]

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training model
model = DecisionTreeClassifier()
model.fit(X_scaled, y)

# Prediksi saat tombol ditekan
if st.button("🔮 Prediksi Jurusan"):
    user_data = np.array([[math, english, science, social, interest_num]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    st.success(f"Rekomendasi jurusan untuk siswa ini adalah: **{prediction}**")

# --- Visualisasi Data ---

st.subheader("📊 Distribusi Minat Siswa")
interest_labels = {0: "IPA", 1: "IPS", 2: "Bahasa"}
df["interest_label"] = df["interest"].map(interest_labels)
fig1, ax1 = plt.subplots()
sns.countplot(x="interest_label", data=df, ax=ax1, palette="pastel")
ax1.set_xlabel("Minat Siswa")
ax1.set_ylabel("Jumlah")
st.pyplot(fig1)

st.subheader("📈 Jurusan yang Direkomendasikan Berdasarkan Minat")
fig2, ax2 = plt.subplots()
sns.countplot(x="jurusan", hue="interest_label", data=df, palette="Set2", ax=ax2)
ax2.set_xlabel("Jurusan")
ax2.set_ylabel("Jumlah")
st.pyplot(fig2)
