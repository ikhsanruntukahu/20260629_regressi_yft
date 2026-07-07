import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from PIL import Image

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# LANDING PAGE
logo = Image.open("_ MDPI Primary Logo.png")
st.set_page_config(
    page_title="Prediksi Ukuran YFT",
    page_icon=logo,
    layout="wide"
)


#Konfigurasi Awal dan Styling Halaman
st.markdown("""
<style>
.block-container{
    padding-top:2rem;
}
</style>
""", unsafe_allow_html=True)

#Import logo/gambar

logo = Image.open("_ MDPI Primary Logo.png")

# Layout logo
col1, col2 = st.columns([1, 4])

st.markdown("""
<style>
.block-container{
    padding-top:2rem;
}
</style>
""", unsafe_allow_html=True)

with col1:
    st.image(logo, width=200)

# Judul
st.markdown(
    """
    <h1 style='text-align:center; font-size:40px; color:#0E4C92;'>
    Model Regresi Random Forest
    </h1>
    <h3 style='text-align:center; color:#444444;'>
    Prediksi Ukuran Panjang Ikan Yellowfin Tuna berdasarkan parameter lingkungan dan teknik penangkapan
    </h3>
    <h4 style='text-align:center; color:#666666;'>
    Studi Kasus Perairan Pulau Buru Tahun 2020-2025
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Informasi tambahan
st.info("""### Informasi Model

Aplikasi ini dibangun untuk memprediksi **ukuran panjang ikan Yellowfin Tuna (*Thunnus albacares*)**
menggunakan algoritma **Random Forest Regression**.

**Variabel prediktor yang digunakan:**
- Bulan
- Sea Surface Temperature (SST)
- Chlorophyll-a (Chl-a)
- Teknik penangkapan
- Kedalaman minimum
- Kedalaman maksimum

**Target prediksi:**
- Panjang ikan Yellowfin Tuna (cm)

**Sumber data:**
Data hasil port sampling **Yayasan MDPI** di **Pulau Buru** periode **2020–2025** yang telah melalui proses preprocessing dan pelatihan model machine learning.""")


#Petunjuk data upload
st.markdown(
    """
    <h3 style="font-size:18px; color:#0E4C92;">
    Petunjuk Struktur Kolom File Excel
    </h3>
    """,
    unsafe_allow_html=True)



petunjuk = pd.DataFrame({
    "Nama Kolom": [
        "Bulan",
        "sst",
        "chl-a",
        "Teknik_penangkapan",
        "kedalaman_min",
        "kedalaman_max",
        "panjang_yft"
    ],
    "Tipe Data": [
        "Teks (Jan, Feb, Mar)",
        "Numerik (°C)",
        "Numerik (mg/m³)",
        "Teks (Rumpon / Non Rumpon)",
        "Numerik (meter)",
        "Numerik (meter)",
        "Numerik (cm)"
    ],
    "Status": [
        "Wajib",
        "Wajib",
        "Wajib",
        "Wajib",
        "Wajib",
        "Wajib",
        "Wajib (untuk evaluasi model)"
    ],
    "Keterangan": [
        "Bulan penangkapan",
        "Sea Surface Temperature",
        "Konsentrasi Chlorophyll-a",
        "Metode penangkapan",
        "Kedalaman minimum penangkapan",
        "Kedalaman maksimum penangkapan",
        "Panjang aktual ikan"
    ]
})

# Menggunakan width='stretch' agar kompatibel penuh dengan Pandas & Streamlit versi 2026 terbaru
st.dataframe(
    petunjuk,
    use_container_width=True, # Melebarkan tabel mengikuti lebar layar
    hide_index=True,          # Menghilangkan angka 0, 1, 2... di sisi kiri
    height=300                # Mengatur tinggi agar ke-7 baris terlihat semua
)


# Load model
my_model = pickle.load(open("20260622_model_regresi_yft_buru.pkl", "rb"))

# Upload Excel
uploaded_file = st.file_uploader(
    "Upload file Excel",
    type=["xlsx"]
)

#Blok pengecekan kondisi awal
if uploaded_file is None:
    st.info("**Silakan masukan data Anda:** Unggah file Excel (.xlsx) yang sesuai dengan struktur kolom di atas untuk memproses analisis model regresi dan melihat hasil prediksi visual.")
    
    # Tampilkan footer standar saat data masih kosong
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; margin-top: 30px; margin-bottom: 20px; color: #666666;'>
            <p style='font-size: 16px;'><strong>Dashboard Prediksi Ukuran Yellowfin Tuna (<i>Thunnus albacares</i>)</strong></p>
            <p style='font-size: 14px;'>Dikembangkan untuk mendukung analisis data dan pengelolaan perikanan tuna yang berkelanjutan.</p>
            <p style='font-size: 12px; margin-top: 10px;'>&copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop() # Menghentikan aplikasi di sini dengan aman sebelum membaca variabel di bawah!
    
# Membaca data
df = pd.read_excel(uploaded_file)

st.subheader("Preview Data")
st.dataframe(df)

# Feature dan target
fitur = [
    "Bulan",
    "sst",
    "chl-a",
    "Teknik_penangkapan",
    "kedalaman_min",
    "kedalaman_max"]

target = "panjang_yft"

# Cek apakah semua kolom tersedia
kolom_harus_ada = fitur + [target]

if not all(col in df.columns for col in kolom_harus_ada):
    st.error("Kolom pada file Excel tidak sesuai.")
    st.stop()

X = df[fitur]
y = df[target]

# Buat informasi statistik

st.subheader("Ringkasan Statistik Data")

#missing value
missing_value = df.isnull().sum()
    
# 1. Statistik Kolom Numerik
st.markdown("##### Variabel Numerik")
    
# Mengambil statistik deskriptif dasar
stat_numerik = df.describe().T 
    
# Menghitung nilai median dan modus
modus_numerik = df.select_dtypes(include='number').mode().iloc[0]
    
# Menambahkan kolom median, modus, dan jumlah kosong (missing values)
stat_numerik['modus'] = modus_numerik
stat_numerik['NA'] = missing_value
    
# Membulatkan semua nilai desimal menjadi 2 angka di belakang koma
stat_numerik = stat_numerik.round(2)
    
# Mengubah nama indeks menjadi kolom agar bisa disembunyikan indeks bawaannya
stat_numerik.insert(0, "Nama Variabel", stat_numerik.index)
    
# Menghitung tinggi dinamis agar pas (jumlah baris * 35px) + 80px buffer header/padding
tinggi_numerik = (len(stat_numerik) * 35) + 80
    
st.dataframe(
    stat_numerik, 
    use_container_width=True, 
    hide_index=True,
    height=int(tinggi_numerik)
    )

# 2. Statistik Kolom Kategorikal / Teks
if df.select_dtypes(include=['object']).shape[1] > 0:
    st.markdown("##### Variabel Kategori")
        
# Mengambil statistik deskriptif kategori
stat_kategori = df.describe(include=['object']).T 
        
# Menambahkan kolom jumlah kosong (missing values) khusus kategori
stat_kategori['NA'] = missing_value
        
#Mengubah nama indeks menjadi kolom
stat_kategori.insert(0, "Nama Variabel", stat_kategori.index)
        
# Menghitung tinggi dinamis untuk tabel kategori
tinggi_kategori = (len(stat_kategori) * 35) + 80
        
st.dataframe(
    stat_kategori, 
    use_container_width=True, 
    hide_index=True,
    height=int(tinggi_kategori))

    # ==========================================


# ==========================================
# Analisis Korelasi Variabel Numerik

st.subheader("Matriks Korelasi")
    
# Menentukan variabel 
corr_vars = [
    'panjang_yft',
    'chl-a',
    'kedalaman_min',
    'kedalaman_max',
    'sst',]

# Memastikan semua kolom tersedua
if all(col in df.columns for col in corr_vars):
        corr_matrix = df[corr_vars].corr()

        # Menggunakan struktur figure dari matplotlib untuk disalurkan ke Streamlit
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='Blues',
            fmt='.2f',
            ax=ax_corr
        )

        ax_corr.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Menampilkan plot ke web Streamlit
        st.pyplot(fig_corr)
        plt.close(fig_corr)
else:
        st.warning("Beberapa kolom yang dibutuhkan untuk korelasi tidak ditemukan.")
    # ==========================================



# Prediksi
prediksi = my_model.predict(X)
df["Prediksi_panjang_yft"] = prediksi

# ===========================
# Evaluasi Model
# ===========================
r2 = r2_score(y, prediksi)
mae = mean_absolute_error(y, prediksi)
rmse = np.sqrt(mean_squared_error(y, prediksi))
n_samples = len(y) # Menghitung jumlah sample

st.markdown("### Evaluasi Model")

# Tambahan CSS agar st.metric terlihat seperti 'Card'
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f7f9fc;
        border: 1px solid #e1e4e8;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #0E4C92; /* Garis aksen biru di sebelah kiri */
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

# Menambahkan argumen 'help' untuk tooltip dan merapikan format label
col1.metric(
        label="R² Score", 
        value=f"{r2:.3f}".replace(".", ","), # Output: 0,733
        help="Koefisien Determinasi: Semakin mendekati 1, model semakin mampu memprediksi dengan akurat."
    )
col2.metric(
        label="MAE", 
        value=f"{mae:.2f}".replace(".", ",") + " cm", # Output: 11,60 cm
        help="Mean Absolute Error: Rata-rata selisih jarak (error) antara prediksi dan nilai aktual."
    )
col3.metric(
        label="RMSE", 
        value=f"{rmse:.2f}".replace(".", ",") + " cm", # Output: 19,30 cm
        help="Root Mean Square Error: Memberikan 'penalti' lebih besar pada error yang melenceng jauh."
    )
col4.metric(
        label="Total Sample", 
        value=f"{n_samples:,}".replace(",", "."), # Mengubah format 15194 menjadi 15.194
        help="Jumlah baris data observasi (ikan) yang dievaluasi."
    )


# Scatter Plot
# ===========================

fig, ax = plt.subplots(figsize=(3, 3))

ax.scatter(y, prediksi, color="orange", alpha=1, s=5 )

# Garis referensi y=x
ax.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--",
    linewidth=0.6
)

# Ukuran label sumbu
ax.set_xlabel("Aktual (cm)", fontsize=6)
ax.set_ylabel("Prediksi (cm)", fontsize=6)

# Ukuran judul
ax.set_title("Scatter plot Aktual vs Prediksi", fontsize=5, fontweight="bold")

# Ukuran angka pada sumbu
ax.tick_params(axis="both", labelsize=5)

# pasang grid
ax.grid(True, linestyle="--", alpha=0.5)

st.pyplot(fig)
plt.close(fig)
   # ===========================
# Feature Importance
# ===========================

st.subheader("Feature Importance")

preprocessor = my_model.named_steps["preprocessing"]
rf_model = my_model.named_steps["rf"]

# bersihkan nama feature
feature_names = [
    f.split("__")[-1]
    for f in preprocessor.get_feature_names_out()
]

importance = rf_model.feature_importances_

fi = (
    pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

fi.insert(0, "Ranking", range(1, len(fi)+1))

st.dataframe(fi, 
             use_container_width=True,
             hide_index=True)


# ===========================
# Distribusi Panjang Ikan (Aktual & Prediksi)

st.subheader("Distribusi Panjang Ikan Yellowfin Tuna")

    # Hitung persentase untuk data AKTUAL
total_akt = len(df)
below_70_akt = (df["panjang_yft"] < 70).sum()
above_70_akt = (df["panjang_yft"] >= 70).sum()
pct_below_akt = below_70_akt / total_akt * 100
pct_above_akt = above_70_akt / total_akt * 100

    # Hitung persentase untuk data PREDIKSI
total_pred = len(df)
below_70_pred = (df["Prediksi_panjang_yft"] < 70).sum()
above_70_pred = (df["Prediksi_panjang_yft"] >= 70).sum()
pct_below_pred = below_70_pred / total_pred * 100
pct_above_pred = above_70_pred / total_pred * 100

    # Tentukan bin range
bin_range = np.arange(10, 200, 1)

    # Plot dengan 1 baris, 2 kolom (bersebelahan)
fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ---------------------------------------------------------
# Grafik 1: Data Aktual (Kiri)

sns.histplot(df["panjang_yft"], bins=bin_range, color="skyblue", kde=True, alpha=0.8, ax=ax1)
ax1.axvline(70, color="red", linestyle="--", linewidth=2, label="Batas Minimum 70 cm")
    
y_max1 = ax1.get_ylim()[1]
ax1.text(10, y_max1 * 0.9, f"{pct_below_akt:.1f}% < 70 cm", fontsize=12, color="red", weight="bold")
ax1.text(80, y_max1 * 0.9, f"{pct_above_akt:.1f}% ≥ 70 cm", fontsize=12, color="green", weight="bold")
    
ax1.set_title("Distribusi Data Aktual", fontsize=14, fontweight="bold")
ax1.set_xlabel("Panjang Ikan (cm)", fontsize=12)
ax1.set_ylabel("Frekuensi", fontsize=12)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.set_xticks(np.arange(10, 200, 20))
ax1.legend()

# ---------------------------------------------------------
# Grafik 2: Data Prediksi (Kanan)

sns.histplot(df["Prediksi_panjang_yft"], bins=bin_range, color="gold", kde=True, alpha=0.8, ax=ax2)
ax2.axvline(70, color="red", linestyle="--", linewidth=2, label="Batas Minimum 70 cm")
    
y_max2 = ax2.get_ylim()[1]
ax2.text(10, y_max2 * 0.9, f"{pct_below_pred:.1f}% < 70 cm", fontsize=12, color="red", weight="bold")
ax2.text(80, y_max2 * 0.9, f"{pct_above_pred:.1f}% ≥ 70 cm", fontsize=12, color="green", weight="bold")
    
ax2.set_title("Distribusi Data Prediksi", fontsize=14, fontweight="bold")
ax2.set_xlabel("Panjang Ikan (cm)", fontsize=12)
ax2.set_ylabel("Frekuensi", fontsize=12)
ax2.grid(axis="y", linestyle="--", alpha=0.5)
ax2.set_xticks(np.arange(10, 200, 20))
ax2.legend()

    # Rapikan jarak antar grafik
plt.tight_layout()
    
    # Tampilkan ke Streamlit
st.pyplot(fig_dist)
plt.close(fig_dist)
# ===========================

# Distribusi Prediksi Berdasarkan Kelompok Suhu (SST)
st.subheader("Pengaruh Suhu Permukaan Laut (SST) terhadap Prediksi Panjang Ikan")

# Membuat DataFrame copy untuk analisis agar tidak mengganggu data utama
df_analisis = df.copy()

# Membuat "Kelompok Suhu" (Binning)
batas_suhu = [28, 29, 30, 33] 
nama_kelompok = ['28 - 29°C', '29 - 30°C', '> 30°C']

df_analisis['Kelompok_Suhu'] = pd.cut(df_analisis['sst'], bins=batas_suhu, labels=nama_kelompok)

# Membuat Visualisasi Boxplot untuk Streamlit
fig_suhu, ax_suhu = plt.subplots(figsize=(8, 5))
    
# Mendefinisikan gradasi warna suhu secara manual (Biru -> Oranye -> Merah)
gradasi_suhu = ["#3498db", "#f39c12", "#e74c3c"] 

sns.boxplot(
    data=df_analisis, 
    x='Kelompok_Suhu', 
    y='Prediksi_panjang_yft', # Sesuai dengan kolom prediksi
    hue='Kelompok_Suhu',      # Menambahkan hue berdasarkan kelompok suhu
    palette=gradasi_suhu,     # Menerapkan gradasi warna kustom
    legend=False,             # Menyembunyikan legend karena sumbu X sudah menjelaskan kategori
    dodge=False,
    ax=ax_suhu
    )

# Merapikan tampilan grafik
ax_suhu.set_title('Distribution of Predicted Tuna Length by Sea Surface Temperature (SST) Groups', fontsize=10, pad=15, fontweight='bold')
ax_suhu.set_xlabel('Sea Surface Temperature (SST) Classes', fontsize=9)
ax_suhu.set_ylabel('Predicted Fork Length (cm)', fontsize=9)
ax_suhu.grid(axis='y', linestyle='--', alpha=0.7)
    
plt.tight_layout()
    
# Menampilkan ke Streamlit
st.pyplot(fig_suhu)
plt.close(fig_suhu)


# ===========================
# Distribusi Prediksi Berdasarkan Kelompok Chlorophyll-a (Chl-a)

st.subheader("Pengaruh Konsentrasi Chlorophyll-a (Chl-a) terhadap Prediksi Panjang Ikan")

# Membuat DataFrame copy untuk analisis Chl-a
df_analisis_chl = df.copy()

# Membuat "Kelompok Klorofil-a" (Binning)
batas_chl = [0.0, 0.15, 0.30, 0.48] 
nama_kelompok_chl = ['Very Low (< 0.15)', 'Low (0.15 - 0.30)', 'Moderate (> 0.30)']

df_analisis_chl['Kelompok_Chl'] = pd.cut(df_analisis_chl['chl-a'], bins=batas_chl, labels=nama_kelompok_chl)

# Membuat Visualisasi Boxplot
fig_chl, ax_chl = plt.subplots(figsize=(8, 5))

# Mendefinisikan gradasi warna hijau (Muda ke Tua) untuk merepresentasikan Klorofil
gradasi_chl = ["#c7e9c0", "#74c476", "#238b45"] 

sns.boxplot(
    data=df_analisis_chl, 
    x='Kelompok_Chl', 
    y='Prediksi_panjang_yft', # Sesuai dengan kolom hasil prediksi
    hue='Kelompok_Chl', 
    palette=gradasi_chl, 
    legend=False,         
    dodge=False,
    ax=ax_chl
    )

# Merapikan tampilan grafik
ax_chl.set_title('Distribution of Predicted Tuna Length by Chlorophyll-a Concentration Groups', fontsize=10, pad=15, fontweight='bold')
ax_chl.set_xlabel('Chlorophyll-a Concentration Classes (mg/m³)', fontsize=9)
ax_chl.set_ylabel('Predicted Fork Length (cm)', fontsize=9) 
ax_chl.grid(axis='y', linestyle='--', alpha=0.7)
    
plt.tight_layout()
    
# Menampilkan ke Streamlit
st.pyplot(fig_chl)
plt.close(fig_chl)


# ===========================
# # Distribusi Prediksi Berdasarkan Bulan Penangkapan
st.subheader("Pengaruh Musiman (Bulan) terhadap Prediksi Panjang Ikan")

# Membuat DataFrame copy untuk analisis bulanan
df_analisis_bulan = df.copy()

# Mendefinisikan urutan bulan yang benar (sesuai kalender)
urutan_bulan = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']

# Mengubah kolom 'Bulan' menjadi tipe kategori yang berurutan agar grafik tidak acak-acakan
df_analisis_bulan['Bulan'] = pd.Categorical(df_analisis_bulan['Bulan'], categories=urutan_bulan, ordered=True)

# Membuat Visualisasi Boxplot
fig_bulan, ax_bulan = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=df_analisis_bulan, 
    x='Bulan', 
    y='Prediksi_panjang_yft', # Sesuai dengan nama kolom hasil prediksi 
    hue='Bulan',
    palette='viridis',        # Gradasi warna
    legend=False,
    dodge=False,
    ax=ax_bulan
    )

# Merapikan tampilan grafik
ax_bulan.set_title('Distribution of Fork Length by Catch Month', fontsize=14, pad=15, fontweight='bold')
ax_bulan.set_xlabel('Month of Observed', fontsize=12)
ax_bulan.set_ylabel('Predicted Fork Length (cm)', fontsize=12)
ax_bulan.grid(axis='y', linestyle='--', alpha=0.7)
    
plt.tight_layout()
    
# Menampilkan ke Streamlit
st.pyplot(fig_bulan)
plt.close(fig_bulan)


# ===========================
# Distribusi Prediksi Berdasarkan Teknik Penangkapan

st.subheader("Pengaruh Teknik Penangkapan terhadap Prediksi Panjang Ikan")

# Membuat DataFrame copy untuk analisis teknik penangkapan
df_analisis_teknik = df.copy()

# Mengubah nama kategori di dalam data untuk keperluan label sumbu X (Rumpon -> FADs, Non Rumpon -> Non-FADs)
kamus_nama_baru = {
    'Rumpon': 'FADs',
    'Non Rumpon': 'Non-FADs'
    }
df_analisis_teknik['Teknik_penangkapan'] = df_analisis_teknik['Teknik_penangkapan'].map(kamus_nama_baru)

# Membuat Visualisasi Boxplot untuk Streamlit
fig_teknik, ax_teknik = plt.subplots(figsize=(8, 5))

# Menyesuaikan kunci warna dengan nama kategori yang baru
warna_custom = {
    'FADs': '#e74c3c',      # Merah (soft crimson)
    'Non-FADs': '#2ecc71'   # Hijau (soft emerald)
    }

sns.boxplot(
    data=df_analisis_teknik, 
    x='Teknik_penangkapan', 
    y='Prediksi_panjang_yft', # Sesuai dengan nama kolom hasil prediksi di atas
    hue='Teknik_penangkapan', 
    palette=warna_custom,     # Menerapkan warna kustom spesifik
    legend=False,
    dodge=False,
    ax=ax_teknik
    )

# Merapikan tampilan grafik 
ax_teknik.set_title('Predicted Tuna Length by Fishing Technique Groups', fontsize=12, pad=15, fontweight='bold')
ax_teknik.set_xlabel('Fishing Technique', fontsize=10)
ax_teknik.set_ylabel('Predicted Fork Length (cm)', fontsize=10)
ax_teknik.grid(axis='y', linestyle='--', alpha=0.7)
    
plt.tight_layout()
    
# Menampilkan ke Streamlit
st.pyplot(fig_teknik)
plt.close(fig_teknik)

# ===========================
# Distribusi Prediksi Berdasarkan Kedalaman Penangkapan (Bersebelahan)

st.subheader("Pengaruh Kedalaman Operasional terhadap Prediksi Panjang Ikan")

# Membuat DataFrame copy untuk analisis kedalaman
df_analisis_kedalaman = df.copy()

# Membuat "Kelompok Kedalaman" (Binning) setiap 20 meter, maksimal 100
batas_kedalaman = [0, 20, 40, 60, 80, 100] 
label_kedalaman = ['0 - 20 m', '20 - 40 m', '40 - 60 m', '60 - 80 m', '80 - 100 m']

df_analisis_kedalaman['Kelompok_Kedalaman_Min'] = pd.cut(df_analisis_kedalaman['kedalaman_min'], bins=batas_kedalaman, labels=label_kedalaman)
df_analisis_kedalaman['Kelompok_Kedalaman_Max'] = pd.cut(df_analisis_kedalaman['kedalaman_max'], bins=batas_kedalaman, labels=label_kedalaman)

# Membuat Frame Gambar untuk 2 Grafik Berdampingan di Streamlit
fig_depth, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Grafik Kiri: Kedalaman Minimum
sns.boxplot(
    data=df_analisis_kedalaman, 
    x='Kelompok_Kedalaman_Min', 
    y='Prediksi_panjang_yft', # Sesuai dengan kolom hasil prediksi
    hue='Kelompok_Kedalaman_Min',
    palette='PuBu',           # Gradasi warna biru
    legend=False,
    dodge=False,
    ax=axes[0])
axes[0].set_title('A. Size Distribution vs. Minimum Depth', fontsize=14, pad=10, fontweight='bold')
axes[0].set_xlabel('Minimum Operational Depth (m)', fontsize=12)
axes[0].set_ylabel('Predicted Fork Length (cm)', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='x', rotation=15)

# Grafik Kanan: Kedalaman Maksimum
sns.boxplot(
    data=df_analisis_kedalaman, 
    x='Kelompok_Kedalaman_Max', 
    y='Prediksi_panjang_yft', 
    hue='Kelompok_Kedalaman_Max',
    palette='PuBu',
    legend=False,
    dodge=False,
     ax=axes[1])
axes[1].set_title('B. Size Distribution vs. Maximum Depth', fontsize=14, pad=10, fontweight='bold')
axes[1].set_xlabel('Maximum Operational Depth (m)', fontsize=12)
axes[1].set_ylabel('') # Dikosongkan karena sumbu Y sudah menyatu dengan grafik kiri (sharey=True)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].tick_params(axis='x', rotation=15)

# Merapikan tampilan keseluruhan
plt.tight_layout()
    
# Menampilkan ke Streamlit
st.pyplot(fig_depth)
plt.close(fig_depth)

# Download
# ===========================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
        "Download Hasil Prediksi",
        csv,
        "hasil_prediksi.csv",
        "text/csv"
    )


# =====================================================================

# Footer Akhir
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; margin-top: 30px; margin-bottom: 20px; color: #666666;'>
        <p style='font-size: 16px;'><strong>Dashboard Prediksi Ukuran Yellowfin Tuna (<i>Thunnus albacares</i>)</strong></p>
        <p style='font-size: 14px;'>Dikembangkan untuk mendukung analisis data dan pengelolaan perikanan tuna yang berkelanjutan</p>
        <p style='font-size: 12px; margin-top: 10px;'>&copy; 2026 Yayasan MDPI. Hak Cipta Dilindungi. <i>Happy People Many Fish</i>.</p>
    </div>
    """,
    unsafe_allow_html=True
)