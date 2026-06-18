# Proyek-Integrasi-Data-Science-Elsa-Mayori-Malik-IF403
Ujian Tengah Semester / Portofolio Integrasi [Elsa Mayori Malik] | 240401020098 | IF403

# Proyek Portofolio Akhir: Pipa Pengolahan Data Science End-to-End
### Implementasi Metodologi CRISP-DM Terintegrasi

* **Nama Lengkap:** Elsa Mayori Malik
* **NIM:** 240401020098
* **Kelas:** IF403
* **Mata Kuliah:** Pengantar Data Science
* **Dosen Koordinator:** Syahid Abdullah, S.Si, M.Kom

---

## 📑 Rangkuman Arsitektur Sistem & Metodologi

Proyek ini menyatukan seluruh kompetensi dasar *Data Science* yang dipelajari dari Pertemuan 1 hingga Pertemuan 7 ke dalam satu pipa eksekusi otomatis (*Automated Data Pipeline*) menggunakan kerangka kerja standard industri **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*).

### 🛠️ Rekapitulasi Pembelajaran Fase per Sesi

```text
┌────────────────────────────────────────────────────────────────────────┐
│               SIKLUS HIDUP PROYEK DATA SCIENCE (CRISP-DM)              │
├──────────────┬─────────────────────────────────────────────────────────┤
│ Sesi 1       │ Pengenalan Lingkungan Kerja (Python, Git, & GitHub)     │
│ Sesi 2 & 3   │ Pembersihan Data (Handling Missing Values & IQR Method) │
│ Sesi 4 & 5   │ Analisis Eksploratif & Visualisasi (EDA, Seaborn, Story)│
│ Sesi 6       │ Transformasi & Persiapan Fitur (Encoding & Split Data)  │
│ Sesi 7       │ Pemodelan & Evaluasi Prediktif (Regresi Linear & RMSE)  │
└──────────────┴─────────────────────────────────────────────────────────┘


---

### 2. Untuk `main.py` (Kode Program Utama Gabungan Komplit)
Salin kode Python di bawah ini ke file bernama `main.py`. Kode ini menggabungkan seluruh logika pemrosesan data, pembersihan, rekayasa fitur, pembuatan grafik otomatis, hingga pemodelan regresi dalam satu kali jalan:

```python
# ==============================================================================
# TUGAS AKHIR / PORTFOLIO INTEGRASI DATA SCIENCE (PERTEMUAN 1 - 7)
# Nama    : Elsa Mayori Malik
# NIM     : 240401020098
# Kelas   : IF403
# ==============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def cetak_tahapan(no_sesi, nama_fase):
    print("\n" + "="*70)
    print(f" [ INTEGRASI SESI {no_sesi} ] : {nama_fase.upper()} ")
    print("="*70)

# ==============================================================================
# SCREEN HEADER IDENTITAS MAHASISWA (PERTEMUAN 1)
# ==============================================================================
print("="*70)
# Pemenuhan standardisasi identitas visual profesional portofolio Elsa
print("         PROYEK INTEGRASI AKHIR PENGANTAR DATA SCIENCE (UTS)       ")
print("="*70)
print(" Nama Lengkap : Elsa Mayori Malik")
print(" NIM          : 240401020098")
print(" Kelas        : IF403")
print(" Status Kode  : Kompilasi Pipeline Sesi 1 s/d Sesi 7 - Sukses Terbaca")
print("="*70)


# ==============================================================================
# PERTEMUAN 2 & 3: DATA INGESTION & DATA CLEANING
# ==============================================================================
cetak_tahapan("2 & 3", "Data Ingestion & Pembersihan Data Mentah (Missing Values & IQR)")

# Simulasi database kotor restoran untuk memprediksi besaran 'Tip' konsumen
raw_restaurant_data = {
    'Total_Tagihan': [15.0, 22.4, np.nan, 45.8, 18.2, 32.1, 120.0, 25.5, np.nan, 28.0], # 120.0 adl Outlier masif
    'Kategori_Hari': ['Weekday', 'Weekend', 'Weekend', 'Weekday', 'Weekend', 'Weekday', 'Weekend', 'Weekday', 'Weekend', 'Weekday'],
    'Nominal_Tip': [2.0, 3.5, 2.5, 6.0, 3.0, 4.5, 5.0, 3.2, 2.8, 3.6]
}

df_raw = pd.DataFrame(raw_restaurant_data)
print("1. Dataset Mentah Berhasil Dimuat:")
print(df_raw)

# A. Penanganan Missing Values pada Fitur Prediktor
print("\n-> Cek Data Hilang:")
print(df_raw.isnull().sum())

median_tagihan = df_raw['Total_Tagihan'].median()
df_raw['Total_Tagihan'] = df_raw['Total_Tagihan'].fillna(median_tagihan)
print(f"\n[Sistem] Imputasi selesai menggunakan nilai Median Tagihan ({median_tagihan} USD).")

# B. Deteksi & Eliminasi Outliers Menggunakan Jarak Pagar IQR
Q1 = df_raw['Total_Tagihan'].quantile(0.25)
Q3 = df_raw['Total_Tagihan'].quantile(0.75)
IQR = Q3 - Q1
df_fences_atas = Q3 + (1.5 * IQR)

print(f"-> Parameter Statistika IQR: Q1={Q1}, Q3={Q3}, Batas Atas Maksimal Valid={df_fences_atas}")

# Filtrasi membuang baris anomali (data baris tagihan 120.0 dibuang demi performa linearitas)
df_clean = df_raw[df_raw['Total_Tagihan'] <= df_fences_atas].reset_index(drop=True)
print("\n2. Dataset Setelah Pembersihan Selesai (Bebas Missing Values & Outliers):")
print(df_clean)


# ==============================================================================
# PERTEMUAN 4 & 5: STATISTIKA DESKRIPTIF & DATA STORYTELLING VISUALIZATION
# ==============================================================================
cetak_tahapan("4 & 5", "Statistika Deskriptif & Visualisasi Data Storytelling")

print("Analisis Deskriptif Fitur Utama:")
print(df_clean.describe().round(3))

print("\n[Sistem] Membuka jendela grafik visualisasi data...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram Sebaran Data Fitur setelah dibersihkan
sns.histplot(df_clean['Total_Tagihan'], kde=True, color='teal', ax=axes[0])
axes[0].set_title('Distribusi Total Tagihan Bersih')
axes[0].set_xlabel('Nilai Tagihan')

# Plot 2: Scatter Plot untuk mendeteksi korelasi awal antara Tagihan dengan Tip
sns.scatterplot(data=df_clean, x='Total_Tagihan', y='Nominal_Tip', hue='Kategori_Hari', s=100, ax=axes[1])
axes[1].set_title('Scatter Korelasi: Total Tagihan vs Nominal Tip')
axes[1].set_xlabel('Total Tagihan ($)')
axes[1].set_ylabel('Tip ($)')

plt.suptitle('Analisis Data Eksploratif (EDA) - Portofolio Terintegrasi Elsa Mayori Malik', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ==============================================================================
# PERTEMUAN 6: PREPROCESSING PIPELINE (ENCODING, SPLITTING, & SCALING)
# ==============================================================================
cetak_tahapan("6", "Persiapan Data (One-Hot Encoding, Train-Test Split, & Scaling)")

# A. One-Hot Encoding pada fitur Kategorikal 'Kategori_Hari'
df_encoded = pd.get_dummies(df_clean, columns=['Kategori_Hari'], drop_first=True)
print("Hasil Transformasi One-Hot Encoding (Variabel Teks -> Angka Biner):")
print(df_encoded.head(3))

# B. Pemisahan Fitur Kontrol (X) dan Target Label (y)
X = df_encoded.drop('Nominal_Tip', axis=1)
y = df_encoded['Nominal_Tip']

# C. Train-Test Split (80% Latih, 20% Uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n-> Matriks X Latih: {X_train.shape} | Matriks X Uji: {X_test.shape}")

# D. Standardisasi Fitur Numerik Kontinu (StandardScaler)
scaler = StandardScaler()
fitur_numerik = ['Total_Tagihan']

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[fitur_numerik] = scaler.fit_transform(X_train[fitur_numerik])
X_test_scaled[fitur_numerik] = scaler.transform(X_test[fitur_numerik])

print("\n3 Baris Pertama X_train Setelah Standardisasi:")
print(X_train_scaled.head(3).round(3))


# ==============================================================================
# PERTEMUAN 7: MACHINE LEARNING MODELING & EVALUATION (REGRESI LINEAR)
# ==============================================================================
cetak_tahapan("7", "Pemodelan Machine Learning Supervised & Evaluasi")

# A. Pelatihan Model Regresi Linear OLS
model_regresi = LinearRegression()
model_regresi.fit(X_train_scaled, y_train)
print("[Sukses] Algoritma Regresi Linear Berhasil Mempelajari Pola Data Latih.")

# B. Pengujian Model pada Data Asing (X_test_scaled)
y_prediksi = model_regresi.predict(X_test_scaled)

# C. Kalkulasi Metrik Evaluasi Kinerja Kuantitatif
nilai_mae = mean_absolute_error(y_test, y_prediksi)
nilai_rmse = np.sqrt(mean_squared_error(y_test, y_prediksi))
nilai_r2 = r2_score(y_test, y_prediksi)

print(f"\n=== Hasil Evaluasi Performa Model ===")
print(f" MAE  (Mean Absolute Error)     : {nilai_mae:.4f}")
print(f" RMSE (Root Mean Squared Error)  : {nilai_rmse:.4f}")
print(f" R-squared Score (Koefisien R2)  : {nilai_r2:.4f}")

# D. Diagnostik Akhir: Plot Residuals Uji Asumsi
nilai_residu = y_test - y_prediksi
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_prediksi, y=nilai_residu, color='crimson', s=80)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot Akhir (Validasi Asumsi Linieritas)\nIdentitas: Elsa Mayori Malik')
plt.xlabel('Nilai Prediksi Model')
plt.ylabel('Sisaan Kesalahan (Residual)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print(" PIPELINE INTEGRASI SELESAI - KODE PROGRAM BERJALAN SEMPURNA 100% ")
print("="*70)
