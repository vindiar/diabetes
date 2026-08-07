"""
app_pages/page_docs.py
──────────────────────
Halaman Dokumentasi Lengkap
Mencakup: CRISP-DM, Preprocessing, Evaluasi, Cara Baca Visualisasi, Algoritma
"""

import streamlit as st

st.markdown("## 📖 Dokumentasi Aplikasi Klasifikasi Diabetes")
st.caption("Panduan lengkap & robust untuk memahami cara kerja, interpretasi, dan seluk-beluk aplikasi ini.")

# ─────────────────────────────────────────────────────────────────────────────
# DAFTAR ISI
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Daftar Isi — klik untuk expand", expanded=False):
    st.markdown("""
    1. [Overview Aplikasi](#1-overview-aplikasi)
    2. [Struktur Dataset](#2-struktur-dataset)
    3. [Metodologi CRISP-DM](#3-metodologi-crisp-dm)
       - Fase 1 — Business Understanding
       - Fase 2 — Data Understanding & EDA
       - Fase 3 — Data Preparation
       - Fase 4 — Modeling
       - Fase 5 — Evaluation
       - Fase 6 — Deployment / Serving
    4. [Algoritma: Logistic Regression](#4-algoritma-logistic-regression)
    5. [Cara Baca Visualisasi](#5-cara-baca-visualisasi)
    6. [Interpretasi Hasil Klasifikasi](#6-interpretasi-hasil-klasifikasi)
    7. [Arsitektur Kode](#7-arsitektur-kode)
    8. [Batasan & Disclaimer](#8-batasan--disclaimer)
    """)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 1. OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 1️⃣ Overview Aplikasi")

st.markdown("#### Mengenal Diabetes Melitus")
st.markdown("""
Diabetes melitus merupakan penyakit gangguan metabolik menahun yang ditandai dengan tingginya kadar
gula di dalam darah. Kondisi ini memerlukan penanganan dan deteksi dini untuk mencegah komplikasi yang
lebih serius. Secara klinis, seseorang umumnya didiagnosis memiliki indikasi diabetes apabila hasil
pemeriksaan kadar gula darah sewaktu mencapai **≥ 200 mg/dl** atau kadar gula darah puasa
**≥ 126 mg/dl**.
""")

st.markdown("#### Peran *Machine Learning*")
st.markdown("""
Machine Learning adalah cabang dari kecerdasan buatan (AI) yang memungkinkan sistem komputer untuk
belajar dari pola data historis tanpa perlu diprogram secara eksplisit. Sistem ini secara spesifik
menerapkan algoritma **Logistic Regression**, yaitu metode klasifikasi prediktif yang sangat andal
untuk mengklasifikasikan keluaran biner (seperti kelas *'Terindikasi Diabetes'* atau *'Normal'*)
berdasarkan evaluasi bobot dari berbagai fitur klinis pasien.
""")

st.markdown("#### Tentang Aplikasi Ini")
st.markdown("""
Aplikasi web ini dibangun menggunakan kerangka kerja Streamlit sebagai instrumen alat bantu skrining
kesehatan. Dengan memasukkan parameter medis seperti umur, indeks massa tubuh (BMI), kadar HbA1c,
hingga glukosa darah, sistem akan memproses data tersebut menggunakan model yang telah dievaluasi
untuk menampilkan probabilitas risiko diabetes secara seketika *(real-time)*. Aplikasi ini dirancang
untuk kemudahan akses masyarakat awam, namun **tidak ditujukan sebagai pengganti vonis diagnosis resmi
dari dokter profesional**.
""")

st.markdown("""
| Item | Detail |
|------|--------|
| 🤖 Algoritma | Logistic Regression (scikit-learn) |
| 📊 Dataset | `data/balanced_dataset.csv` (setelah outlier removal: 12.680 sampel) |
| 🎯 Task | Binary Classification (Diabetes vs Tidak Diabetes) |
| 🖥️ Framework | Streamlit (Python) |
| 💾 Persistence | `models/*.pkl` (Pre-trained model tersimpan) |
| 📐 Split Rasio | **80% Training · 20% Testing** |
| 🔀 Stratifikasi | Ya — distribusi kelas dijaga proporsional di kedua split |
| 🔬 Metodologi | **CRISP-DM** (Cross-Industry Standard Process for Data Mining) |

> **Tahukah Anda?**  
> Proses pelatihan *(training)* ML merupakan komputasi yang berat, sehingga aplikasi web terbaik langsung menyajikan model AI yang **sudah pre-trained** untuk memberikan inferensi dalam hitungan milidetik.
""")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
**Perbandingan Dataset Mentah vs Dibalancing:**
| Metrik / Indikator | Sebelum Balancing (`dataset_diabetes.csv`) | Setelah Balancing (`balanced_dataset.csv`) |
|--------------------|--------------------------------------------|-------------------------------------------|
| **Total Sampel** | **100.000** baris | **12.680** baris (setelah IQR Outlier Removal) |
| **Kasus Diabetes (+)** | 8.500 sampel (8.5%) | 6.340 sampel (50.0%) |
| **Kasus Non-Diabetes (-)** | 91.500 sampel (91.5%) | 6.340 sampel (50.0%) |
| **Status Distribusi** | ⚠️ Imbalanced (1 : 10,7) | ✅ Balanced (1 : 1) |
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Distribusi Setelah Encoding (12 Fitur):**
| Fitur Turunan | Keterangan |
|-------|-----------| 
| `gender_Male` | 1 = Berjenis kelamin laki-laki |
| `smoking_history_...` | Terdapat 5 cabang label merokok (current, former, ever, never, not current) |
| `hypertension` | 1 = Ada tekanan darah tinggi |
| `heart_disease` | 1 = Punya penyakit jantung |
| `diabetes` ⭐ | **Target** — 1 = Diabetes |
""")

with col2:
    st.markdown("""
**Kolom Numerik Kontinu:**
| Fitur Asli | Satuan | Analogi Klinis |
|-------|--------|--------|
| `age` | Tahun | 1–80 |
| `bmi` | kg/m² | Indeks Massa Tubuh |
| `HbA1c_level` | % | Gula darah rata-rata 3 bln |
| `blood_glucose_level` | mg/dL | Gula darah pemeriksaan |
""")


st.info("""
⚠️ **Preprocessing Khusus yang Dilakukan di Notebook:**
- `gender` & `smoking_history` menggunakan **One-Hot Encoding** untuk mencegah hirarki bias.
- 3 fitur numerik (BMI, HbA1c, Glukosa) melalui metode filtrasi IQR **Outlier Removal**.
- Keseluruhan dataset diskalakan menggunakan format `StandardScaler`.
""", icon="🔧")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 3. CRISP-DM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 3️⃣ Metodologi CRISP-DM")
st.markdown("""
**CRISP-DM** *(Cross-Industry Standard Process for Data Mining)* adalah metodologi standar industri
yang paling banyak digunakan dalam proyek data mining dan machine learning. Metodologi ini terdiri dari
**6 fase** yang saling berkaitan dengan alur iteratif:

- Setiap fase dapat kembali ke fase sebelumnya jika diperlukan
- **Evaluation** adalah titik keputusan kritis: jika model **belum memenuhi kriteria** → kembali ke *Business Understanding* untuk iterasi ulang. Jika **sudah sesuai** → lanjut ke *Deployment*
- **Deployment** adalah **fase akhir** — model dipublikasikan kepada pengguna
""")


# CRISP-DM diagram — gambar referensi
col_img_l, col_img_c, col_img_r = st.columns([1, 3, 1])
with col_img_c:
    import os
    _crisp_img = "results/crisp_dm_diagram.png"
    if os.path.exists(_crisp_img):
        st.image(_crisp_img, caption="Diagram Siklus CRISP-DM (Cross-Industry Standard Process for Data Mining)", use_container_width=True)
    else:
        st.info("Diagram CRISP-DM tidak ditemukan di `results/crisp_dm_diagram.png`", icon="🖼️")



tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Fase 1 — Business Understanding",
    "📊 Fase 2 — Data Understanding",
    "🔧 Fase 3 — Data Preparation",
    "🤖 Fase 4 — Modeling",
    "📈 Fase 5 — Evaluation",
    "🚀 Fase 6 — Deployment"
])

with tab1:
    st.markdown("""
#### Fase 1 — Business Understanding
**Tujuan:** Mendefinisikan tujuan bisnis dan kebutuhan proyek dari perspektif klinis.

**Pertanyaan Kunci yang Dijawab:**
- Bagaimana cara mendeteksi risiko diabetes pada pasien secara dini?
- Fitur klinis apa yang paling berpengaruh terhadap indikasi diabetes?
- Bagaimana menyajikan hasil klasifikasi agar mudah dipahami masyarakat awam?

**Output Fase Ini:**
| Item | Hasil |
|------|-------|
| 🎯 Tujuan Bisnis | Sistem skrining kesehatan diabetes berbasis ML |
| 📌 Tujuan Data Mining | Binary Classification: Diabetes vs Tidak Diabetes |
| ✅ Kriteria Keberhasilan | Akurasi ≥ 80%, AUC-ROC ≥ 0.85 |
| 📋 Rencana Proyek | Pipeline: EDA → Preprocessing → Logistic Regression → Evaluasi → Deployment |
""")

with tab2:
    st.markdown("""
#### Fase 2 — Data Understanding & EDA
**File:** `utils/data.py` | **Halaman:** Dashboard & EDA

**Tujuan:** Memahami struktur, kualitas, dan pola distribusi data sebelum pemodelan.

**Langkah:**
1. **Load Dataset** — Membaca `data/balanced_dataset.csv` dengan Pandas, di-cache via `@st.cache_data`
2. **Inspeksi Awal** — Analisis missing value, tipe data, dan statistik deskriptif
3. **Distribusi Variabel** — Visualisasi distribusi kategorikal & numerik
4. **Analisis Korelasi** — Heatmap korelasi Pearson antar fitur klinis

> Hal ini memastikan bahwa data berukuran 12.680 baris yang dikonstruksi secara programatik *sama persis kualitasnya* dengan data ketika dilatih di Jupyter Notebook peneliti.
""")

with tab3:
    st.markdown("""
#### Fase 3 — Data Preparation
**File:** Notebook Jupyter | **Output:** `models/*.pkl`

**Tujuan:** Mentransformasi data mentah menjadi format yang siap dimodelkan.

**Langkah Preprocessing:**
1. **Filter Data** — Fitur jenis kelamin dibersihkan sehingga murni mencakup 'Male' dan 'Female'
2. **Outlier Removal** — Menerapkan teknik rentang **IQR (Q1-Q3)** guna membuang pencilan numerik secara ketat pada fitur BMI, HbA1c, dan Glukosa
3. **One-Hot Encoding** — `gender` & `smoking_history` dikonversi menjadi variabel biner untuk mencegah hirarki bias pada model
4. **Standard Scaling** — Seluruh fitur numerik dinormalisasi menggunakan `StandardScaler` agar model tidak bias terhadap skala

| Artefak | Deskripsi |
|---------|-----------|
| `model_diabetes.pkl` | Bobot model Logistic Regression yang telah dilatih |
| `scaler_diabetes.pkl` | Referensi mean & variance untuk normalisasi |
| `features_diabetes.pkl` | Struktur ekspektasi One-Hot array (12 fitur) |
""")

with tab4:
    st.markdown("""
#### Fase 4 — Modeling
**Algoritma:** Logistic Regression (scikit-learn)

**Tujuan:** Melatih model klasifikasi biner untuk memprediksi risiko diabetes.

**Konfigurasi Model:**
| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| Algoritma | Logistic Regression | Interpretable, probabilistik, efisien untuk data tabular |
| Split Rasio | 80% Train / 20% Test | Standar industri untuk dataset medium |
| Stratifikasi | Ya | Menjaga proporsi kelas di setiap split |
| `random_state` | 42 | Reproducibility hasil |
| Threshold | 0.5 | Titik cut-off probabilitas untuk klasifikasi biner |

**Formula Logistic Regression:**
$$P(y=1|X) = \\frac{1}{1 + e^{-(w_0 + w_1 x_1 + ... + w_n x_n)}}$$

- $w_i$ = koefisien (bobot) setiap fitur klinis
- Jika $P > 0.5$ → Terindikasi Diabetes
- Jika $P \\leq 0.5$ → Tidak Terindikasi Diabetes
""")

with tab5:
    st.markdown("""
#### Fase 5 — Evaluation
**File:** `utils/model.py` (fungsi `evaluate`) | **Halaman:** Performa Model

**Tujuan:** Mengevaluasi apakah model memenuhi kriteria bisnis yang ditetapkan di Fase 1.

**Metrik Evaluasi yang Diaplikasikan:**

| Metrik (Test Set 20%) | Nilai | Keterangan |
|--------|-------|-----------|
| **AUC-ROC** | **0.9338** | Eksekusi Sangat Baik (Excellent) |
| **Akurasi** | **0.8458** | Klasifikasi Tepat Keseluruhan |
| 5-Fold Cross Validation | 0.8562 | Rata-rata akurasi CV — generalisasi baik |

> Di konteks medis, **Recall (Sensitivitas)** juga merupakan indikator krusial dalam mempertimbangkan risiko luput (*False Negative*).
""")

    # Decision gate
    col_ok, col_fail = st.columns(2, gap="small")
    with col_ok:
        st.success("""
**✅ Model Memenuhi Kriteria → Lanjut ke Deployment**
- AUC ≥ 0.85 ✓
- Akurasi ≥ 80% ✓
- Cross-Validation stabil ✓
- → **Fase 6: Deployment**
""", icon="🚀")
    with col_fail:
        st.error("""
**❌ Model Belum Memenuhi Kriteria → Kembali Iterasi**
- Performa di bawah threshold
- Fitur perlu ditinjau ulang
- Strategi preprocessing perlu diubah
- → **Kembali ke Fase 1: Business Understanding**
""", icon="🔄")

with tab6:
    st.markdown("""
#### Fase 6 — Deployment *(Fase Akhir)*
**File:** `utils/model.py` | **Halaman:** Klasifikasi

**Tujuan:** Menyajikan model yang telah lulus evaluasi kepada pengguna akhir. Ini adalah **titik akhir** dari siklus CRISP-DM untuk proyek ini.

**Alur Inferensi (untuk 1 pasien baru):**
1. **Input Mapping** — Pengguna mengisikan Berat serta Tinggi. Sistem otomasi mengekstraksi BMI.
2. **One-Hot Arraying** — Fitur *Gender* dan *Histori Merokok* diadaptasikan ke bentuk One-Hot Array yang tepat menggunakan formasi statik `features_diabetes.pkl`
3. **Standard Scaler** — Vektor input dipusatkan terhadap normal distribusi `scaler`
4. **Scoring Model** — Vektor termutasi masuk ke gerbang `.predict_proba()` Logistic Regression
5. **Output** — Probabilitas dikonversi menjadi persentase medis yang mudah dipahami
6. **Ekspor PDF** — Hasil klasifikasi dapat diunduh sebagai laporan PDF untuk arsip pasien
""")

    st.info("""
**📌 Catatan Penting Alur CRISP-DM pada Proyek Ini:**

Alur proyek ini berhenti di **Fase 6 — Deployment** karena model telah memenuhi semua kriteria evaluasi yang ditetapkan sejak Fase 1 (AUC 0.93, Akurasi 84.58%).

Jika di masa mendatang performa model menurun atau ada data baru yang signifikan, proses akan **kembali ke Fase 1 — Business Understanding** untuk memulai siklus iterasi berikutnya.
""", icon="🔁")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 4. ALGORITMA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 4️⃣ Algoritma: Logistic Regression")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
**Logistic Regression** adalah algoritma klasifikasi yang menghitung probabilitas suatu data masuk ke dalam satu kelas menggunakan **fungsi Sigmoid**:

$$P(y=1|X) = \\frac{1}{1 + e^{-(w_0 + w_1 x_1 + ... + w_n x_n)}}$$

- $w_i$ adalah koefisien (bobot) setiap fitur
- Jika probabilitas > 0.5 → diklasifikasi **Diabetes (1)**
- Jika probabilitas ≤ 0.5 → diklasifikasi **Tidak Diabetes (0)**

**Kenapa Logistic Regression sangat memadai?**
- ✅ **Interpretable** — Evaluasi bobot klinis parameter medis terlihat jelas (+/-)
- ✅ **Tidak memakan sumber daya** — Model Pickle cukup ringan memori.
- ✅ **Probabilistik** — Murni merilis skala 0 sampai 100%, sangat relevan untuk asesmen risiko penanganan pasien klinis di dunia modern.
""")

with col2:
    st.markdown("""
**Rentang Nilai Koefisien:**

| Koefisien | Interpretasi |
|-----------|-------------|
| `+` (positif) | Nilai fitur naik → Risiko diabetes ikut naik |
| `−` (negatif) | Nilai fitur naik → Risiko diabetes turun |
| Besar absolutnya | Semakin kuat pengaruhnya |
""")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 5. CARA BACA VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 5️⃣ Cara Baca Visualisasi")

viz_tabs = st.tabs([
    "📸 Statik Plotting", "📦 Boxplot Outlier",
    "🌡️ Heatmap", "🔲 Confusion Matrix", "📈 ROC Curve", "🏅 Cross Eval"
])

with viz_tabs[0]:
    st.markdown("""
**Arsitektur Penyajian Statik (Optimasi Render)**
- **Letak:** Dashboard & EDA
- Seluruh visualisasi Dashboard **tidak dirender secara langsung menggunakan Matplotlib** oleh Web-Engine (Streamlit). Seluruh gambar di-regenerate oleh *Notebook Script* (`generate_images.py`) dan disimpan stabil dalam `results/`.
- Ini menjamin pengguna dengan gawai ringan dapat mengakses halaman eksplorasi interaktif tanpa macet (*delay processing*).
""")

with viz_tabs[1]:
    st.markdown("""
**Boxplot IQR (*Interquartile Range*)**
- Terdapat visual perbandingan **Sebelum** dan **Sesudah** `Outlier Removal`. Sistem pembersihan ini memotong nilai ekstrim untuk menyajikan struktur data *Gaussian* dengan cara batas ekor pencilan statistik (1.5 × Batas IQR).
""")

with viz_tabs[2]:
    st.markdown("""
**Matriks Korelasi (Heatmap)**
- Menilai kecenderungan hubungan linear parameter.
- Diagonal selalu bernilai **1.00**
- *HbA1c* (Gula darah 3 bln) dan *Blood Glucose* tampil sebagai salah satu prediktor berkorelasi langsung linear tertinggi dengan variabel terikat (target) di struktur kami.
""")

with viz_tabs[3]:
    st.markdown("""
**Confusion Matrix**
- Metrik perbandingan yang memberikan gambaran:
  - **TP**: Pasien sakit. Model bilang Sakit. (Baik)
  - **TN**: Pasien sehat. Model bilang Sehat. (Aman)
  - **FP**: Pasien tidak sakit, dibilang Sakit. (Cemas salah diagnosis).
  - **FN**: Pasien benar-benar sakit, Model meloloskannya (Fatal!).
""")

with viz_tabs[4]:
    st.markdown("""
**ROC Curve**
Memvisualisasi kompromi antara tingkat kepalsuan peringatan dengan tangkapan peringatan sejati. Area AUC menembus 0.93 (*Excellent*) memberikan kapabilitas tangkas terhadap permodelan.
""")

with viz_tabs[5]:
    st.markdown("""
**5-Fold Cross Validation**
Metode pembagian dan perputaran proporsi tes/pelatihan secara serentak dalam 5 kelompok (lipatan). Memberikan estimasi yang minim bias dengan rata-rata **F1** dan **Akurasi** untuk membuktikan bahwa performa yang ditunjukkan tidak kebetulan semata karena set acak yang bernasib baik.
""")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 6. INTERPRETASI KLASIFIKASI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 6️⃣ Interpretasi Hasil Klasifikasi")
col1, col2 = st.columns(2)
with col1:
    st.success("""
**TIDAK TERINDIKASI DIABETES**
- Probabilitas Non-Diabetes > 50%
- Bukan berarti pasien 100% sehat
- Tetap jaga pola makan & olahraga rutin
- Faktor risiko individu tetap harus dipantau
""", icon="✅")

with col2:
    st.error("""
**TERINDIKASI DIABETES**
- Probabilitas Diabetes > 50%
- Ini bukan diagnosis medis resmi
- Segera konsultasikan dengan dokter dan konfirmasi lab.
""", icon="⚠️")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 7. ARSITEKTUR KODE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 7️⃣ Arsitektur Kode")
st.code("""
klasifikasi-diabetes/
├── streamlit_app.py          # Entry point — navigasi & inisialisasi session
├── data/
│   ├── balanced_dataset.csv      # Dataset utama
│   └── dataset_diabetes.csv      # Versi dataset sebelum balancing
├── requirements.txt          # Dependensi Python
├── Makefile                  # Shortcut perintah umum
│
├── .streamlit/
│   └── config.toml           # Konfigurasi tema (dark mode, warna brand)
│
├── models/
│   ├── model_diabetes.pkl    # Pre-trained Logistic Regression
│   ├── scaler_diabetes.pkl   # Pre-trained StandardScaler
│   └── features_diabetes.pkl # Daftar fitur hasil one-hot encoding
│
├── results/                  # Gambar statis Evaluasi Model & EDA dari notebook
│
├── utils/                    # Logika bisnis ML (backend)
│   ├── data.py               # Fase 3 CRISP-DM — Load & Preprocessing (Outlier Removal)
│   ├── model.py              # Fase 4 & 5 CRISP-DM — Load pkl, Evaluasi, Klasifikasi
│   └── plots.py              # Fungsi-fungsi visualisasi UI (sebagian dinamis)
│
└── app_pages/                # Halaman UI Streamlit (frontend)
    ├── page_eda.py           # Dashboard & EDA (akses results/)
    ├── page_model.py         # Performa Model (akses results/)
    ├── page_klasifikasi.py   # Form Klasifikasi
    └── page_docs.py          # Dokumentasi Ini
""", language="text")

st.info("""
**Pemisahan Infrastruktur Statik & Dinamis**
Aplikasi menggunakan metode cerdas di mana bagian kalkulasi evaluasi grafikal tidak dilakukan *On the Fly*. Hanya fungsi klasifikasi risiko individu (*page_klasifikasi*) yang meregulasi probabilitas instan.
""", icon="🏗️")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 8. BATASAN & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 8️⃣ Batasan & Disclaimer")

st.warning("""
**⚠️ Disclaimer Medis**

Aplikasi ini dibuat untuk **tujuan edukasi dan demonstrasi ML Engineering** semata. Hasil probabilitas:
- **BUKAN** diagnosis medis absolut dan mengikat.
- **TIDAK MENGGANTIKAN** verifikasi uji rekam medis dokter dan ahli.

Untuk diagnosa penyakit kronis yang nyata, selalu tanyakan fasilitas tenaga medis terdekat.
""")

st.markdown("""
**Keterbatasan Model:**
| Limitasi | Penjelasan |
|----------|-----------| 
| Fitur Selektif | Model ini hanya menggunakan 12 fitur *One-hot encoded*. Parameter keturunan genetis dan gen komorbiditas tidak diukur. |
| Cut-off 0.5 | Ambang batas positif 0.5 mengedepankan f1-score. Dokter mungkin membutuhkan standar sensitivitas (Recall) yang lebih bias untuk meminimalkan kecolongan deteksi. |
""")
