# 🩺 klasifikasi Risiko Diabetes

Aplikasi web interaktif berbasis **Machine Learning** untuk memklasifikasi risiko penyakit diabetes menggunakan algoritma **Logistic Regression**, dibangun dengan **Streamlit** dan mengikuti alur standar **ML Development Lifecycle**.

> **⚠️ Disclaimer:** Aplikasi ini dibuat untuk **tujuan edukasi dan demonstrasi Machine Learning**. Hasil klasifikasi **bukan** diagnosis medis resmi dan **tidak** menggantikan pemeriksaan dokter atau laboratorium klinis.

---

## 📋 Fitur Utama

| Halaman                | Deskripsi                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| 🏠 **Dashboard & EDA** | Eksplorasi dataset: distribusi kelas, boxplot fitur klinis, heatmap korelasi                  |
| 📊 **Performa Model**  | Metrik evaluasi (Akurasi, Presisi, Recall, F1, AUC-ROC), Confusion Matrix, Feature Importance |
| 🔍 **klasifikasi**     | Form input data pasien → hasil klasifikasi + probabilitas + ringkasan faktor risiko           |
| 📖 **Dokumentasi**     | Penjelasan lengkap ML Lifecycle, cara baca visualisasi, interpretasi hasil                    |

---

## 🚀 Cara Menjalankan (Lokal)

### Prasyarat

- Python **3.9+**
- `git` terinstall
- Koneksi internet (untuk install dependensi pertama kali)

### 1. Clone repository

```bash
git clone https://github.com/<username>/predict-diabetes.git
cd predict-diabetes
```

### 2. Buat virtual environment

```bash
# Buat venv
python -m venv venv

# Aktifkan — Windows (PowerShell / CMD)
venv\Scripts\activate

# Aktifkan — macOS / Linux
source venv/bin/activate
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi tersedia di **`http://localhost:8501`**.

> **🔔 Catatan pertama kali dijalankan:** Aplikasi ini menggunakan **Pre-trained Model** (`model_diabetes.pkl`), **Scaler** (`scaler_diabetes.pkl`), dan daftar fitur turunan (`features_diabetes.pkl`) yang berada di folder `models/` hasil ekspor dari Jupyter Notebook. Tidak memerlukan proses training ulang.

### 5. Menggunakan Makefile (Opsional)

Proyek ini menyertakan `Makefile` untuk menyederhanakan perintah umum:

```bash
make help      # Tampilkan semua perintah yang tersedia
make venv      # Buat virtual environment baru
make install   # Install semua dependensi dari requirements.txt
make run       # Jalankan aplikasi Streamlit
make freeze    # Update requirements.txt dari package yang terinstall
make clean     # Hapus virtual environment
```

---

## ☁️ Deployment ke Streamlit Community Cloud

[Streamlit Community Cloud](https://streamlit.io/cloud) adalah platform hosting gratis untuk aplikasi Streamlit yang terhubung langsung ke GitHub.

### Prasyarat Deployment

1. Akun **GitHub** dengan repository publik yang berisi kode ini
2. Akun **Streamlit Community Cloud** (daftar gratis di [share.streamlit.io](https://share.streamlit.io))
3. File `requirements.txt` yang sudah lengkap ✅

### Langkah-langkah

#### 1. Push kode ke GitHub

Pastikan semua file sudah ter-commit dan ter-push ke repository GitHub Anda:

```bash
git add .
git commit -m "feat: initial commit"
git push origin main
```

> **💡 Tips:** Pastikan folder `models/` beserta file `.pkl` di dalamnya dan fitur `results/` yang menyimpan gambaran statis sudah di-push agar aplikasi berjalan langsung.

Tambahkan baris berikut ke `.gitignore` jika belum ada:

```
# Anda harus me-remove comment ini jika modelnya berubah dan ingin update versi .pkl
# models/*.pkl
```

#### 2. Login ke Streamlit Community Cloud

Buka [share.streamlit.io](https://share.streamlit.io) → klik **"Sign in with GitHub"** → Otorisasi akses ke akun GitHub Anda.

#### 3. Deploy aplikasi baru

1. Klik tombol **"New app"** di dashboard
2. Pilih **"From existing repo"**
3. Isi form deploy:

   | Field                  | Nilai                         |
   | ---------------------- | ----------------------------- |
   | **Repository**         | `<username>/predict-diabetes` |
   | **Branch**             | `main`                        |
   | **Main file path**     | `streamlit_app.py`            |
   | **App URL** (opsional) | `predict-diabetes`            |

4. Klik **"Deploy!"**

Streamlit Cloud akan otomatis:

- Mengkloning repository Anda
- Menginstall semua paket dari `requirements.txt`
- Menjalankan `streamlit_app.py`

#### 4. Pantau proses build

Setelah klik Deploy, Anda akan diarahkan ke halaman log deployment. Proses ini biasanya memakan waktu **2–5 menit** untuk pertama kali (termasuk training model).

Jika berhasil, aplikasi akan dapat diakses di:

```
https://<app-url>.streamlit.app
```

#### 5. Update aplikasi

Setiap kali Anda push commit baru ke branch `main`, Streamlit Cloud akan otomatis **redeploy** aplikasi. Tidak perlu tindakan manual.

### Troubleshooting Deployment

| Masalah                       | Solusi                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError`         | Pastikan semua paket ada di `requirements.txt` dengan versi yang benar                      |
| `MemoryError` saat training   | Free tier memiliki limit RAM ~1 GB; dataset besar mungkin perlu optimasi                    |
| Model tidak tersimpan         | Pastikan folder `models/` ada di repo, atau buat di `streamlit_app.py` dengan `os.makedirs` |
| App tidak update setelah push | Cek branch yang dideploy sudah sesuai; klik "Reboot app" di menu ⋮                          |

---

## 🗂️ Struktur Proyek

```
predict-diabetes/
├── streamlit_app.py          # Entry point — navigasi & inisialisasi session
├── data/
│   ├── balanced_dataset.csv      # Dataset utama
│   └── dataset_diabetes.csv      # Versi dataset sebelum balancing
├── requirements.txt          # Dependensi Python (pinned versions)
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
│   ├── data.py               # Fase 1 — Load & Preprocessing (Outlier Removal)
│   ├── model.py              # Fase 2 & 3 — Load pkl, Evaluasi, klasifikasi
│   └── plots.py              # Fungsi-fungsi visualisasi UI (jika ada yg dinamis)
│
└── app_pages/                # Halaman UI Streamlit (frontend)
    ├── page_eda.py           # Dashboard & EDA (menampilkan gambar dari results/)
    ├── page_model.py         # Performa Model (menampilkan gambar dari results/)
    ├── page_predict.py       # Form klasifikasi
    └── page_docs.py          # Dokumentasi
```

---

## ⚙️ ML Development Lifecycle

Aplikasi ini mengimplementasikan 4 fase standar ML lifecycle:

```
┌─────────────────────────────────────────────────────────┐
│  Fase 1 — DATA           │  utils/data.py               │
│  Load & Cleaning         │  Outlier Removal             │
│  Filtering (Male/Female) │  Penyesuaian Quartile        │
├─────────────────────────────────────────────────────────┤
│  Fase 2 — LOAD MODEL     │  utils/model.py              │
│  Logistic Regression     │  model_diabetes.pkl          │
│  Standard Scaler         │  scaler_diabetes.pkl         │
├─────────────────────────────────────────────────────────┤
│  Fase 3 — EVALUASI       │  utils/model.py              │
│  Akurasi, Presisi, Recall│  app_pages/page_model.py     │
│  F1, AUC-ROC (Static EDA)│  Menggunakan Stratified KFold│
├─────────────────────────────────────────────────────────┤
│  Fase 4 — klasifikasi       │  utils/model.py              │
│  Single-patient inference│  app_pages/page_predict.py   │
│  (One-Hot map format)    │                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Atribut         | Detail                                    |
| --------------- | ----------------------------------------- |
| **Sumber**      | `data/balanced_dataset.csv` (lokal)       |
| **Jumlah data** | `12680` baris (setelah _outlier removal_) |
| **Split Model** | Train 80% / Test 20%                      |
| **Fitur input** | 12 fitur (setelah _One-hot Encoding_)     |
| **ROC AUC**     | **0.9338**                                |
| **Accuracy**    | **0.8458**                                |
| **Target**      | `diabetes` (0 = Tidak, 1 = Diabetes)      |

### Deskripsi Fitur

| Fitur                 | Tipe      | Keterangan                                                     |
| --------------------- | --------- | -------------------------------------------------------------- |
| `gender`              | Kategorik | Jenis kelamin (Male / Female / Other)                          |
| `age`                 | Numerik   | Usia pasien (tahun)                                            |
| `hypertension`        | Biner     | 1 = memiliki tekanan darah tinggi                              |
| `heart_disease`       | Biner     | 1 = memiliki riwayat penyakit jantung                          |
| `smoking_history`     | Kategorik | Riwayat merokok (never / former / current / etc.)              |
| `bmi`                 | Numerik   | Body Mass Index (kg/m²)                                        |
| `HbA1c_level`         | Numerik   | Kadar hemoglobin terglikasi (%) — indikator gula darah 3 bulan |
| `blood_glucose_level` | Numerik   | Kadar gula darah saat pemeriksaan (mg/dL)                      |
| `diabetes` ⭐         | Biner     | **Target** — 1 = Terdiagnosis Diabetes                         |

---

## 📦 Dependensi

| Paket          | Versi Minimum | Fungsi                                    |
| -------------- | ------------- | ----------------------------------------- |
| `streamlit`    | ≥ 1.32.0      | Framework UI web interaktif               |
| `pandas`       | ≥ 2.0.0       | Manipulasi dan analisis data              |
| `numpy`        | ≥ 1.24.0      | Komputasi numerik array                   |
| `scikit-learn` | ≥ 1.3.0       | Model ML & preprocessing pipeline         |
| `matplotlib`   | ≥ 3.7.0       | Visualisasi plot dasar                    |
| `seaborn`      | ≥ 0.12.0      | Visualisasi statistik berbasis matplotlib |
| `joblib`       | ≥ 1.3.0       | Serialisasi dan persistence model         |

---

## 🤝 Kontribusi

Pull request sangat disambut! Untuk perubahan besar, harap buka _issue_ terlebih dahulu untuk mendiskusikan apa yang ingin Anda ubah.

1. Fork repository ini
2. Buat branch fitur: `git checkout -b feat/nama-fitur`
3. Commit perubahan: `git commit -m 'feat: tambah fitur X'`
4. Push ke branch: `git push origin feat/nama-fitur`
5. Buka **Pull Request**

---

## 📄 Lisensi

Proyek ini menggunakan lisensi **MIT**. Lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.
