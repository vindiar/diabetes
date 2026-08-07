"""
app_pages/page_eda.py
─────────────────────
CRISP-DM Fase 2 — DATA UNDERSTANDING
Dashboard & EDA page. Displays static images generated from the notebook.
"""

import streamlit as st

CSS_SECTION = """
<div style="font-size:1.4rem;font-weight:700;color:#a78bfa;
            border-left:4px solid #a78bfa;padding-left:12px;
            margin-bottom:16px;">{}</div>
"""

def _card(value, label, sublabel=""):
    sub_html = f'<div style="font-size:0.75rem;color:#a78bfa;margin-top:2px;">{sublabel}</div>' if sublabel else ''
    return f"""
    <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(139,92,246,0.3);
                border-radius:16px;padding:20px 16px;text-align:center;backdrop-filter:blur(10px);">
      <div style="font-size:2rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#60a5fa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value}</div>
      <div style="font-size:0.78rem;color:#94a3b8;margin-top:4px;
                  text-transform:uppercase;letter-spacing:1px;">{label}</div>
      {sub_html}
    </div>"""


def _overview_card(icon, title, content, badge_color="#8b5cf6"):
    """Render sebuah kartu informasi overview di dashboard."""
    return f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(139,92,246,0.22);
        border-top: 3px solid {badge_color};
        border-radius: 16px;
        padding: 20px 18px;
        height: 100%;
        box-sizing: border-box;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    ">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <span style="font-size:1.6rem;">{icon}</span>
            <div style="font-weight:700;color:#e2e8f0;font-size:1.02rem;line-height:1.3;">{title}</div>
        </div>
        <div style="font-size:0.86rem;color:#cbd5e1;line-height:1.65;text-align:justify;">
            {content}
        </div>
    </div>"""


def _var_card(icon, title, tipe, deskripsi, badge_color="#3b82f6"):
    """Render sebuah kartu penjelasan variabel."""
    return f"""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(139,92,246,0.22);
        border-top: 3px solid {badge_color};
        border-radius: 14px;
        padding: 18px 16px 16px;
        height: 100%;
        box-sizing: border-box;
    ">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="font-size:1.4rem;">{icon}</span>
            <div>
                <div style="font-weight:700;color:#e2e8f0;font-size:0.88rem;line-height:1.2;">{title}</div>
                <span style="font-size:0.68rem;color:#94a3b8;background:rgba(148,163,184,0.12);
                             padding:1px 7px;border-radius:20px;">{tipe}</span>
            </div>
        </div>
        <div style="font-size:0.82rem;color:#cbd5e1;line-height:1.55;">{deskripsi}</div>
    </div>"""


df = st.session_state.df
df_raw = st.session_state.get("df_raw", None)

st.markdown("## 🏠 Dashboard & Eksplorasi Data")
st.caption("Eksplorasi data klinis pasien diabetes & perbandingan dataset sebelum vs sesudah balancing.")
st.divider()

# ── 1. Overview Dashboard (3 Kolom Proporsional) ────────────────────────
st.markdown(CSS_SECTION.format("📌 Informasi & Konteks Aplikasi"), unsafe_allow_html=True)

col_ov1, col_ov2, col_ov3 = st.columns(3, gap="medium")

with col_ov1:
    st.markdown(_overview_card(
        icon="🩺",
        title="Mengenal Diabetes Melitus",
        content="""Diabetes melitus merupakan penyakit gangguan metabolik menahun yang ditandai dengan tingginya kadar gula di dalam darah. Kondisi ini memerlukan penanganan dan deteksi dini untuk mencegah komplikasi yang lebih serius. Secara klinis, seseorang umumnya didiagnosis memiliki indikasi diabetes apabila hasil pemeriksaan kadar gula darah sewaktu mencapai <b>≥ 200 mg/dL</b> atau kadar gula darah puasa <b>≥ 126 mg/dL</b>.""",
        badge_color="#ef4444"
    ), unsafe_allow_html=True)

with col_ov2:
    st.markdown(_overview_card(
        icon="🤖",
        title="Peran Machine Learning",
        content="""Machine Learning adalah cabang dari kecerdasan buatan (AI) yang memungkinkan sistem komputer untuk belajar dari pola data historis tanpa perlu diprogram secara eksplisit. Sistem ini secara spesifik menerapkan algoritma <b>Logistic Regression</b>, yaitu metode klasifikasi prediktif yang sangat andal untuk mengklasifikasikan keluaran biner (seperti kelas 'Terindikasi Diabetes' atau 'Normal') berdasarkan evaluasi bobot dari berbagai fitur klinis pasien.""",
        badge_color="#8b5cf6"
    ), unsafe_allow_html=True)

with col_ov3:
    st.markdown(_overview_card(
        icon="📱",
        title="Tentang Aplikasi Ini",
        content="""Aplikasi web ini dibangun menggunakan kerangka kerja Streamlit sebagai instrumen alat bantu skrining kesehatan. Dengan memasukkan parameter medis seperti umur, indeks massa tubuh (BMI), kadar HbA1c, hingga glukosa darah, sistem akan memproses data tersebut menggunakan model yang telah dievaluasi untuk menampilkan probabilitas risiko diabetes secara seketika <i>(real-time)</i>. Aplikasi ini dirancang untuk kemudahan akses masyarakat awam, namun <b>tidak ditujukan sebagai pengganti vonis diagnosis resmi dari dokter profesional</b>.""",
        badge_color="#3b82f6"
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 2. Perbandingan Dataset Sebelum vs Sesudah Balancing ───────────────────
st.markdown(CSS_SECTION.format("⚖️ Perbandingan Dataset (Sebelum vs Sesudah Balancing)"), unsafe_allow_html=True)

# Calculate raw vs balanced metrics
total_bal = len(df)
pos_bal = int(df["diabetes"].sum())
neg_bal = total_bal - pos_bal
prev_bal = pos_bal / total_bal * 100

if df_raw is not None:
    total_raw = len(df_raw)
    pos_raw = int(df_raw["diabetes"].sum())
    neg_raw = total_raw - pos_raw
    prev_raw = pos_raw / total_raw * 100
else:
    total_raw, pos_raw, neg_raw, prev_raw = 100000, 8500, 91500, 8.5

tab_comp1, tab_comp2 = st.tabs(["📊 Tabel & Kartu Perbandingan", "💡 Alasan & Efek Resampling"])

with tab_comp1:
    col_raw_box, col_bal_box = st.columns(2, gap="medium")

    with col_raw_box:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.3);
                    border-radius:14px;padding:16px;margin-bottom:12px;">
          <div style="font-weight:700;color:#f87171;font-size:1.05rem;">🔴 Sebelum Balancing (Data Mentah)</div>
          <div style="font-size:0.8rem;color:#94a3b8;">File: <code>data/dataset_diabetes.csv</code> (Imbalanced)</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(_card(f"{total_raw:,}", "Total Data"), unsafe_allow_html=True)
        with c2: st.markdown(_card(f"{pos_raw:,}", "Diabetes (+)"), unsafe_allow_html=True)
        with c3: st.markdown(_card(f"{neg_raw:,}", "Non-Diabetes (-)"), unsafe_allow_html=True)
        with c4: st.markdown(_card(f"{prev_raw:.1f}%", "Prevalensi", "Bias Tinggi (1 : 11)"), unsafe_allow_html=True)

    with col_bal_box:
        st.markdown("""
        <div style="background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.3);
                    border-radius:14px;padding:16px;margin-bottom:12px;">
          <div style="font-weight:700;color:#4ade80;font-size:1.05rem;">🟢 Setelah Balancing & Preprocessing</div>
          <div style="font-size:0.8rem;color:#94a3b8;">File: <code>data/balanced_dataset.csv</code> (IQR Cleaned + Balanced)</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(_card(f"{total_bal:,}", "Total Data"), unsafe_allow_html=True)
        with c2: st.markdown(_card(f"{pos_bal:,}", "Diabetes (+)"), unsafe_allow_html=True)
        with c3: st.markdown(_card(f"{neg_bal:,}", "Non-Diabetes (-)"), unsafe_allow_html=True)
        with c4: st.markdown(_card(f"{prev_bal:.1f}%", "Prevalensi", "Ideal (1 : 1)"), unsafe_allow_html=True)

    st.markdown("""
| Indikator | Sebelum Balancing (Raw) | Sesudah Balancing & Preprocessing | Implikasi Klinis / ML |
|-----------|-------------------------|-----------------------------------|-----------------------|
| **File CSV** | `data/dataset_diabetes.csv` | `data/balanced_dataset.csv` | Sumber dataset utama |
| **Jumlah Baris** | **100.000** sampel | **12.680** sampel | Outlier IQR dibuang + Under-sampling |
| **Kelas Diabetes (+)** | 8.500 sampel (8.5%) | 6.340 sampel (50.0%) | Proporsi kelas positif dijaga |
| **Kelas Non-Diabetes (-)** | 91.500 sampel (91.5%) | 6.340 sampel (50.0%) | Mengurangi bias ke kelas mayoritas |
| **Rasio Kelas (+ : -)** | **1 : 10.7 (Sangat Imbalanced)** | **1 : 1 (Ideal / Balanced)** | Menjamin evaluasi F1-Score & Sensitivity adil |
""")

with tab_comp2:
    st.info("""
🎯 **Mengapa Dataset Perlu Dibalancing?**

- **Masalah Imbalanced Data**: Pada dataset asli (`dataset_diabetes.csv`), 91.5% pasien berlabel Non-Diabetes dan hanya 8.5% Diabetes. Jika model langsung dilatih pada data ini, model cenderung terbias untuk selalu menebak **"Non-Diabetes"** dan tetap mendapatkan akurasi 91.5% — namun gagal mendeteksi pasien yang benar-benar sakit *(High False Negative Rate)*.
- **Solusi Resampling & IQR Filtering**: 
  1. Fitur numerik (BMI, HbA1c, Glukosa) dibersihkan dari pencilan (*outliers*) menggunakan batas **IQR (Q1 - 1.5×IQR s/d Q3 + 1.5×IQR)**.
  2. Dataset diseimbangkan (*Under-sampling*) menjadi proporsi **50% Diabetes : 50% Non-Diabetes** (total 12.680 baris) agar model belajar membedakan fitur kedua kelas secara adil.
""", icon="💡")

st.markdown("<br>", unsafe_allow_html=True)

# ── 3. Penjelasan Variabel Dataset ──────────────────────────────────────────
st.markdown(CSS_SECTION.format("📋 Penjelasan Fitur Klinis Dataset"), unsafe_allow_html=True)
st.caption("Deskripsi lengkap setiap fitur klinis yang digunakan dalam model klasifikasi diabetes.")

# Baris 1 — 4 variabel pertama
r1c1, r1c2, r1c3, r1c4 = st.columns(4, gap="small")
with r1c1:
    st.markdown(_var_card(
        icon="👤",
        title="Jenis Kelamin (Gender)",
        tipe="Kategorikal",
        deskripsi="Kategori jenis kelamin pasien: <b>Male</b> (Laki-laki) atau <b>Female</b> (Perempuan). Dikodekan dengan One-Hot Encoding.",
        badge_color="#8b5cf6"
    ), unsafe_allow_html=True)

with r1c2:
    st.markdown(_var_card(
        icon="🎂",
        title="Usia (Age)",
        tipe="Numerik · Tahun",
        deskripsi="Umur pasien dalam tahun. Rentang data antara <b>1–80 tahun</b>. Risiko diabetes umumnya meningkat seiring pertambahan usia.",
        badge_color="#6366f1"
    ), unsafe_allow_html=True)

with r1c3:
    st.markdown(_var_card(
        icon="🫀",
        title="Tekanan Darah (Hypertension)",
        tipe="Biner · 0 / 1",
        deskripsi="Riwayat tekanan darah tinggi pasien. <b>1</b> = memiliki hipertensi, <b>0</b> = tidak ada riwayat hipertensi.",
        badge_color="#3b82f6"
    ), unsafe_allow_html=True)

with r1c4:
    st.markdown(_var_card(
        icon="❤️",
        title="Penyakit Jantung (Heart Disease)",
        tipe="Biner · 0 / 1",
        deskripsi="Indikator riwayat penyakit jantung. <b>1</b> = pernah memiliki penyakit jantung, <b>0</b> = tidak ada riwayat.",
        badge_color="#0ea5e9"
    ), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Baris 2 — 4 variabel berikutnya
r2c1, r2c2, r2c3, r2c4 = st.columns(4, gap="small")
with r2c1:
    st.markdown(_var_card(
        icon="🚬",
        title="Riwayat Merokok (Smoking History)",
        tipe="Kategorikal · 6 kelas",
        deskripsi="Kebiasaan merokok: <b>never</b>, <b>current</b>, <b>former</b>, <b>ever</b>, <b>not current</b>, <b>No Info</b>.",
        badge_color="#10b981"
    ), unsafe_allow_html=True)

with r2c2:
    st.markdown(_var_card(
        icon="⚖️",
        title="BMI (Body Mass Index)",
        tipe="Numerik · kg/m²",
        deskripsi="Indeks massa tubuh. Normal: <b>18,5–24,9</b> · Overweight: <b>25–29,9</b> · Obesitas: <b>≥ 30</b> (Kara et al., 2024).",
        badge_color="#f59e0b"
    ), unsafe_allow_html=True)

with r2c3:
    st.markdown(_var_card(
        icon="🩸",
        title="Tingkat HbA1c (HbA1c Level)",
        tipe="Numerik · %",
        deskripsi="Rata-rata gula darah 2–3 bulan. Normal: <b>&lt;5,7%</b> · Prediabetes: <b>5,7–6,4%</b> · Diabetes: <b>≥ 6,5%</b> (Kara et al., 2024).",
        badge_color="#ef4444"
    ), unsafe_allow_html=True)

with r2c4:
    st.markdown(_var_card(
        icon="💉",
        title="Kadar Gula Darah (Blood Glucose Level)",
        tipe="Numerik · mg/dL",
        deskripsi="Glukosa darah harian. Diabetes jika sewaktu <b>≥ 200 mg/dL</b> atau puasa <b>≥ 126 mg/dL</b> (Jasmani, 2016).",
        badge_color="#ec4899"
    ), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Tabel ringkasan kompak
with st.expander("🔍 Tabel Ringkasan Variabel", expanded=False):
    st.markdown("""
| # | Variabel | Tipe | Rentang / Kategori | Pengaruh Klinis |
|---|----------|------|-------------------|-----------------|
| 1 | Gender | Kategorikal | Male / Female | Faktor demografis |
| 2 | Age | Numerik | 1 – 80 tahun | Risiko meningkat seiring usia |
| 3 | Hypertension | Biner | 0 / 1 | Komorbiditas risiko tinggi |
| 4 | Heart Disease | Biner | 0 / 1 | Komorbiditas risiko tinggi |
| 5 | Smoking History | Kategorikal | 6 kategori | Faktor gaya hidup |
| 6 | BMI | Numerik | kg/m² | Obesitas → risiko naik |
| 7 | HbA1c Level | Numerik | % | **Indikator utama DM** |
| 8 | Blood Glucose | Numerik | mg/dL | **Indikator utama DM** |
""")

st.divider()

# ── 4. Static EDA Images ───────────────────────────────────────────────────

st.markdown(CSS_SECTION.format("Missing Value Matrix"), unsafe_allow_html=True)
st.image("results/1_missing_value.png", width='stretch')

st.markdown(CSS_SECTION.format("Boxplot Setelah Outlier Removal"), unsafe_allow_html=True)
st.image("results/3_boxplot_setelah.png", width='stretch')

st.markdown(CSS_SECTION.format("Distribusi Kategorikal"), unsafe_allow_html=True)
st.image("results/4_dist_kategorikal.png", width='stretch')

st.markdown(CSS_SECTION.format("Distribusi Fitur Numerik"), unsafe_allow_html=True)
st.image("results/5_dist_numerik.png", width='stretch')

st.markdown(CSS_SECTION.format("Heatmap Korelasi"), unsafe_allow_html=True)
st.image("results/6_heatmap_korelasi.png", width='stretch')
