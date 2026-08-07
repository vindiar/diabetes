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

def _card(value, label):
    return f"""
    <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(139,92,246,0.3);
                border-radius:16px;padding:24px;text-align:center;backdrop-filter:blur(10px);">
      <div style="font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#60a5fa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value}</div>
      <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;
                  text-transform:uppercase;letter-spacing:1px;">{label}</div>
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

st.markdown("## 🏠 Dashboard & Eksplorasi Data")
st.caption(f"Dataset: `data/balanced_dataset.csv` — **{len(df):,}** baris, 12 fitur")
st.divider()

# ── Metric cards ──────────────────────────────────────────────────────────
total = len(df)
positif = int(df["diabetes"].sum())
negatif = total - positif
prevalensi = positif / total * 100

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(_card(f"{total:,}", "Total Data"), unsafe_allow_html=True)
with c2: st.markdown(_card(f"{positif:,}", "Diabetes (+)"), unsafe_allow_html=True)
with c3: st.markdown(_card(f"{negatif:,}", "Non-Diabetes (-)"), unsafe_allow_html=True)
with c4: st.markdown(_card(f"{prevalensi:.1f}%", "Prevalensi"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Penjelasan Variabel Dataset ──────────────────────────────────────────
st.markdown(CSS_SECTION.format("📋 Penjelasan Variabel Dataset"), unsafe_allow_html=True)
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

# ── Static EDA Images ───────────────────────────────────────────────────

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
