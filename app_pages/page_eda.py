"""
app_pages/page_eda.py
─────────────────────
ML Lifecycle Phase 1 — DATA EXPLORATION
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
