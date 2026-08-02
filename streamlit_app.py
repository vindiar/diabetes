"""
streamlit_app.py
────────────────
Entry point — ML Development Lifecycle App: klasifikasi Diabetes
Struktur mengikuti skill: developing-with-streamlit
  • st.navigation() dengan app_pages/ (posisi sidebar)
  • Model & data diinisialisasi sekali ke st.session_state
  • Shared CSS & branding diletakkan di sini sebelum page.run()
"""

import os
import streamlit as st
from utils.data import load_dataset
from utils.model import load_artifacts, evaluate_model, MODEL_PATH

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="klasifikasi Diabetes | Logistic Regression",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Override nav link colors */
[data-testid="stSidebarNav"] a { color: #c4b5fd !important; }
[data-testid="stSidebarNav"] a:hover { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State: Initialize Once ───────────────────────────────────────────
# Load dataset
if "df" not in st.session_state:
    st.session_state.df = load_dataset()

# Load models and evaluate
if "preprocessed" not in st.session_state:
    with st.spinner("⏳ Memuat model..."):
        model, scaler, features = load_artifacts()
        eval_results = evaluate_model(st.session_state.df, model, scaler, features)

    st.session_state.model    = model
    st.session_state.scaler   = scaler
    st.session_state.features = features
    st.session_state.eval_results = eval_results
    st.session_state.preprocessed = True  # gate flag

# ── Shared Sidebar Branding ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 DiabetesPredict")
    st.caption("Logistic Regression · ML App")
    st.divider()
    st.markdown(f"""
    **Dataset**
    - `data/balanced_dataset.csv`
    - **{len(st.session_state.df):,}** baris · 12 fitur
    
    **Model**
    - `models/model_diabetes.pkl`
    - Akurasi: **{st.session_state.eval_results['accuracy']*100:.2f}%**
    - AUC: **{st.session_state.eval_results['roc_auc']:.4f}**
    """)
    st.divider()

    # Show model source
    if os.path.exists(MODEL_PATH):
        st.success("Model .pkl di-load dari disk", icon="💾")
    else:
        st.error("Model .pkl tidak ditemukan!", icon="❌")

# ── Navigation (st.navigation — skill pattern) ────────────────────────────────
page = st.navigation(
    [
        st.Page("app_pages/page_eda.py",
                title="Dashboard & EDA",
                icon=":material/dashboard:"),
        st.Page("app_pages/page_model.py",
                title="Performa Model",
                icon=":material/bar_chart:"),
        st.Page("app_pages/page_predict.py",
                title="klasifikasi",
                icon=":material/search:"),
        st.Page("app_pages/page_docs.py",
                title="Dokumentasi",
                icon=":material/menu_book:"),
    ],
    position="sidebar",
)

page.run()
