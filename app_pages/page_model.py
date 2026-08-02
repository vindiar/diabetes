"""
app_pages/page_model.py
───────────────────────
ML Lifecycle Phase 3 — EVALUATION
Shows model performance metrics, confusion matrix, ROC, feature importance.
"""

import streamlit as st

CSS_SECTION = """
<div style="font-size:1.4rem;font-weight:700;color:#a78bfa;
            border-left:4px solid #a78bfa;padding-left:12px;
            margin-bottom:16px;">{}</div>
"""

def _metric_card(value, label):
    return f"""
    <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(139,92,246,0.3);
                border-radius:16px;padding:24px;text-align:center;">
      <div style="font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#60a5fa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value}</div>
      <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;
                  text-transform:uppercase;letter-spacing:1px;">{label}</div>
    </div>"""

model  = st.session_state.model
result = st.session_state.eval_results

st.markdown("## 📊 Performa Model — Logistic Regression")
st.caption("Evaluasi pada **20% test data** (stratified split, random_state=42)")
st.divider()

# ── Scalar metrics ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
items = [
    ("Akurasi",  f"{result['accuracy']*100:.2f}%"),
    ("Presisi",  f"{result['precision']*100:.2f}%"),
    ("Recall",   f"{result['recall']*100:.2f}%"),
    ("F1-Score", f"{result['f1']*100:.2f}%"),
]
for col, (label, val) in zip([c1, c2, c3, c4], items):
    with col:
        st.markdown(_metric_card(val, label), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Confusion Matrix + ROC ────────────────────────────────────────────────
c_l, c_r = st.columns(2)

with c_l:
    st.markdown(CSS_SECTION.format("Confusion Matrix"), unsafe_allow_html=True)
    st.image("results/7_confusion_matrix.png", width='stretch')

    tn, fp, fn, tp = result["confusion_matrix"].ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05);border-radius:12px;padding:16px;margin-top:8px;">
    <table style="width:100%;color:#e2e8f0;font-size:0.9rem;">
      <tr><td>✅ True Positive (TP)</td><td><b>{tp:,}</b></td></tr>
      <tr><td>✅ True Negative (TN)</td><td><b>{tn:,}</b></td></tr>
      <tr><td>❌ False Positive (FP)</td><td><b>{fp:,}</b></td></tr>
      <tr><td>❌ False Negative (FN)</td><td><b>{fn:,}</b></td></tr>
      <tr><td>🔬 Sensitivitas</td><td><b>{sensitivity:.4f}</b></td></tr>
      <tr><td>🔬 Spesifisitas</td><td><b>{specificity:.4f}</b></td></tr>
    </table></div>
    """, unsafe_allow_html=True)

with c_r:
    st.markdown(CSS_SECTION.format("ROC Curve"), unsafe_allow_html=True)
    st.image("results/8_roc_curve.png", width='stretch')

    auc_val = result["roc_auc"]
    interp = (
        "🌟 **Excellent** — performa sangat baik." if auc_val >= 0.9
        else "✅ **Good** — performa baik." if auc_val >= 0.8
        else "⚠️ **Fair** — perlu perbaikan." if auc_val >= 0.7
        else "❌ **Poor** — perlu ditingkatkan."
    )
    st.markdown(f"""
    <div style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.3);
                border-radius:12px;padding:16px;margin-top:8px;">
      <b>AUC = {auc_val:.4f}</b><br>{interp}
    </div>""", unsafe_allow_html=True)

# ── Feature Importance ────────────────────────────────────────────────────
st.divider()
st.markdown(CSS_SECTION.format("Feature Importance (Koefisien Logistic Regression)"),
            unsafe_allow_html=True)

st.image("results/9_feature_importance.png", width='stretch')

# ── Cross Validation ──────────────────────────────────────────────────────
st.divider()
st.markdown(CSS_SECTION.format("Hasil 5-Fold Cross Validation"),
            unsafe_allow_html=True)

st.image("results/10_cross_validation.png", width='stretch')

# ── Classification Report ─────────────────────────────────────────────────
st.divider()
st.markdown(CSS_SECTION.format("Classification Report"), unsafe_allow_html=True)
st.code(result["classification_report"], language="text")
