"""
app_pages/page_klasifikasi.py
───────────────────────────
CRISP-DM Fase 6 — DEPLOYMENT (Serving)
Interactive single-patient classification form with placeholders & PDF export.
"""

import io
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from utils.model import predict_single
from utils import plots


model     = st.session_state.model
scaler    = st.session_state.scaler
features  = st.session_state.features

st.markdown("## 🔍 Klasifikasi Risiko Diabetes")
st.caption("Masukkan data pasien lalu klik **Klasifikasi Sekarang**.")
st.divider()

col_form, col_result = st.columns([1, 1], gap="large")

# ── Input Form (batched — no rerun on every widget change) ────────────────
with col_form:
    st.markdown("#### 📋 Data Pasien")
    with st.form("prediction_form", border=True):
        gender = st.selectbox(
            "👤 Jenis Kelamin",
            ["Perempuan", "Laki-laki"],
        )
        gender_mapped = "Female" if gender == "Perempuan" else "Male"

        age = st.number_input(
            "🎂 Usia (tahun)", min_value=1, max_value=120, value=None, placeholder="Contoh: 35", step=1
        )

        st.markdown("⚖️ **Body Mass Index (BMI)**")
        c_bb, c_tb = st.columns(2)
        with c_bb:
            berat_badan = st.number_input("Berat Badan (Kg)", min_value=10.0, max_value=300.0, value=None, placeholder="Contoh: 75.0", step=0.1, format="%.1f")
        with c_tb:
            tinggi_badan = st.number_input("Tinggi Badan (m)", min_value=0.5, max_value=3.0, value=None, placeholder="Contoh: 1.65", step=0.01, format="%.2f")

        if berat_badan is not None and tinggi_badan is not None and tinggi_badan > 0:
            bmi = berat_badan / (tinggi_badan * tinggi_badan)
            st.caption(f"Hasil otomatis BMI: **{bmi:.1f}**")
        else:
            bmi = None
            st.caption("Hasil otomatis BMI: *-* *(isi berat & tinggi badan)*")

        hba1c = st.number_input(
            "🩸 HbA1c Level (%)", min_value=3.0, max_value=15.0, value=None, placeholder="Contoh: 5.5", step=0.1, format="%.1f",
            help="Rata-rata kadar gula darah 2-3 bulan terakhir dari hasil tes lab. Acuan normal: < 5.7%, Prediabetes: 5.7-6.4%, Diabetes: ≥ 6.5%"
        )
        glucose = st.number_input(
            "💉 Kadar Gula Darah (mg/dL)", min_value=50, max_value=500, value=None, placeholder="Contoh: 130", step=1,
            help="Kadar glukosa darah harian (sewaktu/puasa). Indikasi Diabetes jika sewaktu ≥ 200 mg/dL atau puasa ≥ 126 mg/dL"
        )

        c1, c2 = st.columns(2)
        with c1:
            hypertension = st.radio(
                "🫀 Hipertensi", [0, 1],
                format_func=lambda x: "Ya" if x else "Tidak",
                horizontal=True,
            )
        with c2:
            heart_disease = st.radio(
                "❤️ Penyakit Jantung", [0, 1],
                format_func=lambda x: "Ya" if x else "Tidak",
                horizontal=True,
            )

        smoking_options = {
            "Tidak Pernah": "never",
            "Tidak Ada Informasi": "No Info",
            "Perokok Aktif": "current",
            "Mantan Perokok": "former",
            "Pernah Merokok": "ever",
            "Sudah Berhenti": "not current",
        }
        smoking_label = st.selectbox(
            "🚬 Riwayat Merokok",
            list(smoking_options.keys()),
        )
        smoking = smoking_options[smoking_label]

        submitted = st.form_submit_button(
            "🔍 Klasifikasi Sekarang", use_container_width=True, type="primary"
        )

# ── Result Panel ──────────────────────────────────────────────────────────
with col_result:
    st.markdown("#### 📊 Hasil Klasifikasi")

    if not submitted:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#64748b;">
          <div style="font-size:4rem;">🩺</div>
          <div style="font-size:1rem;margin-top:12px;">
            Isi form pasien di sebelah kiri<br>dan klik <b>Klasifikasi Sekarang</b>
          </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Validate that all required number inputs have values
    missing_fields = []
    if age is None: missing_fields.append("Usia")
    if berat_badan is None: missing_fields.append("Berat Badan")
    if tinggi_badan is None: missing_fields.append("Tinggi Badan")
    if hba1c is None: missing_fields.append("HbA1c Level")
    if glucose is None: missing_fields.append("Kadar Gula Darah")

    if missing_fields:
        st.warning(f"⚠️ Harap lengkapi kolom berikut sebelum melakukan klasifikasi: **{', '.join(missing_fields)}**", icon="⚠️")
        st.stop()

    # Create input dictionary
    input_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose,
        'gender_Male': 1 if gender_mapped == "Male" else 0,
        'smoking_history_current': 1 if smoking == "current" else 0,
        'smoking_history_ever': 1 if smoking == "ever" else 0,
        'smoking_history_former': 1 if smoking == "former" else 0,
        'smoking_history_never': 1 if smoking == "never" else 0,
        'smoking_history_not current': 1 if smoking == "not current" else 0,
    }

    # Run inference
    result = predict_single(model, scaler, features, input_dict=input_dict)

    prediction      = result["prediction"]
    prob_diabetes   = result["prob_diabetes"]
    prob_no_diabetes = result["prob_no_diabetes"]

    # Result banner
    if prediction == 1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(239,68,68,.2),rgba(185,28,28,.2));
                    border:2px solid rgba(239,68,68,.6);border-radius:20px;
                    padding:28px;text-align:center;">
          <div style="font-size:1.9rem;font-weight:800;">⚠️ TERINDIKASI DIABETES</div>
          <div style="color:#cbd5e1;margin-top:6px;">
            Probabilitas Diabetes: <b>{prob_diabetes:.1f}%</b>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(34,197,94,.2),rgba(21,128,61,.2));
                    border:2px solid rgba(34,197,94,.6);border-radius:20px;
                    padding:28px;text-align:center;">
          <div style="font-size:1.9rem;font-weight:800;">✅ TIDAK TERINDIKASI DIABETES</div>
          <div style="color:#cbd5e1;margin-top:6px;">
            Probabilitas Tidak Diabetes: <b>{prob_no_diabetes:.1f}%</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bar
    st.caption("Distribusi Probabilitas")
    st.pyplot(
        plots.fig_probability_bar(prob_no_diabetes, prob_diabetes),
        use_container_width=True,
    )
    plt.close("all")

    # Risk factor table
    st.divider()
    st.caption("📋 Ringkasan Faktor Risiko Pasien")
    bmi_cat = (
        "Obesitas" if bmi >= 30 else
        "Overweight" if bmi >= 25 else
        "Normal" if bmi >= 18.5 else "Underweight"
    )
    hba1c_status = (
        "Tinggi (>=6.5)" if hba1c >= 6.5 else
        "Prediabetes (5.7–6.4)" if hba1c >= 5.7 else "Normal"
    )
    glucose_status = (
        "Tinggi (>=200)" if glucose >= 200 else
        "Prediabetes (140–199)" if glucose >= 140 else "Normal"
    )

    st.markdown(f"""
| Faktor | Nilai | Status |
|--------|-------|--------|
| Berat Badan | {berat_badan:.1f} Kg | - |
| Tinggi Badan | {tinggi_badan:.2f} m | - |
| BMI | {bmi:.1f} | {bmi_cat} |
| HbA1c | {hba1c} | {hba1c_status} |
| Blood Glucose | {glucose} mg/dL | {glucose_status} |
| Hipertensi | {'Ya' if hypertension else 'Tidak'} | {'High Risk' if hypertension else 'Normal'} |
| Penyakit Jantung | {'Ya' if heart_disease else 'Tidak'} | {'High Risk' if heart_disease else 'Normal'} |
    """)

    if prediction == 1:
        st.warning(
            "Hasil ini bersifat **indikatif**. "
            "Konsultasikan dengan dokter untuk diagnosa yang akurat.",
            icon="⚠️",
        )
    else:
        st.success(
            "Pertahankan gaya hidup sehat! Tetap jaga pola makan dan olahraga rutin.",
            icon="🎉",
        )

    # ── PDF Export ────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🖨️ Ekspor Laporan")

    from utils.pdf import generate_pdf_report

    pdf_bytes = generate_pdf_report(
        gender_label=gender,
        age_val=age,
        bb_val=berat_badan,
        tb_val=tinggi_badan,
        bmi_val=bmi,
        bmi_cat_val=bmi_cat,
        hba1c_val=hba1c,
        hba1c_status_val=hba1c_status,
        glucose_val=glucose,
        glucose_status_val=glucose_status,
        hypertension_val=hypertension,
        heart_val=heart_disease,
        smoking_label_val=smoking_label,
        pred_val=prediction,
        prob_diabetes_val=prob_diabetes,
        prob_no_diabetes_val=prob_no_diabetes,
    )


    if pdf_bytes:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"laporan_klasifikasi_diabetes_{ts}.pdf"
        st.download_button(
            label="📄 Unduh Laporan PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
        st.caption("Laporan PDF berisi hasil klasifikasi, data pasien, faktor risiko, dan disclaimer medis.")
    else:
        st.info("Install `fpdf2` untuk mengaktifkan fitur ekspor PDF: `pip install fpdf2`", icon="ℹ️")
