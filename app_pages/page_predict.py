"""
app_pages/page_predict.py
─────────────────────────
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

    def _clean_text(text):
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("≥", ">=").replace("≤", "<=").replace("—", "-").replace("⚠️", "").replace("✅", "").strip()
        return text.encode("latin-1", "ignore").decode("latin-1")

    def _generate_pdf(
        gender_label, age_val, bb_val, tb_val, bmi_val, bmi_cat_val,
        hba1c_val, hba1c_status_val, glucose_val, glucose_status_val,
        hypertension_val, heart_val, smoking_label_val,
        pred_val, prob_diabetes_val, prob_no_diabetes_val
    ):
        """Generate PDF report using fpdf2 and return bytes."""
        try:
            from fpdf import FPDF
        except ImportError:
            return None

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # ── Header ────────────────────────────────────────────────────────
            pdf.set_fill_color(88, 28, 135)   # purple-900
            pdf.rect(0, 0, 210, 30, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_xy(10, 8)
            pdf.cell(0, 10, _clean_text("LAPORAN KLASIFIKASI RISIKO DIABETES"), ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_xy(10, 20)
            pdf.cell(0, 6, _clean_text("DiabetesKlasifikasi | Logistic Regression | CRISP-DM"), ln=True)

            pdf.set_text_color(30, 30, 30)
            pdf.set_y(36)

            # ── Timestamp ─────────────────────────────────────────────────────
            now = datetime.now().strftime("%d %B %Y, %H:%M WIB")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, _clean_text(f"Dicetak pada: {now}"), ln=True, align="R")
            pdf.ln(2)

            # ── Hasil Klasifikasi ──────────────────────────────────────────────
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 8, _clean_text("HASIL KLASIFIKASI"), ln=True)
            pdf.set_draw_color(139, 92, 246)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

            if pred_val == 1:
                pdf.set_fill_color(254, 226, 226)   # red-100
                pdf.set_text_color(185, 28, 28)       # red-700
                verdict = "TERINDIKASI DIABETES"
                prob_text = f"Probabilitas Diabetes: {prob_diabetes_val:.1f}%"
            else:
                pdf.set_fill_color(220, 252, 231)   # green-100
                pdf.set_text_color(21, 128, 61)       # green-700
                verdict = "TIDAK TERINDIKASI DIABETES"
                prob_text = f"Probabilitas Tidak Diabetes: {prob_no_diabetes_val:.1f}%"

            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 12, _clean_text(verdict), ln=True, align="C", fill=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 8, _clean_text(prob_text), ln=True, align="C")
            pdf.ln(4)

            # ── Data Pasien ────────────────────────────────────────────────────
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 8, _clean_text("DATA PASIEN"), ln=True)
            pdf.set_draw_color(139, 92, 246)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

            rows = [
                ("Jenis Kelamin", gender_label),
                ("Usia", f"{age_val} tahun"),
                ("Berat Badan", f"{bb_val:.1f} kg"),
                ("Tinggi Badan", f"{tb_val:.2f} m"),
                ("Riwayat Merokok", smoking_label_val),
                ("Hipertensi", "Ya" if hypertension_val else "Tidak"),
                ("Penyakit Jantung", "Ya" if heart_val else "Tidak"),
            ]
            pdf.set_font("Helvetica", "", 10)
            fill = False
            pdf.set_fill_color(245, 243, 255)
            for label, val in rows:
                pdf.set_fill_color(245, 243, 255) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.cell(80, 8, _clean_text(label), border=1, fill=fill)
                pdf.cell(0, 8, _clean_text(val), border=1, fill=fill, ln=True)
                fill = not fill
            pdf.ln(4)

            # ── Faktor Risiko Klinis ───────────────────────────────────────────
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 8, _clean_text("FAKTOR RISIKO KLINIS"), ln=True)
            pdf.set_draw_color(139, 92, 246)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

            risk_rows = [
                ("BMI", f"{bmi_val:.1f} kg/m2", bmi_cat_val),
                ("HbA1c Level", f"{hba1c_val:.1f}%", hba1c_status_val),
                ("Kadar Gula Darah", f"{glucose_val} mg/dL", glucose_status_val),
            ]
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(88, 28, 135)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(60, 8, _clean_text("Parameter"), border=1, fill=True)
            pdf.cell(60, 8, _clean_text("Nilai"), border=1, fill=True)
            pdf.cell(0, 8, _clean_text("Status Klinis"), border=1, fill=True, ln=True)

            pdf.set_font("Helvetica", "", 10)
            fill = False
            for param, val, status in risk_rows:
                pdf.set_fill_color(245, 243, 255) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(30, 30, 30)
                pdf.cell(60, 8, _clean_text(param), border=1, fill=fill)
                pdf.cell(60, 8, _clean_text(val), border=1, fill=fill)
                pdf.cell(0, 8, _clean_text(status), border=1, fill=fill, ln=True)
                fill = not fill
            pdf.ln(4)

            # ── Distribusi Probabilitas ────────────────────────────────────────
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 8, _clean_text("DISTRIBUSI PROBABILITAS"), ln=True)
            pdf.set_draw_color(139, 92, 246)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)

            pdf.set_font("Helvetica", "", 10)
            bar_y = pdf.get_y()
            bar_w_nd = (prob_no_diabetes_val / 100) * 150
            pdf.set_fill_color(34, 197, 94)
            pdf.rect(30, bar_y, bar_w_nd, 8, 'F')
            pdf.set_xy(10, bar_y)
            pdf.cell(20, 8, "Normal")
            pdf.set_xy(185, bar_y)
            pdf.cell(0, 8, f"{prob_no_diabetes_val:.1f}%", ln=True)

            bar_y2 = pdf.get_y() + 2
            bar_w_d = (prob_diabetes_val / 100) * 150
            pdf.set_fill_color(239, 68, 68)
            pdf.rect(30, bar_y2, bar_w_d, 8, 'F')
            pdf.set_xy(10, bar_y2)
            pdf.cell(20, 8, "Diabetes")
            pdf.set_xy(185, bar_y2)
            pdf.cell(0, 8, f"{prob_diabetes_val:.1f}%", ln=True)
            pdf.ln(8)

            # ── Disclaimer ─────────────────────────────────────────────────────
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.set_fill_color(254, 249, 195)
            pdf.cell(0, 6, _clean_text("DISCLAIMER MEDIS"), ln=True, fill=True)
            pdf.set_font("Helvetica", "", 8)
            disclaimer = (
                "Laporan ini dihasilkan oleh sistem klasifikasi berbasis Machine Learning "
                "dan bersifat INDIKATIF, BUKAN merupakan diagnosis medis resmi. "
                "Hasil ini tidak menggantikan pemeriksaan dan diagnosis oleh dokter atau tenaga medis "
                "berpengalaman. Untuk konfirmasi diagnosis, segera konsultasikan dengan dokter profesional."
            )
            pdf.multi_cell(0, 5, _clean_text(disclaimer))

            return bytes(pdf.output())
        except Exception as e:
            st.error(f"Gagal memformat PDF: {e}")
            return None

    # Build PDF and show download button
    pdf_bytes = _generate_pdf(
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
