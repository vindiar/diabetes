"""
utils/plots.py
──────────────
Reusable chart functions.
All functions return matplotlib.figure.Figure — no st.pyplot() calls here.
UI pages call st.pyplot(fig) themselves, keeping this module testable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#e2e8f0",
    "grid.color": "#1e293b",
    "grid.alpha": 0.5,
})

C_BLUE   = "#60a5fa"
C_RED    = "#f87171"
C_PURPLE = "#a78bfa"
C_GREEN  = "#34d399"


def _base_fig(w=5, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    return fig, ax


# ── EDA Plots ─────────────────────────────────────────────────────────────────

def fig_class_distribution(df) -> plt.Figure:
    n_pos = df["diabetes"].sum()
    n_neg = len(df) - n_pos
    fig, ax = _base_fig(5, 4)
    ax.pie(
        [n_neg, n_pos],
        labels=["Tidak Diabetes", "Diabetes"],
        colors=[C_BLUE, C_RED],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="none"),
    )
    for t in ax.texts:
        t.set_color("#e2e8f0")
    ax.set_title("Distribusi Kelas Target", color="#e2e8f0", fontsize=13, fontweight="bold")
    return fig


def fig_gender_vs_diabetes(df) -> plt.Figure:
    fig, ax = _base_fig(5, 4)
    df_plot = df[df["gender"] != "Other"]
    grp = df_plot.groupby(["gender", "diabetes"]).size().unstack(fill_value=0)
    grp.columns = ["Tidak Diabetes", "Diabetes"]
    grp.plot(kind="bar", ax=ax, color=[C_BLUE, C_RED], edgecolor="none", width=0.6)
    ax.set_xlabel("Gender"); ax.set_ylabel("Jumlah")
    ax.set_title("Gender vs Diabetes", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.tick_params(axis="x", rotation=0)
    return fig


def fig_age_distribution(df) -> plt.Figure:
    fig, ax = _base_fig(5, 4)
    for val, color, label in [(0, C_BLUE, "Tidak Diabetes"), (1, C_RED, "Diabetes")]:
        ax.hist(df[df["diabetes"] == val]["age"], bins=40,
                color=color, alpha=0.7, label=label, edgecolor="none")
    ax.set_xlabel("Usia (tahun)"); ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi Usia", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.grid(True, alpha=0.2)
    return fig


def fig_bmi_boxplot(df) -> plt.Figure:
    fig, ax = _base_fig(5, 4)
    bp = ax.boxplot(
        [df[df["diabetes"] == 0]["bmi"], df[df["diabetes"] == 1]["bmi"]],
        patch_artist=True,
        labels=["Tidak Diabetes", "Diabetes"],
        boxprops=dict(facecolor=C_BLUE, alpha=0.7),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#94a3b8"),
        capprops=dict(color="#94a3b8"),
        flierprops=dict(marker="o", color="#64748b", alpha=0.3, markersize=2),
    )
    bp["boxes"][1].set_facecolor(C_RED)
    ax.set_ylabel("BMI")
    ax.set_title("Distribusi BMI per Kelas", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)
    return fig


def fig_hba1c_boxplot(df) -> plt.Figure:
    fig, ax = _base_fig(5, 4)
    bp = ax.boxplot(
        [df[df["diabetes"] == 0]["HbA1c_level"], df[df["diabetes"] == 1]["HbA1c_level"]],
        patch_artist=True,
        labels=["Tidak Diabetes", "Diabetes"],
        boxprops=dict(facecolor=C_BLUE, alpha=0.7),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#94a3b8"),
        capprops=dict(color="#94a3b8"),
        flierprops=dict(marker="o", color="#64748b", alpha=0.3, markersize=2),
    )
    bp["boxes"][1].set_facecolor(C_RED)
    ax.set_ylabel("HbA1c Level")
    ax.set_title("Distribusi HbA1c per Kelas", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)
    return fig


def fig_glucose_boxplot(df) -> plt.Figure:
    fig, ax = _base_fig(5, 4)
    bp = ax.boxplot(
        [df[df["diabetes"] == 0]["blood_glucose_level"], df[df["diabetes"] == 1]["blood_glucose_level"]],
        patch_artist=True,
        labels=["Tidak Diabetes", "Diabetes"],
        boxprops=dict(facecolor=C_BLUE, alpha=0.7),
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#94a3b8"),
        capprops=dict(color="#94a3b8"),
        flierprops=dict(marker="o", color="#64748b", alpha=0.3, markersize=2),
    )
    bp["boxes"][1].set_facecolor(C_RED)
    ax.set_ylabel("Blood Glucose Level (mg/dL)")
    ax.set_title("Distribusi Blood Glucose per Kelas", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)
    return fig


def fig_correlation_heatmap(df) -> plt.Figure:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # Copy df to encode categoricals just for the correlation matrix
    df_corr = df.copy()
    for col in ["gender", "smoking_history"]:
        if pd.api.types.is_string_dtype(df_corr[col]) or df_corr[col].dtype == "object":
            df_corr[col] = LabelEncoder().fit_transform(df_corr[col])

    cols = ["gender", "age", "hypertension", "heart_disease", "smoking_history",
            "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]
    # .astype(float) ensures Arrow-backed columns are converted to NumPy floats
    corr = df_corr[cols].astype(float).corr()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0)
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f",
        cmap="YlGn", 
        linewidths=0.5, linecolor="#1e293b",
        annot_kws={"size": 10, "color": "#0f172a"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Matriks Korelasi (Pearson)", color="#e2e8f0", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#e2e8f0")
    
    # Put x-axis labels at bottom, rotate 90
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    return fig


# ── Model Evaluation Plots ────────────────────────────────────────────────────

def fig_confusion_matrix(cm) -> plt.Figure:
    fig, ax = _base_fig(5, 4)
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        cmap="Blues",
        xticklabels=["Tidak Diabetes", "Diabetes"],
        yticklabels=["Tidak Diabetes", "Diabetes"],
        linewidths=2, linecolor="#0f172a",
        annot_kws={"size": 16, "weight": "bold", "color": "#0f172a"},
    )
    ax.set_xlabel("klasifikasi", fontsize=11)
    ax.set_ylabel("Aktual", fontsize=11)
    ax.set_title("Confusion Matrix", color="#e2e8f0", fontsize=13, fontweight="bold")
    return fig


def fig_roc_curve(fpr, tpr, roc_auc) -> plt.Figure:
    fig, ax = _base_fig(5, 4)
    ax.plot(fpr, tpr, color=C_PURPLE, lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#475569", lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color=C_PURPLE)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    ax.grid(True, alpha=0.2)
    return fig


def fig_feature_importance(coef, feature_names) -> plt.Figure:
    sorted_idx = np.argsort(np.abs(coef))[::-1]
    # Positive coeff means higher risk -> RED. Negative coeff means lower risk -> GREEN.
    colors = [C_RED if c > 0 else C_GREEN for c in coef[sorted_idx]]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_alpha(0)
    ax.barh([feature_names[i] for i in sorted_idx], coef[sorted_idx],
            color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.set_xlabel("Koefisien")
    ax.set_title("Feature Importance (Koefisien Model)",
                 color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    ax.invert_yaxis()
    pos_p = mpatches.Patch(color=C_RED, label="Meningkatkan risiko")
    neg_p = mpatches.Patch(color=C_GREEN, label="Mengurangi risiko")
    ax.legend(handles=[pos_p, neg_p], facecolor="#1e293b",
              edgecolor="#334155", labelcolor="#e2e8f0")
    return fig


def fig_probability_bar(prob_no_diabetes, prob_diabetes) -> plt.Figure:
    fig, ax = _base_fig(6, 2)
    ax.barh(["Tidak Diabetes", "Diabetes"],
            [prob_no_diabetes, prob_diabetes],
            color=[C_BLUE, C_RED], edgecolor="none", height=0.5)
    for i, v in enumerate([prob_no_diabetes, prob_diabetes]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", color="#e2e8f0", fontweight="bold")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probabilitas (%)")
    ax.grid(True, axis="x", alpha=0.2)
    return fig
