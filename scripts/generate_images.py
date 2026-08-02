import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('data/balanced_dataset.csv')
df = df[df['gender'].isin(['Male', 'Female'])].copy()
df.reset_index(drop=True, inplace=True)

num_cols = ['HbA1c_level', 'bmi', 'blood_glucose_level']
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

mask = pd.Series([True] * len(df), index=df.index)
for col in num_cols:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    mask = mask & (df[col] >= lower) & (df[col] <= upper)

data_clean = df.loc[mask].copy()
data_clean.reset_index(drop=True, inplace=True)


sns.set_style('darkgrid')
plt.rcParams['font.size'] = 10

# 1. Missing Value
print("1. Missing Value")
fig = msno.matrix(df).get_figure()
plt.title('Missing Value Matrix')
plt.savefig('results/1_missing_value.png', bbox_inches='tight', dpi=300)
plt.close('all')

# 2. Boxplot sebelum
print("2. Boxplot Sebelum")
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
sns.boxplot(ax=axes[0], data=df['age'])
axes[0].set_title('Age')
sns.boxplot(ax=axes[1], data=df['HbA1c_level'])
axes[1].set_title('HbA1c Level')
sns.boxplot(ax=axes[2], data=df['bmi'])
axes[2].set_title('BMI')
sns.boxplot(ax=axes[3], data=df['blood_glucose_level'])
axes[3].set_title('Blood Glucose Level')
plt.suptitle('Boxplot Sebelum Outlier Removal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/2_boxplot_sebelum.png', bbox_inches='tight', dpi=300)
plt.close('all')

# 3. Boxplot setelah
print("3. Boxplot Setelah")
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
sns.boxplot(ax=axes[0], data=data_clean['age'])
axes[0].set_title('Age')
sns.boxplot(ax=axes[1], data=data_clean['HbA1c_level'])
axes[1].set_title('HbA1c Level')
sns.boxplot(ax=axes[2], data=data_clean['bmi'])
axes[2].set_title('BMI')
sns.boxplot(ax=axes[3], data=data_clean['blood_glucose_level'])
axes[3].set_title('Blood Glucose Level')
plt.suptitle('Boxplot Setelah Outlier Removal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/3_boxplot_setelah.png', bbox_inches='tight', dpi=300)
plt.close('all')

# 4. Distribusi Kelas Target vs Kategorikal
print("4. Distribusi Kategorikal")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(ax=axes[0], x='diabetes', data=data_clean, hue='gender')
axes[0].set_title('Distribusi Gender vs Diabetes')
sns.countplot(ax=axes[1], x='diabetes', data=data_clean, hue='smoking_history')
axes[1].set_title('Riwayat Merokok vs Diabetes')
axes[1].legend(fontsize=7)
sns.countplot(ax=axes[2], x='diabetes', data=data_clean, hue='heart_disease')
axes[2].set_title('Penyakit Jantung vs Diabetes')
plt.tight_layout()
plt.savefig('results/4_dist_kategorikal.png', bbox_inches='tight', dpi=300)
plt.close('all')

# 5. Distribusi fitur numerik per diabetes
print("5. Distribusi Numerik")
num_feats = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, feat in enumerate(num_feats):
    sns.histplot(data=data_clean, x=feat, hue='diabetes',
                 kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f'Distribusi {feat} per Kelas Diabetes')
plt.tight_layout()
plt.savefig('results/5_dist_numerik.png', bbox_inches='tight', dpi=300)
plt.close('all')

# 6. Heatmap korelasi
print("6. Heatmap Korelasi")
df_encoded = data_clean.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['gender', 'smoking_history'], drop_first=True)
plt.figure(figsize=(14, 10))
corr = df_encoded.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGn',
            linewidths=0.5, annot_kws={'size': 8})
plt.title('Heatmap Korelasi Antar Fitur', fontsize=14)
plt.tight_layout()
plt.savefig('results/6_heatmap_korelasi.png', bbox_inches='tight', dpi=300)
plt.close('all')

# ── MODELING UNTUK MENDAPATKAN PLOT EVALUASI ──
print("Melatih model...")
X = df_encoded.drop(columns=['diabetes'])
y = df_encoded['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_sc, y_train)

y_pred = log_reg.predict(X_test_sc)
y_pred_prob = log_reg.predict_proba(X_test_sc)[:, 1]

# 7. Confusion Matrix
print("7. Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix — Logistic Regression')
plt.tight_layout()
plt.savefig('results/7_confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close('all')

# 8. ROC Curve
print("8. ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/8_roc_curve.png', bbox_inches='tight', dpi=300)
plt.close('all')

# 9. Feature Importance
print("9. Feature Importance")
coef_df = pd.DataFrame({
    'Fitur': X.columns,
    'Koefisien': log_reg.coef_[0]
}).sort_values('Koefisien', ascending=False)
plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['Koefisien']]
plt.barh(coef_df['Fitur'], coef_df['Koefisien'], color=colors)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Nilai Koefisien')
plt.title('Feature Importance — Koefisien Logistic Regression\n(Merah = meningkatkan risiko diabetes, Biru = menurunkan)')
plt.tight_layout()
plt.savefig('results/9_feature_importance.png', bbox_inches='tight', dpi=300)
plt.close('all')

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_dict = {}
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    metrics_dict[metric] = cross_val_score(pipeline, X, y, cv=skf, scoring=metric)

# 10. Cross Validation
print("10. Cross Validation")
cv_df = pd.DataFrame(metrics_dict)
fig, ax = plt.subplots(figsize=(10, 5))
cv_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', width=0.7)
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)], rotation=0)
ax.set_ylabel('Score')
ax.set_ylim(0.7, 1.05)
ax.set_title('Hasil 5-Fold Cross-Validation per Metrik')
ax.legend(loc='lower right')
ax.axhline(y=cv_df.mean().mean(), color='red', linestyle='--', linewidth=1, label=f'Rata-rata keseluruhan')
plt.tight_layout()
plt.savefig('results/10_cross_validation.png', bbox_inches='tight', dpi=300)
plt.close('all')

print("Generating images complete!")
