import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

print('Library berhasil di-import')

df = pd.read_csv('/kaggle/input/datasets/vindiarjohandiputra/balanced-dataset-pred-diabet/balanced_dataset.csv')

# Hanya gunakan gender Male dan Female
df = df[df['gender'].isin(['Male', 'Female'])].copy()
df.reset_index(drop=True, inplace=True)

print(f'Shape data: {df.shape}')
print(f'Distribusi gender: {df["gender"].value_counts().to_dict()}')
df.head()

print('=== Info Dataset ===')
print(df.info())

df.describe()

print('=== Distribusi Label Target ===')
print(df['diabetes'].value_counts())
print(f'\nRasio kelas: {df["diabetes"].value_counts(normalize=True).round(3).to_dict()}')

print('=== Distribusi Fitur Kategorikal ===')
for col in df.select_dtypes(include='object').columns:
    print(f'\n{col}:')
    print(df[col].value_counts())

df.groupby('gender')['diabetes'].value_counts()

print('=== Jumlah Missing Value ===')
print(df.isnull().sum())
msno.matrix(df)
plt.title('Missing Value Matrix')
plt.show()

# FIX: axes[3] sebelumnya salah plot HbA1c lagi — sekarang sudah benar pakai blood_glucose_level
fig, axes = plt.subplots(1, 4, figsize=(14, 5))

sns.boxplot(ax=axes[0], data=df['age'])
axes[0].set_title('Age')

sns.boxplot(ax=axes[1], data=df['HbA1c_level'])
axes[1].set_title('HbA1c Level')

sns.boxplot(ax=axes[2], data=df['bmi'])
axes[2].set_title('BMI')

# ✅ DIPERBAIKI: sebelumnya salah pakai df['HbA1c_level'] lagi
sns.boxplot(ax=axes[3], data=df['blood_glucose_level'])
axes[3].set_title('Blood Glucose Level')

plt.suptitle('Boxplot Sebelum Outlier Removal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

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

print(f'Baris sebelum : {len(df)}')
print(f'Baris setelah : {len(data_clean)}')
print(f'Outlier dihapus: {len(df) - len(data_clean)} ({(len(df)-len(data_clean))/len(df)*100:.1f}%)')

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
plt.show()

sns.set_style('darkgrid')
plt.rcParams['font.size'] = 10

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.countplot(ax=axes[0], x='diabetes', data=data_clean, hue='gender')
axes[0].set_title('Distribusi Gender vs Diabetes')

sns.countplot(ax=axes[1], x='diabetes', data=data_clean, hue='smoking_history')
axes[1].set_title('Riwayat Merokok vs Diabetes')
axes[1].legend(fontsize=7)

sns.countplot(ax=axes[2], x='diabetes', data=data_clean, hue='heart_disease')
axes[2].set_title('Penyakit Jantung vs Diabetes')

plt.tight_layout()
plt.show()

# Distribusi fitur numerik
num_feats = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feat in enumerate(num_feats):
    sns.histplot(data=data_clean, x=feat, hue='diabetes',
                 kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f'Distribusi {feat} per Kelas Diabetes')

plt.tight_layout()
plt.show()

df_encoded = data_clean.copy()

# One-Hot Encoding untuk kolom nominal
df_encoded = pd.get_dummies(df_encoded, columns=['gender', 'smoking_history'], drop_first=True)

print('Kolom setelah encoding:')
print(df_encoded.columns.tolist())
print(f'\nShape: {df_encoded.shape}')
df_encoded.head()

plt.figure(figsize=(14, 10))
corr = df_encoded.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGn',
            linewidths=0.5, annot_kws={'size': 8})
plt.title('Heatmap Korelasi Antar Fitur', fontsize=14)
plt.tight_layout()
plt.show()

# FIX: Gunakan df_encoded (sudah bersih + di-encode), bukan df asli
X = df_encoded.drop(columns=['diabetes'])
y = df_encoded['diabetes']

print(f'Shape X: {X.shape}')
print(f'Shape y: {y.shape}')
print(f'\nFitur yang digunakan: {X.columns.tolist()}')

# FIX: Satu split yang konsisten (80:20) menggantikan 3 split dengan rasio berbeda
# (0.20, 0.30, 0.55) yang tidak memiliki justifikasi metodologis
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f'Data latih : {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.0f}%)')
print(f'Data uji   : {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.0f}%)')

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)  # fit hanya pada data latih
X_test_sc  = sc.transform(X_test)       # transform saja pada data uji
print('Scaling selesai.')

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_sc, y_train)
print('Model berhasil dilatih.')

y_pred      = log_reg.predict(X_test_sc)
y_pred_prob = log_reg.predict_proba(X_test_sc)[:, 1]  # probabilitas kelas positif

acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f} ({acc*100:.2f}%)')

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix — Logistic Regression')
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'True Negative  (TN): {tn}')
print(f'False Positive (FP): {fp}')
print(f'False Negative (FN): {fn}')
print(f'True Positive  (TP): {tp}')

print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# FIX: gunakan y_pred_prob (probabilitas) bukan y_pred (class label)
# Sebelumnya: roc_auc_score(y_test, y_predLR) → kurang akurat
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f'AUC-ROC Score: {roc_auc:.4f}')

coef_df = pd.DataFrame({
    'Fitur': X.columns,
    'Koefisien': log_reg.coef_[0]
}).sort_values('Koefisien', ascending=False)

plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['Koefisien']]
plt.barh(coef_df['Fitur'], coef_df['Koefisien'], color=colors)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Nilai Koefisien')
plt.title('Feature Importance — Koefisien Logistic Regression\n'
          '(Merah = meningkatkan risiko diabetes, Biru = menurunkan)')
plt.tight_layout()
plt.show()

print(coef_df.to_string(index=False))

from sklearn.pipeline import Pipeline

# Pipeline agar scaling tidak bocor ke fold validasi
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

metrics_dict = {}
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=metric)
    metrics_dict[metric] = scores
    print(f'{metric.upper():10s}: {scores.mean():.4f} ± {scores.std():.4f}  '
          f'(fold: {[round(s,4) for s in scores]})')

# Visualisasi hasil cross-validation
cv_df = pd.DataFrame(metrics_dict)

fig, ax = plt.subplots(figsize=(10, 5))
cv_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', width=0.7)
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)], rotation=0)
ax.set_ylabel('Score')
ax.set_ylim(0.7, 1.05)
ax.set_title('Hasil 5-Fold Cross-Validation per Metrik')
ax.legend(loc='lower right')
ax.axhline(y=cv_df.mean().mean(), color='red', linestyle='--', linewidth=1,
           label=f'Rata-rata keseluruhan')
plt.tight_layout()
plt.show()

print('=' * 55)
print('       RINGKASAN EVALUASI MODEL LOGISTIC REGRESSION')
print('=' * 55)
print(f'  Dataset (setelah cleaning)  : {len(data_clean)} sampel')
print(f'  Jumlah fitur                : {X.shape[1]}')
print(f'  Split (train/test)          : 80% / 20%')
print('-' * 55)
print(f'  Accuracy  (test set)        : {accuracy_score(y_test, y_pred):.4f}')
print(f'  AUC-ROC   (test set)        : {roc_auc_score(y_test, y_pred_prob):.4f}')
print('-' * 55)
print('  5-Fold Cross-Validation:')
for metric, scores in metrics_dict.items():
    print(f'    {metric.upper():10s}: {scores.mean():.4f} ± {scores.std():.4f}')
print('=' * 55)

import joblib

# Simpan model dan scaler ke file
joblib.dump(log_reg, 'model_diabetes.pkl')
joblib.dump(sc, 'scaler_diabetes.pkl')
joblib.dump(list(X.columns), 'features_diabetes.pkl')  # simpan nama kolom

print('✅ Model    → model_diabetes.pkl')
print('✅ Scaler   → scaler_diabetes.pkl')
print('✅ Fitur    → features_diabetes.pkl')

import joblib
import pandas as pd

# Load model, scaler, dan nama fitur
model    = joblib.load('model_diabetes.pkl')
scaler   = joblib.load('scaler_diabetes.pkl')
features = joblib.load('features_diabetes.pkl')

print(f'Model berhasil di-load: {type(model).__name__}')
print(f'Fitur yang dibutuhkan ({len(features)}): {features}')

# ── Contoh klasifikasi 1 pasien baru ──────────────────────────────────

pasien_baru = {
    'age'                         : 55,
    'hypertension'                 : 1,
    'heart_disease'                : 0,
    'bmi'                          : 29.4,
    'HbA1c_level'                  : 7.2,
    'blood_glucose_level'          : 155,
    'gender_Male'                  : 1,   # 1 = laki-laki, 0 = perempuan
    'smoking_history_ever'         : 0,
    'smoking_history_former'       : 0,
    'smoking_history_never'        : 1,
    'smoking_history_not current'  : 0,
    'smoking_history_current'      : 0,
}

df_pasien    = pd.DataFrame([pasien_baru])
df_pasien    = df_pasien.reindex(columns=features, fill_value=0)
df_pasien_sc = scaler.transform(df_pasien)

klasifikasi     = model.predict(df_pasien_sc)[0]
probabilitas = model.predict_proba(df_pasien_sc)[0][1]

print('=' * 40)
print('       HASIL klasifikasi PASIEN')
print('=' * 40)
print(f'  klasifikasi       : {"⚠️  DIABETES" if klasifikasi == 1 else "✅ TIDAK DIABETES"}')
print(f'  Probabilitas   : {probabilitas:.2%}')
print(f'  Tingkat risiko : {"Tinggi" if probabilitas >= 0.7 else "Sedang" if probabilitas >= 0.4 else "Rendah"}')
print('=' * 40)


# ── klasifikasi banyak pasien sekaligus (dari DataFrame / CSV) ────────
# Contoh: load dari file CSV
# df_baru = pd.read_csv('data_pasien_baru.csv')
# df_baru_encoded = pd.get_dummies(df_baru, columns=['gender', 'smoking_history'], drop_first=True)
# df_baru_encoded = df_baru_encoded.reindex(columns=features, fill_value=0)
# df_baru_sc = scaler.transform(df_baru_encoded)
# klasifikasi_semua = model.predict(df_baru_sc)
# prob_semua     = model.predict_proba(df_baru_sc)[:, 1]
# df_baru['klasifikasi']     = klasifikasi_semua
# df_baru['probabilitas'] = prob_semua
# df_baru.head()

print('Uncomment kode di atas dan sesuaikan nama file CSV untuk klasifikasi batch.')