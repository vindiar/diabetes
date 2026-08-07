"""
utils/data.py
─────────────
Handles loading and basic cleaning of the diabetes dataset.
"""

import pandas as pd
import streamlit as st

DATASET_PATH = "data/balanced_dataset.csv"
RAW_DATASET_PATH = "data/dataset_diabetes.csv"

@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    """Load original raw dataset before balancing."""
    df_raw = pd.read_csv(RAW_DATASET_PATH)
    return df_raw

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load raw dataset from CSV (cached) and perform basic cleaning."""
    df = pd.read_csv(DATASET_PATH)
    
    # Hanya gunakan gender Male dan Female sesuai notebook
    df = df[df['gender'].isin(['Male', 'Female'])].copy()
    df.reset_index(drop=True, inplace=True)
    
    # Outlier Removal (Sesuai notebook)
    num_cols = ['HbA1c_level', 'bmi', 'blood_glucose_level']
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    mask = pd.Series([True] * len(df), index=df.index)
    for col in num_cols:
        lower = Q1[col] - 1.5 * IQR[col]
        upper = Q3[col] + 1.5 * IQR[col]
        mask = mask & (df[col] >= lower) & (df[col] <= upper)

    df_clean = df.loc[mask].copy()
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean
