"""
utils/model.py
──────────────
Handles loading of pre-trained .pkl artifacts and provides
functions for evaluation and single row inference based on one-hot encoding.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model_diabetes.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_diabetes.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features_diabetes.pkl")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, scaler, and features list from disk."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


@st.cache_data(show_spinner=False)
def evaluate_model(df, _model, _scaler, features):
    """
    Reconstruct the validation set to generate evaluation metrics
    based on the one-hot encoded structure.
    """
    df_clean = df.copy()
    
    # One-hot encode exactly as in notebook
    df_clean = pd.get_dummies(df_clean, columns=['gender', 'smoking_history'], drop_first=False)
    
    # Ensure all required features exist
    for feature in features:
        if feature not in df_clean.columns:
            df_clean[feature] = 0
            
    X = df_clean[features]
    y = df_clean['diabetes']

    # Must match notebook split exactly to reproduce correct test set metrics
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_sc = _scaler.transform(X_test)
    y_pred = _model.predict(X_test_sc)
    y_prob = _model.predict_proba(X_test_sc)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    eval_results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Tidak Diabetes", "Diabetes"]
        ),
    }
    
    return eval_results


def predict_single(model, scaler, features, input_dict: dict) -> dict:
    """
    Run inference on a single patient record.
    The input_dict should exactly match the one-hot expected mapping.
    """
    # Create input 1-row DataFrame directly from dict, selecting only needed features
    input_data = pd.DataFrame([input_dict], columns=features).fillna(0)
    
    X_scaled = scaler.transform(input_data)
    prediction = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]

    return {
        "prediction": prediction,
        "prob_diabetes": float(proba[1]) * 100,
        "prob_no_diabetes": float(proba[0]) * 100,
    }
