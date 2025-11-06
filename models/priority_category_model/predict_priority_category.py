# ===============================================================
# 🎯 Aurigo Hackathon 2025 — Priority_Category Inference Script
# Purpose: Local or Streamlit integration for Project Priority Classification
# ===============================================================

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
import os

# ===============================================================
# 1️⃣ Load Model and Encoders
# ===============================================================

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "priority_category_model.json")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
TARGET_ENCODER_PATH = os.path.join(BASE_DIR, "target_encoder.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

print("⏳ Loading Priority_Category model and encoders...")

model = XGBClassifier()
model.load_model(MODEL_PATH)

encoders = joblib.load(ENCODERS_PATH)
target_encoder = joblib.load(TARGET_ENCODER_PATH)
expected_columns = joblib.load(COLUMNS_PATH)

print("✅ Model and encoders loaded successfully.")


# ===============================================================
# 2️⃣ Preprocessing Function
# ===============================================================

def preprocess_priority_input(user_input: dict) -> pd.DataFrame:
    """Prepare user input for model prediction."""
    df = pd.DataFrame([user_input])

    # === Feature Engineering (same as training) ===
    df["Project_Duration"] = df["End_Year"] - df["Start_Year"]
    df["Budget_per_Year"] = df["Planned_Budget"] / np.maximum(df["Project_Duration"], 1)
    df["Schedule_Risk_Sq"] = df["Schedule_Risk"] ** 2
    df["Cost_Risk_Sq"] = df["Cost_Risk"] ** 2
    df["Funding_Delay_Sq"] = df["Funding_Delay_%"] ** 2
    df["Region_Sector_Interaction"] = df["Region"].astype(str) + "_" + df["Sector"].astype(str)

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
        else:
            print(f"⚠️ Warning: Unknown category in '{col}' — must match training data values.")

    # Ensure consistent columns
    for col in expected_columns:
        if col not in df:
            df[col] = 0
    df = df[expected_columns]
    return df


# ===============================================================
# 3️⃣ Prediction Function
# ===============================================================

def predict_priority(user_input: dict):
    """Predict project priority (High / Medium / Low)."""
    X_input = preprocess_priority_input(user_input)
    class_probs = model.predict_proba(X_input)[0]
    pred_class = np.argmax(class_probs)
    pred_label = target_encoder.inverse_transform([pred_class])[0]
    confidence = round(float(class_probs[pred_class]), 2)
    return pred_label, confidence
