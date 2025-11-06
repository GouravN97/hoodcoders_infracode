# ===============================================================
# 🧩 Aurigo Hackathon 2025 — Inference Script
# File: predict_delay_months.py
# Purpose: Streamlit integration for Delay_Months prediction
# ===============================================================

import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# ===============================================================
# 1️⃣ Load Model and Encoders (Portable Paths)
# ===============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "delay_months_model.json")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

print("⏳ Loading Delay_Months model and encoders...")

model = XGBRegressor()
model.load_model(MODEL_PATH)

encoders = joblib.load(ENCODERS_PATH)

try:
    expected_columns = joblib.load(COLUMNS_PATH)
except FileNotFoundError:
    expected_columns = None
    print("⚠️ model_columns.pkl not found — continuing without strict column alignment.")

print("✅ Model and encoders loaded successfully.\n")

# ===============================================================
# 2️⃣ Preprocessing Function
# ===============================================================
def preprocess_delay_input(user_input: dict) -> pd.DataFrame:
    """Prepares raw user input for model prediction."""
    df = pd.DataFrame([user_input])

    # --- Deterministic Feature Engineering (same as training) ---
    df["Project_Duration"] = df["End_Year"] - df["Start_Year"]
    df["Budget_per_Year"] = df["Planned_Budget"] / np.maximum(df["Project_Duration"], 1)
    df["Schedule_Risk_Sq"] = df["Schedule_Risk"] ** 2
    df["Funding_Delay_Sq"] = df["Funding_Delay_%"] ** 2
    df["Cost_Risk_Sq"] = df["Cost_Risk"] ** 2
    df["Region_Sector_Interaction"] = df["Region"].astype(str) + "_" + df["Sector"].astype(str)

    # --- Encode categorical variables ---
    for col in df.select_dtypes(include="object").columns:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                print(f"⚠️ Unknown category in '{col}' → defaulting to 0.")
                df[col] = 0
        else:
            print(f"⚠️ Warning: Column '{col}' was not encoded during training.")
            df[col] = 0

    # --- Ensure all expected columns exist ---
    if expected_columns is not None:
        for col in expected_columns:
            if col not in df:
                df[col] = 0
        df = df[expected_columns]

    return df

# ===============================================================
# 3️⃣ Prediction Function
# ===============================================================
def predict_delay(user_input: dict) -> float:
    """Predicts project delay (in months) given input features."""
    X_input = preprocess_delay_input(user_input)
    prediction = model.predict(X_input)[0]
    return round(float(prediction), 2)
