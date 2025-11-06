# ================================================================
# 💡 Aurigo Hackathon 2025 — Inference Script
# File: predict_actual_cost.py
# Purpose: Streamlit integration for Actual_Cost prediction
# ================================================================

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# ================================================================
# 1️⃣ Load Model and Encoders
# ================================================================
MODEL_PATH = "actual_cost_model.json"
ENCODERS_PATH = "encoders.pkl"
COLUMNS_PATH = "model_columns.pkl"

print("🔄 Loading model and encoders...")
model = XGBRegressor()
model.load_model(MODEL_PATH)

encoders = joblib.load(ENCODERS_PATH)

try:
    expected_columns = joblib.load(COLUMNS_PATH)
except FileNotFoundError:
    expected_columns = None

print("✅ Model and encoders loaded successfully.")


# ================================================================
# 2️⃣ Helper Functions — Derived Feature Computation
# ================================================================
def compute_risks(df):
    """Compute project risks deterministically."""
    duration = df["End_Year"] - df["Start_Year"]

    df["Schedule_Risk"] = (
        35
        + (duration * 4.2)
        + (df["Funding_Delay_%"] * 0.9)
        - (df["Feasibility_Score"] * 0.35)
        - (df["Sustainability_Score"] * 0.25)
    ).clip(25, 85)

    df["Cost_Risk"] = (
        40
        + (df["Inflation_Rate"] * 1.8)
        + (df["Funding_Delay_%"] * 0.6)
        - (df["Feasibility_Score"] * 0.3)
        - (df["Sustainability_Score"] * 0.25)
    ).clip(30, 80)

    df["Environmental_Risk"] = (
        50
        - (df["Sustainability_Score"] * 0.3)
        + np.where(df["Sector"].astype(str).str.contains("Energy|Industrial", case=False), 10, 0)
    ).clip(25, 80)

    df["Policy_Risk"] = (
        40
        - (df["Public_Benefit_Score"] * 0.25)
        + np.where(df["Region"].astype(str).str.contains("North|East", case=False), 8, 0)
    ).clip(20, 75)

    return df


def compute_funding_features(df):
    """Compute funding ratios and interactions deterministically."""
    df["Funding_Utilization_Ratio"] = (
        0.9
        + 0.0008 * df["Feasibility_Score"]
        + 0.0005 * df["Sustainability_Score"]
        - 0.0010 * df["Schedule_Risk"]
    ).clip(0.6, 1.1)

    df["Funding_Efficiency"] = (
        0.85
        + 0.0006 * df["Feasibility_Score"]
        - 0.0008 * df["Cost_Risk"]
        - 0.0005 * df["Inflation_Index"] * 100
    ).clip(0.6, 1.05)

    df["Funding_Risk_Interaction"] = df["Funding_Utilization_Ratio"] * (100 - df["Schedule_Risk"]) / 100
    df["Funding_Inflation_Interaction"] = df["Funding_Efficiency"] * df["Inflation_Index"]
    return df


# ================================================================
# 3️⃣ Prediction Function
# ================================================================
def predict_actual_cost(input_data: dict):
    """Predict Actual_Cost for a new project based on user input."""
    df = pd.DataFrame([input_data])

    # Step 1 — Compute derived fields
    df = compute_risks(df)
    df = compute_funding_features(df)

    # Step 2 — Encode categorical fields
    for col, le in encoders.items():
        if isinstance(df[col].iloc[0], list):
            df[col] = df[col].apply(lambda x: x[0])
        df[col] = le.transform(df[col])

    # Step 3 — Align with training columns
    if expected_columns:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]

    # Step 4 — Predict
    predicted_cost = model.predict(df)[0]
    return predicted_cost