# ===============================================================
# 💰 Aurigo Hackathon 2025 — ROI_Realized Inference Script
# ===============================================================

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
import os

# ===============================================================
# 1️⃣ Load Model and Encoders
# ===============================================================

# ✅ Corrected base directory (no nested folder)
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "realized_ROI_model.json")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

print("⏳ Loading ROI_Realized model and encoders...")

model = XGBRegressor()
model.load_model(MODEL_PATH)

encoders = joblib.load(ENCODERS_PATH)
expected_columns = joblib.load(COLUMNS_PATH)

print("✅ Model and encoders loaded successfully.")


# ===============================================================
# 2️⃣ Preprocessing Function
# ===============================================================

def preprocess_roi_input(user_input: dict) -> pd.DataFrame:
    """Prepare new project data for ROI prediction."""
    df = pd.DataFrame([user_input])

    # === Feature Engineering (must match training) ===
    df["Project_Duration"] = df["End_Year"] - df["Start_Year"]
    df["Budget_per_Year"] = df["Planned_Budget"] / np.maximum(df["Project_Duration"], 1)
    df["Schedule_Risk_Sq"] = df["Schedule_Risk"] ** 2
    df["Cost_Risk_Sq"] = df["Cost_Risk"] ** 2
    df["Funding_Delay_Sq"] = df["Funding_Delay_%"] ** 2
    df["Region_Sector_Interaction"] = df["Region"].astype(str) + "_" + df["Sector"].astype(str)

    # === Encode categorical features ===
    for col in df.select_dtypes(include="object").columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
        else:
            print(f"⚠️ Warning: Unknown category in '{col}' — must match training data values.")

    # === Ensure consistent feature order ===
    for col in expected_columns:
        if col not in df:
            df[col] = 0  # add missing columns

    df = df[expected_columns]
    return df


# ===============================================================
# 3️⃣ Prediction Function
# ===============================================================

def predict_roi(user_input: dict) -> float:
    """Predict ROI for a given project configuration."""
    X_input = preprocess_roi_input(user_input)
    prediction = model.predict(X_input)[0]
    return round(float(prediction), 4)
