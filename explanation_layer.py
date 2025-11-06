#!/usr/bin/env python3
"""
explanation_layer.py

Unified explanation layer that supports XGBoost JSON models (regressor/classifier) and joblib pickles.
It loads encoders and expected model columns when given, computes derived features exactly like training,
applies encodings, aligns columns, runs model.predict / predict_proba, and produces SHAP explanations.

Usage (example):
  python explanation_layer.py --model models/actual_cost_model/actual_cost_model.json \
      --encoders models/actual_cost_model/encoders.pkl \
      --columns models/actual_cost_model/model_columns.pkl \
      --input-json '{"Sector":"Energy","Region":"South", "Start_Year":2025, "End_Year":2029, "Planned_Budget":10000000, "Funding_Approved":10000000, "Funding_Received":8000000, "Inflation_Rate":0.05, "Inflation_Index":1.05, "Feasibility_Score":80, "Sustainability_Score":75, "Public_Benefit_Score":70, "Stakeholder_Priority":1, "Funding_Delay_%":5.0}'
"""

from __future__ import annotations
import os
import json
import argparse
import logging
from typing import Optional, Tuple, Dict, Any, List

import joblib
import pandas as pd
import numpy as np
import time

# try to import xgboost wrappers
try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:
    XGBRegressor = None
    XGBClassifier = None

# SHAP and plotting
try:
    import shap
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("Please install shap and matplotlib: pip install shap matplotlib") from e

# Logging
logger = logging.getLogger("explanation_layer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)


# -------------------------
# Loaders
# -------------------------
def load_model(model_path: str):
    """
    Load model from XGBoost JSON (try regressor then classifier) or joblib pickle.
    Sets attribute `_is_classifier` on the returned model for downstream logic.
    """
    logger.info("Loading model from %s", model_path)
    lower = str(model_path).lower()
    if lower.endswith((".json", ".model")) and (XGBRegressor is not None or XGBClassifier is not None):
        # Try regressor first
        if XGBRegressor is not None:
            try:
                m = XGBRegressor()
                m.load_model(model_path)
                setattr(m, "_is_classifier", False)
                logger.info("Loaded XGBoost regressor via XGBRegressor.load_model()")
                return m
            except TypeError as te:
                logger.info("XGBRegressor.load_model: type mismatch (%s) - trying classifier", te)
            except Exception as e:
                logger.info("XGBRegressor.load_model raised: %s", e)

        # Then try classifier
        if XGBClassifier is not None:
            try:
                m = XGBClassifier()
                m.load_model(model_path)
                setattr(m, "_is_classifier", True)
                logger.info("Loaded XGBoost classifier via XGBClassifier.load_model()")
                return m
            except Exception as e:
                logger.exception("XGBClassifier.load_model failed")
                raise e

        raise RuntimeError("Unable to load XGBoost JSON model (no suitable wrapper).")

    # Fallback: joblib/pickle
    logger.info("Attempting to load via joblib")
    m = joblib.load(model_path)
    # Try to infer whether it's classifier-like
    tname = type(m).__name__.lower()
    is_clf = "classifier" in tname or getattr(m, "predict_proba", None) is not None and getattr(m, "predict", None) is not None and hasattr(m, "classes_")
    setattr(m, "_is_classifier", bool(is_clf))
    logger.info("Loaded model via joblib; inferred _is_classifier=%s", bool(is_clf))
    return m


def load_encoders(encoders_path: Optional[str]):
    if not encoders_path:
        logger.info("No encoders provided")
        return None
    logger.info("Loading encoders from %s", encoders_path)
    enc = joblib.load(encoders_path)
    logger.info("Encoders loaded.")
    return enc


def load_expected_columns(columns_path: Optional[str]):
    if not columns_path:
        logger.info("No model_columns provided")
        return None
    logger.info("Loading expected columns from %s", columns_path)
    cols = joblib.load(columns_path)
    logger.info("Loaded %d expected columns.", len(cols))
    return cols
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV dataset used as background / examples.
    Returns a pandas DataFrame. Raises FileNotFoundError if the path doesn't exist.
    """
    if not data_path:
        raise ValueError("data_path is required by load_data()")

    logger.info("Loading data from: %s", data_path)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.exception("Data file not found: %s", data_path)
        raise
    except Exception as e:
        logger.exception("Failed to read CSV %s: %s", data_path, e)
        raise

    logger.info("Data loaded: %d rows, %d cols.", df.shape[0], df.shape[1])
    return df


# -------------------------
# Derived feature logic (copied from your predict_actual_cost)
# -------------------------
def compute_risks(df: pd.DataFrame) -> pd.DataFrame:
    # ensure Start_Year/End_Year exist
    if "Start_Year" not in df.columns:
        df["Start_Year"] = 2025
    if "End_Year" not in df.columns:
        df["End_Year"] = df["Start_Year"] + 4

    # ensure numeric
    for c in ["Start_Year", "End_Year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(2025).astype(float)

    df["Project_Duration"] = (df["End_Year"] - df["Start_Year"]).fillna(5).astype(float)

    df["Schedule_Risk"] = (
        35
        + (df["Project_Duration"] * 4.2)
        + (df.get("Funding_Delay_%", 0) * 0.9)
        - (df.get("Feasibility_Score", 50) * 0.35)
        - (df.get("Sustainability_Score", 50) * 0.25)
    )
    df["Schedule_Risk"] = df["Schedule_Risk"].clip(25, 85)

    df["Cost_Risk"] = (
        40
        + (df.get("Inflation_Rate", 0.05) * 1.8)
        + (df.get("Funding_Delay_%", 0) * 0.6)
        - (df.get("Feasibility_Score", 50) * 0.3)
        - (df.get("Sustainability_Score", 50) * 0.25)
    )
    df["Cost_Risk"] = df["Cost_Risk"].clip(30, 80)

    df["Environmental_Risk"] = (
        50
        - (df.get("Sustainability_Score", 50) * 0.3)
        + np.where(df.get("Sector", "").astype(str).str.contains("Energy|Industrial", case=False), 10, 0)
    )
    df["Environmental_Risk"] = df["Environmental_Risk"].clip(25, 80)

    df["Policy_Risk"] = (
        40
        - (df.get("Public_Benefit_Score", 50) * 0.25)
        + np.where(df.get("Region", "").astype(str).str.contains("North|East", case=False), 8, 0)
    )
    df["Policy_Risk"] = df["Policy_Risk"].clip(20, 75)

    return df


def compute_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Funding_Utilization_Ratio"] = (
        0.9
        + 0.0008 * df.get("Feasibility_Score", 50)
        + 0.0005 * df.get("Sustainability_Score", 50)
        - 0.0010 * df.get("Schedule_Risk", 50)
    ).clip(0.6, 1.1)

    df["Funding_Efficiency"] = (
        0.85
        + 0.0006 * df.get("Feasibility_Score", 50)
        - 0.0008 * df.get("Cost_Risk", 50)
        - 0.0005 * df.get("Inflation_Index", 1.0) * 100
    ).clip(0.6, 1.05)

    df["Funding_Risk_Interaction"] = df["Funding_Utilization_Ratio"] * (100 - df["Schedule_Risk"]) / 100
    df["Funding_Inflation_Interaction"] = df["Funding_Efficiency"] * df.get("Inflation_Index", 1.0)
    return df


# -------------------------
# Preprocessing to match training
# -------------------------
def transform_for_model(df: pd.DataFrame,
                        encoders: Optional[Dict[str, Any]] = None,
                        expected_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()

    # compute derived features
    df = compute_risks(df)
    df = compute_funding_features(df)

    # normalize Owner_Agency lists if present
    if "Owner_Agency" in df.columns:
        df["Owner_Agency"] = df["Owner_Agency"].apply(lambda v: v[0] if isinstance(v, (list, tuple)) and v else v)

    # apply encoders
    if encoders:
        for col, encoder in encoders.items():
            # if encoder mapping not present in df create a column
            if col not in df.columns:
                df[col] = ""
            # normalize lists
            df[col] = df[col].apply(lambda v: v[0] if isinstance(v, (list, tuple)) and v else v)
            try:
                df[col] = encoder.transform(df[col])
            except Exception:
                # fallback mapping via classes_ or categories_
                try:
                    if hasattr(encoder, "classes_"):
                        mapping = {c: i for i, c in enumerate(encoder.classes_)}
                        df[col] = df[col].map(mapping).fillna(0).astype(int)
                    elif hasattr(encoder, "categories_"):
                        cats = [list(c) for c in encoder.categories_]
                        mapping = {c: i for i, c in enumerate(cats[0])} if len(cats) > 0 else {}
                        df[col] = df[col].map(mapping).fillna(0).astype(int)
                    else:
                        logger.warning("Encoder for %s could not be applied and has no classes_/categories_", col)
                except Exception:
                    logger.exception("Failed to fallback-encode column %s", col)

    # if expected columns provided, ensure they exist and reorder
    if expected_columns:
        for col in expected_columns:
            if col not in df.columns:
                # create numeric zero default for missing engineered features
                df[col] = 0
        df = df[expected_columns].copy()
        feature_names = list(expected_columns)
    else:
        feature_names = list(df.columns)
        df = df[feature_names]

    # fill NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = df[c].fillna(0)
    nonnum_cols = [c for c in df.columns if c not in numeric_cols]
    for c in nonnum_cols:
        df[c] = df[c].fillna("")

    return df, feature_names


# -------------------------
# SHAP helpers
# -------------------------
def init_shap_explainer(model, background_data: Optional[pd.DataFrame] = None):
    logger.info("Initializing SHAP TreeExplainer...")
    try:
        if getattr(model, "_is_classifier", False):
            # newer shap versions accept model_output param to explain probabilities
            try:
                explainer = shap.TreeExplainer(model, model_output="probability")
            except TypeError:
                explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer ready.")
        return explainer
    except Exception:
        logger.exception("Failed to initialize TreeExplainer, trying booster")
        try:
            explainer = shap.TreeExplainer(model.get_booster())
            return explainer
        except Exception:
            raise


def explain_instance(explainer, instance_df: pd.DataFrame):
    logger.info("Computing SHAP values for the instance...")
    try:
        explanation = explainer(instance_df)
        return explanation
    except Exception:
        try:
            sv = explainer.shap_values(instance_df)
            return sv
        except Exception:
            logger.exception("SHAP explanation failed")
            raise


# -------------------------
# Plotting helpers
# -------------------------
def plot_waterfall(explanation, feature_names: List[str], out_path: str):
    logger.info("Rendering waterfall plot to %s", out_path)
    plt.figure(figsize=(8, 6))
    try:
        shap.plots.waterfall(explanation, show=False)
    except Exception:
        # fallback to horizontal bar of top features
        try:
            arr = np.array(explanation)
            if arr.ndim > 1:
                arr = arr.flatten()
            dfb = pd.DataFrame({"feature": feature_names, "shap": arr})
            dfb["abs_shap"] = dfb["shap"].abs()
            dfb = dfb.sort_values("abs_shap", ascending=False).head(20)
            dfb = dfb.iloc[::-1]
            plt.barh(dfb["feature"], dfb["shap"])
            plt.xlabel("SHAP contribution")
            plt.tight_layout()
        except Exception:
            logger.exception("Fallback waterfall failed")
            raise
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved waterfall plot")


def plot_summary(shap_values_all, X_background, out_path: str):
    logger.info("Rendering summary plot to %s", out_path)
    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_values_all, X_background, show=False)
    except Exception:
        logger.exception("Summary plot failed")
        raise
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved summary plot")


# -------------------------
# JSON conversion
# -------------------------
def _serialize_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer, np.int64, int)):
        return int(v)
    if isinstance(v, (np.floating, np.float64, float)):
        return float(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return str(v)


def explanation_to_dict(explanation, feature_names: List[str], model_prediction: Any, input_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    logger.info("Converting explanation to dict")
    result = {"base_value": None, "prediction": model_prediction, "features": []}

    # If explanation is shap.Explanation
    if hasattr(explanation, "values") and hasattr(explanation, "base_values"):
        vals = np.array(explanation.values)
        # flatten for 1-sample single-class case
        if vals.ndim == 3:
            # (n_samples, n_classes, n_features) -> pick first sample, first class
            vals = vals[0, 0, :]
        elif vals.ndim == 2:
            vals = vals[0, :]
        else:
            vals = vals.reshape(-1)
        try:
            base_raw = explanation.base_values
            base = float(np.array(base_raw).reshape(-1)[0])
        except Exception:
            base = None
        result["base_value"] = base
        for fname, sval in zip(feature_names, vals):
            e = {"feature": fname, "shap_value": float(sval)}
            if input_row is not None and fname in input_row:
                e["input_value"] = _serialize_value(input_row[fname])
            result["features"].append(e)
        return result

    # Fallback: explanation is a raw numpy array (shap_values)
    try:
        arr = np.array(explanation)
        if arr.ndim == 3:
            arr = arr[0, 0, :]
        elif arr.ndim == 2:
            arr = arr[0, :]
        elif arr.ndim == 1:
            arr = arr
        else:
            arr = arr.reshape(-1)
        for fname, sval in zip(feature_names, arr):
            e = {"feature": fname, "shap_value": float(sval)}
            if input_row is not None and fname in input_row:
                e["input_value"] = _serialize_value(input_row[fname])
            result["features"].append(e)
        return result
    except Exception:
        logger.exception("Could not convert shap explanation to dict")
        return result


# -------------------------
# High level API
# -------------------------
def explain_record(model_path: str,
                   encoders_path: Optional[str] = None,
                   columns_path: Optional[str] = None,
                   data_path: Optional[str] = None,
                   index: Optional[int] = None,
                   input_json: Optional[str] = None,
                   output_dir: str = "./explanations",
                   background_sample_size: int = 200,
                   save_plots: bool = True) -> Dict[str, Any]:
    """
    Explain a single record. Returns payload dict and saves plots + explanation.json to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path)
    encoders = load_encoders(encoders_path)
    expected_columns = load_expected_columns(columns_path)

    # load optional background dataset
    df_background = None
    if data_path:
        df_background = load_data(data_path)

    # build instance df
    if input_json:
        inst = json.loads(input_json)
        inst_df = pd.DataFrame([inst])
    elif data_path and index is not None:
        full = load_data(data_path)
        if index < 0 or index >= len(full):
            raise IndexError("index out of range")
        inst_df = full.loc[[index]].copy()
    else:
        raise ValueError("Provide either input_json or both data_path and index")

    # transform input
    transformed_inst_df, feature_names = transform_for_model(inst_df, encoders=encoders, expected_columns=expected_columns)

    # prepare background for shap if available
    background_df_for_shap = None
    if df_background is not None:
        sample = df_background.sample(n=min(background_sample_size, len(df_background)), random_state=42)
        sample_transformed_df, _ = transform_for_model(sample, encoders=encoders, expected_columns=expected_columns)
        background_df_for_shap = sample_transformed_df

    # Prediction
    is_clf = getattr(model, "_is_classifier", False)
    pred_class = None
    pred_proba = None
    pred_value_for_payload = None

    try:
        if is_clf:
            pred_class = model.predict(transformed_inst_df)
            try:
                pred_proba = model.predict_proba(transformed_inst_df)
            except Exception:
                pred_proba = None
            # choose predicted label for payload (numeric)
            pred_value_for_payload = int(pred_class[0]) if hasattr(pred_class, "__len__") else int(pred_class)
            logger.info("Classifier prediction done. class=%s proba-shape=%s", pred_class, None if pred_proba is None else pred_proba.shape)
        else:
            pred = model.predict(transformed_inst_df)
            pred_value_for_payload = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            logger.info("Regressor prediction done. value=%s", pred_value_for_payload)
    except Exception:
        logger.exception("Model prediction failed")
        raise

    # SHAP explanation
    explainer = init_shap_explainer(model, background_data=background_df_for_shap)
    explanation = explain_instance(explainer, transformed_inst_df)

    # extract single per-feature shap array when possible
    shap_array = None
    try:
        if hasattr(explanation, "values"):
            arr = np.array(explanation.values)
            if arr.ndim == 3:
                # multiclass: pick predicted class if available
                if is_clf and pred_proba is not None:
                    class_idx = int(np.argmax(pred_proba, axis=1)[0])
                    shap_array = arr[0, class_idx, :]
                else:
                    shap_array = arr[0, 0, :]
            elif arr.ndim == 2:
                shap_array = arr[0, :]
            else:
                shap_array = arr.reshape(-1)
        else:
            # older API or raw array
            arr2 = np.array(explanation)
            if arr2.ndim == 3:
                if is_clf and pred_proba is not None:
                    class_idx = int(np.argmax(pred_proba, axis=1)[0])
                    shap_array = arr2[0, class_idx, :]
                else:
                    shap_array = arr2[0, 0, :]
            elif arr2.ndim == 2:
                shap_array = arr2[0, :]
            elif arr2.ndim == 1:
                shap_array = arr2
            else:
                shap_array = arr2.reshape(-1)
    except Exception:
        logger.exception("Failed to interpret SHAP explanation object; shap_array=None")
        shap_array = None

    # Save plots
    saved = {}
    if save_plots:
        wf_path = os.path.join(output_dir, "waterfall.png")
        try:
            # prefer to pass Explanation object; plot_waterfall handles raw arrays too
            plot_waterfall(explanation if explanation is not None else shap_array, feature_names, wf_path)
            saved["waterfall"] = wf_path
        except Exception:
            logger.exception("Waterfall plot creation failed")

        if background_df_for_shap is not None:
            summary_path = os.path.join(output_dir, "summary.png")
            try:
                bg_expl = explain_instance(explainer, background_df_for_shap)
                plot_summary(bg_expl, background_df_for_shap, summary_path)
                saved["summary"] = summary_path
            except Exception:
                logger.exception("Summary plot creation failed")

    # Build payload
    input_row_serializable = inst_df.iloc[0].to_dict() if len(inst_df) > 0 else None
    model_pred = pred_value_for_payload

    if shap_array is not None:
        payload = explanation_to_dict(shap_array, feature_names, model_prediction=model_pred, input_row=input_row_serializable)
    else:
        payload = explanation_to_dict(explanation, feature_names, model_prediction=model_pred, input_row=input_row_serializable)

    # Attach classification details if classifier
    if is_clf:
        try:
            payload["predicted_class"] = int(pred_class[0]) if hasattr(pred_class, "__len__") else int(pred_class)
        except Exception:
            payload["predicted_class"] = str(pred_class)
        if pred_proba is not None:
            try:
                payload["prediction_proba"] = pred_proba.tolist()
            except Exception:
                payload["prediction_proba"] = None
    else:
        payload["prediction"] = model_pred

    payload["plots"] = saved

    # Save payload
    payload_path = os.path.join(output_dir, "explanation.json")
    with open(payload_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved explanation payload to %s", payload_path)

    return payload


# -------------------------
# CLI helpers
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Explanation layer for XGBoost models (JSON or joblib)")
    p.add_argument("--model", required=True, help="Path to model file (XGBoost JSON or joblib pickle).")
    p.add_argument("--encoders", help="Path to encoders.pkl (optional).")
    p.add_argument("--columns", help="Path to model_columns.pkl (optional).")
    p.add_argument("--data", help="CSV data file for background sampling (optional).")
    p.add_argument("--index", type=int, help="Row index in CSV to explain (0-based).")
    p.add_argument("--input-json", help="JSON string with raw input values for a single instance.")
    p.add_argument("--output-dir", default="./explanations", help="Directory to write plots and payload.")
    p.add_argument("--background-sample-size", type=int, default=200, help="Sample size for SHAP background.")
    return p.parse_args()


def main():
    args = parse_args()
    payload = explain_record(
        model_path=args.model,
        encoders_path=args.encoders,
        columns_path=args.columns,
        data_path=args.data,
        index=args.index,
        input_json=args.input_json,
        output_dir=args.output_dir,
        background_sample_size=args.background_sample_size,
        save_plots=True
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
