# explainability.py
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def explain_forecast(model, X_instance, feature_names=None, plot_title="SHAP Explanation"):
    """
    model: trained sklearn-like model
    X_instance: single-row DataFrame with features
    Returns shap_values and a matplotlib figure object.
    """
    explainer = shap.Explainer(model, X_instance)  # uses model.predict
    shap_values = explainer(X_instance)
    # create a local force plot or bar plot
    fig = shap.plots.bar(shap_values, show=False)
    # shap.plots may return an embedded plot; to keep it simple, save shap values as dict
    summary = dict(zip(X_instance.columns, shap_values.values[0]))
    return shap_values, summary
