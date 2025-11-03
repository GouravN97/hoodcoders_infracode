# ui_streamlit.py
import streamlit as st
import pandas as pd
import requests
import json
from data_loader import generate_toy_capital_data

st.title("What-If Capital Planning — Explainable AI Demo")

# Local base data preview
df = generate_toy_capital_data()
st.subheader("Historical data (toy)")
st.dataframe(df)

st.sidebar.header("Scenario inputs")
horizon = st.sidebar.slider("Forecast horizon (years)", 1, 10, 5)
capex_mult = st.sidebar.slider("CapEx multiplier", 0.5, 2.0, 1.0, step=0.05)
opex_mult = st.sidebar.slider("OpEx multiplier", 0.5, 2.0, 1.0, step=0.05)
demand_growth = st.sidebar.slider("Annual demand growth", -0.05, 0.20, 0.03, step=0.005)
discount_rate = st.sidebar.slider("Discount rate", 0.01, 0.2, 0.08, step=0.005)

if st.button("Run scenario (local)"):
    # call FastAPI if deployed; here we'll call the model module directly
    import model as model_module
    from api import df_base, mdl
    sc = {'capex_mult': capex_mult, 'opex_mult': opex_mult, 'demand_growth': demand_growth}
    preds = model_module.forecast(model=mdl, base_df=df_base, horizon=horizon, scenario_modifiers=sc)
    st.subheader("Predicted cashflow")
    st.dataframe(preds[['year','revenue','opex','capex']])
    from simulation import compute_kpis
    kpis = compute_kpis(preds[['year','capex','opex','revenue']])
    st.write("KPIs:", kpis)

    st.subheader("Explainability — SHAP feature contributions for next-year revenue")
    from explainability import explain_forecast
    X_next = pd.DataFrame([{
        'year_idx': len(df_base),
        'demand_index': df_base.iloc[-1]['demand_index'] * (1 + demand_growth),
        'capex': df_base.iloc[-1]['capex'] * capex_mult,
        'opex': df_base.iloc[-1]['opex'] * opex_mult,
        'rev_lag1': df_base.iloc[-1]['revenue']
    }])
    shap_vals, summary = explain_forecast(mdl, X_next)
    st.write(summary)
