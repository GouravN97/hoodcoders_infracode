# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd

from data_loader import generate_toy_capital_data
import model as model_module
from model import train_model, load_model, forecast
from explainability import explain_forecast
from simulation import compute_kpis

app = FastAPI(title="Capital Planning What-If API")

# Load or train model on startup
df_base = generate_toy_capital_data(n_years=10)
try:
    mdl = load_model()
except:
    mdl = train_model(df_base)

class ScenarioRequest(BaseModel):
    horizon: int = 5
    capex_mult: float = 1.0
    opex_mult: float = 1.0
    demand_growth: float = 0.03
    discount_rate: float = 0.08

@app.post("/predict_scenario")
def predict_scenario(req: ScenarioRequest):
    sc = {
        'capex_mult': req.capex_mult,
        'opex_mult': req.opex_mult,
        'demand_growth': req.demand_growth
    }
    preds = model_module.forecast(model=mdl, base_df=df_base, horizon=req.horizon, scenario_modifiers=sc)
    # produce cashflow table, compute KPIs
    cashflow = preds[['year','capex','opex','revenue']].copy()
    kpis = compute_kpis(cashflow, discount_rate=req.discount_rate)
    return {"predicted_cashflow": cashflow.to_dict(orient='records'), "kpis": kpis}

@app.post("/explain")
def explain(req: ScenarioRequest):
    sc = {
        'capex_mult': req.capex_mult,
        'opex_mult': req.opex_mult,
        'demand_growth': req.demand_growth
    }
    # build a single input row to explain next-year revenue
    last = df_base.iloc[-1]
    X_next = pd.DataFrame([{
        'year_idx': len(df_base),
        'demand_index': last['demand_index'] * (1 + sc['demand_growth']),
        'capex': last['capex'] * sc['capex_mult'],
        'opex': last['opex'] * sc['opex_mult'],
        'rev_lag1': last['revenue']
    }])
    shap_values, summary = explain_forecast(mdl, X_next)
    # we return numeric summary; for plots you could return image bytes
    return {"shap_summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
