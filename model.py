# model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

MODEL_PATH = "forecast_model.joblib"

def prepare_features(df):
    # features: lagged revenue, year index, demand_index, capex, opex
    df = df.copy()
    df['year_idx'] = np.arange(len(df))
    df['rev_lag1'] = df['revenue'].shift(1).fillna(method='bfill')
    features = df[['year_idx', 'demand_index', 'capex', 'opex', 'rev_lag1']]
    return features, df['revenue']

def train_model(df):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print("Model trained. Test R2:", model.score(X_test, y_test))
    return model

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        raise FileNotFoundError("Model not found. Run train_model first.")

def forecast(model, base_df, horizon=5, scenario_modifiers=None):
    """
    Forecast future revenue for `horizon` years.
    scenario_modifiers: dict with possible keys to override growth, capex multiplier, opex multiplier, demand_growth
    """
    scenario_modifiers = scenario_modifiers or {}
    last = base_df.iloc[-1:].copy().reset_index(drop=True)
    preds = []
    df = base_df.copy()
    for i in range(horizon):
        year_idx = len(df)
        # create synthetic next row
        next_row = {}
        # demand growth scenario
        demand_growth = scenario_modifiers.get('demand_growth', 0.03)
        next_row['demand_index'] = df.iloc[-1]['demand_index'] * (1 + demand_growth)
        # capex and opex modifiers
        capex_mult = scenario_modifiers.get('capex_mult', 1.0)
        opex_mult = scenario_modifiers.get('opex_mult', 1.0)
        next_row['capex'] = df['capex'].mean() * (1 + 0.05 * i) * capex_mult
        next_row['opex'] = df['opex'].mean() * (1 + 0.02 * i) * opex_mult
        next_row['rev_lag1'] = df.iloc[-1]['revenue']
        X_row = pd.DataFrame([{
            'year_idx': year_idx,
            'demand_index': next_row['demand_index'],
            'capex': next_row['capex'],
            'opex': next_row['opex'],
            'rev_lag1': next_row['rev_lag1']
        }])
        pred = model.predict(X_row)[0]
        next_row['revenue'] = float(pred)
        next_row['year'] = int(df['year'].iloc[-1] + 1)
        df = pd.concat([df, pd.DataFrame([next_row])], ignore_index=True)
        preds.append(next_row)
    return pd.DataFrame(preds)
