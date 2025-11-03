# simulation.py
import numpy as np
import pandas as pd

def compute_kpis(cashflow_df, discount_rate=0.08):
    """
    cashflow_df: DataFrame with columns: year, capex, opex, revenue
    Returns NPV, IRR, payback (years)
    """
    # simple yearly cashflow: revenue - opex - capex
    cf = (cashflow_df['revenue'] - cashflow_df['opex'] - cashflow_df['capex']).values
    years = np.arange(len(cf))
    # compute NPV
    npv = sum(cf / ((1 + discount_rate) ** (years + 1)))  # discount from year 1
    # IRR using numpy
    try:
        irr = np.irr(np.concatenate(([-cashflow_df['capex'].iloc[0]], cf)))  # rough
    except Exception:
        irr = np.nan
    # payback: cumulative
    cum = np.cumsum(cf)
    payback_idx = np.where(cum >= 0)[0]
    payback = int(payback_idx[0]) + 1 if len(payback_idx) > 0 else np.inf
    return {'npv': float(npv), 'irr': float(irr) if not np.isnan(irr) else None, 'payback_years': payback}

def monte_carlo(base_df, model, scenario, discount_rate=0.08, n_sims=200):
    """
    Run Monte Carlo sampling over uncertain parameters (demand_growth_sd, revenue_noise).
    scenario: dict contains deterministic modifiers (capex_mult, opex_mult, demand_growth)
    """
    results = []
    for i in range(n_sims):
        # sample uncertain demand growth and revenue noise
        dg = np.random.normal(scenario.get('demand_growth', 0.03), scenario.get('demand_growth_sd', 0.01))
        revenue_noise = np.random.normal(0, scenario.get('revenue_sd', 50_000))
        sc = scenario.copy()
        sc['demand_growth'] = float(dg)
        preds = model.forecast(model=model, base_df=base_df, horizon=scenario.get('horizon',5), scenario_modifiers=sc)
        # add noise
        preds['revenue'] = preds['revenue'] + revenue_noise
        # merge base_df tail + preds for cashflow
        # for simplicity, create a cashflow_df from preds only
        cashflow = preds[['year','capex','opex','revenue']].copy()
        kpis = compute_kpis(cashflow, discount_rate=discount_rate)
        results.append(kpis)
    return pd.DataFrame(results)
