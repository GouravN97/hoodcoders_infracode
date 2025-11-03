# data_loader.py
import pandas as pd
import numpy as np
from datetime import datetime

def generate_toy_capital_data(n_years=10, seed=42):
    np.random.seed(seed)
    years = pd.date_range(start=datetime.now().year, periods=n_years, freq='Y').year
    # columns: year, demand_index, capex, opex, revenue
    demand_index = 1 + np.cumsum(np.random.normal(0.03, 0.02, n_years))  # baseline growth
    capex = np.clip(1_000_000 * (1 + 0.05 * np.arange(n_years)) + np.random.normal(0, 50_000, n_years), 100_000, None)
    opex = 200_000 * demand_index + np.random.normal(0, 10_000, n_years)
    revenue = 1_200_000 * demand_index + np.random.normal(0, 80_000, n_years)
    df = pd.DataFrame({
        'year': years,
        'demand_index': demand_index,
        'capex': capex,
        'opex': opex,
        'revenue': revenue
    })
    return df

if __name__ == "__main__":
    print(generate_toy_capital_data())
