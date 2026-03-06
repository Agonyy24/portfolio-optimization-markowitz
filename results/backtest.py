import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.w_opti_scipy import optimize_portfolios
from src.load_data import tickers_list as tickers
import yfinance as yf
import pandas as pd

results_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(results_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[INFO] Added project root to sys.path: {project_root}")

# Weights
results = optimize_portfolios(risk_free=0.05/252)
w_mvp = np.array(list(results["MVP"]["weights"].values())) / 100
w_tp = np.array(list(results["Tangency"]["weights"].values())) / 100

tickers = list(results["MVP"]["weights"].keys())

# Download prices for backtest
price_frames = []

for ticker in tickers:
    df = yf.download(ticker, start="2024-01-03", end="2025-01-03")
    df.columns = df.columns.get_level_values(0)  # flatten MultiIndex if needed
    df = df[["Open"]]  # select only 'Open'
    df.rename(columns={"Open": ticker}, inplace=True)
    price_frames.append(df)

# Join all tickers by Date index
prices_test_df = pd.concat(price_frames, axis=1)

# Now compute returns
returns_test_df = np.log(prices_test_df / prices_test_df.shift(1)).dropna() # Calculate daily returns, drop NaN

# Compute Returns
mvp_returns = returns_test_df @ w_mvp
tp_returns = returns_test_df @ w_tp

mvp_cum = (mvp_returns + 1).cumprod()
tp_cum = (tp_returns + 1).cumprod()

# Create Plot
plt.figure(figsize=(10, 6))
plt.plot(mvp_cum, label="MVP Portfolio", linewidth=2)
plt.plot(tp_cum, label="Tangency Portfolio", linewidth=2)
plt.title("Backtest: MVP vs Tangency Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.grid(True)

plt.tight_layout()
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, "backtest.png"), dpi=300)
plt.show()
