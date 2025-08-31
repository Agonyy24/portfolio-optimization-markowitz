import sys
import os

results_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(results_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[INFO] Added project root to sys.path: {project_root}")

import numpy as np
import matplotlib.pyplot as plt
from src.w_opti_scipy import optimize_portfolios      # Scipy method
from src.w_opti_matrix import optimize_portfolios_matrix    # Matrix method

results_scipy = optimize_portfolios(risk_free=0.05/252,bounds_type=(-100,100)) #0.05/252
results_matrix = optimize_portfolios_matrix(risk_free=0.05/252,target_return=0.0005)

tickers = list(results_scipy["MVP"]["weights"].keys())

# MVP weights
mvp_scipy = np.array(list(results_scipy["MVP"]["weights"].values()))
mvp_matrix = np.array(list(results_matrix["MVP"]["weights"].values()))

# TP weights
tp_scipy = np.array(list(results_scipy["Tangency"]["weights"].values()))
tp_matrix = np.array(list(results_matrix["Tangency"]["weights"].values()))

# Compare weights
print("\n=== MVP comparison ===")
for ticker, w_s, w_m in zip(tickers, mvp_scipy, mvp_matrix):
    print(f"{ticker}: SciPy={w_s:.2f}%, Matrix={w_m:.2f}%, Δ={(w_s - w_m):.4f}%")

print(f"Return diff: {results_scipy['MVP']['return'] - results_matrix['MVP']['return']:.6f}")
print(f"Vol diff: {results_scipy['MVP']['volatility'] - results_matrix['MVP']['volatility']:.6f}")

print("\n=== Tangency comparison ===")
for ticker, w_s, w_m in zip(tickers, tp_scipy, tp_matrix):
    print(f"{ticker}: SciPy={w_s:.2f}%, Matrix={w_m:.2f}%, Δ={(w_s - w_m):.4f}%")

print(f"Return diff: {results_scipy['Tangency']['return'] - results_matrix['Tangency']['return']:.6f}")
print(f"Vol diff: {results_scipy['Tangency']['volatility'] - results_matrix['Tangency']['volatility']:.6f}")

# Create plot
x = np.arange(len(tickers))
width = 0.35

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# MVP Weights
ax[0].bar(x - width/2, mvp_scipy, width, label='SciPy')
ax[0].bar(x + width/2, mvp_matrix, width, label='Matrix')
ax[0].set_xticks(x)
ax[0].set_xticklabels(tickers)
ax[0].set_title("MVP Weights Comparison")
ax[0].set_ylabel("Weight (%)")
ax[0].legend()
ax[0].grid(axis="y", linestyle="--", alpha=0.6)

# Tangency Weights
ax[1].bar(x - width/2, tp_scipy, width, label='SciPy')
ax[1].bar(x + width/2, tp_matrix, width, label='Matrix')
ax[1].set_xticks(x)
ax[1].set_xticklabels(tickers)
ax[1].set_title("Tangency Portfolio Weights Comparison")
ax[1].set_ylabel("Weight (%)")
ax[1].legend()
ax[1].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, "method_comparison.png"), dpi=300)
plt.show()
