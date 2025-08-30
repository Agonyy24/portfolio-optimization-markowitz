'''

Portfolio optimalization using matrix formula

'''

import numpy as np
from stock_stat import stock_stat

# Download data
returns_df, mean_returns, cov_matrix = stock_stat()

# Numpy conversion
mu = mean_returns
Sigma = cov_matrix
ones = np.ones(len(mu))
Sigma_inv = np.linalg.inv(Sigma)

# Markowitz Constants
A = ones.T @ Sigma_inv @ ones
B = ones.T @ Sigma_inv @ mu
C = mu.T @ Sigma_inv @ mu
D = A * C - B ** 2

print(f"Markowitz Constants: A={A:.4f}, B={B:.4f}, C={C:.4f}, D={D:.4f}")

# Minimal Variance Portfolio (MVP)
w_min_var = (Sigma_inv @ ones) / A
w_min_var_prct = w_min_var * 100
print("\nMVP Weights: ")
for ticker, weight in zip(returns_df.columns,w_min_var_prct):
    print(f"{ticker}: {weight:.2f}%")

# Portfolio with given return
target_return = 0.0001
lambda_1 = (C - B * target_return) / D
lambda_2 = (A * target_return - B) / D
w_target = Sigma_inv @ (lambda_1 * ones + lambda_2 * mu)
w_target_prct = w_target * 100
print(f"\nGiven return portfolio weights {target_return}:")
for ticker, weight in zip(returns_df.columns,w_target_prct):
    print(f"{ticker}: {weight:.2f}%")

# Tangency Portfolio
w_tangency = (Sigma_inv @ mu) / B
w_tangency_prct = w_tangency * 100
print("\nTangency portfolio weights:")
for ticker, weight in zip(returns_df.columns,w_tangency_prct):
    print(f"{ticker}: {weight:.2f}%")

