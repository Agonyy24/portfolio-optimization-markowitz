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

# Weights
A = ones.T @ Sigma_inv @ ones
B = ones.T @ Sigma_inv @ mu
C = mu.T @ Sigma_inv @ mu
D = A * C - B ** 2

print(f"A={A:.4f}, B={B:.4f}, C={C:.4f}, D={D:.4f}")

# Minimal Variance Portfolio (MVP)
w_min_var = (Sigma_inv @ ones) / A
print("\nMVP Weights: ")
print(dict(zip(returns_df.columns, w_min_var)))

# Portfolio with given return
target_return = 0.0005
lambda_1 = (C - B * target_return) / D
lambda_2 = (A * target_return - B) / D
w_target = Sigma_inv @ (lambda_1 * ones + lambda_2 * mu)
print(f"\nGiven return portfolio weights {target_return}:")
print(dict(zip(returns_df.columns, w_target)))

# Tangency Portfolio
w_tangency = (Sigma_inv @ mu) / B
print("\nTangency portfolio weights:")
print(dict(zip(returns_df.columns, w_tangency.round(7))))

