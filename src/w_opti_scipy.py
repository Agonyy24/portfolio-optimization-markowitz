'''
Portfolio optimalization using scipy minimize
'''

import numpy as np
from scipy.optimize import minimize
from stock_stat import stock_stat

returns_df, mean_returns, cov_matrix = stock_stat()

tickers = returns_df.columns
n = len(tickers)

# Helper function
def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix.values @ weights)

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free=0.0):
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return -(ret - risk_free) / vol  # Negative for we need to maximize the sharp ratio

# Setting boundaries
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of stock weights = 1
bounds = tuple((-1, 1) for _ in range(n))  # Boundaries of weights

initial_weights = np.ones(n) / n

# Min. Risk
min_var = minimize(portfolio_volatility,
                   initial_weights,
                   args=(cov_matrix,),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints)

w_min_var = min_var.x

# Max Sharp Ratio
max_sharpe = minimize(neg_sharpe_ratio,
                      initial_weights,
                      args=(mean_returns, cov_matrix, 0.0),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)

w_max_sharpe = max_sharpe.x

# Results
print("\n=== Minimum Variance Portfolio ===")
for ticker, w in zip(tickers, w_min_var):
    print(f"{ticker}: {w*100:.2f}%")
print(f"Portfolio Volatility: {portfolio_volatility(w_min_var, cov_matrix):.4f}")
print(f"Portfolio Return: {portfolio_return(w_min_var, mean_returns):.4f}")

print("\n=== Maximum Sharpe Ratio Portfolio ===")
for ticker, w in zip(tickers, w_max_sharpe):
    print(f"{ticker}: {w*100:.2f}%")
print(f"Portfolio Volatility: {portfolio_volatility(w_max_sharpe, cov_matrix):.4f}")
print(f"Portfolio Return: {portfolio_return(w_max_sharpe, mean_returns):.4f}")
