import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from stock_stat import stock_stat

returns_df, mean_returns, cov_matrix = stock_stat()

tickers = returns_df.columns
n = len(tickers)

def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix.values @ weights)

# Bounds
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(n))

initial_weights = np.ones(n) / n

# --- Efficient Frontier ---
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
frontier_vol = []

for target in target_returns:
    # Dodajemy ograniczenie na oczekiwany zwrot
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, mean_returns) - target}
    )

    res = minimize(portfolio_volatility,
                   initial_weights,
                   args=(cov_matrix,),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=cons)

    if res.success:
        frontier_vol.append(res.fun)
    else:
        frontier_vol.append(np.nan)

# --- MVP: minimum variance portfolio ---
res_mvp = minimize(portfolio_volatility,
                   initial_weights,
                   args=(cov_matrix,),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=constraints)
w_mvp = res_mvp.x
ret_mvp = portfolio_return(w_mvp, mean_returns.values)
vol_mvp = portfolio_volatility(w_mvp, cov_matrix)

# --- TP: tangency (max Sharpe) ---
def neg_sharpe(weights, mean_returns, cov_matrix, rf=0.0):
    ret = np.dot(weights, mean_returns.values)
    vol = np.sqrt(weights @ cov_matrix.values @ weights)
    return -(ret - rf) / vol

rf = np.float64(0.05/252)  # or 0
res_tp = minimize(neg_sharpe,
                  initial_weights,
                  args=(mean_returns, cov_matrix, rf),
                  method='SLSQP',
                  bounds=bounds,
                  constraints=constraints)
w_tp = res_tp.x
ret_tp = portfolio_return(w_tp, mean_returns.values)
vol_tp = portfolio_volatility(w_tp, cov_matrix)

# Plot creation
plt.figure(figsize=(10, 6))
plt.plot(frontier_vol, target_returns, 'g--', label='Efficient Frontier')

# MVP & TP
plt.scatter(vol_mvp, ret_mvp, s=140, marker='o', label='MVP', zorder=3)
plt.scatter(vol_tp, ret_tp, s=180, marker='*', label='Tangency (Max Sharpe)', zorder=3)

plt.annotate('MVP', xy=(vol_mvp, ret_mvp), xytext=(vol_mvp*1.05, ret_mvp),
             arrowprops=dict(arrowstyle='->', lw=1))
plt.annotate('TP', xy=(vol_tp, ret_tp), xytext=(vol_tp*1.05, ret_tp*1.02),
             arrowprops=dict(arrowstyle='->', lw=1))

plt.title('Efficient Frontier')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.grid(True)
plt.legend()
plt.show()