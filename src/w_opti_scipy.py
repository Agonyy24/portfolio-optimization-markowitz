import numpy as np
from scipy.optimize import minimize
from src.stock_stat import stock_stat

def optimize_portfolios(risk_free=0.0, bounds_type=(-1, 1)):

    """
    Optimize portfolios using Markowitz theory:
    - Minimum Variance Portfolio (MVP)
    - Tangency Portfolio (Max Sharpe Ratio)

    Returns:
        dict: {
            'MVP': {'weights': dict, 'return': float, 'volatility': float},
            'Tangency': {'weights': dict, 'return': float, 'volatility': float}
        }
    """
    
    # Get data
    returns_df, mean_returns, cov_matrix = stock_stat()
    tickers = returns_df.columns
    n = len(tickers)

    # Helper functions
    def portfolio_return(weights, mean_returns):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(weights.T @ cov_matrix.values @ weights)

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free=0.0):
        ret = portfolio_return(weights, mean_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        return -(ret - risk_free) / vol # Negative for we need to maximize the sharp ratio

    # Constraints & bounds
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) # Sum of stock weights = 1
    bounds = tuple(bounds_type for _ in range(n)) # Boundaries of weights
    initial_weights = np.ones(n) / n

    # Minimum Variance Portfolio
    min_var = minimize(portfolio_volatility,
                       initial_weights,
                       args=(cov_matrix,),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=constraints)

    w_min_var = min_var.x
    ret_mvp = portfolio_return(w_min_var, mean_returns)
    vol_mvp = portfolio_volatility(w_min_var, cov_matrix)

    # Tangency Portfolio (Max Sharpe Ratio)
    max_sharpe = minimize(neg_sharpe_ratio,
                          initial_weights,
                          args=(mean_returns, cov_matrix, risk_free),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

    w_max_sharpe = max_sharpe.x
    ret_tp = portfolio_return(w_max_sharpe, mean_returns)
    vol_tp = portfolio_volatility(w_max_sharpe, cov_matrix)

    return {
        "MVP": {
            "weights": {ticker: w * 100 for ticker, w in zip(tickers, w_min_var)},
            "return": ret_mvp,
            "volatility": vol_mvp
        },
        "Tangency": {
            "weights": {ticker: w * 100 for ticker, w in zip(tickers, w_max_sharpe)},
            "return": ret_tp,
            "volatility": vol_tp
        }
    }

# Example usage:
# results = optimize_portfolios(risk_free=0.05/252)
# print(results)

if __name__ == "__main__":
    optimize_portfolios()
