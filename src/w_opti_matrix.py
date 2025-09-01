import numpy as np
from src.stock_stat import stock_stat

def optimize_portfolios_matrix(risk_free=0.0,target_return=0.0005):
    """
    Matrix-form Markowitz:
      - MVP
      - Target-return portfolio
      - Tangency portfolio for given risk_free
    All inputs (mu, Sigma, risk_free) must be in the SAME units (daily or annual).
    """

    returns_df, mean_returns, cov_matrix = stock_stat()
    tickers = returns_df.columns

    def port_ret(w, mu): return np.dot(w, mu)
    def port_vol(w, Sigma): return np.sqrt(w @ Sigma.values @ w)

    mu = mean_returns   
    Sigma = cov_matrix
    ones = np.ones(len(mu))
    Sigma_inv = np.linalg.inv(Sigma.values if hasattr(Sigma, "values") else Sigma)

    # Markowitz constants
    A = ones @ Sigma_inv @ ones
    B = ones @ Sigma_inv @ mu
    C = mu   @ Sigma_inv @ mu
    D = A*C - B**2

    # MVP
    w_mvp = (Sigma_inv @ ones) / A
    ret_mvp = port_ret(w_mvp, mu)
    vol_mvp = np.sqrt(w_mvp @ (Sigma.values if hasattr(Sigma, "values") else Sigma) @ w_mvp)

    # Target return
    lam1 = (C - B*target_return)/D
    lam2 = (A*target_return - B)/D
    w_target = Sigma_inv @ (lam1*ones + lam2*mu)
    ret_target = port_ret(w_target, mu)
    vol_target = port_vol(w_target, Sigma)

    # Tangency (including risk-free)
    mu_ex = mu - risk_free*ones
    denom = ones @ Sigma_inv @ mu_ex
    w_tan = (Sigma_inv @ mu_ex) / denom
    ret_tan = port_ret(w_tan, mu)
    vol_tan = port_vol(w_tan, Sigma)

    return {
        "MVP":       {"weights": {t: float(w*100) for t,w in zip(tickers, w_mvp)},    "return": float(ret_mvp),   "volatility": float(vol_mvp)},
        "Target":    {"weights": {t: float(w*100) for t,w in zip(tickers, w_target)}, "return": float(ret_target),"volatility": float(vol_target)},
        "Tangency":  {"weights": {t: float(w*100) for t,w in zip(tickers, w_tan)},    "return": float(ret_tan),   "volatility": float(vol_tan)},
        "Constants": {"A": float(A), "B": float(B), "C": float(C), "D": float(D)}
    }

if __name__ == "__main__":
    optimize_portfolios_matrix()
