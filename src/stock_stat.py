'''
Calculation of stock returns and risks
'''

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import tickers_list as tickers # Tickers are defined in load_data.py

base_dir = os.path.dirname(os.path.abspath(__file__))  # Src dir
data_folder = os.path.join(base_dir, "..", "data") # Higher dir

price_data = {} # Create dictionary

def stock_stat():
    for ticker in tickers:
        file_path = os.path.join(data_folder, f"{ticker}.csv")
        df = pd.read_csv(file_path, parse_dates=["Date"], encoding="utf-8")
        
        df = df.sort_values("Date")
        price_data[ticker] = df["Open"].values

    prices_df = pd.DataFrame(price_data)

    returns_df = np.log(prices_df / prices_df.shift(1))

    mean_returns = returns_df.mean()
    risk = returns_df.std()

    cov_matrix = returns_df.cov()
    
    # Debug
    print("Average daily returns:")
    print(mean_returns)
    print("\nRisk (measured by standard dev.):")
    print(risk)
    print("\nCovariance matrix")
    print(cov_matrix)

    return returns_df, mean_returns, cov_matrix
    
if __name__ == "__main__":
    stock_stat()


