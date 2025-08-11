import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # Src dir
data_folder = os.path.join(base_dir, "..", "data") # Higher dir

tickers = ["AAPL", "JPM", "MSFT", "NDAQ", "NVDA"]

for ticker in tickers:
    file_path = os.path.join(data_folder, f"{ticker}.csv")

    df = pd.read_csv(file_path,parse_dates=["Date"],encoding="utf-8")
    avg_price = df["Open"].mean()
    print(avg_price)