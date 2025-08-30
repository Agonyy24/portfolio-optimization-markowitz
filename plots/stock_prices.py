import sys
import os

plots_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(plots_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[INFO] Added project root to sys.path: {project_root}")

import pandas as pd
import matplotlib.pyplot as plt

data_folder = os.path.abspath(os.path.join(plots_dir, "..", "data"))

from src.load_data import tickers_list as tickers

plt.figure(figsize=(10, 6))

for ticker in tickers:
    file_path = os.path.join(data_folder, f"{ticker}.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], encoding="utf-8")
    
    if "Open" in df.columns and not df.empty:
        plt.plot(df["Date"], df["Open"], label=ticker)
    else:
        print(f"[WARNING] {ticker}.csv is empty or missing 'Open' column")

plt.title("Stock Open Prices")
plt.xlabel("Date")
plt.ylabel("Open Price (USD)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(plots_dir, "open_prices.png"), dpi=300)
plt.show()
