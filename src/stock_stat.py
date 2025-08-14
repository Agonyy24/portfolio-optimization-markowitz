import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))  # Src dir
data_folder = os.path.join(base_dir, "..", "data") # Higher dir

tickers = ["AAPL", "JPM", "MSFT", "NDAQ", "NVDA"]

price_data = {} # Create dictionary

for ticker in tickers:
    file_path = os.path.join(data_folder, f"{ticker}.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"], encoding="utf-8")
    
    df = df.sort_values("Date")
    price_data[ticker] = df["Open"].values

prices_df = pd.DataFrame(price_data)

returns_df = np.log(prices_df.shift(1) / prices_df)

mean_returns = returns_df.mean()
risk = returns_df.std()

cov_matrix = returns_df.cov()

# Results
print("Średnie dzienne zwroty:")
print(mean_returns)
print("\nRyzyko (odchylenie std):")
print(risk)
print("\nMacierz kowariancji:")
print(cov_matrix)

# Załóżmy, że returns to DataFrame ze zwrotami dziennymi
# kolumny = tickery, wiersze = daty

corr_matrix = returns_df.corr()  # korelacja między spółkami

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Macierz korelacji zwrotów", fontsize=14)
plt.show()
