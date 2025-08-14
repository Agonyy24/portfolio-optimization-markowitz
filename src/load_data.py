"""
Module: load_data
Purpose: Download and save historical stock price data
"""

import yfinance as yf
import os

def load_data(tickers: list[str], start_date: str, end_date: str) -> None:
    """
    Downloads historical stock price data for given tickers and saves them as CSV files
    containing only the Date and Open price.

    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols (e.g., ["AAPL", "MSFT"])
    start_date : str
        Start date for historical data in 'YYYY-MM-DD' format
    end_date : str
        End date for historical data in 'YYYY-MM-DD' format
    data_dir : str, optional
        Directory where CSV files will be stored (default is "data")

    Returns
    -------
    None
    """
    
    SRC_DIR = os.path.dirname(__file__)

    PROJECT_ROOT = os.path.dirname(SRC_DIR)

    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    os.makedirs(DATA_DIR, exist_ok=True)

    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        
        # Skip download if file already exists
        if os.path.exists(file_path):
            print(f"[INFO] File already exists: {file_path} — skipping download.")
            continue
        
        print(f"[INFO] Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            print(f"[WARNING] No data found for {ticker}. Skipping.")
            continue

        # Keep only Date and Open columns
        df = df.reset_index()[["Date", "Open"]]
        print(df)
        # Save to CSV
        df.columns = df.columns.get_level_values(0) # Arrange the problem with Multiindex
        df.to_csv(file_path, index=False)
        print(f"[SUCCESS] Saved: {file_path}")

if __name__ == "__main__":
    tickers_list = ["AAPL", "NVDA", "MSFT", "JPM", "NDAQ"]
    load_data(tickers_list, start_date="2015-01-01", end_date="2024-01-01")