import pandas as pd
import yfinance as yf

TICKERS = ["KO", "PEP", "PG", "CL", "KMB", "GIS", "MO",
           "PM", "WMT", "COST", "KR", "SYY", "HSY", "CPB"]

def load(path="prices.csv"):
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        df = yf.download(TICKERS, start="2012-01-01", end="2024-12-31",
                         auto_adjust=True)["Close"]
        df.to_csv(path)   # cache so can reproduce results
        return df