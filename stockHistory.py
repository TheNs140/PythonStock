import yfinance as yf
import pandas as pd

# Fetch historical stock data
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# Example: Fetching data for Tesla
stock_data = get_stock_data("TSLA", "2023-01-01", "2024-01-01")
print(stock_data.head())