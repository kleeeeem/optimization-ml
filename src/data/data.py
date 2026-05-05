import yfinance as yf
import pandas as pd
from pathlib import Path

def download_data():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'NVDA', 'TSLA', 'JPM', 'V', 'JNJ',
               'WMT', 'PG', 'UNH', 'HD', 'MA',
               'DIS', 'NFLX', 'PYPL', 'INTC', 'CSCO']
    
    data = yf.download(tickers, start='2020-01-01', end='2024-12-31')
    
    Path('src/data/raw').mkdir(parents=True, exist_ok=True)
    
    data.to_csv('src/data/raw/stock_prices.csv')
    print(f"Данные сохранены: {data.shape}")
    print(f"Тикеров: {len(tickers)}")

if __name__ == "__main__":
    download_data()