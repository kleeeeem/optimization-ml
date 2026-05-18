import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import cast
import warnings
warnings.filterwarnings('ignore')


def download_raw_data(tickers, start_date, end_date):
    print(f"тикеров: {len(tickers)}")
    print(f"период: {start_date} - {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=True)
    return data


def calculate_features_for_ticker(ticker_df, ticker_name):
    """
    рассчитывает признаки для одного тикера
    принимает df с колонками: Open, High, Low, Close, Adj Close, Volume
    """
    df = ticker_df.copy()
    df.index.name = 'date'
    df.reset_index(inplace=True)
    df['ticker'] = ticker_name

    # нормализация
    col_map = {c.lower(): c for c in df.columns}

    if 'close' not in col_map and 'adj close' not in col_map:
        return None

    close_col = col_map.get('adj close') or col_map.get('close')
    df['close'] = df[close_col]
    df['volume'] = df[col_map.get('volume', close_col)]
    df['high'] = df[col_map.get('high', close_col)]
    df['low'] = df[col_map.get('low', close_col)]
    df['open'] = df[col_map.get('open', close_col)]

    # целевая переменная
    df['return_1d'] = df['close'].pct_change()
    df['target_return'] = df['return_1d'].shift(-1)
    # бинарная классификация: 1 если вырастет, 0 если упадет
    df['target_direction'] = (df['target_return'] > 0).astype(int)

    # ликвидность
    df['dollar_volume'] = df['close'] * df['volume']
    df['log_volume'] = np.log1p(df['volume'])
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['vol_ma20']
    df['dollar_vol_ma20'] = df['dollar_volume'].rolling(20).mean()
    df['dollar_vol_ratio'] = df['dollar_volume'] / df['dollar_vol_ma20']

    # волатильность
    df['volatility_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
    df['volatility_10d'] = df['return_1d'].rolling(10).std() * np.sqrt(252)
    df['true_range'] = df['high'] - df['low']
    df['atr_14'] = df['true_range'].rolling(14).mean()
    df['atr_norm'] = df['atr_14'] / df['close']
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']

    # моментум
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)
    df['momentum_60d'] = df['close'].pct_change(60)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['roc_10'] = df['close'].pct_change(10) * 100

    # тренд
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dev_from_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['dev_from_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['dev_from_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']
    df['sma20_vs_sma50'] = df['sma_20'] / df['sma_50']
    df['sma50_vs_sma200'] = df['sma_50'] / df['sma_200']

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # доп
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    df['price_vs_high20'] = df['close'] / df['high'].rolling(20).max()
    df['price_vs_low20'] = df['close'] / df['low'].rolling(20).min()
    df['volume_z_score'] = (df['volume'] - df['vol_ma20']) / df['vol_ma20'].rolling(20).std()

    return df


def preprocess_data(raw_data):
    """рассчитывает признаки для всех тикеров"""
    # определяем тикеров с мультииндексом
    if isinstance(raw_data.columns, pd.MultiIndex):
        tickers = cast(pd.MultiIndex, raw_data.columns).get_level_values(0).unique().tolist()
    else:
        tickers = raw_data.columns.tolist()

    print(f"тикеров: {len(tickers)}")
    all_features = []
    for i, ticker in enumerate(tickers, 1):
        print(f"   [{i}/{len(tickers)}] {ticker}...", end=" ")
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                ticker_df_raw = raw_data[ticker]
            else:
                ticker_df_raw = raw_data

            if ticker_df_raw.dropna().empty:
                continue

            feat_df = calculate_features_for_ticker(ticker_df_raw, ticker)
            if feat_df is not None and not feat_df.empty:
                all_features.append(feat_df)
        except Exception as e:
            print(f"ошибка: {e}")
            continue

    features = pd.concat(all_features, ignore_index=True)

    # удаляем NaN
    nan_before = len(features)
    features.dropna(inplace=True)
    print(f"удалено {nan_before - len(features)} строк с NaN")

    # удаляем inf в числовых колонках
    numeric_cols = features.select_dtypes(include='number').columns
    inf_before = len(features)
    features = features[~np.isinf(features[numeric_cols]).any(axis=1)]
    print(f"удалено {inf_before - len(features)} строк с inf")

    features.reset_index(drop=True, inplace=True)

    print(f"итого: {features.shape[0]} записей, {features.shape[1]} признаков")
    print(f"тикеров: {features['ticker'].nunique()}")
    print(f"даты: {features['date'].min()} - {features['date'].max()}")

    return features


def save_data(raw_data, features_ds):
    """сохраняет сырые и обработанные данные"""
    Path('../src/data/raw').mkdir(parents=True, exist_ok=True)
    Path('../src/data/processed').mkdir(parents=True, exist_ok=True)
    raw_data.to_csv('../src/data/raw/stock_prices.csv')
    features_ds.to_csv('../src/data/processed/stocks_with_features.csv', index=False)

    # сохраняем датасет
    info_path = '../src/data/processed/dataset_info.md'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("# информация о датасете\n\n")
        f.write(f"**период:** {features_ds['date'].min().date()} - {features_ds['date'].max().date()}\n\n")
        f.write(f"**количество записей:** {features_ds.shape[0]}\n")
        f.write(f"**количество признаков:** {features_ds.shape[1]}\n")
        f.write(f"**количество тикеров:** {features_ds['ticker'].nunique()}\n\n")
        f.write("## признаки:\n\n")
        for col in features_ds.columns:
            if col not in ['date', 'ticker']:
                f.write(f"- {col}\n")


def download_and_preprocess(tickers, start_date, end_date):
    raw_data = download_raw_data(tickers, start_date, end_date)
    feats = preprocess_data(raw_data)
    save_data(raw_data, feats)
    return feats


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'NFLX', 'PYPL', 'INTC', 'CSCO']
features_df = download_and_preprocess(tickers, '2020-01-01', '2024-12-31')