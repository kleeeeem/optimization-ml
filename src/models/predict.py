import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def generate_signals():
    print("📥 Загрузка модели и свежих данных...")
    
    # Пути относительно корня проекта
    root = Path('.')
    model_path = root / 'src/models/price_model.pkl'
    features_path = root / 'src/models/feature_names.pkl'
    data_path = root / 'src/data/processed/stocks_with_features.csv'
    
    if not model_path.exists():
        raise FileNotFoundError("Модель не найдена. Сначала запустите train.py")
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    exclude = ['date', 'ticker', 'target_return', 'target_direction', 
               'return_1d', 'close', 'volume', 'high', 'low', 'open',
               'dollar_volume', 'dollar_vol_ma20']
    valid_features = [f for f in features if f in df.columns and f not in exclude]
    
    print(f"Используем {len(valid_features)} признаков")
    
    max_date = df['date'].max()
    recent = df[df['date'] >= max_date - pd.Timedelta(days=60)].copy()
    
    X = recent[valid_features]
    recent['prob_up'] = model.predict(X)
    
    # агрегируем по тикеру: средняя вероятность, волатильность, ликвидность
    signals = recent.groupby('ticker').agg(
        prob_up=('prob_up', 'mean'),
        volatility=('volatility_20d', 'mean'),
        volume=('volume', 'mean'),
        close=('close', 'last'),
        latest_date=('date', 'max')
    ).reset_index()
    
    signals['expected_return'] = (signals['prob_up'] - 0.5) * 0.003 * 5
    
    # risk_proxy: нормализованная волатильность (используется как σ в матрице ковариации)
    signals['risk'] = signals['volatility'] / 100
    
    # liquidity: средний дневной объем (ограничение доли позиции)
    signals['liquidity'] = signals['volume']
    
    # фильтр: отбрасываем активы, где модель почти не видит роста
    signals = signals[signals['prob_up'] >= 0.45]
    
    out_path = root / 'src/data/processed/ml_signals_for_optimizer.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_path, index=False)
    
    print(f"Сигналы сохранены: {out_path}")
    print(f"Тикеров прошло фильтр: {len(signals)}")
    print("\nВыходных данные:")
    print(signals[['ticker', 'prob_up', 'expected_return', 'risk', 'liquidity']].head())
    
    return signals

if __name__ == '__main__':
    generate_signals()