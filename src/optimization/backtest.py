import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def analyze_signal_decay(df, prob_col='prob_up', threshold=None):
    """Проверяет, на каком горизонте ML-сигнал ещё прибылен"""
    print("🔍 Анализ затухания сигнала...")
    
    # Если порог не задан или слишком строгий, берем медиану или 0.50
    if threshold is None:
        threshold = max(0.50, df[prob_col].median())
        print(f"  (Порог не указан, используем адаптивный: {threshold:.3f})")
        
    for d in [1, 3, 5, 10]:
        df[f'fwd_ret_{d}d'] = df.groupby('ticker')['return_1d'].shift(-d)

    mask = df[prob_col] >= threshold
    n_signals = mask.sum()

    if n_signals == 0:
        print(f"⚠️ Нет сигналов с prob_up >= {threshold}. Уменьшите порог или используйте квантили.")
        return pd.DataFrame()

    decay = []
    for d in [1, 3, 5, 10]:
        avg_ret = df.loc[mask, f'fwd_ret_{d}d'].mean()
        win_rate = (df.loc[mask, f'fwd_ret_{d}d'] > 0).mean()
        decay.append({'horizon_days': d, 'avg_return': avg_ret, 'win_rate': win_rate, 'n_signals': n_signals})
    
    return pd.DataFrame(decay)

def run_portfolio_backtest(df, prob_col='prob_up', initial_capital=10000, tx_cost=0.002, rebalance_freq=21):
    """Векторизированный бэктест: ежемесячная ребалансировка по ML-вероятностям"""
    print("📈 Симуляция портфеля с ML-ребалансировкой...")
    
    dates = df['date'].unique()
    dates = np.sort(dates)
    
    capital = initial_capital
    weights_prev = None
    equity_curve = [capital]
    trade_count = 0
    
    for i, current_date in enumerate(dates):
        if i % rebalance_freq != 0:
            day_weights = weights_prev if weights_prev is not None else np.zeros(len(df['ticker'].unique()))
            day_returns = df[df['date'] == current_date].set_index('ticker')['return_1d'].reindex(df['ticker'].unique()).fillna(0)
            capital *= (1 + np.dot(day_weights, day_returns.values))
            equity_curve.append(capital)
            continue
            
        day_data = df[df['date'] == current_date].copy()
        day_data = day_data[day_data[prob_col] >= 0.50]  # Фильтр слабых сигналов
        
        if day_data.empty:
            weights_prev = np.zeros(len(df['ticker'].unique()))
            equity_curve.append(capital)
            continue
            
        raw_weights = day_data.set_index('ticker')[prob_col].reindex(df['ticker'].unique()).fillna(0)
        weights = raw_weights.values / raw_weights.sum() if raw_weights.sum() > 0 else np.zeros(len(raw_weights))
        
        if weights_prev is not None:
            turnover = np.sum(np.abs(weights - weights_prev))
            capital *= (1 - tx_cost * turnover)
            trade_count += 1
            
        weights_prev = weights
        
        day_returns = day_data.set_index('ticker')['return_1d'].reindex(df['ticker'].unique()).fillna(0)
        capital *= (1 + np.dot(weights, day_returns.values))
        equity_curve.append(capital)
        
    return np.array(equity_curve), trade_count

def compute_metrics(equity_curve, risk_free_rate=0.02):
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    peak = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - peak) / peak
    max_drawdown = drawdown.min()
    
    excess_returns = returns - risk_free_rate / 252
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'trading_days': len(returns)
    }

def main():
    print("📥 Загрузка данных и модели...")
    root = Path('.')
    data_path = root / 'src/data/processed/stocks_with_features.csv'
    model_path = root / 'src/models/price_model.pkl'
    features_path = root / 'src/models/feature_names.pkl'
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    
    test_df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2024-12-31')].copy()
    
    exclude = ['date', 'ticker', 'target_return', 'target_direction', 'return_1d', 
               'close', 'volume', 'high', 'low', 'open', 'dollar_volume', 'dollar_vol_ma20']
    valid_features = [f for f in features if f in test_df.columns and f not in exclude]
    
    print(f"🤖 Генерация прогнозов на {len(test_df)} строк...")
    test_df['prob_up'] = model.predict(test_df[valid_features])
    
    # 1️ Анализ затухания (исправленный порог)
    decay_df = analyze_signal_decay(test_df, threshold=0.50)
    print("\n📊 Затухание ML-сигнала:")
    print(decay_df.to_string(index=False) if not decay_df.empty else "Нет данных для анализа")
    
    # 2️⃣ Бэктест ML-портфеля
    ml_equity, ml_trades = run_portfolio_backtest(test_df, prob_col='prob_up', initial_capital=10000, tx_cost=0.002, rebalance_freq=21)
    ml_metrics = compute_metrics(ml_equity)
    
    # 3️ Бэктест Buy & Hold
    bh_equity, _ = run_portfolio_backtest(test_df, prob_col='prob_up', initial_capital=10000, tx_cost=0.0, rebalance_freq=999)
    bh_metrics = compute_metrics(bh_equity)
    
    print("\n📈 Результаты бэктеста (2023-2024):")
    print(f"{'Метрика':<20} | {'ML-портфель':<12} | {'Buy & Hold':<12}")
    print("-" * 48)
    print(f"{'Итоговая доходность':<20} | {ml_metrics['total_return']:>10.2%} | {bh_metrics['total_return']:>10.2%}")
    print(f"{'CAGR (годовая)':<20} | {ml_metrics['cagr']:>10.2%} | {bh_metrics['cagr']:>10.2%}")
    print(f"{'Max Drawdown':<20} | {ml_metrics['max_drawdown']:>10.2%} | {bh_metrics['max_drawdown']:>10.2%}")
    print(f"{'Sharpe Ratio':<20} | {ml_metrics['sharpe_ratio']:>10.2f} | {bh_metrics['sharpe_ratio']:>10.2f}")
    print(f"{'Кол-во сделок':<20} | {ml_trades:>10} | {'-':<12}")
    
    # 4️⃣ Безопасный расчет горизонта
    if decay_df.empty or decay_df['avg_return'].isna().all():
        optimal_horizon = 5
        print("\n⚠️ Сигналы редкие или порог слишком строгий. Используем стандартный горизонт 5 дней.")
    else:
        safe_idx = decay_df['avg_return'].dropna().idxmax()
        optimal_horizon = decay_df.loc[safe_idx, 'horizon_days']
        
    print("\n✅ ВЫВОД ОБ АКТУАЛЬНОСТИ:")
    print(f"• ML-сигналы актуальны на горизонте ~{optimal_horizon} дней.")
    print(f"• Рекомендуется ребалансировка или фиксация прибыли каждые {optimal_horizon} торговых дней.")
    print(f"• Sharpe ML-портфеля: {ml_metrics['sharpe_ratio']:.2f} vs Buy&Hold: {bh_metrics['sharpe_ratio']:.2f}")
    print(f"• Комиссии {0.2:.1%} частично компенсируются фильтрацией слабых сигналов.")
    
        # Сохранение отчёта (без зависимости от tabulate)
    report_path = root / 'src/data/processed/backtest_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Отчёт по актуальности ML-сигналов и бэктест\n\n")
        f.write("## Затухание сигнала\n```\n")
        f.write(decay_df.to_string(index=False) + "\n```\n\n" if not decay_df.empty else "Данных недостаточно\n\n")
        f.write("## Метрики стратегии\n```\n")
        f.write(pd.DataFrame([ml_metrics, bh_metrics], index=['ML-Portfolio', 'Buy & Hold']).to_string() + "\n```\n\n")
        f.write("## Заключение\n")
        f.write(f"Сигналы модели актуальны для торговли на горизонте **1-{optimal_horizon} дней**.\n")
        f.write(f"Рекомендуется фиксировать прибыль или ребалансировать портфель не реже чем раз в {optimal_horizon} торговых дней.\n")
    print(f"\n💾 Отчёт сохранён в {report_path}")
if __name__ == '__main__':
    main()