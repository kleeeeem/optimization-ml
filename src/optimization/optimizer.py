import pandas as pd
import numpy as np
import cvxpy as cp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def run_optimizer(
    investment_amount: float = 10000.0,
    risk_aversion: float = 2.0,
    tx_cost_pct: float = 0.001,
    max_single_weight: float = 0.15,
    top_n_assets: int = 10
):
    root = Path('.')
    signals_path = root / 'src/data/processed/ml_signals_for_optimizer.csv'
    data_path = root / 'src/data/processed/stocks_with_features.csv'

    # 1. ML-сигналы (ожидаемая доходность, ликвидность, вероятность роста)
    signals = pd.read_csv(signals_path)
    signals = signals.sort_values('prob_up', ascending=False)
    signals = signals.head(top_n_assets)
    print(f" Отбираем топ {top_n_assets} активов для покупки")
    
    tickers = signals['ticker'].tolist()
    mu = signals['expected_return'].values  # вектор матожидания доходности (ML)

    # 2. Историческая ковариационная матрица (риск)
    df = pd.read_csv(data_path, parse_dates=['date'])
    returns_pivot = df.pivot(index='date', columns='ticker', values='return_1d')
    recent_returns = returns_pivot.tail(252).dropna(axis=1, how='all')

    # Проверяем, чтобы тикеры были в данных
    valid_tickers = [t for t in tickers if t in recent_returns.columns]
    tickers = valid_tickers
    signals = signals[signals['ticker'].isin(tickers)]
    mu = signals['expected_return'].values

    cov_matrix = recent_returns[tickers].cov().values


    n = len(tickers)
    w = cp.Variable(n)  # веса портфеля

    # 3. Целевая функция: доходность - риск - транзакционные издержки
    portfolio_return = mu @ w
    portfolio_risk = cp.quad_form(w, cov_matrix)
    tx_costs = tx_cost_pct * cp.sum(cp.abs(w)) # предполагаем вход с нулевой позиции

    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk - tx_costs)

    # 4. Ограничения
    constraints = [
        cp.sum(w) == 1.0,                  # весь капитал инвестирован
        w >= 0.0,                          # только длинные позиции (без шортов)
        w <= max_single_weight             # диверсификация: макс. доля на одну бумагу
    ]

    # 5. Решение задачи
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f" Оптимизация не сошлась. Статус: {prob.status}")

    # 6. Интерпретация результатов
    weights = w.value
    amounts = weights * investment_amount

    result_df = pd.DataFrame({
        'ticker': tickers,
        'weight': np.round(weights, 4),
        'amount': np.round(amounts, 2),
        'prob_up': signals['prob_up'].values,
        'ml_expected_return': mu
    }).sort_values('weight', ascending=False).reset_index(drop=True)

    # Убираем позиции < 0.5% (шум/издержки не окупятся)
    result_df = result_df[result_df['weight'] >= 0.02]

    print("\nРекомендации по распределению:")
    print(result_df.to_string(index=False))
    print(f"\nИтого инвестировано: ${investment_amount:,.2f}")
    print(f"Ожидаемая доходность портфеля (горизонт ~5 дней): {np.dot(weights, mu):.4%}")
    print(f"️ Уровень риска (дисперсия портфеля): {portfolio_risk.value:.6f}")

    # Сохранение
    out_path = root / 'src/data/processed/portfolio_allocation.csv'
    result_df.to_csv(out_path, index=False)
    print(f"\nРаспределение сохранено в {out_path}")
    
    return result_df

if __name__ == '__main__':
    # investment_amount: сумма для инвестирования
    # risk_aversion: 1.0 (агрессивно) ... 5.0 (консервативно)
    # tx_cost_pct: комиссия брокера за сделку (0.001 = 0.1%)
    # max_single_weight: макс. доля одной акции в портфеле
    run_optimizer(
        investment_amount=10000,
        risk_aversion=2.5,
        tx_cost_pct=0.002,
        max_single_weight=0.18
    )