import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def generate_ml_report():

    # загрузка данных
    data_path = Path('../src/data/processed/stocks_with_features.csv')
    df = pd.read_csv(data_path, parse_dates=['date'])

    # подготовка признаков
    # target_direction в exclude, чтобы модель не знала ответ
    exclude = ['date', 'ticker', 'target_return', 'target_direction', 'return_1d', 'close', 'volume', 'high', 'low', 'open']
    features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    df_clean = df[features + ['target_direction']].dropna()
    X = df_clean[features]
    y = df_clean['target_direction']

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importance_series.head(10).index.tolist()


    report_content = f"""
## описание задачи:
необходимо обучить модель **бинарной классификации** для предсказания направления движения цены на следующий день (`target_direction`)
- **0**: цена упадет или останется неизменной
- **1**: цена вырастет

## датасет:
данные находятся в `../src/data/processed/stocks_with_features.csv`
- **объем**: примерно 21,000 записей (20 акций за 2020-2024 годы)
- **формат**: таблица, где каждая строка — день по одной акции
- **временной порядок**: данные отсортированы по дате

## рекомендуемые признаки:
на основе проведенного анализа Feature Importance, для модели следует использовать следующий топ признаков (остальные можно отбросить как шум):

| ранг | название признака | описание |
|------|-------------------|----------|
"""

    for i, feat in enumerate(top_features, 1):
        desc = {
            'atr_norm': 'волатильность (нормализованная)',
            'volume_ratio': 'аномалия объема',
            'rsi_14': 'индикатор RSI',
            'momentum_60d': 'долгосрочный тренд',
            'hl_ratio': 'внутридневная волатильность'
        }
        report_content += f"| {i} | `{feat}` | {desc.get(feat, 'важный признак')} |\n"

    report_content += f"""
> **важно:** не использовать абсолютные цены (Open, Close, High, Low) — модель должна опираться на индикаторы и отношения

## рекомендуемые типы моделей:
учитывая, что финансовые данные шумные и нелинейные, рекомендуется использовать **ансамбли деревьев**:

1. **XGBoost / LightGBM / CatBoost**:
   - лучше всего работают с табличными данными
   - позволяют обрабатывать нелинейные зависимости
   - имеют встроенную регуляризацию (борьба с переобучением)

2. **Random Forest**:
   - хороший бейзлайн
   
## стратегия разделения данных:
**важно:** нельзя перемешивать данные случайным образом
используй **Time Series Split**:
- **Train**: 2020-2023 гг.
- **Validation**: 2023-2024 гг.
- **Test (Out-of-Sample)**: последние 3 месяца 2024 г.

## метрики качества:
точность недостаточна. фокусируйся на:

1. **ROC-AUC**: способность модели ранжировать (отличать рост от падения)
   - цель: > 0.55 (на случайных данных 0.5)
2. **F1-Score**: баланс между Precision и Recall
3. **финансовые метрики**:
   - если модель дает сигнал, какова средняя доходность (Expected Return) на этих сигналах?

## потенциальные проблемы:
- **дисбаланс классов**: рост и падение могут встречаться примерно 50/50, но сила движения разная
- **Look-ahead Bias**: при создании новых признаков убедитесь, что не используется значение `Close` за тот же день для предсказания того же дня
"""


    output_path = Path('../src/data/processed/optimization_report.md')
    output_path.write_text(report_content, encoding='utf-8')


generate_ml_report()