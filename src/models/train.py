import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])

    exclude = ['date', 'ticker', 'target_return', 'target_direction', 
               'return_1d', 'close', 'volume', 'high', 'low', 'open',
               'dollar_volume', 'dollar_vol_ma20']
    
    features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
    
    
    return df, features

def time_split(df, features):
    dates = df['date']
    y = df['target_direction']
    returns = df['target_return']
    X = df[features]
    
    train_mask = dates.dt.year < 2023
    val_mask = dates.dt.year == 2023
    test_mask = dates.dt.year == 2024
    
    return (X[train_mask], y[train_mask], returns[train_mask]), \
           (X[val_mask], y[val_mask], returns[val_mask]), \
           (X[test_mask], y[test_mask], returns[test_mask])

def train_model():
    csv_path = Path('src/data/processed/stocks_with_features.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")
        
    df, features = load_data(csv_path)
    
    print("Временное разделение...")
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, returns_test) = time_split(df, features)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight, 'verbose': -1, 'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # оценка
    probs_test = model.predict(X_test)
    preds_test = (probs_test >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, probs_test)
    f1 = f1_score(y_test, preds_test)
    
    print(f"\nTest (2024) | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
    
    signal_mask = preds_test == 1
    if signal_mask.sum() > 0:
        avg_return = returns_test[signal_mask].mean()
        print(f"Средняя доходность по сигналам 'Buy': {avg_return:.4%}")
    else:
        print("Нет сигналов 'Buy' на тесте")
        
    # Сохранение
    out_dir = Path('src/models')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, out_dir / 'price_model.pkl')
    joblib.dump(features, out_dir / 'feature_names.pkl')
    print(f"\nМодель сохранена в {out_dir}")

if __name__ == '__main__':
    train_model()