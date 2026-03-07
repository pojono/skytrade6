import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
# Let's use the top performing coins to train the model
SYMBOLS = ["BERAUSDT", "ZKUSDT", "ZECUSDT", "HUMAUSDT", "INITUSDT", "NEARUSDT", "SAHARAUSDT", "AGLDUSDT"]

def build_ml_dataset():
    all_events = []
    
    for sym in SYMBOLS:
        file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
        if not os.path.exists(file_path): continue
        
        df = pd.read_parquet(file_path)
        
        rolling_window = 4 * 60
        df['roll_fut_whale'] = df['fut_whale_cvd'].rolling(rolling_window).sum()
        df['roll_fut_retail'] = df['fut_retail_cvd'].rolling(rolling_window).sum()
        
        # Labels
        df['fwd_ret_4h'] = df['price'].shift(-240) / df['price'] - 1
        # Target: 1 if the trade is a winner (> 0.2% to cover fees), 0 if it fails
        df['target'] = (df['fwd_ret_4h'] > 0.002).astype(int)
        
        z_window = 3 * 24 * 60
        df['fut_whale_mean'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_whale_std'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_whale_z'] = (df['roll_fut_whale'] - df['fut_whale_mean']) / df['fut_whale_std']
        
        df['fut_retail_mean'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_retail_std'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_retail_z'] = (df['roll_fut_retail'] - df['fut_retail_mean']) / df['fut_retail_std']
        
        df['spot_whale_1h_avg'] = df['spot_whale_cvd'].abs().rolling(60).mean()
        
        # Base Trigger (The structural state we want to filter)
        bullish_div = (
            (df['fut_retail_z'] < -1.5) & 
            (df['fut_whale_z'] > 1.5) &
            (df['spot_whale_cvd'] > df['spot_whale_1h_avg'] * 3.0) &
            (df['spot_whale_cvd'] > 0)
        )
        
        # Engineer Contextual Features for the ML model
        
        # 1. Trend context (Are we catching a falling knife in a massive downtrend?)
        df['price_change_24h'] = df['price'] / df['price'].shift(24*60) - 1
        df['price_change_4h'] = df['price'] / df['price'].shift(4*60) - 1
        
        # 2. Volatility context
        df['volatility_4h'] = df['price'].rolling(4*60).std() / df['price']
        
        # 3. Absolute Divergence Strength (How extreme is the divergence?)
        df['div_strength'] = df['fut_whale_z'] - df['fut_retail_z']
        
        # 4. Spot Spike Intensity
        df['spot_spike_ratio'] = df['spot_whale_cvd'] / df['spot_whale_1h_avg'].replace(0, 1)
        
        # Filter overlapping signals
        def filter_signals(signals, wait_time=120):
            filtered = pd.Series(False, index=signals.index)
            last_sig_time = None
            for i, (idx, val) in enumerate(signals.items()):
                if val:
                    if last_sig_time is None or (idx - last_sig_time).total_seconds() / 60 > wait_time:
                        filtered[idx] = True
                        last_sig_time = idx
            return filtered

        df['bull_sig'] = filter_signals(bullish_div, 120)
        
        # Extract the exact moments the signal fires
        events = df[df['bull_sig']].dropna(subset=['fwd_ret_4h', 'price_change_24h'])
        events['symbol'] = sym
        
        if len(events) > 0:
            all_events.append(events)
            
    if not all_events: return None
    return pd.concat(all_events)

def train_ml_filter(df_events):
    print(f"\n--- Training ML Execution Filter ({len(df_events)} Historical Triggers) ---")
    
    features = [
        'fut_whale_z', 'fut_retail_z', 'div_strength',
        'spot_spike_ratio', 'price_change_24h', 'price_change_4h', 
        'volatility_4h'
    ]
    
    # Sort chronologically to prevent lookahead bias in cross-validation
    df_events.sort_index(inplace=True)
    
    X = df_events[features]
    y = df_events['target']
    
    # TimeSeries Split for honest evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    
    precisions, accuracies = [], []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Using LightGBM for robust gradient boosting on small tabular datasets
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4, # Keep shallow to prevent overfitting on 200 rows
            class_weight='balanced',
            random_state=42
        )
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        precisions.append(precision_score(y_test, preds, zero_division=0))
        accuracies.append(accuracy_score(y_test, preds))
        
    print(f"Base Strategy Win Rate: {y.mean()*100:.1f}%")
    print(f"ML Filter Precision (Win Rate of approved trades): {np.mean(precisions)*100:.1f}%")
    print(f"ML Filter Overall Accuracy: {np.mean(accuracies)*100:.1f}%\n")
    
    # Train on full dataset to get feature importance
    final_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42)
    final_clf.fit(X, y)
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': final_clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("What matters most when filtering trades? (Feature Importance)")
    print(importance.to_string(index=False))

if __name__ == "__main__":
    df_events = build_ml_dataset()
    if df_events is not None:
        train_ml_filter(df_events)
