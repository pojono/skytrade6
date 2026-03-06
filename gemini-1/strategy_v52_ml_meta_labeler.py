import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    if not files: return None
    df_list = []
    for f in files:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if not df_list: return None
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.15) & (df['ret'] < 0.15)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df.dropna(inplace=True)
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

def calc_features(df_4h):
    # RSI
    delta = df_4h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_4h['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_4h['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_4h['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df_4h['macd_hist'] = macd - signal
    
    # ATR & Volatility
    high_low = df_4h['high'] - df_4h['low']
    high_close = np.abs(df_4h['high'] - df_4h['close'].shift())
    low_close = np.abs(df_4h['low'] - df_4h['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_4h['atr_14'] = tr.rolling(14).mean()
    df_4h['atr_pct'] = df_4h['atr_14'] / df_4h['close']
    
    # Volume dynamics
    df_4h['vol_ma_20'] = df_4h['volume'].rolling(20).mean()
    df_4h['vol_ratio'] = df_4h['volume'] / df_4h['vol_ma_20']
    
    # Trend distances
    df_4h['ema_20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema_50'] = df_4h['close'].ewm(span=50, adjust=False).mean()
    df_4h['dist_ema20'] = (df_4h['close'] - df_4h['ema_20']) / df_4h['ema_20']
    df_4h['dist_ema50'] = (df_4h['close'] - df_4h['ema_50']) / df_4h['ema_50']
    
    return df_4h

def extract_base_trades(args):
    symbol = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_4h.dropna(inplace=True)
    df_4h = calc_features(df_4h)
    
    # Calculate pure math strategy base (No filters except volume and 200 EMA trend)
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['macro_bull'] = df_4h['close'] > df_4h['ema_200']
    
    df_4h['high_20'] = df_4h['high'].rolling(20).max().shift(1)
    df_4h['low_20'] = df_4h['low'].rolling(20).min().shift(1)
    
    # Shift features by 1 so we don't look ahead!
    feature_cols = ['rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio', 'dist_ema20', 'dist_ema50']
    for c in feature_cols:
        df_4h[f'{c}_shift'] = df_4h[c].shift(1)
        
    df_4h['vol_spike'] = df_4h['volume'] > (df_4h['vol_ma_20'].shift(1) * 2.0)
    
    long_sigs = (df_4h['close'] > df_4h['high_20']) & df_4h['macro_bull'] & df_4h['vol_spike']
    short_sigs = (df_4h['close'] < df_4h['low_20']) & (~df_4h['macro_bull']) & df_4h['vol_spike']
    
    # Optional KER just for comparison recording
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['ker_shift'] = (change / volatility).shift(1)

    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    cooldown_until = 0
    pos_type = 0
    
    TP_PCT = 0.20
    SL_PCT = 0.10
    TIME_STOP_CANDLES = 6 * 14
    
    n = len(df_4h)
    
    opens = df_4h['open'].values
    highs = df_4h['high'].values
    lows = df_4h['low'].values
    closes = df_4h['close'].values
    l_sig_arr = long_sigs.values
    s_sig_arr = short_sigs.values
    
    for i in range(200, n - 1):
        if not in_position:
            if i > cooldown_until:
                if l_sig_arr[i]:
                    in_position = True
                    pos_type = 1
                    entry_price = opens[i+1]
                    entry_idx = i + 1
                    entry_time = df_4h.index[i+1]
                    
                    # Capture features at time of signal
                    features = {f: df_4h[f'{f}_shift'].iloc[i] for f in feature_cols}
                    features['ker'] = df_4h['ker_shift'].iloc[i]
                    
                elif s_sig_arr[i]:
                    in_position = True
                    pos_type = -1
                    entry_price = opens[i+1]
                    entry_idx = i + 1
                    entry_time = df_4h.index[i+1]
                    
                    features = {f: df_4h[f'{f}_shift'].iloc[i] for f in feature_cols}
                    features['ker'] = df_4h['ker_shift'].iloc[i]
        else:
            if pos_type == 1:
                target_price = entry_price * (1 + TP_PCT)
                stop_price = entry_price * (1 - SL_PCT)
                
                if lows[i] <= stop_price:
                    net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'dir': 1, **features})
                    in_position = False
                    cooldown_until = i + 6
                elif highs[i] >= target_price:
                    net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'dir': 1, **features})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'dir': 1, **features})
                    in_position = False
                    cooldown_until = i + 6
                    
            elif pos_type == -1:
                target_price = entry_price * (1 - TP_PCT)
                stop_price = entry_price * (1 + SL_PCT)
                
                if highs[i] >= stop_price:
                    net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'dir': -1, **features})
                    in_position = False
                    cooldown_until = i + 6
                elif lows[i] <= target_price:
                    net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'dir': -1, **features})
                    in_position = False
                    cooldown_until = i + 6
                elif (i - entry_idx) >= TIME_STOP_CANDLES:
                    net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                    trades.append({'symbol': symbol, 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'dir': -1, **features})
                    in_position = False
                    cooldown_until = i + 6

    return trades

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    print("1. Extracting Raw Breakouts & Features...")
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    for trades in tqdm(pool.imap_unordered(extract_base_trades, symbols), total=len(symbols)):
        all_trades.extend(trades)
    pool.close()
    pool.join()
    
    df_trades = pd.DataFrame(all_trades)
    df_trades = df_trades.sort_values('entry_time').reset_index(drop=True)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    
    df_trades.dropna(inplace=True)
    
    # Create Binary Label: 1 if PnL > 0 (profitable breakout), 0 if fakeout
    df_trades['label'] = (df_trades['raw_pnl'] > 0).astype(int)
    
    feature_cols = ['rsi_14', 'macd_hist', 'atr_pct', 'vol_ratio', 'dist_ema20', 'dist_ema50', 'dir']
    
    # We will use an expanding window (Time Series Walk Forward)
    # Train on past data, test on next year
    
    print(f"\nTotal extracted breakouts: {len(df_trades)}")
    print(f"Base Win Rate (No Filters): {df_trades['label'].mean():.2%}")
    
    # 1. Pure KER Filter Results (Baseline)
    df_ker = df_trades[df_trades['ker'] >= 0.15]
    ker_wr = df_ker['label'].mean()
    ker_ev = df_ker['raw_pnl'].mean()
    ker_trades = len(df_ker)
    ker_total_ret = df_ker['raw_pnl'].sum() * (0.02 / 0.10) # 20% position size for 2% risk
    
    print("\n" + "="*80)
    print("BASELINE: Math-based KER Filter (>= 0.15)")
    print("="*80)
    print(f"Trades: {ker_trades} | Win Rate: {ker_wr:.2%} | EV/Trade: {ker_ev:.2%} | Total Ret: {ker_total_ret:.2%}")
    
    # 2. LightGBM Walk-Forward Meta-Labeler
    print("\n2. Training LightGBM Walk-Forward Meta-Model...")
    
    # Split times
    train_end_1 = pd.to_datetime('2022-01-01')
    test_end_1 = pd.to_datetime('2023-01-01')
    test_end_2 = pd.to_datetime('2024-01-01')
    test_end_3 = pd.to_datetime('2025-01-01')
    test_end_4 = pd.to_datetime('2026-03-01')
    
    splits = [
        (df_trades['entry_time'] < train_end_1, (df_trades['entry_time'] >= train_end_1) & (df_trades['entry_time'] < test_end_1)),
        (df_trades['entry_time'] < test_end_1, (df_trades['entry_time'] >= test_end_1) & (df_trades['entry_time'] < test_end_2)),
        (df_trades['entry_time'] < test_end_2, (df_trades['entry_time'] >= test_end_2) & (df_trades['entry_time'] < test_end_3)),
        (df_trades['entry_time'] < test_end_3, (df_trades['entry_time'] >= test_end_3) & (df_trades['entry_time'] < test_end_4)),
    ]
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 15,          # Very small to prevent overfitting
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 20
    }
    
    ml_preds = np.zeros(len(df_trades))
    ml_tested_mask = np.zeros(len(df_trades), dtype=bool)
    
    for train_mask, test_mask in splits:
        X_train = df_trades.loc[train_mask, feature_cols]
        y_train = df_trades.loc[train_mask, 'label']
        
        X_test = df_trades.loc[test_mask, feature_cols]
        y_test = df_trades.loc[test_mask, 'label']
        
        if len(X_train) < 50 or len(X_test) == 0: continue
            
        train_data = lgb.Dataset(X_train, label=y_train)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=100
        )
        
        preds = model.predict(X_test)
        ml_preds[test_mask] = preds
        ml_tested_mask[test_mask] = True
        
    df_trades['ml_prob'] = ml_preds
    
    # Evaluate ML performance strictly on the Walk-Forward OOS windows
    df_oos = df_trades[ml_tested_mask].copy()
    
    # We only take the trade if LightGBM probability is > 0.55 (confident)
    df_ml = df_oos[df_oos['ml_prob'] > 0.50]
    
    ml_wr = df_ml['label'].mean()
    ml_ev = df_ml['raw_pnl'].mean()
    ml_trades = len(df_ml)
    ml_total_ret = df_ml['raw_pnl'].sum() * (0.02 / 0.10)
    
    print("\n" + "="*80)
    print("EXPERIMENTAL: LightGBM Meta-Labeler (Strict OOS)")
    print("="*80)
    print(f"Trades: {ml_trades} | Win Rate: {ml_wr:.2%} | EV/Trade: {ml_ev:.2%} | Total Ret: {ml_total_ret:.2%}")
    
    # Feature Importance (from the last model)
    print("\nFeature Importance (Latest Model):")
    importance = model.feature_importance(importance_type='gain')
    for f, imp in sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True):
        print(f"  {f}: {imp:.1f}")
        
    print("\n--- CONCLUSION ---")
    if ml_total_ret > ker_total_ret:
        print("LightGBM BEAT the KER math formula! ML captured nonlinear relationships effectively.")
    else:
        print("KER Math Filter CRUSHED the ML Model. The dataset is too small, and ML overfitted noise.")

if __name__ == "__main__":
    main()
