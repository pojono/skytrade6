import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def audit_single_trade(symbol="BTCUSDT"):
    print(f"--- Re-Evaluating Strict Lookahead Audit on {symbol} ---")
    start_date = "2025-01-01"
    
    kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
    kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
    dfs = []
    for f in kline_files:
        try: dfs.append(pd.read_csv(f, usecols=['open_time', 'open', 'high', 'low', 'close'], engine='c'))
        except: pass
    kline_df = pd.concat(dfs, ignore_index=True)
    kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
    kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
    if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
    kline_df.set_index('timestamp', inplace=True)
    
    metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
    metrics_files = [f for f in metrics_files if f.name >= start_date]
    dfs = []
    for f in metrics_files:
        try: dfs.append(pd.read_csv(f, usecols=['create_time', 'sum_open_interest_value'], engine='c'))
        except: pass
    oi_df = pd.concat(dfs, ignore_index=True)
    oi_df.rename(columns={'create_time': 'timestamp', 'sum_open_interest_value': 'oi_usd'}, inplace=True)
    try: oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp']).astype(np.int64) // 10**6
    except: oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
    if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
    oi_df.set_index('timestamp', inplace=True)
    
    bb_fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
    bb_fr_files = [f for f in bb_fr_files if f.name >= start_date]
    dfs = []
    for f in bb_fr_files:
        try:
            df = pd.read_csv(f, engine='c')
            ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
            val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
            df = df[[ts_col, val_col]]
            df.columns = ['timestamp', 'funding_rate']
            dfs.append(df)
        except: pass
    fr_df = pd.concat(dfs, ignore_index=True)
    fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
    if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
    fr_df.set_index('timestamp', inplace=True)
    fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
    
    m1_df = kline_df.copy()
    m1_df.index = pd.to_datetime(m1_df.index, unit='ms')
    
    merged = kline_df.join(oi_df, how='left').join(fr_df, how='left')
    merged['oi_usd'] = merged['oi_usd'].ffill()
    merged['funding_rate'] = merged['funding_rate'].ffill()
    merged = merged.dropna(subset=['close', 'oi_usd', 'funding_rate'])
    merged = merged[~merged.index.duplicated(keep='last')]
    merged.index = pd.to_datetime(merged.index, unit='ms')
    
    # Original resampling from our backtests
    hourly = merged.resample('1h').agg({
        'close': 'last',
        'oi_usd': 'last',
        'funding_rate': 'last'
    }).dropna()
    
    hourly['oi_z'] = (hourly['oi_usd'] - hourly['oi_usd'].rolling(168).mean()) / hourly['oi_usd'].rolling(168).std()
    hourly['fr_z'] = (hourly['funding_rate'] - hourly['funding_rate'].rolling(168).mean()) / hourly['funding_rate'].rolling(168).std()
    
    hourly['signal'] = 0
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] > 2.0), 'signal'] = -1
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] < -2.0), 'signal'] = 1
    
    trades = hourly[hourly['signal'] != 0].copy()
    if len(trades) == 0: return
        
    first_trade_time = trades.index[0]
    first_trade = trades.iloc[0]
    
    print(f"\nAudit Trade Selected: Row {first_trade_time} | Signal: {first_trade['signal']}")
    
    # In default pandas resample('1h'), row '08:00:00' means data from 08:00:00 to 08:59:59.
    # Therefore, the signal is known at 08:59:59 (or 09:00:00).
    # The entry price in 'hourly' is the close of the 08:59:00 1-minute candle.
    
    print("\n--- Audit Point 1: Execution Price Alignment ---")
    print(f"Hourly Close Price (Assumed Entry Price): {first_trade['close']}")
    
    expected_entry_candle_time = first_trade_time + pd.Timedelta(minutes=59)
    raw_1m_at_execution = m1_df.loc[expected_entry_candle_time]
    print(f"Raw 1m Candle Close at {expected_entry_candle_time}: {raw_1m_at_execution['close']}")
    
    if abs(first_trade['close'] - raw_1m_at_execution['close']) < 0.01:
        print("✅ PASS: The hourly close perfectly matches the raw 1-minute close at the end of the hour.")
    else:
        print("❌ FAIL: Price mismatch.")
        
    print("\n--- Audit Point 2: Execution Path Future Peeking ---")
    # Our backtest engine does: start_time = entry_ts + pd.Timedelta(hours=1)
    start_time = first_trade_time + pd.Timedelta(hours=1)
    end_time = start_time + pd.Timedelta(hours=24)
    
    path = m1_df.loc[start_time:end_time]
    print(f"Signal Generation Known Time: {expected_entry_candle_time} (Closing at {start_time})")
    print(f"Execution Path Start (First tradable minute): {path.index[0]}")
    
    if path.index[0] == start_time:
        print("✅ PASS: The execution path strictly begins at the exact opening of the next hour. No future overlap.")
    else:
        print("❌ FAIL: The execution path overlaps with the signal data. Lookahead bias detected.")

audit_single_trade("BTCUSDT")
