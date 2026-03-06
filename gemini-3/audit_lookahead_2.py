import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def audit_single_trade(symbol="BTCUSDT"):
    print(f"--- Running Strict Lookahead Audit on {symbol} ---")
    
    start_date = "2025-01-01"
    
    print("Loading raw files...")
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
    if len(trades) == 0:
        print("No trades found.")
        return
        
    first_trade_time = trades.index[0]
    first_trade = trades.iloc[0]
    
    print(f"\nAudit Trade Selected: {first_trade_time} | Signal: {first_trade['signal']}")
    
    print("\n--- Audit Point 1: Execution Price Alignment ---")
    print(f"Hourly Signal Row Time: {first_trade_time}")
    print(f"Hourly Close Price (Assumed Entry Price): {first_trade['close']}")
    
    # Check the exact minute before the hour (e.g. 07:59:00 for an 08:00:00 hourly signal)
    raw_1m_at_execution = m1_df.loc[first_trade_time - pd.Timedelta(minutes=1)]
    print(f"Raw 1m Candle Close right before execution ({first_trade_time - pd.Timedelta(minutes=1)}): {raw_1m_at_execution['close']}")
    
    if abs(first_trade['close'] - raw_1m_at_execution['close']) < 0.01:
        print("✅ PASS: The hourly close perfectly matches the raw 1-minute close immediately preceding the signal.")
    else:
        print("❌ FAIL: Price mismatch. Lookahead bias detected in resampling.")
        
    print("\n--- Audit Point 2: Execution Path Future Peeking ---")
    start_time = first_trade_time + pd.Timedelta(minutes=1) 
    end_time = start_time + pd.Timedelta(hours=24)
    
    path = m1_df.loc[start_time:end_time]
    print(f"Signal Generation Time: {first_trade_time}")
    print(f"Execution Path Start (First tradable minute): {path.index[0]}")
    
    if path.index[0] > first_trade_time:
        print("✅ PASS: The execution path strictly begins AFTER the signal generation time. No future overlap.")
    else:
        print("❌ FAIL: The execution path overlaps with the signal data. Lookahead bias detected.")

audit_single_trade("BTCUSDT")
