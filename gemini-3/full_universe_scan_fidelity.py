import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings
from backtest_core_high_fidelity import run_high_fidelity_backtest

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def get_all_symbols(exchange='binance'):
    symbols = []
    base_dir = DATALAKE / exchange
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.endswith('USDT'):
                symbols.append(item.name)
    return sorted(symbols)

def load_data(symbol, start_date="2025-01-01"):
    try:
        # Load Binance Metrics (Open Interest)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None, None
        
        dfs = []
        for f in metrics_files:
            try: dfs.append(pd.read_csv(f, usecols=['create_time', 'sum_open_interest_value'], engine='c'))
            except: pass
        if not dfs: return None, None
        
        oi_df = pd.concat(dfs, ignore_index=True)
        oi_df.rename(columns={'create_time': 'timestamp', 'sum_open_interest_value': 'oi_usd'}, inplace=True)
        # Fix datetime parsing
        try:
            oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp']).astype(np.int64) // 10**6
        except:
            oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
        if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
        oi_df.set_index('timestamp', inplace=True)
        
        # Load Bybit Funding Rate
        bb_fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
        bb_fr_files = [f for f in bb_fr_files if f.name >= start_date]
        if not bb_fr_files: return None, None
        
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
        if not dfs: return None, None
        fr_df = pd.concat(dfs, ignore_index=True)
        fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
        if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
        fr_df.set_index('timestamp', inplace=True)
        fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
        
        # Load Klines
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None, None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'high', 'low', 'close'], engine='c'))
            except: pass
        if not dfs: return None, None
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        m1_df = kline_df.copy() # Save the high resolution 1m data for the executor
        m1_df.index = pd.to_datetime(m1_df.index, unit='ms')
        
        # Merge hourly signals
        merged = kline_df.join(oi_df, how='left').join(fr_df, how='left')
        merged['oi_usd'] = merged['oi_usd'].ffill()
        merged['funding_rate'] = merged['funding_rate'].ffill()
        merged = merged.dropna(subset=['close', 'oi_usd', 'funding_rate'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        # Convert merged index to datetime for resampling
        merged.index = pd.to_datetime(merged.index, unit='ms')
        
        prev_close = merged['close'].shift(1)
        tr1 = merged['high'] - merged['low']
        tr2 = (merged['high'] - prev_close).abs()
        tr3 = (merged['low'] - prev_close).abs()
        merged['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        hourly = merged.resample('1h').agg({
            'close': 'last',
            'oi_usd': 'last',
            'funding_rate': 'last',
            'tr': 'sum'
        }).dropna()
        
        hourly['atr_pct'] = (hourly['tr'].rolling(24).mean() * 24) / hourly['close']
        
        return hourly, m1_df
    except: return None, None

def analyze_symbol(symbol):
    hourly, m1_df = load_data(symbol)
    if hourly is None or m1_df is None or len(hourly) < 500: return None
    
    # Generate Signals
    hourly['oi_z'] = (hourly['oi_usd'] - hourly['oi_usd'].rolling(168).mean()) / hourly['oi_usd'].rolling(168).std()
    hourly['fr_z'] = (hourly['funding_rate'] - hourly['funding_rate'].rolling(168).mean()) / hourly['funding_rate'].rolling(168).std()
    
    hourly['signal'] = 0
    # Powder Keg -> Short
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] > 2.0), 'signal'] = -1
    # Despair Pit -> Long
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] < -2.0), 'signal'] = 1
    
    hourly['signal_shifted'] = hourly['signal'].shift(1).fillna(0)
    hourly.loc[hourly['signal'] == hourly['signal_shifted'], 'signal'] = 0
    
    # Run High Fidelity Backtest using 1m High/Low prices
    # Trailing Stop: 400 bps (4%), TP: 1200 bps (12%)
    results = run_high_fidelity_backtest(hourly, m1_df, max_hold_hours=48, trailing_stop_bps=400.0, take_profit_bps=1200.0, fee_bps=5.0, use_volatility_sizing=True)
    
    if results['events'] > 0:
        results['symbol'] = symbol
        return results
    return None

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SUIUSDT', 'WLDUSDT', 'LINKUSDT', 'DYDXUSDT', 'ENAUSDT']
    print(f"--- High Fidelity Tick/1M Trailing Stop Validation ({len(symbols)} coins) ---")
    
    res_list = []
    with Pool(min(4, os.cpu_count() or 4)) as p:
        for i, r in enumerate(p.imap_unordered(analyze_symbol, symbols)):
            if r: res_list.append(r)
            if (i+1) % 2 == 0: print(f"Processed {i+1}/{len(symbols)}...")
            
    if res_list:
        res_df = pd.DataFrame(res_list)
        res_df = res_df.sort_values('total_net_ret_%', ascending=False)
        
        cols = ['symbol', 'events', 'win_rate', 'total_net_ret_%', 'avg_net_ret_bps', 'max_drawdown_%', 'sharpe']
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(res_df[cols].to_string(index=False))
        print("\n--- Aggregate Portfolio Performance ---")
        print(f"Total Trades: {res_df['events'].sum()}")
        print(f"Avg Win Rate: {res_df['win_rate'].mean():.2%}")
        print(f"Total Net Return: {res_df['total_net_ret_%'].sum():.2f}%")
        print(f"Avg Sharpe: {res_df['sharpe'].mean():.2f}")

