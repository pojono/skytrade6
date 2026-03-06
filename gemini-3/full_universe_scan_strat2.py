import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings
from backtest_core_v2 import run_backtest_advanced

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
        # Load Binance Metrics (LS Ratio)
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None
        
        dfs = []
        for f in metrics_files:
            try: dfs.append(pd.read_csv(f, usecols=['create_time', 'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio'], engine='c'))
            except: pass
        if not dfs: return None
        
        m_df = pd.concat(dfs, ignore_index=True)
        m_df.rename(columns={'create_time': 'timestamp', 
                             'count_toptrader_long_short_ratio': 'count_ls',
                             'sum_toptrader_long_short_ratio': 'vol_ls'}, inplace=True)
        try: m_df['timestamp'] = pd.to_datetime(m_df['timestamp']).astype(np.int64) // 10**6
        except: m_df['timestamp'] = pd.to_numeric(m_df['timestamp'])
        if m_df['timestamp'].max() < 1e11: m_df['timestamp'] *= 1000
        m_df.set_index('timestamp', inplace=True)
        
        # Load Klines
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'high', 'low', 'close'], engine='c'))
            except: pass
        if not dfs: return None
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        # Merge and Resample to 5-minute
        merged = kline_df.join(m_df, how='left')
        merged['count_ls'] = merged['count_ls'].ffill()
        merged['vol_ls'] = merged['vol_ls'].ffill()
        merged = merged.dropna(subset=['close', 'count_ls', 'vol_ls'])
        merged = merged[~merged.index.duplicated(keep='last')]
        
        merged.index = pd.to_datetime(merged.index, unit='ms')
        
        # ATR logic
        prev_close = merged['close'].shift(1)
        tr1 = merged['high'] - merged['low']
        tr2 = (merged['high'] - prev_close).abs()
        tr3 = (merged['low'] - prev_close).abs()
        merged['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        m5 = merged.resample('5min').agg({
            'close': 'last',
            'count_ls': 'mean',
            'vol_ls': 'mean',
            'tr': 'sum'
        }).dropna()
        
        # Daily ATR in 5m periods = 288 periods
        m5['atr_pct'] = (m5['tr'].rolling(288).mean() * 288) / m5['close']
        
        return m5
    except: return None

def analyze_symbol(symbol):
    df = load_data(symbol)
    if df is None or len(df) < 2000: return None
    
    # 24-hour rolling window (288 periods for 5-min data)
    df['count_z'] = (df['count_ls'] - df['count_ls'].rolling(288).mean()) / df['count_ls'].rolling(288).std()
    df['vol_z'] = (df['vol_ls'] - df['vol_ls'].rolling(288).mean()) / df['vol_ls'].rolling(288).std()
    
    # Momentum filter (6 hours = 72 periods)
    df['mom_6h'] = df['close'] / df['close'].shift(72) - 1
    
    df['signal'] = 0
    # Bearish Divergence
    df.loc[(df['count_z'] > 2.0) & (df['vol_z'] < -2.0) & (df['mom_6h'] < 0.05), 'signal'] = -1
    # Bullish Divergence
    df.loc[(df['count_z'] < -2.0) & (df['vol_z'] > 2.0) & (df['mom_6h'] > -0.05), 'signal'] = 1
    
    df['signal_shifted'] = df['signal'].shift(1).fillna(0)
    df.loc[df['signal'] == df['signal_shifted'], 'signal'] = 0
    
    # Hold for 4 hours (48 periods), 5bps fee, Vol sizing ON
    # Trailing stop 300 bps (3%), TP 600 bps (6%)
    results = run_backtest_advanced(df, max_hold_periods=48, trailing_stop_bps=300.0, take_profit_bps=600.0, fee_bps=5.0, use_volatility_sizing=True)
    
    if results['events'] > 0:
        results['symbol'] = symbol
        return results
    return None

if __name__ == "__main__":
    symbols = get_all_symbols('binance')
    print(f"--- Running Whale Shadow V2 Full Universe Scan ({len(symbols)} coins) ---")
    
    res_list = []
    with Pool(min(8, os.cpu_count() or 8)) as p:
        for i, r in enumerate(p.imap_unordered(analyze_symbol, symbols)):
            if r: res_list.append(r)
            if (i+1) % 20 == 0: print(f"Processed {i+1}/{len(symbols)}...")
            
    if res_list:
        res_df = pd.DataFrame(res_list)
        res_df = res_df[res_df['events'] >= 10]
        res_df = res_df.sort_values('total_net_ret_%', ascending=False)
        
        cols = ['symbol', 'events', 'win_rate', 'total_net_ret_%', 'avg_net_ret_bps', 'max_drawdown_%', 'sharpe']
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        print(res_df[cols].to_string(index=False))
        print("\n--- Aggregate Portfolio Performance ---")
        print(f"Total Trades: {res_df['events'].sum()}")
        print(f"Avg Win Rate: {res_df['win_rate'].mean():.2%}")
        print(f"Total Net Return: {res_df['total_net_ret_%'].sum():.2f}%")
        print(f"Avg Sharpe: {res_df['sharpe'].mean():.2f}")
