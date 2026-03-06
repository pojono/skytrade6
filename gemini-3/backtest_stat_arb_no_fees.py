import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_pair_data(symbol, start_date="2025-01-01"):
    sources = {
        'binance': (DATALAKE / "binance" / symbol, "*_kline_1m.csv", 'open_time', 'close'),
        'bybit': (DATALAKE / "bybit" / symbol, "*_kline_1m.csv", 'startTime', 'close')
    }
    
    dfs = []
    for name, (path_dir, pattern, time_col, close_col) in sources.items():
        files = sorted(list(path_dir.glob(pattern)))
        files = [f for f in files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name]
        files = [f for f in files if f.name >= start_date]
        
        daily_dfs = []
        for f in files:
            try:
                df_day = pd.read_csv(f, usecols=[time_col, close_col], engine='c')
                daily_dfs.append(df_day)
            except:
                pass
                
        if not daily_dfs:
            return None
            
        df = pd.concat(daily_dfs, ignore_index=True)
        df = df.rename(columns={time_col: 'timestamp', close_col: name})
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        
        if df['timestamp'].max() < 1e11:
            df['timestamp'] = df['timestamp'] * 1000
        elif df['timestamp'].max() > 1e14:
            df['timestamp'] = df['timestamp'] // 1000
            
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        dfs.append(df)
        
    if len(dfs) != 2:
        return None
        
    merged = pd.concat(dfs, axis=1, join='inner').dropna()
    return merged

def backtest_stat_arb(df, z_entry=2.0, z_exit=0.0, window=60):
    # Calculate log prices
    df['log_binance'] = np.log(df['binance'])
    df['log_bybit'] = np.log(df['bybit'])
    
    # Spread is the log difference
    df['spread'] = df['log_binance'] - df['log_bybit']
    
    # Calculate rolling mean and std
    df['spread_mean'] = df['spread'].rolling(window=window).mean()
    df['spread_std'] = df['spread'].rolling(window=window).std()
    
    # Z-score of spread
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    
    position = 0
    entry_price_binance = 0
    entry_price_bybit = 0
    
    trades = []
    
    zscores = df['zscore'].values
    binance_px = df['binance'].values
    bybit_px = df['bybit'].values
    timestamps = df.index.values
    
    for i in range(window, len(df)):
        z = zscores[i]
        
        if np.isnan(z):
            continue
            
        # Entry logic
        if position == 0:
            if z > z_entry:
                position = -1
                entry_price_binance = binance_px[i]
                entry_price_bybit = bybit_px[i]
                entry_time = timestamps[i]
            elif z < -z_entry:
                position = 1
                entry_price_binance = binance_px[i]
                entry_price_bybit = bybit_px[i]
                entry_time = timestamps[i]
                
        # Exit logic
        elif position == -1:
            if z <= z_exit:
                ret_binance = (entry_price_binance - binance_px[i]) / entry_price_binance 
                ret_bybit = (bybit_px[i] - entry_price_bybit) / entry_price_bybit
                trade_ret = (ret_binance + ret_bybit) / 2.0 
                
                trades.append({
                    'type': 'short_spread',
                    'return': trade_ret,
                    'hold_time': (timestamps[i] - entry_time) / 60000 
                })
                position = 0
                
        elif position == 1:
            if z >= -z_exit:
                ret_binance = (binance_px[i] - entry_price_binance) / entry_price_binance 
                ret_bybit = (entry_price_bybit - bybit_px[i]) / entry_price_bybit 
                trade_ret = (ret_binance + ret_bybit) / 2.0
                
                trades.append({
                    'type': 'long_spread',
                    'return': trade_ret,
                    'hold_time': (timestamps[i] - entry_time) / 60000
                })
                position = 0
                
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return {'num_trades': 0, 'win_rate': 0, 'total_return': 0, 'avg_return': 0}
        
    return {
        'num_trades': len(trades_df),
        'win_rate': (trades_df['return'] > 0).mean(),
        'total_return': trades_df['return'].sum(),
        'avg_return': trades_df['return'].mean(),
        'avg_hold_mins': trades_df['hold_time'].mean()
    }

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'DYDXUSDT', 'WLDUSDT']
    
    results = []
    print("Running without fees to find gross edge...")
    for sym in symbols:
        df = load_pair_data(sym)
        if df is not None:
            res = backtest_stat_arb(df, z_entry=2.5, z_exit=0.0, window=60)
            res['symbol'] = sym
            results.append(res)
            print(f"{sym}: {res['num_trades']} trades | Gross WR: {res['win_rate']:.1%} | Gross Net: {res['total_return']*100:.2f}% | Gross Avg: {res['avg_return']*10000:.2f} bps | Avg Hold: {res['avg_hold_mins']:.1f}m")
            
    res_df = pd.DataFrame(results)
    print("\n--- Summary (NO FEES) ---")
    print(res_df.to_string())
