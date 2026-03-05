import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

def get_all_symbols():
    return sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])

def process_symbol(symbol):
    try:
        files_kline = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
        if not files_kline: return None
        
        dfs_k = []
        for f in files_kline:
            try:
                df = pd.read_csv(f, usecols=['startTime', 'open', 'high', 'low', 'close', 'volume'])
                dfs_k.append(df)
            except: pass
        if not dfs_k: return None
        kline = pd.concat(dfs_k, ignore_index=True).dropna()
        kline['startTime'] = pd.to_numeric(kline['startTime'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            kline[col] = pd.to_numeric(kline[col], errors='coerce')
        kline = kline[kline['close'] > 0]
        kline['datetime'] = pd.to_datetime(kline['startTime'], unit='ms')
        kline = kline.set_index('datetime').sort_index()
        kline = kline[~kline.index.duplicated(keep='first')]
        
        # We need 5m for signals, 1m for precise execution
        kline_5m = kline.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
        
        files_oi = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_open_interest_5min.csv")
        dfs_o = []
        for f in files_oi:
            try:
                df = pd.read_csv(f)
                ts_col = 'timestamp' if 'timestamp' in df.columns else 'startTime' if 'startTime' in df.columns else None
                oi_col = 'openInterest' if 'openInterest' in df.columns else 'OpenInterest' if 'OpenInterest' in df.columns else None
                if not ts_col or not oi_col: continue
                df = df[[ts_col, oi_col]].dropna()
                df.columns = ['startTime', 'oi']
                dfs_o.append(df)
            except: pass
        
        if not dfs_o: return None
        
        oi_df = pd.concat(dfs_o, ignore_index=True)
        oi_df['startTime'] = pd.to_numeric(oi_df['startTime'], errors='coerce')
        oi_df['oi'] = pd.to_numeric(oi_df['oi'], errors='coerce')
        oi_df['datetime'] = pd.to_datetime(oi_df['startTime'], unit='ms')
        oi_df = oi_df.set_index('datetime').sort_index()
        oi_df = oi_df[~oi_df.index.duplicated(keep='first')]
        oi_df = oi_df.resample('5min').last().ffill()
        
        df_sig = pd.concat([kline_5m, oi_df['oi']], axis=1).ffill().dropna()
        
        # Calculate signal parameters exactly on past data
        df_sig['ret_15m'] = df_sig['close'] / df_sig['close'].shift(3) - 1
        df_sig['oi_chg_15m'] = df_sig['oi'] / df_sig['oi'].shift(3) - 1
        
        # Signal triggers at the end of the candle where condition is met
        df_sig['signal'] = ((df_sig['oi_chg_15m'] < -0.10) & (df_sig['ret_15m'] < -0.05)).astype(int)
        
        trades = []
        signal_times = df_sig[df_sig['signal'] == 1].index
        
        for t_sig in signal_times:
            # t_sig is the timestamp of the 5m candle close. e.g. 10:05:00
            # Our entry will be exactly at this close price
            entry_price = df_sig.loc[t_sig, 'close']
            
            # Slice 1m data starting from t_sig + 1min for realistic path simulation
            # (No lookahead: we start looking from the very next minute)
            end_time = t_sig + pd.Timedelta(hours=4)
            path_df = kline.loc[t_sig + pd.Timedelta(minutes=1) : end_time]
            
            if len(path_df) == 0: continue
            
            trades.append({
                'symbol': symbol,
                'entry_time': t_sig,
                'entry_price': entry_price,
                'path_high': path_df['high'].values.tolist(),
                'path_low': path_df['low'].values.tolist(),
                'path_close': path_df['close'].values.tolist(),
                'path_times': path_df.index.tolist()
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Running strict 1m path data extraction on {len(symbols)} symbols...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols), desc="Extracting"):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    print(f"Found {len(trades)} trades. Running exact execution simulator...")
    
    FEE_BPS = 20
    TP = 0.20
    SL = -0.10
    
    executed_trades = []
    
    for t in trades:
        ep = t['entry_price']
        highs = t['path_high']
        lows = t['path_low']
        closes = t['path_close']
        times = t['path_times']
        
        exit_price = None
        exit_time = None
        exit_reason = None
        
        for i in range(len(highs)):
            h = highs[i]
            l = lows[i]
            
            ret_h = (h - ep) / ep
            ret_l = (l - ep) / ep
            
            if ret_l <= SL:
                exit_price = ep * (1 + SL)
                exit_time = times[i]
                exit_reason = 'SL'
                break
            elif ret_h >= TP:
                exit_price = ep * (1 + TP)
                exit_time = times[i]
                exit_reason = 'TP'
                break
                
        # If no TP/SL hit, exit at the end of the 4h window
        if exit_price is None:
            exit_price = closes[-1]
            exit_time = times[-1]
            exit_reason = 'Time'
            
        net_ret = (exit_price - ep) / ep - (FEE_BPS/10000)
        
        executed_trades.append({
            'symbol': t['symbol'],
            'entry_time': t['entry_time'],
            'exit_time': exit_time,
            'entry_price': ep,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'net_ret': net_ret
        })
        
    df_exec = pd.DataFrame(executed_trades).sort_values('entry_time')
    
    print("\n=== Robust Backtest Results (1m path precision) ===")
    print(f"Total Trades: {len(df_exec)}")
    print(f"Win Rate: {(df_exec['net_ret'] > 0).mean() * 100:.1f}%")
    print(f"Mean Net Return per Trade: {df_exec['net_ret'].mean() * 10000:.2f} bps")
    print(f"Exit Reasons: {df_exec['exit_reason'].value_counts().to_dict()}")
    
    # Analyze clustering
    df_exec['date'] = df_exec['entry_time'].dt.date
    daily_counts = df_exec.groupby('date').size()
    print(f"Trades occurred on {len(daily_counts)} unique days out of dataset.")
    print(f"Top 5 days by trade count:\n{daily_counts.sort_values(ascending=False).head(5)}")
    
    # Portfolio Sim - 10k capital, max 5 positions
    # To properly simulate, we do a chronological walk
    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    MAX_POSITIONS = 5
    pos_size = INITIAL_CAPITAL / MAX_POSITIONS
    
    events = []
    for idx, t in df_exec.iterrows():
        events.append({'time': t['entry_time'], 'type': 'enter', 'trade_idx': idx})
        events.append({'time': t['exit_time'], 'type': 'exit', 'trade_idx': idx})
        
    events.sort(key=lambda x: x['time'])
    
    active_trades = set()
    eq_curve = []
    portfolio_trades = []
    
    for ev in events:
        tid = ev['trade_idx']
        t = df_exec.loc[tid]
        
        if ev['type'] == 'enter':
            if len(active_trades) < MAX_POSITIONS:
                # To prevent same symbol from filling all slots simultaneously
                symbols_active = [df_exec.loc[at_id, 'symbol'] for at_id in active_trades]
                if t['symbol'] not in symbols_active:
                    active_trades.add(tid)
        elif ev['type'] == 'exit':
            if tid in active_trades:
                active_trades.remove(tid)
                pnl = pos_size * t['net_ret']
                capital += pnl
                eq_curve.append({'time': ev['time'], 'capital': capital})
                
                t_copy = t.to_dict()
                t_copy['pnl_usd'] = pnl
                portfolio_trades.append(t_copy)
                
    df_port = pd.DataFrame(portfolio_trades)
    df_eq = pd.DataFrame(eq_curve)
    
    print("\n=== Realistic Portfolio Simulation ===")
    print(f"Max Positions: {MAX_POSITIONS}, Max 1 per symbol, Fixed Size: ${pos_size:,.2f}")
    print(f"Total Return: {(capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
    if len(df_port) > 0:
        print(f"Trades Taken: {len(df_port)}")
        print(f"Portfolio Win Rate: {(df_port['net_ret'] > 0).mean() * 100:.1f}%")
        max_dd = ((df_eq['capital'].cummax() - df_eq['capital']) / df_eq['capital'].cummax()).max() * 100
        print(f"Max Drawdown: {max_dd:.2f}%")
    
    df_exec.to_csv('/home/ubuntu/Projects/skytrade6/gemini-2/flush_trades.csv', index=False)

