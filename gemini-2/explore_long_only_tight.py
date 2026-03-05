import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
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
        
        if dfs_o:
            oi_df = pd.concat(dfs_o, ignore_index=True)
            oi_df['startTime'] = pd.to_numeric(oi_df['startTime'], errors='coerce')
            oi_df['oi'] = pd.to_numeric(oi_df['oi'], errors='coerce')
            oi_df['datetime'] = pd.to_datetime(oi_df['startTime'], unit='ms')
            oi_df = oi_df.set_index('datetime').sort_index()
            oi_df = oi_df[~oi_df.index.duplicated(keep='first')]
            oi_df = oi_df.resample('5min').last().ffill()
            df = pd.concat([kline_5m, oi_df['oi']], axis=1).ffill().dropna()
        else:
            return None
            
        df['ret_15m'] = df['close'].pct_change(3)
        df['oi_chg_15m'] = df['oi'].pct_change(3)
        
        # Stricter Long Signal: OI drops > 15%, Price drops > 8%
        df['sig_long'] = ((df['oi_chg_15m'] < -0.15) & (df['ret_15m'] < -0.08)).astype(int)
        
        trades = []
        
        # Parameters
        TP_LONG = 0.30
        SL_LONG = -0.10
        
        HOLD = 24 # 2 hours to avoid bleeding out
        FEE_BPS = 20
        
        indices = np.where(df['sig_long'] == 1)[0]
        for i in indices:
            if i + HOLD >= len(df): continue
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            path_df = df.iloc[i+1 : i+1+HOLD]
            highs = path_df['high'].values
            lows = path_df['low'].values
            closes = path_df['close'].values
            
            exit_price = closes[-1]
            exit_time = path_df.index[-1]
            
            for j in range(len(highs)):
                ret_h = (highs[j] - entry_price) / entry_price
                ret_l = (lows[j] - entry_price) / entry_price
                
                if ret_l <= SL_LONG:
                    exit_price = entry_price * (1 + SL_LONG)
                    exit_time = path_df.index[j]
                    break
                elif ret_h >= TP_LONG:
                    exit_price = entry_price * (1 + TP_LONG)
                    exit_time = path_df.index[j]
                    break
                    
            net_ret = (exit_price - entry_price) / entry_price - (FEE_BPS/10000)
            
            trades.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'net_ret': net_ret
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Testing ULTRA-STRICT Long Flush...")
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in tqdm(pool.imap_unordered(process_symbol, symbols), total=len(symbols)):
            if res_list:
                trades.extend(res_list)
                
    if not trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(trades).sort_values('entry_time')
    
    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    MAX_POSITIONS = 5
    pos_size = INITIAL_CAPITAL / MAX_POSITIONS
    
    events = []
    for idx, t in df_trades.iterrows():
        events.append({'time': t['entry_time'], 'type': 'enter', 'trade_idx': idx})
        events.append({'time': t['exit_time'], 'type': 'exit', 'trade_idx': idx})
        
    events.sort(key=lambda x: x['time'])
    
    active_trades = set()
    eq_curve = []
    portfolio_trades = []
    
    for ev in events:
        tid = ev['trade_idx']
        t = df_trades.loc[tid]
        
        if ev['type'] == 'enter':
            if len(active_trades) < MAX_POSITIONS:
                symbols_active = [df_trades.loc[at_id, 'symbol'] for at_id in active_trades]
                if t['symbol'] not in symbols_active:
                    active_trades.add(tid)
        elif ev['type'] == 'exit':
            if tid in active_trades:
                active_trades.remove(tid)
                pnl = pos_size * t['net_ret']
                capital += pnl
                eq_curve.append({'time': ev['time'], 'capital': capital})
                
                t_dict = t.to_dict()
                t_dict['pnl_usd'] = pnl
                portfolio_trades.append(t_dict)
                
    df_port = pd.DataFrame(portfolio_trades)
    df_eq = pd.DataFrame(eq_curve)
    
    total_ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
    
    print("\n=== ULTRA-STRICT LONG FLUSH ===")
    print(f"Entry: OI drops > 15%, Price drops > 8% in 15m")
    print(f"Exit: Hold 2h, TP 30%, SL 10%")
    print(f"Total Return: {total_ret_pct:.2f}%")
    if len(df_port) > 0:
        max_dd = ((df_eq['capital'].cummax() - df_eq['capital']) / df_eq['capital'].cummax()).max() * 100
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Trades Taken: {len(df_port)}")
        print(f"Win Rate: {(df_port['net_ret'] > 0).mean() * 100:.1f}%")
        print(f"Average Net Return per Trade: {df_port['net_ret'].mean() * 10000:.2f} bps")

