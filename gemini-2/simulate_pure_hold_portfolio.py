import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"

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
        
        # Original baseline
        df['sig_long'] = ((df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.05)).astype(int)
        
        trades = []
        indices = np.where(df['sig_long'] == 1)[0]
        
        HOLD = 24 # 2 hours
        FEE_BPS = 20
        
        for i in indices:
            if i + HOLD >= len(df): continue 
            
            entry_time = df.index[i]
            entry_price = df['close'].iloc[i]
            
            fwd_window = df.iloc[i+1 : i+1+HOLD]
            exit_price = fwd_window['close'].iloc[-1]
            exit_time = fwd_window.index[-1]
            
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
    symbols = sorted([d for d in os.listdir(DATALAKE_DIR) if os.path.isdir(os.path.join(DATALAKE_DIR, d)) and d.endswith('USDT')])
    
    trades = []
    with Pool(processes=12) as pool:
        for res_list in pool.imap_unordered(process_symbol, symbols):
            if res_list: trades.extend(res_list)
                
    df_trades = pd.DataFrame(trades).sort_values('entry_time')
    
    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    MAX_POSITIONS = 5
    
    events = []
    for idx, t in df_trades.iterrows():
        events.append({'time': t['entry_time'], 'type': 'enter', 'trade_idx': idx})
        events.append({'time': t['exit_time'], 'type': 'exit', 'trade_idx': idx})
        
    events.sort(key=lambda x: x['time'])
    
    active_trades = {}
    eq_curve = []
    portfolio_trades = []
    
    for ev in events:
        tid = ev['trade_idx']
        t = df_trades.loc[tid]
        
        if ev['type'] == 'enter':
            if len(active_trades) < MAX_POSITIONS:
                symbols_active = [df_trades.loc[at_id, 'symbol'] for at_id in active_trades.keys()]
                if t['symbol'] not in symbols_active:
                    pos_size = capital / MAX_POSITIONS # Compounding
                    active_trades[tid] = pos_size
        elif ev['type'] == 'exit':
            if tid in active_trades:
                pos_size = active_trades.pop(tid)
                pnl = pos_size * t['net_ret']
                capital += pnl
                eq_curve.append({'time': ev['time'], 'capital': capital})
                
                tr_dict = t.to_dict()
                tr_dict['pnl_usd'] = pnl
                tr_dict['pos_size'] = pos_size
                portfolio_trades.append(tr_dict)
                
    df_port = pd.DataFrame(portfolio_trades)
    df_eq = pd.DataFrame(eq_curve)
    
    total_ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
    max_dd = ((df_eq['capital'].cummax() - df_eq['capital']) / df_eq['capital'].cummax()).max() * 100
    
    print("\n=== PURE HOLD 2 HOURS (NO SL/TP) ===")
    print(f"Total Trades Taken: {len(df_port)}")
    print(f"Win Rate: {(df_port['net_ret'] > 0).mean() * 100:.1f}%")
    print(f"Mean Net Return per Trade: {df_port['net_ret'].mean() * 10000:.2f} bps")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital: ${capital:,.2f}")
    print(f"Total Return: {total_ret_pct:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}%")
    
    df_port['month'] = pd.to_datetime(df_port['entry_time']).dt.to_period('M')
    monthly = df_port.groupby('month').agg(
        trades=('pnl_usd', 'count'),
        win_rate=('net_ret', lambda x: (x > 0).mean() * 100),
        net_pnl_usd=('pnl_usd', 'sum')
    ).reset_index()

    print("\n--- Monthly Breakdown ---")
    print(monthly.to_string(index=False))

    plt.figure(figsize=(12, 6))
    plt.plot(df_eq['time'], df_eq['capital'], label='Equity Curve', color='blue', linewidth=2)
    plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital', alpha=0.7)
    plt.title('Pure Hold 2H (No SL/TP) - Compounded Equity', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Account Balance ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Projects/skytrade6/gemini-2/equity_curve_pure_hold.png')

