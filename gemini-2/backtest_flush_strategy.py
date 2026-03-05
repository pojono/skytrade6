import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import warnings
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
        kline = kline.set_index('datetime').resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
        
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
            oi_df = oi_df.set_index('datetime').resample('5min').last().ffill()
            df = pd.concat([kline, oi_df['oi']], axis=1).ffill().dropna()
        else:
            return None
            
        df['ret_15m'] = df['close'].pct_change(3)
        df['oi_chg_15m'] = df['oi'].pct_change(3)
        
        # Targets
        df['fwd_ret_2h'] = df['close'].shift(-24) / df['close'] - 1
        df['fwd_ret_4h'] = df['close'].shift(-48) / df['close'] - 1
        
        df = df.dropna(subset=['fwd_ret_4h', 'ret_15m', 'oi_chg_15m'])
        
        # Extreme Flush
        mask_extreme = (df['oi_chg_15m'] < -0.10) & (df['ret_15m'] < -0.05)
        
        trades = []
        for idx, row in df[mask_extreme].iterrows():
            trades.append({
                'symbol': symbol,
                'entry_time': idx,
                'entry_price': row['close'],
                'exit_time': idx + pd.Timedelta(hours=4),
                'exit_price': df.loc[idx:].iloc[48]['close'] if len(df.loc[idx:]) > 48 else np.nan,
                'ret_15m': row['ret_15m'],
                'oi_chg_15m': row['oi_chg_15m'],
                'fwd_ret_4h': row['fwd_ret_4h']
            })

        return trades
    except Exception as e:
        return None

if __name__ == '__main__':
    symbols = get_all_symbols()
    print(f"Running backtest on {len(symbols)} symbols...")
    
    with Pool(processes=12) as pool:
        all_results = pool.map(process_symbol, symbols)
        
    trades = []
    for res_list in all_results:
        if res_list:
            trades.extend(res_list)
            
    df_trades = pd.DataFrame(trades).dropna()
    df_trades = df_trades.sort_values('entry_time')
    
    # Portfolio Simulation
    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    MAX_POSITIONS = 5 # Max concurrent open positions
    POSITION_SIZE = INITIAL_CAPITAL / MAX_POSITIONS # USD per trade
    FEE_BPS = 20 # taker in + taker out
    
    open_positions = []
    executed_trades = []
    equity_curve = []
    
    # We will iterate through trades in order of entry time
    # To properly simulate, we need a timeline. Since trades are just entry/exit, 
    # we can process them as events.
    events = []
    for _, t in df_trades.iterrows():
        events.append({'time': t['entry_time'], 'type': 'entry', 'trade': t})
        events.append({'time': t['exit_time'], 'type': 'exit', 'trade': t})
        
    events.sort(key=lambda x: x['time'])
    
    active_trades = set()
    current_invested = 0
    
    for ev in events:
        t = ev['trade']
        trade_id = f"{t['symbol']}_{t['entry_time']}"
        
        if ev['type'] == 'entry':
            if len(active_trades) < MAX_POSITIONS:
                # We can take this trade
                active_trades.add(trade_id)
                # Not doing complex compounding per trade, just a fixed size for simplicity
                # but we will track total PnL
            else:
                pass # Skipped due to lack of capital
        elif ev['type'] == 'exit':
            if trade_id in active_trades:
                # Close trade
                active_trades.remove(trade_id)
                
                # Calculate PnL
                raw_ret = (t['exit_price'] - t['entry_price']) / t['entry_price']
                net_ret = raw_ret - (FEE_BPS / 10000)
                pnl_usd = POSITION_SIZE * net_ret
                
                capital += pnl_usd
                
                t_exec = t.copy()
                t_exec['net_ret'] = net_ret
                t_exec['pnl_usd'] = pnl_usd
                executed_trades.append(t_exec)
                
                equity_curve.append({'time': ev['time'], 'capital': capital})
                
    exec_df = pd.DataFrame(executed_trades)
    eq_df = pd.DataFrame(equity_curve)
    
    print("\n=== Backtest Results (Portfolio constraints: Max 5 positions) ===")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital: ${capital:,.2f}")
    print(f"Total Return: {(capital / INITIAL_CAPITAL - 1) * 100:.2f}%")
    
    if len(exec_df) > 0:
        print(f"Total Trades Executed: {len(exec_df)} (out of {len(df_trades)} signals)")
        print(f"Win Rate: {(exec_df['net_ret'] > 0).mean() * 100:.1f}%")
        print(f"Average Net Return per Trade: {exec_df['net_ret'].mean() * 10000:.2f} bps")
        print(f"Max Drawdown of Capital: {((eq_df['capital'].cummax() - eq_df['capital']) / eq_df['capital'].cummax()).max() * 100:.2f}%")
    else:
        print("No trades executed.")
        
    print("\nNote: The Extreme Flush strategy focuses on capturing rapid mean reversion after forced liquidations. Holding period fixed at 4 hours.")

