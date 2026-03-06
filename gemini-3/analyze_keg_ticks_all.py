import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_trades_around_event(symbol, event_time_str):
    event_time = pd.to_datetime(event_time_str)
    date_str = event_time.strftime('%Y-%m-%d')
    
    tick_file = DATALAKE / f"bybit/{symbol}/{date_str}_trades.csv.gz"
    if not tick_file.exists(): return None
        
    try:
        df = pd.read_csv(tick_file, usecols=['timestamp', 'side', 'size', 'price'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        start_time = event_time
        end_time = event_time + pd.Timedelta(hours=24)
        
        next_day = end_time.strftime('%Y-%m-%d')
        if next_day != date_str:
            next_file = DATALAKE / f"bybit/{symbol}/{next_day}_trades.csv.gz"
            if next_file.exists():
                df_next = pd.read_csv(next_file, usecols=['timestamp', 'side', 'size', 'price'])
                df_next['datetime'] = pd.to_datetime(df_next['timestamp'], unit='s')
                df_next.set_index('datetime', inplace=True)
                df_next.sort_index(inplace=True)
                df = pd.concat([df, df_next])
        
        window_df = df.loc[start_time:end_time].copy()
        return window_df if not window_df.empty else None
    except: return None

def process_single_event(row):
    symbol = row['symbol']
    event_time = row['timestamp']
    signal = row['signal']
    
    ticks = load_trades_around_event(symbol, event_time)
    if ticks is None or ticks.empty: return None
        
    ticks['signed_vol'] = np.where(ticks['side'] == 'Buy', ticks['size'], -ticks['size'])
    
    sec_df = ticks.resample('1s').agg({'price': 'last', 'size': 'sum', 'signed_vol': 'sum'}).ffill()
    
    # 10s rolling CVD on 1s bins for fast ignition
    sec_df['vol_imb_10s'] = sec_df['signed_vol'].rolling(10).sum() 
    sec_df['total_vol_10s'] = sec_df['size'].rolling(10).sum()
    sec_df['imb_ratio'] = sec_df['vol_imb_10s'] / (sec_df['total_vol_10s'] + 1e-8)
    
    entry_idx = None
    entry_price = 0
    
    # Threshold: >85% one-sided volume over a 10s burst
    for idx, sec_row in sec_df.iterrows():
        if pd.isna(sec_row['imb_ratio']) or sec_row['total_vol_10s'] == 0: continue
            
        if signal == -1 and sec_row['imb_ratio'] < -0.85:
            entry_idx = idx
            entry_price = sec_row['price']
            break
        elif signal == 1 and sec_row['imb_ratio'] > 0.85:
            entry_idx = idx
            entry_price = sec_row['price']
            break
            
    if entry_idx is None: return None
        
    # Simulate 30ms latency fill
    fill_time = entry_idx + pd.Timedelta(milliseconds=30)
    post_latency_ticks = ticks.loc[fill_time:]
    if post_latency_ticks.empty: return None
        
    actual_fill_price = post_latency_ticks.iloc[0]['price']
    latency_slippage = (actual_fill_price - entry_price) / entry_price * signal
    
    # Hold for 30 mins
    exit_time = fill_time + pd.Timedelta(minutes=30)
    post_trade = ticks.loc[fill_time:exit_time]
    
    if post_trade.empty: return None
        
    max_pnl, min_pnl = 0.0, 0.0
    exit_price = post_trade.iloc[-1]['price']
    exit_reason = "Time (30m)"
    
    prices = post_trade['price'].values
    pnls = (prices - actual_fill_price) / actual_fill_price * signal
    
    # Let the runner run, cut the losers fast
    for i, pnl in enumerate(pnls):
        if pnl > max_pnl: max_pnl = pnl
        if pnl < min_pnl: min_pnl = pnl
        
        if pnl <= -0.01: # 1% SL
            exit_price = prices[i]
            exit_reason = "Stop Loss (1%)"
            break
        if pnl >= 0.03: # 3% TP
            exit_price = prices[i]
            exit_reason = "Take Profit (3%)"
            break
            
    # Include 10bps total taker fees
    net_ret = ((exit_price - actual_fill_price) / actual_fill_price * signal) - 0.001
    
    return {
        'symbol': symbol,
        'signal': signal,
        'entry_time': fill_time,
        'fill_price': actual_fill_price,
        'slippage_bps': latency_slippage * 10000,
        'exit_reason': exit_reason,
        'net_ret_pct': net_ret * 100,
        'max_runup_pct': max_pnl * 100,
        'max_dd_pct': min_pnl * 100
    }

def run_all():
    events = pd.read_csv("keg_events_all_relaxed.csv")
    print(f"Loaded {len(events)} events for massively scaled backtest.")
    rows = [row for _, row in events.iterrows()]
    
    results = []
    # Use max CPU for the massive tick crunch
    with Pool(min(32, os.cpu_count() or 16)) as p:
        for r in p.imap_unordered(process_single_event, rows):
            if r is not None:
                results.append(r)
                
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv("massive_hft_results.csv", index=False)
        print("\n=== MASSIVE SCALE HFT SNIPER (141 Coins, >3000 Setups) ===")
        print(f"Total Triggers Hit: {len(res_df)}")
        print(f"Average Slippage (30ms latency): {res_df['slippage_bps'].mean():.2f} bps")
        print(f"Win Rate (>0% Net Return): {(res_df['net_ret_pct'] > 0).mean()*100:.1f}%")
        print(f"Total Net Return: {res_df['net_ret_pct'].sum():.2f}%")
        print(f"Average Return per Trade: {res_df['net_ret_pct'].mean():.2f}%")
        print(f"Sharpe Ratio (per trade): {res_df['net_ret_pct'].mean() / res_df['net_ret_pct'].std() * np.sqrt(len(res_df)):.2f}")
        
        print("\nExit Reasons:")
        print(res_df['exit_reason'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

if __name__ == "__main__":
    run_all()
