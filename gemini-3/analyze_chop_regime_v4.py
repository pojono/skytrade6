import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def extract_macro_state():
    print("Extracting BTC Macro State...")
    kline_files = sorted(list((DATALAKE / f"bybit/BTCUSDT").glob("*_kline_1m.csv")))
    kline_files = [f for f in kline_files if "2024-01-01" <= f.name[:10] <= "2024-12-31" and "mark" not in f.name and "index" not in f.name and "premium" not in f.name]
    
    dfs = []
    for f in kline_files:
        try: dfs.append(pd.read_csv(f, usecols=['startTime', 'close'], engine='c'))
        except: pass
        
    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    if df['timestamp'].max() < 1e11: df['timestamp'] *= 1000
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    
    hourly = df.resample('1h').agg({'close': 'last'}).dropna()
    
    hourly['sma_200'] = hourly['close'].rolling(200).mean()
    hourly['sma_200_slope'] = (hourly['sma_200'] - hourly['sma_200'].shift(24)) / hourly['sma_200'].shift(24) * 100
    hourly['hv_7d'] = hourly['close'].pct_change().rolling(168).std() * np.sqrt(24 * 365) * 100
    
    return hourly[['close', 'sma_200', 'sma_200_slope', 'hv_7d']]

if __name__ == "__main__":
    macro_state = extract_macro_state()
    
    from run_honest_oos import GOLDEN_CLUSTER, analyze_oos_symbol
    from multiprocessing import Pool
    import os
    
    print("Loading OOS trades...")
    all_trades = []
    with Pool(min(6, os.cpu_count() or 6)) as p:
        for t_list in p.imap_unordered(analyze_oos_symbol, GOLDEN_CLUSTER):
            if t_list: all_trades.extend(t_list)
            
    df_trades = pd.DataFrame(all_trades)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades = df_trades.sort_values('entry_time')
    
    df_trades = pd.merge_asof(df_trades, macro_state, left_on='entry_time', right_index=True, direction='backward')
    
    chop_trades = df_trades[(df_trades['entry_time'] >= '2024-08-01') & (df_trades['entry_time'] <= '2024-10-31')]
    other_trades = df_trades[~((df_trades['entry_time'] >= '2024-08-01') & (df_trades['entry_time'] <= '2024-10-31'))]
    
    print("\n--- Trade Environment Analysis ---")
    print(f"Non-Chop Trades Win Rate: {(other_trades['net_ret'] > 0).mean() * 100:.1f}% ({len(other_trades)} trades)")
    print(f"Q3 Chop Trades Win Rate: {(chop_trades['net_ret'] > 0).mean() * 100:.1f}% ({len(chop_trades)} trades)")
    
    print("\nAverage BTC Macro State during WINNING vs LOSING trades:")
    win_trades = df_trades[df_trades['net_ret'] > 0]
    loss_trades = df_trades[df_trades['net_ret'] <= 0]
    
    print(f"Winners: Avg BTC 200h Slope = {win_trades['sma_200_slope'].mean():.3f}% | Avg HV = {win_trades['hv_7d'].mean():.1f}%")
    print(f"Losers:  Avg BTC 200h Slope = {loss_trades['sma_200_slope'].mean():.3f}% | Avg HV = {loss_trades['hv_7d'].mean():.1f}%")
    
    print(f"\nWinners where HV < 45%: {len(win_trades[win_trades['hv_7d'] < 45])}")
    print(f"Losers where HV < 45%: {len(loss_trades[loss_trades['hv_7d'] < 45])}")
    
    # Test HV Filter
    filtered_trades = df_trades[df_trades['hv_7d'] >= 45.0]
    print(f"\n--- Applying Volatility Gate (HV >= 45%) ---")
    print(f"Remaining Trades: {len(filtered_trades)} (Filtered out {len(df_trades) - len(filtered_trades)})")
    print(f"New Win Rate: {(filtered_trades['net_ret'] > 0).mean() * 100:.1f}%")
    
    # Let's test a slope filter (absolute slope > 0.1% per day)
    filtered_trades_slope = df_trades[df_trades['sma_200_slope'].abs() >= 0.15]
    print(f"\n--- Applying Trend Velocity Gate (|SMA 200 Slope| >= 0.15%) ---")
    print(f"Remaining Trades: {len(filtered_trades_slope)} (Filtered out {len(df_trades) - len(filtered_trades_slope)})")
    print(f"New Win Rate: {(filtered_trades_slope['net_ret'] > 0).mean() * 100:.1f}%")

    # Combine both
    combined = df_trades[(df_trades['hv_7d'] >= 40.0) & (df_trades['sma_200_slope'].abs() >= 0.10)]
    print(f"\n--- Applying Combined Gate (HV >= 40 AND |Slope| >= 0.10%) ---")
    print(f"Remaining Trades: {len(combined)} (Filtered out {len(df_trades) - len(combined)})")
    print(f"New Win Rate: {(combined['net_ret'] > 0).mean() * 100:.1f}%")
    
    STARTING_CAPITAL = 1000.0
    RISK_PER_TRADE = 0.02 
    
    equity = STARTING_CAPITAL
    for idx, row in combined.iterrows():
        atr = row['atr_pct'] 
        size_multiplier = RISK_PER_TRADE / atr
        size_multiplier = min(size_multiplier, 3.0) 
        trade_return_pct = row['net_ret']
        pnl_usd = equity * size_multiplier * trade_return_pct
        equity += pnl_usd
        
    print(f"New Total Net Profit: ${equity - STARTING_CAPITAL:.2f} ({((equity - STARTING_CAPITAL)/STARTING_CAPITAL * 100):.2f}%)")
