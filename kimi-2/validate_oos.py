#!/usr/bin/env python3
"""Out-of-sample validation for 4h mean reversion edge"""
import pandas as pd
import numpy as np
from pathlib import Path

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')

# Top profitable symbols from September
top_symbols = ['0GUSDT', 'HUSDT', 'XMRUSDT', 'LAUSDT', 'KAVAUSDT', 'ADAUSDT']

print('='*70)
print('OUT-OF-SAMPLE VALIDATION - October 2025')
print('='*70)

all_results = []

for sym in top_symbols:
    print(f'\n{sym}:', flush=True)
    
    # Load October data
    dfs = []
    for day in range(1, 32):
        f = DATALAKE / sym / f'2025-10-{day:02d}_kline_1m.csv'
        if f.exists():
            dfs.append(pd.read_csv(f))
    
    if len(dfs) < 10:
        print(f'  No October data ({len(dfs)} files)')
        continue
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f'  Loaded {len(df)} records ({len(df)/1440:.1f} days)')
    
    # Resample to 4h
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    if len(df_4h) < 50:
        continue
    
    df_4h['ret'] = df_4h['close'].pct_change() * 10000
    
    # Test 4h mean reversion with 200bps threshold
    extreme = df_4h[abs(df_4h['ret']) > 200]
    
    if len(extreme) < 2:
        print(f'  Only {len(extreme)} extreme moves')
        continue
    
    trades = []
    for idx in extreme.index:
        idx_pos = df_4h.index.get_loc(idx)
        if idx_pos + 4 >= len(df_4h):
            continue
        
        move = df_4h.loc[idx, 'ret']
        future_ret = (df_4h.iloc[idx_pos + 4]['close'] / df_4h.loc[idx, 'close'] - 1) * 10000
        
        if move > 200:
            pnl = -future_ret - 20  # short
        else:
            pnl = future_ret - 20   # long
        trades.append(pnl)
    
    if len(trades) > 0:
        avg = np.mean(trades)
        wr = sum(1 for t in trades if t > 0) / len(trades) * 100
        total = sum(trades)
        print(f'  Trades: {len(trades)} | WR: {wr:.1f}% | Avg: {avg:.1f}bps | Total: {total:.1f}bps')
        all_results.append((sym, len(trades), wr, avg, total))

print('\n' + '='*70)
print('OUT-OF-SAMPLE RESULTS (October 2025)')
print('='*70)

if all_results:
    # Show profitable only
    profitable = [r for r in all_results if r[3] > 0]
    if profitable:
        print(f'Profitable symbols: {len(profitable)}/{len(all_results)}')
        profitable.sort(key=lambda x: x[3], reverse=True)
        for r in profitable:
            monthly_est = r[3] / 10000 * 10000 * (r[1] / 31) * 30
            print(f"  {r[0]:12s} | {r[1]:3d} trades | {r[2]:5.1f}% WR | {r[3]:6.1f}bps avg | ${monthly_est:.0f}/mo est")
    else:
        print('No profitable symbols in October')
        print('Edge may have decayed or was overfit to September')
        
    # Show all for comparison
    print('\nAll results:')
    for r in all_results:
        status = 'PROFIT' if r[3] > 0 else 'LOSS'
        print(f"  {r[0]:12s} | {status:6s} | {r[1]:3d} trades | {r[2]:5.1f}% WR | {r[3]:6.1f}bps avg")

print('='*70)
