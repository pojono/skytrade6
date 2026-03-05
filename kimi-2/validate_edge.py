#!/usr/bin/env python3
"""Validate 4h mean reversion edge on extended data"""
import pandas as pd
import numpy as np
from pathlib import Path

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')

print('='*70)
print('VALIDATING 4H MEAN REVERSION - Extended Test (Sep 2025)')
print('='*70)

symbols = ['LINKUSDT', 'ETHUSDT', 'ADAUSDT', 'BTCUSDT', 'DOGEUSDT']
all_results = []

for sym in symbols:
    print(f'\n{sym}:', flush=True)
    
    # Load all September data
    dfs = []
    for day in range(1, 31):
        f = DATALAKE / sym / f'2025-09-{day:02d}_kline_1m.csv'
        if f.exists():
            dfs.append(pd.read_csv(f))
    
    if len(dfs) == 0:
        print('  No data')
        continue
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f'  Loaded {len(df)} 1m records ({len(df)/1440:.1f} days)')
    
    # Resample to 4h
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    df_4h['ret'] = df_4h['close'].pct_change() * 10000
    
    # Test best config: thresh=200, hold=4
    extreme = df_4h[abs(df_4h['ret']) > 200]
    
    if len(extreme) < 2:
        print(f'  Only {len(extreme)} extreme moves found')
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
print('VALIDATION RESULTS')
print('='*70)

if all_results:
    print('Profitable on extended data:')
    all_results.sort(key=lambda x: x[4], reverse=True)
    for r in all_results:
        print(f"  {r[0]:10s} | {r[1]:3d} trades | {r[2]:5.1f}% WR | {r[3]:6.1f}bps avg | {r[4]:7.1f}bps total")
    
    # Calculate daily P&L potential
    print('\nDAILY P&L POTENTIAL (assuming $10k per trade):')
    for r in all_results:
        days = 30  # September has ~30 days
        trades_per_day = r[1] / days
        daily_pnl = trades_per_day * r[3] / 10000 * 10000  # $ per day
        monthly = daily_pnl * 30
        print(f"  {r[0]:10s} | {trades_per_day:.1f} trades/day | ${daily_pnl:.2f}/day | ${monthly:.2f}/month")

print('='*70)
