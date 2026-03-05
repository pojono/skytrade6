#!/usr/bin/env python3
"""Fast edge search with visible progress"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines

print('='*70)
print('EDGE SEARCH - Fixed Data Loading')
print('='*70)

symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'ADAUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT']
profitable_found = []

for i, sym in enumerate(symbols, 1):
    print(f'[{i}/{len(symbols)}] Loading {sym}... ', end='', flush=True)
    
    df = load_klines(sym, '2025-09-01', '2025-12-31')
    print(f'{len(df)} records', flush=True)
    
    if len(df) < 1000:
        continue
    
    df['ret'] = df['close'].pct_change()
    df['ret_bps'] = df['ret'] * 10000
    
    # Test mean reversion
    for thresh in [50, 100, 150]:
        extreme = df[abs(df['ret_bps']) > thresh]
        if len(extreme) < 3:
            continue
        
        for hold in [1, 5, 10]:
            trades = []
            for idx in extreme.index:
                if idx + hold >= len(df):
                    continue
                move = df.loc[idx, 'ret_bps']
                future_ret = (df.loc[idx + hold, 'close'] / df.loc[idx, 'close'] - 1) * 10000
                
                if move > thresh:
                    pnl = -future_ret - 20
                else:
                    pnl = future_ret - 20
                trades.append(pnl)
            
            if len(trades) >= 3:
                avg = np.mean(trades)
                if avg > 0:
                    wr = sum(1 for t in trades if t > 0) / len(trades) * 100
                    print(f'  >>> {sym}: thresh={thresh}, hold={hold}, WR={wr:.1f}%, Avg={avg:.2f}bps')
                    profitable_found.append((sym, thresh, hold, len(trades), wr, avg))

print('='*70)
print('RESULTS')
print('='*70)

if profitable_found:
    print(f'Found {len(profitable_found)} profitable configurations:')
    profitable_found.sort(key=lambda x: x[5], reverse=True)
    for pf in profitable_found[:10]:
        print(f"  {pf[0]:10s} | thresh={pf[1]:3d} | hold={pf[2]:2d} | {pf[3]:3d} trades | {pf[4]:5.1f}% WR | {pf[5]:6.2f}bps")
else:
    print('No profitable strategies found')

print('='*70)
