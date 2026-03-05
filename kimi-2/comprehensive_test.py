#!/usr/bin/env python3
"""Comprehensive findings document"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')

print('='*70)
print('COMPREHENSIVE STRATEGY TEST - ALL APPROACHES')
print('='*70)

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT']
results = {
    'mean_reversion': [],
    'momentum': [],
    'breakout': [],
    'grid': []
}

for sym in symbols:
    print(f'\n{sym}:', flush=True)
    
    # Load data
    dfs = []
    for day in range(1, 15):  # Just first 2 weeks of Sept
        f = DATALAKE / sym / f'2025-09-{day:02d}_kline_1m.csv'
        if f.exists():
            dfs.append(pd.read_csv(f))
    
    if len(dfs) < 5:
        continue
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Resample to 1h for faster testing
    df_h = df.set_index('timestamp').resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    if len(df_h) < 10:
        continue
    
    df_h['ret'] = df_h['close'].pct_change() * 10000
    df_h['sma'] = df_h['close'].rolling(5).mean()
    df_h['trend'] = (df_h['close'] - df_h['sma']) / df_h['sma'] * 10000
    
    # 1. Mean Reversion: Extreme move reverses
    extreme = df_h[abs(df_h['ret']) > 50]
    mr_trades = []
    for idx in extreme.index:
        idx_pos = df_h.index.get_loc(idx)
        if idx_pos + 1 >= len(df_h):
            continue
        move = df_h.loc[idx, 'ret']
        future = (df_h.iloc[idx_pos + 1]['close'] / df_h.loc[idx, 'close'] - 1) * 10000
        pnl = -future - 20 if move > 50 else future - 20
        mr_trades.append(pnl)
    
    if len(mr_trades) >= 3:
        avg = np.mean(mr_trades)
        results['mean_reversion'].append((sym, len(mr_trades), avg))
        print(f'  Mean Rev: {len(mr_trades):2d} trades, {avg:6.1f}bps avg')
    
    # 2. Momentum: Trend continuation
    mom_trades = []
    for i in range(5, len(df_h) - 1):
        trend = df_h.iloc[i]['trend']
        ret = df_h.iloc[i]['ret']
        
        if abs(trend) > 20 and abs(ret) > 30:
            future = (df_h.iloc[i + 1]['close'] / df_h.iloc[i]['close'] - 1) * 10000
            pnl = future - 20 if trend > 0 else -future - 20
            mom_trades.append(pnl)
    
    if len(mom_trades) >= 3:
        avg = np.mean(mom_trades)
        results['momentum'].append((sym, len(mom_trades), avg))
        print(f'  Momentum: {len(mom_trades):2d} trades, {avg:6.1f}bps avg')
    
    # 3. Breakout: New high/low
    breakout_trades = []
    for i in range(10, len(df_h) - 1):
        high_10 = df_h.iloc[i-10:i]['high'].max()
        low_10 = df_h.iloc[i-10:i]['low'].min()
        
        if df_h.iloc[i]['close'] > high_10:
            future = (df_h.iloc[i + 1]['close'] / df_h.iloc[i]['close'] - 1) * 10000
            breakout_trades.append(future - 20)
        elif df_h.iloc[i]['close'] < low_10:
            future = (df_h.iloc[i + 1]['close'] / df_h.iloc[i]['close'] - 1) * 10000
            breakout_trades.append(-future - 20)
    
    if len(breakout_trades) >= 3:
        avg = np.mean(breakout_trades)
        results['breakout'].append((sym, len(breakout_trades), avg))
        print(f'  Breakout: {len(breakout_trades):2d} trades, {avg:6.1f}bps avg')

print('\n' + '='*70)
print('SUMMARY - ALL STRATEGIES')
print('='*70)

for strategy, trades in results.items():
    profitable = [t for t in trades if t[2] > 0]
    print(f'{strategy:15s}: {len(profitable)}/{len(trades)} symbols profitable')
    for t in trades:
        status = 'PROFIT' if t[2] > 0 else 'LOSS'
        print(f'  {t[0]:10s} | {t[1]:2d} trades | {t[2]:6.1f}bps | {status}')

print('='*70)
