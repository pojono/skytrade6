"""
Final Strategy Tests - Grid Trading & Long-Term Holds
Last attempt before documenting findings
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines

FEE_MAKER = 0.0008  # 8 bps
FEE_TAKER = 0.002   # 20 bps


def test_grid_trading(symbol, start, end):
    """Grid trading around moving average"""
    df = load_klines(symbol, start, end)
    if len(df) < 1000:
        return None
    
    df['sma'] = df['close'].rolling(100).mean()
    df['dev'] = (df['close'] - df['sma']) / df['sma'] * 10000  # bps from mean
    
    results = []
    
    for entry_bps in [-200, -150, -100]:
        for exit_bps in [100, 150, 200]:
            trades = []
            in_pos = False
            entry_price = None
            
            for i in range(100, len(df)):
                dev = df.iloc[i]['dev']
                
                if not in_pos and dev <= entry_bps:
                    in_pos = True
                    entry_price = df.iloc[i]['close']
                
                elif in_pos and dev >= exit_bps:
                    exit_price = df.iloc[i]['close']
                    ret = (exit_price - entry_price) / entry_price
                    
                    gross = ret * 10000  # bps
                    net = gross - 8  # maker fees
                    
                    trades.append(net)
                    in_pos = False
            
            if len(trades) >= 10:
                total = sum(trades)
                if total > 0:
                    results.append({
                        'entry': entry_bps,
                        'exit': exit_bps,
                        'trades': len(trades),
                        'win_rate': sum(1 for t in trades if t > 0) / len(trades) * 100,
                        'total_bps': total,
                        'avg_bps': total / len(trades)
                    })
    
    return results


def test_trend_4h(symbol, start, end):
    """Simple trend following on 4h"""
    df = load_klines(symbol, start, end)
    if len(df) < 1000:
        return None
    
    # Resample to 4h
    df = df.set_index('timestamp')
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(df_4h) < 50:
        return None
    
    df_4h['sma20'] = df_4h['close'].rolling(20).mean()
    df_4h['ret'] = df_4h['close'].pct_change() * 10000
    
    results = []
    
    for hold in [1, 2, 4, 8]:  # 4h bars
        trades = []
        
        for i in range(20, len(df_4h) - hold):
            # Enter when price > SMA and previous was below
            if df_4h.iloc[i]['close'] > df_4h.iloc[i]['sma20']:
                if df_4h.iloc[i-1]['close'] <= df_4h.iloc[i-1]['sma20']:
                    ret = (df_4h.iloc[i + hold]['close'] - df_4h.iloc[i]['close']) / df_4h.iloc[i]['close'] * 10000
                    net = ret - 20  # taker fees
                    trades.append(net)
        
        if len(trades) >= 5:
            total = sum(trades)
            if total > 0:
                results.append({
                    'hold': hold,
                    'trades': len(trades),
                    'win_rate': sum(1 for t in trades if t > 0) / len(trades) * 100,
                    'total_bps': total
                })
    
    return results


if __name__ == '__main__':
    print('='*70)
    print('FINAL STRATEGY TESTS - Grid & 4h Trend')
    print('='*70)
    
    symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'ADAUSDT', 'XRPUSDT']
    
    all_results = []
    
    # Grid trading
    print('\n1. GRID TRADING (Maker Fees - 8 bps)')
    print('-'*70)
    for sym in symbols:
        res = test_grid_trading(sym, '2025-01-01', '2025-12-31')
        if res:
            for r in res:
                print(f"  {sym}: Entry {r['entry']:4d}bps, Exit {r['exit']:4d}bps | "
                      f"Trades={r['trades']:3d}, WR={r['win_rate']:5.1f}%, Net={r['total_bps']:7.1f}bps")
                all_results.append(('grid', sym, r))
    
    # 4h trend
    print('\n2. 4H TREND FOLLOWING (Taker Fees - 20 bps)')
    print('-'*70)
    for sym in symbols:
        res = test_trend_4h(sym, '2025-01-01', '2025-12-31')
        if res:
            for r in res:
                print(f"  {sym}: Hold {r['hold']} bars | "
                      f"Trades={r['trades']:3d}, WR={r['win_rate']:5.1f}%, Net={r['total_bps']:7.1f}bps")
                all_results.append(('trend_4h', sym, r))
    
    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    
    if all_results:
        print(f'Found {len(all_results)} potentially profitable configurations:')
        all_results.sort(key=lambda x: x[2]['total_bps'], reverse=True)
        for strat, sym, r in all_results[:10]:
            print(f"  {strat:10s} | {sym:10s} | Net={r['total_bps']:8.1f}bps | WR={r['win_rate']:5.1f}%")
    else:
        print('NO PROFITABLE STRATEGIES FOUND')
    
    print('='*70)
