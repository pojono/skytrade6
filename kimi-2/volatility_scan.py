"""
Fast Volatility Scan with Progress Output
Shows real-time progress as it tests symbols
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines, get_available_symbols

def main():
    symbols = get_available_symbols()
    print(f'Total symbols: {len(symbols)}')
    print('Scanning top 30 symbols for volatility...')
    print('='*70)
    
    volatility_scores = []
    
    for i, sym in enumerate(symbols[:30]):
        print(f'[{i+1}/30] Loading {sym}...', end=' ')
        
        try:
            df = load_klines(sym, '2025-06-01', '2025-12-31')
            if len(df) > 1000:
                df['ret'] = df['close'].pct_change()
                vol = df['ret'].std() * np.sqrt(365 * 24 * 60)
                max_move = abs(df['ret']).max() * 10000
                
                volatility_scores.append({
                    'symbol': sym,
                    'volatility': vol,
                    'max_move_bps': max_move,
                    'records': len(df)
                })
                print(f'OK - Vol={vol:.1f}, Max={max_move:.1f}bps')
            else:
                print(f'SKIP - Only {len(df)} records')
        except Exception as e:
            print(f'ERROR - {str(e)[:30]}')
    
    print('='*70)
    print(f'Scanned {len(volatility_scores)} symbols successfully')
    
    volatility_scores.sort(key=lambda x: x['volatility'], reverse=True)
    
    print('\nTop 15 Most Volatile Symbols (Jun-Dec 2025):')
    print('-'*70)
    for v in volatility_scores[:15]:
        print(f"  {v['symbol']:15s} | Vol={v['volatility']:8.1f} | Max Move={v['max_move_bps']:8.1f}bps")
    
    print('-'*70)
    print('Testing mean reversion on top 5 volatile symbols...')
    print('-'*70)
    
    profitable_found = []
    
    for v in volatility_scores[:5]:
        sym = v['symbol']
        print(f'\nTesting {sym} (vol={v["volatility"]:.1f})...')
        
        try:
            df = load_klines(sym, '2025-06-01', '2025-12-31')
            if len(df) < 100:
                print('  SKIP - insufficient data')
                continue
            
            df['ret'] = df['close'].pct_change() * 10000
            
            for threshold in [100, 150, 200]:
                extreme_moves = df[abs(df['ret']) > threshold]
                
                if len(extreme_moves) < 3:
                    continue
                
                for hold in [1, 5, 10]:
                    reversion_trades = []
                    
                    for idx in extreme_moves.index:
                        if idx + hold < len(df):
                            move = df.loc[idx, 'ret']
                            future_ret = df.loc[idx + hold, 'close'] / df.loc[idx, 'close'] - 1
                            
                            if move > threshold:
                                reversion = -future_ret * 10000 - 20
                            else:
                                reversion = future_ret * 10000 - 20
                            
                            reversion_trades.append(reversion)
                    
                    if len(reversion_trades) >= 3:
                        avg = np.mean(reversion_trades)
                        win_rate = sum(1 for t in reversion_trades if t > 0) / len(reversion_trades) * 100
                        
                        if avg > 0:
                            print(f"  PROFITABLE: thresh={threshold}, hold={hold} | Trades={len(reversion_trades)}, WR={win_rate:.1f}%, Avg={avg:.2f}bps")
                            profitable_found.append((sym, threshold, hold, len(reversion_trades), win_rate, avg))
        except Exception as e:
            print(f'  ERROR: {str(e)[:50]}')
    
    print('='*70)
    if profitable_found:
        print(f'SUCCESS: Found {len(profitable_found)} profitable mean reversion setups!')
        print('-'*70)
        for pf in profitable_found:
            print(f"  {pf[0]}: thresh={pf[1]}, hold={pf[2]} | {pf[3]} trades, {pf[4]:.1f}% WR, {pf[5]:.2f}bps avg")
    else:
        print('RESULT: No profitable mean reversion found on volatile symbols')
    
    print('='*70)

if __name__ == '__main__':
    main()
