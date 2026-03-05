"""
Higher Timeframe & Longer Hold Strategy Search
Testing 4h/1d bars and multi-day holds where moves can overcome 20 bps fees
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines, get_available_symbols

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')
FEE_RT = 0.002  # 20 bps


def resample_to_higher_tf(df: pd.DataFrame, minutes: int = 240) -> pd.DataFrame:
    """Resample 1m data to higher timeframe (4h = 240m)"""
    df = df.set_index('timestamp')
    
    resampled = pd.DataFrame({
        'open': df['open'].resample(f'{minutes}min').first(),
        'high': df['high'].resample(f'{minutes}min').max(),
        'low': df['low'].resample(f'{minutes}min').min(),
        'close': df['close'].resample(f'{minutes}min').last(),
        'volume': df['volume'].resample(f'{minutes}min').sum(),
    }).dropna()
    
    resampled = resampled.reset_index()
    return resampled


def test_trend_following_4h(symbol: str, start: str, end: str) -> dict:
    """
    Test trend following on 4h bars with longer holds
    """
    klines = load_klines(symbol, start, end)
    if len(klines) < 1000:
        return None
    
    # Resample to 4h
    df = resample_to_higher_tf(klines, 240)
    if len(df) < 50:
        return None
    
    # Calculate indicators
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 10000  # bps
    
    results = []
    
    # Test EMA crossover
    for fast in [10, 20]:
        for slow in [30, 50]:
            for hold in [1, 2, 4, 8]:  # 4h bars = 4h, 8h, 16h, 32h hold
                trades = []
                in_pos = False
                entry_price = None
                
                for i in range(slow, len(df) - hold):
                    if not in_pos:
                        # Entry: Fast EMA crosses above slow
                        if df.iloc[i]['ema20'] > df.iloc[i]['ema50']:
                            if df.iloc[i-1]['ema20'] <= df.iloc[i-1]['ema50']:
                                in_pos = True
                                entry_price = df.iloc[i]['close']
                    else:
                        # Time-based exit
                        exit_price = df.iloc[i + hold]['close']
                        ret = (exit_price - entry_price) / entry_price
                        
                        position_value = 10000
                        gross = ret * position_value
                        fees = position_value * FEE_RT
                        net = gross - fees
                        
                        trades.append(net)
                        in_pos = False
                
                if len(trades) >= 5:
                    total = sum(trades)
                    wins = len([t for t in trades if t > 0])
                    results.append({
                        'type': 'ema_cross',
                        'fast': fast,
                        'slow': slow,
                        'hold': hold,
                        'trades': len(trades),
                        'win_rate': wins / len(trades) * 100,
                        'total_net': total,
                        'avg_net': total / len(trades),
                        'pf': abs(sum([t for t in trades if t > 0]) / sum([t for t in trades if t < 0])) if sum([t for t in trades if t < 0]) != 0 else 999
                    })
    
    # Test ATR breakout (volatility expansion)
    for atr_mult in [1.5, 2.0, 3.0]:
        for hold in [1, 2, 4]:
            trades = []
            
            for i in range(50, len(df) - hold):
                atr = df.iloc[i]['atr_pct']
                
                # Breakout: price moves > ATR * mult in one bar
                bar_range = (df.iloc[i]['high'] - df.iloc[i]['low']) / df.iloc[i]['open'] * 10000
                
                if bar_range > atr * atr_mult:
                    # Directional breakout
                    body = (df.iloc[i]['close'] - df.iloc[i]['open']) / df.iloc[i]['open'] * 10000
                    
                    if body > 0:  # Up breakout
                        entry = df.iloc[i]['close']
                        exit_p = df.iloc[i + hold]['close']
                        ret = (exit_p - entry) / entry
                    else:  # Down breakout - skip or go short
                        continue
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_RT
                    net = gross - fees
                    
                    trades.append(net)
            
            if len(trades) >= 5:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                results.append({
                    'type': 'atr_breakout',
                    'atr_mult': atr_mult,
                    'hold': hold,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_net': total / len(trades),
                    'pf': abs(sum([t for t in trades if t > 0]) / sum([t for t in trades if t < 0])) if sum([t for t in trades if t < 0]) != 0 else 999
                })
    
    return results


def test_multi_day_hold(symbol: str, start: str, end: str) -> dict:
    """
    Test multi-day hold strategies (swing trading)
    """
    klines = load_klines(symbol, start, end)
    if len(klines) < 2000:
        return None
    
    # Use daily data
    df = resample_to_higher_tf(klines, 1440)  # 1 day
    if len(df) < 30:
        return None
    
    # Calculate indicators
    df['sma20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365)  # Annualized
    
    results = []
    
    # Test: Enter after 3+ consecutive down days, hold for reversal
    df['down_days'] = (df['close'] < df['close'].shift(1)).astype(int)
    df['consec_down'] = df['down_days'].groupby((df['down_days'] != df['down_days'].shift()).cumsum()).cumcount() + 1
    df['consec_down'] = df['consec_down'] * df['down_days']
    
    for consec in [2, 3, 4, 5]:  # Consecutive down days
        for hold in [1, 2, 3, 5, 10]:  # Hold days
            trades = []
            
            for i in range(20, len(df) - hold):
                if df.iloc[i]['consec_down'] >= consec:
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold]['close']
                    ret = (exit_p - entry) / entry
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_RT
                    net = gross - fees
                    
                    trades.append(net)
            
            if len(trades) >= 5:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                results.append({
                    'type': 'reversal',
                    'consec_down': consec,
                    'hold_days': hold,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_net': total / len(trades)
                })
    
    # Test: Momentum - buy when price > SMA20, hold
    for hold in [2, 5, 10, 20]:
        trades = []
        
        for i in range(20, len(df) - hold):
            if df.iloc[i]['close'] > df.iloc[i]['sma20']:
                if df.iloc[i-1]['close'] <= df.iloc[i-1]['sma20']:  # Cross above
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold]['close']
                    ret = (exit_p - entry) / entry
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_RT
                    net = gross - fees
                    
                    trades.append(net)
        
        if len(trades) >= 5:
            total = sum(trades)
            wins = len([t for t in trades if t > 0])
            results.append({
                'type': 'momentum',
                'hold_days': hold,
                'trades': len(trades),
                'win_rate': wins / len(trades) * 100,
                'total_net': total,
                'avg_net': total / len(trades)
            })
    
    return results


def test_volatility_regime_4h(symbol: str, start: str, end: str) -> dict:
    """
    Test trading only in high volatility regimes on 4h
    """
    klines = load_klines(symbol, start, end)
    if len(klines) < 1000:
        return None
    
    df = resample_to_higher_tf(klines, 240)
    if len(df) < 50:
        return None
    
    # Calculate volatility
    df['range'] = (df['high'] - df['low']) / df['open'] * 10000  # bps
    df['range_pctile'] = df['range'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan, raw=False)
    
    results = []
    
    for vol_thresh in [0.7, 0.8, 0.9]:
        for hold in [1, 2, 4]:
            trades = []
            
            for i in range(30, len(df) - hold):
                if pd.isna(df.iloc[i]['range_pctile']):
                    continue
                
                # Only trade in high volatility
                if df.iloc[i]['range_pctile'] < vol_thresh:
                    continue
                
                # Momentum direction
                body = (df.iloc[i]['close'] - df.iloc[i]['open']) / df.iloc[i]['open']
                
                if body > 0:
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold]['close']
                    ret = (exit_p - entry) / entry
                else:
                    continue
                
                position_value = 10000
                gross = ret * position_value
                fees = position_value * FEE_RT
                net = gross - fees
                
                trades.append(net)
            
            if len(trades) >= 5:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                results.append({
                    'type': 'high_vol_momentum',
                    'vol_thresh': vol_thresh,
                    'hold': hold,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_net': total / len(trades)
                })
    
    return results


if __name__ == '__main__':
    print('='*70)
    print('HIGHER TIMEFRAME STRATEGY SEARCH (4h / Daily)')
    print('='*70)
    
    symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 
               'DOTUSDT', 'MATICUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'UNIUSDT']
    
    # Test different periods
    periods = [
        ('2024-01-01', '2024-12-31', '2024 Full Year'),
        ('2025-01-01', '2025-12-31', '2025 Full Year'),
        ('2025-06-01', '2025-12-31', '2025 H2 (High Vol)'),
    ]
    
    all_results = []
    
    for start, end, label in periods:
        print(f'\n{"="*70}')
        print(f'PERIOD: {label}')
        print('='*70)
        
        for sym in symbols:
            # 4h strategies
            res = test_trend_following_4h(sym, start, end)
            if res:
                profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 5]
                for p in profitable:
                    all_results.append({
                        'period': label,
                        'symbol': sym,
                        'strategy': p['type'],
                        'config': f"hold={p.get('hold', p.get('hold_days', 'N/A'))}",
                        'trades': p['trades'],
                        'win_rate': p['win_rate'],
                        'total_net': p['total_net'],
                        'pf': p.get('pf', 0)
                    })
            
            # Daily strategies
            res = test_multi_day_hold(sym, start, end)
            if res:
                profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 5]
                for p in profitable:
                    all_results.append({
                        'period': label,
                        'symbol': sym,
                        'strategy': p['type'],
                        'config': f"consec={p.get('consec_down', 'N/A')},hold={p.get('hold_days', 'N/A')}",
                        'trades': p['trades'],
                        'win_rate': p['win_rate'],
                        'total_net': p['total_net'],
                        'pf': 0
                    })
            
            # Vol regime
            res = test_volatility_regime_4h(sym, start, end)
            if res:
                profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 5]
                for p in profitable:
                    all_results.append({
                        'period': label,
                        'symbol': sym,
                        'strategy': 'vol_regime',
                        'config': f"thresh={p['vol_thresh']},hold={p['hold']}",
                        'trades': p['trades'],
                        'win_rate': p['win_rate'],
                        'total_net': p['total_net'],
                        'pf': 0
                    })
    
    # Final summary
    print('\n' + '='*70)
    print('FINAL SUMMARY - ALL PROFITABLE CONFIGURATIONS')
    print('='*70)
    
    if all_results:
        all_results.sort(key=lambda x: x['total_net'], reverse=True)
        
        print(f'Top 15 profitable configurations:')
        for r in all_results[:15]:
            print(f"  {r['period'][:10]:10s} | {r['symbol']:10s} | {r['strategy']:12s} | "
                  f"{r['config']:20s} | Trades={r['trades']:3d} | WR={r['win_rate']:5.1f}% | "
                  f"Net=${r['total_net']:8.2f} | PF={r['pf']:.2f}")
        
        print('\n' + '='*70)
        print(f'TOTAL PROFITABLE CONFIGURATIONS: {len(all_results)}')
        print('='*70)
    else:
        print('NO PROFITABLE STRATEGIES FOUND ON HIGHER TIMEFRAMES')
        print('\nThis suggests:')
        print('- Market is efficient even at 4h/daily timeframes')
        print('- 20 bps fees are very difficult to overcome')
        print('- Need to explore: maker-only strategies, cross-exchange arb, or exotic coins')
        print('='*70)
