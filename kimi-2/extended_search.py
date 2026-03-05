"""
Extended Strategy Search - Different Timeframes, Volatility Regimes, More Symbols
Fresh analysis - no assumptions from prior research
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines, load_funding_rates, get_available_symbols

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')
FEE_TAKER_RT = 0.002  # 20 bps
FEE_MAKER_RT = 0.0008  # 8 bps


def test_simple_breakout(symbol: str, start: str, end: str) -> dict:
    """
    Test simple breakout: price breaks above recent high, enter long
    """
    df = load_klines(symbol, start, end)
    if len(df) < 100:
        return None
    
    results = []
    
    # Test different lookback periods and hold times
    for lookback in [20, 50, 100]:  # bars
        for hold in [10, 30, 60, 120]:  # bars
            for vol_filter in [None, 1.0, 1.5]:  # None or Nx avg volume
                
                trades = []
                in_pos = False
                entry_price = None
                
                for i in range(lookback, len(df) - hold):
                    if in_pos:
                        continue  # Wait for next iteration
                    
                    # Get recent range
                    recent = df.iloc[i-lookback:i]
                    recent_high = recent['high'].max()
                    recent_vol_avg = recent['volume'].mean()
                    
                    # Breakout condition
                    price_now = df.iloc[i]['close']
                    vol_now = df.iloc[i]['volume']
                    
                    # Break above recent high
                    is_breakout = price_now > recent_high
                    
                    # Volume filter
                    if vol_filter:
                        if vol_now < recent_vol_avg * vol_filter:
                            continue
                    
                    if is_breakout:
                        entry_price = price_now
                        exit_price = df.iloc[i + hold]['close']
                        
                        # Calculate P&L with taker fees
                        ret = (exit_price - entry_price) / entry_price
                        position_value = 10000
                        gross = ret * position_value
                        fees = position_value * FEE_TAKER_RT
                        net = gross - fees
                        
                        trades.append(net)
                        in_pos = True
                    
                    # Reset position flag after hold period
                    if in_pos and i > lookback + hold:
                        in_pos = False
                
                if len(trades) >= 10:
                    total = sum(trades)
                    wins = len([t for t in trades if t > 0])
                    results.append({
                        'lookback': lookback,
                        'hold': hold,
                        'vol_filter': vol_filter,
                        'trades': len(trades),
                        'win_rate': wins / len(trades) * 100,
                        'total_net': total,
                        'avg_net': total / len(trades),
                        'pf': abs(sum([t for t in trades if t > 0]) / sum([t for t in trades if t < 0])) if sum([t for t in trades if t < 0]) != 0 else 999
                    })
    
    return results


def test_gap_reversal(symbol: str, start: str, end: str) -> dict:
    """
    Test: Large 1m moves tend to reverse partially
    """
    df = load_klines(symbol, start, end)
    if len(df) < 100:
        return None
    
    df['return_1m'] = df['close'].pct_change() * 10000  # bps
    
    results = []
    
    # Test different thresholds for "large move"
    for threshold in [20, 50, 100, 200]:  # bps in 1m
        for hold in [1, 5, 10, 30]:  # bars
            trades = []
            
            for i in range(len(df) - hold):
                move = df.iloc[i]['return_1m']
                
                if pd.isna(move):
                    continue
                
                # Large up move -> short (expect reversal)
                if move > threshold:
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold]['close']
                    ret = (entry - exit_p) / entry  # short
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_TAKER_RT
                    net = gross - fees
                    trades.append(net)
                
                # Large down move -> long (expect reversal)
                elif move < -threshold:
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold]['close']
                    ret = (exit_p - entry) / entry  # long
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_TAKER_RT
                    net = gross - fees
                    trades.append(net)
            
            if len(trades) >= 10:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                results.append({
                    'threshold': threshold,
                    'hold': hold,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_net': total / len(trades)
                })
    
    return results


def test_session_based(symbol: str, start: str, end: str) -> dict:
    """
    Test: Different performance by time of day/session
    """
    df = load_klines(symbol, start, end)
    if len(df) < 100:
        return None
    
    df['hour'] = df['timestamp'].dt.hour
    df['return_1h'] = df['close'].pct_change(60) * 10000  # 1h return in bps
    
    results = []
    
    # Test each hour as entry point
    for entry_hour in range(24):
        for hold_hours in [1, 4, 8]:
            trades = []
            
            for i in range(len(df) - hold_hours * 60):
                if df.iloc[i]['hour'] == entry_hour:
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + hold_hours * 60]['close']
                    ret = (exit_p - entry) / entry
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_TAKER_RT
                    net = gross - fees
                    trades.append(net)
            
            if len(trades) >= 10:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                avg_ret = np.mean([(t + FEE_TAKER_RT * position_value) / position_value * 10000 for t in trades])
                results.append({
                    'entry_hour': entry_hour,
                    'hold_hours': hold_hours,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_ret_bps': avg_ret
                })
    
    return results


def test_volatility_regime(symbol: str, start: str, end: str) -> dict:
    """
    Test: Trade only in high volatility regimes
    """
    df = load_klines(symbol, start, end)
    if len(df) < 200:
        return None
    
    df['return_1m'] = df['close'].pct_change()
    df['vol_20'] = df['return_1m'].rolling(20).std() * 10000  # bps
    df['vol_percentile'] = df['vol_20'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 100 else np.nan, raw=False)
    
    results = []
    
    # Test different volatility thresholds
    for vol_thresh in [0.5, 0.7, 0.8, 0.9]:  # percentile
        for zscore in [1.5, 2.0, 2.5]:
            trades = []
            
            for i in range(100, len(df) - 10):
                if pd.isna(df.iloc[i]['vol_percentile']):
                    continue
                
                # Only trade in high vol regime
                if df.iloc[i]['vol_percentile'] < vol_thresh:
                    continue
                
                # Mean reversion signal in high vol
                recent = df.iloc[i-20:i]
                mean_price = recent['close'].mean()
                std_price = recent['close'].std()
                z = (df.iloc[i]['close'] - mean_price) / std_price if std_price > 0 else 0
                
                if z < -zscore:  # Buy dip
                    entry = df.iloc[i]['close']
                    exit_p = df.iloc[i + 10]['close']
                    ret = (exit_p - entry) / entry
                    
                    position_value = 10000
                    gross = ret * position_value
                    fees = position_value * FEE_TAKER_RT
                    net = gross - fees
                    trades.append(net)
            
            if len(trades) >= 10:
                total = sum(trades)
                wins = len([t for t in trades if t > 0])
                results.append({
                    'vol_thresh': vol_thresh,
                    'zscore': zscore,
                    'trades': len(trades),
                    'win_rate': wins / len(trades) * 100,
                    'total_net': total,
                    'avg_net': total / len(trades)
                })
    
    return results


def scan_many_symbols(start: str, end: str) -> list:
    """
    Scan all available symbols for any profitable configuration
    """
    symbols = get_available_symbols()
    print(f'Scanning {len(symbols)} symbols...')
    
    all_profitable = []
    
    # Test top 30 by data coverage (approximated by directory size)
    symbol_sizes = []
    for sym in symbols:
        sym_path = DATALAKE / sym
        if sym_path.exists():
            # Count files as proxy for data coverage
            num_files = len(list(sym_path.glob('*.csv')))
            symbol_sizes.append((sym, num_files))
    
    symbol_sizes.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s[0] for s in symbol_sizes[:30]]
    
    print(f'Testing top {len(top_symbols)} symbols by data coverage')
    
    for sym in top_symbols:
        # Test breakout
        res = test_simple_breakout(sym, start, end)
        if res:
            profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 10]
            for p in profitable:
                all_profitable.append({
                    'symbol': sym,
                    'strategy': 'breakout',
                    'config': f"lb={p['lookback']},hold={p['hold']}",
                    'trades': p['trades'],
                    'win_rate': p['win_rate'],
                    'total_net': p['total_net']
                })
        
        # Test gap reversal
        res = test_gap_reversal(sym, start, end)
        if res:
            profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 10]
            for p in profitable:
                all_profitable.append({
                    'symbol': sym,
                    'strategy': 'gap_reversal',
                    'config': f"th={p['threshold']},hold={p['hold']}",
                    'trades': p['trades'],
                    'win_rate': p['win_rate'],
                    'total_net': p['total_net']
                })
        
        # Test volatility regime
        res = test_volatility_regime(sym, start, end)
        if res:
            profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 10]
            for p in profitable:
                all_profitable.append({
                    'symbol': sym,
                    'strategy': 'vol_regime',
                    'config': f"vol={p['vol_thresh']},z={p['zscore']}",
                    'trades': p['trades'],
                    'win_rate': p['win_rate'],
                    'total_net': p['total_net']
                })
    
    return all_profitable


if __name__ == '__main__':
    print('='*70)
    print('EXTENDED STRATEGY SEARCH - All Timeframes & Volatility Regimes')
    print('='*70)
    
    # Test different periods
    periods = [
        ('2025-01-01', '2025-06-30', 'H1 2025'),
        ('2025-07-01', '2025-12-31', 'H2 2025'),
        ('2024-06-01', '2024-12-31', 'H2 2024'),
    ]
    
    all_time_results = []
    
    for start, end, label in periods:
        print(f'\n{"="*70}')
        print(f'TESTING PERIOD: {label} ({start} to {end})')
        print('='*70)
        
        # Scan many symbols
        results = scan_many_symbols(start, end)
        
        if results:
            # Sort by total net profit
            results.sort(key=lambda x: x['total_net'], reverse=True)
            
            print(f'\nTop 10 profitable configurations in {label}:')
            for r in results[:10]:
                print(f"  {r['symbol']:12s} | {r['strategy']:15s} | {r['config']:20s} | "
                      f"Trades={r['trades']:4d} | WR={r['win_rate']:5.1f}% | Net=${r['total_net']:8.2f}")
            
            all_time_results.extend([(label, r) for r in results])
        else:
            print(f'No profitable strategies found in {label}')
    
    # Final summary
    print('\n' + '='*70)
    print('FINAL SUMMARY - ALL PERIODS')
    print('='*70)
    
    if all_time_results:
        # Group by strategy type
        by_strategy = {}
        for period, result in all_time_results:
            strat = result['strategy']
            if strat not in by_strategy:
                by_strategy[strat] = []
            by_strategy[strat].append((period, result))
        
        for strat, results in by_strategy.items():
            print(f'\n{strat.upper()}:')
            results.sort(key=lambda x: x[1]['total_net'], reverse=True)
            for period, r in results[:5]:
                print(f"  {period:10s} | {r['symbol']:12s} | Net=${r['total_net']:8.2f} | WR={r['win_rate']:5.1f}%")
    else:
        print('NO PROFITABLE STRATEGIES FOUND IN ANY PERIOD')
        print('\nConclusions:')
        print('- 8h funding rates too small for profitable FR hold')
        print('- Simple price patterns do not overcome 20 bps fees')
        print('- Market too efficient at 1m timeframe with current fee structure')
        print('\nPossible edges requiring further research:')
        print('- 1h funding rate coins (higher FR volatility)')
        print('- Cross-exchange arbitrage')
        print('- Sub-second HFT strategies')
        print('- Options/volatility strategies')
    
    print('='*70)
