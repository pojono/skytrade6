"""
Deep Strategy Search - Maker Fees & Microstructure
Test EVERYTHING fresh - no trust in prior conclusions
"""

import pandas as pd
import numpy as np
import gzip
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import load_klines, load_funding_rates, load_open_interest, BacktestResult, Trade

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')

# Maker fee = 4 bps per leg, 8 bps round-trip (vs 20 bps taker)
FEE_MAKER_RT = 0.0008
FEE_TAKER_RT = 0.002


def test_maker_fr_hold(symbol: str, start: str, end: str) -> dict:
    """
    Test FR hold with maker fees (8 bps vs 20 bps)
    Entry/exit via limit orders
    """
    klines = load_klines(symbol, start, end)
    funding = load_funding_rates(symbol, start, end)
    
    if len(klines) < 1000 or len(funding) < 50:
        return None
    
    # Merge funding to klines
    klines = klines.set_index('timestamp')
    funding = funding.set_index('timestamp')
    df = klines.join(funding[['fundingRate']], how='left')
    df['fundingRate'] = df['fundingRate'].fillna(method='ffill')
    df = df.reset_index()
    
    results = []
    
    # Test various entry/exit thresholds
    for entry_th in [5, 10, 15, 20, 30, 50]:  # bps
        for exit_th in [2, 5, 10]:  # bps
            for max_holds in [1, 2, 3, 5]:  # funding periods
                trades = []
                in_pos = False
                pos_start = None
                pos_entry = None
                holds = 0
                fr_collected = 0
                
                for i, row in df.iterrows():
                    fr = row['fundingRate'] * 10000 if pd.notna(row['fundingRate']) else 0
                    
                    # Entry: FR <= -threshold (extreme negative = get paid as long)
                    if not in_pos and fr <= -entry_th:
                        in_pos = True
                        pos_start = row['timestamp']
                        pos_entry = row['close']
                        holds = 0
                        fr_collected = 0
                    
                    elif in_pos:
                        holds += 1
                        # Collect funding (assume we hold through settlement)
                        if fr < 0:
                            fr_collected += abs(fr)  # bps collected
                        
                        # Exit conditions
                        exit_now = (fr >= -exit_th) or (holds >= max_holds)
                        
                        if exit_now or i == len(df) - 1:
                            exit_price = row['close']
                            
                            # Calculate P&L
                            price_ret = (exit_price - pos_entry) / pos_entry
                            
                            # For maker fees: 8 bps RT
                            position_value = 10000
                            fees = position_value * FEE_MAKER_RT
                            
                            # Funding income: FR * position * time
                            # Assume 8h periods, so fr_collected bps over hold period
                            funding_pnl = (fr_collected / 10000) * position_value
                            price_pnl = price_ret * position_value
                            
                            gross = funding_pnl + price_pnl
                            net = gross - fees
                            
                            trades.append({
                                'net': net,
                                'gross': gross,
                                'fees': fees,
                                'fr_collected': fr_collected,
                                'price_ret': price_ret,
                                'holds': holds
                            })
                            
                            in_pos = False
                
                if len(trades) >= 5:
                    total_net = sum(t['net'] for t in trades)
                    wins = len([t for t in trades if t['net'] > 0])
                    results.append({
                        'symbol': symbol,
                        'entry': entry_th,
                        'exit': exit_th,
                        'max_holds': max_holds,
                        'trades': len(trades),
                        'win_rate': wins / len(trades) * 100,
                        'total_net': total_net,
                        'avg_net': total_net / len(trades),
                        'avg_fr': np.mean([t['fr_collected'] for t in trades]),
                        'avg_price': np.mean([t['price_ret'] * 10000 for t in trades])  # bps
                    })
    
    return results


def analyze_trade_data_microstructure(symbol: str, date: str) -> dict:
    """
    Analyze tick-level trade data for microstructure patterns
    """
    trade_file = DATALAKE / symbol / f"{date}_trades.csv.gz"
    
    if not trade_file.exists():
        return None
    
    try:
        df = pd.read_csv(trade_file, compression='gzip')
        # Columns: timestamp, price, size, side
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        
        # Calculate microstructure metrics
        results = {}
        
        # 1. Trade flow imbalance over short windows
        df['minute'] = df['timestamp'].dt.floor('1min')
        minute_stats = df.groupby('minute').agg({
            'size': ['sum', 'count'],
            'side': lambda x: (x == 'Buy').sum() / len(x) if len(x) > 0 else 0.5
        }).reset_index()
        
        # 2. Large trade analysis
        large_threshold = df['size'].quantile(0.95)
        large_trades = df[df['size'] >= large_threshold]
        
        # 3. Price impact of large trades
        df['price_chg_1m'] = df['price'].shift(-10).pct_change() * 10000  # bps, ~1min later
        large_trades_impact = df[df['size'] >= large_threshold]['price_chg_1m'].mean()
        
        results = {
            'total_trades': len(df),
            'large_trade_threshold': large_threshold,
            'num_large_trades': len(large_trades),
            'avg_price_impact_large_bps': large_trades_impact if pd.notna(large_trades_impact) else 0,
            'buy_ratio_avg': minute_stats[('side', '<lambda>')].mean(),
            'volume_per_minute': df['size'].sum() / (df['timestamp'].max() - df['timestamp'].min()).total_seconds() * 60
        }
        
        return results
    except Exception as e:
        return {'error': str(e)}


def test_simple_market_making(symbol: str, start: str, end: str) -> dict:
    """
    Test simple market making: buy bid, sell ask, profit from spread
    With maker fees only (8 bps RT)
    """
    klines = load_klines(symbol, start, end)
    
    if len(klines) == 0:
        return None
    
    # Estimate spread from OHLC
    klines['spread_est'] = (klines['high'] - klines['low']) / klines['close'] * 10000  # bps
    klines['bar_return'] = klines['close'].pct_change() * 10000  # bps
    
    results = []
    
    # Test different "hold until profitable" windows
    for hold_minutes in [1, 5, 10, 30]:
        trades = []
        
        for i in range(len(klines) - hold_minutes):
            entry = klines.iloc[i]['close']
            exit_p = klines.iloc[i + hold_minutes]['close']
            
            # Assume we capture 50% of the high-low range as spread profit
            spread = klines.iloc[i]['spread_est'] * 0.5
            
            # Market making: buy at bid (below mid), sell at ask (above mid)
            # Profit = spread captured - fees
            position_value = 10000
            
            # Spread profit in bps
            spread_profit_bps = spread
            
            # Maker fees: 8 bps
            fees_bps = 8
            
            net_profit_bps = spread_profit_bps - fees_bps
            net_profit = net_profit_bps / 10000 * position_value
            
            trades.append(net_profit)
        
        if len(trades) > 0:
            total = sum(trades)
            wins = len([t for t in trades if t > 0])
            results.append({
                'hold_min': hold_minutes,
                'trades': len(trades),
                'total_net': total,
                'win_rate': wins / len(trades) * 100,
                'avg_profit_bps': np.mean(trades) / position_value * 10000
            })
    
    return results


def test_multi_factor_signal(symbol: str, start: str, end: str) -> dict:
    """
    Test combined signal: OI change + Funding + Price momentum
    """
    klines = load_klines(symbol, start, end)
    funding = load_funding_rates(symbol, start, end)
    oi = load_open_interest(symbol, start, end)
    
    if len(klines) == 0 or len(funding) == 0 or len(oi) == 0:
        return None
    
    # Merge all data
    klines = klines.set_index('timestamp')
    funding = funding.set_index('timestamp')
    oi = oi.set_index('timestamp')
    
    df = klines.join(funding[['fundingRate']], how='left')
    df = df.join(oi[['openInterest']], how='left')
    df = df.reset_index()
    
    # Fill forward
    df['fundingRate'] = df['fundingRate'].fillna(method='ffill')
    df['openInterest'] = df['openInterest'].fillna(method='ffill')
    
    # Calculate features
    df['oi_chg_1h'] = df['openInterest'].pct_change(60) * 100  # % change over 1h
    df['price_chg_1h'] = df['close'].pct_change(60) * 100  # % change over 1h
    df['fr_bps'] = df['fundingRate'] * 10000
    
    # Signal: OI rising + price flat/down + FR negative = long buildup, potential squeeze
    df['signal'] = 0
    df.loc[(df['oi_chg_1h'] > 2) & (df['price_chg_1h'] < 1) & (df['fr_bps'] < 0), 'signal'] = 1
    
    # Signal: OI falling + price rising = distribution, potential drop
    df.loc[(df['oi_chg_1h'] < -2) & (df['price_chg_1h'] > 2), 'signal'] = -1
    
    results = []
    
    for hold_bars in [30, 60, 120]:  # 30m, 1h, 2h
        trades = []
        
        for i in range(len(df) - hold_bars):
            sig = df.iloc[i]['signal']
            if sig == 0:
                continue
            
            entry = df.iloc[i]['close']
            exit_p = df.iloc[i + hold_bars]['close']
            
            position_value = 10000
            
            if sig == 1:  # Long
                ret = (exit_p - entry) / entry
            else:  # Short
                ret = (entry - exit_p) / entry
            
            gross = ret * position_value
            fees = position_value * FEE_MAKER_RT
            net = gross - fees
            
            trades.append({
                'net': net,
                'side': 'long' if sig == 1 else 'short',
                'ret_bps': ret * 10000
            })
        
        if len(trades) >= 5:
            total_net = sum(t['net'] for t in trades)
            wins = len([t for t in trades if t['net'] > 0])
            avg_ret = np.mean([t['ret_bps'] for t in trades])
            
            results.append({
                'hold_bars': hold_bars,
                'trades': len(trades),
                'win_rate': wins / len(trades) * 100,
                'total_net': total_net,
                'avg_ret_bps': avg_ret
            })
    
    return results


if __name__ == '__main__':
    print('='*70)
    print('DEEP STRATEGY SEARCH - Maker Fees & Microstructure')
    print('='*70)
    
    symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT']
    start = '2025-06-01'
    end = '2025-12-31'
    
    all_results = []
    
    # 1. Test maker fee FR hold
    print('\n1. MAKER FEE FR HOLD STRATEGY')
    print('-'*70)
    for sym in symbols:
        print(f'Testing {sym}...')
        res = test_maker_fr_hold(sym, start, end)
        if res:
            profitable = [r for r in res if r['total_net'] > 0 and r['trades'] >= 10]
            if profitable:
                best = max(profitable, key=lambda x: x['total_net'])
                print(f'  PROFITABLE: entry={best["entry"]}, exit={best["exit"]}, holds={best["max_holds"]}')
                print(f'    Trades: {best["trades"]}, WR: {best["win_rate"]:.1f}%, Net: ${best["total_net"]:.2f}')
                print(f'    Avg FR: {best["avg_fr"]:.1f} bps, Price drag: {best["avg_price"]:.1f} bps')
                all_results.append(('maker_fr_hold', sym, best))
    
    # 2. Test multi-factor signal
    print('\n2. MULTI-FACTOR SIGNAL (OI + Funding + Price)')
    print('-'*70)
    for sym in symbols:
        print(f'Testing {sym}...')
        res = test_multi_factor_signal(sym, start, end)
        if res:
            profitable = [r for r in res if r['total_net'] > 0]
            if profitable:
                best = max(profitable, key=lambda x: x['total_net'])
                print(f'  PROFITABLE: hold={best["hold_bars"]} bars')
                print(f'    Trades: {best["trades"]}, WR: {best["win_rate"]:.1f}%, Net: ${best["total_net"]:.2f}')
                all_results.append(('multi_factor', sym, best))
    
    # 3. Test trade data microstructure
    print('\n3. MICROSTRUCTURE ANALYSIS (Trade Data)')
    print('-'*70)
    for sym in symbols[:3]:
        date = '2025-02-01'
        print(f'Analyzing {sym} on {date}...')
        res = analyze_trade_data_microstructure(sym, date)
        if res and 'error' not in res:
            print(f'  Trades: {res["total_trades"]:,}')
            print(f'  Large trades (>95th): {res["num_large_trades"]}')
            print(f'  Avg price impact (large): {res["avg_price_impact_large_bps"]:.2f} bps')
            print(f'  Buy ratio: {res["buy_ratio_avg"]:.2%}')
    
    # Summary
    print('\n' + '='*70)
    print('SUMMARY OF ALL PROFITABLE FINDINGS')
    print('='*70)
    
    if all_results:
        for strategy, symbol, result in all_results:
            print(f'{strategy:20s} | {symbol:10s} | Net=${result["total_net"]:.2f} | WR={result.get("win_rate", 0):.1f}%')
    else:
        print('NO PROFITABLE STRATEGIES FOUND')
        print('\nPossible next steps:')
        print('- Test different date ranges (market regimes change)')
        print('- Look at tick-level data for HFT strategies')
        print('- Try cross-exchange arbitrage')
        print('- Look for 1h funding rate coins (different fee schedule)')
    
    print('='*70)
