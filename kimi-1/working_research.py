"""
Kimi-1 Strategy Research - Working Implementation
Based on proven edges from prior research.
"""
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Fees: Maker 0.04%, Taker 0.1% (round trip 0.2% with taker)
ROUND_TRIP_FEE = 0.002

def load_data(symbol, date, exchange='bybit'):
    """Load klines and funding for a symbol on a specific date."""
    base_path = f'/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}'
    
    # Load klines
    kline_file = f'{base_path}/{date}_kline_1m.csv'
    klines = pd.read_csv(kline_file)
    klines['timestamp'] = pd.to_datetime(klines['startTime'], unit='ms')
    
    # Load funding
    fr_file = f'{base_path}/{date}_funding_rate.csv'
    funding = pd.read_csv(fr_file)
    funding['timestamp'] = pd.to_datetime(funding['timestamp'], unit='ms')
    funding['fr_bps'] = funding['fundingRate'] * 10000
    
    return klines, funding


def test_fr_hold(symbol, date, entry_bps=1.0, exit_bps=0.3, max_hold_hours=24):
    """
    Test funding rate hold strategy.
    Long when FR >= entry_bps, exit when FR <= exit_bps or max hold.
    """
    try:
        klines, funding = load_data(symbol, date)
    except Exception as e:
        return None
    
    if len(funding) < 2:
        return None
    
    trades = []
    position = 0
    entry_time = None
    entry_price = 0
    entry_fr = 0
    
    for i, fr_row in funding.iterrows():
        fr = fr_row['fr_bps']
        ts = fr_row['timestamp']
        
        # Get price at this timestamp
        price_rows = klines[klines['timestamp'] >= ts]
        if price_rows.empty:
            continue
        price = price_rows.iloc[0]['close']
        
        # Entry logic
        if position == 0 and fr >= entry_bps:
            position = 1
            entry_time = ts
            entry_price = price
            entry_fr = fr
        
        # Exit logic
        elif position == 1:
            hours_held = (ts - entry_time).total_seconds() / 3600
            
            exit_signal = (fr <= exit_bps) or (hours_held >= max_hold_hours)
            
            if exit_signal:
                # Calculate P&L
                pnl_pct = (price - entry_price) / entry_price
                pnl_gross = pnl_pct * 10000  # $10k position
                fees = 10000 * ROUND_TRIP_FEE
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': ts,
                    'hours_held': hours_held,
                    'entry_fr': entry_fr,
                    'exit_fr': fr,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_gross': pnl_gross,
                    'fees': fees,
                    'pnl_net': pnl_net,
                    'won': pnl_net > 0
                })
                position = 0
    
    if len(trades) == 0:
        return None
    
    total_pnl = sum(t['pnl_net'] for t in trades)
    total_fees = sum(t['fees'] for t in trades)
    wins = sum(t['won'] for t in trades)
    
    return {
        'symbol': symbol,
        'date': date,
        'entry_bps': entry_bps,
        'exit_bps': exit_bps,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'edge': total_pnl - total_fees,
        'avg_trade': total_pnl / len(trades),
        'trades_data': trades
    }


def test_price_momentum(symbol, date, lookback=30, threshold=0.01):
    """
    Test simple momentum: go long after price increase > threshold.
    """
    try:
        klines, _ = load_data(symbol, date)
    except:
        return None
    
    if len(klines) < lookback + 10:
        return None
    
    klines['returns'] = klines['close'].pct_change()
    klines['cum_ret'] = klines['returns'].rolling(window=lookback).sum()
    klines['signal'] = klines['cum_ret'].shift(1)  # No lookahead
    
    trades = []
    position = 0
    entry_price = 0
    bars_held = 0
    max_bars = 60  # 1 hour max hold
    
    for i in range(lookback + 1, len(klines)):
        price = klines['close'].iloc[i]
        signal = klines['signal'].iloc[i]
        
        if position == 0 and signal > threshold:
            position = 1
            entry_price = price
            bars_held = 0
        
        elif position == 1:
            bars_held += 1
            
            # Exit on profit target, stop loss, or max hold
            pnl_pct = (price - entry_price) / entry_price
            
            exit_signal = (pnl_pct > 0.005) or (pnl_pct < -0.005) or (bars_held >= max_bars)
            
            if exit_signal:
                pnl_gross = pnl_pct * 10000
                fees = 10000 * ROUND_TRIP_FEE
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'pnl_net': pnl_net,
                    'won': pnl_net > 0
                })
                position = 0
    
    if len(trades) < 3:
        return None
    
    total_pnl = sum(t['pnl_net'] for t in trades)
    wins = sum(t['won'] for t in trades)
    
    return {
        'symbol': symbol,
        'date': date,
        'strategy': 'momentum',
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'total_fees': sum(10000 * ROUND_TRIP_FEE for _ in trades),
        'edge': total_pnl - sum(10000 * ROUND_TRIP_FEE for _ in trades),
        'avg_trade': total_pnl / len(trades)
    }


def scan_multiple_days(symbol, start_date, end_date, strategy='fr_hold'):
    """Scan multiple days for a symbol."""
    results = []
    
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        
        if strategy == 'fr_hold':
            # Test different thresholds
            for entry_bps in [0.5, 1.0, 1.5]:
                for exit_bps in [0.2, 0.3, 0.5]:
                    result = test_fr_hold(symbol, date_str, entry_bps, exit_bps)
                    if result and result['trades'] >= 2:
                        results.append(result)
        
        elif strategy == 'momentum':
            result = test_price_momentum(symbol, date_str)
            if result and result['trades'] >= 3:
                results.append(result)
        
        current += timedelta(days=1)
    
    return results


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 WORKING STRATEGY RESEARCH")
    print("=" * 100)
    
    # Test on a few symbols over a week
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    start_date = '2025-10-01'
    end_date = '2025-10-07'
    
    all_results = []
    
    print(f"\nScanning {len(symbols)} symbols from {start_date} to {end_date}...")
    print("-" * 100)
    
    for symbol in symbols:
        print(f"\n{symbol} - FR Hold Strategy:")
        results = scan_multiple_days(symbol, start_date, end_date, 'fr_hold')
        
        if results:
            # Show best config for this symbol
            best = max(results, key=lambda x: x['edge'])
            print(f"  Best: E={best['entry_bps']:.1f} X={best['exit_bps']:.1f} | "
                  f"{best['trades']} trades | WR={best['win_rate']:.0%} | "
                  f"Net=${best['total_pnl']:.2f} | Edge=${best['edge']:.2f}")
            all_results.extend(results)
        else:
            print(f"  No valid results")
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Aggregate by symbol
        print("\nAggregated by symbol (best config per symbol):")
        for symbol in symbols:
            sym_results = [r for r in all_results if r['symbol'] == symbol]
            if sym_results:
                best = max(sym_results, key=lambda x: x['edge'])
                total_trades = sum(r['trades'] for r in sym_results)
                total_pnl = sum(r['total_pnl'] for r in sym_results)
                total_edge = sum(r['edge'] for r in sym_results)
                print(f"  {symbol:12s} | {total_trades:3d} total trades | "
                      f"Net=${total_pnl:8.2f} | Edge=${total_edge:8.2f}")
        
        # Overall stats
        total_trades = sum(r['trades'] for r in all_results)
        total_pnl = sum(r['total_pnl'] for r in all_results)
        total_edge = sum(r['edge'] for r in all_results)
        
        print(f"\nOverall: {total_trades} trades, ${total_pnl:.2f} net, ${total_edge:.2f} edge after fees")
        
        # Profitable configs
        profitable = [r for r in all_results if r['edge'] > 0]
        print(f"\nProfitable configs: {len(profitable)}/{len(all_results)}")
        
        if profitable:
            print("\nTop 5 by edge:")
            for r in sorted(profitable, key=lambda x: x['edge'], reverse=True)[:5]:
                print(f"  {r['symbol']:12s} | E={r['entry_bps']:.1f} X={r['exit_bps']:.1f} | "
                      f"{r['trades']} trades | Edge=${r['edge']:.2f}")
    else:
        print("\nNo results generated.")
        print("8h funding rates are too small (1 bps) to overcome 20 bps fees.")
        print("Need 1h funding coins or different strategy type.")
    
    print("\n" + "=" * 100)
