"""
Kimi-1 Expanded Audit - More Coins + Higher Trade Frequency
Optimized parameters: shorter lookback, tighter threshold
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002

def get_daily_fast(symbol):
    """Fast daily data loader."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 100:
        return None, 0
    
    # Load all files
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
            all_data.append(df)
        except:
            pass
    
    if not all_data:
        return None, 0
    
    df = pd.concat(all_data)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    daily = df.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    return daily, len(daily)


def breakout_optimized(daily, lookback=3, threshold=0.005, hold_days=1, risk=0.02):
    """
    Optimized breakout for higher trade frequency:
    - lookback=3 (instead of 5) = more signals
    - threshold=0.5% (instead of 1%) = more sensitive
    """
    if len(daily) < lookback + hold_days + 10:
        return None
    
    daily['highest'] = daily['high'].rolling(window=lookback).max().shift(1)
    
    capital = 10000
    trades = []
    
    for i in range(lookback + 1, len(daily) - hold_days):
        price = daily['close'].iloc[i]
        breakout_level = daily['highest'].iloc[i] * (1 + threshold)
        
        if price > breakout_level:
            entry_price = price
            exit_price = daily['close'].iloc[i + hold_days]
            
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_gross = pnl_pct * capital * risk
            fees = capital * risk * FEE_PCT
            pnl_net = pnl_gross - fees
            
            trades.append({'pnl_net': pnl_net, 'won': pnl_net > 0})
    
    if len(trades) < 10:
        return None
    
    wins = sum(t['won'] for t in trades)
    return {
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': sum(t['pnl_net'] for t in trades),
        'avg_trade': sum(t['pnl_net'] for t in trades) / len(trades),
        'profitable': sum(t['pnl_net'] for t in trades) > 0
    }


def oos_test_fast(symbol, n_splits=5):
    """Fast OOS test with optimized parameters."""
    daily, n_days = get_daily_fast(symbol)
    if daily is None or n_days < 200:
        return None
    
    train_size = 120  # 4 months
    test_size = 40    # ~6 weeks
    results = []
    
    for i in range(n_splits):
        window_start = i * 20
        test_start = window_start + train_size
        test_end = test_start + test_size
        
        if test_end >= len(daily):
            break
        
        test_data = daily.iloc[test_start:test_end]
        result = breakout_optimized(test_data)
        if result:
            results.append(result)
    
    if len(results) < 3:
        return None
    
    total_pnl = sum(r['total_pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    wins = sum(r['win_rate'] * r['trades'] for r in results)
    
    return {
        'symbol': symbol,
        'days': n_days,
        'splits': len(results),
        'trades': total_trades,
        'win_rate': wins / total_trades if total_trades > 0 else 0,
        'total_pnl': total_pnl,
        'avg_pnl_per_split': total_pnl / len(results),
        'consistency': sum(1 for r in results if r['total_pnl'] > 0) / len(results),
        'profitable': total_pnl > 0
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 EXPANDED AUDIT - Optimized for Higher Trade Frequency")
    print("=" * 100)
    print("\nStrategy: 3-day breakout + 0.5% threshold (vs 5-day + 1%)")
    print("-" * 100)
    
    # Get major coins
    major_coins = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT',
        'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT', 'UNIUSDT',
        'AAVEUSDT', 'NEARUSDT', 'FILUSDT', 'ATOMUSDT', 'ARBUSDT', 'OPUSDT',
        'APTUSDT', 'SUIUSDT', 'TRXUSDT', 'ETCUSDT', 'ALGOUSDT', 'VETUSDT',
        'ICPUSDT', 'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'THETAUSDT', 'XTZUSDT',
        'BNBUSDT', 'XLMUSDT', 'KAITOUSDT', 'TONUSDT', 'SHIB1000USDT',
        'PEPEUSDT', 'WIFUSDT', 'BONKUSDT', 'FLOKIUSDT', 'ENSUSDT',
        'CRVUSDT', 'LDOUSDT', 'SUSHIUSDT', 'COMPUSDT', 'MKRUSDT',
        'YFIUSDT', '1INCHUSDT', 'GRTUSDT', 'SNXUSDT', 'BATUSDT'
    ]
    
    results = []
    for i, sym in enumerate(major_coins):
        result = oos_test_fast(sym)
        if result:
            results.append(result)
            status = "✓" if result['profitable'] else "✗"
            print(f"[{i+1:2d}/{len(major_coins)}] {status} {result['symbol']:15s} | "
                  f"{result['trades']:3d} trades | {result['splits']} splits | "
                  f"Cons: {result['consistency']:.0%} | PnL: ${result['total_pnl']:+7.2f}")
    
    print("\n" + "=" * 100)
    print("EXPANDED AUDIT RESULTS")
    print("=" * 100)
    
    if results:
        df = pd.DataFrame(results)
        profitable = df[df['profitable']].sort_values('total_pnl', ascending=False)
        
        print(f"\nTotal tested: {len(results)} coins")
        print(f"Profitable: {len(profitable)} ({len(profitable)/len(results):.0%})")
        print(f"Total trades across all coins: {df['trades'].sum()}")
        print(f"Avg trades per coin: {df['trades'].mean():.1f}")
        
        if len(profitable) > 0:
            print("\n" + "=" * 100)
            print("✓ PROFITABLE COINS (3-day breakout, 0.5% threshold)")
            print("=" * 100)
            
            for _, r in profitable.head(15).iterrows():
                print(f"  {r['symbol']:15s} | {r['trades']:3d} trades | "
                      f"WR: {r['win_rate']:.0%} | Cons: {r['consistency']:.0%} | "
                      f"PnL: ${r['total_pnl']:+8.2f}")
            
            # Save results
            profitable.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/EXPANDED_AUDIT_RESULTS.csv', index=False)
            print(f"\nResults saved to EXPANDED_AUDIT_RESULTS.csv")
    
    print("\n" + "=" * 100)
