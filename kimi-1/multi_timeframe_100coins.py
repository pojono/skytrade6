"""
Kimi-1 Multi-Timeframe Test - 100+ Coins Full Power
Test breakout strategy on 1h, 2h, 4h, and daily timeframes
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002


def get_all_bybit_symbols():
    """Get all symbols with sufficient data from datalake."""
    path = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')
    symbols = []
    
    for d in path.iterdir():
        if d.is_dir():
            sym = d.name
            # Quick check - at least 100 kline files
            files = list(d.glob('*_kline_1m.csv'))
            if len(files) >= 100:
                symbols.append(sym)
    
    return sorted(symbols)


def load_ohlcv(symbol, timeframe='1D'):
    """
    Load and resample data to specified timeframe.
    timeframe: '1H', '2H', '4H', '1D'
    """
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 100:
        return None, 0
    
    all_data = []
    # Load first 300 files max for speed
    for f in files[:300]:
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
    
    # Resample to target timeframe
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled, len(resampled)


def breakout_backtest(df, lookback=5, threshold=0.01, hold_bars=1, risk=0.02):
    """
    Simple breakout backtest.
    Returns dict with results or None if insufficient data.
    """
    if len(df) < lookback + hold_bars + 20:
        return None
    
    df = df.copy()
    df['highest'] = df['high'].rolling(window=lookback).max().shift(1)
    
    capital = 10000
    trades = []
    
    for i in range(lookback + 1, len(df) - hold_bars):
        price = df['close'].iloc[i]
        breakout_level = df['highest'].iloc[i] * (1 + threshold)
        
        if price > breakout_level:
            entry = price
            exit_price = df['close'].iloc[i + hold_bars]
            
            pnl_pct = (exit_price - entry) / entry
            pnl_gross = pnl_pct * capital * risk
            fees = capital * risk * FEE_PCT
            pnl_net = pnl_gross - fees
            
            trades.append(pnl_net)
    
    if len(trades) < 10:
        return None
    
    wins = sum(1 for p in trades if p > 0)
    return {
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': sum(trades),
        'avg_trade': sum(trades) / len(trades),
        'profitable': sum(trades) > 0,
        'sharpe': np.mean(trades) / np.std(trades) if np.std(trades) > 0 else 0
    }


def optimize_params(timeframe):
    """Get optimized lookback/threshold for each timeframe."""
    # Adjust lookback based on timeframe (more bars for shorter timeframes)
    params = {
        '1H': {'lookback': 24, 'threshold': 0.008, 'hold_bars': 4},   # 24h lookback, hold 4h
        '2H': {'lookback': 12, 'threshold': 0.01, 'hold_bars': 2},    # 24h lookback, hold 4h
        '4H': {'lookback': 6, 'threshold': 0.012, 'hold_bars': 1},    # 24h lookback, hold 4h
        '1D': {'lookback': 5, 'threshold': 0.01, 'hold_bars': 1},    # 5-day lookback, hold 1d
    }
    return params.get(timeframe, {'lookback': 5, 'threshold': 0.01, 'hold_bars': 1})


def test_symbol_timeframe(symbol, timeframe):
    """Test single symbol on single timeframe."""
    df, n_bars = load_ohlcv(symbol, timeframe)
    if df is None or n_bars < 100:
        return None
    
    params = optimize_params(timeframe)
    result = breakout_backtest(df, **params)
    
    if result:
        result['symbol'] = symbol
        result['timeframe'] = timeframe
        result['bars'] = n_bars
        return result
    return None


def run_full_test(max_coins=None):
    """Run comprehensive test on all available coins."""
    print("=" * 100)
    print("KIMI-1 MULTI-TIMEFRAME TEST - 100+ COINS FULL POWER")
    print("=" * 100)
    
    symbols = get_all_bybit_symbols()
    if max_coins:
        symbols = symbols[:max_coins]
    
    print(f"\nFound {len(symbols)} symbols with sufficient data")
    print("Testing timeframes: 1H, 2H, 4H, 1D")
    print("-" * 100)
    
    timeframes = ['1H', '2H', '4H', '1D']
    all_results = []
    
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol}")
        symbol_results = []
        
        for tf in timeframes:
            result = test_symbol_timeframe(symbol, tf)
            if result:
                symbol_results.append(result)
                all_results.append(result)
                status = "✓" if result['profitable'] else "✗"
                print(f"  {status} {tf}: {result['trades']:3d}T ${result['total_pnl']:+6.0f} ({result['win_rate']:.0%})")
        
        if not symbol_results:
            print(f"     No results (insufficient data)")
    
    # Analysis
    print("\n" + "=" * 100)
    print("RESULTS BY TIMEFRAME")
    print("=" * 100)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        for tf in timeframes:
            tf_data = df[df['timeframe'] == tf]
            if len(tf_data) == 0:
                continue
            
            prof = tf_data[tf_data['profitable']]
            print(f"\n{tf}:")
            print(f"  Tested: {len(tf_data)} coin-tests")
            print(f"  Profitable: {len(prof)} ({len(prof)/len(tf_data):.0%})")
            print(f"  Total trades: {tf_data['trades'].sum()}")
            print(f"  Avg trades/coin: {tf_data['trades'].mean():.1f}")
            print(f"  Avg PnL: ${tf_data['total_pnl'].mean():+.2f}")
            
            if len(prof) > 0:
                top = prof.sort_values('total_pnl', ascending=False).head(5)
                print(f"  Top 5:")
                for _, row in top.iterrows():
                    print(f"    {row['symbol']:12s} ${row['total_pnl']:+7.2f} ({row['trades']}T)")
        
        # Best overall
        print("\n" + "=" * 100)
        print("BEST PERFORMERS (All Timeframes)")
        print("=" * 100)
        
        prof_all = df[df['profitable']].sort_values('total_pnl', ascending=False).head(15)
        for _, row in prof_all.iterrows():
            print(f"  {row['symbol']:12s} {row['timeframe']:3s} | "
                  f"${row['total_pnl']:+7.2f} | {row['trades']:3d}T | {row['win_rate']:.0%} WR")
        
        # Save results
        df.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/MULTI_TIMEFRAME_100COINS.csv', index=False)
        print(f"\n✓ Results saved to MULTI_TIMEFRAME_100COINS.csv")
    
    print("\n" + "=" * 100)
    return all_results


if __name__ == '__main__':
    # Test all available coins (or limit for faster testing)
    results = run_full_test(max_coins=None)  # Set to 50 for quick test, None for all
