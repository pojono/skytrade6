"""
Daily Breakout Strategy - Broad Universe Validation
Test validated daily breakout on full 100+ coin universe with OOS walk-forward
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
            files = list(d.glob('*_kline_1m.csv'))
            if len(files) >= 100:
                symbols.append(sym)
    
    return sorted(symbols)


def load_ohlcv_daily(symbol):
    """Load and resample data to daily timeframe."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 100:
        return None, None
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
            all_data.append(df)
        except:
            pass
    
    if not all_data:
        return None, None
    
    df = pd.concat(all_data)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    # Resample to daily
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return daily, daily.index.min()


def breakout_backtest_daily(df, lookback=5, threshold=0.01, hold_bars=1, risk=0.02):
    """
    Daily breakout backtest with walk-forward OOS splits.
    Returns dict with results or None if insufficient data.
    """
    if len(df) < lookback + hold_bars + 50:
        return None
    
    df = df.copy()
    df['highest'] = df['high'].rolling(window=lookback).max().shift(1)
    
    # Walk-forward: 3 splits
    n = len(df)
    splits = [
        (0, int(n * 0.5)),           # First 50%
        (int(n * 0.25), int(n * 0.75)),  # Middle 50%
        (int(n * 0.5), n)            # Last 50%
    ]
    
    all_trades = []
    split_results = []
    
    for start, end in splits:
        if end - start < lookback + hold_bars + 10:
            continue
            
        split_df = df.iloc[start:end].copy()
        trades = []
        
        for i in range(lookback + 1, len(split_df) - hold_bars):
            price = split_df['close'].iloc[i]
            breakout_level = split_df['highest'].iloc[i] * (1 + threshold)
            
            if price > breakout_level:
                entry = price
                exit_price = split_df['close'].iloc[i + hold_bars]
                
                pnl_pct = (exit_price - entry) / entry
                pnl_gross = pnl_pct * 10000 * risk
                fees = 10000 * risk * FEE_PCT
                pnl_net = pnl_gross - fees
                
                trades.append(pnl_net)
        
        if len(trades) >= 3:
            split_pnl = sum(trades)
            split_results.append({
                'profitable': split_pnl > 0,
                'trades': len(trades),
                'pnl': split_pnl
            })
            all_trades.extend(trades)
    
    if len(all_trades) < 10 or len(split_results) < 2:
        return None
    
    wins = sum(1 for p in all_trades if p > 0)
    oos_profitable = all(s['profitable'] for s in split_results)
    
    return {
        'trades': len(all_trades),
        'win_rate': wins / len(all_trades),
        'total_pnl': sum(all_trades),
        'avg_trade': sum(all_trades) / len(all_trades),
        'profitable': sum(all_trades) > 0,
        'oos_profitable': oos_profitable,
        'oos_splits_profitable': sum(1 for s in split_results if s['profitable']),
        'oos_total_splits': len(split_results),
        'sharpe': np.mean(all_trades) / np.std(all_trades) if np.std(all_trades) > 0 else 0,
        'start_date': df.index.min().strftime('%Y-%m'),
        'end_date': df.index.max().strftime('%Y-%m'),
        'days': len(df)
    }


def test_daily_strategy(symbol, params=None):
    """Test daily breakout on single symbol."""
    if params is None:
        params = {'lookback': 5, 'threshold': 0.01, 'hold_bars': 1, 'risk': 0.02}
    
    df, start_date = load_ohlcv_daily(symbol)
    if df is None or len(df) < 100:
        return None
    
    result = breakout_backtest_daily(df, **params)
    if result:
        result['symbol'] = symbol
        result['params'] = params
        return result
    return None


def run_daily_broad_test():
    """Run daily breakout on full coin universe."""
    print("=" * 100)
    print("DAILY BREAKOUT - BROAD UNIVERSE VALIDATION")
    print("=" * 100)
    print("Strategy: 5-day breakout, 1% threshold, 1-day hold")
    print("Validation: OOS walk-forward (3 splits)")
    print("-" * 100)
    
    symbols = get_all_bybit_symbols()
    print(f"\nFound {len(symbols)} symbols")
    
    results = []
    oos_robust = []  # Profitable in ALL OOS splits
    
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol}", end=" ")
        
        result = test_daily_strategy(symbol)
        if result:
            results.append(result)
            status = "✓" if result['profitable'] else "✗"
            oos_mark = "★" if result['oos_profitable'] else ""
            print(f"{status}{oos_mark} ${result['total_pnl']:+6.0f} ({result['trades']}T, {result['win_rate']:.0%} WR, OOS: {result['oos_splits_profitable']}/{result['oos_total_splits']})")
            
            if result['oos_profitable']:
                oos_robust.append(result)
        else:
            print("- No data/insufficient")
    
    # Summary
    print("\n" + "=" * 100)
    print("BROAD UNIVERSE RESULTS")
    print("=" * 100)
    
    if results:
        df = pd.DataFrame(results)
        prof = df[df['profitable']]
        oos = df[df['oos_profitable']]
        
        print(f"\nTotal symbols tested: {len(results)}")
        print(f"Profitable (gross): {len(prof)} ({len(prof)/len(df)*100:.1f}%)")
        print(f"OOS Robust (all splits profitable): {len(oos)} ({len(oos)/len(df)*100:.1f}%)")
        print(f"Total trades: {df['trades'].sum()}")
        print(f"Average trades/coin: {df['trades'].mean():.1f}")
        print(f"Average PnL: ${df['total_pnl'].mean():+.2f}")
        
        if len(prof) > 0:
            print(f"\n--- Top 10 Gross Profitable ---")
            top = prof.sort_values('total_pnl', ascending=False).head(10)
            for _, row in top.iterrows():
                oos_badge = "[OOS]" if row['oos_profitable'] else ""
                print(f"  {row['symbol']:12s} ${row['total_pnl']:+7.2f} | {row['trades']:3d}T | {row['win_rate']:.0%} WR {oos_badge}")
        
        if len(oos) > 0:
            print(f"\n--- OOS ROBUST (All Splits Profitable) ---")
            top_oos = oos.sort_values('total_pnl', ascending=False)
            for _, row in top_oos.iterrows():
                print(f"  {row['symbol']:12s} ${row['total_pnl']:+7.2f} | {row['trades']:3d}T | {row['win_rate']:.0%} WR")
            print(f"\nOOS Robust average PnL: ${oos['total_pnl'].mean():+.2f}")
        else:
            print(f"\n⚠️ No OOS-robust coins found!")
        
        # Save results
        df.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/DAILY_BROAD_UNIVERSE.csv', index=False)
        print(f"\n✓ Results saved to DAILY_BROAD_UNIVERSE.csv")
    
    print("\n" + "=" * 100)
    return results


if __name__ == '__main__':
    results = run_daily_broad_test()
