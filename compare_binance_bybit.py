#!/usr/bin/env python3
"""
Compare Binance vs Bybit data quality and coverage.
Analyze price differences, volume patterns, and data reliability.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from datetime import datetime
import matplotlib.pyplot as plt

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def load_binance_klines(symbol):
    """Load Binance 1h klines."""
    file = Path(f"data_binance/{symbol}/binance_klines_1h.csv.gz")
    if not file.exists():
        return None
    
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['source'] = 'binance'
    
    return df[['open', 'high', 'low', 'close', 'volume']]

def load_bybit_ticker_aggregated(symbol):
    """Load and aggregate Bybit 5-second ticker to 1h bars."""
    ticker_dir = Path(f"data/{symbol}")
    ticker_files = sorted(ticker_dir.glob("ticker_*.jsonl.gz"))
    
    if not ticker_files:
        return None
    
    print(f"  Loading {len(ticker_files)} Bybit ticker files for {symbol}...")
    
    records = []
    for i, file in enumerate(ticker_files):
        if i % 200 == 0:
            print(f"    Progress: {i}/{len(ticker_files)}...")
        
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    result = data['result']['list'][0]
                    
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'last_price': float(result['lastPrice']),
                        'volume_24h': float(result['volume24h']),
                    })
                except:
                    continue
    
    df = pd.DataFrame(records)
    df = df.set_index('timestamp')
    
    # Aggregate to 1h bars
    hourly = df.resample('1H').agg({
        'last_price': ['first', 'max', 'min', 'last'],
        'volume_24h': 'last'
    })
    
    hourly.columns = ['open', 'high', 'low', 'close', 'volume_24h']
    hourly['source'] = 'bybit'
    
    return hourly

def compare_symbols(symbol):
    """Compare Binance vs Bybit data for a symbol."""
    print(f"\n{'='*70}")
    print(f"Comparing {symbol}")
    print(f"{'='*70}")
    
    # Load data
    binance_df = load_binance_klines(symbol)
    bybit_df = load_bybit_ticker_aggregated(symbol)
    
    if binance_df is None or bybit_df is None:
        print(f"  ✗ Missing data for {symbol}")
        return None
    
    print(f"\nData Coverage:")
    print(f"  Binance: {len(binance_df)} bars, {binance_df.index.min()} to {binance_df.index.max()}")
    print(f"  Bybit:   {len(bybit_df)} bars, {bybit_df.index.min()} to {bybit_df.index.max()}")
    
    # Merge on timestamp
    merged = pd.merge(
        binance_df,
        bybit_df,
        left_index=True,
        right_index=True,
        suffixes=('_binance', '_bybit'),
        how='inner'
    )
    
    print(f"\nOverlapping bars: {len(merged)}")
    
    if len(merged) == 0:
        print(f"  ✗ No overlapping data")
        return None
    
    # Calculate price differences
    merged['close_diff'] = merged['close_binance'] - merged['close_bybit']
    merged['close_diff_pct'] = (merged['close_diff'] / merged['close_binance']) * 100
    
    merged['high_diff_pct'] = ((merged['high_binance'] - merged['high_bybit']) / merged['high_binance']) * 100
    merged['low_diff_pct'] = ((merged['low_binance'] - merged['low_bybit']) / merged['low_binance']) * 100
    
    # Statistics
    print(f"\nPrice Comparison (Close):")
    print(f"  Mean difference: {merged['close_diff_pct'].mean():.4f}%")
    print(f"  Std difference:  {merged['close_diff_pct'].std():.4f}%")
    print(f"  Max difference:  {merged['close_diff_pct'].max():.4f}%")
    print(f"  Min difference:  {merged['close_diff_pct'].min():.4f}%")
    print(f"  Median diff:     {merged['close_diff_pct'].median():.4f}%")
    
    # Correlation
    corr = merged['close_binance'].corr(merged['close_bybit'])
    print(f"\nPrice Correlation: {corr:.6f}")
    
    # Large discrepancies
    large_diff = merged[abs(merged['close_diff_pct']) > 0.1]
    if len(large_diff) > 0:
        print(f"\nLarge discrepancies (>0.1%):")
        print(f"  Count: {len(large_diff)}")
        print(f"  Max: {large_diff['close_diff_pct'].abs().max():.4f}%")
        print(f"  Examples:")
        for idx in large_diff.head(3).index:
            print(f"    {idx}: Binance=${merged.loc[idx, 'close_binance']:.2f}, "
                  f"Bybit=${merged.loc[idx, 'close_bybit']:.2f}, "
                  f"Diff={merged.loc[idx, 'close_diff_pct']:.4f}%")
    
    # Returns comparison
    merged['ret_binance'] = merged['close_binance'].pct_change()
    merged['ret_bybit'] = merged['close_bybit'].pct_change()
    
    ret_corr = merged['ret_binance'].corr(merged['ret_bybit'])
    print(f"\nReturns Correlation: {ret_corr:.6f}")
    
    # Volatility comparison
    vol_binance = merged['ret_binance'].std() * np.sqrt(365 * 24)
    vol_bybit = merged['ret_bybit'].std() * np.sqrt(365 * 24)
    
    print(f"\nAnnualized Volatility:")
    print(f"  Binance: {vol_binance*100:.2f}%")
    print(f"  Bybit:   {vol_bybit*100:.2f}%")
    print(f"  Ratio:   {vol_binance/vol_bybit:.4f}")
    
    return merged

def main():
    print("="*70)
    print("BINANCE vs BYBIT DATA COMPARISON")
    print("="*70)
    print(f"Period: May 11 - Aug 10, 2025")
    print(f"Resolution: 1 hour")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print("="*70)
    
    results = {}
    
    for symbol in SYMBOLS:
        merged = compare_symbols(symbol)
        if merged is not None:
            results[symbol] = merged
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for symbol, df in results.items():
        print(f"\n{symbol}:")
        print(f"  Overlapping bars: {len(df)}")
        print(f"  Price correlation: {df['close_binance'].corr(df['close_bybit']):.6f}")
        print(f"  Returns correlation: {df['ret_binance'].corr(df['ret_bybit']):.6f}")
        print(f"  Mean price diff: {df['close_diff_pct'].mean():.4f}%")
        print(f"  Std price diff: {df['close_diff_pct'].std():.4f}%")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    # Overall assessment
    all_corrs = [df['close_binance'].corr(df['close_bybit']) for df in results.values()]
    avg_corr = np.mean(all_corrs)
    
    all_diffs = [df['close_diff_pct'].abs().mean() for df in results.values()]
    avg_diff = np.mean(all_diffs)
    
    print(f"\nAverage price correlation: {avg_corr:.6f}")
    print(f"Average absolute price difference: {avg_diff:.4f}%")
    
    if avg_corr > 0.9999:
        print("\n✅ EXCELLENT: Binance and Bybit data are nearly identical")
        print("   Both exchanges can be used interchangeably for research")
    elif avg_corr > 0.999:
        print("\n✅ GOOD: Binance and Bybit data are highly correlated")
        print("   Minor differences exist but both are reliable")
    elif avg_corr > 0.99:
        print("\n⚠️  MODERATE: Some differences between exchanges")
        print("   Consider using both for validation")
    else:
        print("\n❌ POOR: Significant differences between exchanges")
        print("   Data quality issues may exist")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
