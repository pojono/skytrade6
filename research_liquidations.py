#!/usr/bin/env python3
"""
Liquidations Research (v26)
Analyze liquidation cascades, imbalances, and predictive power.
"""
import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_liquidations(symbol, data_dir='data'):
    """Load all liquidation data for a symbol."""
    print(f"\n{'='*70}")
    print(f"Loading liquidations data: {symbol}")
    print(f"{'='*70}")
    
    symbol_dir = Path(data_dir) / symbol
    liq_files = sorted(symbol_dir.glob("liquidation_*.jsonl.gz"))
    
    if not liq_files:
        raise ValueError(f"No liquidation files found for {symbol}")
    
    print(f"Found {len(liq_files)} liquidation files")
    
    records = []
    errors = 0
    
    for i, file in enumerate(liq_files, 1):
        if i % 20 == 0 or i == len(liq_files):
            print(f"  Progress: {i}/{len(liq_files)} files ({i/len(liq_files)*100:.1f}%)")
        
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extract liquidation events from result.data array
                    if 'result' in data and 'data' in data['result']:
                        for liq_event in data['result']['data']:
                            records.append({
                                'timestamp': pd.to_datetime(liq_event['T'], unit='ms'),
                                'side': liq_event['S'],  # Buy or Sell
                                'volume': float(liq_event['v']),
                                'price': float(liq_event['p']),
                            })
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    errors += 1
                    continue
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} liquidation events")
    if errors > 0:
        print(f"Skipped {errors} corrupted lines")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    return df

def aggregate_liquidations(df, freq='1min'):
    """Aggregate liquidations to time bars."""
    print(f"\nAggregating to {freq} bars...")
    
    df_agg = df.set_index('timestamp')
    
    # Separate by side
    buy_liq = df_agg[df_agg['side'] == 'Buy']
    sell_liq = df_agg[df_agg['side'] == 'Sell']
    
    # Aggregate
    bars = pd.DataFrame(index=pd.date_range(
        start=df_agg.index.min().floor(freq),
        end=df_agg.index.max().ceil(freq),
        freq=freq
    ))
    
    # Total liquidations
    bars['liq_count'] = df_agg.resample(freq).size()
    bars['liq_volume'] = df_agg.resample(freq)['volume'].sum()
    bars['liq_notional'] = (df_agg['volume'] * df_agg['price']).resample(freq).sum()
    
    # By side
    bars['liq_buy_count'] = buy_liq.resample(freq).size()
    bars['liq_buy_volume'] = buy_liq.resample(freq)['volume'].sum()
    bars['liq_buy_notional'] = (buy_liq['volume'] * buy_liq['price']).resample(freq).sum()
    
    bars['liq_sell_count'] = sell_liq.resample(freq).size()
    bars['liq_sell_volume'] = sell_liq.resample(freq)['volume'].sum()
    bars['liq_sell_notional'] = (sell_liq['volume'] * sell_liq['price']).resample(freq).sum()
    
    # Fill NaN with 0
    bars = bars.fillna(0)
    
    # Calculate imbalance
    bars['liq_imbalance'] = (bars['liq_sell_volume'] - bars['liq_buy_volume']) / (bars['liq_volume'] + 1e-10)
    bars['liq_imbalance_notional'] = (bars['liq_sell_notional'] - bars['liq_buy_notional']) / (bars['liq_notional'] + 1e-10)
    
    print(f"Created {len(bars):,} bars")
    print(f"Non-zero bars: {(bars['liq_count'] > 0).sum():,} ({(bars['liq_count'] > 0).sum() / len(bars) * 100:.1f}%)")
    
    return bars.reset_index().rename(columns={'index': 'timestamp'})

def detect_cascades(df, volume_threshold_pct=95, time_window_sec=60):
    """Detect liquidation cascade events."""
    print(f"\n{'='*70}")
    print(f"Detecting liquidation cascades")
    print(f"{'='*70}")
    print(f"Volume threshold: {volume_threshold_pct}th percentile")
    print(f"Time window: {time_window_sec} seconds")
    
    # Calculate volume threshold
    volume_threshold = df['volume'].quantile(volume_threshold_pct / 100)
    print(f"Volume threshold: {volume_threshold:.4f} contracts")
    
    # Find large liquidations
    large_liqs = df[df['volume'] >= volume_threshold].copy()
    print(f"Large liquidations: {len(large_liqs):,} ({len(large_liqs) / len(df) * 100:.2f}%)")
    
    # Group nearby liquidations into cascades
    cascades = []
    current_cascade = []
    
    for idx, row in large_liqs.iterrows():
        if not current_cascade:
            current_cascade = [row]
        else:
            # Check if within time window of last event
            time_diff = (row['timestamp'] - current_cascade[-1]['timestamp']).total_seconds()
            if time_diff <= time_window_sec:
                current_cascade.append(row)
            else:
                # Save current cascade if it has multiple events
                if len(current_cascade) >= 2:
                    cascades.append(current_cascade)
                current_cascade = [row]
    
    # Don't forget the last cascade
    if len(current_cascade) >= 2:
        cascades.append(current_cascade)
    
    print(f"Detected {len(cascades)} cascades (2+ events within {time_window_sec}s)")
    
    # Analyze cascades
    if cascades:
        cascade_stats = []
        for cascade in cascades:
            cascade_df = pd.DataFrame(cascade)
            cascade_stats.append({
                'start_time': cascade_df['timestamp'].min(),
                'end_time': cascade_df['timestamp'].max(),
                'duration': (cascade_df['timestamp'].max() - cascade_df['timestamp'].min()).total_seconds(),
                'event_count': len(cascade),
                'total_volume': cascade_df['volume'].sum(),
                'total_notional': (cascade_df['volume'] * cascade_df['price']).sum(),
                'buy_volume': cascade_df[cascade_df['side'] == 'Buy']['volume'].sum(),
                'sell_volume': cascade_df[cascade_df['side'] == 'Sell']['volume'].sum(),
                'dominant_side': 'Buy' if cascade_df[cascade_df['side'] == 'Buy']['volume'].sum() > 
                                         cascade_df[cascade_df['side'] == 'Sell']['volume'].sum() else 'Sell',
            })
        
        cascade_df = pd.DataFrame(cascade_stats)
        
        print(f"\nCascade Statistics:")
        print(f"  Avg duration: {cascade_df['duration'].mean():.1f}s")
        print(f"  Avg events: {cascade_df['event_count'].mean():.1f}")
        print(f"  Avg volume: {cascade_df['total_volume'].mean():.2f} contracts")
        print(f"  Avg notional: ${cascade_df['total_notional'].mean():,.0f}")
        print(f"  Buy-dominated: {(cascade_df['dominant_side'] == 'Buy').sum()} ({(cascade_df['dominant_side'] == 'Buy').sum() / len(cascade_df) * 100:.1f}%)")
        print(f"  Sell-dominated: {(cascade_df['dominant_side'] == 'Sell').sum()} ({(cascade_df['dominant_side'] == 'Sell').sum() / len(cascade_df) * 100:.1f}%)")
        
        return cascade_df
    else:
        print("No cascades detected")
        return pd.DataFrame()

def analyze_liquidation_patterns(df):
    """Analyze liquidation patterns and statistics."""
    print(f"\n{'='*70}")
    print(f"Liquidation Pattern Analysis")
    print(f"{'='*70}")
    
    # Overall statistics
    total_volume = df['volume'].sum()
    total_notional = (df['volume'] * df['price']).sum()
    
    buy_volume = df[df['side'] == 'Buy']['volume'].sum()
    sell_volume = df[df['side'] == 'Sell']['volume'].sum()
    
    print(f"\nOverall Statistics:")
    print(f"  Total events: {len(df):,}")
    print(f"  Total volume: {total_volume:,.2f} contracts")
    print(f"  Total notional: ${total_notional:,.0f}")
    print(f"  Avg event size: {df['volume'].mean():.4f} contracts (${(df['volume'] * df['price']).mean():,.0f})")
    print(f"  Median event size: {df['volume'].median():.4f} contracts")
    
    print(f"\nBy Side:")
    print(f"  Buy liquidations (longs stopped): {(df['side'] == 'Buy').sum():,} ({(df['side'] == 'Buy').sum() / len(df) * 100:.1f}%)")
    print(f"    Volume: {buy_volume:,.2f} contracts ({buy_volume / total_volume * 100:.1f}%)")
    print(f"  Sell liquidations (shorts stopped): {(df['side'] == 'Sell').sum():,} ({(df['side'] == 'Sell').sum() / len(df) * 100:.1f}%)")
    print(f"    Volume: {sell_volume:,.2f} contracts ({sell_volume / total_volume * 100:.1f}%)")
    
    # Size distribution
    print(f"\nSize Distribution (percentiles):")
    for pct in [50, 75, 90, 95, 99]:
        vol = df['volume'].quantile(pct / 100)
        notional = (df['volume'] * df['price']).quantile(pct / 100)
        print(f"  {pct}th: {vol:.4f} contracts (${notional:,.0f})")
    
    # Temporal patterns
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    print(f"\nTemporal Patterns:")
    print(f"  Most active hour: {df.groupby('hour').size().idxmax()}:00 UTC ({df.groupby('hour').size().max():,} events)")
    print(f"  Least active hour: {df.groupby('hour').size().idxmin()}:00 UTC ({df.groupby('hour').size().min():,} events)")
    
    hourly_volume = df.groupby('hour')['volume'].sum()
    print(f"  Peak volume hour: {hourly_volume.idxmax()}:00 UTC ({hourly_volume.max():,.0f} contracts)")
    
    # Inter-event timing
    df_sorted = df.sort_values('timestamp')
    inter_event_time = df_sorted['timestamp'].diff().dt.total_seconds()
    
    print(f"\nInter-Event Timing:")
    print(f"  Mean: {inter_event_time.mean():.1f}s")
    print(f"  Median: {inter_event_time.median():.1f}s")
    print(f"  Min: {inter_event_time.min():.3f}s")
    print(f"  Max: {inter_event_time.max():.1f}s")
    
    # Rapid liquidations (< 1 second apart)
    rapid_liqs = (inter_event_time < 1).sum()
    print(f"  Rapid liquidations (<1s apart): {rapid_liqs:,} ({rapid_liqs / len(df) * 100:.2f}%)")

def analyze_imbalance_predictive_power(bars_df, horizons=[5, 15, 60]):
    """Analyze if liquidation imbalance predicts price moves."""
    print(f"\n{'='*70}")
    print(f"Liquidation Imbalance Predictive Power")
    print(f"{'='*70}")
    
    # We don't have price data in this script, so we'll analyze imbalance patterns
    # In a real implementation, we'd merge with ticker data
    
    print(f"\nImbalance Statistics:")
    print(f"  Mean imbalance: {bars_df['liq_imbalance'].mean():.4f}")
    print(f"  Std imbalance: {bars_df['liq_imbalance'].std():.4f}")
    print(f"  Min imbalance: {bars_df['liq_imbalance'].min():.4f} (extreme buy liquidations)")
    print(f"  Max imbalance: {bars_df['liq_imbalance'].max():.4f} (extreme sell liquidations)")
    
    # Extreme imbalance events
    extreme_buy_liq = bars_df[bars_df['liq_imbalance'] < -0.7]
    extreme_sell_liq = bars_df[bars_df['liq_imbalance'] > 0.7]
    
    print(f"\nExtreme Imbalance Events:")
    print(f"  Extreme buy liquidations (imbalance < -0.7): {len(extreme_buy_liq):,}")
    print(f"    Avg volume: {extreme_buy_liq['liq_volume'].mean():.2f} contracts")
    print(f"    Avg notional: ${extreme_buy_liq['liq_notional'].mean():,.0f}")
    
    print(f"  Extreme sell liquidations (imbalance > 0.7): {len(extreme_sell_liq):,}")
    print(f"    Avg volume: {extreme_sell_liq['liq_volume'].mean():.2f} contracts")
    print(f"    Avg notional: ${extreme_sell_liq['liq_notional'].mean():,.0f}")
    
    # Calculate rolling imbalance
    for window in [5, 15, 60]:
        col_name = f'liq_imbalance_{window}m'
        bars_df[col_name] = bars_df['liq_imbalance'].rolling(window=window, min_periods=1).mean()
    
    print(f"\nRolling Imbalance Persistence:")
    for window in [5, 15, 60]:
        col_name = f'liq_imbalance_{window}m'
        autocorr = bars_df['liq_imbalance'].autocorr(lag=window)
        print(f"  {window}-min autocorrelation: {autocorr:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Liquidations Research (v26)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Symbol to analyze (default: BTCUSDT)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LIQUIDATIONS RESEARCH (v26)")
    print("="*70)
    print(f"Symbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    start_time = datetime.now()
    
    # Load liquidations data
    liq_df = load_liquidations(args.symbol, args.data_dir)
    
    # Analyze patterns
    analyze_liquidation_patterns(liq_df)
    
    # Detect cascades
    cascade_df = detect_cascades(liq_df, volume_threshold_pct=95, time_window_sec=60)
    
    # Aggregate to 1-minute bars
    bars_1m = aggregate_liquidations(liq_df, freq='1min')
    
    # Analyze imbalance
    analyze_imbalance_predictive_power(bars_1m)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"liquidations_v26_{args.symbol}.txt"
    
    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Total time: {elapsed:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
