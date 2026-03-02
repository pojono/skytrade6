#!/usr/bin/env python3
"""
Multi-scale pre-event feature extraction.

Instead of fixed 5m window, extract features at multiple timescales:
- t-10s: micro exhaustion
- t-30s: flow buildup  
- t-2m: liquidity regime
- t-5m: positioning
- t-15m: trend context

Goal: Find which timescale separates continuation vs reversal regimes.

Hypothesis:
- Continuation determined by LONG horizon (trend, positioning)
- Exhaustion determined by SHORT horizon (micro buildup, liquidity stress)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")

# Multi-scale windows (in seconds)
WINDOWS = {
    '10s': 10,
    '30s': 30,
    '2m': 120,
    '5m': 300,
    '15m': 900
}

def load_trades_window(date_str, start_ts, end_ts):
    """Load trades in a time window."""
    trades = []
    
    start_dt = pd.to_datetime(start_ts, unit='ms')
    end_dt = pd.to_datetime(end_ts, unit='ms')
    
    # Determine which hours to load
    hours = set()
    current = start_dt
    while current <= end_dt:
        hours.add(current.hour)
        current += pd.Timedelta(hours=1)
    
    for hour in sorted(hours):
        trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
        
        if not trade_file.exists():
            continue
        
        try:
            with gzip.open(trade_file, 'rt') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for trade in data['result']['data']:
                                ts = int(trade['T'])
                                if start_ts <= ts <= end_ts:
                                    trades.append({
                                        'timestamp': ts,
                                        'price': float(trade['p']),
                                        'volume': float(trade['v']),
                                        'side': trade['S']
                                    })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(trades).sort_values('timestamp') if trades else pd.DataFrame()

def calculate_window_features(trades_df, event_price, window_name):
    """Calculate features for a specific time window."""
    if len(trades_df) == 0:
        return {}
    
    prefix = f"{window_name}_"
    features = {}
    
    # 1. Volatility
    if len(trades_df) > 1:
        returns = trades_df['price'].pct_change().dropna() * 10000  # bps
        features[f'{prefix}vol'] = returns.std()
        features[f'{prefix}vol_mean'] = returns.mean()
    else:
        features[f'{prefix}vol'] = 0
        features[f'{prefix}vol_mean'] = 0
    
    # 2. Price drift
    if len(trades_df) > 0:
        first_price = trades_df['price'].iloc[0]
        last_price = trades_df['price'].iloc[-1]
        features[f'{prefix}drift'] = (last_price - first_price) / first_price * 10000  # bps
        
        # Distance from event price
        features[f'{prefix}distance'] = (event_price - last_price) / last_price * 10000
    else:
        features[f'{prefix}drift'] = 0
        features[f'{prefix}distance'] = 0
    
    # 3. Range
    if len(trades_df) > 0:
        high = trades_df['price'].max()
        low = trades_df['price'].min()
        range_bps = (high - low) / trades_df['price'].mean() * 10000
        features[f'{prefix}range'] = range_bps
        
        # Price position in range
        if high > low:
            features[f'{prefix}position'] = (event_price - low) / (high - low)
        else:
            features[f'{prefix}position'] = 0.5
    else:
        features[f'{prefix}range'] = 0
        features[f'{prefix}position'] = 0.5
    
    # 4. Activity
    duration_s = (trades_df['timestamp'].max() - trades_df['timestamp'].min()) / 1000
    if duration_s > 0:
        features[f'{prefix}trade_rate'] = len(trades_df) / duration_s
    else:
        features[f'{prefix}trade_rate'] = 0
    
    features[f'{prefix}volume'] = trades_df['volume'].sum()
    
    # 5. Buy/Sell imbalance
    buy_vol = trades_df[trades_df['side'] == 'Buy']['volume'].sum()
    sell_vol = trades_df[trades_df['side'] == 'Sell']['volume'].sum()
    total_vol = buy_vol + sell_vol
    if total_vol > 0:
        features[f'{prefix}imbalance'] = (buy_vol - sell_vol) / total_vol
    else:
        features[f'{prefix}imbalance'] = 0
    
    # 6. Trend strength (for longer windows)
    if len(trades_df) > 10:
        trades_df_copy = trades_df.copy()
        trades_df_copy['time_norm'] = (trades_df_copy['timestamp'] - trades_df_copy['timestamp'].min()) / 1000
        slope = np.polyfit(trades_df_copy['time_norm'], trades_df_copy['price'], 1)[0]
        features[f'{prefix}slope'] = slope / event_price * 10000  # bps per second
    else:
        features[f'{prefix}slope'] = 0
    
    return features

def build_multiscale_features(events_file, sample_name):
    """Build multi-scale features for all events."""
    print(f"\n{'='*80}", flush=True)
    print(f"🔬 MULTI-SCALE FEATURE EXTRACTION: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Load events
    df = pd.read_csv(events_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    
    print(f"Total events: {len(df)}", flush=True)
    
    # Filter events with classification
    df = df[df['classification'].notna()].copy()
    print(f"Events with classification: {len(df)}", flush=True)
    
    print(f"\n📋 Classification:", flush=True)
    for cls, count in df['classification'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
    
    # Extract multi-scale features
    print(f"\n🔧 Extracting multi-scale features...", flush=True)
    print(f"   Windows: {', '.join(WINDOWS.keys())}", flush=True)
    
    all_features = []
    
    for idx, event in df.iterrows():
        if (idx + 1) % 25 == 0:
            print(f"   Processed {idx + 1}/{len(df)} events...", flush=True)
        
        event_ts = event['timestamp']
        event_price = event['event_price']
        event_date = event['date']
        date_str = event_date.strftime('%Y-%m-%d')
        
        # Base features
        event_features = {
            'timestamp': event_ts,
            'datetime': event['datetime'],
            'classification': event['classification'],
            'ret_30s': event['ret_30s'],
            'flow_impact': event['flow_impact'],
            'imbalance': event['imbalance'],
            'direction': event['direction'],
            'hour': pd.to_datetime(event['datetime']).hour
        }
        
        # Extract features for each window
        for window_name, window_seconds in WINDOWS.items():
            start_ts = event_ts - (window_seconds * 1000)
            end_ts = event_ts
            
            trades_df = load_trades_window(date_str, start_ts, end_ts)
            
            if len(trades_df) >= 5:  # Minimum trades required
                window_features = calculate_window_features(trades_df, event_price, window_name)
                event_features.update(window_features)
            else:
                # Fill with NaN if not enough data
                for key in ['vol', 'vol_mean', 'drift', 'distance', 'range', 'position', 
                           'trade_rate', 'volume', 'imbalance', 'slope']:
                    event_features[f'{window_name}_{key}'] = np.nan
        
        all_features.append(event_features)
    
    print(f"\n✅ Extracted multi-scale features for {len(all_features)} events", flush=True)
    
    return pd.DataFrame(all_features)

def analyze_timescale_separation(df, sample_name):
    """Analyze which timescale best separates continuation vs reversal."""
    print(f"\n{'='*80}", flush=True)
    print(f"📊 TIMESCALE SEPARATION ANALYSIS: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Split by outcome
    reversal = df[df['classification'] == 'Reversal']
    continuation = df[df['classification'] == 'Continuation']
    
    print(f"Reversal events: {len(reversal)}", flush=True)
    print(f"Continuation events: {len(continuation)}", flush=True)
    
    if len(reversal) == 0 or len(continuation) == 0:
        print(f"\n⚠️  Not enough events for comparison", flush=True)
        return
    
    # Analyze each feature across timescales
    print(f"\n{'='*80}", flush=True)
    print(f"VOLATILITY ACROSS TIMESCALES", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    print(f"{'Window':>10} | {'Rev Vol':>10} | {'Cont Vol':>10} | {'Diff':>10} | {'Separation':>12}", flush=True)
    print(f"{'-'*70}", flush=True)
    
    for window in WINDOWS.keys():
        col = f'{window}_vol'
        if col in df.columns:
            rev_val = reversal[col].mean()
            cont_val = continuation[col].mean()
            diff = cont_val - rev_val
            
            # Separation score (normalized difference)
            pooled_std = np.sqrt((reversal[col].std()**2 + continuation[col].std()**2) / 2)
            separation = abs(diff) / pooled_std if pooled_std > 0 else 0
            
            print(f"{window:>10} | {rev_val:>10.2f} | {cont_val:>10.2f} | {diff:>+10.2f} | {separation:>12.2f}", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"DRIFT ACROSS TIMESCALES", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    print(f"{'Window':>10} | {'Rev Drift':>10} | {'Cont Drift':>10} | {'Diff':>10} | {'Separation':>12}", flush=True)
    print(f"{'-'*70}", flush=True)
    
    for window in WINDOWS.keys():
        col = f'{window}_drift'
        if col in df.columns:
            rev_val = reversal[col].mean()
            cont_val = continuation[col].mean()
            diff = cont_val - rev_val
            
            pooled_std = np.sqrt((reversal[col].std()**2 + continuation[col].std()**2) / 2)
            separation = abs(diff) / pooled_std if pooled_std > 0 else 0
            
            print(f"{window:>10} | {rev_val:>10.2f} | {cont_val:>10.2f} | {diff:>+10.2f} | {separation:>12.2f}", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"RANGE ACROSS TIMESCALES", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    print(f"{'Window':>10} | {'Rev Range':>10} | {'Cont Range':>10} | {'Diff':>10} | {'Separation':>12}", flush=True)
    print(f"{'-'*70}", flush=True)
    
    for window in WINDOWS.keys():
        col = f'{window}_range'
        if col in df.columns:
            rev_val = reversal[col].mean()
            cont_val = continuation[col].mean()
            diff = cont_val - rev_val
            
            pooled_std = np.sqrt((reversal[col].std()**2 + continuation[col].std()**2) / 2)
            separation = abs(diff) / pooled_std if pooled_std > 0 else 0
            
            print(f"{window:>10} | {rev_val:>10.2f} | {cont_val:>10.2f} | {diff:>+10.2f} | {separation:>12.2f}", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"KEY INSIGHT: Which timescale shows strongest separation?", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Find best separating features
    separations = []
    for window in WINDOWS.keys():
        for feature in ['vol', 'drift', 'range', 'slope', 'imbalance']:
            col = f'{window}_{feature}'
            if col in df.columns and df[col].notna().sum() > 10:
                rev_val = reversal[col].mean()
                cont_val = continuation[col].mean()
                diff = cont_val - rev_val
                
                pooled_std = np.sqrt((reversal[col].std()**2 + continuation[col].std()**2) / 2)
                separation = abs(diff) / pooled_std if pooled_std > 0 else 0
                
                separations.append({
                    'feature': col,
                    'window': window,
                    'metric': feature,
                    'separation': separation,
                    'diff': diff
                })
    
    sep_df = pd.DataFrame(separations).sort_values('separation', ascending=False)
    
    print(f"Top 10 separating features:", flush=True)
    print(f"\n{'Feature':>20} | {'Window':>10} | {'Separation':>12} | {'Diff':>10}", flush=True)
    print(f"{'-'*70}", flush=True)
    
    for _, row in sep_df.head(10).iterrows():
        print(f"{row['metric']:>20} | {row['window']:>10} | {row['separation']:>12.2f} | {row['diff']:>+10.2f}", flush=True)

def main():
    print("="*80, flush=True)
    print("🔬 MULTI-SCALE REGIME FEATURE EXTRACTION", flush=True)
    print("="*80, flush=True)
    print("\nGoal: Find which timescale separates continuation vs reversal", flush=True)
    print("Hypothesis: Continuation = long horizon, Exhaustion = short horizon", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Sample 1
    sample1_file = Path("results/sample1_with_returns.csv")
    if sample1_file.exists():
        df1 = build_multiscale_features(sample1_file, "Sample 1 (May 18-24)")
        analyze_timescale_separation(df1, "Sample 1")
        df1.to_csv("results/sample1_multiscale.csv", index=False)
        print(f"\n💾 Saved: results/sample1_multiscale.csv", flush=True)
    else:
        print(f"⚠️  Sample 1 not found", flush=True)
        df1 = None
    
    # Sample 2
    sample2_file = Path("results/sample2_with_returns.csv")
    if sample2_file.exists():
        df2 = build_multiscale_features(sample2_file, "Sample 2 (Jul 29 - Aug 4)")
        analyze_timescale_separation(df2, "Sample 2")
        df2.to_csv("results/sample2_multiscale.csv", index=False)
        print(f"\n💾 Saved: results/sample2_multiscale.csv", flush=True)
    else:
        print(f"⚠️  Sample 2 not found", flush=True)
        df2 = None
    
    # Combined analysis
    if df1 is not None and df2 is not None:
        print(f"\n{'='*80}", flush=True)
        print(f"📊 COMBINED TIMESCALE ANALYSIS", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        combined = pd.concat([df1, df2])
        analyze_timescale_separation(combined, "Combined OOS")
        combined.to_csv("results/combined_multiscale.csv", index=False)
        print(f"\n💾 Saved: results/combined_multiscale.csv", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
