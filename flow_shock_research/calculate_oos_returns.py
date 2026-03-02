#!/usr/bin/env python3
"""
Calculate forward returns for OOS samples to validate reversal hypothesis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
HORIZONS = [5, 15, 30, 60, 120, 300]  # seconds

def load_trades_for_date(date_str):
    """Load all trades for a specific date."""
    print(f"   Loading trades for {date_str}...", flush=True)
    trades = []
    
    for hour in range(24):
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
                                trades.append({
                                    'timestamp': int(trade['T']),
                                    'price': float(trade['p']),
                                    'side': trade['S']
                                })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(trades).sort_values('timestamp') if trades else pd.DataFrame()

def get_price_at_time(trades_df, target_ts, window_ms=1000):
    """Get price closest to target timestamp."""
    mask = (trades_df['timestamp'] >= target_ts - window_ms) & (trades_df['timestamp'] <= target_ts + window_ms)
    nearby = trades_df[mask].copy()
    
    if len(nearby) == 0:
        return None
    
    nearby['diff'] = abs(nearby['timestamp'] - target_ts)
    closest = nearby.loc[nearby['diff'].idxmin()]
    return closest['price']

def calculate_returns_for_sample(sample_file, sample_name):
    """Calculate forward returns for all events in a sample."""
    print(f"\n{'='*80}", flush=True)
    print(f"📈 CALCULATING RETURNS: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Load events
    events_df = pd.read_csv(sample_file)
    events_df['datetime'] = pd.to_datetime(events_df['datetime'])
    events_df['date'] = events_df['datetime'].dt.date
    
    print(f"Total events: {len(events_df)}", flush=True)
    
    # Get unique dates
    unique_dates = sorted(events_df['date'].unique())
    print(f"Unique dates: {len(unique_dates)}", flush=True)
    for date in unique_dates:
        print(f"   {date}", flush=True)
    print(flush=True)
    
    # Load trades for each date
    print(f"Loading trade data...", flush=True)
    trades_by_date = {}
    for date in unique_dates:
        date_str = date.strftime('%Y-%m-%d')
        trades_by_date[date_str] = load_trades_for_date(date_str)
        print(f"   {date_str}: {len(trades_by_date[date_str]):,} trades", flush=True)
    
    # Calculate returns
    print(f"\nCalculating forward returns...", flush=True)
    results = []
    
    for idx, event in events_df.iterrows():
        if (idx + 1) % 50 == 0:
            print(f"   Processed {idx + 1}/{len(events_df)} events...", flush=True)
        
        event_ts = event['timestamp']
        event_price = event['price']
        event_date = event['date']
        event_direction = event['direction']
        
        date_str = event_date.strftime('%Y-%m-%d')
        
        if date_str not in trades_by_date:
            continue
        
        trades_df = trades_by_date[date_str]
        
        # Calculate returns at each horizon
        returns = {
            'timestamp': event_ts,
            'datetime': event['datetime'],
            'event_price': event_price,
            'direction': event_direction,
            'flow_impact': event['flow_impact'],
            'imbalance': event['imbalance'],
            'max_run': event['max_run']
        }
        
        for horizon in HORIZONS:
            target_ts = event_ts + (horizon * 1000)
            future_price = get_price_at_time(trades_df, target_ts)
            
            if future_price:
                # Calculate return in bps
                ret_bps = ((future_price - event_price) / event_price) * 10000
                
                # Directional return (positive if price moves with flow direction)
                if event_direction == 'Sell':
                    ret_bps = -ret_bps  # Invert for sell events
                
                returns[f'ret_{horizon}s'] = ret_bps
                returns[f'price_{horizon}s'] = future_price
            else:
                returns[f'ret_{horizon}s'] = None
                returns[f'price_{horizon}s'] = None
        
        results.append(returns)
    
    print(f"\n✅ Calculated returns for {len(results)} events", flush=True)
    
    return pd.DataFrame(results)

def analyze_returns(df, sample_name):
    """Analyze returns and classify events."""
    print(f"\n{'='*80}", flush=True)
    print(f"📊 ANALYZING RETURNS: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Classification (add before filtering)
    def classify(row):
        if pd.isna(row.get('ret_30s')):
            return None
        ret_30s = row['ret_30s']
        if ret_30s > 10:
            return 'Continuation'
        elif ret_30s < -10:
            return 'Reversal'
        else:
            return 'Exhaustion'
    
    df['classification'] = df.apply(classify, axis=1)
    
    # Remove events without t+30s data
    df = df[df['ret_30s'].notna()].copy()
    print(f"Events with t+30s data: {len(df)}", flush=True)
    
    # Classification breakdown
    print(f"\n📋 Event Classification (t+30s):", flush=True)
    class_counts = df['classification'].value_counts()
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
    
    # Average returns by horizon
    print(f"\n📈 Average Returns by Horizon (all events):", flush=True)
    for horizon in HORIZONS:
        col = f'ret_{horizon}s'
        if col in df.columns:
            mean_ret = df[col].mean()
            median_ret = df[col].median()
            print(f"   t+{horizon:3d}s: {mean_ret:+7.2f} bps (median: {median_ret:+7.2f} bps)", flush=True)
    
    # Win rate
    print(f"\n📊 Win Rate (positive directional return):", flush=True)
    for horizon in HORIZONS:
        col = f'ret_{horizon}s'
        if col in df.columns:
            wins = (df[col] > 0).sum()
            total = df[col].notna().sum()
            wr = wins / total * 100 if total > 0 else 0
            print(f"   t+{horizon:3d}s: {wr:5.1f}% ({wins}/{total})", flush=True)
    
    # Quality filter analysis
    print(f"\n{'='*80}", flush=True)
    print(f"QUALITY FILTER ANALYSIS (impact>100 + imb>0.8)", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    quality = df[(df['flow_impact'] >= 100) & (df['imbalance'] >= 0.8)].copy()
    print(f"Quality events: {len(quality)} ({len(quality)/len(df)*100:.1f}%)", flush=True)
    
    if len(quality) > 0:
        print(f"\n📋 Classification (quality events):", flush=True)
        q_class = quality['classification'].value_counts()
        for cls, count in q_class.items():
            pct = count / len(quality) * 100
            print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
        
        print(f"\n📈 Returns (quality events):", flush=True)
        for horizon in HORIZONS:
            col = f'ret_{horizon}s'
            if col in quality.columns:
                mean_ret = quality[col].mean()
                print(f"   t+{horizon:3d}s: {mean_ret:+7.2f} bps", flush=True)
    
    return df

def main():
    print("="*80, flush=True)
    print("📈 OOS RETURNS CALCULATION", flush=True)
    print("="*80, flush=True)
    print("\nCalculating forward returns for OOS samples...", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Sample 1
    sample1_file = Path("results/sample1_may2025.csv")
    if sample1_file.exists():
        df1 = calculate_returns_for_sample(sample1_file, "Sample 1 (May 18-24)")
        df1_analyzed = analyze_returns(df1, "Sample 1")
        df1.to_csv("results/sample1_with_returns.csv", index=False)
        print(f"\n💾 Saved: results/sample1_with_returns.csv", flush=True)
    else:
        print(f"⚠️  Sample 1 not found", flush=True)
        df1_analyzed = None
    
    # Sample 2
    sample2_file = Path("results/sample2_jul2025.csv")
    if sample2_file.exists():
        df2 = calculate_returns_for_sample(sample2_file, "Sample 2 (Jul 29 - Aug 4)")
        df2_analyzed = analyze_returns(df2, "Sample 2")
        df2.to_csv("results/sample2_with_returns.csv", index=False)
        print(f"\n💾 Saved: results/sample2_with_returns.csv", flush=True)
    else:
        print(f"⚠️  Sample 2 not found", flush=True)
        df2_analyzed = None
    
    # Combined summary
    if df1_analyzed is not None and df2_analyzed is not None:
        print(f"\n{'='*80}", flush=True)
        print(f"📊 COMBINED OOS SUMMARY", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        combined = pd.concat([df1_analyzed, df2_analyzed])
        
        print(f"Total events: {len(combined)}", flush=True)
        print(f"Reversal rate: {(combined['classification'] == 'Reversal').sum() / len(combined) * 100:.1f}%", flush=True)
        print(f"Mean return t+30s: {combined['ret_30s'].mean():.2f} bps", flush=True)
        
        # Quality filter
        quality_combined = combined[(combined['flow_impact'] >= 100) & (combined['imbalance'] >= 0.8)]
        print(f"\nQuality events (impact>100 + imb>0.8): {len(quality_combined)}", flush=True)
        print(f"Reversal rate (quality): {(quality_combined['classification'] == 'Reversal').sum() / len(quality_combined) * 100:.1f}%", flush=True)
        print(f"Mean return t+30s (quality): {quality_combined['ret_30s'].mean():.2f} bps", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
