#!/usr/bin/env python3
"""
Post-Event Study - Analyze price behavior after Flow Pressure events.

For each detected event, calculate forward returns at:
- t+5s, t+15s, t+30s, t+60s, t+2m, t+5m

Classify events:
- Continuation: price continues in flow direction
- Exhaustion: price stops moving
- Reversal: price reverses against flow direction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from datetime import datetime, timedelta

# Load detected events
EVENTS_FILE = Path("flow_shock_research/results/flow_pressure_v3.csv")
DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")

# Forward return horizons (seconds)
HORIZONS = [5, 15, 30, 60, 120, 300]

def load_trade_data_for_date(date_str):
    """Load all trades for a specific date."""
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
                                    'volume': float(trade['v']),
                                    'side': trade['S']
                                })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(trades).sort_values('timestamp')

def get_price_at_time(trades_df, target_ts, window_ms=1000):
    """Get price closest to target timestamp."""
    # Find trades within window
    mask = (trades_df['timestamp'] >= target_ts - window_ms) & (trades_df['timestamp'] <= target_ts + window_ms)
    nearby = trades_df[mask]
    
    if len(nearby) == 0:
        return None
    
    # Return closest trade price
    nearby['diff'] = abs(nearby['timestamp'] - target_ts)
    closest = nearby.loc[nearby['diff'].idxmin()]
    return closest['price']

def calculate_forward_returns(events_df, trades_by_date):
    """Calculate forward returns for all events."""
    results = []
    
    print("📊 Calculating forward returns...")
    
    for idx, event in events_df.iterrows():
        event_ts = event['timestamp']
        event_price = event['price']
        event_date = pd.to_datetime(event['datetime']).date()
        event_direction = event['direction']
        
        # Get trades for this date
        date_str = event_date.strftime('%Y-%m-%d')
        if date_str not in trades_by_date:
            continue
        
        trades_df = trades_by_date[date_str]
        
        # Calculate returns at each horizon
        returns = {
            'event_idx': idx,
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
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1}/{len(events_df)} events...")
    
    return pd.DataFrame(results)

def classify_events(results_df):
    """Classify events based on forward returns."""
    # Use t+30s as primary classification horizon
    df = results_df.copy()
    
    # Remove events without t+30s data
    df = df[df['ret_30s'].notna()].copy()
    
    # Classification logic
    def classify(row):
        ret_30s = row['ret_30s']
        
        # Continuation: price continues in flow direction (positive return)
        if ret_30s > 10:  # >10 bps continuation
            return 'Continuation'
        
        # Reversal: price reverses against flow direction (negative return)
        elif ret_30s < -10:  # <-10 bps reversal
            return 'Reversal'
        
        # Exhaustion: price doesn't move much
        else:
            return 'Exhaustion'
    
    df['classification'] = df.apply(classify, axis=1)
    
    return df

def analyze_post_event():
    """Main analysis."""
    print("="*80)
    print("📊 POST-EVENT STUDY - Price Behavior Analysis")
    print("="*80)
    
    # Load events (filter for impact > 70)
    print("\n📂 Loading events...")
    events_df = pd.read_csv(EVENTS_FILE)
    events_df = events_df[events_df['flow_impact'] >= 70].copy()
    print(f"   Loaded {len(events_df)} events (impact > 70)")
    
    # Get unique dates
    events_df['datetime'] = pd.to_datetime(events_df['datetime'])
    events_df['date'] = events_df['datetime'].dt.date
    unique_dates = sorted(events_df['date'].unique())
    print(f"   Dates: {len(unique_dates)} days")
    
    # Load trade data for each date
    print("\n📂 Loading trade data...")
    trades_by_date = {}
    for date in unique_dates:
        date_str = date.strftime('%Y-%m-%d')
        print(f"   Loading {date_str}...")
        trades_by_date[date_str] = load_trade_data_for_date(date_str)
    
    # Calculate forward returns
    results_df = calculate_forward_returns(events_df, trades_by_date)
    
    # Classify events
    print("\n🔍 Classifying events...")
    classified_df = classify_events(results_df)
    
    # Analysis
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    print(f"\nTotal events analyzed: {len(classified_df)}")
    
    # Classification breakdown
    print("\n📋 Event Classification (t+30s):")
    class_counts = classified_df['classification'].value_counts()
    for cls, count in class_counts.items():
        pct = count / len(classified_df) * 100
        print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Average returns by horizon
    print("\n📈 Average Returns by Horizon (all events):")
    for horizon in HORIZONS:
        col = f'ret_{horizon}s'
        if col in classified_df.columns:
            mean_ret = classified_df[col].mean()
            median_ret = classified_df[col].median()
            print(f"   t+{horizon:3d}s: {mean_ret:+7.2f} bps (median: {median_ret:+7.2f} bps)")
    
    # Returns by classification
    print("\n📊 Average Returns by Classification:")
    for cls in ['Continuation', 'Exhaustion', 'Reversal']:
        subset = classified_df[classified_df['classification'] == cls]
        if len(subset) > 0:
            print(f"\n   {cls}:")
            for horizon in HORIZONS:
                col = f'ret_{horizon}s'
                if col in subset.columns:
                    mean_ret = subset[col].mean()
                    print(f"      t+{horizon:3d}s: {mean_ret:+7.2f} bps")
    
    # Win rate analysis
    print("\n📊 Win Rate Analysis (positive directional return):")
    for horizon in HORIZONS:
        col = f'ret_{horizon}s'
        if col in classified_df.columns:
            wins = (classified_df[col] > 0).sum()
            total = classified_df[col].notna().sum()
            wr = wins / total * 100 if total > 0 else 0
            print(f"   t+{horizon:3d}s: {wr:5.1f}% ({wins}/{total})")
    
    # Save results
    output_file = Path("flow_shock_research/results/post_event_study.csv")
    classified_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file}")
    
    # Generate summary
    generate_summary(classified_df)
    
    return classified_df

def generate_summary(df):
    """Generate summary report."""
    output = Path("flow_shock_research/results/POST_EVENT_FINDINGS.md")
    
    with open(output, 'w') as f:
        f.write("# Post-Event Study - Findings\n\n")
        f.write(f"**Events Analyzed:** {len(df)}\n")
        f.write(f"**Date Range:** {df['datetime'].min()} to {df['datetime'].max()}\n\n")
        
        f.write("## Event Classification (t+30s)\n\n")
        class_counts = df['classification'].value_counts()
        f.write("| Classification | Count | Percentage |\n")
        f.write("|----------------|-------|------------|\n")
        for cls, count in class_counts.items():
            pct = count / len(df) * 100
            f.write(f"| {cls} | {count} | {pct:.1f}% |\n")
        
        f.write("\n## Average Returns by Horizon\n\n")
        f.write("| Horizon | Mean Return | Median Return | Win Rate |\n")
        f.write("|---------|-------------|---------------|----------|\n")
        for horizon in HORIZONS:
            col = f'ret_{horizon}s'
            mean_ret = df[col].mean()
            median_ret = df[col].median()
            wins = (df[col] > 0).sum()
            total = df[col].notna().sum()
            wr = wins / total * 100 if total > 0 else 0
            f.write(f"| t+{horizon}s | {mean_ret:+.2f} bps | {median_ret:+.2f} bps | {wr:.1f}% |\n")
        
        f.write("\n## Returns by Classification\n\n")
        for cls in ['Continuation', 'Exhaustion', 'Reversal']:
            subset = df[df['classification'] == cls]
            if len(subset) > 0:
                f.write(f"\n### {cls} ({len(subset)} events)\n\n")
                f.write("| Horizon | Mean Return |\n")
                f.write("|---------|-------------|\n")
                for horizon in HORIZONS:
                    col = f'ret_{horizon}s'
                    mean_ret = subset[col].mean()
                    f.write(f"| t+{horizon}s | {mean_ret:+.2f} bps |\n")
    
    print(f"📄 Summary saved: {output}")

if __name__ == "__main__":
    analyze_post_event()
