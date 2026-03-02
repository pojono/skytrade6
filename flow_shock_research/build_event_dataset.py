#!/usr/bin/env python3
"""
Step 1: Build baseline event dataset.

Freeze Stage-1 and classifier parameters, extract all events with:
- Flow metrics (FlowImpact, imbalance, run, agg_trades)
- Orderbook state (spread, depth L1-L5, depth_drop, spread_ratio)
- Regime features (pre_vol_15m, pre_range_10s, pre_drift_2m, etc.)
- Classification label (FOLLOW/NO_TRADE)
- Forward returns

Save to parquet for fast iteration.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys
from datetime import datetime

# FROZEN PARAMETERS
STAGE1_PARAMS = {
    'flow_impact_threshold': 70,
    'imbalance_threshold': 0.6,
    'min_agg_trades': 20,
    'same_side_share': 0.75
}

CLASSIFIER_PARAMS = {
    'vol_15m_q30': 0.51,
    'range_10s_q80': 61.34,
    'drift_2m_q70': -41.87
}

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")

def load_orderbook_snapshot(date_str, target_ts):
    """Load orderbook snapshot closest to target timestamp."""
    target_dt = pd.to_datetime(target_ts, unit='ms')
    hour = target_dt.hour
    
    ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook.500" / "symbol=SOLUSDT" / "data.jsonl.gz"
    
    if not ob_file.exists():
        return None
    
    closest_snapshot = None
    min_diff = float('inf')
    
    try:
        with gzip.open(ob_file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        ob_data = data['result']['data']
                        ts = int(ob_data['T'])
                        
                        diff = abs(ts - target_ts)
                        if diff < min_diff:
                            min_diff = diff
                            closest_snapshot = ob_data
                            
                            if diff < 100:  # Within 100ms is good enough
                                break
                except:
                    continue
    except:
        return None
    
    return closest_snapshot

def extract_orderbook_features(ob_snapshot):
    """Extract orderbook features from snapshot."""
    if not ob_snapshot:
        return {}
    
    try:
        bids = ob_snapshot.get('b', [])
        asks = ob_snapshot.get('a', [])
        
        if not bids or not asks:
            return {}
        
        # Best bid/ask
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        # Spread
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000
        
        # Depth at each level (L1-L5)
        depth_levels = []
        for i in range(min(5, len(bids))):
            bid_size = float(bids[i][1])
            ask_size = float(asks[i][1]) if i < len(asks) else 0
            depth_levels.append(bid_size + ask_size)
        
        # Pad to 5 levels
        while len(depth_levels) < 5:
            depth_levels.append(0)
        
        # Top depth (L1)
        top_depth = depth_levels[0]
        
        return {
            'spread': spread,
            'spread_bps': spread_bps,
            'depth_l1': depth_levels[0],
            'depth_l2': depth_levels[1],
            'depth_l3': depth_levels[2],
            'depth_l4': depth_levels[3],
            'depth_l5': depth_levels[4],
            'top_depth': top_depth
        }
    except:
        return {}

def build_event_dataset(sample_file, sample_name):
    """Build comprehensive event dataset for a sample."""
    print(f"\n{'='*80}", flush=True)
    print(f"📦 BUILDING EVENT DATASET: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Load events with regime features and classification
    df = pd.read_csv(sample_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    
    print(f"Total events: {len(df)}", flush=True)
    
    # Build comprehensive dataset
    events = []
    
    for idx, row in df.iterrows():
        if (idx + 1) % 20 == 0:
            print(f"   Processed {idx + 1}/{len(df)} events...", flush=True)
        
        event_ts = row['timestamp']
        event_date = row['date']
        date_str = event_date.strftime('%Y-%m-%d')
        
        # Base event data
        event = {
            # Identifiers
            'timestamp': event_ts,
            'datetime': row['datetime'],
            'date': date_str,
            'hour': pd.to_datetime(row['datetime']).hour,
            
            # Flow metrics (Stage-1)
            'flow_impact': row.get('flow_impact', np.nan),
            'imbalance': row.get('imbalance', np.nan),
            'max_run': row.get('max_run', np.nan),
            'direction': row.get('direction', ''),
            
            # Regime features (multi-scale)
            'pre_vol_15m': row.get('15m_vol', np.nan),
            'pre_range_15m': row.get('15m_range', np.nan),
            'pre_vol_5m': row.get('5m_vol', np.nan),
            'pre_range_5m': row.get('5m_range', np.nan),
            'pre_vol_2m': row.get('2m_vol', np.nan),
            'pre_drift_2m': row.get('2m_drift', np.nan),
            'pre_vol_30s': row.get('30s_vol', np.nan),
            'pre_range_30s': row.get('30s_range', np.nan),
            'pre_imbalance_30s': row.get('30s_imbalance', np.nan),
            'pre_vol_10s': row.get('10s_vol', np.nan),
            'pre_range_10s': row.get('10s_range', np.nan),
            
            # Classification
            'regime_pred': row.get('regime_pred', 'NO_TRADE'),
            'classification': row.get('classification', ''),
            
            # Returns
            'ret_5s': row.get('ret_5s', np.nan),
            'ret_15s': row.get('ret_15s', np.nan),
            'ret_30s': row.get('ret_30s', np.nan),
            'ret_60s': row.get('ret_60s', np.nan),
            'ret_120s': row.get('ret_120s', np.nan),
            'ret_300s': row.get('ret_300s', np.nan),
        }
        
        # Get orderbook snapshot at event time
        ob_snapshot = load_orderbook_snapshot(date_str, event_ts)
        ob_features = extract_orderbook_features(ob_snapshot)
        event.update(ob_features)
        
        events.append(event)
    
    result_df = pd.DataFrame(events)
    
    print(f"\n✅ Built dataset: {len(result_df)} events", flush=True)
    print(f"   Columns: {len(result_df.columns)}", flush=True)
    print(f"   FOLLOW events: {(result_df['regime_pred'] == 'FOLLOW').sum()}", flush=True)
    
    return result_df

def main():
    print("="*80, flush=True)
    print("📦 STEP 1: BUILD BASELINE EVENT DATASET", flush=True)
    print("="*80, flush=True)
    print("\nFrozen parameters:", flush=True)
    print(f"   Stage-1: {STAGE1_PARAMS}", flush=True)
    print(f"   Classifier: {CLASSIFIER_PARAMS}", flush=True)
    print("="*80 + "\n", flush=True)
    
    all_events = []
    
    # Process all samples
    for name, file in [
        ('Sample 1 (May)', "results/sample1_global.csv"),
        ('Sample 2 (Jul-Aug)', "results/sample2_global.csv"),
        ('Sample 3 (Jun)', "results/sample3_global.csv")
    ]:
        if Path(file).exists():
            df = build_event_dataset(file, name)
            all_events.append(df)
    
    # Combine all samples
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        
        print(f"\n{'='*80}", flush=True)
        print(f"COMBINED DATASET", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        print(f"Total events: {len(combined)}", flush=True)
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}", flush=True)
        print(f"Total days: {combined['date'].nunique()}", flush=True)
        
        print(f"\nRegime distribution:", flush=True)
        for regime, count in combined['regime_pred'].value_counts().items():
            pct = count / len(combined) * 100
            print(f"   {regime:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
        
        print(f"\nClassification distribution:", flush=True)
        for cls, count in combined['classification'].value_counts().items():
            pct = count / len(combined) * 100
            print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
        
        # Save to parquet
        output_file = "results/events_baseline.parquet"
        combined.to_parquet(output_file, index=False)
        print(f"\n💾 Saved: {output_file}", flush=True)
        
        # Also save CSV for inspection
        combined.to_csv("results/events_baseline.csv", index=False)
        print(f"💾 Saved: results/events_baseline.csv", flush=True)
        
        # Summary stats
        print(f"\n{'='*80}", flush=True)
        print(f"DATASET SUMMARY", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        print(f"Features available:", flush=True)
        print(f"   Flow metrics: flow_impact, imbalance, max_run, direction", flush=True)
        print(f"   Orderbook: spread, depth_l1-l5, top_depth", flush=True)
        print(f"   Regime: pre_vol_15m, pre_range_10s, pre_drift_2m, etc.", flush=True)
        print(f"   Labels: regime_pred, classification", flush=True)
        print(f"   Returns: ret_5s through ret_300s", flush=True)
        
        # Check data quality
        print(f"\nData quality:", flush=True)
        has_ob = 'spread' in combined.columns and combined['spread'].notna().sum() > 0
        if has_ob:
            print(f"   Events with orderbook: {combined['spread'].notna().sum()} ({combined['spread'].notna().mean():.1%})", flush=True)
        else:
            print(f"   Events with orderbook: 0 (orderbook data not available for these hours)", flush=True)
        print(f"   Events with returns: {combined['ret_30s'].notna().sum()} ({combined['ret_30s'].notna().mean():.1%})", flush=True)
        print(f"   FOLLOW events: {(combined['regime_pred'] == 'FOLLOW').sum()}", flush=True)
        
        # Show hour distribution
        print(f"\nHour distribution:", flush=True)
        for hour, count in combined['hour'].value_counts().sort_index().items():
            pct = count / len(combined) * 100
            print(f"   Hour {hour:02d}: {count:3d} ({pct:5.1f}%)", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"✅ STEP 1 COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
