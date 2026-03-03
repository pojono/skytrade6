#!/usr/bin/env python3
"""
Enrich v2 events: add regime features + forward returns.
Only processes filtered events (not all 14K).
Parallel by event.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys
import time
from bisect import bisect_right
from multiprocessing import Pool

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")

FLOW_IMPACT_MIN = 200
BURST_Z_MIN = 5.0
WINDOW_MS = 15_000

# Regime classifier thresholds
FOLLOW_THRESHOLDS = {
    'vol_15m_q30': 0.51,
    'range_10s_q80': 61.34,
    'drift_2m_q70': 41.87,
}

# Cache trades by date to avoid re-loading
_trade_cache = {}

def get_trades(date_str):
    """Load and cache trades for a date."""
    if date_str in _trade_cache:
        return _trade_cache[date_str]
    
    trades = []
    for hour in range(24):
        tf = (DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}"
              / "exchange=bybit" / "source=ws" / "market=linear"
              / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz")
        if not tf.exists():
            continue
        try:
            with gzip.open(tf, 'rt') as f:
                for line in f:
                    try:
                        msg = json.loads(line)
                        r = msg.get('result')
                        if not r or 'data' not in r:
                            continue
                        for t in r['data']:
                            trades.append((int(t['T']), float(t['p']),
                                          float(t['v']), t['S']))
                    except:
                        continue
        except:
            continue
    
    trades.sort(key=lambda x: x[0])
    _trade_cache[date_str] = trades
    return trades


def process_event(args):
    """Add regime features + returns for one event."""
    idx, row = args
    date_str = row['date']
    event_ts = row['timestamp']
    event_price = row['event_price']
    
    trades = get_trades(date_str)
    if not trades:
        return {}
    
    trade_ts = [t[0] for t in trades]
    
    # ── Regime features (Fix 4: end at t0 - WINDOW_MS, no leakage) ──
    t0_features = event_ts - WINDOW_MS
    
    windows = {'10s': 10_000, '30s': 30_000, '2m': 120_000,
               '5m': 300_000, '15m': 900_000}
    features = {}
    
    for name, ms in windows.items():
        start = t0_features - ms
        end = t0_features
        
        lo = bisect_right(trade_ts, start - 1)
        hi = bisect_right(trade_ts, end)
        window = trades[lo:hi]
        
        if len(window) < 5:
            continue
        
        prices = [t[1] for t in window]
        volumes = [t[2] for t in window]
        sides = [t[3] for t in window]
        
        prefix = f'pre_{name}_'
        
        if len(prices) > 1:
            rets = [(prices[i] - prices[i-1]) / prices[i-1] * 10_000
                    for i in range(1, len(prices))]
            features[prefix + 'vol'] = float(np.std(rets))
        
        hi_p, lo_p = max(prices), min(prices)
        features[prefix + 'range'] = (hi_p - lo_p) / np.mean(prices) * 10_000
        features[prefix + 'drift'] = (prices[-1] - prices[0]) / prices[0] * 10_000
        
        buy_v = sum(v for v, s in zip(volumes, sides) if s == 'Buy')
        sell_v = sum(v for v, s in zip(volumes, sides) if s == 'Sell')
        total_v = buy_v + sell_v
        if total_v > 0:
            features[prefix + 'imbalance'] = (buy_v - sell_v) / total_v
    
    # ── Forward returns ──
    for ret_name, ret_ms in [('5s', 5000), ('15s', 15000), ('30s', 30000),
                              ('60s', 60000), ('120s', 120000), ('300s', 300000)]:
        target_ts = event_ts + ret_ms
        hi = bisect_right(trade_ts, target_ts)
        if hi > 0 and hi <= len(trades):
            future_price = trades[hi - 1][1]
            features[f'ret_{ret_name}'] = (future_price - event_price) / event_price * 10_000
    
    # ── Classification ──
    vol_15m = features.get('pre_15m_vol', np.nan)
    range_10s = features.get('pre_10s_range', np.nan)
    drift_2m = features.get('pre_2m_drift', np.nan)
    
    if (not pd.isna(vol_15m) and not pd.isna(range_10s) and not pd.isna(drift_2m)
            and vol_15m < FOLLOW_THRESHOLDS['vol_15m_q30']
            and range_10s > FOLLOW_THRESHOLDS['range_10s_q80']
            and abs(drift_2m) > FOLLOW_THRESHOLDS['drift_2m_q70']):
        features['label'] = 'FOLLOW'
    else:
        features['label'] = 'NO_TRADE'
    
    return features


def process_date_batch(date_events):
    """Process all events for one date (loads trades once)."""
    date_str, events_df = date_events
    
    trades = get_trades(date_str)
    if not trades:
        return []
    
    results = []
    for _, row in events_df.iterrows():
        feat = process_event((0, row))
        results.append(feat)
    
    return results


def main():
    print("="*80, flush=True)
    print("📊 ENRICH V2 EVENTS — Regime + Returns", flush=True)
    print("="*80, flush=True)
    
    # Load and filter
    df = pd.read_parquet("results/v2_events_raw.parquet")
    mask = (df['flow_impact'] >= FLOW_IMPACT_MIN) & (df['burst_zscore'] >= BURST_Z_MIN)
    df = df[mask].reset_index(drop=True)
    
    print(f"Filtered events: {len(df)} (FI>={FLOW_IMPACT_MIN}, BZ>={BURST_Z_MIN})", flush=True)
    print(f"Events/day: {len(df)/df['date'].nunique():.1f}", flush=True)
    print(f"\nProcessing by date (sequential, trades cached)...\n", flush=True)
    
    t0 = time.time()
    
    all_features = []
    dates = df['date'].unique()
    
    for i, date_str in enumerate(sorted(dates)):
        date_df = df[df['date'] == date_str]
        
        # Load trades once per date
        trades = get_trades(date_str)
        
        for _, row in date_df.iterrows():
            feat = process_event((0, row))
            all_features.append(feat)
        
        # Free cache to save memory
        if date_str in _trade_cache:
            del _trade_cache[date_str]
        
        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(dates)}] {date_str} — {elapsed:.0f}s", flush=True)
    
    # Merge features into df
    feat_df = pd.DataFrame(all_features)
    for col in feat_df.columns:
        df[col] = feat_df[col].values
    
    elapsed = time.time() - t0
    
    # Save
    df.to_parquet("results/v2_events_enriched.parquet", index=False)
    df.to_csv("results/v2_events_enriched.csv", index=False)
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"✅ DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"   Events: {len(df)}", flush=True)
    print(f"   FOLLOW: {(df['label']=='FOLLOW').sum()}", flush=True)
    print(f"   NO_TRADE: {(df['label']=='NO_TRADE').sum()}", flush=True)
    
    # FOLLOW performance
    follow = df[df['label'] == 'FOLLOW'].copy()
    if len(follow) > 0 and 'ret_30s' in follow.columns:
        follow['ret_dir'] = follow.apply(
            lambda r: r['ret_30s'] if r['direction'] == 'Buy' else -r['ret_30s'], axis=1)
        
        print(f"\n   FOLLOW performance (n={len(follow)}):", flush=True)
        print(f"      Gross 30s: {follow['ret_dir'].mean():+.2f} bps", flush=True)
        print(f"      Win rate:  {(follow['ret_dir'] > 0).mean():.1%}", flush=True)
        print(f"      Net maker: {follow['ret_dir'].mean() - 8:+.2f} bps", flush=True)
        
        # By month
        follow['month'] = pd.to_datetime(follow['date']).dt.to_period('M')
        print(f"\n   FOLLOW by month:", flush=True)
        for m, sub in follow.groupby('month'):
            sub_dir = sub.apply(lambda r: r['ret_30s'] if r['direction'] == 'Buy' else -r['ret_30s'], axis=1)
            print(f"      {m}: n={len(sub):3d}, gross={sub_dir.mean():+.2f} bps, "
                  f"wr={( sub_dir > 0).mean():.0%}", flush=True)
    
    print(f"\n💾 results/v2_events_enriched.parquet", flush=True)
    print(f"{'='*80}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
