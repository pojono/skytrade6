#!/usr/bin/env python3
"""
ORDERBOOK IMBALANCE SIGNAL
============================
Tests if bid/ask imbalance in L2 orderbook predicts direction.

Data: Bybit orderbook snapshots (200 levels), Aug 2025 → Mar 2026.
Signal: Bid depth >> Ask depth → long (buying pressure), and vice versa.
Measured at top-5, top-20, top-50 levels.

Approach: Sample snapshots every 5 min, compute imbalance, measure forward returns.
"""
import sys, os, gzip, json
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from pathlib import Path
from data_loader import load_kline, progress_bar, RT_TAKER_BPS

BYBIT = Path("/home/ubuntu/Projects/skytrade6/datalake/bybit")
OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'

# Orderbook data available from ~Aug 2025
START = '2025-09-01'
END = '2026-03-04'

# Test on liquid alts with orderbook data
OB_ALTS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
           'AVAXUSDT', 'LINKUSDT', 'SUIUSDT', 'APTUSDT', 'ARBUSDT']

# Sample days to keep processing manageable
SAMPLE_DAYS = 30  # random sample of days
np.random.seed(42)


def parse_orderbook_snapshots(fpath, sample_interval_sec=300):
    """Parse JSONL.gz orderbook file, keeping only snapshots at ~5min intervals.
    Returns list of dicts with timestamp, bid_depth, ask_depth at various levels."""
    results = []
    last_ts = 0
    
    # We need to reconstruct from snapshots. Just use snapshots, skip deltas.
    try:
        with gzip.open(fpath, 'rt') as f:
            for line in f:
                try:
                    msg = json.loads(line)
                except:
                    continue
                
                if msg.get('type') != 'snapshot':
                    continue
                
                ts_ms = msg.get('ts', 0)
                ts_sec = ts_ms / 1000
                
                # Sample at interval
                if ts_sec - last_ts < sample_interval_sec:
                    continue
                last_ts = ts_sec
                
                data = msg.get('data', {})
                bids = data.get('b', [])
                asks = data.get('a', [])
                
                if not bids or not asks:
                    continue
                
                # Compute depth at various levels
                row = {'ts': pd.Timestamp(ts_ms, unit='ms')}
                
                for n_levels in [5, 20, 50]:
                    bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:n_levels])
                    ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:n_levels])
                    total = bid_depth + ask_depth
                    
                    if total > 0:
                        row[f'imbalance_{n_levels}'] = (bid_depth - ask_depth) / total
                    else:
                        row[f'imbalance_{n_levels}'] = 0
                
                # Best bid/ask spread
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid = (best_bid + best_ask) / 2
                row['spread_bps'] = (best_ask - best_bid) / mid * 10000
                row['mid_price'] = mid
                
                results.append(row)
    except Exception as e:
        pass
    
    return results


def analyze_symbol(sym, start, end, sample_days):
    """Analyze orderbook imbalance signal for one symbol."""
    sym_dir = BYBIT / sym
    
    # Find available orderbook files (futures only, exclude spot)
    all_files = os.listdir(sym_dir)
    ob_files = sorted(f for f in all_files
                     if f.endswith('_orderbook.jsonl.gz') and '_orderbook_spot' not in f
                     and f[:10] >= start and f[:10] <= end)
    
    if not ob_files:
        return pd.DataFrame()
    
    # Sample days
    if len(ob_files) > sample_days:
        indices = np.random.choice(len(ob_files), sample_days, replace=False)
        ob_files = [ob_files[i] for i in sorted(indices)]
    
    # Load klines for forward returns
    kline = load_kline(sym, start, end)
    if kline.empty or len(kline) < 5000:
        return pd.DataFrame()
    k1m = kline[['ts', 'close']].set_index('ts').sort_index()
    k1m = k1m[~k1m.index.duplicated(keep='first')]
    
    all_obs = []
    for fname in ob_files:
        fpath = sym_dir / fname
        snapshots = parse_orderbook_snapshots(str(fpath))
        all_obs.extend(snapshots)
    
    if not all_obs:
        return pd.DataFrame()
    
    ob_df = pd.DataFrame(all_obs).set_index('ts').sort_index()
    
    # Merge with kline forward returns
    merged = ob_df.join(k1m, how='inner')
    if len(merged) < 50:
        return pd.DataFrame()
    
    # Forward returns
    for h in [30, 60, 240]:
        merged[f'fwd_{h}m'] = (merged['close'].shift(-h) / merged['close'] - 1) * 10000
    
    # Z-score the imbalance (rolling)
    for n in [5, 20, 50]:
        col = f'imbalance_{n}'
        merged[f'{col}_z'] = (merged[col] - merged[col].rolling(48).mean()) / merged[col].rolling(48).std().clip(lower=1e-6)
    
    merged['symbol'] = sym
    return merged


def main():
    print("=" * 75)
    print("  ORDERBOOK IMBALANCE SIGNAL")
    print(f"  Period: {START} → {END} | Symbols: {len(OB_ALTS)}")
    print(f"  Sampling {SAMPLE_DAYS} days per symbol")
    print("=" * 75)
    
    all_data = []
    t0 = time.time()
    
    for si, sym in enumerate(OB_ALTS):
        progress_bar(si, len(OB_ALTS), prefix='  Processing', start_time=t0)
        df = analyze_symbol(sym, START, END, SAMPLE_DAYS)
        if not df.empty:
            all_data.append(df)
            print(f"    {sym}: {len(df)} snapshots")
    
    progress_bar(len(OB_ALTS), len(OB_ALTS), prefix='  Processing', start_time=t0)
    
    if not all_data:
        print("  ❌ No orderbook data processed")
        return
    
    merged = pd.concat(all_data)
    print(f"\n  Total: {len(merged)} observations across {merged['symbol'].nunique()} symbols")
    
    # ============================================================
    # SIGNAL ANALYSIS
    # ============================================================
    print(f"\n{'='*75}")
    print("  IMBALANCE → FORWARD RETURN CORRELATION")
    print(f"{'='*75}")
    
    print(f"\n  {'Feature':<20s} │ {'Corr 30m':>8s} │ {'Corr 60m':>8s} │ {'Corr 240m':>9s} │ {'N':>6s}")
    print(f"  {'─'*20}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*6}")
    
    for feat in ['imbalance_5', 'imbalance_20', 'imbalance_50',
                 'imbalance_5_z', 'imbalance_20_z', 'imbalance_50_z', 'spread_bps']:
        if feat not in merged.columns:
            continue
        valid = merged[[feat, 'fwd_30m', 'fwd_60m', 'fwd_240m']].dropna()
        if len(valid) < 50:
            continue
        c30 = valid[feat].corr(valid['fwd_30m'])
        c60 = valid[feat].corr(valid['fwd_60m'])
        c240 = valid[feat].corr(valid['fwd_240m'])
        print(f"  {feat:<20s} │ {c30:>+7.4f} │ {c60:>+7.4f} │ {c240:>+8.4f} │ {len(valid):>6d}")
    
    # Quintile analysis
    print(f"\n{'='*75}")
    print("  QUINTILE ANALYSIS: imbalance_20_z → forward returns")
    print(f"{'='*75}")
    
    feat = 'imbalance_20_z'
    if feat in merged.columns:
        valid = merged[[feat, 'fwd_30m', 'fwd_60m', 'fwd_240m']].dropna()
        valid['quintile'] = pd.qcut(valid[feat], 5, labels=['Q1(asks)', 'Q2', 'Q3', 'Q4', 'Q5(bids)'])
        
        print(f"\n  {'Quintile':<12s} │ {'Imb range':>12s} │ {'fwd 30m':>8s} │ {'fwd 60m':>8s} │ {'fwd 240m':>9s} │ {'N':>5s}")
        print(f"  {'─'*12}─┼─{'─'*12}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*5}")
        
        for q in ['Q1(asks)', 'Q2', 'Q3', 'Q4', 'Q5(bids)']:
            sub = valid[valid['quintile'] == q]
            imb_lo = sub[feat].min()
            imb_hi = sub[feat].max()
            print(f"  {q:<12s} │ {imb_lo:>+5.1f}→{imb_hi:>+4.1f} │ {sub['fwd_30m'].mean():>+7.0f} │ {sub['fwd_60m'].mean():>+7.0f} │ {sub['fwd_240m'].mean():>+8.0f} │ {len(sub):>5d}")
        
        # Q5 - Q1 spread
        q1 = valid[valid['quintile'] == 'Q1(asks)']
        q5 = valid[valid['quintile'] == 'Q5(bids)']
        for h in ['fwd_30m', 'fwd_60m', 'fwd_240m']:
            spread = q5[h].mean() - q1[h].mean()
            net = spread - RT_TAKER_BPS
            print(f"\n  Q5-Q1 spread at {h}: {spread:+.0f} bps raw, {net:+.0f} bps net")
    
    # Extreme imbalance as signal
    print(f"\n{'='*75}")
    print("  EXTREME IMBALANCE SIGNAL (z > 2 or z < -2)")
    print(f"{'='*75}")
    
    for feat in ['imbalance_5_z', 'imbalance_20_z', 'imbalance_50_z']:
        if feat not in merged.columns:
            continue
        valid = merged[[feat, 'fwd_30m', 'fwd_60m', 'fwd_240m']].dropna()
        
        for thresh in [1.5, 2.0, 3.0]:
            long_sig = valid[feat] > thresh
            short_sig = valid[feat] < -thresh
            
            long_rets = valid.loc[long_sig, 'fwd_240m']
            short_rets = -valid.loc[short_sig, 'fwd_240m']
            
            if len(long_rets) < 10 and len(short_rets) < 10:
                continue
            
            all_rets = pd.concat([long_rets, short_rets])
            net = all_rets.mean() - RT_TAKER_BPS
            wr = (all_rets > 0).mean() * 100
            
            status = '✅' if net > 20 else '⚠️' if net > 0 else '❌'
            print(f"  {status} {feat} |z|>{thresh}: n={len(all_rets)}, net={net:+.0f} bps, WR={wr:.0f}%")
    
    merged.to_csv(f'{OUT}/orderbook_signal.csv', index=True)
    print(f"\n⏱ Total: {time.time()-t0:.0f}s")
    print(f"✅ Saved: orderbook_signal.csv")


if __name__ == '__main__':
    main()
