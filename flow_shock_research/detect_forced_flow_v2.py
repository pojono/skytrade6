#!/usr/bin/env python3
"""
Forced Flow Detector v2 — Production-grade, all 5 fixes applied.

Fix 1: OB sync — use closest snapshot by timestamp (bisect), not every-10th
Fix 2: Check by timer (500ms intervals), not every 50th trade
Fix 3: Burst intensity normalized by local trade rate (1h rolling median)
Fix 4: pre_* windows end at t0 - WINDOW_SECONDS*1000 (no leakage)
Fix 5: All units verified (SOL for both volume and depth)

Parallelized by date using multiprocessing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys
import time
from bisect import bisect_right
from collections import deque
from multiprocessing import Pool

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")

# ── Stage-1 thresholds ──
WINDOW_MS = 15_000          # 15-second rolling window
CHECK_INTERVAL_MS = 500     # Fix 2: check every 500ms, not every N trades
FLOW_IMPACT_THRESHOLD = 70  # agg_vol / top_depth_L5 * 100
IMBALANCE_THRESHOLD = 0.60
SAME_SIDE_THRESHOLD = 0.75
MIN_AGG_TRADES = 20
COOLDOWN_MS = 30_000        # no re-trigger within 30s of last event
BURST_ZSCORE_MIN = 1.5      # Fix 3: trades_per_sec must be 1.5 MAD above 1h median

# ── Regime classifier thresholds (frozen) ──
FOLLOW_THRESHOLDS = {
    'vol_15m_q30': 0.51,
    'range_10s_q80': 61.34,
    'drift_2m_q70': 41.87,
}


def load_all_orderbook(date_str):
    """Load ALL orderbook.500 snapshots for a date, sorted by timestamp.
    Returns (timestamps[], snapshots[]) for bisect lookup.
    """
    ob_timestamps = []
    ob_snapshots = []

    for hour in range(24):
        ob_file = (DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}"
                   / "exchange=bybit" / "source=ws" / "market=linear"
                   / "stream=orderbook.500" / "symbol=SOLUSDT" / "data.jsonl.gz")
        if not ob_file.exists():
            continue
        try:
            with gzip.open(ob_file, 'rt') as f:
                for line in f:
                    try:
                        msg = json.loads(line)
                        if 'result' not in msg or 'data' not in msg['result']:
                            continue
                        ts = int(msg['result']['ts'])
                        ob = msg['result']['data']
                        bids = ob.get('b', [])[:5]
                        asks = ob.get('a', [])[:5]
                        if not bids or not asks:
                            continue
                        ob_timestamps.append(ts)
                        ob_snapshots.append((bids, asks))
                    except Exception:
                        continue
        except Exception:
            continue

    return ob_timestamps, ob_snapshots


def get_ob_at(ts, ob_timestamps, ob_snapshots):
    """Fix 1: Get the last orderbook snapshot with timestamp <= ts (bisect)."""
    idx = bisect_right(ob_timestamps, ts) - 1
    if idx < 0:
        return None
    return ob_snapshots[idx], ob_timestamps[idx]


def ob_depth_l5(bids, asks):
    """Sum of bid+ask sizes for L1..L5. Units: SOL."""
    total = 0.0
    for i in range(min(5, len(bids))):
        total += float(bids[i][1])
    for i in range(min(5, len(asks))):
        total += float(asks[i][1])
    return total


def ob_features(bids, asks):
    """Extract orderbook features from snapshot."""
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    spread = best_ask - best_bid
    spread_bps = spread / best_bid * 10_000

    depth_levels = []
    for i in range(min(5, len(bids), len(asks))):
        depth_levels.append(float(bids[i][1]) + float(asks[i][1]))
    while len(depth_levels) < 5:
        depth_levels.append(0.0)

    return {
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread': spread,
        'spread_bps': spread_bps,
        'depth_l1': depth_levels[0],
        'depth_l2': depth_levels[1],
        'depth_l3': depth_levels[2],
        'depth_l4': depth_levels[3],
        'depth_l5': depth_levels[4],
        'top_depth': depth_levels[0],
        'total_depth_l5': sum(depth_levels),
    }


def load_all_trades(date_str):
    """Load all trades for a date, sorted by timestamp.
    Returns list of dicts with keys: T, p, v, S (all native types).
    """
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
                        if 'result' not in msg or 'data' not in msg['result']:
                            continue
                        for t in msg['result']['data']:
                            trades.append((int(t['T']), float(t['p']),
                                           float(t['v']), t['S']))
                    except Exception:
                        continue
        except Exception:
            continue
    trades.sort(key=lambda x: x[0])
    return trades


def compute_regime_features(trades_sorted, event_ts, event_price):
    """Fix 4: Pre-event features. Windows end at (event_ts - WINDOW_MS) to
    guarantee zero overlap with the 15s event window.
    t0_features = event_ts - WINDOW_MS  (the start of forced flow)
    """
    t0 = event_ts - WINDOW_MS  # features end BEFORE forced-flow window

    windows = {'10s': 10_000, '30s': 30_000, '2m': 120_000,
               '5m': 300_000, '15m': 900_000}
    features = {}

    for name, ms in windows.items():
        start = t0 - ms
        end = t0

        # Binary search for start/end indices
        lo = bisect_right([t[0] for t in trades_sorted], start - 1)
        hi = bisect_right([t[0] for t in trades_sorted], end)
        window_trades = trades_sorted[lo:hi]

        if len(window_trades) < 5:
            continue

        prices = [t[1] for t in window_trades]
        volumes = [t[2] for t in window_trades]
        sides = [t[3] for t in window_trades]

        prefix = f'pre_{name}_'

        # Volatility
        if len(prices) > 1:
            rets = [(prices[i] - prices[i-1]) / prices[i-1] * 10_000
                    for i in range(1, len(prices))]
            features[prefix + 'vol'] = float(np.std(rets))

        # Range
        hi_p, lo_p = max(prices), min(prices)
        mean_p = np.mean(prices)
        features[prefix + 'range'] = (hi_p - lo_p) / mean_p * 10_000

        # Drift
        features[prefix + 'drift'] = ((prices[-1] - prices[0])
                                       / prices[0] * 10_000)

        # Imbalance
        buy_v = sum(v for v, s in zip(volumes, sides) if s == 'Buy')
        sell_v = sum(v for v, s in zip(volumes, sides) if s == 'Sell')
        total_v = buy_v + sell_v
        if total_v > 0:
            features[prefix + 'imbalance'] = (buy_v - sell_v) / total_v

    return features


def classify_follow(features):
    """Regime classifier — FOLLOW or NO_TRADE."""
    vol_15m = features.get('pre_15m_vol', np.nan)
    range_10s = features.get('pre_10s_range', np.nan)
    drift_2m = features.get('pre_2m_drift', np.nan)

    if pd.isna(vol_15m) or pd.isna(range_10s) or pd.isna(drift_2m):
        return 'NO_TRADE'

    if (vol_15m < FOLLOW_THRESHOLDS['vol_15m_q30']
            and range_10s > FOLLOW_THRESHOLDS['range_10s_q80']
            and abs(drift_2m) > FOLLOW_THRESHOLDS['drift_2m_q70']):
        return 'FOLLOW'
    return 'NO_TRADE'


def process_date(date_str):
    """Process one date: detect events, extract features. Returns list of dicts."""
    t_start = time.time()

    # ── Load data ──
    ob_timestamps, ob_snapshots = load_all_orderbook(date_str)
    if not ob_timestamps:
        print(f"[{date_str}] No OB data, skip", flush=True)
        return []

    trades = load_all_trades(date_str)
    if not trades:
        print(f"[{date_str}] No trade data, skip", flush=True)
        return []

    # ── Precompute trade-timestamps for bisect ──
    trade_ts = [t[0] for t in trades]

    # ── Fix 3: Rolling 1-hour trade-rate statistics ──
    # Build per-second trade counts for entire day
    if trades:
        first_sec = trades[0][0] // 1000
        last_sec = trades[-1][0] // 1000
        n_secs = last_sec - first_sec + 1
        sec_counts = np.zeros(n_secs, dtype=np.int32)
        for t in trades:
            sec_idx = t[0] // 1000 - first_sec
            if 0 <= sec_idx < n_secs:
                sec_counts[sec_idx] += 1
    else:
        first_sec = 0
        sec_counts = np.array([])

    def get_burst_zscore(check_ts, agg_trades_count):
        """Fix 3: Is current trade rate abnormal vs last 1 hour?"""
        tps_now = agg_trades_count / (WINDOW_MS / 1000)
        sec_idx = check_ts // 1000 - first_sec
        hour_start = max(0, sec_idx - 3600)
        hour_end = sec_idx
        if hour_end <= hour_start + 60:
            return 0.0  # not enough history
        hourly = sec_counts[hour_start:hour_end]
        med = np.median(hourly)
        mad = np.median(np.abs(hourly - med))
        if mad < 0.01:
            mad = 1.0
        return (tps_now - med) / (1.4826 * mad)

    # ── Fix 2: Timer-based checking ──
    events = []
    last_event_ts = 0

    # Generate check timestamps at 500ms intervals covering the day
    day_start_ts = trades[0][0]
    day_end_ts = trades[-1][0]
    check_ts = day_start_ts + WINDOW_MS  # first possible check

    while check_ts <= day_end_ts:
        # Cooldown
        if check_ts - last_event_ts < COOLDOWN_MS:
            check_ts += CHECK_INTERVAL_MS
            continue

        # Get trades in [check_ts - WINDOW_MS, check_ts]
        window_start = check_ts - WINDOW_MS
        lo = bisect_right(trade_ts, window_start - 1)
        hi = bisect_right(trade_ts, check_ts)
        window = trades[lo:hi]

        if len(window) < MIN_AGG_TRADES:
            check_ts += CHECK_INTERVAL_MS
            continue

        # Aggregate
        buy_vol = sum(t[2] for t in window if t[3] == 'Buy')
        sell_vol = sum(t[2] for t in window if t[3] == 'Sell')
        agg_vol = buy_vol + sell_vol

        if agg_vol == 0:
            check_ts += CHECK_INTERVAL_MS
            continue

        imbalance = abs(buy_vol - sell_vol) / agg_vol
        dominant_vol = max(buy_vol, sell_vol)
        same_side = dominant_vol / agg_vol
        direction = 'Buy' if buy_vol > sell_vol else 'Sell'

        # Fix 1: Get OB closest to check_ts
        ob_result = get_ob_at(check_ts, ob_timestamps, ob_snapshots)
        if ob_result is None:
            check_ts += CHECK_INTERVAL_MS
            continue
        (bids, asks), ob_ts = ob_result
        top_depth_l5 = ob_depth_l5(bids, asks)
        if top_depth_l5 == 0:
            check_ts += CHECK_INTERVAL_MS
            continue

        flow_impact = agg_vol / top_depth_l5 * 100

        # Fix 3: Burst intensity
        burst_z = get_burst_zscore(check_ts, len(window))

        # Max run
        max_run = 1
        cur_run = 1
        for i in range(1, len(window)):
            if window[i][3] == window[i-1][3]:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 1

        # ── TRIGGER ──
        if (flow_impact >= FLOW_IMPACT_THRESHOLD
                and imbalance >= IMBALANCE_THRESHOLD
                and same_side >= SAME_SIDE_THRESHOLD
                and len(window) >= MIN_AGG_TRADES
                and burst_z >= BURST_ZSCORE_MIN):

            event_price = window[-1][1]

            # OB features
            obf = ob_features(bids, asks)
            obf['ob_delay_ms'] = check_ts - ob_ts

            # Regime features (Fix 4: no leakage)
            regime = compute_regime_features(trades, check_ts, event_price)

            # Classification
            label = classify_follow(regime)

            event = {
                'timestamp': check_ts,
                'datetime': pd.to_datetime(check_ts, unit='ms').strftime(
                    '%Y-%m-%d %H:%M:%S.%f')[:-3],
                'date': date_str,
                'hour': pd.to_datetime(check_ts, unit='ms').hour,
                'direction': direction,
                'event_price': event_price,
                'flow_impact': flow_impact,
                'agg_vol': agg_vol,
                'agg_trades_count': len(window),
                'imbalance': imbalance,
                'same_side_share': same_side,
                'max_run': max_run,
                'burst_zscore': burst_z,
                'label': label,
            }
            event.update(obf)
            event.update(regime)

            events.append(event)
            last_event_ts = check_ts

        check_ts += CHECK_INTERVAL_MS

    elapsed = time.time() - t_start
    print(f"[{date_str}] {len(events):3d} events  "
          f"({len(trades):,} trades, {len(ob_timestamps):,} OB snaps, "
          f"{elapsed:.0f}s)", flush=True)
    return events


def main():
    print("=" * 80, flush=True)
    print("⚡ FORCED FLOW DETECTOR v2 — PRODUCTION GRADE", flush=True)
    print("=" * 80, flush=True)
    print(f"  Window:        {WINDOW_MS}ms ({WINDOW_MS//1000}s)", flush=True)
    print(f"  Check every:   {CHECK_INTERVAL_MS}ms (timer, not trade-count)", flush=True)
    print(f"  FlowImpact:    >= {FLOW_IMPACT_THRESHOLD}%", flush=True)
    print(f"  Imbalance:     >= {IMBALANCE_THRESHOLD}", flush=True)
    print(f"  SameSide:      >= {SAME_SIDE_THRESHOLD}", flush=True)
    print(f"  MinTrades:     >= {MIN_AGG_TRADES}", flush=True)
    print(f"  BurstZscore:   >= {BURST_ZSCORE_MIN}", flush=True)
    print(f"  Cooldown:      {COOLDOWN_MS}ms", flush=True)
    print(f"  OB sync:       bisect (closest snapshot <= t)", flush=True)
    print(f"  Pre-features:  end at t0 - {WINDOW_MS}ms (no leakage)", flush=True)
    print("=" * 80, flush=True)

    # Get all dates with OB data
    ob_dir = DATA_DIR_OB
    dates = sorted([d.name.replace('dt=', '')
                    for d in ob_dir.iterdir() if d.name.startswith('dt=')])

    print(f"\nDates to process: {len(dates)}", flush=True)
    print(f"Range: {dates[0]} → {dates[-1]}", flush=True)
    print(f"Workers: 6\n", flush=True)

    t0 = time.time()
    with Pool(6) as pool:
        results = pool.map(process_date, dates)

    all_events = []
    for r in results:
        all_events.extend(r)

    df = pd.DataFrame(all_events).sort_values('timestamp').reset_index(drop=True)
    elapsed = time.time() - t0

    # ── Save ──
    out_path = Path("results/v2_forced_flow_all.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    df.to_csv("results/v2_forced_flow_all.csv", index=False)

    # ── Summary ──
    print(f"\n{'='*80}", flush=True)
    print(f"✅ DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"   Total events: {len(df)}", flush=True)
    print(f"   Days with events: {df['date'].nunique()}", flush=True)
    if len(df) > 0:
        print(f"   Events/day: {len(df)/df['date'].nunique():.1f}", flush=True)
        print(f"   FOLLOW: {(df['label']=='FOLLOW').sum()}", flush=True)
        print(f"   NO_TRADE: {(df['label']=='NO_TRADE').sum()}", flush=True)

        print(f"\n   Direction:", flush=True)
        for d, c in df['direction'].value_counts().items():
            print(f"      {d}: {c}", flush=True)

        print(f"\n   By month:", flush=True)
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        for m, sub in df.groupby('month'):
            follow_n = (sub['label'] == 'FOLLOW').sum()
            print(f"      {m}: {len(sub)} events ({follow_n} FOLLOW)", flush=True)

        print(f"\n   OB delay stats (ms):", flush=True)
        print(f"      mean: {df['ob_delay_ms'].mean():.0f}", flush=True)
        print(f"      p50:  {df['ob_delay_ms'].median():.0f}", flush=True)
        print(f"      p95:  {df['ob_delay_ms'].quantile(0.95):.0f}", flush=True)

    print(f"\n💾 results/v2_forced_flow_all.parquet", flush=True)
    print(f"💾 results/v2_forced_flow_all.csv", flush=True)
    print(f"{'='*80}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
