#!/usr/bin/env python3
"""
Forced Flow Detector v2 — FAST version.

Fixes applied:
1. OB sync: bisect closest snapshot <= t (sampled every 50th for speed)
2. Timer-based check every 500ms (not every Nth trade)
3. Burst intensity: normalized by 1h trade rate
4. Cooldown 30s between events
5. No regime features inline — those are added AFTER on events only

Speed target: <30s per date (vs 250s in v2).
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
DATA_DIR_OB = Path("data_bybit/SOLUSDT/orderbook_all/dataminer/data/archive/raw")

WINDOW_MS = 15_000
CHECK_INTERVAL_MS = 500
FLOW_IMPACT_THRESHOLD = 70
IMBALANCE_THRESHOLD = 0.60
SAME_SIDE_THRESHOLD = 0.75
MIN_AGG_TRADES = 20
COOLDOWN_MS = 30_000
BURST_ZSCORE_MIN = 2.0  # tightened from 1.5


def load_ob_sampled(date_str, sample_rate=50):
    """Load OB snapshots sampled every Nth line. Returns numpy arrays for fast bisect."""
    ts_list = []
    depth_list = []  # total L1-L5 depth
    spread_list = []

    for hour in range(24):
        ob_file = (DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}"
                   / "exchange=bybit" / "source=ws" / "market=linear"
                   / "stream=orderbook.500" / "symbol=SOLUSDT" / "data.jsonl.gz")
        if not ob_file.exists():
            continue
        try:
            with gzip.open(ob_file, 'rt') as f:
                for i, line in enumerate(f):
                    if i % sample_rate != 0:
                        continue
                    try:
                        msg = json.loads(line)
                        r = msg.get('result')
                        if not r or 'data' not in r:
                            continue
                        ts = int(r['ts'])
                        ob = r['data']
                        bids = ob.get('b', [])[:5]
                        asks = ob.get('a', [])[:5]
                        if not bids or not asks:
                            continue

                        depth = 0.0
                        for b in bids:
                            depth += float(b[1])
                        for a in asks:
                            depth += float(a[1])

                        spread = float(asks[0][0]) - float(bids[0][0])

                        ts_list.append(ts)
                        depth_list.append(depth)
                        spread_list.append(spread)
                    except Exception:
                        continue
        except Exception:
            continue

    return np.array(ts_list), np.array(depth_list), np.array(spread_list)


def load_trades_fast(date_str):
    """Load all trades as numpy arrays: ts, price, vol, side_is_buy."""
    ts_l, price_l, vol_l, buy_l = [], [], [], []

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
                            ts_l.append(int(t['T']))
                            price_l.append(float(t['p']))
                            vol_l.append(float(t['v']))
                            buy_l.append(1 if t['S'] == 'Buy' else 0)
                    except Exception:
                        continue
        except Exception:
            continue

    if not ts_l:
        return None

    idx = np.argsort(ts_l)
    return {
        'ts': np.array(ts_l)[idx],
        'price': np.array(price_l)[idx],
        'vol': np.array(vol_l)[idx],
        'buy': np.array(buy_l, dtype=np.int8)[idx],
    }


def process_date(date_str):
    """Detect forced flow events for one date."""
    t0 = time.time()

    ob_ts, ob_depth, ob_spread = load_ob_sampled(date_str)
    if len(ob_ts) == 0:
        print(f"[{date_str}] No OB data, skip", flush=True)
        return []

    td = load_trades_fast(date_str)
    if td is None:
        print(f"[{date_str}] No trade data, skip", flush=True)
        return []

    trade_ts = td['ts']
    trade_price = td['price']
    trade_vol = td['vol']
    trade_buy = td['buy']
    n_trades = len(trade_ts)

    # Pre-compute per-second trade counts for burst detection
    first_sec = int(trade_ts[0] // 1000)
    last_sec = int(trade_ts[-1] // 1000)
    n_secs = last_sec - first_sec + 1
    sec_counts = np.zeros(n_secs, dtype=np.int32)
    sec_idx_arr = (trade_ts // 1000 - first_sec).astype(np.int32)
    for i in range(n_trades):
        si = sec_idx_arr[i]
        if 0 <= si < n_secs:
            sec_counts[si] += 1

    events = []
    last_event_ts = 0

    # Timer-based check
    check_ts = int(trade_ts[0]) + WINDOW_MS
    end_ts = int(trade_ts[-1])

    while check_ts <= end_ts:
        if check_ts - last_event_ts < COOLDOWN_MS:
            check_ts += CHECK_INTERVAL_MS
            continue

        # Window bounds
        w_start = check_ts - WINDOW_MS
        lo = int(np.searchsorted(trade_ts, w_start, side='left'))
        hi = int(np.searchsorted(trade_ts, check_ts, side='right'))

        n_window = hi - lo
        if n_window < MIN_AGG_TRADES:
            check_ts += CHECK_INTERVAL_MS
            continue

        # Aggregate volume
        w_vol = trade_vol[lo:hi]
        w_buy = trade_buy[lo:hi]
        buy_vol = float(np.sum(w_vol * w_buy))
        sell_vol = float(np.sum(w_vol * (1 - w_buy)))
        agg_vol = buy_vol + sell_vol

        if agg_vol == 0:
            check_ts += CHECK_INTERVAL_MS
            continue

        imbalance = abs(buy_vol - sell_vol) / agg_vol
        if imbalance < IMBALANCE_THRESHOLD:
            check_ts += CHECK_INTERVAL_MS
            continue

        same_side = max(buy_vol, sell_vol) / agg_vol
        if same_side < SAME_SIDE_THRESHOLD:
            check_ts += CHECK_INTERVAL_MS
            continue

        # OB depth at check_ts (bisect)
        ob_idx = int(np.searchsorted(ob_ts, check_ts, side='right')) - 1
        if ob_idx < 0:
            check_ts += CHECK_INTERVAL_MS
            continue

        depth = float(ob_depth[ob_idx])
        if depth == 0:
            check_ts += CHECK_INTERVAL_MS
            continue

        flow_impact = agg_vol / depth * 100
        if flow_impact < FLOW_IMPACT_THRESHOLD:
            check_ts += CHECK_INTERVAL_MS
            continue

        # Burst z-score
        tps_now = n_window / (WINDOW_MS / 1000)
        sec_idx = int(check_ts // 1000 - first_sec)
        h_start = max(0, sec_idx - 3600)
        if sec_idx - h_start < 60:
            check_ts += CHECK_INTERVAL_MS
            continue
        hourly = sec_counts[h_start:sec_idx]
        med = float(np.median(hourly))
        mad = float(np.median(np.abs(hourly.astype(np.float64) - med)))
        if mad < 0.5:
            mad = 0.5
        burst_z = (tps_now - med) / (1.4826 * mad)

        if burst_z < BURST_ZSCORE_MIN:
            check_ts += CHECK_INTERVAL_MS
            continue

        # Max run
        w_sides = trade_buy[lo:hi]
        max_run = 1
        cur_run = 1
        for i in range(1, n_window):
            if w_sides[i] == w_sides[i-1]:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 1

        direction = 'Buy' if buy_vol > sell_vol else 'Sell'
        event_price = float(trade_price[hi - 1])
        ob_delay = int(check_ts - ob_ts[ob_idx])
        spread = float(ob_spread[ob_idx])

        events.append({
            'timestamp': check_ts,
            'datetime': pd.to_datetime(check_ts, unit='ms').strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'date': date_str,
            'hour': pd.to_datetime(check_ts, unit='ms').hour,
            'direction': direction,
            'event_price': event_price,
            'flow_impact': flow_impact,
            'agg_vol': agg_vol,
            'agg_trades_count': n_window,
            'imbalance': imbalance,
            'same_side_share': same_side,
            'max_run': max_run,
            'burst_zscore': burst_z,
            'ob_depth_l5': depth,
            'ob_spread': spread,
            'ob_delay_ms': ob_delay,
        })
        last_event_ts = check_ts

        check_ts += CHECK_INTERVAL_MS

    elapsed = time.time() - t0
    print(f"[{date_str}] {len(events):3d} events  "
          f"({n_trades:,} trades, {len(ob_ts):,} OB, {elapsed:.0f}s)",
          flush=True)
    return events


def main():
    print("=" * 80, flush=True)
    print("⚡ FORCED FLOW DETECTOR v2 — FAST", flush=True)
    print("=" * 80, flush=True)
    print(f"  Window:      {WINDOW_MS}ms  |  Check: {CHECK_INTERVAL_MS}ms", flush=True)
    print(f"  FlowImpact:  >= {FLOW_IMPACT_THRESHOLD}%", flush=True)
    print(f"  Imbalance:   >= {IMBALANCE_THRESHOLD}  |  SameSide: >= {SAME_SIDE_THRESHOLD}", flush=True)
    print(f"  MinTrades:   >= {MIN_AGG_TRADES}  |  BurstZ: >= {BURST_ZSCORE_MIN}", flush=True)
    print(f"  Cooldown:    {COOLDOWN_MS}ms  |  OB sample: 1/50", flush=True)
    print("=" * 80, flush=True)

    ob_dir = DATA_DIR_OB
    dates = sorted([d.name.replace('dt=', '')
                    for d in ob_dir.iterdir() if d.name.startswith('dt=')])

    print(f"\nDates: {len(dates)} | Workers: 6\n", flush=True)

    t0 = time.time()
    with Pool(6) as pool:
        results = pool.map(process_date, dates)

    all_events = []
    for r in results:
        all_events.extend(r)

    df = pd.DataFrame(all_events).sort_values('timestamp').reset_index(drop=True)
    elapsed = time.time() - t0

    Path("results").mkdir(exist_ok=True)
    df.to_parquet("results/v2_events_raw.parquet", index=False)
    df.to_csv("results/v2_events_raw.csv", index=False)

    print(f"\n{'='*80}", flush=True)
    print(f"✅ DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"   Events: {len(df)}", flush=True)
    if len(df) > 0:
        print(f"   Days: {df['date'].nunique()}", flush=True)
        print(f"   Events/day: {len(df)/df['date'].nunique():.1f}", flush=True)
        print(f"   Buy: {(df['direction']=='Buy').sum()}  Sell: {(df['direction']=='Sell').sum()}", flush=True)
        print(f"   FlowImpact: mean={df['flow_impact'].mean():.0f} p50={df['flow_impact'].median():.0f}", flush=True)
        print(f"   BurstZ: mean={df['burst_zscore'].mean():.1f} p50={df['burst_zscore'].median():.1f}", flush=True)
        print(f"   OB delay: mean={df['ob_delay_ms'].mean():.0f}ms p95={df['ob_delay_ms'].quantile(0.95):.0f}ms", flush=True)

        print(f"\n   By month:", flush=True)
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        for m, sub in df.groupby('month'):
            print(f"      {m}: {len(sub)} events ({len(sub)/sub['date'].nunique():.1f}/day)", flush=True)
    print(f"\n💾 results/v2_events_raw.parquet", flush=True)
    print(f"{'='*80}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
