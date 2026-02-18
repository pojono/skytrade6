#!/usr/bin/env python3
"""
v28: Long-Horizon Liquidation Buildup Before Regime Switches

v27c showed liquidations lead by 20-600s at second resolution.
But the P75 first-crossing showed 594s (10min) median lead.
Question: is there a GRADUAL buildup over hours before the switch?

EXPERIMENT DESIGN:
==================

1. Use the same tick-level data (BTC, 7 days, second resolution)
2. Look at MUCH wider windows: ±2 hours around each switch
3. Compute multiple timescales of liq activity:
   - Raw per-second liq count/notional
   - Rolling 10s, 30s, 60s, 5min, 15min, 30min, 1h windows
4. Normalize each switch individually (relative to its own baseline)
   to avoid averaging out different magnitude events
5. Measure: at what timescale does the buildup become detectable?
6. Build a "regime switch probability" curve: P(switch in next T | liq features)
7. Test on held-out data

Key insight to test: maybe individual liquidations are small/noisy,
but the RATE of liquidation events gradually increases over 30-60min
before the big cascade that triggers the switch.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import time
import json
import gzip
import warnings
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/BTCUSDT")
SYMBOL = "BTCUSDT"


def mem_gb():
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label=""):
    used, avail = mem_gb()
    print(f"  [RAM] used={used:.1f}GB avail={avail:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING (reuse from v27c — chunked, RAM-safe)
# ============================================================================

def load_liquidations_all(dates):
    all_liqs = []
    for date_str in dates:
        pattern = f"liquidation_{date_str}_hr*.jsonl.gz"
        files = sorted((DATA_DIR / "bybit" / "liquidations").glob(pattern))
        if not files:
            files = sorted(DATA_DIR.glob(pattern))
        day_count = 0
        for f in files:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for ev in data['result']['data']:
                                all_liqs.append({
                                    'ts_ms': int(ev['T']),
                                    'side': ev['S'],
                                    'volume': float(ev['v']),
                                    'price': float(ev['p']),
                                })
                                day_count += 1
                    except:
                        continue
        print(f"  {date_str}: {day_count:>6,} liquidations", flush=True)
    if not all_liqs:
        return pd.DataFrame()
    liqs = pd.DataFrame(all_liqs)
    liqs['ts_s'] = liqs['ts_ms'] // 1000
    liqs['notional'] = liqs['volume'] * liqs['price']
    liqs['is_buy'] = (liqs['side'] == 'Buy').astype(np.int8)
    print(f"  Total: {len(liqs):,} liquidations")
    return liqs


def aggregate_day_to_seconds(date_str, liqs):
    path = DATA_DIR / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        return None
    t0 = time.time()
    df = pd.read_csv(path, compression='gzip', usecols=['timestamp', 'price', 'size', 'side'])
    n_trades = len(df)
    ts_s = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.float64)
    notionals = prices * sizes
    del df; gc.collect()

    day_start = ts_s.min()
    day_end = ts_s.max()
    n_seconds = day_end - day_start + 1
    offsets = ts_s - day_start

    trade_count = np.bincount(offsets, minlength=n_seconds).astype(np.int32)
    trade_notional = np.bincount(offsets, weights=notionals, minlength=n_seconds)

    price_last = np.full(n_seconds, np.nan, dtype=np.float64)
    _, last_indices = np.unique(offsets[::-1], return_index=True)
    last_indices = len(offsets) - 1 - last_indices
    for ui, li in zip(np.unique(offsets), last_indices):
        price_last[ui] = prices[li]

    del ts_s, prices, sizes, notionals, offsets; gc.collect()

    mask = np.isnan(price_last)
    if not mask.all():
        first_valid = np.argmin(mask)
        for i in range(first_valid + 1, n_seconds):
            if mask[i]:
                price_last[i] = price_last[i - 1]

    liq_count = np.zeros(n_seconds, dtype=np.int32)
    liq_notional = np.zeros(n_seconds, dtype=np.float64)
    if len(liqs) > 0:
        liq_day = liqs[(liqs['ts_s'] >= day_start) & (liqs['ts_s'] <= day_end)]
        if len(liq_day) > 0:
            l_offsets = (liq_day['ts_s'].values - day_start).astype(np.int64)
            valid = (l_offsets >= 0) & (l_offsets < n_seconds)
            l_offsets = l_offsets[valid]
            l_not = liq_day['notional'].values[valid]
            liq_count = np.bincount(l_offsets, minlength=n_seconds).astype(np.int32)
            liq_notional = np.bincount(l_offsets, weights=l_not, minlength=n_seconds)

    elapsed = time.time() - t0
    print(f"  {date_str}: {n_trades:,} trades → {n_seconds:,}s ({elapsed:.1f}s)", flush=True)
    return {
        'day_start': day_start, 'n_seconds': n_seconds,
        'trade_count': trade_count, 'trade_notional': trade_notional,
        'price_last': price_last,
        'liq_count': liq_count, 'liq_notional': liq_notional,
    }


def build_second_series(dates, liqs):
    day_results = []
    for date_str in dates:
        result = aggregate_day_to_seconds(date_str, liqs)
        if result is not None:
            day_results.append(result)
        gc.collect()

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n_seconds'] - 1
    n = ts_end - ts_start + 1

    trade_count = np.zeros(n, dtype=np.int32)
    trade_notional = np.zeros(n, dtype=np.float64)
    price_last = np.full(n, np.nan, dtype=np.float64)
    liq_count = np.zeros(n, dtype=np.int32)
    liq_notional = np.zeros(n, dtype=np.float64)

    for dr in day_results:
        o = dr['day_start'] - ts_start
        l = dr['n_seconds']
        trade_count[o:o+l] = dr['trade_count']
        trade_notional[o:o+l] = dr['trade_notional']
        price_last[o:o+l] = dr['price_last']
        liq_count[o:o+l] = dr['liq_count']
        liq_notional[o:o+l] = dr['liq_notional']
    del day_results; gc.collect()

    for i in range(1, n):
        if np.isnan(price_last[i]):
            price_last[i] = price_last[i - 1]

    # Log returns
    log_ret = np.zeros(n, dtype=np.float64)
    valid = (price_last[1:] > 0) & (price_last[:-1] > 0)
    log_ret[1:][valid] = np.log(price_last[1:][valid] / price_last[:-1][valid])

    def fast_rolling_std(arr, window):
        cs = np.cumsum(arr); cs2 = np.cumsum(arr ** 2)
        result = np.full(len(arr), np.nan, dtype=np.float64)
        s = cs[window:] - cs[:-window]
        s2 = cs2[window:] - cs2[:-window]
        var = s2 / window - (s / window) ** 2
        np.clip(var, 0, None, out=var)
        result[window:] = np.sqrt(var)
        return result

    def fast_rolling_sum(arr, window):
        cs = np.cumsum(arr.astype(np.float64))
        result = np.zeros(len(arr), dtype=np.float64)
        result[window:] = cs[window:] - cs[:-window]
        return result

    print_mem("computing features")

    vol_60s = fast_rolling_std(log_ret, 60)

    # Multi-timescale liq rolling sums
    liq_10s = fast_rolling_sum(liq_count, 10)
    liq_30s = fast_rolling_sum(liq_count, 30)
    liq_60s = fast_rolling_sum(liq_count, 60)
    liq_300s = fast_rolling_sum(liq_count, 300)
    liq_900s = fast_rolling_sum(liq_count, 900)
    liq_1800s = fast_rolling_sum(liq_count, 1800)
    liq_3600s = fast_rolling_sum(liq_count, 3600)

    liq_not_60s = fast_rolling_sum(liq_notional, 60)
    liq_not_300s = fast_rolling_sum(liq_notional, 300)
    liq_not_900s = fast_rolling_sum(liq_notional, 900)
    liq_not_3600s = fast_rolling_sum(liq_notional, 3600)

    trade_10s = fast_rolling_sum(trade_count, 10)
    trade_60s = fast_rolling_sum(trade_count, 60)

    seconds = np.arange(ts_start, ts_end + 1)

    data = {
        'ts_s': seconds, 'price': price_last, 'log_ret': log_ret,
        'vol_60s': vol_60s,
        'trade_count': trade_count, 'trade_10s': trade_10s, 'trade_60s': trade_60s,
        'liq_count': liq_count, 'liq_notional': liq_notional,
        'liq_10s': liq_10s, 'liq_30s': liq_30s, 'liq_60s': liq_60s,
        'liq_300s': liq_300s, 'liq_900s': liq_900s,
        'liq_1800s': liq_1800s, 'liq_3600s': liq_3600s,
        'liq_not_60s': liq_not_60s, 'liq_not_300s': liq_not_300s,
        'liq_not_900s': liq_not_900s, 'liq_not_3600s': liq_not_3600s,
    }

    # Free intermediates
    del (trade_count, trade_notional, price_last, log_ret, vol_60s,
         liq_count, liq_notional, liq_10s, liq_30s, liq_60s,
         liq_300s, liq_900s, liq_1800s, liq_3600s,
         liq_not_60s, liq_not_300s, liq_not_900s, liq_not_3600s,
         trade_10s, trade_60s, seconds)
    gc.collect()

    sec_df = pd.DataFrame(data)
    del data; gc.collect()

    print(f"  Built {len(sec_df):,} seconds")
    print_mem("done")
    return sec_df, ts_start


def detect_regime_switches(sec_df, vol_col='vol_60s', median_window=3600,
                           threshold=2.0, min_gap=1800):
    vol = sec_df[vol_col].values
    n = len(vol)
    vol_median = np.full(n, np.nan)
    for i in range(median_window, n):
        window = vol[i-median_window:i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > median_window // 2:
            vol_median[i] = np.median(valid_w)

    switches = []
    in_volatile = False
    for i in range(median_window + 1, n):
        if np.isnan(vol[i]) or np.isnan(vol_median[i]):
            continue
        if not in_volatile and vol[i] > vol_median[i] * threshold:
            if not switches or (sec_df['ts_s'].iloc[i] - switches[-1]) > min_gap:
                switches.append(sec_df['ts_s'].iloc[i])
            in_volatile = True
        elif in_volatile and vol[i] < vol_median[i] * 1.2:
            in_volatile = False
    return switches


# ============================================================================
# ANALYSIS 1: Wide-window profiles (±2 hours) at multiple timescales
# ============================================================================

def analysis_wide_profiles(sec_df, switches):
    """Average profiles ±2h at 30-second resolution, multiple liq timescales."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: Wide Profiles ±2h Around {len(switches)} Switches")
    print(f"{'='*70}")

    window = 7200  # ±2 hours
    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    cols = ['vol_60s', 'liq_60s', 'liq_300s', 'liq_900s', 'liq_1800s', 'liq_3600s',
            'liq_not_60s', 'liq_not_900s', 'trade_10s']
    profiles = {c: [] for c in cols}

    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < window or idx + window >= len(ts_arr):
            continue
        for c in cols:
            profiles[c].append(sec_df[c].values[idx-window:idx+window])

    n_valid = len(profiles['vol_60s'])
    print(f"  Valid switches: {n_valid}")

    avgs = {c: np.nanmean(profiles[c], axis=0) for c in cols}

    def norm(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        return (arr - mn) / (mx - mn + 1e-15)

    # Print at 1-minute resolution for the wide view
    step = 60
    print(f"\n  {'Offset':>8s}  {'vol_60s':>10s}  {'v_norm':>7s}  "
          f"{'liq_60s':>8s}  {'l60_n':>6s}  {'liq_5m':>7s}  {'l5m_n':>6s}  "
          f"{'liq_15m':>8s}  {'l15m_n':>7s}  {'liq_30m':>8s}  {'l30m_n':>7s}  "
          f"{'liq_1h':>7s}  {'l1h_n':>6s}")
    print(f"  {'-'*115}")

    vol_n = norm(avgs['vol_60s'])
    l60_n = norm(avgs['liq_60s'])
    l300_n = norm(avgs['liq_300s'])
    l900_n = norm(avgs['liq_900s'])
    l1800_n = norm(avgs['liq_1800s'])
    l3600_n = norm(avgs['liq_3600s'])

    for i in range(0, len(avgs['vol_60s']), step):
        offset = i - window
        marker = " ← SWITCH" if abs(offset) < step // 2 else ""
        print(f"  {offset:>+7d}s  {avgs['vol_60s'][i]:>10.6f}  {vol_n[i]:>7.3f}  "
              f"{avgs['liq_60s'][i]:>8.2f}  {l60_n[i]:>6.3f}  "
              f"{avgs['liq_300s'][i]:>7.1f}  {l300_n[i]:>6.3f}  "
              f"{avgs['liq_900s'][i]:>8.1f}  {l900_n[i]:>7.3f}  "
              f"{avgs['liq_1800s'][i]:>8.1f}  {l1800_n[i]:>7.3f}  "
              f"{avgs['liq_3600s'][i]:>7.1f}  {l3600_n[i]:>6.3f}{marker}")

    # When does each timescale first reach 25%, 50% of peak?
    print(f"\n  First time each signal reaches threshold (before switch):")
    print(f"  {'Signal':>15s}  {'25% of peak':>12s}  {'50% of peak':>12s}  {'75% of peak':>12s}")
    print(f"  {'-'*55}")

    for name, arr_n in [('vol_60s', vol_n), ('liq_60s', l60_n),
                         ('liq_5min', l300_n), ('liq_15min', l900_n),
                         ('liq_30min', l1800_n), ('liq_1h', l3600_n)]:
        results = {}
        for pct, label in [(0.25, '25% of peak'), (0.50, '50% of peak'), (0.75, '75% of peak')]:
            first = None
            for j in range(window):  # only before switch
                if arr_n[j] >= pct:
                    first = j - window
                    break
            results[label] = f"{first}s" if first is not None else "never"
        print(f"  {name:>15s}  {results['25% of peak']:>12s}  {results['50% of peak']:>12s}  {results['75% of peak']:>12s}")

    return avgs, profiles


# ============================================================================
# ANALYSIS 2: Per-switch normalized buildup
# ============================================================================

def analysis_normalized_buildup(sec_df, switches):
    """Normalize each switch individually to see the buildup pattern."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: Per-Switch Normalized Buildup (±2h)")
    print(f"{'='*70}")

    window = 7200
    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    # For each switch, compute the liq_300s profile and normalize
    # by the BASELINE (average of -2h to -1h before switch)
    normalized_profiles = []
    baseline_window = (window - 3600, window - 600)  # -2h to -1h10m (avoid contamination near switch)

    liq_300s = sec_df['liq_300s'].values
    liq_900s = sec_df['liq_900s'].values

    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < window or idx + window >= len(ts_arr):
            continue

        profile = liq_900s[idx-window:idx+window].copy()
        baseline = np.mean(profile[baseline_window[0]:baseline_window[1]])

        if baseline > 0.5:  # need some baseline activity
            normalized = profile / baseline
            normalized_profiles.append(normalized)

    print(f"  Switches with sufficient baseline: {len(normalized_profiles)}")

    if not normalized_profiles:
        return

    avg_norm = np.mean(normalized_profiles, axis=0)
    med_norm = np.median(normalized_profiles, axis=0)
    p75_norm = np.percentile(normalized_profiles, 75, axis=0)
    p25_norm = np.percentile(normalized_profiles, 25, axis=0)

    step = 120  # 2-minute resolution
    print(f"\n  Liq_15min relative to baseline (-2h to -1h average = 1.0x)")
    print(f"  {'Offset':>8s}  {'Mean':>8s}  {'Median':>8s}  {'P25':>8s}  {'P75':>8s}  {'Visual':>30s}")
    print(f"  {'-'*75}")

    for i in range(0, len(avg_norm), step):
        offset = i - window
        bar_len = int(min(avg_norm[i], 10) * 3)
        bar = '█' * max(bar_len, 0)
        marker = " ← SWITCH" if abs(offset) < step // 2 else ""
        print(f"  {offset:>+7d}s  {avg_norm[i]:>8.2f}x  {med_norm[i]:>8.2f}x  "
              f"{p25_norm[i]:>8.2f}x  {p75_norm[i]:>8.2f}x  {bar}{marker}")


# ============================================================================
# ANALYSIS 3: Gradual rate increase detection
# ============================================================================

def analysis_rate_gradient(sec_df, switches):
    """Is the liq rate INCREASING in the hour before a switch?"""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 3: Liq Rate Gradient Before Switches")
    print(f"{'='*70}")

    window = 7200
    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    liq_900s = sec_df['liq_900s'].values

    # For each switch, compute the slope of liq_900s in different windows before
    windows_before = [
        (-3600, -1800, "1h→30m before"),
        (-1800, -600,  "30m→10m before"),
        (-600, -60,    "10m→1m before"),
        (-3600, -60,   "1h→1m before (full)"),
    ]

    results = {w[2]: [] for w in windows_before}
    non_switch_results = {w[2]: [] for w in windows_before}

    valid_switch_indices = []
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is not None and idx >= window and idx + window < len(ts_arr):
            valid_switch_indices.append(idx)

    for idx in valid_switch_indices:
        for start, end, label in windows_before:
            vals = liq_900s[idx+start:idx+end]
            if len(vals) > 10:
                baseline = np.mean(vals[:60])
                endpoint = np.mean(vals[-60:])
                slope = (endpoint - baseline) / max(baseline, 0.1)
                results[label].append(slope)

    # Compare with random non-switch points — use exclusion zones of ±30min
    np.random.seed(42)
    excl_mask = np.zeros(len(ts_arr), dtype=bool)
    for idx in valid_switch_indices:
        lo = max(0, idx - 1800)
        hi = min(len(ts_arr), idx + 1801)
        excl_mask[lo:hi] = True

    candidate_indices = np.arange(window, len(ts_arr) - window)
    non_switch_pool = candidate_indices[~excl_mask[candidate_indices]]
    print(f"  Non-switch pool: {len(non_switch_pool):,} seconds (excl. ±30min around switches)")

    sample_size = min(len(non_switch_pool), len(valid_switch_indices) * 5)
    sample = np.random.choice(non_switch_pool, size=sample_size, replace=False)
    print(f"  Random sample: {sample_size:,}")

    for idx in sample:
        for start, end, label in windows_before:
            vals = liq_900s[idx+start:idx+end]
            if len(vals) > 10:
                baseline = np.mean(vals[:60])
                endpoint = np.mean(vals[-60:])
                slope = (endpoint - baseline) / max(baseline, 0.1)
                non_switch_results[label].append(slope)

    print(f"\n  Liq rate change (relative) before switch vs random:")
    print(f"  {'Window':>25s}  {'Switch mean':>12s}  {'Random mean':>12s}  {'Diff':>8s}  {'p-value':>10s}")
    print(f"  {'-'*75}")

    for start, end, label in windows_before:
        sw_vals = np.array(results[label])
        ns_vals = np.array(non_switch_results[label])
        if len(sw_vals) > 10 and len(ns_vals) > 10:
            sw_vals = sw_vals[np.isfinite(sw_vals)]
            ns_vals = ns_vals[np.isfinite(ns_vals)]
            stat, pval = mannwhitneyu(sw_vals, ns_vals, alternative='greater')
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            print(f"  {label:>25s}  {np.mean(sw_vals):>+12.3f}  {np.mean(ns_vals):>+12.3f}  "
                  f"{np.mean(sw_vals)-np.mean(ns_vals):>+8.3f}  {pval:>9.6f}{sig}")


# ============================================================================
# ANALYSIS 4: Conditional probability curve
# ============================================================================

def analysis_probability_curve(sec_df, switches):
    """P(switch within T | current liq features) at different horizons."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 4: P(Switch Within T | Liq State)")
    print(f"{'='*70}")

    ts_arr = sec_df['ts_s'].values
    switch_set = set(switches)

    # For each second, compute: is there a switch within next T seconds?
    # Then bin by current liq_900s level

    liq_900s = sec_df['liq_900s'].values
    vol_60s = sec_df['vol_60s'].values

    # Use test set only (last 30%)
    split = int(len(ts_arr) * 0.7)

    for horizon in [60, 300, 900, 1800, 3600]:
        # Build target: switch within next `horizon` seconds
        target = np.zeros(len(ts_arr), dtype=np.int8)
        for sw in switches:
            idx_sw = np.searchsorted(ts_arr, sw)
            start = max(0, idx_sw - horizon)
            for j in range(start, idx_sw):
                target[j] = 1

        # Only test set
        liq_test = liq_900s[split:]
        tgt_test = target[split:]
        vol_test = vol_60s[split:]

        # Remove nans
        valid = ~np.isnan(liq_test) & ~np.isnan(vol_test)
        liq_v = liq_test[valid]
        tgt_v = tgt_test[valid]

        if len(liq_v) < 1000:
            continue

        base_rate = tgt_v.mean()

        # Bin by liq_900s percentile
        try:
            pcts = [0, 50, 75, 90, 95, 99, 100]
            thresholds = np.percentile(liq_v, pcts)
        except:
            continue

        print(f"\n  Horizon: {horizon}s ({horizon//60}min) — base rate: {base_rate:.4f} ({base_rate*100:.2f}%)")
        print(f"  {'Liq_15m range':>20s}  {'P(switch)':>10s}  {'Lift':>8s}  {'Count':>10s}")
        print(f"  {'-'*55}")

        for j in range(len(pcts) - 1):
            mask = (liq_v >= thresholds[j]) & (liq_v < thresholds[j+1])
            if mask.sum() < 50:
                continue
            prob = tgt_v[mask].mean()
            lift = prob / max(base_rate, 1e-6)
            label = f"P{pcts[j]}-P{pcts[j+1]}"
            print(f"  {label:>20s}  {prob:>10.4f}  {lift:>7.1f}x  {mask.sum():>10,}")


# ============================================================================
# ANALYSIS 5: Earliest detectable signal
# ============================================================================

def analysis_earliest_signal(sec_df, switches):
    """At what time before the switch does liq become statistically elevated?"""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 5: Earliest Detectable Liq Signal Before Switch")
    print(f"{'='*70}")

    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    liq_cols = ['liq_60s', 'liq_300s', 'liq_900s', 'liq_1800s', 'liq_3600s']
    col_labels = ['1min', '5min', '15min', '30min', '1h']

    # For each time offset before switch, test if liq is significantly elevated
    offsets_to_test = [-7200, -3600, -1800, -900, -600, -300, -180, -120, -60, -30, -10, 0]

    # Collect valid switch indices
    valid_switch_indices = []
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is not None and idx >= 7200 and idx + 7200 < len(ts_arr):
            valid_switch_indices.append(idx)

    # Build exclusion zones (±30min around switches) using numpy boolean mask
    excl_mask = np.zeros(len(ts_arr), dtype=bool)
    for idx in valid_switch_indices:
        lo = max(0, idx - 1800)
        hi = min(len(ts_arr), idx + 1801)
        excl_mask[lo:hi] = True

    np.random.seed(42)
    candidate_indices = np.arange(7200, len(ts_arr) - 7200)
    non_switch_pool = candidate_indices[~excl_mask[candidate_indices]]
    sample = np.random.choice(non_switch_pool,
                               size=min(len(non_switch_pool), len(valid_switch_indices) * 5),
                               replace=False)
    print(f"  Valid switches: {len(valid_switch_indices)}, random sample: {len(sample)}")

    for col, label in zip(liq_cols, col_labels):
        vals = sec_df[col].values
        print(f"\n  Signal: liq_{label}")
        print(f"  {'Offset':>8s}  {'Switch mean':>12s}  {'Random mean':>12s}  {'Ratio':>8s}  {'p-value':>10s}")
        print(f"  {'-'*60}")

        for offset in offsets_to_test:
            sw_vals = []
            for idx in valid_switch_indices:
                if 0 <= idx + offset < len(vals):
                    v = vals[idx + offset]
                    if not np.isnan(v):
                        sw_vals.append(v)

            ns_vals = [vals[i + offset] for i in sample
                       if 0 <= i + offset < len(vals) and not np.isnan(vals[i + offset])]

            if len(sw_vals) > 20 and len(ns_vals) > 20:
                sw_arr = np.array(sw_vals)
                ns_arr = np.array(ns_vals)
                stat, pval = mannwhitneyu(sw_arr, ns_arr, alternative='greater')
                sw_mean = np.mean(sw_arr)
                ns_mean = np.mean(ns_arr)
                ratio = sw_mean / max(ns_mean, 1e-6)
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"  {offset:>+7d}s  {sw_mean:>12.2f}  {ns_mean:>12.2f}  {ratio:>7.2f}x  {pval:>9.6f}{sig}")


# ============================================================================
# ANALYSIS 6: Individual liq event frequency (not rolling)
# ============================================================================

def analysis_event_frequency(sec_df, switches, liqs):
    """Look at raw liq EVENT timestamps relative to switches."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 6: Raw Liquidation Event Frequency Around Switches")
    print(f"{'='*70}")

    # For each switch, count liq events in time bins before/after
    bins_sec = [-7200, -3600, -1800, -900, -600, -300, -120, -60, -30, -10,
                0, 10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200]

    liq_ts = liqs['ts_s'].values
    liq_not = liqs['notional'].values

    bin_counts = np.zeros(len(bins_sec) - 1, dtype=np.float64)
    bin_notionals = np.zeros(len(bins_sec) - 1, dtype=np.float64)
    n_valid = 0

    for sw in switches:
        for b in range(len(bins_sec) - 1):
            t_start = sw + bins_sec[b]
            t_end = sw + bins_sec[b + 1]
            mask = (liq_ts >= t_start) & (liq_ts < t_end)
            duration = t_end - t_start
            bin_counts[b] += mask.sum() / duration  # events per second
            bin_notionals[b] += liq_not[mask].sum() / duration  # $ per second
        n_valid += 1

    if n_valid == 0:
        return

    bin_counts /= n_valid
    bin_notionals /= n_valid

    print(f"\n  Average liquidation RATE (events/sec) in time bins around {n_valid} switches:")
    print(f"  {'Bin':>20s}  {'Duration':>10s}  {'Liq/sec':>10s}  {'$/sec':>12s}  {'Visual':>30s}")
    print(f"  {'-'*90}")

    max_rate = max(bin_counts)
    for b in range(len(bins_sec) - 1):
        t_start = bins_sec[b]
        t_end = bins_sec[b + 1]
        duration = t_end - t_start
        label = f"{t_start:+d}s→{t_end:+d}s"
        bar_len = int(bin_counts[b] / max(max_rate, 1e-10) * 25)
        bar = '█' * max(bar_len, 0)
        marker = " ← SWITCH" if t_start == 0 else ""
        print(f"  {label:>20s}  {duration:>8d}s  {bin_counts[b]:>10.4f}  "
              f"${bin_notionals[b]:>11.2f}  {bar}{marker}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()
    dates = [f"2025-05-{d:02d}" for d in range(12, 19)]

    print(f"\n{'='*70}")
    print(f"  v28: LONG-HORIZON LIQ BUILDUP — {SYMBOL}")
    print(f"  Dates: {dates[0]} to {dates[-1]} (7 days)")
    print(f"{'='*70}")
    print_mem("start")

    print(f"\n  --- Loading Liquidations ---")
    liqs = load_liquidations_all(dates)
    print_mem("after liqs")

    print(f"\n  --- Building Second-Level Timeseries ---")
    sec_df, ts_start_val = build_second_series(dates, liqs)

    print(f"\n  --- Detecting Regime Switches ---")
    switches = detect_regime_switches(sec_df)
    print(f"  Found {len(switches)} regime switches")

    if len(switches) < 10:
        switches = detect_regime_switches(sec_df, threshold=1.5)
        print(f"  Retried with threshold=1.5: {len(switches)} switches")

    print_mem("before analyses")

    # Run all analyses
    analysis_wide_profiles(sec_df, switches)
    analysis_normalized_buildup(sec_df, switches)
    analysis_rate_gradient(sec_df, switches)
    analysis_probability_curve(sec_df, switches)
    analysis_earliest_signal(sec_df, switches)
    analysis_event_frequency(sec_df, switches, liqs)

    print_mem("all done")
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
