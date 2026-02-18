#!/usr/bin/env python3
"""
v29-multi: Combined Tick-Level Analysis across 5 symbols × 89 days

Processes all data where trades + liquidations + ticker overlap:
  May 11 – Aug 7, 2025 (89 days) × BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT

RAM-safe: processes in 7-day chunks per symbol, accumulates only summary stats.

Analyses (per symbol, then cross-symbol):
  1. Combined profiles around regime switches (±30min)
  2. OI dynamics (direction, magnitude, vs random)
  3. FR dynamics (level, funding cycle timing)
  4. Cross-correlations (all pairs, lead-lag)
  5. Novel combined signals (OI+liq, FR+liq, etc.)
  6. ML prediction (feature set comparison)
  7. Cross-symbol comparison & meta-analysis
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import time
import json
import gzip
import warnings
import numpy as np
import psutil
from pathlib import Path
from collections import defaultdict
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# Date range where all 4 streams overlap (May 11 – Aug 7, 2025)
from datetime import date, timedelta

def _make_dates():
    start = date(2025, 5, 11)
    end = date(2025, 8, 7)
    dates = []
    d = start
    while d <= end:
        dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates

ALL_DATES = _make_dates()


def mem_gb():
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label=""):
    used, avail = mem_gb()
    print(f"  [RAM] used={used:.1f}GB avail={avail:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING (per-chunk)
# ============================================================================

def load_liquidations_chunk(symbol, dates):
    all_liqs = []
    for date_str in dates:
        pattern = f"liquidation_{date_str}_hr*.jsonl.gz"
        files = sorted((DATA_DIR / symbol / "bybit" / "liquidations").glob(pattern))
        for f in files:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for ev in data['result']['data']:
                                all_liqs.append((
                                    int(ev['T']),       # ts_ms
                                    ev['S'],            # side
                                    float(ev['v']),     # volume
                                    float(ev['p']),     # price
                                ))
                    except:
                        continue
    if not all_liqs:
        return np.array([]), np.array([]), np.array([])
    arr = np.array([(t, v * p, 1 if s == 'Buy' else 0)
                    for t, s, v, p in all_liqs])
    ts_s = (arr[:, 0] // 1000).astype(np.int64)
    notional = arr[:, 1]
    return ts_s, notional, arr[:, 2]


def load_ticker_chunk(symbol, dates):
    oi_ts = []; oi_val = []
    fr_ts = []; fr_val = []
    for date_str in dates:
        pattern = f"ticker_{date_str}_hr*.jsonl.gz"
        files = sorted((DATA_DIR / symbol / "bybit" / "ticker").glob(pattern))
        for f in files:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        ts_ms = d.get('ts', 0)
                        data = d.get('result', {}).get('data', {})
                        if 'openInterestValue' in data:
                            oi_ts.append(ts_ms // 1000)
                            oi_val.append(float(data['openInterestValue']))
                        if 'fundingRate' in data:
                            fr_ts.append(ts_ms // 1000)
                            fr_val.append(float(data['fundingRate']))
                    except:
                        continue
    return (np.array(oi_ts, dtype=np.int64), np.array(oi_val, dtype=np.float64),
            np.array(fr_ts, dtype=np.int64), np.array(fr_val, dtype=np.float64))


def build_second_arrays(symbol, dates, liq_ts, liq_not, oi_ts, oi_val, fr_ts, fr_val):
    """Build per-second arrays for a chunk of dates. Returns dict of arrays + ts_start."""
    # Load trades day by day, aggregate to seconds
    all_trade_count = []
    all_trade_not = []
    all_price = []
    day_ranges = []

    for date_str in dates:
        path = DATA_DIR / symbol / "bybit" / "futures" / f"{symbol}{date_str}.csv.gz"
        if not path.exists():
            continue
        import pandas as pd
        df = pd.read_csv(path, compression='gzip', usecols=['timestamp', 'price', 'size', 'side'])
        ts_s = df['timestamp'].values.astype(np.int64)
        prices = df['price'].values.astype(np.float64)
        sizes = df['size'].values.astype(np.float64)
        notionals = prices * sizes
        del df

        day_start = ts_s.min()
        day_end = ts_s.max()
        n_sec = day_end - day_start + 1
        offsets = ts_s - day_start

        tc = np.bincount(offsets, minlength=n_sec).astype(np.int32)
        tn = np.bincount(offsets, weights=notionals, minlength=n_sec)

        pl = np.full(n_sec, np.nan, dtype=np.float64)
        _, last_idx = np.unique(offsets[::-1], return_index=True)
        last_idx = len(offsets) - 1 - last_idx
        unique_offs = np.unique(offsets)
        for uo, li in zip(unique_offs, last_idx):
            pl[uo] = prices[li]

        del ts_s, prices, sizes, notionals, offsets
        day_ranges.append((day_start, n_sec))
        all_trade_count.append(tc)
        all_trade_not.append(tn)
        all_price.append(pl)
        gc.collect()

    if not day_ranges:
        return None

    ts_start = day_ranges[0][0]
    ts_end = day_ranges[-1][0] + day_ranges[-1][1] - 1
    n = ts_end - ts_start + 1

    trade_count = np.zeros(n, dtype=np.int32)
    trade_notional = np.zeros(n, dtype=np.float64)
    price_last = np.full(n, np.nan, dtype=np.float64)
    liq_count = np.zeros(n, dtype=np.int32)
    liq_notional_arr = np.zeros(n, dtype=np.float64)

    for (ds, ns), tc, tn, pl in zip(day_ranges, all_trade_count, all_trade_not, all_price):
        o = ds - ts_start
        trade_count[o:o+ns] = tc
        trade_notional[o:o+ns] = tn
        price_last[o:o+ns] = pl
    del all_trade_count, all_trade_not, all_price; gc.collect()

    # Forward-fill prices (vectorized via pandas)
    import pandas as pd
    mask = np.isnan(price_last)
    if not mask.all():
        price_last = pd.Series(price_last).ffill().values

    # Liq aggregation
    if len(liq_ts) > 0:
        l_off = (liq_ts - ts_start).astype(np.int64)
        valid = (l_off >= 0) & (l_off < n)
        l_off = l_off[valid]
        l_not = liq_not[valid]
        liq_count = np.bincount(l_off, minlength=n).astype(np.int32)
        liq_notional_arr = np.bincount(l_off, weights=l_not, minlength=n)

    # OI: forward-fill (vectorized)
    oi_arr = np.full(n, np.nan, dtype=np.float64)
    if len(oi_ts) > 0:
        oi_off = (oi_ts - ts_start).astype(np.int64)
        valid = (oi_off >= 0) & (oi_off < n)
        oi_arr[oi_off[valid]] = oi_val[valid]
        oi_arr = pd.Series(oi_arr).ffill().values

    # FR: forward-fill (vectorized)
    fr_arr = np.full(n, np.nan, dtype=np.float64)
    if len(fr_ts) > 0:
        fr_off = (fr_ts - ts_start).astype(np.int64)
        valid = (fr_off >= 0) & (fr_off < n)
        fr_arr[fr_off[valid]] = fr_val[valid]
        fr_arr = pd.Series(fr_arr).ffill().values

    # Compute features
    log_ret = np.zeros(n, dtype=np.float64)
    v = (price_last[1:] > 0) & (price_last[:-1] > 0)
    log_ret[1:][v] = np.log(price_last[1:][v] / price_last[:-1][v])

    def rolling_std(arr, w):
        cs = np.cumsum(arr); cs2 = np.cumsum(arr**2)
        r = np.full(len(arr), np.nan)
        s = cs[w:] - cs[:-w]; s2 = cs2[w:] - cs2[:-w]
        var = s2/w - (s/w)**2; np.clip(var, 0, None, out=var)
        r[w:] = np.sqrt(var)
        return r

    def rolling_sum(arr, w):
        cs = np.cumsum(arr.astype(np.float64))
        r = np.zeros(len(arr))
        r[w:] = cs[w:] - cs[:-w]
        return r

    vol_60s = rolling_std(log_ret, 60)
    liq_60s = rolling_sum(liq_count, 60)
    liq_300s = rolling_sum(liq_count, 300)
    liq_not_60s = rolling_sum(liq_notional_arr, 60)
    trade_10s = rolling_sum(trade_count, 10)

    oi_delta_60s = np.zeros(n); oi_delta_300s = np.zeros(n)
    if not np.all(np.isnan(oi_arr)):
        oi_delta_60s[60:] = oi_arr[60:] - oi_arr[:-60]
        oi_delta_300s[300:] = oi_arr[300:] - oi_arr[:-300]

    # FR time to funding (vectorized)
    seconds = np.arange(ts_start, ts_end + 1)
    tod = seconds % 86400
    # Next funding at 0, 28800, 57600 UTC. Compute distance to next.
    fr_ttf = np.minimum(
        np.minimum(28800 - tod, 57600 - tod),
        86400 - tod
    )
    fr_ttf = np.where(tod < 28800, 28800 - tod,
              np.where(tod < 57600, 57600 - tod, 86400 - tod))

    return {
        'ts_start': ts_start, 'n': n,
        'vol_60s': vol_60s, 'liq_60s': liq_60s, 'liq_300s': liq_300s,
        'liq_not_60s': liq_not_60s, 'trade_10s': trade_10s,
        'liq_count': liq_count, 'liq_notional': liq_notional_arr,
        'oi': oi_arr, 'oi_delta_60s': oi_delta_60s, 'oi_delta_300s': oi_delta_300s,
        'fr': fr_arr, 'fr_ttf': fr_ttf,
        'price': price_last, 'trade_count': trade_count,
    }


def detect_switches(vol, ts_start, median_window=3600, threshold=2.0, min_gap=1800):
    import pandas as pd
    n = len(vol)
    # Pandas rolling median is C-optimized, much faster than Python loop
    vol_median = pd.Series(vol).rolling(median_window, min_periods=median_window//4).median().values

    # Vectorized switch detection
    valid = ~np.isnan(vol) & ~np.isnan(vol_median)
    above = valid & (vol > vol_median * threshold)
    below = valid & (vol < vol_median * 1.2)

    switches = []
    in_volatile = False
    last_sw = -999999
    for i in range(median_window + 1, n):
        if not valid[i]:
            continue
        if not in_volatile and above[i]:
            if (ts_start + i) - last_sw > min_gap:
                switches.append(ts_start + i)
                last_sw = ts_start + i
            in_volatile = True
        elif in_volatile and below[i]:
            in_volatile = False
    return switches


# ============================================================================
# PER-SYMBOL ANALYSIS: collect summary stats from chunks
# ============================================================================

def process_symbol(symbol):
    """Process one symbol across all dates in 7-day chunks. Returns summary dict."""
    print(f"\n{'#'*80}")
    print(f"  PROCESSING: {symbol}")
    print(f"{'#'*80}")
    t0 = time.time()

    # Split dates into 7-day chunks
    chunk_size = 7
    date_chunks = [ALL_DATES[i:i+chunk_size] for i in range(0, len(ALL_DATES), chunk_size)]

    # Accumulators
    all_switches = []
    # Profile accumulators (±1800s = 3600 bins)
    window = 1800
    profile_cols = ['vol_60s', 'liq_60s', 'liq_not_60s', 'trade_10s',
                    'oi_delta_60s', 'oi_delta_300s', 'fr', 'fr_ttf']
    profile_sums = {c: np.zeros(2 * window) for c in profile_cols}
    profile_count = 0

    # OI dynamics accumulators
    oi_deltas_before = []  # OI_delta_300s at switch
    oi_deltas_at = []      # OI_delta_60s at switch
    oi_price_dir = []      # (oi_rising, price_up)

    # FR accumulators
    switch_fr = []
    switch_ttf = []
    random_fr = []
    random_ttf = []

    # Cross-correlation accumulators (sampled at 10s)
    xcorr_signals = {k: [] for k in ['vol_60s', 'liq_60s', 'oi_delta_60s', 'fr', 'trade_10s']}

    # Novel signal accumulators
    novel_fires = defaultdict(int)
    novel_hits = defaultdict(lambda: defaultdict(int))

    # ML accumulators: per-chunk AUC scores (no concatenation)
    ml_chunk_aucs = defaultdict(list)  # {feature_set_name: [auc_chunk1, auc_chunk2, ...]}
    ml_chunk_importances = []  # list of dicts per chunk
    ml_chunk_base_rates = []
    ml_chunk_sizes = []

    total_seconds = 0
    total_trades = 0
    total_liqs = 0

    for ci, chunk_dates in enumerate(date_chunks):
        ct0 = time.time()
        print(f"\n  Chunk {ci+1}/{len(date_chunks)}: {chunk_dates[0]} to {chunk_dates[-1]} ({len(chunk_dates)} days)", flush=True)

        # Load data
        liq_ts, liq_not, liq_buy = load_liquidations_chunk(symbol, chunk_dates)
        oi_ts, oi_val, fr_ts_arr, fr_val_arr = load_ticker_chunk(symbol, chunk_dates)

        total_liqs += len(liq_ts)

        # Build second arrays
        arrays = build_second_arrays(symbol, chunk_dates, liq_ts, liq_not,
                                     oi_ts, oi_val, fr_ts_arr, fr_val_arr)
        del liq_ts, liq_not, liq_buy, oi_ts, oi_val, fr_ts_arr, fr_val_arr
        gc.collect()

        if arrays is None:
            print(f"    No trade data, skipping")
            continue

        n = arrays['n']
        ts_start = arrays['ts_start']
        total_seconds += n
        total_trades += int(arrays['trade_count'].sum())

        # Detect switches in this chunk
        switches = detect_switches(arrays['vol_60s'], ts_start)
        all_switches.extend(switches)
        print(f"    {n:,}s, {int(arrays['trade_count'].sum()):,} trades, "
              f"{int(arrays['liq_count'].sum()):,} liqs, {len(switches)} switches "
              f"({time.time()-ct0:.1f}s)", flush=True)

        # --- Accumulate profiles ---
        for sw in switches:
            idx = sw - ts_start
            if idx < window or idx + window >= n:
                continue
            for c in profile_cols:
                profile_sums[c] += arrays[c][idx-window:idx+window]
            profile_count += 1

            # OI dynamics
            oi_deltas_before.append(arrays['oi_delta_300s'][idx])
            oi_deltas_at.append(arrays['oi_delta_60s'][idx])

            # Price direction
            if idx >= 30 and idx + 30 < n:
                price_change = arrays['price'][idx+30] - arrays['price'][idx-30]
                oi_d = arrays['oi_delta_300s'][idx]
                if not np.isnan(oi_d) and not np.isnan(price_change):
                    oi_price_dir.append((oi_d > 0, price_change > 0))

            # FR at switch
            fr_val_sw = arrays['fr'][idx]
            if not np.isnan(fr_val_sw):
                switch_fr.append(fr_val_sw)
                switch_ttf.append(arrays['fr_ttf'][idx])

        # --- Random samples for FR comparison ---
        np.random.seed(42 + ci)
        excl = np.zeros(n, dtype=bool)
        for sw in switches:
            idx = sw - ts_start
            lo = max(0, idx - 1800); hi = min(n, idx + 1801)
            excl[lo:hi] = True
        pool = np.arange(window, n - window)
        pool = pool[~excl[pool]]
        if len(pool) > 500:
            samp = np.random.choice(pool, size=500, replace=False)
            for si in samp:
                fv = arrays['fr'][si]
                if not np.isnan(fv):
                    random_fr.append(fv)
                    random_ttf.append(arrays['fr_ttf'][si])

        # --- Cross-correlation samples (every 10s) ---
        step = 10
        indices = np.arange(0, n, step)
        for k in xcorr_signals:
            xcorr_signals[k].append(arrays[k][indices])

        # --- Novel signals (test on last 30% of chunk) ---
        split = int(n * 0.7)
        vol = arrays['vol_60s']
        liq = arrays['liq_60s']
        oi_d = arrays['oi_delta_300s']
        fr_a = arrays['fr']

        # Compute thresholds from training portion
        liq_pos = liq[:split][liq[:split] > 0]
        liq_p90 = np.percentile(liq_pos, 90) if len(liq_pos) > 100 else 1
        liq_p95 = np.percentile(liq_pos, 95) if len(liq_pos) > 100 else 1
        oi_valid = oi_d[:split][np.isfinite(oi_d[:split])]
        oi_p10 = np.percentile(oi_valid, 10) if len(oi_valid) > 100 else -1e9

        # Switch proximity for test portion
        sw_within_60 = np.zeros(n, dtype=bool)
        sw_within_300 = np.zeros(n, dtype=bool)
        for sw in switches:
            idx = sw - ts_start
            if idx is not None:
                sw_within_60[max(0, idx-60):idx+1] = True
                sw_within_300[max(0, idx-300):idx+1] = True

        test_range = np.arange(split, n - 300)
        for i in test_range[::10]:  # sample every 10s for speed
            if np.isnan(vol[i]):
                continue
            # Liq spike only
            if liq[i] >= liq_p95:
                novel_fires['liq_spike'] += 1
                if sw_within_60[i]: novel_hits['liq_spike'][60] += 1
                if sw_within_300[i]: novel_hits['liq_spike'][300] += 1
            # OI drop only
            if np.isfinite(oi_d[i]) and oi_d[i] < oi_p10:
                novel_fires['oi_drop'] += 1
                if sw_within_60[i]: novel_hits['oi_drop'][60] += 1
                if sw_within_300[i]: novel_hits['oi_drop'][300] += 1
            # OI drop + liq spike
            if np.isfinite(oi_d[i]) and oi_d[i] < oi_p10 and liq[i] >= liq_p90:
                novel_fires['oi_drop+liq'] += 1
                if sw_within_60[i]: novel_hits['oi_drop+liq'][60] += 1
                if sw_within_300[i]: novel_hits['oi_drop+liq'][300] += 1

        # --- ML: per-chunk train/test (no accumulation) ---
        feat_cols = ['vol_60s', 'liq_60s', 'liq_300s', 'liq_not_60s',
                     'oi_delta_60s', 'oi_delta_300s', 'fr', 'fr_ttf', 'trade_10s']
        target_300 = sw_within_300

        ml_indices = np.arange(3600, n - 300, 10)
        ml_split = int(len(ml_indices) * 0.7)

        X_chunk = np.column_stack([arrays[c][ml_indices] for c in feat_cols])
        y_chunk = target_300[ml_indices].astype(np.int8)

        valid_ml = np.all(np.isfinite(X_chunk), axis=1)
        X_chunk = X_chunk[valid_ml]; y_chunk = y_chunk[valid_ml]

        if len(X_chunk) > 500 and y_chunk[:ml_split].sum() > 5 and y_chunk[ml_split:].sum() > 5:
            chunk_ml = run_ml_chunk(X_chunk[:ml_split], y_chunk[:ml_split],
                                    X_chunk[ml_split:], y_chunk[ml_split:], feat_cols)
            for fs_name, auc_val in chunk_ml['aucs'].items():
                ml_chunk_aucs[fs_name].append(auc_val)
            if 'importances' in chunk_ml:
                ml_chunk_importances.append(chunk_ml['importances'])
            ml_chunk_base_rates.append(float(y_chunk[ml_split:].mean()))
            ml_chunk_sizes.append(len(y_chunk))

        del arrays; gc.collect()

    elapsed = time.time() - t0
    print(f"\n  {symbol} complete: {total_seconds:,}s, {total_trades:,} trades, "
          f"{total_liqs:,} liqs, {len(all_switches)} switches in {elapsed:.0f}s")
    print_mem(f"{symbol} done")

    # Compile results
    results = {
        'symbol': symbol,
        'total_seconds': total_seconds,
        'total_trades': total_trades,
        'total_liqs': total_liqs,
        'n_switches': len(all_switches),
        'profile_count': profile_count,
    }

    # --- Compute final statistics ---

    # 1. Profiles
    if profile_count > 0:
        results['profiles'] = {c: profile_sums[c] / profile_count for c in profile_cols}
    else:
        results['profiles'] = None

    # 2. OI dynamics
    oi_before = np.array(oi_deltas_before)
    oi_at = np.array(oi_deltas_at)
    valid_b = np.isfinite(oi_before)
    valid_a = np.isfinite(oi_at)
    results['oi'] = {
        'mean_before': np.mean(oi_before[valid_b]) if valid_b.sum() > 0 else np.nan,
        'median_before': np.median(oi_before[valid_b]) if valid_b.sum() > 0 else np.nan,
        'mean_at': np.mean(oi_at[valid_a]) if valid_a.sum() > 0 else np.nan,
        'n_rising': sum(1 for d in oi_before[valid_b] if d > 1e6),
        'n_falling': sum(1 for d in oi_before[valid_b] if d < -1e6),
        'n_flat': sum(1 for d in oi_before[valid_b] if abs(d) <= 1e6),
    }

    # OI direction vs price
    if oi_price_dir:
        oi_up_p_up = sum(1 for o, p in oi_price_dir if o and p)
        oi_up_p_dn = sum(1 for o, p in oi_price_dir if o and not p)
        oi_dn_p_up = sum(1 for o, p in oi_price_dir if not o and p)
        oi_dn_p_dn = sum(1 for o, p in oi_price_dir if not o and not p)
        results['oi_price'] = (oi_up_p_up, oi_up_p_dn, oi_dn_p_up, oi_dn_p_dn)
    else:
        results['oi_price'] = None

    # 3. FR
    switch_fr_arr = np.array(switch_fr)
    random_fr_arr = np.array(random_fr)
    results['fr'] = {
        'switch_mean': np.mean(switch_fr_arr) if len(switch_fr_arr) > 0 else np.nan,
        'random_mean': np.mean(random_fr_arr) if len(random_fr_arr) > 0 else np.nan,
        'switch_ttf': switch_ttf,
        'random_ttf': random_ttf,
    }
    if len(switch_fr_arr) > 20 and len(random_fr_arr) > 20:
        _, pval = mannwhitneyu(np.abs(switch_fr_arr), np.abs(random_fr_arr), alternative='greater')
        results['fr']['pval'] = pval
    else:
        results['fr']['pval'] = np.nan

    # 4. Cross-correlations
    xcorr_combined = {}
    for k in xcorr_signals:
        if xcorr_signals[k]:
            xcorr_combined[k] = np.concatenate(xcorr_signals[k])
    results['xcorr'] = xcorr_combined

    # 5. Novel signals
    results['novel'] = {'fires': dict(novel_fires), 'hits': {k: dict(v) for k, v in novel_hits.items()}}

    # 6. ML — aggregate per-chunk results
    if ml_chunk_aucs:
        ml_result = {}
        for fs_name, aucs in ml_chunk_aucs.items():
            ml_result[fs_name] = {
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs)),
                'auc_min': float(np.min(aucs)),
                'auc_max': float(np.max(aucs)),
                'n_chunks': len(aucs),
            }
        if ml_chunk_importances:
            avg_imp = {}
            for k in ml_chunk_importances[0]:
                avg_imp[k] = float(np.mean([d[k] for d in ml_chunk_importances]))
            ml_result['importances'] = avg_imp
        ml_result['base_rate'] = float(np.mean(ml_chunk_base_rates))
        ml_result['total_samples'] = sum(ml_chunk_sizes)
        results['ml'] = ml_result
    else:
        results['ml'] = None

    gc.collect()
    return results


def run_ml_chunk(X_train, y_train, X_test, y_test, feat_cols):
    """Run ML on one chunk (~40K samples). Fast: uses small GBM."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    subsets = {
        'vol_only': [0],
        'vol+liq': [0, 1, 2],
        'vol+liq+OI': [0, 1, 2, 4, 5],
        'vol+liq+OI+FR': [0, 1, 2, 4, 5, 6, 7],
        'all': list(range(len(feat_cols))),
    }

    result = {'aucs': {}}
    for name, cols in subsets.items():
        Xtr = X_train[:, cols]; Xte = X_test[:, cols]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(Xtr_s, y_train)
        y_prob = model.predict_proba(Xte_s)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5
        result['aucs'][name] = auc

        if name == 'all':
            result['importances'] = dict(zip(
                [feat_cols[c] for c in cols],
                model.feature_importances_.tolist()
            ))

    return result


# ============================================================================
# REPORTING
# ============================================================================

def print_symbol_results(r):
    """Print results for one symbol."""
    sym = r['symbol']
    print(f"\n{'='*80}")
    print(f"  {sym}: {r['total_seconds']:,}s ({r['total_seconds']/86400:.0f} days), "
          f"{r['total_trades']:,} trades, {r['total_liqs']:,} liqs, "
          f"{r['n_switches']} switches ({r['profile_count']} profiled)")
    print(f"{'='*80}")

    # Profiles (key offsets only)
    if r['profiles'] is not None:
        w = 1800
        print(f"\n  --- Profiles (±30min, {r['profile_count']} switches) ---")
        print(f"  {'Offset':>8s}  {'Vol_60s':>10s}  {'Liq_60s':>8s}  {'LiqNot$':>10s}  "
              f"{'OI_Δ60s$':>12s}  {'OI_Δ5m$':>12s}  {'FR':>10s}")
        print(f"  {'-'*75}")

        offsets = [-1800, -900, -600, -300, -120, -60, -30, 0, 30, 60, 120, 300, 600, 900, 1800]
        for off in offsets:
            i = w + off
            if 0 <= i < 2 * w:
                p = r['profiles']
                marker = " ← SW" if off == 0 else ""
                fr_s = f"{p['fr'][i]:.8f}" if not np.isnan(p['fr'][i]) else "nan"
                print(f"  {off:>+7d}s  {p['vol_60s'][i]:>10.6f}  {p['liq_60s'][i]:>8.1f}  "
                      f"${p['liq_not_60s'][i]:>9,.0f}  ${p['oi_delta_60s'][i]:>11,.0f}  "
                      f"${p['oi_delta_300s'][i]:>11,.0f}  {fr_s}{marker}")

    # OI dynamics
    oi = r['oi']
    total_oi = oi['n_rising'] + oi['n_falling'] + oi['n_flat']
    if total_oi > 0:
        print(f"\n  --- OI Dynamics ---")
        print(f"  OI_Δ5min before: mean=${oi['mean_before']:+,.0f}, median=${oi['median_before']:+,.0f}")
        print(f"  OI_Δ60s at switch: mean=${oi['mean_at']:+,.0f}")
        print(f"  Rising: {oi['n_rising']} ({oi['n_rising']/total_oi*100:.0f}%), "
              f"Falling: {oi['n_falling']} ({oi['n_falling']/total_oi*100:.0f}%), "
              f"Flat: {oi['n_flat']} ({oi['n_flat']/total_oi*100:.0f}%)")

    if r['oi_price']:
        uu, ud, du, dd = r['oi_price']
        total = uu + ud + du + dd
        if total > 0:
            print(f"  OI↑+Price↑: {uu} ({uu/total*100:.0f}%), OI↑+Price↓: {ud} ({ud/total*100:.0f}%), "
                  f"OI↓+Price↑: {du} ({du/total*100:.0f}%), OI↓+Price↓: {dd} ({dd/total*100:.0f}%)")

    # FR
    fr = r['fr']
    print(f"\n  --- Funding Rate ---")
    print(f"  FR at switch: {fr['switch_mean']:.8f}, at random: {fr['random_mean']:.8f}, "
          f"|FR| p={fr['pval']:.4f} {'***' if fr['pval'] < 0.001 else '**' if fr['pval'] < 0.01 else '*' if fr['pval'] < 0.05 else 'ns'}")

    # Funding cycle timing
    if fr['switch_ttf'] and fr['random_ttf']:
        bins = [0, 1800, 3600, 7200, 14400, 28800]
        labels = ['0-30m', '30m-1h', '1h-2h', '2h-4h', '4h-8h']
        sw_h = np.histogram(fr['switch_ttf'], bins=bins)[0]
        rnd_h = np.histogram(fr['random_ttf'], bins=bins)[0]
        print(f"  {'ToFunding':>10s}  {'Switch%':>8s}  {'Random%':>8s}  {'Lift':>6s}")
        for i, lab in enumerate(labels):
            sp = sw_h[i] / max(len(fr['switch_ttf']), 1) * 100
            rp = rnd_h[i] / max(len(fr['random_ttf']), 1) * 100
            lift = sp / max(rp, 0.01)
            print(f"  {lab:>10s}  {sp:>7.1f}%  {rp:>7.1f}%  {lift:>5.2f}x")

    # Novel signals
    if r['novel']['fires']:
        print(f"\n  --- Novel Signals ---")
        print(f"  {'Signal':>20s}  {'Fires':>8s}  {'P(sw 1m)':>10s}  {'Lift':>6s}  {'P(sw 5m)':>10s}  {'Lift':>6s}")
        for sig in ['liq_spike', 'oi_drop', 'oi_drop+liq']:
            fires = r['novel']['fires'].get(sig, 0)
            if fires < 10:
                continue
            h60 = r['novel']['hits'].get(sig, {}).get(60, 0)
            h300 = r['novel']['hits'].get(sig, {}).get(300, 0)
            p60 = h60 / fires; p300 = h300 / fires
            # Approximate base rate from ML results
            br = r['ml']['base_rate'] if r['ml'] else 0.1
            l60 = p60 / max(br * 60/300, 0.001)
            l300 = p300 / max(br, 0.001)
            print(f"  {sig:>20s}  {fires:>8,}  {p60:>10.4f}  {l60:>5.1f}x  {p300:>10.4f}  {l300:>5.1f}x")

    # ML
    if r['ml']:
        ml = r['ml']
        print(f"\n  --- ML Prediction (per-chunk walk-forward, switch within 5min) ---")
        print(f"  Total samples: {ml['total_samples']:,}, Base rate: {ml['base_rate']:.4f}")
        print(f"  {'Feature Set':>20s}  {'AUC mean':>10s}  {'±std':>8s}  {'min':>8s}  {'max':>8s}  {'chunks':>8s}")
        for name in ['vol_only', 'vol+liq', 'vol+liq+OI', 'vol+liq+OI+FR', 'all']:
            if name in ml:
                d = ml[name]
                print(f"  {name:>20s}  {d['auc_mean']:>10.4f}  {d['auc_std']:>8.4f}  "
                      f"{d['auc_min']:>8.4f}  {d['auc_max']:>8.4f}  {d['n_chunks']:>8d}")
        if 'importances' in ml:
            print(f"\n  Feature Importance (avg across chunks):")
            for fname, imp in sorted(ml['importances'].items(), key=lambda x: -x[1]):
                bar = '█' * int(imp * 40)
                print(f"    {fname:>20s}  {imp:.4f}  {bar}")


def print_cross_symbol(all_results):
    """Cross-symbol comparison."""
    print(f"\n\n{'#'*80}")
    print(f"  CROSS-SYMBOL COMPARISON")
    print(f"{'#'*80}")

    # Summary table
    print(f"\n  {'Symbol':>10s}  {'Days':>6s}  {'Trades':>12s}  {'Liqs':>8s}  {'Switches':>10s}  "
          f"{'Sw/day':>8s}  {'OI_Δ60s@sw':>14s}  {'FR p-val':>10s}")
    print(f"  {'-'*85}")
    for r in all_results:
        days = r['total_seconds'] / 86400
        sw_day = r['n_switches'] / max(days, 1)
        oi_at = r['oi']['mean_at']
        fr_p = r['fr']['pval']
        print(f"  {r['symbol']:>10s}  {days:>6.0f}  {r['total_trades']:>12,}  {r['total_liqs']:>8,}  "
              f"{r['n_switches']:>10d}  {sw_day:>8.1f}  ${oi_at:>13,.0f}  {fr_p:>10.4f}")

    # ML comparison (mean AUC ± std across chunks)
    print(f"\n  --- ML AUC Comparison (mean ± std across weekly chunks) ---")
    print(f"  {'Symbol':>10s}  {'vol':>12s}  {'v+l':>12s}  {'v+l+OI':>12s}  {'v+l+OI+FR':>12s}  {'all':>12s}  {'base':>6s}  {'#chk':>5s}")
    print(f"  {'-'*90}")
    for r in all_results:
        if r['ml']:
            ml = r['ml']
            parts = [f"  {r['symbol']:>10s}"]
            for name in ['vol_only', 'vol+liq', 'vol+liq+OI', 'vol+liq+OI+FR', 'all']:
                d = ml.get(name, {})
                if d:
                    parts.append(f"{d['auc_mean']:.3f}±{d['auc_std']:.3f}")
                else:
                    parts.append("    n/a")
            parts.append(f"{ml['base_rate']:.3f}")
            n_chk = ml.get('vol_only', {}).get('n_chunks', 0)
            parts.append(f"{n_chk:>3d}")
            print("  ".join(parts))

    # OI dynamics comparison
    print(f"\n  --- OI Dynamics Comparison ---")
    print(f"  {'Symbol':>10s}  {'OI_Δ5m mean':>14s}  {'OI_Δ60s@sw':>14s}  "
          f"{'Rising%':>8s}  {'Falling%':>8s}  {'Flat%':>8s}")
    print(f"  {'-'*65}")
    for r in all_results:
        oi = r['oi']
        total = oi['n_rising'] + oi['n_falling'] + oi['n_flat']
        if total > 0:
            print(f"  {r['symbol']:>10s}  ${oi['mean_before']:>13,.0f}  ${oi['mean_at']:>13,.0f}  "
                  f"{oi['n_rising']/total*100:>7.0f}%  {oi['n_falling']/total*100:>7.0f}%  "
                  f"{oi['n_flat']/total*100:>7.0f}%")

    # Funding cycle comparison
    print(f"\n  --- Funding Cycle Timing (Lift: switch vs random) ---")
    bins = [0, 1800, 3600, 7200, 14400, 28800]
    labels = ['0-30m', '30m-1h', '1h-2h', '2h-4h', '4h-8h']
    print(f"  {'Symbol':>10s}  " + "  ".join(f"{l:>8s}" for l in labels))
    print(f"  {'-'*60}")
    for r in all_results:
        fr = r['fr']
        if fr['switch_ttf'] and fr['random_ttf']:
            sw_h = np.histogram(fr['switch_ttf'], bins=bins)[0]
            rnd_h = np.histogram(fr['random_ttf'], bins=bins)[0]
            lifts = []
            for i in range(len(labels)):
                sp = sw_h[i] / max(len(fr['switch_ttf']), 1) * 100
                rp = rnd_h[i] / max(len(fr['random_ttf']), 1) * 100
                lifts.append(sp / max(rp, 0.01))
            print(f"  {r['symbol']:>10s}  " + "  ".join(f"{l:>7.2f}x" for l in lifts))

    # Cross-correlation summary
    print(f"\n  --- Key Cross-Correlations (Spearman, contemporaneous) ---")
    print(f"  {'Symbol':>10s}  {'vol↔liq':>10s}  {'vol↔OI_Δ':>10s}  {'vol↔FR':>10s}  "
          f"{'liq↔OI_Δ':>10s}  {'liq↔FR':>10s}")
    print(f"  {'-'*65}")
    for r in all_results:
        xc = r['xcorr']
        if not xc:
            continue
        # Concatenate all chunks
        sigs = {}
        for k in ['vol_60s', 'liq_60s', 'oi_delta_60s', 'fr', 'trade_10s']:
            if k in xc:
                sigs[k] = xc[k]
        valid = np.ones(len(sigs.get('vol_60s', [])), dtype=bool)
        for s in sigs.values():
            valid &= ~np.isnan(s) & np.isfinite(s)

        if valid.sum() < 100:
            continue

        pairs = [
            ('vol_60s', 'liq_60s'),
            ('vol_60s', 'oi_delta_60s'),
            ('vol_60s', 'fr'),
            ('liq_60s', 'oi_delta_60s'),
            ('liq_60s', 'fr'),
        ]
        corrs = []
        for a, b in pairs:
            rho, _ = spearmanr(sigs[a][valid], sigs[b][valid])
            corrs.append(rho)
        print(f"  {r['symbol']:>10s}  " + "  ".join(f"{c:>+10.4f}" for c in corrs))

    # Feature importance comparison
    print(f"\n  --- Feature Importance (all_features model) ---")
    feat_names = ['vol_60s', 'liq_60s', 'liq_300s', 'liq_not_60s',
                  'oi_delta_60s', 'oi_delta_300s', 'fr', 'fr_ttf', 'trade_10s']
    print(f"  {'Feature':>20s}  " + "  ".join(f"{r['symbol']:>10s}" for r in all_results))
    print(f"  {'-'*(20 + 12 * len(all_results))}")
    for fn in feat_names:
        vals = []
        for r in all_results:
            if r['ml'] and 'importances' in r['ml']:
                vals.append(r['ml']['importances'].get(fn, 0))
            else:
                vals.append(0)
        print(f"  {fn:>20s}  " + "  ".join(f"{v:>10.4f}" for v in vals))


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    print(f"\n{'#'*80}")
    print(f"  v29-MULTI: COMBINED TICK-LEVEL ANALYSIS")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Dates: {ALL_DATES[0]} to {ALL_DATES[-1]} ({len(ALL_DATES)} days)")
    print(f"  Streams: Trades + Liquidations + Open Interest + Funding Rate")
    print(f"{'#'*80}")
    print_mem("start")

    all_results = []
    for symbol in SYMBOLS:
        r = process_symbol(symbol)
        print_symbol_results(r)
        all_results.append(r)
        gc.collect()

    print_cross_symbol(all_results)

    elapsed = time.time() - t_start
    print(f"\n{'#'*80}")
    print(f"  ALL COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
