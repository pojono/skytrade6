#!/usr/bin/env python3
"""
v30: Tradeable Prediction Targets

v29-rich found that regime switches are predictable (AUC 0.60) but the target
("switch within 5min") isn't directly tradeable. v30 tests 5 alternative targets
that map to real trading strategies.

EMPIRICAL FACTS (from v29 analysis):
  - Switches are NOT directional: 52/48 up/down split
  - Pre-switch momentum does NOT predict direction (rho=0.08, p=0.15)
  - But switches ARE high-range: mean 17.7 bps range in 5 min
  - Moderate continuation: 62.6% (30s→300s same direction)
  - |ret_300s| mean=9.0 bps, P90=20.9 bps

TARGETS:
  T1: Vol expansion — range in next 5min > median (straddle strategy)
  T2: Big absolute move — |ret_300s| > P75 (selective straddle)
  T3: Continuation — given 30s move started, does it continue? (momentum)
  T4: Straight-line move — sign consistency > 0.75 (trend-follow)
  T5: Liq-side direction — predict direction from liq buy/sell ratio

FEATURES: Reuse v29-rich top features at 900s/3600s horizons + liq side features.

DATA: BTCUSDT, May 11-24 2025 (14 days), week 1 train / week 2 test.
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
from collections import defaultdict

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
SYMBOL = "ETHUSDT"
DATES = [f"2025-05-{d:02d}" for d in range(11, 18)]  # ETH week 1
HORIZONS = [60, 300, 900, 3600]  # Skip 10s/30s — they're noise


def mem_gb():
    return psutil.virtual_memory().used / 1e9, psutil.virtual_memory().available / 1e9

def print_mem(label=""):
    u, a = mem_gb()
    print(f"  [RAM] used={u:.1f}GB avail={a:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING (reuse from v29-rich)
# ============================================================================

def load_trades_day(date_str):
    path = DATA_DIR / SYMBOL / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, compression='gzip',
                     usecols=['timestamp', 'price', 'size', 'side', 'tickDirection'])
    ts_s = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float32)
    sizes = df['size'].values.astype(np.float32)
    notionals = prices * sizes
    is_buy = (df['side'] == 'Buy').values.astype(np.float32)
    is_plus_tick = df['tickDirection'].isin(['PlusTick', 'ZeroPlusTick']).values.astype(np.float32)
    del df; gc.collect()

    day_start = int(ts_s.min()); day_end = int(ts_s.max())
    n = day_end - day_start + 1
    off = (ts_s - day_start).astype(np.int32)
    del ts_s

    trade_count = np.bincount(off, minlength=n).astype(np.float32)
    trade_notional = np.bincount(off, weights=notionals, minlength=n).astype(np.float32)
    buy_notional = np.bincount(off, weights=notionals * is_buy, minlength=n).astype(np.float32)
    buy_count = np.bincount(off, weights=is_buy, minlength=n).astype(np.float32)
    plus_ticks = np.bincount(off, weights=is_plus_tick, minlength=n).astype(np.float32)
    p90_not = np.percentile(notionals, 90)
    large_count = np.bincount(off, weights=(notionals > p90_not).astype(np.float32), minlength=n).astype(np.float32)
    vwap_num = np.bincount(off, weights=prices * sizes, minlength=n).astype(np.float32)
    vwap_den = np.bincount(off, weights=sizes, minlength=n).astype(np.float32)

    price_last = np.full(n, np.nan, dtype=np.float32)
    _, last_idx = np.unique(off[::-1], return_index=True)
    last_idx = len(off) - 1 - last_idx
    for uo, li in zip(np.unique(off), last_idx):
        price_last[uo] = prices[li]

    del prices, sizes, notionals, is_buy, is_plus_tick, off; gc.collect()
    return {
        'day_start': day_start, 'n': n,
        'trade_count': trade_count, 'trade_notional': trade_notional,
        'buy_notional': buy_notional, 'buy_count': buy_count,
        'plus_ticks': plus_ticks, 'large_count': large_count,
        'vwap_num': vwap_num, 'vwap_den': vwap_den,
        'price_last': price_last,
    }


def load_liquidations(dates):
    records = []
    for date_str in dates:
        for f in sorted((DATA_DIR / SYMBOL / "bybit" / "liquidations").glob(
                f"liquidation_{date_str}_hr*.jsonl.gz")):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        if 'result' in d and 'data' in d['result']:
                            for ev in d['result']['data']:
                                records.append((
                                    int(ev['T']) // 1000,
                                    1.0 if ev['S'] == 'Buy' else 0.0,
                                    float(ev['v']) * float(ev['p']),
                                ))
                    except:
                        continue
    if not records:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(records)
    return arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2]


def load_ticker_into_arrays(dates, ts_start, n):
    global _ticker_t0
    _ticker_t0 = time.time()
    field_map = {
        'openInterestValue': 'oi', 'fundingRate': 'fr',
        'bid1Price': 'bid', 'bid1Size': 'bid_sz',
        'ask1Price': 'ask', 'ask1Size': 'ask_sz',
        'markPrice': 'mark', 'indexPrice': 'index',
    }
    arrays = {v: np.full(n, np.nan, dtype=np.float32) for v in field_map.values()}
    count = 0
    for di, date_str in enumerate(dates):
        files = sorted((DATA_DIR / SYMBOL / "bybit" / "ticker").glob(
                f"ticker_{date_str}_hr*.jsonl.gz"))
        day_count = 0
        for f in files:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        ts_s = d.get('ts', 0) // 1000
                        idx = ts_s - ts_start
                        if idx < 0 or idx >= n:
                            continue
                        data = d.get('result', {}).get('data', {})
                        for src, dst in field_map.items():
                            if src in data:
                                arrays[dst][idx] = float(data[src])
                        day_count += 1
                    except:
                        continue
            del d, data, line  # help gc
        gc.collect()
        count += day_count
        u, a = mem_gb()
        eta = (time.time() - _ticker_t0) / (di + 1) * (len(dates) - di - 1)
        print(f"    {date_str}: {day_count:,} updates (RAM={u:.1f}/{a:.1f}GB, ETA={eta:.0f}s)", flush=True)
    for k in arrays:
        arrays[k] = pd.Series(arrays[k]).ffill().values
    print(f"  Ticker: {count:,} total updates", flush=True)
    return arrays


# ============================================================================
# BUILD ARRAYS & FEATURES
# ============================================================================

def build_all(dates):
    t0 = time.time()

    # Trades
    print("  Loading trades...", flush=True)
    day_results = []
    for date_str in dates:
        dr = load_trades_day(date_str)
        if dr:
            day_results.append(dr)
            print(f"    {date_str}: {int(dr['trade_count'].sum()):,} trades", flush=True)
        gc.collect()

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n'] - 1
    n = ts_end - ts_start + 1

    raw_keys = ['trade_count', 'trade_notional', 'buy_notional', 'buy_count',
                'plus_ticks', 'large_count', 'vwap_num', 'vwap_den']
    raw = {k: np.zeros(n, dtype=np.float32) for k in raw_keys}
    raw['price_last'] = np.full(n, np.nan, dtype=np.float32)

    for dr in day_results:
        o = dr['day_start'] - ts_start
        for k in raw_keys:
            raw[k][o:o+dr['n']] = dr[k]
        raw['price_last'][o:o+dr['n']] = dr['price_last']
    del day_results; gc.collect()
    raw['price_last'] = pd.Series(raw['price_last']).ffill().values

    # Liquidations
    print("  Loading liquidations...", flush=True)
    liq_ts, liq_is_buy, liq_not = load_liquidations(dates)
    raw['liq_count'] = np.zeros(n, dtype=np.float32)
    raw['liq_notional'] = np.zeros(n, dtype=np.float32)
    raw['liq_buy_count'] = np.zeros(n, dtype=np.float32)
    raw['liq_buy_notional'] = np.zeros(n, dtype=np.float32)
    if len(liq_ts) > 0:
        l_off = (liq_ts - ts_start).astype(np.int64)
        valid = (l_off >= 0) & (l_off < n)
        l_off_v = l_off[valid]
        raw['liq_count'] = np.bincount(l_off_v, minlength=n).astype(np.float32)
        raw['liq_notional'] = np.bincount(l_off_v, weights=liq_not[valid], minlength=n).astype(np.float32)
        raw['liq_buy_count'] = np.bincount(l_off_v, weights=liq_is_buy[valid], minlength=n).astype(np.float32)
        raw['liq_buy_notional'] = np.bincount(l_off_v, weights=(liq_not * liq_is_buy)[valid], minlength=n).astype(np.float32)
    del liq_ts, liq_is_buy, liq_not; gc.collect()
    print_mem('after liqs')

    # Ticker
    print("  Loading ticker...", flush=True)
    print_mem("before ticker")
    ticker = load_ticker_into_arrays(dates, ts_start, n)
    raw.update(ticker)
    del ticker; gc.collect()

    print(f"  Built {n:,} seconds ({time.time()-t0:.0f}s)", flush=True)
    print_mem("raw done")
    return raw, ts_start, n


def compute_features(raw, ts_start, n):
    t0 = time.time()

    def rsum(arr, w):
        cs = np.cumsum(arr)
        r = np.zeros(len(arr)); r[w:] = cs[w:] - cs[:-w]
        return r

    def rstd(arr, w):
        cs = np.cumsum(arr); cs2 = np.cumsum(arr**2)
        r = np.full(len(arr), np.nan)
        s = cs[w:] - cs[:-w]; s2 = cs2[w:] - cs2[:-w]
        var = s2/w - (s/w)**2; np.clip(var, 0, None, out=var)
        r[w:] = np.sqrt(var)
        return r

    def rmean(arr, w):
        cs = np.cumsum(arr)
        r = np.full(len(arr), np.nan); r[w:] = (cs[w:] - cs[:-w]) / w
        return r

    def delta(arr, w):
        r = np.full(len(arr), np.nan); r[w:] = arr[w:] - arr[:-w]
        return r

    p = raw['price_last']
    log_ret = np.zeros(n)
    v = (p[1:] > 0) & (p[:-1] > 0)
    log_ret[1:][v] = np.log(p[1:][v] / p[:-1][v])

    features = {}

    for h in HORIZONS:
        hs = f"_{h}s"
        # Trades
        features[f'vol{hs}'] = rstd(log_ret, h)
        features[f'trade_count{hs}'] = rsum(raw['trade_count'], h)
        features[f'trade_notional{hs}'] = rsum(raw['trade_notional'], h)
        features[f'large_count{hs}'] = rsum(raw['large_count'], h)

        buy_sum = rsum(raw['buy_count'], h)
        tc_sum = rsum(raw['trade_count'], h)
        features[f'buy_ratio{hs}'] = np.where(tc_sum > 0, buy_sum / tc_sum, 0.5)

        buy_not_sum = rsum(raw['buy_notional'], h)
        tn_sum = rsum(raw['trade_notional'], h)
        features[f'buy_not_ratio{hs}'] = np.where(tn_sum > 0, buy_not_sum / tn_sum, 0.5)

        features[f'avg_trade_size{hs}'] = np.where(tc_sum > 0, tn_sum / tc_sum, 0)

        plus_sum = rsum(raw['plus_ticks'], h)
        tick_total = rsum(raw['plus_ticks'] + (raw['trade_count'] - raw['plus_ticks']), h)
        features[f'tick_imbalance{hs}'] = np.where(tick_total > 0,
            (2 * plus_sum - tick_total) / tick_total, 0)

        vwap_n = rsum(raw['vwap_num'], h)
        vwap_d = rsum(raw['vwap_den'], h)
        vwap = np.where(vwap_d > 0, vwap_n / vwap_d, p)
        features[f'vwap_dev{hs}'] = np.where(vwap > 0, (p - vwap) / vwap * 10000, 0)

        # Liquidations — including SIDE features
        features[f'liq_count{hs}'] = rsum(raw['liq_count'], h)
        features[f'liq_notional{hs}'] = rsum(raw['liq_notional'], h)

        liq_buy_sum = rsum(raw['liq_buy_count'], h)
        liq_c_sum = features[f'liq_count{hs}']
        features[f'liq_buy_ratio{hs}'] = np.where(liq_c_sum > 0, liq_buy_sum / liq_c_sum, 0.5)

        # Liq side imbalance: positive = more buy liqs (longs liquidated → price going down)
        liq_sell_sum = liq_c_sum - liq_buy_sum
        features[f'liq_side_imbalance{hs}'] = np.where(liq_c_sum > 0,
            (liq_buy_sum - liq_sell_sum) / liq_c_sum, 0)

        # Liq $ imbalance
        liq_buy_not = rsum(raw['liq_buy_notional'], h)
        liq_total_not = features[f'liq_notional{hs}']
        liq_sell_not = liq_total_not - liq_buy_not
        features[f'liq_not_imbalance{hs}'] = np.where(liq_total_not > 0,
            (liq_buy_not - liq_sell_not) / liq_total_not, 0)

        # OI
        features[f'oi_delta{hs}'] = delta(raw['oi'], h)

        # FR
        features[f'fr{hs}'] = rmean(raw['fr'], h)
        features[f'fr_abs{hs}'] = rmean(np.abs(np.nan_to_num(raw['fr'])), h)

        # Spread
        spread = raw['ask'] - raw['bid']
        mid = (raw['ask'] + raw['bid']) / 2
        features[f'spread_bps{hs}'] = rmean(
            np.where(mid > 0, spread / mid * 10000, np.nan), h)

        # Book imbalance
        total_sz = raw['bid_sz'] + raw['ask_sz']
        book_imb = np.where(total_sz > 0, (raw['bid_sz'] - raw['ask_sz']) / total_sz, 0)
        features[f'book_imbalance{hs}'] = rmean(book_imb, h)

        # Basis
        basis = raw['mark'] - raw['index']
        features[f'basis_bps{hs}'] = rmean(
            np.where(raw['index'] > 0, basis / raw['index'] * 10000, np.nan), h)

    # Clock
    seconds = np.arange(ts_start, ts_start + n)
    tod = seconds % 86400
    features['hour_of_day'] = (tod / 3600).astype(np.float64)
    features['fr_time_to_funding'] = np.where(
        tod < 28800, 28800 - tod,
        np.where(tod < 57600, 57600 - tod, 86400 - tod)).astype(np.float64)

    # Return features (for direction targets)
    features['ret_10s'] = np.zeros(n)
    features['ret_30s'] = np.zeros(n)
    features['ret_60s'] = np.zeros(n)
    for lag in [10, 30, 60]:
        r = np.zeros(n)
        v = (p[lag:] > 0) & (p[:-lag] > 0)
        r[lag:][v] = (p[lag:][v] - p[:-lag][v]) / p[:-lag][v] * 10000
        features[f'ret_{lag}s'] = r

    elapsed = time.time() - t0
    feat_df = pd.DataFrame(features)
    print(f"  Computed {len(feat_df.columns)} features in {elapsed:.1f}s", flush=True)
    return feat_df


# ============================================================================
# TARGET CONSTRUCTION
# ============================================================================

def build_targets(price, n, vol_60s, ts_start):
    """Build all 5 prediction targets."""
    t0 = time.time()

    # First: detect regime switches (same as v29)
    vol_median = pd.Series(vol_60s).rolling(3600, min_periods=900).median().values
    valid = ~np.isnan(vol_60s) & ~np.isnan(vol_median)
    above = valid & (vol_60s > vol_median * 2.0)
    below = valid & (vol_60s < vol_median * 1.2)

    switches = []
    in_volatile = False
    last_sw = -999999
    for i in range(3601, n):
        if not valid[i]: continue
        if not in_volatile and above[i]:
            if (ts_start + i) - last_sw > 1800:
                switches.append(i)
                last_sw = ts_start + i
            in_volatile = True
        elif in_volatile and below[i]:
            in_volatile = False

    print(f"  Detected {len(switches)} switches", flush=True)

    # Compute forward returns and range for every second
    # (only need at sampled points, but compute for full array for simplicity)
    # Forward 300s return
    fwd_ret_300 = np.full(n, np.nan)
    v = (price[300:] > 0) & (price[:-300] > 0)
    fwd_ret_300[:-300][v] = (price[300:][v] - price[:-300][v]) / price[:-300][v] * 10000

    # Forward 300s range (max - min)
    # Compute efficiently using rolling max/min on reversed suffix
    fwd_range_300 = np.full(n, np.nan)
    fwd_max_300 = np.full(n, np.nan)
    fwd_min_300 = np.full(n, np.nan)
    # Use pandas rolling on reversed array for forward-looking
    price_series = pd.Series(price[::-1])
    roll_max = price_series.rolling(300, min_periods=1).max().values[::-1]
    roll_min = price_series.rolling(300, min_periods=1).min().values[::-1]
    valid_p = price > 0
    fwd_max_300[valid_p] = (roll_max[valid_p] - price[valid_p]) / price[valid_p] * 10000
    fwd_min_300[valid_p] = (roll_min[valid_p] - price[valid_p]) / price[valid_p] * 10000
    fwd_range_300 = fwd_max_300 - fwd_min_300

    # Forward sign consistency (30, 60, 120, 300s)
    fwd_signs = np.full(n, np.nan)
    for i in range(n - 300):
        if price[i] <= 0: continue
        signs = []
        for h in [30, 60, 120, 300]:
            if i + h < n and price[i + h] > 0:
                signs.append(np.sign(price[i + h] - price[i]))
        if len(signs) == 4:
            fwd_signs[i] = abs(sum(signs)) / 4

    # Compute this more efficiently — vectorized
    print("  Computing forward targets (vectorized)...", flush=True)
    fwd_signs = np.full(n, np.nan)
    for h in [30, 60, 120, 300]:
        pass  # Already have fwd_ret_300, compute others

    fwd_ret_30 = np.full(n, np.nan)
    fwd_ret_60 = np.full(n, np.nan)
    fwd_ret_120 = np.full(n, np.nan)
    for lag, arr in [(30, fwd_ret_30), (60, fwd_ret_60), (120, fwd_ret_120), (300, fwd_ret_300)]:
        if lag == 300:
            continue  # already computed
        v = (price[lag:] > 0) & (price[:-lag] > 0)
        arr[:-lag][v] = (price[lag:][v] - price[:-lag][v]) / price[:-lag][v] * 10000

    # Sign consistency
    sign_30 = np.sign(np.nan_to_num(fwd_ret_30))
    sign_60 = np.sign(np.nan_to_num(fwd_ret_60))
    sign_120 = np.sign(np.nan_to_num(fwd_ret_120))
    sign_300 = np.sign(np.nan_to_num(fwd_ret_300))
    fwd_signs = np.abs(sign_30 + sign_60 + sign_120 + sign_300) / 4

    # Median range for threshold
    valid_range = fwd_range_300[np.isfinite(fwd_range_300) & (fwd_range_300 > 0)]
    range_median = np.median(valid_range) if len(valid_range) > 1000 else 10
    range_p75 = np.percentile(valid_range, 75) if len(valid_range) > 1000 else 15

    abs_ret_300 = np.abs(np.nan_to_num(fwd_ret_300))
    abs_p75 = np.percentile(abs_ret_300[abs_ret_300 > 0], 75) if (abs_ret_300 > 0).sum() > 1000 else 10

    print(f"  Range median={range_median:.1f} bps, P75={range_p75:.1f} bps", flush=True)
    print(f"  |ret_300s| P75={abs_p75:.1f} bps", flush=True)

    # Build targets
    targets = {}

    # T1: Vol expansion — range > median
    targets['T1_vol_expansion'] = (fwd_range_300 > range_median).astype(np.int8)

    # T2: Big move — |ret_300s| > P75
    targets['T2_big_move'] = (abs_ret_300 > abs_p75).astype(np.int8)

    # T3: Continuation — given recent 30s move, does 300s continue?
    # Only defined where there IS a recent move (|ret_30s| > 1 bps)
    recent_ret = np.zeros(n)
    v = (price[:-0] if False else np.ones(n, dtype=bool))  # placeholder
    # Use backward 30s return as "recent move"
    back_ret_30 = np.zeros(n)
    v = (price[30:] > 0) & (price[:-30] > 0)
    back_ret_30[30:][v] = (price[30:][v] - price[:-30][v]) / price[:-30][v] * 10000
    has_move = np.abs(back_ret_30) > 1.0  # at least 1 bps move
    continuation = (np.sign(back_ret_30) == np.sign(np.nan_to_num(fwd_ret_300))) & has_move
    targets['T3_continuation'] = continuation.astype(np.int8)
    targets['T3_mask'] = has_move  # only evaluate where there's a move

    # T4: Straight-line — sign consistency > 0.75
    targets['T4_straight_line'] = (fwd_signs > 0.75).astype(np.int8)

    # T5: Direction — predict UP vs DOWN (only where |ret_300s| > 2 bps)
    has_direction = abs_ret_300 > 2.0
    targets['T5_direction_up'] = (np.nan_to_num(fwd_ret_300) > 0).astype(np.int8)
    targets['T5_mask'] = has_direction

    # Stats
    for name, arr in targets.items():
        if 'mask' in name:
            continue
        valid_mask = np.isfinite(fwd_range_300)
        if name.startswith('T3'):
            valid_mask = valid_mask & targets['T3_mask']
        elif name.startswith('T5'):
            valid_mask = valid_mask & targets['T5_mask']
        pos_rate = arr[valid_mask].mean() if valid_mask.sum() > 0 else 0
        print(f"  {name}: {pos_rate:.4f} positive rate ({valid_mask.sum():,} valid)", flush=True)

    targets['switches'] = switches
    targets['fwd_ret_300'] = fwd_ret_300
    targets['fwd_range_300'] = fwd_range_300
    targets['fwd_signs'] = fwd_signs
    targets['back_ret_30'] = back_ret_30

    print(f"  Targets built ({time.time()-t0:.1f}s)", flush=True)
    return targets


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiments(feat_df, targets, n):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    print(f"\n{'#'*80}")
    print(f"  v30: TRADEABLE TARGET EXPERIMENTS")
    print(f"{'#'*80}")

    # Sample at 10s, week1 train / week2 test
    sample_idx = np.arange(3600, n - 300, 10)
    split = n // 2
    train_idx = sample_idx[sample_idx < split]
    test_idx = sample_idx[sample_idx >= split]

    all_cols = list(feat_df.columns)
    # Exclude ret_ features from ML input (they're targets, not features)
    feat_cols = [c for c in all_cols if not c.startswith('ret_')]
    X_all = feat_df[feat_cols].values

    # NaN handling
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)

    # Remove zero-variance
    var = X_all[train_idx].var(axis=0)
    good_cols = var > 1e-12
    feat_cols = [c for c, g in zip(feat_cols, good_cols) if g]
    X_all = X_all[:, good_cols]

    X_train = X_all[train_idx]
    X_test = X_all[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    print(f"  Features: {len(feat_cols)}")
    print(f"  Train: {len(train_idx):,}, Test: {len(test_idx):,}")

    # =====================================================================
    # Run each target
    # =====================================================================

    target_configs = [
        ('T1_vol_expansion', 'Vol Expansion (range > median)', None, 'straddle'),
        ('T2_big_move', 'Big Move (|ret| > P75)', None, 'selective straddle'),
        ('T3_continuation', 'Continuation (30s→300s)', 'T3_mask', 'momentum'),
        ('T4_straight_line', 'Straight-Line (consistency > 0.75)', None, 'trend-follow'),
        ('T5_direction_up', 'Direction UP', 'T5_mask', 'directional'),
    ]

    all_results = {}

    for target_name, description, mask_name, strategy in target_configs:
        print(f"\n{'='*80}")
        print(f"  TARGET: {description}")
        print(f"  Strategy: {strategy}")
        print(f"{'='*80}")

        y_all = targets[target_name]

        # Apply mask if needed
        if mask_name:
            mask = targets[mask_name]
            tr_mask = mask[train_idx]
            te_mask = mask[test_idx]
            y_tr = y_all[train_idx][tr_mask]
            y_te = y_all[test_idx][te_mask]
            X_tr = X_tr_s[tr_mask]
            X_te = X_te_s[te_mask]
        else:
            y_tr = y_all[train_idx]
            y_te = y_all[test_idx]
            X_tr = X_tr_s
            X_te = X_te_s

        pos_rate_tr = y_tr.mean()
        pos_rate_te = y_te.mean()
        print(f"  Train: {len(y_tr):,} samples, {y_tr.sum():,} pos ({pos_rate_tr:.4f})")
        print(f"  Test:  {len(y_te):,} samples, {y_te.sum():,} pos ({pos_rate_te:.4f})")

        if y_tr.sum() < 50 or y_te.sum() < 50:
            print(f"  SKIPPED: too few positives")
            continue

        # Train GBM
        print(f"  Training GBM...", flush=True)
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, min_samples_leaf=50
        )
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]

        try:
            auc = roc_auc_score(y_te, y_prob)
        except:
            auc = 0.5
        print(f"  AUC: {auc:.4f}")

        # Precision at various thresholds
        precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
        print(f"\n  --- Precision-Recall Trade-off ---")
        print(f"  {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>8s}  {'#Signals':>10s}  {'Lift':>6s}")
        print(f"  {'-'*50}")
        for pct in [90, 80, 70, 50]:
            thresh = np.percentile(y_prob, pct)
            pred_pos = y_prob >= thresh
            if pred_pos.sum() > 0:
                prec = y_te[pred_pos].mean()
                rec = y_te[pred_pos].sum() / max(y_te.sum(), 1)
                lift = prec / max(pos_rate_te, 0.001)
                n_signals = pred_pos.sum()
                print(f"  P{pct:>2d}={thresh:.3f}  {prec:>10.4f}  {rec:>8.4f}  {n_signals:>10,}  {lift:>5.2f}x")

        # Feature importance
        importances = model.feature_importances_
        imp_order = np.argsort(importances)[::-1]
        print(f"\n  --- Top 15 Features ---")
        for rank, idx in enumerate(imp_order[:15]):
            print(f"    {rank+1:>2d}. {feat_cols[idx]:>35s}  {importances[idx]:.4f}")

        # Strategy simulation (simple)
        if target_name in ('T1_vol_expansion', 'T2_big_move'):
            # Straddle: when model fires, expect large range
            # Profit ≈ actual range - cost (spread + slippage)
            print(f"\n  --- Strategy Simulation (straddle) ---")
            fwd_range = targets['fwd_range_300'][test_idx]
            if mask_name:
                fwd_range = fwd_range[te_mask]
            # Top 10% signals
            thresh = np.percentile(y_prob, 90)
            signal = y_prob >= thresh
            if signal.sum() > 0:
                avg_range_signal = np.nanmean(fwd_range[signal])
                avg_range_all = np.nanmean(fwd_range)
                cost = 2.0  # assume 2 bps round-trip cost
                print(f"  Avg range (all):    {avg_range_all:.1f} bps")
                print(f"  Avg range (signal): {avg_range_signal:.1f} bps")
                print(f"  Range lift: {avg_range_signal/max(avg_range_all,0.1):.2f}x")
                print(f"  Net after cost ({cost} bps): {avg_range_signal - cost:.1f} bps per trade")
                print(f"  Signals: {signal.sum():,} ({signal.sum()/len(y_prob)*100:.1f}% of time)")

        elif target_name == 'T5_direction_up':
            # Directional: when model says UP, go long; when DOWN, go short
            print(f"\n  --- Strategy Simulation (directional) ---")
            fwd_ret = targets['fwd_ret_300'][test_idx]
            if mask_name:
                fwd_ret = fwd_ret[te_mask]
            # Model prediction: >0.5 = UP, <0.5 = DOWN
            pred_up = y_prob > 0.55
            pred_dn = y_prob < 0.45
            if pred_up.sum() > 0 and pred_dn.sum() > 0:
                avg_ret_up = np.nanmean(fwd_ret[pred_up])
                avg_ret_dn = np.nanmean(fwd_ret[pred_dn])
                print(f"  Pred UP ({pred_up.sum():,}):  avg fwd ret = {avg_ret_up:+.2f} bps")
                print(f"  Pred DN ({pred_dn.sum():,}):  avg fwd ret = {avg_ret_dn:+.2f} bps")
                print(f"  Long-short spread: {avg_ret_up - avg_ret_dn:.2f} bps")

        elif target_name == 'T3_continuation':
            # Momentum: when model says continue, enter in direction of recent move
            print(f"\n  --- Strategy Simulation (momentum) ---")
            fwd_ret = targets['fwd_ret_300'][test_idx]
            back_ret = targets['back_ret_30'][test_idx]
            if mask_name:
                fwd_ret = fwd_ret[te_mask]
                back_ret = back_ret[te_mask]
            # Top 20% continuation signals
            thresh = np.percentile(y_prob, 80)
            signal = y_prob >= thresh
            if signal.sum() > 0:
                # PnL = sign(back_ret) * fwd_ret (go with momentum)
                pnl = np.sign(back_ret[signal]) * fwd_ret[signal]
                avg_pnl = np.nanmean(pnl)
                win_rate = (pnl > 0).sum() / max(len(pnl), 1)
                print(f"  Signals: {signal.sum():,}")
                print(f"  Avg PnL (momentum): {avg_pnl:+.2f} bps")
                print(f"  Win rate: {win_rate*100:.1f}%")
                # Baseline (all moves)
                pnl_all = np.sign(back_ret) * fwd_ret
                avg_pnl_all = np.nanmean(pnl_all)
                print(f"  Baseline PnL (all moves): {avg_pnl_all:+.2f} bps")

        all_results[target_name] = {
            'auc': auc,
            'pos_rate': float(pos_rate_te),
            'n_test': len(y_te),
        }

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n\n{'#'*80}")
    print(f"  SUMMARY: ALL TARGETS")
    print(f"{'#'*80}")
    print(f"  {'Target':>25s}  {'AUC':>8s}  {'Base Rate':>10s}  {'Strategy':>15s}")
    print(f"  {'-'*65}")
    for target_name, description, mask_name, strategy in target_configs:
        if target_name in all_results:
            r = all_results[target_name]
            print(f"  {description[:25]:>25s}  {r['auc']:>8.4f}  {r['pos_rate']:>10.4f}  {strategy:>15s}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    print(f"\n{'#'*80}")
    print(f"  v30: TRADEABLE PREDICTION TARGETS")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Dates: {DATES[0]} to {DATES[-1]} ({len(DATES)} days)")
    print(f"{'#'*80}")
    print_mem("start")

    # Build data
    raw, ts_start, n = build_all(DATES)

    # Features
    print(f"\n  --- Feature Engineering ---")
    feat_df = compute_features(raw, ts_start, n)

    # Targets
    print(f"\n  --- Building Targets ---")
    vol_60s = feat_df['vol_60s'].values
    price = raw['price_last']
    targets = build_targets(price, n, vol_60s, ts_start)
    del raw; gc.collect()

    # Experiments
    run_experiments(feat_df, targets, n)

    elapsed = time.time() - t_start
    print(f"\n{'#'*80}")
    print(f"  COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
