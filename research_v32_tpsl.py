#!/usr/bin/env python3
"""
v32: Asymmetric TP/SL Prediction

Can we predict when TP will be hit before SL?

Setup: At each second, consider two trades:
  LONG:  TP = +X bps, SL = -X/2 bps
  SHORT: TP = -X bps, SL = +X/2 bps

Target: predict which direction (long or short) has TP hit first.
If we can predict this, we have a 2:1 reward-risk strategy.

Baseline (unconditional BTC):
  X=5 bps, 300s: Long TP hit 33.6%, SL hit 60.3% → EV ≈ +0.24 bps
  The 2:1 ratio compensates for lower win rate, but barely.
  If we can push TP hit from 33% → 40%+, strategy becomes very profitable.

We test multiple X values and time limits to find the sweet spot.
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

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
SYMBOL = "BTCUSDT"
DATES = [f"2025-05-{d:02d}" for d in range(11, 18)]  # 7 days
HORIZONS = [300, 900, 3600]  # Skip 60s — minimal signal, save RAM

# TP/SL configs to test: (tp_bps, sl_bps, time_limit_s)
TPSL_CONFIGS = [
    (5, 2.5, 300),    # tight, 5min
    (5, 2.5, 600),    # tight, 10min
    (10, 5, 300),     # medium, 5min
    (10, 5, 600),     # medium, 10min
    (10, 5, 900),     # medium, 15min
    (15, 7.5, 600),   # wide, 10min
    (15, 7.5, 900),   # wide, 15min
    (20, 10, 900),    # very wide, 15min
]


def mem_gb():
    return psutil.virtual_memory().used / 1e9, psutil.virtual_memory().available / 1e9

def print_mem(label=""):
    u, a = mem_gb()
    print(f"  [RAM] used={u:.1f}GB avail={a:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING (from v30)
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
    t0 = time.time()
    field_map = {
        'openInterestValue': 'oi', 'fundingRate': 'fr',
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
        gc.collect()
        count += day_count
        u, a = mem_gb()
        eta = (time.time() - t0) / (di + 1) * (len(dates) - di - 1)
        print(f"    {date_str}: {day_count:,} ticker (RAM={u:.1f}/{a:.1f}GB, ETA={eta:.0f}s)", flush=True)
    for k in arrays:
        arrays[k] = pd.Series(arrays[k]).ffill().values
    print(f"  Ticker: {count:,} total updates", flush=True)
    return arrays


# ============================================================================
# BUILD ARRAYS & FEATURES
# ============================================================================

def build_all(dates):
    t0 = time.time()

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
                'plus_ticks', 'large_count']
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
    if len(liq_ts) > 0:
        l_off = (liq_ts - ts_start).astype(np.int64)
        valid = (l_off >= 0) & (l_off < n)
        l_off_v = l_off[valid]
        raw['liq_count'] = np.bincount(l_off_v, minlength=n).astype(np.float32)
        raw['liq_notional'] = np.bincount(l_off_v, weights=liq_not[valid], minlength=n).astype(np.float32)
        raw['liq_buy_count'] = np.bincount(l_off_v, weights=liq_is_buy[valid], minlength=n).astype(np.float32)
    del liq_ts, liq_is_buy, liq_not; gc.collect()
    print_mem('after liqs')

    # Skip ticker — v29-rich showed trades-only matches full model AUC
    # This saves ~2GB RAM from JSON parsing
    print(f"  Built {n:,} seconds ({time.time()-t0:.0f}s)", flush=True)
    print_mem("raw done")
    return raw, ts_start, n


def compute_features(raw, ts_start, n):
    t0 = time.time()

    def rsum(arr, w):
        cs = np.cumsum(arr)
        r = np.zeros(len(arr), dtype=np.float32); r[w:] = cs[w:] - cs[:-w]
        return r

    def rstd(arr, w):
        cs = np.cumsum(arr); cs2 = np.cumsum(arr**2)
        r = np.full(len(arr), np.nan, dtype=np.float32)
        s = cs[w:] - cs[:-w]; s2 = cs2[w:] - cs2[:-w]
        var = s2/w - (s/w)**2; np.clip(var, 0, None, out=var)
        r[w:] = np.sqrt(var)
        return r

    def rmean(arr, w):
        cs = np.cumsum(arr)
        r = np.full(len(arr), np.nan, dtype=np.float32); r[w:] = (cs[w:] - cs[:-w]) / w
        return r

    def delta(arr, w):
        r = np.full(len(arr), np.nan, dtype=np.float32); r[w:] = arr[w:] - arr[:-w]
        return r

    p = raw['price_last']
    log_ret = np.zeros(n, dtype=np.float32)
    v = (p[1:] > 0) & (p[:-1] > 0)
    log_ret[1:][v] = np.log(p[1:][v] / p[:-1][v])

    features = {}

    for h in HORIZONS:
        hs = f"_{h}s"
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
        features[f'tick_imbalance{hs}'] = np.where(tc_sum > 0,
            (2 * plus_sum - tc_sum) / tc_sum, 0)

        # Liquidations
        features[f'liq_count{hs}'] = rsum(raw['liq_count'], h)
        features[f'liq_notional{hs}'] = rsum(raw['liq_notional'], h)
        liq_buy_sum = rsum(raw['liq_buy_count'], h)
        liq_c_sum = features[f'liq_count{hs}']
        features[f'liq_buy_ratio{hs}'] = np.where(liq_c_sum > 0, liq_buy_sum / liq_c_sum, 0.5)
        liq_sell_sum = liq_c_sum - liq_buy_sum
        features[f'liq_side_imbalance{hs}'] = np.where(liq_c_sum > 0,
            (liq_buy_sum - liq_sell_sum) / liq_c_sum, 0)



    # Recent returns (directional features — important for TP/SL prediction)
    for lag in [10, 30, 60, 300]:
        r = np.zeros(n, dtype=np.float32)
        v = (p[lag:] > 0) & (p[:-lag] > 0)
        r[lag:][v] = (p[lag:][v] - p[:-lag][v]) / p[:-lag][v] * 10000
        features[f'ret_{lag}s'] = r

    # Clock
    seconds = np.arange(ts_start, ts_start + n)
    tod = seconds % 86400
    features['hour_of_day'] = (tod / 3600).astype(np.float32)
    features['fr_time_to_funding'] = np.where(
        tod < 28800, 28800 - tod,
        np.where(tod < 57600, 57600 - tod, 86400 - tod)).astype(np.float32)

    # Vol (60s) — useful even though we skip 60s horizon for other features
    features['vol_60s'] = rstd(log_ret, 60)

    elapsed = time.time() - t0
    feat_df = pd.DataFrame(features)
    print(f"  Computed {len(feat_df.columns)} features in {elapsed:.1f}s", flush=True)
    return feat_df


# ============================================================================
# TP/SL TARGET CONSTRUCTION
# ============================================================================

def compute_tpsl_outcomes(price, n, sample_idx, tp_bps, sl_bps, time_limit):
    """Vectorized TP/SL outcome computation.
    For each sample: does price hit +tp_bps or -sl_bps first within time_limit?
    Uses forward-scanning with early termination per chunk.
    """
    tp_frac = tp_bps / 10000
    sl_frac = sl_bps / 10000
    ns = len(sample_idx)

    long_win = np.full(ns, np.nan, dtype=np.float32)
    short_win = np.full(ns, np.nan, dtype=np.float32)
    best_side = np.full(ns, np.nan, dtype=np.float32)

    # Process in chunks of seconds: for each offset t=1..time_limit,
    # check all unresolved samples at once
    p0 = price[sample_idx].astype(np.float64)
    valid = (p0 > 0) & ~np.isnan(p0)

    long_tp_price = p0 * (1 + tp_frac)
    long_sl_price = p0 * (1 - sl_frac)
    short_tp_price = p0 * (1 - tp_frac)
    short_sl_price = p0 * (1 + sl_frac)

    l_resolved = ~valid  # invalid samples are "resolved" (will stay NaN)
    s_resolved = ~valid

    for t in range(1, time_limit + 1):
        if l_resolved.all() and s_resolved.all():
            break

        # Get price at sample_idx + t (clip to bounds)
        fwd_idx = np.clip(sample_idx + t, 0, n - 1)
        pt = price[fwd_idx].astype(np.float64)

        # Long side
        if not l_resolved.all():
            l_tp_hit = (~l_resolved) & (pt >= long_tp_price)
            l_sl_hit = (~l_resolved) & (pt <= long_sl_price)
            long_win[l_tp_hit] = 1.0
            long_win[l_sl_hit] = 0.0
            l_resolved |= l_tp_hit | l_sl_hit

        # Short side
        if not s_resolved.all():
            s_tp_hit = (~s_resolved) & (pt <= short_tp_price)
            s_sl_hit = (~s_resolved) & (pt >= short_sl_price)
            short_win[s_tp_hit] = 1.0
            short_win[s_sl_hit] = 0.0
            s_resolved |= s_tp_hit | s_sl_hit

    # Best side
    best_side[long_win == 1.0] = 1.0   # long TP hit
    best_side[short_win == 1.0] = 0.0  # short TP hit
    both_lost = (long_win == 0.0) & (short_win == 0.0)
    best_side[both_lost] = 0.5

    return long_win, short_win, best_side


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(feat_df, price, n, ts_start):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    print(f"\n{'#'*80}")
    print(f"  v32: ASYMMETRIC TP/SL PREDICTION")
    print(f"{'#'*80}")

    # Sample every 10s, split at midpoint
    sample_idx = np.arange(3600, n - 900, 10)
    split = n // 2
    train_mask = sample_idx < split
    test_mask = sample_idx >= split

    # Prepare features — index at sample points
    all_cols = list(feat_df.columns)
    X_all = feat_df.values[sample_idx].astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
    del feat_df; gc.collect()

    # Remove zero-variance
    var = X_all[train_mask].var(axis=0)
    good_cols = var > 1e-12
    feat_cols = [c for c, g in zip(all_cols, good_cols) if g]
    X_all = X_all[:, good_cols]

    print(f"  Features: {len(feat_cols)}")
    print(f"  Train: {train_mask.sum():,}, Test: {test_mask.sum():,}")

    # =====================================================================
    # Phase 1: Baseline stats for all configs
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 1: Baseline TP/SL Statistics")
    print(f"{'='*80}")
    print(f"  {'TP bps':>7s}  {'SL bps':>7s}  {'TL':>5s}  {'L-TP%':>7s}  {'L-SL%':>7s}  {'S-TP%':>7s}  {'S-SL%':>7s}  {'TO%':>6s}  {'EV':>7s}")
    print(f"  {'-'*65}")

    config_results = {}
    for tp_bps, sl_bps, tl in TPSL_CONFIGS:
        t0 = time.time()
        long_win, short_win, best_side = compute_tpsl_outcomes(
            price, n, sample_idx, tp_bps, sl_bps, tl)

        # Stats
        valid_l = ~np.isnan(long_win)
        valid_s = ~np.isnan(short_win)
        l_tp = (long_win[valid_l] == 1).mean() * 100
        l_sl = (long_win[valid_l] == 0).mean() * 100
        s_tp = (short_win[valid_s] == 1).mean() * 100
        s_sl = (short_win[valid_s] == 0).mean() * 100
        timeout = (~valid_l).mean() * 100

        # EV: for each sample, take the better side
        ev_samples = []
        for li, si in zip(long_win, short_win):
            if li == 1:
                ev_samples.append(tp_bps)
            elif si == 1:
                ev_samples.append(tp_bps)
            elif li == 0 and si == 0:
                ev_samples.append(-sl_bps)  # both SL hit
            elif li == 0:
                ev_samples.append(-sl_bps)
            elif si == 0:
                ev_samples.append(-sl_bps)
        ev = np.mean(ev_samples) if ev_samples else 0

        elapsed = time.time() - t0
        print(f"  {tp_bps:>7.1f}  {sl_bps:>7.1f}  {tl:>5d}  {l_tp:>6.1f}%  {l_sl:>6.1f}%  {s_tp:>6.1f}%  {s_sl:>6.1f}%  {timeout:>5.1f}%  {ev:>+6.2f}  ({elapsed:.0f}s)")

        config_results[(tp_bps, sl_bps, tl)] = {
            'long_win': long_win, 'short_win': short_win, 'best_side': best_side,
            'l_tp_pct': l_tp, 's_tp_pct': s_tp, 'timeout_pct': timeout,
        }

    # =====================================================================
    # Phase 2: ML prediction — can we predict which side wins?
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 2: ML Prediction — Which Side Wins?")
    print(f"{'='*80}")

    summary_rows = []

    for tp_bps, sl_bps, tl in TPSL_CONFIGS:
        cr = config_results[(tp_bps, sl_bps, tl)]
        long_win = cr['long_win']
        short_win = cr['short_win']
        best_side = cr['best_side']

        config_label = f"TP={tp_bps} SL={sl_bps} TL={tl}s"
        print(f"\n  --- {config_label} ---")

        # Target A: predict long TP hit (binary: 1=TP, 0=SL, exclude timeouts)
        valid = ~np.isnan(long_win)
        if valid.sum() < 1000:
            print(f"  SKIP: too few valid samples ({valid.sum()})")
            continue

        y_long = long_win[valid].astype(np.int8)
        X_valid = X_all[valid]
        tr = train_mask[valid]
        te = test_mask[valid]

        if tr.sum() < 500 or te.sum() < 500:
            print(f"  SKIP: too few train/test")
            continue

        X_tr, X_te = X_valid[tr], X_valid[te]
        y_tr, y_te = y_long[tr], y_long[te]

        base_rate = y_te.mean()
        print(f"  Long TP base rate: {base_rate:.4f} (train: {y_tr.mean():.4f})")

        # Train GBM
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42, min_samples_leaf=50
        )
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]

        try:
            auc = roc_auc_score(y_te, y_prob)
        except:
            auc = 0.5
        print(f"  AUC (long TP): {auc:.4f}")

        # Strategy simulation: when model says high prob of long TP, go long
        # When model says low prob (= high prob of short TP), go short
        print(f"\n  Strategy: go long when P(long TP) > threshold, short when < (1-threshold)")
        print(f"  {'Threshold':>10s}  {'#Long':>7s}  {'L-Win%':>7s}  {'#Short':>7s}  {'S-Win%':>7s}  {'EV/trade':>9s}  {'Lift':>6s}")
        print(f"  {'-'*55}")

        for pct in [90, 80, 70, 60]:
            hi_thresh = np.percentile(y_prob, pct)
            lo_thresh = np.percentile(y_prob, 100 - pct)

            long_signals = y_prob >= hi_thresh
            short_signals = y_prob <= lo_thresh

            if long_signals.sum() > 0:
                l_win_rate = y_te[long_signals].mean()
                l_ev = l_win_rate * tp_bps - (1 - l_win_rate) * sl_bps
            else:
                l_win_rate = 0; l_ev = 0

            if short_signals.sum() > 0:
                # For short: y_te=0 means short TP hit
                s_win_rate = (1 - y_te[short_signals]).mean()
                s_ev = s_win_rate * tp_bps - (1 - s_win_rate) * sl_bps
            else:
                s_win_rate = 0; s_ev = 0

            combined_ev = (l_ev + s_ev) / 2 if (long_signals.sum() > 0 and short_signals.sum() > 0) else 0
            baseline_ev = base_rate * tp_bps - (1 - base_rate) * sl_bps

            print(f"  P{pct:>2d}        {long_signals.sum():>7,}  {l_win_rate*100:>6.1f}%  {short_signals.sum():>7,}  {s_win_rate*100:>6.1f}%  {combined_ev:>+8.2f}  {combined_ev/max(abs(baseline_ev),0.01):>5.1f}x")

        # Top features
        importances = model.feature_importances_
        imp_order = np.argsort(importances)[::-1]
        print(f"\n  Top 10 features:")
        for rank, idx in enumerate(imp_order[:10]):
            print(f"    {rank+1:>2d}. {feat_cols[idx]:>30s}  {importances[idx]:.4f}")

        summary_rows.append({
            'config': config_label,
            'auc': auc,
            'base_rate': base_rate,
            'tp_bps': tp_bps,
            'sl_bps': sl_bps,
            'tl': tl,
        })

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n\n{'#'*80}")
    print(f"  SUMMARY: ALL CONFIGS")
    print(f"{'#'*80}")
    print(f"  {'Config':>30s}  {'AUC':>8s}  {'Base Rate':>10s}  {'2:1 EV':>8s}")
    print(f"  {'-'*60}")
    for r in summary_rows:
        baseline_ev = r['base_rate'] * r['tp_bps'] - (1 - r['base_rate']) * r['sl_bps']
        print(f"  {r['config']:>30s}  {r['auc']:>8.4f}  {r['base_rate']:>10.4f}  {baseline_ev:>+7.2f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    print(f"\n{'#'*80}")
    print(f"  v32: ASYMMETRIC TP/SL PREDICTION")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Dates: {DATES[0]} to {DATES[-1]} ({len(DATES)} days)")
    print(f"  Configs: {len(TPSL_CONFIGS)} TP/SL combinations")
    print(f"{'#'*80}")
    print_mem("start")

    # Build data
    raw, ts_start, n = build_all(DATES)

    # Features
    print(f"\n  --- Feature Engineering ---")
    feat_df = compute_features(raw, ts_start, n)
    price = raw['price_last'].copy()
    del raw; gc.collect()
    print_mem("features done")

    # Experiment
    run_experiment(feat_df, price, n, ts_start)

    elapsed = time.time() - t_start
    print(f"\n{'#'*80}")
    print(f"  COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
