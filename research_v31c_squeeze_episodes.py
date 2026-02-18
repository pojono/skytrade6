#!/usr/bin/env python3
"""
v31c: Squeeze Episode Detection & Continuation

v31b showed per-second OI divergence is too noisy, BUT:
  - Long squeeze (shorts liq'd): 55.3% cont at 300s, +2.22bps — real signal
  - Short squeeze (longs liq'd): no signal
  - Need EPISODE-based detection, not per-second sampling

APPROACH:
  1. Detect squeeze ONSET: moment when OI starts dropping while price is moving
  2. Require minimum OI drop rate + price momentum to trigger
  3. Non-overlapping episodes with cooldown period
  4. Measure forward continuation from episode onset
  5. Also measure: how long does the squeeze last? What predicts duration?

EPISODE DETECTION:
  - Compute rolling OI velocity and price momentum at multiple windows
  - Trigger when both cross thresholds simultaneously
  - Require minimum gap between episodes (cooldown = lookback window)
  - Confirm with liquidation activity

TESTS:
  1. Episode continuation probability vs random baseline
  2. Episode duration distribution
  3. What predicts squeeze strength? (OI drop rate, liq intensity, volume)
  4. Long vs short squeeze asymmetry
  5. Cross-asset (BTC vs ETH) and cross-period stability
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import time
import json
import gzip
import warnings
import argparse
import numpy as np
import pandas as pd
import psutil
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
SYMBOL = "BTCUSDT"
DATES = [f"2025-05-{d:02d}" for d in range(12, 19)]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# UTILITIES
# ============================================================================

def mem_gb():
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label=""):
    u, a = mem_gb()
    print(f"  [RAM] used={u:.1f}GB avail={a:.1f}GB {label}", flush=True)

def fmt_time(seconds):
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds/60:.1f}m"


# ============================================================================
# DATA LOADING (reuse from v31b — trades, ticker OI, liquidations)
# ============================================================================

def load_trades_day(date_str):
    path = DATA_DIR / SYMBOL / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, compression='gzip',
                     usecols=['timestamp', 'price', 'size', 'side'])
    ts_s = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.float64)
    notionals = prices * sizes
    is_buy = (df['side'] == 'Buy').values.astype(np.float32)
    n_raw = len(df)
    del df; gc.collect()

    day_start = int(ts_s.min()); day_end = int(ts_s.max())
    n = day_end - day_start + 1
    off = (ts_s - day_start).astype(np.int32)

    trade_count = np.bincount(off, minlength=n).astype(np.float32)
    trade_notional = np.bincount(off, weights=notionals, minlength=n).astype(np.float32)
    buy_notional = np.bincount(off, weights=notionals * is_buy, minlength=n).astype(np.float32)

    price_last = np.full(n, np.nan, dtype=np.float32)
    _, last_idx = np.unique(off[::-1], return_index=True)
    last_idx = len(off) - 1 - last_idx
    for uo, li in zip(np.unique(off), last_idx):
        price_last[uo] = prices[li]

    del ts_s, prices, sizes, notionals, is_buy, off; gc.collect()
    return {'day_start': day_start, 'n': n, 'n_raw': n_raw,
            'trade_count': trade_count, 'trade_notional': trade_notional,
            'buy_notional': buy_notional, 'price_last': price_last}


def load_ticker_oi(dates, ts_start, n):
    arrays = {k: np.full(n, np.nan, dtype=np.float32) for k in ['oi', 'fr']}
    count = 0
    for date_str in dates:
        for f in sorted((DATA_DIR / SYMBOL / "bybit" / "ticker").glob(
                f"ticker_{date_str}_hr*.jsonl.gz")):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        ts_s = d.get('ts', 0) // 1000
                        idx = ts_s - ts_start
                        if idx < 0 or idx >= n:
                            continue
                        data = d.get('result', {}).get('data', {})
                        if 'openInterestValue' in data:
                            arrays['oi'][idx] = float(data['openInterestValue'])
                        if 'fundingRate' in data:
                            arrays['fr'][idx] = float(data['fundingRate'])
                        count += 1
                    except:
                        continue
        print(f"    {date_str}: loaded ticker", flush=True)
    for k in arrays:
        arr = arrays[k]; mask = np.isnan(arr)
        if not mask.all():
            fv = np.argmin(mask)
            for i in range(fv + 1, n):
                if mask[i]: arr[i] = arr[i - 1]
    print(f"  Ticker: {count:,} updates → {n:,}s arrays", flush=True)
    return arrays


def load_liquidations(dates):
    records = []
    for date_str in dates:
        day_count = 0
        for f in sorted((DATA_DIR / SYMBOL / "bybit" / "liquidations").glob(
                f"liquidation_{date_str}_hr*.jsonl.gz")):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        if 'result' in d and 'data' in d['result']:
                            for ev in d['result']['data']:
                                records.append((int(ev['T']) // 1000,
                                    1.0 if ev['S'] == 'Buy' else 0.0,
                                    float(ev['v']) * float(ev['p'])))
                                day_count += 1
                    except: continue
        print(f"  {date_str}: {day_count:>6,} liquidations", flush=True)
    if not records:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(records)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.float32), arr[:, 2].astype(np.float32)


def build_arrays(dates):
    t0 = time.time(); print_mem("start")
    print(f"\n[1/3] Loading liquidations...", flush=True)
    liq_ts, liq_is_buy, liq_not = load_liquidations(dates)
    print(f"  Total: {len(liq_ts):,} liquidation events", flush=True)

    print(f"\n[2/3] Loading trades...", flush=True)
    day_results = []
    for i, date_str in enumerate(dates):
        dr = load_trades_day(date_str)
        if dr:
            day_results.append(dr)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f"    {date_str}: {dr['n_raw']:,} trades → {dr['n']:,}s "
                  f"({elapsed:.1f}s, ETA {fmt_time(eta)})", flush=True)
        gc.collect()
    if not day_results: return None

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n'] - 1
    n = ts_end - ts_start + 1

    raw = {k: np.zeros(n, dtype=np.float32) for k in
           ['trade_count', 'trade_notional', 'buy_notional']}
    raw['price_last'] = np.full(n, np.nan, dtype=np.float32)
    for dr in day_results:
        o = dr['day_start'] - ts_start; l = dr['n']
        for k in ['trade_count', 'trade_notional', 'buy_notional']:
            raw[k][o:o+l] = dr[k]
        raw['price_last'][o:o+l] = dr['price_last']
    del day_results; gc.collect()

    p = raw['price_last']; mask = np.isnan(p)
    if not mask.all():
        fv = np.argmin(mask)
        for i in range(fv + 1, n):
            if mask[i]: p[i] = p[i - 1]

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

    print_mem("before ticker")
    print(f"\n[3/3] Loading ticker (OI/FR)...", flush=True)
    ticker = load_ticker_oi(dates, ts_start, n)
    raw.update(ticker); del ticker; gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Built {n:,} seconds ({n/86400:.1f} days) in {elapsed:.1f}s", flush=True)
    print_mem("arrays done")
    return raw, ts_start, n


# ============================================================================
# EPISODE DETECTION
# ============================================================================

def detect_squeeze_episodes(raw, ts_start, n):
    """
    Detect discrete squeeze episodes.

    A squeeze episode starts when:
      1. Price momentum over lookback exceeds threshold
      2. OI velocity over lookback is negative (below threshold)
      3. Minimum cooldown since last episode

    Returns list of episode dicts with onset time, direction, features.
    """
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"SQUEEZE EPISODE DETECTION")
    print(f"{'='*70}")

    p = raw['price_last'].astype(np.float64)
    oi = raw['oi'].astype(np.float64)

    # Precompute rolling sums for liquidations and volume
    def rsum(arr, w):
        cs = np.cumsum(arr.astype(np.float64))
        r = np.zeros(len(arr), dtype=np.float64)
        r[w:] = cs[w:] - cs[:-w]
        return r

    all_episodes = []

    # Test multiple lookback windows
    for lb in [60, 120, 300]:
        # Price momentum
        price_mom = np.full(n, np.nan, dtype=np.float64)
        v = (p[lb:] > 0) & (p[:-lb] > 0)
        price_mom[lb:][v] = (p[lb:][v] - p[:-lb][v]) / p[:-lb][v]

        # OI velocity
        oi_vel = np.full(n, np.nan, dtype=np.float64)
        v_oi = (oi[lb:] > 0) & (oi[:-lb] > 0)
        oi_vel[lb:][v_oi] = (oi[lb:][v_oi] - oi[:-lb][v_oi]) / oi[:-lb][v_oi]

        # Rolling liq count and notional
        liq_sum = rsum(raw['liq_count'], lb)
        liq_not_sum = rsum(raw['liq_notional'], lb)
        liq_buy_sum = rsum(raw['liq_buy_count'], lb)
        trade_not_sum = rsum(raw['trade_notional'], lb)
        buy_not_sum = rsum(raw['buy_notional'], lb)

        # Thresholds: use percentiles of the actual data
        pm_abs = np.abs(np.nan_to_num(price_mom))
        pm_valid = pm_abs[pm_abs > 0]
        if len(pm_valid) < 100:
            continue

        ov_valid = oi_vel[~np.isnan(oi_vel)]
        if len(ov_valid) < 100:
            continue

        # Multiple threshold levels
        for pm_pct, ov_pct, label in [
            (75, 50, 'moderate'),
            (75, 75, 'strong'),
            (90, 75, 'extreme'),
        ]:
            pm_thresh = np.percentile(pm_valid, pm_pct)
            ov_thresh = np.percentile(np.abs(ov_valid), ov_pct)

            # Detect onset: price moving AND OI dropping
            # Long squeeze: price UP + OI DOWN
            long_trigger = (price_mom > pm_thresh) & (oi_vel < -ov_thresh)
            # Short squeeze: price DOWN + OI DOWN
            short_trigger = (price_mom < -pm_thresh) & (oi_vel < -ov_thresh)

            # Extract non-overlapping episodes with cooldown
            cooldown = lb * 2  # Minimum gap between episodes

            for direction, trigger, dir_name in [
                (1, long_trigger, 'long'),
                (-1, short_trigger, 'short'),
            ]:
                # Find trigger points
                trigger_idx = np.where(trigger)[0]
                if len(trigger_idx) == 0:
                    continue

                # Apply cooldown: keep only first trigger in each cluster
                episodes_idx = [trigger_idx[0]]
                for idx in trigger_idx[1:]:
                    if idx - episodes_idx[-1] >= cooldown:
                        episodes_idx.append(idx)

                for onset in episodes_idx:
                    if onset + 600 >= n:  # Need at least 600s forward
                        continue

                    ep = {
                        'onset': onset,
                        'onset_ts': ts_start + onset,
                        'direction': direction,
                        'dir_name': dir_name,
                        'lookback': lb,
                        'strength': label,
                        'price_mom': price_mom[onset],
                        'oi_vel': oi_vel[onset],
                        'price_at_onset': p[onset],
                        'oi_at_onset': oi[onset],
                        'liq_count_lb': liq_sum[onset],
                        'liq_notional_lb': liq_not_sum[onset],
                        'liq_buy_ratio': liq_buy_sum[onset] / max(liq_sum[onset], 1),
                        'trade_notional_lb': trade_not_sum[onset],
                        'buy_ratio_lb': buy_not_sum[onset] / max(trade_not_sum[onset], 1),
                    }

                    # Forward returns at multiple horizons
                    for fwd in [30, 60, 120, 300, 600]:
                        if onset + fwd < n and p[onset] > 0 and p[onset + fwd] > 0:
                            fwd_ret = (p[onset + fwd] - p[onset]) / p[onset]
                            ep[f'fwd_ret_{fwd}s'] = fwd_ret
                            ep[f'fwd_dir_ret_{fwd}s'] = fwd_ret * direction
                        else:
                            ep[f'fwd_ret_{fwd}s'] = np.nan
                            ep[f'fwd_dir_ret_{fwd}s'] = np.nan

                    # OI change forward (does OI keep dropping?)
                    for fwd in [60, 300]:
                        if onset + fwd < n and oi[onset] > 0 and oi[onset + fwd] > 0:
                            ep[f'fwd_oi_chg_{fwd}s'] = (oi[onset + fwd] - oi[onset]) / oi[onset]
                        else:
                            ep[f'fwd_oi_chg_{fwd}s'] = np.nan

                    # Liq count forward
                    for fwd in [60, 300]:
                        if onset + fwd < n:
                            ep[f'fwd_liq_count_{fwd}s'] = raw['liq_count'][onset:onset+fwd].sum()
                        else:
                            ep[f'fwd_liq_count_{fwd}s'] = np.nan

                    all_episodes.append(ep)

        gc.collect()

    episodes = pd.DataFrame(all_episodes)
    elapsed = time.time() - t0
    print(f"  Detected {len(episodes):,} squeeze episodes in {elapsed:.1f}s", flush=True)

    if len(episodes) > 0:
        print(f"\n  Episode breakdown:")
        for lb in episodes['lookback'].unique():
            for strength in episodes[episodes['lookback'] == lb]['strength'].unique():
                mask = (episodes['lookback'] == lb) & (episodes['strength'] == strength)
                sub = episodes[mask]
                n_long = (sub['direction'] == 1).sum()
                n_short = (sub['direction'] == -1).sum()
                print(f"    lb={lb:>3d}s {strength:>8s}: {len(sub):>4d} episodes "
                      f"(long={n_long}, short={n_short})")

    return episodes


# ============================================================================
# EPISODE ANALYSIS
# ============================================================================

def analyze_episodes(episodes):
    """Analyze squeeze episodes: continuation, duration, predictors."""
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"EPISODE CONTINUATION ANALYSIS")
    print(f"{'='*70}")

    if len(episodes) == 0:
        print("  No episodes to analyze")
        return

    fwd_cols = [c for c in episodes.columns if c.startswith('fwd_dir_ret_')]
    fwd_horizons = sorted(set(int(c.split('_')[-1].replace('s', '')) for c in fwd_cols))

    # Overall continuation by direction and strength
    for dir_name in ['long', 'short']:
        dir_eps = episodes[episodes['dir_name'] == dir_name]
        if len(dir_eps) < 5:
            continue

        print(f"\n  === {dir_name.upper()} SQUEEZE ({len(dir_eps)} episodes) ===")

        for strength in ['moderate', 'strong', 'extreme']:
            sub = dir_eps[dir_eps['strength'] == strength]
            if len(sub) < 5:
                continue

            print(f"\n    [{strength}] ({len(sub)} episodes)")
            print(f"    {'Fwd':>5s}  {'Cont%':>6s}  {'DirRet':>10s}  {'|Ret|':>8s}  {'Std':>8s}  "
                  f"{'OI chg':>8s}  {'Fwd Liqs':>8s}  {'N':>5s}")
            print(f"    {'-'*70}")

            for fwd in fwd_horizons:
                col = f'fwd_dir_ret_{fwd}s'
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna()
                if len(vals) < 5:
                    continue

                cont = (vals > 0).mean()
                mean_ret = vals.mean() * 10000
                abs_ret = vals.abs().mean() * 10000
                std_ret = vals.std() * 10000

                # OI change forward
                oi_col = f'fwd_oi_chg_{fwd}s' if f'fwd_oi_chg_{fwd}s' in sub.columns else None
                oi_chg = sub[oi_col].mean() * 10000 if oi_col and oi_col in sub.columns else np.nan

                # Liq count forward
                liq_col = f'fwd_liq_count_{fwd}s' if f'fwd_liq_count_{fwd}s' in sub.columns else None
                fwd_liqs = sub[liq_col].mean() if liq_col and liq_col in sub.columns else np.nan

                oi_str = f"{oi_chg:+7.1f}bp" if not np.isnan(oi_chg) else "     n/a"
                liq_str = f"{fwd_liqs:7.1f}" if not np.isnan(fwd_liqs) else "     n/a"

                print(f"    {fwd:>4d}s  {cont:5.1%}  {mean_ret:+9.2f}bps  {abs_ret:7.2f}bps  "
                      f"{std_ret:7.2f}bps  {oi_str}  {liq_str}  {len(vals):>4d}")

    # Lookback comparison
    print(f"\n  === LOOKBACK COMPARISON (all directions, 300s forward) ===")
    print(f"  {'LB':>4s}  {'Strength':>8s}  {'Cont%':>6s}  {'DirRet':>10s}  {'N':>5s}")
    print(f"  {'-'*45}")

    for lb in sorted(episodes['lookback'].unique()):
        for strength in ['moderate', 'strong', 'extreme']:
            sub = episodes[(episodes['lookback'] == lb) & (episodes['strength'] == strength)]
            vals = sub['fwd_dir_ret_300s'].dropna()
            if len(vals) < 5:
                continue
            cont = (vals > 0).mean()
            mean_ret = vals.mean() * 10000
            print(f"  {lb:>4d}  {strength:>8s}  {cont:5.1%}  {mean_ret:+9.2f}bps  {len(vals):>4d}")

    elapsed = time.time() - t0
    print(f"\n  Episode analysis done in {elapsed:.1f}s", flush=True)


# ============================================================================
# PREDICTOR ANALYSIS: What predicts squeeze strength?
# ============================================================================

def analyze_predictors(episodes):
    """What features at onset predict stronger continuation?"""
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"SQUEEZE STRENGTH PREDICTORS")
    print(f"{'='*70}")

    if len(episodes) < 20:
        print("  Not enough episodes for predictor analysis")
        return

    # Focus on 300s forward directional return as target
    target_col = 'fwd_dir_ret_300s'
    valid = episodes[target_col].notna()
    df = episodes[valid].copy()

    if len(df) < 20:
        print("  Not enough valid episodes")
        return

    # Predictors available at onset
    predictors = ['price_mom', 'oi_vel', 'liq_count_lb', 'liq_notional_lb',
                  'liq_buy_ratio', 'trade_notional_lb', 'buy_ratio_lb']

    target = df[target_col].values

    print(f"\n  Correlation with 300s forward directional return (n={len(df)}):")
    print(f"  {'Predictor':>25s}  {'Corr':>7s}  {'p-value':>8s}")
    print(f"  {'-'*45}")

    from scipy import stats

    for pred in predictors:
        if pred not in df.columns:
            continue
        x = df[pred].values
        valid_mask = ~np.isnan(x) & ~np.isnan(target)
        if valid_mask.sum() < 10:
            continue
        corr, pval = stats.pearsonr(x[valid_mask], target[valid_mask])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {pred:>25s}  {corr:+6.3f}  {pval:8.4f} {sig}")

    # Bin by OI velocity quintiles
    print(f"\n  Continuation by OI velocity quintile:")
    oi_vel = df['oi_vel'].values
    valid_oi = ~np.isnan(oi_vel)
    if valid_oi.sum() > 20:
        pcts = [0, 20, 40, 60, 80, 100]
        edges = np.percentile(oi_vel[valid_oi], pcts)
        print(f"  {'OI Vel Quintile':>25s}  {'Cont%':>6s}  {'DirRet':>10s}  {'N':>5s}")
        print(f"  {'-'*50}")
        for i in range(len(pcts) - 1):
            lo, hi = edges[i], edges[i+1]
            if i == len(pcts) - 2:
                mask = valid_oi & (oi_vel >= lo) & (oi_vel <= hi)
            else:
                mask = valid_oi & (oi_vel >= lo) & (oi_vel < hi)
            if mask.sum() < 3:
                continue
            rets = target[mask]
            cont = (rets > 0).mean()
            mean_ret = np.mean(rets) * 10000
            print(f"  Q{i+1} ({lo*10000:+.1f} to {hi*10000:+.1f}bps)  "
                  f"{cont:5.1%}  {mean_ret:+9.2f}bps  {mask.sum():>4d}")

    # Bin by liquidation count
    print(f"\n  Continuation by liquidation count in lookback:")
    liq_c = df['liq_count_lb'].values
    valid_liq = ~np.isnan(liq_c)
    if valid_liq.sum() > 20:
        # Use fixed bins: 0, 1-5, 6-20, 20+
        bins = [(0, 0, '0 liqs'), (1, 5, '1-5 liqs'), (6, 20, '6-20 liqs'), (21, 9999, '20+ liqs')]
        print(f"  {'Liq Count':>25s}  {'Cont%':>6s}  {'DirRet':>10s}  {'N':>5s}")
        print(f"  {'-'*50}")
        for lo, hi, label in bins:
            mask = valid_liq & (liq_c >= lo) & (liq_c <= hi)
            if mask.sum() < 3:
                continue
            rets = target[mask]
            cont = (rets > 0).mean()
            mean_ret = np.mean(rets) * 10000
            print(f"  {label:>25s}  {cont:5.1%}  {mean_ret:+9.2f}bps  {mask.sum():>4d}")

    elapsed = time.time() - t0
    print(f"\n  Predictor analysis done in {elapsed:.1f}s", flush=True)


# ============================================================================
# RANDOM BASELINE
# ============================================================================

def compute_baseline(raw, ts_start, n):
    """Compute baseline continuation probability for random entry points."""
    print(f"\n{'='*70}")
    print(f"RANDOM BASELINE")
    print(f"{'='*70}")

    p = raw['price_last'].astype(np.float64)

    # Sample random points (every 300s to match episode density)
    sample_idx = np.arange(300, n - 600, 300)
    np.random.seed(42)
    if len(sample_idx) > 5000:
        sample_idx = np.random.choice(sample_idx, 5000, replace=False)

    print(f"  Random sample: {len(sample_idx)} points")
    print(f"\n  {'Fwd':>5s}  {'Up%':>6s}  {'Mean Ret':>10s}  {'|Ret|':>8s}  {'Std':>8s}")
    print(f"  {'-'*45}")

    for fwd in [30, 60, 120, 300, 600]:
        rets = []
        for idx in sample_idx:
            if idx + fwd < n and p[idx] > 0 and p[idx + fwd] > 0:
                rets.append((p[idx + fwd] - p[idx]) / p[idx])
        rets = np.array(rets)
        if len(rets) < 10:
            continue
        up_pct = (rets > 0).mean()
        mean_ret = np.mean(rets) * 10000
        abs_ret = np.mean(np.abs(rets)) * 10000
        std_ret = np.std(rets) * 10000
        print(f"  {fwd:>4d}s  {up_pct:5.1%}  {mean_ret:+9.2f}bps  {abs_ret:7.2f}bps  {std_ret:7.2f}bps")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global SYMBOL, DATES

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=SYMBOL)
    parser.add_argument('--start', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', help='End date YYYY-MM-DD')
    parser.add_argument('--days', type=int, help='Number of days from start')
    args = parser.parse_args()

    SYMBOL = args.symbol
    if args.start:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(args.start, '%Y-%m-%d')
        end_dt = (datetime.strptime(args.end, '%Y-%m-%d') if args.end
                  else start_dt + timedelta(days=(args.days or 7) - 1))
        DATES = []
        dt = start_dt
        while dt <= end_dt:
            DATES.append(dt.strftime('%Y-%m-%d'))
            dt += timedelta(days=1)

    t_start = time.time()
    print(f"{'='*70}")
    print(f"v31c: Squeeze Episode Detection & Continuation")
    print(f"{'='*70}")
    print(f"Symbol: {SYMBOL}")
    print(f"Dates:  {DATES[0]} to {DATES[-1]} ({len(DATES)} days)")
    print_mem("start")

    result = build_arrays(DATES)
    if result is None:
        return
    raw, ts_start, n = result

    # Detect episodes
    episodes = detect_squeeze_episodes(raw, ts_start, n)

    # Random baseline
    compute_baseline(raw, ts_start, n)

    # Analyze episodes
    analyze_episodes(episodes)

    # Predictor analysis
    analyze_predictors(episodes)

    # Save
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {fmt_time(total_time)}")
    print_mem("final")

    date_tag = f"{DATES[0]}_to_{DATES[-1]}"
    output_file = RESULTS_DIR / f"v31c_squeeze_episodes_{SYMBOL}_{date_tag}.txt"

    import io
    buf = io.StringIO()
    buf.write(f"v31c: Squeeze Episode Detection — {SYMBOL}\n")
    buf.write(f"{'='*70}\n")
    buf.write(f"Dates: {DATES[0]} to {DATES[-1]} ({len(DATES)} days)\n\n")

    if len(episodes) > 0:
        buf.write(f"EPISODES DETECTED: {len(episodes)}\n\n")

        for dir_name in ['long', 'short']:
            dir_eps = episodes[episodes['dir_name'] == dir_name]
            if len(dir_eps) < 5:
                continue
            buf.write(f"\n{dir_name.upper()} SQUEEZE ({len(dir_eps)} episodes):\n")
            for strength in ['moderate', 'strong', 'extreme']:
                sub = dir_eps[dir_eps['strength'] == strength]
                if len(sub) < 5:
                    continue
                buf.write(f"  [{strength}]:\n")
                for fwd in [30, 60, 120, 300, 600]:
                    col = f'fwd_dir_ret_{fwd}s'
                    vals = sub[col].dropna()
                    if len(vals) < 3:
                        continue
                    cont = (vals > 0).mean()
                    mean_ret = vals.mean() * 10000
                    buf.write(f"    fwd={fwd:>4d}s: cont={cont:.1%} ret={mean_ret:+.1f}bps n={len(vals)}\n")

    with open(output_file, 'w') as f:
        f.write(buf.getvalue())
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
