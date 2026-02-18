#!/usr/bin/env python3
"""
v31b: OI Divergence as Squeeze Continuation Signal

HYPOTHESIS:
  When price moves directionally AND OI drops simultaneously, this signals
  forced liquidation (not voluntary positioning). The positive feedback loop
  (price up → shorts liquidated → forced buys → price up) creates predictable
  continuation until liquidations exhaust.

SIGNAL:
  price_momentum > +threshold  AND  oi_velocity < -threshold  → long continuation
  price_momentum < -threshold  AND  oi_velocity < -threshold  → short continuation

TESTS:
  1. Raw continuation probability: P(next N bars same direction | signal)
  2. Varying lookback windows for momentum/OI (30s, 60s, 120s, 300s)
  3. Varying forward horizons (30s, 60s, 120s, 300s, 600s)
  4. Compare: OI-divergence signal vs pure momentum vs random
  5. Strength: does stronger OI drop → stronger continuation?
  6. Asymmetry: long squeeze vs short squeeze differences

DATA: BTCUSDT tick-level (trades + ticker), 7 days first, then expand.
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
DATES = [f"2025-05-{d:02d}" for d in range(12, 19)]  # 7 days

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Lookback windows for signal detection (seconds)
LOOKBACKS = [30, 60, 120, 300]
# Forward horizons for measuring continuation (seconds)
FORWARDS = [30, 60, 120, 300, 600]


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
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds/60:.1f}m"


# ============================================================================
# DATA LOADING — trades (per-second, day by day)
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

    day_start = int(ts_s.min())
    day_end = int(ts_s.max())
    n = day_end - day_start + 1
    off = (ts_s - day_start).astype(np.int32)

    trade_count = np.bincount(off, minlength=n).astype(np.float32)
    trade_notional = np.bincount(off, weights=notionals, minlength=n).astype(np.float32)
    buy_notional = np.bincount(off, weights=notionals * is_buy, minlength=n).astype(np.float32)

    # Last price per second
    price_last = np.full(n, np.nan, dtype=np.float32)
    _, last_idx = np.unique(off[::-1], return_index=True)
    last_idx = len(off) - 1 - last_idx
    for uo, li in zip(np.unique(off), last_idx):
        price_last[uo] = prices[li]

    del ts_s, prices, sizes, notionals, is_buy, off; gc.collect()
    return {
        'day_start': day_start, 'n': n, 'n_raw': n_raw,
        'trade_count': trade_count, 'trade_notional': trade_notional,
        'buy_notional': buy_notional,
        'price_last': price_last,
    }


# ============================================================================
# DATA LOADING — ticker (OI only, RAM-efficient)
# ============================================================================

def load_ticker_oi(dates, ts_start, n):
    """Load only OI and liquidation-relevant fields from ticker."""
    arrays = {
        'oi': np.full(n, np.nan, dtype=np.float32),
        'fr': np.full(n, np.nan, dtype=np.float32),
    }
    count = 0
    for date_str in dates:
        files = sorted((DATA_DIR / SYMBOL / "bybit" / "ticker").glob(
            f"ticker_{date_str}_hr*.jsonl.gz"))
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
                        if 'openInterestValue' in data:
                            arrays['oi'][idx] = float(data['openInterestValue'])
                        if 'fundingRate' in data:
                            arrays['fr'][idx] = float(data['fundingRate'])
                        count += 1
                    except:
                        continue
        print(f"    {date_str}: loaded ticker", flush=True)

    # Forward-fill
    for k in arrays:
        arr = arrays[k]
        mask = np.isnan(arr)
        if not mask.all():
            fv = np.argmin(mask)
            for i in range(fv + 1, n):
                if mask[i]:
                    arr[i] = arr[i - 1]

    print(f"  Ticker: {count:,} updates → {n:,}s arrays", flush=True)
    return arrays


# ============================================================================
# DATA LOADING — liquidations
# ============================================================================

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
                                records.append((
                                    int(ev['T']) // 1000,
                                    1.0 if ev['S'] == 'Buy' else 0.0,
                                    float(ev['v']) * float(ev['p']),
                                ))
                                day_count += 1
                    except:
                        continue
        print(f"  {date_str}: {day_count:>6,} liquidations", flush=True)
    if not records:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(records)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.float32), arr[:, 2].astype(np.float32)


# ============================================================================
# BUILD ARRAYS
# ============================================================================

def build_arrays(dates):
    t0 = time.time()
    print_mem("start")

    # Liquidations
    print(f"\n[1/3] Loading liquidations...", flush=True)
    liq_ts, liq_is_buy, liq_not = load_liquidations(dates)
    print(f"  Total: {len(liq_ts):,} liquidation events", flush=True)

    # Trades
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

    if not day_results:
        return None

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n'] - 1
    n = ts_end - ts_start + 1

    raw = {
        'trade_count': np.zeros(n, dtype=np.float32),
        'trade_notional': np.zeros(n, dtype=np.float32),
        'buy_notional': np.zeros(n, dtype=np.float32),
        'price_last': np.full(n, np.nan, dtype=np.float32),
    }

    for dr in day_results:
        o = dr['day_start'] - ts_start
        l = dr['n']
        for k in ['trade_count', 'trade_notional', 'buy_notional']:
            raw[k][o:o+l] = dr[k]
        raw['price_last'][o:o+l] = dr['price_last']
    del day_results; gc.collect()

    # Forward-fill price
    p = raw['price_last']
    mask = np.isnan(p)
    if not mask.all():
        fv = np.argmin(mask)
        for i in range(fv + 1, n):
            if mask[i]:
                p[i] = p[i - 1]

    # Liquidations per-second
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

    # Ticker (OI + FR)
    print(f"\n[3/3] Loading ticker (OI/FR)...", flush=True)
    ticker = load_ticker_oi(dates, ts_start, n)
    raw.update(ticker)
    del ticker; gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Built {n:,} seconds ({n/86400:.1f} days) in {elapsed:.1f}s", flush=True)
    print_mem("arrays done")
    return raw, ts_start, n


# ============================================================================
# CORE ANALYSIS: OI DIVERGENCE CONTINUATION
# ============================================================================

def analyze_oi_divergence(raw, ts_start, n):
    """
    Main analysis: test OI divergence as continuation signal.

    For each lookback window:
      1. Compute price_momentum = (price_now - price_Nago) / price_Nago
      2. Compute oi_velocity = (OI_now - OI_Nago) / OI_Nago
      3. Identify divergence events: |price_mom| > threshold AND oi_vel < -threshold
      4. Measure forward continuation probability and return
    """
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"OI DIVERGENCE CONTINUATION ANALYSIS")
    print(f"{'='*70}")

    p = raw['price_last'].astype(np.float64)
    oi = raw['oi'].astype(np.float64)

    results = {}

    for lb in LOOKBACKS:
        print(f"\n  --- Lookback: {lb}s ---", flush=True)

        # Price momentum (% change over lookback)
        price_mom = np.full(n, np.nan, dtype=np.float64)
        valid = (p[lb:] > 0) & (p[:-lb] > 0)
        price_mom[lb:][valid] = (p[lb:][valid] - p[:-lb][valid]) / p[:-lb][valid]

        # OI velocity (% change over lookback)
        oi_vel = np.full(n, np.nan, dtype=np.float64)
        valid_oi = (oi[lb:] > 0) & (oi[:-lb] > 0)
        oi_vel[lb:][valid_oi] = (oi[lb:][valid_oi] - oi[:-lb][valid_oi]) / oi[:-lb][valid_oi]

        # Rolling liquidation count over lookback
        liq_cs = np.cumsum(raw['liq_count'].astype(np.float64))
        liq_sum = np.zeros(n, dtype=np.float64)
        liq_sum[lb:] = liq_cs[lb:] - liq_cs[:-lb]

        # Rolling trade notional
        tn_cs = np.cumsum(raw['trade_notional'].astype(np.float64))
        tn_sum = np.zeros(n, dtype=np.float64)
        tn_sum[lb:] = tn_cs[lb:] - tn_cs[:-lb]

        # Adaptive thresholds based on distribution
        pm_abs = np.abs(np.nan_to_num(price_mom))
        ov_abs = np.abs(np.nan_to_num(oi_vel))

        # Use percentile thresholds
        pm_p50 = np.nanpercentile(pm_abs[pm_abs > 0], 50)
        pm_p75 = np.nanpercentile(pm_abs[pm_abs > 0], 75)
        ov_p50 = np.nanpercentile(ov_abs[ov_abs > 0], 50)
        ov_p75 = np.nanpercentile(ov_abs[ov_abs > 0], 75)

        print(f"    Price momentum |%|: P50={pm_p50*10000:.1f}bps  P75={pm_p75*10000:.1f}bps")
        print(f"    OI velocity |%|:    P50={ov_p50*10000:.1f}bps  P75={ov_p75*10000:.1f}bps")

        # Test multiple threshold combinations
        for pm_thresh_name, pm_thresh in [('P50', pm_p50), ('P75', pm_p75)]:
            for ov_thresh_name, ov_thresh in [('P50', ov_p50), ('P75', ov_p75)]:

                # DIVERGENCE: price moving AND OI dropping
                # Long squeeze: price UP, OI DOWN
                long_squeeze = (price_mom > pm_thresh) & (oi_vel < -ov_thresh)
                # Short squeeze: price DOWN, OI DOWN
                short_squeeze = (price_mom < -pm_thresh) & (oi_vel < -ov_thresh)
                # Combined (either direction)
                any_squeeze = long_squeeze | short_squeeze
                # Direction sign
                direction = np.where(long_squeeze, 1.0, np.where(short_squeeze, -1.0, 0.0))

                # BASELINE: pure momentum (no OI condition)
                long_mom = price_mom > pm_thresh
                short_mom = price_mom < -pm_thresh
                any_mom = long_mom | short_mom
                mom_direction = np.where(long_mom, 1.0, np.where(short_mom, -1.0, 0.0))

                # ANTI-SIGNAL: price moving AND OI RISING (new positions, not forced)
                long_anti = (price_mom > pm_thresh) & (oi_vel > ov_thresh)
                short_anti = (price_mom < -pm_thresh) & (oi_vel > ov_thresh)
                any_anti = long_anti | short_anti
                anti_direction = np.where(long_anti, 1.0, np.where(short_anti, -1.0, 0.0))

                n_squeeze = any_squeeze.sum()
                n_mom = any_mom.sum()
                n_anti = any_anti.sum()

                if n_squeeze < 10:
                    continue

                label = f"lb{lb}_pm{pm_thresh_name}_ov{ov_thresh_name}"

                print(f"\n    [{label}] Squeeze events: {n_squeeze:,}  "
                      f"(long={long_squeeze.sum():,}, short={short_squeeze.sum():,})")
                print(f"    [{label}] Momentum-only:  {n_mom:,}  |  Anti-signal: {n_anti:,}")

                # Measure forward continuation for each forward horizon
                for fwd in FORWARDS:
                    if fwd + lb >= n:
                        continue

                    # Forward return
                    fwd_ret = np.full(n, np.nan, dtype=np.float64)
                    valid_fwd = p[fwd:] > 0
                    fwd_ret[:-fwd][valid_fwd] = (p[fwd:][valid_fwd] - p[:-fwd][valid_fwd]) / p[:-fwd][valid_fwd]

                    # Directional forward return (positive = continuation)
                    dir_fwd_ret_squeeze = fwd_ret * direction
                    dir_fwd_ret_mom = fwd_ret * mom_direction
                    dir_fwd_ret_anti = fwd_ret * anti_direction

                    # Squeeze signal
                    sq_mask = any_squeeze & ~np.isnan(fwd_ret)
                    if sq_mask.sum() < 5:
                        continue
                    sq_dir_ret = dir_fwd_ret_squeeze[sq_mask]
                    sq_cont_prob = (sq_dir_ret > 0).mean()
                    sq_mean_ret = np.nanmean(sq_dir_ret) * 10000  # bps
                    sq_mean_abs = np.nanmean(np.abs(fwd_ret[sq_mask])) * 10000

                    # Momentum baseline
                    mom_mask = any_mom & ~np.isnan(fwd_ret)
                    mom_dir_ret = dir_fwd_ret_mom[mom_mask]
                    mom_cont_prob = (mom_dir_ret > 0).mean() if mom_mask.sum() > 0 else np.nan
                    mom_mean_ret = np.nanmean(mom_dir_ret) * 10000 if mom_mask.sum() > 0 else np.nan

                    # Anti-signal
                    anti_mask = any_anti & ~np.isnan(fwd_ret)
                    anti_dir_ret = dir_fwd_ret_anti[anti_mask]
                    anti_cont_prob = (anti_dir_ret > 0).mean() if anti_mask.sum() > 0 else np.nan
                    anti_mean_ret = np.nanmean(anti_dir_ret) * 10000 if anti_mask.sum() > 0 else np.nan

                    # Liq count during squeeze events
                    sq_liq = liq_sum[sq_mask]
                    avg_liq = np.mean(sq_liq)

                    edge = sq_cont_prob - mom_cont_prob if not np.isnan(mom_cont_prob) else 0

                    print(f"      fwd={fwd:>4d}s: "
                          f"SQUEEZE cont={sq_cont_prob:.1%} ret={sq_mean_ret:+.1f}bps (n={sq_mask.sum():>5d}) | "
                          f"MOM cont={mom_cont_prob:.1%} ret={mom_mean_ret:+.1f}bps (n={mom_mask.sum():>5d}) | "
                          f"ANTI cont={anti_cont_prob:.1%} ret={anti_mean_ret:+.1f}bps (n={anti_mask.sum():>5d}) | "
                          f"edge={edge:+.1%} | avg_liq={avg_liq:.1f}")

                    results[(lb, pm_thresh_name, ov_thresh_name, fwd)] = {
                        'n_squeeze': sq_mask.sum(),
                        'sq_cont_prob': sq_cont_prob,
                        'sq_mean_ret': sq_mean_ret,
                        'mom_cont_prob': mom_cont_prob,
                        'mom_mean_ret': mom_mean_ret,
                        'anti_cont_prob': anti_cont_prob,
                        'anti_mean_ret': anti_mean_ret,
                        'edge': edge,
                        'avg_liq': avg_liq,
                    }

        gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Analysis done in {elapsed:.1f}s", flush=True)
    return results


# ============================================================================
# STRENGTH ANALYSIS: Does stronger OI drop → stronger continuation?
# ============================================================================

def analyze_oi_strength(raw, ts_start, n):
    """Bin OI velocity into quintiles and measure continuation by bin."""
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"OI VELOCITY STRENGTH ANALYSIS")
    print(f"{'='*70}")

    p = raw['price_last'].astype(np.float64)
    oi = raw['oi'].astype(np.float64)

    # Use 120s lookback (good balance of signal/noise from initial tests)
    lb = 120
    fwd = 300  # 5 min forward

    price_mom = np.full(n, np.nan, dtype=np.float64)
    valid = (p[lb:] > 0) & (p[:-lb] > 0)
    price_mom[lb:][valid] = (p[lb:][valid] - p[:-lb][valid]) / p[:-lb][valid]

    oi_vel = np.full(n, np.nan, dtype=np.float64)
    valid_oi = (oi[lb:] > 0) & (oi[:-lb] > 0)
    oi_vel[lb:][valid_oi] = (oi[lb:][valid_oi] - oi[:-lb][valid_oi]) / oi[:-lb][valid_oi]

    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    valid_fwd = p[fwd:] > 0
    fwd_ret[:-fwd][valid_fwd] = (p[fwd:][valid_fwd] - p[:-fwd][valid_fwd]) / p[:-fwd][valid_fwd]

    # Direction
    direction = np.sign(price_mom)
    dir_fwd_ret = fwd_ret * direction

    # Filter: only when price is moving (|mom| > P50)
    pm_abs = np.abs(np.nan_to_num(price_mom))
    pm_thresh = np.nanpercentile(pm_abs[pm_abs > 0], 50)
    moving = pm_abs > pm_thresh

    # Valid mask
    valid_all = moving & ~np.isnan(oi_vel) & ~np.isnan(dir_fwd_ret)
    if valid_all.sum() < 100:
        print("  Not enough data for strength analysis")
        return

    oi_vel_valid = oi_vel[valid_all]
    dir_ret_valid = dir_fwd_ret[valid_all]

    # Bin OI velocity into quintiles
    pcts = [0, 10, 25, 50, 75, 90, 100]
    edges = np.nanpercentile(oi_vel_valid, pcts)

    print(f"\n  Lookback={lb}s, Forward={fwd}s, Price moving (|mom|>P50)")
    print(f"  OI velocity percentiles: {[f'{e*10000:.1f}bps' for e in edges]}")
    print(f"\n  {'OI Vel Range':>25s}  {'Cont%':>6s}  {'Mean Ret':>10s}  {'|Ret|':>8s}  {'N':>7s}")
    print(f"  {'-'*65}")

    for i in range(len(pcts) - 1):
        lo, hi = edges[i], edges[i+1]
        if i == len(pcts) - 2:
            mask = (oi_vel_valid >= lo) & (oi_vel_valid <= hi)
        else:
            mask = (oi_vel_valid >= lo) & (oi_vel_valid < hi)

        if mask.sum() < 5:
            continue

        rets = dir_ret_valid[mask]
        cont = (rets > 0).mean()
        mean_ret = np.mean(rets) * 10000
        abs_ret = np.mean(np.abs(rets)) * 10000

        label = f"P{pcts[i]:02d}-P{pcts[i+1]:02d} ({lo*10000:+.1f} to {hi*10000:+.1f}bps)"
        print(f"  {label:>25s}  {cont:5.1%}  {mean_ret:+9.2f}bps  {abs_ret:7.2f}bps  {mask.sum():>6d}")

    elapsed = time.time() - t0
    print(f"\n  Strength analysis done in {elapsed:.1f}s", flush=True)


# ============================================================================
# ASYMMETRY: Long squeeze vs Short squeeze
# ============================================================================

def analyze_asymmetry(raw, ts_start, n):
    """Compare long squeeze (shorts liquidated) vs short squeeze (longs liquidated)."""
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"LONG vs SHORT SQUEEZE ASYMMETRY")
    print(f"{'='*70}")

    p = raw['price_last'].astype(np.float64)
    oi = raw['oi'].astype(np.float64)

    lb = 120
    price_mom = np.full(n, np.nan, dtype=np.float64)
    valid = (p[lb:] > 0) & (p[:-lb] > 0)
    price_mom[lb:][valid] = (p[lb:][valid] - p[:-lb][valid]) / p[:-lb][valid]

    oi_vel = np.full(n, np.nan, dtype=np.float64)
    valid_oi = (oi[lb:] > 0) & (oi[:-lb] > 0)
    oi_vel[lb:][valid_oi] = (oi[lb:][valid_oi] - oi[:-lb][valid_oi]) / oi[:-lb][valid_oi]

    # Liquidation data
    liq_cs = np.cumsum(raw['liq_count'].astype(np.float64))
    liq_sum = np.zeros(n, dtype=np.float64)
    liq_sum[lb:] = liq_cs[lb:] - liq_cs[:-lb]

    liq_buy_cs = np.cumsum(raw['liq_buy_count'].astype(np.float64))
    liq_buy_sum = np.zeros(n, dtype=np.float64)
    liq_buy_sum[lb:] = liq_buy_cs[lb:] - liq_buy_cs[:-lb]

    liq_not_cs = np.cumsum(raw['liq_notional'].astype(np.float64))
    liq_not_sum = np.zeros(n, dtype=np.float64)
    liq_not_sum[lb:] = liq_not_cs[lb:] - liq_not_cs[:-lb]

    pm_abs = np.abs(np.nan_to_num(price_mom))
    pm_p75 = np.nanpercentile(pm_abs[pm_abs > 0], 75)
    ov_abs = np.abs(np.nan_to_num(oi_vel))
    ov_p50 = np.nanpercentile(ov_abs[ov_abs > 0], 50)

    # Long squeeze: price UP + OI DOWN (shorts being liquidated)
    long_sq = (price_mom > pm_p75) & (oi_vel < -ov_p50)
    # Short squeeze: price DOWN + OI DOWN (longs being liquidated)
    short_sq = (price_mom < -pm_p75) & (oi_vel < -ov_p50)

    print(f"\n  Lookback={lb}s, Thresholds: price>P75, OI_vel<-P50")
    print(f"  Long squeeze events:  {long_sq.sum():,}")
    print(f"  Short squeeze events: {short_sq.sum():,}")

    for label, mask in [("LONG SQUEEZE (shorts liq'd)", long_sq),
                        ("SHORT SQUEEZE (longs liq'd)", short_sq)]:
        if mask.sum() < 10:
            print(f"\n  {label}: too few events ({mask.sum()})")
            continue

        print(f"\n  {label} ({mask.sum():,} events):")

        # Liquidation characteristics
        avg_liq = np.mean(liq_sum[mask])
        avg_liq_not = np.mean(liq_not_sum[mask])
        liq_buy_ratio = np.mean(liq_buy_sum[mask] / np.maximum(liq_sum[mask], 1))

        print(f"    Avg liquidations in window: {avg_liq:.1f} events, ${avg_liq_not:,.0f} notional")
        print(f"    Liq buy ratio: {liq_buy_ratio:.1%}")

        print(f"\n    {'Fwd':>5s}  {'Cont%':>6s}  {'Mean Ret':>10s}  {'|Ret|':>8s}  {'Std':>8s}  {'N':>6s}")
        print(f"    {'-'*52}")

        for fwd in FORWARDS:
            fwd_ret = np.full(n, np.nan, dtype=np.float64)
            valid_fwd = p[fwd:] > 0
            fwd_ret[:-fwd][valid_fwd] = (p[fwd:][valid_fwd] - p[:-fwd][valid_fwd]) / p[:-fwd][valid_fwd]

            direction = np.sign(price_mom)
            dir_fwd = fwd_ret * direction

            sq_valid = mask & ~np.isnan(dir_fwd)
            if sq_valid.sum() < 5:
                continue

            rets = dir_fwd[sq_valid]
            cont = (rets > 0).mean()
            mean_ret = np.mean(rets) * 10000
            abs_ret = np.mean(np.abs(rets)) * 10000
            std_ret = np.std(rets) * 10000

            print(f"    {fwd:>4d}s  {cont:5.1%}  {mean_ret:+9.2f}bps  {abs_ret:7.2f}bps  {std_ret:7.2f}bps  {sq_valid.sum():>5d}")

    elapsed = time.time() - t0
    print(f"\n  Asymmetry analysis done in {elapsed:.1f}s", flush=True)


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
        if args.end:
            end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        elif args.days:
            end_dt = start_dt + timedelta(days=args.days - 1)
        else:
            end_dt = start_dt + timedelta(days=6)
        DATES = []
        dt = start_dt
        while dt <= end_dt:
            DATES.append(dt.strftime('%Y-%m-%d'))
            dt += timedelta(days=1)

    t_start = time.time()
    print(f"{'='*70}")
    print(f"v31b: OI Divergence as Squeeze Continuation Signal")
    print(f"{'='*70}")
    print(f"Symbol:   {SYMBOL}")
    print(f"Dates:    {DATES[0]} to {DATES[-1]} ({len(DATES)} days)")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Forwards:  {FORWARDS}")
    print_mem("start")

    # Build arrays
    result = build_arrays(DATES)
    if result is None:
        return
    raw, ts_start, n = result

    # Core analysis: OI divergence continuation
    results = analyze_oi_divergence(raw, ts_start, n)

    # Strength analysis: does stronger OI drop → stronger continuation?
    analyze_oi_strength(raw, ts_start, n)

    # Asymmetry: long vs short squeeze
    analyze_asymmetry(raw, ts_start, n)

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"SUMMARY — Best configurations")
    print(f"{'='*70}")

    # Find best edge configs
    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('edge', 0), reverse=True)
        print(f"\n  Top 10 by edge (squeeze cont% - momentum cont%):")
        print(f"  {'Config':>35s}  {'Squeeze':>8s}  {'Momentum':>8s}  {'Edge':>7s}  {'Ret':>8s}  {'N':>6s}")
        print(f"  {'-'*80}")
        for (lb, pm, ov, fwd), r in sorted_results[:10]:
            print(f"  lb={lb:>3d} pm={pm} ov={ov} fwd={fwd:>3d}  "
                  f"{r['sq_cont_prob']:7.1%}  {r['mom_cont_prob']:7.1%}  "
                  f"{r['edge']:+6.1%}  {r['sq_mean_ret']:+7.1f}bps  {r['n_squeeze']:>5d}")

        # Best by absolute return
        sorted_by_ret = sorted(results.items(), key=lambda x: x[1].get('sq_mean_ret', 0), reverse=True)
        print(f"\n  Top 10 by mean directional return:")
        print(f"  {'Config':>35s}  {'Squeeze':>8s}  {'Momentum':>8s}  {'Ret':>8s}  {'Anti':>8s}  {'N':>6s}")
        print(f"  {'-'*80}")
        for (lb, pm, ov, fwd), r in sorted_by_ret[:10]:
            print(f"  lb={lb:>3d} pm={pm} ov={ov} fwd={fwd:>3d}  "
                  f"{r['sq_cont_prob']:7.1%}  {r['mom_cont_prob']:7.1%}  "
                  f"{r['sq_mean_ret']:+7.1f}bps  {r['anti_mean_ret']:+7.1f}bps  {r['n_squeeze']:>5d}")

    print(f"\nTOTAL TIME: {fmt_time(total_time)}")
    print_mem("final")

    # Save results
    date_tag = f"{DATES[0]}_to_{DATES[-1]}"
    output_file = RESULTS_DIR / f"v31b_oi_squeeze_{SYMBOL}_{date_tag}.txt"

    import io
    # Re-run summary to file
    buf = io.StringIO()
    buf.write(f"v31b: OI Divergence Squeeze Continuation — {SYMBOL}\n")
    buf.write(f"{'='*70}\n")
    buf.write(f"Dates: {DATES[0]} to {DATES[-1]} ({len(DATES)} days)\n")
    buf.write(f"Time: {fmt_time(total_time)}\n\n")

    if results:
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('edge', 0), reverse=True)
        buf.write(f"TOP CONFIGURATIONS BY EDGE:\n")
        for (lb, pm, ov, fwd), r in sorted_results[:15]:
            buf.write(f"  lb={lb:>3d} pm={pm} ov={ov} fwd={fwd:>3d}: "
                      f"squeeze={r['sq_cont_prob']:.1%} mom={r['mom_cont_prob']:.1%} "
                      f"edge={r['edge']:+.1%} ret={r['sq_mean_ret']:+.1f}bps "
                      f"anti={r['anti_mean_ret']:+.1f}bps n={r['n_squeeze']}\n")

        buf.write(f"\nTOP CONFIGURATIONS BY RETURN:\n")
        sorted_by_ret = sorted(results.items(), key=lambda x: x[1].get('sq_mean_ret', 0), reverse=True)
        for (lb, pm, ov, fwd), r in sorted_by_ret[:15]:
            buf.write(f"  lb={lb:>3d} pm={pm} ov={ov} fwd={fwd:>3d}: "
                      f"squeeze={r['sq_cont_prob']:.1%} ret={r['sq_mean_ret']:+.1f}bps "
                      f"anti={r['anti_mean_ret']:+.1f}bps n={r['n_squeeze']}\n")

    with open(output_file, 'w') as f:
        f.write(buf.getvalue())
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
