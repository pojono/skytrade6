#!/usr/bin/env python3
"""
Analyse price drift BEFORE settlement for extreme FR coins.

Question: Does price drift up (anticipating FR payment) before settlement?
If yes → limit buy is risky (price runs away from bid)
If no/random → limit buy at bid has good fill probability

Uses Bybit 5s ticker data with ms-precision timestamps.
Measures price movement in windows: T-30s, T-20s, T-10s, T-5s, T-2s, T-1s
relative to settlement for coins with extreme negative FR.
"""
import sys
import time
import gc
from pathlib import Path

import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data_all"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
MIN_FR_BPS = 20.0      # minimum |FR| to consider extreme
FR_BUCKETS = [20, 40, 60, 80, 100]  # analyse by FR magnitude

print("=" * 100)
print("ANALYSE: Pre-Settlement Price Drift for Extreme FR Coins")
print("=" * 100)
print(f"  Min |FR|: {MIN_FR_BPS} bps")
print(f"  FR buckets: {FR_BUCKETS}")
print()

t_global = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Build settlement schedule from Binance FR data
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 1: Building settlement schedule...")
sys.stdout.flush()
t1 = time.time()

bn_fr = pd.read_parquet(DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "nextFundingTime"])
print(f"  Loaded Binance FR: {len(bn_fr):,} rows [{time.time()-t1:.1f}s]")
sys.stdout.flush()

# Downsample to 1-min
bn_fr["ts_1m"] = bn_fr["ts"].dt.floor("1min")
bn_fr_1m = bn_fr.groupby(["ts_1m", "symbol"]).agg(
    fr=("lastFundingRate", "last"),
    nft=("nextFundingTime", "last"),
).reset_index()
del bn_fr; gc.collect()

# Detect settlement: nextFundingTime changes
bn_fr_1m = bn_fr_1m.sort_values(["symbol", "ts_1m"])
bn_fr_1m["nft_prev"] = bn_fr_1m.groupby("symbol")["nft"].shift(1)
bn_fr_1m["is_settle"] = (bn_fr_1m["nft"] != bn_fr_1m["nft_prev"]) & bn_fr_1m["nft_prev"].notna()
bn_fr_1m["fr_prev"] = bn_fr_1m.groupby("symbol")["fr"].shift(1)

settle_rows = bn_fr_1m[bn_fr_1m["is_settle"]].copy()
settle_rows = settle_rows.rename(columns={"ts_1m": "settle_time"})
settle_rows["fr_paid_bps"] = settle_rows["fr_prev"] * 10000  # signed

settle_df = settle_rows[["settle_time", "symbol", "fr_paid_bps"]].dropna().copy()
settle_df = settle_df.sort_values("settle_time").reset_index(drop=True)

# Only negative FR (we go LONG to collect)
settle_neg = settle_df[settle_df["fr_paid_bps"] <= -MIN_FR_BPS].copy()
settle_neg["fr_abs_bps"] = settle_neg["fr_paid_bps"].abs()

print(f"  Total settlements: {len(settle_df):,}")
print(f"  Negative FR <= -{MIN_FR_BPS} bps: {len(settle_neg):,}")
print(f"  Unique coins: {settle_neg['symbol'].nunique()}")
print(f"  Date range: {settle_df['settle_time'].min()} to {settle_df['settle_time'].max()}")
print(f"  Phase 1 done [{time.time()-t1:.1f}s]")
print()
sys.stdout.flush()

del bn_fr_1m; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Load Bybit tick data
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 2: Loading Bybit tick data...")
sys.stdout.flush()
t2 = time.time()

_bb = pd.read_parquet(DATA / "bybit" / "ticker.parquet",
    columns=["ts", "symbol", "lastPrice", "bid1Price", "ask1Price"])
bb_ts = _bb["ts"].values
bb_sym = _bb["symbol"].values.astype(str)
bb_last = _bb["lastPrice"].values
bb_bid = _bb["bid1Price"].values
bb_ask = _bb["ask1Price"].values
del _bb; gc.collect()
print(f"  Bybit ticker: {len(bb_ts):,} rows [{time.time()-t2:.1f}s]")
print()
sys.stdout.flush()


def get_ticks(symbol, settle_time_ns, before_s=35, after_s=10):
    """Get all tick data for a symbol in a window around settlement."""
    t_lo = settle_time_ns - np.timedelta64(before_s, "s")
    t_hi = settle_time_ns + np.timedelta64(after_s, "s")
    i0 = np.searchsorted(bb_ts, t_lo, side="left")
    i1 = np.searchsorted(bb_ts, t_hi, side="right")
    if i0 >= i1:
        return None
    sl = slice(i0, i1)
    mask = bb_sym[sl] == symbol
    if mask.sum() == 0:
        return None
    ts = bb_ts[sl][mask]
    return {
        "offset_ms": (ts - settle_time_ns) / np.timedelta64(1, "ms"),
        "last": bb_last[sl][mask],
        "bid": bb_bid[sl][mask],
        "ask": bb_ask[sl][mask],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Measure pre-settlement price drift
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 3: Measuring pre-settlement price drift...")
sys.stdout.flush()
t3 = time.time()

# Time windows to measure (ms before settlement)
WINDOWS_MS = [-30000, -20000, -10000, -5000, -2000, -1000, -500, 0, 500, 1000, 2000, 5000]

# For each settlement, measure mid-price at each window relative to T-10s anchor
records = []
n_no_data = 0
n_processed = 0

for idx, row in settle_neg.iterrows():
    n_processed += 1
    if n_processed % 200 == 0 or n_processed == len(settle_neg):
        elapsed = time.time() - t3
        rate = n_processed / elapsed if elapsed > 0 else 0
        eta = (len(settle_neg) - n_processed) / rate if rate > 0 else 0
        print(f"  [{n_processed:,}/{len(settle_neg):,}] {elapsed:.0f}s elapsed, "
              f"~{eta:.0f}s ETA, no_data={n_no_data}", flush=True)

    symbol = row["symbol"]
    settle_time = row["settle_time"]
    fr_abs = row["fr_abs_bps"]
    settle_ns = np.datetime64(settle_time, "ns")

    data = get_ticks(symbol, settle_ns, before_s=35, after_s=10)
    if data is None:
        n_no_data += 1
        continue

    offsets = data["offset_ms"]
    mids = (data["bid"] + data["ask"]) / 2
    bids = data["bid"]
    asks = data["ask"]

    # Anchor: mid-price at T-30s (or closest available tick before T-25s)
    anchor_mask = (offsets >= -32000) & (offsets <= -28000)
    if anchor_mask.sum() == 0:
        # Try any tick before T-20s
        anchor_mask = offsets <= -20000
    if anchor_mask.sum() == 0:
        n_no_data += 1
        continue

    anchor_idx = np.where(anchor_mask)[0][-1]  # last tick in anchor window
    anchor_mid = float(mids[anchor_idx])
    anchor_bid = float(bids[anchor_idx])
    anchor_ask = float(asks[anchor_idx])
    if anchor_mid <= 0 or np.isnan(anchor_mid):
        n_no_data += 1
        continue

    anchor_spread_bps = (anchor_ask - anchor_bid) / anchor_mid * 10000

    # Measure mid-price at each window
    rec = {
        "symbol": symbol,
        "settle_time": settle_time,
        "fr_abs_bps": fr_abs,
        "anchor_mid": anchor_mid,
        "anchor_spread_bps": anchor_spread_bps,
    }

    for w_ms in WINDOWS_MS:
        # Find closest tick to this offset
        target_lo = w_ms - 1500  # 1.5s tolerance
        target_hi = w_ms + 1500
        w_mask = (offsets >= target_lo) & (offsets <= target_hi)
        if w_mask.sum() == 0:
            rec[f"mid_{w_ms}ms"] = np.nan
            rec[f"bid_{w_ms}ms"] = np.nan
            rec[f"ask_{w_ms}ms"] = np.nan
            rec[f"drift_{w_ms}ms_bps"] = np.nan
            continue

        # Closest tick to target
        w_indices = np.where(w_mask)[0]
        closest = w_indices[np.argmin(np.abs(offsets[w_mask] - w_ms))]

        mid_val = float(mids[closest])
        bid_val = float(bids[closest])
        ask_val = float(asks[closest])

        rec[f"mid_{w_ms}ms"] = mid_val
        rec[f"bid_{w_ms}ms"] = bid_val
        rec[f"ask_{w_ms}ms"] = ask_val
        rec[f"drift_{w_ms}ms_bps"] = (mid_val - anchor_mid) / anchor_mid * 10000

    records.append(rec)

df = pd.DataFrame(records)
print(f"\n  Processed {n_processed:,} settlements, got data for {len(df):,} ({n_no_data:,} no data)")
print(f"  Phase 3 done [{time.time()-t3:.1f}s]")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Statistical analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 100)
print("RESULTS: Pre-Settlement Price Drift (bps relative to T-30s anchor)")
print("=" * 100)

drift_cols = [c for c in df.columns if c.startswith("drift_") and c.endswith("_bps")]

# Overall stats
print("\n── ALL EXTREME FR SETTLEMENTS (|FR| >= 20 bps) ──")
print(f"   N = {len(df):,}")
print(f"   {'Window':>12}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'P10':>8}  {'P90':>8}  {'Up%':>6}")
print(f"   {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
for col in drift_cols:
    w_ms = col.replace("drift_", "").replace("ms_bps", "")
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    arr = vals.values
    label = f"T{int(w_ms):+d}ms" if int(w_ms) != 0 else "T+0ms"
    print(f"   {label:>12}  {arr.mean():>+7.2f}  {np.median(arr):>+7.2f}  "
          f"{arr.std():>7.2f}  {np.percentile(arr,10):>+7.2f}  {np.percentile(arr,90):>+7.2f}  "
          f"{(arr>0).mean()*100:>5.1f}%")

# By FR magnitude bucket
for i in range(len(FR_BUCKETS)):
    lo = FR_BUCKETS[i]
    hi = FR_BUCKETS[i + 1] if i + 1 < len(FR_BUCKETS) else 9999
    subset = df[(df["fr_abs_bps"] >= lo) & (df["fr_abs_bps"] < hi)]
    if len(subset) < 10:
        continue

    label = f"|FR| {lo}-{hi} bps" if hi < 9999 else f"|FR| >= {lo} bps"
    print(f"\n── {label} (N={len(subset):,}) ──")
    print(f"   {'Window':>12}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'P10':>8}  {'P90':>8}  {'Up%':>6}")
    print(f"   {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
    for col in drift_cols:
        w_ms = col.replace("drift_", "").replace("ms_bps", "")
        vals = subset[col].dropna()
        if len(vals) == 0:
            continue
        arr = vals.values
        w_label = f"T{int(w_ms):+d}ms" if int(w_ms) != 0 else "T+0ms"
        print(f"   {w_label:>12}  {arr.mean():>+7.2f}  {np.median(arr):>+7.2f}  "
              f"{arr.std():>7.2f}  {np.percentile(arr,10):>+7.2f}  {np.percentile(arr,90):>+7.2f}  "
              f"{(arr>0).mean()*100:>5.1f}%")

# Spread analysis
print(f"\n── SPREAD AT T-30s ──")
spreads = df["anchor_spread_bps"].dropna()
print(f"   N={len(spreads):,}  Mean={spreads.mean():.2f}  Median={spreads.median():.2f}  "
      f"P10={spreads.quantile(0.1):.2f}  P90={spreads.quantile(0.9):.2f}")

# Key question: is mid-price at T-1s vs T-10s consistently higher (anticipating FR)?
print(f"\n── KEY QUESTION: Does price drift UP before settlement (anticipating FR payment)? ──")
if "drift_-1000ms_bps" in df.columns and "drift_-10000ms_bps" in df.columns:
    d10 = df["drift_-10000ms_bps"].dropna()
    d1 = df["drift_-1000ms_bps"].dropna()
    d0 = df["drift_0ms_bps"].dropna()

    common = df[["drift_-10000ms_bps", "drift_-1000ms_bps", "drift_0ms_bps"]].dropna()
    last_10s_drift = common["drift_-1000ms_bps"] - common["drift_-10000ms_bps"]
    last_1s_drift = common["drift_0ms_bps"] - common["drift_-1000ms_bps"]

    print(f"   Price move T-10s → T-1s:  mean={last_10s_drift.mean():+.2f} bps  "
          f"median={last_10s_drift.median():+.2f}  std={last_10s_drift.std():.2f}  "
          f"up%={( last_10s_drift > 0).mean()*100:.1f}%")
    print(f"   Price move T-1s → T+0s:   mean={last_1s_drift.mean():+.2f} bps  "
          f"median={last_1s_drift.median():+.2f}  std={last_1s_drift.std():.2f}  "
          f"up%={(last_1s_drift > 0).mean()*100:.1f}%")

    # For extreme FR (>= 60 bps)
    extreme = df[df["fr_abs_bps"] >= 60][["drift_-10000ms_bps", "drift_-1000ms_bps", "drift_0ms_bps"]].dropna()
    if len(extreme) > 10:
        e_last10 = extreme["drift_-1000ms_bps"] - extreme["drift_-10000ms_bps"]
        e_last1 = extreme["drift_0ms_bps"] - extreme["drift_-1000ms_bps"]
        print(f"\n   For |FR| >= 60 bps (N={len(extreme)}):")
        print(f"   Price move T-10s → T-1s:  mean={e_last10.mean():+.2f} bps  "
              f"median={e_last10.median():+.2f}  std={e_last10.std():.2f}  "
              f"up%={(e_last10 > 0).mean()*100:.1f}%")
        print(f"   Price move T-1s → T+0s:   mean={e_last1.mean():+.2f} bps  "
              f"median={e_last1.median():+.2f}  std={e_last1.std():.2f}  "
              f"up%={(e_last1 > 0).mean()*100:.1f}%")

# Limit order fill probability analysis
print(f"\n── LIMIT ORDER FILL PROBABILITY ──")
print("   If we place limit buy at best bid at T-10s, would price ever touch it before T-0?")
# Price touching bid = mid drops to bid level = drift becomes negative by at least half-spread
if "drift_-10000ms_bps" in df.columns:
    for w_ms in [-5000, -2000, -1000, -500, 0]:
        col = f"drift_{w_ms}ms_bps"
        if col not in df.columns:
            continue
        ref_col = "drift_-10000ms_bps"
        common = df[[ref_col, col, "anchor_spread_bps"]].dropna()
        move = common[col] - common[ref_col]
        half_spread = common["anchor_spread_bps"] / 2
        # Price dropped at all (any sell pressure)
        dropped = (move < 0).mean() * 100
        # Price dropped by at least half spread (limit at bid would fill)
        filled_est = (move < -half_spread).mean() * 100
        w_label = f"T{w_ms:+d}ms" if w_ms != 0 else "T+0ms"
        print(f"   By {w_label}: dropped={dropped:.1f}%  dropped>½sprd={filled_est:.1f}%  "
              f"mean_move={move.mean():+.2f}bps")

elapsed = time.time() - t_global
print(f"\n{'='*100}")
print(f"DONE — {elapsed:.1f}s total")
print(f"{'='*100}")
