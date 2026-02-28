#!/usr/bin/env python3
"""
Analyse price drift BEFORE settlement using individual Bybit trades (ms precision).

Uses parquet/{SYMBOL}/trades/bybit_futures/{date}.parquet
Each row = one trade: timestamp_us, price, quantity, side (+1=buy, -1=sell)

Measures:
  - VWAP drift in sub-second windows: T-10s, T-5s, T-2s, T-1s, T-500ms, T-200ms, T-100ms
  - Buy/sell aggression (volume imbalance) in each window
  - Spread proxy from trade prices
  - Whether a limit buy at T-10s mid would have filled by T-0
"""
import sys
import time
import gc
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data_all"
PARQUET = REPO_ROOT / "parquet"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
MIN_FR_BPS = 5.0  # lower threshold since we only have 5 large-cap coins with trade data
# Sub-second windows (ms before settlement T=0)
WINDOWS_MS = [-10000, -5000, -2000, -1000, -500, -200, -100, 0, 100, 200, 500, 1000, 2000, 5000]

print("=" * 110)
print("ANALYSE: Pre-Settlement Price Drift — Individual Trades (ms precision)")
print("=" * 110)
print(f"  Min |FR|:      {MIN_FR_BPS} bps")
print(f"  Trade source:  parquet/*/trades/bybit_futures/")
print(f"  Windows:       {WINDOWS_MS}")
print()

t_global = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Build settlement schedule from Bybit historical FR
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 1: Building settlement schedule (Bybit historical FR)...")
sys.stdout.flush()
t1 = time.time()

hfr = pd.read_parquet(DATA / "historical_fr" / "bybit_fr_history.parquet")
print(f"  Loaded Bybit historical FR: {len(hfr):,} rows [{time.time()-t1:.1f}s]")
print(f"  Date range: {hfr['fundingTime'].min()} to {hfr['fundingTime'].max()}")

# fundingTime = exact settlement time, fundingRate = the settled rate
hfr["fr_bps"] = hfr["fundingRate"] * 10000

# Both directions: |FR| >= threshold
# Negative FR: shorts pay longs (we go long) / Positive FR: longs pay shorts (we go short)
settle_neg = hfr[hfr["fr_bps"].abs() >= MIN_FR_BPS].copy()
settle_neg = settle_neg.rename(columns={"fundingTime": "settle_time"})
settle_neg["fr_abs_bps"] = settle_neg["fr_bps"].abs()
settle_neg["direction"] = np.where(settle_neg["fr_bps"] < 0, "long", "short")

# Filter to coins that have trade data
available_coins = set(d.name for d in PARQUET.iterdir() if d.is_dir())
settle_neg = settle_neg[settle_neg["symbol"].isin(available_coins)].copy()

# Filter to dates where we have trade parquet files (up to ~2026-02-16)
settle_neg = settle_neg[settle_neg["settle_time"] <= "2026-02-16"].copy()
settle_neg = settle_neg.sort_values("settle_time").reset_index(drop=True)

print(f"  Total FR records: {len(hfr):,}")
print(f"  |FR| >= {MIN_FR_BPS} bps with trade data: {len(settle_neg):,}")
print(f"  Unique coins: {settle_neg['symbol'].nunique()}")
print(f"  Date range: {settle_neg['settle_time'].min()} to {settle_neg['settle_time'].max()}")
print(f"  Phase 1 done [{time.time()-t1:.1f}s]")
print()
sys.stdout.flush()

del hfr; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: For each settlement, load trades around it and measure drift
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 2: Analysing individual trades around settlement...")
sys.stdout.flush()
t2 = time.time()

# Pre-group by (symbol, date) for efficient loading
settle_neg["date_str"] = settle_neg["settle_time"].dt.strftime("%Y-%m-%d")
# fundingTime from Bybit is already the exact settlement timestamp
settle_neg["settle_us"] = (settle_neg["settle_time"].astype(np.int64) // 1000).astype(np.int64)

# Cache loaded trade files
trade_cache = {}

def load_trades(symbol, date_str):
    """Load trades for a symbol/date, with caching."""
    key = (symbol, date_str)
    if key not in trade_cache:
        path = PARQUET / symbol / "trades" / "bybit_futures" / f"{date_str}.parquet"
        if not path.exists():
            trade_cache[key] = None
        else:
            df = pd.read_parquet(path, columns=["timestamp_us", "price", "quantity", "side"])
            trade_cache[key] = df
    return trade_cache[key]

def clear_cache():
    global trade_cache
    trade_cache.clear()
    gc.collect()


records = []
n_no_data = 0
n_processed = 0
n_total = len(settle_neg)
last_date = None

for idx, row in settle_neg.iterrows():
    n_processed += 1
    if n_processed % 100 == 0 or n_processed == n_total:
        elapsed = time.time() - t2
        rate = n_processed / elapsed if elapsed > 0 else 0
        eta = (n_total - n_processed) / rate if rate > 0 else 0
        print(f"  [{n_processed:,}/{n_total:,}] {elapsed:.0f}s elapsed, "
              f"~{eta:.0f}s ETA, no_data={n_no_data}, records={len(records)}", flush=True)

    symbol = row["symbol"]
    date_str = row["date_str"]
    settle_us = row["settle_us"]
    fr_abs = row["fr_abs_bps"]

    # Clear cache when date changes to limit memory
    if date_str != last_date:
        clear_cache()
        last_date = date_str

    trades = load_trades(symbol, date_str)
    if trades is None:
        n_no_data += 1
        continue

    ts = trades["timestamp_us"].values
    prices = trades["price"].values
    qtys = trades["quantity"].values
    sides = trades["side"].values

    # Window: T-30s to T+10s
    t_lo = settle_us - 30_000_000  # 30s before in us
    t_hi = settle_us + 10_000_000  # 10s after in us
    i0 = np.searchsorted(ts, t_lo, side="left")
    i1 = np.searchsorted(ts, t_hi, side="right")

    if i1 - i0 < 10:
        n_no_data += 1
        continue

    sl = slice(i0, i1)
    w_ts = ts[sl]
    w_price = prices[sl]
    w_qty = qtys[sl]
    w_side = sides[sl]

    # Offsets in ms relative to settlement
    offsets_ms = (w_ts - settle_us) / 1000.0  # us → ms

    # Anchor: VWAP of trades in [-12s, -8s] window
    anchor_mask = (offsets_ms >= -12000) & (offsets_ms <= -8000)
    if anchor_mask.sum() < 3:
        n_no_data += 1
        continue

    anchor_vwap = np.average(w_price[anchor_mask], weights=w_qty[anchor_mask])
    if anchor_vwap <= 0 or np.isnan(anchor_vwap):
        n_no_data += 1
        continue

    direction = row["direction"]  # "long" or "short"
    # sign_flip: for longs, price up = adverse (flip sign); for shorts, price up = favorable
    sign_flip = -1 if direction == "long" else +1

    rec = {
        "symbol": symbol,
        "settle_time": row["settle_time"],
        "fr_abs_bps": fr_abs,
        "direction": direction,
        "sign_flip": sign_flip,
        "anchor_vwap": anchor_vwap,
        "n_trades_window": int(i1 - i0),
    }

    # For each window, compute VWAP and buy/sell volume
    for w_ms in WINDOWS_MS:
        # 200ms bucket around target
        bucket_lo = w_ms - 100  # ms
        bucket_hi = w_ms + 100
        w_mask = (offsets_ms >= bucket_lo) & (offsets_ms <= bucket_hi)

        if w_mask.sum() == 0:
            rec[f"vwap_{w_ms}ms"] = np.nan
            rec[f"drift_{w_ms}ms_bps"] = np.nan
            rec[f"buy_vol_{w_ms}ms"] = np.nan
            rec[f"sell_vol_{w_ms}ms"] = np.nan
            rec[f"n_trades_{w_ms}ms"] = 0
            continue

        w_p = w_price[w_mask]
        w_q = w_qty[w_mask]
        w_s = w_side[w_mask]

        vwap = np.average(w_p, weights=w_q)
        drift_bps = (vwap - anchor_vwap) / anchor_vwap * 10000

        buy_vol = w_q[w_s == 1].sum()
        sell_vol = w_q[w_s == -1].sum()

        rec[f"vwap_{w_ms}ms"] = vwap
        rec[f"drift_{w_ms}ms_bps"] = drift_bps
        rec[f"buy_vol_{w_ms}ms"] = buy_vol
        rec[f"sell_vol_{w_ms}ms"] = sell_vol
        rec[f"n_trades_{w_ms}ms"] = int(w_mask.sum())

    # Limit order analysis: if we place buy at anchor_vwap at T-10s,
    # does price ever drop to that level before T-0?
    pre_settle = (offsets_ms >= -10000) & (offsets_ms < 0)
    if pre_settle.sum() > 0:
        min_price_pre = w_price[pre_settle].min()
        rec["min_price_pre_10s"] = min_price_pre
        rec["min_drift_pre_10s_bps"] = (min_price_pre - anchor_vwap) / anchor_vwap * 10000
        rec["limit_at_anchor_filled"] = min_price_pre <= anchor_vwap
    else:
        rec["min_price_pre_10s"] = np.nan
        rec["min_drift_pre_10s_bps"] = np.nan
        rec["limit_at_anchor_filled"] = False

    # Also check T-5s and T-2s windows
    for pre_s, label in [(5, "5s"), (2, "2s"), (1, "1s")]:
        pre_mask = (offsets_ms >= -pre_s * 1000) & (offsets_ms < 0)
        if pre_mask.sum() > 0:
            min_p = w_price[pre_mask].min()
            rec[f"min_drift_pre_{label}_bps"] = (min_p - anchor_vwap) / anchor_vwap * 10000
        else:
            rec[f"min_drift_pre_{label}_bps"] = np.nan

    # ── REALISTIC LIMIT ORDER SIMULATION ──
    # For each submission time T-Xs: place limit buy at VWAP around submission,
    # check if price trades at/below that level between submission and T+0.
    # Also compare fill price to market fill at T-300ms.
    # For shorts: place limit sell at VWAP, check if price trades at/above.
    is_long = (direction == "long")

    # Market entry benchmark: VWAP at T-300ms (±50ms bucket)
    mkt_mask = (offsets_ms >= -350) & (offsets_ms <= -250)
    if mkt_mask.sum() > 0:
        mkt_entry_price = np.average(w_price[mkt_mask], weights=w_qty[mkt_mask])
    else:
        # fallback: last trade before T-300ms
        before_300 = offsets_ms <= -300
        if before_300.sum() > 0:
            mkt_entry_price = w_price[before_300][-1]
        else:
            mkt_entry_price = anchor_vwap
    rec["mkt_entry_price"] = mkt_entry_price

    for submit_s in [10, 5, 3, 2, 1]:
        submit_ms = -submit_s * 1000
        # VWAP in ±1s bucket around submission time = our limit price
        # (wider bucket needed for sparse altcoin trading)
        half_w = 1000  # 1s each side
        sub_mask = (offsets_ms >= submit_ms - half_w) & (offsets_ms <= submit_ms + half_w)
        if sub_mask.sum() < 2:
            rec[f"lim_fill_{submit_s}s"] = np.nan
            rec[f"lim_improve_{submit_s}s_bps"] = np.nan
            rec[f"lim_price_{submit_s}s"] = np.nan
            continue

        limit_price = np.average(w_price[sub_mask], weights=w_qty[sub_mask])
        rec[f"lim_price_{submit_s}s"] = limit_price

        # Check fill: between end of submit bucket and T+0, does price touch our limit?
        fill_window = (offsets_ms > submit_ms + half_w) & (offsets_ms <= 0)
        if fill_window.sum() == 0:
            rec[f"lim_fill_{submit_s}s"] = False
            rec[f"lim_improve_{submit_s}s_bps"] = np.nan
            continue

        if is_long:
            # Limit buy at limit_price: fills if any trade <= limit_price
            filled = w_price[fill_window].min() <= limit_price
        else:
            # Limit sell at limit_price: fills if any trade >= limit_price
            filled = w_price[fill_window].max() >= limit_price

        rec[f"lim_fill_{submit_s}s"] = filled

        # Improvement: how much better is limit fill vs market at T-300ms?
        # For longs: lower price = better → improvement = (mkt - limit) / mkt * 10000
        # For shorts: higher price = better → improvement = (limit - mkt) / mkt * 10000
        if filled:
            if is_long:
                improve = (mkt_entry_price - limit_price) / mkt_entry_price * 10000
            else:
                improve = (limit_price - mkt_entry_price) / mkt_entry_price * 10000
            rec[f"lim_improve_{submit_s}s_bps"] = improve
        else:
            rec[f"lim_improve_{submit_s}s_bps"] = np.nan

    # Buy/sell imbalance in last 10s, 5s, 2s, 1s before settlement
    for pre_s, label in [(10, "10s"), (5, "5s"), (2, "2s"), (1, "1s")]:
        pre_mask = (offsets_ms >= -pre_s * 1000) & (offsets_ms < 0)
        if pre_mask.sum() > 0:
            bv = w_qty[pre_mask & (w_side == 1)].sum()
            sv = w_qty[pre_mask & (w_side == -1)].sum()
            total = bv + sv
            rec[f"vol_imb_pre_{label}"] = (bv - sv) / total if total > 0 else 0
            rec[f"buy_pct_pre_{label}"] = bv / total * 100 if total > 0 else 50
        else:
            rec[f"vol_imb_pre_{label}"] = np.nan
            rec[f"buy_pct_pre_{label}"] = np.nan

    records.append(rec)

clear_cache()
df = pd.DataFrame(records)
print(f"\n  Processed {n_processed:,}, got data for {len(df):,} ({n_no_data:,} no data)")
print(f"  Phase 2 done [{time.time()-t2:.1f}s]")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Results — DIRECTION-AWARE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 110)
print("RESULTS: Pre-Settlement Price Drift — DIRECTION-AWARE (ms precision)")
print("=" * 110)
print()
print("  SIGN CONVENTION:")
print("    Raw drift:       positive = price went UP from anchor")
print("    Favorable drift: positive = price moved IN YOUR FAVOR for entry")
print("      → Long (neg FR):  favorable = price DOWN  (you buy cheaper)")
print("      → Short (pos FR): favorable = price UP    (you sell higher)")
print("    Adverse drift:   positive = price moved AGAINST YOU")
print("      → Long (neg FR):  adverse = price UP      (you buy more expensive)")
print("      → Short (pos FR): adverse = price DOWN    (you sell cheaper)")
print()

if len(df) == 0:
    print("\n  NO DATA — no settlements matched trade files.")
    print(f"  Available coins: {available_coins & set(settle_neg['symbol'].unique())}")
    sys.exit(0)

drift_cols = sorted([c for c in df.columns if c.startswith("drift_") and c.endswith("ms_bps")],
                    key=lambda x: int(x.split("_")[1].replace("ms", "").replace("bps", "")))

# Compute direction-aware "favorable drift" columns
# favorable = raw_drift * sign_flip  (positive = in your favor)
for col in drift_cols:
    fav_col = col.replace("drift_", "fdrift_")
    df[fav_col] = df[col] * df["sign_flip"]

fav_drift_cols = sorted([c for c in df.columns if c.startswith("fdrift_") and c.endswith("ms_bps")],
                        key=lambda x: int(x.split("_")[1].replace("ms", "").replace("bps", "")))

n_long = (df["direction"] == "long").sum()
n_short = (df["direction"] == "short").sum()
print(f"  Total: {len(df):,}  (long={n_long:,}, short={n_short:,})")

# ── RAW PRICE DRIFT (up=positive, no direction awareness) ──
def print_raw_drift_table(subset, label):
    print(f"\n── {label} (N={len(subset):,}) ──")
    print(f"   {'Window':>12}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'P10':>8}  {'P90':>8}  {'Up%':>6}  {'NTrades':>8}")
    print(f"   {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}")
    for col in drift_cols:
        w_ms = col.replace("drift_", "").replace("ms_bps", "")
        vals = subset[col].dropna()
        if len(vals) < 5:
            continue
        arr = vals.values
        nt_col = f"n_trades_{w_ms}ms"
        avg_nt = subset[nt_col].mean() if nt_col in subset.columns else 0
        label_w = f"T{int(w_ms):+d}ms" if int(w_ms) != 0 else "T+0ms"
        print(f"   {label_w:>12}  {arr.mean():>+7.2f}  {np.median(arr):>+7.2f}  "
              f"{arr.std():>7.2f}  {np.percentile(arr,10):>+7.2f}  {np.percentile(arr,90):>+7.2f}  "
              f"{(arr>0).mean()*100:>5.1f}%  {avg_nt:>7.1f}")

# ── DIRECTION-AWARE FAVORABLE DRIFT (positive = in your favor) ──
def print_fav_drift_table(subset, label):
    print(f"\n── {label} (N={len(subset):,}) ──")
    print(f"   {'Window':>12}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'P10':>8}  {'P90':>8}  {'Fav%':>6}  {'NTrades':>8}")
    print(f"   {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}")
    for col in fav_drift_cols:
        w_ms = col.replace("fdrift_", "").replace("ms_bps", "")
        vals = subset[col].dropna()
        if len(vals) < 5:
            continue
        arr = vals.values
        nt_col = f"n_trades_{w_ms}ms"
        avg_nt = subset[nt_col].mean() if nt_col in subset.columns else 0
        label_w = f"T{int(w_ms):+d}ms" if int(w_ms) != 0 else "T+0ms"
        print(f"   {label_w:>12}  {arr.mean():>+7.2f}  {np.median(arr):>+7.2f}  "
              f"{arr.std():>7.2f}  {np.percentile(arr,10):>+7.2f}  {np.percentile(arr,90):>+7.2f}  "
              f"{(arr>0).mean()*100:>5.1f}%  {avg_nt:>7.1f}")

# ────────────────────────────────────────────────────────
# A. RAW DRIFT — shows actual price direction
# ────────────────────────────────────────────────────────
print("\n" + "━" * 110)
print("  A. RAW PRICE DRIFT (positive = price UP from anchor)")
print("━" * 110)

print_raw_drift_table(df, f"ALL — Raw drift")

for dir_label, dir_val in [("LONG (neg FR — we buy)", "long"), ("SHORT (pos FR — we sell)", "short")]:
    subset = df[df["direction"] == dir_val]
    if len(subset) >= 10:
        print_raw_drift_table(subset, f"{dir_label} — Raw drift")

# ────────────────────────────────────────────────────────
# B. DIRECTION-AWARE FAVORABLE DRIFT
# ────────────────────────────────────────────────────────
print("\n" + "━" * 110)
print("  B. FAVORABLE DRIFT (positive = price moved IN YOUR FAVOR for entry)")
print("     Long: favorable = price DOWN | Short: favorable = price UP")
print("━" * 110)

print_fav_drift_table(df, f"ALL — Favorable drift")

for dir_label, dir_val in [("LONG entries (neg FR)", "long"), ("SHORT entries (pos FR)", "short")]:
    subset = df[df["direction"] == dir_val]
    if len(subset) >= 10:
        print_fav_drift_table(subset, f"{dir_label} — Favorable drift")

# By FR magnitude
for lo, hi in [(5, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 9999)]:
    subset = df[(df["fr_abs_bps"] >= lo) & (df["fr_abs_bps"] < hi)]
    if len(subset) < 10:
        continue
    label = f"|FR| {lo}-{hi} bps" if hi < 9999 else f"|FR| >= {lo} bps"
    n_l = (subset["direction"] == "long").sum()
    n_s = (subset["direction"] == "short").sum()
    print_fav_drift_table(subset, f"{label} (long={n_l}, short={n_s}) — Favorable drift")

# ────────────────────────────────────────────────────────
# C. Buy/sell aggression — split by direction
# ────────────────────────────────────────────────────────
print("\n" + "━" * 110)
print("  C. BUY/SELL VOLUME IMBALANCE BEFORE SETTLEMENT")
print("━" * 110)
print(f"   (>50% buy = buying pressure, <50% = selling pressure)")
print(f"   {'Window':>12}  {'ALL Buy%':>10}  {'LONG Buy%':>10}  {'SHORT Buy%':>10}")
print(f"   {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")
for label in ["10s", "5s", "2s", "1s"]:
    col = f"buy_pct_pre_{label}"
    if col not in df.columns:
        continue
    all_v = df[col].dropna()
    long_v = df[df["direction"] == "long"][col].dropna()
    short_v = df[df["direction"] == "short"][col].dropna()
    if len(all_v) < 5:
        continue
    print(f"   {'T-'+label:>12}  {all_v.mean():>9.1f}%  "
          f"{long_v.mean():>9.1f}%  {short_v.mean():>9.1f}%")

# ────────────────────────────────────────────────────────
# D. KEY QUESTION: Does price move AGAINST or FOR you before settlement?
# ────────────────────────────────────────────────────────
print("\n" + "━" * 110)
print("  D. KEY QUESTION: Does price move AGAINST or FOR you before settlement?")
print("     (positive = price moves AGAINST you = you get worse entry)")
print("━" * 110)

for dir_label, dir_val in [("ALL", None), ("LONG (neg FR)", "long"), ("SHORT (pos FR)", "short")]:
    subset = df if dir_val is None else df[df["direction"] == dir_val]
    if len(subset) < 10:
        continue
    print(f"\n   {dir_label} (N={len(subset):,}):")
    for pre_s in [10, 5, 2, 1]:
        w1 = f"fdrift_{-pre_s*1000}ms_bps"
        w2 = "fdrift_0ms_bps"
        if w1 in subset.columns and w2 in subset.columns:
            common = subset[[w1, w2]].dropna()
            if len(common) < 5:
                continue
            # move from window to T+0, in favorable terms
            # negative = price moved against you in that interval
            move = common[w2] - common[w1]
            # But we want "adverse drift" = how much worse entry got
            # adverse = -(favorable_drift), so adverse = -move means entry got worse
            adverse = -move
            print(f"     T-{pre_s}s → T+0: adverse_drift mean={adverse.mean():>+.2f} bps  "
                  f"median={adverse.median():>+.2f}  "
                  f"worse%={(adverse>0).mean()*100:.1f}%  better%={(adverse<0).mean()*100:.1f}%")

# ────────────────────────────────────────────────────────
# E. LIMIT ORDER VIABILITY — anchor-based (legacy)
# ────────────────────────────────────────────────────────
print("\n" + "━" * 110)
print("  E. LIMIT ORDER VIABILITY (anchor VWAP T-10s)")
print("━" * 110)

for dir_label, dir_val in [("ALL", None), ("LONG (neg FR)", "long"), ("SHORT (pos FR)", "short")]:
    subset = df if dir_val is None else df[df["direction"] == dir_val]
    if len(subset) < 5:
        continue
    filled = subset["limit_at_anchor_filled"].dropna()
    print(f"\n   {dir_label} (N={len(subset):,}): "
          f"fill rate = {filled.mean()*100:.1f}%")
    for lo, hi in [(20, 40), (40, 60), (60, 100), (100, 9999)]:
        sub2 = subset[(subset["fr_abs_bps"] >= lo) & (subset["fr_abs_bps"] < hi)]
        if len(sub2) < 5:
            continue
        fr_label = f"|FR|{lo}-{hi}" if hi < 9999 else f"|FR|>={lo}"
        rate = sub2["limit_at_anchor_filled"].mean() * 100
        print(f"     {fr_label:>12}: {rate:.1f}%  (N={len(sub2)})")

# ────────────────────────────────────────────────────────
# F. REALISTIC LIMIT vs MARKET — fee-aware comparison
# ────────────────────────────────────────────────────────
TAKER_FEE_BPS = 10.0  # 0.1% taker fee
MAKER_FEE_BPS = 2.0   # 0.02% maker fee (PostOnly)

print("\n" + "━" * 110)
print("  F. REALISTIC LIMIT vs MARKET ORDER COMPARISON")
print(f"     Taker fee: {TAKER_FEE_BPS:.1f} bps | Maker fee: {MAKER_FEE_BPS:.1f} bps")
print("     Limit = PostOnly at VWAP when submitted | Market = taker at T-300ms")
print("━" * 110)

# Show NaN rate first so we understand data coverage
for submit_s in [10, 5, 3, 2, 1]:
    fill_col = f"lim_fill_{submit_s}s"
    if fill_col in df.columns:
        n_valid = df[fill_col].notna().sum()
        n_total = len(df)
        print(f"   T-{submit_s:>2d}s: {n_valid:,}/{n_total:,} settlements had enough trades for limit sim "
              f"({n_valid/n_total*100:.1f}%)")

print(f"\n   {'Submit':>8s}  {'N':>6s}  {'Fill%':>6s}  {'PriceImprove':>13s}  {'FeeSave':>8s}  "
      f"{'EV(lim+fb)':>10s}  {'EV(mkt)':>10s}  {'Verdict':>12s}")
print(f"   {'─'*8}  {'─'*6}  {'─'*6}  {'─'*13}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}")

for submit_s in [10, 5, 3, 2, 1]:
    fill_col = f"lim_fill_{submit_s}s"
    improve_col = f"lim_improve_{submit_s}s_bps"
    if fill_col not in df.columns:
        continue

    fills = df[fill_col].dropna()
    if len(fills) < 5:
        continue
    fill_rate = fills.mean()

    improves = df[improve_col].dropna()
    avg_improve = improves.mean() if len(improves) > 0 else 0
    med_improve = improves.median() if len(improves) > 0 else 0

    fee_save = TAKER_FEE_BPS - MAKER_FEE_BPS  # 8 bps saved per fill

    # Settlement edge: use actual data for this subset
    t0_col = "fdrift_0ms_bps"
    edge_vals = df.loc[fills.index, t0_col].dropna() if t0_col in df.columns else pd.Series()
    avg_edge = edge_vals.mean() if len(edge_vals) > 10 else 36.0

    # Scenario: limit with market fallback at T-300ms if not filled
    ev_limit_fb = fill_rate * (avg_edge + fee_save + avg_improve) + (1 - fill_rate) * avg_edge
    ev_market = avg_edge

    print(f"   T-{submit_s:>2d}s    {len(fills):>5,}  {fill_rate*100:>5.1f}%  "
          f"  {avg_improve:>+6.2f} ({med_improve:>+.1f})  {fee_save:>6.1f}bp  "
          f"  {ev_limit_fb:>+7.2f}bp  {ev_market:>+7.2f}bp  "
          f"{'✓ LIMIT' if ev_limit_fb > ev_market else '✗ MARKET':>12s}")

# Detailed breakdown by FR bucket and direction
print(f"\n   ── By FR bucket (LONG only, with market fallback) ──")
print(f"   {'FR Bucket':>12s}  {'Submit':>6s}  {'N':>5s}  {'Fill%':>6s}  {'Improve':>8s}  "
      f"{'Edge':>6s}  {'EV_lim':>8s}  {'EV_mkt':>8s}  {'Verdict':>10s}")
print(f"   {'─'*12}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")

long_df = df[df["direction"] == "long"]
for lo, hi in [(10, 20), (20, 40), (40, 60), (60, 100), (100, 9999)]:
    subset = long_df[(long_df["fr_abs_bps"] >= lo) & (long_df["fr_abs_bps"] < hi)]
    if len(subset) < 10:
        continue
    fr_label = f"|FR|{lo}-{hi}" if hi < 9999 else f"|FR|>={lo}"

    t0_col = "fdrift_0ms_bps"
    if t0_col in subset.columns:
        edge = subset[t0_col].dropna().mean()
    else:
        edge = 36.0

    for submit_s in [5, 3, 2, 1]:
        fill_col = f"lim_fill_{submit_s}s"
        improve_col = f"lim_improve_{submit_s}s_bps"
        if fill_col not in subset.columns:
            continue
        fills = subset[fill_col].dropna()
        if len(fills) < 5:
            continue
        fill_rate = fills.mean()
        improves = subset[improve_col].dropna()
        avg_improve = improves.mean() if len(improves) > 0 else 0
        fee_save = TAKER_FEE_BPS - MAKER_FEE_BPS

        ev_lim = fill_rate * (edge + fee_save + avg_improve) + (1 - fill_rate) * edge
        ev_mkt = edge

        print(f"   {fr_label:>12s}   T-{submit_s}s  {len(fills):>4}  {fill_rate*100:>5.1f}%  "
              f"{avg_improve:>+7.2f}  {edge:>+5.0f}  {ev_lim:>+7.1f}  {ev_mkt:>+7.1f}  "
              f"{'✓ LIMIT' if ev_lim > ev_mkt else '✗ MARKET':>10s}")

# ────────────────────────────────────────────────────────
# G. BOTTOM LINE
# ────────────────────────────────────────────────────────
print("\n" + "━" * 110)
print("  G. BOTTOM LINE")
print("━" * 110)
print(f"     Taker fee = {TAKER_FEE_BPS:.0f} bps | Maker fee = {MAKER_FEE_BPS:.0f} bps | Fee saving = {TAKER_FEE_BPS - MAKER_FEE_BPS:.0f} bps per leg")
print()

# Best strategy: limit with fallback
best_ev = -999
best_label = ""
for submit_s in [10, 5, 3, 2, 1]:
    fill_col = f"lim_fill_{submit_s}s"
    improve_col = f"lim_improve_{submit_s}s_bps"
    if fill_col not in df.columns:
        continue
    fills = df[fill_col].dropna()
    if len(fills) < 10:
        continue
    fill_rate = fills.mean()
    improves = df[improve_col].dropna()
    avg_improve = improves.mean() if len(improves) > 0 else 0
    fee_save = TAKER_FEE_BPS - MAKER_FEE_BPS
    ev = fill_rate * (36.0 + fee_save + avg_improve) + (1 - fill_rate) * 36.0
    if ev > best_ev:
        best_ev = ev
        best_label = f"T-{submit_s}s"

print(f"     Best limit submit time: {best_label} (EV={best_ev:+.2f} vs market EV=+36.00)")
print(f"     Strategy: PostOnly limit at {best_label}, fallback to market at T-300ms if not filled")

elapsed = time.time() - t_global
print(f"\n{'='*110}")
print(f"DONE — {elapsed:.1f}s total, {len(df):,} settlements analysed")
print(f"{'='*110}")
