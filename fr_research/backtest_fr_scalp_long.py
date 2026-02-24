#!/usr/bin/env python3
"""
Backtest: Long Futures FR Scalp

Strategy:
  - When FR is very negative (shorts pay longs), go LONG futures just before settlement
  - Collect the FR payment at settlement
  - Exit immediately: TP with limit order if price goes up, SL with market if down
  
Fee model (Bybit VIP0):
  - Entry: market order = 5.5 bps taker
  - Exit TP: limit order = 2.0 bps maker
  - Exit SL: market order = 5.5 bps taker
  - Best case RT: 7.5 bps | Worst case RT: 11 bps

Uses 5-second Bybit ticker data with bid/ask for realistic fills.
Uses Binance fundingRate data for settlement detection (settled FR = lastFundingRate).
"""
import sys
import time
import gc
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data_all"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
NOTIONAL = 10_000

# Fee model
ENTRY_FEE_BPS = 5.5    # market order taker
TP_FEE_BPS = 2.0       # limit order maker
SL_FEE_BPS = 5.5       # market order taker

# Timing
ENTRY_BEFORE_S = 5     # enter X seconds before settlement
MAX_HOLD_S = 300       # max hold time after settlement (5 min)

# FR threshold
MIN_FR_BPS = 20.0      # minimum |FR| to enter

print("=" * 90)
print("BACKTEST: Long Futures FR Scalp (collect negative FR)")
print("=" * 90)
print(f"  Notional:      ${NOTIONAL:,}")
print(f"  Entry fee:     {ENTRY_FEE_BPS} bps (market)")
print(f"  TP fee:        {TP_FEE_BPS} bps (limit)")
print(f"  SL fee:        {SL_FEE_BPS} bps (market)")
print(f"  Entry timing:  {ENTRY_BEFORE_S}s before settlement")
print(f"  Max hold:      {MAX_HOLD_S}s after settlement")
print(f"  Min |FR|:      {MIN_FR_BPS} bps")
print("=" * 90)
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
settle_rows["fr_paid_bps"] = settle_rows["fr_prev"] * 10000  # signed: negative = shorts pay longs

settle_df = settle_rows[["settle_time", "symbol", "fr_paid_bps"]].dropna().copy()
settle_df = settle_df.sort_values("settle_time").reset_index(drop=True)

# Only negative FR (we go LONG, so we collect when FR is negative)
settle_neg = settle_df[settle_df["fr_paid_bps"] <= -MIN_FR_BPS].copy()

print(f"  Total settlements: {len(settle_df):,}")
print(f"  Negative FR <= -{MIN_FR_BPS} bps: {len(settle_neg):,} ({len(settle_neg)/len(settle_df)*100:.1f}%)")
print(f"  Unique coins: {settle_neg['symbol'].nunique()}")
print(f"  Date range: {settle_df['settle_time'].min()} to {settle_df['settle_time'].max()}")
print(f"  Phase 1 done [{time.time()-t1:.1f}s]")
print()
sys.stdout.flush()

del bn_fr_1m; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Load Bybit tick data (has bid/ask for realistic fills)
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
sys.stdout.flush()


def get_prices_around(symbol, settle_time_ns, before_s, after_s):
    """Get all tick data for a symbol in a window around settlement.
    Returns arrays of (ts_offset_s, last, bid, ask) relative to settle_time."""
    t_lo = settle_time_ns - np.timedelta64(before_s, "s")
    t_hi = settle_time_ns + np.timedelta64(after_s, "s")
    i0 = np.searchsorted(bb_ts, t_lo, side="left")
    i1 = np.searchsorted(bb_ts, t_hi, side="right")
    if i0 >= i1:
        return None
    sl = slice(i0, i1)
    sym_mask = bb_sym[sl] == symbol
    if sym_mask.sum() == 0:
        return None
    ts = bb_ts[sl][sym_mask]
    offsets = (ts - settle_time_ns) / np.timedelta64(1, "s")
    return {
        "offset_s": offsets,
        "last": bb_last[sl][sym_mask],
        "bid": bb_bid[sl][sym_mask],
        "ask": bb_ask[sl][sym_mask],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Backtest — scan all negative FR settlements
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nPHASE 3: Running backtest ({len(settle_neg):,} settlements)...")
sys.stdout.flush()
t3 = time.time()

# We'll test multiple SL/TP configs simultaneously
configs = [
    # (name, sl_bps, tp_bps)
    ("SL10_TP0",   10, 0),     # 10 bps SL, no TP (hold max_hold then exit)
    ("SL20_TP0",   20, 0),
    ("SL30_TP0",   30, 0),
    ("SL50_TP0",   50, 0),
    ("SL10_TP10",  10, 10),
    ("SL20_TP20",  20, 20),
    ("SL30_TP10",  30, 10),
    ("SL50_TP20",  50, 20),
    ("SL100_TP0", 100, 0),    # wide SL, just hold
    ("NO_SL_TP0",  9999, 0),  # no SL, exit at max_hold
]

results = {name: [] for name, _, _ in configs}
n_no_data = 0
n_processed = 0

for idx, row in settle_neg.iterrows():
    n_processed += 1
    if n_processed % 500 == 0 or n_processed == len(settle_neg):
        elapsed = time.time() - t3
        rate = n_processed / elapsed if elapsed > 0 else 0
        eta = (len(settle_neg) - n_processed) / rate if rate > 0 else 0
        print(f"  [{n_processed:,}/{len(settle_neg):,}] {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA, no_data={n_no_data}", flush=True)

    symbol = row["symbol"]
    settle_time = row["settle_time"]
    fr_bps = row["fr_paid_bps"]  # negative number
    fr_collected_bps = abs(fr_bps)  # we go long, collect positive payment

    settle_ns = np.datetime64(settle_time, "ns")

    # Get tick data: enter ENTRY_BEFORE_S before, hold up to MAX_HOLD_S after
    data = get_prices_around(symbol, settle_ns, before_s=30, after_s=MAX_HOLD_S + 30)
    if data is None:
        n_no_data += 1
        continue

    # Entry: buy at ask price at ENTRY_BEFORE_S before settlement
    entry_mask = data["offset_s"] <= -ENTRY_BEFORE_S
    if entry_mask.sum() == 0:
        # Try closest tick before settlement
        pre_mask = data["offset_s"] < 0
        if pre_mask.sum() == 0:
            n_no_data += 1
            continue
        entry_idx = np.where(pre_mask)[0][-1]
    else:
        entry_idx = np.where(entry_mask)[0][-1]

    entry_ask = float(data["ask"][entry_idx])
    if entry_ask <= 0 or np.isnan(entry_ask):
        n_no_data += 1
        continue

    # Post-settlement ticks
    post_mask = data["offset_s"] > 0
    if post_mask.sum() < 2:
        n_no_data += 1
        continue

    post_offsets = data["offset_s"][post_mask]
    post_last = data["last"][post_mask]
    post_bid = data["bid"][post_mask]
    post_ask = data["ask"][post_mask]

    # For each config, simulate exit
    for name, sl_bps, tp_bps in configs:
        sl_price = entry_ask * (1 - sl_bps / 10000) if sl_bps < 9000 else 0
        tp_price = entry_ask * (1 + tp_bps / 10000) if tp_bps > 0 else float("inf")

        exit_price = None
        exit_type = "timeout"
        exit_fee = SL_FEE_BPS  # default: market exit on timeout

        for j in range(len(post_last)):
            tick_bid = float(post_bid[j])
            tick_last = float(post_last[j])

            if tick_bid <= 0 or np.isnan(tick_bid):
                continue

            # Check SL first (price drops below SL → market sell at bid)
            if sl_bps < 9000 and tick_bid <= sl_price:
                exit_price = tick_bid
                exit_type = "SL"
                exit_fee = SL_FEE_BPS
                break

            # Check TP (price rises above TP → limit sell, fills at TP price)
            if tp_bps > 0 and tick_bid >= tp_price:
                exit_price = tp_price  # limit order fills at our price
                exit_type = "TP"
                exit_fee = TP_FEE_BPS
                break

        # Timeout: exit at last available bid
        if exit_price is None:
            valid_bids = post_bid[~np.isnan(post_bid) & (post_bid > 0)]
            if len(valid_bids) == 0:
                continue
            exit_price = float(valid_bids[-1])
            exit_fee = SL_FEE_BPS  # market exit

        # P&L calculation
        price_pnl_bps = (exit_price - entry_ask) / entry_ask * 10000
        total_fee_bps = ENTRY_FEE_BPS + exit_fee
        net_bps = fr_collected_bps + price_pnl_bps - total_fee_bps
        net_usd = net_bps / 10000 * NOTIONAL

        results[name].append({
            "symbol": symbol,
            "settle_time": settle_time,
            "fr_bps": fr_collected_bps,
            "price_pnl_bps": price_pnl_bps,
            "fee_bps": total_fee_bps,
            "net_bps": net_bps,
            "net_usd": net_usd,
            "exit_type": exit_type,
        })

print(f"\n  Processed: {n_processed:,}, no_data: {n_no_data:,}")
print(f"  Phase 3 done [{time.time()-t3:.1f}s]")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Results
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("RESULTS")
print("=" * 90)

# Date range for daily P&L
t_min = settle_neg["settle_time"].min()
t_max = settle_neg["settle_time"].max()
days = (t_max - t_min).total_seconds() / 86400

print(f"Period: {t_min} to {t_max} ({days:.0f} days)")
print(f"Settlements with |FR| >= {MIN_FR_BPS} bps (negative): {len(settle_neg):,}")
print()

print(f"{'Config':>14} {'Trades':>7} {'WR':>5} {'Avg bps':>8} {'Daily':>8} {'ROI/yr':>8}  "
      f"{'TP%':>5} {'SL%':>5} {'TO%':>5} {'AvgFR':>7} {'AvgPrice':>9} {'AvgFee':>7}")
print("─" * 120)

for name, sl_bps, tp_bps in configs:
    trades = results[name]
    if not trades:
        print(f"{name:>14}       0")
        continue

    df = pd.DataFrame(trades)
    n = len(df)
    wr = (df["net_usd"] > 0).mean() * 100
    avg_bps = df["net_bps"].mean()
    total_usd = df["net_usd"].sum()
    daily = total_usd / days
    capital = NOTIONAL  # only futures, no spot leg!
    roi_yr = daily / capital * 365 * 100

    tp_pct = (df["exit_type"] == "TP").mean() * 100
    sl_pct = (df["exit_type"] == "SL").mean() * 100
    to_pct = (df["exit_type"] == "timeout").mean() * 100
    avg_fr = df["fr_bps"].mean()
    avg_price = df["price_pnl_bps"].mean()
    avg_fee = df["fee_bps"].mean()

    print(f"{name:>14} {n:>7} {wr:>4.0f}% {avg_bps:>+7.1f} ${daily:>+7,.0f} {roi_yr:>7.0f}%  "
          f"{tp_pct:>4.0f}% {sl_pct:>4.0f}% {to_pct:>4.0f}% {avg_fr:>+6.1f} {avg_price:>+8.1f} {avg_fee:>6.1f}")

# Detailed analysis of best config
print()
print("=" * 90)
print("DETAILED: Best configs")
print("=" * 90)

for name in ["SL50_TP0", "SL30_TP10", "NO_SL_TP0"]:
    trades = results[name]
    if not trades:
        continue
    df = pd.DataFrame(trades)
    df["settle_time"] = pd.to_datetime(df["settle_time"])

    print(f"\n  --- {name} ---")
    print(f"  Trades: {len(df):,}")
    print(f"  Win rate: {(df['net_usd'] > 0).mean()*100:.1f}%")
    print(f"  Total P&L: ${df['net_usd'].sum():+,.0f}")
    print(f"  Daily P&L: ${df['net_usd'].sum()/days:+,.0f}")
    print(f"  Capital: ${NOTIONAL:,} (futures only, no spot)")
    print(f"  Annual ROI: {df['net_usd'].sum()/days/NOTIONAL*365*100:+,.0f}%")

    # Distribution
    print(f"\n  P&L distribution (bps):")
    for pct in [5, 25, 50, 75, 95]:
        print(f"    {pct}th percentile: {df['net_bps'].quantile(pct/100):+.1f} bps")

    # Monthly
    df["month"] = df["settle_time"].dt.to_period("M")
    print(f"\n  Monthly:")
    for m in sorted(df["month"].unique()):
        mdf = df[df["month"] == m]
        mdays = max((mdf["settle_time"].max() - mdf["settle_time"].min()).total_seconds() / 86400, 1)
        print(f"    {str(m):>10}: {len(mdf):>4} trades, ${mdf['net_usd'].sum():>+8,.0f} "
              f"(${mdf['net_usd'].sum()/mdays:>+,.0f}/day, {(mdf['net_usd']>0).mean()*100:.0f}% WR)")

    # By FR bucket
    print(f"\n  By FR magnitude:")
    for lo, hi in [(20,30), (30,50), (50,100), (100,500)]:
        bucket = df[(df["fr_bps"] >= lo) & (df["fr_bps"] < hi)]
        if len(bucket) == 0:
            continue
        wr = (bucket["net_usd"] > 0).mean() * 100
        avg = bucket["net_bps"].mean()
        print(f"    FR {lo:>3}-{hi:>3} bps: {len(bucket):>5} trades, {wr:.0f}% WR, avg {avg:+.1f} bps")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Compare to delta-neutral strategy
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 90)
print("COMPARISON: FR Scalp Long vs Delta-Neutral")
print("=" * 90)

best_scalp = None
best_daily = -float("inf")
for name, _, _ in configs:
    trades = results[name]
    if trades:
        daily = sum(t["net_usd"] for t in trades) / days
        if daily > best_daily:
            best_daily = daily
            best_scalp = name

if best_scalp:
    print(f"\n  Best scalp config: {best_scalp}")
    print(f"  Scalp daily:    ${best_daily:+,.0f} on ${NOTIONAL:,} capital = {best_daily/NOTIONAL*365*100:.0f}% annual")
    print(f"  Delta-neutral:  ~$218/day on $20,000 capital = ~398% annual (Binance 1h, from audit)")
    print(f"\n  Key differences:")
    print(f"    Scalp: {NOTIONAL/1000:.0f}k capital (futures only), directional risk, lower fees")
    print(f"    DN:    20k capital (spot+futures), no directional risk, higher fees")

print(f"\n[{time.time()-t_global:.0f}s total]")
