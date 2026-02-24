#!/usr/bin/env python3
# ruff: noqa: E501
"""
Backtest: FR Flash Scalp — Enter just before settlement, exit just after.

Strategy:
  - When FR is very negative: LONG futures ~1min before settlement
  - FR settles: longs receive the payment
  - Exit ~1min after settlement
  - Total exposure: ~2 minutes → minimal directional risk
  - NO spot leg, NO margin borrowing needed

Fee model:
  - Entry: market 5.5 bps taker
  - Exit: market 5.5 bps taker (need to get out fast)
  - OR exit: limit 2.0 bps maker (if we can get filled quickly)
  - Total RT: 11 bps (market/market) or 7.5 bps (market/limit)

The question: does FR income (avg ~40 bps) cover the ~11 bps fees + the 
~2 min of price noise?

Uses settlement_klines.parquet with ±10min of 1m Bybit candles.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

import builtins
_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True)
    _print(*a, **k)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data_all"

NOTIONAL = 10_000
ENTRY_FEE_BPS = 5.5   # market taker
EXIT_MARKET_BPS = 5.5  # market taker
EXIT_LIMIT_BPS = 2.0   # limit maker
MIN_FR_BPS = 15.0      # test with lower threshold too

t0 = time.time()
print("=" * 100)
print("BACKTEST: FR Flash Scalp — LONG before settlement, EXIT after (~2min exposure)")
print("=" * 100)

# Load data
print("\n1. Loading data...", flush=True)

klines = pd.read_parquet(DATA / "historical_fr" / "settlement_klines.parquet")
bb_fr = pd.read_parquet(DATA / "historical_fr" / "bybit_fr_history.parquet")
bb_fr["fr_bps"] = bb_fr["fundingRate"] * 10000
bb_fr["settle_ts_ms"] = bb_fr["fundingTime"].astype(np.int64) // 10**6

klines = klines.merge(
    bb_fr[["symbol", "settle_ts_ms", "fr_bps"]].drop_duplicates(),
    on=["symbol", "settle_ts_ms"], how="inner"
)
klines["offset_min"] = (klines["ts_ms"] - klines["settle_ts_ms"]) / 60000
klines = klines.sort_values(["symbol", "settle_ts_ms", "ts_ms"])

days = (bb_fr["fundingTime"].max() - bb_fr["fundingTime"].min()).total_seconds() / 86400
print(f"   Klines: {len(klines):,}, Period: {days:.0f} days")

# Pre-group
print("   Pre-grouping...", flush=True)
groups = {}
for (sym, sts), g in klines.groupby(["symbol", "settle_ts_ms"]):
    g = g.sort_values("ts_ms")
    groups[(sym, sts)] = {
        "offset": g["offset_min"].values,
        "open": g["open"].values,
        "high": g["high"].values,
        "low": g["low"].values,
        "close": g["close"].values,
        "fr_bps": float(g["fr_bps"].iloc[0]),
    }
print(f"   {len(groups):,} settlement windows")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Flash scalp simulation
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("2. Flash Scalp Simulation")
print("=" * 100, flush=True)

# Config: (name, entry_offset_min, exit_offset_min, exit_fee_bps, min_fr)
# entry_offset: minutes before settlement (negative = before)
# exit_offset: minutes after settlement (positive = after)
configs = [
    # Quick scalp: enter 1min before, exit 1min after
    ("E-1_X+1_mkt",    -1, 1, EXIT_MARKET_BPS, 20),
    ("E-1_X+1_lmt",    -1, 1, EXIT_LIMIT_BPS,  20),
    ("E-1_X+2_mkt",    -1, 2, EXIT_MARKET_BPS, 20),
    ("E-1_X+2_lmt",    -1, 2, EXIT_LIMIT_BPS,  20),
    ("E-2_X+1_mkt",    -2, 1, EXIT_MARKET_BPS, 20),
    ("E-2_X+2_mkt",    -2, 2, EXIT_MARKET_BPS, 20),
    ("E-1_X+3_mkt",    -1, 3, EXIT_MARKET_BPS, 20),
    ("E-1_X+5_mkt",    -1, 5, EXIT_MARKET_BPS, 20),
    # Lower FR threshold
    ("E-1_X+1_fr15",   -1, 1, EXIT_MARKET_BPS, 15),
    ("E-1_X+1_fr30",   -1, 1, EXIT_MARKET_BPS, 30),
    ("E-1_X+1_fr50",   -1, 1, EXIT_MARKET_BPS, 50),
    # Enter even closer
    ("E-0_X+1_mkt",     0, 1, EXIT_MARKET_BPS, 20),  # enter at settlement candle open
]

results = {}

for name, entry_off, exit_off, exit_fee, min_fr in configs:
    trades = []
    
    for (sym, sts), g in groups.items():
        fr = g["fr_bps"]
        if fr > -min_fr:  # only negative FR (we go long to collect)
            continue
        
        fr_collected = abs(fr)
        offsets = g["offset"]
        
        # Find entry candle: closest to entry_offset
        entry_idx = np.argmin(np.abs(offsets - entry_off))
        if abs(offsets[entry_idx] - entry_off) > 0.6:
            continue
        
        # Find exit candle: closest to exit_offset  
        exit_idx = np.argmin(np.abs(offsets - exit_off))
        if abs(offsets[exit_idx] - exit_off) > 0.6:
            continue
        
        entry_price = g["close"][entry_idx]  # enter at close of entry candle
        exit_price = g["close"][exit_idx]    # exit at close of exit candle
        
        if entry_price <= 0 or exit_price <= 0:
            continue
        if np.isnan(entry_price) or np.isnan(exit_price):
            continue
        
        # Also track worst drawdown during hold
        hold_slice = slice(entry_idx, exit_idx + 1)
        hold_lows = g["low"][hold_slice]
        if len(hold_lows) > 0:
            worst_low = np.nanmin(hold_lows)
            max_dd_bps = (entry_price - worst_low) / entry_price * 10000
        else:
            max_dd_bps = 0
        
        price_pnl = (exit_price - entry_price) / entry_price * 10000
        total_fee = ENTRY_FEE_BPS + exit_fee
        net = fr_collected + price_pnl - total_fee
        
        trades.append({
            "symbol": sym, "settle_ts_ms": sts, "fr_bps": fr_collected,
            "price_pnl_bps": price_pnl, "fee_bps": total_fee,
            "net_bps": net, "net_usd": net / 10000 * NOTIONAL,
            "max_dd_bps": max_dd_bps,
        })
    
    results[name] = trades

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"RESULTS — {days:.0f} days")
print("=" * 100)

print(f"\n{'Config':>18} {'N':>6} {'WR':>5} {'AvgNet':>8} {'Daily':>8} {'ROI/yr':>8}  "
      f"{'AvgFR':>7} {'AvgPx':>8} {'Fee':>5} {'AvgDD':>7}")
print("─" * 105)

sorted_results = []
for name, entry_off, exit_off, exit_fee, min_fr in configs:
    trades = results[name]
    if not trades:
        sorted_results.append((name, 0, []))
        continue
    df = pd.DataFrame(trades)
    daily = df["net_usd"].sum() / days
    sorted_results.append((name, daily, trades))

sorted_results.sort(key=lambda x: -x[1])

for name, daily, trades in sorted_results:
    if not trades:
        print(f"{name:>18}      0")
        continue
    df = pd.DataFrame(trades)
    n = len(df)
    wr = (df["net_usd"] > 0).mean() * 100
    avg = df["net_bps"].mean()
    roi = daily / NOTIONAL * 365 * 100
    avg_fr = df["fr_bps"].mean()
    avg_px = df["price_pnl_bps"].mean()
    fee = df["fee_bps"].iloc[0]
    avg_dd = df["max_dd_bps"].mean()
    marker = " <<<" if daily > 0 else ""
    print(f"{name:>18} {n:>6} {wr:>4.0f}% {avg:>+7.1f} ${daily:>+7,.0f} {roi:>7.0f}%  "
          f"{avg_fr:>+6.1f} {avg_px:>+7.1f} {fee:>4.1f} {avg_dd:>6.1f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
# DETAILED on best configs
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("DETAILED ANALYSIS — Top Configs")
print("=" * 100)

top = [x[0] for x in sorted_results[:4] if x[2]]

for name in top:
    trades = results[name]
    df = pd.DataFrame(trades)
    daily = df["net_usd"].sum() / days

    print(f"\n  --- {name} ({len(df):,} trades, ${daily:+,.0f}/day, {daily/NOTIONAL*365*100:+,.0f}% annual) ---")
    
    # P&L decomposition
    print(f"  P&L decomposition:")
    print(f"    FR income:   {df['fr_bps'].mean():+.1f} bps avg")
    print(f"    Price move:  {df['price_pnl_bps'].mean():+.1f} bps avg (std: {df['price_pnl_bps'].std():.1f})")
    print(f"    Fees:        -{df['fee_bps'].iloc[0]:.1f} bps")
    print(f"    Net:         {df['net_bps'].mean():+.1f} bps avg")
    
    print(f"  Distribution (bps): "
          f"5th={df['net_bps'].quantile(0.05):+.0f}, "
          f"25th={df['net_bps'].quantile(0.25):+.0f}, "
          f"med={df['net_bps'].median():+.0f}, "
          f"75th={df['net_bps'].quantile(0.75):+.0f}, "
          f"95th={df['net_bps'].quantile(0.95):+.0f}")
    print(f"  Worst: {df['net_bps'].min():+.0f} bps | Best: {df['net_bps'].max():+.0f} bps")
    print(f"  Avg max drawdown during hold: {df['max_dd_bps'].mean():.1f} bps")

    # By FR magnitude
    print(f"  By FR magnitude:")
    for lo, hi in [(15,20), (20,30), (30,50), (50,100), (100,500)]:
        b = df[(df["fr_bps"] >= lo) & (df["fr_bps"] < hi)]
        if len(b) == 0: continue
        wr = (b["net_usd"] > 0).mean() * 100
        print(f"    FR {lo:>3}-{hi:>3}: {len(b):>5} trades, {wr:.0f}% WR, "
              f"net {b['net_bps'].mean():+.1f}, px {b['price_pnl_bps'].mean():+.1f}")

    # Monthly
    df["month"] = pd.to_datetime(df["settle_ts_ms"], unit="ms").dt.to_period("M")
    monthly = df.groupby("month").agg(
        n=("net_usd", "count"), total=("net_usd", "sum"),
        wr=("net_usd", lambda x: (x > 0).mean() * 100),
    )
    all_pos = all(monthly["total"] > 0)
    pos_months = (monthly["total"] > 0).sum()
    print(f"  Monthly ({pos_months}/{len(monthly)} positive):")
    for m, r in monthly.iterrows():
        d = r["total"] / 30
        print(f"    {str(m):>10}: {int(r['n']):>4} trades, ${r['total']:>+8,.0f} "
              f"(${d:>+,.0f}/day, {r['wr']:.0f}% WR)")

# ═══════════════════════════════════════════════════════════════════════════════
# RISK ANALYSIS: What happens in the ~2min window?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("RISK ANALYSIS: Price behavior in the ~2min settlement window")
print("=" * 100)

# Use the E-1_X+1_mkt trades
trades = results.get("E-1_X+1_mkt", [])
if trades:
    df = pd.DataFrame(trades)
    px = df["price_pnl_bps"]
    print(f"\n  Price move distribution (E-1 to X+1, ~2min):")
    print(f"    Mean:   {px.mean():+.1f} bps")
    print(f"    Std:    {px.std():.1f} bps")
    print(f"    1st:    {px.quantile(0.01):+.0f} bps")
    print(f"    5th:    {px.quantile(0.05):+.0f} bps")
    print(f"    25th:   {px.quantile(0.25):+.0f} bps")
    print(f"    Median: {px.median():+.0f} bps")
    print(f"    75th:   {px.quantile(0.75):+.0f} bps")
    print(f"    95th:   {px.quantile(0.95):+.0f} bps")
    print(f"    99th:   {px.quantile(0.99):+.0f} bps")
    
    # What % of trades have price move > FR (i.e., FR doesn't cover the loss)?
    net_neg = df[df["net_bps"] < 0]
    print(f"\n  Losing trades: {len(net_neg)} ({len(net_neg)/len(df)*100:.0f}%)")
    if len(net_neg) > 0:
        print(f"    Avg loss: {net_neg['net_bps'].mean():+.1f} bps")
        print(f"    Worst loss: {net_neg['net_bps'].min():+.0f} bps")
    
    # Profit factor
    wins = df[df["net_usd"] > 0]["net_usd"].sum()
    losses = abs(df[df["net_usd"] < 0]["net_usd"].sum())
    pf = wins / losses if losses > 0 else float("inf")
    print(f"\n  Profit factor: {pf:.2f}")
    print(f"  Win $: ${wins:,.0f} | Loss $: ${losses:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# CAPACITY: How many trades per day?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("CAPACITY & SCALING")
print("=" * 100)

trades = results.get("E-1_X+1_mkt", [])
if trades:
    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["settle_ts_ms"], unit="ms").dt.date
    daily_counts = df.groupby("date").size()
    
    print(f"\n  Trades per day: mean={daily_counts.mean():.0f}, "
          f"median={daily_counts.median():.0f}, max={daily_counts.max()}")
    print(f"\n  At $10k per trade:")
    d10k = df["net_usd"].sum() / days
    print(f"    Daily: ${d10k:+,.0f}")
    print(f"    Annual: ${d10k*365:+,.0f}")
    
    # What if we run 3-5 simultaneous positions?
    # Settlements happen at the same time, so we CAN do multiple coins
    for max_pos in [1, 3, 5, 10]:
        # Limit to top-N coins per settlement by FR magnitude
        df2 = df.copy()
        df2["settle_hour"] = pd.to_datetime(df2["settle_ts_ms"], unit="ms").dt.floor("h")
        
        selected = df2.sort_values("fr_bps", ascending=False).groupby("settle_hour").head(max_pos)
        daily_pnl = selected["net_usd"].sum() / days
        print(f"\n  Top {max_pos} per settlement hour:")
        print(f"    Trades: {len(selected):,} ({len(selected)/len(df)*100:.0f}% of all)")
        print(f"    Daily: ${daily_pnl:+,.0f}")
        print(f"    Capital: ${max_pos * NOTIONAL:,}")
        print(f"    Annual ROI: {daily_pnl / (max_pos * NOTIONAL) * 365 * 100:.0f}%")

# Final comparison
print(f"\n{'='*100}")
print(f"FINAL COMPARISON")
print("=" * 100)

best = sorted_results[0] if sorted_results[0][1] > 0 else None
print(f"\n  {'Strategy':>40} {'Daily':>8} {'Capital':>8} {'ROI/yr':>8} {'Borrow?':>8}")
print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
if best:
    d = best[1]
    print(f"  {'FR Flash Scalp (best)':>40} ${d:>+7,.0f} $    10k {d/10000*365*100:>7.0f}%       NO")
print(f"  {'Naked long full-hold':>40} $    -45 $    10k    -164%       NO")
print(f"  {'Delta-neutral (audit, theoretical)':>40} $   +273 $    20k     498%      YES")
print(f"  {'Delta-neutral (real borrow ~$200)':>40} $     +5 $      0.4k   498%      YES")

print(f"\n  * Flash scalp needs NO margin borrowing — pure futures trade")
print(f"  * Can scale across multiple coins simultaneously at each settlement")

print(f"\n[{time.time()-t0:.0f}s total]")
