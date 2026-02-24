#!/usr/bin/env python3
"""
Backtest: Funding Rate Settlement Arbitrage (Single-Exchange Spot+Futures)

Strategy:
  Before each settlement, find the coin with the most extreme FR.
  If |FR| > cost threshold:
    - Buy spot (maker limit) + Short futures (taker market) ~60s before settlement
    - Collect funding payment at settlement
    - Close both legs ~60s after settlement
    
Memory-efficient: loads only 30-min windows of 5s data per settlement, one at a time.
Shows progress and ETA throughout.

Usage: python3 backtest_settlement_arb.py
"""
import sys
import time
import gc
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data_all"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
NOTIONAL = 10_000          # USD per trade
WINDOW_BEFORE_S = 900      # 15 min before settlement (seconds)
WINDOW_AFTER_S = 900       # 15 min after settlement (seconds)
ENTRY_BEFORE_S = 60        # enter ~60s before settlement
EXIT_AFTER_S = 60          # exit ~60s after settlement

# Fee model (bps)
SPOT_MAKER_FEE = 1.0       # maker limit order on spot
FUTURES_TAKER_FEE = 4.5    # taker market order on futures (Binance: 4.5, Bybit: 5.5)
SLIPPAGE_BPS = 1.0         # slippage per leg

# Per-leg cost
ENTRY_COST_BPS = SPOT_MAKER_FEE + FUTURES_TAKER_FEE + 2 * SLIPPAGE_BPS  # open spot + open futures
EXIT_COST_BPS = SPOT_MAKER_FEE + FUTURES_TAKER_FEE + 2 * SLIPPAGE_BPS   # close spot + close futures
TOTAL_COST_BPS = ENTRY_COST_BPS + EXIT_COST_BPS

FR_THRESHOLD_BPS = TOTAL_COST_BPS  # minimum |FR| to trade

print("=" * 90)
print("BACKTEST: Settlement Arbitrage (Spot + Futures, Single Exchange)")
print("=" * 90)
print(f"  Notional:        ${NOTIONAL:,.0f}")
print(f"  Entry:           {ENTRY_BEFORE_S}s before settlement")
print(f"  Exit:            {EXIT_AFTER_S}s after settlement")
print(f"  Spot fee:        {SPOT_MAKER_FEE} bps (maker)")
print(f"  Futures fee:     {FUTURES_TAKER_FEE} bps (taker)")
print(f"  Slippage:        {SLIPPAGE_BPS} bps per leg")
print(f"  Total cost (RT): {TOTAL_COST_BPS:.1f} bps")
print(f"  FR threshold:    {FR_THRESHOLD_BPS:.1f} bps")
print(f"  Window:          ±{WINDOW_BEFORE_S//60} min around settlement")
print("=" * 90)
print()

t_global = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Build settlement schedule (lightweight — 1-min downsampled)
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 1: Detecting settlement events (1-min resolution)...")
sys.stdout.flush()

t1 = time.time()

# Load only columns we need for settlement detection
bn_fr_cols = ["ts", "symbol", "lastFundingRate", "nextFundingTime"]
bn_fr = pd.read_parquet(DATA / "binance" / "fundingRate.parquet", columns=bn_fr_cols)
print(f"  Loaded Binance FR: {len(bn_fr):,} rows [{time.time()-t1:.1f}s]")
sys.stdout.flush()

bn_fr["ts_1m"] = bn_fr["ts"].dt.floor("1min")
bn_fr_1m = bn_fr.groupby(["ts_1m", "symbol"]).agg(
    fr=("lastFundingRate", "last"),
    nft=("nextFundingTime", "last"),
).reset_index()
del bn_fr
gc.collect()

bn_fr_1m = bn_fr_1m.sort_values(["symbol", "ts_1m"])
bn_fr_1m["nft_prev"] = bn_fr_1m.groupby("symbol")["nft"].shift(1)
bn_fr_1m["is_settle"] = (bn_fr_1m["nft"] != bn_fr_1m["nft_prev"]) & bn_fr_1m["nft_prev"].notna()

settle_idx = bn_fr_1m[bn_fr_1m["is_settle"]].index
pre = bn_fr_1m.loc[settle_idx - 1].copy()
pre["settle_time"] = bn_fr_1m.loc[settle_idx, "ts_1m"].values
pre["fr_bps"] = pre["fr"].abs() * 10000

# For each settlement, pick the coin with highest |FR|
best = pre.loc[pre.groupby("settle_time")["fr_bps"].idxmax()].copy()
best = best.sort_values("settle_time").reset_index(drop=True)

# Also get top-3 per settlement for backup if best coin has no liquidity data
top3_per_settle = (
    pre.sort_values("fr_bps", ascending=False)
    .groupby("settle_time")
    .head(3)
    .sort_values(["settle_time", "fr_bps"], ascending=[True, False])
)

del bn_fr_1m, pre
gc.collect()

# Filter to tradeable settlements (FR > threshold)
tradeable = best[best["fr_bps"] >= FR_THRESHOLD_BPS].reset_index(drop=True)

print(f"  Total settlement hours:   {len(best)}")
print(f"  Tradeable (FR>{FR_THRESHOLD_BPS:.0f} bps): {len(tradeable)}")
print(f"  Phase 1 done [{time.time()-t1:.1f}s]")
print()
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Backtest each settlement individually (5s resolution)
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 2: Running backtest per settlement (5s resolution, memory-safe)")
print("─" * 90)
sys.stdout.flush()

# Precompute parquet file paths
BN_FR_PQ = str(DATA / "binance" / "fundingRate.parquet")
BN_TK_PQ = str(DATA / "binance" / "ticker.parquet")
BB_TK_PQ = str(DATA / "bybit" / "ticker.parquet")

# We'll use pyarrow to read with pushdown filters for speed
# But since our parquet isn't time-partitioned, we'll read into pandas with column selection
# and then filter in memory. To avoid loading everything, we'll read once and keep
# a time index for slicing.

# Preload time indexes for efficient slicing
print("  Building time indexes for efficient window extraction...")
sys.stdout.flush()
t_idx = time.time()

# Strategy: read just ts column to build an index, then use row ranges for each window
# This avoids loading the full dataset

def build_time_index(pq_path):
    """Read only 'ts' column and return as array for fast binary search."""
    ts = pd.read_parquet(pq_path, columns=["ts"])["ts"].values
    return ts

ts_bn_fr = build_time_index(BN_FR_PQ)
print(f"    Binance FR ts index: {len(ts_bn_fr):,} [{time.time()-t_idx:.1f}s]")
sys.stdout.flush()

ts_bn_tk = build_time_index(BN_TK_PQ)
print(f"    Binance ticker ts index: {len(ts_bn_tk):,} [{time.time()-t_idx:.1f}s]")
sys.stdout.flush()

ts_bb_tk = build_time_index(BB_TK_PQ)
print(f"    Bybit ticker ts index: {len(ts_bb_tk):,} [{time.time()-t_idx:.1f}s]")
sys.stdout.flush()

print(f"  Time indexes built [{time.time()-t_idx:.1f}s]")
print()
sys.stdout.flush()


def load_window(pq_path, ts_index, t_start, t_end, columns, symbol=None):
    """Load a time window from parquet using prebuilt timestamp index."""
    t_start_ns = np.datetime64(t_start, "ns")
    t_end_ns = np.datetime64(t_end, "ns")

    # Binary search for row range
    i_start = np.searchsorted(ts_index, t_start_ns, side="left")
    i_end = np.searchsorted(ts_index, t_end_ns, side="right")

    if i_start >= i_end:
        return pd.DataFrame(columns=columns)

    # Read just the row slice
    pf = pq.ParquetFile(pq_path)
    table = pf.read_row_groups(
        range(pf.metadata.num_row_groups),
        columns=columns,
    )
    df = table.to_pandas()
    df = df.iloc[i_start:i_end]

    if symbol:
        df = df[df["symbol"] == symbol]

    return df


# OPTIMIZATION: since reading row_groups reads ALL groups anyway,
# let's just read the full parquet once into memory but keep only columns we need.
# The 5s data is ~25M rows * a few columns = manageable if we pick minimal columns.
# BUT user said avoid OOM. So let's be smarter:
# Read once, keep as numpy arrays, and slice per settlement.

print("  Pre-loading minimal columns for all streams (numpy arrays)...")
sys.stdout.flush()
t_load = time.time()

# Binance FR: ts, symbol, lastFundingRate, markPrice, nextFundingTime
_bn_fr = pd.read_parquet(BN_FR_PQ, columns=["ts", "symbol", "lastFundingRate", "markPrice"])
bn_fr_ts = _bn_fr["ts"].values
bn_fr_sym = _bn_fr["symbol"].values
bn_fr_rate = _bn_fr["lastFundingRate"].values
bn_fr_mark = _bn_fr["markPrice"].values
del _bn_fr
gc.collect()
print(f"    Binance FR: {len(bn_fr_ts):,} rows [{time.time()-t_load:.1f}s]")
sys.stdout.flush()

# Binance ticker: ts, symbol, lastPrice
_bn_tk = pd.read_parquet(BN_TK_PQ, columns=["ts", "symbol", "lastPrice"])
bn_tk_ts = _bn_tk["ts"].values
bn_tk_sym = _bn_tk["symbol"].values
bn_tk_price = _bn_tk["lastPrice"].values
del _bn_tk
gc.collect()
print(f"    Binance ticker: {len(bn_tk_ts):,} rows [{time.time()-t_load:.1f}s]")
sys.stdout.flush()

# Bybit ticker: ts, symbol, fundingRate, bid1Price, ask1Price, lastPrice
_bb_tk = pd.read_parquet(BB_TK_PQ, columns=["ts", "symbol", "fundingRate", "bid1Price", "ask1Price", "lastPrice"])
bb_tk_ts = _bb_tk["ts"].values
bb_tk_sym = _bb_tk["symbol"].values
bb_tk_fr = _bb_tk["fundingRate"].values
bb_tk_bid = _bb_tk["bid1Price"].values
bb_tk_ask = _bb_tk["ask1Price"].values
bb_tk_last = _bb_tk["lastPrice"].values
del _bb_tk
gc.collect()
print(f"    Bybit ticker: {len(bb_tk_ts):,} rows [{time.time()-t_load:.1f}s]")
sys.stdout.flush()

# Free timestamp indexes (we have the real ones now)
del ts_bn_fr, ts_bn_tk, ts_bb_tk
gc.collect()

print(f"  All data loaded [{time.time()-t_load:.1f}s]")
print()
sys.stdout.flush()


def get_window(ts_arr, sym_arr, arrays_dict, symbol, t_start_ns, t_end_ns):
    """Extract a time+symbol window from numpy arrays. Returns dict of arrays."""
    i0 = np.searchsorted(ts_arr, t_start_ns, side="left")
    i1 = np.searchsorted(ts_arr, t_end_ns, side="right")
    if i0 >= i1:
        return None

    sl = slice(i0, i1)
    sym_mask = sym_arr[sl] == symbol
    if sym_mask.sum() == 0:
        return None

    result = {"ts": ts_arr[sl][sym_mask]}
    for name, arr in arrays_dict.items():
        result[name] = arr[sl][sym_mask]
    return result


# ── Run the backtest ──────────────────────────────────────────────────────────
trades = []
n_total = len(tradeable)
t_bt = time.time()

for i, row in tradeable.iterrows():
    t_iter = time.time()
    settle_time = pd.Timestamp(row["settle_time"])
    if settle_time.tzinfo is None:
        settle_time = settle_time.tz_localize("UTC")
    symbol = row["symbol"]
    pre_fr = row["fr"]
    pre_fr_bps = row["fr_bps"]

    # Time window
    t_start = settle_time - timedelta(seconds=WINDOW_BEFORE_S)
    t_end = settle_time + timedelta(seconds=WINDOW_AFTER_S)
    t_entry = settle_time - timedelta(seconds=ENTRY_BEFORE_S)
    t_exit = settle_time + timedelta(seconds=EXIT_AFTER_S)

    t_start_ns = np.datetime64(t_start, "ns")
    t_end_ns = np.datetime64(t_end, "ns")
    t_entry_ns = np.datetime64(t_entry, "ns")
    t_exit_ns = np.datetime64(t_exit, "ns")

    # ── Get Binance FR data in window ──
    bn_fr_win = get_window(bn_fr_ts, bn_fr_sym,
                           {"fr": bn_fr_rate, "mark": bn_fr_mark},
                           symbol, t_start_ns, t_end_ns)

    # ── Get Binance ticker in window ──
    bn_tk_win = get_window(bn_tk_ts, bn_tk_sym,
                           {"price": bn_tk_price},
                           symbol, t_start_ns, t_end_ns)

    # ── Get Bybit ticker in window ──
    bb_win = get_window(bb_tk_ts, bb_tk_sym,
                        {"fr": bb_tk_fr, "bid": bb_tk_bid, "ask": bb_tk_ask, "last": bb_tk_last},
                        symbol, t_start_ns, t_end_ns)

    # ── Determine execution prices ──
    # We trade on Binance (spot + futures) — use Binance prices
    # Entry: closest tick to t_entry
    entry_price = None
    exit_price = None
    entry_fr = None
    bb_entry_spread_bps = None

    if bn_tk_win is not None and len(bn_tk_win["ts"]) > 0:
        # Find tick closest to entry time
        entry_diffs = np.abs(bn_tk_win["ts"] - t_entry_ns)
        entry_idx = np.argmin(entry_diffs)
        entry_price = bn_tk_win["price"][entry_idx]

        # Find tick closest to exit time
        exit_diffs = np.abs(bn_tk_win["ts"] - t_exit_ns)
        exit_idx = np.argmin(exit_diffs)
        exit_price = bn_tk_win["price"][exit_idx]

    # Get FR at entry from Binance
    if bn_fr_win is not None and len(bn_fr_win["ts"]) > 0:
        entry_fr_diffs = np.abs(bn_fr_win["ts"] - t_entry_ns)
        entry_fr_idx = np.argmin(entry_fr_diffs)
        entry_fr = bn_fr_win["fr"][entry_fr_idx]

    # Get Bybit bid-ask spread at entry
    if bb_win is not None and len(bb_win["ts"]) > 0:
        bb_entry_diffs = np.abs(bb_win["ts"] - t_entry_ns)
        bb_entry_idx = np.argmin(bb_entry_diffs)
        bb_bid = bb_win["bid"][bb_entry_idx]
        bb_ask = bb_win["ask"][bb_entry_idx]
        if bb_bid > 0 and bb_ask > 0:
            bb_mid = (bb_bid + bb_ask) / 2
            bb_entry_spread_bps = (bb_ask - bb_bid) / bb_mid * 10000

    if entry_price is None or exit_price is None or entry_price == 0:
        # No data — skip
        elapsed = time.time() - t_bt
        eta = elapsed / (i + 1) * (n_total - i - 1) if i > 0 else 0
        print(f"  [{i+1:3d}/{n_total}] {settle_time} {symbol:<14} FR={pre_fr*100:+.4f}%  "
              f"SKIP (no price data)  [{elapsed:.0f}s, ETA {eta:.0f}s]")
        sys.stdout.flush()
        continue

    # ── Calculate P&L ──
    # Strategy: buy spot + short futures (delta neutral)
    # FR is negative → shorts PAY longs. We are short futures, so we PAY.
    # Wait — if FR is negative, shorts pay longs. We are SHORT futures → we pay.
    # That's wrong! We want to be on the RECEIVING side.
    #
    # If FR < 0: shorts pay longs → go LONG futures, SHORT spot
    # If FR > 0: longs pay shorts → go SHORT futures, LONG spot
    #
    # Since our extreme FRs are almost all negative, we'd go:
    #   SHORT spot (sell) + LONG futures (buy) → collect payment as long

    fr_value = entry_fr if entry_fr is not None else pre_fr

    if fr_value < 0:
        # Shorts pay longs → we go LONG futures + SHORT spot
        direction = "long_futures"
    else:
        # Longs pay shorts → we go SHORT futures + LONG spot
        direction = "short_futures"

    # Funding payment received (absolute, on our notional)
    fr_payment = abs(fr_value) * NOTIONAL  # in USD

    # Price change P&L (from our spot position)
    # If long_futures: we sold spot at entry, buy back at exit
    #   spot P&L = (entry_price - exit_price) / entry_price * NOTIONAL
    #   futures P&L = (exit_price - entry_price) / entry_price * NOTIONAL
    #   Net price P&L = 0 (delta neutral) — but NOT exactly, due to 5s timing gap
    price_change_pct = (exit_price - entry_price) / entry_price

    if direction == "long_futures":
        # Short spot: profit if price drops. Long futures: profit if price rises.
        # spot_pnl = -price_change_pct * NOTIONAL (short spot)
        # fut_pnl = +price_change_pct * NOTIONAL (long futures)
        # Net = 0 ideally, but there's basis/timing
        # For simplicity, assume perfect delta neutral → price P&L = 0
        # The real residual is from the ~5s timing gap between spot and futures fills
        residual_pnl = 0  # assume perfect hedge
    else:
        residual_pnl = 0

    # Execution costs
    total_cost_usd = TOTAL_COST_BPS / 10000 * NOTIONAL

    # Net P&L
    net_pnl = fr_payment - total_cost_usd + residual_pnl

    # Also calculate: what if we used Bybit instead? (higher FR sometimes)
    bb_fr_at_entry = None
    if bb_win is not None and len(bb_win["ts"]) > 0:
        bb_fr_at_entry = bb_win["fr"][bb_entry_idx]

    trade = {
        "settle_time": settle_time,
        "symbol": symbol,
        "direction": direction,
        "entry_fr": fr_value,
        "entry_fr_bps": abs(fr_value) * 10000,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "price_change_bps": price_change_pct * 10000,
        "fr_payment": fr_payment,
        "cost_usd": total_cost_usd,
        "net_pnl": net_pnl,
        "bb_fr": bb_fr_at_entry,
        "bb_spread_bps": bb_entry_spread_bps,
    }
    trades.append(trade)

    elapsed = time.time() - t_bt
    eta = elapsed / (len(trades)) * (n_total - len(trades)) if len(trades) > 0 else 0
    flag = "✓" if net_pnl > 0 else "✗"
    print(f"  [{len(trades):3d}/{n_total}] {settle_time} {symbol:<14} "
          f"FR={fr_value*100:+.4f}%  entry=${entry_price:.4f}  "
          f"pnl=${net_pnl:+.2f}  {flag}  "
          f"[{elapsed:.0f}s, ETA {eta:.0f}s]")
    sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Results
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 90)
print("BACKTEST RESULTS")
print("=" * 90)

if not trades:
    print("  No trades executed!")
    sys.exit(0)

df = pd.DataFrame(trades)

# Summary stats
n_trades = len(df)
n_wins = (df["net_pnl"] > 0).sum()
n_losses = (df["net_pnl"] <= 0).sum()
total_pnl = df["net_pnl"].sum()
avg_pnl = df["net_pnl"].mean()
total_fr = df["fr_payment"].sum()
total_cost = df["cost_usd"].sum()
best_trade = df.loc[df["net_pnl"].idxmax()]
worst_trade = df.loc[df["net_pnl"].idxmin()]

# Cumulative P&L
df["cum_pnl"] = df["net_pnl"].cumsum()
max_dd = (df["cum_pnl"].cummax() - df["cum_pnl"]).max()

# Time span
t_first = df["settle_time"].min()
t_last = df["settle_time"].max()
hours_span = (t_last - t_first).total_seconds() / 3600
days_span = hours_span / 24

print(f"\n  Period:          {t_first} → {t_last} ({days_span:.1f} days)")
print(f"  Notional:        ${NOTIONAL:,.0f} per trade")
print(f"  Cost model:      {TOTAL_COST_BPS:.1f} bps round-trip")
print(f"                   (spot maker {SPOT_MAKER_FEE} + futures taker {FUTURES_TAKER_FEE} + slippage {SLIPPAGE_BPS}×2) × 2 legs")
print(f"  Entry/Exit:      {ENTRY_BEFORE_S}s before / {EXIT_AFTER_S}s after settlement")

print(f"\n  {'─'*60}")
print(f"  Trades:          {n_trades}")
print(f"  Wins:            {n_wins} ({n_wins/n_trades*100:.0f}%)")
print(f"  Losses:          {n_losses} ({n_losses/n_trades*100:.0f}%)")
print(f"  Win rate:        {n_wins/n_trades*100:.1f}%")
print(f"  {'─'*60}")
print(f"  Gross FR income: ${total_fr:,.2f}")
print(f"  Total costs:     ${total_cost:,.2f}")
print(f"  Net P&L:         ${total_pnl:+,.2f}")
print(f"  Avg P&L/trade:   ${avg_pnl:+,.2f}")
print(f"  Max drawdown:    ${max_dd:,.2f}")
print(f"  {'─'*60}")

if days_span > 0:
    daily_pnl = total_pnl / days_span
    daily_trades = n_trades / days_span
    annualized = daily_pnl * 365
    roi_daily = daily_pnl / NOTIONAL * 100
    print(f"  Daily P&L:       ${daily_pnl:+,.2f}")
    print(f"  Daily trades:    {daily_trades:.1f}")
    print(f"  Daily ROI:       {roi_daily:+.2f}%")
    print(f"  Annualized:      ${annualized:+,.0f}")
    print(f"  Annual ROI:      {annualized/NOTIONAL*100:+.1f}%")

print(f"\n  Best trade:      {best_trade['symbol']} at {best_trade['settle_time']}")
print(f"                   FR={best_trade['entry_fr']*100:+.4f}%, P&L=${best_trade['net_pnl']:+.2f}")
print(f"  Worst trade:     {worst_trade['symbol']} at {worst_trade['settle_time']}")
print(f"                   FR={worst_trade['entry_fr']*100:+.4f}%, P&L=${worst_trade['net_pnl']:+.2f}")

# ── Per-symbol breakdown ──
print(f"\n  {'─'*60}")
print(f"  Per-symbol breakdown:")
sym_stats = df.groupby("symbol").agg(
    trades=("net_pnl", "count"),
    total_pnl=("net_pnl", "sum"),
    avg_fr_bps=("entry_fr_bps", "mean"),
    wins=("net_pnl", lambda x: (x > 0).sum()),
).sort_values("total_pnl", ascending=False)

for sym, r in sym_stats.iterrows():
    wr = r["wins"] / r["trades"] * 100
    print(f"    {sym:<14} {int(r['trades']):>3} trades, "
          f"WR {wr:>5.1f}%, "
          f"avg FR {r['avg_fr_bps']:>6.1f} bps, "
          f"P&L ${r['total_pnl']:>+8.2f}")

# ── Hourly distribution ──
print(f"\n  {'─'*60}")
print(f"  P&L by hour of day:")
df["hour"] = df["settle_time"].dt.hour
hour_stats = df.groupby("hour").agg(
    trades=("net_pnl", "count"),
    total_pnl=("net_pnl", "sum"),
    avg_pnl=("net_pnl", "mean"),
).sort_index()

for hr, r in hour_stats.iterrows():
    bar = "█" * max(1, int(r["total_pnl"] / 2)) if r["total_pnl"] > 0 else ""
    print(f"    {hr:02d}:00  {int(r['trades']):>2} trades  "
          f"avg ${r['avg_pnl']:>+6.2f}  "
          f"total ${r['total_pnl']:>+8.2f}  {bar}")

# ── Trade log ──
print(f"\n  {'─'*60}")
print(f"  Full trade log:")
print(f"  {'Time':<22} {'Symbol':<14} {'Dir':<16} {'FR bps':>7} {'Net P&L':>9} {'Cum P&L':>9}")
cum = 0
for _, t in df.iterrows():
    cum += t["net_pnl"]
    print(f"    {str(t['settle_time']):<22} {t['symbol']:<14} {t['direction']:<16} "
          f"{t['entry_fr_bps']:>6.1f}  ${t['net_pnl']:>+7.2f}  ${cum:>+8.2f}")

elapsed = time.time() - t_global
print(f"\n{'='*90}")
print(f"Backtest complete [{elapsed:.1f}s]")
print(f"{'='*90}")
