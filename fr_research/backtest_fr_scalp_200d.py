#!/usr/bin/env python3
# ruff: noqa: E501
"""
Backtest FR Scalp strategies using 200 days of 1m kline data.

Uses settlement_klines.parquet (downloaded via download_settlement_klines.py)
which has ±10min of 1m candles around each extreme FR settlement on Bybit.

Strategies tested:
  A) SHORT after neg FR settlement — ride the dump, no FR collection
  B) LONG after neg FR settlement — collect FR at NEXT settlement (autocorrelation)
  C) Comparison baseline numbers from delta-neutral audit

For SHORT: entry at close of first candle after settlement, simulate SL/TP on subsequent candles.
For LONG: same entry, but hold and check if next settlement FR is also negative.

1m candle limitation: SL/TP can only be checked at candle OHLC, not tick-level.
We use candle low for SL triggers (long) or candle high for SL triggers (short).
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
TP_FEE_BPS = 2.0      # limit maker
SL_FEE_BPS = 5.5      # market taker
MIN_FR_BPS = 20.0

t0 = time.time()

print("=" * 100)
print("BACKTEST: FR Scalp Strategies — 200 days, 1m Bybit klines (vectorized)")
print("=" * 100)
print(f"  Notional: ${NOTIONAL:,}  |  Entry: {ENTRY_FEE_BPS}bps  |  TP: {TP_FEE_BPS}bps  |  SL: {SL_FEE_BPS}bps")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("1. Loading data...", flush=True)

klines = pd.read_parquet(DATA / "historical_fr" / "settlement_klines.parquet")
print(f"   Klines: {len(klines):,} rows, {klines['symbol'].nunique()} symbols")

bb_fr = pd.read_parquet(DATA / "historical_fr" / "bybit_fr_history.parquet")
bb_fr["fr_bps"] = bb_fr["fundingRate"] * 10000
bb_fr["settle_ts_ms"] = bb_fr["fundingTime"].astype(np.int64) // 10**6
print(f"   Bybit FR: {len(bb_fr):,} records")

# Merge FR with klines
klines = klines.merge(
    bb_fr[["symbol", "settle_ts_ms", "fr_bps"]].drop_duplicates(),
    on=["symbol", "settle_ts_ms"],
    how="inner"
)
klines["offset_min"] = (klines["ts_ms"] - klines["settle_ts_ms"]) / 60000
klines = klines.sort_values(["symbol", "settle_ts_ms", "ts_ms"])

days = (bb_fr["fundingTime"].max() - bb_fr["fundingTime"].min()).total_seconds() / 86400

neg_mask = klines["fr_bps"] <= -MIN_FR_BPS
pos_mask = klines["fr_bps"] >= MIN_FR_BPS
n_neg = klines.loc[neg_mask, ["symbol", "settle_ts_ms"]].drop_duplicates().shape[0]
n_pos = klines.loc[pos_mask, ["symbol", "settle_ts_ms"]].drop_duplicates().shape[0]
print(f"   Neg FR (<= -{MIN_FR_BPS}bps): {n_neg:,} settlements")
print(f"   Pos FR (>= +{MIN_FR_BPS}bps): {n_pos:,} settlements")
print(f"   Period: {days:.0f} days")

# ═══════════════════════════════════════════════════════════════════════════════
# Pre-group klines by (symbol, settle_ts_ms) for fast lookup
# ═══════════════════════════════════════════════════════════════════════════════
print("\n   Pre-grouping klines...", flush=True)
t_grp = time.time()

# Build dict: (symbol, settle_ts_ms) -> {offset_min, open, high, low, close} as numpy arrays
groups = {}
for (sym, sts), g in klines.groupby(["symbol", "settle_ts_ms"]):
    post = g[g["offset_min"] > 0].sort_values("ts_ms")
    if len(post) < 2:
        continue
    groups[(sym, sts)] = {
        "offset": post["offset_min"].values,
        "open": post["open"].values,
        "high": post["high"].values,
        "low": post["low"].values,
        "close": post["close"].values,
        "fr_bps": float(g["fr_bps"].iloc[0]),
    }

print(f"   {len(groups):,} settlement windows with post data [{time.time()-t_grp:.1f}s]")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Raw price moves (vectorized)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n2. Raw price moves after settlement...", flush=True)

for label, fr_filter, direction in [
    ("NEG FR → SHORT after", lambda fr: fr <= -MIN_FR_BPS, "short"),
    ("POS FR → LONG after",  lambda fr: fr >= MIN_FR_BPS,  "long"),
]:
    moves = {t: [] for t in [1, 2, 3, 5, 10]}
    count = 0
    for key, g in groups.items():
        if not fr_filter(g["fr_bps"]):
            continue
        count += 1
        entry = g["close"][0]
        if entry <= 0 or np.isnan(entry):
            continue
        for h in moves:
            idx = np.argmin(np.abs(g["offset"] - h))
            if abs(g["offset"][idx] - h) > 0.6:
                continue
            ex = g["close"][idx]
            if direction == "short":
                moves[h].append((entry - ex) / entry * 10000)
            else:
                moves[h].append((ex - entry) / entry * 10000)

    print(f"\n  === {label} ({count} settlements) ===")
    print(f"  {'Horizon':>10} {'N':>6} {'Mean':>8} {'Med':>8} {'P25':>8} {'P75':>8} {'>0%':>6}")
    for h, vals in moves.items():
        if not vals:
            continue
        a = np.array(vals)
        print(f"  {h:>8}min {len(a):>6} {a.mean():>+7.1f} {np.median(a):>+7.1f} "
              f"{np.percentile(a,25):>+7.1f} {np.percentile(a,75):>+7.1f} {(a>0).mean()*100:>5.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: SHORT after negative FR — vectorized SL/TP simulation
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("3. SHORT after negative FR — SL/TP configs")
print("=" * 100, flush=True)

configs = [
    ("SL10_TP5_3m",     10,  5,  3),
    ("SL10_TP10_5m",    10, 10,  5),
    ("SL15_TP10_5m",    15, 10,  5),
    ("SL20_TP10_5m",    20, 10,  5),
    ("SL20_TP15_5m",    20, 15,  5),
    ("SL20_TP20_10m",   20, 20, 10),
    ("SL30_TP15_5m",    30, 15,  5),
    ("SL30_TP20_10m",   30, 20, 10),
    ("SL50_TP20_10m",   50, 20, 10),
    ("SL50_TP50_10m",   50, 50, 10),
    ("NO_SL_5m",      9999,  0,  5),
    ("NO_SL_10m",     9999,  0, 10),
]

def sim_short(g, sl_bps, tp_bps, max_hold_min):
    """Simulate a SHORT trade on pre-grouped numpy arrays. Returns (net_bps, exit_type, price_pnl)."""
    entry = g["close"][0]
    if entry <= 0 or np.isnan(entry):
        return None
    sl_px = entry * (1 + sl_bps / 10000) if sl_bps < 9000 else 1e18
    tp_px = entry * (1 - tp_bps / 10000) if tp_bps > 0 else 0

    offsets, highs, lows, closes = g["offset"], g["high"], g["low"], g["close"]
    for i in range(1, len(offsets)):
        if offsets[i] > max_hold_min:
            break
        if sl_bps < 9000 and highs[i] >= sl_px:
            px_pnl = (entry - sl_px) / entry * 10000
            return (px_pnl - ENTRY_FEE_BPS - SL_FEE_BPS, "SL", px_pnl)
        if tp_bps > 0 and lows[i] <= tp_px:
            px_pnl = (entry - tp_px) / entry * 10000
            return (px_pnl - ENTRY_FEE_BPS - TP_FEE_BPS, "TP", px_pnl)

    # timeout
    valid = offsets <= max_hold_min
    if valid.sum() == 0:
        return None
    last_close = closes[np.where(valid)[0][-1]]
    px_pnl = (entry - last_close) / entry * 10000
    return (px_pnl - ENTRY_FEE_BPS - SL_FEE_BPS, "timeout", px_pnl)

results = {}
t3 = time.time()

neg_groups = {k: v for k, v in groups.items() if v["fr_bps"] <= -MIN_FR_BPS}
print(f"  Processing {len(neg_groups):,} neg-FR settlements x {len(configs)} configs...")

for name, sl, tp, max_hold in configs:
    trades = []
    for (sym, sts), g in neg_groups.items():
        r = sim_short(g, sl, tp, max_hold)
        if r is None:
            continue
        net_bps, exit_type, px_pnl = r
        trades.append({
            "symbol": sym, "settle_ts_ms": sts, "fr_bps": abs(g["fr_bps"]),
            "price_pnl_bps": px_pnl, "fee_bps": ENTRY_FEE_BPS + (TP_FEE_BPS if exit_type == "TP" else SL_FEE_BPS),
            "net_bps": net_bps, "net_usd": net_bps / 10000 * NOTIONAL,
            "exit_type": exit_type,
        })
    results[name] = trades

print(f"  Done [{time.time()-t3:.1f}s]")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: LONG after neg FR — collect FR at NEXT settlement
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("4. LONG after negative FR — hold to NEXT settlement, collect FR")
print("=" * 100, flush=True)

# Build next-settlement FR lookup
bb_fr_sorted = bb_fr.sort_values(["symbol", "settle_ts_ms"])
bb_fr_sorted["next_fr_bps"] = bb_fr_sorted.groupby("symbol")["fr_bps"].shift(-1)
next_fr_ser = bb_fr_sorted.dropna(subset=["next_fr_bps"]).set_index(["symbol", "settle_ts_ms"])["next_fr_bps"]

long_trades = []
for (sym, sts), g in neg_groups.items():
    if (sym, sts) not in next_fr_ser.index:
        continue
    next_fr = next_fr_ser.loc[(sym, sts)]
    entry = g["close"][0]
    if entry <= 0 or np.isnan(entry):
        continue
    exit_close = g["close"][-1]
    price_pnl = (exit_close - entry) / entry * 10000
    fr_income = -next_fr  # neg FR → longs receive
    total_fee = ENTRY_FEE_BPS + SL_FEE_BPS
    net = price_pnl + fr_income - total_fee
    long_trades.append({
        "symbol": sym, "settle_ts_ms": sts, "fr_bps": abs(g["fr_bps"]),
        "next_fr_bps": next_fr, "fr_income_bps": fr_income,
        "price_pnl_bps": price_pnl, "fee_bps": total_fee,
        "net_bps": net, "net_usd": net / 10000 * NOTIONAL,
        "exit_type": "next_settle",
    })

results["LONG_hold_next_FR"] = long_trades
print(f"  {len(long_trades):,} trades")

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("RESULTS — ALL CONFIGS (sorted by daily P&L)")
print("=" * 100)
print(f"Period: {days:.0f} days | Neg FR settlements: {len(neg_groups):,}")
print()

print(f"{'Config':>24} {'N':>6} {'WR':>5} {'Avg':>8} {'Daily':>8} {'ROI/yr':>8}  "
      f"{'TP%':>5} {'SL%':>5} {'TO%':>5} {'AvgPx':>8}")
print("─" * 115)

sorted_results = []
for name, trades in results.items():
    if not trades:
        sorted_results.append((name, 0, []))
        continue
    df = pd.DataFrame(trades)
    daily = df["net_usd"].sum() / days
    sorted_results.append((name, daily, trades))

sorted_results.sort(key=lambda x: -x[1])

for name, daily, trades in sorted_results:
    if not trades:
        print(f"{name:>24}      0")
        continue
    df = pd.DataFrame(trades)
    n = len(df)
    wr = (df["net_usd"] > 0).mean() * 100
    avg = df["net_bps"].mean()
    roi = daily / NOTIONAL * 365 * 100
    tp_pct = (df["exit_type"] == "TP").mean() * 100
    sl_pct = (df["exit_type"] == "SL").mean() * 100
    to_pct = (df["exit_type"].isin(["timeout", "next_settle"])).mean() * 100
    avg_px = df["price_pnl_bps"].mean()
    marker = " <<<" if daily > 0 else ""
    print(f"{name:>24} {n:>6} {wr:>4.0f}% {avg:>+7.1f} ${daily:>+7,.0f} {roi:>7.0f}%  "
          f"{tp_pct:>4.0f}% {sl_pct:>4.0f}% {to_pct:>4.0f}% {avg_px:>+7.1f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
# DETAILED on top configs
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("DETAILED: Top Configs")
print("=" * 100)

top_names = [x[0] for x in sorted_results[:5] if x[2]]

for name in top_names:
    trades = results[name]
    if not trades:
        continue
    df = pd.DataFrame(trades)
    daily = df["net_usd"].sum() / days

    print(f"\n  --- {name} ---")
    print(f"  Trades: {len(df):,} | WR: {(df['net_usd']>0).mean()*100:.0f}% | "
          f"Daily: ${daily:+,.0f} | ROI: {daily/NOTIONAL*365*100:+,.0f}%")

    print(f"  P&L distribution (bps): "
          f"5th={df['net_bps'].quantile(0.05):+.0f}, "
          f"25th={df['net_bps'].quantile(0.25):+.0f}, "
          f"median={df['net_bps'].median():+.0f}, "
          f"75th={df['net_bps'].quantile(0.75):+.0f}, "
          f"95th={df['net_bps'].quantile(0.95):+.0f}")

    # By FR magnitude
    print(f"  By FR magnitude:")
    for lo, hi in [(20,30), (30,50), (50,100), (100,500)]:
        bucket = df[(df["fr_bps"] >= lo) & (df["fr_bps"] < hi)]
        if len(bucket) == 0:
            continue
        wr = (bucket["net_usd"] > 0).mean() * 100
        print(f"    FR {lo:>3}-{hi:>3}: {len(bucket):>5} trades, {wr:.0f}% WR, "
              f"avg net {bucket['net_bps'].mean():+.1f}, price {bucket['price_pnl_bps'].mean():+.1f}")

    # Monthly
    df["month"] = pd.to_datetime(df["settle_ts_ms"], unit="ms").dt.to_period("M")
    monthly = df.groupby("month").agg(
        n=("net_usd", "count"),
        total=("net_usd", "sum"),
        wr=("net_usd", lambda x: (x > 0).mean() * 100)
    )
    print(f"  Monthly:")
    for m, r in monthly.iterrows():
        mdays = 30  # approximate
        print(f"    {str(m):>10}: {int(r['n']):>4} trades, ${r['total']:>+8,.0f} "
              f"(${r['total']/mdays:>+,.0f}/day, {r['wr']:.0f}% WR)")

# For LONG_hold_next_FR, show additional FR stats
if "LONG_hold_next_FR" in results and results["LONG_hold_next_FR"]:
    df = pd.DataFrame(results["LONG_hold_next_FR"])
    print(f"\n  --- LONG_hold_next_FR: FR autocorrelation analysis ---")
    print(f"  Next FR also negative: {(df['next_fr_bps'] < 0).mean()*100:.0f}%")
    print(f"  Next FR also <= -20: {(df['next_fr_bps'] <= -20).mean()*100:.0f}%")
    print(f"  Avg FR income: {df['fr_income_bps'].mean():+.1f} bps")
    print(f"  Avg price move (10min): {df['price_pnl_bps'].mean():+.1f} bps")
    print(f"  NOTE: Price P&L is only the 10min window, not full hold to next settlement.")
    print(f"        Actual directional exposure would be much larger over 1-8 hours.")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("STRATEGY COMPARISON (all on {:.0f} days)".format(days))
print("=" * 100)

best = sorted_results[0] if sorted_results[0][1] > 0 else None
print(f"\n  {'Strategy':>35} {'Daily':>8} {'Capital':>8} {'ROI/yr':>8}")
print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8}")

if best:
    d = best[1]
    print(f"  {'Best FR scalp':>35} ${d:>+7,.0f} ${'10k':>7} {d/10000*365*100:>7.0f}%  ({best[0]})")

print(f"  {'Delta-neutral Bybit 1h (audit)':>35} $   +273 $    20k     498%")
print(f"  {'Delta-neutral Binance 1h (audit)':>35} $   +218 $    20k     398%")
print(f"  {'4-pool combined (audit)':>35} $   +879 $    80k     401%")

print(f"\n[{time.time()-t0:.0f}s total]")
