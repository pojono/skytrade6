#!/usr/bin/env python3
"""
Backtest: Post-Settlement Directional Trades

Hypothesis: After extreme negative FR settlements, price tends to dump (the crowd
is bearish, shorts are paying longs). Instead of collecting FR, we ride the move:

Strategy A: SHORT after negative FR settlement (ride the dump)
Strategy B: LONG after positive FR settlement (ride the pump) [rare]
Strategy C: LONG before settlement + collect FR (from previous backtest, for comparison)

No FR collection — pure directional scalp timed to settlement events.

Fee model (Bybit VIP0):
  - Entry market: 5.5 bps taker
  - Exit TP limit: 2.0 bps maker  
  - Exit SL market: 5.5 bps taker
"""
import sys
import time
import gc
from pathlib import Path

import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data_all"

NOTIONAL = 10_000
ENTRY_FEE_BPS = 5.5
TP_FEE_BPS = 2.0
SL_FEE_BPS = 5.5

MIN_FR_BPS = 20.0

print("=" * 100)
print("BACKTEST: Post-Settlement Directional Scalp")
print("=" * 100)
print(f"  Notional: ${NOTIONAL:,}  |  Entry: {ENTRY_FEE_BPS}bps market  |  TP: {TP_FEE_BPS}bps limit  |  SL: {SL_FEE_BPS}bps market")
print(f"  Min |FR|: {MIN_FR_BPS} bps")
print("=" * 100)
print()

t_global = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Settlement schedule
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 1: Building settlement schedule...", flush=True)
t1 = time.time()

bn_fr = pd.read_parquet(DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "nextFundingTime"])
print(f"  Loaded Binance FR: {len(bn_fr):,} rows [{time.time()-t1:.1f}s]", flush=True)

bn_fr["ts_1m"] = bn_fr["ts"].dt.floor("1min")
bn_fr_1m = bn_fr.groupby(["ts_1m", "symbol"]).agg(
    fr=("lastFundingRate", "last"),
    nft=("nextFundingTime", "last"),
).reset_index()
del bn_fr; gc.collect()

bn_fr_1m = bn_fr_1m.sort_values(["symbol", "ts_1m"])
bn_fr_1m["nft_prev"] = bn_fr_1m.groupby("symbol")["nft"].shift(1)
bn_fr_1m["is_settle"] = (bn_fr_1m["nft"] != bn_fr_1m["nft_prev"]) & bn_fr_1m["nft_prev"].notna()
bn_fr_1m["fr_prev"] = bn_fr_1m.groupby("symbol")["fr"].shift(1)

settle_rows = bn_fr_1m[bn_fr_1m["is_settle"]].copy()
settle_rows = settle_rows.rename(columns={"ts_1m": "settle_time"})
settle_rows["fr_paid_bps"] = settle_rows["fr_prev"] * 10000

settle_df = settle_rows[["settle_time", "symbol", "fr_paid_bps"]].dropna().copy()
settle_df = settle_df.sort_values("settle_time").reset_index(drop=True)

neg_fr = settle_df[settle_df["fr_paid_bps"] <= -MIN_FR_BPS].copy()
pos_fr = settle_df[settle_df["fr_paid_bps"] >= MIN_FR_BPS].copy()

print(f"  Total settlements: {len(settle_df):,}")
print(f"  Negative FR <= -{MIN_FR_BPS}bps: {len(neg_fr):,} ({neg_fr['symbol'].nunique()} coins)")
print(f"  Positive FR >= +{MIN_FR_BPS}bps: {len(pos_fr):,} ({pos_fr['symbol'].nunique()} coins)")
print(f"  Date range: {settle_df['settle_time'].min()} to {settle_df['settle_time'].max()}")
print(f"  [{time.time()-t1:.1f}s]", flush=True)

del bn_fr_1m, settle_rows; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Load Bybit tick data
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPHASE 2: Loading Bybit tick data...", flush=True)
t2 = time.time()

_bb = pd.read_parquet(DATA / "bybit" / "ticker.parquet",
    columns=["ts", "symbol", "lastPrice", "bid1Price", "ask1Price"])
bb_ts = _bb["ts"].values
bb_sym = _bb["symbol"].values.astype(str)
bb_last = _bb["lastPrice"].values
bb_bid = _bb["bid1Price"].values
bb_ask = _bb["ask1Price"].values
del _bb; gc.collect()
print(f"  {len(bb_ts):,} rows [{time.time()-t2:.1f}s]", flush=True)


def get_ticks(symbol, ref_time_ns, before_s, after_s):
    t_lo = ref_time_ns - np.timedelta64(before_s, "s")
    t_hi = ref_time_ns + np.timedelta64(after_s, "s")
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
        "offset_s": (ts - ref_time_ns) / np.timedelta64(1, "s"),
        "last": bb_last[sl][mask],
        "bid": bb_bid[sl][mask],
        "ask": bb_ask[sl][mask],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Measure raw price moves after settlements (no SL/TP, just observe)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPHASE 3: Measuring raw price moves after settlement...", flush=True)
t3 = time.time()

for label, subset, direction in [("NEGATIVE FR (short after)", neg_fr, "short"),
                                  ("POSITIVE FR (long after)", pos_fr, "long")]:
    print(f"\n  === {label}: {len(subset)} settlements ===")
    if len(subset) == 0:
        continue
    
    moves = {t: [] for t in [5, 10, 15, 30, 60, 120, 300]}
    n_data = 0
    
    for _, row in subset.iterrows():
        sym = row["symbol"]
        st = row["settle_time"]
        st_ns = np.datetime64(st, "ns")
        
        data = get_ticks(sym, st_ns, before_s=10, after_s=310)
        if data is None:
            continue
        
        # Entry: first tick AFTER settlement (offset > 0)
        post = data["offset_s"] > 0
        if post.sum() < 3:
            continue
        
        # Entry at ask (for long) or bid (for short) right after settlement
        first_post = np.where(post)[0][0]
        if direction == "short":
            entry_price = float(data["bid"][first_post])
        else:
            entry_price = float(data["ask"][first_post])
        
        if entry_price <= 0 or np.isnan(entry_price):
            continue
        n_data += 1
        
        # Measure price at various horizons
        for horizon_s in moves.keys():
            horizon_mask = (data["offset_s"] >= horizon_s - 3) & (data["offset_s"] <= horizon_s + 3)
            if horizon_mask.sum() == 0:
                continue
            idx = np.where(horizon_mask)[0][-1]
            mid = (float(data["bid"][idx]) + float(data["ask"][idx])) / 2
            if mid <= 0 or np.isnan(mid):
                continue
            
            if direction == "short":
                move_bps = (entry_price - mid) / entry_price * 10000
            else:
                move_bps = (mid - entry_price) / entry_price * 10000
            moves[horizon_s].append(move_bps)
    
    print(f"  Trades with data: {n_data}")
    print(f"  {'Horizon':>10} {'N':>5} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'>0%':>6}")
    for h, vals in moves.items():
        if not vals:
            continue
        arr = np.array(vals)
        print(f"  {h:>8}s {len(arr):>5} {arr.mean():>+7.1f} {np.median(arr):>+7.1f} "
              f"{np.percentile(arr,25):>+7.1f} {np.percentile(arr,75):>+7.1f} {(arr>0).mean()*100:>5.0f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Backtest with SL/TP — SHORT after negative FR
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("PHASE 4: SHORT after negative FR settlement — with SL/TP")
print("=" * 100, flush=True)
t4 = time.time()

configs = [
    # (name, sl_bps, tp_bps, max_hold_s, entry_delay_s)
    ("SL10_TP5_30s",    10, 5,   30,   1),
    ("SL10_TP10_30s",   10, 10,  30,   1),
    ("SL10_TP10_60s",   10, 10,  60,   1),
    ("SL15_TP10_60s",   15, 10,  60,   1),
    ("SL20_TP10_60s",   20, 10,  60,   1),
    ("SL20_TP15_120s",  20, 15, 120,   1),
    ("SL20_TP20_120s",  20, 20, 120,   1),
    ("SL30_TP15_120s",  30, 15, 120,   1),
    ("SL30_TP20_300s",  30, 20, 300,   1),
    ("SL50_TP30_300s",  50, 30, 300,   1),
    ("SL50_TP50_300s",  50, 50, 300,   1),
    ("SL100_TP50_300s",100, 50, 300,   1),
    ("NO_SL_300s",    9999,  0, 300,   1),
    # Also test entering a few seconds later
    ("SL20_TP10_60s_d5",  20, 10,  60,   5),
    ("SL20_TP10_60s_d10", 20, 10,  60,  10),
]

# Also test LONG before settlement + collect FR (combined strategy)
long_configs = [
    # Collect FR AND ride the move: enter long before, hold through settlement, exit after
    ("LONG_FR_SL10_TP10_60s", 10, 10, 60),
]

results = {}

for name, sl, tp, max_hold, delay in configs:
    trades = []
    
    for _, row in neg_fr.iterrows():
        sym = row["symbol"]
        st = row["settle_time"]
        fr_bps = abs(row["fr_paid_bps"])
        st_ns = np.datetime64(st, "ns")
        
        data = get_ticks(sym, st_ns, before_s=10, after_s=max_hold + 30)
        if data is None:
            continue
        
        # Entry: short at bid, delay seconds after settlement
        entry_mask = data["offset_s"] >= delay
        if entry_mask.sum() < 3:
            continue
        
        entry_idx = np.where(entry_mask)[0][0]
        entry_price = float(data["bid"][entry_idx])  # short at bid
        if entry_price <= 0 or np.isnan(entry_price):
            continue
        
        sl_price = entry_price * (1 + sl / 10000) if sl < 9000 else float("inf")
        tp_price = entry_price * (1 - tp / 10000) if tp > 0 else 0
        
        exit_price = None
        exit_type = "timeout"
        exit_fee = SL_FEE_BPS
        
        # Scan post-entry ticks
        post_entry = np.where(data["offset_s"] > data["offset_s"][entry_idx])[0]
        deadline = data["offset_s"][entry_idx] + max_hold
        
        for j in post_entry:
            if data["offset_s"][j] > deadline:
                break
            tick_ask = float(data["ask"][j])  # to cover short, buy at ask
            tick_bid = float(data["bid"][j])
            
            if tick_ask <= 0 or np.isnan(tick_ask):
                continue
            
            # SL: price goes UP above SL → buy to cover at ask
            if sl < 9000 and tick_ask >= sl_price:
                exit_price = tick_ask
                exit_type = "SL"
                exit_fee = SL_FEE_BPS
                break
            
            # TP: price drops below TP → limit buy at TP price
            if tp > 0 and tick_ask <= tp_price:
                exit_price = tp_price
                exit_type = "TP"
                exit_fee = TP_FEE_BPS
                break
        
        # Timeout: cover at ask
        if exit_price is None:
            valid = post_entry[post_entry < len(data["ask"])]
            if len(valid) == 0:
                continue
            last_valid = valid[-1]
            exit_price = float(data["ask"][last_valid])
            if exit_price <= 0 or np.isnan(exit_price):
                continue
        
        # PnL for short: (entry - exit) / entry
        price_pnl_bps = (entry_price - exit_price) / entry_price * 10000
        total_fee = ENTRY_FEE_BPS + exit_fee
        net_bps = price_pnl_bps - total_fee
        
        trades.append({
            "symbol": sym, "settle_time": st, "fr_bps": fr_bps,
            "price_pnl_bps": price_pnl_bps, "fee_bps": total_fee,
            "net_bps": net_bps, "net_usd": net_bps / 10000 * NOTIONAL,
            "exit_type": exit_type,
        })
    
    results[name] = trades

# LONG before settlement — collect FR + ride
for name, sl, tp, max_hold in long_configs:
    trades = []
    
    for _, row in neg_fr.iterrows():
        sym = row["symbol"]
        st = row["settle_time"]
        fr_bps = abs(row["fr_paid_bps"])
        st_ns = np.datetime64(st, "ns")
        
        data = get_ticks(sym, st_ns, before_s=30, after_s=max_hold + 30)
        if data is None:
            continue
        
        # Entry: long at ask, 5s before settlement
        pre_mask = data["offset_s"] <= -3
        if pre_mask.sum() == 0:
            continue
        entry_idx = np.where(pre_mask)[0][-1]
        entry_price = float(data["ask"][entry_idx])  # buy at ask
        if entry_price <= 0 or np.isnan(entry_price):
            continue
        
        sl_price = entry_price * (1 - sl / 10000)
        tp_price = entry_price * (1 + tp / 10000) if tp > 0 else float("inf")
        
        exit_price = None
        exit_type = "timeout"
        exit_fee = SL_FEE_BPS
        
        # Scan POST-settlement ticks
        post_settle = np.where(data["offset_s"] > 0)[0]
        
        for j in post_settle:
            if data["offset_s"][j] > max_hold:
                break
            tick_bid = float(data["bid"][j])
            if tick_bid <= 0 or np.isnan(tick_bid):
                continue
            
            if tick_bid <= sl_price:
                exit_price = tick_bid
                exit_type = "SL"
                exit_fee = SL_FEE_BPS
                break
            if tp > 0 and tick_bid >= tp_price:
                exit_price = tp_price
                exit_type = "TP"
                exit_fee = TP_FEE_BPS
                break
        
        if exit_price is None:
            valid_bids = data["bid"][post_settle]
            valid_bids = valid_bids[(valid_bids > 0) & ~np.isnan(valid_bids)]
            if len(valid_bids) == 0:
                continue
            exit_price = float(valid_bids[-1])
        
        price_pnl_bps = (exit_price - entry_price) / entry_price * 10000
        total_fee = ENTRY_FEE_BPS + exit_fee
        net_bps = fr_bps + price_pnl_bps - total_fee  # FR collected!
        
        trades.append({
            "symbol": sym, "settle_time": st, "fr_bps": fr_bps,
            "price_pnl_bps": price_pnl_bps, "fee_bps": total_fee,
            "net_bps": net_bps, "net_usd": net_bps / 10000 * NOTIONAL,
            "exit_type": exit_type,
        })
    
    results[name] = trades

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Results
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("RESULTS: All Configs")
print("=" * 100)

t_min = neg_fr["settle_time"].min()
t_max = neg_fr["settle_time"].max()
days = max((t_max - t_min).total_seconds() / 86400, 1)

print(f"Period: {days:.0f} days | Neg FR settlements: {len(neg_fr)}")
print()

print(f"{'Config':>24} {'N':>5} {'WR':>5} {'Avg':>8} {'Daily':>8} {'ROI/yr':>8}  "
      f"{'TP%':>5} {'SL%':>5} {'TO%':>5} {'AvgPx':>8}")
print("─" * 110)

# Sort by daily P&L
sorted_configs = []
for name in results:
    trades = results[name]
    if not trades:
        sorted_configs.append((name, 0, []))
        continue
    df = pd.DataFrame(trades)
    daily = df["net_usd"].sum() / days
    sorted_configs.append((name, daily, trades))

sorted_configs.sort(key=lambda x: -x[1])

for name, daily, trades in sorted_configs:
    if not trades:
        print(f"{name:>24}     0")
        continue
    df = pd.DataFrame(trades)
    n = len(df)
    wr = (df["net_usd"] > 0).mean() * 100
    avg = df["net_bps"].mean()
    roi = daily / NOTIONAL * 365 * 100
    tp_pct = (df["exit_type"] == "TP").mean() * 100
    sl_pct = (df["exit_type"] == "SL").mean() * 100
    to_pct = (df["exit_type"] == "timeout").mean() * 100
    avg_px = df["price_pnl_bps"].mean()

    marker = " <<<" if daily > 0 else ""
    print(f"{name:>24} {n:>5} {wr:>4.0f}% {avg:>+7.1f} ${daily:>+7,.0f} {roi:>7.0f}%  "
          f"{tp_pct:>4.0f}% {sl_pct:>4.0f}% {to_pct:>4.0f}% {avg_px:>+7.1f}{marker}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Detailed breakdown of top configs  
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("DETAILED: Top 3 Configs")
print("=" * 100)

top3 = [x[0] for x in sorted_configs[:3] if x[2]]

for name in top3:
    trades = results[name]
    if not trades:
        continue
    df = pd.DataFrame(trades)
    df["settle_time"] = pd.to_datetime(df["settle_time"])

    daily = df["net_usd"].sum() / days
    print(f"\n  --- {name} ---")
    print(f"  Trades: {len(df)} | WR: {(df['net_usd']>0).mean()*100:.0f}% | "
          f"Daily: ${daily:+,.0f} | ROI: {daily/NOTIONAL*365*100:+,.0f}%")
    
    print(f"\n  P&L distribution (bps):")
    for pct in [5, 25, 50, 75, 95]:
        print(f"    {pct}th: {df['net_bps'].quantile(pct/100):+.1f}")
    
    print(f"\n  By FR magnitude:")
    for lo, hi in [(20,30), (30,50), (50,100), (100,500)]:
        bucket = df[(df["fr_bps"] >= lo) & (df["fr_bps"] < hi)]
        if len(bucket) == 0:
            continue
        wr = (bucket["net_usd"] > 0).mean() * 100
        print(f"    FR {lo:>3}-{hi:>3}: {len(bucket):>4} trades, {wr:.0f}% WR, "
              f"avg {bucket['net_bps'].mean():+.1f} bps, price {bucket['price_pnl_bps'].mean():+.1f}")
    
    # Exit type breakdown
    for et in ["TP", "SL", "timeout"]:
        sub = df[df["exit_type"] == et]
        if len(sub) == 0:
            continue
        print(f"\n  {et} exits ({len(sub)}):")
        print(f"    Avg net: {sub['net_bps'].mean():+.1f} bps, Avg price move: {sub['price_pnl_bps'].mean():+.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print("STRATEGY COMPARISON")
print("=" * 100)
print(f"\n  {'Strategy':>30} {'Daily':>8} {'Capital':>10} {'ROI/yr':>8} {'Notes'}")
print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*8} {'─'*30}")

best_short = sorted_configs[0] if sorted_configs[0][1] > 0 else None
if best_short and best_short[2]:
    d = best_short[1]
    print(f"  {'SHORT after neg FR':>30} ${d:>+7,.0f} ${'10k':>9} {d/10000*365*100:>7.0f}%  {best_short[0]}")

if "LONG_FR_SL10_TP10_60s" in results and results["LONG_FR_SL10_TP10_60s"]:
    d2 = sum(t["net_usd"] for t in results["LONG_FR_SL10_TP10_60s"]) / days
    print(f"  {'LONG before + collect FR':>30} ${d2:>+7,.0f} ${'10k':>9} {d2/10000*365*100:>7.0f}%  LONG_FR_SL10_TP10_60s")

print(f"  {'Delta-neutral (Binance 1h)':>30} $   +218 $      20k     398%  from audit (106 days)")
print(f"  {'Delta-neutral (Bybit 1h)':>30} $   +273 $      20k     498%  from audit (106 days)")

print(f"\n  ⚠ CAVEAT: Post-settlement results based on only {days:.0f} days of tick data!")
print(f"\n[{time.time()-t_global:.0f}s total]")
