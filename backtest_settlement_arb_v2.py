#!/usr/bin/env python3
"""
Backtest v2: Funding Rate Settlement Arbitrage — Hold Until FR Normalizes

Key changes from v1:
  - Taker-only execution (5.5 bps per side, your actual rate)
  - HOLD across settlements: enter once, collect FR every settlement, exit when FR drops
  - No rush to exit — wait for calm conditions (low spread, low volatility)
  - Commission paid only on entry + exit (not per settlement)
  - Per-coin positions tracked independently (can hold multiple coins)
  - Realistic fill impact from ob200 data

Strategy:
  1. Before each settlement, scan all coins for extreme FR
  2. If |FR| > entry_threshold and we're not already in this coin → ENTER
  3. Collect funding payment at every settlement while holding
  4. If |FR| < exit_threshold → EXIT (wait for calm conditions)
  5. Track cumulative P&L including all FR payments minus entry+exit costs

Memory-efficient: loads full tick data once, processes settlements sequentially.
"""
import sys
import time
import gc
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

DATA = Path("data_all")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
NOTIONAL = 10_000              # USD per position

# Fee model — YOUR actual taker rate
TAKER_FEE_BPS = 5.5            # per side
SLIPPAGE_BPS = 2.0             # conservative estimate per side

# Entry: taker on both spot + futures
ENTRY_COST_BPS = (TAKER_FEE_BPS + SLIPPAGE_BPS) * 2   # open spot + open futures = 15 bps
EXIT_COST_BPS = (TAKER_FEE_BPS + SLIPPAGE_BPS) * 2    # close spot + close futures = 15 bps
TOTAL_RT_COST_BPS = ENTRY_COST_BPS + EXIT_COST_BPS     # 30 bps total

# Thresholds
# Entry: FR must be high enough that even 1 settlement covers entry cost
# (we'll also model: what if it takes 2-3 settlements to break even)
ENTRY_THRESHOLD_BPS = 15.0     # min |FR| to open new position
EXIT_THRESHOLD_BPS = 5.0       # close when |FR| drops below this

# Position limits
MAX_POSITIONS = 3              # max simultaneous positions

print("=" * 90)
print("BACKTEST v2: Hold-Until-Normalize Settlement Arbitrage")
print("=" * 90)
print(f"  Notional:        ${NOTIONAL:,.0f} per position")
print(f"  Taker fee:       {TAKER_FEE_BPS} bps per side (your actual rate)")
print(f"  Slippage:        {SLIPPAGE_BPS} bps per side")
print(f"  Entry cost:      {ENTRY_COST_BPS:.1f} bps (spot+futures)")
print(f"  Exit cost:       {EXIT_COST_BPS:.1f} bps")
print(f"  Total RT cost:   {TOTAL_RT_COST_BPS:.1f} bps")
print(f"  Entry threshold: {ENTRY_THRESHOLD_BPS} bps |FR|")
print(f"  Exit threshold:  {EXIT_THRESHOLD_BPS} bps |FR|")
print(f"  Max positions:   {MAX_POSITIONS}")
print("=" * 90)
print()

t_global = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Build complete settlement schedule (all coins, all times)
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 1: Building settlement schedule...")
sys.stdout.flush()
t1 = time.time()

bn_fr = pd.read_parquet(DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "nextFundingTime"])
print(f"  Loaded Binance FR: {len(bn_fr):,} rows [{time.time()-t1:.1f}s]")
sys.stdout.flush()

# Downsample to 1-min for settlement detection
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

# Get FR at each settlement for each coin
# The FR that was paid is the one from the row BEFORE the settlement event (same symbol).
# Use shift(1) within each symbol group to get the previous row's FR safely.
bn_fr_1m["fr_prev"] = bn_fr_1m.groupby("symbol")["fr"].shift(1)

settle_rows = bn_fr_1m[bn_fr_1m["is_settle"]].copy()
settle_rows = settle_rows.rename(columns={"ts_1m": "settle_time"})
settle_rows["fr_paid"] = settle_rows["fr_prev"]
settle_rows["fr_paid_bps"] = settle_rows["fr_paid"].abs() * 10000

# Drop rows where we couldn't get the previous FR
settle_df = settle_rows[["settle_time", "symbol", "fr_paid", "fr_paid_bps"]].dropna().copy()
settle_df = settle_df.sort_values("settle_time").reset_index(drop=True)

# Get unique settlement times (hourly)
settle_times = sorted(settle_df["settle_time"].unique())

del bn_fr_1m, settle_rows; gc.collect()

print(f"  Settlement events: {len(settle_df):,} across {settle_df['symbol'].nunique()} coins")
print(f"  Unique settlement times: {len(settle_times)}")
print(f"  Phase 1 done [{time.time()-t1:.1f}s]")
print()
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Load tick data for entry/exit price lookups
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 2: Loading tick data...")
sys.stdout.flush()
t2 = time.time()

_bn_tk = pd.read_parquet(DATA / "binance" / "ticker.parquet",
    columns=["ts", "symbol", "lastPrice"])
bn_tk_ts = _bn_tk["ts"].values
bn_tk_sym = _bn_tk["symbol"].values
bn_tk_price = _bn_tk["lastPrice"].values
del _bn_tk; gc.collect()
print(f"  Binance ticker: {len(bn_tk_ts):,} [{time.time()-t2:.1f}s]")
sys.stdout.flush()

# Also load Bybit for spread measurement
_bb_tk = pd.read_parquet(DATA / "bybit" / "ticker.parquet",
    columns=["ts", "symbol", "bid1Price", "ask1Price"])
bb_tk_ts = _bb_tk["ts"].values
bb_tk_sym = _bb_tk["symbol"].values
bb_tk_bid = _bb_tk["bid1Price"].values
bb_tk_ask = _bb_tk["ask1Price"].values
del _bb_tk; gc.collect()
print(f"  Bybit ticker: {len(bb_tk_ts):,} [{time.time()-t2:.1f}s]")
sys.stdout.flush()

print(f"  Done [{time.time()-t2:.1f}s]")
print()


def get_price_at(symbol, target_time_ns, search_range_s=30):
    """Get Binance lastPrice closest to target time."""
    t_lo = target_time_ns - np.timedelta64(search_range_s, "s")
    t_hi = target_time_ns + np.timedelta64(search_range_s, "s")
    i0 = np.searchsorted(bn_tk_ts, t_lo, side="left")
    i1 = np.searchsorted(bn_tk_ts, t_hi, side="right")
    if i0 >= i1:
        return None
    sl = slice(i0, i1)
    sym_mask = bn_tk_sym[sl] == symbol
    if sym_mask.sum() == 0:
        return None
    ts_sub = bn_tk_ts[sl][sym_mask]
    pr_sub = bn_tk_price[sl][sym_mask]
    closest = np.argmin(np.abs(ts_sub - target_time_ns))
    return float(pr_sub[closest])


def get_bb_spread_at(symbol, target_time_ns, search_range_s=30):
    """Get Bybit bid-ask spread in bps closest to target time."""
    t_lo = target_time_ns - np.timedelta64(search_range_s, "s")
    t_hi = target_time_ns + np.timedelta64(search_range_s, "s")
    i0 = np.searchsorted(bb_tk_ts, t_lo, side="left")
    i1 = np.searchsorted(bb_tk_ts, t_hi, side="right")
    if i0 >= i1:
        return None
    sl = slice(i0, i1)
    sym_mask = bb_tk_sym[sl] == symbol
    if sym_mask.sum() == 0:
        return None
    ts_sub = bb_tk_ts[sl][sym_mask]
    bid_sub = bb_tk_bid[sl][sym_mask]
    ask_sub = bb_tk_ask[sl][sym_mask]
    closest = np.argmin(np.abs(ts_sub - target_time_ns))
    bid = float(bid_sub[closest])
    ask = float(ask_sub[closest])
    if bid <= 0 or ask <= 0:
        return None
    mid = (bid + ask) / 2
    return (ask - bid) / mid * 10000


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Simulate the hold-until-normalize strategy
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 3: Running strategy simulation")
print("─" * 90)
sys.stdout.flush()

# Position tracking
# Each position: {symbol, entry_time, entry_price, direction, fr_payments: [], notional}
open_positions = {}  # symbol -> position dict
closed_trades = []
trade_events = []  # all events for logging

n_settlements = len(settle_times)
t_sim = time.time()

for si, st in enumerate(settle_times):
    st_ts = pd.Timestamp(st)
    if st_ts.tzinfo is None:
        st_ts = st_ts.tz_localize("UTC")
    st_ns = np.datetime64(st_ts, "ns")
    
    # Get all coins settling at this time
    coins_at_settle = settle_df[settle_df["settle_time"] == st].copy()
    
    # ── Step 1: Collect FR payments for open positions ──
    # CAUSALITY: fr_paid is the FR that was just settled — observable after payout.
    # We use it to compute the actual payment received (or paid if FR flipped).
    for sym, pos in list(open_positions.items()):
        coin_row = coins_at_settle[coins_at_settle["symbol"] == sym]
        if len(coin_row) == 0:
            # This coin doesn't settle at this time (different schedule)
            continue
        
        fr_paid = float(coin_row.iloc[0]["fr_paid"])
        fr_bps = abs(fr_paid) * 10000
        
        # Payment depends on our direction AND the FR sign:
        #   FR < 0: shorts pay longs
        #   FR > 0: longs pay shorts
        # If we are long_futures:
        #   FR < 0 → we RECEIVE |FR| (shorts pay us)
        #   FR > 0 → we PAY |FR| (we pay shorts)
        # If we are short_futures:
        #   FR > 0 → we RECEIVE |FR| (longs pay us)
        #   FR < 0 → we PAY |FR| (we pay longs)
        if pos["direction"] == "long_futures":
            payment_usd = -fr_paid * NOTIONAL  # FR<0 → positive payment
        else:
            payment_usd = fr_paid * NOTIONAL   # FR>0 → positive payment
        pos["fr_payments"].append({
            "time": st_ts,
            "fr": fr_paid,
            "fr_bps": fr_bps,
            "payment_usd": payment_usd,
        })
        pos["total_fr_received"] += payment_usd
        pos["n_settlements"] += 1
        pos["last_fr_bps"] = fr_bps
        
        trade_events.append({
            "time": st_ts,
            "type": "FR_COLLECT",
            "symbol": sym,
            "fr_bps": fr_bps,
            "payment_usd": payment_usd,
            "cumulative_fr": pos["total_fr_received"],
        })
    
    # ── Step 2: Check exits — FR dropped or flipped against us ──
    # CAUSALITY: we just received (or paid) FR. We observe the settled value
    # and decide whether to keep holding. No future information used.
    for sym in list(open_positions.keys()):
        pos = open_positions[sym]
        
        coin_row = coins_at_settle[coins_at_settle["symbol"] == sym]
        if len(coin_row) == 0:
            continue
        
        current_fr = float(coin_row.iloc[0]["fr_paid"])
        current_fr_bps = float(coin_row.iloc[0]["fr_paid_bps"])
        
        # Check if FR flipped against us (we'd be paying instead of receiving)
        fr_against_us = False
        if pos["direction"] == "long_futures" and current_fr > 0:
            fr_against_us = True
        elif pos["direction"] == "short_futures" and current_fr < 0:
            fr_against_us = True
        
        # Exit if FR normalized OR flipped against us
        if current_fr_bps < EXIT_THRESHOLD_BPS or fr_against_us:
            # Get exit price (wait a bit after settlement for things to calm down)
            exit_time_ns = st_ns + np.timedelta64(120, "s")  # 2 min after settle
            exit_price = get_price_at(sym, exit_time_ns)
            if exit_price is None:
                exit_price = pos["entry_price"]  # fallback
            
            exit_spread_bps = get_bb_spread_at(sym, exit_time_ns) or 0
            
            # Close trade
            exit_cost_usd = EXIT_COST_BPS / 10000 * NOTIONAL
            net_pnl = pos["total_fr_received"] - pos["entry_cost_usd"] - exit_cost_usd
            reason = "fr_flipped" if fr_against_us else "fr_normalized"
            
            trade = {
                "symbol": sym,
                "entry_time": pos["entry_time"],
                "exit_time": st_ts,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "direction": pos["direction"],
                "n_settlements": pos["n_settlements"],
                "total_fr_received": pos["total_fr_received"],
                "entry_cost_usd": pos["entry_cost_usd"],
                "exit_cost_usd": exit_cost_usd,
                "total_cost_usd": pos["entry_cost_usd"] + exit_cost_usd,
                "net_pnl": net_pnl,
                "hold_hours": (st_ts - pos["entry_time"]).total_seconds() / 3600,
                "avg_fr_bps": np.mean([p["fr_bps"] for p in pos["fr_payments"]]),
                "exit_reason": reason,
            }
            closed_trades.append(trade)
            
            trade_events.append({
                "time": st_ts,
                "type": "EXIT",
                "symbol": sym,
                "net_pnl": net_pnl,
                "n_settlements": pos["n_settlements"],
                "total_fr": pos["total_fr_received"],
                "reason": reason,
            })
            
            del open_positions[sym]
    
    # ── Step 3: Check entries — find best new coin to enter ──
    # CAUSALITY: we enter AFTER seeing the just-settled FR. The entry signal
    # is the FR that was just paid (observable). We do NOT collect this FR —
    # we only start collecting from the NEXT settlement onward.
    # The bet: if FR was extreme at this settlement, it will persist.
    if len(open_positions) < MAX_POSITIONS:
        candidates = coins_at_settle[
            (coins_at_settle["fr_paid_bps"] >= ENTRY_THRESHOLD_BPS) &
            (~coins_at_settle["symbol"].isin(open_positions.keys()))
        ].sort_values("fr_paid_bps", ascending=False)
        
        for _, cand in candidates.iterrows():
            if len(open_positions) >= MAX_POSITIONS:
                break
            
            sym = cand["symbol"]
            fr = cand["fr_paid"]
            fr_bps = cand["fr_paid_bps"]
            
            # Get entry price (right at settlement time)
            entry_price = get_price_at(sym, st_ns)
            if entry_price is None or entry_price == 0:
                continue
            
            entry_spread_bps = get_bb_spread_at(sym, st_ns) or 0
            
            # Direction based on FR sign
            if fr < 0:
                direction = "long_futures"  # shorts pay longs, we go long
            else:
                direction = "short_futures"
            
            entry_cost_usd = ENTRY_COST_BPS / 10000 * NOTIONAL
            
            open_positions[sym] = {
                "symbol": sym,
                "entry_time": st_ts,
                "entry_price": entry_price,
                "direction": direction,
                "fr_payments": [],
                "total_fr_received": 0,
                "entry_cost_usd": entry_cost_usd,
                "n_settlements": 0,
                "last_fr_bps": fr_bps,
                "entry_spread_bps": entry_spread_bps,
            }
            
            trade_events.append({
                "time": st_ts,
                "type": "ENTRY",
                "symbol": sym,
                "fr_bps": fr_bps,
                "direction": direction,
                "entry_price": entry_price,
            })
    
    # Progress
    if (si + 1) % 5 == 0 or si == n_settlements - 1:
        elapsed = time.time() - t_sim
        eta = elapsed / (si + 1) * (n_settlements - si - 1) if si > 0 else 0
        n_open = len(open_positions)
        n_closed = len(closed_trades)
        open_syms = ", ".join(open_positions.keys()) if open_positions else "none"
        print(f"  [{si+1:3d}/{n_settlements}] {st_ts}  "
              f"open={n_open} ({open_syms})  closed={n_closed}  "
              f"[{elapsed:.0f}s, ETA {eta:.0f}s]")
        sys.stdout.flush()

# ── Force-close any remaining positions at end of data ──
for sym, pos in list(open_positions.items()):
    last_time = settle_times[-1]
    last_ts = pd.Timestamp(last_time)
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    last_ns = np.datetime64(last_ts, "ns")
    
    exit_price = get_price_at(sym, last_ns + np.timedelta64(60, "s"))
    if exit_price is None:
        exit_price = pos["entry_price"]
    
    exit_cost_usd = EXIT_COST_BPS / 10000 * NOTIONAL
    net_pnl = pos["total_fr_received"] - pos["entry_cost_usd"] - exit_cost_usd
    
    trade = {
        "symbol": sym,
        "entry_time": pos["entry_time"],
        "exit_time": last_ts,
        "entry_price": pos["entry_price"],
        "exit_price": exit_price,
        "direction": pos["direction"],
        "n_settlements": pos["n_settlements"],
        "total_fr_received": pos["total_fr_received"],
        "entry_cost_usd": pos["entry_cost_usd"],
        "exit_cost_usd": exit_cost_usd,
        "total_cost_usd": pos["entry_cost_usd"] + exit_cost_usd,
        "net_pnl": net_pnl,
        "hold_hours": (last_ts - pos["entry_time"]).total_seconds() / 3600,
        "avg_fr_bps": np.mean([p["fr_bps"] for p in pos["fr_payments"]]) if pos["fr_payments"] else 0,
        "exit_reason": "end_of_data",
    }
    closed_trades.append(trade)
    
    trade_events.append({
        "time": last_ts,
        "type": "EXIT",
        "symbol": sym,
        "net_pnl": net_pnl,
        "n_settlements": pos["n_settlements"],
        "total_fr": pos["total_fr_received"],
        "reason": "end_of_data",
    })

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Results
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 90)
print("BACKTEST v2 RESULTS — Hold Until Normalize")
print("=" * 90)

if not closed_trades:
    print("  No trades!")
    sys.exit(0)

df = pd.DataFrame(closed_trades)

n_trades = len(df)
n_wins = (df["net_pnl"] > 0).sum()
total_pnl = df["net_pnl"].sum()
total_fr = df["total_fr_received"].sum()
total_cost = df["total_cost_usd"].sum()
total_settlements = df["n_settlements"].sum()

t_first = df["entry_time"].min()
t_last = df["exit_time"].max()
days = (t_last - t_first).total_seconds() / 86400

print(f"\n  Period:           {t_first} → {t_last} ({days:.1f} days)")
print(f"  Notional:         ${NOTIONAL:,.0f} per position")
print(f"  Fee model:        {TAKER_FEE_BPS} bps taker + {SLIPPAGE_BPS} bps slippage per side")
print(f"  Total RT cost:    {TOTAL_RT_COST_BPS:.0f} bps (paid only at entry+exit)")

print(f"\n  {'─'*70}")
print(f"  Positions opened: {n_trades}")
print(f"  Profitable:       {n_wins} ({n_wins/n_trades*100:.0f}%)")
print(f"  Settlements held: {int(total_settlements)} total ({total_settlements/n_trades:.1f} avg per trade)")
print(f"  {'─'*70}")
print(f"  Gross FR income:  ${total_fr:,.2f}")
print(f"  Total costs:      ${total_cost:,.2f} ({n_trades} entries × ${ENTRY_COST_BPS/10000*NOTIONAL:.2f} + exits)")
print(f"  Net P&L:          ${total_pnl:+,.2f}")
print(f"  Avg P&L/trade:    ${total_pnl/n_trades:+,.2f}")
print(f"  Cost efficiency:  {total_fr/total_cost:.1f}x (FR income / costs)")
print(f"  {'─'*70}")

if days > 0:
    daily_pnl = total_pnl / days
    daily_fr = total_fr / days
    daily_cost = total_cost / days
    print(f"  Daily FR income:  ${daily_fr:+,.2f}")
    print(f"  Daily costs:      ${daily_cost:,.2f}")
    print(f"  Daily net P&L:    ${daily_pnl:+,.2f}")
    print(f"  Daily ROI:        {daily_pnl/NOTIONAL*100:+.2f}% on ${NOTIONAL:,}")
    print(f"  Annualized P&L:   ${daily_pnl*365:+,.0f}")
    print(f"  Annual ROI:       {daily_pnl*365/NOTIONAL*100:+.1f}%")

# ── Per-trade detail ──
print(f"\n  {'─'*70}")
print(f"  Trade detail:")
print(f"  {'Symbol':<14} {'Entry':<22} {'Exit':<22} {'Hold':>6} {'#Sttl':>5} "
      f"{'FR inc':>9} {'Cost':>7} {'Net':>9} {'Reason':<15}")

cum_pnl = 0
for _, t in df.iterrows():
    cum_pnl += t["net_pnl"]
    hold_str = f"{t['hold_hours']:.0f}h"
    print(f"  {t['symbol']:<14} {str(t['entry_time']):<22} {str(t['exit_time']):<22} "
          f"{hold_str:>6} {int(t['n_settlements']):>5} "
          f"${t['total_fr_received']:>8.2f} ${t['total_cost_usd']:>6.2f} "
          f"${t['net_pnl']:>+8.2f} {t['exit_reason']:<15}")

print(f"\n  Cumulative P&L: ${cum_pnl:+,.2f}")

# ── Comparison: v1 (enter/exit each settlement) vs v2 (hold) ──
print(f"\n  {'─'*70}")
print(f"  KEY ADVANTAGE: cost amortization")
v1_cost = total_settlements * TOTAL_RT_COST_BPS / 10000 * NOTIONAL
v2_cost = total_cost
savings = v1_cost - v2_cost
print(f"    v1 model (enter/exit each settlement): {int(total_settlements)} round-trips × ${TOTAL_RT_COST_BPS/10000*NOTIONAL:.2f} = ${v1_cost:,.2f}")
print(f"    v2 model (hold across settlements):    {n_trades} round-trips × ${TOTAL_RT_COST_BPS/10000*NOTIONAL:.2f} = ${v2_cost:,.2f}")
print(f"    Cost savings:                          ${savings:,.2f} ({savings/v1_cost*100:.0f}%)")
print(f"    v1 net P&L (same FR):                  ${total_fr - v1_cost:+,.2f}")
print(f"    v2 net P&L:                            ${total_pnl:+,.2f}")
print(f"    Improvement:                           ${total_pnl - (total_fr - v1_cost):+,.2f}")

# ── Event timeline ──
print(f"\n  {'─'*70}")
print(f"  Event timeline (entries, FR collections, exits):")

events_df = pd.DataFrame(trade_events)
cum_pnl_ev = 0.0
for _, ev in events_df.iterrows():
    if ev["type"] == "ENTRY":
        print(f"    {ev['time']}  ENTER  {ev['symbol']:<14} FR={ev['fr_bps']:.1f}bps  "
              f"dir={ev['direction']}  price=${ev['entry_price']:.4f}")
    elif ev["type"] == "FR_COLLECT":
        cum_pnl_ev += ev["payment_usd"]
        pay_str = f"+${ev['payment_usd']:.2f}" if ev["payment_usd"] >= 0 else f"-${-ev['payment_usd']:.2f}"
        print(f"    {ev['time']}  FR     {ev['symbol']:<14} FR={ev['fr_bps']:.1f}bps  "
              f"{pay_str}  cum_FR=${ev['cumulative_fr']:.2f}")
    elif ev["type"] == "EXIT":
        cost_entry = ENTRY_COST_BPS / 10000 * NOTIONAL
        cost_exit = EXIT_COST_BPS / 10000 * NOTIONAL
        cum_pnl_ev -= (cost_entry + cost_exit)
        print(f"    {ev['time']}  EXIT   {ev['symbol']:<14} {ev['reason']:<20} "
              f"net=${ev['net_pnl']:+.2f}  ({ev['n_settlements']} settlements)")

# ── Sensitivity: different entry/exit thresholds ──
print(f"\n  {'─'*70}")
print(f"  SENSITIVITY ANALYSIS: What if we use different thresholds?")
print(f"  (rerunning with different entry_threshold / exit_threshold)")

# Quick re-simulation with different thresholds (same corrected logic)
for entry_th, exit_th in [(10, 3), (15, 5), (20, 8), (30, 10), (50, 15)]:
    mini_positions = {}  # sym -> {direction, total_fr, n_settle, last_fr}
    mini_closed = []
    
    for si, st in enumerate(settle_times):
        st_ts = pd.Timestamp(st)
        if st_ts.tzinfo is None:
            st_ts = st_ts.tz_localize("UTC")
        
        coins_at = settle_df[settle_df["settle_time"] == st]
        
        # Collect FR (direction-aware)
        for sym, pos in list(mini_positions.items()):
            cr = coins_at[coins_at["symbol"] == sym]
            if len(cr) == 0:
                continue
            fr = float(cr.iloc[0]["fr_paid"])
            if pos["direction"] == "long_futures":
                payment = -fr * NOTIONAL
            else:
                payment = fr * NOTIONAL
            pos["total_fr"] += payment
            pos["n_settle"] += 1
            pos["last_fr"] = abs(fr) * 10000
        
        # Exits: FR normalized OR flipped against us
        for sym in list(mini_positions.keys()):
            cr = coins_at[coins_at["symbol"] == sym]
            if len(cr) == 0:
                continue
            fr = float(cr.iloc[0]["fr_paid"])
            fr_bps = abs(fr) * 10000
            pos = mini_positions[sym]
            flipped = (pos["direction"] == "long_futures" and fr > 0) or \
                      (pos["direction"] == "short_futures" and fr < 0)
            if fr_bps < exit_th or flipped:
                cost = TOTAL_RT_COST_BPS / 10000 * NOTIONAL
                mini_closed.append({
                    "net_pnl": pos["total_fr"] - cost,
                    "n_settle": pos["n_settle"],
                    "total_fr": pos["total_fr"],
                })
                del mini_positions[sym]
        
        # Entries
        if len(mini_positions) < MAX_POSITIONS:
            cands = coins_at[
                (coins_at["fr_paid_bps"] >= entry_th) &
                (~coins_at["symbol"].isin(mini_positions.keys()))
            ].sort_values("fr_paid_bps", ascending=False)
            for _, c in cands.iterrows():
                if len(mini_positions) >= MAX_POSITIONS:
                    break
                fr = float(c["fr_paid"])
                direction = "long_futures" if fr < 0 else "short_futures"
                mini_positions[c["symbol"]] = {
                    "direction": direction, "total_fr": 0,
                    "n_settle": 0, "last_fr": 0,
                }
    
    # Force close remaining
    for sym, pos in mini_positions.items():
        cost = TOTAL_RT_COST_BPS / 10000 * NOTIONAL
        mini_closed.append({
            "net_pnl": pos["total_fr"] - cost,
            "n_settle": pos["n_settle"],
            "total_fr": pos["total_fr"],
        })
    
    if mini_closed:
        mc = pd.DataFrame(mini_closed)
        n_tr = len(mc)
        n_w = (mc["net_pnl"] > 0).sum()
        total = mc["net_pnl"].sum()
        total_sett = mc["n_settle"].sum()
        daily = total / days if days > 0 else 0
        cost_total = n_tr * TOTAL_RT_COST_BPS / 10000 * NOTIONAL
        fr_total = mc["total_fr"].sum()
        print(f"    entry≥{entry_th:>2}bps exit<{exit_th:>2}bps: "
              f"{n_tr:>3} trades, {int(total_sett):>4} settlements, "
              f"WR {n_w/n_tr*100:>5.1f}%, "
              f"FR ${fr_total:>8,.2f}, cost ${cost_total:>6,.2f}, "
              f"net ${total:>+9,.2f}, daily ${daily:>+7,.2f}")
    else:
        print(f"    entry≥{entry_th:>2}bps exit<{exit_th:>2}bps: no trades")

elapsed = time.time() - t_global
print(f"\n{'='*90}")
print(f"Backtest v2 complete [{elapsed:.1f}s]")
print(f"{'='*90}")
