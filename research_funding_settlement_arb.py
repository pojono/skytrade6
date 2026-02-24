#!/usr/bin/env python3
"""
Research: Funding Rate Settlement Arbitrage

Strategy: Open delta-neutral position just BEFORE settlement, collect FR, close immediately after.
- If FR is negative on exchange A: go LONG on A (you GET paid)
- Hedge with SHORT on exchange B (or vice versa)
- Net profit = |FR_A - FR_B| minus execution costs

Execution costs per leg:
- Taker fee: ~2.5 bps (Bybit VIP0), ~2 bps (Binance VIP0)  
- Bid-ask spread: ~half the spread per leg
- Slippage: assume 1 bps per leg for liquid coins

We need to identify EVERY settlement event, the FR at that moment, and compute net P&L.

Key insight: Bybit 1h coins settle 24x/day, Binance 8h coins settle 3x/day.
When BOTH settle at the same time, we can do single-settlement arb.
When only ONE settles, we can still collect on that exchange (but need hedge on the other).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

DATA = Path("data_all")
t0 = time.time()

# ── Cost assumptions ──────────────────────────────────────────────────────────
# Per-leg costs (we open+close = 2 trades per leg, 2 legs = 4 trades total)
TAKER_FEE_BPS = 2.5      # bps per trade (conservative)
SLIPPAGE_BPS = 1.0        # bps per trade
# Total round-trip cost = 4 trades × (taker + slippage) = 4 × 3.5 = 14 bps
# Plus bid-ask spread on each leg (measured from data)
FIXED_COST_PER_TRADE_BPS = TAKER_FEE_BPS + SLIPPAGE_BPS  # 3.5 bps
TOTAL_FIXED_COST_BPS = 4 * FIXED_COST_PER_TRADE_BPS       # 14 bps = 0.0014

print("=" * 100)
print("FUNDING RATE SETTLEMENT ARBITRAGE ANALYSIS")
print("=" * 100)
print(f"Cost model: {TAKER_FEE_BPS} bps taker + {SLIPPAGE_BPS} bps slippage per trade")
print(f"Round-trip (open+close both legs): {TOTAL_FIXED_COST_BPS:.0f} bps fixed + bid-ask spread")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA (memory-efficient)
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data...")

# Binance funding rate — need nextFundingTime to detect settlements
bn_fr = pd.read_parquet(
    DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "markPrice", "nextFundingTime"],
)
print(f"  Binance FR: {len(bn_fr):,} rows")

# Bybit ticker — has fundingRate + nextFundingTime + bid/ask
bb_tk = pd.read_parquet(
    DATA / "bybit" / "ticker.parquet",
    columns=["ts", "symbol", "fundingRate", "lastPrice", "markPrice",
             "bid1Price", "ask1Price", "nextFundingTime", "fundingIntervalHour",
             "openInterestValue", "volume24h"],
)
print(f"  Bybit ticker: {len(bb_tk):,} rows")

# Binance ticker — need lastPrice for price spread
bn_tk = pd.read_parquet(
    DATA / "binance" / "ticker.parquet",
    columns=["ts", "symbol", "lastPrice"],
)
print(f"  Binance ticker: {len(bn_tk):,} rows")

# Common symbols
common = sorted(set(bn_fr["symbol"].unique()) & set(bb_tk["symbol"].unique()))
bn_fr = bn_fr[bn_fr["symbol"].isin(common)]
bb_tk = bb_tk[bb_tk["symbol"].isin(common)]
bn_tk = bn_tk[bn_tk["symbol"].isin(common)]
print(f"  Common symbols: {len(common)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DETECT SETTLEMENT EVENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("DETECTING SETTLEMENT EVENTS")
print("=" * 100)

# Strategy: for each symbol, find when nextFundingTime changes — that's a settlement.
# We want the snapshot JUST BEFORE settlement (last snapshot where nextFundingTime = T)
# and the FR at that moment.

def detect_settlements(df, ts_col, symbol_col, next_funding_col, fr_col, exchange_name):
    """Detect settlement events by finding when nextFundingTime jumps."""
    print(f"\n  Detecting {exchange_name} settlements...")
    
    # Downsample to 1-min to speed up
    df = df.copy()
    df["ts_1m"] = df[ts_col].dt.floor("1min")
    df = df.sort_values([symbol_col, "ts_1m"])
    
    # Take last observation per minute per symbol
    df = df.groupby(["ts_1m", symbol_col]).last().reset_index()
    
    # For each symbol, detect when nextFundingTime changes
    df["next_funding_shift"] = df.groupby(symbol_col)[next_funding_col].shift(1)
    df["settlement"] = (df[next_funding_col] != df["next_funding_shift"]) & df["next_funding_shift"].notna()
    
    # The settlement happened at the PREVIOUS row's time
    # We want the row JUST BEFORE the change (that's when FR was active)
    # Actually: the row where settlement=True means nextFundingTime just changed,
    # so the PREVIOUS row had the old nextFundingTime and the FR that was just settled.
    
    # Get the row just before each settlement
    settlements = []
    for sym in df[df["settlement"]][symbol_col].unique():
        sym_df = df[df[symbol_col] == sym].reset_index(drop=True)
        settle_idx = sym_df[sym_df["settlement"]].index
        
        for idx in settle_idx:
            if idx > 0:
                pre = sym_df.iloc[idx - 1]  # last row before settlement
                post = sym_df.iloc[idx]      # first row after settlement
                settlements.append({
                    "symbol": sym,
                    "settlement_time": post["ts_1m"],  # when we detected the change
                    f"{exchange_name}_fr": pre[fr_col],
                    f"{exchange_name}_markPrice": pre["markPrice"],
                    f"{exchange_name}_nextFunding_before": pre[next_funding_col],
                })
    
    result = pd.DataFrame(settlements)
    print(f"    Found {len(result):,} settlement events across {result['symbol'].nunique()} symbols")
    return result

# Detect Binance settlements
bn_settlements = detect_settlements(
    bn_fr, "ts", "symbol", "nextFundingTime", "lastFundingRate", "bn"
)

# Detect Bybit settlements
bb_settlements = detect_settlements(
    bb_tk, "ts", "symbol", "nextFundingTime", "fundingRate", "bb"
)

# Also grab Bybit bid/ask and OI at settlement time
bb_tk["ts_1m"] = bb_tk["ts"].dt.floor("1min")
bb_snap = bb_tk.groupby(["ts_1m", "symbol"]).agg(
    bb_bid1=("bid1Price", "last"),
    bb_ask1=("ask1Price", "last"),
    bb_lastPrice=("lastPrice", "last"),
    bb_oi=("openInterestValue", "last"),
    bb_vol24h=("volume24h", "last"),
    bb_interval=("fundingIntervalHour", "last"),
).reset_index()

bn_tk["ts_1m"] = bn_tk["ts"].dt.floor("1min")
bn_tk_snap = bn_tk.groupby(["ts_1m", "symbol"]).agg(
    bn_lastPrice=("lastPrice", "last"),
).reset_index()

del bn_fr, bb_tk, bn_tk
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANALYZE SETTLEMENT OPPORTUNITIES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("SETTLEMENT OPPORTUNITIES")
print("=" * 100)

# Case A: BOTH exchanges settle at the same time (within 2 min window)
# This is the cleanest arb — collect FR on both sides simultaneously

# Case B: Only ONE exchange settles
# You collect FR on settling exchange, but need hedge on the other
# The hedge costs you the round-trip on both exchanges

# Let's merge settlements with market data at settlement time

# Enrich Bybit settlements with bid/ask data
bb_settlements = bb_settlements.merge(
    bb_snap, left_on=["settlement_time", "symbol"], right_on=["ts_1m", "symbol"], how="left"
)
bb_settlements = bb_settlements.merge(
    bn_tk_snap, left_on=["settlement_time", "symbol"], right_on=["ts_1m", "symbol"], how="left"
)

# Enrich Binance settlements similarly
bn_settlements = bn_settlements.merge(
    bb_snap, left_on=["settlement_time", "symbol"], right_on=["ts_1m", "symbol"], how="left"
)
bn_settlements = bn_settlements.merge(
    bn_tk_snap, left_on=["settlement_time", "symbol"], right_on=["ts_1m", "symbol"], how="left"
)

print(f"\nBinance settlement events: {len(bn_settlements):,}")
print(f"Bybit settlement events:  {len(bb_settlements):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CASE A: SIMULTANEOUS SETTLEMENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("CASE A: SIMULTANEOUS SETTLEMENTS (both exchanges settle within 2 min)")
print("=" * 100)

# Find settlements that happen at the same time (±2 min) for the same symbol
bn_s = bn_settlements[["symbol", "settlement_time", "bn_fr"]].copy()
bb_s = bb_settlements[["symbol", "settlement_time", "bb_fr", "bb_bid1", "bb_ask1",
                         "bb_lastPrice", "bn_lastPrice", "bb_oi", "bb_interval"]].copy()

# Round to 2-min windows for matching
bn_s["settle_2m"] = bn_s["settlement_time"].dt.floor("2min")
bb_s["settle_2m"] = bb_s["settlement_time"].dt.floor("2min")

simultaneous = bn_s.merge(bb_s, on=["symbol", "settle_2m"], suffixes=("_bn", "_bb"))

print(f"Simultaneous settlements: {len(simultaneous):,} across {simultaneous['symbol'].nunique()} symbols")

if len(simultaneous) > 0:
    # Compute the arb P&L
    # Strategy: if bn_fr > bb_fr, go short BN (pay bn_fr) + long BB (receive bb_fr)... wait no.
    # 
    # FR payment: if you are LONG and FR is positive, you PAY. If LONG and FR negative, you RECEIVE.
    # If you are SHORT and FR is positive, you RECEIVE. If SHORT and FR negative, you PAY.
    #
    # So for delta-neutral arb at settlement:
    # Option 1: Long on exchange with more negative FR (receive more), Short on other
    # Option 2: Short on exchange with more positive FR (receive more), Long on other
    #
    # Net FR collected = |bn_fr - bb_fr| (the spread)
    # But we also need to account for the FR we pay on the hedge leg.
    #
    # Actually simpler: 
    # If we go LONG BN + SHORT BB:
    #   FR P&L = -bn_fr + bb_fr  (long pays positive FR, short receives positive FR)
    # If we go SHORT BN + LONG BB:
    #   FR P&L = bn_fr - bb_fr
    # We pick whichever is positive = |bn_fr - bb_fr|
    
    simultaneous["fr_spread"] = (simultaneous["bn_fr"] - simultaneous["bb_fr"]).abs()
    simultaneous["fr_spread_bps"] = simultaneous["fr_spread"] * 10000
    
    # Execution costs
    # Bid-ask spread cost (half spread per entry + half spread per exit, per leg)
    simultaneous["bb_spread_bps"] = (simultaneous["bb_ask1"] - simultaneous["bb_bid1"]) / simultaneous["bb_bid1"] * 10000
    # Assume Binance spread ≈ Bybit spread (we don't have BN order book)
    simultaneous["total_spread_cost_bps"] = simultaneous["bb_spread_bps"] * 2  # both legs
    
    # Total cost
    simultaneous["total_cost_bps"] = TOTAL_FIXED_COST_BPS + simultaneous["total_spread_cost_bps"]
    
    # Net P&L
    simultaneous["net_pnl_bps"] = simultaneous["fr_spread_bps"] - simultaneous["total_cost_bps"]
    simultaneous["profitable"] = simultaneous["net_pnl_bps"] > 0
    
    n_profitable = simultaneous["profitable"].sum()
    n_total = len(simultaneous)
    
    print(f"\nProfitable: {n_profitable} / {n_total} ({100*n_profitable/n_total:.1f}%)")
    print(f"\nFR spread distribution (bps):")
    print(simultaneous["fr_spread_bps"].describe().to_string())
    print(f"\nExecution cost distribution (bps):")
    print(simultaneous["total_cost_bps"].describe().to_string())
    print(f"\nNet P&L distribution (bps):")
    print(simultaneous["net_pnl_bps"].describe().to_string())
    
    if n_profitable > 0:
        print(f"\n--- Top 30 PROFITABLE simultaneous settlements ---")
        top = simultaneous[simultaneous["profitable"]].nlargest(30, "net_pnl_bps")
        print(top[["symbol", "settlement_time_bn", "bn_fr", "bb_fr", "fr_spread_bps",
                    "total_cost_bps", "net_pnl_bps", "bb_oi", "bb_interval"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CASE B: SINGLE-EXCHANGE SETTLEMENTS (Bybit settles, Binance doesn't)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("CASE B: BYBIT-ONLY SETTLEMENTS (Bybit settles, hedge on Binance)")
print("=" * 100)

# For Bybit-only settlements, we:
# - Go LONG Bybit if FR is negative (we receive FR)
# - Go SHORT Bybit if FR is positive (we receive FR)
# - Hedge with opposite on Binance (no FR event, just delta hedge)
# - Close both after settlement
#
# Revenue = |bb_fr| (the FR we collect on Bybit)
# Cost = round-trip execution on both exchanges
#
# This is simpler — we just need |bb_fr| > execution cost

# Find Bybit settlements that DON'T coincide with Binance settlements
bb_only = bb_settlements.copy()
bb_only["settle_2m"] = bb_only["settlement_time"].dt.floor("2min")

# Remove those that are simultaneous
if len(simultaneous) > 0:
    simul_keys = set(zip(simultaneous["symbol"], simultaneous["settle_2m"]))
    bb_only["is_simul"] = bb_only.apply(lambda r: (r["symbol"], r["settle_2m"]) in simul_keys, axis=1)
    bb_only = bb_only[~bb_only["is_simul"]].copy()

print(f"Bybit-only settlements: {len(bb_only):,} across {bb_only['symbol'].nunique()} symbols")

# Compute P&L
bb_only["fr_abs_bps"] = bb_only["bb_fr"].abs() * 10000
bb_only["bb_spread_bps"] = (bb_only["bb_ask1"] - bb_only["bb_bid1"]) / bb_only["bb_bid1"] * 10000
bb_only["total_spread_cost_bps"] = bb_only["bb_spread_bps"] * 2  # both legs
bb_only["total_cost_bps"] = TOTAL_FIXED_COST_BPS + bb_only["total_spread_cost_bps"]
bb_only["net_pnl_bps"] = bb_only["fr_abs_bps"] - bb_only["total_cost_bps"]
bb_only["profitable"] = bb_only["net_pnl_bps"] > 0

n_profitable_bb = bb_only["profitable"].sum()
n_total_bb = len(bb_only)

print(f"\nProfitable: {n_profitable_bb} / {n_total_bb} ({100*n_profitable_bb/n_total_bb:.1f}%)")
print(f"\nBybit FR (absolute, bps) distribution:")
print(bb_only["fr_abs_bps"].describe().to_string())
print(f"\nExecution cost distribution (bps):")
print(bb_only["total_cost_bps"].describe().to_string())
print(f"\nNet P&L distribution (bps):")
print(bb_only["net_pnl_bps"].describe().to_string())

if n_profitable_bb > 0:
    print(f"\n--- Top 40 PROFITABLE Bybit-only settlements ---")
    top_bb = bb_only[bb_only["profitable"]].nlargest(40, "net_pnl_bps")
    print(top_bb[["symbol", "settlement_time", "bb_fr", "fr_abs_bps",
                   "bb_spread_bps", "total_cost_bps", "net_pnl_bps",
                   "bb_oi", "bb_interval"]].to_string(index=False))
    
    # Summary by symbol
    print(f"\n--- Profitable Bybit-only settlements by symbol ---")
    sym_summary = bb_only[bb_only["profitable"]].groupby("symbol").agg(
        count=("net_pnl_bps", "count"),
        total_pnl_bps=("net_pnl_bps", "sum"),
        avg_pnl_bps=("net_pnl_bps", "mean"),
        avg_fr_bps=("fr_abs_bps", "mean"),
        avg_cost_bps=("total_cost_bps", "mean"),
        avg_oi=("bb_oi", "mean"),
        interval=("bb_interval", "first"),
    ).sort_values("total_pnl_bps", ascending=False)
    print(sym_summary.head(30).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 6. CASE C: BINANCE-ONLY SETTLEMENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("CASE C: BINANCE-ONLY SETTLEMENTS (Binance settles, hedge on Bybit)")
print("=" * 100)

bn_only = bn_settlements.copy()
bn_only["settle_2m"] = bn_only["settlement_time"].dt.floor("2min")

if len(simultaneous) > 0:
    bn_only["is_simul"] = bn_only.apply(lambda r: (r["symbol"], r["settle_2m"]) in simul_keys, axis=1)
    bn_only = bn_only[~bn_only["is_simul"]].copy()

print(f"Binance-only settlements: {len(bn_only):,} across {bn_only['symbol'].nunique()} symbols")

bn_only["fr_abs_bps"] = bn_only["bn_fr"].abs() * 10000
bn_only["bb_spread_bps"] = (bn_only["bb_ask1"] - bn_only["bb_bid1"]) / bn_only["bb_bid1"] * 10000
bn_only["total_spread_cost_bps"] = bn_only["bb_spread_bps"] * 2
bn_only["total_cost_bps"] = TOTAL_FIXED_COST_BPS + bn_only["total_spread_cost_bps"]
bn_only["net_pnl_bps"] = bn_only["fr_abs_bps"] - bn_only["total_cost_bps"]
bn_only["profitable"] = bn_only["net_pnl_bps"] > 0

n_profitable_bn = bn_only["profitable"].sum()
n_total_bn = len(bn_only)

print(f"\nProfitable: {n_profitable_bn} / {n_total_bn} ({100*n_profitable_bn/n_total_bn:.1f}%)")
print(f"\nBinance FR (absolute, bps) distribution:")
print(bn_only["fr_abs_bps"].describe().to_string())
print(f"\nNet P&L distribution (bps):")
print(bn_only["net_pnl_bps"].describe().to_string())

if n_profitable_bn > 0:
    print(f"\n--- Top 40 PROFITABLE Binance-only settlements ---")
    top_bn = bn_only[bn_only["profitable"]].nlargest(40, "net_pnl_bps")
    print(top_bn[["symbol", "settlement_time", "bn_fr", "fr_abs_bps",
                   "bb_spread_bps", "total_cost_bps", "net_pnl_bps",
                   "bb_oi", "bb_interval"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 7. GRAND SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("GRAND SUMMARY — ALL SETTLEMENT ARBITRAGE OPPORTUNITIES")
print("=" * 100)

total_settlements = len(simultaneous) + n_total_bb + n_total_bn
total_profitable = (simultaneous["profitable"].sum() if len(simultaneous) > 0 else 0) + n_profitable_bb + n_profitable_bn

print(f"\nTotal settlement events (2 days): {total_settlements:,}")
print(f"  Simultaneous (both):  {len(simultaneous):,} → {simultaneous['profitable'].sum() if len(simultaneous) > 0 else 0} profitable")
print(f"  Bybit-only:           {n_total_bb:,} → {n_profitable_bb} profitable")
print(f"  Binance-only:         {n_total_bn:,} → {n_profitable_bn} profitable")
print(f"  TOTAL PROFITABLE:     {total_profitable} / {total_settlements}")

# Aggregate all profitable trades
all_profitable = []
if len(simultaneous) > 0 and simultaneous["profitable"].sum() > 0:
    s = simultaneous[simultaneous["profitable"]].copy()
    s["type"] = "simultaneous"
    s["settlement_time"] = s["settlement_time_bn"]
    s["fr_collected_bps"] = s["fr_spread_bps"]
    all_profitable.append(s[["symbol", "settlement_time", "type", "fr_collected_bps", "total_cost_bps", "net_pnl_bps", "bb_oi"]])

if n_profitable_bb > 0:
    b = bb_only[bb_only["profitable"]].copy()
    b["type"] = "bybit_only"
    b["fr_collected_bps"] = b["fr_abs_bps"]
    all_profitable.append(b[["symbol", "settlement_time", "type", "fr_collected_bps", "total_cost_bps", "net_pnl_bps", "bb_oi"]])

if n_profitable_bn > 0:
    b = bn_only[bn_only["profitable"]].copy()
    b["type"] = "binance_only"
    b["fr_collected_bps"] = b["fr_abs_bps"]
    all_profitable.append(b[["symbol", "settlement_time", "type", "fr_collected_bps", "total_cost_bps", "net_pnl_bps", "bb_oi"]])

if all_profitable:
    all_prof = pd.concat(all_profitable, ignore_index=True)
    
    print(f"\n--- P&L Summary (all profitable trades) ---")
    print(f"  Total trades:        {len(all_prof)}")
    print(f"  Total net P&L:       {all_prof['net_pnl_bps'].sum():.1f} bps")
    print(f"  Avg net P&L/trade:   {all_prof['net_pnl_bps'].mean():.1f} bps")
    print(f"  Median net P&L:      {all_prof['net_pnl_bps'].median():.1f} bps")
    print(f"  Max single trade:    {all_prof['net_pnl_bps'].max():.1f} bps")
    
    # If we traded $10K per trade
    notional = 10000
    all_prof["pnl_usd"] = all_prof["net_pnl_bps"] / 10000 * notional
    print(f"\n  At $10K notional per trade:")
    print(f"    Total P&L:         ${all_prof['pnl_usd'].sum():.2f}")
    print(f"    Avg P&L/trade:     ${all_prof['pnl_usd'].mean():.2f}")
    print(f"    Trades/day:        {len(all_prof) / 2:.0f}")
    
    # By type
    print(f"\n--- Breakdown by type ---")
    type_summary = all_prof.groupby("type").agg(
        trades=("net_pnl_bps", "count"),
        total_pnl_bps=("net_pnl_bps", "sum"),
        avg_pnl_bps=("net_pnl_bps", "mean"),
        total_pnl_usd=("pnl_usd", "sum"),
    )
    print(type_summary.to_string())
    
    # Timeline — how are opportunities distributed?
    print(f"\n--- Opportunities by hour ---")
    all_prof["hour"] = all_prof["settlement_time"].dt.hour
    hourly = all_prof.groupby("hour").agg(
        trades=("net_pnl_bps", "count"),
        total_pnl_bps=("net_pnl_bps", "sum"),
    )
    print(hourly.to_string())

    # Capacity analysis — what's the realistic notional?
    print(f"\n--- Capacity analysis (OI-based) ---")
    # Assume we can trade up to 0.1% of OI without moving the market
    all_prof["max_notional"] = all_prof["bb_oi"] * 0.001
    all_prof["realistic_pnl"] = all_prof["net_pnl_bps"] / 10000 * all_prof["max_notional"]
    print(f"  Total realistic P&L (0.1% of OI): ${all_prof['realistic_pnl'].sum():.2f}")
    print(f"  Avg realistic P&L/trade:           ${all_prof['realistic_pnl'].mean():.2f}")

elapsed = time.time() - t0
print(f"\n\nAnalysis completed in {elapsed:.1f}s")
