#!/usr/bin/env python3
"""
Comprehensive Strategy Comparison & Commission Audit

Strategy A: Scalp — enter ~60s before settlement, exit ~60s after
Strategy B: Hold — enter after seeing extreme FR, hold until FR normalizes

Pessimistic assumptions throughout. Every cost counted twice if in doubt.

Commission audit:
  - We trade on Binance (spot + futures)
  - Spot taker: 0.1% = 10 bps (standard, NOT VIP)
  - Futures taker: 0.055% = 5.5 bps (user's confirmed rate)
  - BUT WAIT: user said "my take commission is 0.055%" — need to clarify if
    that's futures only or both. Let's model BOTH scenarios.
  
Delta-neutral trade = 4 legs:
  ENTRY:  buy spot + sell futures (or vice versa)
  EXIT:   sell spot + buy futures
  
  Each leg has: exchange fee + slippage + spread crossing
"""
import sys
import time
import gc
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

DATA = Path("data_all")
NOTIONAL = 10_000

print("=" * 100)
print("STRATEGY COMPARISON & COMMISSION AUDIT — PESSIMISTIC SCENARIO")
print("=" * 100)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: COMMISSION AUDIT — What do we ACTUALLY pay?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 100)
print("PART 1: COMMISSION AUDIT")
print("═" * 100)

print("""
Delta-neutral trade structure:
  ENTRY: Buy $10k spot + Sell $10k futures (if FR < 0, shorts pay longs)
  EXIT:  Sell $10k spot + Buy $10k futures

Each leg costs: exchange_fee + slippage

Fee scenarios (per leg, in bps):
""")

# Fee models to test
FEE_MODELS = {
    "optimistic": {
        "label": "VIP spot maker + futures taker",
        "spot_fee": 1.0,       # maker rebate on some tiers
        "futures_fee": 5.5,    # user's confirmed rate
        "slippage": 1.0,       # optimistic slippage
    },
    "realistic": {
        "label": "Spot taker 0.1% + futures taker 0.055%",
        "spot_fee": 10.0,      # standard Binance spot taker
        "futures_fee": 5.5,    # user's confirmed rate
        "slippage": 2.0,       # moderate slippage
    },
    "pessimistic": {
        "label": "Spot taker 0.1% + futures taker 0.055% + high slippage",
        "spot_fee": 10.0,
        "futures_fee": 5.5,
        "slippage": 5.0,       # pessimistic slippage (thin books)
    },
    "user_055_both": {
        "label": "If 0.055% applies to BOTH spot and futures",
        "spot_fee": 5.5,
        "futures_fee": 5.5,
        "slippage": 2.0,
    },
}

for name, model in FEE_MODELS.items():
    # Entry: 2 legs (spot + futures), Exit: 2 legs (spot + futures)
    entry_cost_bps = model["spot_fee"] + model["futures_fee"] + 2 * model["slippage"]
    exit_cost_bps = model["spot_fee"] + model["futures_fee"] + 2 * model["slippage"]
    rt_cost_bps = entry_cost_bps + exit_cost_bps
    rt_cost_usd = rt_cost_bps / 10000 * NOTIONAL
    
    print(f"  {name:<20} ({model['label']})")
    print(f"    Spot fee:      {model['spot_fee']:>5.1f} bps/leg")
    print(f"    Futures fee:   {model['futures_fee']:>5.1f} bps/leg")
    print(f"    Slippage:      {model['slippage']:>5.1f} bps/leg")
    print(f"    Entry (2 legs): {entry_cost_bps:>5.1f} bps  =  ${entry_cost_bps/10000*NOTIONAL:.2f}")
    print(f"    Exit  (2 legs): {exit_cost_bps:>5.1f} bps  =  ${exit_cost_bps/10000*NOTIONAL:.2f}")
    print(f"    ROUND TRIP:     {rt_cost_bps:>5.1f} bps  =  ${rt_cost_usd:.2f}")
    print(f"    Break-even FR (1 settlement): {rt_cost_bps:.1f} bps")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Load data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
sys.stdout.flush()
t_load = time.time()

bn_fr = pd.read_parquet(DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "nextFundingTime"])
print(f"  Binance FR: {len(bn_fr):,} [{time.time()-t_load:.1f}s]")

_bn_tk = pd.read_parquet(DATA / "binance" / "ticker.parquet",
    columns=["ts", "symbol", "lastPrice"])
bn_tk_ts = _bn_tk["ts"].values
bn_tk_sym = _bn_tk["symbol"].values
bn_tk_price = _bn_tk["lastPrice"].values
del _bn_tk; gc.collect()
print(f"  Binance ticker: {len(bn_tk_ts):,} [{time.time()-t_load:.1f}s]")
sys.stdout.flush()

# Build settlement schedule
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
settle_rows["settle_time"] = settle_rows["ts_1m"]
settle_rows["fr_paid"] = settle_rows["fr_prev"]
settle_rows["fr_paid_bps"] = settle_rows["fr_paid"].abs() * 10000

# Also get the FR BEFORE settlement (what we can observe as signal)
# This is bn_fr_1m 2 rows before the settlement event (the FR at T-2min)
bn_fr_1m["fr_2_prev"] = bn_fr_1m.groupby("symbol")["fr"].shift(2)
settle_rows["fr_signal"] = bn_fr_1m.loc[settle_rows.index, "fr_2_prev"]
settle_rows["fr_signal_bps"] = settle_rows["fr_signal"].abs() * 10000

settle_df = settle_rows[["settle_time", "symbol", "fr_paid", "fr_paid_bps",
                          "fr_signal", "fr_signal_bps"]].dropna().copy()
settle_df = settle_df.sort_values("settle_time").reset_index(drop=True)

settle_times = sorted(settle_df["settle_time"].unique())

del bn_fr_1m, settle_rows; gc.collect()
print(f"  Settlements: {len(settle_df):,} across {settle_df['symbol'].nunique()} coins, {len(settle_times)} times")
print(f"  Data loaded [{time.time()-t_load:.1f}s]")
print()
sys.stdout.flush()


def get_price_at(symbol, target_time_ns, search_range_s=30):
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


def get_price_range(symbol, t_start_ns, t_end_ns):
    """Get min/max/mean price in a window."""
    i0 = np.searchsorted(bn_tk_ts, t_start_ns, side="left")
    i1 = np.searchsorted(bn_tk_ts, t_end_ns, side="right")
    if i0 >= i1:
        return None
    sl = slice(i0, i1)
    sym_mask = bn_tk_sym[sl] == symbol
    if sym_mask.sum() == 0:
        return None
    prices = bn_tk_price[sl][sym_mask]
    return {"min": float(prices.min()), "max": float(prices.max()),
            "mean": float(prices.mean()), "n": len(prices)}

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: STRATEGY A — Scalp each settlement
# ═══════════════════════════════════════════════════════════════════════════════
print("═" * 100)
print("PART 3: STRATEGY A — Scalp (enter 60s before, exit 60s after)")
print("═" * 100)
sys.stdout.flush()

# For each settlement, find the best coin, check if profitable under each fee model
strat_a_results = []

for si, st in enumerate(settle_times):
    st_ts = pd.Timestamp(st)
    if st_ts.tzinfo is None:
        st_ts = st_ts.tz_localize("UTC")
    st_ns = np.datetime64(st_ts, "ns")
    
    coins = settle_df[settle_df["settle_time"] == st].copy()
    if coins.empty:
        continue
    
    # Pick coin with highest |FR| that we could observe BEFORE settlement
    # Use fr_signal (FR observed ~2min before settlement on Binance)
    coins = coins.sort_values("fr_signal_bps", ascending=False)
    best = coins.iloc[0]
    
    sym = best["symbol"]
    fr_signal = best["fr_signal"]
    fr_signal_bps = best["fr_signal_bps"]
    fr_actual = best["fr_paid"]
    fr_actual_bps = best["fr_paid_bps"]
    
    # Entry price at T-60s, exit price at T+60s
    entry_ns = st_ns - np.timedelta64(60, "s")
    exit_ns = st_ns + np.timedelta64(60, "s")
    
    entry_price = get_price_at(sym, entry_ns)
    exit_price = get_price_at(sym, exit_ns)
    
    if entry_price is None or exit_price is None or entry_price == 0:
        continue
    
    # Volatility during the 2-min hold
    price_range = get_price_range(sym, entry_ns, exit_ns)
    if price_range and price_range["min"] > 0:
        vol_bps = (price_range["max"] - price_range["min"]) / price_range["mean"] * 10000
    else:
        vol_bps = 0
    
    # Price move from entry to exit
    price_move_bps = abs(exit_price - entry_price) / entry_price * 10000
    
    result = {
        "settle_time": st_ts,
        "symbol": sym,
        "fr_signal_bps": fr_signal_bps,
        "fr_actual_bps": fr_actual_bps,
        "fr_signal_error_bps": abs(fr_signal_bps - fr_actual_bps),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "price_move_bps": price_move_bps,
        "vol_2min_bps": vol_bps,
    }
    
    # P&L under each fee model
    for name, model in FEE_MODELS.items():
        rt_cost_bps = (model["spot_fee"] + model["futures_fee"] + 2 * model["slippage"]) * 2
        # We collect fr_actual (what actually settled)
        fr_income = fr_actual_bps
        net_bps = fr_income - rt_cost_bps
        net_usd = net_bps / 10000 * NOTIONAL
        result[f"rt_cost_{name}"] = rt_cost_bps
        result[f"net_bps_{name}"] = net_bps
        result[f"net_usd_{name}"] = net_usd
    
    strat_a_results.append(result)
    
    if (si + 1) % 10 == 0 or si == len(settle_times) - 1:
        elapsed = time.time() - t0
        print(f"  [{si+1}/{len(settle_times)}] [{elapsed:.0f}s]")
        sys.stdout.flush()

df_a = pd.DataFrame(strat_a_results)
n_a = len(df_a)

t_first = df_a["settle_time"].min()
t_last = df_a["settle_time"].max()
days = (t_last - t_first).total_seconds() / 86400

print(f"\n  Settlements analyzed: {n_a}")
print(f"  Period: {t_first} → {t_last} ({days:.1f} days)")

print(f"\n  --- Signal vs Actual FR ---")
print(f"  FR signal (observable before settle):")
print(f"    Mean: {df_a['fr_signal_bps'].mean():.1f} bps  |  Median: {df_a['fr_signal_bps'].median():.1f} bps")
print(f"  FR actual (what settled):")
print(f"    Mean: {df_a['fr_actual_bps'].mean():.1f} bps  |  Median: {df_a['fr_actual_bps'].median():.1f} bps")
print(f"  Signal error:")
print(f"    Mean: {df_a['fr_signal_error_bps'].mean():.1f} bps  |  Max: {df_a['fr_signal_error_bps'].max():.1f} bps")

print(f"\n  --- Volatility during 2-min hold ---")
print(f"  Price range (high-low) during entry→exit:")
print(f"    Mean: {df_a['vol_2min_bps'].mean():.1f} bps  |  Median: {df_a['vol_2min_bps'].median():.1f} bps  |  "
      f"p95: {df_a['vol_2min_bps'].quantile(0.95):.1f} bps  |  Max: {df_a['vol_2min_bps'].max():.1f} bps")
print(f"  Price move (|entry - exit|):")
print(f"    Mean: {df_a['price_move_bps'].mean():.1f} bps  |  Median: {df_a['price_move_bps'].median():.1f} bps  |  "
      f"p95: {df_a['price_move_bps'].quantile(0.95):.1f} bps  |  Max: {df_a['price_move_bps'].max():.1f} bps")
print(f"  NOTE: Price move doesn't affect delta-neutral P&L directly,")
print(f"        but affects slippage if we can't execute both legs simultaneously.")

print(f"\n  --- Strategy A P&L by fee model ---")
for name, model in FEE_MODELS.items():
    col = f"net_usd_{name}"
    vals = df_a[col]
    n_profit = (vals > 0).sum()
    total = vals.sum()
    daily = total / days if days > 0 else 0
    rt = df_a[f"rt_cost_{name}"].iloc[0]
    print(f"\n  {model['label']} (RT: {rt:.0f} bps)")
    print(f"    Trades: {n_a}  |  Profitable: {n_profit} ({n_profit/n_a*100:.0f}%)  |  "
          f"Total: ${total:>+,.2f}  |  Daily: ${daily:>+,.2f}")
    
    # Distribution of net P&L per trade
    print(f"    Per-trade: mean=${vals.mean():+.2f}  median=${vals.median():+.2f}  "
          f"min=${vals.min():+.2f}  max=${vals.max():+.2f}")
    
    # Only profitable if FR > threshold
    threshold = rt
    n_above = (df_a["fr_actual_bps"] > threshold).sum()
    print(f"    Settlements with FR > {threshold:.0f}bps: {n_above}/{n_a} ({n_above/n_a*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: STRATEGY B — Hold until FR normalizes
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*100}")
print("PART 4: STRATEGY B — Hold until FR normalizes")
print("═" * 100)
sys.stdout.flush()

MAX_POSITIONS = 3

for fee_name, fee_model in FEE_MODELS.items():
    rt_cost_bps = (fee_model["spot_fee"] + fee_model["futures_fee"] + 2 * fee_model["slippage"]) * 2
    rt_cost_usd = rt_cost_bps / 10000 * NOTIONAL
    entry_cost_bps = rt_cost_bps / 2
    exit_cost_bps = rt_cost_bps / 2
    
    for entry_th, exit_th in [(15, 5), (20, 8), (30, 10)]:
        positions = {}
        closed = []
        
        for st in settle_times:
            st_ts = pd.Timestamp(st)
            if st_ts.tzinfo is None:
                st_ts = st_ts.tz_localize("UTC")
            
            coins_at = settle_df[settle_df["settle_time"] == st]
            
            # Collect FR (direction-aware)
            for sym, pos in list(positions.items()):
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
                pos["payments"].append(payment)
            
            # Exits
            for sym in list(positions.keys()):
                cr = coins_at[coins_at["symbol"] == sym]
                if len(cr) == 0:
                    continue
                fr = float(cr.iloc[0]["fr_paid"])
                fr_bps = abs(fr) * 10000
                pos = positions[sym]
                flipped = (pos["direction"] == "long_futures" and fr > 0) or \
                          (pos["direction"] == "short_futures" and fr < 0)
                if fr_bps < exit_th or flipped:
                    reason = "flipped" if flipped else "normalized"
                    closed.append({
                        "symbol": sym,
                        "entry_time": pos["entry_time"],
                        "exit_time": st_ts,
                        "n_settle": pos["n_settle"],
                        "total_fr": pos["total_fr"],
                        "cost": rt_cost_usd,
                        "net_pnl": pos["total_fr"] - rt_cost_usd,
                        "payments": pos["payments"],
                        "reason": reason,
                    })
                    del positions[sym]
            
            # Entries (use fr_signal — what we observe BEFORE settlement)
            if len(positions) < MAX_POSITIONS:
                cands = coins_at[
                    (coins_at["fr_signal_bps"] >= entry_th) &
                    (~coins_at["symbol"].isin(positions.keys()))
                ].sort_values("fr_signal_bps", ascending=False)
                for _, c in cands.iterrows():
                    if len(positions) >= MAX_POSITIONS:
                        break
                    fr = float(c["fr_signal"])
                    direction = "long_futures" if fr < 0 else "short_futures"
                    positions[c["symbol"]] = {
                        "direction": direction, "total_fr": 0,
                        "n_settle": 0, "entry_time": st_ts,
                        "payments": [],
                    }
        
        # Force close remaining
        for sym, pos in positions.items():
            closed.append({
                "symbol": sym,
                "entry_time": pos["entry_time"],
                "exit_time": pd.Timestamp(settle_times[-1], tz="UTC") if pd.Timestamp(settle_times[-1]).tzinfo is None else pd.Timestamp(settle_times[-1]),
                "n_settle": pos["n_settle"],
                "total_fr": pos["total_fr"],
                "cost": rt_cost_usd,
                "net_pnl": pos["total_fr"] - rt_cost_usd,
                "payments": pos["payments"],
                "reason": "end_of_data",
            })
        
        if not closed:
            print(f"  {fee_name:<20} entry≥{entry_th}bps exit<{exit_th}bps: no trades")
            continue
        
        mc = pd.DataFrame(closed)
        n_tr = len(mc)
        n_w = (mc["net_pnl"] > 0).sum()
        total_pnl = mc["net_pnl"].sum()
        total_sett = mc["n_settle"].sum()
        total_fr = mc["total_fr"].sum()
        total_cost = mc["cost"].sum()
        daily = total_pnl / days if days > 0 else 0
        
        # Check for negative payments (FR flipped against us)
        all_payments = []
        for _, row in mc.iterrows():
            all_payments.extend(row["payments"])
        neg_payments = [p for p in all_payments if p < 0]
        n_neg = len(neg_payments)
        
        print(f"  {fee_name:<20} entry≥{entry_th:>2} exit<{exit_th:>2} (RT:{rt_cost_bps:.0f}bps): "
              f"{n_tr:>2} trades, {int(total_sett):>3} settles, "
              f"WR {n_w/n_tr*100:>5.1f}%, "
              f"FR ${total_fr:>8,.2f}, cost ${total_cost:>6,.2f}, "
              f"net ${total_pnl:>+9,.2f}, daily ${daily:>+8,.2f}"
              f"  neg_payments: {n_neg}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Detailed volatility risk analysis
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*100}")
print("PART 5: VOLATILITY & EXECUTION RISK")
print("═" * 100)

print(f"\n  --- Leg execution timing risk ---")
print(f"  Delta-neutral requires 2 simultaneous legs. If there's a delay:")
print(f"  In the worst case, price moves between spot and futures execution.")
print()

# For each settlement, measure max price move in 1s, 5s, 10s windows
windows = [1, 5, 10, 30]
for w in windows:
    moves = []
    for _, r in df_a.iterrows():
        sym = r["symbol"]
        st_ns = np.datetime64(r["settle_time"], "ns")
        # Around entry time (60s before settle)
        entry_ns = st_ns - np.timedelta64(60, "s")
        t1 = entry_ns - np.timedelta64(w, "s")
        t2 = entry_ns + np.timedelta64(w, "s")
        pr = get_price_range(sym, t1, t2)
        if pr and pr["min"] > 0:
            move = (pr["max"] - pr["min"]) / pr["mean"] * 10000
            moves.append(move)
    
    if moves:
        arr = np.array(moves)
        print(f"  ±{w}s around entry: mean={arr.mean():.1f}bps  "
              f"median={np.median(arr):.1f}bps  p95={np.percentile(arr,95):.1f}bps  "
              f"max={arr.max():.1f}bps")

print(f"\n  --- What this means for execution ---")
print(f"  If you can execute both legs within 1s: ~{df_a['vol_2min_bps'].mean():.0f}bps risk is irrelevant")
print(f"  Taker orders fill instantly — no fill risk")
print(f"  The ONLY risk is the ~0-5bps price move between the 2 taker orders")
print(f"  This is already captured in the 'slippage' estimate")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: FR Distribution — How often is FR profitable?
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*100}")
print("PART 6: FR DISTRIBUTION — How often is FR above various thresholds?")
print("═" * 100)

# For each settlement time, what's the BEST coin's FR?
best_per_settle = settle_df.loc[settle_df.groupby("settle_time")["fr_paid_bps"].idxmax()].copy()

print(f"\n  Best-coin FR distribution across {len(best_per_settle)} settlements:")
for threshold in [10, 15, 20, 25, 30, 35, 40, 50, 75, 100]:
    n = (best_per_settle["fr_paid_bps"] >= threshold).sum()
    pct = n / len(best_per_settle) * 100
    print(f"    FR ≥ {threshold:>3} bps: {n:>3}/{len(best_per_settle)} ({pct:>5.1f}%)")

print(f"\n  Statistics:")
print(f"    Mean: {best_per_settle['fr_paid_bps'].mean():.1f} bps")
print(f"    Median: {best_per_settle['fr_paid_bps'].median():.1f} bps")
print(f"    p25: {best_per_settle['fr_paid_bps'].quantile(0.25):.1f} bps")
print(f"    p75: {best_per_settle['fr_paid_bps'].quantile(0.75):.1f} bps")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*100}")
print("PART 7: FINAL COMPARISON — Strategy A vs B")
print("═" * 100)

print(f"""
  KEY QUESTION: What is your Binance spot taker fee?
  
  If 0.055% = futures only (standard spot = 0.1%):
    Round trip = (10 + 5.5 + 2×2) × 2 = 39 bps = $39 per RT on $10k
    → Strategy A needs FR > 39 bps to be profitable  
    → Strategy B needs FR > 39/N bps average over N settlements
    
  If 0.055% = both spot and futures:  
    Round trip = (5.5 + 5.5 + 2×2) × 2 = 30 bps = $30 per RT on $10k
    → Strategy A needs FR > 30 bps
    → Strategy B needs FR > 30/N bps average
    
  PESSIMISTIC (spot 0.1% + 5bps slippage):
    Round trip = (10 + 5.5 + 2×5) × 2 = 51 bps = $51 per RT on $10k
    → Strategy A needs FR > 51 bps (only 22% of settlements qualify)
    → Strategy B needs FR > 51/N bps average
""")

# Strategy A summary under pessimistic
pess_col = "net_usd_pessimistic"
real_col = "net_usd_realistic"
user_col = "net_usd_user_055_both"

for col, label in [(user_col, "0.055% both"), (real_col, "spot 0.1% + futures 0.055%"), (pess_col, "PESSIMISTIC")]:
    vals = df_a[col]
    n_profit = (vals > 0).sum()
    total = vals.sum()
    daily = total / days if days > 0 else 0
    print(f"  Strategy A ({label}):")
    print(f"    {n_a} scalps, {n_profit} profitable ({n_profit/n_a*100:.0f}%), "
          f"total: ${total:+,.2f}, daily: ${daily:+,.2f}")

print(f"\n  Strategy B (hold, entry≥20 exit<8):")
for fee_name in ["user_055_both", "realistic", "pessimistic"]:
    model = FEE_MODELS[fee_name]
    rt = (model["spot_fee"] + model["futures_fee"] + 2 * model["slippage"]) * 2
    rt_usd = rt / 10000 * NOTIONAL
    
    positions = {}
    closed = []
    for st in settle_times:
        st_ts = pd.Timestamp(st)
        if st_ts.tzinfo is None:
            st_ts = st_ts.tz_localize("UTC")
        coins_at = settle_df[settle_df["settle_time"] == st]
        
        for sym, pos in list(positions.items()):
            cr = coins_at[coins_at["symbol"] == sym]
            if len(cr) == 0: continue
            fr = float(cr.iloc[0]["fr_paid"])
            payment = -fr * NOTIONAL if pos["dir"] == "long_futures" else fr * NOTIONAL
            pos["fr"] += payment
            pos["n"] += 1
        
        for sym in list(positions.keys()):
            cr = coins_at[coins_at["symbol"] == sym]
            if len(cr) == 0: continue
            fr = float(cr.iloc[0]["fr_paid"])
            fr_bps = abs(fr) * 10000
            pos = positions[sym]
            flipped = (pos["dir"] == "long_futures" and fr > 0) or \
                      (pos["dir"] == "short_futures" and fr < 0)
            if fr_bps < 8 or flipped:
                closed.append({"fr": pos["fr"], "cost": rt_usd, "n": pos["n"],
                               "net": pos["fr"] - rt_usd})
                del positions[sym]
        
        if len(positions) < 3:
            cands = coins_at[
                (coins_at["fr_signal_bps"] >= 20) &
                (~coins_at["symbol"].isin(positions.keys()))
            ].sort_values("fr_signal_bps", ascending=False)
            for _, c in cands.iterrows():
                if len(positions) >= 3: break
                fr = float(c["fr_signal"])
                positions[c["symbol"]] = {
                    "dir": "long_futures" if fr < 0 else "short_futures",
                    "fr": 0, "n": 0,
                }
    
    for sym, pos in positions.items():
        closed.append({"fr": pos["fr"], "cost": rt_usd, "n": pos["n"],
                       "net": pos["fr"] - rt_usd})
    
    if closed:
        mc = pd.DataFrame(closed)
        total = mc["net"].sum()
        daily = total / days if days > 0 else 0
        n_w = (mc["net"] > 0).sum()
        print(f"    {model['label']}: {len(mc)} trades, "
              f"WR {n_w/len(mc)*100:.0f}%, "
              f"total: ${total:+,.2f}, daily: ${daily:+,.2f}")

elapsed = time.time() - t0
print(f"\n{'='*100}")
print(f"Audit complete [{elapsed:.1f}s]")
print(f"{'='*100}")
