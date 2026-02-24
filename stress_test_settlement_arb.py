#!/usr/bin/env python3
"""
Stress Test: Settlement Arbitrage Strategy

Measures actual risk factors from real data:
  1. Price impact — how much does price move in 5s windows around entry/exit
  2. Orderbook capacity — how much size is available in the ob200 book near settlement
  3. Maker fill probability — how often does price touch our limit order level
  4. Realistic slippage from orderbook depth
  5. Combined stress P&L with worst-case assumptions

Memory-efficient: processes one settlement at a time, loads ob200 per-day.
"""
import sys
import time
import gc
import json
import re
import zipfile
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

DATA = Path("data_all")
OB_DATA = Path("data")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
NOTIONAL = 10_000
ENTRY_BEFORE_S = 60
EXIT_AFTER_S = 60
WINDOW_S = 900  # 15 min each side

# Fee model (bps)
SPOT_MAKER_FEE = 1.0
SPOT_TAKER_FEE = 5.0
FUTURES_TAKER_FEE = 4.5
SLIPPAGE_BPS = 1.0

# Scenarios
SCENARIOS = {
    "optimistic": {
        "spot_fee": SPOT_MAKER_FEE,
        "futures_fee": FUTURES_TAKER_FEE,
        "slippage": 0.5,
        "label": "Maker spot, low slippage",
    },
    "base": {
        "spot_fee": SPOT_MAKER_FEE,
        "futures_fee": FUTURES_TAKER_FEE,
        "slippage": SLIPPAGE_BPS,
        "label": "Maker spot, normal slippage",
    },
    "taker": {
        "spot_fee": SPOT_TAKER_FEE,
        "futures_fee": FUTURES_TAKER_FEE,
        "slippage": SLIPPAGE_BPS,
        "label": "Taker both sides",
    },
    "worst": {
        "spot_fee": SPOT_TAKER_FEE,
        "futures_fee": FUTURES_TAKER_FEE,
        "slippage": 3.0,
        "label": "Taker + high slippage",
    },
}

TARGET_SYMBOLS = ["POWERUSDT", "AWEUSDT", "LAUSDT", "BELUSDT", "AGLDUSDT", "AXSUSDT"]

print("=" * 90)
print("STRESS TEST: Settlement Arbitrage Strategy")
print("=" * 90)

t_global = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Detect settlements (reuse from backtest)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPHASE 1: Detecting settlement events...")
sys.stdout.flush()

t1 = time.time()
bn_fr = pd.read_parquet(DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "nextFundingTime"])
print(f"  Loaded Binance FR: {len(bn_fr):,} rows [{time.time()-t1:.1f}s]")
sys.stdout.flush()

bn_fr["ts_1m"] = bn_fr["ts"].dt.floor("1min")
bn_fr_1m = bn_fr.groupby(["ts_1m", "symbol"]).agg(
    fr=("lastFundingRate", "last"),
    nft=("nextFundingTime", "last"),
).reset_index()
del bn_fr; gc.collect()

bn_fr_1m = bn_fr_1m.sort_values(["symbol", "ts_1m"])
bn_fr_1m["nft_prev"] = bn_fr_1m.groupby("symbol")["nft"].shift(1)
bn_fr_1m["is_settle"] = (bn_fr_1m["nft"] != bn_fr_1m["nft_prev"]) & bn_fr_1m["nft_prev"].notna()

settle_idx = bn_fr_1m[bn_fr_1m["is_settle"]].index
pre = bn_fr_1m.loc[settle_idx - 1].copy()
pre["settle_time"] = bn_fr_1m.loc[settle_idx, "ts_1m"].values
pre["fr_bps"] = pre["fr"].abs() * 10000

# Filter to target symbols and |FR| > 10 bps
pre = pre[pre["symbol"].isin(TARGET_SYMBOLS)]
best = pre.loc[pre.groupby("settle_time")["fr_bps"].idxmax()].copy()
best = best.sort_values("settle_time").reset_index(drop=True)

# Keep all with FR > 10 bps (we'll test profitability under each scenario)
tradeable = best[best["fr_bps"] >= 10].reset_index(drop=True)

del bn_fr_1m, pre; gc.collect()
print(f"  Tradeable settlements (FR>10bps, target symbols): {len(tradeable)}")
print(f"  Phase 1 done [{time.time()-t1:.1f}s]")
sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Load tick data arrays (same as backtest)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPHASE 2: Loading tick data...")
sys.stdout.flush()
t2 = time.time()

_bn_fr = pd.read_parquet(DATA / "binance" / "fundingRate.parquet",
    columns=["ts", "symbol", "lastFundingRate", "markPrice"])
bn_fr_ts = _bn_fr["ts"].values
bn_fr_sym = _bn_fr["symbol"].values
bn_fr_rate = _bn_fr["lastFundingRate"].values
bn_fr_mark = _bn_fr["markPrice"].values
del _bn_fr; gc.collect()
print(f"  Binance FR: {len(bn_fr_ts):,} [{time.time()-t2:.1f}s]")
sys.stdout.flush()

_bn_tk = pd.read_parquet(DATA / "binance" / "ticker.parquet",
    columns=["ts", "symbol", "lastPrice"])
bn_tk_ts = _bn_tk["ts"].values
bn_tk_sym = _bn_tk["symbol"].values
bn_tk_price = _bn_tk["lastPrice"].values
del _bn_tk; gc.collect()
print(f"  Binance ticker: {len(bn_tk_ts):,} [{time.time()-t2:.1f}s]")
sys.stdout.flush()

_bb_tk = pd.read_parquet(DATA / "bybit" / "ticker.parquet",
    columns=["ts", "symbol", "fundingRate", "bid1Price", "ask1Price", "lastPrice"])
bb_tk_ts = _bb_tk["ts"].values
bb_tk_sym = _bb_tk["symbol"].values
bb_tk_fr = _bb_tk["fundingRate"].values
bb_tk_bid = _bb_tk["bid1Price"].values
bb_tk_ask = _bb_tk["ask1Price"].values
bb_tk_last = _bb_tk["lastPrice"].values
del _bb_tk; gc.collect()
print(f"  Bybit ticker: {len(bb_tk_ts):,} [{time.time()-t2:.1f}s]")
sys.stdout.flush()

print(f"  Data loaded [{time.time()-t2:.1f}s]")
print()


def get_window(ts_arr, sym_arr, arrays_dict, symbol, t_start_ns, t_end_ns):
    """Extract time+symbol window from numpy arrays."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Orderbook depth extraction around settlements
# ═══════════════════════════════════════════════════════════════════════════════
print("PHASE 3: Extracting orderbook depth around settlements")
print("─" * 90)
sys.stdout.flush()


def load_ob_window(symbol, settle_time, window_s=300):
    """Load ob200 data for a symbol around a settlement time.
    
    Returns list of (ts_ms, bids_dict, asks_dict) tuples within window,
    or None if no data available.
    """
    date_str = settle_time.strftime("%Y-%m-%d")
    zip_path = OB_DATA / symbol / "bybit" / "orderbook_futures" / f"{date_str}_{symbol}_ob200.data.zip"
    
    if not zip_path.exists():
        return None
    
    settle_ts_ms = int(settle_time.timestamp() * 1000)
    t_start_ms = settle_ts_ms - window_s * 1000
    t_end_ms = settle_ts_ms + window_s * 1000
    
    # Parse the ob200 zip — reconstruct book state, sample around settlement
    book_bids = {}
    book_asks = {}
    initialized = False
    snapshots = []
    
    try:
        with zipfile.ZipFile(zip_path) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as f:
                buffer = ""
                chunk_size = 20 * 1024 * 1024
                
                while True:
                    raw = f.read(chunk_size)
                    if not raw:
                        break
                    buffer += raw.decode("utf-8", errors="replace")
                    parts = re.split(r"\}\s*\{", buffer)
                    
                    for i, part in enumerate(parts[:-1]):
                        if i == 0:
                            text = part + "}"
                        else:
                            text = "{" + part + "}"
                        
                        try:
                            obj = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        
                        ts_ms = obj.get("ts", 0)
                        msg_type = obj.get("type", "")
                        data = obj.get("data", {})
                        bids = data.get("b", [])
                        asks = data.get("a", [])
                        
                        if msg_type == "snapshot":
                            book_bids = {}
                            book_asks = {}
                            for p_s, s_s in bids:
                                p, s = float(p_s), float(s_s)
                                if s > 0: book_bids[p] = s
                            for p_s, s_s in asks:
                                p, s = float(p_s), float(s_s)
                                if s > 0: book_asks[p] = s
                            initialized = True
                        elif msg_type == "delta" and initialized:
                            for p_s, s_s in bids:
                                p, s = float(p_s), float(s_s)
                                if s == 0: book_bids.pop(p, None)
                                else: book_bids[p] = s
                            for p_s, s_s in asks:
                                p, s = float(p_s), float(s_s)
                                if s == 0: book_asks.pop(p, None)
                                else: book_asks[p] = s
                        
                        # Only save snapshots in our window
                        if initialized and t_start_ms <= ts_ms <= t_end_ms:
                            # Sample every ~1 second (avoid storing every delta)
                            if not snapshots or ts_ms - snapshots[-1][0] >= 1000:
                                snapshots.append((ts_ms, dict(book_bids), dict(book_asks)))
                        
                        # Early exit if we're past our window
                        if ts_ms > t_end_ms + 60000:
                            return snapshots
                    
                    buffer = "{" + parts[-1] if len(parts) > 1 else parts[-1]
    except Exception as e:
        print(f"    WARNING: Error reading {zip_path}: {e}")
        return None
    
    return snapshots if snapshots else None


def compute_ob_stats(snapshots, notional_usd):
    """Compute orderbook statistics from snapshots.
    
    Returns dict with:
      - spread_bps: mean/min/max spread
      - depth_usd_Nbps: available USD depth at N bps from mid
      - fill_price_impact_bps: price impact to fill $notional_usd
      - available_size_contracts: total size near best bid/ask
    """
    spreads = []
    depths_5bps = []
    depths_10bps = []
    depths_25bps = []
    fill_impacts = []
    best_bid_sizes = []
    best_ask_sizes = []
    
    for ts_ms, bids, asks in snapshots:
        if not bids or not asks:
            continue
        
        bid_prices = sorted(bids.keys(), reverse=True)
        ask_prices = sorted(asks.keys())
        
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid = (best_bid + best_ask) / 2
        
        if mid <= 0:
            continue
        
        spread_bps = (best_ask - best_bid) / mid * 10000
        spreads.append(spread_bps)
        
        # Depth within N bps of mid (in USD)
        for bps_level, depth_list in [(5, depths_5bps), (10, depths_10bps), (25, depths_25bps)]:
            total_bid_usd = 0
            for p in bid_prices:
                dist_bps = (mid - p) / mid * 10000
                if dist_bps <= bps_level:
                    total_bid_usd += bids[p] * p
            total_ask_usd = 0
            for p in ask_prices:
                dist_bps = (p - mid) / mid * 10000
                if dist_bps <= bps_level:
                    total_ask_usd += asks[p] * p
            depth_list.append(total_bid_usd + total_ask_usd)
        
        # Best level sizes
        best_bid_sizes.append(bids[best_bid] * best_bid)
        best_ask_sizes.append(asks[best_ask] * best_ask)
        
        # Price impact to fill $notional_usd on ask side (buying)
        remaining = notional_usd
        fill_cost = 0
        for p in ask_prices:
            size_usd = asks[p] * p
            if size_usd >= remaining:
                fill_cost += remaining
                remaining = 0
                break
            else:
                fill_cost += size_usd
                remaining -= size_usd
        
        if remaining == 0:
            vwap = fill_cost / notional_usd * (notional_usd / fill_cost)  # this is just 1
            # Proper VWAP
            filled = 0
            cost = 0
            for p in ask_prices:
                size_usd = asks[p] * p
                take = min(size_usd, notional_usd - filled)
                cost += take / p * p  # this is just take
                filled += take
                if filled >= notional_usd:
                    break
            # Actually calculate properly
            filled_qty = 0
            cost_total = 0
            for p in ask_prices:
                size_contracts = asks[p]
                take_usd = min(size_contracts * p, notional_usd - cost_total)
                take_contracts = take_usd / p
                cost_total += take_contracts * p
                filled_qty += take_contracts
                if cost_total >= notional_usd:
                    break
            if filled_qty > 0:
                vwap_price = cost_total / filled_qty
                impact_bps = (vwap_price - best_ask) / mid * 10000
                fill_impacts.append(impact_bps)
        else:
            fill_impacts.append(float("nan"))  # not enough liquidity
    
    if not spreads:
        return None
    
    return {
        "n_snapshots": len(snapshots),
        "spread_mean_bps": np.nanmean(spreads),
        "spread_min_bps": np.nanmin(spreads),
        "spread_max_bps": np.nanmax(spreads),
        "spread_p95_bps": np.nanpercentile(spreads, 95),
        "depth_5bps_mean_usd": np.nanmean(depths_5bps),
        "depth_10bps_mean_usd": np.nanmean(depths_10bps),
        "depth_25bps_mean_usd": np.nanmean(depths_25bps),
        "best_bid_size_mean_usd": np.nanmean(best_bid_sizes),
        "best_ask_size_mean_usd": np.nanmean(best_ask_sizes),
        "fill_impact_mean_bps": np.nanmean(fill_impacts),
        "fill_impact_max_bps": np.nanmax(fill_impacts) if fill_impacts else float("nan"),
        "fill_impact_p95_bps": np.nanpercentile(fill_impacts, 95) if fill_impacts else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Run comprehensive analysis per settlement
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPHASE 4: Analyzing each settlement")
print("─" * 90)
sys.stdout.flush()

results = []
n_total = len(tradeable)
t_phase4 = time.time()

for idx, row in tradeable.iterrows():
    t_iter = time.time()
    settle_time = pd.Timestamp(row["settle_time"])
    if settle_time.tzinfo is None:
        settle_time = settle_time.tz_localize("UTC")
    symbol = row["symbol"]
    pre_fr = row["fr"]
    pre_fr_bps = row["fr_bps"]
    
    t_start = settle_time - timedelta(seconds=WINDOW_S)
    t_end = settle_time + timedelta(seconds=WINDOW_S)
    t_entry = settle_time - timedelta(seconds=ENTRY_BEFORE_S)
    t_exit = settle_time + timedelta(seconds=EXIT_AFTER_S)
    
    t_start_ns = np.datetime64(t_start, "ns")
    t_end_ns = np.datetime64(t_end, "ns")
    t_entry_ns = np.datetime64(t_entry, "ns")
    t_exit_ns = np.datetime64(t_exit, "ns")
    
    # ── 1. Price impact analysis (from 5s tick data) ──
    bn_tk_win = get_window(bn_tk_ts, bn_tk_sym,
                           {"price": bn_tk_price},
                           symbol, t_start_ns, t_end_ns)
    
    bb_win = get_window(bb_tk_ts, bb_tk_sym,
                        {"fr": bb_tk_fr, "bid": bb_tk_bid, "ask": bb_tk_ask, "last": bb_tk_last},
                        symbol, t_start_ns, t_end_ns)
    
    # Get Binance FR
    bn_fr_win = get_window(bn_fr_ts, bn_fr_sym,
                           {"fr": bn_fr_rate, "mark": bn_fr_mark},
                           symbol, t_start_ns, t_end_ns)
    
    entry_price = None
    exit_price = None
    entry_fr = None
    
    if bn_tk_win is not None and len(bn_tk_win["ts"]) > 0:
        entry_diffs = np.abs(bn_tk_win["ts"] - t_entry_ns)
        entry_idx = np.argmin(entry_diffs)
        entry_price = bn_tk_win["price"][entry_idx]
        
        exit_diffs = np.abs(bn_tk_win["ts"] - t_exit_ns)
        exit_idx = np.argmin(exit_diffs)
        exit_price = bn_tk_win["price"][exit_idx]
    
    if bn_fr_win is not None and len(bn_fr_win["ts"]) > 0:
        entry_fr_diffs = np.abs(bn_fr_win["ts"] - t_entry_ns)
        entry_fr = bn_fr_win["fr"][np.argmin(entry_fr_diffs)]
    
    if entry_price is None or exit_price is None or entry_price == 0:
        elapsed = time.time() - t_phase4
        eta = elapsed / max(len(results) + 1, 1) * (n_total - len(results) - 1)
        print(f"  [{len(results)+1:3d}/{n_total}] {settle_time} {symbol:<14} SKIP (no price) [{elapsed:.0f}s, ETA {eta:.0f}s]")
        sys.stdout.flush()
        continue
    
    # Price changes in different windows around entry/exit
    # 5s, 10s, 30s, 60s windows
    price_impacts = {}
    for label, offset_s in [("5s", 5), ("10s", 10), ("30s", 30), ("60s", 60)]:
        # Around entry time
        t_pre = t_entry_ns - np.timedelta64(offset_s, "s")
        t_post = t_entry_ns + np.timedelta64(offset_s, "s")
        
        mask_pre = (bn_tk_win["ts"] >= t_pre) & (bn_tk_win["ts"] <= t_entry_ns)
        mask_post = (bn_tk_win["ts"] >= t_entry_ns) & (bn_tk_win["ts"] <= t_post)
        
        if mask_pre.sum() > 0 and mask_post.sum() > 0:
            p_pre = bn_tk_win["price"][mask_pre]
            p_post = bn_tk_win["price"][mask_post]
            # Max adverse move in the window
            changes = np.abs(np.diff(np.concatenate([p_pre, p_post]))) / entry_price * 10000
            price_impacts[f"entry_{label}_max_bps"] = float(np.max(changes)) if len(changes) > 0 else 0
            price_impacts[f"entry_{label}_mean_bps"] = float(np.mean(changes)) if len(changes) > 0 else 0
            # Total move from start to end of window
            total_move = abs(p_post[-1] - p_pre[0]) / entry_price * 10000
            price_impacts[f"entry_{label}_total_bps"] = float(total_move)
        
        # Around exit time
        t_pre = t_exit_ns - np.timedelta64(offset_s, "s")
        t_post = t_exit_ns + np.timedelta64(offset_s, "s")
        
        mask_pre = (bn_tk_win["ts"] >= t_pre) & (bn_tk_win["ts"] <= t_exit_ns)
        mask_post = (bn_tk_win["ts"] >= t_exit_ns) & (bn_tk_win["ts"] <= t_post)
        
        if mask_pre.sum() > 0 and mask_post.sum() > 0:
            p_pre = bn_tk_win["price"][mask_pre]
            p_post = bn_tk_win["price"][mask_post]
            changes = np.abs(np.diff(np.concatenate([p_pre, p_post]))) / exit_price * 10000
            price_impacts[f"exit_{label}_max_bps"] = float(np.max(changes)) if len(changes) > 0 else 0
            price_impacts[f"exit_{label}_mean_bps"] = float(np.mean(changes)) if len(changes) > 0 else 0
            total_move = abs(p_post[-1] - p_pre[0]) / exit_price * 10000
            price_impacts[f"exit_{label}_total_bps"] = float(total_move)
    
    # Price move from entry to exit (the hedge residual)
    hedge_residual_bps = (exit_price - entry_price) / entry_price * 10000
    
    # ── 2. Maker fill probability ──
    # If we place a maker limit buy at best_bid at entry time, does the ask come down to fill us?
    # We check: in the 60s before settlement, how often does price touch a level N bps below mid
    maker_fill_probs = {}
    if bb_win is not None and len(bb_win["ts"]) > 0:
        # Use Bybit bid/ask for maker analysis
        entry_mask = (bb_win["ts"] >= t_entry_ns - np.timedelta64(10, "s")) & \
                     (bb_win["ts"] <= t_entry_ns + np.timedelta64(10, "s"))
        if entry_mask.sum() > 0:
            bb_bid_at_entry = bb_win["bid"][entry_mask][0]
            bb_ask_at_entry = bb_win["ask"][entry_mask][0]
            bb_mid_at_entry = (bb_bid_at_entry + bb_ask_at_entry) / 2
            
            # Check: in the 60s before settlement, what fraction of time does
            # ask price stay at or above our limit buy (best_bid)?
            pre_settle_mask = (bb_win["ts"] >= t_entry_ns) & (bb_win["ts"] <= np.datetime64(settle_time, "ns"))
            if pre_settle_mask.sum() > 0:
                asks_in_window = bb_win["ask"][pre_settle_mask]
                # Maker buy at best_bid_at_entry: filled if ask drops to our level
                # With maker, we post at best_bid and wait
                maker_fill_probs["bid_fill_pct"] = float((asks_in_window <= bb_bid_at_entry).sum() / len(asks_in_window) * 100)
                # More aggressive: post at mid
                maker_fill_probs["mid_fill_pct"] = float((asks_in_window <= bb_mid_at_entry).sum() / len(asks_in_window) * 100)
                # Even more: post 1 bps above mid
                aggressive_price = bb_mid_at_entry * (1 + 1 / 10000)
                maker_fill_probs["aggressive_fill_pct"] = float((asks_in_window <= aggressive_price).sum() / len(asks_in_window) * 100)
    
    # ── 3. Bybit bid-ask spread ──
    bb_spread_entry = None
    bb_spread_settle = None
    if bb_win is not None and len(bb_win["ts"]) > 0:
        entry_mask = np.abs(bb_win["ts"] - t_entry_ns) < np.timedelta64(10, "s")
        if entry_mask.sum() > 0:
            bids = bb_win["bid"][entry_mask]
            asks = bb_win["ask"][entry_mask]
            valid = (bids > 0) & (asks > 0)
            if valid.sum() > 0:
                mids = (bids[valid] + asks[valid]) / 2
                bb_spread_entry = float(np.mean((asks[valid] - bids[valid]) / mids * 10000))
        
        settle_ns = np.datetime64(settle_time, "ns")
        settle_mask = np.abs(bb_win["ts"] - settle_ns) < np.timedelta64(10, "s")
        if settle_mask.sum() > 0:
            bids = bb_win["bid"][settle_mask]
            asks = bb_win["ask"][settle_mask]
            valid = (bids > 0) & (asks > 0)
            if valid.sum() > 0:
                mids = (bids[valid] + asks[valid]) / 2
                bb_spread_settle = float(np.mean((asks[valid] - bids[valid]) / mids * 10000))
    
    # ── 4. Orderbook depth (from ob200) ──
    ob_stats = None
    ob_loaded = False
    try:
        ob_snaps = load_ob_window(symbol, settle_time, window_s=120)
        if ob_snaps:
            ob_stats = compute_ob_stats(ob_snaps, NOTIONAL)
            ob_loaded = True
    except Exception as e:
        pass
    
    # ── 5. Build result ──
    fr_value = entry_fr if entry_fr is not None else pre_fr
    fr_payment_usd = abs(fr_value) * NOTIONAL
    
    result = {
        "settle_time": settle_time,
        "symbol": symbol,
        "fr_bps": abs(fr_value) * 10000,
        "fr_payment_usd": fr_payment_usd,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "hedge_residual_bps": hedge_residual_bps,
        "bb_spread_entry_bps": bb_spread_entry,
        "bb_spread_settle_bps": bb_spread_settle,
        "ob_loaded": ob_loaded,
    }
    result.update(price_impacts)
    result.update(maker_fill_probs)
    if ob_stats:
        for k, v in ob_stats.items():
            result[f"ob_{k}"] = v
    
    # Compute scenario P&Ls
    for sc_name, sc in SCENARIOS.items():
        cost_bps = (sc["spot_fee"] + sc["futures_fee"] + 2 * sc["slippage"]) * 2
        cost_usd = cost_bps / 10000 * NOTIONAL
        net = fr_payment_usd - cost_usd
        result[f"pnl_{sc_name}"] = net
    
    results.append(result)
    
    elapsed = time.time() - t_phase4
    eta = elapsed / len(results) * (n_total - len(results))
    ob_flag = "OB" if ob_loaded else "  "
    print(f"  [{len(results):3d}/{n_total}] {settle_time} {symbol:<14} "
          f"FR={fr_value*100:+.4f}%  hedge_res={hedge_residual_bps:+.1f}bps  "
          f"bb_sprd={bb_spread_entry if bb_spread_entry else 0:.1f}bps  {ob_flag}  "
          f"[{elapsed:.0f}s, ETA {eta:.0f}s]")
    sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Aggregate results and print report
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 90)
print("STRESS TEST RESULTS")
print("=" * 90)

if not results:
    print("No trades analyzed!")
    sys.exit(0)

df = pd.DataFrame(results)
n = len(df)

# ── A. Price Impact Analysis ──
print("\n" + "─" * 90)
print("A. PRICE IMPACT (how much does price move in N-second windows)")
print("─" * 90)

for window in ["5s", "10s", "30s", "60s"]:
    cols_entry = [c for c in df.columns if c.startswith(f"entry_{window}")]
    cols_exit = [c for c in df.columns if c.startswith(f"exit_{window}")]
    
    if not cols_entry:
        continue
    
    total_col = f"entry_{window}_total_bps"
    max_col = f"entry_{window}_max_bps"
    
    if total_col in df.columns:
        vals = df[total_col].dropna()
        print(f"\n  Entry ±{window} total price move:")
        print(f"    Mean: {vals.mean():6.2f} bps  |  Median: {vals.median():6.2f} bps  |  "
              f"p95: {vals.quantile(0.95):6.2f} bps  |  Max: {vals.max():6.2f} bps")
    
    total_col_exit = f"exit_{window}_total_bps"
    if total_col_exit in df.columns:
        vals = df[total_col_exit].dropna()
        print(f"  Exit  ±{window} total price move:")
        print(f"    Mean: {vals.mean():6.2f} bps  |  Median: {vals.median():6.2f} bps  |  "
              f"p95: {vals.quantile(0.95):6.2f} bps  |  Max: {vals.max():6.2f} bps")

print(f"\n  Hedge residual (entry→exit price change, ~2min hold):")
vals = df["hedge_residual_bps"].dropna()
print(f"    Mean: {vals.mean():+6.2f} bps  |  Median: {vals.median():+6.2f} bps  |  "
      f"Std: {vals.std():6.2f} bps  |  Min: {vals.min():+6.2f}  |  Max: {vals.max():+6.2f}")

# ── B. Bybit Bid-Ask Spread ──
print("\n" + "─" * 90)
print("B. BYBIT BID-ASK SPREAD (actual from tick data)")
print("─" * 90)

if "bb_spread_entry_bps" in df.columns:
    vals = df["bb_spread_entry_bps"].dropna()
    print(f"  At entry (60s before settlement):")
    print(f"    Mean: {vals.mean():6.2f} bps  |  Median: {vals.median():6.2f} bps  |  "
          f"p95: {vals.quantile(0.95):6.2f} bps  |  Max: {vals.max():6.2f} bps")

if "bb_spread_settle_bps" in df.columns:
    vals = df["bb_spread_settle_bps"].dropna()
    print(f"  At settlement:")
    print(f"    Mean: {vals.mean():6.2f} bps  |  Median: {vals.median():6.2f} bps  |  "
          f"p95: {vals.quantile(0.95):6.2f} bps  |  Max: {vals.max():6.2f} bps")

# ── C. Maker Fill Probability ──
print("\n" + "─" * 90)
print("C. MAKER FILL PROBABILITY (60s before settlement)")
print("─" * 90)

for fill_col, label in [("bid_fill_pct", "At best bid"), ("mid_fill_pct", "At mid price"), ("aggressive_fill_pct", "1 bps above mid")]:
    if fill_col in df.columns:
        vals = df[fill_col].dropna()
        if len(vals) > 0:
            print(f"  {label}:")
            print(f"    Mean: {vals.mean():5.1f}%  |  Median: {vals.median():5.1f}%  |  "
                  f"Min: {vals.min():5.1f}%  |  Max: {vals.max():5.1f}%")

# ── D. Orderbook Depth ──
print("\n" + "─" * 90)
print("D. ORDERBOOK DEPTH (from Bybit ob200, ±2min around settlement)")
print("─" * 90)

ob_df = df[df["ob_loaded"]].copy()
if len(ob_df) > 0:
    for col, label in [
        ("ob_spread_mean_bps", "OB spread (mean)"),
        ("ob_spread_p95_bps", "OB spread (p95)"),
        ("ob_depth_5bps_mean_usd", "Depth within 5 bps ($)"),
        ("ob_depth_10bps_mean_usd", "Depth within 10 bps ($)"),
        ("ob_depth_25bps_mean_usd", "Depth within 25 bps ($)"),
        ("ob_best_bid_size_mean_usd", "Best bid size ($)"),
        ("ob_best_ask_size_mean_usd", "Best ask size ($)"),
        ("ob_fill_impact_mean_bps", "Fill impact $10k (mean bps)"),
        ("ob_fill_impact_p95_bps", "Fill impact $10k (p95 bps)"),
        ("ob_fill_impact_max_bps", "Fill impact $10k (max bps)"),
    ]:
        if col in ob_df.columns:
            vals = ob_df[col].dropna()
            if len(vals) > 0:
                print(f"  {label:<35} mean: {vals.mean():>10,.2f}  |  min: {vals.min():>10,.2f}  |  max: {vals.max():>10,.2f}")
    
    # Per-symbol breakdown
    print(f"\n  Per-symbol depth (mean within 10 bps of mid):")
    if "ob_depth_10bps_mean_usd" in ob_df.columns:
        for sym in ob_df["symbol"].unique():
            sym_df = ob_df[ob_df["symbol"] == sym]
            d = sym_df["ob_depth_10bps_mean_usd"].mean()
            s = sym_df["ob_spread_mean_bps"].mean()
            fi = sym_df["ob_fill_impact_mean_bps"].mean() if "ob_fill_impact_mean_bps" in sym_df.columns else 0
            cap = d / 2  # rough: we can take half the visible depth
            print(f"    {sym:<14} depth_10bps: ${d:>10,.0f}  |  spread: {s:>5.1f} bps  |  "
                  f"fill_impact: {fi:>5.2f} bps  |  est capacity: ${cap:>10,.0f}")
else:
    print("  No orderbook data loaded")

# ── E. Scenario P&L ──
print("\n" + "─" * 90)
print("E. SCENARIO P&L COMPARISON")
print("─" * 90)

t_first = df["settle_time"].min()
t_last = df["settle_time"].max()
days = (t_last - t_first).total_seconds() / 86400

for sc_name, sc in SCENARIOS.items():
    col = f"pnl_{sc_name}"
    vals = df[col]
    n_profit = (vals > 0).sum()
    total = vals.sum()
    daily = total / days if days > 0 else 0
    cost_bps = (sc["spot_fee"] + sc["futures_fee"] + 2 * sc["slippage"]) * 2
    
    print(f"\n  {sc['label']} (RT cost: {cost_bps:.0f} bps)")
    print(f"    Trades: {n}  |  Profitable: {n_profit} ({n_profit/n*100:.0f}%)  |  "
          f"Total: ${total:>+10,.2f}  |  Daily: ${daily:>+8,.2f}")

# ── F. Worst-case stress test ──
print("\n" + "─" * 90)
print("F. WORST-CASE STRESS TEST")
print("─" * 90)

# Combine: worst price impact + worst slippage + taker fees
p5s = df.get("entry_5s_total_bps", pd.Series(dtype=float))
p_impact_p95 = p5s.quantile(0.95) if len(p5s.dropna()) > 0 else 5.0
hedge_std = df["hedge_residual_bps"].std()

print(f"  Price impact (5s window p95):     {p_impact_p95:.2f} bps")
print(f"  Hedge residual (std):             {hedge_std:.2f} bps")

worst_cost_bps = (SPOT_TAKER_FEE + FUTURES_TAKER_FEE + 2 * 3.0) * 2  # taker + 3bps slippage
print(f"  Worst-case RT cost:               {worst_cost_bps:.0f} bps")
print(f"  Total worst-case overhead:        {worst_cost_bps + p_impact_p95 + hedge_std:.1f} bps")

# How many trades still profitable under worst case?
min_fr_needed = worst_cost_bps + p_impact_p95 + hedge_std
n_still_profit = (df["fr_bps"] > min_fr_needed).sum()
print(f"\n  FR needed to be profitable:       {min_fr_needed:.1f} bps")
print(f"  Trades still profitable:          {n_still_profit}/{n} ({n_still_profit/n*100:.0f}%)")
print(f"  Worst-case total P&L:             ${df[f'pnl_worst'].sum():+,.2f}")

if days > 0:
    print(f"  Worst-case daily P&L:             ${df[f'pnl_worst'].sum()/days:+,.2f}")

# ── G. Per-trade audit ──
print("\n" + "─" * 90)
print("G. PER-TRADE DETAIL (sorted by FR)")
print("─" * 90)
print(f"  {'Time':<22} {'Symbol':<14} {'FR bps':>7} {'Hedge':>7} {'BbSprd':>7} "
      f"{'5s imp':>7} {'Optim':>8} {'Base':>8} {'Taker':>8} {'Worst':>8}")

df_sorted = df.sort_values("fr_bps", ascending=False)
for _, r in df_sorted.iterrows():
    p5s_val = r.get("entry_5s_total_bps", 0) or 0
    bb_sprd = r.get("bb_spread_entry_bps", 0) or 0
    print(f"  {str(r['settle_time']):<22} {r['symbol']:<14} {r['fr_bps']:>6.1f} "
          f"{r['hedge_residual_bps']:>+6.1f} {bb_sprd:>6.1f} {p5s_val:>6.1f} "
          f"${r['pnl_optimistic']:>+7.2f} ${r['pnl_base']:>+7.2f} "
          f"${r['pnl_taker']:>+7.2f} ${r['pnl_worst']:>+7.2f}")

elapsed = time.time() - t_global
print(f"\n{'='*90}")
print(f"Stress test complete [{elapsed:.1f}s]")
print(f"{'='*90}")
