#!/usr/bin/env python3
"""
Multi-Coin Stages 2-4 Analysis Pipeline.

For each coin's Stage-1 events:
  Stage-2: Lifecycle reconstruction (VacuumScore, RefillScore, DepthRecovery, Spread)
  Stage-3: Causal entry after Refill, returns at 5s/30s/60s/5m/15m/60m
  Stage-4: Hour horizon study (ret_15m/30m/60m, realized vol, regime persistence)

Processes ONE DAY at a time to control RAM.
Loads OB+trades for [day, day+1] (for events near midnight), processes events, frees.

Usage:
  python multicoin_analysis.py ATOMUSDT
  python multicoin_analysis.py ALL          # all 8 coins sequentially
  python multicoin_analysis.py ALL --skip-existing
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

ALL_SYMBOLS = [
    "1000RATSUSDT", "1000BONKUSDT", "1000TURBOUSDT",
    "ARBUSDT", "APTUSDT", "ATOMUSDT",
    "ARCUSDT", "AIXBTUSDT",
]

OB_LEVELS_K = 5
TICK_MS = 100

# Return horizons (seconds)
RETURN_HORIZONS = [5, 30, 60, 300, 900, 3600]
HORIZON_LABELS = ["5s", "30s", "60s", "5m", "15m", "60m"]

# Refill detection params
TH_REFILL = 0.5
STATE_HOLD_TICKS = 3
MAX_REFILL_SEARCH_S = 5.0

# Spread quality filter
MAX_SPREAD_BPS = 200  # exclude events with spread > 200 bps at t0


# -----------------------------------------------------------------------
# Data loading (streaming, minimal RAM)
# -----------------------------------------------------------------------

def load_ob_l5(filepath: Path):
    """Load OB snapshots. Returns dict of numpy arrays."""
    ts_l, bb_l, ba_l, bd_l, ad_l = [], [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            obj = json.loads(line)
            ts = obj["ts"] / 1000.0
            data = obj["data"]
            bids = data["b"][:OB_LEVELS_K]
            asks = data["a"][:OB_LEVELS_K]
            bb = float(bids[0][0]) if bids else 0.0
            ba = float(asks[0][0]) if asks else 0.0
            bid_n = sum(float(b[1]) * float(b[0]) for b in bids)
            ask_n = sum(float(a[1]) * float(a[0]) for a in asks)
            ts_l.append(ts)
            bb_l.append(bb)
            ba_l.append(ba)
            bd_l.append(bid_n)
            ad_l.append(ask_n)
    return {
        "ts": np.array(ts_l, dtype=np.float64),
        "bb": np.array(bb_l, dtype=np.float64),
        "ba": np.array(ba_l, dtype=np.float64),
        "bd": np.array(bd_l, dtype=np.float64),
        "ad": np.array(ad_l, dtype=np.float64),
    }


def load_trades(filepath: Path):
    """Load trades. Returns dict of numpy arrays."""
    ts_l, side_l, not_l, price_l = [], [], [], []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_l.append(float(row["timestamp"]))
            side_l.append(1 if row["side"] == "Buy" else -1)
            not_l.append(float(row["foreignNotional"]))
            price_l.append(float(row["price"]))
    return {
        "ts": np.array(ts_l, dtype=np.float64),
        "side": np.array(side_l, dtype=np.int8),
        "notional": np.array(not_l, dtype=np.float64),
        "price": np.array(price_l, dtype=np.float64),
    }


def load_day_data(symbol, date_str, next_date_str=None):
    """Load OB+trades for one day (+ optionally next day for boundary events).
    Returns (ob, tr) or (None, None) if files missing."""
    ob_path = DATA_DIR / symbol / f"{date_str}_orderbook.jsonl"
    tr_path = DATA_DIR / symbol / f"{date_str}_trades.csv"
    if not ob_path.exists() or not tr_path.exists():
        return None, None

    ob = load_ob_l5(ob_path)
    tr = load_trades(tr_path)

    if next_date_str:
        nob = DATA_DIR / symbol / f"{next_date_str}_orderbook.jsonl"
        ntr = DATA_DIR / symbol / f"{next_date_str}_trades.csv"
        if nob.exists() and ntr.exists():
            ob_n = load_ob_l5(nob)
            tr_n = load_trades(ntr)
            for k in ob:
                ob[k] = np.concatenate([ob[k], ob_n[k]])
            for k in tr:
                tr[k] = np.concatenate([tr[k], tr_n[k]])

    return ob, tr


# -----------------------------------------------------------------------
# OB helpers
# -----------------------------------------------------------------------

def ob_snap(ob, t):
    """Get OB state at time t. Returns dict or None if invalid."""
    idx = np.searchsorted(ob["ts"], t, side="right") - 1
    if idx < 0:
        return None
    bb = ob["bb"][idx]
    ba = ob["ba"][idx]
    if bb <= 0 or ba <= 0 or ba <= bb:
        return None
    mid = (bb + ba) / 2.0
    spread = ba - bb
    return {
        "bb": bb, "ba": ba, "mid": mid, "spread": spread,
        "bd": ob["bd"][idx], "ad": ob["ad"][idx],
        "spread_bp": spread / mid * 10000,
        "depth": ob["bd"][idx] + ob["ad"][idx],
    }


def find_pre_shock_depth(ob, t0, lookback=5.0):
    """Find valid-spread depth baseline before t0."""
    i0 = np.searchsorted(ob["ts"], t0 - lookback, side="left")
    i1 = np.searchsorted(ob["ts"], t0, side="left")
    for idx in range(i1 - 1, max(i0 - 1, -1), -1):
        sp = ob["ba"][idx] - ob["bb"][idx]
        if sp > 0 and ob["bb"][idx] > 0:
            return float(ob["bd"][idx] + ob["ad"][idx])
    return None


# -----------------------------------------------------------------------
# Stage-2: Lifecycle metrics
# -----------------------------------------------------------------------

def compute_lifecycle(ob, tr, t0, direction_sign, mid_t0):
    """Compute lifecycle metrics for one event."""
    result = {}

    # Pre-shock depth baseline
    depth_base = find_pre_shock_depth(ob, t0)
    result["depth_base"] = depth_base if depth_base else 0

    # Spread at t0
    snap0 = ob_snap(ob, t0)
    if snap0:
        result["spread_t0_bp"] = snap0["spread_bp"]
        result["depth_t0"] = snap0["depth"]
    else:
        result["spread_t0_bp"] = 0
        result["depth_t0"] = 0

    # VacuumScore: depth drop in first 500ms
    snap_05 = ob_snap(ob, t0 + 0.5)
    if snap_05 and depth_base and depth_base > 0:
        result["vacuum_score"] = 1.0 - snap_05["depth"] / depth_base
    else:
        result["vacuum_score"] = 0

    # DepthRecovery at various times
    for t_off, label in [(1, "1s"), (2, "2s"), (5, "5s"), (10, "10s"), (30, "30s")]:
        snap_t = ob_snap(ob, t0 + t_off)
        if snap_t and depth_base and depth_base > 0:
            result[f"depth_rec_{label}"] = snap_t["depth"] / depth_base
        else:
            result[f"depth_rec_{label}"] = None

    # Spread at various times
    for t_off, label in [(0.5, "500ms"), (1, "1s"), (2, "2s"), (5, "5s"), (30, "30s")]:
        snap_t = ob_snap(ob, t0 + t_off)
        if snap_t:
            result[f"spread_{label}_bp"] = snap_t["spread_bp"]
        else:
            result[f"spread_{label}_bp"] = None

    # Flow in first 1s
    tr_i0 = np.searchsorted(tr["ts"], t0, side="left")
    tr_i1 = np.searchsorted(tr["ts"], t0 + 1.0, side="left")
    if tr_i0 < tr_i1:
        sides = tr["side"][tr_i0:tr_i1]
        nots = tr["notional"][tr_i0:tr_i1]
        buy_vol = float(nots[sides == 1].sum())
        sell_vol = float(nots[sides == -1].sum())
        total = buy_vol + sell_vol
        result["imbalance_1s"] = (buy_vol - sell_vol) / total if total > 0 else 0
        result["flow_1s"] = total
    else:
        result["imbalance_1s"] = 0
        result["flow_1s"] = 0

    return result


# -----------------------------------------------------------------------
# Refill detection (causal)
# -----------------------------------------------------------------------

def detect_refill(ob, tr, t0, direction_sign):
    """Find t_refill causally. Returns t_refill or None."""
    depth_base = find_pre_shock_depth(ob, t0)
    if depth_base is None or depth_base < 100:
        return None

    hold = 0
    for k in range(0, int(MAX_REFILL_SEARCH_S / 0.1) + 1):
        t_k = t0 + k * 0.1
        idx = np.searchsorted(ob["ts"], t_k, side="right") - 1
        if idx < 0:
            hold = 0
            continue
        bb = ob["bb"][idx]
        ba = ob["ba"][idx]
        if bb <= 0 or ba <= 0 or ba <= bb:
            hold = 0
            continue

        depth_k = ob["bd"][idx] + ob["ad"][idx]
        depth_rec = depth_k / (depth_base + 1e-9)

        # Flow decay
        tr_i0 = np.searchsorted(tr["ts"], t_k - 1.0, side="left")
        tr_i1 = np.searchsorted(tr["ts"], t_k, side="left")
        total_flow = float(tr["notional"][tr_i0:tr_i1].sum()) if tr_i0 < tr_i1 else 0
        tr_i0p = np.searchsorted(tr["ts"], t_k - 2.0, side="left")
        prev_flow = float(tr["notional"][tr_i0p:tr_i0].sum()) if tr_i0p < tr_i0 else 0
        peak_flow = max(total_flow, prev_flow, 1.0)
        flow_frac = total_flow / peak_flow

        depth_rising = np.clip((depth_rec - 0.3) / 0.7, 0, 1)
        flow_decay = np.clip(1.0 - flow_frac, 0, 1)
        refill_score = 0.65 * depth_rising + 0.35 * flow_decay

        if refill_score > TH_REFILL:
            hold += 1
            if hold >= STATE_HOLD_TICKS:
                return t_k
        else:
            hold = 0

    return None


# -----------------------------------------------------------------------
# Stage-3: Causal returns at multiple horizons
# -----------------------------------------------------------------------

def compute_returns(ob, tr, t_entry, direction_sign, mid_t0):
    """Compute signed returns at multiple horizons after causal entry.
    Returns are in bps, signed so that positive = continuation direction."""
    result = {}

    # Entry mid price
    snap_entry = ob_snap(ob, t_entry)
    if snap_entry is None:
        return None
    mid_entry = snap_entry["mid"]
    result["mid_entry"] = mid_entry
    result["spread_entry_bp"] = snap_entry["spread_bp"]

    for horizon_s, label in zip(RETURN_HORIZONS, HORIZON_LABELS):
        t_h = t_entry + horizon_s

        # Mid at horizon
        snap_h = ob_snap(ob, t_h)
        if snap_h is None:
            # Fallback: use last trade price
            tr_idx = np.searchsorted(tr["ts"], t_h, side="right") - 1
            if tr_idx >= 0 and tr["ts"][tr_idx] > t_entry:
                price_h = tr["price"][tr_idx]
            else:
                result[f"ret_{label}_bp"] = None
                result[f"ret_{label}_dir_bp"] = None
                continue
        else:
            price_h = snap_h["mid"]

        ret_bp = (price_h - mid_entry) / mid_entry * 10000
        # dir_bp: positive = event direction (continuation), negative = MR
        ret_dir_bp = ret_bp * direction_sign
        result[f"ret_{label}_bp"] = round(ret_bp, 2)
        result[f"ret_{label}_dir_bp"] = round(ret_dir_bp, 2)

    # MFE/MAE in first 60s (by mid)
    mfe = 0.0
    mae = 0.0
    i_start = np.searchsorted(ob["ts"], t_entry, side="left")
    i_end = np.searchsorted(ob["ts"], t_entry + 60.0, side="right")
    for i in range(i_start, min(i_end, i_start + 600)):
        bb = ob["bb"][i]
        ba = ob["ba"][i]
        if bb <= 0 or ba <= 0 or ba <= bb:
            continue
        mid_i = (bb + ba) / 2.0
        pnl = (mid_i - mid_entry) / mid_entry * 10000
        # MFE in event direction (continuation), MAE opposite
        pnl_dir = pnl * direction_sign
        mfe = max(mfe, pnl_dir)
        mae = min(mae, pnl_dir)
    result["mfe_60s_dir_bp"] = round(mfe, 2)
    result["mae_60s_dir_bp"] = round(mae, 2)

    # Touch t0 within 60s?
    touched = False
    for i in range(i_start, min(i_end, i_start + 600)):
        bb = ob["bb"][i]
        ba = ob["ba"][i]
        if bb <= 0 or ba <= 0 or ba <= bb:
            continue
        mid_i = (bb + ba) / 2.0
        # Touch = mid crosses back to t0 level
        if direction_sign == 1:
            if mid_i <= mid_t0:
                touched = True
                result["touch_t0_time_s"] = round(ob["ts"][i] - t_entry, 2)
                break
        else:
            if mid_i >= mid_t0:
                touched = True
                result["touch_t0_time_s"] = round(ob["ts"][i] - t_entry, 2)
                break
    result["touch_t0_60s"] = 1 if touched else 0
    if not touched:
        result["touch_t0_time_s"] = None

    return result


# -----------------------------------------------------------------------
# Stage-4: Hour horizon metrics
# -----------------------------------------------------------------------

def compute_hour_metrics(ob, tr, t0, t_entry, direction_sign, mid_t0):
    """Compute hour-horizon metrics: RV, regime persistence, continuation."""
    result = {}

    # Realized volatility in next 1h (from t0)
    tr_i0 = np.searchsorted(tr["ts"], t0, side="left")
    tr_i1 = np.searchsorted(tr["ts"], t0 + 3600, side="left")
    if tr_i1 - tr_i0 > 10:
        prices = tr["price"][tr_i0:tr_i1]
        # Sample at ~1s intervals for RV
        n_samples = min(len(prices), 3600)
        step = max(1, len(prices) // n_samples)
        sampled = prices[::step]
        if len(sampled) > 1:
            log_rets = np.diff(np.log(sampled))
            result["rv_1h_bp"] = round(np.std(log_rets) * 10000 * np.sqrt(len(log_rets)), 2)
        else:
            result["rv_1h_bp"] = None
    else:
        result["rv_1h_bp"] = None

    # Max continuation in 1h (from entry)
    max_cont = 0.0
    max_mr = 0.0
    if t_entry:
        snap_entry = ob_snap(ob, t_entry)
        if snap_entry:
            mid_entry = snap_entry["mid"]
            # Sample OB at ~1s intervals for 1h
            for dt in range(0, 3600, 1):
                t_s = t_entry + dt
                idx = np.searchsorted(ob["ts"], t_s, side="right") - 1
                if idx < 0 or idx >= len(ob["ts"]):
                    continue
                bb = ob["bb"][idx]
                ba = ob["ba"][idx]
                if bb <= 0 or ba <= 0 or ba <= bb:
                    continue
                mid_s = (bb + ba) / 2.0
                ret = (mid_s - mid_entry) / mid_entry * 10000 * direction_sign
                max_cont = max(max_cont, ret)
                max_mr = min(max_mr, ret)

    result["max_cont_1h_bp"] = round(max_cont, 2)
    result["max_mr_1h_bp"] = round(max_mr, 2)

    # Regime: is the move still in event direction at 15m/30m/60m?
    for horizon_s, label in [(900, "15m"), (1800, "30m"), (3600, "60m")]:
        snap_h = ob_snap(ob, t0 + horizon_s)
        if snap_h:
            ret = (snap_h["mid"] - mid_t0) / mid_t0 * 10000 * direction_sign
            result[f"regime_{label}_dir_bp"] = round(ret, 2)
            result[f"regime_{label}_same_dir"] = 1 if ret > 0 else 0
        else:
            result[f"regime_{label}_dir_bp"] = None
            result[f"regime_{label}_same_dir"] = None

    return result


# -----------------------------------------------------------------------
# Process one event
# -----------------------------------------------------------------------

def process_event(ob, tr, ev):
    """Process one Stage-1 event through Stages 2-4."""
    t0 = ev["t0"]
    direction = ev["direction"]
    direction_sign = 1 if direction == "BUY" else -1

    # Get mid at t0
    snap0 = ob_snap(ob, t0)
    if snap0 is None:
        return None
    mid_t0 = snap0["mid"]

    # Quality filter: spread
    if snap0["spread_bp"] > MAX_SPREAD_BPS:
        return None

    row = {
        "event_id": ev["event_id"],
        "t0": t0,
        "t0_iso": ev.get("t0_iso", ""),
        "direction": direction,
        "mid_t0": round(mid_t0, 8),
        "spread_t0_bp": round(snap0["spread_bp"], 2),
    }

    # Stage-2: Lifecycle
    lifecycle = compute_lifecycle(ob, tr, t0, direction_sign, mid_t0)
    row.update(lifecycle)

    # Detect refill (causal)
    t_refill = detect_refill(ob, tr, t0, direction_sign)
    row["has_refill"] = 1 if t_refill else 0
    row["t_refill_s"] = round(t_refill - t0, 3) if t_refill else None

    # Stage-3: Causal returns (entry at t_refill if found, else t0+1s as fallback)
    t_entry = t_refill if t_refill else t0 + 1.0
    row["t_entry_s"] = round(t_entry - t0, 3)

    returns = compute_returns(ob, tr, t_entry, direction_sign, mid_t0)
    if returns:
        row.update(returns)
    else:
        return None

    # Stage-4: Hour horizon
    hour_metrics = compute_hour_metrics(ob, tr, t0, t_entry, direction_sign, mid_t0)
    row.update(hour_metrics)

    return row


# -----------------------------------------------------------------------
# Process one day for one symbol
# -----------------------------------------------------------------------

def process_day(symbol, date_str, events_day, next_date_str=None):
    """Process all events for one day. Returns list of result dicts."""
    t_start = time.monotonic()

    ob, tr = load_day_data(symbol, date_str, next_date_str)
    if ob is None:
        return []

    t_load = time.monotonic() - t_start
    results = []

    for _, ev in events_day.iterrows():
        row = process_event(ob, tr, ev)
        if row:
            results.append(row)

    t_total = time.monotonic() - t_start
    print(f"    [{date_str}] {len(results)}/{len(events_day)} events "
          f"(load={t_load:.1f}s total={t_total:.1f}s)", flush=True)

    # Explicitly free
    del ob, tr
    gc.collect()

    return results


# -----------------------------------------------------------------------
# Run all stages for one symbol
# -----------------------------------------------------------------------

def run_symbol(symbol, skip_existing=False):
    """Run Stages 2-4 for one symbol."""
    out_dir = OUTPUT_DIR / symbol
    out_path = out_dir / "events_enriched.parquet"

    if skip_existing and out_path.exists():
        print(f"  {symbol}: SKIP (exists)", flush=True)
        return

    events_path = out_dir / "events_stage1.parquet"
    if not events_path.exists():
        print(f"  {symbol}: SKIP (no Stage-1)", flush=True)
        return

    df = pd.read_parquet(events_path)
    df["date"] = df["t0_iso"].str[:10]
    dates = sorted(df["date"].unique())

    print(f"\n{'='*60}")
    print(f"  {symbol}: {len(df)} events across {len(dates)} days")
    print(f"{'='*60}", flush=True)

    # Build next-day map
    date_next = {}
    for i, d in enumerate(dates):
        if i + 1 < len(dates):
            d_dt = datetime.strptime(d, "%Y-%m-%d")
            n_dt = datetime.strptime(dates[i + 1], "%Y-%m-%d")
            if (n_dt - d_dt).days == 1:
                date_next[d] = dates[i + 1]
        # Also check if next calendar day exists
        d_dt = datetime.strptime(d, "%Y-%m-%d")
        n_str = (d_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        ob_next = DATA_DIR / symbol / f"{n_str}_orderbook.jsonl"
        if ob_next.exists():
            date_next[d] = n_str

    all_results = []
    t_total = time.monotonic()

    for i, d in enumerate(dates):
        ev_day = df[df["date"] == d]
        # Always load next day for hour-horizon metrics
        next_d = date_next.get(d)
        results = process_day(symbol, d, ev_day, next_d)
        all_results.extend(results)

        # RAM check every 5 days
        if (i + 1) % 5 == 0:
            import psutil
            mem = psutil.virtual_memory()
            print(f"    >> {i+1}/{len(dates)} days, {len(all_results)} events, "
                  f"RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f}GB "
                  f"({mem.percent}%)", flush=True)

    elapsed = time.monotonic() - t_total
    print(f"  {symbol}: DONE — {len(all_results)}/{len(df)} enriched in {elapsed:.0f}s",
          flush=True)

    if not all_results:
        print(f"  {symbol}: NO RESULTS", flush=True)
        return

    result_df = pd.DataFrame(all_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_path, index=False)
    result_df.to_csv(out_dir / "events_enriched.csv", index=False)
    print(f"  Saved: {out_path}", flush=True)

    # Quick summary
    print_summary(symbol, result_df)

    return result_df


# -----------------------------------------------------------------------
# Summary per coin
# -----------------------------------------------------------------------

def print_summary(symbol, df):
    """Print quick summary for one coin."""
    n = len(df)
    n_refill = df["has_refill"].sum()

    print(f"\n  --- {symbol} Summary ({n} events) ---")
    print(f"  Refill detected: {n_refill}/{n} ({n_refill/n*100:.0f}%)")
    print(f"  Spread at t0: med={df['spread_t0_bp'].median():.1f}bp "
          f"p95={df['spread_t0_bp'].quantile(0.95):.1f}bp")
    print(f"  VacuumScore: med={df['vacuum_score'].median():.2f}")

    # Returns at each horizon
    print(f"\n  Direction-signed returns (positive=continuation):")
    for label in HORIZON_LABELS:
        col = f"ret_{label}_dir_bp"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                pos_pct = (vals > 0).mean() * 100
                print(f"    {label:>5s}: med={vals.median():+6.1f}bp  "
                      f"mean={vals.mean():+6.1f}bp  "
                      f"p25={vals.quantile(0.25):+6.1f}bp  "
                      f"p75={vals.quantile(0.75):+6.1f}bp  "
                      f"cont%={pos_pct:.0f}%  n={len(vals)}")

    # Hour regime persistence
    print(f"\n  Regime persistence (% still in event direction):")
    for label in ["15m", "30m", "60m"]:
        col = f"regime_{label}_same_dir"
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"    {label}: {vals.mean()*100:.1f}%  "
                      f"(dir_ret: med={df[f'regime_{label}_dir_bp'].dropna().median():+.1f}bp)")

    # Touch t0 within 60s
    touch = df["touch_t0_60s"].dropna()
    if len(touch) > 0:
        print(f"\n  Touch t0 within 60s: {touch.mean()*100:.1f}%")
        touch_times = df["touch_t0_time_s"].dropna()
        if len(touch_times) > 0:
            print(f"    Time to touch: med={touch_times.median():.1f}s "
                  f"p75={touch_times.quantile(0.75):.1f}s")

    # MFE/MAE
    mfe = df["mfe_60s_dir_bp"].dropna()
    mae = df["mae_60s_dir_bp"].dropna()
    if len(mfe) > 0:
        print(f"\n  MFE/MAE (60s, dir):")
        print(f"    MFE: med={mfe.median():+.1f}bp  mean={mfe.mean():+.1f}bp")
        print(f"    MAE: med={mae.median():+.1f}bp  mean={mae.mean():+.1f}bp")

    # Tail asymmetry
    if "ret_60m_dir_bp" in df.columns:
        r60 = df["ret_60m_dir_bp"].dropna()
        if len(r60) > 10:
            up_tail = r60.quantile(0.95)
            down_tail = abs(r60.quantile(0.05))
            asym = up_tail / down_tail if down_tail > 0 else 0
            print(f"\n  Tail asymmetry (60m): p95={up_tail:+.1f}bp  p5={r60.quantile(0.05):+.1f}bp  "
                  f"ratio={asym:.2f}x")

    print()


# -----------------------------------------------------------------------
# Cross-coin comparison
# -----------------------------------------------------------------------

def cross_coin_report():
    """Generate cross-coin comparison from all enriched parquets."""
    print(f"\n{'='*80}")
    print("CROSS-COIN COMPARISON")
    print(f"{'='*80}")

    rows = []
    for sym in ALL_SYMBOLS:
        path = OUTPUT_DIR / sym / "events_enriched.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        n = len(df)
        if n < 10:
            continue

        row = {"symbol": sym, "n_events": n}
        row["events_per_day"] = round(n / 30, 1)
        row["spread_med_bp"] = round(df["spread_t0_bp"].median(), 1)
        row["refill_pct"] = round(df["has_refill"].mean() * 100, 0)

        # Returns
        for label in HORIZON_LABELS:
            col = f"ret_{label}_dir_bp"
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    row[f"ret_{label}_med"] = round(vals.median(), 1)
                    row[f"ret_{label}_mean"] = round(vals.mean(), 1)
                    row[f"cont_{label}_pct"] = round((vals > 0).mean() * 100, 0)

        # Regime persistence
        for label in ["15m", "30m", "60m"]:
            col = f"regime_{label}_same_dir"
            if col in df.columns:
                vals = df[col].dropna()
                row[f"regime_{label}_pct"] = round(vals.mean() * 100, 0) if len(vals) > 0 else None

        # Tail asymmetry at 60m
        if "ret_60m_dir_bp" in df.columns:
            r60 = df["ret_60m_dir_bp"].dropna()
            if len(r60) > 10:
                up = r60.quantile(0.95)
                down = abs(r60.quantile(0.05))
                row["tail_asym_60m"] = round(up / down, 2) if down > 0 else None

        # Touch t0
        touch = df["touch_t0_60s"].dropna()
        row["touch_t0_pct"] = round(touch.mean() * 100, 0) if len(touch) > 0 else None

        # RV 1h
        rv = df["rv_1h_bp"].dropna()
        row["rv_1h_med"] = round(rv.median(), 0) if len(rv) > 0 else None

        # Pass/fail criteria
        passes = []
        r60_med = row.get("ret_60m_med")
        if r60_med and r60_med >= 25:
            passes.append("med_ret≥25bp")
        tail = row.get("tail_asym_60m")
        if tail and tail >= 2.0:
            passes.append("tail≥2x")
        # AUC will be computed separately
        row["criteria_met"] = ", ".join(passes) if passes else "NONE"

        rows.append(row)

    if not rows:
        print("  No enriched data found.")
        return

    comp_df = pd.DataFrame(rows)

    # Print table
    print(f"\n{'Symbol':>16s} {'N':>6s} {'ev/d':>5s} {'spr':>5s} {'ref%':>4s} "
          f"{'5s':>6s} {'60s':>6s} {'5m':>6s} {'15m':>6s} {'60m':>6s} "
          f"{'c60m':>4s} {'tail':>5s} {'rvol':>5s} {'Criteria':>20s}")
    print("-" * 120)
    for _, r in comp_df.iterrows():
        print(f"{r['symbol']:>16s} {r['n_events']:6d} {r['events_per_day']:5.1f} "
              f"{r['spread_med_bp']:5.1f} {r['refill_pct']:4.0f} "
              f"{r.get('ret_5s_med', ''):>6} {r.get('ret_60s_med', ''):>6} "
              f"{r.get('ret_5m_med', ''):>6} {r.get('ret_15m_med', ''):>6} "
              f"{r.get('ret_60m_med', ''):>6} "
              f"{r.get('cont_60m_pct', ''):>4} {r.get('tail_asym_60m', ''):>5} "
              f"{r.get('rv_1h_med', ''):>5} {r['criteria_met']:>20s}")

    # Save
    comp_path = OUTPUT_DIR / "cross_coin_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\nSaved: {comp_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Coin Stages 2-4 Analysis")
    parser.add_argument("symbol", help="Symbol or ALL")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.symbol.upper() == "ALL":
        symbols = ALL_SYMBOLS
    else:
        symbols = [args.symbol.upper()]

    for sym in symbols:
        try:
            run_symbol(sym, skip_existing=args.skip_existing)
        except Exception as e:
            import traceback
            print(f"  {sym}: ERROR — {e}", flush=True)
            traceback.print_exc()
        gc.collect()

    if len(symbols) > 1:
        cross_coin_report()


if __name__ == "__main__":
    main()
