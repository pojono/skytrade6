#!/usr/bin/env python3
"""
Stage-4.1: Event-Based Exit Feasibility (Causal + Queue-Realistic)

Tests whether the MR-leg (retrace) can be captured with event-based TP
(touch of t0 level) before continuation resumes.

Two scenarios:
  A) TAKER_MR: market order at t_refill, exit at TP/SL/TO
  B) MAKER_MR: limit order at t_refill, queue-realistic fill, exit at TP/SL/TO

All entries are OPPOSITE to event direction (betting on retrace to t0).
All features strictly causal (data ≤ t only).

Usage:
  python stage41_event_exits.py DOGEUSDT [--workers 8]
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

OB_LEVELS_K = 5
TICK_MS = 100
MAX_SIM_S = 60.0   # max simulation window (covers all TOs)

# Score thresholds (from Stage-4)
TH_REFILL = 0.5
STATE_HOLD_TICKS = 3  # 300ms persistence

# Scenario grid (start with 6 priority configs, then expand)
SCENARIOS = [
    # (scenario, limit_type, tp_type, sl_bp, to_s, delay_ms)
    ("TAKER_MR", "NA",  "TP0",  20, 5,  0),
    ("TAKER_MR", "NA",  "TP0",  30, 10, 0),
    ("TAKER_MR", "NA",  "TP0",  20, 10, 0),
    ("TAKER_MR", "NA",  "TP0",  30, 30, 0),
    ("TAKER_MR", "NA",  "TP5",  20, 10, 0),
    ("TAKER_MR", "NA",  "TP10", 30, 30, 0),
    ("MAKER_MR", "L1",  "TP0",  20, 10, 0),
    ("MAKER_MR", "L1",  "TP0",  30, 20, 0),
    ("MAKER_MR", "L1",  "TP0",  20, 30, 0),
    ("MAKER_MR", "MID", "TP0",  20, 10, 0),
    ("MAKER_MR", "MID", "TP0",  30, 30, 0),
    ("MAKER_MR", "L1",  "TP5",  20, 10, 0),
    # Delay sensitivity
    ("TAKER_MR", "NA",  "TP0",  20, 10, 100),
    ("TAKER_MR", "NA",  "TP0",  20, 10, 300),
    ("MAKER_MR", "L1",  "TP0",  20, 10, 100),
    ("MAKER_MR", "L1",  "TP0",  20, 10, 300),
]

FEES = {"TAKER_MR": 20.0, "MAKER_MR": 4.0}  # bps round-trip

TP_OFFSETS = {"TP0": 0, "TP2": 2, "TP5": 5, "TP10": 10}


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_ob_l5(filepath: Path):
    ts_l, bb_l, ba_l, bd_l, ad_l = [], [], [], [], []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
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
            if (i + 1) % 200000 == 0:
                print(f"    OB {i+1:,}...", flush=True)
    return {
        "ts": np.array(ts_l, dtype=np.float64),
        "best_bid": np.array(bb_l, dtype=np.float64),
        "best_ask": np.array(ba_l, dtype=np.float64),
        "bid_depth": np.array(bd_l, dtype=np.float64),
        "ask_depth": np.array(ad_l, dtype=np.float64),
    }


def load_trades(filepath: Path):
    ts_l, side_l, not_l, price_l = [], [], [], []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            ts_l.append(float(row["timestamp"]))
            side_l.append(1 if row["side"] == "Buy" else -1)
            not_l.append(float(row["foreignNotional"]))
            price_l.append(float(row["price"]))
            if (i + 1) % 500000 == 0:
                print(f"    Trades {i+1:,}...", flush=True)
    return {
        "ts": np.array(ts_l, dtype=np.float64),
        "side": np.array(side_l, dtype=np.int8),
        "notional": np.array(not_l, dtype=np.float64),
        "price": np.array(price_l, dtype=np.float64),
    }


# -----------------------------------------------------------------------
# Find pre-shock depth baseline (from Stage-4)
# -----------------------------------------------------------------------

def find_pre_shock_depth(ob, t0, lookback=5.0):
    t_start = t0 - lookback
    i0 = np.searchsorted(ob["ts"], t_start, side="left")
    i1 = np.searchsorted(ob["ts"], t0, side="left")
    if i0 >= i1:
        return None
    for idx in range(i1 - 1, i0 - 1, -1):
        sp = ob["best_ask"][idx] - ob["best_bid"][idx]
        if sp > 0 and ob["best_bid"][idx] > 0:
            return float(ob["bid_depth"][idx] + ob["ask_depth"][idx])
    return None


# -----------------------------------------------------------------------
# Detect refill time (causal, from OB stream)
# -----------------------------------------------------------------------

def detect_refill_time(ob, tr, t0, direction_sign):
    """Find t_refill: first time RefillScore stays > threshold for 300ms.
    Simplified: depth_recovery towards pre-shock level + flow decay."""
    depth_base = find_pre_shock_depth(ob, t0)
    if depth_base is None or depth_base < 100:
        return None

    hold = 0
    for k in range(0, 51):  # 0..5s
        t_k = t0 + k * 0.1
        idx = np.searchsorted(ob["ts"], t_k, side="right") - 1
        if idx < 0:
            hold = 0
            continue
        bb = ob["best_bid"][idx]
        ba = ob["best_ask"][idx]
        if bb <= 0 or ba <= 0 or ba <= bb:
            hold = 0
            continue
        depth_k = ob["bid_depth"][idx] + ob["ask_depth"][idx]
        depth_rec = depth_k / (depth_base + 1e-9)

        # Flow in last 1s
        tr_i0 = np.searchsorted(tr["ts"], t_k - 1.0, side="left")
        tr_i1 = np.searchsorted(tr["ts"], t_k, side="left")
        if tr_i0 < tr_i1:
            total_flow = float(tr["notional"][tr_i0:tr_i1].sum())
        else:
            total_flow = 0

        # Flow in prev 1s
        tr_i0p = np.searchsorted(tr["ts"], t_k - 2.0, side="left")
        if tr_i0p < tr_i0:
            prev_flow = float(tr["notional"][tr_i0p:tr_i0].sum())
        else:
            prev_flow = 0

        peak_flow = max(total_flow, prev_flow, 1.0)
        flow_rate_frac = total_flow / peak_flow

        # Simplified refill score
        depth_rising = np.clip((depth_rec - 0.3) / 0.7, 0, 1)
        flow_not_rising = np.clip(1.0 - flow_rate_frac, 0, 1)
        refill_score = 0.65 * depth_rising + 0.35 * flow_not_rising

        if refill_score > TH_REFILL:
            hold += 1
            if hold >= STATE_HOLD_TICKS:
                return t_k
        else:
            hold = 0

    return None


# -----------------------------------------------------------------------
# OB snapshot at time t
# -----------------------------------------------------------------------

def ob_at(ob, t):
    """Get OB state at time t. Returns (bb, ba, bid_depth, ask_depth, spread, mid, valid)."""
    idx = np.searchsorted(ob["ts"], t, side="right") - 1
    if idx < 0:
        return None
    bb = ob["best_bid"][idx]
    ba = ob["best_ask"][idx]
    if bb <= 0 or ba <= 0 or ba <= bb:
        return None
    spread = ba - bb
    mid = (ba + bb) / 2.0
    age_ms = (t - ob["ts"][idx]) * 1000.0
    return {
        "bb": bb, "ba": ba,
        "bid_depth": ob["bid_depth"][idx],
        "ask_depth": ob["ask_depth"][idx],
        "spread": spread, "mid": mid,
        "age_ms": age_ms,
    }


# -----------------------------------------------------------------------
# Queue fill check for maker
# -----------------------------------------------------------------------

def check_maker_fill(tr, t_place, limit_price, mr_side, queue_ahead, max_wait_s=30.0):
    """Check if maker limit order fills within max_wait_s.
    mr_side: +1 = we BUY (MR on sell-shock), -1 = we SELL (MR on buy-shock).
    Returns (filled, fill_time, vol_through)."""
    tr_i0 = np.searchsorted(tr["ts"], t_place, side="left")
    tr_i1 = np.searchsorted(tr["ts"], t_place + max_wait_s, side="left")
    if tr_i0 >= tr_i1:
        return False, None, 0.0

    ts_w = tr["ts"][tr_i0:tr_i1]
    sides_w = tr["side"][tr_i0:tr_i1]
    nots_w = tr["notional"][tr_i0:tr_i1]
    prices_w = tr["price"][tr_i0:tr_i1]

    # For BUY limit: need SELL aggressor trades at <= limit_price
    # For SELL limit: need BUY aggressor trades at >= limit_price
    if mr_side == 1:  # we BUY
        mask = (sides_w == -1) & (prices_w <= limit_price)
    else:  # we SELL
        mask = (sides_w == 1) & (prices_w >= limit_price)

    if not mask.any():
        return False, None, 0.0

    cum_vol = np.cumsum(nots_w[mask])
    vol_through = float(cum_vol[-1]) if len(cum_vol) > 0 else 0.0

    if vol_through < max(queue_ahead, 100.0):
        return False, None, vol_through

    # Find fill time
    fill_idx_in_mask = np.searchsorted(cum_vol, max(queue_ahead, 100.0))
    fill_indices = np.where(mask)[0]
    if fill_idx_in_mask < len(fill_indices):
        fill_trade_idx = fill_indices[fill_idx_in_mask] + tr_i0
        fill_time = tr["ts"][fill_trade_idx]
    else:
        fill_time = ts_w[-1]

    return True, fill_time, vol_through


# -----------------------------------------------------------------------
# Simulate one trade (entry → TP/SL/TO)
# -----------------------------------------------------------------------

def simulate_trade(ob, tr, t_entry, entry_price, mr_side, mid_t0,
                   tp_type, sl_bp, to_s, fee_bp):
    """
    Simulate from t_entry forward. mr_side: +1=BUY(MR), -1=SELL(MR).
    TP: touch of mid(t0) + tp_offset in MR direction.
    SL: adverse move from entry_price.
    TO: time-out.
    Returns dict with exit info.
    """
    tp_offset_bp = TP_OFFSETS.get(tp_type, 0)

    # TP level: mid(t0) ± offset in MR direction
    # MR direction = opposite to event direction = mr_side direction
    # If mr_side=+1 (we bought), TP is when price goes UP to mid(t0)+offset
    # If mr_side=-1 (we sold), TP is when price goes DOWN to mid(t0)-offset
    tp_price = mid_t0 + mr_side * tp_offset_bp * mid_t0 / 10000.0

    # SL level: entry_price ± sl_bp against us
    sl_price = entry_price - mr_side * sl_bp * entry_price / 10000.0

    t_end = t_entry + to_s
    i_start = np.searchsorted(ob["ts"], t_entry, side="left")
    i_end = np.searchsorted(ob["ts"], t_end, side="right")

    if i_start >= i_end:
        return {"exit_reason": "NODATA", "net_pnl_bp": 0, "hold_ms": 0}

    # Track MFE/MAE from entry
    best_pnl = -999999.0
    worst_pnl = 999999.0

    for i in range(i_start, i_end):
        t_i = ob["ts"][i]
        bb_i = ob["best_bid"][i]
        ba_i = ob["best_ask"][i]

        # Skip invalid snapshots
        if bb_i <= 0 or ba_i <= 0 or ba_i <= bb_i:
            continue

        # Executable exit prices:
        # If we're LONG (mr_side=+1), we SELL to exit → get bid
        # If we're SHORT (mr_side=-1), we BUY to exit → pay ask
        if mr_side == 1:
            exit_exec = bb_i  # sell at bid
        else:
            exit_exec = ba_i  # buy at ask

        # PnL from entry (direction-aware)
        pnl_bp = mr_side * (exit_exec - entry_price) / entry_price * 10000
        best_pnl = max(best_pnl, pnl_bp)
        worst_pnl = min(worst_pnl, pnl_bp)

        hold_ms = (t_i - t_entry) * 1000.0

        # Check TP: has executable price reached TP level?
        if mr_side == 1:
            tp_hit = bb_i >= tp_price  # can sell at bid >= tp
        else:
            tp_hit = ba_i <= tp_price  # can buy at ask <= tp

        if tp_hit:
            gross = mr_side * (tp_price - entry_price) / entry_price * 10000
            return {
                "exit_reason": "TP",
                "exit_time_ms": round(hold_ms),
                "exit_price": round(tp_price, 8),
                "gross_pnl_bp": round(gross, 2),
                "net_pnl_bp": round(gross - fee_bp, 2),
                "mfe_bp": round(best_pnl, 2),
                "mae_bp": round(worst_pnl, 2),
                "hold_ms": round(hold_ms),
            }

        # Check SL: has price moved against us by sl_bp?
        if pnl_bp <= -sl_bp:
            return {
                "exit_reason": "SL",
                "exit_time_ms": round(hold_ms),
                "exit_price": round(exit_exec, 8),
                "gross_pnl_bp": round(pnl_bp, 2),
                "net_pnl_bp": round(pnl_bp - fee_bp, 2),
                "mfe_bp": round(best_pnl, 2),
                "mae_bp": round(worst_pnl, 2),
                "hold_ms": round(hold_ms),
            }

    # Time-out: exit at last valid snapshot
    # Find last valid OB in window
    for i in range(i_end - 1, i_start - 1, -1):
        bb_i = ob["best_bid"][i]
        ba_i = ob["best_ask"][i]
        if bb_i > 0 and ba_i > 0 and ba_i > bb_i:
            if mr_side == 1:
                exit_exec = bb_i
            else:
                exit_exec = ba_i
            pnl_bp = mr_side * (exit_exec - entry_price) / entry_price * 10000
            hold_ms = (ob["ts"][i] - t_entry) * 1000.0
            return {
                "exit_reason": "TO",
                "exit_time_ms": round(hold_ms),
                "exit_price": round(exit_exec, 8),
                "gross_pnl_bp": round(pnl_bp, 2),
                "net_pnl_bp": round(pnl_bp - fee_bp, 2),
                "mfe_bp": round(best_pnl if best_pnl > -999998 else 0, 2),
                "mae_bp": round(worst_pnl if worst_pnl < 999998 else 0, 2),
                "hold_ms": round(hold_ms),
            }

    return {"exit_reason": "NODATA", "net_pnl_bp": 0, "hold_ms": 0}


# -----------------------------------------------------------------------
# Process one event across all scenarios
# -----------------------------------------------------------------------

def process_event(ob, tr, ev, t_refill):
    """Run all scenarios for one event. Returns list of result dicts."""
    t0 = ev["t0"]
    direction_sign = 1 if ev["direction"] == "BUY" else -1
    mr_side = -direction_sign  # opposite to event
    mid_t0 = ev["mid"] if ev["mid"] > 0 else ev["price_at_t0"]

    if mid_t0 <= 0:
        return []

    results = []

    for scenario, limit_type, tp_type, sl_bp, to_s, delay_ms in SCENARIOS:
        fee_bp = FEES[scenario]
        t_place = t_refill + delay_ms / 1000.0

        base = {
            "event_id": ev["event_id"],
            "t0": t0,
            "direction": ev["direction"],
            "mid_t0": round(mid_t0, 8),
            "t_refill_detect": round(t_refill - t0, 4),
            "t_place": round(t_place - t0, 4),
            "scenario": scenario,
            "limit_type": limit_type,
            "tp_type": tp_type,
            "sl_bp": sl_bp,
            "to_s": to_s,
            "decision_delay_ms": delay_ms,
            "fees_bp": fee_bp,
        }

        snap_place = ob_at(ob, t_place)
        if snap_place is None:
            base.update({"filled": 0, "exit_reason": "NOFILL", "net_pnl_bp": 0})
            results.append(base)
            continue

        base["entry_spread_bp"] = round(snap_place["spread"] / mid_t0 * 10000, 2)
        base["entry_depth"] = round(snap_place["bid_depth"] + snap_place["ask_depth"], 0)
        base["entry_book_age_ms"] = round(snap_place["age_ms"], 1)

        if scenario == "TAKER_MR":
            # Market order: buy at ask, sell at bid
            if mr_side == 1:
                entry_price = snap_place["ba"]  # buy at ask
            else:
                entry_price = snap_place["bb"]  # sell at bid

            base["filled"] = 1
            base["fill_time_ms"] = 0
            base["fill_price"] = round(entry_price, 8)
            base["queue_ahead"] = 0
            base["exec_through_level"] = 0
            base["t_entry"] = round(t_place - t0, 4)

            result = simulate_trade(ob, tr, t_place, entry_price, mr_side,
                                    mid_t0, tp_type, sl_bp, to_s, fee_bp)
            base.update(result)
            results.append(base)

        elif scenario == "MAKER_MR":
            # Limit order
            if limit_type == "L1":
                if mr_side == 1:
                    limit_price = snap_place["bb"]  # buy at bid
                else:
                    limit_price = snap_place["ba"]  # sell at ask
            elif limit_type == "MID":
                limit_price = snap_place["mid"]
            else:  # MID_HALFSPREAD
                if mr_side == 1:
                    limit_price = snap_place["mid"] - snap_place["spread"] / 2
                else:
                    limit_price = snap_place["mid"] + snap_place["spread"] / 2

            # Queue ahead: visible depth at our level
            if mr_side == 1:
                queue_ahead = snap_place["bid_depth"]
            else:
                queue_ahead = snap_place["ask_depth"]

            base["fill_price"] = round(limit_price, 8)
            base["queue_ahead"] = round(queue_ahead, 2)

            filled, fill_time, vol_through = check_maker_fill(
                tr, t_place, limit_price, mr_side, queue_ahead, max_wait_s=to_s
            )

            base["exec_through_level"] = round(vol_through, 2)
            base["filled"] = 1 if filled else 0

            if not filled:
                base["exit_reason"] = "NOFILL"
                base["net_pnl_bp"] = 0
                base["fill_time_ms"] = 0
                base["t_entry"] = None
                results.append(base)
                continue

            base["fill_time_ms"] = round((fill_time - t_place) * 1000)
            base["t_entry"] = round(fill_time - t0, 4)

            # Remaining time for TP/SL/TO
            remaining_to = to_s - (fill_time - t_place)
            if remaining_to <= 0:
                base["exit_reason"] = "TO"
                base["net_pnl_bp"] = -fee_bp
                base["gross_pnl_bp"] = 0
                base["hold_ms"] = 0
                results.append(base)
                continue

            result = simulate_trade(ob, tr, fill_time, limit_price, mr_side,
                                    mid_t0, tp_type, sl_bp, remaining_to, fee_bp)
            base.update(result)
            results.append(base)

    return results


# -----------------------------------------------------------------------
# Process one day
# -----------------------------------------------------------------------

def process_day(symbol, date_str, events_df, next_date_str=None):
    t0_wall = time.monotonic()

    ob_path = DATA_DIR / symbol / f"{date_str}_orderbook.jsonl"
    tr_path = DATA_DIR / symbol / f"{date_str}_trades.csv"
    if not ob_path.exists() or not tr_path.exists():
        print(f"  [{date_str}] SKIP: missing data", flush=True)
        return []

    print(f"  [{date_str}] Loading...", flush=True)
    ob = load_ob_l5(ob_path)
    tr = load_trades(tr_path)

    # Next day boundary
    if next_date_str:
        nob = DATA_DIR / symbol / f"{next_date_str}_orderbook.jsonl"
        ntr = DATA_DIR / symbol / f"{next_date_str}_trades.csv"
        if nob.exists() and ntr.exists():
            print(f"  [{date_str}] + next day...", flush=True)
            ob_n = load_ob_l5(nob)
            tr_n = load_trades(ntr)
            for k in ob:
                ob[k] = np.concatenate([ob[k], ob_n[k]])
            for k in tr:
                tr[k] = np.concatenate([tr[k], tr_n[k]])

    print(f"  [{date_str}] Processing {len(events_df)} events...", flush=True)
    all_results = []
    refill_found = 0

    for _, ev in events_df.iterrows():
        t0 = ev["t0"]
        direction_sign = 1 if ev["direction"] == "BUY" else -1

        # Detect refill causally
        t_refill = detect_refill_time(ob, tr, t0, direction_sign)
        if t_refill is None:
            continue
        refill_found += 1

        # Quality filter: valid spread at entry
        snap = ob_at(ob, t_refill)
        if snap is None:
            continue

        results = process_event(ob, tr, ev, t_refill)
        all_results.extend(results)

    elapsed = time.monotonic() - t0_wall
    print(f"  [{date_str}] DONE: {refill_found}/{len(events_df)} with refill, "
          f"{len(all_results)} sims, {elapsed:.1f}s", flush=True)
    return all_results


# -----------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------

def report_a_profitability(df):
    """Report A: Profitability surface."""
    print(f"\n{'='*80}")
    print("REPORT A: PROFITABILITY SURFACE")
    print(f"{'='*80}")

    group_cols = ["scenario", "limit_type", "tp_type", "sl_bp", "to_s", "decision_delay_ms"]

    for _, grp in df.groupby(group_cols):
        row = grp.iloc[0]
        label = (f"{row['scenario']:10s} {row['limit_type']:4s} {row['tp_type']:4s} "
                 f"SL{row['sl_bp']:2.0f} TO{row['to_s']:2.0f}s "
                 f"d{row['decision_delay_ms']:.0f}ms")

        n_signals = len(grp)
        n_filled = grp["filled"].sum()
        fill_rate = n_filled / n_signals if n_signals > 0 else 0

        filled = grp[grp["filled"] == 1]
        if len(filled) == 0:
            print(f"  {label}: n={n_signals:5d}  fill={fill_rate:.0%}  NO FILLS")
            continue

        # Exit reasons
        tp_count = (filled["exit_reason"] == "TP").sum()
        sl_count = (filled["exit_reason"] == "SL").sum()
        to_count = (filled["exit_reason"] == "TO").sum()
        tp_rate = tp_count / len(filled)

        net = filled["net_pnl_bp"]
        gross = filled["gross_pnl_bp"].dropna()

        # EV per signal (not per trade)
        ev_per_signal = net.sum() / n_signals

        hold = filled["hold_ms"].dropna()

        print(f"  {label}: "
              f"n={n_signals:5d}  fill={fill_rate:.0%}  "
              f"TP={tp_rate:.0%}({tp_count}) SL={sl_count} TO={to_count}  "
              f"net_med={net.median():+5.1f}bp  net_mean={net.mean():+5.1f}bp  "
              f"p5={net.quantile(0.05):+5.1f}bp  "
              f"hold_med={hold.median():5.0f}ms  "
              f"EV/sig={ev_per_signal:+5.2f}bp")


def report_b_time_to_touch(df):
    """Report B: Time-to-touch distributions."""
    print(f"\n{'='*80}")
    print("REPORT B: TIME-TO-TOUCH (TP events only)")
    print(f"{'='*80}")

    for tp_type in ["TP0", "TP2", "TP5", "TP10"]:
        tp_events = df[(df["tp_type"] == tp_type) & (df["exit_reason"] == "TP") &
                       (df["filled"] == 1)]
        if len(tp_events) == 0:
            continue

        # Group by scenario for comparison
        for scenario in ["TAKER_MR", "MAKER_MR"]:
            sub = tp_events[tp_events["scenario"] == scenario]
            if len(sub) == 0:
                continue
            t = sub["exit_time_ms"].dropna()
            # Also count total filled for TP rate
            total_filled = df[(df["tp_type"] == tp_type) &
                              (df["scenario"] == scenario) &
                              (df["filled"] == 1)]
            tp_rate = len(sub) / len(total_filled) if len(total_filled) > 0 else 0

            print(f"  {tp_type} {scenario:10s}: "
                  f"n={len(sub):5d}  TP_rate={tp_rate:.1%}  "
                  f"med={t.median():5.0f}ms  p75={t.quantile(0.75):5.0f}ms  "
                  f"p90={t.quantile(0.90):5.0f}ms  "
                  f"P(<2s)={(t<2000).mean():.1%}  "
                  f"P(<5s)={(t<5000).mean():.1%}  "
                  f"P(<10s)={(t<10000).mean():.1%}")


def report_c_makers_curse(df):
    """Report C: Maker's curse diagnostic."""
    print(f"\n{'='*80}")
    print("REPORT C: MAKER'S CURSE DIAGNOSTIC")
    print(f"{'='*80}")

    maker = df[df["scenario"] == "MAKER_MR"]
    if len(maker) == 0:
        print("  No maker events.")
        return

    for (lt, tp, sl, to, delay), grp in maker.groupby(
            ["limit_type", "tp_type", "sl_bp", "to_s", "decision_delay_ms"]):
        label = f"{lt:4s} {tp:4s} SL{sl:2.0f} TO{to:2.0f}s d{delay:.0f}ms"
        filled = grp[grp["filled"] == 1]
        not_filled = grp[grp["filled"] == 0]

        if len(filled) < 5 or len(not_filled) < 5:
            continue

        # For not-filled: what WOULD have been the outcome?
        # We can't directly measure, but we can note that not-filled means
        # price didn't reach our limit = it moved in event direction (continuation)
        # which is the opposite of what we wanted.
        # So by construction, not-filled events had better MR outcomes (price didn't come to us).

        net_f = filled["net_pnl_bp"]
        print(f"  {label}: "
              f"filled={len(filled):4d}({len(filled)/len(grp):.0%})  "
              f"net_med={net_f.median():+5.1f}bp  net_mean={net_f.mean():+5.1f}bp  "
              f"TP_rate={(filled['exit_reason']=='TP').mean():.0%}  "
              f"SL_rate={(filled['exit_reason']=='SL').mean():.0%}")


def report_d_delay_sensitivity(df):
    """Report D: Sensitivity to decision delay."""
    print(f"\n{'='*80}")
    print("REPORT D: DELAY SENSITIVITY")
    print(f"{'='*80}")

    for scenario in ["TAKER_MR", "MAKER_MR"]:
        sub = df[(df["scenario"] == scenario) & (df["tp_type"] == "TP0") &
                 (df["sl_bp"] == 20) & (df["to_s"] == 10)]
        if len(sub) == 0:
            continue
        print(f"\n  {scenario} TP0 SL20 TO10:")
        for delay in sorted(sub["decision_delay_ms"].unique()):
            d_sub = sub[sub["decision_delay_ms"] == delay]
            filled = d_sub[d_sub["filled"] == 1]
            if len(filled) == 0:
                continue
            net = filled["net_pnl_bp"]
            n_sig = len(d_sub)
            ev_sig = net.sum() / n_sig
            tp_rate = (filled["exit_reason"] == "TP").mean()
            print(f"    delay={delay:3.0f}ms: "
                  f"n={n_sig:5d}  fill={len(filled)/n_sig:.0%}  "
                  f"TP={tp_rate:.0%}  "
                  f"net_med={net.median():+5.1f}bp  "
                  f"EV/sig={ev_sig:+5.2f}bp")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-4.1: Event-Based Exits")
    parser.add_argument("symbol")
    parser.add_argument("--workers", "-w", type=int, default=0)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    workers = args.workers or max(1, (os.cpu_count() or 1) // 2)

    events_path = OUTPUT_DIR / symbol / "events_stage1.parquet"
    if not events_path.exists():
        print(f"ERROR: {events_path} not found.")
        sys.exit(1)

    df = pd.read_parquet(events_path)
    df["date"] = df["t0_iso"].str[:10]
    dates = sorted(df["date"].unique())

    print(f"Symbol:  {symbol}")
    print(f"Stage-1 events: {len(df):,}")
    print(f"Days: {len(dates)}, Workers: {workers}")
    print(f"Scenarios: {len(SCENARIOS)} configs")
    print(f"Tick: {TICK_MS}ms, Max sim: {MAX_SIM_S}s")
    print("=" * 60, flush=True)

    t_total = time.monotonic()

    date_next = {}
    for i, d in enumerate(dates):
        if i + 1 < len(dates):
            d_dt = datetime.strptime(d, "%Y-%m-%d")
            n_dt = datetime.strptime(dates[i + 1], "%Y-%m-%d")
            if (n_dt - d_dt).days == 1:
                date_next[d] = dates[i + 1]

    all_results = []
    total_days = len(dates)
    with ProcessPoolExecutor(max_workers=min(workers, total_days)) as executor:
        futures = {}
        for d in dates:
            ev_day = df[df["date"] == d].copy()
            need_next = False
            if len(ev_day) > 0:
                last_t0 = ev_day["t0"].max()
                day_end = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).timestamp()
                if last_t0 + 10 + MAX_SIM_S > day_end:
                    need_next = True
            next_d = date_next.get(d) if need_next else None
            futures[executor.submit(process_day, symbol, d, ev_day, next_d)] = d

        done_count = 0
        for future in as_completed(futures):
            d = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                import traceback
                print(f"  [{d}] ERROR: {exc}", flush=True)
                traceback.print_exc()
            done_count += 1
            elapsed = time.monotonic() - t_total
            rate = elapsed / done_count
            eta = rate * (total_days - done_count)
            print(f"  >> {done_count}/{total_days} days  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"sims={len(all_results):,}",
                  flush=True)

    total_elapsed = time.monotonic() - t_total

    sim_df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / symbol / "tradesim_stage41.parquet"
    sim_df.to_parquet(out_path, index=False)

    # Also CSV for inspection
    csv_path = OUTPUT_DIR / symbol / "tradesim_stage41.csv"
    sim_df.to_csv(csv_path, index=False)

    n_events = sim_df["event_id"].nunique()
    n_sims = len(sim_df)
    print(f"\nDone: {n_events:,} events × {len(SCENARIOS)} scenarios = {n_sims:,} sims "
          f"in {total_elapsed:.0f}s")
    print(f"Saved: {out_path}")
    print(f"Saved: {csv_path}")

    # ===== REPORTS =====
    report_a_profitability(sim_df)
    report_b_time_to_touch(sim_df)
    report_c_makers_curse(sim_df)
    report_d_delay_sensitivity(sim_df)

    # ===== VERDICT =====
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    # Check if any config has positive net EV per signal
    group_cols = ["scenario", "limit_type", "tp_type", "sl_bp", "to_s", "decision_delay_ms"]
    best_ev = -999
    best_config = None
    for key, grp in sim_df.groupby(group_cols):
        n_sig = len(grp)
        filled = grp[grp["filled"] == 1]
        if len(filled) == 0:
            continue
        ev = filled["net_pnl_bp"].sum() / n_sig
        if ev > best_ev:
            best_ev = ev
            best_config = key

    if best_ev > 0:
        print(f"  ✅ POSITIVE EV found: {best_ev:+.2f} bp/signal")
        print(f"     Config: {best_config}")
        print(f"     MR-class: ALIVE (conditional on further validation)")
    else:
        print(f"  ❌ Best EV: {best_ev:+.2f} bp/signal")
        print(f"     Config: {best_config}")
        print(f"  MR-class: CLOSED at current fee structure")


if __name__ == "__main__":
    main()
