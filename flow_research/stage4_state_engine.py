#!/usr/bin/env python3
"""
Stage-4: Online State Engine (100ms ticks)

For each Stage-1 event, builds a 100ms-resolution feature grid from t0 to t0+5s,
computes online scores (Exhaustion, Refill, OneSided), runs a state machine,
and evaluates separation against outcome labels at 60s/300s.

Trade simulations (taker retrace + maker MR) test feasibility.

All features are CAUSAL: only data ≤ t_k is used at tick k.
VacuumScore is NOT used as a trigger.

Usage:
  python stage4_state_engine.py DOGEUSDT [--workers 8]
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

# Grid parameters
TICK_MS = 100                    # 100ms ticks
N_TICKS_DENSE = 50              # 5s dense grid (50 × 100ms)
OUTCOME_HORIZON_S = 300.0       # outcomes measured up to 300s
OB_LEVELS_K = 5                 # L1-L5 for depth features

# Score thresholds (initial, tuned by separation not PnL)
TH_EXHAUST = 0.5
TH_REFILL = 0.5
TH_ONESIDED = 0.5
STATE_HOLD_TICKS = 3            # 300ms persistence required


# -----------------------------------------------------------------------
# Data loading (reused from prior stages, trimmed to essentials)
# -----------------------------------------------------------------------

def load_ob_arrays_l5(filepath: Path):
    """Load OB as arrays. Returns ts, bid_depth_l5, ask_depth_l5,
    best_bid, best_ask, spread, mid, total_depth_l5."""
    ts_l, bd_l, ad_l, bb_l, ba_l, sp_l, mi_l = [], [], [], [], [], [], []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            ts = obj["ts"] / 1000.0
            data = obj["data"]
            bids = data["b"][:OB_LEVELS_K]
            asks = data["a"][:OB_LEVELS_K]
            bid_n = sum(float(b[1]) * float(b[0]) for b in bids)
            ask_n = sum(float(a[1]) * float(a[0]) for a in asks)
            bb = float(bids[0][0]) if bids else 0.0
            ba = float(asks[0][0]) if asks else 0.0
            spread = max(ba - bb, 0.0)
            mid = (ba + bb) / 2.0 if bb > 0 and ba > 0 else max(ba, bb)
            ts_l.append(ts)
            bd_l.append(bid_n)
            ad_l.append(ask_n)
            bb_l.append(bb)
            ba_l.append(ba)
            sp_l.append(spread)
            mi_l.append(mid)
            if (i + 1) % 200000 == 0:
                print(f"    OB {i+1:,}...", flush=True)
    return {
        "ts": np.array(ts_l, dtype=np.float64),
        "bid_depth": np.array(bd_l, dtype=np.float64),
        "ask_depth": np.array(ad_l, dtype=np.float64),
        "best_bid": np.array(bb_l, dtype=np.float64),
        "best_ask": np.array(ba_l, dtype=np.float64),
        "spread": np.array(sp_l, dtype=np.float64),
        "mid": np.array(mi_l, dtype=np.float64),
    }


def load_trades_arrays(filepath: Path):
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
# Feature computation at a single tick (all data ≤ t_k)
# -----------------------------------------------------------------------

def get_ob_at_t(ob, t):
    """Get latest OB snapshot at or before time t. Returns dict or None."""
    idx = np.searchsorted(ob["ts"], t, side="right") - 1
    if idx < 0:
        return None
    return {
        "ts": ob["ts"][idx],
        "bid_depth": ob["bid_depth"][idx],
        "ask_depth": ob["ask_depth"][idx],
        "best_bid": ob["best_bid"][idx],
        "best_ask": ob["best_ask"][idx],
        "spread": ob["spread"][idx],
        "mid": ob["mid"][idx],
        "age_ms": (t - ob["ts"][idx]) * 1000.0,
    }


def agg_trades_window(tr, t_start, t_end):
    """Aggregate trades in [t_start, t_end). Returns buy_not, sell_not, count."""
    i0 = np.searchsorted(tr["ts"], t_start, side="left")
    i1 = np.searchsorted(tr["ts"], t_end, side="left")
    if i0 >= i1:
        return 0.0, 0.0, 0
    sides = tr["side"][i0:i1]
    nots = tr["notional"][i0:i1]
    buy_mask = sides == 1
    sell_mask = sides == -1
    return float(nots[buy_mask].sum()), float(nots[sell_mask].sum()), int(i1 - i0)


def find_pre_shock_depth(ob, t0, lookback=5.0):
    """Find pre-shock depth baseline: last valid-spread snapshot in [t0-lookback, t0)."""
    t_start = t0 - lookback
    i0 = np.searchsorted(ob["ts"], t_start, side="left")
    i1 = np.searchsorted(ob["ts"], t0, side="left")
    if i0 >= i1:
        return None
    # Search backward from t0 for valid spread
    for idx in range(i1 - 1, i0 - 1, -1):
        if ob["spread"][idx] > 0:
            return float(ob["bid_depth"][idx] + ob["ask_depth"][idx])
    return None


def compute_tick_features(ob, tr, t_k, t0, direction_sign, ob_at_t0, depth_baseline):
    """Compute all features at tick t_k. Strictly causal: uses data ≤ t_k."""
    feat = {}
    feat["t_k"] = t_k
    feat["elapsed_ms"] = round((t_k - t0) * 1000)

    # --- OB features ---
    snap = get_ob_at_t(ob, t_k)
    if snap is None:
        return None

    top_depth_k = snap["bid_depth"] + snap["ask_depth"]
    feat["top_depth_k"] = top_depth_k
    feat["spread_k"] = snap["spread"]
    feat["mid_k"] = snap["mid"]
    feat["book_age_ms"] = snap["age_ms"]
    feat["best_bid_k"] = snap["best_bid"]
    feat["best_ask_k"] = snap["best_ask"]

    # Depth recovery relative to PRE-SHOCK baseline (not blown-book t0)
    feat["depth_recovery"] = top_depth_k / (depth_baseline + 1e-9)

    # Spread normalized to max spread seen so far in [t0, t_k]
    # (computed later in batch for efficiency — placeholder)
    feat["spread_raw"] = snap["spread"]

    # --- Trade features (various windows) ---
    # Last 100ms
    b100, s100, n100 = agg_trades_window(tr, t_k - 0.1, t_k)
    feat["agg_buy_100ms"] = b100
    feat["agg_sell_100ms"] = s100
    feat["agg_total_100ms"] = b100 + s100

    # Last 500ms
    b500, s500, n500 = agg_trades_window(tr, t_k - 0.5, t_k)
    feat["agg_buy_500ms"] = b500
    feat["agg_sell_500ms"] = s500
    feat["agg_total_500ms"] = b500 + s500

    # Last 1s
    b1s, s1s, n1s = agg_trades_window(tr, t_k - 1.0, t_k)
    feat["agg_buy_1s"] = b1s
    feat["agg_sell_1s"] = s1s
    feat["agg_total_1s"] = b1s + s1s

    # Last 3s
    b3s, s3s, n3s = agg_trades_window(tr, t_k - 3.0, t_k)
    feat["agg_buy_3s"] = b3s
    feat["agg_sell_3s"] = s3s
    feat["agg_total_3s"] = b3s + s3s

    # Imbalance (1s)
    total_1s = b1s + s1s
    feat["imbalance_flow_1s"] = (b1s - s1s) / (total_1s + 1e-9) * direction_sign
    # same_side_share (1s)
    if direction_sign == 1:
        feat["same_side_share_1s"] = b1s / (total_1s + 1e-9)
    else:
        feat["same_side_share_1s"] = s1s / (total_1s + 1e-9)

    # FlowRate_1s ($/sec)
    feat["flow_rate_1s"] = total_1s

    # FlowDecay: last 500ms / prev 500ms
    b_prev500, s_prev500, _ = agg_trades_window(tr, t_k - 1.0, t_k - 0.5)
    prev_total = b_prev500 + s_prev500
    feat["flow_decay"] = (b500 + s500) / (prev_total + 1e-9) if prev_total > 0 else 1.0
    feat["flow_decay"] = min(feat["flow_decay"], 5.0)  # clip extreme

    return feat


# -----------------------------------------------------------------------
# Batch feature computation for one event (dense 100ms grid)
# -----------------------------------------------------------------------

def compute_event_grid(ob, tr, t0, direction_sign, mid_t0):
    """Build 100ms feature grid for one event. Returns list of dicts."""
    ob_at_t0 = get_ob_at_t(ob, t0)
    if ob_at_t0 is None:
        return None

    # Pre-shock depth baseline (valid-spread snapshot before t0)
    depth_baseline = find_pre_shock_depth(ob, t0)
    if depth_baseline is None or depth_baseline < 100:
        # Fallback: use t0 depth if pre-shock not available
        depth_baseline = ob_at_t0["bid_depth"] + ob_at_t0["ask_depth"]
    if depth_baseline < 100:
        return None

    ticks = []
    max_spread_so_far = 0.0
    prev_depth = None
    prev_mid = None

    for k in range(N_TICKS_DENSE + 1):  # 0..50 = 0ms..5000ms
        t_k = t0 + k * (TICK_MS / 1000.0)
        feat = compute_tick_features(ob, tr, t_k, t0, direction_sign, ob_at_t0, depth_baseline)
        if feat is None:
            ticks.append(None)
            continue

        # SpreadNorm: spread / max(spread seen so far)
        max_spread_so_far = max(max_spread_so_far, feat["spread_raw"])
        feat["spread_norm"] = feat["spread_raw"] / (max_spread_so_far + 1e-12)

        # SpreadContraction: prev_spread - current_spread
        if k > 0 and ticks[-1] is not None:
            feat["spread_contraction"] = ticks[-1]["spread_raw"] - feat["spread_raw"]
        else:
            feat["spread_contraction"] = 0.0

        # dDepth: depth change rate
        if prev_depth is not None:
            feat["d_depth"] = (feat["top_depth_k"] - prev_depth) / (TICK_MS / 1000.0)
        else:
            feat["d_depth"] = 0.0
        prev_depth = feat["top_depth_k"]

        # MidVel: mid velocity in bp/sec (direction-signed)
        if prev_mid is not None and prev_mid > 0:
            mid_change_bps = (feat["mid_k"] - prev_mid) / prev_mid * 10000
            feat["mid_vel_bps"] = mid_change_bps / (TICK_MS / 1000.0) * direction_sign
        else:
            feat["mid_vel_bps"] = 0.0
        prev_mid = feat["mid_k"]

        # Price return from t0 (direction-signed)
        if mid_t0 > 0:
            feat["ret_from_t0_bps"] = direction_sign * (feat["mid_k"] - mid_t0) / mid_t0 * 10000
        else:
            feat["ret_from_t0_bps"] = 0.0

        ticks.append(feat)

    return ticks


# -----------------------------------------------------------------------
# Online scores (no ML, robust z-score-like normalization)
# -----------------------------------------------------------------------

def compute_scores(ticks):
    """Compute ExhaustionScore, RefillScore, OneSidedScore for each tick.
    Normalization: use running stats from ticks seen so far (causal)."""
    if not ticks or ticks[0] is None:
        return ticks

    # Collect raw signal arrays
    n = len(ticks)
    flow_decay = np.array([t["flow_decay"] if t else 1.0 for t in ticks])
    flow_rate = np.array([t["flow_rate_1s"] if t else 0.0 for t in ticks])
    depth_rec = np.array([t["depth_recovery"] if t else 1.0 for t in ticks])
    d_depth = np.array([t["d_depth"] if t else 0.0 for t in ticks])
    spread_norm = np.array([t["spread_norm"] if t else 1.0 for t in ticks])
    spread_contr = np.array([t["spread_contraction"] if t else 0.0 for t in ticks])
    imb_flow = np.array([t["imbalance_flow_1s"] if t else 0.0 for t in ticks])
    same_side = np.array([t["same_side_share_1s"] if t else 0.5 for t in ticks])
    mid_vel = np.array([t["mid_vel_bps"] if t else 0.0 for t in ticks])

    for k in range(n):
        if ticks[k] is None:
            continue

        # --- ExhaustionScore ---
        # High when: flow_decay low (aggression fading), flow_rate dropping,
        #            depth_recovery still low (MM haven't returned)
        # Signals: low flow_decay, low flow_rate relative to peak, low depth_recovery
        peak_flow_rate = max(flow_rate[:k+1].max(), 1.0)
        flow_rate_frac = flow_rate[k] / peak_flow_rate  # 0..1, lower = more exhausted
        decay_signal = np.clip(1.0 - flow_decay[k], 0, 1)  # high when decay < 1
        depth_low = np.clip(1.0 - depth_rec[k], 0, 1)  # high when depth still low

        exhaust = (0.4 * decay_signal +
                   0.3 * (1.0 - flow_rate_frac) +
                   0.3 * depth_low)
        ticks[k]["exhaustion_score"] = round(exhaust, 4)

        # --- RefillScore ---
        # High when: depth recovering TOWARDS pre-shock level, d_depth positive,
        #            spread contracting, flow_rate NOT increasing
        # depth_rec uses pre-shock baseline, so 1.0 = fully recovered
        # Below 0.3 = very depleted, above 0.8 = mostly recovered
        depth_rising = np.clip((depth_rec[k] - 0.3) / 0.7, 0, 1)  # 0 at ≤0.3, 1 at ≥1.0
        # d_depth positive = depth actively growing ($/sec, typical ~$100K-1M/s)
        d_depth_pos = np.clip(d_depth[k] / 5e5, 0, 1)
        spread_shrink = np.clip(1.0 - spread_norm[k], 0, 1)
        flow_not_rising = np.clip(1.0 - flow_rate_frac, 0, 1)

        refill = (0.35 * depth_rising +
                  0.25 * d_depth_pos +
                  0.25 * spread_shrink +
                  0.15 * flow_not_rising)
        refill = np.clip(refill, 0, 1)
        ticks[k]["refill_score"] = round(refill, 4)

        # --- OneSidedScore ---
        # High when: imbalance still in event direction, same_side high
        imb_signal = np.clip(imb_flow[k], 0, 1)
        same_signal = np.clip((same_side[k] - 0.5) * 2, 0, 1)

        onesided = 0.5 * imb_signal + 0.5 * same_signal
        ticks[k]["onesided_score"] = round(onesided, 4)

    return ticks


# -----------------------------------------------------------------------
# State machine
# -----------------------------------------------------------------------

def run_state_machine(ticks):
    """Assign states: Shock, Exhaustion, Refill, RetraceLeg, Continuation."""
    if not ticks:
        return ticks

    n = len(ticks)
    exhaust_count = 0
    refill_count = 0
    current_state = "Shock"

    for k in range(n):
        if ticks[k] is None:
            ticks[k] = {"state": "Unknown"}
            continue

        e_score = ticks[k].get("exhaustion_score", 0)
        r_score = ticks[k].get("refill_score", 0)
        o_score = ticks[k].get("onesided_score", 0)
        mid_vel = ticks[k].get("mid_vel_bps", 0)

        # Shock phase: first 200ms
        if k <= 1:
            ticks[k]["state"] = "Shock"
            continue

        # Count consecutive ticks above threshold
        if e_score > TH_EXHAUST:
            exhaust_count += 1
        else:
            exhaust_count = 0

        if r_score > TH_REFILL:
            refill_count += 1
        else:
            refill_count = 0

        # State transitions
        if current_state == "Shock":
            if exhaust_count >= STATE_HOLD_TICKS:
                current_state = "Exhaustion"
            elif o_score > TH_ONESIDED and exhaust_count < 1:
                current_state = "Continuation"

        elif current_state == "Exhaustion":
            if refill_count >= STATE_HOLD_TICKS:
                current_state = "Refill"
            elif o_score > TH_ONESIDED and e_score < TH_EXHAUST:
                current_state = "Continuation"

        elif current_state == "Refill":
            # RetraceLeg: mid moving against event direction + refill active
            if mid_vel < -5.0 and r_score > TH_REFILL * 0.8:
                current_state = "RetraceLeg"
            elif o_score > TH_ONESIDED and r_score < TH_REFILL * 0.5:
                current_state = "Continuation"

        elif current_state == "RetraceLeg":
            if o_score > TH_ONESIDED and r_score < TH_REFILL * 0.5:
                current_state = "Continuation"

        elif current_state == "Continuation":
            if exhaust_count >= STATE_HOLD_TICKS:
                current_state = "Exhaustion"

        ticks[k]["state"] = current_state

    return ticks


# -----------------------------------------------------------------------
# Outcome labels (post-hoc, only for evaluation)
# -----------------------------------------------------------------------

def compute_outcomes(ob, tr, t0, direction_sign, mid_t0):
    """Compute outcome labels at 60s and 300s horizons."""
    outcomes = {}

    # Get OB path for outcomes (60s and 300s from t0)
    for hz, label in [(60, "60s"), (300, "300s")]:
        t_end = t0 + hz
        i0 = np.searchsorted(ob["ts"], t0, side="left")
        i1 = np.searchsorted(ob["ts"], t_end, side="right")
        if i0 >= i1 or mid_t0 <= 0:
            outcomes[f"ret_dir_{label}"] = None
            outcomes[f"ret_opp_{label}"] = None
            outcomes[f"touch_t0_{label}"] = None
            outcomes[f"mae_{label}"] = None
            outcomes[f"mfe_{label}"] = None
            continue

        path_mid = ob["mid"][i0:i1]
        path_spread = ob["spread"][i0:i1]
        valid = path_spread > 0
        if valid.sum() < 5:
            outcomes[f"ret_dir_{label}"] = None
            outcomes[f"ret_opp_{label}"] = None
            outcomes[f"touch_t0_{label}"] = None
            outcomes[f"mae_{label}"] = None
            outcomes[f"mfe_{label}"] = None
            continue

        path_mid_v = path_mid[valid]
        rets = direction_sign * (path_mid_v - mid_t0) / mid_t0 * 10000

        # ret in event direction (positive = continuation)
        final_ret = float(rets[-1]) if len(rets) > 0 else 0
        outcomes[f"ret_dir_{label}"] = round(final_ret, 2)
        # ret in opposite direction (mean reversion leg — negative of above)
        outcomes[f"ret_opp_{label}"] = round(-final_ret, 2)
        # Touch t0: did price cross back to t0 level?
        # direction_sign * (mid - mid_t0) < 0 means price went opposite to event
        outcomes[f"touch_t0_{label}"] = bool((rets < 0).any())
        outcomes[f"mfe_{label}"] = round(float(rets.max()), 2)
        outcomes[f"mae_{label}"] = round(float(rets.min()), 2)

    # MR leg metrics: max move in opposite direction within 60s
    i0 = np.searchsorted(ob["ts"], t0, side="left")
    i1 = np.searchsorted(ob["ts"], t0 + 60, side="right")
    if i0 < i1 and mid_t0 > 0:
        pm = ob["mid"][i0:i1]
        ps = ob["spread"][i0:i1]
        valid = ps > 0
        if valid.sum() > 5:
            rets_opp = -direction_sign * (pm[valid] - mid_t0) / mid_t0 * 10000
            outcomes["mr_max_60s"] = round(float(rets_opp.max()), 2)
            outcomes["mr_med_60s"] = round(float(np.median(rets_opp)), 2)
        else:
            outcomes["mr_max_60s"] = None
            outcomes["mr_med_60s"] = None
    else:
        outcomes["mr_max_60s"] = None
        outcomes["mr_med_60s"] = None

    return outcomes


# -----------------------------------------------------------------------
# Trade simulations
# -----------------------------------------------------------------------

def simulate_trades(ticks, ob, tr, t0, direction_sign, mid_t0, outcomes):
    """Run Scenario A (taker) and Scenario B (maker) trade simulations."""
    sims = {}
    if not ticks:
        return sims

    # Find t_refill: first tick where refill_score stays above threshold for 300ms
    t_refill = None
    refill_hold = 0
    for tick in ticks:
        if tick is None or "refill_score" not in tick:
            refill_hold = 0
            continue
        if tick["refill_score"] > TH_REFILL:
            refill_hold += 1
            if refill_hold >= STATE_HOLD_TICKS and t_refill is None:
                t_refill = tick["t_k"]
        else:
            refill_hold = 0

    # Find t_exhaust similarly
    t_exhaust = None
    exhaust_hold = 0
    for tick in ticks:
        if tick is None or "exhaustion_score" not in tick:
            exhaust_hold = 0
            continue
        if tick["exhaustion_score"] > TH_EXHAUST:
            exhaust_hold += 1
            if exhaust_hold >= STATE_HOLD_TICKS and t_exhaust is None:
                t_exhaust = tick["t_k"]
        else:
            exhaust_hold = 0

    sims["t_exhaust"] = round(t_exhaust - t0, 3) if t_exhaust else None
    sims["t_refill"] = round(t_refill - t0, 3) if t_refill else None

    # --- Scenario A: Taker retrace entry ---
    # Entry: after Refill confirmed, enter OPPOSITE to event direction (catch retrace)
    # TP: touch mid(t0) | SL: fixed bp
    TAKER_FEE_RT = 20.0  # bps
    SL_LEVELS = [20, 30, 40]

    if t_refill and mid_t0 > 0:
        # Entry at t_refill, opposite direction
        snap_entry = get_ob_at_t(ob, t_refill)
        if snap_entry and snap_entry["mid"] > 0 and snap_entry["spread"] > 0:
            entry_mid = snap_entry["mid"]
            # Return in OPPOSITE direction from entry (we're betting on retrace)
            # "retrace" = price moving AGAINST the event direction = towards t0
            # If event was BUY (price went up), retrace = price goes down
            # We SHORT (opposite to event) and profit if price drops back to t0

            # Track path from t_refill
            i0 = np.searchsorted(ob["ts"], t_refill, side="left")
            for hz_label, hz in [("30s", 30), ("60s", 60)]:
                i1 = np.searchsorted(ob["ts"], t_refill + hz, side="right")
                if i0 >= i1:
                    continue
                pm = ob["mid"][i0:i1]
                ps = ob["spread"][i0:i1]
                valid = ps > 0
                if valid.sum() < 5:
                    continue
                pm_v = pm[valid]
                # Returns in OPPOSITE direction from entry
                # If event BUY (dir=+1), we SHORT: profit = -(mid - entry)/entry
                # = -direction_sign * (mid - entry) / entry * 10000
                rets_opp = -direction_sign * (pm_v - entry_mid) / entry_mid * 10000

                # TP at touch of mid(t0)
                # Distance from entry to t0 in opposite direction
                tp_dist = -direction_sign * (mid_t0 - entry_mid) / entry_mid * 10000
                # Did it hit TP?
                hit_tp = (rets_opp >= tp_dist).any() if tp_dist > 0 else False

                sims[f"scA_{hz_label}_tp_dist"] = round(tp_dist, 2)
                sims[f"scA_{hz_label}_hit_tp"] = bool(hit_tp)
                sims[f"scA_{hz_label}_mfe"] = round(float(rets_opp.max()), 2)
                sims[f"scA_{hz_label}_mae"] = round(float(rets_opp.min()), 2)
                sims[f"scA_{hz_label}_final"] = round(float(rets_opp[-1]), 2)
                sims[f"scA_{hz_label}_gross"] = round(float(rets_opp[-1]), 2)
                sims[f"scA_{hz_label}_net"] = round(float(rets_opp[-1]) - TAKER_FEE_RT, 2)

                # With SL
                for sl in SL_LEVELS:
                    hit_sl = (rets_opp <= -sl).any()
                    if hit_sl and not hit_tp:
                        pnl = -sl - TAKER_FEE_RT
                    elif hit_tp:
                        pnl = tp_dist - TAKER_FEE_RT
                    else:
                        pnl = float(rets_opp[-1]) - TAKER_FEE_RT
                    sims[f"scA_{hz_label}_sl{sl}_pnl"] = round(pnl, 2)

    # --- Scenario B: Maker MR entry ---
    # Place limit order at mid(t0) AFTER refill signal, opposite direction
    # Fill model: volume through level >= visible depth
    MAKER_FEE_RT = 4.0  # bps

    if t_refill and mid_t0 > 0:
        order_time = t_refill
        entry_price = mid_t0  # limit at t0 level

        # Queue ahead estimate: visible depth at the level at order time
        snap_order = get_ob_at_t(ob, order_time)
        if snap_order:
            if direction_sign == 1:
                # Event was BUY, we want to SELL at mid(t0) (which is below current)
                # Queue = ask depth at that level
                queue_ahead = snap_order["ask_depth"]
            else:
                queue_ahead = snap_order["bid_depth"]

            # Track volume through the level in 60s after order
            queue_window = 60.0
            tr_i0 = np.searchsorted(tr["ts"], order_time, side="left")
            tr_i1 = np.searchsorted(tr["ts"], order_time + queue_window, side="left")

            if tr_i0 < tr_i1:
                w_sides = tr["side"][tr_i0:tr_i1]
                w_nots = tr["notional"][tr_i0:tr_i1]
                w_prices = tr["price"][tr_i0:tr_i1]

                # Volume of opposite-direction trades at or through our level
                if direction_sign == 1:  # we SELL, need BUY trades at >= entry_price
                    fill_mask = (w_sides == 1) & (w_prices >= entry_price)
                else:  # we BUY, need SELL trades at <= entry_price
                    fill_mask = (w_sides == -1) & (w_prices <= entry_price)

                vol_through = float(w_nots[fill_mask].sum())
                fill_ratio = vol_through / max(queue_ahead, 1000.0)
                filled = fill_ratio > 1.0

                sims["scB_queue_ahead"] = round(queue_ahead, 2)
                sims["scB_vol_through"] = round(vol_through, 2)
                sims["scB_fill_ratio"] = round(fill_ratio, 4)
                sims["scB_filled"] = bool(filled)

                # If filled, compute PnL
                if filled:
                    # Find approximate fill time (when cumulative vol > queue)
                    cum_vol = np.cumsum(w_nots[fill_mask])
                    fill_idx_in_mask = np.searchsorted(cum_vol, queue_ahead)
                    fill_indices = np.where(fill_mask)[0]
                    if fill_idx_in_mask < len(fill_indices):
                        fill_trade_idx = fill_indices[fill_idx_in_mask] + tr_i0
                        fill_time = tr["ts"][fill_trade_idx]
                    else:
                        fill_time = order_time + queue_window
                    sims["scB_fill_time_s"] = round(fill_time - t0, 2)

                    # Return from entry_price to various horizons after fill
                    for hz in [30, 60, 300]:
                        ob_idx = np.searchsorted(ob["ts"], fill_time + hz, side="right") - 1
                        if 0 <= ob_idx < len(ob["ts"]) and ob["spread"][ob_idx] > 0:
                            exit_mid = ob["mid"][ob_idx]
                            # We entered opposite to event direction
                            ret = -direction_sign * (exit_mid - entry_price) / entry_price * 10000
                            sims[f"scB_ret_{hz}s"] = round(ret, 2)
                            sims[f"scB_net_{hz}s"] = round(ret - MAKER_FEE_RT, 2)

    return sims


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
    ob = load_ob_arrays_l5(ob_path)
    tr = load_trades_arrays(tr_path)

    # Append next-day data for boundary events
    if next_date_str:
        next_ob = DATA_DIR / symbol / f"{next_date_str}_orderbook.jsonl"
        next_tr = DATA_DIR / symbol / f"{next_date_str}_trades.csv"
        if next_ob.exists() and next_tr.exists():
            print(f"  [{date_str}] + next day boundary...", flush=True)
            ob_n = load_ob_arrays_l5(next_ob)
            tr_n = load_trades_arrays(next_tr)
            for k in ob:
                ob[k] = np.concatenate([ob[k], ob_n[k]])
            for k in tr:
                tr[k] = np.concatenate([tr[k], tr_n[k]])

    print(f"  [{date_str}] Processing {len(events_df)} events...", flush=True)

    results = []
    for _, ev in events_df.iterrows():
        t0 = ev["t0"]
        direction_sign = 1 if ev["direction"] == "BUY" else -1
        mid_t0 = ev["mid"] if ev["mid"] > 0 else ev["price_at_t0"]

        # Skip if mid_t0 looks bad
        if mid_t0 <= 0:
            results.append({"event_id": ev["event_id"], "engine_ok": False})
            continue

        # 1. Feature grid (0-5s, 100ms)
        ticks = compute_event_grid(ob, tr, t0, direction_sign, mid_t0)
        if ticks is None:
            results.append({"event_id": ev["event_id"], "engine_ok": False})
            continue

        # 2. Scores
        ticks = compute_scores(ticks)

        # 3. State machine
        ticks = run_state_machine(ticks)

        # 4. Outcomes
        outcomes = compute_outcomes(ob, tr, t0, direction_sign, mid_t0)

        # 5. Trade sims
        sims = simulate_trades(ticks, ob, tr, t0, direction_sign, mid_t0, outcomes)

        # Aggregate tick-level info
        valid_ticks = [t for t in ticks if t is not None and "exhaustion_score" in t]
        if not valid_ticks:
            results.append({"event_id": ev["event_id"], "engine_ok": False})
            continue

        # Summary features from tick grid
        agg = {
            "event_id": ev["event_id"],
            "engine_ok": True,
            "direction": ev["direction"],
            "t0": t0,
            "mid_t0": mid_t0,
            "flow_impact": ev["flow_impact"],
            "imbalance": ev["imbalance"],
            "same_side_share": ev["same_side_share"],
            "agg_total": ev["agg_total"],
            "top_depth_t0": ev["top_depth"],
            "spread_t0": ev["spread"],
        }

        # Score time series summary
        e_scores = [t["exhaustion_score"] for t in valid_ticks]
        r_scores = [t["refill_score"] for t in valid_ticks]
        o_scores = [t["onesided_score"] for t in valid_ticks]

        agg["exhaust_max"] = round(max(e_scores), 4)
        agg["exhaust_mean"] = round(np.mean(e_scores), 4)
        agg["refill_max"] = round(max(r_scores), 4)
        agg["refill_mean"] = round(np.mean(r_scores), 4)
        agg["onesided_max"] = round(max(o_scores), 4)
        agg["onesided_mean"] = round(np.mean(o_scores), 4)

        # State summary
        states = [t.get("state", "Unknown") for t in valid_ticks]
        for s in ["Shock", "Exhaustion", "Refill", "RetraceLeg", "Continuation"]:
            agg[f"state_{s}_frac"] = round(states.count(s) / len(states), 4)

        # Final state
        agg["final_state"] = states[-1] if states else "Unknown"

        # Depth/spread at tick 10 (1s) and 50 (5s) if available
        for tk_idx, tk_label in [(10, "1s"), (30, "3s"), (50, "5s")]:
            if tk_idx < len(ticks) and ticks[tk_idx] is not None:
                agg[f"depth_recovery_{tk_label}"] = ticks[tk_idx].get("depth_recovery", None)
                agg[f"flow_rate_{tk_label}"] = ticks[tk_idx].get("flow_rate_1s", None)
                agg[f"flow_decay_{tk_label}"] = ticks[tk_idx].get("flow_decay", None)
                agg[f"spread_norm_{tk_label}"] = ticks[tk_idx].get("spread_norm", None)

        # Merge outcomes and sims
        agg.update(outcomes)
        agg.update(sims)

        results.append(agg)

    elapsed = time.monotonic() - t0_wall
    ok_count = sum(1 for r in results if r.get("engine_ok"))
    print(f"  [{date_str}] DONE: {ok_count}/{len(results)}, {elapsed:.1f}s", flush=True)
    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage-4: Online State Engine")
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
    print(f"ALL Stage-1 events: {len(df):,}")
    print(f"Days: {len(dates)}, Workers: {workers}")
    print(f"Tick: {TICK_MS}ms, Dense grid: {N_TICKS_DENSE} ticks (5s)")
    print(f"Thresholds: Exhaust={TH_EXHAUST}, Refill={TH_REFILL}, OneSided={TH_ONESIDED}")
    print("=" * 60, flush=True)

    t_total = time.monotonic()

    # Date -> next date
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
                if last_t0 + 5 + OUTCOME_HORIZON_S > day_end:
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
            print(f"  >> {done_count}/{total_days} days  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    total_elapsed = time.monotonic() - t_total

    # Build result DataFrame
    res_df = pd.DataFrame(all_results)
    ok_df = res_df[res_df["engine_ok"] == True].copy()

    out_path = OUTPUT_DIR / symbol / "events_stage4.parquet"
    ok_df.to_parquet(out_path, index=False)
    print(f"\nProcessed: {len(ok_df):,} / {len(res_df):,} in {total_elapsed:.0f}s")
    print(f"Saved: {out_path}")

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    analyze(ok_df)


def analyze(df):
    print(f"\n{'='*60}")
    print("STAGE-4: ONLINE STATE ENGINE ANALYSIS")
    print(f"{'='*60}")
    n = len(df)

    # --- 1. Score distributions ---
    print(f"\n--- 1. SCORE DISTRIBUTIONS (n={n}) ---")
    for col in ["exhaust_max", "exhaust_mean", "refill_max", "refill_mean",
                 "onesided_max", "onesided_mean"]:
        v = df[col].dropna()
        print(f"  {col:20s}: med={v.median():.3f}  p25={v.quantile(0.25):.3f}  p75={v.quantile(0.75):.3f}")

    # --- 2. State fractions ---
    print(f"\n--- 2. STATE FRACTIONS ---")
    for s in ["Shock", "Exhaustion", "Refill", "RetraceLeg", "Continuation"]:
        col = f"state_{s}_frac"
        if col in df.columns:
            v = df[col].dropna()
            print(f"  {s:15s}: mean={v.mean():.3f}  med={v.median():.3f}")
    print(f"\n  Final state distribution:")
    if "final_state" in df.columns:
        print(df["final_state"].value_counts().to_string(header=False))

    # --- 3. Timing of state transitions ---
    print(f"\n--- 3. STATE TRANSITION TIMING ---")
    for col in ["t_exhaust", "t_refill"]:
        v = df[col].dropna()
        if len(v) > 0:
            print(f"  {col}: n={len(v)} ({len(v)/n:.1%})  "
                  f"med={v.median():.2f}s  p25={v.quantile(0.25):.2f}  p75={v.quantile(0.75):.2f}")

    # --- 4. SEPARATION: scores vs outcomes ---
    print(f"\n--- 4. SEPARATION: Online scores vs outcomes ---")

    # TouchT0_60s
    touch_col = "touch_t0_60s"
    if touch_col in df.columns:
        valid = df[df[touch_col].notna()].copy()
        print(f"\n  TouchT0_60s: {valid[touch_col].mean():.1%} baseline (n={len(valid)})")

        # AUC for predicting TouchT0_60s from scores
        for score_col in ["exhaust_max", "refill_max", "onesided_max",
                          "exhaust_mean", "refill_mean",
                          "depth_recovery_1s", "flow_decay_1s"]:
            if score_col not in valid.columns:
                continue
            sv = valid[[score_col, touch_col]].dropna()
            if len(sv) < 20:
                continue
            # Simple AUC approximation via rank correlation
            from scipy import stats
            pos = sv[sv[touch_col] == True][score_col]
            neg = sv[sv[touch_col] == False][score_col]
            if len(pos) < 5 or len(neg) < 5:
                continue
            # Mann-Whitney U → AUC
            u_stat, p_val = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            auc = u_stat / (len(pos) * len(neg))
            # Make sure AUC is directional (we may need to flip)
            auc_adj = max(auc, 1 - auc)
            print(f"    {score_col:25s}: AUC={auc_adj:.3f}  p={p_val:.4f}  "
                  f"pos_med={pos.median():.3f}  neg_med={neg.median():.3f}")

    # Uplift: conditioned WR for mean reversion
    print(f"\n  Uplift: WR for retrace conditioned on early refill")
    mr_col = "mr_max_60s"
    if mr_col in df.columns:
        valid = df[df[mr_col].notna()].copy()
        baseline_wr = (valid[mr_col] > 10).mean()
        print(f"    Baseline P(MR>10bp in 60s): {baseline_wr:.1%}")

        for score_col in ["refill_max", "exhaust_max", "depth_recovery_1s"]:
            if score_col not in valid.columns:
                continue
            sv = valid[valid[score_col].notna()]
            if len(sv) < 20:
                continue
            p75 = sv[score_col].quantile(0.75)
            high = sv[sv[score_col] >= p75]
            low = sv[sv[score_col] < sv[score_col].quantile(0.25)]
            wr_high = (high[mr_col] > 10).mean() if len(high) > 0 else 0
            wr_low = (low[mr_col] > 10).mean() if len(low) > 0 else 0
            print(f"    {score_col:25s}: high_q={wr_high:.1%}  low_q={wr_low:.1%}  "
                  f"uplift={wr_high-wr_low:+.1%}")

    # Median MR move conditioned on refill timing
    print(f"\n  MR move by refill timing:")
    if "t_refill" in df.columns and mr_col in df.columns:
        valid = df[df["t_refill"].notna() & df[mr_col].notna()]
        for lo, hi, label in [(0, 1, "fast 0-1s"), (1, 2, "mid 1-2s"),
                               (2, 3, "late 2-3s"), (3, 5, "very late 3-5s")]:
            sub = valid[(valid["t_refill"] >= lo) & (valid["t_refill"] < hi)]
            if len(sub) < 10:
                continue
            mr = sub[mr_col]
            print(f"    {label:15s}: n={len(sub):5d}  MR_max_med={mr.median():+5.1f}bp  "
                  f"P(MR>10)={(mr>10).mean():.1%}  P(MR>20)={(mr>20).mean():.1%}")

    # No refill detected
    no_refill = df[df["t_refill"].isna()]
    if len(no_refill) > 0 and mr_col in df.columns:
        mr_nr = no_refill[mr_col].dropna()
        if len(mr_nr) > 0:
            print(f"    {'no refill':15s}: n={len(mr_nr):5d}  MR_max_med={mr_nr.median():+5.1f}bp  "
                  f"P(MR>10)={(mr_nr>10).mean():.1%}")

    # --- 5. TRADE SIMULATIONS ---
    print(f"\n--- 5. TRADE SIMULATIONS ---")

    # Scenario A: Taker retrace
    print(f"\n  Scenario A: Taker retrace entry (opposite to event after Refill)")
    for hz in ["30s", "60s"]:
        net_col = f"scA_{hz}_net"
        if net_col not in df.columns:
            continue
        v = df[net_col].dropna()
        if len(v) == 0:
            continue
        print(f"    {hz}: n={len(v)}  med={v.median():+.1f}bp  mean={v.mean():+.1f}bp  "
              f"WR={(v>0).mean():.1%}  "
              f"n/day={len(v)/30:.1f}")

        # With SL
        for sl in [20, 30, 40]:
            sl_col = f"scA_{hz}_sl{sl}_pnl"
            if sl_col in df.columns:
                sv = df[sl_col].dropna()
                if len(sv) > 0:
                    print(f"      SL={sl}: med={sv.median():+.1f}bp  mean={sv.mean():+.1f}bp  "
                          f"WR={(sv>0).mean():.1%}")

    # Scenario B: Maker MR
    print(f"\n  Scenario B: Maker MR (limit at t0 after Refill)")
    if "scB_filled" in df.columns:
        v = df["scB_filled"].dropna()
        fill_rate = v.mean()
        n_filled = v.sum()
        print(f"    Fill rate: {fill_rate:.1%} ({int(n_filled)}/{len(v)})")

        if "scB_fill_ratio" in df.columns:
            fr = df["scB_fill_ratio"].dropna()
            print(f"    Fill ratio: med={fr.median():.2f}  mean={fr.mean():.2f}")

        filled = df[df["scB_filled"] == True]
        if len(filled) > 0:
            for hz in [30, 60, 300]:
                net_col = f"scB_net_{hz}s"
                if net_col in filled.columns:
                    sv = filled[net_col].dropna()
                    if len(sv) > 0:
                        print(f"    Filled, {hz}s: n={len(sv)}  "
                              f"med={sv.median():+.1f}bp  mean={sv.mean():+.1f}bp  "
                              f"WR={(sv>0).mean():.1%}  "
                              f"n/day={len(sv)/30:.1f}")

    # --- 6. AVERAGE TRAJECTORIES: winners vs losers ---
    print(f"\n--- 6. WINNERS vs LOSERS PROFILE ---")
    if "touch_t0_60s" in df.columns:
        winners = df[df["touch_t0_60s"] == True]
        losers = df[df["touch_t0_60s"] == False]
        print(f"    Winners (touch t0): {len(winners)}  Losers (no touch): {len(losers)}")
        for col in ["exhaust_mean", "refill_mean", "onesided_mean",
                     "depth_recovery_1s", "depth_recovery_3s",
                     "flow_decay_1s", "flow_decay_3s",
                     "t_exhaust", "t_refill"]:
            if col not in df.columns:
                continue
            w = winners[col].dropna()
            l = losers[col].dropna()
            if len(w) < 10 or len(l) < 10:
                continue
            print(f"    {col:25s}: win={w.median():.3f}  lose={l.median():.3f}  "
                  f"diff={w.median()-l.median():+.3f}")

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
