#!/usr/bin/env python3
"""
Exit ML Feature Experiments
============================
Systematically test feature additions one-by-one against baseline.

Each experiment:
  1. Build tick features with new feature group enabled
  2. Train LogReg + HGBC on same 70/30 temporal split
  3. Report Test AUC, LOSO AUC, overfit gap
  4. Backtest single-exit with LOSO predictions

Usage:
    python3 research_exit_ml_experiments.py
"""

import copy
import json
import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOCAL_DATA_DIR = Path("charts_settlement")
TICK_MS = 100
MAX_POST_MS = 60000


# ═══════════════════════════════════════════════════════════════════════
# Parse JSONL — extended to rebuild OB.50 snapshots
# ═══════════════════════════════════════════════════════════════════════

def parse_jsonl_full(fp):
    """Parse JSONL with trades, OB.1, OB.50 raw deltas, tickers.

    OB.50 returns raw deltas — caller maintains dict state and computes
    features at tick time. No sorted snapshots stored = fast.
    """
    trades, ob1_updates, ob50_events, tickers = [], [], [], []

    with open(fp) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except:
                continue
            t = msg.get("_t_ms", 0)
            topic = msg.get("topic", "")
            data = msg.get("data", {})
            mtype = msg.get("type", "")

            if "publicTrade" in topic:
                for tr in (data if isinstance(data, list) else [data]):
                    p = float(tr.get("p", 0))
                    q = float(tr.get("v", 0))
                    s = tr.get("S", "")
                    trades.append((t, p, q, s, p * q))

            elif "orderbook.1." in topic:
                b = data.get("b", [])
                a = data.get("a", [])
                if b and a:
                    ob1_updates.append((t, float(b[0][0]), float(b[0][1]),
                                        float(a[0][0]), float(a[0][1])))

            elif "orderbook.50." in topic:
                # Store raw event: (t, type, bid_updates, ask_updates)
                bids_raw = [(float(p), float(q)) for p, q in data.get("b", [])]
                asks_raw = [(float(p), float(q)) for p, q in data.get("a", [])]
                ob50_events.append((t, mtype, bids_raw, asks_raw))

            elif "tickers" in topic:
                fr = float(data.get("fundingRate", 0))
                oi = float(data.get("openInterest", 0))
                tickers.append((t, fr, oi))

    return (sorted(trades, key=lambda x: x[0]),
            sorted(ob1_updates, key=lambda x: x[0]),
            sorted(ob50_events, key=lambda x: x[0]),
            sorted(tickers, key=lambda x: x[0]))


def _compute_ob50_features(ob_bids, ob_asks, ref_price):
    """Compute OB depth features from bid/ask dicts. No expensive sorting —
    uses numpy vectorized ops on dict values."""
    if not ob_bids or not ob_asks:
        return {k: 0 for k in ["ob50_total_bid_depth", "ob50_total_ask_depth",
                                "ob50_depth_imbalance",
                                "ob_bid_depth_10bps", "ob_ask_depth_10bps", "ob_imbalance_10bps",
                                "ob_bid_depth_25bps", "ob_ask_depth_25bps", "ob_imbalance_25bps",
                                "ob_bid_depth_50bps", "ob_ask_depth_50bps", "ob_imbalance_50bps",
                                "ob_bid_wall_ratio", "ob_ask_wall_ratio",
                                "ob_bid_top5_pct", "ob_ask_top5_pct"]}

    # Vectorize: convert dicts to numpy arrays
    bp = np.fromiter(ob_bids.keys(), dtype=np.float64)
    bq = np.fromiter(ob_bids.values(), dtype=np.float64)
    ap = np.fromiter(ob_asks.keys(), dtype=np.float64)
    aq = np.fromiter(ob_asks.values(), dtype=np.float64)

    # Notional depth
    b_notional = bp * bq
    a_notional = ap * aq
    bid_depth = b_notional.sum()
    ask_depth = a_notional.sum()
    total = bid_depth + ask_depth

    feat = {}
    feat["ob50_total_bid_depth"] = bid_depth
    feat["ob50_total_ask_depth"] = ask_depth
    feat["ob50_depth_imbalance"] = (bid_depth - ask_depth) / total if total > 0 else 0

    # Mid price for range calculations
    best_bid = bp.max()
    best_ask = ap.min()
    mid = (best_bid + best_ask) / 2

    # Depth within X bps of mid (vectorized filter)
    for bps_range, rlabel in [(10, "10bps"), (25, "25bps"), (50, "50bps")]:
        lo = mid * (1 - bps_range / 10000)
        hi = mid * (1 + bps_range / 10000)
        bid_in = b_notional[bp >= lo].sum()
        ask_in = a_notional[ap <= hi].sum()
        total_in = bid_in + ask_in
        feat[f"ob_bid_depth_{rlabel}"] = bid_in
        feat[f"ob_ask_depth_{rlabel}"] = ask_in
        feat[f"ob_imbalance_{rlabel}"] = (bid_in - ask_in) / total_in if total_in > 0 else 0

    # Wall detection (max qty vs avg qty — no sort needed)
    avg_bq = bq.mean()
    avg_aq = aq.mean()
    feat["ob_bid_wall_ratio"] = bq.max() / avg_bq if avg_bq > 0 else 0
    feat["ob_ask_wall_ratio"] = aq.max() / avg_aq if avg_aq > 0 else 0

    # Top 5 concentration (partial sort via argpartition — O(n) not O(n log n))
    if len(b_notional) >= 5:
        top5_idx = np.argpartition(-b_notional, 5)[:5]
        feat["ob_bid_top5_pct"] = b_notional[top5_idx].sum() / bid_depth if bid_depth > 0 else 0
    else:
        feat["ob_bid_top5_pct"] = 1.0
    if len(a_notional) >= 5:
        top5_idx = np.argpartition(-a_notional, 5)[:5]
        feat["ob_ask_top5_pct"] = a_notional[top5_idx].sum() / ask_depth if ask_depth > 0 else 0
    else:
        feat["ob_ask_top5_pct"] = 1.0

    return feat


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction — modular by experiment
# ═══════════════════════════════════════════════════════════════════════

def build_tick_features_extended(fp, experiments=None):
    """Build tick features with optional experiment feature groups.

    experiments: set of enabled experiment names, e.g. {"ob_depth", "cvd", "sequence"}
    """
    if experiments is None:
        experiments = set()

    need_ob50 = "ob_depth" in experiments

    if need_ob50:
        trades, ob1, ob50_events, tickers = parse_jsonl_full(fp)
    else:
        # Faster path — skip OB.50 parsing
        from research_exit_ml_v2 import parse_jsonl
        trades, ob1, tickers = parse_jsonl(fp)
        ob50_events = []

    if not trades:
        return None

    stem = fp.stem
    parts = stem.split("_")
    symbol = parts[0]

    pre_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t < 0]
    post_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t >= 0]

    if not pre_trades or len(post_trades) < 10:
        return None

    ref_price = pre_trades[-1][1]
    bps = lambda p: (p / ref_price - 1) * 10000

    # FR
    pre_tickers = [tk for tk in tickers if tk[0] < 0]
    fr_bps = pre_tickers[-1][1] * 10000 if pre_tickers else 0

    # Pre-settlement stats
    pre_10s = [(t, p, q, s, n) for t, p, q, s, n in pre_trades if t >= -10000]
    pre_vol_rate = sum(n for _, _, _, _, n in pre_10s) / 10.0 if pre_10s else 1.0
    pre_trade_rate = len(pre_10s) / 10.0

    pre_ob1 = [o for o in ob1 if -5000 <= o[0] < 0]
    pre_spread_bps = 0
    if pre_ob1:
        _, bp, bq, ap, aq = pre_ob1[-1]
        pre_spread_bps = (ap - bp) / ref_price * 10000

    # Post-trade arrays
    post_sorted = sorted(post_trades, key=lambda x: x[0])
    post_times = np.array([t for t, _, _, _, _ in post_sorted])
    post_prices_bps = np.array([bps(p) for _, p, _, _, _ in post_sorted])
    post_sides = np.array([1 if s == "Sell" else 0 for _, _, _, s, _ in post_sorted])
    post_notionals = np.array([n for _, _, _, _, n in post_sorted])
    post_sizes = np.array([q for _, _, q, _, _ in post_sorted])

    max_t = min(post_sorted[-1][0], MAX_POST_MS)

    # OB.1 arrays
    ob1_post = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1 if t >= 0]
    ob1_times = np.array([t for t, _, _, _, _ in ob1_post]) if ob1_post else np.array([])

    # OB.50: prepare event replay index for efficient seek
    ob50_bids_state = {}  # maintained during tick loop
    ob50_asks_state = {}
    ob50_initialized = False
    ob50_event_idx = 0  # current position in event list

    # Global min for targets
    global_min_bps = post_prices_bps.min()

    # CVD tracking
    cvd_running = 0.0
    cvd_values = []  # (t, cvd)
    if "cvd" in experiments:
        for t, p, q, s, n in post_sorted:
            if s == "Sell":
                cvd_running -= n
            else:
                cvd_running += n
            cvd_values.append((t, cvd_running))

    rows = []
    running_min_bps = 0.0
    last_new_low_t = 0

    # Sequence tracking
    bounce_count = 0
    consecutive_new_lows = 0
    prev_was_new_low = False

    for tick_t in range(0, int(max_t), TICK_MS):
        mask_up_to = post_times <= tick_t
        if mask_up_to.sum() < 2:
            continue

        current_prices = post_prices_bps[mask_up_to]
        current_times = post_times[mask_up_to]
        current_price = current_prices[-1]

        # Running minimum
        new_min = current_prices.min()
        is_new_low = new_min < running_min_bps - 0.5
        if new_min < running_min_bps:
            running_min_bps = new_min
            last_new_low_t = tick_t
        elif tick_t == 0:
            running_min_bps = current_price

        # Sequence tracking
        if is_new_low:
            if not prev_was_new_low:
                consecutive_new_lows = 1
            else:
                consecutive_new_lows += 1
            prev_was_new_low = True
        else:
            if prev_was_new_low and current_price > running_min_bps + 3:
                bounce_count += 1
            consecutive_new_lows = 0
            prev_was_new_low = False

        # ══════════════════════════════════════════════════════════
        # BASELINE FEATURES (same as v2)
        # ══════════════════════════════════════════════════════════
        feat = {}
        feat["t_ms"] = tick_t
        feat["price_bps"] = current_price
        feat["running_min_bps"] = running_min_bps
        feat["distance_from_low_bps"] = current_price - running_min_bps
        feat["new_low"] = 1 if current_price <= running_min_bps + 0.5 else 0
        feat["time_since_new_low_ms"] = tick_t - last_new_low_t
        feat["pct_of_window_elapsed"] = tick_t / MAX_POST_MS

        # Price velocity
        for window, label in [(500, "500ms"), (1000, "1s"), (2000, "2s")]:
            w_mask = (current_times > tick_t - window) & (current_times <= tick_t)
            w_prices = post_prices_bps[mask_up_to][-w_mask.sum():] if w_mask.sum() > 0 else np.array([])
            if len(w_prices) >= 2:
                feat[f"price_velocity_{label}"] = w_prices[-1] - w_prices[0]
            else:
                feat[f"price_velocity_{label}"] = 0

        # Price acceleration
        mask_early = (current_times > tick_t - 1000) & (current_times <= tick_t - 500)
        mask_late = (current_times > tick_t - 500) & (current_times <= tick_t)
        if mask_early.sum() >= 1 and mask_late.sum() >= 1:
            early_p = post_prices_bps[mask_up_to][-mask_late.sum() - mask_early.sum():-mask_late.sum()]
            late_p = post_prices_bps[mask_up_to][-mask_late.sum():]
            v_early = early_p[-1] - early_p[0] if len(early_p) > 1 else 0
            v_late = late_p[-1] - late_p[0] if len(late_p) > 1 else 0
            feat["price_accel"] = v_late - v_early
        else:
            feat["price_accel"] = 0

        # Drop rate
        feat["drop_rate_bps_per_s"] = running_min_bps / (tick_t / 1000) if tick_t > 0 else 0

        # Trade flow
        for window, label in [(500, "500ms"), (1000, "1s"), (2000, "2s"), (5000, "5s")]:
            w_mask = (post_times > tick_t - window) & (post_times <= tick_t)
            w_sides = post_sides[w_mask]
            w_notionals = post_notionals[w_mask]
            w_sizes = post_sizes[w_mask]
            n_trades = w_mask.sum()
            total_vol = w_notionals.sum()
            sell_vol = w_notionals[w_sides == 1].sum()

            feat[f"sell_ratio_{label}"] = sell_vol / total_vol if total_vol > 0 else 0.5
            feat[f"trade_rate_{label}"] = n_trades / (window / 1000)
            feat[f"vol_rate_{label}"] = total_vol / (window / 1000)
            feat[f"avg_size_{label}"] = w_sizes.mean() if n_trades > 0 else 0
            if n_trades > 2:
                feat[f"large_trade_pct_{label}"] = (w_sizes > 2 * np.median(w_sizes)).mean()
            else:
                feat[f"large_trade_pct_{label}"] = 0

        # Surge
        feat["vol_surge"] = feat["vol_rate_1s"] / pre_vol_rate if pre_vol_rate > 0 else 1
        feat["trade_surge"] = feat["trade_rate_1s"] / pre_trade_rate if pre_trade_rate > 0 else 1

        # Buy/sell ratio
        mask_1s = (post_times > tick_t - 1000) & (post_times <= tick_t)
        buy_v = post_notionals[mask_1s & (post_sides == 0)].sum()
        sell_v = post_notionals[mask_1s & (post_sides == 1)].sum()
        feat["buy_sell_ratio_1s"] = buy_v / sell_v if sell_v > 0 else 1.0

        # Trade rate accel
        if tick_t >= 2000:
            r_early = ((post_times > tick_t - 2000) & (post_times <= tick_t - 1000)).sum()
            r_late = ((post_times > tick_t - 1000) & (post_times <= tick_t)).sum()
            feat["trade_rate_accel"] = r_late - r_early
        else:
            feat["trade_rate_accel"] = 0

        # OB.1
        if len(ob1_times) > 0:
            ob_mask = ob1_times <= tick_t
            if ob_mask.sum() > 0:
                idx = ob_mask.sum() - 1
                _, bp, bq, ap, aq = ob1_post[idx]
                spread = (ap - bp) / ref_price * 10000
                feat["spread_bps"] = spread
                feat["spread_change"] = spread - pre_spread_bps
                feat["ob1_bid_qty"] = bq
                feat["ob1_ask_qty"] = aq
                feat["ob1_imbalance"] = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0
                if idx > 0:
                    _, _, bq2, _, aq2 = ob1_post[idx - 1]
                    feat["bid_qty_change"] = bq - bq2
                    feat["ask_qty_change"] = aq - aq2
                else:
                    feat["bid_qty_change"] = feat["ask_qty_change"] = 0
            else:
                for k in ["spread_bps", "spread_change", "ob1_bid_qty", "ob1_ask_qty",
                           "ob1_imbalance", "bid_qty_change", "ask_qty_change"]:
                    feat[k] = 0
        else:
            for k in ["spread_bps", "spread_change", "ob1_bid_qty", "ob1_ask_qty",
                       "ob1_imbalance", "bid_qty_change", "ask_qty_change"]:
                feat[k] = 0

        # Static
        feat["fr_bps"] = fr_bps
        feat["fr_abs_bps"] = abs(fr_bps)
        feat["t_seconds"] = tick_t / 1000.0
        feat["log_t"] = np.log1p(tick_t)
        feat["phase"] = (0 if tick_t < 1000 else 1 if tick_t < 5000 else
                         2 if tick_t < 10000 else 3 if tick_t < 30000 else 4)

        # ══════════════════════════════════════════════════════════
        # EXPERIMENT 1: OB DEPTH FEATURES
        # Replay all OB.50 deltas up to tick_t, then compute from dict
        # ══════════════════════════════════════════════════════════
        if "ob_depth" in experiments and ob50_events:
            # Advance OB state to current tick (replay deltas)
            while ob50_event_idx < len(ob50_events) and ob50_events[ob50_event_idx][0] <= tick_t:
                _, etype, bids_raw, asks_raw = ob50_events[ob50_event_idx]
                if etype == "snapshot":
                    ob50_bids_state = {p: q for p, q in bids_raw}
                    ob50_asks_state = {p: q for p, q in asks_raw}
                    ob50_initialized = True
                elif etype == "delta" and ob50_initialized:
                    for p, q in bids_raw:
                        if q == 0:
                            ob50_bids_state.pop(p, None)
                        else:
                            ob50_bids_state[p] = q
                    for p, q in asks_raw:
                        if q == 0:
                            ob50_asks_state.pop(p, None)
                        else:
                            ob50_asks_state[p] = q
                ob50_event_idx += 1

            if ob50_initialized:
                ob_feat = _compute_ob50_features(ob50_bids_state, ob50_asks_state, ref_price)
                feat.update(ob_feat)
            else:
                feat.update(_compute_ob50_features({}, {}, ref_price))

        # ══════════════════════════════════════════════════════════
        # EXPERIMENT 2: CVD (Cumulative Volume Delta)
        # ══════════════════════════════════════════════════════════
        if "cvd" in experiments and cvd_values:
            # Current CVD
            cvd_at_tick = [v for t_v, v in cvd_values if t_v <= tick_t]
            feat["cvd"] = cvd_at_tick[-1] if cvd_at_tick else 0

            # CVD velocity (change in last 1s, 2s, 5s)
            for window, label in [(1000, "1s"), (2000, "2s"), (5000, "5s")]:
                cvd_before = [v for t_v, v in cvd_values if t_v <= tick_t - window]
                if cvd_before:
                    feat[f"cvd_velocity_{label}"] = feat["cvd"] - cvd_before[-1]
                else:
                    feat[f"cvd_velocity_{label}"] = feat["cvd"]

            # CVD acceleration (2nd derivative)
            cvd_1s_ago = [v for t_v, v in cvd_values if t_v <= tick_t - 1000]
            cvd_2s_ago = [v for t_v, v in cvd_values if t_v <= tick_t - 2000]
            if cvd_1s_ago and cvd_2s_ago:
                v_recent = feat["cvd"] - cvd_1s_ago[-1]
                v_old = cvd_1s_ago[-1] - cvd_2s_ago[-1]
                feat["cvd_accel"] = v_recent - v_old
            else:
                feat["cvd_accel"] = 0

            # CVD relative to total volume (normalized)
            total_vol_so_far = post_notionals[mask_up_to].sum()
            feat["cvd_normalized"] = feat["cvd"] / total_vol_so_far if total_vol_so_far > 0 else 0

        # ══════════════════════════════════════════════════════════
        # EXPERIMENT 3: SEQUENCE FEATURES
        # ══════════════════════════════════════════════════════════
        if "sequence" in experiments:
            feat["bounce_count"] = bounce_count
            feat["consecutive_new_lows"] = consecutive_new_lows

            # Price range in recent windows
            for window, label in [(2000, "2s"), (5000, "5s")]:
                w_mask = (current_times > tick_t - window) & (current_times <= tick_t)
                w_prices = post_prices_bps[mask_up_to][-(w_mask.sum()):]
                if len(w_prices) >= 2:
                    feat[f"price_range_{label}"] = w_prices.max() - w_prices.min()
                    feat[f"price_std_{label}"] = w_prices.std()
                else:
                    feat[f"price_range_{label}"] = 0
                    feat[f"price_std_{label}"] = 0

            # Inter-trade time (avg time between trades in last 1s)
            recent_times = current_times[(current_times > tick_t - 1000) & (current_times <= tick_t)]
            if len(recent_times) >= 2:
                diffs = np.diff(recent_times)
                feat["avg_inter_trade_ms"] = diffs.mean()
                feat["max_inter_trade_ms"] = diffs.max()
            else:
                feat["avg_inter_trade_ms"] = 1000
                feat["max_inter_trade_ms"] = 1000

            # Number of price reversals (direction changes in 2s window)
            w_mask = (current_times > tick_t - 2000) & (current_times <= tick_t)
            w_prices = post_prices_bps[mask_up_to][-(w_mask.sum()):]
            if len(w_prices) >= 3:
                diffs = np.diff(w_prices)
                signs = np.sign(diffs)
                signs = signs[signs != 0]
                reversals = (np.diff(signs) != 0).sum() if len(signs) >= 2 else 0
                feat["reversals_2s"] = reversals
            else:
                feat["reversals_2s"] = 0

        # ══════════════════════════════════════════════════════════
        # EXPERIMENT 5: FR REGIME INTERACTIONS
        # ══════════════════════════════════════════════════════════
        if "fr_regime" in experiments:
            fa = abs(fr_bps)
            feat["fr_x_distance"] = fa * feat["distance_from_low_bps"]
            feat["fr_x_velocity_1s"] = fa * feat.get("price_velocity_1s", 0)
            feat["fr_x_time_since_low"] = fa * feat["time_since_new_low_ms"]
            feat["fr_x_vol_rate_1s"] = fa * feat.get("vol_rate_1s", 0)
            feat["fr_x_pct_elapsed"] = fa * feat["pct_of_window_elapsed"]
            feat["fr_regime"] = 0 if fa < 30 else 1 if fa < 60 else 2 if fa < 100 else 3

        # ══════════════════════════════════════════════════════════
        # TARGETS (same as v2)
        # ══════════════════════════════════════════════════════════
        future_mask = post_times > tick_t
        if future_mask.sum() > 0:
            future_prices = post_prices_bps[future_mask]
            future_min = future_prices.min()
            feat["target_drop_remaining"] = current_price - future_min
            feat["target_near_bottom_5"] = 1 if feat["target_drop_remaining"] < 5 else 0
            feat["target_near_bottom_10"] = 1 if feat["target_drop_remaining"] < 10 else 0
            feat["target_near_bottom_15"] = 1 if feat["target_drop_remaining"] < 15 else 0
            feat["target_bottom_passed"] = 1 if running_min_bps <= global_min_bps + 0.5 else 0
        else:
            feat["target_drop_remaining"] = 0
            feat["target_near_bottom_5"] = 1
            feat["target_near_bottom_10"] = 1
            feat["target_near_bottom_15"] = 1
            feat["target_bottom_passed"] = 1

        feat["symbol"] = symbol
        feat["settle_id"] = stem

        rows.append(feat)

    return pd.DataFrame(rows) if rows else None


# ═══════════════════════════════════════════════════════════════════════
# Evaluate a single experiment
# ═══════════════════════════════════════════════════════════════════════

TARGET_COLS = {"target_drop_remaining", "target_near_bottom_5", "target_near_bottom_10",
               "target_near_bottom_15", "target_bottom_passed"}
META_COLS = {"symbol", "settle_id", "t_ms", "phase"}
ALL_SKIP = TARGET_COLS | META_COLS


def evaluate_experiment(df, name="baseline", run_loso=False):
    """Train LogReg + HGBC, return metrics dict.
    run_loso=True adds expensive LOSO CV + backtest (only for baseline/final)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import gc

    feature_cols = [c for c in df.columns if c not in ALL_SKIP]
    X = df[feature_cols].values.astype(np.float32)  # float32 saves 50% memory
    y = df["target_near_bottom_10"].values
    symbols = df["symbol"].values

    # 70/30 temporal split
    unique_settle = df["settle_id"].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    train_mask = df["settle_id"].isin(train_settles).values
    test_mask = ~train_mask

    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    results = {"name": name, "n_features": len(feature_cols)}

    # LogReg
    lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=5000)),
    ])
    lr.fit(X_tr, y_tr)
    auc_tr = roc_auc_score(y_tr, lr.predict_proba(X_tr)[:, 1])
    auc_te = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    results["lr_auc_train"] = auc_tr
    results["lr_auc_test"] = auc_te
    results["lr_gap"] = auc_tr - auc_te
    del lr; gc.collect()

    # HGBC
    hgbc = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    hgbc.fit(X_tr, y_tr)
    auc_tr = roc_auc_score(y_tr, hgbc.predict_proba(X_tr)[:, 1])
    auc_te = roc_auc_score(y_te, hgbc.predict_proba(X_te)[:, 1])
    results["hgbc_auc_train"] = auc_tr
    results["hgbc_auc_test"] = auc_te
    results["hgbc_gap"] = auc_tr - auc_te

    # Permutation importance (top features)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(hgbc, X_te, y_te, n_repeats=3, random_state=42, scoring="roc_auc")
    sorted_idx = np.argsort(-perm.importances_mean)
    results["top_features"] = [(feature_cols[i], perm.importances_mean[i]) for i in sorted_idx[:8]]
    del hgbc; gc.collect()

    # LOSO + Backtest (expensive — only for baseline and final)
    results["loso_auc"] = float("nan")
    results["bt_ml_loso_50_avg"] = float("nan")
    results["bt_ml_loso_50_total"] = float("nan")
    results["bt_fixed_10s_avg"] = float("nan")
    results["bt_oracle_avg"] = float("nan")

    if run_loso:
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
        print(f"    Running LOSO CV ({len(np.unique(symbols))} symbols)...")
        logo = LeaveOneGroupOut()
        hgbc2 = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, min_samples_leaf=30,
            learning_rate=0.05, l2_regularization=1.0, random_state=42,
        )
        y_pred_loso = cross_val_predict(hgbc2, X, y, cv=logo, groups=symbols, method="predict_proba")[:, 1]
        results["loso_auc"] = roc_auc_score(y, y_pred_loso)
        del hgbc2; gc.collect()

        # Backtest with LOSO predictions
        df_bt = df[["settle_id", "t_ms", "price_bps"]].copy()
        df_bt["ml_prob_loso"] = y_pred_loso

        for strat_name, threshold in [("ml_loso_50", 0.50)]:
            pnls = []
            for sid, sdf in df_bt.groupby("settle_id"):
                sdf = sdf.sort_values("t_ms")
                exit_price = sdf.iloc[-1]["price_bps"]
                for _, row in sdf.iterrows():
                    if row["t_ms"] < 1000:
                        continue
                    if row["ml_prob_loso"] > threshold:
                        exit_price = row["price_bps"]
                        break
                pnls.append(-exit_price - 20)
            pnls = np.array(pnls)
            results[f"bt_{strat_name}_avg"] = pnls.mean()
            results[f"bt_{strat_name}_total"] = pnls.sum()

        # Fixed + oracle
        for strat_name, t_exit in [("fixed_10s", 10000)]:
            pnls = []
            for sid, sdf in df_bt.groupby("settle_id"):
                sdf = sdf.sort_values("t_ms")
                at_exit = sdf[sdf["t_ms"] <= t_exit]
                exit_price = at_exit.iloc[-1]["price_bps"] if len(at_exit) > 0 else sdf.iloc[-1]["price_bps"]
                pnls.append(-exit_price - 20)
            results[f"bt_{strat_name}_avg"] = np.mean(pnls)

        pnls = []
        for sid, sdf in df_bt.groupby("settle_id"):
            pnls.append(-sdf["price_bps"].min() - 20)
        results["bt_oracle_avg"] = np.mean(pnls)

        del df_bt, y_pred_loso; gc.collect()

    return results


# ═══════════════════════════════════════════════════════════════════════
# Build datasets for each experiment
# ═══════════════════════════════════════════════════════════════════════

def build_dataset(experiment_set, label=""):
    """Build tick dataset with specified experiments enabled."""
    jsonl_files = sorted(LOCAL_DATA_DIR.glob("*.jsonl"))
    print(f"\n  Building [{label}] from {len(jsonl_files)} files...")

    all_dfs = []
    t0 = _time.time()
    for i, fp in enumerate(jsonl_files, 1):
        tick_df = build_tick_features_extended(fp, experiments=experiment_set)
        if tick_df is not None:
            all_dfs.append(tick_df)
        if i % 30 == 0:
            n = sum(len(d) for d in all_dfs)
            print(f"    [{i}/{len(jsonl_files)}] {len(all_dfs)} valid, {n} ticks, {_time.time()-t0:.1f}s")

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    elapsed = _time.time() - t0
    feature_cols = [c for c in df.columns if c not in ALL_SKIP]
    print(f"    ✅ {len(df)} ticks, {df['settle_id'].nunique()} settlements, "
          f"{len(feature_cols)} features [{elapsed:.1f}s]")
    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   EXIT ML FEATURE EXPERIMENTS                                   ║")
    print("║   Test each feature group one-by-one against baseline           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── PHASE 1: Build ONE full dataset with ALL features in single pass ──
    # Process each file once with all experiments enabled, discard raw data immediately

    print(f"\n{'='*70}")
    print(f"PHASE 1: BUILD FULL DATASET (single pass, all features)")
    print(f"{'='*70}")

    all_experiments = {"ob_depth", "cvd", "sequence", "fr_regime"}
    df_full = build_dataset(all_experiments, label="all_features")
    if df_full is None:
        print("  ✗ Failed"); return

    all_feature_cols = [c for c in df_full.columns if c not in ALL_SKIP]
    print(f"\n  Full dataset: {len(df_full)} ticks, {len(all_feature_cols)} features")

    # ── PHASE 2: Define experiments as column subsets ──────────────
    baseline_cols = [c for c in all_feature_cols
                     if not c.startswith("ob5") and not c.startswith("ob_")
                     and not c.startswith("cvd") and c not in (
                         "bounce_count", "consecutive_new_lows",
                         "price_range_2s", "price_range_5s", "price_std_2s", "price_std_5s",
                         "avg_inter_trade_ms", "max_inter_trade_ms", "reversals_2s",
                         "fr_x_distance", "fr_x_velocity_1s", "fr_x_time_since_low",
                         "fr_x_vol_rate_1s", "fr_x_pct_elapsed", "fr_regime")]

    cvd_cols = [c for c in all_feature_cols if c.startswith("cvd")]
    seq_cols = ["bounce_count", "consecutive_new_lows",
                "price_range_2s", "price_range_5s", "price_std_2s", "price_std_5s",
                "avg_inter_trade_ms", "max_inter_trade_ms", "reversals_2s"]
    seq_cols = [c for c in seq_cols if c in all_feature_cols]
    fr_cols = [c for c in all_feature_cols if c.startswith("fr_x_") or c == "fr_regime"]
    ob_cols = [c for c in all_feature_cols if c.startswith("ob5") or c.startswith("ob_")]

    experiments = [
        ("baseline",    baseline_cols,                          "Baseline v2",       True),
        ("+ OB depth",  baseline_cols + ob_cols,                "+ OB depth (L5-L50)", False),
        ("+ CVD",       baseline_cols + cvd_cols,               "+ CVD (cum vol delta)", False),
        ("+ Sequence",  baseline_cols + seq_cols,               "+ Sequence features", False),
        ("+ FR regime", baseline_cols + fr_cols,                "+ FR regime interactions", False),
        ("ALL combined", all_feature_cols,                      "ALL new features",  True),
    ]

    print(f"\n  Experiments defined:")
    for name, cols, desc, loso in experiments:
        loso_s = " [+LOSO]" if loso else ""
        print(f"    {desc:<30s} {len(cols)} features{loso_s}")

    # ── PHASE 3: Evaluate each experiment ─────────────────────────
    all_results = []

    for exp_name, exp_cols, exp_desc, do_loso in experiments:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {exp_desc} ({len(exp_cols)} features)")
        print(f"{'='*70}")

        # Slice to only the experiment's columns + meta + targets
        keep_cols = list(ALL_SKIP) + exp_cols
        df_exp = df_full[[c for c in keep_cols if c in df_full.columns]].copy()

        results = evaluate_experiment(df_exp, name=exp_name, run_loso=do_loso)
        results["desc"] = exp_desc
        all_results.append(results)

        print(f"\n  LogReg:  Train={results['lr_auc_train']:.4f}  Test={results['lr_auc_test']:.4f}  "
              f"Gap={results['lr_gap']:+.3f}")
        print(f"  HGBC:   Train={results['hgbc_auc_train']:.4f}  Test={results['hgbc_auc_test']:.4f}  "
              f"Gap={results['hgbc_gap']:+.3f}")
        if do_loso:
            print(f"  LOSO:   {results['loso_auc']:.4f}")
            print(f"  Backtest: ML_LOSO_50={results['bt_ml_loso_50_avg']:+.1f} bps  "
                  f"Fixed10s={results['bt_fixed_10s_avg']:+.1f}  Oracle={results['bt_oracle_avg']:+.1f}")
        print(f"  Top features: {', '.join(f'{n}({v:.3f})' for n, v in results['top_features'][:5])}")

    # ── COMPARISON TABLE ──────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"EXPERIMENT COMPARISON")
    print(f"{'='*90}")
    print(f"\n{'Experiment':<25s} {'#F':>3s} {'LR Test':>8s} {'HGBC Te':>8s} {'LOSO':>7s} "
          f"{'LR Gap':>7s} {'LOSO50':>8s} {'vs Base':>8s}")
    print(f"{'-'*25} {'-'*3} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    baseline_pnl = all_results[0]["bt_ml_loso_50_avg"] if all_results else 0

    for r in all_results:
        delta_pnl = r["bt_ml_loso_50_avg"] - baseline_pnl
        print(f"{r['desc']:<25s} {r['n_features']:3d} {r['lr_auc_test']:8.4f} {r['hgbc_auc_test']:8.4f} "
              f"{r['loso_auc']:7.4f} {r['lr_gap']:+7.3f} {r['bt_ml_loso_50_avg']:+8.1f} "
              f"{delta_pnl:+8.1f}")

    print(f"\n\nTotal time: {_time.time()-t0:.1f}s")
    return all_results


if __name__ == "__main__":
    results = main()
