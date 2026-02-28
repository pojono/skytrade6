#!/usr/bin/env python3
"""
Real-Time Exit ML Research — Microstructure-based exit signal
==============================================================
For each settlement recording (JSONL), create a time series of 100ms ticks
with microstructure features and a target: "will price drop ≥5 bps more in next 1s?"

Then train ML models and backtest exit strategies.

Usage:
    python3 research_exit_ml.py
"""

import json
import sys
import time as _time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOCAL_DATA_DIR = Path("charts_settlement")
TICK_MS = 100          # feature computation interval
LOOKAHEAD_MS = 1000    # predict 1s ahead
MIN_DROP_BPS = 5       # "further drop" threshold
MAX_POST_MS = 60000    # analyze up to 60s post-settlement


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Parse JSONL into event streams
# ═══════════════════════════════════════════════════════════════════════

def parse_jsonl(fp):
    """Parse JSONL into sorted event streams."""
    trades = []
    ob1 = []
    tickers = []

    with open(fp) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except:
                continue

            t = msg.get("_t_ms", 0)
            topic = msg.get("topic", "")
            data = msg.get("data", {})

            if "publicTrade" in topic:
                for tr in (data if isinstance(data, list) else [data]):
                    p = float(tr.get("p", 0))
                    q = float(tr.get("v", 0))
                    s = tr.get("S", "")
                    trades.append((t, p, q, s, p * q))

            elif topic.startswith("orderbook.1."):
                b = data.get("b", [])
                a = data.get("a", [])
                if b and a:
                    ob1.append((t, float(b[0][0]), float(b[0][1]),
                                float(a[0][0]), float(a[0][1])))

            elif "tickers" in topic:
                fr = float(data.get("fundingRate", 0))
                oi = float(data.get("openInterest", 0))
                tickers.append((t, fr, oi))

    return (sorted(trades, key=lambda x: x[0]),
            sorted(ob1, key=lambda x: x[0]),
            sorted(tickers, key=lambda x: x[0]))


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Build time-series features at 100ms ticks
# ═══════════════════════════════════════════════════════════════════════

def build_tick_features(fp):
    """Build feature matrix for one settlement recording."""
    trades, ob1, tickers, = parse_jsonl(fp)

    if not trades:
        return None

    stem = fp.stem
    parts = stem.split("_")
    symbol = parts[0]

    # Reference price = last trade before settlement
    pre_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t < 0]
    post_trades = [(t, p, q, s, n) for t, p, q, s, n in trades if t >= 0]

    if not pre_trades or len(post_trades) < 5:
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

    # Pre-settlement OB snapshot
    pre_ob1 = [o for o in ob1 if -5000 <= o[0] < 0]
    if pre_ob1:
        _, bp, bq, ap, aq = pre_ob1[-1]
        pre_spread_bps = (ap - bp) / ref_price * 10000
    else:
        pre_spread_bps = 0

    # Max time in recording
    max_t = min(post_trades[-1][0], MAX_POST_MS)

    # Pre-compute: all post-trade prices sorted by time for efficient lookups
    post_sorted = sorted(post_trades, key=lambda x: x[0])
    post_times = np.array([t for t, _, _, _, _ in post_sorted])
    post_prices_bps = np.array([bps(p) for _, p, _, _, _ in post_sorted])
    post_sides = np.array([1 if s == "Sell" else 0 for _, _, _, s, _ in post_sorted])
    post_notionals = np.array([n for _, _, _, _, n in post_sorted])
    post_sizes = np.array([q for _, _, q, _, _ in post_sorted])

    # OB.1 indexed by time
    ob1_post = [(t, bp, bq, ap, aq) for t, bp, bq, ap, aq in ob1 if t >= 0]
    ob1_times = np.array([t for t, _, _, _, _ in ob1_post]) if ob1_post else np.array([])

    rows = []

    # Tick through every TICK_MS
    running_min_bps = 0.0
    last_new_low_t = 0

    for tick_t in range(0, int(max_t) - LOOKAHEAD_MS, TICK_MS):
        # Trades up to this tick
        mask = post_times <= tick_t
        if mask.sum() < 2:
            continue

        current_prices = post_prices_bps[mask]
        current_sides = post_sides[mask]
        current_notionals = post_notionals[mask]
        current_times = post_times[mask]
        current_sizes = post_sizes[mask]

        current_price = current_prices[-1]

        # Running minimum
        new_min = current_prices.min()
        if new_min < running_min_bps:
            running_min_bps = new_min
            last_new_low_t = tick_t
        elif tick_t == 0:
            running_min_bps = current_price

        # ── Price features ───────────────────────────────────────
        feat = {}
        feat["t_ms"] = tick_t
        feat["price_bps"] = current_price
        feat["running_min_bps"] = running_min_bps
        feat["distance_from_low_bps"] = current_price - running_min_bps
        feat["new_low"] = 1 if current_price <= running_min_bps + 0.5 else 0
        feat["time_since_new_low_ms"] = tick_t - last_new_low_t

        # Price velocity (last 500ms)
        mask_500 = (current_times > tick_t - 500) & (current_times <= tick_t)
        if mask_500.sum() >= 2:
            prices_500 = post_prices_bps[mask][-mask_500.sum():]
            feat["price_velocity_500ms"] = prices_500[-1] - prices_500[0]
        else:
            feat["price_velocity_500ms"] = 0

        # Price velocity (last 1s)
        mask_1s = (current_times > tick_t - 1000) & (current_times <= tick_t)
        if mask_1s.sum() >= 2:
            prices_1s = post_prices_bps[mask][-mask_1s.sum():]
            feat["price_velocity_1s"] = prices_1s[-1] - prices_1s[0]
        else:
            feat["price_velocity_1s"] = 0

        # Price acceleration
        mask_early = (current_times > tick_t - 1000) & (current_times <= tick_t - 500)
        mask_late = (current_times > tick_t - 500) & (current_times <= tick_t)
        if mask_early.sum() >= 1 and mask_late.sum() >= 1:
            early_prices = post_prices_bps[mask][-mask_1s.sum():][:mask_early.sum()]
            late_prices = post_prices_bps[mask][-mask_late.sum():]
            v_early = early_prices[-1] - early_prices[0] if len(early_prices) > 1 else 0
            v_late = late_prices[-1] - late_prices[0] if len(late_prices) > 1 else 0
            feat["price_accel"] = v_late - v_early
        else:
            feat["price_accel"] = 0

        # ── Trade flow features ──────────────────────────────────
        for window, label in [(500, "500ms"), (1000, "1s"), (2000, "2s")]:
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

            # Large trades (> 2x median)
            if n_trades > 2:
                med = np.median(w_sizes)
                feat[f"large_trade_pct_{label}"] = (w_sizes > 2 * med).mean()
            else:
                feat[f"large_trade_pct_{label}"] = 0

        # Volume surge vs pre-settlement
        feat["vol_surge"] = feat["vol_rate_1s"] / pre_vol_rate if pre_vol_rate > 0 else 1
        feat["trade_surge"] = feat["trade_rate_1s"] / pre_trade_rate if pre_trade_rate > 0 else 1

        # Trade rate acceleration
        if tick_t >= 2000:
            mask_tr_early = (post_times > tick_t - 2000) & (post_times <= tick_t - 1000)
            mask_tr_late = (post_times > tick_t - 1000) & (post_times <= tick_t)
            rate_early = mask_tr_early.sum()
            rate_late = mask_tr_late.sum()
            feat["trade_rate_accel"] = rate_late - rate_early
        else:
            feat["trade_rate_accel"] = 0

        # Buy volume surge (buyers stepping in = bottom signal)
        buy_vol_1s = w_notionals[w_sides == 0].sum() if 'w_sides' in dir() else 0
        mask_1s_check = (post_times > tick_t - 1000) & (post_times <= tick_t)
        buy_v = post_notionals[mask_1s_check & (post_sides == 0)].sum()
        sell_v = post_notionals[mask_1s_check & (post_sides == 1)].sum()
        feat["buy_sell_ratio_1s"] = buy_v / sell_v if sell_v > 0 else 1.0

        # ── Orderbook features ───────────────────────────────────
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

                # OB change from previous snapshot
                if idx > 0:
                    _, bp2, bq2, ap2, aq2 = ob1_post[idx - 1]
                    feat["bid_qty_change"] = bq - bq2
                    feat["ask_qty_change"] = aq - aq2
                else:
                    feat["bid_qty_change"] = 0
                    feat["ask_qty_change"] = 0
            else:
                feat["spread_bps"] = pre_spread_bps
                feat["spread_change"] = 0
                feat["ob1_bid_qty"] = feat["ob1_ask_qty"] = 0
                feat["ob1_imbalance"] = 0
                feat["bid_qty_change"] = feat["ask_qty_change"] = 0
        else:
            feat["spread_bps"] = pre_spread_bps
            feat["spread_change"] = 0
            feat["ob1_bid_qty"] = feat["ob1_ask_qty"] = 0
            feat["ob1_imbalance"] = 0
            feat["bid_qty_change"] = feat["ask_qty_change"] = 0

        # ── Static features ──────────────────────────────────────
        feat["fr_bps"] = fr_bps
        feat["fr_abs_bps"] = abs(fr_bps)

        # ── Time features ────────────────────────────────────────
        feat["t_seconds"] = tick_t / 1000.0
        feat["log_t"] = np.log1p(tick_t)
        feat["phase"] = (0 if tick_t < 1000 else
                         1 if tick_t < 5000 else
                         2 if tick_t < 10000 else
                         3 if tick_t < 30000 else 4)

        # ── TARGET ───────────────────────────────────────────────
        # Will price drop at least MIN_DROP_BPS more in next LOOKAHEAD_MS?
        future_mask = (post_times > tick_t) & (post_times <= tick_t + LOOKAHEAD_MS)
        if future_mask.sum() > 0:
            future_min = post_prices_bps[future_mask].min()
            feat["target_further_drop"] = 1 if future_min < current_price - MIN_DROP_BPS else 0
            feat["target_future_min_bps"] = future_min
            feat["target_future_drop_bps"] = current_price - future_min
        else:
            feat["target_further_drop"] = 0
            feat["target_future_min_bps"] = current_price
            feat["target_future_drop_bps"] = 0

        # Meta
        feat["symbol"] = symbol
        feat["settle_id"] = stem

        rows.append(feat)

    if not rows:
        return None

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Train and evaluate
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(df):
    """Train ML models with efficient validation."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    feature_cols = [c for c in df.columns if c not in
                    ("symbol", "settle_id", "t_ms",
                     "target_further_drop", "target_future_min_bps",
                     "target_future_drop_bps")]

    X = df[feature_cols].values
    y = df["target_further_drop"].values
    groups = df["settle_id"].values
    symbols = df["symbol"].values

    print(f"\n{'='*70}")
    print(f"ML TRAINING — {len(df)} ticks, {df['settle_id'].nunique()} settlements, {len(feature_cols)} features")
    print(f"{'='*70}")
    print(f"  Target balance: {y.mean()*100:.1f}% positive (further drop)")
    print(f"  Ticks per settlement: {len(df) / df['settle_id'].nunique():.0f} avg")

    # ── Validation: 70/30 temporal split by settlement ────────────
    unique_settle = df['settle_id'].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    test_settles = set(unique_settle[n_train:])

    train_mask = df['settle_id'].isin(train_settles).values
    test_mask = df['settle_id'].isin(test_settles).values

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    n_train_sym = df.loc[train_mask, 'symbol'].nunique()
    n_test_sym = df.loc[test_mask, 'symbol'].nunique()
    overlap = len(set(df.loc[train_mask, 'symbol']) & set(df.loc[test_mask, 'symbol']))
    print(f"  Split: train={train_mask.sum()} ticks ({len(train_settles)} settle), test={test_mask.sum()} ({len(test_settles)} settle)")
    print(f"  Train symbols: {n_train_sym}, Test symbols: {n_test_sym}, Overlap: {overlap}")

    # ── Models ────────────────────────────────────────────────────
    models = {
        "LogReg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=5000)),
        ]),
        "HGBC_light": HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, min_samples_leaf=50,
            learning_rate=0.05, l2_regularization=1.0, random_state=42,
        ),
        "HGBC_deep": HistGradientBoostingClassifier(
            max_iter=500, max_depth=7, min_samples_leaf=20,
            learning_rate=0.03, l2_regularization=0.5, random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        import copy
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        # Train metrics
        y_prob_tr = m.predict_proba(X_tr)[:, 1]
        auc_tr = roc_auc_score(y_tr, y_prob_tr)

        # Test metrics
        y_prob_te = m.predict_proba(X_te)[:, 1]
        auc_te = roc_auc_score(y_te, y_prob_te)
        y_pred_te = (y_prob_te > 0.5).astype(int)
        acc_te = accuracy_score(y_te, y_pred_te)
        f1_te = f1_score(y_te, y_pred_te)

        print(f"  Train AUC: {auc_tr:.4f}")
        print(f"  Test AUC:  {auc_te:.4f}  Acc={acc_te:.3f}  F1={f1_te:.3f}")
        print(f"  Overfit gap: {auc_tr - auc_te:.4f}")

        # AUC by time phase (test set)
        print(f"  Test AUC by phase:")
        for phase, label in [(0, "0-1s"), (1, "1-5s"), (2, "5-10s"), (3, "10-30s"), (4, "30-60s")]:
            pmask = (df["phase"].values == phase) & test_mask
            if pmask.sum() > 50 and y[pmask].mean() > 0.01 and y[pmask].mean() < 0.99:
                auc_phase = roc_auc_score(y[pmask], m.predict_proba(X[pmask])[:, 1])
                print(f"    {label:>8s}: AUC={auc_phase:.3f} (N={pmask.sum()}, pos={y[pmask].mean():.1%})")

        # Full predictions for backtest (train on full, predict on full)
        m_full = copy.deepcopy(model)
        m_full.fit(X, y)
        y_pred_all = m_full.predict_proba(X)[:, 1]

        results[name] = {
            "auc_train": auc_tr,
            "auc_test": auc_te,
            "acc_test": acc_te,
            "f1_test": f1_te,
            "y_pred": y_pred_all,  # for backtest
            "y_pred_test": y_prob_te,  # honest test-only
            "test_mask": test_mask,
        }

    # ── LOSO (symbol) for best model only ─────────────────────────
    print(f"\n--- Leave-One-Symbol-Out (HGBC_light, honest) ---")
    logo = LeaveOneGroupOut()
    best = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, min_samples_leaf=50,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    y_pred_loso = cross_val_predict(best, X, y, cv=logo, groups=symbols, method="predict_proba")[:, 1]
    auc_loso = roc_auc_score(y, y_pred_loso)
    print(f"  LOSO (symbol) AUC: {auc_loso:.4f}")
    results["HGBC_light"]["auc_loso"] = auc_loso
    results["HGBC_light"]["y_pred_loso"] = y_pred_loso

    # ── Feature importance (from permutation importance) ────────
    print(f"\n--- Feature Importance (HGBC_light, permutation) ---")
    from sklearn.inspection import permutation_importance
    best_full = HistGradientBoostingClassifier(
        max_iter=200, max_depth=5, min_samples_leaf=50,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    best_full.fit(X_tr, y_tr)
    perm = permutation_importance(best_full, X_te, y_te, n_repeats=5, random_state=42, scoring="roc_auc")
    sorted_idx = np.argsort(-perm.importances_mean)
    for i in sorted_idx[:20]:
        print(f"  {feature_cols[i]:30s}: {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}")

    return results, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Backtest exit strategies
# ═══════════════════════════════════════════════════════════════════════

def backtest_exits(df, results):
    """Compare exit strategies on each settlement."""
    FEE_BPS = 20

    print(f"\n{'='*70}")
    print(f"BACKTEST EXIT STRATEGIES")
    print(f"{'='*70}")

    best_pred = results["HGBC_light"]["y_pred"]
    df["ml_prob"] = best_pred

    settlements = df.groupby("settle_id")
    strats = {
        "fixed_5s": [],
        "fixed_10s": [],
        "fixed_30s": [],
        "time_tiers": [],
        "trailing_15bps": [],
        "ml_exit_30": [],
        "ml_exit_40": [],
        "ml_exit_50": [],
        "ml_plus_trail": [],
        "oracle": [],
    }

    for sid, sdf in settlements:
        sdf = sdf.sort_values("t_ms")
        fr = sdf["fr_abs_bps"].iloc[0]

        # Fixed exits — price at that time
        for name, t_exit in [("fixed_5s", 5000), ("fixed_10s", 10000), ("fixed_30s", 30000)]:
            at_exit = sdf[sdf["t_ms"] <= t_exit]
            if len(at_exit) > 0:
                exit_price = at_exit.iloc[-1]["price_bps"]
            else:
                exit_price = sdf.iloc[-1]["price_bps"]
            strats[name].append(-exit_price - FEE_BPS)

        # Time tiers by FR
        if fr < 25:
            tier_time = 0  # skip
        elif fr < 50:
            tier_time = 5000
        elif fr < 80:
            tier_time = 10000
        else:
            tier_time = 25000
        if tier_time == 0:
            strats["time_tiers"].append(0)  # skipped
        else:
            at_exit = sdf[sdf["t_ms"] <= tier_time]
            if len(at_exit) > 0:
                exit_price = at_exit.iloc[-1]["price_bps"]
            else:
                exit_price = sdf.iloc[-1]["price_bps"]
            strats["time_tiers"].append(-exit_price - FEE_BPS)

        # Trailing stop: exit when price bounces 15 bps from running min
        running_min = 0
        trail_exit_price = sdf.iloc[-1]["price_bps"]
        for _, row in sdf.iterrows():
            p = row["price_bps"]
            if p < running_min:
                running_min = p
            if p > running_min + 15:
                trail_exit_price = p
                break
        strats["trailing_15bps"].append(-trail_exit_price - FEE_BPS)

        # ML exits: exit when P(further_drop) < threshold
        for threshold, name in [(0.30, "ml_exit_30"), (0.40, "ml_exit_40"), (0.50, "ml_exit_50")]:
            ml_exit_price = sdf.iloc[-1]["price_bps"]
            for _, row in sdf.iterrows():
                if row["t_ms"] < 500:  # minimum hold time
                    continue
                if row["ml_prob"] < threshold:
                    ml_exit_price = row["price_bps"]
                    break
            strats[name].append(-ml_exit_price - FEE_BPS)

        # ML + trailing stop combo
        running_min = 0
        combo_exit = sdf.iloc[-1]["price_bps"]
        for _, row in sdf.iterrows():
            p = row["price_bps"]
            if p < running_min:
                running_min = p
            if row["t_ms"] < 500:
                continue
            # Exit if ML says no more drop OR trailing stop triggered
            if row["ml_prob"] < 0.35 or p > running_min + 15:
                combo_exit = p
                break
        strats["ml_plus_trail"].append(-combo_exit - FEE_BPS)

        # Oracle: exit at the actual bottom
        oracle_price = sdf["price_bps"].min()
        strats["oracle"].append(-oracle_price - FEE_BPS)

    # Summary
    print(f"\n  {'Strategy':25s} {'Trades':>7s} {'Avg PnL':>8s} {'Med PnL':>8s} {'WR':>6s} {'Total':>10s}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")

    for name, pnls in strats.items():
        pnls = np.array(pnls)
        # Time tiers: 0 means skipped
        if name == "time_tiers":
            traded = pnls != 0
            if traded.sum() > 0:
                active_pnls = pnls[traded]
                print(f"  {name:25s} {traded.sum():7d} {active_pnls.mean():+8.1f} {np.median(active_pnls):+8.1f} {(active_pnls > 0).mean()*100:5.0f}% {active_pnls.sum():+10.1f}")
            continue

        print(f"  {name:25s} {len(pnls):7d} {pnls.mean():+8.1f} {np.median(pnls):+8.1f} {(pnls > 0).mean()*100:5.0f}% {pnls.sum():+10.1f}")

    return strats


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   MICROSTRUCTURE EXIT ML RESEARCH                              ║")
    print("║   Predict: will price drop ≥5bps more in next 1s?              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Step 1: Build tick dataset from all JSONL files
    jsonl_files = sorted(LOCAL_DATA_DIR.glob("*.jsonl"))
    print(f"\nProcessing {len(jsonl_files)} recordings...")

    all_dfs = []
    for i, fp in enumerate(jsonl_files, 1):
        df = build_tick_features(fp)
        if df is not None:
            all_dfs.append(df)
        if i % 20 == 0:
            n_ticks = sum(len(d) for d in all_dfs)
            print(f"  [{i}/{len(jsonl_files)}] {len(all_dfs)} valid, {n_ticks} ticks, {_time.time()-t0:.1f}s")

    if not all_dfs:
        print("No valid data")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    n_ticks = len(df)
    n_settle = df["settle_id"].nunique()
    n_symbols = df["symbol"].nunique()

    print(f"\n  Dataset: {n_ticks} ticks × {len(df.columns)} columns")
    print(f"  Settlements: {n_settle} | Symbols: {n_symbols}")
    print(f"  Ticks/settlement: {n_ticks/n_settle:.0f} avg")
    print(f"  Target positive rate: {df['target_further_drop'].mean()*100:.1f}%")

    # Quick feature stats
    print(f"\n  Feature ranges:")
    feat_cols = [c for c in df.columns if c not in
                 ("symbol", "settle_id", "t_ms", "target_further_drop",
                  "target_future_min_bps", "target_future_drop_bps")]
    for c in feat_cols[:10]:
        v = df[c].dropna()
        print(f"    {c:30s}: [{v.min():.2f}, {v.max():.2f}] mean={v.mean():.2f}")

    # Step 2: Train and evaluate
    results, feature_cols = train_and_evaluate(df)

    # Step 3: Backtest exit strategies
    strats = backtest_exits(df, results)

    elapsed = _time.time() - t0
    print(f"\n{'='*70}")
    print(f"RESEARCH COMPLETE [{elapsed:.1f}s]")
    print(f"{'='*70}")

    # Save dataset for further analysis
    out = Path("exit_ml_ticks.parquet")
    df.to_parquet(out, index=False)
    print(f"  Tick dataset saved to: {out}")


if __name__ == "__main__":
    main()
