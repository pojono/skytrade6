#!/usr/bin/env python3
"""
Exit ML v3 — Predict the BOTTOM + sequence features
=====================================================
Builds on v2 (predict-the-bottom) by adding the 9 winning sequence features
from the v3 feature experiments:
  - bounce_count, consecutive_new_lows
  - price_range_2s/5s, price_std_2s/5s
  - avg_inter_trade_ms, max_inter_trade_ms, reversals_2s

These were the ONLY feature group that improved BOTH LogReg and HGBC:
  LR: +0.0052 AUC, HGBC: +0.0026 AUC (zero overfit risk)

Usage:
    python3 research_exit_ml_v3.py
"""

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
# Parse JSONL
# ═══════════════════════════════════════════════════════════════════════

def parse_jsonl(fp):
    trades, ob1, tickers = [], [], []
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
# Build tick features with CORRECT targets
# ═══════════════════════════════════════════════════════════════════════

def build_tick_features(fp):
    trades, ob1, tickers = parse_jsonl(fp)
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

    # Precompute: the GLOBAL minimum price from each tick onwards
    # This is what we need for the "is this the bottom?" target
    global_min_bps = post_prices_bps.min()

    rows = []
    running_min_bps = 0.0
    last_new_low_t = 0

    # Sequence tracking (v3)
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

        # Sequence tracking (v3)
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

        # ── FEATURES ─────────────────────────────────────────────
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

        # Cumulative drop rate (bps per second elapsed)
        if tick_t > 0:
            feat["drop_rate_bps_per_s"] = running_min_bps / (tick_t / 1000)
        else:
            feat["drop_rate_bps_per_s"] = 0

        # Trade flow features
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

        # Volume/trade surge
        feat["vol_surge"] = feat["vol_rate_1s"] / pre_vol_rate if pre_vol_rate > 0 else 1
        feat["trade_surge"] = feat["trade_rate_1s"] / pre_trade_rate if pre_trade_rate > 0 else 1

        # Buy/sell ratio
        mask_1s = (post_times > tick_t - 1000) & (post_times <= tick_t)
        buy_v = post_notionals[mask_1s & (post_sides == 0)].sum()
        sell_v = post_notionals[mask_1s & (post_sides == 1)].sum()
        feat["buy_sell_ratio_1s"] = buy_v / sell_v if sell_v > 0 else 1.0

        # Trade rate acceleration
        if tick_t >= 2000:
            r_early = ((post_times > tick_t - 2000) & (post_times <= tick_t - 1000)).sum()
            r_late = ((post_times > tick_t - 1000) & (post_times <= tick_t)).sum()
            feat["trade_rate_accel"] = r_late - r_early
        else:
            feat["trade_rate_accel"] = 0

        # Orderbook
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

        # ── SEQUENCE FEATURES (v3) ──────────────────────────────
        feat["bounce_count"] = bounce_count
        feat["consecutive_new_lows"] = consecutive_new_lows

        for window, label in [(2000, "2s"), (5000, "5s")]:
            w_mask = (current_times > tick_t - window) & (current_times <= tick_t)
            w_prices = post_prices_bps[mask_up_to][-(w_mask.sum()):]
            if len(w_prices) >= 2:
                feat[f"price_range_{label}"] = w_prices.max() - w_prices.min()
                feat[f"price_std_{label}"] = w_prices.std()
            else:
                feat[f"price_range_{label}"] = 0
                feat[f"price_std_{label}"] = 0

        # Inter-trade time
        recent_times = current_times[(current_times > tick_t - 1000) & (current_times <= tick_t)]
        if len(recent_times) >= 2:
            diffs = np.diff(recent_times)
            feat["avg_inter_trade_ms"] = diffs.mean()
            feat["max_inter_trade_ms"] = diffs.max()
        else:
            feat["avg_inter_trade_ms"] = 1000
            feat["max_inter_trade_ms"] = 1000

        # Price reversals in 2s
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

        # Static features
        feat["fr_bps"] = fr_bps
        feat["fr_abs_bps"] = abs(fr_bps)

        # Time features
        feat["t_seconds"] = tick_t / 1000.0
        feat["log_t"] = np.log1p(tick_t)
        feat["phase"] = (0 if tick_t < 1000 else
                         1 if tick_t < 5000 else
                         2 if tick_t < 10000 else
                         3 if tick_t < 30000 else 4)

        # ── TARGETS: THE REAL QUESTION ───────────────────────────
        # "How much MORE will the price drop from HERE?"
        future_mask = post_times > tick_t
        if future_mask.sum() > 0:
            future_prices = post_prices_bps[future_mask]
            future_min = future_prices.min()

            # Regression: how many more bps of drop remain?
            feat["target_drop_remaining"] = current_price - future_min  # positive = more drop ahead
            # If current_price = -50 and future_min = -80, drop_remaining = 30 (30 more bps to go)

            # Classification: are we near the bottom?
            feat["target_near_bottom_5"] = 1 if feat["target_drop_remaining"] < 5 else 0
            feat["target_near_bottom_10"] = 1 if feat["target_drop_remaining"] < 10 else 0
            feat["target_near_bottom_15"] = 1 if feat["target_drop_remaining"] < 15 else 0

            # Is the actual global minimum already behind us?
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
# Train and evaluate
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(df):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error, r2_score
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import copy

    feature_cols = [c for c in df.columns if c not in
                    ("symbol", "settle_id", "t_ms",
                     "target_drop_remaining", "target_near_bottom_5",
                     "target_near_bottom_10", "target_near_bottom_15",
                     "target_bottom_passed")]

    X = df[feature_cols].values
    symbols = df["symbol"].values

    # 70/30 temporal split by settlement
    unique_settle = df['settle_id'].unique()
    n_train = int(len(unique_settle) * 0.7)
    train_settles = set(unique_settle[:n_train])
    train_mask = df['settle_id'].isin(train_settles).values
    test_mask = ~train_mask

    X_tr, X_te = X[train_mask], X[test_mask]

    n_train_sym = df.loc[train_mask, 'symbol'].nunique()
    n_test_sym = df.loc[test_mask, 'symbol'].nunique()
    overlap = len(set(df.loc[train_mask, 'symbol']) & set(df.loc[test_mask, 'symbol']))

    print(f"\n{'='*70}")
    print(f"ML TRAINING v2 — {len(df)} ticks, {df['settle_id'].nunique()} settlements, {len(feature_cols)} features")
    print(f"{'='*70}")
    print(f"  Split: train={train_mask.sum()} ({len(train_settles)} settle), test={test_mask.sum()} ({len(unique_settle)-n_train} settle)")
    print(f"  Symbols: train={n_train_sym}, test={n_test_sym}, overlap={overlap}")

    results = {}

    # ── A: REGRESSION — predict how many more bps of drop remain ──
    print(f"\n--- A: REGRESSION: future drop remaining (bps) ---")
    y_reg = df["target_drop_remaining"].values
    y_tr_r, y_te_r = y_reg[train_mask], y_reg[test_mask]
    print(f"  Target stats: mean={y_reg.mean():.1f}, median={np.median(y_reg):.1f}, p90={np.percentile(y_reg, 90):.1f}")

    reg_models = {
        "Ridge": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "HGBR": HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, min_samples_leaf=30,
            learning_rate=0.05, l2_regularization=1.0, random_state=42,
        ),
    }

    for name, model in reg_models.items():
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr_r)

        y_pred_tr = m.predict(X_tr)
        y_pred_te = m.predict(X_te)

        mae_tr = mean_absolute_error(y_tr_r, y_pred_tr)
        mae_te = mean_absolute_error(y_te_r, y_pred_te)
        r2_tr = r2_score(y_tr_r, y_pred_tr)
        r2_te = r2_score(y_te_r, y_pred_te)

        print(f"  {name:10s}: Train MAE={mae_tr:.1f} R²={r2_tr:.3f} | Test MAE={mae_te:.1f} R²={r2_te:.3f}")
        results[f"reg_{name}"] = {"mae_train": mae_tr, "mae_test": mae_te, "r2_train": r2_tr, "r2_test": r2_te}

    # ── B: CLASSIFICATION — is this near the bottom? ──────────────
    for threshold, label in [(5, "5bps"), (10, "10bps"), (15, "15bps")]:
        target_col = f"target_near_bottom_{threshold}"
        y_clf = df[target_col].values
        y_tr_c, y_te_c = y_clf[train_mask], y_clf[test_mask]
        pos_rate = y_clf.mean()

        print(f"\n--- B: CLASSIFICATION: near_bottom_{label} (pos={pos_rate:.1%}) ---")

        clf_models = {
            "LogReg": Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=5000)),
            ]),
            "HGBC": HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, min_samples_leaf=30,
                learning_rate=0.05, l2_regularization=1.0, random_state=42,
            ),
        }

        for name, model in clf_models.items():
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr_c)

            y_prob_tr = m.predict_proba(X_tr)[:, 1]
            y_prob_te = m.predict_proba(X_te)[:, 1]

            auc_tr = roc_auc_score(y_tr_c, y_prob_tr)
            auc_te = roc_auc_score(y_te_c, y_prob_te)
            y_pred_te = (y_prob_te > 0.5).astype(int)
            acc_te = accuracy_score(y_te_c, y_pred_te)
            f1_te = f1_score(y_te_c, y_pred_te)

            print(f"  {name:10s}: Train AUC={auc_tr:.4f} | Test AUC={auc_te:.4f}  Acc={acc_te:.3f}  F1={f1_te:.3f}  Gap={auc_tr-auc_te:.3f}")
            results[f"clf_{label}_{name}"] = {
                "auc_train": auc_tr, "auc_test": auc_te,
                "acc_test": acc_te, "f1_test": f1_te,
            }

        # AUC by phase for best model (HGBC)
        m_best = copy.deepcopy(clf_models["HGBC"])
        m_best.fit(X_tr, y_tr_c)
        print(f"  Test AUC by phase (HGBC):")
        for phase, plabel in [(0, "0-1s"), (1, "1-5s"), (2, "5-10s"), (3, "10-30s"), (4, "30-60s")]:
            pmask = (df["phase"].values == phase) & test_mask
            if pmask.sum() > 50:
                y_p = y_clf[pmask]
                if y_p.mean() > 0.01 and y_p.mean() < 0.99:
                    auc = roc_auc_score(y_p, m_best.predict_proba(X[pmask])[:, 1])
                    print(f"    {plabel:>8s}: AUC={auc:.3f} (N={pmask.sum()}, pos={y_p.mean():.1%})")

    # ── C: LOSO (symbol) for best model ───────────────────────────
    print(f"\n--- C: Leave-One-Symbol-Out (HGBC, near_bottom_10) ---")
    y_nb10 = df["target_near_bottom_10"].values
    logo = LeaveOneGroupOut()
    best = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    y_pred_loso = cross_val_predict(best, X, y_nb10, cv=logo, groups=symbols, method="predict_proba")[:, 1]
    auc_loso = roc_auc_score(y_nb10, y_pred_loso)
    print(f"  LOSO (symbol) AUC: {auc_loso:.4f}")
    results["loso_auc"] = auc_loso

    # ── D: Permutation importance ─────────────────────────────────
    print(f"\n--- D: Feature Importance (permutation, HGBC near_bottom_10) ---")
    from sklearn.inspection import permutation_importance
    m_imp = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    m_imp.fit(X_tr, y_nb10[train_mask])
    perm = permutation_importance(m_imp, X_te, y_nb10[test_mask], n_repeats=5, random_state=42, scoring="roc_auc")
    sorted_idx = np.argsort(-perm.importances_mean)
    for i in sorted_idx[:15]:
        print(f"  {feature_cols[i]:30s}: {perm.importances_mean[i]:+.4f} ± {perm.importances_std[i]:.4f}")

    # Store full-data predictions for backtest
    m_full = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, min_samples_leaf=30,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    m_full.fit(X, y_nb10)
    results["y_pred_full_nb10"] = m_full.predict_proba(X)[:, 1]

    # Also LOSO predictions for honest backtest
    results["y_pred_loso_nb10"] = y_pred_loso

    return results, feature_cols


# ═══════════════════════════════════════════════════════════════════════
# Backtest: SINGLE EXIT per settlement
# ═══════════════════════════════════════════════════════════════════════

def backtest_single_exit(df, results):
    """Each strategy gets exactly ONE exit per settlement."""
    FEE_BPS = 20

    print(f"\n{'='*70}")
    print(f"BACKTEST — SINGLE EXIT PER SETTLEMENT")
    print(f"{'='*70}")

    df["ml_prob_nb10"] = results["y_pred_full_nb10"]
    df["ml_prob_loso"] = results["y_pred_loso_nb10"]

    settlements = df.groupby("settle_id")

    strats = {
        "fixed_5s": [],
        "fixed_10s": [],
        "fixed_30s": [],
        "time_tiers_fr": [],
        "trailing_15bps": [],
        "ml_nb10_70": [],        # exit when P(near_bottom) > 0.70
        "ml_nb10_60": [],        # exit when P(near_bottom) > 0.60
        "ml_nb10_50": [],        # exit when P(near_bottom) > 0.50
        "ml_loso_70": [],        # honest LOSO predictions
        "ml_loso_60": [],
        "ml_loso_50": [],
        "ml_reg_then_nb": [],    # hybrid: wait until model says <10bps remaining, then use nb10
        "oracle": [],
    }

    exit_times = {k: [] for k in strats}

    for sid, sdf in settlements:
        sdf = sdf.sort_values("t_ms")
        fr = sdf["fr_abs_bps"].iloc[0]

        # Fixed exits
        for name, t_exit in [("fixed_5s", 5000), ("fixed_10s", 10000), ("fixed_30s", 30000)]:
            at_exit = sdf[sdf["t_ms"] <= t_exit]
            if len(at_exit) > 0:
                exit_price = at_exit.iloc[-1]["price_bps"]
                exit_t = at_exit.iloc[-1]["t_ms"]
            else:
                exit_price = sdf.iloc[-1]["price_bps"]
                exit_t = sdf.iloc[-1]["t_ms"]
            strats[name].append(-exit_price - FEE_BPS)
            exit_times[name].append(exit_t)

        # Time tiers by FR
        if fr < 25:
            tier_time = 5000
        elif fr < 50:
            tier_time = 5000
        elif fr < 80:
            tier_time = 10000
        else:
            tier_time = 25000
        at_exit = sdf[sdf["t_ms"] <= tier_time]
        exit_price = at_exit.iloc[-1]["price_bps"] if len(at_exit) > 0 else sdf.iloc[-1]["price_bps"]
        exit_t = at_exit.iloc[-1]["t_ms"] if len(at_exit) > 0 else sdf.iloc[-1]["t_ms"]
        strats["time_tiers_fr"].append(-exit_price - FEE_BPS)
        exit_times["time_tiers_fr"].append(exit_t)

        # Trailing stop
        running_min = 0
        trail_exit_price = sdf.iloc[-1]["price_bps"]
        trail_exit_t = sdf.iloc[-1]["t_ms"]
        for _, row in sdf.iterrows():
            p = row["price_bps"]
            if p < running_min:
                running_min = p
            if p > running_min + 15:
                trail_exit_price = p
                trail_exit_t = row["t_ms"]
                break
        strats["trailing_15bps"].append(-trail_exit_price - FEE_BPS)
        exit_times["trailing_15bps"].append(trail_exit_t)

        # ML exits: exit at FIRST tick where P(near_bottom_10) > threshold
        for prob_col, thresholds in [
            ("ml_prob_nb10", [(0.70, "ml_nb10_70"), (0.60, "ml_nb10_60"), (0.50, "ml_nb10_50")]),
            ("ml_prob_loso", [(0.70, "ml_loso_70"), (0.60, "ml_loso_60"), (0.50, "ml_loso_50")]),
        ]:
            for threshold, name in thresholds:
                ml_exit_price = sdf.iloc[-1]["price_bps"]
                ml_exit_t = sdf.iloc[-1]["t_ms"]
                for _, row in sdf.iterrows():
                    if row["t_ms"] < 1000:  # min hold 1s
                        continue
                    if row[prob_col] > threshold:
                        ml_exit_price = row["price_bps"]
                        ml_exit_t = row["t_ms"]
                        break
                strats[name].append(-ml_exit_price - FEE_BPS)
                exit_times[name].append(ml_exit_t)

        # Oracle
        oracle_price = sdf["price_bps"].min()
        strats["oracle"].append(-oracle_price - FEE_BPS)
        oracle_idx = sdf["price_bps"].idxmin()
        exit_times["oracle"].append(sdf.loc[oracle_idx, "t_ms"])

    # Summary
    print(f"\n  {'Strategy':25s} {'Trades':>7s} {'Avg PnL':>8s} {'Med PnL':>8s} {'WR':>6s} {'Total':>10s} {'Avg Exit@':>10s}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")

    for name in ["oracle", "ml_loso_70", "ml_loso_60", "ml_loso_50",
                  "ml_nb10_70", "ml_nb10_60", "ml_nb10_50",
                  "time_tiers_fr", "fixed_10s", "fixed_5s", "fixed_30s", "trailing_15bps"]:
        pnls = np.array(strats[name])
        times = np.array(exit_times[name])
        avg_t = times.mean() / 1000
        print(f"  {name:25s} {len(pnls):7d} {pnls.mean():+8.1f} {np.median(pnls):+8.1f} {(pnls > 0).mean()*100:5.0f}% {pnls.sum():+10.1f} {avg_t:9.1f}s")

    return strats, exit_times


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   EXIT ML v2 — PREDICT THE BOTTOM                              ║")
    print("║   Target: 'Is this near the deepest point in remaining window?' ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    jsonl_files = sorted(LOCAL_DATA_DIR.glob("*.jsonl"))
    print(f"\nProcessing {len(jsonl_files)} recordings...")

    all_dfs = []
    for i, fp in enumerate(jsonl_files, 1):
        tick_df = build_tick_features(fp)
        if tick_df is not None:
            all_dfs.append(tick_df)
        if i % 30 == 0:
            n_ticks = sum(len(d) for d in all_dfs)
            print(f"  [{i}/{len(jsonl_files)}] {len(all_dfs)} valid, {n_ticks} ticks, {_time.time()-t0:.1f}s")

    if not all_dfs:
        print("No valid data")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Dataset: {len(df)} ticks, {df['settle_id'].nunique()} settlements, {df['symbol'].nunique()} symbols")
    print(f"  Target stats (drop_remaining_bps): mean={df['target_drop_remaining'].mean():.1f}, median={df['target_drop_remaining'].median():.1f}")
    print(f"  Near bottom (10bps): {df['target_near_bottom_10'].mean()*100:.1f}% of ticks")
    print(f"  Bottom already passed: {df['target_bottom_passed'].mean()*100:.1f}% of ticks")

    # Train
    results, feature_cols = train_and_evaluate(df)

    # Backtest
    strats, exit_times = backtest_single_exit(df, results)

    elapsed = _time.time() - t0
    print(f"\n{'='*70}")
    print(f"RESEARCH COMPLETE [{elapsed:.1f}s]")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
