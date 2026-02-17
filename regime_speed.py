#!/usr/bin/env python3
"""
Regime Detection Speed & Prediction Experiment (v20b)

Q1: How fast can we detect a regime switch after it happens?
    - Ground truth: full-sample GMM labels
    - Online: classify using only past N bars, measure lag to correct detection

Q2: Can we predict regime switches before they happen?
    - Build features that might lead regime transitions
    - Train classifier: "will regime switch in next N bars?"
    - Evaluate precision/recall at various horizons
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)

warnings.filterwarnings("ignore")

from regime_detection import load_bars, compute_regime_features

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
START = "2025-01-01"
END = "2026-01-31"

CLUSTER_FEATURES = [
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "parkvol_1h", "parkvol_4h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h", "vol_ratio_bar",
    "efficiency_1h", "efficiency_2h", "efficiency_4h", "efficiency_8h",
    "ret_autocorr_1h", "ret_autocorr_4h",
    "adx_2h", "adx_4h",
    "trade_intensity_ratio",
    "bar_eff_1h", "bar_eff_4h",
    "imbalance_persistence",
    "large_trade_1h", "iti_cv_1h",
    "price_vs_sma_4h", "price_vs_sma_8h", "price_vs_sma_24h",
    "momentum_1h", "momentum_2h", "momentum_4h",
    "sign_persist_1h", "sign_persist_2h",
    "vol_sma_24h",
]


def prepare_and_label(df):
    """Prepare features and get ground-truth regime labels from full-sample GMM."""
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[cols].copy()
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    idx = X.index

    means = X.mean()
    stds = X.std().clip(lower=1e-10)
    X_scaled = (X - means) / stds

    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                          n_init=5, random_state=42)
    labels = gmm.fit_predict(X_scaled.values)

    # Ensure regime 0 = quiet (lower rvol_1h)
    r0_vol = df.loc[idx, "rvol_1h"].values[labels == 0].mean()
    r1_vol = df.loc[idx, "rvol_1h"].values[labels == 1].mean()
    if r0_vol > r1_vol:
        labels = 1 - labels

    return X_scaled, idx, labels, means, stds, gmm, cols


def find_transitions(labels):
    """Find indices where regime changes."""
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions.append(i)
    return transitions


# =========================================================================
# EXPERIMENT 1: Detection Latency
# =========================================================================
def exp_detection_latency(df, X_scaled, idx, labels, means, stds, gmm, cols, symbol):
    """
    How fast can we detect a regime switch?

    Method: At each bar, classify using a rolling window of past bars.
    After a ground-truth transition, count how many bars until the
    online classifier agrees with the new regime.
    """
    print(f"\n{'='*70}")
    print(f"  DETECTION LATENCY — {symbol}")
    print(f"{'='*70}")

    transitions = find_transitions(labels)
    print(f"  Total transitions: {len(transitions)}")

    # Method 1: Single-bar GMM probability
    # At each bar, use the already-fitted GMM to get P(regime)
    # This is the fastest possible — just evaluate the bar's features
    probs = gmm.predict_proba(X_scaled.values)
    # probs[:, 0] = P(quiet), probs[:, 1] = P(volatile)
    # But GMM components may be swapped — align with our labels
    # Check: for bars labeled 0, which component has higher prob?
    r0_mask = labels == 0
    if probs[r0_mask, 0].mean() < probs[r0_mask, 1].mean():
        probs = probs[:, ::-1]

    # Method 2: Rolling confidence threshold
    # After transition, how many bars until P(new_regime) > threshold?
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n  Method 1: Single-bar GMM posterior probability")
    print(f"  After a transition, how many bars until P(new_regime) > threshold?")
    print(f"  {'Threshold':>10s}  {'Median lag':>10s}  {'Mean lag':>10s}  {'P90 lag':>10s}  {'Max lag':>10s}  {'% detected <3':>14s}  {'% detected <12':>14s}")
    print(f"  {'-'*84}")

    for thresh in thresholds:
        lags = []
        for t_idx in transitions:
            new_regime = labels[t_idx]
            # Count bars until P(new_regime) > threshold
            for lag in range(0, min(100, len(labels) - t_idx)):
                p = probs[t_idx + lag, new_regime]
                if p >= thresh:
                    lags.append(lag)
                    break
            else:
                lags.append(100)  # not detected within 100 bars

        lags = np.array(lags)
        pct_lt3 = (lags < 3).mean() * 100
        pct_lt12 = (lags < 12).mean() * 100
        print(f"  {thresh:>10.1f}  {np.median(lags):>10.0f}  {np.mean(lags):>10.1f}  "
              f"{np.percentile(lags, 90):>10.0f}  {np.max(lags):>10.0f}  "
              f"{pct_lt3:>13.1f}%  {pct_lt12:>13.1f}%")

    # Method 3: Rolling window re-fit
    # Use a short rolling window to fit a local GMM and classify
    print(f"\n  Method 2: Rolling window classification")
    print(f"  Classify each bar using GMM fitted on last N bars")

    window_sizes = [12, 24, 48, 96, 144, 288]  # 1h, 2h, 4h, 8h, 12h, 24h

    print(f"  {'Window':>10s}  {'Accuracy':>10s}  {'Median lag':>10s}  {'Mean lag':>10s}  {'P90 lag':>10s}  {'% <3 bars':>10s}  {'% <12 bars':>10s}")
    print(f"  {'-'*74}")

    for win in window_sizes:
        # Classify each bar using rolling window
        online_labels = np.full(len(labels), -1)
        step = max(1, win // 6)  # evaluate every few bars for speed

        for i in range(win, len(labels), step):
            window_data = X_scaled.values[i - win:i]
            try:
                local_gmm = GaussianMixture(n_components=2, covariance_type="diag",
                                            n_init=1, random_state=42, max_iter=50)
                local_labels = local_gmm.fit_predict(window_data)

                # Align: check which local label corresponds to quiet
                local_rvol = X_scaled.values[i - win:i, 0]  # rvol_1h is first feature
                r0_mean = local_rvol[local_labels == 0].mean() if (local_labels == 0).any() else 0
                r1_mean = local_rvol[local_labels == 1].mean() if (local_labels == 1).any() else 0

                # Predict current bar
                pred = local_gmm.predict(X_scaled.values[i:i+1])[0]
                if r0_mean > r1_mean:
                    pred = 1 - pred
                online_labels[i] = pred
            except:
                pass

        # Fill gaps with nearest valid prediction
        valid = online_labels >= 0
        if valid.sum() < 100:
            continue

        # Compute accuracy on evaluated bars
        eval_mask = valid & (np.arange(len(labels)) >= win)
        acc = (online_labels[eval_mask] == labels[eval_mask]).mean()

        # Compute detection lag at transitions
        lags = []
        for t_idx in transitions:
            if t_idx < win:
                continue
            new_regime = labels[t_idx]
            for lag in range(0, min(100, len(labels) - t_idx)):
                check_idx = t_idx + lag
                if online_labels[check_idx] >= 0 and online_labels[check_idx] == new_regime:
                    lags.append(lag)
                    break
            else:
                lags.append(100)

        if len(lags) == 0:
            continue

        lags = np.array(lags)
        pct_lt3 = (lags < 3).mean() * 100
        pct_lt12 = (lags < 12).mean() * 100
        hrs = win * 5 / 60
        print(f"  {hrs:>8.0f}h  {acc:>10.3f}  {np.median(lags):>10.0f}  "
              f"{np.mean(lags):>10.1f}  {np.percentile(lags, 90):>10.0f}  "
              f"{pct_lt3:>9.1f}%  {pct_lt12:>9.1f}%")

    # Method 3: Exponential moving average of GMM probability
    print(f"\n  Method 3: EMA-smoothed GMM probability")
    print(f"  Smooth P(volatile) with EMA, trigger switch when crossing threshold")

    ema_spans = [3, 6, 12, 24]  # bars
    trigger_thresh = 0.5

    print(f"  {'EMA span':>10s}  {'Accuracy':>10s}  {'Median lag':>10s}  {'Mean lag':>10s}  {'P90 lag':>10s}  {'False sw/day':>12s}")
    print(f"  {'-'*66}")

    for span in ema_spans:
        # EMA of P(volatile)
        p_volatile = pd.Series(probs[:, 1])
        ema = p_volatile.ewm(span=span, adjust=False).mean().values

        ema_labels = (ema >= trigger_thresh).astype(int)

        # Accuracy
        acc = (ema_labels == labels).mean()

        # Detection lag
        lags = []
        for t_idx in transitions:
            new_regime = labels[t_idx]
            for lag in range(0, min(100, len(labels) - t_idx)):
                if ema_labels[t_idx + lag] == new_regime:
                    lags.append(lag)
                    break
            else:
                lags.append(100)

        lags = np.array(lags)

        # False switches per day
        ema_transitions = np.sum(np.diff(ema_labels) != 0)
        gt_transitions = len(transitions)
        false_switches = max(0, ema_transitions - gt_transitions)
        days = len(labels) / 288  # 288 bars per day
        false_per_day = false_switches / days

        mins = span * 5
        print(f"  {mins:>8.0f}m  {acc:>10.3f}  {np.median(lags):>10.0f}  "
              f"{np.mean(lags):>10.1f}  {np.percentile(lags, 90):>10.0f}  "
              f"{false_per_day:>11.1f}")

    # Summary: regime episode statistics
    print(f"\n  Regime episode statistics:")
    episodes = []
    current_regime = labels[0]
    ep_start = 0
    for i in range(1, len(labels)):
        if labels[i] != current_regime:
            episodes.append({"regime": current_regime, "start": ep_start,
                             "end": i - 1, "length": i - ep_start})
            current_regime = labels[i]
            ep_start = i
    episodes.append({"regime": current_regime, "start": ep_start,
                     "end": len(labels) - 1, "length": len(labels) - ep_start})

    ep_df = pd.DataFrame(episodes)
    for r in [0, 1]:
        rname = "quiet" if r == 0 else "volatile"
        subset = ep_df[ep_df["regime"] == r]["length"]
        print(f"    {rname:>10s}: n={len(subset):>5d}, median={subset.median():>5.0f} bars ({subset.median()*5:.0f}m), "
              f"mean={subset.mean():>6.1f}, P10={subset.quantile(0.1):>4.0f}, P90={subset.quantile(0.9):>5.0f}")

    very_short = ep_df[ep_df["length"] <= 3]
    print(f"    Episodes ≤3 bars (≤15min): {len(very_short)} ({len(very_short)/len(ep_df)*100:.1f}%) — these are noise/flickers")

    return probs, transitions, ep_df


# =========================================================================
# EXPERIMENT 2: Regime Switch Prediction
# =========================================================================
def exp_predict_switch(df, X_scaled, idx, labels, cols, symbol):
    """
    Can we predict regime switches before they happen?

    Approach:
    - Target: will regime switch in next N bars? (binary classification)
    - Features: current bar features + rate-of-change features
    - Train/test: 70/30 time split
    - Evaluate at multiple horizons: 6, 12, 24, 48 bars (30m to 4h ahead)
    """
    print(f"\n{'='*70}")
    print(f"  REGIME SWITCH PREDICTION — {symbol}")
    print(f"{'='*70}")

    transitions = set(find_transitions(labels))
    horizons = [6, 12, 24, 48]  # bars ahead

    # Build enhanced features for prediction
    # Add rate-of-change of key features (these might lead transitions)
    feature_df = pd.DataFrame(X_scaled.values, columns=cols)

    # Rate of change features
    for lookback in [3, 6, 12]:
        for feat in ["rvol_1h", "parkvol_1h", "trade_intensity_ratio", "vol_ratio_1h_24h"]:
            if feat in cols:
                fi = cols.index(feat)
                vals = X_scaled.values[:, fi]
                roc = np.zeros_like(vals)
                roc[lookback:] = vals[lookback:] - vals[:-lookback]
                feature_df[f"{feat}_roc{lookback}"] = roc

    # Acceleration (second derivative)
    for feat in ["rvol_1h", "parkvol_1h"]:
        if feat in cols:
            fi = cols.index(feat)
            vals = X_scaled.values[:, fi]
            d1 = np.zeros_like(vals)
            d1[1:] = vals[1:] - vals[:-1]
            d2 = np.zeros_like(vals)
            d2[1:] = d1[1:] - d1[:-1]
            feature_df[f"{feat}_accel"] = d2

    # Distance from regime boundary (GMM posterior)
    # Already have this from the GMM — add as feature
    # We'll compute it inline

    X_pred = feature_df.values
    n = len(X_pred)

    print(f"  Features for prediction: {X_pred.shape[1]}")
    print(f"  Total transitions: {len(transitions)}")

    # Train/test split (time-based)
    split = int(n * 0.7)

    for horizon in horizons:
        # Create target: will regime switch in next `horizon` bars?
        target = np.zeros(n, dtype=int)
        for t in transitions:
            # Mark bars in the window before the transition
            start = max(0, t - horizon)
            for j in range(start, t):
                target[j] = 1

        # Base rate
        base_rate = target.mean()
        train_rate = target[:split].mean()
        test_rate = target[split:].mean()

        X_train, X_test = X_pred[:split], X_pred[split:]
        y_train, y_test = target[:split], target[split:]

        # Skip warmup
        warmup = 288  # 1 day
        X_train = X_train[warmup:]
        y_train = y_train[warmup:]

        print(f"\n  Horizon: {horizon} bars ({horizon*5}min = {horizon*5/60:.1f}h)")
        print(f"  Base rate (switch within horizon): {base_rate:.3f} ({base_rate*100:.1f}%)")
        print(f"  Train: {len(X_train)} bars, Test: {len(X_test)} bars")

        # Model 1: Logistic Regression
        lr = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced", random_state=42)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_prob = lr.predict_proba(X_test)[:, 1]

        # Model 2: Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=50,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]

        # Model 3: Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_leaf=50,
                                        subsample=0.8, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_prob = gb.predict_proba(X_test)[:, 1]

        print(f"\n  {'Model':>20s}  {'Accuracy':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'AUC':>10s}")
        print(f"  {'-'*72}")

        for name, pred, prob in [("Logistic Reg", lr_pred, lr_prob),
                                  ("Random Forest", rf_pred, rf_prob),
                                  ("Gradient Boost", gb_pred, gb_prob)]:
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, prob)
            except:
                auc = 0.5
            print(f"  {name:>20s}  {acc:>10.3f}  {prec:>10.3f}  {rec:>10.3f}  {f1:>10.3f}  {auc:>10.3f}")

        # Feature importance from RF
        if horizon == 12:  # only print for 1h horizon
            importances = rf.feature_importances_
            feat_names = list(feature_df.columns)
            top_idx = np.argsort(importances)[::-1][:15]
            print(f"\n  Top 15 features for predicting switch (RF, {horizon*5}min horizon):")
            for rank, fi in enumerate(top_idx):
                print(f"    {rank+1:>2d}. {feat_names[fi]:>30s}  importance={importances[fi]:.4f}")

    # Practical analysis: early warning signal
    print(f"\n  --- Practical Early Warning Analysis ---")
    print(f"  Using best model (GB) at 12-bar (1h) horizon:")

    # Refit GB for 12-bar horizon
    horizon = 12
    target = np.zeros(n, dtype=int)
    for t in transitions:
        start = max(0, t - horizon)
        for j in range(start, t):
            target[j] = 1

    X_train, X_test = X_pred[:split], X_pred[split:]
    y_train, y_test = target[:split], target[split:]
    X_train = X_train[288:]
    y_train = y_train[288:]

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, min_samples_leaf=50,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    gb_prob = gb.predict_proba(X_test)[:, 1]

    # At various probability thresholds, what's the precision/recall?
    print(f"\n  {'P(switch) threshold':>20s}  {'Precision':>10s}  {'Recall':>10s}  {'Alerts/day':>10s}  {'True alerts/day':>15s}")
    print(f"  {'-'*67}")

    days = len(X_test) / 288
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (gb_prob >= thresh).astype(int)
        if pred.sum() == 0:
            continue
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        alerts_per_day = pred.sum() / days
        true_alerts_per_day = (pred & y_test).sum() / days
        print(f"  {thresh:>20.1f}  {prec:>10.3f}  {rec:>10.3f}  {alerts_per_day:>10.1f}  {true_alerts_per_day:>14.1f}")

    # How much advance warning do we actually get?
    print(f"\n  Advance warning analysis (bars before actual transition):")
    test_transitions = [t - split for t in transitions if t >= split and t < n]
    if test_transitions:
        advance_warnings = []
        for t_idx in test_transitions:
            if t_idx < 0 or t_idx >= len(gb_prob):
                continue
            # Look backwards from transition: when did P(switch) first exceed 0.3?
            for lookback in range(min(48, t_idx), -1, -1):
                check = t_idx - lookback
                if check >= 0 and gb_prob[check] >= 0.3:
                    advance_warnings.append(lookback)
                    break

        if advance_warnings:
            aw = np.array(advance_warnings)
            print(f"    Transitions in test set: {len(test_transitions)}")
            print(f"    Detected (P>0.3 before): {len(advance_warnings)} ({len(advance_warnings)/len(test_transitions)*100:.0f}%)")
            print(f"    Advance warning: median={np.median(aw):.0f} bars ({np.median(aw)*5:.0f}min), "
                  f"mean={np.mean(aw):.1f} bars ({np.mean(aw)*5:.0f}min)")
            print(f"    P10={np.percentile(aw, 10):.0f} bars, P90={np.percentile(aw, 90):.0f} bars")


# =========================================================================
# Main
# =========================================================================
def main():
    t0 = time.time()

    for symbol in SYMBOLS:
        sym_t0 = time.time()
        print(f"\n{'*'*70}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'*'*70}")

        print("  Loading bars...")
        df = load_bars(symbol, START, END)
        print("  Computing features...")
        df = compute_regime_features(df)

        print("  Preparing features and clustering...")
        X_scaled, idx, labels, means, stds, gmm, cols = prepare_and_label(df)
        print(f"  Bars: {len(labels)}, Transitions: {len(find_transitions(labels))}")

        # Q1: Detection latency
        probs, transitions, ep_df = exp_detection_latency(
            df, X_scaled, idx, labels, means, stds, gmm, cols, symbol)

        # Q2: Switch prediction
        exp_predict_switch(df, X_scaled, idx, labels, cols, symbol)

        elapsed = time.time() - sym_t0
        print(f"\n  {symbol} completed in {elapsed:.0f}s")

    total = time.time() - t0
    print(f"\n✅ All done in {total:.0f}s")


if __name__ == "__main__":
    main()
