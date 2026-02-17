#!/usr/bin/env python3
"""
Regime Classification Experiment (v20)

Questions answered:
1. How many distinct market regimes exist? (data-driven, not hand-picked)
2. What features distinguish one regime from another?
3. How much history (lookback) do we need to classify the current regime?
4. Are regimes consistent across assets?
5. How long do regimes last? How predictable are transitions?

Approach:
- Load 13 months of 5m bars with features (reuse regime_detection.py infra)
- Unsupervised clustering: GMM and KMeans with K=2..10
- Model selection: BIC, silhouette, Calinski-Harabasz, stability
- Feature importance: which features separate regimes best
- Minimum history: test classification accuracy with shrinking lookback windows
- Cross-asset: do the same regimes appear in BTC, ETH, SOL?
- Regime profiling: what does each regime look like? duration, vol, returns
- Signal performance per regime: do v19 signals work in specific regimes?
"""

import sys
import time
import argparse
import psutil
import warnings
import os
from pathlib import Path
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Reuse infrastructure from regime_detection.py
from regime_detection import load_bars, compute_regime_features, label_regimes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
START = "2025-01-01"
END = "2026-01-31"

# Features to use for clustering (backward-looking, computed in regime_detection.py)
CLUSTER_FEATURES = [
    # Volatility (most important from v8/v9)
    "rvol_1h", "rvol_2h", "rvol_4h", "rvol_8h", "rvol_24h",
    "parkvol_1h", "parkvol_4h",
    "vol_ratio_1h_24h", "vol_ratio_2h_24h", "vol_ratio_1h_8h",
    "vol_accel_1h", "vol_accel_4h",
    "vol_ratio_bar",
    # Trend/efficiency
    "efficiency_1h", "efficiency_2h", "efficiency_4h", "efficiency_8h",
    "ret_autocorr_1h", "ret_autocorr_4h",
    "adx_2h", "adx_4h",
    # Microstructure
    "trade_intensity_ratio",
    "bar_eff_1h", "bar_eff_4h",
    "imbalance_persistence",
    "large_trade_1h", "iti_cv_1h",
    # Price structure
    "price_vs_sma_4h", "price_vs_sma_8h", "price_vs_sma_24h",
    "momentum_1h", "momentum_2h", "momentum_4h",
    "sign_persist_1h", "sign_persist_2h",
    # Volume
    "vol_sma_24h",
]


def prepare_features(df):
    """Extract and standardize clustering features from bar dataframe."""
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[cols].copy()
    # Drop rows with NaN (warmup period)
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    idx = X.index

    # Standardize (z-score) — critical for clustering
    means = X.mean()
    stds = X.std().clip(lower=1e-10)
    X_scaled = (X - means) / stds

    return X_scaled, idx, cols, means, stds


# ---------------------------------------------------------------------------
# Experiment 1: Optimal number of regimes
# ---------------------------------------------------------------------------
def exp1_optimal_k(X_scaled, symbol):
    """Find optimal number of clusters using multiple criteria."""
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    print(f"\n{'='*70}")
    print(f"  EXP 1: OPTIMAL NUMBER OF REGIMES — {symbol}")
    print(f"  Samples: {len(X_scaled)}, Features: {X_scaled.shape[1]}")
    print(f"{'='*70}")

    K_range = range(2, 9)
    results = {
        "k": [], "kmeans_inertia": [], "kmeans_silhouette": [], "kmeans_ch": [],
        "gmm_bic": [], "gmm_aic": [], "gmm_silhouette": [], "gmm_ch": [],
    }

    X_arr = X_scaled.values

    for k in K_range:
        print(f"  K={k}...", end=" ", flush=True)

        # KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        km_labels = km.fit_predict(X_arr)
        km_sil = silhouette_score(X_arr, km_labels, sample_size=min(10000, len(X_arr)))
        km_ch = calinski_harabasz_score(X_arr, km_labels)

        # GMM (diag covariance for speed with 36 features)
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                               n_init=3, random_state=42, max_iter=200)
        gmm_labels = gmm.fit_predict(X_arr)
        gmm_bic = gmm.bic(X_arr)
        gmm_aic = gmm.aic(X_arr)
        gmm_sil = silhouette_score(X_arr, gmm_labels, sample_size=min(10000, len(X_arr)))
        gmm_ch = calinski_harabasz_score(X_arr, gmm_labels)

        results["k"].append(k)
        results["kmeans_inertia"].append(km.inertia_)
        results["kmeans_silhouette"].append(km_sil)
        results["kmeans_ch"].append(km_ch)
        results["gmm_bic"].append(gmm_bic)
        results["gmm_aic"].append(gmm_aic)
        results["gmm_silhouette"].append(gmm_sil)
        results["gmm_ch"].append(gmm_ch)

        print(f"KM_sil={km_sil:.3f} GMM_BIC={gmm_bic:.0f} GMM_sil={gmm_sil:.3f}")

    res_df = pd.DataFrame(results)

    # Find optimal K by each criterion
    best_km_sil = res_df.loc[res_df["kmeans_silhouette"].idxmax(), "k"]
    best_gmm_bic = res_df.loc[res_df["gmm_bic"].idxmin(), "k"]
    best_gmm_sil = res_df.loc[res_df["gmm_silhouette"].idxmax(), "k"]
    best_km_ch = res_df.loc[res_df["kmeans_ch"].idxmax(), "k"]

    print(f"\n  Optimal K by criterion:")
    print(f"    KMeans Silhouette → K={best_km_sil}")
    print(f"    KMeans Calinski-Harabasz → K={best_km_ch}")
    print(f"    GMM BIC → K={best_gmm_bic}")
    print(f"    GMM Silhouette → K={best_gmm_sil}")

    # Elbow detection for KMeans inertia
    inertias = res_df["kmeans_inertia"].values
    diffs = -np.diff(inertias)
    diffs2 = -np.diff(diffs)
    elbow_k = int(res_df["k"].values[1 + np.argmax(diffs2)]) if len(diffs2) > 0 else 3
    print(f"    KMeans Elbow → K={elbow_k}")

    # Consensus
    votes = [best_km_sil, best_gmm_bic, best_gmm_sil, best_km_ch, elbow_k]
    from collections import Counter
    vote_counts = Counter(votes)
    consensus_k = vote_counts.most_common(1)[0][0]
    print(f"\n  ★ CONSENSUS K = {consensus_k} (votes: {dict(vote_counts)})")

    return res_df, consensus_k


# ---------------------------------------------------------------------------
# Experiment 2: Cluster stability (bootstrap)
# ---------------------------------------------------------------------------
def exp2_cluster_stability(X_scaled, k_values, symbol):
    """Test cluster stability via bootstrap resampling."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    print(f"\n{'='*70}")
    print(f"  EXP 2: CLUSTER STABILITY (Bootstrap) — {symbol}")
    print(f"{'='*70}")

    X_arr = X_scaled.values
    n = len(X_arr)
    n_bootstrap = 20

    for k in k_values:
        # Reference clustering on full data
        ref_km = KMeans(n_clusters=k, n_init=10, random_state=42)
        ref_labels = ref_km.fit_predict(X_arr)

        ari_scores = []
        for b in range(n_bootstrap):
            # Bootstrap sample (80% of data)
            idx = np.random.RandomState(b).choice(n, size=int(0.8 * n), replace=False)
            km_b = KMeans(n_clusters=k, n_init=5, random_state=b)
            labels_b = km_b.fit_predict(X_arr[idx])

            # Compare: assign full data to bootstrap centroids
            from sklearn.metrics import pairwise_distances_argmin
            full_labels_b = pairwise_distances_argmin(X_arr, km_b.cluster_centers_)
            ari = adjusted_rand_score(ref_labels, full_labels_b)
            ari_scores.append(ari)

        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        print(f"  K={k}: ARI={mean_ari:.3f} ± {std_ari:.3f} "
              f"({'STABLE' if mean_ari > 0.7 else 'UNSTABLE' if mean_ari < 0.5 else 'MODERATE'})")


# ---------------------------------------------------------------------------
# Experiment 3: Regime profiling
# ---------------------------------------------------------------------------
def exp3_regime_profiles(df, X_scaled, idx, best_k, feature_cols, symbol):
    """Profile each regime: what does it look like?"""
    from sklearn.mixture import GaussianMixture

    print(f"\n{'='*70}")
    print(f"  EXP 3: REGIME PROFILES (K={best_k}) — {symbol}")
    print(f"{'='*70}")

    X_arr = X_scaled.values
    gmm = GaussianMixture(n_components=best_k, covariance_type="diag",
                           n_init=5, random_state=42, max_iter=300)
    labels = gmm.fit_predict(X_arr)
    probs = gmm.predict_proba(X_arr)

    # Assign labels back to dataframe
    df_sub = df.loc[idx].copy()
    df_sub["regime_cluster"] = labels
    df_sub["regime_confidence"] = probs.max(axis=1)

    # --- Regime sizes ---
    print(f"\n  Regime sizes:")
    for c in range(best_k):
        mask = labels == c
        pct = mask.mean() * 100
        print(f"    Regime {c}: {mask.sum():,} bars ({pct:.1f}%)")

    # --- Profile each regime by key features ---
    profile_features = [
        "rvol_1h", "rvol_4h", "rvol_24h",
        "parkvol_1h",
        "vol_ratio_1h_24h",
        "efficiency_1h", "efficiency_4h",
        "adx_4h",
        "bar_eff_4h",
        "trade_intensity_ratio",
        "momentum_4h",
        "price_vs_sma_24h",
    ]
    profile_features = [f for f in profile_features if f in df_sub.columns]

    print(f"\n  Regime profiles (mean values, raw scale):")
    header = f"  {'Feature':30s}" + "".join(f"  R{c:d}" + " " * 8 for c in range(best_k))
    print(header)
    print(f"  {'-'*30}" + "-" * 12 * best_k)

    for feat in profile_features:
        vals = []
        for c in range(best_k):
            mask = labels == c
            v = df_sub.loc[df_sub["regime_cluster"] == c, feat].mean()
            vals.append(v)
        val_str = "".join(f"  {v:+10.6f}" for v in vals)
        print(f"  {feat:30s}{val_str}")

    # --- Forward returns per regime ---
    print(f"\n  Forward returns per regime (next 48 bars = 4h):")
    if "fwd_vol" in df_sub.columns:
        for c in range(best_k):
            mask = df_sub["regime_cluster"] == c
            sub = df_sub[mask]
            fwd_v = sub["fwd_vol"].dropna()
            fwd_e = sub["fwd_efficiency"].dropna() if "fwd_efficiency" in sub.columns else pd.Series()
            # Compute forward return
            ret_4h = sub["returns"].rolling(48).sum().shift(-48).dropna()
            print(f"    Regime {c}: fwd_vol={fwd_v.mean():.6f} fwd_eff={fwd_e.mean():.3f} "
                  f"avg_4h_ret={ret_4h.mean()*10000:.1f}bps n={mask.sum()}")

    # --- Regime durations ---
    print(f"\n  Regime durations (consecutive bars in same regime):")
    run_lengths = {c: [] for c in range(best_k)}
    current_regime = labels[0]
    current_run = 1
    for i in range(1, len(labels)):
        if labels[i] == current_regime:
            current_run += 1
        else:
            run_lengths[current_regime].append(current_run)
            current_regime = labels[i]
            current_run = 1
    run_lengths[current_regime].append(current_run)

    for c in range(best_k):
        runs = run_lengths[c]
        if runs:
            print(f"    Regime {c}: median={np.median(runs):.0f} bars ({np.median(runs)*5:.0f} min), "
                  f"mean={np.mean(runs):.0f}, max={np.max(runs)}, "
                  f"n_episodes={len(runs)}")

    # --- Transition matrix ---
    print(f"\n  Transition matrix (row=from, col=to, %):")
    trans = np.zeros((best_k, best_k))
    for i in range(len(labels) - 1):
        trans[labels[i], labels[i+1]] += 1
    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_pct = trans / np.maximum(row_sums, 1) * 100

    header = "       " + "".join(f"  R{c:d}   " for c in range(best_k))
    print(f"  {header}")
    for r in range(best_k):
        row_str = "".join(f"  {trans_pct[r, c]:5.1f}%" for c in range(best_k))
        print(f"    R{r}{row_str}")

    # --- Confidence distribution ---
    print(f"\n  Classification confidence:")
    for c in range(best_k):
        mask = labels == c
        conf = probs[mask, c]
        print(f"    Regime {c}: mean_conf={conf.mean():.3f}, "
              f"P(conf>0.8)={( conf > 0.8).mean():.1%}, "
              f"P(conf>0.9)={( conf > 0.9).mean():.1%}")

    # Name the regimes based on profiles
    regime_names = _name_regimes(df_sub, labels, best_k, profile_features)
    print(f"\n  ★ Regime names (auto-assigned):")
    for c, name in regime_names.items():
        mask = labels == c
        print(f"    Regime {c} → {name} ({mask.sum():,} bars, {mask.mean()*100:.1f}%)")

    return gmm, labels, df_sub, regime_names


def _name_regimes(df_sub, labels, k, profile_features):
    """Auto-name regimes based on their feature profiles."""
    names = {}
    # Compute z-scores of regime means
    regime_means = {}
    for c in range(k):
        mask = df_sub["regime_cluster"] == c
        means = {}
        for f in ["rvol_4h", "efficiency_4h", "trade_intensity_ratio", "vol_ratio_1h_24h"]:
            if f in df_sub.columns:
                means[f] = df_sub.loc[mask, f].mean()
        regime_means[c] = means

    # Rank regimes by volatility
    vol_key = "rvol_4h"
    if vol_key in df_sub.columns:
        vol_ranks = sorted(regime_means.keys(), key=lambda c: regime_means[c].get(vol_key, 0))
    else:
        vol_ranks = list(range(k))

    eff_key = "efficiency_4h"

    for c in range(k):
        vol_rank = vol_ranks.index(c)  # 0=lowest vol, k-1=highest vol
        eff = regime_means[c].get(eff_key, 0.5)
        vol_ratio = regime_means[c].get("vol_ratio_1h_24h", 1.0)
        intensity = regime_means[c].get("trade_intensity_ratio", 1.0)

        # Naming logic
        if vol_rank <= k // 4:
            vol_label = "quiet"
        elif vol_rank >= k - k // 4 - 1:
            vol_label = "volatile"
        else:
            vol_label = "normal"

        if eff > 0.35:
            trend_label = "trending"
        elif eff < 0.20:
            trend_label = "choppy"
        else:
            trend_label = "ranging"

        if vol_ratio > 1.3:
            extra = " (vol expanding)"
        elif vol_ratio < 0.7:
            extra = " (vol contracting)"
        else:
            extra = ""

        names[c] = f"{vol_label}_{trend_label}{extra}"

    return names


# ---------------------------------------------------------------------------
# Experiment 4: Feature importance for regime separation
# ---------------------------------------------------------------------------
def exp4_feature_importance(X_scaled, labels, feature_cols, best_k, symbol):
    """Which features matter most for distinguishing regimes?"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    print(f"\n{'='*70}")
    print(f"  EXP 4: FEATURE IMPORTANCE — {symbol}")
    print(f"{'='*70}")

    X_arr = X_scaled.values

    # Method 1: ANOVA F-statistic (univariate)
    print(f"\n  Method 1: ANOVA F-statistic (univariate separability)")
    f_scores = []
    for i, col in enumerate(feature_cols):
        groups = [X_arr[labels == c, i] for c in range(best_k)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            f_stat, p_val = scipy_stats.f_oneway(*groups)
            f_scores.append((col, f_stat, p_val))

    f_scores.sort(key=lambda x: -x[1])
    print(f"  {'Feature':30s} {'F-stat':>10s} {'p-value':>12s}")
    print(f"  {'-'*55}")
    for col, f, p in f_scores[:15]:
        print(f"  {col:30s} {f:10.1f} {p:12.2e}")

    # Method 2: Random Forest feature importance (multivariate)
    print(f"\n  Method 2: Random Forest importance (multivariate)")
    # Use time-based split to avoid lookahead
    n = len(X_arr)
    train_idx = np.arange(int(0.7 * n))
    test_idx = np.arange(int(0.7 * n), n)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_arr[train_idx], labels[train_idx])
    train_acc = rf.score(X_arr[train_idx], labels[train_idx])
    test_acc = rf.score(X_arr[test_idx], labels[test_idx])
    print(f"  RF accuracy: train={train_acc:.3f}, test={test_acc:.3f}")

    importances = list(zip(feature_cols, rf.feature_importances_))
    importances.sort(key=lambda x: -x[1])
    print(f"\n  {'Feature':30s} {'Importance':>12s}")
    print(f"  {'-'*45}")
    for col, imp in importances[:15]:
        print(f"  {col:30s} {imp:12.4f}")

    # Method 3: Minimal feature set — how few features do we need?
    print(f"\n  Method 3: Minimal feature set analysis")
    top_features_ordered = [x[0] for x in importances]

    for n_feat in [3, 5, 8, 10, 15, 20, len(feature_cols)]:
        n_feat = min(n_feat, len(feature_cols))
        feat_idx = [feature_cols.index(f) for f in top_features_ordered[:n_feat]]
        X_sub = X_arr[:, feat_idx]
        rf_sub = RandomForestClassifier(n_estimators=100, max_depth=8,
                                         class_weight="balanced", random_state=42, n_jobs=-1)
        rf_sub.fit(X_sub[train_idx], labels[train_idx])
        acc = rf_sub.score(X_sub[test_idx], labels[test_idx])
        print(f"    Top {n_feat:2d} features → test accuracy: {acc:.3f}")

    return f_scores, importances


# ---------------------------------------------------------------------------
# Experiment 5: Minimum history needed
# ---------------------------------------------------------------------------
def exp5_minimum_history(df, X_scaled, idx, labels, feature_cols, best_k, symbol):
    """How much lookback history do we need to classify the current regime?"""
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score

    print(f"\n{'='*70}")
    print(f"  EXP 5: MINIMUM HISTORY NEEDED — {symbol}")
    print(f"{'='*70}")

    X_arr = X_scaled.values
    n = len(X_arr)

    # Reference: full-data GMM labels
    ref_labels = labels

    # Test: train GMM on different amounts of initial history, then classify rest
    history_sizes = [
        (288, "1 day"),
        (288 * 3, "3 days"),
        (288 * 7, "1 week"),
        (288 * 14, "2 weeks"),
        (288 * 30, "1 month"),
        (288 * 60, "2 months"),
        (288 * 90, "3 months"),
        (288 * 180, "6 months"),
    ]

    print(f"\n  Training GMM on different history lengths, evaluating on remaining data:")
    print(f"  {'History':15s} {'Bars':>8s} {'Train ARI':>10s} {'Test ARI':>10s} {'Test Acc':>10s}")
    print(f"  {'-'*60}")

    for hist_bars, hist_label in history_sizes:
        if hist_bars >= n * 0.9:
            continue

        # Train on first hist_bars
        X_train = X_arr[:hist_bars]
        X_test = X_arr[hist_bars:]
        ref_train = ref_labels[:hist_bars]
        ref_test = ref_labels[hist_bars:]

        gmm_h = GaussianMixture(n_components=best_k, covariance_type="diag",
                                  n_init=5, random_state=42, max_iter=300)
        gmm_h.fit(X_train)
        train_labels = gmm_h.predict(X_train)
        test_labels = gmm_h.predict(X_test)

        # ARI vs reference (need to handle label permutation)
        train_ari = adjusted_rand_score(ref_train, train_labels)
        test_ari = adjusted_rand_score(ref_test, test_labels)

        # Direct accuracy (after finding best label mapping)
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((best_k, best_k))
        for i in range(best_k):
            for j in range(best_k):
                cost[i, j] = -np.sum((test_labels == i) & (ref_test == j))
        row_ind, col_ind = linear_sum_assignment(cost)
        mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        mapped_labels = np.array([mapping.get(l, l) for l in test_labels])
        test_acc = (mapped_labels == ref_test).mean()

        print(f"  {hist_label:15s} {hist_bars:8d} {train_ari:10.3f} {test_ari:10.3f} {test_acc:10.3f}")

    # Also test: rolling window classification
    print(f"\n  Rolling window classification (train on last N bars, classify next bar):")
    window_sizes = [288, 288*3, 288*7, 288*14, 288*30]
    window_labels = ["1d", "3d", "1w", "2w", "1m"]

    for ws, wl in zip(window_sizes, window_labels):
        if ws + 288 >= n:
            continue

        # Sample every 288 bars (once per day) to keep it fast
        correct = 0
        total = 0
        sample_points = range(ws, n - 1, 288)

        for t in sample_points:
            X_window = X_arr[t - ws:t]
            gmm_w = GaussianMixture(n_components=best_k, covariance_type="diag",
                                     n_init=1, random_state=42, max_iter=100)
            try:
                gmm_w.fit(X_window)
                pred = gmm_w.predict(X_arr[t:t+1])[0]
                # Map to reference
                window_ref = ref_labels[t - ws:t]
                window_pred = gmm_w.predict(X_window)
                # Find mapping
                cost = np.zeros((best_k, best_k))
                for i in range(best_k):
                    for j in range(best_k):
                        cost[i, j] = -np.sum((window_pred == i) & (window_ref == j))
                row_ind, col_ind = linear_sum_assignment(cost)
                mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
                mapped_pred = mapping.get(pred, pred)
                if mapped_pred == ref_labels[t]:
                    correct += 1
                total += 1
            except Exception:
                pass

        acc = correct / max(total, 1)
        print(f"    Window={wl:4s} ({ws:6d} bars): accuracy={acc:.3f} (n={total})")


# ---------------------------------------------------------------------------
# Experiment 6: Cross-asset regime consistency
# ---------------------------------------------------------------------------
def exp6_cross_asset(all_data, best_k):
    """Do the same regimes appear across different assets?"""
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import adjusted_rand_score

    print(f"\n{'='*70}")
    print(f"  EXP 6: CROSS-ASSET REGIME CONSISTENCY")
    print(f"{'='*70}")

    if len(all_data) < 2:
        print("  Need at least 2 symbols for cross-asset comparison")
        return

    # Train GMM on each asset, then cross-predict
    models = {}
    labels_dict = {}
    X_dict = {}

    for sym, (X_scaled, idx, cols) in all_data.items():
        gmm = GaussianMixture(n_components=best_k, covariance_type="diag",
                               n_init=5, random_state=42, max_iter=300)
        labels = gmm.fit_predict(X_scaled.values)
        models[sym] = gmm
        labels_dict[sym] = labels
        X_dict[sym] = X_scaled.values

    # Cross-prediction: train on A, predict B
    print(f"\n  Cross-prediction accuracy (train on row, predict col):")
    symbols = list(all_data.keys())
    train_pred = "Train\\Predict"
    header = f"  {train_pred:15s}" + "".join(f"  {s:>10s}" for s in symbols)
    print(header)
    print(f"  {'-'*15}" + "-" * 12 * len(symbols))

    for sym_train in symbols:
        row = f"  {sym_train:15s}"
        for sym_test in symbols:
            pred_labels = models[sym_train].predict(X_dict[sym_test])
            ari = adjusted_rand_score(labels_dict[sym_test], pred_labels)
            row += f"  {ari:10.3f}"
        print(row)

    # Regime distribution comparison
    print(f"\n  Regime distribution comparison:")
    for sym in symbols:
        labels = labels_dict[sym]
        dist = [f"R{c}={np.mean(labels == c):.1%}" for c in range(best_k)]
        print(f"    {sym:10s}: {', '.join(dist)}")


# ---------------------------------------------------------------------------
# Experiment 7: Signal performance per regime
# ---------------------------------------------------------------------------
def exp7_signal_per_regime(df_sub, labels, best_k, regime_names, symbol):
    """Do any v19 signals work in specific regimes?"""
    print(f"\n{'='*70}")
    print(f"  EXP 7: SIGNAL EDGE PER REGIME — {symbol}")
    print(f"{'='*70}")

    # Compute simple signals and test per regime
    ret = df_sub["returns"].values
    n = len(ret)

    # Forward returns at different horizons
    fwd_1h = np.full(n, np.nan)
    fwd_4h = np.full(n, np.nan)
    for i in range(n - 48):
        fwd_1h[i] = np.sum(ret[i+1:i+13])  # 1h forward
        fwd_4h[i] = np.sum(ret[i+1:i+49])  # 4h forward

    # Simple signals
    signals = {}
    if "vol_imbalance" in df_sub.columns:
        vi = df_sub["vol_imbalance"].rolling(12).mean().values
        vi_z = (vi - np.nanmean(vi)) / max(np.nanstd(vi), 1e-10)
        signals["contrarian_imbalance"] = -vi_z  # fade imbalance

    if "momentum_4h" in df_sub.columns:
        mom = df_sub["momentum_4h"].values
        mom_z = (mom - np.nanmean(mom)) / max(np.nanstd(mom), 1e-10)
        signals["momentum_4h"] = mom_z

    if "bar_eff_4h" in df_sub.columns and "momentum_4h" in df_sub.columns:
        eff = df_sub["efficiency_4h"].values if "efficiency_4h" in df_sub.columns else df_sub["bar_eff_4h"].values
        mom = df_sub["momentum_4h"].values
        signals["efficiency_momentum"] = np.sign(mom) * eff

    print(f"\n  Signal IC (Information Coefficient) per regime:")
    print(f"  {'Signal':30s} {'Overall':>8s}" + "".join(f"  R{c}({regime_names.get(c,'?')[:12]:12s})" for c in range(best_k)))
    print(f"  {'-'*30}{'-'*10}" + "-" * 16 * best_k)

    FEE_BPS = 7  # round-trip fee

    for sig_name, sig_vals in signals.items():
        # Overall IC
        valid = ~np.isnan(sig_vals) & ~np.isnan(fwd_4h)
        if valid.sum() < 100:
            continue
        overall_ic = np.corrcoef(sig_vals[valid], fwd_4h[valid])[0, 1]

        regime_ics = []
        for c in range(best_k):
            mask = (labels == c) & valid
            if mask.sum() > 50:
                ic = np.corrcoef(sig_vals[mask], fwd_4h[mask])[0, 1]
                regime_ics.append(f"{ic:+.4f}")
            else:
                regime_ics.append("   N/A ")

        print(f"  {sig_name:30s} {overall_ic:+.4f}  " + "  ".join(f"{r:>14s}" for r in regime_ics))

    # Per-regime simple backtest: go long when signal > 1 std, short when < -1 std
    print(f"\n  Simple backtest per regime (threshold=1.0 z-score, 4h hold, 7bps fee):")
    for sig_name, sig_vals in signals.items():
        valid = ~np.isnan(sig_vals) & ~np.isnan(fwd_4h)
        if valid.sum() < 100:
            continue

        print(f"\n  {sig_name}:")
        for c in range(best_k):
            mask = (labels == c) & valid
            if mask.sum() < 50:
                continue
            sv = sig_vals[mask]
            fv = fwd_4h[mask]

            # Long signals
            long_mask = sv > 1.0
            short_mask = sv < -1.0

            if long_mask.sum() > 10:
                long_pnl = fv[long_mask] * 10000 - FEE_BPS
                avg_long = long_pnl.mean()
            else:
                avg_long = np.nan

            if short_mask.sum() > 10:
                short_pnl = -fv[short_mask] * 10000 - FEE_BPS
                avg_short = short_pnl.mean()
            else:
                avg_short = np.nan

            n_trades = long_mask.sum() + short_mask.sum()
            combined = np.nan
            if not np.isnan(avg_long) and not np.isnan(avg_short):
                combined = (avg_long * long_mask.sum() + avg_short * short_mask.sum()) / max(n_trades, 1)

            rname = regime_names.get(c, f"R{c}")[:20]
            print(f"    R{c} ({rname:20s}): trades={n_trades:5d}, "
                  f"avg_long={avg_long:+.1f}bps, avg_short={avg_short:+.1f}bps, "
                  f"combined={combined:+.1f}bps")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Regime Classification (v20)")
    parser.add_argument("--symbol", default="all", help="Symbol or 'all'")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()

    symbols = SYMBOLS if args.symbol == "all" else [args.symbol]

    print(f"{'#'*70}")
    print(f"  REGIME CLASSIFICATION EXPERIMENT (v20)")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {args.start} → {args.end}")
    print(f"  Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'#'*70}")

    all_data = {}  # symbol → (X_scaled, idx, cols)
    all_results = {}  # symbol → results dict
    consensus_ks = []

    for sym in symbols:
        t0 = time.time()
        print(f"\n\n{'*'*70}")
        print(f"  SYMBOL: {sym}")
        print(f"{'*'*70}")

        # Load and prepare data
        print(f"\n  Loading bars...")
        df = load_bars(sym, args.start, args.end)
        if df.empty:
            print(f"  No data for {sym}, skipping")
            continue

        print(f"  Computing features...")
        df = compute_regime_features(df)
        df = label_regimes(df, forward_window=48)

        print(f"  Preparing clustering features...")
        X_scaled, idx, cols, means, stds = prepare_features(df)
        print(f"  Features: {len(cols)}, Valid bars: {len(X_scaled)}")

        all_data[sym] = (X_scaled, idx, cols)

        # Exp 1: Optimal K
        res_df, consensus_k = exp1_optimal_k(X_scaled, sym)
        consensus_ks.append(consensus_k)

        # Exp 2: Stability for top candidates
        candidates = sorted(set([consensus_k, max(2, consensus_k-1), consensus_k+1]))
        candidates = [k for k in candidates if 2 <= k <= 8]
        exp2_cluster_stability(X_scaled, candidates, sym)

        # Exp 3: Regime profiles
        gmm, labels, df_sub, regime_names = exp3_regime_profiles(
            df, X_scaled, idx, consensus_k, cols, sym)

        # Exp 4: Feature importance
        f_scores, importances = exp4_feature_importance(
            X_scaled, labels, cols, consensus_k, sym)

        # Exp 5: Minimum history
        exp5_minimum_history(df, X_scaled, idx, labels, cols, consensus_k, sym)

        # Exp 7: Signal per regime
        exp7_signal_per_regime(df_sub, labels, consensus_k, regime_names, sym)

        elapsed = time.time() - t0
        print(f"\n  {sym} completed in {elapsed:.0f}s")
        mem = psutil.virtual_memory().used / (1024**3)
        print(f"  RAM: {mem:.1f}GB")

        all_results[sym] = {
            "consensus_k": consensus_k,
            "regime_names": regime_names,
            "labels": labels,
            "gmm": gmm,
        }

    # Exp 6: Cross-asset consistency
    if len(all_data) >= 2:
        # Use consensus K across all symbols
        from collections import Counter
        overall_k = Counter(consensus_ks).most_common(1)[0][0]
        print(f"\n  Overall consensus K across symbols: {overall_k}")
        exp6_cross_asset(all_data, overall_k)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    for sym, res in all_results.items():
        print(f"\n  {sym}: K={res['consensus_k']}")
        for c, name in res["regime_names"].items():
            mask = res["labels"] == c
            print(f"    R{c}: {name} ({mask.mean()*100:.1f}%)")

    print(f"\n✅ All done")


if __name__ == "__main__":
    main()
