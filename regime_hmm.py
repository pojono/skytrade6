#!/usr/bin/env python3
"""
HMM Regime Classification & Detection (v21)

Compare Hidden Markov Model vs Gaussian Mixture Model for:
1. Regime classification — does temporal modeling find better regimes?
2. Detection speed — does HMM filter noise and reduce false switches?
3. Prediction — does the transition matrix predict regime switches?

Key advantage of HMM over GMM:
- GMM treats each bar independently: P(state | features_t)
- HMM models transitions: P(state_t | state_{t-1}, features_t)
- This penalizes rapid switching → should reduce noise flickers
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

warnings.filterwarnings("ignore")

from regime_detection import load_bars, compute_regime_features

# ---------------------------------------------------------------------------
# Config — start with BTC only for quick evaluation
# ---------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT"]
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


def prepare_features(df):
    """Extract and standardize features."""
    cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[cols].copy()
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    idx = X.index

    means = X.mean()
    stds = X.std().clip(lower=1e-10)
    X_scaled = (X - means) / stds

    return X_scaled, idx, cols, means, stds


def align_labels(labels, df, idx, ref_col="rvol_1h"):
    """Ensure regime 0 = quiet (lower volatility)."""
    r0_vol = df.loc[idx, ref_col].values[labels == 0].mean()
    r1_vol = df.loc[idx, ref_col].values[labels == 1].mean()
    if r0_vol > r1_vol:
        labels = 1 - labels
    return labels


def find_transitions(labels):
    """Find indices where regime changes."""
    transitions = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions.append(i)
    return transitions


def episode_stats(labels):
    """Compute episode statistics."""
    episodes = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            episodes.append({"regime": current, "length": i - start})
            current = labels[i]
            start = i
    episodes.append({"regime": current, "length": len(labels) - start})
    return pd.DataFrame(episodes)


# =========================================================================
# EXPERIMENT 1: HMM vs GMM Classification
# =========================================================================
def exp1_classification(X_scaled, idx, df, symbol):
    """Compare HMM and GMM clustering quality."""
    print(f"\n{'='*70}")
    print(f"  EXP 1: HMM vs GMM CLASSIFICATION — {symbol}")
    print(f"  Samples: {len(X_scaled)}, Features: {X_scaled.shape[1]}")
    print(f"{'='*70}")

    X_arr = X_scaled.values

    results = {}

    # --- GMM baseline (same as v20) ---
    print(f"\n  --- GMM Baseline ---")
    for k in [2, 3, 4]:
        print(f"  GMM K={k}...", end=" ", flush=True)
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              n_init=5, random_state=42)
        gmm.fit(X_arr)
        labels = gmm.predict(X_arr)

        bic = gmm.bic(X_arr)
        aic = gmm.aic(X_arr)
        sil = silhouette_score(X_arr, labels, sample_size=min(10000, len(X_arr)))
        ll = gmm.score(X_arr) * len(X_arr)

        # Episode stats
        ep = episode_stats(labels)
        n_transitions = len(find_transitions(labels))
        short_pct = (ep["length"] <= 3).mean() * 100

        print(f"BIC={bic:.0f} AIC={aic:.0f} Sil={sil:.3f} "
              f"Transitions={n_transitions} Short≤3bars={short_pct:.1f}%")

        results[f"gmm_k{k}"] = {
            "model": gmm, "labels": labels, "bic": bic, "aic": aic,
            "sil": sil, "ll": ll, "transitions": n_transitions,
            "short_pct": short_pct, "episodes": ep
        }

    # --- HMM ---
    print(f"\n  --- HMM (Gaussian, diagonal covariance) ---")
    for k in [2, 3, 4]:
        print(f"  HMM K={k}...", end=" ", flush=True)
        t0 = time.time()

        best_hmm = None
        best_score = -np.inf

        # Multiple random inits (HMM is sensitive to initialization)
        for init in range(5):
            try:
                hmm = GaussianHMM(n_components=k, covariance_type="diag",
                                  n_iter=200, tol=1e-4, random_state=42 + init,
                                  verbose=False)
                hmm.fit(X_arr)
                score = hmm.score(X_arr)
                if score > best_score:
                    best_score = score
                    best_hmm = hmm
            except Exception as e:
                pass

        if best_hmm is None:
            print(f"FAILED")
            continue

        hmm = best_hmm
        labels = hmm.predict(X_arr)  # Viterbi decoding

        # BIC for HMM: -2*LL + k*log(n)
        # Number of free params: k*k-1 (transitions) + k*d (means) + k*d (diag covs) + k-1 (start probs)
        d = X_arr.shape[1]
        n_params = k * (k - 1) + k * d + k * d + (k - 1)
        ll = hmm.score(X_arr) * len(X_arr)
        bic = -2 * ll + n_params * np.log(len(X_arr))
        aic = -2 * ll + 2 * n_params

        sil = silhouette_score(X_arr, labels, sample_size=min(10000, len(X_arr)))

        ep = episode_stats(labels)
        n_transitions = len(find_transitions(labels))
        short_pct = (ep["length"] <= 3).mean() * 100

        elapsed = time.time() - t0
        print(f"BIC={bic:.0f} AIC={aic:.0f} Sil={sil:.3f} "
              f"Transitions={n_transitions} Short≤3bars={short_pct:.1f}% ({elapsed:.0f}s)")

        # Print transition matrix
        print(f"    Transition matrix:")
        for i in range(k):
            row = " ".join(f"{hmm.transmat_[i, j]:.4f}" for j in range(k))
            print(f"      State {i}: [{row}]")

        # Self-transition probability = expected regime duration
        for i in range(k):
            expected_dur = 1.0 / (1.0 - hmm.transmat_[i, i])
            print(f"      State {i} expected duration: {expected_dur:.1f} bars ({expected_dur*5:.0f}min)")

        results[f"hmm_k{k}"] = {
            "model": hmm, "labels": labels, "bic": bic, "aic": aic,
            "sil": sil, "ll": ll, "transitions": n_transitions,
            "short_pct": short_pct, "episodes": ep
        }

    # --- Compare GMM K=2 vs HMM K=2 ---
    print(f"\n  --- Head-to-Head: GMM K=2 vs HMM K=2 ---")
    gmm_labels = results["gmm_k2"]["labels"]
    hmm_labels = results["hmm_k2"]["labels"]

    # Align both to quiet=0
    gmm_labels = align_labels(gmm_labels.copy(), df, idx)
    hmm_labels = align_labels(hmm_labels.copy(), df, idx)

    # Agreement
    ari = adjusted_rand_score(gmm_labels, hmm_labels)
    agree_pct = (gmm_labels == hmm_labels).mean() * 100
    print(f"  Agreement: {agree_pct:.1f}% (ARI={ari:.3f})")

    # Episode comparison
    for name, labels in [("GMM", gmm_labels), ("HMM", hmm_labels)]:
        ep = episode_stats(labels)
        n_trans = len(find_transitions(labels))
        short = (ep["length"] <= 3).mean() * 100
        for r in [0, 1]:
            rname = "quiet" if r == 0 else "volatile"
            subset = ep[ep["regime"] == r]["length"]
            if len(subset) == 0:
                continue
            print(f"    {name} {rname:>10s}: n={len(subset):>5d}, "
                  f"median={subset.median():>5.0f} bars ({subset.median()*5:.0f}m), "
                  f"mean={subset.mean():>6.1f}, P10={subset.quantile(0.1):>4.0f}, "
                  f"P90={subset.quantile(0.9):>5.0f}")
        print(f"    {name} transitions: {n_trans}, short≤3bars: {short:.1f}%")

    # Regime profiles
    print(f"\n  --- Regime Profiles (HMM K=2) ---")
    for r in [0, 1]:
        rname = "quiet" if r == 0 else "volatile"
        mask = hmm_labels == r
        pct = mask.mean() * 100
        print(f"  Regime {r} ({rname}): {pct:.1f}% of bars")
        for feat in ["rvol_1h", "parkvol_1h", "trade_intensity_ratio",
                     "efficiency_1h", "vol_ratio_1h_24h"]:
            if feat in df.columns:
                vals = df.loc[idx, feat].values
                print(f"    {feat:>25s}: mean={vals[mask].mean():.6f} "
                      f"(quiet={vals[gmm_labels==0].mean():.6f} volatile={vals[gmm_labels==1].mean():.6f})")

    return results, gmm_labels, hmm_labels


# =========================================================================
# EXPERIMENT 2: Detection Speed — HMM Forward vs GMM Posterior
# =========================================================================
def exp2_detection(X_scaled, idx, df, results, gmm_labels, hmm_labels, symbol):
    """Compare detection latency: HMM forward algorithm vs GMM posterior."""
    print(f"\n{'='*70}")
    print(f"  EXP 2: DETECTION SPEED — HMM vs GMM — {symbol}")
    print(f"{'='*70}")

    X_arr = X_scaled.values
    n = len(X_arr)

    gmm = results["gmm_k2"]["model"]
    hmm = results["hmm_k2"]["model"]

    # --- GMM: single-bar posterior (baseline from v20) ---
    print(f"\n  --- Method A: GMM single-bar posterior ---")
    gmm_probs = gmm.predict_proba(X_arr)
    # Align probabilities with our label convention
    r0_mask = gmm_labels == 0
    if gmm_probs[r0_mask, 0].mean() < gmm_probs[r0_mask, 1].mean():
        gmm_probs = gmm_probs[:, ::-1]

    gmm_pred = (gmm_probs[:, 1] >= 0.5).astype(int)

    # --- HMM: Viterbi (global optimal) ---
    print(f"  --- Method B: HMM Viterbi (global, offline) ---")
    hmm_viterbi = hmm.predict(X_arr)
    hmm_viterbi = align_labels(hmm_viterbi.copy(), df, idx)

    # --- HMM: Forward algorithm (online, causal) ---
    print(f"  --- Method C: HMM Forward-filtered (online, causal) ---")
    # The forward algorithm gives P(state_t | obs_1..t) — truly online
    # hmmlearn doesn't expose forward directly, so we implement it
    hmm_forward_probs = _hmm_forward_filter(hmm, X_arr)

    # Align: check which state corresponds to quiet
    r0_mask_hmm = hmm_labels == 0
    if hmm_forward_probs[r0_mask_hmm, 0].mean() < hmm_forward_probs[r0_mask_hmm, 1].mean():
        hmm_forward_probs = hmm_forward_probs[:, ::-1]

    hmm_forward_pred = (hmm_forward_probs[:, 1] >= 0.5).astype(int)

    # --- HMM: Forward + EMA smoothing ---
    print(f"  --- Method D: HMM Forward + EMA(3) ---")
    p_vol_hmm = pd.Series(hmm_forward_probs[:, 1])
    hmm_ema3 = p_vol_hmm.ewm(span=3, adjust=False).mean().values
    hmm_ema3_pred = (hmm_ema3 >= 0.5).astype(int)

    # --- GMM + EMA(3) baseline ---
    p_vol_gmm = pd.Series(gmm_probs[:, 1])
    gmm_ema3 = p_vol_gmm.ewm(span=3, adjust=False).mean().values
    gmm_ema3_pred = (gmm_ema3 >= 0.5).astype(int)

    # --- Use HMM Viterbi as ground truth (it's the best full-sample labeling) ---
    # But also compare against GMM labels as ground truth
    for gt_name, gt_labels in [("GMM full-sample", gmm_labels),
                                ("HMM Viterbi full-sample", hmm_labels)]:
        print(f"\n  Ground truth: {gt_name}")
        transitions = find_transitions(gt_labels)
        print(f"  Transitions: {len(transitions)}")

        methods = [
            ("GMM posterior", gmm_pred, gmm_probs[:, 1]),
            ("GMM + EMA(3)", gmm_ema3_pred, gmm_ema3),
            ("HMM forward", hmm_forward_pred, hmm_forward_probs[:, 1]),
            ("HMM forward+EMA(3)", hmm_ema3_pred, hmm_ema3),
        ]

        print(f"\n  {'Method':>22s}  {'Accuracy':>8s}  {'Med lag':>7s}  {'Mean lag':>8s}  "
              f"{'P90 lag':>7s}  {'%<3bar':>6s}  {'%<12bar':>7s}  {'FalsSw/d':>8s}  {'Trans':>5s}")
        print(f"  {'-'*90}")

        for mname, pred, prob in methods:
            acc = (pred == gt_labels).mean()

            # Detection lag
            lags = []
            for t_idx in transitions:
                new_regime = gt_labels[t_idx]
                for lag in range(0, min(100, len(gt_labels) - t_idx)):
                    if pred[t_idx + lag] == new_regime:
                        lags.append(lag)
                        break
                else:
                    lags.append(100)

            lags = np.array(lags)
            pct_lt3 = (lags < 3).mean() * 100
            pct_lt12 = (lags < 12).mean() * 100

            # False switches
            pred_trans = np.sum(np.diff(pred) != 0)
            gt_trans = len(transitions)
            false_sw = max(0, pred_trans - gt_trans)
            days = len(gt_labels) / 288
            false_per_day = false_sw / days

            print(f"  {mname:>22s}  {acc:>8.3f}  {np.median(lags):>7.0f}  {np.mean(lags):>8.1f}  "
                  f"{np.percentile(lags, 90):>7.0f}  {pct_lt3:>5.1f}%  {pct_lt12:>6.1f}%  "
                  f"{false_per_day:>8.1f}  {pred_trans:>5d}")

    # --- Detailed threshold analysis for HMM forward ---
    print(f"\n  --- HMM Forward: Threshold Sensitivity ---")
    print(f"  Ground truth: HMM Viterbi")
    transitions = find_transitions(hmm_labels)

    print(f"  {'Threshold':>10s}  {'Accuracy':>8s}  {'Med lag':>7s}  {'Mean lag':>8s}  "
          f"{'P90 lag':>7s}  {'%<3bar':>6s}  {'FalsSw/d':>8s}")
    print(f"  {'-'*60}")

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (hmm_forward_probs[:, 1] >= thresh).astype(int)
        acc = (pred == hmm_labels).mean()

        lags = []
        for t_idx in transitions:
            new_regime = hmm_labels[t_idx]
            for lag in range(0, min(100, len(hmm_labels) - t_idx)):
                p = hmm_forward_probs[t_idx + lag, new_regime]
                if p >= thresh:
                    lags.append(lag)
                    break
            else:
                lags.append(100)

        lags = np.array(lags)
        pct_lt3 = (lags < 3).mean() * 100

        pred_trans = np.sum(np.diff(pred) != 0)
        false_sw = max(0, pred_trans - len(transitions))
        days = len(hmm_labels) / 288
        false_per_day = false_sw / days

        print(f"  {thresh:>10.1f}  {acc:>8.3f}  {np.median(lags):>7.0f}  {np.mean(lags):>8.1f}  "
              f"{np.percentile(lags, 90):>7.0f}  {pct_lt3:>5.1f}%  {false_per_day:>8.1f}")

    return hmm_forward_probs, gmm_probs


def _hmm_forward_filter(hmm, X):
    """
    Run HMM forward algorithm to get P(state_t | obs_1..t).
    This is the online/causal version — only uses past observations.
    """
    n = len(X)
    k = hmm.n_components

    # Get log emission probabilities for each observation
    # hmmlearn stores means_ and covars_ for Gaussian emissions
    from scipy.stats import multivariate_normal

    # Pre-compute log-likelihoods for all observations
    log_lik = np.zeros((n, k))
    for j in range(k):
        mean = hmm.means_[j]
        cov = np.diag(hmm.covars_[j]) if hmm.covars_.ndim == 2 else hmm.covars_[j]
        try:
            log_lik[:, j] = multivariate_normal.logpdf(X, mean=mean, cov=cov)
        except:
            # Fallback: compute element-wise for numerical stability
            diff = X - mean
            if hmm.covars_.ndim == 2:
                var = hmm.covars_[j]
            else:
                var = np.diag(hmm.covars_[j])
            log_lik[:, j] = -0.5 * np.sum(diff**2 / var, axis=1) - 0.5 * np.sum(np.log(var)) - 0.5 * X.shape[1] * np.log(2 * np.pi)

    # Forward pass
    log_startprob = np.log(hmm.startprob_ + 1e-300)
    log_transmat = np.log(hmm.transmat_ + 1e-300)

    # alpha[t, j] = P(obs_1..t, state_t=j)
    log_alpha = np.zeros((n, k))

    # Initialize
    log_alpha[0] = log_startprob + log_lik[0]

    # Forward recursion
    for t in range(1, n):
        for j in range(k):
            # log_alpha[t, j] = log_lik[t, j] + logsumexp(log_alpha[t-1, :] + log_transmat[:, j])
            log_alpha[t, j] = log_lik[t, j] + _logsumexp(log_alpha[t-1] + log_transmat[:, j])

    # Convert to posterior: P(state_t | obs_1..t) = alpha[t] / sum(alpha[t])
    posteriors = np.zeros((n, k))
    for t in range(n):
        log_norm = _logsumexp(log_alpha[t])
        posteriors[t] = np.exp(log_alpha[t] - log_norm)

    return posteriors


def _logsumexp(x):
    """Numerically stable log-sum-exp."""
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


# =========================================================================
# EXPERIMENT 3: HMM Transition Matrix for Prediction
# =========================================================================
def exp3_prediction(X_scaled, idx, df, results, hmm_labels, hmm_forward_probs, symbol):
    """
    Can HMM transition probabilities predict regime switches?

    The HMM gives us P(switch) = 1 - P(stay in current state) from the
    transition matrix. Combined with forward-filtered state probabilities,
    this gives a principled switch probability at each bar.
    """
    print(f"\n{'='*70}")
    print(f"  EXP 3: HMM-BASED PREDICTION — {symbol}")
    print(f"{'='*70}")

    hmm = results["hmm_k2"]["model"]
    X_arr = X_scaled.values
    n = len(X_arr)
    transitions = set(find_transitions(hmm_labels))
    cols = list(X_scaled.columns)

    # --- Method 1: Pure HMM transition probability ---
    # P(switch at t+1) = sum_j P(state_t=j) * P(state_{t+1}!=j | state_t=j)
    print(f"\n  --- Method 1: HMM Transition Probability ---")
    p_switch_hmm = np.zeros(n)
    for t in range(n):
        for j in range(hmm.n_components):
            p_stay = hmm.transmat_[j, j]
            p_switch_hmm[t] += hmm_forward_probs[t, j] * (1.0 - p_stay)

    # Evaluate as a predictor
    horizons = [6, 12, 24, 48]
    split = int(n * 0.7)

    for horizon in horizons:
        target = np.zeros(n, dtype=int)
        for t in transitions:
            start = max(0, t - horizon)
            for j in range(start, t):
                target[j] = 1

        base_rate = target[split:].mean()
        print(f"\n  Horizon: {horizon} bars ({horizon*5}min), Base rate: {base_rate:.3f}")

        # HMM transition probability as predictor
        for thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
            pred = (p_switch_hmm[split:] >= thresh).astype(int)
            if pred.sum() == 0 or pred.sum() == len(pred):
                continue
            prec = precision_score(target[split:], pred, zero_division=0)
            rec = recall_score(target[split:], pred, zero_division=0)
            f1 = f1_score(target[split:], pred, zero_division=0)
            try:
                auc = roc_auc_score(target[split:], p_switch_hmm[split:])
            except:
                auc = 0.5
            alerts_per_day = pred.sum() / (len(pred) / 288)
            print(f"    P(switch)>{thresh:.2f}: Prec={prec:.3f} Rec={rec:.3f} "
                  f"F1={f1:.3f} AUC={auc:.3f} Alerts/day={alerts_per_day:.0f}")

    # --- Method 2: HMM features + ML (same as v20 but with HMM-derived features) ---
    print(f"\n  --- Method 2: ML with HMM-augmented features ---")

    # Build feature matrix: original features + HMM-derived
    feature_df = pd.DataFrame(X_arr, columns=cols)

    # Add HMM-derived features
    feature_df["hmm_p_volatile"] = hmm_forward_probs[:, 1]
    feature_df["hmm_p_switch"] = p_switch_hmm
    feature_df["hmm_p_volatile_ema3"] = pd.Series(hmm_forward_probs[:, 1]).ewm(span=3).mean().values
    feature_df["hmm_p_volatile_ema12"] = pd.Series(hmm_forward_probs[:, 1]).ewm(span=12).mean().values

    # Rate of change of HMM probability
    for lb in [3, 6, 12]:
        vals = hmm_forward_probs[:, 1]
        roc = np.zeros_like(vals)
        roc[lb:] = vals[lb:] - vals[:-lb]
        feature_df[f"hmm_p_vol_roc{lb}"] = roc

    # Rate of change of key features
    for lb in [3, 6, 12]:
        for feat in ["rvol_1h", "parkvol_1h", "trade_intensity_ratio", "vol_ratio_1h_24h"]:
            if feat in cols:
                fi = cols.index(feat)
                vals = X_arr[:, fi]
                roc = np.zeros_like(vals)
                roc[lb:] = vals[lb:] - vals[:-lb]
                feature_df[f"{feat}_roc{lb}"] = roc

    # Acceleration
    for feat in ["rvol_1h", "parkvol_1h"]:
        if feat in cols:
            fi = cols.index(feat)
            vals = X_arr[:, fi]
            d1 = np.zeros_like(vals)
            d1[1:] = vals[1:] - vals[:-1]
            d2 = np.zeros_like(vals)
            d2[1:] = d1[1:] - d1[:-1]
            feature_df[f"{feat}_accel"] = d2

    X_pred = feature_df.values
    n_feat = X_pred.shape[1]
    print(f"  Total features: {n_feat} (35 base + {n_feat - 35} HMM-derived + rate-of-change)")

    # Also build a version WITHOUT HMM features for fair comparison
    base_feat_count = len(cols)
    # Add same ROC/accel features as v20 (without HMM)
    v20_feature_df = pd.DataFrame(X_arr, columns=cols)
    for lb in [3, 6, 12]:
        for feat in ["rvol_1h", "parkvol_1h", "trade_intensity_ratio", "vol_ratio_1h_24h"]:
            if feat in cols:
                fi = cols.index(feat)
                vals = X_arr[:, fi]
                roc = np.zeros_like(vals)
                roc[lb:] = vals[lb:] - vals[:-lb]
                v20_feature_df[f"{feat}_roc{lb}"] = roc
    for feat in ["rvol_1h", "parkvol_1h"]:
        if feat in cols:
            fi = cols.index(feat)
            vals = X_arr[:, fi]
            d1 = np.zeros_like(vals)
            d1[1:] = vals[1:] - vals[:-1]
            d2 = np.zeros_like(vals)
            d2[1:] = d1[1:] - d1[:-1]
            v20_feature_df[f"{feat}_accel"] = d2
    X_v20 = v20_feature_df.values

    warmup = 288

    print(f"\n  Comparing: v20 features ({X_v20.shape[1]}) vs v21 HMM-augmented ({X_pred.shape[1]})")

    for horizon in [12, 24]:
        target = np.zeros(n, dtype=int)
        for t in transitions:
            start = max(0, t - horizon)
            for j in range(start, t):
                target[j] = 1

        base_rate = target[split:].mean()

        X_train_v20, X_test_v20 = X_v20[warmup:split], X_v20[split:]
        X_train_v21, X_test_v21 = X_pred[warmup:split], X_pred[split:]
        y_train, y_test = target[warmup:split], target[split:]

        print(f"\n  Horizon: {horizon} bars ({horizon*5}min = {horizon*5/60:.1f}h), "
              f"Base rate: {base_rate:.3f}")
        print(f"  {'Features':>12s}  {'Model':>16s}  {'Accuracy':>8s}  {'Precision':>9s}  "
              f"{'Recall':>6s}  {'F1':>6s}  {'AUC':>6s}")
        print(f"  {'-'*72}")

        for feat_name, X_tr, X_te in [("v20 (no HMM)", X_train_v20, X_test_v20),
                                       ("v21 (+HMM)", X_train_v21, X_test_v21)]:
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                        min_samples_leaf=50, class_weight="balanced",
                                        random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_train)
            rf_pred = rf.predict(X_te)
            rf_prob = rf.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_test, rf_pred)
            prec = precision_score(y_test, rf_pred, zero_division=0)
            rec = recall_score(y_test, rf_pred, zero_division=0)
            f1 = f1_score(y_test, rf_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, rf_prob)
            except:
                auc = 0.5

            print(f"  {feat_name:>12s}  {'Random Forest':>16s}  {acc:>8.3f}  {prec:>9.3f}  "
                  f"{rec:>6.3f}  {f1:>6.3f}  {auc:>6.3f}")

            # Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                            min_samples_leaf=50, subsample=0.8,
                                            random_state=42)
            gb.fit(X_tr, y_train)
            gb_pred = gb.predict(X_te)
            gb_prob = gb.predict_proba(X_te)[:, 1]

            acc = accuracy_score(y_test, gb_pred)
            prec = precision_score(y_test, gb_pred, zero_division=0)
            rec = recall_score(y_test, gb_pred, zero_division=0)
            f1 = f1_score(y_test, gb_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, gb_prob)
            except:
                auc = 0.5

            print(f"  {feat_name:>12s}  {'Gradient Boost':>16s}  {acc:>8.3f}  {prec:>9.3f}  "
                  f"{rec:>6.3f}  {f1:>6.3f}  {auc:>6.3f}")

        # Feature importance for v21 RF at 12-bar horizon
        if horizon == 12:
            rf21 = RandomForestClassifier(n_estimators=100, max_depth=8,
                                          min_samples_leaf=50, class_weight="balanced",
                                          random_state=42, n_jobs=-1)
            rf21.fit(X_train_v21, y_train)
            importances = rf21.feature_importances_
            feat_names = list(feature_df.columns)
            top_idx = np.argsort(importances)[::-1][:15]
            print(f"\n  Top 15 features (v21 RF, 1h horizon):")
            for rank, fi in enumerate(top_idx):
                marker = " ★" if "hmm" in feat_names[fi] else ""
                print(f"    {rank+1:>2d}. {feat_names[fi]:>30s}  importance={importances[fi]:.4f}{marker}")


# =========================================================================
# EXPERIMENT 4: Noise Filtering — Does HMM Solve the Flicker Problem?
# =========================================================================
def exp4_noise(results, gmm_labels, hmm_labels, hmm_forward_probs, symbol):
    """Analyze whether HMM reduces the 53% noise flicker problem."""
    print(f"\n{'='*70}")
    print(f"  EXP 4: NOISE FILTERING — {symbol}")
    print(f"{'='*70}")

    # Compare episode distributions
    for name, labels in [("GMM (v20)", gmm_labels), ("HMM Viterbi", hmm_labels)]:
        ep = episode_stats(labels)
        n_trans = len(find_transitions(labels))

        print(f"\n  {name}:")
        print(f"    Total transitions: {n_trans}")
        print(f"    Transitions/day: {n_trans / (len(labels)/288):.1f}")

        for threshold in [1, 2, 3, 5, 10, 20]:
            short = (ep["length"] <= threshold).mean() * 100
            print(f"    Episodes ≤{threshold:>2d} bars (≤{threshold*5:>3d}min): {short:.1f}%")

    # HMM forward with different thresholds
    print(f"\n  HMM Forward-filtered with confirmation:")
    for confirm_bars in [0, 1, 2, 3, 5]:
        # Require P(new_regime) > 0.5 for confirm_bars consecutive bars
        pred = np.zeros(len(hmm_labels), dtype=int)
        current = 0
        count = 0
        for t in range(len(hmm_labels)):
            new = 1 if hmm_forward_probs[t, 1] >= 0.5 else 0
            if new != current:
                count += 1
                if count >= confirm_bars:
                    current = new
                    count = 0
            else:
                count = 0
            pred[t] = current

        ep = episode_stats(pred)
        n_trans = len(find_transitions(pred))
        short = (ep["length"] <= 3).mean() * 100
        acc = (pred == hmm_labels).mean()

        # Detection lag
        transitions = find_transitions(hmm_labels)
        lags = []
        for t_idx in transitions:
            new_regime = hmm_labels[t_idx]
            for lag in range(0, min(100, len(hmm_labels) - t_idx)):
                if pred[t_idx + lag] == new_regime:
                    lags.append(lag)
                    break
            else:
                lags.append(100)
        lags = np.array(lags)

        print(f"    Confirm={confirm_bars}: Acc={acc:.3f} Trans={n_trans} "
              f"Short≤3={short:.1f}% MedLag={np.median(lags):.0f} MeanLag={np.mean(lags):.1f}")


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

        print("  Preparing features...")
        X_scaled, idx, cols, means, stds = prepare_features(df)
        print(f"  Bars: {len(X_scaled)}, Features: {X_scaled.shape[1]}")

        # Exp 1: Classification
        results, gmm_labels, hmm_labels = exp1_classification(X_scaled, idx, df, symbol)

        # Exp 2: Detection speed
        hmm_fwd_probs, gmm_probs = exp2_detection(
            X_scaled, idx, df, results, gmm_labels, hmm_labels, symbol)

        # Exp 3: Prediction
        exp3_prediction(X_scaled, idx, df, results, hmm_labels, hmm_fwd_probs, symbol)

        # Exp 4: Noise filtering
        exp4_noise(results, gmm_labels, hmm_labels, hmm_fwd_probs, symbol)

        elapsed = time.time() - sym_t0
        print(f"\n  {symbol} completed in {elapsed:.0f}s")

    total = time.time() - t0
    print(f"\n✅ All done in {total:.0f}s")


if __name__ == "__main__":
    main()
