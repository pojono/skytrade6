#!/usr/bin/env python3
"""
OI Regime Lead/Lag Analysis

Core question: Does OI velocity spike BEFORE OHLCV volatility rises?
If yes, OI could provide early warning of regime transitions,
beating HMM's 10-15 minute detection lag.

Causal hypothesis:
  Positions open (OI spikes) → Price moves (OHLCV vol rises) → Regime detector catches it

Method:
1. Identify all quiet→volatile regime transitions (ground truth from GMM/HMM)
2. For each transition, look at OI velocity in the bars BEFORE the transition
3. Compare: when does OI velocity first spike vs when does OHLCV vol first spike?
4. Measure lead time in bars (each bar = 5 min)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path("./parquet")
SOURCE = "bybit_futures"
INTERVAL_5M_US = 300_000_000

SYMBOL = "BTCUSDT"
START_DATE = "2025-05-12"
END_DATE = "2025-08-08"


def load_ohlcv_bars():
    """Load 5-min OHLCV bars with microstructure features."""
    from regime_detection import load_bars, compute_regime_features
    print("Loading OHLCV bars...")
    df = load_bars(SYMBOL, START_DATE, END_DATE)
    if df.empty:
        return df
    df = compute_regime_features(df)
    print(f"  OHLCV: {len(df)} bars")
    return df


def load_ticker_5m():
    """Load pre-built ticker 5-min features."""
    ticker_dir = PARQUET_DIR / SYMBOL / "ticker"
    dates = pd.date_range(START_DATE, END_DATE)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = ticker_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Ticker: {len(df)} rows (5-second)")
    return df


def build_oi_velocity_5m(ticker_df):
    """Build OI velocity features at 5-min resolution from 5-second data."""
    df = ticker_df.copy()
    df["bucket"] = (df["timestamp_us"].values // INTERVAL_5M_US) * INTERVAL_5M_US
    df["oi_pct_change"] = df["open_interest"].pct_change() * 100

    features = []
    for bkt, grp in df.groupby("bucket"):
        if len(grp) < 10:
            continue
        oi = grp["open_interest"].values
        oi_pct = grp["oi_pct_change"].values
        oi_pct_clean = oi_pct[~np.isnan(oi_pct)]

        if len(oi_pct_clean) < 5:
            continue

        features.append({
            "timestamp_us": bkt,
            "oi_vel_mean": np.mean(np.abs(oi_pct_clean)),
            "oi_vel_max": np.max(np.abs(oi_pct_clean)),
            "oi_vel_std": np.std(oi_pct_clean),
            "oi_spike_count": np.sum(np.abs(oi_pct_clean) > 0.05),
            "oi_large_spike_count": np.sum(np.abs(oi_pct_clean) > 0.10),
            "oi_change_5m": (oi[-1] - oi[0]) / max(oi[0], 1) * 100,
            "oi_level": oi[-1],
            "funding_rate": grp["funding_rate"].iloc[-1],
            "mark_index_spread": np.mean(
                (grp["mark_price"].values - grp["index_price"].values) /
                grp["index_price"].values * 10000
            ),
        })

    feat_df = pd.DataFrame(features)

    # Rolling OI velocity features
    for w, name in [(3, "15m"), (12, "1h"), (48, "4h")]:
        feat_df[f"oi_vel_mean_{name}"] = feat_df["oi_vel_mean"].rolling(w, min_periods=1).mean()
        feat_df[f"oi_spike_count_{name}"] = feat_df["oi_spike_count"].rolling(w, min_periods=1).sum()

    print(f"  OI velocity features: {len(feat_df)} bars")
    return feat_df


def identify_regime_transitions(ohlcv_df):
    """Identify quiet→volatile transitions using GMM."""
    print("\nFitting GMM for regime labels...")

    regime_cols = [c for c in [
        "rvol_1h", "rvol_4h", "rvol_24h", "parkvol_1h",
        "vol_ratio_1h_24h", "efficiency_1h", "efficiency_4h",
        "adx_4h", "bar_eff_4h", "trade_intensity_ratio",
    ] if c in ohlcv_df.columns]

    valid = ohlcv_df[regime_cols].notna().all(axis=1)
    X = ohlcv_df.loc[valid, regime_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                           n_init=10, random_state=42, max_iter=300)
    labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    # Ensure label 0 = quiet, 1 = volatile
    if np.mean(ohlcv_df.loc[valid, "rvol_1h"].values[labels == 0]) > \
       np.mean(ohlcv_df.loc[valid, "rvol_1h"].values[labels == 1]):
        labels = 1 - labels
        probs = probs[:, ::-1]

    regime = np.full(len(ohlcv_df), -1, dtype=np.int8)
    regime[valid.values] = labels
    p_volatile = np.full(len(ohlcv_df), np.nan)
    p_volatile[valid.values] = probs[:, 1]

    ohlcv_df["regime"] = regime
    ohlcv_df["p_volatile"] = p_volatile

    # Find quiet→volatile transitions
    transitions = []
    for i in range(1, len(regime)):
        if regime[i-1] == 0 and regime[i] == 1:
            transitions.append(i)

    n_quiet = (regime == 0).sum()
    n_vol = (regime == 1).sum()
    print(f"  Quiet: {n_quiet} bars ({n_quiet/len(regime)*100:.1f}%)")
    print(f"  Volatile: {n_vol} bars ({n_vol/len(regime)*100:.1f}%)")
    print(f"  Transitions quiet→volatile: {len(transitions)}")

    return transitions


def analyze_lead_lag(ohlcv_df, oi_df, transitions):
    """
    For each quiet→volatile transition:
    - Look at bars [-24, ..., -1, 0, +1, ..., +12] around the transition
    - Track when OI velocity first spikes vs when OHLCV vol first spikes
    """
    print(f"\n{'='*70}")
    print(f"  LEAD/LAG ANALYSIS: OI Velocity vs OHLCV Volatility")
    print(f"  Around {len(transitions)} quiet→volatile transitions")
    print(f"{'='*70}")

    # Merge OI features into OHLCV
    merged = pd.merge_asof(
        ohlcv_df.sort_values("timestamp_us"),
        oi_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    # Features to track
    oi_features = ["oi_vel_mean", "oi_vel_max", "oi_spike_count",
                   "oi_large_spike_count", "oi_change_5m",
                   "oi_vel_mean_15m", "oi_vel_mean_1h",
                   "oi_spike_count_15m", "oi_spike_count_1h"]
    ohlcv_features = ["rvol_1h", "parkvol_1h", "trade_intensity_ratio",
                      "vol_ratio_1h_24h"]

    all_features = oi_features + ohlcv_features

    # Window around transitions
    lookback = 24  # 2 hours before
    lookforward = 12  # 1 hour after
    window = range(-lookback, lookforward + 1)

    # Collect feature values around transitions
    profiles = {f: {t: [] for t in window} for f in all_features}

    valid_transitions = 0
    for trans_idx in transitions:
        if trans_idx < lookback or trans_idx + lookforward >= len(merged):
            continue
        valid_transitions += 1
        for offset in window:
            idx = trans_idx + offset
            for feat in all_features:
                val = merged.iloc[idx].get(feat, np.nan)
                if not np.isnan(val) if isinstance(val, float) else True:
                    profiles[feat][offset].append(float(val))

    print(f"  Valid transitions (with full window): {valid_transitions}")

    # Compute z-scores relative to quiet baseline (bars -24 to -13)
    print(f"\n  Z-score of features around transitions (baseline = bars -24 to -13):")
    print(f"  Positive z = feature is elevated above quiet baseline")
    print(f"  {'Feature':30s} | {'−12':>6s} {'−9':>6s} {'−6':>6s} {'−3':>6s} {'−2':>6s} {'−1':>6s} {'  0':>6s} {'+1':>6s} {'+3':>6s} {'+6':>6s}")
    print(f"  {'-'*30}-+-{'-'*66}")

    # Track first-spike bar for each feature
    first_spike = {}

    for feat in all_features:
        # Baseline: bars -24 to -13
        baseline_vals = []
        for t in range(-24, -12):
            baseline_vals.extend(profiles[feat][t])
        if not baseline_vals:
            continue
        bl_mean = np.mean(baseline_vals)
        bl_std = np.std(baseline_vals)
        if bl_std < 1e-10:
            continue

        zscores = {}
        for t in window:
            vals = profiles[feat][t]
            if vals:
                zscores[t] = (np.mean(vals) - bl_mean) / bl_std
            else:
                zscores[t] = 0

        # Print selected bars
        tag = "[OI]" if feat in oi_features else "[OH]"
        selected = [-12, -9, -6, -3, -2, -1, 0, 1, 3, 6]
        vals_str = " ".join(f"{zscores.get(t, 0):>+6.2f}" for t in selected)
        print(f"  {tag} {feat:25s} | {vals_str}")

        # Find first bar where z > 1.0 (sustained for 2+ bars)
        for t in range(-lookback, lookforward + 1):
            if zscores.get(t, 0) > 1.0:
                # Check if next bar also elevated
                if zscores.get(t + 1, 0) > 0.5:
                    first_spike[feat] = t
                    break

    # Summary: which features spike first?
    print(f"\n  First sustained spike (z > 1.0 for 2+ bars):")
    print(f"  {'Feature':30s} {'First bar':>10s} {'Time':>12s} {'Type':>6s}")
    print(f"  {'-'*60}")

    oi_leads = []
    ohlcv_leads = []
    for feat in sorted(first_spike.keys(), key=lambda f: first_spike[f]):
        bar = first_spike[feat]
        minutes = bar * 5
        ftype = "OI" if feat in oi_features else "OHLCV"
        flag = "⚡" if bar < 0 else "  "
        print(f"  {flag} {feat:28s} {bar:>+10d} {minutes:>+10d} min  {ftype:>6s}")
        if feat in oi_features:
            oi_leads.append(bar)
        else:
            ohlcv_leads.append(bar)

    if oi_leads and ohlcv_leads:
        oi_avg = np.mean(oi_leads)
        ohlcv_avg = np.mean(ohlcv_leads)
        lead = ohlcv_avg - oi_avg
        print(f"\n  Average first-spike bar:")
        print(f"    OI features:    bar {oi_avg:+.1f} ({oi_avg*5:+.0f} min)")
        print(f"    OHLCV features: bar {ohlcv_avg:+.1f} ({ohlcv_avg*5:+.0f} min)")
        print(f"    OI leads OHLCV by: {lead:.1f} bars ({lead*5:.0f} min)")

    return profiles, first_spike


def analyze_oi_as_early_warning(ohlcv_df, oi_df, transitions):
    """
    Build a simple OI-based early warning detector and compare lag to GMM/HMM.
    """
    print(f"\n{'='*70}")
    print(f"  OI-BASED EARLY WARNING DETECTOR")
    print(f"  Can OI velocity detect regime changes faster than OHLCV-based methods?")
    print(f"{'='*70}")

    merged = pd.merge_asof(
        ohlcv_df.sort_values("timestamp_us"),
        oi_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    # Build OI-based regime detector: z-score of OI velocity
    for feat in ["oi_vel_mean", "oi_spike_count"]:
        if feat not in merged.columns:
            continue
        rolling_mean = merged[feat].rolling(288, min_periods=48).mean()
        rolling_std = merged[feat].rolling(288, min_periods=48).std()
        merged[f"{feat}_zscore"] = (merged[feat] - rolling_mean) / rolling_std.replace(0, np.nan)

    # For each transition, measure detection lag for different methods
    print(f"\n  Detection lag comparison across {len(transitions)} transitions:")
    print(f"  (How many bars BEFORE the GMM transition does each method fire?)")

    methods = {
        "GMM p_volatile > 0.5": lambda df, i: df.iloc[i]["p_volatile"] > 0.5 if not np.isnan(df.iloc[i].get("p_volatile", np.nan)) else False,
        "GMM p_volatile > 0.7": lambda df, i: df.iloc[i]["p_volatile"] > 0.7 if not np.isnan(df.iloc[i].get("p_volatile", np.nan)) else False,
        "OI vel z > 1.0": lambda df, i: df.iloc[i].get("oi_vel_mean_zscore", 0) > 1.0 if not np.isnan(df.iloc[i].get("oi_vel_mean_zscore", np.nan)) else False,
        "OI vel z > 1.5": lambda df, i: df.iloc[i].get("oi_vel_mean_zscore", 0) > 1.5 if not np.isnan(df.iloc[i].get("oi_vel_mean_zscore", np.nan)) else False,
        "OI vel z > 2.0": lambda df, i: df.iloc[i].get("oi_vel_mean_zscore", 0) > 2.0 if not np.isnan(df.iloc[i].get("oi_vel_mean_zscore", np.nan)) else False,
        "OI spikes z > 1.0": lambda df, i: df.iloc[i].get("oi_spike_count_zscore", 0) > 1.0 if not np.isnan(df.iloc[i].get("oi_spike_count_zscore", np.nan)) else False,
        "OI spikes z > 1.5": lambda df, i: df.iloc[i].get("oi_spike_count_zscore", 0) > 1.5 if not np.isnan(df.iloc[i].get("oi_spike_count_zscore", np.nan)) else False,
        "parkvol_1h z > 1.0": lambda df, i: _zscore_check(df, i, "parkvol_1h", 1.0),
        "rvol_1h z > 1.0": lambda df, i: _zscore_check(df, i, "rvol_1h", 1.0),
    }

    # Pre-compute OHLCV z-scores
    for feat in ["parkvol_1h", "rvol_1h"]:
        if feat in merged.columns:
            rm = merged[feat].rolling(288, min_periods=48).mean()
            rs = merged[feat].rolling(288, min_periods=48).std()
            merged[f"{feat}_zscore"] = (merged[feat] - rm) / rs.replace(0, np.nan)

    lookback = 24  # check up to 2 hours before

    results = {m: {"leads": [], "detected": 0, "false_per_day": 0} for m in methods}

    for trans_idx in transitions:
        if trans_idx < lookback + 288:  # need warmup for z-scores
            continue

        for method_name, check_fn in methods.items():
            # Look backwards from transition to find first detection
            lead_bars = None
            for offset in range(0, lookback + 1):
                idx = trans_idx - offset
                try:
                    if check_fn(merged, idx):
                        lead_bars = offset
                    else:
                        if lead_bars is not None:
                            break  # found the start of the detection
                except:
                    continue

            if lead_bars is not None:
                results[method_name]["leads"].append(lead_bars)
                results[method_name]["detected"] += 1

    # Count false positives: how often does each method fire when NOT near a transition?
    transition_set = set(transitions)
    total_bars = len(merged)
    near_transition = set()
    for t in transitions:
        for offset in range(-6, 7):
            near_transition.add(t + offset)

    for method_name, check_fn in methods.items():
        false_fires = 0
        checked = 0
        for i in range(288, total_bars):
            if i in near_transition:
                continue
            checked += 1
            try:
                if check_fn(merged, i):
                    false_fires += 1
            except:
                continue
        days = total_bars / 288
        results[method_name]["false_per_day"] = false_fires / max(days, 1)
        results[method_name]["false_rate"] = false_fires / max(checked, 1) * 100

    # Print results
    n_transitions = sum(1 for t in transitions if t >= lookback + 288)
    print(f"\n  {'Method':25s} {'Detected':>10s} {'Med Lead':>10s} {'Mean Lead':>10s} {'P90 Lead':>10s} {'False/Day':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for method_name in methods:
        r = results[method_name]
        leads = r["leads"]
        detected = r["detected"]
        det_pct = detected / max(n_transitions, 1) * 100

        if leads:
            med = np.median(leads)
            mean = np.mean(leads)
            p90 = np.percentile(leads, 90)
            print(f"  {method_name:25s} {det_pct:>9.0f}% {med:>9.0f}bar {mean:>9.1f}bar {p90:>9.0f}bar {r['false_per_day']:>9.1f}")
        else:
            print(f"  {method_name:25s} {det_pct:>9.0f}% {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {r['false_per_day']:>9.1f}")

    print(f"\n  (Lead = how many bars BEFORE the GMM transition the method first fires)")
    print(f"  (Higher lead = earlier detection = better)")
    print(f"  (False/Day = false alarms when no transition is happening)")

    # Direct comparison: for each transition, does OI fire before OHLCV?
    print(f"\n  Head-to-head: OI vel z>1.0 vs parkvol_1h z>1.0")
    oi_leads_list = results["OI vel z > 1.0"]["leads"]
    pv_leads_list = results["parkvol_1h z > 1.0"]["leads"]

    if oi_leads_list and pv_leads_list:
        # Pair up transitions where both detected
        min_len = min(len(oi_leads_list), len(pv_leads_list))
        oi_arr = np.array(oi_leads_list[:min_len])
        pv_arr = np.array(pv_leads_list[:min_len])
        oi_wins = (oi_arr > pv_arr).sum()
        pv_wins = (pv_arr > oi_arr).sum()
        ties = (oi_arr == pv_arr).sum()
        oi_advantage = np.mean(oi_arr - pv_arr)

        print(f"    OI fires first: {oi_wins} times ({oi_wins/min_len*100:.0f}%)")
        print(f"    OHLCV fires first: {pv_wins} times ({pv_wins/min_len*100:.0f}%)")
        print(f"    Tie: {ties} times ({ties/min_len*100:.0f}%)")
        print(f"    Average OI advantage: {oi_advantage:+.1f} bars ({oi_advantage*5:+.0f} min)")

    return results


def _zscore_check(df, i, feat, threshold):
    col = f"{feat}_zscore"
    val = df.iloc[i].get(col, np.nan)
    if pd.isna(val):
        return False
    return val > threshold


def analyze_combined_detector(ohlcv_df, oi_df, transitions):
    """
    Test: OI velocity as pre-filter + GMM confirmation = faster + fewer false positives?
    """
    print(f"\n{'='*70}")
    print(f"  COMBINED DETECTOR: OI early warning + GMM confirmation")
    print(f"{'='*70}")

    merged = pd.merge_asof(
        ohlcv_df.sort_values("timestamp_us"),
        oi_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    # Pre-compute z-scores
    for feat in ["oi_vel_mean", "oi_spike_count"]:
        if feat in merged.columns:
            rm = merged[feat].rolling(288, min_periods=48).mean()
            rs = merged[feat].rolling(288, min_periods=48).std()
            merged[f"{feat}_zscore"] = (merged[feat] - rm) / rs.replace(0, np.nan)

    # Detectors to compare
    # 1. GMM alone (baseline): p_volatile > 0.5 for 3 consecutive bars
    # 2. OI pre-alert + GMM confirm: OI z > 1.0 → then wait for GMM p > 0.5 (1 bar confirm)
    # 3. OI + GMM combined: OI z > 1.0 AND GMM p > 0.3 → switch (no confirmation needed)

    n = len(merged)
    regime_gmm_3bar = np.zeros(n, dtype=int)
    regime_oi_gmm = np.zeros(n, dtype=int)
    regime_oi_fast = np.zeros(n, dtype=int)

    # Method 1: GMM + 3-bar confirmation
    confirm_count = 0
    current_regime = 0
    for i in range(n):
        pv = merged.iloc[i].get("p_volatile", 0)
        if pd.isna(pv):
            pv = 0
        new_regime = 1 if pv > 0.5 else 0
        if new_regime != current_regime:
            confirm_count += 1
            if confirm_count >= 3:
                current_regime = new_regime
                confirm_count = 0
        else:
            confirm_count = 0
        regime_gmm_3bar[i] = current_regime

    # Method 2: OI pre-alert + GMM 1-bar confirm
    oi_alert = False
    current_regime = 0
    for i in range(n):
        oi_z = merged.iloc[i].get("oi_vel_mean_zscore", 0)
        if pd.isna(oi_z):
            oi_z = 0
        pv = merged.iloc[i].get("p_volatile", 0)
        if pd.isna(pv):
            pv = 0

        if current_regime == 0:
            if oi_z > 1.0:
                oi_alert = True
            if oi_alert and pv > 0.5:
                current_regime = 1
                oi_alert = False
        else:
            if pv < 0.3:
                current_regime = 0
                oi_alert = False
        regime_oi_gmm[i] = current_regime

    # Method 3: OI z > 1.0 AND GMM p > 0.3 (fast combined)
    current_regime = 0
    for i in range(n):
        oi_z = merged.iloc[i].get("oi_vel_mean_zscore", 0)
        if pd.isna(oi_z):
            oi_z = 0
        pv = merged.iloc[i].get("p_volatile", 0)
        if pd.isna(pv):
            pv = 0

        if current_regime == 0:
            if oi_z > 1.0 and pv > 0.3:
                current_regime = 1
        else:
            if pv < 0.2 and oi_z < 0.5:
                current_regime = 0
        regime_oi_fast[i] = current_regime

    # Evaluate each detector against GMM ground truth
    ground_truth = merged["regime"].values

    detectors = {
        "GMM + 3-bar confirm": regime_gmm_3bar,
        "OI alert + GMM 1-bar": regime_oi_gmm,
        "OI z>1 + GMM p>0.3": regime_oi_fast,
    }

    print(f"\n  {'Method':25s} {'Accuracy':>10s} {'Med Lag':>10s} {'Mean Lag':>10s} {'P90 Lag':>10s} {'FalseSw/d':>10s} {'Trans/d':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, pred in detectors.items():
        # Accuracy
        valid = (ground_truth >= 0)
        acc = (pred[valid] == ground_truth[valid]).mean()

        # Detection lag for quiet→volatile transitions
        lags = []
        for trans_idx in transitions:
            if trans_idx < 288:
                continue
            # Find when detector first switches to volatile after this transition
            for offset in range(0, 50):
                if trans_idx + offset >= n:
                    break
                if pred[trans_idx + offset] == 1:
                    lags.append(offset)
                    break

        # False switches
        pred_changes = np.sum(np.diff(pred) != 0)
        gt_changes = len(transitions)
        days = n / 288
        trans_per_day = pred_changes / max(days, 1)

        # False switches: detector changes not near a real transition
        transition_set = set(transitions)
        false_switches = 0
        for i in range(1, n):
            if pred[i] != pred[i-1]:
                near = any(abs(i - t) <= 6 for t in transitions)
                if not near:
                    false_switches += 1
        false_sw_day = false_switches / max(days, 1)

        if lags:
            med_lag = np.median(lags)
            mean_lag = np.mean(lags)
            p90_lag = np.percentile(lags, 90)
            detected_pct = len(lags) / max(len([t for t in transitions if t >= 288]), 1) * 100
            print(f"  {name:25s} {acc:>9.1%} {med_lag:>9.0f}bar {mean_lag:>9.1f}bar {p90_lag:>9.0f}bar {false_sw_day:>9.1f} {trans_per_day:>9.1f}")
        else:
            print(f"  {name:25s} {acc:>9.1%} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {false_sw_day:>9.1f} {trans_per_day:>9.1f}")

    print(f"\n  (Lag = bars AFTER the GMM ground-truth transition until detector fires)")
    print(f"  (Lower lag = faster detection = better)")
    print(f"  (1 bar = 5 minutes)")


def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  OI REGIME LEAD/LAG ANALYSIS — {SYMBOL}")
    print(f"  Does OI velocity spike BEFORE OHLCV volatility?")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"{'='*70}")

    ohlcv_df = load_ohlcv_bars()
    if ohlcv_df.empty:
        print("ERROR: No OHLCV data")
        return

    ticker_df = load_ticker_5m()
    if ticker_df.empty:
        print("ERROR: No ticker data")
        return

    oi_df = build_oi_velocity_5m(ticker_df)
    del ticker_df

    transitions = identify_regime_transitions(ohlcv_df)

    profiles, first_spike = analyze_lead_lag(ohlcv_df, oi_df, transitions)
    early_warning = analyze_oi_as_early_warning(ohlcv_df, oi_df, transitions)
    analyze_combined_detector(ohlcv_df, oi_df, transitions)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()
    SYMBOL = args.symbol
    main()
