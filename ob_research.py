#!/usr/bin/env python3
"""
Orderbook Research — v23

Experiments:
  1. OB feature profiling: what do OB features look like in quiet vs volatile regimes?
  2. Regime detection improvement: do OB features improve GMM/HMM regime classification?
  3. Regime switch prediction: do OB features improve prediction of regime transitions?
  4. Directional signal: does depth imbalance predict short-term returns?
  5. Volatility prediction: do OB features improve vol forecasting?
  6. Futures-spot basis: does the basis between futures and spot OB predict anything?

Data: BTC Dec 2025 (31 days), 5-min bars with OB features merged.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from scipy import stats as scipy_stats

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
SYMBOL = "BTCUSDT"
SOURCE = "bybit_futures"
SPOT_SOURCE = "bybit_spot"
START_DATE = "2025-12-01"
END_DATE = "2025-12-31"

# Features from regime_detection.py that we already know work
OHLCV_FEATURES = [
    "rvol_1h", "rvol_4h", "rvol_24h",
    "parkvol_1h",
    "vol_ratio_1h_24h",
    "efficiency_1h", "efficiency_4h",
    "adx_4h",
    "bar_eff_4h",
    "trade_intensity_ratio",
    "momentum_4h",
    "price_vs_sma_24h",
    "ret_autocorr_1h",
    "sign_persist_1h",
]

# OB features to use (selected from the 62 available)
OB_FEATURES_CORE = [
    "ob_spread_mean", "ob_spread_std", "ob_spread_max",
    "ob_imb_0.5bps_mean", "ob_imb_1bps_mean", "ob_imb_2bps_mean",
    "ob_imb_5bps_mean",
    "ob_imb_1bps_std", "ob_imb_2bps_std",
    "ob_imb_1bps_trend", "ob_imb_2bps_trend",
    "ob_bid_depth_1bps_mean", "ob_ask_depth_1bps_mean",
    "ob_bid_depth_2bps_mean", "ob_ask_depth_2bps_mean",
    "ob_bid_depth_5bps_mean", "ob_ask_depth_5bps_mean",
    "ob_total_depth_mean",
    "ob_bid_depth_change", "ob_ask_depth_change",
    "ob_bid_depth_cv", "ob_ask_depth_cv",
    "ob_bid_wall_ratio", "ob_ask_wall_ratio",
    "ob_bid_wall_frac", "ob_ask_wall_frac",
    "ob_bid_slope", "ob_ask_slope",
    "ob_microprice_dev_bps",
    "ob_mid_return_bps", "ob_mid_volatility",
]

FEE_BPS = 7  # round-trip fee


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv_bars():
    """Load 5-min OHLCV bars with microstructure features for Dec 2025."""
    from regime_detection import load_bars, compute_regime_features
    print("Loading OHLCV bars...")
    df = load_bars(SYMBOL, START_DATE, END_DATE)
    df = compute_regime_features(df)
    print(f"  OHLCV: {len(df)} bars, {df.columns.size} cols")
    return df


def load_ob_features(market="futures"):
    """Load 5-min OB features for Dec 2025."""
    feat_dir = PARQUET_DIR / SYMBOL / "ob_features_5m" / f"bybit_{market}"
    dates = pd.date_range(START_DATE, END_DATE)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = feat_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        print(f"  No OB features found for {market}")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    print(f"  OB features ({market}): {len(df)} bars, {df.columns.size} cols")
    return df


def merge_ohlcv_ob(ohlcv_df, ob_df):
    """Merge OHLCV bars with OB features on 5-min timestamp."""
    # OHLCV bars have a datetime index; OB features have timestamp_us
    # Need to align them

    # Convert OHLCV index to timestamp_us
    if "timestamp_us" not in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df["timestamp_us"] = ohlcv_df.index.astype(np.int64) // 1000

    # Merge on timestamp_us (5-min aligned)
    # OB timestamps are bar-start aligned from build_ob_features
    merged = pd.merge_asof(
        ohlcv_df.sort_values("timestamp_us"),
        ob_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,  # 5 min tolerance
        direction="nearest",
    )
    n_matched = merged[OB_FEATURES_CORE[0]].notna().sum()
    print(f"  Merged: {len(merged)} bars, {n_matched} with OB data ({n_matched/len(merged)*100:.1f}%)")
    return merged


def load_spot_ob_features():
    """Load spot OB features and compute futures-spot basis features."""
    return load_ob_features("spot")


# ---------------------------------------------------------------------------
# Experiment 1: OB Feature Profiling by Regime
# ---------------------------------------------------------------------------

def exp1_ob_regime_profiles(df, labels, regime_names):
    """Profile OB features in quiet vs volatile regimes."""
    print(f"\n{'='*70}")
    print(f"  EXP 1: OB FEATURE PROFILES BY REGIME")
    print(f"{'='*70}")

    ob_cols = [c for c in OB_FEATURES_CORE if c in df.columns]
    if not ob_cols:
        print("  No OB features found in merged data!")
        return

    print(f"\n  {'Feature':35s} {'Quiet':>12s} {'Volatile':>12s} {'Ratio':>8s} {'T-stat':>8s} {'P-val':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")

    significant = []
    for col in ob_cols:
        vals = df[col].values
        valid = ~np.isnan(vals) & (labels >= 0)
        if valid.sum() < 100:
            continue

        quiet_vals = vals[valid & (labels == 0)]
        vol_vals = vals[valid & (labels == 1)]

        if len(quiet_vals) < 30 or len(vol_vals) < 30:
            continue

        q_mean = np.mean(quiet_vals)
        v_mean = np.mean(vol_vals)
        ratio = v_mean / q_mean if abs(q_mean) > 1e-10 else float('inf')

        t_stat, p_val = scipy_stats.ttest_ind(quiet_vals, vol_vals, equal_var=False)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {col:35s} {q_mean:>12.4f} {v_mean:>12.4f} {ratio:>8.2f} {t_stat:>8.2f} {p_val:>10.2e} {sig}")

        if p_val < 0.05:
            significant.append((col, abs(t_stat), ratio))

    significant.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top discriminating OB features (p<0.05): {len(significant)}/{len(ob_cols)}")
    for col, t, r in significant[:10]:
        print(f"    {col:35s} |t|={t:.2f}  ratio={r:.2f}")


# ---------------------------------------------------------------------------
# Experiment 2: Regime Detection Improvement
# ---------------------------------------------------------------------------

def exp2_regime_detection(df, labels):
    """Compare regime detection accuracy with and without OB features."""
    print(f"\n{'='*70}")
    print(f"  EXP 2: REGIME DETECTION — OHLCV vs OHLCV+OB")
    print(f"{'='*70}")

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    ob_cols = [c for c in OB_FEATURES_CORE if c in df.columns]

    # Valid rows: have both OHLCV and OB features and labels
    valid = df[ohlcv_cols + ob_cols].notna().all(axis=1).values & (labels >= 0)
    df_valid = df[valid].copy()
    y = labels[valid]
    n = len(df_valid)

    if n < 200:
        print(f"  Not enough valid data: {n} rows")
        return

    # Train/test split: first 70% train, last 30% test
    split = int(n * 0.7)
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)

    results = {}

    for name, feature_cols in [
        ("OHLCV only", ohlcv_cols),
        ("OHLCV + OB", ohlcv_cols + ob_cols),
        ("OB only", ob_cols),
    ]:
        X = df_valid[feature_cols].values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        # GMM (unsupervised)
        gmm = GaussianMixture(n_components=2, covariance_type="diag",
                               n_init=5, random_state=42, max_iter=300)
        gmm.fit(X_train)
        gmm_labels = gmm.predict(X_test)
        # Align labels
        if np.mean(X_test[gmm_labels == 0, 0]) > np.mean(X_test[gmm_labels == 1, 0]):
            gmm_labels = 1 - gmm_labels
        gmm_acc = max(accuracy_score(y_test, gmm_labels),
                      accuracy_score(y_test, 1 - gmm_labels))

        # Logistic Regression (supervised)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_proba = lr.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_proba)
        lr_acc = accuracy_score(y_test, lr.predict(X_test))

        # Gradient Boosting (supervised)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                         random_state=42, subsample=0.8)
        gb.fit(X_train, y_train)
        gb_proba = gb.predict_proba(X_test)[:, 1]
        gb_auc = roc_auc_score(y_test, gb_proba)
        gb_acc = accuracy_score(y_test, gb.predict(X_test))

        results[name] = {
            "gmm_acc": gmm_acc,
            "lr_acc": lr_acc, "lr_auc": lr_auc,
            "gb_acc": gb_acc, "gb_auc": gb_auc,
            "n_features": len(feature_cols),
        }

        print(f"\n  {name} ({len(feature_cols)} features):")
        print(f"    GMM accuracy:  {gmm_acc:.4f}")
        print(f"    LR accuracy:   {lr_acc:.4f}  AUC: {lr_auc:.4f}")
        print(f"    GB accuracy:   {gb_acc:.4f}  AUC: {gb_auc:.4f}")

    # Feature importance from GB with all features
    print(f"\n  Top 15 features (GB, OHLCV+OB):")
    all_cols = ohlcv_cols + ob_cols
    X_all = df_valid[all_cols].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx])
    y_train = y[train_idx]
    gb_all = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                         random_state=42, subsample=0.8)
    gb_all.fit(X_train, y_train)
    importances = gb_all.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    for rank, idx in enumerate(top_idx, 1):
        is_ob = "OB" if all_cols[idx].startswith("ob_") else "  "
        print(f"    {rank:2d}. [{is_ob}] {all_cols[idx]:35s} importance={importances[idx]:.4f}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Regime Switch Prediction
# ---------------------------------------------------------------------------

def exp3_regime_switch_prediction(df, labels):
    """Does OB data improve prediction of regime switches?"""
    print(f"\n{'='*70}")
    print(f"  EXP 3: REGIME SWITCH PREDICTION — OHLCV vs OHLCV+OB")
    print(f"{'='*70}")

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    ob_cols = [c for c in OB_FEATURES_CORE if c in df.columns]

    valid = df[ohlcv_cols + ob_cols].notna().all(axis=1).values & (labels >= 0)
    df_valid = df[valid].copy()
    y_regime = labels[valid]
    n = len(df_valid)

    if n < 200:
        print(f"  Not enough data: {n}")
        return

    # Target: will regime switch in next N bars?
    horizons = {"30min": 6, "1h": 12, "2h": 24}

    for hz_name, hz_bars in horizons.items():
        # Create target
        y_switch = np.zeros(n, dtype=np.int8)
        for i in range(n - hz_bars):
            if np.any(y_regime[i+1:i+1+hz_bars] != y_regime[i]):
                y_switch[i] = 1

        # Only use rows where we have a valid target
        valid_target = np.arange(n - hz_bars)
        split = int(len(valid_target) * 0.7)
        train_idx = valid_target[:split]
        test_idx = valid_target[split:]

        print(f"\n  Horizon: {hz_name} ({hz_bars} bars)")
        print(f"    Switch rate: {y_switch[valid_target].mean():.3f}")

        for name, feature_cols in [
            ("OHLCV only", ohlcv_cols),
            ("OHLCV + OB", ohlcv_cols + ob_cols),
        ]:
            X = df_valid[feature_cols].values
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train = y_switch[train_idx]
            y_test = y_switch[test_idx]

            gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                             random_state=42, subsample=0.8)
            gb.fit(X_train, y_train)
            proba = gb.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            acc = accuracy_score(y_test, gb.predict(X_test))

            print(f"    {name:20s}: AUC={auc:.4f}  Acc={acc:.4f}")


# ---------------------------------------------------------------------------
# Experiment 4: Directional Signal from Depth Imbalance
# ---------------------------------------------------------------------------

def exp4_imbalance_signal(df):
    """Does depth imbalance predict short-term returns?"""
    print(f"\n{'='*70}")
    print(f"  EXP 4: DEPTH IMBALANCE → DIRECTIONAL SIGNAL")
    print(f"{'='*70}")

    if "returns" not in df.columns:
        print("  No returns column!")
        return

    ret = df["returns"].values
    n = len(ret)

    # Forward returns at various horizons
    horizons = {"5min": 1, "15min": 3, "1h": 12, "4h": 48}

    imb_cols = [c for c in df.columns if c.startswith("ob_imb_") and c.endswith("_mean")]

    print(f"\n  Information Coefficient (IC) — imbalance vs forward returns:")
    print(f"  {'Feature':35s}" + "".join(f" {h:>8s}" for h in horizons.keys()))
    print(f"  {'-'*35}" + "-" * 10 * len(horizons))

    best_ic = {}
    for col in imb_cols:
        vals = df[col].values
        ics = []
        for hz_name, hz_bars in horizons.items():
            fwd = np.full(n, np.nan)
            for i in range(n - hz_bars):
                fwd[i] = np.sum(ret[i+1:i+1+hz_bars])
            valid = ~np.isnan(vals) & ~np.isnan(fwd)
            if valid.sum() > 100:
                ic = np.corrcoef(vals[valid], fwd[valid])[0, 1]
                ics.append(ic)
                if hz_name not in best_ic or abs(ic) > abs(best_ic[hz_name][1]):
                    best_ic[hz_name] = (col, ic)
            else:
                ics.append(np.nan)
        ic_str = "".join(f" {ic:>+8.4f}" if not np.isnan(ic) else "      N/A" for ic in ics)
        print(f"  {col:35s}{ic_str}")

    print(f"\n  Best IC per horizon:")
    for hz, (col, ic) in best_ic.items():
        print(f"    {hz:>6s}: {col:35s} IC={ic:+.4f}")

    # Simple backtest: trade on imbalance signal
    print(f"\n  Simple backtest — trade on imbalance (4h hold, 7bps fee):")
    for col in imb_cols:
        vals = df[col].values
        valid = ~np.isnan(vals)
        if valid.sum() < 100:
            continue

        # Z-score the signal
        v = vals[valid]
        z = (v - np.mean(v)) / max(np.std(v), 1e-10)

        # Forward 4h returns
        fwd_4h = np.full(n, np.nan)
        for i in range(n - 48):
            fwd_4h[i] = np.sum(ret[i+1:i+49])

        fwd_valid = fwd_4h[valid]
        z_valid = z

        # Long when imbalance > 1 std (more bids), short when < -1 std
        long_mask = z_valid > 1.0
        short_mask = z_valid < -1.0

        if long_mask.sum() < 10 or short_mask.sum() < 10:
            continue

        long_pnl = fwd_valid[long_mask & ~np.isnan(fwd_valid)] * 10000 - FEE_BPS
        short_pnl = -fwd_valid[short_mask & ~np.isnan(fwd_valid)] * 10000 - FEE_BPS

        n_trades = len(long_pnl) + len(short_pnl)
        if n_trades < 20:
            continue

        all_pnl = np.concatenate([long_pnl, short_pnl])
        avg = np.mean(all_pnl)
        wr = np.mean(all_pnl > 0)

        print(f"    {col:35s}: trades={n_trades:4d}, avg={avg:+.1f}bps, wr={wr:.1%}, "
              f"long={np.mean(long_pnl):+.1f}bps({len(long_pnl)}), "
              f"short={np.mean(short_pnl):+.1f}bps({len(short_pnl)})")


# ---------------------------------------------------------------------------
# Experiment 5: Volatility Prediction
# ---------------------------------------------------------------------------

def exp5_vol_prediction(df):
    """Do OB features improve volatility prediction?"""
    print(f"\n{'='*70}")
    print(f"  EXP 5: VOLATILITY PREDICTION — OHLCV vs OHLCV+OB")
    print(f"{'='*70}")

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    ob_cols = [c for c in OB_FEATURES_CORE if c in df.columns]

    if "returns" not in df.columns:
        print("  No returns column!")
        return

    ret = df["returns"].values
    n = len(ret)

    # Target: realized vol over next 1h (12 bars)
    fwd_vol = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_vol[i] = np.std(ret[i+1:i+13])

    valid = df[ohlcv_cols + ob_cols].notna().all(axis=1).values & ~np.isnan(fwd_vol)
    idx = np.where(valid)[0]

    if len(idx) < 200:
        print(f"  Not enough data: {len(idx)}")
        return

    split = int(len(idx) * 0.7)
    train_idx = idx[:split]
    test_idx = idx[split:]

    y_train = fwd_vol[train_idx]
    y_test = fwd_vol[test_idx]

    for name, feature_cols in [
        ("OHLCV only", ohlcv_cols),
        ("OHLCV + OB", ohlcv_cols + ob_cols),
        ("OB only", ob_cols),
    ]:
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        # Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        pred = ridge.predict(X_test)
        r2 = r2_score(y_test, pred)
        corr = np.corrcoef(y_test, pred)[0, 1]

        # GB regression
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                        random_state=42, subsample=0.8)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_corr = np.corrcoef(y_test, gb_pred)[0, 1]

        print(f"\n  {name} ({len(feature_cols)} features):")
        print(f"    Ridge:  R²={r2:.4f}  corr={corr:.4f}")
        print(f"    GB:     R²={gb_r2:.4f}  corr={gb_corr:.4f}")

    # Feature importance
    all_cols = ohlcv_cols + ob_cols
    X_all = df[all_cols].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx])
    gb_all = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                        random_state=42, subsample=0.8)
    gb_all.fit(X_train, fwd_vol[train_idx])
    importances = gb_all.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"\n  Top 15 features for vol prediction (GB, OHLCV+OB):")
    for rank, idx_val in enumerate(top_idx, 1):
        is_ob = "OB" if all_cols[idx_val].startswith("ob_") else "  "
        print(f"    {rank:2d}. [{is_ob}] {all_cols[idx_val]:35s} importance={importances[idx_val]:.4f}")


# ---------------------------------------------------------------------------
# Experiment 6: Futures-Spot Basis
# ---------------------------------------------------------------------------

def exp6_basis_signal(df_futures_ob, df_spot_ob, ohlcv_df):
    """Does the futures-spot basis predict returns?"""
    print(f"\n{'='*70}")
    print(f"  EXP 6: FUTURES-SPOT BASIS SIGNAL")
    print(f"{'='*70}")

    if df_spot_ob.empty:
        print("  No spot OB data available — skipping")
        return

    # Merge futures and spot OB on timestamp
    fut = df_futures_ob[["timestamp_us", "ob_spread_mean", "ob_imb_1bps_mean",
                          "ob_imb_2bps_mean", "ob_total_depth_mean"]].copy()
    fut.columns = ["timestamp_us"] + [f"fut_{c}" for c in fut.columns[1:]]

    spot_cols = ["timestamp_us", "ob_spread_mean", "ob_imb_1bps_mean",
                 "ob_imb_2bps_mean", "ob_total_depth_mean"]
    spot_cols = [c for c in spot_cols if c in df_spot_ob.columns]
    spot = df_spot_ob[spot_cols].copy()
    spot.columns = ["timestamp_us"] + [f"spot_{c}" for c in spot.columns[1:]]

    merged = pd.merge_asof(
        fut.sort_values("timestamp_us"),
        spot.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    n_matched = merged["spot_ob_spread_mean"].notna().sum() if "spot_ob_spread_mean" in merged.columns else 0
    print(f"  Merged futures+spot: {len(merged)} bars, {n_matched} matched")

    if n_matched < 100:
        print("  Not enough matched data")
        return

    # Compute basis features
    if "fut_ob_imb_1bps_mean" in merged.columns and "spot_ob_imb_1bps_mean" in merged.columns:
        merged["basis_imb_1bps"] = merged["fut_ob_imb_1bps_mean"] - merged["spot_ob_imb_1bps_mean"]
        merged["basis_imb_2bps"] = merged["fut_ob_imb_2bps_mean"] - merged["spot_ob_imb_2bps_mean"]
        merged["basis_spread"] = merged["fut_ob_spread_mean"] - merged["spot_ob_spread_mean"]
        merged["basis_depth_ratio"] = merged["fut_ob_total_depth_mean"] / merged["spot_ob_total_depth_mean"].clip(lower=0.01)

        print(f"\n  Basis feature stats:")
        for col in ["basis_imb_1bps", "basis_imb_2bps", "basis_spread", "basis_depth_ratio"]:
            if col in merged.columns:
                v = merged[col].dropna()
                print(f"    {col:25s}: mean={v.mean():.4f}, std={v.std():.4f}")

    # Merge with OHLCV to get returns
    if "returns" in ohlcv_df.columns:
        if "timestamp_us" not in ohlcv_df.columns:
            ohlcv_df = ohlcv_df.copy()
            ohlcv_df["timestamp_us"] = ohlcv_df.index.astype(np.int64) // 1000

        basis_merged = pd.merge_asof(
            merged.sort_values("timestamp_us"),
            ohlcv_df[["timestamp_us", "returns"]].sort_values("timestamp_us"),
            on="timestamp_us",
            tolerance=300_000_000,
            direction="nearest",
        )

        ret = basis_merged["returns"].values
        n = len(ret)

        # IC of basis features vs forward returns
        print(f"\n  IC of basis features vs forward returns:")
        basis_cols = [c for c in ["basis_imb_1bps", "basis_imb_2bps", "basis_spread", "basis_depth_ratio"]
                      if c in basis_merged.columns]

        for hz_name, hz_bars in [("5min", 1), ("1h", 12), ("4h", 48)]:
            fwd = np.full(n, np.nan)
            for i in range(n - hz_bars):
                fwd[i] = np.sum(ret[i+1:i+1+hz_bars])

            print(f"    {hz_name}:")
            for col in basis_cols:
                vals = basis_merged[col].values
                valid = ~np.isnan(vals) & ~np.isnan(fwd)
                if valid.sum() > 50:
                    ic = np.corrcoef(vals[valid], fwd[valid])[0, 1]
                    print(f"      {col:25s}: IC={ic:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print(f"  ORDERBOOK RESEARCH — {SYMBOL} Dec 2025")
    print(f"  Data: 5-min bars + OB features (futures + spot)")
    print("=" * 70)

    # Load data
    ohlcv_df = load_ohlcv_bars()
    ob_futures = load_ob_features("futures")
    ob_spot = load_ob_features("spot")

    # Merge OHLCV with futures OB
    df = merge_ohlcv_ob(ohlcv_df, ob_futures)

    # Create regime labels using GMM on OHLCV features (ground truth)
    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    X_regime = df[ohlcv_cols].copy()
    valid_regime = X_regime.notna().all(axis=1)
    X_valid = X_regime[valid_regime]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                           n_init=10, random_state=42, max_iter=300)
    labels_raw = gmm.fit_predict(X_scaled)

    # Align: volatile = higher rvol
    rvol_col = "rvol_1h" if "rvol_1h" in X_valid.columns else ohlcv_cols[0]
    if np.mean(X_valid[rvol_col].values[labels_raw == 0]) > np.mean(X_valid[rvol_col].values[labels_raw == 1]):
        labels_raw = 1 - labels_raw

    # Map back to full df
    labels = np.full(len(df), -1, dtype=np.int8)
    labels[valid_regime.values] = labels_raw

    regime_names = {0: "quiet", 1: "volatile"}
    n_quiet = (labels == 0).sum()
    n_vol = (labels == 1).sum()
    print(f"\n  Regime labels: quiet={n_quiet} ({n_quiet/len(df)*100:.1f}%), "
          f"volatile={n_vol} ({n_vol/len(df)*100:.1f}%)")

    # Run experiments
    exp1_ob_regime_profiles(df, labels, regime_names)
    exp2_results = exp2_regime_detection(df, labels)
    exp3_regime_switch_prediction(df, labels)
    exp4_imbalance_signal(df)
    exp5_vol_prediction(df)

    # Load spot features for basis experiment
    if not ob_spot.empty:
        # Build spot 5-min features if not already done
        spot_feat_dir = PARQUET_DIR / SYMBOL / "ob_features_5m" / "bybit_spot"
        if spot_feat_dir.exists():
            spot_features = load_ob_features("spot")
            # Actually we need the raw spot features, not re-loaded
            # The load_ob_features already loads from ob_features_5m
            exp6_basis_signal(ob_futures, spot_features, ohlcv_df)
        else:
            print("\n  Spot OB features not yet computed — run build_ob_features.py first")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
