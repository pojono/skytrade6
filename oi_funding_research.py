#!/usr/bin/env python3
"""
Open Interest & Funding Rate Research — v24

Experiments:
  1. Feature profiling: OI/funding features in quiet vs volatile regimes
  2. Regime detection: do OI/funding features improve regime classification?
  3. Directional signal: do OI changes, funding extremes predict returns?
  4. Volatility prediction: do OI/funding features improve vol forecasting?
  5. Crowding/extreme detection: do positioning extremes predict reversals?
  6. Combined with OB: does OI/funding + OB outperform either alone?

Data: Binance metrics (OI, LS ratios, taker ratio) + premium index (funding basis)
      at 5-min resolution, Dec 2025.
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
START_DATE = "2025-12-01"
END_DATE = "2025-12-31"

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

# OI/Funding raw columns from Binance metrics
METRICS_RAW = [
    "open_interest", "open_interest_value",
    "top_trader_ls_ratio_accounts", "top_trader_ls_ratio_positions",
    "global_ls_ratio", "taker_buy_sell_ratio",
]

# Engineered OI/Funding features we'll compute
OI_FUNDING_FEATURES = []  # populated in build_oi_funding_features()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv_bars():
    """Load 5-min OHLCV bars with microstructure features."""
    from regime_detection import load_bars, compute_regime_features
    print("Loading OHLCV bars...")
    df = load_bars(SYMBOL, START_DATE, END_DATE)
    df = compute_regime_features(df)
    print(f"  OHLCV: {len(df)} bars, {df.columns.size} cols")
    return df


def load_ob_features(market="futures"):
    """Load 5-min OB features."""
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


def load_metrics():
    """Load Binance metrics (OI, LS ratios, taker ratio) at 5-min."""
    metrics_dir = PARQUET_DIR / SYMBOL / "binance" / "metrics"
    dates = pd.date_range(START_DATE, END_DATE)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = metrics_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        print("  No metrics data found!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Metrics: {len(df)} bars, cols={list(df.columns)}")
    return df


def load_premium_index():
    """Load Binance premium index klines at 5-min (funding basis)."""
    prem_dir = PARQUET_DIR / SYMBOL / "binance" / "premium_index_klines_futures" / "5m"
    dates = pd.date_range(START_DATE, END_DATE)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = prem_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        print("  No premium index data found!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # Rename to avoid collision with OHLCV columns
    df = df.rename(columns={
        "open": "prem_open", "high": "prem_high",
        "low": "prem_low", "close": "prem_close",
    })
    keep = ["timestamp_us", "prem_open", "prem_high", "prem_low", "prem_close"]
    df = df[keep]
    print(f"  Premium index: {len(df)} bars, premium range [{df['prem_close'].min():.6f}, {df['prem_close'].max():.6f}]")
    return df


def build_oi_funding_features(metrics_df, premium_df):
    """
    Engineer features from raw OI/funding data.

    Features:
    - OI change (5m, 1h, 4h, 24h) — rate of new position opening
    - OI acceleration — change in OI change rate
    - OI z-score (rolling) — how unusual is current OI level
    - Funding rate (premium close) — current funding basis
    - Funding rate change — momentum of funding
    - Funding z-score — extreme funding detection
    - LS ratio changes — shift in positioning
    - Taker ratio — aggressive buying vs selling
    - Taker ratio z-score — extreme taker activity
    """
    global OI_FUNDING_FEATURES

    # Merge metrics and premium on timestamp
    df = pd.merge_asof(
        metrics_df.sort_values("timestamp_us"),
        premium_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    # --- OI features ---
    # OI change (pct) over various windows
    for window, name in [(1, "5m"), (12, "1h"), (48, "4h"), (288, "24h")]:
        df[f"oi_change_{name}"] = df["open_interest"].pct_change(window) * 100
    # OI acceleration (change in 1h change)
    df["oi_accel_1h"] = df["oi_change_1h"].diff(12)
    # OI z-score (rolling 24h)
    oi_mean = df["open_interest"].rolling(288, min_periods=48).mean()
    oi_std = df["open_interest"].rolling(288, min_periods=48).std()
    df["oi_zscore_24h"] = (df["open_interest"] - oi_mean) / oi_std.replace(0, np.nan)
    # OI value change (captures both OI and price movement)
    df["oi_value_change_1h"] = df["open_interest_value"].pct_change(12) * 100

    # --- Funding/premium features ---
    df["funding_rate"] = df["prem_close"]  # premium index ≈ continuous funding
    df["funding_abs"] = df["prem_close"].abs()
    # Funding change
    df["funding_change_1h"] = df["prem_close"].diff(12)
    df["funding_change_4h"] = df["prem_close"].diff(48)
    # Funding z-score (rolling 24h)
    fr_mean = df["prem_close"].rolling(288, min_periods=48).mean()
    fr_std = df["prem_close"].rolling(288, min_periods=48).std()
    df["funding_zscore_24h"] = (df["prem_close"] - fr_mean) / fr_std.replace(0, np.nan)
    # Funding cumulative (8h rolling sum ≈ actual funding payment)
    df["funding_cum_8h"] = df["prem_close"].rolling(96, min_periods=12).sum()

    # --- Long/Short ratio features ---
    df["ls_ratio_top"] = df["top_trader_ls_ratio_accounts"]
    df["ls_ratio_global"] = df["global_ls_ratio"]
    # LS ratio changes
    df["ls_top_change_1h"] = df["top_trader_ls_ratio_accounts"].pct_change(12) * 100
    df["ls_global_change_1h"] = df["global_ls_ratio"].pct_change(12) * 100
    # LS ratio z-scores
    ls_mean = df["top_trader_ls_ratio_accounts"].rolling(288, min_periods=48).mean()
    ls_std = df["top_trader_ls_ratio_accounts"].rolling(288, min_periods=48).std()
    df["ls_top_zscore_24h"] = (df["top_trader_ls_ratio_accounts"] - ls_mean) / ls_std.replace(0, np.nan)
    ls_g_mean = df["global_ls_ratio"].rolling(288, min_periods=48).mean()
    ls_g_std = df["global_ls_ratio"].rolling(288, min_periods=48).std()
    df["ls_global_zscore_24h"] = (df["global_ls_ratio"] - ls_g_mean) / ls_g_std.replace(0, np.nan)

    # --- Taker buy/sell ratio features ---
    df["taker_ratio"] = df["taker_buy_sell_ratio"]
    df["taker_ratio_1h"] = df["taker_buy_sell_ratio"].rolling(12, min_periods=3).mean()
    df["taker_ratio_4h"] = df["taker_buy_sell_ratio"].rolling(48, min_periods=12).mean()
    tk_mean = df["taker_buy_sell_ratio"].rolling(288, min_periods=48).mean()
    tk_std = df["taker_buy_sell_ratio"].rolling(288, min_periods=48).std()
    df["taker_zscore_24h"] = (df["taker_buy_sell_ratio"] - tk_mean) / tk_std.replace(0, np.nan)

    # --- Cross features ---
    # OI rising + funding positive = crowded long
    df["oi_x_funding"] = df["oi_change_1h"] * df["funding_rate"] * 1000
    # OI rising + taker buying = aggressive long entry
    df["oi_x_taker"] = df["oi_change_1h"] * (df["taker_buy_sell_ratio"] - 1)

    OI_FUNDING_FEATURES = [
        "oi_change_5m", "oi_change_1h", "oi_change_4h", "oi_change_24h",
        "oi_accel_1h", "oi_zscore_24h", "oi_value_change_1h",
        "funding_rate", "funding_abs", "funding_change_1h", "funding_change_4h",
        "funding_zscore_24h", "funding_cum_8h",
        "ls_ratio_top", "ls_ratio_global",
        "ls_top_change_1h", "ls_global_change_1h",
        "ls_top_zscore_24h", "ls_global_zscore_24h",
        "taker_ratio", "taker_ratio_1h", "taker_ratio_4h", "taker_zscore_24h",
        "oi_x_funding", "oi_x_taker",
    ]

    print(f"  Engineered {len(OI_FUNDING_FEATURES)} OI/funding features")
    return df


def merge_all(ohlcv_df, ob_df, oi_df):
    """Merge OHLCV + OB + OI/Funding on timestamp_us."""
    # Ensure OHLCV has timestamp_us
    if "timestamp_us" not in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df["timestamp_us"] = ohlcv_df.index.astype(np.int64) // 1000

    # Merge OHLCV + OB
    merged = pd.merge_asof(
        ohlcv_df.sort_values("timestamp_us"),
        ob_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    # Merge with OI/Funding
    oi_cols = ["timestamp_us"] + OI_FUNDING_FEATURES
    oi_subset = oi_df[oi_cols].copy()
    merged = pd.merge_asof(
        merged.sort_values("timestamp_us"),
        oi_subset.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    ob_matched = merged[OB_FEATURES_CORE[0]].notna().sum() if OB_FEATURES_CORE[0] in merged.columns else 0
    oi_matched = merged["oi_change_1h"].notna().sum()
    print(f"  Merged: {len(merged)} bars, OB={ob_matched}, OI/F={oi_matched}")

    # Compute forward returns
    merged["fwd_ret_5m"] = merged["close"].pct_change(1).shift(-1) * 10000  # bps
    merged["fwd_ret_15m"] = merged["close"].pct_change(3).shift(-3) * 10000
    merged["fwd_ret_1h"] = merged["close"].pct_change(12).shift(-12) * 10000
    merged["fwd_ret_4h"] = merged["close"].pct_change(48).shift(-48) * 10000
    merged["fwd_rvol_1h"] = merged["close"].pct_change(1).abs().rolling(12).sum().shift(-12) * 10000

    return merged


# ---------------------------------------------------------------------------
# Experiment 1: OI/Funding Feature Profiling by Regime
# ---------------------------------------------------------------------------

def exp1_regime_profiles(df, labels, regime_names):
    """Profile OI/funding features in quiet vs volatile regimes."""
    print(f"\n{'='*70}")
    print(f"  EXP 1: OI/FUNDING FEATURE PROFILES BY REGIME")
    print(f"{'='*70}")

    feat_cols = [c for c in OI_FUNDING_FEATURES if c in df.columns]
    if not feat_cols:
        print("  No OI/funding features found!")
        return

    print(f"\n  {'Feature':35s} {'Quiet':>12s} {'Volatile':>12s} {'Ratio':>8s} {'T-stat':>8s} {'P-val':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")

    significant = []
    for col in feat_cols:
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
    print(f"\n  Top discriminating OI/funding features (p<0.05): {len(significant)}/{len(feat_cols)}")
    for col, t, r in significant[:10]:
        print(f"    {col:35s} |t|={t:.2f}  ratio={r:.2f}")


# ---------------------------------------------------------------------------
# Experiment 2: Regime Detection — Does OI/Funding Improve It?
# ---------------------------------------------------------------------------

def exp2_regime_detection(df, labels):
    """Test if OI/funding features improve regime classification."""
    print(f"\n{'='*70}")
    print(f"  EXP 2: REGIME DETECTION — OHLCV vs OHLCV+OI/FUNDING")
    print(f"{'='*70}")

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    oi_cols = [c for c in OI_FUNDING_FEATURES if c in df.columns]
    ob_cols = [c for c in OB_FEATURES_CORE if c in df.columns]

    valid = df[ohlcv_cols + oi_cols].notna().all(axis=1).values & (labels >= 0)
    df_valid = df[valid].copy()
    y = labels[valid]

    if len(df_valid) < 200:
        print("  Not enough valid data!")
        return

    feature_sets = {
        "OHLCV only": ohlcv_cols,
        "OHLCV + OI/F": ohlcv_cols + oi_cols,
        "OI/F only": oi_cols,
    }

    # Add OB combos if available
    if ob_cols:
        valid_ob = df[ohlcv_cols + oi_cols + ob_cols].notna().all(axis=1).values & (labels >= 0)
        if valid_ob.sum() > 200:
            feature_sets["OHLCV + OB"] = ohlcv_cols + ob_cols
            feature_sets["OHLCV + OB + OI/F"] = ohlcv_cols + ob_cols + oi_cols
            # Use the intersection of valid rows
            valid = valid & valid_ob
            df_valid = df[valid].copy()
            y = labels[valid]

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    for name, cols in feature_sets.items():
        X = df_valid[cols].values
        scaler = StandardScaler()

        # GMM
        X_scaled = scaler.fit_transform(X)
        gmm = GaussianMixture(n_components=2, covariance_type="diag",
                               n_init=10, random_state=42, max_iter=300)
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_acc = max(accuracy_score(y, gmm_labels), accuracy_score(y, 1 - gmm_labels))

        # Supervised (time-series CV)
        lr_aucs, gb_aucs = [], []
        for train_idx, test_idx in tscv.split(X):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train = y[train_idx]
            y_test = y[test_idx]

            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_train, y_train)
            lr_aucs.append(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))

            gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            gb.fit(X_train, y_train)
            gb_aucs.append(roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1]))

        print(f"\n  {name} ({len(cols)} features):")
        print(f"    GMM accuracy:  {gmm_acc:.4f}")
        print(f"    LR AUC:        {np.mean(lr_aucs):.4f}")
        print(f"    GB AUC:        {np.mean(gb_aucs):.4f}")

    # Feature importance from best GB model
    all_cols = list(feature_sets.values())[-1]  # last (most complete) set
    X_all = df_valid[all_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(X_scaled, y)
    importances = sorted(zip(all_cols, gb.feature_importances_), key=lambda x: -x[1])

    print(f"\n  Top 20 features (GB, all combined):")
    for i, (col, imp) in enumerate(importances[:20], 1):
        tag = "[OI]" if col in oi_cols else "[OB]" if col in ob_cols else "[  ]"
        print(f"    {i:>2}. {tag} {col:40s} importance={imp:.4f}")


# ---------------------------------------------------------------------------
# Experiment 3: Directional Signal — OI/Funding → Returns
# ---------------------------------------------------------------------------

def exp3_directional_signal(df):
    """Test if OI changes and funding extremes predict returns."""
    print(f"\n{'='*70}")
    print(f"  EXP 3: OI/FUNDING → DIRECTIONAL SIGNAL")
    print(f"{'='*70}")

    oi_cols = [c for c in OI_FUNDING_FEATURES if c in df.columns]
    horizons = {
        "5min": "fwd_ret_5m",
        "15min": "fwd_ret_15m",
        "1h": "fwd_ret_1h",
        "4h": "fwd_ret_4h",
    }

    # IC analysis
    print(f"\n  Information Coefficient (IC) — OI/funding vs forward returns:")
    print(f"  {'Feature':35s} {'5min':>8s} {'15min':>8s} {'1h':>8s} {'4h':>8s}")
    print(f"  {'-'*67}")

    best_per_horizon = {}
    for col in oi_cols:
        ics = []
        for h_name, h_col in horizons.items():
            valid = df[[col, h_col]].notna().all(axis=1)
            if valid.sum() < 100:
                ics.append(np.nan)
                continue
            ic = df.loc[valid, col].corr(df.loc[valid, h_col])
            ics.append(ic)
            if h_name not in best_per_horizon or abs(ic) > abs(best_per_horizon[h_name][1]):
                best_per_horizon[h_name] = (col, ic)

        print(f"  {col:35s} {ics[0]:>+8.4f} {ics[1]:>+8.4f} {ics[2]:>+8.4f} {ics[3]:>+8.4f}")

    print(f"\n  Best IC per horizon:")
    for h_name, (col, ic) in sorted(best_per_horizon.items(), key=lambda x: list(horizons.keys()).index(x[0])):
        print(f"    {h_name:>6s}: {col:35s} IC={ic:+.4f}")

    # Simple backtest on best signals
    print(f"\n  Simple backtest — trade on OI/funding signal (4h hold, {FEE_BPS}bps fee):")

    signal_cols = [
        "oi_change_1h", "oi_change_4h", "oi_zscore_24h",
        "funding_rate", "funding_zscore_24h", "funding_cum_8h",
        "ls_top_zscore_24h", "ls_global_zscore_24h",
        "taker_zscore_24h", "oi_x_funding", "oi_x_taker",
    ]

    for col in signal_cols:
        if col not in df.columns:
            continue
        valid = df[[col, "fwd_ret_4h"]].notna().all(axis=1)
        sub = df[valid].copy()
        if len(sub) < 100:
            continue

        vals = sub[col].values
        rets = sub["fwd_ret_4h"].values

        # Z-score threshold trading
        z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-10)

        # Long when signal > +1 std, short when < -1 std
        long_mask = z > 1.0
        short_mask = z < -1.0
        trade_mask = long_mask | short_mask

        if trade_mask.sum() < 10:
            continue

        trade_rets = np.where(long_mask[trade_mask], rets[trade_mask], -rets[trade_mask]) - FEE_BPS
        n_trades = trade_mask.sum()
        avg_pnl = np.mean(trade_rets)
        wr = (trade_rets > 0).mean() * 100

        n_long = long_mask.sum()
        n_short = short_mask.sum()
        long_avg = np.mean(rets[long_mask] - FEE_BPS) if n_long > 0 else 0
        short_avg = np.mean(-rets[short_mask] - FEE_BPS) if n_short > 0 else 0

        flag = "✅" if avg_pnl > 0 else "  "
        print(f"  {flag} {col:35s}: trades={n_trades:>4d}, avg={avg_pnl:>+.1f}bps, wr={wr:.1f}%, "
              f"long={long_avg:>+.1f}bps({n_long}), short={short_avg:>+.1f}bps({n_short})")


# ---------------------------------------------------------------------------
# Experiment 4: Volatility Prediction
# ---------------------------------------------------------------------------

def exp4_vol_prediction(df):
    """Test if OI/funding features improve vol forecasting."""
    print(f"\n{'='*70}")
    print(f"  EXP 4: VOLATILITY PREDICTION — OHLCV vs +OI/F vs +OB vs +ALL")
    print(f"{'='*70}")

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    oi_cols = [c for c in OI_FUNDING_FEATURES if c in df.columns]
    ob_cols = [c for c in OB_FEATURES_CORE if c in df.columns]

    target = "fwd_rvol_1h"

    feature_sets = {
        "OHLCV only": ohlcv_cols,
        "OHLCV + OI/F": ohlcv_cols + oi_cols,
        "OI/F only": oi_cols,
    }
    if ob_cols:
        feature_sets["OHLCV + OB"] = ohlcv_cols + ob_cols
        feature_sets["OHLCV + OB + OI/F"] = ohlcv_cols + ob_cols + oi_cols

    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    for name, cols in feature_sets.items():
        valid = df[cols + [target]].notna().all(axis=1)
        X = df.loc[valid, cols].values
        y = df.loc[valid, target].values

        if len(X) < 200:
            continue

        ridge_r2s, gb_r2s, ridge_corrs, gb_corrs = [], [], [], []
        for train_idx, test_idx in tscv.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            ridge = Ridge(alpha=10)
            ridge.fit(X_train, y_train)
            pred_r = ridge.predict(X_test)
            ridge_r2s.append(r2_score(y_test, pred_r))
            ridge_corrs.append(np.corrcoef(y_test, pred_r)[0, 1])

            gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            gb.fit(X_train, y_train)
            pred_g = gb.predict(X_test)
            gb_r2s.append(r2_score(y_test, pred_g))
            gb_corrs.append(np.corrcoef(y_test, pred_g)[0, 1])

        print(f"\n  {name} ({len(cols)} features):")
        print(f"    Ridge:  R²={np.mean(ridge_r2s):.4f}  corr={np.mean(ridge_corrs):.4f}")
        print(f"    GB:     R²={np.mean(gb_r2s):.4f}  corr={np.mean(gb_corrs):.4f}")

    # Feature importance from full model
    all_cols = list(feature_sets.values())[-1]
    valid = df[all_cols + [target]].notna().all(axis=1)
    X_all = df.loc[valid, all_cols].values
    y_all = df.loc[valid, target].values
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(StandardScaler().fit_transform(X_all), y_all)
    importances = sorted(zip(all_cols, gb.feature_importances_), key=lambda x: -x[1])

    print(f"\n  Top 20 features for vol prediction (GB, all combined):")
    for i, (col, imp) in enumerate(importances[:20], 1):
        tag = "[OI]" if col in oi_cols else "[OB]" if col in ob_cols else "[  ]"
        print(f"    {i:>2}. {tag} {col:40s} importance={imp:.4f}")


# ---------------------------------------------------------------------------
# Experiment 5: Crowding & Extreme Detection
# ---------------------------------------------------------------------------

def exp5_crowding_extremes(df):
    """Test if positioning extremes predict reversals."""
    print(f"\n{'='*70}")
    print(f"  EXP 5: CROWDING & EXTREME DETECTION → REVERSALS")
    print(f"{'='*70}")

    # Test: when OI is extreme + funding is extreme → reversal?
    signals = [
        ("oi_zscore_24h", "OI z-score"),
        ("funding_zscore_24h", "Funding z-score"),
        ("ls_top_zscore_24h", "Top trader LS z-score"),
        ("ls_global_zscore_24h", "Global LS z-score"),
        ("taker_zscore_24h", "Taker ratio z-score"),
    ]

    for col, label in signals:
        if col not in df.columns:
            continue

        valid = df[[col, "fwd_ret_4h"]].notna().all(axis=1)
        sub = df[valid].copy()
        if len(sub) < 200:
            continue

        z = sub[col].values
        rets = sub["fwd_ret_4h"].values

        print(f"\n  {label} ({col}):")
        print(f"    {'Condition':25s} {'N':>6s} {'Avg Ret':>10s} {'WR':>8s} {'Sharpe':>8s}")
        print(f"    {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")

        for thresh, direction, name in [
            (2.0, "contra", "z > +2.0 → short"),
            (1.5, "contra", "z > +1.5 → short"),
            (1.0, "contra", "z > +1.0 → short"),
            (-1.0, "contra", "z < -1.0 → long"),
            (-1.5, "contra", "z < -1.5 → long"),
            (-2.0, "contra", "z < -2.0 → long"),
        ]:
            if thresh > 0:
                mask = z > thresh
                trade_rets = -rets[mask] - FEE_BPS  # contra = short when high
            else:
                mask = z < thresh
                trade_rets = rets[mask] - FEE_BPS  # contra = long when low

            n = mask.sum()
            if n < 5:
                continue

            avg = np.mean(trade_rets)
            wr = (trade_rets > 0).mean() * 100
            sharpe = np.mean(trade_rets) / (np.std(trade_rets) + 1e-10) * np.sqrt(n)
            flag = "✅" if avg > 0 else "  "
            print(f"  {flag} {name:25s} {n:>6d} {avg:>+10.1f}bps {wr:>7.1f}% {sharpe:>+8.2f}")


# ---------------------------------------------------------------------------
# Experiment 6: Walk-Forward Combined Signal
# ---------------------------------------------------------------------------

def exp6_walkforward_combined(df):
    """Walk-forward test of combined OI/funding + OHLCV + OB signal."""
    print(f"\n{'='*70}")
    print(f"  EXP 6: WALK-FORWARD — OI/FUNDING COMBINED SIGNAL")
    print(f"{'='*70}")

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    oi_cols = [c for c in OI_FUNDING_FEATURES if c in df.columns]
    ob_cols = [c for c in OB_FEATURES_CORE if c in df.columns]

    target = "fwd_ret_4h"

    feature_combos = {
        "OHLCV only": ohlcv_cols,
        "OHLCV + OI/F": ohlcv_cols + oi_cols,
    }
    if ob_cols:
        feature_combos["OHLCV + OB + OI/F"] = ohlcv_cols + ob_cols + oi_cols

    for name, cols in feature_combos.items():
        valid = df[cols + [target]].notna().all(axis=1)
        sub = df[valid].reset_index(drop=True)
        X = sub[cols].values
        y = sub[target].values

        if len(X) < 500:
            continue

        # Expanding window walk-forward
        train_size = len(X) // 3  # first third for initial training
        preds = np.full(len(X), np.nan)

        step = 48  # retrain every 4h
        for start in range(train_size, len(X), step):
            end = min(start + step, len(X))
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[:start])
            X_test = scaler.transform(X[start:end])

            ridge = Ridge(alpha=10)
            ridge.fit(X_train, y[:start])
            preds[start:end] = ridge.predict(X_test)

        # Evaluate
        valid_pred = ~np.isnan(preds)
        if valid_pred.sum() < 100:
            continue

        pred_vals = preds[valid_pred]
        actual_vals = y[valid_pred]

        ic = np.corrcoef(pred_vals, actual_vals)[0, 1]
        rank_ic = scipy_stats.spearmanr(pred_vals, actual_vals).correlation

        print(f"\n  {name} ({len(cols)} features):")
        print(f"    IC={ic:+.4f}  rank_IC={rank_ic:+.4f}  n={valid_pred.sum()}")

        # Quintile analysis
        n_valid = valid_pred.sum()
        sorted_idx = np.argsort(pred_vals)
        q_size = n_valid // 5
        for q in range(5):
            q_start = q * q_size
            q_end = (q + 1) * q_size if q < 4 else n_valid
            q_rets = actual_vals[sorted_idx[q_start:q_end]]
            q_avg = np.mean(q_rets)
            q_wr = (q_rets > 0).mean() * 100
            label = ["Bottom", "Q2", "Q3", "Q4", "Top"][q]
            print(f"    {label:8s}: avg={q_avg:>+8.1f}bps  wr={q_wr:.1f}%  n={q_end-q_start}")

        # Long top quintile, short bottom quintile
        top_rets = actual_vals[sorted_idx[-q_size:]] - FEE_BPS
        bot_rets = -actual_vals[sorted_idx[:q_size]] - FEE_BPS
        ls_rets = np.concatenate([top_rets, bot_rets])
        ls_avg = np.mean(ls_rets)
        ls_wr = (ls_rets > 0).mean() * 100
        ls_sharpe = np.mean(ls_rets) / (np.std(ls_rets) + 1e-10) * np.sqrt(len(ls_rets))
        flag = "✅" if ls_avg > 0 else "  "
        print(f"  {flag} Long-short (Q5-Q1): avg={ls_avg:>+.1f}bps  wr={ls_wr:.1f}%  sharpe={ls_sharpe:+.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  OI/FUNDING RESEARCH — {SYMBOL} Dec 2025")
    print(f"  Data: 5-min bars + OI/funding + OB features")
    print(f"{'='*70}")

    # Load all data
    ohlcv_df = load_ohlcv_bars()
    ob_futures = load_ob_features("futures")
    metrics_df = load_metrics()
    premium_df = load_premium_index()

    # Build OI/funding features
    oi_df = build_oi_funding_features(metrics_df, premium_df)

    # Merge everything
    df = merge_all(ohlcv_df, ob_futures, oi_df)

    # Regime labels (same as ob_research.py)
    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    valid_regime = df[ohlcv_cols].notna().all(axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.loc[valid_regime, ohlcv_cols])

    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                           n_init=10, random_state=42, max_iter=300)
    labels_raw = gmm.fit_predict(X_scaled)

    rvol_col = "rvol_1h" if "rvol_1h" in df.columns else ohlcv_cols[0]
    if np.mean(df.loc[valid_regime, rvol_col].values[labels_raw == 0]) > \
       np.mean(df.loc[valid_regime, rvol_col].values[labels_raw == 1]):
        labels_raw = 1 - labels_raw

    labels = np.full(len(df), -1, dtype=np.int8)
    labels[valid_regime.values] = labels_raw

    regime_names = {0: "quiet", 1: "volatile"}
    n_quiet = (labels == 0).sum()
    n_vol = (labels == 1).sum()
    print(f"\n  Regime labels: quiet={n_quiet} ({n_quiet/len(df)*100:.1f}%), "
          f"volatile={n_vol} ({n_vol/len(df)*100:.1f}%)")

    # Run experiments
    exp1_regime_profiles(df, labels, regime_names)
    exp2_regime_detection(df, labels)
    exp3_directional_signal(df)
    exp4_vol_prediction(df)
    exp5_crowding_extremes(df)
    exp6_walkforward_combined(df)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()
    SYMBOL = args.symbol
    main()
