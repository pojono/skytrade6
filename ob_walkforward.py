#!/usr/bin/env python3
"""
Walk-forward tests for orderbook signals — v23b

Tests:
  1. Futures-spot depth ratio as medium-term (4h) signal
  2. ob_mid_volatility for vol prediction (walk-forward Ridge)
  3. Combined OB signal (depth ratio + vol + imbalance)

Walk-forward: train on expanding window, predict out-of-sample.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path("./parquet")
SYMBOL = "BTCUSDT"
FEE_BPS = 7  # round-trip taker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ob_features(symbol, market="futures"):
    feat_dir = PARQUET_DIR / symbol / "ob_features_5m" / f"bybit_{market}"
    dfs = []
    for f in sorted(feat_dir.glob("*.parquet")):
        dfs.append(pd.read_parquet(f))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_ohlcv():
    from regime_detection import load_bars, compute_regime_features
    df = load_bars(SYMBOL, "2025-12-01", "2025-12-31")
    df = compute_regime_features(df)
    return df


def merge_all():
    """Load OHLCV + futures OB + spot OB, merge everything."""
    print("Loading data...")
    ohlcv = load_ohlcv()
    fut = load_ob_features(SYMBOL, "futures")
    spot = load_ob_features(SYMBOL, "spot")

    print(f"  OHLCV: {len(ohlcv)} bars")
    print(f"  Futures OB: {len(fut)} bars")
    print(f"  Spot OB: {len(spot)} bars")

    ohlcv = ohlcv.copy()
    # timestamp_us already exists in OHLCV from load_bars

    # Merge futures OB
    fut_cols = {c: f"fut_{c}" for c in fut.columns if c != "timestamp_us"}
    fut_renamed = fut.rename(columns=fut_cols)
    df = pd.merge_asof(
        ohlcv.sort_values("timestamp_us"),
        fut_renamed.sort_values("timestamp_us"),
        on="timestamp_us", tolerance=300_000_000, direction="nearest",
    )

    # Merge spot OB
    spot_cols = {c: f"spot_{c}" for c in spot.columns if c != "timestamp_us"}
    spot_renamed = spot.rename(columns=spot_cols)
    df = pd.merge_asof(
        df.sort_values("timestamp_us"),
        spot_renamed.sort_values("timestamp_us"),
        on="timestamp_us", tolerance=300_000_000, direction="nearest",
    )

    # Compute basis features
    df["basis_depth_ratio"] = df["fut_ob_total_depth_mean"] / df["spot_ob_total_depth_mean"].clip(lower=0.01)
    df["basis_imb_1bps"] = df["fut_ob_imb_1bps_mean"] - df["spot_ob_imb_1bps_mean"]
    df["basis_imb_2bps"] = df["fut_ob_imb_2bps_mean"] - df["spot_ob_imb_2bps_mean"]
    df["basis_spread"] = df["fut_ob_spread_mean"] - df["spot_ob_spread_mean"]

    n_matched = df["fut_ob_spread_mean"].notna().sum()
    print(f"  Merged: {len(df)} bars, {n_matched} with OB data")
    return df


# ---------------------------------------------------------------------------
# Walk-forward test 1: Depth ratio signal
# ---------------------------------------------------------------------------

def wf_depth_ratio_signal(df):
    """Walk-forward test: trade on futures-spot depth ratio."""
    print(f"\n{'='*70}")
    print(f"  WF TEST 1: FUTURES-SPOT DEPTH RATIO SIGNAL")
    print(f"{'='*70}")

    ret = df["returns"].values
    n = len(ret)

    # Forward returns at various horizons
    horizons = {"1h": 12, "2h": 24, "4h": 48}

    for hz_name, hz_bars in horizons.items():
        fwd = np.full(n, np.nan)
        for i in range(n - hz_bars):
            fwd[i] = np.sum(ret[i+1:i+1+hz_bars])

        # Walk-forward: use expanding window z-score
        depth_ratio = df["basis_depth_ratio"].values
        valid = ~np.isnan(depth_ratio) & ~np.isnan(fwd)

        # Need at least 1 day warmup (288 bars)
        warmup = 288 * 3  # 3 days warmup
        if valid.sum() < warmup + 100:
            print(f"  {hz_name}: not enough data")
            continue

        # Walk-forward z-score: use expanding mean/std
        z_scores = np.full(n, np.nan)
        for i in range(warmup, n):
            window = depth_ratio[max(0, i-288*7):i]  # 7-day lookback for stats
            window = window[~np.isnan(window)]
            if len(window) > 50:
                z_scores[i] = (depth_ratio[i] - np.mean(window)) / max(np.std(window), 1e-10)

        # Strategy: short when depth ratio z > threshold (excess futures depth = bearish)
        #           long when depth ratio z < -threshold
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            long_pnl = []
            short_pnl = []
            long_bars = []
            short_bars = []

            i = warmup
            while i < n - hz_bars:
                if np.isnan(z_scores[i]):
                    i += 1
                    continue

                if z_scores[i] > threshold:
                    # Short signal
                    pnl_bps = -fwd[i] * 10000 - FEE_BPS
                    short_pnl.append(pnl_bps)
                    short_bars.append(i)
                    i += hz_bars  # hold for full horizon, no overlap
                elif z_scores[i] < -threshold:
                    # Long signal
                    pnl_bps = fwd[i] * 10000 - FEE_BPS
                    long_pnl.append(pnl_bps)
                    long_bars.append(i)
                    i += hz_bars
                else:
                    i += 1

            all_pnl = long_pnl + short_pnl
            n_trades = len(all_pnl)
            if n_trades < 10:
                continue

            avg = np.mean(all_pnl)
            wr = np.mean(np.array(all_pnl) > 0)
            total = np.sum(all_pnl)
            sharpe = np.mean(all_pnl) / max(np.std(all_pnl), 1e-10) * np.sqrt(252 * 288 / hz_bars)

            long_avg = np.mean(long_pnl) if long_pnl else 0
            short_avg = np.mean(short_pnl) if short_pnl else 0

            marker = "✅" if avg > 0 else "  "
            print(f"  {marker} {hz_name} z>{threshold:.1f}: trades={n_trades:4d}, "
                  f"avg={avg:+.1f}bps, wr={wr:.1%}, total={total:+.0f}bps, "
                  f"sharpe={sharpe:.2f}, "
                  f"long={long_avg:+.1f}({len(long_pnl)}), short={short_avg:+.1f}({len(short_pnl)})")


# ---------------------------------------------------------------------------
# Walk-forward test 2: Vol prediction with OB features
# ---------------------------------------------------------------------------

def wf_vol_prediction(df):
    """Walk-forward Ridge vol prediction: OHLCV vs OHLCV+OB."""
    print(f"\n{'='*70}")
    print(f"  WF TEST 2: VOL PREDICTION — OHLCV vs OHLCV+OB")
    print(f"{'='*70}")

    from ob_research import OHLCV_FEATURES

    ob_vol_features = [
        "fut_ob_mid_volatility", "fut_ob_spread_std", "fut_ob_spread_max",
        "fut_ob_bid_depth_cv", "fut_ob_ask_depth_cv",
        "fut_ob_bid_wall_frac", "fut_ob_ask_wall_frac",
        "fut_ob_total_depth_mean",
        "fut_ob_imb_1bps_std", "fut_ob_imb_2bps_std",
    ]

    ret = df["returns"].values
    n = len(ret)

    # Target: 1h forward realized vol
    fwd_vol = np.full(n, np.nan)
    for i in range(n - 12):
        fwd_vol[i] = np.std(ret[i+1:i+13])

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    ob_cols = [c for c in ob_vol_features if c in df.columns]

    warmup = 288 * 3  # 3 days
    retrain_interval = 288  # retrain daily

    for name, feature_cols in [
        ("OHLCV only", ohlcv_cols),
        ("OHLCV + OB", ohlcv_cols + ob_cols),
        ("OB only", ob_cols),
    ]:
        X = df[feature_cols].values
        y = fwd_vol

        predictions = np.full(n, np.nan)
        scaler = StandardScaler()
        model = Ridge(alpha=1.0)
        last_train = -retrain_interval

        for i in range(warmup, n):
            if i - last_train >= retrain_interval:
                valid_mask = ~np.isnan(y[:i])
                X_train = np.nan_to_num(X[:i][valid_mask], nan=0, posinf=0, neginf=0)
                y_train = y[:i][valid_mask]
                if len(y_train) < 100:
                    continue
                scaler.fit(X_train)
                model.fit(scaler.transform(X_train), y_train)
                last_train = i

            x_i = np.nan_to_num(X[i:i+1], nan=0, posinf=0, neginf=0)
            predictions[i] = max(model.predict(scaler.transform(x_i))[0], 1e-8)

        # Evaluate
        valid = ~np.isnan(predictions) & ~np.isnan(fwd_vol)
        if valid.sum() < 100:
            print(f"  {name}: not enough predictions")
            continue

        pred_valid = predictions[valid]
        actual_valid = fwd_vol[valid]

        corr = np.corrcoef(pred_valid, actual_valid)[0, 1]
        ss_res = np.sum((actual_valid - pred_valid) ** 2)
        ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        # Rank correlation (more robust)
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(pred_valid, actual_valid)

        print(f"  {name:20s} ({len(feature_cols):2d} features): "
              f"R²={r2:.4f}  corr={corr:.4f}  rank_corr={rank_corr:.4f}  "
              f"n={valid.sum()}")

    return predictions  # return OHLCV+OB predictions for grid bot use


# ---------------------------------------------------------------------------
# Walk-forward test 3: Combined OB signal for direction
# ---------------------------------------------------------------------------

def wf_combined_ob_signal(df):
    """Walk-forward Ridge: predict 4h returns from OB features."""
    print(f"\n{'='*70}")
    print(f"  WF TEST 3: COMBINED OB SIGNAL → 4h RETURNS")
    print(f"{'='*70}")

    ob_signal_features = [
        "basis_depth_ratio", "basis_imb_1bps", "basis_imb_2bps", "basis_spread",
        "fut_ob_imb_0.5bps_mean", "fut_ob_imb_1bps_mean", "fut_ob_imb_2bps_mean",
        "fut_ob_imb_5bps_mean",
        "fut_ob_imb_1bps_trend", "fut_ob_imb_2bps_trend",
        "fut_ob_microprice_dev_bps",
        "fut_ob_bid_depth_change", "fut_ob_ask_depth_change",
        "fut_ob_bid_wall_ratio", "fut_ob_ask_wall_ratio",
        "fut_ob_mid_volatility", "fut_ob_spread_std",
    ]

    feature_cols = [c for c in ob_signal_features if c in df.columns]
    ret = df["returns"].values
    n = len(ret)

    # Target: 4h forward return
    fwd_4h = np.full(n, np.nan)
    for i in range(n - 48):
        fwd_4h[i] = np.sum(ret[i+1:i+49])

    X = df[feature_cols].values
    y = fwd_4h

    warmup = 288 * 3
    retrain_interval = 288

    predictions = np.full(n, np.nan)
    scaler = StandardScaler()
    model = Ridge(alpha=10.0)  # higher regularization for noisy target
    last_train = -retrain_interval

    for i in range(warmup, n):
        if i - last_train >= retrain_interval:
            valid_mask = ~np.isnan(y[:i])
            X_train = np.nan_to_num(X[:i][valid_mask], nan=0, posinf=0, neginf=0)
            y_train = y[:i][valid_mask]
            if len(y_train) < 100:
                continue
            scaler.fit(X_train)
            model.fit(scaler.transform(X_train), y_train)
            last_train = i

        x_i = np.nan_to_num(X[i:i+1], nan=0, posinf=0, neginf=0)
        predictions[i] = model.predict(scaler.transform(x_i))[0]

    # Evaluate IC
    valid = ~np.isnan(predictions) & ~np.isnan(fwd_4h)
    if valid.sum() < 100:
        print(f"  Not enough predictions")
        return

    pred_valid = predictions[valid]
    actual_valid = fwd_4h[valid]
    ic = np.corrcoef(pred_valid, actual_valid)[0, 1]
    from scipy.stats import spearmanr
    rank_ic, _ = spearmanr(pred_valid, actual_valid)

    print(f"  Ridge (α=10): IC={ic:.4f}  rank_IC={rank_ic:.4f}  n={valid.sum()}")

    # Backtest: trade on prediction sign
    for threshold_pct in [50, 60, 70, 80]:
        # Only trade when prediction is in top/bottom percentile
        pred_abs = np.abs(pred_valid)
        thresh = np.percentile(pred_abs, threshold_pct)

        trade_mask = pred_abs >= thresh
        trade_pred = pred_valid[trade_mask]
        trade_actual = actual_valid[trade_mask]

        if len(trade_pred) < 20:
            continue

        # PnL: sign(prediction) * actual_return - fees
        pnl_bps = np.sign(trade_pred) * trade_actual * 10000 - FEE_BPS
        avg = np.mean(pnl_bps)
        wr = np.mean(pnl_bps > 0)
        total = np.sum(pnl_bps)
        n_trades = len(pnl_bps)

        # Approximate Sharpe (annualized)
        trades_per_year = n_trades * (365 / 31)  # scale from 31 days
        if np.std(pnl_bps) > 0:
            sharpe = np.mean(pnl_bps) / np.std(pnl_bps) * np.sqrt(trades_per_year)
        else:
            sharpe = 0

        marker = "✅" if avg > 0 else "  "
        print(f"  {marker} top {100-threshold_pct}%: trades={n_trades:4d}, "
              f"avg={avg:+.1f}bps, wr={wr:.1%}, total={total:+.0f}bps, sharpe={sharpe:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print(f"  WALK-FORWARD OB TESTS — {SYMBOL} Dec 2025")
    print("=" * 70)

    df = merge_all()

    wf_depth_ratio_signal(df)
    vol_pred = wf_vol_prediction(df)
    wf_combined_ob_signal(df)

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
