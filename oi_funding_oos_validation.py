#!/usr/bin/env python3
"""
LS Ratio OOS Validation — v24 OOS

Validates the extraordinary v24 findings (LS ratio IC=0.20, Sharpe 9+)
on a completely out-of-sample period using Binance data warehouse metrics.

In-sample (v24):  Dec 2025 (31 days)
Out-of-sample:    May 1 - Oct 31, 2025 (184 days, 6 months earlier)

Methodology: Exact replication of v24 experiments 3, 5, 6 on OOS data.
Data: Binance metrics (LS ratios, OI, taker ratio) + Binance 5m klines.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy import stats as scipy_stats

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
SYMBOLS = ["BTCUSDT", "SOLUSDT"]

# OOS period — completely disjoint from v24's Dec 2025
OOS_START = "2025-05-01"
OOS_END = "2025-10-31"

# Also run on the original in-sample period for direct comparison
IS_START = "2025-11-01"
IS_END = "2026-01-31"

FEE_BPS = 7  # round-trip fee (same as v24)

# Metrics columns from Binance data warehouse
METRICS_RAW = [
    "open_interest", "open_interest_value",
    "top_trader_ls_ratio_accounts", "top_trader_ls_ratio_positions",
    "global_ls_ratio", "taker_buy_sell_ratio",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_binance_klines_5m(symbol, start_date, end_date):
    """Load Binance 5m klines from parquet."""
    kline_dir = PARQUET_DIR / symbol / "binance" / "klines_futures" / "5m"
    dates = pd.date_range(start_date, end_date)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = kline_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        print(f"  No klines found for {symbol}!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Klines: {len(df)} bars, {df['timestamp_us'].min()} to {df['timestamp_us'].max()}")
    return df


def load_binance_metrics(symbol, start_date, end_date):
    """Load Binance metrics (OI, LS ratios, taker ratio) from parquet."""
    metrics_dir = PARQUET_DIR / symbol / "binance" / "metrics"
    dates = pd.date_range(start_date, end_date)
    dfs = []
    loaded = 0
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = metrics_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
            loaded += 1
    if not dfs:
        print(f"  No metrics found for {symbol}!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Metrics: {loaded} days, {len(df)} rows, cols={list(df.columns)}")
    return df


def build_features(klines_df, metrics_df):
    """
    Merge klines + metrics and engineer OI/funding features.
    Replicates v24's build_oi_funding_features() exactly.
    """
    # Merge on timestamp
    merged = pd.merge_asof(
        klines_df.sort_values("timestamp_us"),
        metrics_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,  # 5 min tolerance
        direction="nearest",
    )

    n_matched = merged["open_interest"].notna().sum()
    print(f"  Merged: {len(merged)} bars, {n_matched} with metrics ({100*n_matched/len(merged):.1f}%)")

    # --- OI features ---
    for window, name in [(1, "5m"), (12, "1h"), (48, "4h"), (288, "24h")]:
        merged[f"oi_change_{name}"] = merged["open_interest"].pct_change(window) * 100
    merged["oi_accel_1h"] = merged["oi_change_1h"].diff(12)
    oi_mean = merged["open_interest"].rolling(288, min_periods=48).mean()
    oi_std = merged["open_interest"].rolling(288, min_periods=48).std()
    merged["oi_zscore_24h"] = (merged["open_interest"] - oi_mean) / oi_std.replace(0, np.nan)
    merged["oi_value_change_1h"] = merged["open_interest_value"].pct_change(12) * 100

    # --- LS ratio features ---
    merged["ls_ratio_top"] = merged["top_trader_ls_ratio_accounts"]
    merged["ls_ratio_global"] = merged["global_ls_ratio"]
    merged["ls_top_change_1h"] = merged["top_trader_ls_ratio_accounts"].pct_change(12) * 100
    merged["ls_global_change_1h"] = merged["global_ls_ratio"].pct_change(12) * 100

    # LS z-scores (rolling 24h)
    ls_mean = merged["top_trader_ls_ratio_accounts"].rolling(288, min_periods=48).mean()
    ls_std = merged["top_trader_ls_ratio_accounts"].rolling(288, min_periods=48).std()
    merged["ls_top_zscore_24h"] = (merged["top_trader_ls_ratio_accounts"] - ls_mean) / ls_std.replace(0, np.nan)
    ls_g_mean = merged["global_ls_ratio"].rolling(288, min_periods=48).mean()
    ls_g_std = merged["global_ls_ratio"].rolling(288, min_periods=48).std()
    merged["ls_global_zscore_24h"] = (merged["global_ls_ratio"] - ls_g_mean) / ls_g_std.replace(0, np.nan)

    # --- Taker features ---
    merged["taker_ratio"] = merged["taker_buy_sell_ratio"]
    merged["taker_ratio_1h"] = merged["taker_buy_sell_ratio"].rolling(12, min_periods=3).mean()
    merged["taker_ratio_4h"] = merged["taker_buy_sell_ratio"].rolling(48, min_periods=12).mean()
    tk_mean = merged["taker_buy_sell_ratio"].rolling(288, min_periods=48).mean()
    tk_std = merged["taker_buy_sell_ratio"].rolling(288, min_periods=48).std()
    merged["taker_zscore_24h"] = (merged["taker_buy_sell_ratio"] - tk_mean) / tk_std.replace(0, np.nan)

    # --- Cross features ---
    merged["oi_x_taker"] = merged["oi_change_1h"] * (merged["taker_buy_sell_ratio"] - 1)

    # --- OHLCV-derived features (basic, for walk-forward) ---
    merged["ret_1bar"] = merged["close"].pct_change(1) * 10000  # bps
    merged["rvol_1h"] = merged["ret_1bar"].rolling(12, min_periods=6).std()
    merged["rvol_4h"] = merged["ret_1bar"].rolling(48, min_periods=12).std()
    merged["rvol_24h"] = merged["ret_1bar"].rolling(288, min_periods=48).std()
    merged["momentum_1h"] = merged["close"].pct_change(12) * 10000
    merged["momentum_4h"] = merged["close"].pct_change(48) * 10000

    # Efficiency ratio
    for w, name in [(12, "1h"), (48, "4h")]:
        net_move = (merged["close"] - merged["close"].shift(w)).abs()
        sum_moves = merged["ret_1bar"].abs().rolling(w, min_periods=w//2).sum()
        merged[f"efficiency_{name}"] = net_move / (sum_moves + 1e-10)

    merged["price_vs_sma_24h"] = (merged["close"] / merged["close"].rolling(288, min_periods=48).mean() - 1) * 100

    # --- Forward returns ---
    merged["fwd_ret_5m"] = merged["close"].pct_change(1).shift(-1) * 10000
    merged["fwd_ret_1h"] = merged["close"].pct_change(12).shift(-12) * 10000
    merged["fwd_ret_4h"] = merged["close"].pct_change(48).shift(-48) * 10000

    return merged


# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

OI_FUNDING_FEATURES = [
    "oi_change_5m", "oi_change_1h", "oi_change_4h", "oi_change_24h",
    "oi_accel_1h", "oi_zscore_24h", "oi_value_change_1h",
    "ls_ratio_top", "ls_ratio_global",
    "ls_top_change_1h", "ls_global_change_1h",
    "ls_top_zscore_24h", "ls_global_zscore_24h",
    "taker_ratio", "taker_ratio_1h", "taker_ratio_4h", "taker_zscore_24h",
    "oi_x_taker",
]

OHLCV_FEATURES = [
    "rvol_1h", "rvol_4h", "rvol_24h",
    "momentum_1h", "momentum_4h",
    "efficiency_1h", "efficiency_4h",
    "price_vs_sma_24h",
]


# ---------------------------------------------------------------------------
# Experiment 3: Directional Signal (IC + Simple Backtest)
# ---------------------------------------------------------------------------

def exp3_directional_signal(df, label):
    """IC analysis and simple backtest — replicates v24 Exp 3."""
    print(f"\n{'='*70}")
    print(f"  EXP 3: DIRECTIONAL SIGNAL — {label}")
    print(f"{'='*70}")

    oi_cols = [c for c in OI_FUNDING_FEATURES if c in df.columns]
    horizons = {
        "5min": "fwd_ret_5m",
        "1h": "fwd_ret_1h",
        "4h": "fwd_ret_4h",
    }

    # IC analysis
    print(f"\n  Information Coefficient (IC) — OI/funding vs forward returns:")
    print(f"  {'Feature':35s} {'5min':>8s} {'1h':>8s} {'4h':>8s}")
    print(f"  {'-'*59}")

    ic_results = {}
    for col in oi_cols:
        ics = []
        for h_name, h_col in horizons.items():
            valid = df[[col, h_col]].notna().all(axis=1)
            if valid.sum() < 100:
                ics.append(np.nan)
                continue
            ic = df.loc[valid, col].corr(df.loc[valid, h_col])
            ics.append(ic)
        ic_results[col] = ics
        print(f"  {col:35s} {ics[0]:>+8.4f} {ics[1]:>+8.4f} {ics[2]:>+8.4f}")

    # Highlight best 4h IC
    best_4h = sorted(ic_results.items(), key=lambda x: abs(x[1][2]) if not np.isnan(x[1][2]) else 0, reverse=True)
    print(f"\n  Top 5 features by |IC| at 4h:")
    for col, ics in best_4h[:5]:
        print(f"    {col:35s} IC_4h={ics[2]:+.4f}")

    # Simple backtest on key signals
    print(f"\n  Simple backtest — z>1 long, z<-1 short (4h hold, {FEE_BPS}bps fee):")
    print(f"  {'Signal':35s} {'Trades':>7s} {'Avg PnL':>10s} {'WR':>7s} {'Long':>12s} {'Short':>12s}")
    print(f"  {'-'*83}")

    signal_cols = [
        "ls_top_zscore_24h", "ls_global_zscore_24h",
        "oi_zscore_24h", "taker_zscore_24h",
        "oi_change_4h", "oi_x_taker",
    ]

    backtest_results = {}
    for col in signal_cols:
        if col not in df.columns:
            continue
        valid = df[[col, "fwd_ret_4h"]].notna().all(axis=1)
        sub = df[valid].copy()
        if len(sub) < 100:
            continue

        vals = sub[col].values
        rets = sub["fwd_ret_4h"].values
        # Handle inf values
        vals = np.where(np.isfinite(vals), vals, np.nan)
        z = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-10)

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
        print(f"  {flag} {col:33s} {n_trades:>7d} {avg_pnl:>+10.1f}bps {wr:>6.1f}% "
              f"L={long_avg:>+.1f}({n_long}) S={short_avg:>+.1f}({n_short})")

        backtest_results[col] = {
            "trades": n_trades, "avg_pnl": avg_pnl, "wr": wr,
            "long_avg": long_avg, "short_avg": short_avg,
            "n_long": n_long, "n_short": n_short,
        }

    return ic_results, backtest_results


# ---------------------------------------------------------------------------
# Experiment 5: Crowding & Extreme Detection
# ---------------------------------------------------------------------------

def exp5_crowding_extremes(df, label):
    """Test positioning extremes as contrarian/momentum signals."""
    print(f"\n{'='*70}")
    print(f"  EXP 5: CROWDING & EXTREMES — {label}")
    print(f"{'='*70}")

    signals = [
        ("oi_zscore_24h", "OI z-score"),
        ("ls_top_zscore_24h", "Top trader LS z-score"),
        ("ls_global_zscore_24h", "Global LS z-score"),
        ("taker_zscore_24h", "Taker ratio z-score"),
    ]

    results = {}
    for col, col_label in signals:
        if col not in df.columns:
            continue

        valid = df[[col, "fwd_ret_4h"]].notna().all(axis=1)
        sub = df[valid].copy()
        if len(sub) < 200:
            continue

        z = sub[col].values
        rets = sub["fwd_ret_4h"].values

        print(f"\n  {col_label} ({col}):")
        print(f"    {'Condition':25s} {'N':>6s} {'Avg Ret':>10s} {'WR':>8s} {'Sharpe':>8s}")
        print(f"    {'-'*59}")

        col_results = {}
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
                trade_rets = -rets[mask] - FEE_BPS
            else:
                mask = z < thresh
                trade_rets = rets[mask] - FEE_BPS

            n = mask.sum()
            if n < 5:
                continue

            avg = np.mean(trade_rets)
            wr = (trade_rets > 0).mean() * 100
            sharpe = np.mean(trade_rets) / (np.std(trade_rets) + 1e-10) * np.sqrt(n)
            flag = "✅" if avg > 0 else "  "
            print(f"  {flag} {name:25s} {n:>6d} {avg:>+10.1f}bps {wr:>7.1f}% {sharpe:>+8.2f}")
            col_results[name] = {"n": n, "avg": avg, "wr": wr, "sharpe": sharpe}

        # Also test MOMENTUM (not contrarian) for LS ratios
        if "ls_" in col:
            print(f"\n    MOMENTUM test (follow the crowd):")
            for thresh, name in [
                (1.0, "z > +1.0 → LONG"),
                (1.5, "z > +1.5 → LONG"),
                (2.0, "z > +2.0 → LONG"),
                (-1.0, "z < -1.0 → SHORT"),
                (-1.5, "z < -1.5 → SHORT"),
                (-2.0, "z < -2.0 → SHORT"),
            ]:
                if thresh > 0:
                    mask = z > thresh
                    trade_rets = rets[mask] - FEE_BPS  # momentum = long when high
                else:
                    mask = z < thresh
                    trade_rets = -rets[mask] - FEE_BPS  # momentum = short when low

                n = mask.sum()
                if n < 5:
                    continue

                avg = np.mean(trade_rets)
                wr = (trade_rets > 0).mean() * 100
                sharpe = np.mean(trade_rets) / (np.std(trade_rets) + 1e-10) * np.sqrt(n)
                flag = "✅" if avg > 0 else "  "
                print(f"  {flag} {name:25s} {n:>6d} {avg:>+10.1f}bps {wr:>7.1f}% {sharpe:>+8.2f}")
                col_results[f"mom_{name}"] = {"n": n, "avg": avg, "wr": wr, "sharpe": sharpe}

        results[col] = col_results

    return results


# ---------------------------------------------------------------------------
# Experiment 6: Walk-Forward Combined Signal
# ---------------------------------------------------------------------------

def exp6_walkforward_combined(df, label):
    """Walk-forward quintile L/S — replicates v24 Exp 6."""
    print(f"\n{'='*70}")
    print(f"  EXP 6: WALK-FORWARD COMBINED SIGNAL — {label}")
    print(f"{'='*70}")

    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]
    oi_cols = [c for c in OI_FUNDING_FEATURES if c in df.columns]

    target = "fwd_ret_4h"

    feature_combos = {
        "OHLCV only": ohlcv_cols,
        "OHLCV + OI/F": ohlcv_cols + oi_cols,
    }

    wf_results = {}
    for name, cols in feature_combos.items():
        valid = df[cols + [target]].notna().all(axis=1)
        sub = df[valid].reset_index(drop=True)
        X = sub[cols].values
        y = sub[target].values

        if len(X) < 500:
            print(f"\n  {name}: Not enough data ({len(X)} bars)")
            continue

        # Expanding window walk-forward (same as v24)
        train_size = len(X) // 3
        preds = np.full(len(X), np.nan)

        step = 48  # retrain every 4h
        n_retrains = 0
        for start in range(train_size, len(X), step):
            end = min(start + step, len(X))
            scaler = StandardScaler()
            X_tr = X[:start].copy()
            X_te = X[start:end].copy()
            y_tr = y[:start].copy()

            # Replace inf with nan, then fill nan with 0
            X_tr = np.where(np.isfinite(X_tr), X_tr, np.nan)
            X_te = np.where(np.isfinite(X_te), X_te, np.nan)
            X_tr = np.nan_to_num(X_tr, nan=0.0)
            X_te = np.nan_to_num(X_te, nan=0.0)
            y_tr = np.nan_to_num(y_tr, nan=0.0)

            X_train = scaler.fit_transform(X_tr)
            X_test = scaler.transform(X_te)

            ridge = Ridge(alpha=10)
            ridge.fit(X_train, y_tr)
            preds[start:end] = ridge.predict(X_test)
            n_retrains += 1

        # Evaluate
        valid_pred = ~np.isnan(preds)
        if valid_pred.sum() < 100:
            continue

        pred_vals = preds[valid_pred]
        actual_vals = y[valid_pred]

        ic = np.corrcoef(pred_vals, actual_vals)[0, 1]
        rank_ic = scipy_stats.spearmanr(pred_vals, actual_vals).correlation

        print(f"\n  {name} ({len(cols)} features, {n_retrains} retrains):")
        print(f"    IC={ic:+.4f}  rank_IC={rank_ic:+.4f}  n={valid_pred.sum()}")

        # Quintile analysis
        n_valid = valid_pred.sum()
        sorted_idx = np.argsort(pred_vals)
        q_size = n_valid // 5

        quintile_avgs = []
        for q in range(5):
            q_start = q * q_size
            q_end = (q + 1) * q_size if q < 4 else n_valid
            q_rets = actual_vals[sorted_idx[q_start:q_end]]
            q_avg = np.mean(q_rets)
            q_wr = (q_rets > 0).mean() * 100
            q_label = ["Bottom", "Q2", "Q3", "Q4", "Top"][q]
            print(f"    {q_label:8s}: avg={q_avg:>+8.1f}bps  wr={q_wr:.1f}%  n={q_end-q_start}")
            quintile_avgs.append(q_avg)

        # Long-short
        top_rets = actual_vals[sorted_idx[-q_size:]] - FEE_BPS
        bot_rets = -actual_vals[sorted_idx[:q_size]] - FEE_BPS
        ls_rets = np.concatenate([top_rets, bot_rets])
        ls_avg = np.mean(ls_rets)
        ls_wr = (ls_rets > 0).mean() * 100
        ls_sharpe = np.mean(ls_rets) / (np.std(ls_rets) + 1e-10) * np.sqrt(len(ls_rets))
        flag = "✅" if ls_avg > 0 else "  "
        print(f"  {flag} Long-short (Q5-Q1): avg={ls_avg:>+.1f}bps  wr={ls_wr:.1f}%  sharpe={ls_sharpe:+.2f}")

        # Monthly breakdown
        if "timestamp_us" in sub.columns:
            sub_valid = sub[valid_pred].copy()
            sub_valid["pred"] = pred_vals
            sub_valid["actual"] = actual_vals
            sub_valid["dt"] = pd.to_datetime(sub_valid["timestamp_us"], unit="us")
            sub_valid["month"] = sub_valid["dt"].dt.to_period("M")

            print(f"\n    Monthly IC breakdown:")
            for month, grp in sub_valid.groupby("month"):
                if len(grp) < 50:
                    continue
                m_ic = grp["pred"].corr(grp["actual"])
                m_avg = grp["actual"].mean()
                print(f"      {month}: IC={m_ic:+.4f}  avg_ret={m_avg:+.1f}bps  n={len(grp)}")

        wf_results[name] = {
            "ic": ic, "rank_ic": rank_ic, "n": int(valid_pred.sum()),
            "ls_avg": ls_avg, "ls_wr": ls_wr, "ls_sharpe": ls_sharpe,
            "quintile_avgs": quintile_avgs,
        }

    return wf_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_period(symbol, start_date, end_date, period_label):
    """Run all experiments for one symbol and period."""
    print(f"\n{'#'*70}")
    print(f"  {symbol} — {period_label}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"{'#'*70}")

    t0 = time.time()

    # Load data
    print(f"\nLoading data...")
    klines = load_binance_klines_5m(symbol, start_date, end_date)
    metrics = load_binance_metrics(symbol, start_date, end_date)

    if klines.empty or metrics.empty:
        print(f"  SKIPPING — missing data")
        return None

    # Build features
    print(f"\nBuilding features...")
    df = build_features(klines, metrics)

    # Data summary
    n_total = len(df)
    n_with_ls = df["ls_ratio_top"].notna().sum()
    n_with_fwd = df["fwd_ret_4h"].notna().sum()
    print(f"\n  Data summary:")
    print(f"    Total bars:      {n_total}")
    print(f"    With LS ratio:   {n_with_ls} ({100*n_with_ls/n_total:.1f}%)")
    print(f"    With fwd_ret_4h: {n_with_fwd} ({100*n_with_fwd/n_total:.1f}%)")

    if n_with_ls < 100:
        print(f"  SKIPPING — not enough LS ratio data")
        return None

    # LS ratio basic stats
    ls_top = df["ls_ratio_top"].dropna()
    print(f"\n  LS ratio stats:")
    print(f"    Mean:   {ls_top.mean():.4f}")
    print(f"    Std:    {ls_top.std():.4f}")
    print(f"    Min:    {ls_top.min():.4f}")
    print(f"    Max:    {ls_top.max():.4f}")
    print(f"    Median: {ls_top.median():.4f}")

    # Run experiments
    ic_results, bt_results = exp3_directional_signal(df, f"{symbol} {period_label}")
    ext_results = exp5_crowding_extremes(df, f"{symbol} {period_label}")
    wf_results = exp6_walkforward_combined(df, f"{symbol} {period_label}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")

    return {
        "ic": ic_results,
        "backtest": bt_results,
        "extremes": ext_results,
        "walkforward": wf_results,
    }


def print_comparison(is_results, oos_results, symbol):
    """Print side-by-side comparison of IS vs OOS results."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {symbol} — In-Sample vs Out-of-Sample")
    print(f"{'='*70}")

    # IC comparison
    print(f"\n  IC at 4h horizon:")
    print(f"  {'Feature':35s} {'IS (Nov-Jan)':>12s} {'OOS (May-Oct)':>14s} {'Δ':>8s}")
    print(f"  {'-'*71}")

    if is_results and oos_results and is_results.get("ic") and oos_results.get("ic"):
        for col in OI_FUNDING_FEATURES:
            if col in is_results["ic"] and col in oos_results["ic"]:
                is_ic = is_results["ic"][col][2]  # 4h is index 2
                oos_ic = oos_results["ic"][col][2]
                if not np.isnan(is_ic) and not np.isnan(oos_ic):
                    delta = oos_ic - is_ic
                    flag = "✅" if np.sign(is_ic) == np.sign(oos_ic) and abs(oos_ic) > 0.05 else "⚠️" if np.sign(is_ic) == np.sign(oos_ic) else "❌"
                    print(f"  {flag} {col:33s} {is_ic:>+12.4f} {oos_ic:>+14.4f} {delta:>+8.4f}")

    # Walk-forward comparison
    print(f"\n  Walk-forward L/S results:")
    print(f"  {'Feature Set':25s} {'IS IC':>8s} {'IS Sharpe':>10s} {'OOS IC':>8s} {'OOS Sharpe':>11s}")
    print(f"  {'-'*64}")

    if is_results and oos_results:
        is_wf = is_results.get("walkforward", {})
        oos_wf = oos_results.get("walkforward", {})
        for name in ["OHLCV only", "OHLCV + OI/F"]:
            is_r = is_wf.get(name, {})
            oos_r = oos_wf.get(name, {})
            is_ic = is_r.get("ic", float("nan"))
            is_sh = is_r.get("ls_sharpe", float("nan"))
            oos_ic = oos_r.get("ic", float("nan"))
            oos_sh = oos_r.get("ls_sharpe", float("nan"))
            print(f"  {name:25s} {is_ic:>+8.4f} {is_sh:>+10.2f} {oos_ic:>+8.4f} {oos_sh:>+11.2f}")

    # Backtest comparison for key signals
    print(f"\n  Simple backtest (z-threshold, 4h hold, {FEE_BPS}bps fee):")
    print(f"  {'Signal':35s} {'IS Avg':>8s} {'IS WR':>7s} {'OOS Avg':>9s} {'OOS WR':>8s}")
    print(f"  {'-'*69}")

    if is_results and oos_results:
        is_bt = is_results.get("backtest", {})
        oos_bt = oos_results.get("backtest", {})
        for col in ["ls_top_zscore_24h", "ls_global_zscore_24h", "oi_zscore_24h", "taker_zscore_24h"]:
            is_r = is_bt.get(col, {})
            oos_r = oos_bt.get(col, {})
            is_avg = is_r.get("avg_pnl", float("nan"))
            is_wr = is_r.get("wr", float("nan"))
            oos_avg = oos_r.get("avg_pnl", float("nan"))
            oos_wr = oos_r.get("wr", float("nan"))
            flag = "✅" if oos_avg > 0 else "❌"
            print(f"  {flag} {col:33s} {is_avg:>+8.1f} {is_wr:>6.1f}% {oos_avg:>+9.1f} {oos_wr:>7.1f}%")


def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  LS RATIO OOS VALIDATION")
    print(f"  In-sample:      Nov 2025 - Jan 2026 (92 days)")
    print(f"  Out-of-sample:  May - Oct 2025 (184 days)")
    print(f"  Symbols:        {', '.join(SYMBOLS)}")
    print(f"{'='*70}")

    all_results = {}

    for symbol in SYMBOLS:
        # OOS period first
        oos_results = run_period(symbol, OOS_START, OOS_END, "OOS (May-Oct 2025)")

        # In-sample period for comparison
        is_results = run_period(symbol, IS_START, IS_END, "IS (Nov 2025 - Jan 2026)")

        all_results[symbol] = {"is": is_results, "oos": oos_results}

        # Print comparison
        print_comparison(is_results, oos_results, symbol)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")

    for symbol in SYMBOLS:
        is_r = all_results[symbol]["is"]
        oos_r = all_results[symbol]["oos"]

        print(f"\n  {symbol}:")

        # Key metric: ls_top_zscore_24h IC at 4h
        if is_r and oos_r and is_r.get("ic") and oos_r.get("ic"):
            is_ic = is_r["ic"].get("ls_top_zscore_24h", [0,0,0])[2]
            oos_ic = oos_r["ic"].get("ls_top_zscore_24h", [0,0,0])[2]
            print(f"    ls_top_zscore_24h IC@4h:  IS={is_ic:+.4f}  OOS={oos_ic:+.4f}")

        if is_r and oos_r:
            is_wf = is_r.get("walkforward", {}).get("OHLCV + OI/F", {})
            oos_wf = oos_r.get("walkforward", {}).get("OHLCV + OI/F", {})
            if is_wf and oos_wf:
                print(f"    Walk-forward OHLCV+OI/F:  IS Sharpe={is_wf.get('ls_sharpe', 0):+.2f}  "
                      f"OOS Sharpe={oos_wf.get('ls_sharpe', 0):+.2f}")

        if oos_r and oos_r.get("backtest"):
            bt = oos_r["backtest"].get("ls_top_zscore_24h", {})
            if bt:
                print(f"    Simple backtest (OOS):    avg={bt.get('avg_pnl', 0):+.1f}bps  "
                      f"wr={bt.get('wr', 0):.1f}%  trades={bt.get('trades', 0)}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
