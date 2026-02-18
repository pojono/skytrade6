#!/usr/bin/env python3
"""
Online Learning for LS Ratio Signal — v24d

Addresses the core problem from v24c: the LS ratio distribution is non-stationary
(mean doubled between May-Oct and Nov-Jan). Static models trained on one period
fail on the next.

Solution: Online learning with SGDRegressor.partial_fit() — each new bar updates
the model incrementally, so it continuously adapts to the current distribution.

Approach:
1. Stream bars chronologically across the full May 2025 - Jan 2026 period
2. At each bar t: predict fwd_ret_4h using current model
3. When bar t+48 arrives: observe actual return, update model via partial_fit()
4. Track predictions vs actuals — every prediction is genuinely out-of-sample

Also tests multiple learning rates, forgetting factors, and feature sets.
Also compares EWMA-based features (faster adaptation) vs fixed rolling windows.

Data: Binance metrics (LS ratios, OI, taker ratio) + Binance 5m klines.
"""

import sys
import time
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
SYMBOLS = ["BTCUSDT", "SOLUSDT"]

# Full continuous period
START_DATE = "2025-05-01"
END_DATE = "2026-01-31"

FEE_BPS = 7  # round-trip fee
FWD_HORIZON = 48  # 4h = 48 bars of 5m

# Online learning configs to test
CONFIGS = [
    {"name": "SGD_fast",    "alpha": 0.001, "lr": "invscaling", "eta0": 0.01,  "power_t": 0.25},
    {"name": "SGD_medium",  "alpha": 0.01,  "lr": "invscaling", "eta0": 0.005, "power_t": 0.25},
    {"name": "SGD_slow",    "alpha": 0.1,   "lr": "invscaling", "eta0": 0.001, "power_t": 0.25},
    {"name": "SGD_constant","alpha": 0.01,  "lr": "constant",   "eta0": 0.001, "power_t": 0.25},
    {"name": "SGD_adaptive","alpha": 0.01,  "lr": "adaptive",   "eta0": 0.005, "power_t": 0.25},
]

# ---------------------------------------------------------------------------
# Data loading (reused from v24c)
# ---------------------------------------------------------------------------

def load_binance_klines_5m(symbol, start_date, end_date):
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
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Klines: {len(df)} bars")
    return df


def load_binance_metrics(symbol, start_date, end_date):
    metrics_dir = PARQUET_DIR / symbol / "binance" / "metrics"
    dates = pd.date_range(start_date, end_date)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = metrics_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        print(f"  No metrics found for {symbol}!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Metrics: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Feature engineering — both fixed-window and EWMA variants
# ---------------------------------------------------------------------------

def build_features_dual(klines_df, metrics_df):
    """
    Build features with both fixed rolling windows AND EWMA variants.
    EWMA features adapt faster to distribution shifts.
    """
    merged = pd.merge_asof(
        klines_df.sort_values("timestamp_us"),
        metrics_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )
    n_matched = merged["open_interest"].notna().sum()
    print(f"  Merged: {len(merged)} bars, {n_matched} with metrics ({100*n_matched/len(merged):.1f}%)")

    # --- Fixed rolling features (same as v24c) ---
    for window, name in [(1, "5m"), (12, "1h"), (48, "4h"), (288, "24h")]:
        merged[f"oi_change_{name}"] = merged["open_interest"].pct_change(window) * 100

    merged["oi_accel_1h"] = merged["oi_change_1h"].diff(12)

    oi_mean = merged["open_interest"].rolling(288, min_periods=48).mean()
    oi_std = merged["open_interest"].rolling(288, min_periods=48).std()
    merged["oi_zscore_24h"] = (merged["open_interest"] - oi_mean) / oi_std.replace(0, np.nan)
    merged["oi_value_change_1h"] = merged["open_interest_value"].pct_change(12) * 100

    # LS ratio features (fixed)
    merged["ls_ratio_top"] = merged["top_trader_ls_ratio_accounts"]
    merged["ls_ratio_global"] = merged["global_ls_ratio"]
    merged["ls_top_change_1h"] = merged["top_trader_ls_ratio_accounts"].pct_change(12) * 100
    merged["ls_global_change_1h"] = merged["global_ls_ratio"].pct_change(12) * 100

    ls_mean = merged["top_trader_ls_ratio_accounts"].rolling(288, min_periods=48).mean()
    ls_std = merged["top_trader_ls_ratio_accounts"].rolling(288, min_periods=48).std()
    merged["ls_top_zscore_24h"] = (merged["top_trader_ls_ratio_accounts"] - ls_mean) / ls_std.replace(0, np.nan)

    ls_g_mean = merged["global_ls_ratio"].rolling(288, min_periods=48).mean()
    ls_g_std = merged["global_ls_ratio"].rolling(288, min_periods=48).std()
    merged["ls_global_zscore_24h"] = (merged["global_ls_ratio"] - ls_g_mean) / ls_g_std.replace(0, np.nan)

    # Taker features (fixed)
    merged["taker_ratio"] = merged["taker_buy_sell_ratio"]
    merged["taker_ratio_1h"] = merged["taker_buy_sell_ratio"].rolling(12, min_periods=3).mean()
    merged["taker_ratio_4h"] = merged["taker_buy_sell_ratio"].rolling(48, min_periods=12).mean()
    tk_mean = merged["taker_buy_sell_ratio"].rolling(288, min_periods=48).mean()
    tk_std = merged["taker_buy_sell_ratio"].rolling(288, min_periods=48).std()
    merged["taker_zscore_24h"] = (merged["taker_buy_sell_ratio"] - tk_mean) / tk_std.replace(0, np.nan)

    merged["oi_x_taker"] = merged["oi_change_1h"] * (merged["taker_buy_sell_ratio"] - 1)

    # --- EWMA features (faster adaptation) ---
    # EWMA z-scores with halflife = 4h (48 bars) instead of fixed 24h window
    hl = 48  # halflife in bars

    oi_ewm_mean = merged["open_interest"].ewm(halflife=hl, min_periods=24).mean()
    oi_ewm_std = merged["open_interest"].ewm(halflife=hl, min_periods=24).std()
    merged["oi_zscore_ewma"] = (merged["open_interest"] - oi_ewm_mean) / oi_ewm_std.replace(0, np.nan)

    ls_ewm_mean = merged["top_trader_ls_ratio_accounts"].ewm(halflife=hl, min_periods=24).mean()
    ls_ewm_std = merged["top_trader_ls_ratio_accounts"].ewm(halflife=hl, min_periods=24).std()
    merged["ls_top_zscore_ewma"] = (merged["top_trader_ls_ratio_accounts"] - ls_ewm_mean) / ls_ewm_std.replace(0, np.nan)

    ls_g_ewm_mean = merged["global_ls_ratio"].ewm(halflife=hl, min_periods=24).mean()
    ls_g_ewm_std = merged["global_ls_ratio"].ewm(halflife=hl, min_periods=24).std()
    merged["ls_global_zscore_ewma"] = (merged["global_ls_ratio"] - ls_g_ewm_mean) / ls_g_ewm_std.replace(0, np.nan)

    tk_ewm_mean = merged["taker_buy_sell_ratio"].ewm(halflife=hl, min_periods=24).mean()
    tk_ewm_std = merged["taker_buy_sell_ratio"].ewm(halflife=hl, min_periods=24).std()
    merged["taker_zscore_ewma"] = (merged["taker_buy_sell_ratio"] - tk_ewm_mean) / tk_ewm_std.replace(0, np.nan)

    # EWMA LS ratio change (exponentially weighted momentum)
    merged["ls_top_ewma_fast"] = merged["top_trader_ls_ratio_accounts"].ewm(halflife=12).mean()
    merged["ls_top_ewma_slow"] = merged["top_trader_ls_ratio_accounts"].ewm(halflife=96).mean()
    merged["ls_top_ewma_cross"] = merged["ls_top_ewma_fast"] - merged["ls_top_ewma_slow"]

    # --- OHLCV features ---
    merged["ret_1bar"] = merged["close"].pct_change(1) * 10000
    merged["rvol_1h"] = merged["ret_1bar"].rolling(12, min_periods=6).std()
    merged["rvol_4h"] = merged["ret_1bar"].rolling(48, min_periods=12).std()
    merged["rvol_24h"] = merged["ret_1bar"].rolling(288, min_periods=48).std()
    merged["momentum_1h"] = merged["close"].pct_change(12) * 10000
    merged["momentum_4h"] = merged["close"].pct_change(48) * 10000

    for w, name in [(12, "1h"), (48, "4h")]:
        net_move = (merged["close"] - merged["close"].shift(w)).abs()
        sum_moves = merged["ret_1bar"].abs().rolling(w, min_periods=w//2).sum()
        merged[f"efficiency_{name}"] = net_move / (sum_moves + 1e-10)

    merged["price_vs_sma_24h"] = (merged["close"] / merged["close"].rolling(288, min_periods=48).mean() - 1) * 100

    # EWMA vol
    merged["rvol_ewma_1h"] = merged["ret_1bar"].ewm(halflife=12, min_periods=6).std()
    merged["rvol_ewma_4h"] = merged["ret_1bar"].ewm(halflife=48, min_periods=12).std()

    # --- Forward returns ---
    merged["fwd_ret_4h"] = merged["close"].pct_change(FWD_HORIZON).shift(-FWD_HORIZON) * 10000

    return merged


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

FEAT_FIXED = [
    "oi_zscore_24h", "oi_value_change_1h",
    "ls_ratio_top", "ls_ratio_global",
    "ls_top_change_1h", "ls_global_change_1h",
    "ls_top_zscore_24h", "ls_global_zscore_24h",
    "taker_ratio", "taker_ratio_1h", "taker_ratio_4h", "taker_zscore_24h",
    "oi_x_taker",
    "rvol_1h", "rvol_4h", "rvol_24h",
    "momentum_1h", "momentum_4h",
    "efficiency_1h", "efficiency_4h",
    "price_vs_sma_24h",
]

FEAT_EWMA = [
    "oi_zscore_ewma",
    "ls_ratio_top", "ls_ratio_global",
    "ls_top_zscore_ewma", "ls_global_zscore_ewma",
    "ls_top_ewma_cross",
    "taker_ratio", "taker_zscore_ewma",
    "oi_x_taker",
    "rvol_ewma_1h", "rvol_ewma_4h",
    "momentum_1h", "momentum_4h",
    "efficiency_1h", "efficiency_4h",
    "price_vs_sma_24h",
]

FEAT_LS_ONLY = [
    "ls_ratio_top", "ls_ratio_global",
    "ls_top_zscore_ewma", "ls_global_zscore_ewma",
    "ls_top_ewma_cross",
]

FEATURE_SETS = {
    "fixed_all": FEAT_FIXED,
    "ewma_all": FEAT_EWMA,
    "ls_only_ewma": FEAT_LS_ONLY,
}


# ---------------------------------------------------------------------------
# Online Scaler — incremental mean/std tracking
# ---------------------------------------------------------------------------

class OnlineScaler:
    """Incremental StandardScaler using Welford's algorithm."""

    def __init__(self, n_features, decay=None):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)
        self.decay = decay  # if set, use exponential weighting

    def partial_fit(self, x):
        """Update stats with a single sample (1D array)."""
        self.n += 1
        if self.decay and self.n > 1:
            # Exponential decay: downweight old stats
            w = self.decay
            delta = x - self.mean
            self.mean = self.mean * (1 - w) + x * w
            self.M2 = self.M2 * (1 - w) + w * delta * (x - self.mean)
        else:
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def transform(self, x):
        """Scale a single sample."""
        if self.n < 2:
            return np.zeros_like(x)
        if self.decay:
            std = np.sqrt(self.M2 + 1e-10)
        else:
            std = np.sqrt(self.M2 / (self.n - 1) + 1e-10)
        return (x - self.mean) / std


# ---------------------------------------------------------------------------
# Online learning engine
# ---------------------------------------------------------------------------

def run_online_learning(df, feature_cols, config, label=""):
    """
    True online learning: stream bars, predict, then update when label arrives.

    At bar t:
      1. Extract features X_t
      2. Predict y_hat_t = model.predict(X_t)
      3. Store (X_t, t) in pending queue
      4. If bar t-48 label is now available: update model with partial_fit(X_{t-48}, y_{t-48})
      5. Record prediction for evaluation
    """
    name = config["name"]
    alpha = config["alpha"]
    lr = config["lr"]
    eta0 = config["eta0"]
    power_t = config["power_t"]

    n_features = len(feature_cols)

    # Initialize model
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=alpha,
        learning_rate=lr,
        eta0=eta0,
        power_t=power_t,
        warm_start=True,
        random_state=42,
    )

    # Online scaler with exponential decay
    scaler = OnlineScaler(n_features, decay=0.005)

    # Pending queue: (features, index) waiting for labels
    pending = deque()

    # Results storage
    predictions = np.full(len(df), np.nan)
    actuals = df["fwd_ret_4h"].values

    # Extract feature matrix
    X_all = df[feature_cols].values
    y_all = df["fwd_ret_4h"].values

    # Warmup: need some data before predictions are meaningful
    WARMUP = 288  # 24h of data for scaler warmup
    model_initialized = False
    n_updates = 0
    n_predictions = 0

    t0 = time.time()

    for t in range(len(df)):
        x_t = X_all[t]

        # Skip if features have NaN/inf
        if not np.all(np.isfinite(x_t)):
            continue

        # Update scaler
        scaler.partial_fit(x_t)

        if t < WARMUP:
            continue

        # Scale features
        x_scaled = scaler.transform(x_t).reshape(1, -1)

        # Predict (if model is initialized)
        if model_initialized:
            pred = model.predict(x_scaled)[0]
            predictions[t] = pred
            n_predictions += 1

        # Check if any pending labels are now available
        # Label for bar t-48 is the return from t-48 to t
        while pending and pending[0][1] + FWD_HORIZON <= t:
            x_old, t_old = pending.popleft()
            y_old = y_all[t_old]

            if np.isfinite(y_old):
                x_old_scaled = scaler.transform(x_old).reshape(1, -1)
                if not model_initialized:
                    # First fit
                    model.partial_fit(x_old_scaled, [y_old])
                    model_initialized = True
                else:
                    model.partial_fit(x_old_scaled, [y_old])
                n_updates += 1

        # Add current bar to pending queue
        pending.append((x_t.copy(), t))

        # Progress
        if t % 10000 == 0 and t > 0:
            elapsed = time.time() - t0
            pct = 100 * t / len(df)
            print(f"    [{name}] {pct:.0f}% ({t}/{len(df)}) updates={n_updates} preds={n_predictions} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"    [{name}] Done: {n_updates} updates, {n_predictions} predictions [{elapsed:.1f}s]")

    return predictions, actuals


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(predictions, actuals, df, config_name, feat_set_name, symbol):
    """Evaluate online predictions with IC, quintile analysis, and L/S backtest."""
    valid = np.isfinite(predictions) & np.isfinite(actuals)
    if valid.sum() < 500:
        print(f"    Not enough valid predictions ({valid.sum()})")
        return None

    pred = predictions[valid]
    actual = actuals[valid]
    n = len(pred)

    # IC
    ic = np.corrcoef(pred, actual)[0, 1]
    rank_ic = scipy_stats.spearmanr(pred, actual).correlation

    # Quintile analysis
    sorted_idx = np.argsort(pred)
    q_size = n // 5
    quintile_avgs = []
    for q in range(5):
        qs = q * q_size
        qe = (q + 1) * q_size if q < 4 else n
        q_avg = np.mean(actual[sorted_idx[qs:qe]])
        quintile_avgs.append(q_avg)

    # Long-short
    top_rets = actual[sorted_idx[-q_size:]] - FEE_BPS
    bot_rets = -actual[sorted_idx[:q_size]] - FEE_BPS
    ls_rets = np.concatenate([top_rets, bot_rets])
    ls_avg = np.mean(ls_rets)
    ls_wr = (ls_rets > 0).mean() * 100
    ls_sharpe = np.mean(ls_rets) / (np.std(ls_rets) + 1e-10) * np.sqrt(len(ls_rets))

    # Monthly breakdown
    ts = df["timestamp_us"].values[valid]
    dt = pd.to_datetime(ts, unit="us")
    months = dt.to_period("M")
    monthly_ics = {}
    for m in months.unique():
        mask = months == m
        if mask.sum() < 100:
            continue
        m_ic = np.corrcoef(pred[mask], actual[mask])[0, 1]
        monthly_ics[str(m)] = m_ic

    # Print results
    print(f"\n  {symbol} | {feat_set_name} | {config_name}")
    print(f"    IC={ic:+.4f}  rank_IC={rank_ic:+.4f}  n={n}")
    q_labels = ["Bottom", "Q2", "Q3", "Q4", "Top"]
    for i, (ql, qa) in enumerate(zip(q_labels, quintile_avgs)):
        print(f"    {ql:8s}: avg={qa:>+8.1f}bps")
    flag = "✅" if ls_avg > 0 else "  "
    print(f"  {flag} L/S: avg={ls_avg:>+.1f}bps  wr={ls_wr:.1f}%  sharpe={ls_sharpe:+.2f}")

    # Monthly
    print(f"    Monthly IC:")
    for m, mic in sorted(monthly_ics.items()):
        flag_m = "✅" if mic > 0.02 else "  "
        print(f"    {flag_m} {m}: IC={mic:+.4f}")

    # Monotonicity check
    monotonic = all(quintile_avgs[i] <= quintile_avgs[i+1] for i in range(4))
    print(f"    Quintile monotonic: {'YES ✅' if monotonic else 'NO'}")

    return {
        "ic": ic, "rank_ic": rank_ic, "n": n,
        "ls_avg": ls_avg, "ls_wr": ls_wr, "ls_sharpe": ls_sharpe,
        "quintile_avgs": quintile_avgs, "monthly_ics": monthly_ics,
        "monotonic": monotonic,
    }


def evaluate_by_period(predictions, actuals, df, config_name, feat_set_name, symbol):
    """Split evaluation into OOS (May-Oct) and IS (Nov-Jan) sub-periods."""
    ts = df["timestamp_us"].values
    # OOS: May-Oct 2025
    oos_cutoff = int(pd.Timestamp("2025-11-01").timestamp() * 1e6)
    oos_mask = ts < oos_cutoff
    is_mask = ts >= oos_cutoff

    results = {}
    for period_name, mask in [("OOS (May-Oct)", oos_mask), ("IS (Nov-Jan)", is_mask)]:
        p = predictions.copy()
        a = actuals.copy()
        p[~mask] = np.nan
        valid = np.isfinite(p) & np.isfinite(a)
        if valid.sum() < 200:
            print(f"    {period_name}: not enough data ({valid.sum()})")
            continue

        pred = p[valid]
        actual = a[valid]
        n = len(pred)

        ic = np.corrcoef(pred, actual)[0, 1]
        rank_ic = scipy_stats.spearmanr(pred, actual).correlation

        sorted_idx = np.argsort(pred)
        q_size = n // 5
        top_rets = actual[sorted_idx[-q_size:]] - FEE_BPS
        bot_rets = -actual[sorted_idx[:q_size]] - FEE_BPS
        ls_rets = np.concatenate([top_rets, bot_rets])
        ls_avg = np.mean(ls_rets)
        ls_sharpe = np.mean(ls_rets) / (np.std(ls_rets) + 1e-10) * np.sqrt(len(ls_rets))

        flag = "✅" if ls_avg > 0 else "  "
        print(f"  {flag} {period_name}: IC={ic:+.4f}  L/S avg={ls_avg:+.1f}bps  sharpe={ls_sharpe:+.2f}  n={n}")
        results[period_name] = {"ic": ic, "rank_ic": rank_ic, "ls_avg": ls_avg, "ls_sharpe": ls_sharpe, "n": n}

    return results


# ---------------------------------------------------------------------------
# Simple online signal (no ML, just adaptive z-score threshold)
# ---------------------------------------------------------------------------

def run_adaptive_signal(df, symbol):
    """
    Non-ML online signal: use EWMA z-score of LS ratio with adaptive threshold.
    Trade when z-score exceeds rolling percentile threshold.
    """
    print(f"\n{'='*70}")
    print(f"  ADAPTIVE SIGNAL (non-ML) — {symbol}")
    print(f"{'='*70}")

    ls = df["top_trader_ls_ratio_accounts"].values
    fwd = df["fwd_ret_4h"].values
    ts = df["timestamp_us"].values

    # EWMA z-score with different halflifes
    for hl_name, hl in [("4h", 48), ("8h", 96), ("24h", 288)]:
        ewm_mean = df["top_trader_ls_ratio_accounts"].ewm(halflife=hl, min_periods=hl//2).mean().values
        ewm_std = df["top_trader_ls_ratio_accounts"].ewm(halflife=hl, min_periods=hl//2).std().values
        z = (ls - ewm_mean) / (ewm_std + 1e-10)

        # Momentum: z > 1 → long, z < -1 → short
        for thresh in [0.5, 1.0, 1.5, 2.0]:
            long_mask = (z > thresh) & np.isfinite(fwd)
            short_mask = (z < -thresh) & np.isfinite(fwd)
            trade_mask = long_mask | short_mask

            if trade_mask.sum() < 100:
                continue

            # Momentum direction
            trade_rets_mom = np.where(
                long_mask[trade_mask], fwd[trade_mask], -fwd[trade_mask]
            ) - FEE_BPS

            n = trade_mask.sum()
            avg = np.mean(trade_rets_mom)
            wr = (trade_rets_mom > 0).mean() * 100

            # Split by period
            oos_cutoff = int(pd.Timestamp("2025-11-01").timestamp() * 1e6)
            oos_trade = trade_mask & (ts < oos_cutoff)
            is_trade = trade_mask & (ts >= oos_cutoff)

            oos_rets = np.where(
                long_mask[oos_trade], fwd[oos_trade], -fwd[oos_trade]
            ) - FEE_BPS if oos_trade.sum() > 10 else np.array([])

            is_rets = np.where(
                long_mask[is_trade], fwd[is_trade], -fwd[is_trade]
            ) - FEE_BPS if is_trade.sum() > 10 else np.array([])

            oos_avg = np.mean(oos_rets) if len(oos_rets) > 0 else float("nan")
            is_avg = np.mean(is_rets) if len(is_rets) > 0 else float("nan")

            flag = "✅" if avg > 0 else "  "
            print(f"  {flag} EWMA({hl_name}) z>{thresh:.1f} mom: "
                  f"avg={avg:+.1f}bps wr={wr:.1f}% n={n}  "
                  f"OOS={oos_avg:+.1f}bps IS={is_avg:+.1f}bps")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  ONLINE LEARNING — LS RATIO SIGNAL")
    print(f"  Period: {START_DATE} to {END_DATE} (continuous stream)")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  SGD configs: {len(CONFIGS)}")
    print(f"  Feature sets: {list(FEATURE_SETS.keys())}")
    print(f"{'='*70}")

    all_results = {}

    for symbol in SYMBOLS:
        print(f"\n{'#'*70}")
        print(f"  {symbol}")
        print(f"{'#'*70}")

        # Load data
        print(f"\nLoading data...")
        klines = load_binance_klines_5m(symbol, START_DATE, END_DATE)
        metrics = load_binance_metrics(symbol, START_DATE, END_DATE)

        if klines.empty or metrics.empty:
            print(f"  SKIPPING — missing data")
            continue

        # Build features
        print(f"\nBuilding features...")
        df = build_features_dual(klines, metrics)

        n_total = len(df)
        n_ls = df["ls_ratio_top"].notna().sum()
        print(f"  Total bars: {n_total}, with LS ratio: {n_ls} ({100*n_ls/n_total:.1f}%)")

        # LS ratio stats over time
        print(f"\n  LS ratio distribution over time:")
        df["_dt"] = pd.to_datetime(df["timestamp_us"], unit="us")
        df["_month"] = df["_dt"].dt.to_period("M")
        for m, grp in df.groupby("_month"):
            ls_vals = grp["ls_ratio_top"].dropna()
            if len(ls_vals) > 0:
                print(f"    {m}: mean={ls_vals.mean():.3f} std={ls_vals.std():.3f} n={len(ls_vals)}")

        # --- Run adaptive signal (non-ML baseline) ---
        run_adaptive_signal(df, symbol)

        # --- Run online learning experiments ---
        symbol_results = {}
        for feat_name, feat_cols in FEATURE_SETS.items():
            available = [c for c in feat_cols if c in df.columns]
            if len(available) < len(feat_cols):
                missing = set(feat_cols) - set(available)
                print(f"\n  WARNING: Missing features for {feat_name}: {missing}")
            feat_cols_use = available

            for config in CONFIGS:
                print(f"\n  Running: {symbol} | {feat_name} | {config['name']}")
                predictions, actuals = run_online_learning(df, feat_cols_use, config, label=f"{symbol}_{feat_name}")

                # Full period evaluation
                result = evaluate_predictions(predictions, actuals, df, config["name"], feat_name, symbol)

                # Period split
                period_results = evaluate_by_period(predictions, actuals, df, config["name"], feat_name, symbol)

                key = f"{feat_name}_{config['name']}"
                symbol_results[key] = {
                    "full": result,
                    "periods": period_results,
                }

        all_results[symbol] = symbol_results

    # ---------------------------------------------------------------------------
    # Final comparison table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Symbol':8s} {'Features':14s} {'Config':14s} "
          f"{'Full IC':>8s} {'Full Sharpe':>12s} "
          f"{'OOS IC':>8s} {'OOS Sharpe':>11s} "
          f"{'IS IC':>7s} {'IS Sharpe':>10s}")
    print(f"  {'-'*98}")

    for symbol in SYMBOLS:
        if symbol not in all_results:
            continue
        for key, res in sorted(all_results[symbol].items()):
            parts = key.rsplit("_", 1)
            feat_name = parts[0] if len(parts) > 1 else key
            config_name = parts[-1] if len(parts) > 1 else ""

            full = res.get("full", {}) or {}
            periods = res.get("periods", {}) or {}
            oos = periods.get("OOS (May-Oct)", {})
            is_ = periods.get("IS (Nov-Jan)", {})

            full_ic = full.get("ic", float("nan"))
            full_sh = full.get("ls_sharpe", float("nan"))
            oos_ic = oos.get("ic", float("nan"))
            oos_sh = oos.get("ls_sharpe", float("nan"))
            is_ic = is_.get("ic", float("nan"))
            is_sh = is_.get("ls_sharpe", float("nan"))

            flag = "✅" if full_sh > 0 and oos_sh > 0 else "  "
            # Parse feat_name and config_name from key properly
            key_parts = key.split("_SGD_")
            fn = key_parts[0] if len(key_parts) > 0 else key
            cn = "SGD_" + key_parts[1] if len(key_parts) > 1 else ""

            print(f"{flag} {symbol:8s} {fn:14s} {cn:14s} "
                  f"{full_ic:>+8.4f} {full_sh:>+12.2f} "
                  f"{oos_ic:>+8.4f} {oos_sh:>+11.2f} "
                  f"{is_ic:>+7.4f} {is_sh:>+10.2f}")

    # Compare to v24c static results
    print(f"\n  COMPARISON TO v24c STATIC WALK-FORWARD:")
    print(f"  (v24c used Ridge with expanding window, retrained every 4h)")
    print(f"  BTC v24c: OOS IC=-0.018, OOS Sharpe=-12.40 | IS IC=-0.019, IS Sharpe=-5.71")
    print(f"  SOL v24c: OOS IC=+0.032, OOS Sharpe=+0.55  | IS IC=+0.077, IS Sharpe=+2.31")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
