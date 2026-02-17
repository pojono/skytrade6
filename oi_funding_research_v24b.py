#!/usr/bin/env python3
"""
OI/Funding Research v24b — Out-of-Sample Validation + Sub-5min OI Spikes

Part A: Validate LS ratio directional signal on May-Aug 2025 (Bybit ticker data)
        - Bybit ticker has OI + funding at 5-second resolution
        - No LS ratios (Binance-specific) — but we can validate OI/funding signals
        - Build OHLCV from Bybit trades, merge with ticker OI/funding

Part B: Engineer sub-5min OI spike features from 5-second ticker data
        - OI velocity (change per second)
        - OI spike detection (sudden jumps)
        - Funding rate micro-dynamics
        - Bid/ask size dynamics

Data: Bybit ticker (5s) + Bybit trades → 5-min bars, May 12 - Aug 8 2025
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, roc_auc_score
from scipy import stats as scipy_stats

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
SYMBOL = "BTCUSDT"
SOURCE = "bybit_futures"
# Use full clean date range (skip partial first/last days)
START_DATE = "2025-05-12"
END_DATE = "2025-08-08"

INTERVAL_5M_US = 300_000_000  # 5 min in microseconds

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

FEE_BPS = 7


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ohlcv_bars():
    """Load 5-min OHLCV bars with microstructure features for May-Aug 2025."""
    from regime_detection import load_bars, compute_regime_features
    print("Loading OHLCV bars (from Bybit trades)...")
    df = load_bars(SYMBOL, START_DATE, END_DATE)
    if df.empty:
        print("  ERROR: No OHLCV bars found!")
        return df
    df = compute_regime_features(df)
    print(f"  OHLCV: {len(df)} bars, {df.columns.size} cols")
    return df


def load_ticker_data():
    """Load parsed Bybit ticker data (5-second resolution)."""
    ticker_dir = PARQUET_DIR / SYMBOL / "ticker"
    dates = pd.date_range(START_DATE, END_DATE)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = ticker_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        print("  No ticker data found!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Ticker: {len(df)} rows ({len(dfs)} days), "
          f"OI range [{df['open_interest'].min():.0f}, {df['open_interest'].max():.0f}]")
    return df


def build_ticker_5m_features(ticker_df):
    """
    Aggregate 5-second ticker data into 5-minute features.

    Standard features (comparable to Binance metrics):
      - OI level, OI change, OI z-score
      - Funding rate, funding z-score
      - Mark-index spread (basis proxy)

    Sub-5min features (NEW — only possible with 5s data):
      - OI velocity (max change rate within 5-min bar)
      - OI spike count (number of >0.1% jumps in 5s)
      - OI acceleration (change in velocity)
      - Funding rate volatility within bar
      - Bid/ask size dynamics
    """
    print("Building 5-min ticker features...")
    t0 = time.time()

    df = ticker_df.copy()
    df["bucket"] = (df["timestamp_us"].values // INTERVAL_5M_US) * INTERVAL_5M_US

    # Pre-compute per-row features
    df["oi_pct_change"] = df["open_interest"].pct_change() * 100
    df["mark_index_spread"] = (df["mark_price"] - df["index_price"]) / df["index_price"] * 10000  # bps
    df["bid_ask_spread_bps"] = (df["ask1_price"] - df["bid1_price"]) / df["bid1_price"] * 10000
    df["bid_ask_size_ratio"] = df["bid1_size"] / (df["ask1_size"] + 1e-10)

    features = []
    groups = df.groupby("bucket")
    n_groups = len(groups)

    for i, (bkt, grp) in enumerate(groups):
        if len(grp) < 10:  # need at least ~50s of data
            continue

        oi = grp["open_interest"].values
        fr = grp["funding_rate"].values
        price = grp["last_price"].values
        oi_pct = grp["oi_pct_change"].values
        mis = grp["mark_index_spread"].values
        bas = grp["bid_ask_spread_bps"].values
        ba_ratio = grp["bid_ask_size_ratio"].values

        # --- Standard 5-min aggregates ---
        oi_start, oi_end = oi[0], oi[-1]
        oi_change_5m = (oi_end - oi_start) / max(oi_start, 1) * 100
        funding_mean = np.mean(fr)
        mis_mean = np.mean(mis)

        # --- Sub-5min OI dynamics (NEW) ---
        # OI velocity: max absolute change rate (pct per 5s tick)
        oi_pct_clean = oi_pct[~np.isnan(oi_pct)]
        if len(oi_pct_clean) > 5:
            oi_velocity_max = np.max(np.abs(oi_pct_clean))
            oi_velocity_mean = np.mean(np.abs(oi_pct_clean))
            oi_velocity_std = np.std(oi_pct_clean)

            # OI spike count: number of >0.05% jumps in 5 seconds
            oi_spike_count = np.sum(np.abs(oi_pct_clean) > 0.05)
            # Large spike count: >0.1%
            oi_large_spike_count = np.sum(np.abs(oi_pct_clean) > 0.10)

            # OI direction within bar: net positive or negative changes
            oi_up_count = np.sum(oi_pct_clean > 0.01)
            oi_down_count = np.sum(oi_pct_clean < -0.01)
            oi_direction = (oi_up_count - oi_down_count) / max(len(oi_pct_clean), 1)

            # OI acceleration: velocity in second half vs first half
            half = len(oi_pct_clean) // 2
            v1 = np.mean(np.abs(oi_pct_clean[:half]))
            v2 = np.mean(np.abs(oi_pct_clean[half:]))
            oi_accel_intrabar = v2 - v1
        else:
            oi_velocity_max = 0
            oi_velocity_mean = 0
            oi_velocity_std = 0
            oi_spike_count = 0
            oi_large_spike_count = 0
            oi_direction = 0
            oi_accel_intrabar = 0

        # --- Sub-5min funding dynamics ---
        funding_std = np.std(fr) if len(fr) > 5 else 0
        funding_range = np.max(fr) - np.min(fr) if len(fr) > 1 else 0

        # --- Sub-5min mark-index spread dynamics ---
        mis_std = np.std(mis) if len(mis) > 5 else 0
        mis_max_abs = np.max(np.abs(mis)) if len(mis) > 0 else 0

        # --- Sub-5min bid/ask dynamics ---
        bas_mean = np.mean(bas)
        bas_std = np.std(bas) if len(bas) > 5 else 0
        ba_ratio_mean = np.mean(ba_ratio)
        ba_ratio_std = np.std(ba_ratio) if len(ba_ratio) > 5 else 0

        features.append({
            "timestamp_us": bkt,
            # Standard
            "tk_oi": oi_end,
            "tk_oi_change_5m": oi_change_5m,
            "tk_funding_rate": funding_mean,
            "tk_mark_index_spread": mis_mean,
            # Sub-5min OI dynamics
            "tk_oi_velocity_max": oi_velocity_max,
            "tk_oi_velocity_mean": oi_velocity_mean,
            "tk_oi_velocity_std": oi_velocity_std,
            "tk_oi_spike_count": oi_spike_count,
            "tk_oi_large_spike_count": oi_large_spike_count,
            "tk_oi_direction": oi_direction,
            "tk_oi_accel_intrabar": oi_accel_intrabar,
            # Sub-5min funding
            "tk_funding_std": funding_std,
            "tk_funding_range": funding_range,
            # Sub-5min basis
            "tk_mis_std": mis_std,
            "tk_mis_max_abs": mis_max_abs,
            # Sub-5min microstructure
            "tk_spread_mean": bas_mean,
            "tk_spread_std": bas_std,
            "tk_ba_ratio_mean": ba_ratio_mean,
            "tk_ba_ratio_std": ba_ratio_std,
        })

        if (i + 1) % 5000 == 0:
            print(f"    [{i+1}/{n_groups}] bars processed...", flush=True)

    feat_df = pd.DataFrame(features)
    print(f"  Built {len(feat_df)} 5-min bars with {len(feat_df.columns)-1} ticker features "
          f"({time.time()-t0:.0f}s)")
    return feat_df


def build_rolling_features(feat_df):
    """Add rolling features on top of the 5-min ticker features."""
    df = feat_df.copy()

    # OI rolling features (same as v24 but from Bybit data)
    for window, name in [(12, "1h"), (48, "4h"), (288, "24h")]:
        df[f"tk_oi_change_{name}"] = df["tk_oi"].pct_change(window) * 100

    # OI z-score
    oi_mean = df["tk_oi"].rolling(288, min_periods=48).mean()
    oi_std = df["tk_oi"].rolling(288, min_periods=48).std()
    df["tk_oi_zscore_24h"] = (df["tk_oi"] - oi_mean) / oi_std.replace(0, np.nan)

    # OI acceleration
    df["tk_oi_accel_1h"] = df["tk_oi_change_1h"].diff(12)

    # Funding rolling
    df["tk_funding_abs"] = df["tk_funding_rate"].abs()
    fr_mean = df["tk_funding_rate"].rolling(288, min_periods=48).mean()
    fr_std = df["tk_funding_rate"].rolling(288, min_periods=48).std()
    df["tk_funding_zscore_24h"] = (df["tk_funding_rate"] - fr_mean) / fr_std.replace(0, np.nan)
    df["tk_funding_cum_8h"] = df["tk_funding_rate"].rolling(96, min_periods=12).sum()

    # Mark-index spread rolling
    df["tk_mis_1h"] = df["tk_mark_index_spread"].rolling(12, min_periods=3).mean()
    df["tk_mis_4h"] = df["tk_mark_index_spread"].rolling(48, min_periods=12).mean()
    mis_mean = df["tk_mark_index_spread"].rolling(288, min_periods=48).mean()
    mis_std = df["tk_mark_index_spread"].rolling(288, min_periods=48).std()
    df["tk_mis_zscore_24h"] = (df["tk_mark_index_spread"] - mis_mean) / mis_std.replace(0, np.nan)

    # OI spike rolling (sum of spikes over windows)
    df["tk_oi_spikes_1h"] = df["tk_oi_spike_count"].rolling(12, min_periods=3).sum()
    df["tk_oi_spikes_4h"] = df["tk_oi_spike_count"].rolling(48, min_periods=12).sum()
    df["tk_oi_large_spikes_1h"] = df["tk_oi_large_spike_count"].rolling(12, min_periods=3).sum()

    # OI velocity rolling
    df["tk_oi_vel_max_1h"] = df["tk_oi_velocity_max"].rolling(12, min_periods=3).max()
    df["tk_oi_vel_mean_1h"] = df["tk_oi_velocity_mean"].rolling(12, min_periods=3).mean()
    df["tk_oi_vel_mean_4h"] = df["tk_oi_velocity_mean"].rolling(48, min_periods=12).mean()

    # Bid/ask ratio rolling
    df["tk_ba_ratio_1h"] = df["tk_ba_ratio_mean"].rolling(12, min_periods=3).mean()

    print(f"  Rolling features: {len(df.columns)} total columns")
    return df


# Feature lists
TICKER_STANDARD = [
    "tk_oi_change_5m", "tk_oi_change_1h", "tk_oi_change_4h", "tk_oi_change_24h",
    "tk_oi_zscore_24h", "tk_oi_accel_1h",
    "tk_funding_rate", "tk_funding_abs", "tk_funding_zscore_24h", "tk_funding_cum_8h",
    "tk_mark_index_spread", "tk_mis_1h", "tk_mis_4h", "tk_mis_zscore_24h",
]

TICKER_SUB5MIN = [
    "tk_oi_velocity_max", "tk_oi_velocity_mean", "tk_oi_velocity_std",
    "tk_oi_spike_count", "tk_oi_large_spike_count",
    "tk_oi_direction", "tk_oi_accel_intrabar",
    "tk_funding_std", "tk_funding_range",
    "tk_mis_std", "tk_mis_max_abs",
    "tk_spread_mean", "tk_spread_std",
    "tk_ba_ratio_mean", "tk_ba_ratio_std",
    "tk_oi_spikes_1h", "tk_oi_spikes_4h", "tk_oi_large_spikes_1h",
    "tk_oi_vel_max_1h", "tk_oi_vel_mean_1h", "tk_oi_vel_mean_4h",
    "tk_ba_ratio_1h",
]

ALL_TICKER_FEATURES = TICKER_STANDARD + TICKER_SUB5MIN


def merge_ohlcv_ticker(ohlcv_df, ticker_df):
    """Merge OHLCV bars with ticker features on timestamp_us."""
    if "timestamp_us" not in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df["timestamp_us"] = ohlcv_df.index.astype(np.int64) // 1000

    merged = pd.merge_asof(
        ohlcv_df.sort_values("timestamp_us"),
        ticker_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )

    tk_matched = merged["tk_oi"].notna().sum()
    print(f"  Merged: {len(merged)} bars, {tk_matched} with ticker data ({tk_matched/len(merged)*100:.1f}%)")

    # Forward returns
    merged["fwd_ret_5m"] = merged["close"].pct_change(1).shift(-1) * 10000
    merged["fwd_ret_15m"] = merged["close"].pct_change(3).shift(-3) * 10000
    merged["fwd_ret_1h"] = merged["close"].pct_change(12).shift(-12) * 10000
    merged["fwd_ret_4h"] = merged["close"].pct_change(48).shift(-48) * 10000
    merged["fwd_rvol_1h"] = merged["close"].pct_change(1).abs().rolling(12).sum().shift(-12) * 10000

    return merged


# ---------------------------------------------------------------------------
# Part A: Validate OI/Funding Signals (Out-of-Sample)
# ---------------------------------------------------------------------------

def part_a_validate_signals(df):
    """Validate OI/funding directional signals from v24 on May-Aug data."""
    print(f"\n{'='*70}")
    print(f"  PART A: OUT-OF-SAMPLE VALIDATION (May-Aug 2025)")
    print(f"  Validating v24 findings using Bybit ticker data")
    print(f"{'='*70}")

    # Note: We don't have LS ratios from Bybit — that's Binance-specific
    # But we can validate: OI signals, funding signals, mark-index spread

    tk_cols = [c for c in TICKER_STANDARD if c in df.columns]

    # --- IC Analysis ---
    horizons = {
        "5min": "fwd_ret_5m", "15min": "fwd_ret_15m",
        "1h": "fwd_ret_1h", "4h": "fwd_ret_4h",
    }

    print(f"\n  Information Coefficient (IC) — Bybit ticker features vs forward returns:")
    print(f"  {'Feature':35s} {'5min':>8s} {'15min':>8s} {'1h':>8s} {'4h':>8s}")
    print(f"  {'-'*67}")

    best_per_horizon = {}
    for col in tk_cols:
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
    for h_name in ["5min", "15min", "1h", "4h"]:
        if h_name in best_per_horizon:
            col, ic = best_per_horizon[h_name]
            print(f"    {h_name:>6s}: {col:35s} IC={ic:+.4f}")

    # --- OI Extreme Contrarian (replicating v24 Exp 5) ---
    print(f"\n  OI Z-Score Extreme Contrarian (replicating v24 finding):")
    col = "tk_oi_zscore_24h"
    if col in df.columns:
        valid = df[[col, "fwd_ret_4h"]].notna().all(axis=1)
        sub = df[valid]
        z = sub[col].values
        rets = sub["fwd_ret_4h"].values

        print(f"    {'Condition':25s} {'N':>6s} {'Avg Ret':>10s} {'WR':>8s} {'Sharpe':>8s}")
        print(f"    {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")

        for thresh, name in [
            (2.0, "z > +2.0 → short"),
            (1.5, "z > +1.5 → short"),
            (1.0, "z > +1.0 → short"),
            (-1.0, "z < -1.0 → long"),
            (-1.5, "z < -1.5 → long"),
            (-2.0, "z < -2.0 → long"),
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

    # --- Funding Extreme Contrarian ---
    print(f"\n  Funding Z-Score Extreme Contrarian:")
    col = "tk_funding_zscore_24h"
    if col in df.columns:
        valid = df[[col, "fwd_ret_4h"]].notna().all(axis=1)
        sub = df[valid]
        z = sub[col].values
        rets = sub["fwd_ret_4h"].values

        print(f"    {'Condition':25s} {'N':>6s} {'Avg Ret':>10s} {'WR':>8s} {'Sharpe':>8s}")
        print(f"    {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")

        for thresh, name in [
            (2.0, "z > +2.0 → short"),
            (1.5, "z > +1.5 → short"),
            (1.0, "z > +1.0 → short"),
            (-1.0, "z < -1.0 → long"),
            (-1.5, "z < -1.5 → long"),
            (-2.0, "z < -2.0 → long"),
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

    # --- Mark-Index Spread (Basis) Signal ---
    print(f"\n  Mark-Index Spread Signals:")
    for col in ["tk_mark_index_spread", "tk_mis_1h", "tk_mis_4h", "tk_mis_zscore_24h"]:
        if col not in df.columns:
            continue
        valid = df[[col, "fwd_ret_4h"]].notna().all(axis=1)
        if valid.sum() < 200:
            continue
        sub = df[valid]
        vals = sub[col].values
        rets = sub["fwd_ret_4h"].values

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
        flag = "✅" if avg_pnl > 0 else "  "
        print(f"  {flag} {col:35s}: trades={n_trades:>4d}, avg={avg_pnl:>+.1f}bps, wr={wr:.1f}%")

    # --- Walk-Forward (OHLCV + Ticker Standard) ---
    print(f"\n  Walk-Forward Quintile L/S (4h returns):")
    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]

    feature_combos = {
        "OHLCV only": ohlcv_cols,
        "OHLCV + Ticker Std": ohlcv_cols + [c for c in TICKER_STANDARD if c in df.columns],
    }

    target = "fwd_ret_4h"
    for name, cols in feature_combos.items():
        valid = df[cols + [target]].notna().all(axis=1)
        sub = df[valid].reset_index(drop=True)
        X = sub[cols].values
        y = sub[target].values

        if len(X) < 500:
            print(f"  {name}: not enough data ({len(X)} rows)")
            continue

        train_size = len(X) // 3
        preds = np.full(len(X), np.nan)
        step = 48

        for start in range(train_size, len(X), step):
            end = min(start + step, len(X))
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[:start])
            X_test = scaler.transform(X[start:end])
            ridge = Ridge(alpha=10)
            ridge.fit(X_train, y[:start])
            preds[start:end] = ridge.predict(X_test)

        valid_pred = ~np.isnan(preds)
        if valid_pred.sum() < 100:
            continue

        pred_vals = preds[valid_pred]
        actual_vals = y[valid_pred]
        ic = np.corrcoef(pred_vals, actual_vals)[0, 1]
        rank_ic = scipy_stats.spearmanr(pred_vals, actual_vals).correlation

        print(f"\n  {name} ({len(cols)} features):")
        print(f"    IC={ic:+.4f}  rank_IC={rank_ic:+.4f}  n={valid_pred.sum()}")

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

        top_rets = actual_vals[sorted_idx[-q_size:]] - FEE_BPS
        bot_rets = -actual_vals[sorted_idx[:q_size]] - FEE_BPS
        ls_rets = np.concatenate([top_rets, bot_rets])
        ls_avg = np.mean(ls_rets)
        ls_wr = (ls_rets > 0).mean() * 100
        ls_sharpe = np.mean(ls_rets) / (np.std(ls_rets) + 1e-10) * np.sqrt(len(ls_rets))
        flag = "✅" if ls_avg > 0 else "  "
        print(f"  {flag} Long-short (Q5-Q1): avg={ls_avg:>+.1f}bps  wr={ls_wr:.1f}%  sharpe={ls_sharpe:+.2f}")


# ---------------------------------------------------------------------------
# Part B: Sub-5min OI Spike Features
# ---------------------------------------------------------------------------

def part_b_oi_spike_research(df):
    """Test if sub-5min OI spike features add predictive value."""
    print(f"\n{'='*70}")
    print(f"  PART B: SUB-5MIN OI SPIKE FEATURES")
    print(f"  Testing if 5-second OI dynamics add value over 5-min aggregates")
    print(f"{'='*70}")

    sub5_cols = [c for c in TICKER_SUB5MIN if c in df.columns]
    std_cols = [c for c in TICKER_STANDARD if c in df.columns]
    ohlcv_cols = [c for c in OHLCV_FEATURES if c in df.columns]

    # --- IC Analysis for sub-5min features ---
    horizons = {
        "5min": "fwd_ret_5m", "15min": "fwd_ret_15m",
        "1h": "fwd_ret_1h", "4h": "fwd_ret_4h",
    }

    print(f"\n  Information Coefficient — Sub-5min features:")
    print(f"  {'Feature':35s} {'5min':>8s} {'15min':>8s} {'1h':>8s} {'4h':>8s}")
    print(f"  {'-'*67}")

    for col in sub5_cols:
        ics = []
        for h_name, h_col in horizons.items():
            valid = df[[col, h_col]].notna().all(axis=1)
            if valid.sum() < 100:
                ics.append(np.nan)
                continue
            ic = df.loc[valid, col].corr(df.loc[valid, h_col])
            ics.append(ic)
        print(f"  {col:35s} {ics[0]:>+8.4f} {ics[1]:>+8.4f} {ics[2]:>+8.4f} {ics[3]:>+8.4f}")

    # --- Regime Profiling ---
    print(f"\n  Sub-5min features by regime:")
    valid_regime = df[ohlcv_cols].notna().all(axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.loc[valid_regime, ohlcv_cols])

    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                           n_init=10, random_state=42, max_iter=300)
    labels_raw = gmm.fit_predict(X_scaled)

    rvol_col = "rvol_1h" if "rvol_1h" in df.columns else ohlcv_cols[0]
    if np.mean(df.loc[valid_regime, rvol_col].values[labels_raw == 0]) > \
       np.mean(df.loc[valid_regime, rvol_col].values[labels_raw == 1]):
        labels_raw = 1 - labels_raw

    labels = np.full(len(df), -1, dtype=np.int8)
    labels[valid_regime.values] = labels_raw

    print(f"  {'Feature':35s} {'Quiet':>12s} {'Volatile':>12s} {'Ratio':>8s} {'T-stat':>8s} {'P-val':>10s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")

    significant = []
    for col in sub5_cols:
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
            significant.append((col, abs(t_stat)))

    print(f"\n  Significant sub-5min features: {len(significant)}/{len(sub5_cols)}")

    # --- Vol Prediction: Does sub-5min help? ---
    print(f"\n  Volatility Prediction — sub-5min features:")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    target = "fwd_rvol_1h"

    feature_sets = {
        "OHLCV only": ohlcv_cols,
        "OHLCV + Ticker Std": ohlcv_cols + std_cols,
        "OHLCV + Ticker All": ohlcv_cols + std_cols + sub5_cols,
        "OHLCV + Sub-5min only": ohlcv_cols + sub5_cols,
    }

    for name, cols in feature_sets.items():
        valid = df[cols + [target]].notna().all(axis=1)
        X = df.loc[valid, cols].values
        y = df.loc[valid, target].values
        if len(X) < 200:
            continue

        ridge_r2s, gb_r2s = [], []
        for train_idx, test_idx in tscv.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            ridge = Ridge(alpha=10)
            ridge.fit(X_train, y_train)
            ridge_r2s.append(r2_score(y_test, ridge.predict(X_test)))

            gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            gb.fit(X_train, y_train)
            gb_r2s.append(r2_score(y_test, gb.predict(X_test)))

        print(f"    {name:30s}: Ridge R²={np.mean(ridge_r2s):.4f}  GB R²={np.mean(gb_r2s):.4f}")

    # --- Direction Prediction: Does sub-5min help? ---
    print(f"\n  Walk-Forward Direction (4h) — sub-5min features:")
    target = "fwd_ret_4h"

    feature_combos = {
        "OHLCV only": ohlcv_cols,
        "OHLCV + Ticker Std": ohlcv_cols + std_cols,
        "OHLCV + Ticker All": ohlcv_cols + std_cols + sub5_cols,
        "OHLCV + Sub-5min only": ohlcv_cols + sub5_cols,
    }

    for name, cols in feature_combos.items():
        valid = df[cols + [target]].notna().all(axis=1)
        sub = df[valid].reset_index(drop=True)
        X = sub[cols].values
        y = sub[target].values

        if len(X) < 500:
            continue

        train_size = len(X) // 3
        preds = np.full(len(X), np.nan)
        step = 48

        for start in range(train_size, len(X), step):
            end = min(start + step, len(X))
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[:start])
            X_test = scaler.transform(X[start:end])
            ridge = Ridge(alpha=10)
            ridge.fit(X_train, y[:start])
            preds[start:end] = ridge.predict(X_test)

        valid_pred = ~np.isnan(preds)
        if valid_pred.sum() < 100:
            continue

        pred_vals = preds[valid_pred]
        actual_vals = y[valid_pred]
        ic = np.corrcoef(pred_vals, actual_vals)[0, 1]

        n_valid = valid_pred.sum()
        sorted_idx = np.argsort(pred_vals)
        q_size = n_valid // 5

        top_rets = actual_vals[sorted_idx[-q_size:]] - FEE_BPS
        bot_rets = -actual_vals[sorted_idx[:q_size]] - FEE_BPS
        ls_rets = np.concatenate([top_rets, bot_rets])
        ls_avg = np.mean(ls_rets)
        ls_sharpe = np.mean(ls_rets) / (np.std(ls_rets) + 1e-10) * np.sqrt(len(ls_rets))
        flag = "✅" if ls_avg > 0 else "  "
        print(f"  {flag} {name:30s}: IC={ic:+.4f}  L/S avg={ls_avg:>+.1f}bps  sharpe={ls_sharpe:+.2f}")

    # --- Feature importance from best model ---
    all_cols = ohlcv_cols + std_cols + sub5_cols
    valid = df[all_cols + [target]].notna().all(axis=1)
    X_all = df.loc[valid, all_cols].values
    y_all = df.loc[valid, target].values

    if len(X_all) > 200:
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gb.fit(StandardScaler().fit_transform(X_all), y_all)
        importances = sorted(zip(all_cols, gb.feature_importances_), key=lambda x: -x[1])

        print(f"\n  Top 20 features for 4h return prediction (GB):")
        for i, (col, imp) in enumerate(importances[:20], 1):
            tag = "[S5]" if col in sub5_cols else "[TK]" if col in std_cols else "[  ]"
            print(f"    {i:>2}. {tag} {col:40s} importance={imp:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  OI/FUNDING RESEARCH v24b — {SYMBOL}")
    print(f"  Out-of-Sample: {START_DATE} to {END_DATE}")
    print(f"  Data: Bybit ticker (5s) + Bybit trades → 5-min bars")
    print(f"{'='*70}")

    # Load data
    ohlcv_df = load_ohlcv_bars()
    if ohlcv_df.empty:
        print("ERROR: No OHLCV data. Aborting.")
        return

    ticker_df = load_ticker_data()
    if ticker_df.empty:
        print("ERROR: No ticker data. Aborting.")
        return

    # Build ticker features
    ticker_5m = build_ticker_5m_features(ticker_df)
    del ticker_df  # free memory
    ticker_5m = build_rolling_features(ticker_5m)

    # Merge
    df = merge_ohlcv_ticker(ohlcv_df, ticker_5m)

    # Run experiments
    part_a_validate_signals(df)
    part_b_oi_spike_research(df)

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
