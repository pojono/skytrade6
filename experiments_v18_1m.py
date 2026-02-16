#!/usr/bin/env python3
"""
v18 — 1-Minute Bar Experiments

Tests whether 1m bars improve our best strategies vs 5m bars.
Scope: Dec 2025 (30 days), Bybit futures, BTC/ETH/SOL.

Phases:
  0. Build 1m bars from tick data (cached to parquet)
  1. Run existing signal experiments (E01, E03, E09) at 1m resolution
  2. Precision entry: use 1m features to time entry within 5m signal window
  3. Short-horizon mean reversion at 1m with maker fees (4 bps RT)
  4. Vol prediction comparison (1m vs 5m Ridge regression)

Memory safety: processes 1 day at a time, deletes after use.
"""

import sys
import time
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
SOURCE = "bybit_futures"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PERIOD_30D = ("2025-12-01", "2025-12-30")

INTERVAL_1M_US = 60_000_000
INTERVAL_5M_US = 300_000_000

TAKER_FEE_BPS = 7.0
MAKER_FEE_BPS = 4.0   # 2 bps per fill × 2 fills


def mem_gb():
    return psutil.virtual_memory().used / (1024**3)


# ---------------------------------------------------------------------------
# Phase 0: Aggregate ticks → 1m bars (day-by-day, cached)
# ---------------------------------------------------------------------------

def aggregate_ticks_to_bars(trades, interval_us):
    """Aggregate tick trades into OHLCV + microstructure bars."""
    bucket = (trades["timestamp_us"].values // interval_us) * interval_us
    trades = trades.copy()
    trades["bucket"] = bucket

    features = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values
        t = grp["timestamp_us"].values
        n = len(grp)
        if n < 2:
            continue

        buy_mask = s == 1
        sell_mask = s == -1
        buy_vol = q[buy_mask].sum()
        sell_vol = q[sell_mask].sum()
        total_vol = q.sum()
        buy_quote = qq[buy_mask].sum()
        sell_quote = qq[sell_mask].sum()

        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)
        dollar_imbalance = (buy_quote - sell_quote) / max(buy_quote + sell_quote, 1e-10)

        q90 = np.percentile(q, 90) if n >= 10 else q.max()
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)

        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        duration_s = max((t[-1] - t[0]) / 1e6, 0.001)
        arrival_rate = n / duration_s

        if n > 2:
            iti = np.diff(t).astype(np.float64)
            iti_cv = iti.std() / max(iti.mean(), 1)
        else:
            iti_cv = 0.0

        mid_t = (t[0] + t[-1]) / 2
        first_half = int((t < mid_t).sum())
        trade_acceleration = (n - first_half - first_half) / max(n, 1)

        vwap = qq.sum() / max(total_vol, 1e-10)
        price_range = (p.max() - p.min()) / max(vwap, 1e-10)
        close_vs_vwap = (p[-1] - vwap) / max(vwap, 1e-10)

        if n > 10:
            signed_vol = q * s
            price_changes = np.diff(p)
            if len(price_changes) > 1 and signed_vol[1:].std() > 0:
                kyle_lambda = float(np.corrcoef(signed_vol[1:], price_changes)[0, 1])
            else:
                kyle_lambda = 0.0
        else:
            kyle_lambda = 0.0

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        full_range = high_p - low_p
        if full_range > 0:
            upper_wick = (high_p - max(open_p, close_p)) / full_range
            lower_wick = (min(open_p, close_p) - low_p) / full_range
            body_pct = abs(close_p - open_p) / full_range
        else:
            upper_wick = 0.0; lower_wick = 0.0; body_pct = 0.0

        avg_buy_size = buy_vol / max(buy_count, 1)
        avg_sell_size = sell_vol / max(sell_count, 1)
        size_imbalance = (avg_buy_size - avg_sell_size) / max(avg_buy_size + avg_sell_size, 1e-10)

        features.append({
            "timestamp_us": bkt,
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "large_imbalance": large_imbalance,
            "count_imbalance": count_imbalance,
            "arrival_rate": arrival_rate,
            "iti_cv": iti_cv,
            "trade_acceleration": trade_acceleration,
            "price_range": price_range,
            "close_vs_vwap": close_vs_vwap,
            "kyle_lambda": kyle_lambda,
            "vol_profile_skew": 0.0,  # simplified for speed
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_pct": body_pct,
            "size_imbalance": size_imbalance,
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "buy_volume": buy_vol, "sell_volume": sell_vol,
            "quote_volume": buy_quote + sell_quote, "trade_count": n,
        })

    return pd.DataFrame(features)


def build_bars(symbol, start_date, end_date, interval_us, interval_label):
    """Build bars from tick data, day-by-day with caching."""
    cache_dir = PARQUET_DIR / symbol / f"v18_{interval_label}_cache" / SOURCE
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, end_date)
    all_bars = []
    t0 = time.time()
    new_count = 0
    cache_count = 0

    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        cache_path = cache_dir / f"{ds}.parquet"

        if cache_path.exists():
            bars = pd.read_parquet(cache_path)
            all_bars.append(bars)
            cache_count += 1
        else:
            tick_path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{ds}.parquet"
            if not tick_path.exists():
                continue
            trades = pd.read_parquet(tick_path)
            bars = aggregate_ticks_to_bars(trades, interval_us)
            del trades
            gc.collect()
            if not bars.empty:
                bars.to_parquet(cache_path, index=False, compression="snappy")
                all_bars.append(bars)
            new_count += 1

        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.1)
        eta = (len(dates) - i) / max(rate, 0.01)

        if i % 5 == 0 or i == len(dates) or i == 1:
            print(f"    [{i:2d}/{len(dates)}] {ds}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"RAM={mem_gb():.1f}GB  new={new_count} cached={cache_count}",
                  flush=True)

    if not all_bars:
        return pd.DataFrame()

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df["returns"] = df["close"].pct_change()
    print(f"  → {len(df)} bars loaded ({interval_label}), RAM={mem_gb():.1f}GB", flush=True)
    return df


# ---------------------------------------------------------------------------
# Derived features (rolling windows adapted for 1m vs 5m)
# ---------------------------------------------------------------------------

def add_derived_features(df, bars_per_hour):
    """Add rolling features. bars_per_hour=60 for 1m, 12 for 5m."""
    bph = bars_per_hour

    # Rolling volatility at different horizons
    df["rvol_1h"] = df["returns"].rolling(bph).std()
    df["rvol_5h"] = df["returns"].rolling(5 * bph).std()
    df["rvol_24h"] = df["returns"].rolling(24 * bph).std()
    df["vol_ratio"] = df["rvol_1h"] / df["rvol_24h"].clip(lower=1e-10)

    # Volume z-score (24h lookback)
    w24 = 24 * bph
    df["vol_zscore"] = (df["volume"] - df["volume"].rolling(w24).mean()) / \
                       df["volume"].rolling(w24).std().clip(lower=1e-10)

    # Arrival rate z-score
    df["rate_zscore"] = (df["arrival_rate"] - df["arrival_rate"].rolling(w24).mean()) / \
                        df["arrival_rate"].rolling(w24).std().clip(lower=1e-10)

    # Price momentum
    df["mom_1h"] = df["close"].pct_change(bph)
    df["mom_5h"] = df["close"].pct_change(5 * bph)

    # Mean-reversion z-scores
    df["price_zscore_5h"] = (df["close"] - df["close"].rolling(5 * bph).mean()) / \
                            df["close"].rolling(5 * bph).std().clip(lower=1e-10)
    df["price_zscore_24h"] = (df["close"] - df["close"].rolling(w24).mean()) / \
                             df["close"].rolling(w24).std().clip(lower=1e-10)

    # Range expansion
    df["range_zscore"] = (df["price_range"] - df["price_range"].rolling(w24).mean()) / \
                         df["price_range"].rolling(w24).std().clip(lower=1e-10)

    # Cumulative imbalance (rolling sum over 30m and 1h)
    df["cum_imbalance_30m"] = df["vol_imbalance"].rolling(bph // 2).sum()
    df["cum_imbalance_1h"] = df["vol_imbalance"].rolling(bph).sum()

    # VWAP deviation z-score
    df["vwap_zscore"] = (df["close_vs_vwap"] - df["close_vs_vwap"].rolling(w24).mean()) / \
                        df["close_vs_vwap"].rolling(w24).std().clip(lower=1e-10)

    # Parkinson volatility (1h, 4h)
    log_hl = np.log(df["high"] / df["low"].clip(lower=1e-10))
    df["parkvol_1h"] = (log_hl**2).rolling(bph).mean().apply(np.sqrt) / (4 * np.log(2))**0.5
    df["parkvol_4h"] = (log_hl**2).rolling(4 * bph).mean().apply(np.sqrt) / (4 * np.log(2))**0.5

    # Efficiency ratio (1h, 4h)
    for h, label in [(bph, "1h"), (4 * bph, "4h")]:
        net_move = (df["close"] - df["close"].shift(h)).abs()
        sum_moves = df["returns"].abs().rolling(h).sum() * df["close"]
        df[f"efficiency_{label}"] = net_move / sum_moves.clip(lower=1e-10)

    # Sign persistence (autocorrelation of trade sign proxy)
    sign = np.sign(df["returns"])
    df["sign_ac_1h"] = sign.rolling(bph).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 2 else 0, raw=False)

    return df


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_signal(df, signal_col, entry_threshold, holding_bars, fee_bps,
                    direction="contrarian"):
    """Generic backtest. Returns array of per-trade PnL in bps."""
    data = df.dropna(subset=[signal_col]).copy()
    signals = data[signal_col].values
    closes = data["close"].values
    n = len(data)

    pnls = []
    in_trade = False
    entry_idx = 0
    trade_dir = 0

    for i in range(n - holding_bars):
        if in_trade and i - entry_idx >= holding_bars:
            raw = (closes[i] / closes[entry_idx] - 1) * 10000 * trade_dir
            pnls.append(raw - fee_bps)
            in_trade = False

        if not in_trade:
            if direction == "contrarian":
                if signals[i] > entry_threshold:
                    in_trade = True; entry_idx = i; trade_dir = -1
                elif signals[i] < -entry_threshold:
                    in_trade = True; entry_idx = i; trade_dir = 1
            else:
                if signals[i] > entry_threshold:
                    in_trade = True; entry_idx = i; trade_dir = 1
                elif signals[i] < -entry_threshold:
                    in_trade = True; entry_idx = i; trade_dir = -1

    return np.array(pnls) if pnls else np.array([])


def zscore_signal(series, window, min_periods=None):
    """Rolling z-score."""
    if min_periods is None:
        min_periods = window // 3
    mu = series.rolling(window, min_periods=min_periods).mean()
    sd = series.rolling(window, min_periods=min_periods).std().clip(lower=1e-10)
    return (series - mu) / sd


def rank_composite(df, cols, window, min_periods=None):
    """Rank-based composite z-score of multiple columns."""
    if min_periods is None:
        min_periods = window // 3
    rank_cols = []
    for col in cols:
        rc = f"_rank_{col}"
        df[rc] = df[col].rolling(window, min_periods=min_periods).rank(pct=True)
        rank_cols.append(rc)
    composite = df[rank_cols].mean(axis=1)
    signal = zscore_signal(composite, window, min_periods)
    df.drop(columns=rank_cols, inplace=True)
    return signal


def summarize_pnl(pnls, label):
    """Print summary of PnL array."""
    if len(pnls) == 0:
        print(f"  {label}: NO TRADES")
        return
    avg = pnls.mean()
    total = pnls.sum()
    wr = (pnls > 0).mean() * 100
    sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else 0
    print(f"  {label}: trades={len(pnls):4d}  avg={avg:+.2f}bps  "
          f"total={total:+.0f}bps  WR={wr:.0f}%  sharpe={sharpe:.3f}")


# ---------------------------------------------------------------------------
# Phase 1: Same experiments at 1m vs 5m
# ---------------------------------------------------------------------------

def run_signal_experiments(df, bars_per_hour, fee_bps, label_prefix):
    """Run E01, E03, E09 experiments. Returns dict of results."""
    bph = bars_per_hour
    results = {}

    # Holding periods scaled to real time
    hold_configs = {
        "1h": bph,
        "2h": 2 * bph,
        "4h": 4 * bph,
    }

    # Z-score window = 3 days
    zw = 3 * 24 * bph

    print(f"\n  --- E01: Contrarian Imbalance ({label_prefix}) ---")
    cols = ["vol_imbalance", "dollar_imbalance", "large_imbalance",
            "count_imbalance", "close_vs_vwap"]
    df["e01_signal"] = rank_composite(df, cols, zw)
    for thresh in [1.0, 1.5]:
        for hold_name, hold_bars in hold_configs.items():
            pnls = backtest_signal(df, "e01_signal", thresh, hold_bars, fee_bps, "contrarian")
            key = f"E01_{label_prefix}_t{thresh}_{hold_name}"
            results[key] = pnls
            summarize_pnl(pnls, f"E01 t={thresh} hold={hold_name}")

    print(f"\n  --- E03: Vol Breakout ({label_prefix}) ---")
    df["e03_signal"] = df["range_zscore"]
    for thresh in [1.0, 1.5]:
        for hold_name, hold_bars in hold_configs.items():
            pnls = backtest_signal(df, "e03_signal", thresh, hold_bars, fee_bps, "momentum")
            key = f"E03_{label_prefix}_t{thresh}_{hold_name}"
            results[key] = pnls
            summarize_pnl(pnls, f"E03 t={thresh} hold={hold_name}")

    print(f"\n  --- E09: Cumulative Imbalance Momentum ({label_prefix}) ---")
    df["e09_signal"] = zscore_signal(df["cum_imbalance_1h"], zw)
    for thresh in [1.0, 1.5]:
        for hold_name, hold_bars in hold_configs.items():
            pnls = backtest_signal(df, "e09_signal", thresh, hold_bars, fee_bps, "momentum")
            key = f"E09_{label_prefix}_t{thresh}_{hold_name}"
            results[key] = pnls
            summarize_pnl(pnls, f"E09 t={thresh} hold={hold_name}")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Precision entry (1m timing within 5m window)
# ---------------------------------------------------------------------------

def run_precision_entry(df_1m, df_5m, bars_per_hour_1m, fee_bps):
    """Use 5m signal as trigger, 1m features for entry timing."""
    print("\n  --- Phase 2: Precision Entry ---")
    results = {}

    # Build 5m signal: E01 contrarian
    zw_5m = 3 * 24 * 12  # 3 days at 5m
    cols = ["vol_imbalance", "dollar_imbalance", "large_imbalance",
            "count_imbalance", "close_vs_vwap"]
    df_5m["e01_signal"] = rank_composite(df_5m, cols, zw_5m)

    # Build 5m signal: E09 momentum
    df_5m["cum_imbalance_12"] = df_5m["vol_imbalance"].rolling(12).sum()
    df_5m["e09_signal"] = zscore_signal(df_5m["cum_imbalance_12"], zw_5m)

    # For each 5m bar that triggers, find the best 1m entry within that window
    for sig_name, sig_col, direction, thresh in [
        ("E01", "e01_signal", "contrarian", 1.0),
        ("E09", "e09_signal", "momentum", 1.5),
    ]:
        hold_bars_5m = 48  # 4h at 5m
        hold_bars_1m = 240  # 4h at 1m

        # Baseline: enter at 5m close
        pnls_baseline = backtest_signal(df_5m, sig_col, thresh, hold_bars_5m, fee_bps, direction)

        # Precision: for each 5m trigger, find best 1m entry
        data_5m = df_5m.dropna(subset=[sig_col]).copy()
        signals_5m = data_5m[sig_col].values
        ts_5m = data_5m["timestamp_us"].values
        closes_5m = data_5m["close"].values

        ts_1m = df_1m["timestamp_us"].values
        closes_1m = df_1m["close"].values
        imb_1m = df_1m["vol_imbalance"].values

        pnls_best_1m = []
        pnls_worst_1m = []
        pnls_imb_1m = []  # enter at 1m bar with most favorable imbalance

        in_trade = False
        entry_idx_5m = 0
        trade_dir = 0
        n_5m = len(data_5m)

        for i in range(n_5m - hold_bars_5m):
            if in_trade and i - entry_idx_5m >= hold_bars_5m:
                # Find exit price at 4h later in 1m data
                exit_ts = ts_5m[i]
                exit_idx_1m = np.searchsorted(ts_1m, exit_ts)
                if exit_idx_1m < len(closes_1m):
                    exit_price = closes_1m[min(exit_idx_1m, len(closes_1m) - 1)]
                else:
                    exit_price = closes_5m[i]

                # Best 1m entry (lowest for long, highest for short)
                entry_ts = ts_5m[entry_idx_5m]
                start_1m = np.searchsorted(ts_1m, entry_ts)
                end_1m = min(start_1m + 5, len(closes_1m))  # 5 bars of 1m = 1 bar of 5m

                if start_1m < end_1m:
                    window_prices = closes_1m[start_1m:end_1m]
                    window_imb = imb_1m[start_1m:end_1m]

                    if trade_dir == 1:  # long
                        best_entry = window_prices.min()
                        worst_entry = window_prices.max()
                        # Enter at bar with most negative imbalance (selling pressure = good for contrarian long)
                        imb_idx = np.argmin(window_imb) if direction == "contrarian" else np.argmax(window_imb)
                    else:  # short
                        best_entry = window_prices.max()
                        worst_entry = window_prices.min()
                        imb_idx = np.argmax(window_imb) if direction == "contrarian" else np.argmin(window_imb)

                    imb_entry = window_prices[imb_idx]

                    pnl_best = (exit_price / best_entry - 1) * 10000 * trade_dir - fee_bps
                    pnl_worst = (exit_price / worst_entry - 1) * 10000 * trade_dir - fee_bps
                    pnl_imb = (exit_price / imb_entry - 1) * 10000 * trade_dir - fee_bps

                    pnls_best_1m.append(pnl_best)
                    pnls_worst_1m.append(pnl_worst)
                    pnls_imb_1m.append(pnl_imb)

                in_trade = False

            if not in_trade:
                if direction == "contrarian":
                    if signals_5m[i] > thresh:
                        in_trade = True; entry_idx_5m = i; trade_dir = -1
                    elif signals_5m[i] < -thresh:
                        in_trade = True; entry_idx_5m = i; trade_dir = 1
                else:
                    if signals_5m[i] > thresh:
                        in_trade = True; entry_idx_5m = i; trade_dir = 1
                    elif signals_5m[i] < -thresh:
                        in_trade = True; entry_idx_5m = i; trade_dir = -1

        pnls_best_1m = np.array(pnls_best_1m) if pnls_best_1m else np.array([])
        pnls_worst_1m = np.array(pnls_worst_1m) if pnls_worst_1m else np.array([])
        pnls_imb_1m = np.array(pnls_imb_1m) if pnls_imb_1m else np.array([])

        summarize_pnl(pnls_baseline, f"{sig_name} baseline (5m close)")
        summarize_pnl(pnls_best_1m, f"{sig_name} best 1m entry (oracle)")
        summarize_pnl(pnls_worst_1m, f"{sig_name} worst 1m entry")
        summarize_pnl(pnls_imb_1m, f"{sig_name} imbalance-timed 1m entry")

        if len(pnls_baseline) > 0 and len(pnls_imb_1m) > 0:
            delta = pnls_imb_1m.mean() - pnls_baseline.mean()
            print(f"  → Imbalance timing delta: {delta:+.2f} bps/trade")
            oracle_delta = pnls_best_1m.mean() - pnls_baseline.mean()
            print(f"  → Oracle (best possible) delta: {oracle_delta:+.2f} bps/trade")

        results[f"precision_{sig_name}_baseline"] = pnls_baseline
        results[f"precision_{sig_name}_imb"] = pnls_imb_1m
        results[f"precision_{sig_name}_oracle"] = pnls_best_1m

    return results


# ---------------------------------------------------------------------------
# Phase 3: Short-horizon mean reversion at 1m
# ---------------------------------------------------------------------------

def run_short_horizon(df_1m, bars_per_hour):
    """Test short-horizon strategies only possible at 1m."""
    print("\n  --- Phase 3: Short-Horizon Mean Reversion (1m, maker fees) ---")
    bph = bars_per_hour
    results = {}

    # Short holding periods (in real time)
    hold_configs = {
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
    }

    # Z-score window = 1h (fast-reacting)
    zw_fast = bph

    # Strategy A: Contrarian imbalance (short-term fade)
    print(f"\n  A: Short-term contrarian imbalance")
    cols = ["vol_imbalance", "dollar_imbalance", "count_imbalance", "close_vs_vwap"]
    df_1m["st_signal"] = rank_composite(df_1m, cols, zw_fast)
    for thresh in [1.0, 1.5, 2.0]:
        for hold_name, hold_bars in hold_configs.items():
            pnls = backtest_signal(df_1m, "st_signal", thresh, hold_bars, MAKER_FEE_BPS, "contrarian")
            key = f"ST_A_{hold_name}_t{thresh}"
            results[key] = pnls
            summarize_pnl(pnls, f"A t={thresh} hold={hold_name}")

    # Strategy B: VWAP reversion (short-term)
    print(f"\n  B: Short-term VWAP reversion")
    df_1m["vwap_st"] = zscore_signal(df_1m["close_vs_vwap"], zw_fast)
    for thresh in [1.0, 1.5, 2.0]:
        for hold_name, hold_bars in hold_configs.items():
            pnls = backtest_signal(df_1m, "vwap_st", thresh, hold_bars, MAKER_FEE_BPS, "contrarian")
            key = f"ST_B_{hold_name}_t{thresh}"
            results[key] = pnls
            summarize_pnl(pnls, f"B t={thresh} hold={hold_name}")

    # Strategy C: Price mean-reversion (z-score vs 1h MA)
    print(f"\n  C: Short-term price mean-reversion (vs 1h MA)")
    df_1m["price_mr"] = (df_1m["close"] - df_1m["close"].rolling(bph).mean()) / \
                         df_1m["close"].rolling(bph).std().clip(lower=1e-10)
    for thresh in [1.5, 2.0, 2.5]:
        for hold_name, hold_bars in hold_configs.items():
            pnls = backtest_signal(df_1m, "price_mr", thresh, hold_bars, MAKER_FEE_BPS, "contrarian")
            key = f"ST_C_{hold_name}_t{thresh}"
            results[key] = pnls
            summarize_pnl(pnls, f"C t={thresh} hold={hold_name}")

    # Strategy D: Range compression → breakout (momentum, short hold)
    print(f"\n  D: Range compression breakout (momentum)")
    df_1m["range_brk"] = df_1m["range_zscore"]
    for thresh in [1.5, 2.0]:
        for hold_name, hold_bars in hold_configs.items():
            pnls = backtest_signal(df_1m, "range_brk", thresh, hold_bars, MAKER_FEE_BPS, "momentum")
            key = f"ST_D_{hold_name}_t{thresh}"
            results[key] = pnls
            summarize_pnl(pnls, f"D t={thresh} hold={hold_name}")

    return results


# ---------------------------------------------------------------------------
# Phase 4: Vol prediction comparison (1m vs 5m)
# ---------------------------------------------------------------------------

def run_vol_prediction(df, bars_per_hour, label):
    """Simple Ridge regression vol prediction, compare 1m vs 5m."""
    print(f"\n  --- Phase 4: Vol Prediction ({label}) ---")
    bph = bars_per_hour

    # Forward vol targets
    for horizon_name, horizon_bars in [("1h", bph), ("4h", 4 * bph)]:
        fwd_vol = df["returns"].rolling(horizon_bars).std().shift(-horizon_bars)
        df[f"fwd_vol_{horizon_name}"] = fwd_vol

    # Features for prediction
    feat_cols = [c for c in df.columns if c.startswith(("rvol_", "parkvol_", "vol_ratio",
                 "efficiency_", "sign_ac", "vol_zscore", "rate_zscore", "range_zscore"))]

    if not feat_cols:
        print("  No feature columns found, skipping vol prediction")
        return {}

    results = {}

    for horizon_name in ["1h", "4h"]:
        target = f"fwd_vol_{horizon_name}"
        valid = df.dropna(subset=feat_cols + [target])

        if len(valid) < 1000:
            print(f"  {horizon_name}: too few valid rows ({len(valid)}), skipping")
            continue

        X = valid[feat_cols].values
        y = valid[target].values

        # Walk-forward: train on first 70%, test on last 30%
        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Standardize
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd[sd < 1e-10] = 1.0
        X_train = (X_train - mu) / sd
        X_test = (X_test - mu) / sd

        # Ridge regression (manual, no sklearn needed)
        alpha = 1.0
        XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        try:
            w = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            print(f"  {horizon_name}: Ridge solve failed, skipping")
            continue

        y_pred = X_test @ w
        corr = np.corrcoef(y_test, y_pred)[0, 1] if y_test.std() > 0 else 0
        ss_res = ((y_test - y_pred) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        print(f"  {label} → {horizon_name}: R²={r2:.3f}  corr={corr:.3f}  "
              f"n_train={split}  n_test={len(X_test)}  features={len(feat_cols)}")
        results[f"vol_{label}_{horizon_name}"] = {"r2": r2, "corr": corr}

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_symbol(symbol, start_date, end_date, output_file):
    """Run all phases for one symbol."""
    print(f"\n{'='*70}")
    print(f"  v18 1-Minute Experiment: {symbol}")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  RAM: {mem_gb():.1f}GB")
    print(f"{'='*70}")

    t0_total = time.time()

    # --- Phase 0: Build bars ---
    print(f"\n[Phase 0] Building 1m bars from tick data...")
    df_1m = build_bars(symbol, start_date, end_date, INTERVAL_1M_US, "1m")
    if df_1m.empty:
        print("  ERROR: No 1m bars built. Skipping symbol.")
        return

    print(f"\n[Phase 0] Building 5m bars from tick data...")
    df_5m = build_bars(symbol, start_date, end_date, INTERVAL_5M_US, "5m")
    if df_5m.empty:
        print("  ERROR: No 5m bars built. Skipping symbol.")
        return

    print(f"\n  1m bars: {len(df_1m):,}  |  5m bars: {len(df_5m):,}  |  ratio: {len(df_1m)/max(len(df_5m),1):.1f}x")

    # --- Add derived features ---
    print(f"\n[Phase 0b] Computing derived features...")
    t1 = time.time()
    df_1m = add_derived_features(df_1m, bars_per_hour=60)
    print(f"  1m features done in {time.time()-t1:.0f}s, RAM={mem_gb():.1f}GB")
    t1 = time.time()
    df_5m = add_derived_features(df_5m, bars_per_hour=12)
    print(f"  5m features done in {time.time()-t1:.0f}s, RAM={mem_gb():.1f}GB")

    all_results = {}

    # --- Phase 1: Signal experiments at both resolutions ---
    print(f"\n{'='*70}")
    print(f"[Phase 1] Signal experiments: 1m vs 5m")
    print(f"{'='*70}")

    print(f"\n  === 5m bars (baseline, 7 bps taker fees) ===")
    res_5m = run_signal_experiments(df_5m, 12, TAKER_FEE_BPS, "5m")
    all_results.update(res_5m)

    print(f"\n  === 1m bars (7 bps taker fees) ===")
    res_1m = run_signal_experiments(df_1m, 60, TAKER_FEE_BPS, "1m")
    all_results.update(res_1m)

    print(f"\n  === 1m bars (4 bps maker fees) ===")
    res_1m_maker = run_signal_experiments(df_1m, 60, MAKER_FEE_BPS, "1m_maker")
    all_results.update(res_1m_maker)

    # --- Phase 1 comparison ---
    print(f"\n  --- Phase 1 Summary: 1m vs 5m ---")
    for exp in ["E01", "E03", "E09"]:
        for thresh in [1.0, 1.5]:
            key_5m = f"{exp}_5m_t{thresh}_4h"
            key_1m = f"{exp}_1m_t{thresh}_4h"
            key_1m_mk = f"{exp}_1m_maker_t{thresh}_4h"
            for k, label in [(key_5m, "5m/taker"), (key_1m, "1m/taker"), (key_1m_mk, "1m/maker")]:
                if k in all_results and len(all_results[k]) > 0:
                    avg = all_results[k].mean()
                    n = len(all_results[k])
                    print(f"  {exp} t={thresh} 4h [{label:10s}]: avg={avg:+.2f}bps  n={n}")

    # --- Phase 2: Precision entry ---
    print(f"\n{'='*70}")
    print(f"[Phase 2] Precision Entry (1m timing within 5m window)")
    print(f"{'='*70}")
    res_prec = run_precision_entry(df_1m, df_5m, 60, TAKER_FEE_BPS)
    all_results.update(res_prec)

    # --- Phase 3: Short-horizon ---
    print(f"\n{'='*70}")
    print(f"[Phase 3] Short-Horizon Mean Reversion (1m, maker fees)")
    print(f"{'='*70}")
    res_short = run_short_horizon(df_1m, 60)
    all_results.update(res_short)

    # --- Phase 4: Vol prediction ---
    print(f"\n{'='*70}")
    print(f"[Phase 4] Vol Prediction Comparison")
    print(f"{'='*70}")
    res_vol_5m = run_vol_prediction(df_5m, 12, "5m")
    res_vol_1m = run_vol_prediction(df_1m, 60, "1m")
    all_results.update(res_vol_5m)
    all_results.update(res_vol_1m)

    # --- Summary ---
    elapsed_total = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"  {symbol} COMPLETE in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"  RAM: {mem_gb():.1f}GB")
    print(f"{'='*70}")

    # Free memory
    del df_1m, df_5m
    gc.collect()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="v18: 1-Minute Bar Experiments")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        help="Symbols to test (default: BTC ETH SOL)")
    parser.add_argument("--start", default=PERIOD_30D[0], help="Start date")
    parser.add_argument("--end", default=PERIOD_30D[1], help="End date")
    args = parser.parse_args()

    print(f"v18 — 1-Minute Bar Experiments")
    print(f"Symbols: {args.symbols}")
    print(f"Period: {args.start} → {args.end}")
    print(f"RAM: {mem_gb():.1f}GB / {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"=" * 70)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    t0 = time.time()
    for symbol in args.symbols:
        output_file = results_dir / f"v18_1m_{symbol}.txt"

        # Redirect stdout to both console and file
        import io

        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        f = open(output_file, "w")
        old_stdout = sys.stdout
        sys.stdout = Tee(old_stdout, f)

        try:
            run_symbol(symbol, args.start, args.end, output_file)
        finally:
            sys.stdout = old_stdout
            f.close()

        print(f"\nResults saved to {output_file}")

    total = time.time() - t0
    print(f"\nAll symbols done in {total:.0f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
