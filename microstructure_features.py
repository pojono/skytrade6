#!/usr/bin/env python3
"""
Aggregate tick-level Bybit futures trades into microstructure features
at multiple timeframes (15m, 30m, 1h, 2h, 4h).

Memory-efficient: processes one day at a time, slices into candle bins,
computes features per candle, then frees the raw ticks.

Usage:
    python3 microstructure_features.py BTCUSDT 2026-01-01 2026-01-31
    python3 microstructure_features.py BTCUSDT 2026-01-01 2026-01-31 --timeframes 1h 4h
    python3 microstructure_features.py BTCUSDT 2026-01-01 2026-01-31 --data-dir ./data --output-dir ./results
"""

import argparse
import gc
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert as scipy_hilbert
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore", category=scipy_stats.ConstantInputWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEFRAMES = {
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
}

LARGE_TRADE_PERCENTILE = 95  # trades above this percentile = "large"

# Only load columns we actually need — saves ~40% RAM
USECOLS = ["timestamp", "side", "size", "price", "tickDirection",
           "foreignNotional"]


# ---------------------------------------------------------------------------
# Data loading — one day at a time, minimal columns
# ---------------------------------------------------------------------------


def load_day(path: Path) -> pd.DataFrame:
    """Load one day of Bybit futures tick data with minimal memory."""
    df = pd.read_csv(path, usecols=USECOLS)
    # Downcast where possible
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["side_sign"] = np.where(df["side"] == "Buy", np.int8(1), np.int8(-1))
    df["signed_volume"] = df["side_sign"].astype(np.float32) * df["size"].astype(np.float32)
    df["log_price"] = np.log(df["price"])
    df["tick_return"] = df["log_price"].diff().astype(np.float32)
    df["abs_price_change"] = df["price"].diff().abs().astype(np.float32)
    # Drop columns we no longer need in raw form
    df.drop(columns=["side"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Fast helpers (avoid scipy overhead per candle)
# ---------------------------------------------------------------------------


def _safe_corr(a, b):
    """Correlation that returns 0.0 when either array is constant or result is NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(r) or np.isinf(r) else float(r)


def _fast_skew(x):
    n = len(x)
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / s ** 3)


def _fast_kurtosis(x):
    n = len(x)
    if n < 4:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 4).mean() / s ** 4 - 3.0)


def _max_run(mask):
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return int((ends - starts).max()) if len(starts) > 0 else 0


def _hurst_rs(x, max_k=4):
    """Estimate Hurst exponent via rescaled range (R/S) method."""
    n_x = len(x)
    rs_list = []
    ns = []
    for k in range(1, max_k + 1):
        sub_n = n_x // (2 ** k)
        if sub_n < 4:
            break
        rs_vals = []
        for i in range(2 ** k):
            sub = x[i * sub_n:(i + 1) * sub_n]
            m = sub.mean()
            y = np.cumsum(sub - m)
            r = y.max() - y.min()
            s = sub.std()
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            ns.append(sub_n)
    if len(ns) >= 2:
        log_n = np.log(ns)
        log_rs = np.log(rs_list)
        slope = np.polyfit(log_n, log_rs, 1)[0]
        return float(np.clip(slope, 0, 1))
    return 0.5


def _fractal_dim(p, ks=(1, 2, 4)):
    """Estimate fractal dimension via Higuchi-like method."""
    lengths = []
    valid_ks = []
    for k in ks:
        if k >= len(p):
            continue
        l_sum = np.sum(np.abs(np.diff(p[::k]))) * (len(p) - 1) / (k * ((len(p) - 1) // k) * k)
        if l_sum > 0:
            lengths.append(np.log(l_sum))
            valid_ks.append(np.log(1.0 / k))
    if len(valid_ks) >= 2:
        return float(-np.polyfit(valid_ks, lengths, 1)[0])
    return 1.0


# ---------------------------------------------------------------------------
# Feature computation per candle group
# ---------------------------------------------------------------------------


def compute_features(group: pd.DataFrame) -> dict:
    """Compute all microstructure features for one candle."""
    n = len(group)
    if n == 0:
        return {}

    prices = group["price"].values
    sizes = group["size"].values
    notional = group["foreignNotional"].values
    sides = group["side_sign"].values
    ts = group["ts"]
    timestamps_s = group["timestamp"].values  # unix seconds
    tick_returns = group["tick_return"].values
    signed_vol = group["signed_volume"].values

    buy_mask = sides == 1
    sell_mask = sides == -1

    buy_sizes = sizes[buy_mask]
    sell_sizes = sizes[sell_mask]
    buy_notional = notional[buy_mask]
    sell_notional = notional[sell_mask]
    buy_prices = prices[buy_mask]
    sell_prices = prices[sell_mask]
    buy_count = buy_mask.sum()
    sell_count = sell_mask.sum()

    total_vol = sizes.sum()
    buy_vol = buy_sizes.sum() if len(buy_sizes) > 0 else 0.0
    sell_vol = sell_sizes.sum() if len(sell_sizes) > 0 else 0.0
    total_notional = notional.sum()

    o, h, l, c = prices[0], prices.max(), prices.min(), prices[-1]
    price_range = h - l

    # Time calculations
    candle_start_s = timestamps_s[0]
    candle_end_s = timestamps_s[-1]
    candle_duration_s = candle_end_s - candle_start_s
    if candle_duration_s <= 0:
        candle_duration_s = 1e-6  # avoid division by zero

    features = {}

    # -----------------------------------------------------------------------
    # 1. Basic OHLCV
    # -----------------------------------------------------------------------
    features["open"] = o
    features["high"] = h
    features["low"] = l
    features["close"] = c
    features["range"] = price_range
    features["return"] = np.log(c / o) if o > 0 else 0.0
    features["total_volume"] = total_vol
    features["total_notional"] = total_notional
    features["total_trades"] = n

    # -----------------------------------------------------------------------
    # 2. Volume by side
    # -----------------------------------------------------------------------
    features["buy_volume"] = buy_vol
    features["sell_volume"] = sell_vol
    features["net_volume_delta"] = buy_vol - sell_vol
    features["buy_sell_volume_ratio"] = buy_vol / sell_vol if sell_vol > 0 else np.nan
    features["buy_trades"] = buy_count
    features["sell_trades"] = sell_count
    features["buy_sell_trade_ratio"] = buy_count / sell_count if sell_count > 0 else np.nan
    features["order_flow_imbalance"] = (buy_vol - sell_vol) / (buy_vol + sell_vol) if total_vol > 0 else 0.0
    features["trade_imbalance"] = (buy_count - sell_count) / (buy_count + sell_count) if n > 0 else 0.0

    # -----------------------------------------------------------------------
    # 3. Trade size statistics
    # -----------------------------------------------------------------------
    features["avg_trade_size"] = total_vol / n
    features["median_trade_size"] = np.median(sizes)
    features["max_trade_size"] = sizes.max()
    features["std_trade_size"] = np.std(sizes) if n > 1 else 0.0
    features["avg_buy_size"] = np.mean(buy_sizes) if len(buy_sizes) > 0 else 0.0
    features["avg_sell_size"] = np.mean(sell_sizes) if len(sell_sizes) > 0 else 0.0
    features["size_imbalance"] = (features["avg_buy_size"] / features["avg_sell_size"]
                                  if features["avg_sell_size"] > 0 else np.nan)

    # Large trades (above 95th percentile of this candle)
    if n >= 20:
        large_threshold = np.percentile(sizes, LARGE_TRADE_PERCENTILE)
        large_mask = sizes >= large_threshold
        features["large_trade_count"] = large_mask.sum()
        features["large_trade_volume_pct"] = sizes[large_mask].sum() / total_vol if total_vol > 0 else 0.0
        features["large_buy_pct"] = (buy_mask & large_mask).sum() / large_mask.sum() if large_mask.sum() > 0 else 0.5
    else:
        features["large_trade_count"] = 0
        features["large_trade_volume_pct"] = 0.0
        features["large_buy_pct"] = 0.5

    # -----------------------------------------------------------------------
    # 4. VWAP / TWAP
    # -----------------------------------------------------------------------
    vwap = np.average(prices, weights=sizes) if total_vol > 0 else c
    features["vwap"] = vwap
    features["close_to_vwap"] = (c - vwap) / vwap if vwap > 0 else 0.0

    if len(buy_sizes) > 0:
        features["vwap_buy"] = np.average(buy_prices, weights=buy_sizes)
    else:
        features["vwap_buy"] = vwap
    if len(sell_sizes) > 0:
        features["vwap_sell"] = np.average(sell_prices, weights=sell_sizes)
    else:
        features["vwap_sell"] = vwap
    features["vwap_spread"] = features["vwap_buy"] - features["vwap_sell"]
    features["vwap_spread_bps"] = (features["vwap_spread"] / vwap * 10000) if vwap > 0 else 0.0

    # TWAP — time-weighted average price
    if n > 1:
        dt = np.diff(timestamps_s)
        dt = np.append(dt, dt[-1])  # repeat last interval
        twap = np.average(prices, weights=dt)
    else:
        twap = c
    features["twap"] = twap
    features["twap_vwap_spread_bps"] = ((twap - vwap) / vwap * 10000) if vwap > 0 else 0.0

    # -----------------------------------------------------------------------
    # 5. Time distribution within candle
    # -----------------------------------------------------------------------
    # Relative position of each trade in candle [0, 1]
    rel_time = (timestamps_s - candle_start_s) / candle_duration_s if candle_duration_s > 0 else np.zeros(n)

    # Time-weighted position in range
    if price_range > 0 and n > 1:
        price_position = (prices - l) / price_range  # [0, 1]
        dt = np.diff(timestamps_s)
        dt = np.append(dt, dt[-1])
        features["time_weighted_position"] = np.average(price_position, weights=dt)
        features["time_at_high_pct"] = (dt[price_position >= 0.75].sum() / dt.sum()) if dt.sum() > 0 else 0.0
        features["time_at_low_pct"] = (dt[price_position <= 0.25].sum() / dt.sum()) if dt.sum() > 0 else 0.0
    else:
        features["time_weighted_position"] = 0.5
        features["time_at_high_pct"] = 0.0
        features["time_at_low_pct"] = 0.0

    # Volume distribution across candle quarters
    q1_mask = rel_time < 0.25
    q2_mask = (rel_time >= 0.25) & (rel_time < 0.5)
    q3_mask = (rel_time >= 0.5) & (rel_time < 0.75)
    q4_mask = rel_time >= 0.75
    features["volume_q1_pct"] = sizes[q1_mask].sum() / total_vol if total_vol > 0 else 0.25
    features["volume_q2_pct"] = sizes[q2_mask].sum() / total_vol if total_vol > 0 else 0.25
    features["volume_q3_pct"] = sizes[q3_mask].sum() / total_vol if total_vol > 0 else 0.25
    features["volume_q4_pct"] = sizes[q4_mask].sum() / total_vol if total_vol > 0 else 0.25
    features["first_half_volume_pct"] = features["volume_q1_pct"] + features["volume_q2_pct"]

    # Buy concentration in time
    if buy_count > 1:
        features["buy_time_std"] = np.std(rel_time[buy_mask])
    else:
        features["buy_time_std"] = 0.0
    if sell_count > 1:
        features["sell_time_std"] = np.std(rel_time[sell_mask])
    else:
        features["sell_time_std"] = 0.0

    # Time to high / low
    high_idx = np.argmax(prices)
    low_idx = np.argmin(prices)
    features["time_to_high_pct"] = rel_time[high_idx] if n > 1 else 0.5
    features["time_to_low_pct"] = rel_time[low_idx] if n > 1 else 0.5
    features["high_before_low"] = 1 if high_idx < low_idx else 0

    # -----------------------------------------------------------------------
    # 6. Trade arrival / intensity
    # -----------------------------------------------------------------------
    if n > 1:
        inter_trade = np.diff(timestamps_s)
        inter_trade_pos = inter_trade[inter_trade > 0]
        features["trades_per_second"] = n / candle_duration_s
        features["median_inter_trade_ms"] = np.median(inter_trade) * 1000
        features["inter_trade_std"] = np.std(inter_trade)
        if len(inter_trade_pos) > 2:
            features["inter_trade_skew"] = _fast_skew(inter_trade_pos)
        else:
            features["inter_trade_skew"] = 0.0

        # Peak burst rate: trades per 1-second bin
        second_bins = np.floor(timestamps_s - candle_start_s).astype(int)
        bin_counts = np.bincount(second_bins)
        bin_counts = bin_counts[bin_counts > 0]
        features["max_trades_per_second"] = bin_counts.max() if len(bin_counts) > 0 else 0
        features["trade_rate_std"] = np.std(bin_counts) if len(bin_counts) > 1 else 0.0

        # Volume per second std
        vol_per_sec = np.zeros(int(candle_duration_s) + 1)
        sec_idx = np.floor(timestamps_s - candle_start_s).astype(int)
        np.add.at(vol_per_sec, sec_idx, sizes)
        nonzero_vol = vol_per_sec[vol_per_sec > 0]
        features["volume_per_second_std"] = np.std(nonzero_vol) if len(nonzero_vol) > 1 else 0.0
    else:
        features["trades_per_second"] = 0.0
        features["median_inter_trade_ms"] = 0.0
        features["inter_trade_std"] = 0.0
        features["inter_trade_skew"] = 0.0
        features["max_trades_per_second"] = 0
        features["trade_rate_std"] = 0.0
        features["volume_per_second_std"] = 0.0

    # -----------------------------------------------------------------------
    # 7. Consecutive runs / bursts
    # -----------------------------------------------------------------------
    if n > 1:
        side_changes = np.diff(sides) != 0
        features["side_switches"] = side_changes.sum()
        features["side_switch_rate"] = side_changes.sum() / (n - 1)

        # Max consecutive buy/sell runs (vectorized)
        features["max_buy_run"] = _max_run(buy_mask)
        features["max_sell_run"] = _max_run(sell_mask)
    else:
        features["side_switches"] = 0
        features["side_switch_rate"] = 0.0
        features["max_buy_run"] = 0
        features["max_sell_run"] = 0

    # -----------------------------------------------------------------------
    # 8. Cumulative Volume Delta (CVD) intra-candle
    # -----------------------------------------------------------------------
    cvd = np.cumsum(signed_vol)
    features["cvd_final"] = cvd[-1]
    features["cvd_max"] = cvd.max()
    features["cvd_min"] = cvd.min()
    features["cvd_range"] = cvd.max() - cvd.min()
    cvd_range_val = features["cvd_range"]
    if cvd_range_val > 0:
        features["cvd_close_vs_range"] = (cvd[-1] - cvd.min()) / cvd_range_val
    else:
        features["cvd_close_vs_range"] = 0.5

    # -----------------------------------------------------------------------
    # 9. Volatility measures
    # -----------------------------------------------------------------------
    valid_returns = tick_returns[~np.isnan(tick_returns)]
    if len(valid_returns) > 2:
        features["realized_vol"] = np.std(valid_returns) * np.sqrt(len(valid_returns))
        features["return_skewness"] = _fast_skew(valid_returns)
        features["return_kurtosis"] = _fast_kurtosis(valid_returns)

        # Autocorrelation lag-1
        if len(valid_returns) > 10:
            features["return_autocorr_1"] = _safe_corr(valid_returns[:-1], valid_returns[1:])
        else:
            features["return_autocorr_1"] = 0.0

        # Up vol vs down vol
        up_ret = valid_returns[valid_returns > 0]
        down_ret = valid_returns[valid_returns < 0]
        features["up_volatility"] = np.std(up_ret) if len(up_ret) > 1 else 0.0
        features["down_volatility"] = np.std(down_ret) if len(down_ret) > 1 else 0.0
        features["vol_asymmetry"] = (features["up_volatility"] - features["down_volatility"]) if (features["up_volatility"] + features["down_volatility"]) > 0 else 0.0
    else:
        features["realized_vol"] = 0.0
        features["return_skewness"] = 0.0
        features["return_kurtosis"] = 0.0
        features["return_autocorr_1"] = 0.0
        features["up_volatility"] = 0.0
        features["down_volatility"] = 0.0
        features["vol_asymmetry"] = 0.0

    # Garman-Klass volatility
    if o > 0 and h > 0 and l > 0 and c > 0 and h > l:
        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        features["garman_klass_vol"] = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
    else:
        features["garman_klass_vol"] = 0.0

    # Parkinson volatility
    if h > l and l > 0:
        features["parkinson_vol"] = np.log(h / l) / (2 * np.sqrt(np.log(2)))
    else:
        features["parkinson_vol"] = 0.0

    # -----------------------------------------------------------------------
    # 10. Price path efficiency
    # -----------------------------------------------------------------------
    if n > 1:
        abs_changes = group["abs_price_change"].values
        valid_changes = abs_changes[~np.isnan(abs_changes)]
        path_length = valid_changes.sum()
        net_move = abs(c - o)
        features["price_path_length"] = path_length
        features["efficiency_ratio"] = net_move / path_length if path_length > 0 else 0.0
        # Normalized path length (relative to range)
        features["path_length_over_range"] = path_length / price_range if price_range > 0 else 0.0
    else:
        features["price_path_length"] = 0.0
        features["efficiency_ratio"] = 0.0
        features["path_length_over_range"] = 0.0

    # -----------------------------------------------------------------------
    # 11. Tick direction analysis
    # -----------------------------------------------------------------------
    tick_dir = group["tickDirection"].values if "tickDirection" in group.columns else None
    if tick_dir is not None:
        plus_ticks = np.isin(tick_dir, ["PlusTick", "ZeroPlusTick"]).sum()
        minus_ticks = np.isin(tick_dir, ["MinusTick", "ZeroMinusTick"]).sum()
        features["uptick_pct"] = plus_ticks / n
        features["downtick_pct"] = minus_ticks / n
        features["net_tick_direction"] = (plus_ticks - minus_ticks) / n
    else:
        features["uptick_pct"] = 0.5
        features["downtick_pct"] = 0.5
        features["net_tick_direction"] = 0.0

    # -----------------------------------------------------------------------
    # 12. Liquidity / market quality
    # -----------------------------------------------------------------------
    # Kyle's Lambda: price impact per unit of signed volume
    if n > 10:
        valid_mask = ~np.isnan(tick_returns)
        if valid_mask.sum() > 10:
            sv = signed_vol[valid_mask]
            tr = tick_returns[valid_mask]
            # Simple regression: tick_return = lambda * signed_volume + epsilon
            if np.std(sv) > 0:
                features["kyle_lambda"] = np.cov(tr, sv)[0, 1] / np.var(sv)
            else:
                features["kyle_lambda"] = 0.0
        else:
            features["kyle_lambda"] = 0.0
    else:
        features["kyle_lambda"] = 0.0

    # Amihud illiquidity: |return| / volume
    candle_return = abs(np.log(c / o)) if o > 0 and c > 0 else 0.0
    features["amihud_illiquidity"] = candle_return / total_notional if total_notional > 0 else 0.0

    # Roll's implied spread
    if len(valid_returns) > 10:
        cov_1 = np.cov(valid_returns[:-1], valid_returns[1:])[0, 1]
        if cov_1 < 0:
            features["roll_spread"] = 2 * np.sqrt(-cov_1)
        else:
            features["roll_spread"] = 0.0
    else:
        features["roll_spread"] = 0.0

    features["avg_notional_per_trade"] = total_notional / n

    # -----------------------------------------------------------------------
    # 13. Trade size distribution
    # -----------------------------------------------------------------------
    if n >= 10:
        features["volume_skewness"] = _fast_skew(sizes)
        features["volume_kurtosis"] = _fast_kurtosis(sizes)
    else:
        features["volume_skewness"] = 0.0
        features["volume_kurtosis"] = 0.0

    # Percentiles of trade size
    features["trade_size_p25"] = np.percentile(sizes, 25)
    features["trade_size_p75"] = np.percentile(sizes, 75)
    features["trade_size_iqr"] = features["trade_size_p75"] - features["trade_size_p25"]

    # -----------------------------------------------------------------------
    # 14. Intra-candle momentum / mean-reversion signals
    # -----------------------------------------------------------------------
    # First-half return vs second-half return
    mid_idx = n // 2
    if mid_idx > 0 and mid_idx < n - 1:
        first_half_ret = np.log(prices[mid_idx] / prices[0]) if prices[0] > 0 else 0.0
        second_half_ret = np.log(prices[-1] / prices[mid_idx]) if prices[mid_idx] > 0 else 0.0
        features["first_half_return"] = first_half_ret
        features["second_half_return"] = second_half_ret
        features["intracandle_reversal"] = -np.sign(first_half_ret) * second_half_ret
    else:
        features["first_half_return"] = 0.0
        features["second_half_return"] = 0.0
        features["intracandle_reversal"] = 0.0

    # OFI (Order Flow Imbalance) in first vs second half
    buy_vol_h1 = sizes[(buy_mask) & (rel_time < 0.5)].sum()
    sell_vol_h1 = sizes[(sell_mask) & (rel_time < 0.5)].sum()
    buy_vol_h2 = sizes[(buy_mask) & (rel_time >= 0.5)].sum()
    sell_vol_h2 = sizes[(sell_mask) & (rel_time >= 0.5)].sum()
    tot_h1 = buy_vol_h1 + sell_vol_h1
    tot_h2 = buy_vol_h2 + sell_vol_h2
    features["ofi_first_half"] = (buy_vol_h1 - sell_vol_h1) / tot_h1 if tot_h1 > 0 else 0.0
    features["ofi_second_half"] = (buy_vol_h2 - sell_vol_h2) / tot_h2 if tot_h2 > 0 else 0.0
    features["ofi_shift"] = features["ofi_second_half"] - features["ofi_first_half"]

    # -----------------------------------------------------------------------
    # 15. Entropy & information-theoretic features
    # -----------------------------------------------------------------------
    # Volume entropy — Shannon entropy of trade size distribution (10 bins)
    if n >= 20:
        size_hist, _ = np.histogram(sizes, bins=10)
        size_probs = size_hist / size_hist.sum()
        size_probs = size_probs[size_probs > 0]
        features["volume_entropy"] = float(-np.sum(size_probs * np.log2(size_probs)))
    else:
        features["volume_entropy"] = 0.0

    # Inter-trade time entropy
    if n > 10:
        it = np.diff(timestamps_s)
        it_pos = it[it > 0]
        if len(it_pos) >= 10:
            it_hist, _ = np.histogram(it_pos, bins=10)
            it_probs = it_hist / it_hist.sum()
            it_probs = it_probs[it_probs > 0]
            features["inter_trade_entropy"] = float(-np.sum(it_probs * np.log2(it_probs)))
        else:
            features["inter_trade_entropy"] = 0.0
    else:
        features["inter_trade_entropy"] = 0.0

    # Side sequence entropy — bigram entropy of buy/sell sequences
    if n >= 20:
        bigrams = sides[:-1] * 2 + sides[1:]  # encode (prev, cur) as single int
        unique, counts = np.unique(bigrams, return_counts=True)
        bg_probs = counts / counts.sum()
        features["side_sequence_entropy"] = float(-np.sum(bg_probs * np.log2(bg_probs)))
    else:
        features["side_sequence_entropy"] = 0.0

    # Price tick entropy
    if len(valid_returns) >= 20:
        ret_hist, _ = np.histogram(valid_returns, bins=10)
        ret_probs = ret_hist / ret_hist.sum()
        ret_probs = ret_probs[ret_probs > 0]
        features["price_tick_entropy"] = float(-np.sum(ret_probs * np.log2(ret_probs)))
    else:
        features["price_tick_entropy"] = 0.0

    # -----------------------------------------------------------------------
    # 16. Toxicity / adverse selection metrics
    # -----------------------------------------------------------------------
    # VPIN — Volume-Synchronized Probability of Informed Trading
    if n >= 20 and total_vol > 0:
        n_buckets = min(10, n // 5)
        if n_buckets >= 2:
            cum_vol = np.cumsum(sizes)
            bucket_size = total_vol / n_buckets
            bucket_edges = np.arange(1, n_buckets) * bucket_size
            bucket_idx = np.searchsorted(cum_vol, bucket_edges)
            bucket_idx = np.clip(bucket_idx, 0, n - 1)
            bucket_starts = np.concatenate(([0], bucket_idx))
            bucket_ends = np.concatenate((bucket_idx, [n]))
            vpin_sum = 0.0
            for bs, be in zip(bucket_starts, bucket_ends):
                if be > bs:
                    bv = buy_sizes_slice = sizes[bs:be][buy_mask[bs:be]].sum()
                    sv = sizes[bs:be][sell_mask[bs:be]].sum()
                    bucket_total = bv + sv
                    if bucket_total > 0:
                        vpin_sum += abs(bv - sv) / bucket_total
            features["vpin"] = vpin_sum / n_buckets
        else:
            features["vpin"] = abs(buy_vol - sell_vol) / total_vol
    else:
        features["vpin"] = 0.0

    # Toxic flow ratio — fraction of volume that trades against VWAP
    if n >= 10 and total_vol > 0:
        buy_above_vwap = sizes[(buy_mask) & (prices > vwap)].sum()
        sell_below_vwap = sizes[(sell_mask) & (prices < vwap)].sum()
        features["toxic_flow_ratio"] = (buy_above_vwap + sell_below_vwap) / total_vol
    else:
        features["toxic_flow_ratio"] = 0.0

    # Effective spread — avg |trade_price - vwap| weighted by size
    if total_vol > 0 and vwap > 0:
        features["effective_spread_bps"] = float(np.average(np.abs(prices - vwap), weights=sizes) / vwap * 10000)
    else:
        features["effective_spread_bps"] = 0.0

    # Price impact asymmetry — buy impact vs sell impact
    if buy_count >= 5 and sell_count >= 5 and buy_vol > 0 and sell_vol > 0:
        buy_tr = tick_returns[buy_mask]
        sell_tr = tick_returns[sell_mask]
        buy_tr_valid = buy_tr[~np.isnan(buy_tr)]
        sell_tr_valid = sell_tr[~np.isnan(sell_tr)]
        buy_impact = np.mean(buy_tr_valid) if len(buy_tr_valid) > 0 else 0.0
        sell_impact = np.mean(sell_tr_valid) if len(sell_tr_valid) > 0 else 0.0
        total_impact = abs(buy_impact) + abs(sell_impact)
        features["price_impact_asymmetry"] = (abs(buy_impact) - abs(sell_impact)) / total_impact if total_impact > 0 else 0.0
    else:
        features["price_impact_asymmetry"] = 0.0

    # -----------------------------------------------------------------------
    # 17. Clustering / herding features
    # -----------------------------------------------------------------------
    # Trade size clustering — fraction at round sizes
    if n >= 10:
        # Round sizes: multiples of 0.001, 0.01, 0.1, 1.0
        round_mask = np.zeros(n, dtype=bool)
        for r in [1.0, 0.1, 0.01]:
            round_mask |= (np.abs(sizes / r - np.round(sizes / r)) < 1e-9)
        features["size_clustering"] = round_mask.sum() / n
    else:
        features["size_clustering"] = 0.0

    # Price level clustering — fraction at round price levels ($10, $100, $500)
    if n >= 10:
        price_round_mask = np.zeros(n, dtype=bool)
        for r in [10.0, 50.0, 100.0, 500.0]:
            price_round_mask |= (np.abs(prices % r) < 0.01)
        features["price_clustering"] = price_round_mask.sum() / n
    else:
        features["price_clustering"] = 0.0

    # Temporal clustering (Hawkes-like) — ratio of trades within 100ms after a trade vs baseline
    if n > 20:
        it_all = np.diff(timestamps_s)
        fast_trades = (it_all < 0.1).sum()  # within 100ms
        features["temporal_clustering"] = fast_trades / (n - 1)
    else:
        features["temporal_clustering"] = 0.0

    # Volume autocorrelation — lag-1 autocorr of trade sizes
    if n > 10:
        s1, s2 = sizes[:-1], sizes[1:]
        if np.std(s1) > 0 and np.std(s2) > 0:
            features["volume_autocorr"] = _safe_corr(s1, s2)
        else:
            features["volume_autocorr"] = 0.0
    else:
        features["volume_autocorr"] = 0.0

    # -----------------------------------------------------------------------
    # 18. Acceleration / momentum derivatives
    # -----------------------------------------------------------------------
    # Volume acceleration — change in volume rate between halves
    vol_h1 = tot_h1
    vol_h2 = tot_h2
    vol_avg = (vol_h1 + vol_h2) / 2
    if vol_avg > 0:
        features["volume_acceleration"] = (vol_h2 - vol_h1) / vol_avg
    else:
        features["volume_acceleration"] = 0.0

    # OFI acceleration
    ofi_h1 = features["ofi_first_half"]
    ofi_h2 = features["ofi_second_half"]
    features["ofi_acceleration"] = ofi_h2 - ofi_h1  # same as ofi_shift but kept for clarity

    # Price acceleration — curvature of price path (split into 3 segments)
    if n >= 9:
        t1 = n // 3
        t2 = 2 * n // 3
        r1 = np.log(prices[t1] / prices[0]) if prices[0] > 0 else 0.0
        r2 = np.log(prices[t2] / prices[t1]) if prices[t1] > 0 else 0.0
        r3 = np.log(prices[-1] / prices[t2]) if prices[t2] > 0 else 0.0
        features["price_acceleration"] = (r3 - r1)  # positive = convex (accelerating), negative = concave
        features["price_curvature"] = r3 - 2 * r2 + r1  # discrete second derivative
    else:
        features["price_acceleration"] = 0.0
        features["price_curvature"] = 0.0

    # Volatility of volatility — std of sub-window realized vols
    if n >= 40:
        n_sub = 4
        sub_size = n // n_sub
        sub_vols = []
        for i in range(n_sub):
            sub_ret = valid_returns[i * sub_size:(i + 1) * sub_size] if i < n_sub - 1 else valid_returns[i * sub_size:]
            if len(sub_ret) > 2:
                sub_vols.append(np.std(sub_ret))
        if len(sub_vols) >= 2:
            features["vol_of_vol"] = float(np.std(sub_vols))
        else:
            features["vol_of_vol"] = 0.0
    else:
        features["vol_of_vol"] = 0.0

    # -----------------------------------------------------------------------
    # 19. Cross-side interaction features
    # -----------------------------------------------------------------------
    # Absorption ratio — heavy volume but small price move
    if price_range > 0 and total_vol > 0:
        features["absorption_ratio"] = total_vol / price_range
    else:
        features["absorption_ratio"] = 0.0

    # Aggression imbalance — buy vol above VWAP vs sell vol below VWAP
    if total_vol > 0:
        aggressive_buy = sizes[(buy_mask) & (prices > vwap)].sum()
        aggressive_sell = sizes[(sell_mask) & (prices < vwap)].sum()
        agg_total = aggressive_buy + aggressive_sell
        features["aggression_imbalance"] = (aggressive_buy - aggressive_sell) / agg_total if agg_total > 0 else 0.0
    else:
        features["aggression_imbalance"] = 0.0

    # Response asymmetry — avg price move per buy trade vs per sell trade
    if buy_count >= 5 and sell_count >= 5:
        buy_abs_ret = np.abs(tick_returns[buy_mask])
        sell_abs_ret = np.abs(tick_returns[sell_mask])
        buy_resp = np.nanmean(buy_abs_ret)
        sell_resp = np.nanmean(sell_abs_ret)
        total_resp = buy_resp + sell_resp
        features["response_asymmetry"] = (buy_resp - sell_resp) / total_resp if total_resp > 0 else 0.0
    else:
        features["response_asymmetry"] = 0.0

    # Sweep detection — rapid monotonic price moves (< 200ms gaps, same direction)
    if n > 10:
        it_all = np.diff(timestamps_s)
        pr_diff = np.diff(prices)
        fast = it_all < 0.2  # < 200ms
        up_sweep = fast & (pr_diff > 0)
        down_sweep = fast & (pr_diff < 0)
        # Count max consecutive sweeps
        features["max_up_sweep"] = _max_run(up_sweep) if len(up_sweep) > 0 else 0
        features["max_down_sweep"] = _max_run(down_sweep) if len(down_sweep) > 0 else 0
        features["sweep_count"] = max(features["max_up_sweep"], features["max_down_sweep"])
    else:
        features["max_up_sweep"] = 0
        features["max_down_sweep"] = 0
        features["sweep_count"] = 0

    # -----------------------------------------------------------------------
    # 20. Fractal / multi-scale features
    # -----------------------------------------------------------------------
    # Hurst exponent estimate (R/S method on tick returns)
    if len(valid_returns) >= 20:
        features["hurst_exponent"] = _hurst_rs(valid_returns)
    else:
        features["hurst_exponent"] = 0.5

    # Fractal dimension estimate (from price path)
    if n >= 20 and price_range > 0:
        features["fractal_dimension"] = _fractal_dim(prices)
    else:
        features["fractal_dimension"] = 1.0

    # Sub-candle regime count — split into 4 windows, classify trending vs ranging
    if n >= 20:
        sub_n = n // 4
        regimes = []
        for i in range(4):
            sub_p = prices[i * sub_n:(i + 1) * sub_n] if i < 3 else prices[i * sub_n:]
            if len(sub_p) >= 3:
                sub_ret = np.log(sub_p[-1] / sub_p[0]) if sub_p[0] > 0 else 0.0
                sub_range = sub_p.max() - sub_p.min()
                eff = abs(sub_ret) / (np.log(sub_p.max() / sub_p.min()) if sub_p.min() > 0 and sub_p.max() > sub_p.min() else 1.0)
                regimes.append(1 if eff > 0.5 else 0)  # 1=trending, 0=ranging
            else:
                regimes.append(0)
        features["sub_regime_trending_pct"] = sum(regimes) / len(regimes)
        features["sub_regime_transitions"] = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
    else:
        features["sub_regime_trending_pct"] = 0.0
        features["sub_regime_transitions"] = 0

    # -----------------------------------------------------------------------
    # 21. Tail / extreme event features
    # -----------------------------------------------------------------------
    # Max drawdown / drawup within candle
    if n >= 5:
        cum_max = np.maximum.accumulate(prices)
        drawdowns = (prices - cum_max) / cum_max
        features["max_drawdown"] = float(drawdowns.min())

        cum_min = np.minimum.accumulate(prices)
        drawups = (prices - cum_min) / np.where(cum_min > 0, cum_min, 1.0)
        features["max_drawup"] = float(drawups.max())

        features["drawdown_drawup_asymmetry"] = abs(features["max_drawdown"]) - features["max_drawup"]
    else:
        features["max_drawdown"] = 0.0
        features["max_drawup"] = 0.0
        features["drawdown_drawup_asymmetry"] = 0.0

    # Tail ratio — volume in top 1% of price moves vs bottom 1%
    if len(valid_returns) >= 100:
        p99 = np.percentile(valid_returns, 99)
        p01 = np.percentile(valid_returns, 1)
        not_nan = ~np.isnan(tick_returns)
        tail_up_mask = not_nan & (tick_returns >= p99)
        tail_down_mask = not_nan & (tick_returns <= p01)
        tail_up_vol = sizes[tail_up_mask].sum()
        tail_down_vol = sizes[tail_down_mask].sum()
        tail_total = tail_up_vol + tail_down_vol
        features["tail_volume_ratio"] = (tail_up_vol - tail_down_vol) / tail_total if tail_total > 0 else 0.0
    else:
        features["tail_volume_ratio"] = 0.0

    # Flash event detection — count of >3σ moves within 1-second windows
    if len(valid_returns) > 20:
        ret_std = np.std(valid_returns)
        if ret_std > 0:
            extreme_mask = np.abs(valid_returns) > 3 * ret_std
            features["flash_event_count"] = int(extreme_mask.sum())
            features["flash_event_pct"] = extreme_mask.sum() / len(valid_returns)
        else:
            features["flash_event_count"] = 0
            features["flash_event_pct"] = 0.0
    else:
        features["flash_event_count"] = 0
        features["flash_event_pct"] = 0.0

    # -----------------------------------------------------------------------
    # 22. Time-of-day / seasonality features
    # -----------------------------------------------------------------------
    candle_ts = ts.iloc[0]
    candle_hour = candle_ts.hour if hasattr(candle_ts, 'hour') else 0
    candle_minute = candle_ts.minute if hasattr(candle_ts, 'minute') else 0
    hour_frac = candle_hour + candle_minute / 60.0

    # Cyclical encoding
    features["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    # Day of week (0=Mon, 6=Sun)
    dow = candle_ts.dayofweek if hasattr(candle_ts, 'dayofweek') else 0
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Distance to next funding rate (8h cycle: 00:00, 08:00, 16:00 UTC)
    funding_hours = [0, 8, 16, 24]
    dist_to_funding = min(abs(fh - hour_frac) for fh in funding_hours)
    features["dist_to_funding_hrs"] = dist_to_funding

    # Trading session flags — DST-aware (all times in UTC)
    # Determine DST: US DST is Mar second Sun – Nov first Sun
    # EU DST is Mar last Sun – Oct last Sun
    month = candle_ts.month
    day = candle_ts.day
    dow_num = candle_ts.dayofweek  # 0=Mon, 6=Sun

    # Approximate DST detection (good enough for session flags)
    # US DST: roughly Mar 8-14 (second Sun) to Nov 1-7 (first Sun)
    us_dst = (month > 3 and month < 11) or \
             (month == 3 and day >= 8) or \
             (month == 11 and day < 7)
    # EU DST: roughly Mar 25-31 (last Sun) to Oct 25-31 (last Sun)
    eu_dst = (month > 3 and month < 10) or \
             (month == 3 and day >= 25) or \
             (month == 10 and day < 25)

    # Session times in UTC (shift 1h earlier in summer)
    # Asia/Tokyo: 00:00-09:00 UTC (no DST in Japan)
    # Europe/London: 08:00-16:30 UTC winter, 07:00-15:30 UTC summer
    # US/New York: 14:30-21:00 UTC winter, 13:30-20:00 UTC summer
    asia_start, asia_end = 0.0, 9.0
    eu_start = 7.0 if eu_dst else 8.0
    eu_end = 15.5 if eu_dst else 16.5
    us_start = 13.5 if us_dst else 14.5
    us_end = 20.0 if us_dst else 21.0

    features["session_asia"] = 1.0 if asia_start <= hour_frac < asia_end else 0.0
    features["session_europe"] = 1.0 if eu_start <= hour_frac < eu_end else 0.0
    features["session_us"] = 1.0 if us_start <= hour_frac < us_end else 0.0

    # Overlap flags
    features["overlap_asia_europe"] = 1.0 if (eu_start <= hour_frac < asia_end) else 0.0
    features["overlap_europe_us"] = 1.0 if (us_start <= hour_frac < eu_end) else 0.0
    features["overlap_count"] = features["overlap_asia_europe"] + features["overlap_europe_us"]

    # -----------------------------------------------------------------------
    # 23. Volume Profile & Fair Value (intra-candle)
    # -----------------------------------------------------------------------
    # Build volume profile: distribute volume across price bins
    if n >= 20 and price_range > 0:
        n_bins = min(50, max(10, n // 100))
        bin_edges = np.linspace(l, h, n_bins + 1)
        bin_idx = np.clip(np.digitize(prices, bin_edges) - 1, 0, n_bins - 1)

        # Volume at each price level
        vol_profile = np.zeros(n_bins)
        np.add.at(vol_profile, bin_idx, sizes)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Point of Control (POC) — price level with highest volume
        poc_idx = np.argmax(vol_profile)
        features["poc_price"] = float(bin_mids[poc_idx])
        features["close_to_poc"] = (c - features["poc_price"]) / features["poc_price"] if features["poc_price"] > 0 else 0.0
        features["close_to_poc_bps"] = features["close_to_poc"] * 10000

        # Value Area (70% of volume) — find tightest range containing 70% vol
        total_profile_vol = vol_profile.sum()
        va_threshold = 0.70 * total_profile_vol
        # Expand from POC outward
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        va_vol = vol_profile[poc_idx]
        while va_vol < va_threshold and (va_low_idx > 0 or va_high_idx < n_bins - 1):
            expand_low = vol_profile[va_low_idx - 1] if va_low_idx > 0 else 0
            expand_high = vol_profile[va_high_idx + 1] if va_high_idx < n_bins - 1 else 0
            if expand_low >= expand_high and va_low_idx > 0:
                va_low_idx -= 1
                va_vol += vol_profile[va_low_idx]
            elif va_high_idx < n_bins - 1:
                va_high_idx += 1
                va_vol += vol_profile[va_high_idx]
            else:
                va_low_idx -= 1
                va_vol += vol_profile[va_low_idx]

        features["value_area_low"] = float(bin_edges[va_low_idx])
        features["value_area_high"] = float(bin_edges[va_high_idx + 1])
        va_width = features["value_area_high"] - features["value_area_low"]
        features["value_area_width_bps"] = (va_width / vwap * 10000) if vwap > 0 else 0.0

        # Close position within value area
        if va_width > 0:
            features["close_in_value_area"] = (c - features["value_area_low"]) / va_width
        else:
            features["close_in_value_area"] = 0.5
        features["close_above_value_area"] = 1.0 if c > features["value_area_high"] else 0.0
        features["close_below_value_area"] = 1.0 if c < features["value_area_low"] else 0.0

        # Volume profile shape — how concentrated is volume?
        # Low node ratio: volume in bottom 25% of bins / total
        sorted_vp = np.sort(vol_profile)
        low_node_cutoff = int(n_bins * 0.25)
        features["low_volume_node_pct"] = sorted_vp[:low_node_cutoff].sum() / total_profile_vol if total_profile_vol > 0 else 0.0
        features["high_volume_node_pct"] = sorted_vp[-low_node_cutoff:].sum() / total_profile_vol if total_profile_vol > 0 else 0.0

        # Volume profile skew — is volume concentrated at top or bottom of range?
        if total_profile_vol > 0:
            vp_mean_price = np.average(bin_mids, weights=vol_profile)
            features["volume_profile_skew"] = (vp_mean_price - (h + l) / 2) / price_range
        else:
            features["volume_profile_skew"] = 0.0

        # Fair price — volume-weighted median price
        cum_vol_profile = np.cumsum(vol_profile)
        median_vol_idx = np.searchsorted(cum_vol_profile, total_profile_vol * 0.5)
        median_vol_idx = min(median_vol_idx, n_bins - 1)
        features["fair_price"] = float(bin_mids[median_vol_idx])
        features["close_to_fair_price"] = (c - features["fair_price"]) / features["fair_price"] if features["fair_price"] > 0 else 0.0
        features["close_to_fair_price_bps"] = features["close_to_fair_price"] * 10000

        # Fair value = average of POC and VWAP (two estimates of "true" value)
        features["fair_value"] = (features["poc_price"] + vwap) / 2
        features["close_to_fair_value_bps"] = ((c - features["fair_value"]) / features["fair_value"] * 10000) if features["fair_value"] > 0 else 0.0
    else:
        features["poc_price"] = c
        features["close_to_poc"] = 0.0
        features["close_to_poc_bps"] = 0.0
        features["value_area_low"] = l
        features["value_area_high"] = h
        features["value_area_width_bps"] = 0.0
        features["close_in_value_area"] = 0.5
        features["close_above_value_area"] = 0.0
        features["close_below_value_area"] = 0.0
        features["low_volume_node_pct"] = 0.0
        features["high_volume_node_pct"] = 0.0
        features["volume_profile_skew"] = 0.0
        features["fair_price"] = c
        features["close_to_fair_price"] = 0.0
        features["close_to_fair_price_bps"] = 0.0
        features["fair_value"] = c
        features["close_to_fair_value_bps"] = 0.0

    # -----------------------------------------------------------------------
    # 24. Physics-inspired features — Newtonian Mechanics
    # -----------------------------------------------------------------------
    # Momentum = mass (volume) × velocity (return)
    candle_ret = features["return"]
    features["market_momentum"] = total_vol * candle_ret

    # Force = Δmomentum / Δt  (split candle into halves)
    if mid_idx > 0 and mid_idx < n - 1:
        vol_h1 = sizes[:mid_idx].sum()
        vol_h2 = sizes[mid_idx:].sum()
        ret_h1 = features["first_half_return"]
        ret_h2 = features["second_half_return"]
        mom_h1 = vol_h1 * ret_h1
        mom_h2 = vol_h2 * ret_h2
        half_duration = candle_duration_s / 2
        features["market_force"] = (mom_h2 - mom_h1) / half_duration if half_duration > 0 else 0.0
    else:
        features["market_force"] = 0.0

    # Impulse = total signed volume × duration (total directional energy)
    features["market_impulse"] = float(np.sum(signed_vol)) * candle_duration_s

    # Kinetic energy = ½ × volume × return²
    features["kinetic_energy"] = 0.5 * total_vol * candle_ret ** 2

    # Potential energy = distance from VWAP × volume (stored mean-reversion energy)
    dist_from_vwap = abs(c - vwap) / vwap if vwap > 0 else 0.0
    features["potential_energy"] = dist_from_vwap * total_vol

    # Total energy = kinetic + potential (conservation of energy analogy)
    features["total_energy"] = features["kinetic_energy"] + features["potential_energy"]

    # Energy ratio: what fraction is kinetic vs potential?
    if features["total_energy"] > 0:
        features["energy_ratio"] = features["kinetic_energy"] / features["total_energy"]
    else:
        features["energy_ratio"] = 0.5

    # Inertia = resistance to direction change = volume × (1 - efficiency_ratio)
    features["market_inertia"] = total_vol * (1 - features.get("efficiency_ratio", 0.0))

    # -----------------------------------------------------------------------
    # 25. Physics-inspired — Thermodynamics
    # -----------------------------------------------------------------------
    # Temperature = realized volatility (kinetic energy of tick movements)
    features["market_temperature"] = features.get("realized_vol", 0.0)

    # Pressure = volume / range (compressed volume = high pressure)
    features["market_pressure"] = total_vol / price_range if price_range > 0 else 0.0

    # Pressure-volume work = pressure × ΔV (range) — energy released
    features["pv_work"] = features["market_pressure"] * price_range

    # Heat capacity = Δvolume / Δtemperature
    # Approximate: how much extra volume is needed per unit of volatility
    if features["market_temperature"] > 0:
        features["heat_capacity"] = total_vol / features["market_temperature"]
    else:
        features["heat_capacity"] = 0.0

    # Boltzmann entropy = log(number of distinct price levels visited)
    if n >= 10:
        unique_prices = len(np.unique(np.round(prices, 2)))
        features["boltzmann_entropy"] = np.log(unique_prices) if unique_prices > 1 else 0.0
    else:
        features["boltzmann_entropy"] = 0.0

    # Thermal equilibrium deviation: is the market "cooling" or "heating"?
    # Compare vol in first half vs second half
    if n >= 10:
        ret_h1_arr = valid_returns[:len(valid_returns) // 2]
        ret_h2_arr = valid_returns[len(valid_returns) // 2:]
        vol_h1_temp = np.std(ret_h1_arr) if len(ret_h1_arr) > 2 else 0.0
        vol_h2_temp = np.std(ret_h2_arr) if len(ret_h2_arr) > 2 else 0.0
        features["temperature_change"] = vol_h2_temp - vol_h1_temp
        features["heating_rate"] = (vol_h2_temp - vol_h1_temp) / (vol_h1_temp + 1e-12)
    else:
        features["temperature_change"] = 0.0
        features["heating_rate"] = 0.0

    # -----------------------------------------------------------------------
    # 26. Physics-inspired — Electromagnetism / Field Theory
    # -----------------------------------------------------------------------
    # Buy/sell field strength = OFI gradient across candle (dOFI/dt)
    if n >= 10:
        q1_ofi = features.get("ofi_first_half", 0.0)
        q2_ofi = features.get("ofi_second_half", 0.0)
        features["ofi_field_gradient"] = (q2_ofi - q1_ofi) / (candle_duration_s / 2) if candle_duration_s > 0 else 0.0
    else:
        features["ofi_field_gradient"] = 0.0

    # Dipole moment = separation between buy and sell centers of mass in time
    if buy_count > 0 and sell_count > 0:
        buy_com_time = np.mean(rel_time[buy_mask])  # center of mass of buys in time
        sell_com_time = np.mean(rel_time[sell_mask])
        features["temporal_dipole"] = buy_com_time - sell_com_time  # +ve = buys later, -ve = buys earlier

        # Dipole strength = separation × volume difference
        features["dipole_strength"] = features["temporal_dipole"] * abs(buy_vol - sell_vol)

        # Spatial dipole: buy vs sell center of mass in price
        buy_com_price = np.mean(buy_prices)
        sell_com_price = np.mean(sell_prices)
        features["price_dipole"] = (buy_com_price - sell_com_price) / vwap * 10000 if vwap > 0 else 0.0
    else:
        features["temporal_dipole"] = 0.0
        features["dipole_strength"] = 0.0
        features["price_dipole"] = 0.0

    # Flux = net signed volume through VWAP level
    above_vwap = signed_vol[prices >= vwap].sum() if vwap > 0 else 0.0
    below_vwap = signed_vol[prices < vwap].sum() if vwap > 0 else 0.0
    features["vwap_flux"] = above_vwap - below_vwap
    features["vwap_flux_normalized"] = features["vwap_flux"] / total_vol if total_vol > 0 else 0.0

    # -----------------------------------------------------------------------
    # 27. Physics-inspired — Fluid Dynamics
    # -----------------------------------------------------------------------
    # Viscosity = resistance to flow = effective spread / volume rate
    vol_rate = total_vol / candle_duration_s if candle_duration_s > 0 else 0.0
    eff_spread = features.get("effective_spread_bps", 0.0)
    features["market_viscosity"] = eff_spread / vol_rate if vol_rate > 0 else 0.0

    # Reynolds number = inertial / viscous = (volume × |return|) / (spread × duration)
    if eff_spread > 0 and candle_duration_s > 0:
        features["reynolds_number"] = (total_vol * abs(candle_ret)) / (eff_spread * candle_duration_s / 10000)
    else:
        features["reynolds_number"] = 0.0

    # Turbulence intensity = vol_of_vol / realized_vol
    rv = features.get("realized_vol", 0.0)
    vov = features.get("vol_of_vol", 0.0)
    features["turbulence"] = vov / rv if rv > 0 else 0.0

    # Flow velocity = net volume delta / cross-sectional area (range)
    features["flow_velocity"] = (buy_vol - sell_vol) / price_range if price_range > 0 else 0.0

    # Bernoulli pressure: P + ½ρv² = const → trade-off between pressure and velocity
    # High pressure + low velocity = about to move; low pressure + high velocity = moving
    features["bernoulli"] = features["market_pressure"] + 0.5 * total_vol * (candle_ret ** 2)

    # -----------------------------------------------------------------------
    # 28. Physics-inspired — Wave Physics
    # -----------------------------------------------------------------------
    # Amplitude = range / 2
    features["wave_amplitude"] = price_range / 2

    # Frequency = zero-crossings of tick returns (oscillation rate)
    if len(valid_returns) > 5:
        sign_changes = np.diff(np.sign(valid_returns))
        zero_crossings = np.count_nonzero(sign_changes)
        features["wave_frequency"] = zero_crossings / candle_duration_s if candle_duration_s > 0 else 0.0
        features["zero_crossing_rate"] = zero_crossings / len(valid_returns)
    else:
        features["wave_frequency"] = 0.0
        features["zero_crossing_rate"] = 0.0

    # Wavelength = avg distance between direction changes (in seconds)
    if n > 5:
        price_dir = np.sign(np.diff(prices))
        dir_changes = np.where(np.diff(price_dir) != 0)[0]
        if len(dir_changes) >= 2:
            avg_wavelength_ticks = np.mean(np.diff(dir_changes))
            features["wave_wavelength"] = avg_wavelength_ticks * (candle_duration_s / n) if n > 0 else 0.0
        else:
            features["wave_wavelength"] = candle_duration_s  # no oscillation = one long wave
    else:
        features["wave_wavelength"] = 0.0

    # Standing wave ratio = |CVD_max| / |CVD_min| (reflection coefficient)
    cvd_abs_max = abs(features.get("cvd_max", 0.0))
    cvd_abs_min = abs(features.get("cvd_min", 0.0))
    if cvd_abs_min > 0:
        features["standing_wave_ratio"] = cvd_abs_max / cvd_abs_min
    elif cvd_abs_max > 0:
        features["standing_wave_ratio"] = cvd_abs_max  # infinite SWR → use max
    else:
        features["standing_wave_ratio"] = 1.0

    # Wave energy = amplitude² × frequency² (proportional to energy in a wave)
    features["wave_energy"] = (features["wave_amplitude"] ** 2) * (features["wave_frequency"] ** 2)

    # -----------------------------------------------------------------------
    # 29. Physics-inspired — Gravity / Orbital Mechanics
    # -----------------------------------------------------------------------
    # Gravitational pull to VWAP = volume / distance² (inverse square law)
    dist_to_vwap = abs(c - vwap) if vwap > 0 else 1.0
    if dist_to_vwap > 0:
        features["vwap_gravity"] = total_vol / (dist_to_vwap ** 2)
    else:
        features["vwap_gravity"] = total_vol  # at VWAP = max gravity

    # Escape velocity from value area
    # v_escape = sqrt(2 × G × M / r) → sqrt(2 × volume × range / VA_width)
    va_width = features.get("value_area_width_bps", 0.0) / 10000 * vwap if vwap > 0 else 0.0
    if va_width > 0:
        features["escape_velocity"] = np.sqrt(2 * total_vol * price_range / va_width)
    else:
        features["escape_velocity"] = 0.0

    # Orbital energy = kinetic + potential relative to fair value
    fv = features.get("fair_value", c)
    dist_to_fv = abs(c - fv) / fv if fv > 0 else 0.0
    features["orbital_energy"] = 0.5 * total_vol * candle_ret ** 2 - total_vol * dist_to_fv

    # Gravitational binding energy: how "bound" is price to the value area?
    # High binding = price stuck in VA, low binding = breaking out
    if features.get("value_area_width_bps", 0.0) > 0:
        features["binding_energy"] = total_vol / features["value_area_width_bps"]
    else:
        features["binding_energy"] = 0.0

    # Centripetal acceleration: rate of price oscillation around VWAP
    if n > 10 and vwap > 0:
        deviations = prices - vwap
        features["centripetal_accel"] = float(np.std(deviations) / vwap * 10000)
    else:
        features["centripetal_accel"] = 0.0

    # -----------------------------------------------------------------------
    # 30. Psychology — Anchoring Bias
    # -----------------------------------------------------------------------
    # Distance from nearest round number ($100, $500, $1000 increments)
    for rnd, label in [(100, "100"), (500, "500"), (1000, "1k")]:
        nearest_round = round(c / rnd) * rnd
        features[f"anchor_dist_{label}_bps"] = (c - nearest_round) / c * 10000 if c > 0 else 0.0

    # Magnetic effect: how much time did price spend near round numbers?
    if n >= 20 and c > 0:
        near_round_100 = np.abs(prices % 100)
        # Within 0.1% of a round $100 level
        threshold = c * 0.001
        features["round_level_magnet_pct"] = (near_round_100 < threshold).sum() / n
    else:
        features["round_level_magnet_pct"] = 0.0

    # Open anchoring: distance from candle open (how far have we drifted?)
    features["drift_from_open_bps"] = ((c - o) / o * 10000) if o > 0 else 0.0

    # -----------------------------------------------------------------------
    # 31. Psychology — Loss Aversion / Prospect Theory
    # -----------------------------------------------------------------------
    # Sell urgency vs buy urgency: avg inter-trade time for sells vs buys
    if buy_count >= 5 and sell_count >= 5 and n > 10:
        buy_times = timestamps_s[buy_mask]
        sell_times = timestamps_s[sell_mask]
        buy_gaps = np.diff(buy_times)
        sell_gaps = np.diff(sell_times)
        avg_buy_gap = np.mean(buy_gaps) if len(buy_gaps) > 0 else candle_duration_s
        avg_sell_gap = np.mean(sell_gaps) if len(sell_gaps) > 0 else candle_duration_s
        # Urgency ratio: < 1 means sells are more urgent (faster)
        features["sell_urgency_ratio"] = avg_sell_gap / avg_buy_gap if avg_buy_gap > 0 else 1.0
    else:
        features["sell_urgency_ratio"] = 1.0

    # Panic selling: sell volume acceleration when price is dropping
    if n >= 20:
        # Split into down-move ticks and up-move ticks
        down_ticks = (tick_returns < 0) & ~np.isnan(tick_returns)
        up_ticks = (tick_returns > 0) & ~np.isnan(tick_returns)
        sell_on_down = sizes[down_ticks & sell_mask].sum()
        sell_on_up = sizes[up_ticks & sell_mask].sum()
        buy_on_up = sizes[up_ticks & buy_mask].sum()
        buy_on_down = sizes[down_ticks & buy_mask].sum()
        total_directional = sell_on_down + sell_on_up + buy_on_up + buy_on_down
        if total_directional > 0:
            # Panic = sells accelerate on down-moves
            features["panic_sell_ratio"] = sell_on_down / total_directional
            # FOMO buy = buys accelerate on up-moves
            features["fomo_buy_ratio"] = buy_on_up / total_directional
            # Contrarian = buys on down, sells on up
            features["contrarian_ratio"] = (buy_on_down + sell_on_up) / total_directional
        else:
            features["panic_sell_ratio"] = 0.25
            features["fomo_buy_ratio"] = 0.25
            features["contrarian_ratio"] = 0.5
    else:
        features["panic_sell_ratio"] = 0.25
        features["fomo_buy_ratio"] = 0.25
        features["contrarian_ratio"] = 0.5

    # Loss aversion asymmetry: volume on down-moves / volume on up-moves
    if n >= 20:
        vol_on_down = sizes[down_ticks].sum()
        vol_on_up = sizes[up_ticks].sum()
        total_dir_vol = vol_on_down + vol_on_up
        features["loss_aversion_ratio"] = vol_on_down / vol_on_up if vol_on_up > 0 else 1.0
        features["down_move_vol_pct"] = vol_on_down / total_dir_vol if total_dir_vol > 0 else 0.5
    else:
        features["loss_aversion_ratio"] = 1.0
        features["down_move_vol_pct"] = 0.5

    # -----------------------------------------------------------------------
    # 32. Psychology — Herding / FOMO
    # -----------------------------------------------------------------------
    # Volume-price feedback: correlation between |price change| and subsequent volume
    if n >= 30:
        abs_ret = np.abs(tick_returns[1:])  # skip first NaN
        next_vol = sizes[1:]
        valid_fb = ~np.isnan(abs_ret)
        if valid_fb.sum() > 10:
            features["vol_price_feedback"] = _safe_corr(
                abs_ret[valid_fb], next_vol[valid_fb]
            )
        else:
            features["vol_price_feedback"] = 0.0
    else:
        features["vol_price_feedback"] = 0.0

    # Momentum chasing: buy volume after up-ticks / sell volume after down-ticks
    if n >= 20:
        # Shift: does the NEXT trade chase the current tick direction?
        prev_ret = tick_returns[:-1]
        next_side = sides[1:]
        next_size = sizes[1:]
        valid_chase = ~np.isnan(prev_ret)
        if valid_chase.sum() > 10:
            chase_buy = next_size[(prev_ret > 0) & (next_side == 1) & valid_chase].sum()
            chase_sell = next_size[(prev_ret < 0) & (next_side == -1) & valid_chase].sum()
            contra_buy = next_size[(prev_ret < 0) & (next_side == 1) & valid_chase].sum()
            contra_sell = next_size[(prev_ret > 0) & (next_side == -1) & valid_chase].sum()
            total_chase = chase_buy + chase_sell + contra_buy + contra_sell
            features["herding_ratio"] = (chase_buy + chase_sell) / total_chase if total_chase > 0 else 0.5
        else:
            features["herding_ratio"] = 0.5
    else:
        features["herding_ratio"] = 0.5

    # Trade arrival acceleration after moves: do big moves attract more trades?
    if n >= 30:
        # Split candle into 10 time slices, measure correlation(|return|, next_slice_trade_count)
        n_slices = 10
        slice_size = n // n_slices
        slice_rets = []
        slice_next_counts = []
        for i in range(n_slices - 1):
            s_start = i * slice_size
            s_end = (i + 1) * slice_size
            s_ret = abs(np.log(prices[s_end] / prices[s_start])) if prices[s_start] > 0 else 0.0
            next_count = min((i + 2) * slice_size, n) - (i + 1) * slice_size
            slice_rets.append(s_ret)
            slice_next_counts.append(next_count)
        if len(slice_rets) >= 5 and np.std(slice_rets) > 0:
            features["attention_effect"] = _safe_corr(slice_rets, slice_next_counts)
        else:
            features["attention_effect"] = 0.0
    else:
        features["attention_effect"] = 0.0

    # -----------------------------------------------------------------------
    # 33. Psychology — Disposition Effect
    # -----------------------------------------------------------------------
    # Profit-taking at highs: sell volume near candle high vs buy volume near candle low
    if n >= 20 and price_range > 0:
        price_pct = (prices - l) / price_range  # 0=low, 1=high
        near_high = price_pct >= 0.8
        near_low = price_pct <= 0.2

        sell_at_high = sizes[near_high & sell_mask].sum()
        buy_at_low = sizes[near_low & buy_mask].sum()
        sell_at_low = sizes[near_low & sell_mask].sum()
        buy_at_high = sizes[near_high & buy_mask].sum()

        # Disposition = sell winners + hold losers
        disposition_sell = sell_at_high / (sell_at_high + sell_at_low) if (sell_at_high + sell_at_low) > 0 else 0.5
        disposition_buy = buy_at_low / (buy_at_low + buy_at_high) if (buy_at_low + buy_at_high) > 0 else 0.5
        features["disposition_sell"] = disposition_sell  # high = selling at highs (profit-taking)
        features["disposition_buy"] = disposition_buy    # high = buying at lows (bargain-hunting)
        features["disposition_effect"] = disposition_sell - (1 - disposition_buy)  # net disposition
    else:
        features["disposition_sell"] = 0.5
        features["disposition_buy"] = 0.5
        features["disposition_effect"] = 0.0

    # -----------------------------------------------------------------------
    # 34. Psychology — Regret Aversion / Hesitation / Decision Fatigue
    # -----------------------------------------------------------------------
    # Trade size shrinkage after large moves
    if n >= 30 and len(valid_returns) >= 20:
        abs_ret_arr = np.abs(tick_returns)
        # After big moves (>2σ), do trade sizes shrink?
        ret_std_val = np.std(valid_returns)
        if ret_std_val > 0:
            big_move_mask = abs_ret_arr > 2 * ret_std_val
            big_move_mask[0] = False  # skip first
            # Average size of trades AFTER big moves vs normal
            after_big = np.zeros(n, dtype=bool)
            big_indices = np.where(big_move_mask)[0]
            for bi in big_indices:
                if bi + 1 < n:
                    after_big[bi + 1] = True
            if after_big.sum() >= 3 and (~after_big).sum() >= 3:
                avg_size_after_big = sizes[after_big].mean()
                avg_size_normal = sizes[~after_big].mean()
                features["post_shock_size_ratio"] = avg_size_after_big / avg_size_normal if avg_size_normal > 0 else 1.0
            else:
                features["post_shock_size_ratio"] = 1.0
        else:
            features["post_shock_size_ratio"] = 1.0
    else:
        features["post_shock_size_ratio"] = 1.0

    # Decision fatigue: declining trade rate through candle (regression slope)
    if n >= 20:
        n_quarters = 4
        q_size = n // n_quarters
        q_rates = []
        for qi in range(n_quarters):
            qs = qi * q_size
            qe = (qi + 1) * q_size if qi < n_quarters - 1 else n
            q_duration = timestamps_s[min(qe - 1, n - 1)] - timestamps_s[qs]
            q_count = qe - qs
            q_rates.append(q_count / q_duration if q_duration > 0 else 0.0)
        if np.std(q_rates) > 0:
            features["trade_rate_decay"] = float(np.polyfit(range(len(q_rates)), q_rates, 1)[0])
        else:
            features["trade_rate_decay"] = 0.0
    else:
        features["trade_rate_decay"] = 0.0

    # Hesitation: inter-trade time increase after volatility (post-shock pause)
    if n >= 30 and len(valid_returns) >= 20:
        it_arr = np.diff(timestamps_s)
        if ret_std_val > 0 and len(it_arr) > 10:
            big_ret_mask_it = np.abs(tick_returns[1:-1]) > 2 * ret_std_val
            if big_ret_mask_it.sum() >= 3:
                pause_after = it_arr[1:][big_ret_mask_it[:len(it_arr) - 1]].mean() if big_ret_mask_it[:len(it_arr) - 1].sum() > 0 else 0.0
                normal_gap = it_arr.mean()
                features["post_shock_pause_ratio"] = pause_after / normal_gap if normal_gap > 0 else 1.0
            else:
                features["post_shock_pause_ratio"] = 1.0
        else:
            features["post_shock_pause_ratio"] = 1.0
    else:
        features["post_shock_pause_ratio"] = 1.0

    # -----------------------------------------------------------------------
    # 35. Psychology — Overconfidence
    # -----------------------------------------------------------------------
    # Size escalation: are trade sizes growing through the candle?
    if n >= 20:
        half1_avg_size = sizes[:mid_idx].mean() if mid_idx > 0 else sizes.mean()
        half2_avg_size = sizes[mid_idx:].mean() if mid_idx < n else sizes.mean()
        features["size_escalation"] = half2_avg_size / half1_avg_size if half1_avg_size > 0 else 1.0
    else:
        features["size_escalation"] = 1.0

    # Large trade concentration: Gini coefficient of trade sizes
    if n >= 20:
        sorted_sizes = np.sort(sizes)
        cumsum = np.cumsum(sorted_sizes)
        n_s = len(sorted_sizes)
        features["size_gini"] = float(1 - 2 * np.sum(cumsum) / (n_s * cumsum[-1]) + 1 / n_s) if cumsum[-1] > 0 else 0.0
    else:
        features["size_gini"] = 0.0

    # Aggressive pricing: fraction of trades at extreme prices (top/bottom 10% of range)
    if n >= 20 and price_range > 0:
        price_pct_all = (prices - l) / price_range
        extreme_trades = ((price_pct_all >= 0.9) | (price_pct_all <= 0.1)).sum()
        features["extreme_price_pct"] = extreme_trades / n
    else:
        features["extreme_price_pct"] = 0.0

    # -----------------------------------------------------------------------
    # 36. Psychology — Micro Fear & Greed Index
    # -----------------------------------------------------------------------
    # Composite of fear vs greed signals within the candle
    # Fear signals: sell urgency, panic selling, spread widening, vol spike
    # Greed signals: buy urgency, FOMO buying, spread tightening, momentum
    fear_score = 0.0
    greed_score = 0.0

    # Sell urgency (fear)
    su = features.get("sell_urgency_ratio", 1.0)
    if su < 0.8:
        fear_score += (0.8 - su) / 0.8  # faster sells = more fear
    elif su > 1.2:
        greed_score += (su - 1.2) / 1.2  # faster buys = more greed

    # Panic/FOMO ratios
    fear_score += max(0, features.get("panic_sell_ratio", 0.25) - 0.3)
    greed_score += max(0, features.get("fomo_buy_ratio", 0.25) - 0.3)

    # Volatility heating (fear = heating up)
    hr = features.get("heating_rate", 0.0)
    if hr > 0.2:
        fear_score += min(hr, 1.0)
    elif hr < -0.2:
        greed_score += min(-hr, 1.0)

    # Order flow (greed = strong buying)
    ofi = features.get("order_flow_imbalance", 0.0)
    if ofi > 0.2:
        greed_score += ofi
    elif ofi < -0.2:
        fear_score += -ofi

    total_fg = fear_score + greed_score
    if total_fg > 0:
        features["micro_fear_greed"] = (greed_score - fear_score) / total_fg  # [-1, 1]: -1=fear, +1=greed
    else:
        features["micro_fear_greed"] = 0.0
    features["micro_fear_score"] = fear_score
    features["micro_greed_score"] = greed_score

    # -----------------------------------------------------------------------
    # 37. Psychology — Attention / Surprise
    # -----------------------------------------------------------------------
    # Volume response to price shocks: volume surge after big moves
    if n >= 30 and len(valid_returns) >= 20 and ret_std_val > 0:
        shock_mask = np.abs(tick_returns) > 2 * ret_std_val
        shock_mask[0] = False
        shock_indices = np.where(shock_mask)[0]
        if len(shock_indices) >= 2:
            # Average volume in 5 trades after shock vs 5 trades before
            post_vols = []
            pre_vols = []
            for si in shock_indices:
                if si >= 5 and si + 5 < n:
                    post_vols.append(sizes[si + 1:si + 6].mean())
                    pre_vols.append(sizes[si - 5:si].mean())
            if len(post_vols) >= 2:
                avg_post = np.mean(post_vols)
                avg_pre = np.mean(pre_vols)
                features["shock_attention_ratio"] = avg_post / avg_pre if avg_pre > 0 else 1.0
            else:
                features["shock_attention_ratio"] = 1.0
        else:
            features["shock_attention_ratio"] = 1.0
    else:
        features["shock_attention_ratio"] = 1.0

    # Rubber-necking: volume at extreme prices vs volume at normal prices
    if n >= 20 and price_range > 0:
        price_pct_rn = (prices - l) / price_range
        extreme_mask_rn = (price_pct_rn >= 0.85) | (price_pct_rn <= 0.15)
        normal_mask_rn = ~extreme_mask_rn
        vol_extreme = sizes[extreme_mask_rn].sum()
        vol_normal = sizes[normal_mask_rn].sum()
        features["rubberneck_ratio"] = vol_extreme / vol_normal if vol_normal > 0 else 1.0
    else:
        features["rubberneck_ratio"] = 1.0

    # Surprise magnitude: how much did the candle deviate from "expected" move?
    # Expected = 0 (random walk), so surprise = |return| / realized_vol
    rv_val = features.get("realized_vol", 0.0)
    if rv_val > 0:
        features["surprise_magnitude"] = abs(candle_ret) / rv_val
    else:
        features["surprise_magnitude"] = 0.0

    # -----------------------------------------------------------------------
    # 38. Trade Event Temporal Distribution
    # -----------------------------------------------------------------------
    if n >= 20 and candle_duration_s > 0:
        # --- Temporal centroid (center of mass of trades in time) ---
        # 0.0 = all at start, 0.5 = uniform, 1.0 = all at end
        features["activity_centroid"] = float(np.mean(rel_time))
        features["activity_centroid_vol"] = float(np.average(rel_time, weights=sizes))

        # Buy vs sell centroid: who trades earlier/later?
        if buy_count >= 5:
            features["buy_centroid"] = float(np.mean(rel_time[buy_mask]))
        else:
            features["buy_centroid"] = 0.5
        if sell_count >= 5:
            features["sell_centroid"] = float(np.mean(rel_time[sell_mask]))
        else:
            features["sell_centroid"] = 0.5
        features["centroid_buy_sell_gap"] = features["buy_centroid"] - features["sell_centroid"]

        # --- Spikiness vs flatness ---
        # Split candle into 20 time bins, count trades per bin
        n_tbins = 20
        tbin_edges = np.linspace(0, 1, n_tbins + 1)
        tbin_idx = np.clip(np.digitize(rel_time, tbin_edges) - 1, 0, n_tbins - 1)

        tbin_counts = np.zeros(n_tbins)
        np.add.at(tbin_counts, tbin_idx, 1)
        tbin_vol = np.zeros(n_tbins)
        np.add.at(tbin_vol, tbin_idx, sizes)

        mean_count = tbin_counts.mean()
        max_count = tbin_counts.max()
        features["activity_peak_to_mean"] = max_count / mean_count if mean_count > 0 else 1.0

        mean_vol_bin = tbin_vol.mean()
        max_vol_bin = tbin_vol.max()
        features["volume_peak_to_mean"] = max_vol_bin / mean_vol_bin if mean_vol_bin > 0 else 1.0

        # Temporal kurtosis of trade arrival times
        features["arrival_time_kurtosis"] = _fast_kurtosis(rel_time)

        # Burstiness coefficient: (σ - μ) / (σ + μ) of inter-trade times
        # -1 = perfectly periodic, 0 = Poisson, +1 = extremely bursty
        it_all = np.diff(timestamps_s)
        it_pos = it_all[it_all > 0]
        if len(it_pos) >= 5:
            it_mean = np.mean(it_pos)
            it_std = np.std(it_pos)
            features["burstiness"] = (it_std - it_mean) / (it_std + it_mean) if (it_std + it_mean) > 0 else 0.0
        else:
            features["burstiness"] = 0.0

        # --- Quartile analysis ---
        q_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        q_counts = np.zeros(4)
        q_vols = np.zeros(4)
        for qi in range(4):
            q_mask = (rel_time >= q_edges[qi]) & (rel_time < q_edges[qi + 1])
            if qi == 3:  # include endpoint
                q_mask = (rel_time >= q_edges[qi]) & (rel_time <= q_edges[qi + 1])
            q_counts[qi] = q_mask.sum()
            q_vols[qi] = sizes[q_mask].sum()

        total_q_counts = q_counts.sum()
        total_q_vol = q_vols.sum()
        for qi, ql in enumerate(["q1", "q2", "q3", "q4"]):
            features[f"trade_pct_{ql}"] = q_counts[qi] / total_q_counts if total_q_counts > 0 else 0.25
            features[f"volume_pct_{ql}"] = q_vols[qi] / total_q_vol if total_q_vol > 0 else 0.25

        # Max/min quartile ratio
        if q_counts.min() > 0:
            features["quartile_count_ratio"] = q_counts.max() / q_counts.min()
        else:
            features["quartile_count_ratio"] = q_counts.max() / max(q_counts.min(), 1)
        if q_vols.min() > 0:
            features["quartile_vol_ratio"] = q_vols.max() / q_vols.min()
        else:
            features["quartile_vol_ratio"] = q_vols.max() / max(q_vols.min(), 1e-12)

        # Which quartile is busiest? (0-3 → 1-4)
        features["busiest_quartile"] = int(np.argmax(q_counts)) + 1
        features["busiest_vol_quartile"] = int(np.argmax(q_vols)) + 1

        # Front-heavy vs back-heavy: (Q1+Q2 vol) / (Q3+Q4 vol)
        front_vol = q_vols[0] + q_vols[1]
        back_vol = q_vols[2] + q_vols[3]
        features["front_back_vol_ratio"] = front_vol / back_vol if back_vol > 0 else 1.0

        # Middle-heavy: (Q2+Q3) / (Q1+Q4)
        middle_vol = q_vols[1] + q_vols[2]
        edge_vol = q_vols[0] + q_vols[3]
        features["middle_edge_vol_ratio"] = middle_vol / edge_vol if edge_vol > 0 else 1.0

        # --- Temporal entropy ---
        # How uniform is the trade distribution across time bins?
        count_probs = tbin_counts / tbin_counts.sum() if tbin_counts.sum() > 0 else np.ones(n_tbins) / n_tbins
        count_probs = count_probs[count_probs > 0]
        max_entropy = np.log2(n_tbins)
        features["trade_time_entropy"] = float(-np.sum(count_probs * np.log2(count_probs)))
        features["trade_time_uniformity"] = features["trade_time_entropy"] / max_entropy if max_entropy > 0 else 1.0

        vol_probs = tbin_vol / tbin_vol.sum() if tbin_vol.sum() > 0 else np.ones(n_tbins) / n_tbins
        vol_probs = vol_probs[vol_probs > 0]
        features["volume_time_entropy"] = float(-np.sum(vol_probs * np.log2(vol_probs)))
        features["volume_time_uniformity"] = features["volume_time_entropy"] / max_entropy if max_entropy > 0 else 1.0

        # --- Activity acceleration profile ---
        # Linear regression of bin counts over time → slope = ramp direction
        bin_x = np.arange(n_tbins, dtype=float)
        if np.std(tbin_counts) > 0:
            features["activity_ramp"] = float(np.polyfit(bin_x, tbin_counts, 1)[0])
        else:
            features["activity_ramp"] = 0.0
        if np.std(tbin_vol) > 0:
            features["volume_ramp"] = float(np.polyfit(bin_x, tbin_vol, 1)[0])
        else:
            features["volume_ramp"] = 0.0

        # Curvature: fit quadratic, coefficient of x² tells us if ramp is accelerating
        if np.std(tbin_counts) > 0 and n_tbins >= 5:
            poly = np.polyfit(bin_x, tbin_counts, 2)
            features["activity_curvature"] = float(poly[0])  # +ve = U-shape, -ve = ∩-shape
        else:
            features["activity_curvature"] = 0.0

    else:
        features["activity_centroid"] = 0.5
        features["activity_centroid_vol"] = 0.5
        features["buy_centroid"] = 0.5
        features["sell_centroid"] = 0.5
        features["centroid_buy_sell_gap"] = 0.0
        features["activity_peak_to_mean"] = 1.0
        features["volume_peak_to_mean"] = 1.0
        features["arrival_time_kurtosis"] = 0.0
        features["burstiness"] = 0.0
        for ql in ["q1", "q2", "q3", "q4"]:
            features[f"trade_pct_{ql}"] = 0.25
            features[f"volume_pct_{ql}"] = 0.25
        features["quartile_count_ratio"] = 1.0
        features["quartile_vol_ratio"] = 1.0
        features["busiest_quartile"] = 1
        features["busiest_vol_quartile"] = 1
        features["front_back_vol_ratio"] = 1.0
        features["middle_edge_vol_ratio"] = 1.0
        features["trade_time_entropy"] = 0.0
        features["trade_time_uniformity"] = 1.0
        features["volume_time_entropy"] = 0.0
        features["volume_time_uniformity"] = 1.0
        features["activity_ramp"] = 0.0
        features["volume_ramp"] = 0.0
        features["activity_curvature"] = 0.0

    # -----------------------------------------------------------------------
    # 39. Math — Linear Algebra (price-volume covariance structure)
    # -----------------------------------------------------------------------
    if n >= 20 and len(valid_returns) >= 10:
        # 2×2 covariance of (tick_return, log_size)
        tr_valid = tick_returns[1:]  # skip first NaN
        sz_valid = np.log1p(sizes[1:])
        mask_valid = ~np.isnan(tr_valid)
        if mask_valid.sum() >= 10:
            tr_v = tr_valid[mask_valid]
            sz_v = sz_valid[mask_valid]
            cov_mat = np.cov(tr_v, sz_v)
            if cov_mat.shape == (2, 2):
                eigvals = np.linalg.eigvalsh(cov_mat)
                eigvals = np.sort(eigvals)[::-1]  # descending
                features["eigen_ratio"] = eigvals[0] / eigvals[1] if eigvals[1] > 0 else eigvals[0]
                features["eigen_sum"] = float(eigvals[0] + eigvals[1])  # total variance
                # Principal axis angle: arctan of eigenvector direction
                # Positive angle = price and volume move together
                if cov_mat[0, 0] != cov_mat[1, 1]:
                    angle = 0.5 * np.arctan2(2 * cov_mat[0, 1], cov_mat[0, 0] - cov_mat[1, 1])
                else:
                    angle = np.pi / 4 if cov_mat[0, 1] > 0 else -np.pi / 4
                features["pv_principal_angle"] = float(angle)
            else:
                features["eigen_ratio"] = 1.0
                features["eigen_sum"] = 0.0
                features["pv_principal_angle"] = 0.0
        else:
            features["eigen_ratio"] = 1.0
            features["eigen_sum"] = 0.0
            features["pv_principal_angle"] = 0.0
    else:
        features["eigen_ratio"] = 1.0
        features["eigen_sum"] = 0.0
        features["pv_principal_angle"] = 0.0

    # -----------------------------------------------------------------------
    # 40. Math — Geometry (price path shape)
    # -----------------------------------------------------------------------
    if n >= 10 and price_range > 0 and candle_duration_s > 0:
        # Normalize to [0,1] × [0,1] for shape analysis
        norm_time = rel_time
        norm_price = (prices - l) / price_range

        # Convex hull area (using Shoelace formula on sorted boundary)
        try:
            pts = np.column_stack([norm_time, norm_price])
            if len(np.unique(pts, axis=0)) >= 3:
                hull = ConvexHull(pts)
                features["convex_hull_area"] = float(hull.volume)  # 2D: volume = area
                # Path efficiency: how much of the hull does the path fill?
                # Approximate path area using trapezoidal rule
                path_area = float(np.trapezoid(norm_price, norm_time))
                features["path_hull_ratio"] = path_area / hull.volume if hull.volume > 0 else 0.0
            else:
                features["convex_hull_area"] = 0.0
                features["path_hull_ratio"] = 0.0
        except Exception:
            features["convex_hull_area"] = 0.0
            features["path_hull_ratio"] = 0.0

        # Triangle area from OHLC: area of triangle (open, high, low)
        # Normalized: open at t=0, close at t=1, high/low at their relative times
        o_norm = (o - l) / price_range
        c_norm = (c - l) / price_range
        # Shoelace for triangle (0, o_norm), (0.5, 1), (1, c_norm)
        features["ohlc_triangle_area"] = abs(0 * (1 - c_norm) + 0.5 * (c_norm - o_norm) + 1 * (o_norm - 1)) / 2

        # Aspect ratio: range / duration (in normalized units → bps per second)
        features["candle_aspect_ratio"] = (price_range / vwap * 10000) / candle_duration_s if vwap > 0 else 0.0
    else:
        features["convex_hull_area"] = 0.0
        features["path_hull_ratio"] = 0.0
        features["ohlc_triangle_area"] = 0.0
        features["candle_aspect_ratio"] = 0.0

    # -----------------------------------------------------------------------
    # 41. Math — Angles & Slopes
    # -----------------------------------------------------------------------
    # Price slope angle (in degrees): steepness of the move
    if candle_duration_s > 0 and vwap > 0:
        # Normalize return to bps, time to minutes
        ret_bps = candle_ret * 10000
        duration_min = candle_duration_s / 60
        features["price_slope_angle"] = float(np.degrees(np.arctan2(ret_bps, duration_min)))
    else:
        features["price_slope_angle"] = 0.0

    # Angle between price direction and volume direction
    # Price vector: (1, return), Volume vector: (1, volume_change_normalized)
    if n >= 20 and candle_duration_s > 0:
        vol_h1_t = sizes[:mid_idx].sum()
        vol_h2_t = sizes[mid_idx:].sum()
        vol_change = (vol_h2_t - vol_h1_t) / (vol_h1_t + vol_h2_t) if (vol_h1_t + vol_h2_t) > 0 else 0.0
        # Cosine of angle between (1, return) and (1, vol_change)
        dot = 1 + candle_ret * vol_change
        mag1 = np.sqrt(1 + candle_ret ** 2)
        mag2 = np.sqrt(1 + vol_change ** 2)
        features["price_volume_angle_cos"] = dot / (mag1 * mag2) if (mag1 * mag2) > 0 else 0.0
    else:
        features["price_volume_angle_cos"] = 0.0

    # -----------------------------------------------------------------------
    # 42. Math — Distance Metrics
    # -----------------------------------------------------------------------
    if n >= 10:
        # Manhattan distance of price path (sum of |Δprice| + |Δtime|)
        dp = np.abs(np.diff(prices))
        dt = np.abs(np.diff(timestamps_s))
        # Normalize price to same scale as time
        if vwap > 0 and candle_duration_s > 0:
            dp_norm = dp / vwap * candle_duration_s  # scale price changes to time units
            features["manhattan_distance"] = float(np.sum(dp_norm + dt))
            features["chebyshev_distance"] = float(np.max(np.maximum(dp_norm, dt)))
        else:
            features["manhattan_distance"] = 0.0
            features["chebyshev_distance"] = 0.0

        # Cosine similarity between first-half and second-half price vectors
        p_h1 = prices[:mid_idx]
        p_h2 = prices[mid_idx:]
        if len(p_h1) >= 5 and len(p_h2) >= 5:
            # Resample to same length for comparison
            min_len = min(len(p_h1), len(p_h2))
            v1 = np.diff(p_h1[:min_len])
            v2 = np.diff(p_h2[:min_len])
            if len(v1) > 0 and len(v2) > 0:
                dot_p = np.dot(v1, v2)
                m1 = np.linalg.norm(v1)
                m2 = np.linalg.norm(v2)
                features["half_cosine_similarity"] = float(dot_p / (m1 * m2)) if (m1 * m2) > 0 else 0.0
            else:
                features["half_cosine_similarity"] = 0.0
        else:
            features["half_cosine_similarity"] = 0.0
    else:
        features["manhattan_distance"] = 0.0
        features["chebyshev_distance"] = 0.0
        features["half_cosine_similarity"] = 0.0

    # -----------------------------------------------------------------------
    # 43. Math — Calculus (integrals, derivatives)
    # -----------------------------------------------------------------------
    if n >= 10 and candle_duration_s > 0 and vwap > 0:
        # Signed area: integral of (price - VWAP) over time
        deviation = prices - vwap
        # Trapezoidal integration over relative time
        features["signed_area"] = float(np.trapezoid(deviation, rel_time))
        features["signed_area_bps"] = features["signed_area"] / vwap * 10000

        # Positive area vs negative area (above/below VWAP)
        pos_dev = np.maximum(deviation, 0)
        neg_dev = np.minimum(deviation, 0)
        pos_area = float(np.trapezoid(pos_dev, rel_time))
        neg_area = float(np.trapezoid(neg_dev, rel_time))
        total_area = pos_area - neg_area  # both positive
        features["area_above_vwap_pct"] = pos_area / total_area if total_area > 0 else 0.5

        # Jerk: third derivative of price (rate of change of acceleration)
        if n >= 12:
            t1, t2, t3, t4 = n // 4, n // 2, 3 * n // 4, n - 1
            r1 = np.log(prices[t1] / prices[0]) if prices[0] > 0 else 0.0
            r2 = np.log(prices[t2] / prices[t1]) if prices[t1] > 0 else 0.0
            r3 = np.log(prices[t3] / prices[t2]) if prices[t2] > 0 else 0.0
            r4 = np.log(prices[t4] / prices[t3]) if prices[t3] > 0 else 0.0
            accel_1 = r2 - r1
            accel_2 = r4 - r3
            features["price_jerk"] = accel_2 - accel_1  # change in acceleration
        else:
            features["price_jerk"] = 0.0
    else:
        features["signed_area"] = 0.0
        features["signed_area_bps"] = 0.0
        features["area_above_vwap_pct"] = 0.5
        features["price_jerk"] = 0.0

    # -----------------------------------------------------------------------
    # 44. Math — Topology / Shape Analysis
    # -----------------------------------------------------------------------
    if n >= 10:
        # Number of local extrema (peaks + valleys)
        if len(prices) >= 3:
            dp_sign = np.sign(np.diff(prices))
            dp_sign_nz = dp_sign[dp_sign != 0]  # remove flats
            if len(dp_sign_nz) >= 2:
                direction_changes = np.sum(np.diff(dp_sign_nz) != 0)
                features["num_extrema"] = int(direction_changes)
                features["extrema_density"] = direction_changes / candle_duration_s if candle_duration_s > 0 else 0.0
            else:
                features["num_extrema"] = 0
                features["extrema_density"] = 0.0
        else:
            features["num_extrema"] = 0
            features["extrema_density"] = 0.0

        # Monotonicity score: longest monotonic run / total ticks
        if len(prices) >= 3:
            dp_dir = np.sign(np.diff(prices))
            up_run = _max_run(dp_dir > 0)
            down_run = _max_run(dp_dir < 0)
            features["max_monotonic_run"] = max(up_run, down_run)
            features["monotonicity_score"] = features["max_monotonic_run"] / (n - 1) if n > 1 else 0.0
        else:
            features["max_monotonic_run"] = 0
            features["monotonicity_score"] = 0.0

        # Tortuosity: path length / straight-line distance
        straight_dist = abs(prices[-1] - prices[0])
        if straight_dist > 0:
            features["tortuosity"] = features.get("price_path_length", 0.0) / straight_dist
        else:
            features["tortuosity"] = 1.0
    else:
        features["num_extrema"] = 0
        features["extrema_density"] = 0.0
        features["max_monotonic_run"] = 0
        features["monotonicity_score"] = 0.0
        features["tortuosity"] = 1.0

    # -----------------------------------------------------------------------
    # 45. Deep Fibonacci / Golden Ratio Analysis
    # -----------------------------------------------------------------------
    phi = 1.618033988749895
    inv_phi = 0.618033988749895  # 1/φ = φ-1
    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    fib_ratios = [phi, inv_phi, phi**2, 1/phi**2, phi**3, 1/phi**3]

    # --- 45a. Fibonacci Price Retracement (close position) ---
    if price_range > 0:
        close_pct = (c - l) / price_range
        fib_distances = [abs(close_pct - fl) for fl in fib_levels]
        features["fib_nearest_dist"] = min(fib_distances)
        features["fib_nearest_level"] = fib_levels[int(np.argmin(fib_distances))]
        features["fib_proximity"] = 1.0 if features["fib_nearest_dist"] < 0.05 else 0.0
    else:
        features["fib_nearest_dist"] = 0.5
        features["fib_nearest_level"] = 0.5
        features["fib_proximity"] = 0.0

    # --- 45b. Intra-candle Fibonacci level interaction ---
    if n >= 20 and price_range > 0:
        price_pct_fib = (prices - l) / price_range  # normalize to [0,1]

        # How many times does price cross each Fib level?
        fib_crosses = 0
        fib_vol_at_level = 0.0
        fib_vol_away = 0.0
        fib_threshold = 0.02  # within 2% of range = "at" a Fib level

        for fl in fib_levels:
            near_fib = np.abs(price_pct_fib - fl) < fib_threshold
            fib_vol_at_level += sizes[near_fib].sum()
            # Count crossings: sign changes of (price_pct - fib_level)
            diff_from_fib = price_pct_fib - fl
            sign_changes = np.diff(np.sign(diff_from_fib))
            fib_crosses += np.count_nonzero(sign_changes)

        fib_vol_away = total_vol - fib_vol_at_level
        features["fib_total_crosses"] = fib_crosses
        features["fib_crosses_per_level"] = fib_crosses / len(fib_levels)
        features["fib_vol_concentration"] = fib_vol_at_level / total_vol if total_vol > 0 else 0.0
        features["fib_vol_ratio"] = fib_vol_at_level / fib_vol_away if fib_vol_away > 0 else 1.0

        # Fib level "respect" score: do prices bounce at Fib levels?
        # Vectorized: precompute direction changes, then check near-Fib points
        before_diff = np.zeros(n)
        after_diff = np.zeros(n)
        before_diff[2:] = prices[2:] - prices[:-2]
        after_diff[:n-2] = prices[2:] - prices[:n-2]
        reversal_mask = (before_diff * after_diff < 0)  # direction reversal at each point

        fib_bounces = 0
        for fl in fib_levels:
            near = np.abs(price_pct_fib - fl) < fib_threshold
            near[0:2] = False
            near[n-2:] = False
            fib_bounces += int((near & reversal_mask).sum())
        features["fib_bounce_count"] = fib_bounces
        features["fib_respect_score"] = fib_bounces / max(fib_crosses, 1)
    else:
        features["fib_total_crosses"] = 0
        features["fib_crosses_per_level"] = 0.0
        features["fib_vol_concentration"] = 0.0
        features["fib_vol_ratio"] = 0.0
        features["fib_bounce_count"] = 0
        features["fib_respect_score"] = 0.0

    # --- 45c. Fibonacci Time Zones ---
    if n >= 20 and candle_duration_s > 0:
        # Fibonacci time fractions: 1/21, 1/13, 1/8, 2/13, 3/13, 5/13, 8/13, 13/21
        fib_time_fracs = [1/21, 1/13, 2/21, 1/8, 3/21, 2/13, 5/21, 3/13,
                          8/21, 5/13, 8/13, 13/21]
        fib_time_fracs = sorted(set([f for f in fib_time_fracs if 0 < f < 1]))

        # Volume at Fibonacci time zones (within 2% of candle duration)
        fib_time_vol = 0.0
        fib_time_trades = 0
        time_threshold = 0.02
        for ftf in fib_time_fracs:
            near_fib_time = np.abs(rel_time - ftf) < time_threshold
            fib_time_vol += sizes[near_fib_time].sum()
            fib_time_trades += near_fib_time.sum()

        features["fib_time_vol_pct"] = fib_time_vol / total_vol if total_vol > 0 else 0.0
        features["fib_time_trade_pct"] = fib_time_trades / n

        # Expected if uniform: len(fib_time_fracs) * 2 * time_threshold * total_vol
        expected_fib_vol = len(fib_time_fracs) * 2 * time_threshold * total_vol
        features["fib_time_vol_surprise"] = fib_time_vol / expected_fib_vol if expected_fib_vol > 0 else 1.0
    else:
        features["fib_time_vol_pct"] = 0.0
        features["fib_time_trade_pct"] = 0.0
        features["fib_time_vol_surprise"] = 1.0

    # --- 45d. Golden Ratio in Volume Splits ---
    if total_vol > 0:
        # Buy/sell volume ratio vs φ
        bs_ratio = buy_vol / sell_vol if sell_vol > 0 else 0.0
        features["golden_ratio_bs_dist"] = min(abs(bs_ratio - phi), abs(bs_ratio - inv_phi))

        # First-half / second-half volume ratio vs φ
        vol_h1_fib = sizes[:mid_idx].sum() if mid_idx > 0 else 0.0
        vol_h2_fib = sizes[mid_idx:].sum() if mid_idx < n else total_vol
        hh_ratio = vol_h1_fib / vol_h2_fib if vol_h2_fib > 0 else 0.0
        features["golden_ratio_half_dist"] = min(abs(hh_ratio - phi), abs(hh_ratio - inv_phi))

        # Quartile volume ratios: how many consecutive pairs are near φ?
        if n >= 20:
            q_size_fib = n // 4
            q_vols_fib = [sizes[i*q_size_fib:(i+1)*q_size_fib].sum() for i in range(4)]
            golden_pairs = 0
            for i in range(3):
                if q_vols_fib[i+1] > 0:
                    qr = q_vols_fib[i] / q_vols_fib[i+1]
                    if min(abs(qr - phi), abs(qr - inv_phi)) < 0.15:
                        golden_pairs += 1
            features["golden_quartile_pairs"] = golden_pairs
        else:
            features["golden_quartile_pairs"] = 0

        # Composite golden ratio proximity (average of all ratio distances)
        features["golden_ratio_composite"] = (
            features["golden_ratio_bs_dist"] +
            features["golden_ratio_half_dist"]
        ) / 2
    else:
        features["golden_ratio_bs_dist"] = 1.0
        features["golden_ratio_half_dist"] = 1.0
        features["golden_quartile_pairs"] = 0
        features["golden_ratio_composite"] = 1.0

    # --- 45e. Fibonacci in Trade Size Ratios ---
    if n >= 30:
        # Consecutive trade size ratios
        size_ratios = sizes[1:] / np.maximum(sizes[:-1], 1e-12)
        # How many consecutive size ratios are near a Fibonacci ratio?
        fib_size_count = 0
        for fr in [phi, inv_phi, phi**2, 1/phi**2]:
            fib_size_count += (np.abs(size_ratios - fr) < 0.1 * fr).sum()
        features["fib_size_ratio_count"] = fib_size_count
        features["fib_size_ratio_pct"] = fib_size_count / len(size_ratios)

        # Largest trade / median trade — distance to nearest Fibonacci number
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        max_med_ratio = sizes.max() / np.median(sizes) if np.median(sizes) > 0 else 0.0
        fib_num_dists = [abs(max_med_ratio - fn) / fn for fn in fib_numbers if fn > 0]
        features["fib_size_max_med_dist"] = min(fib_num_dists) if fib_num_dists else 1.0
    else:
        features["fib_size_ratio_count"] = 0
        features["fib_size_ratio_pct"] = 0.0
        features["fib_size_max_med_dist"] = 1.0

    # --- 45f. Fibonacci Wave Structure (Elliott-like) ---
    if n >= 30 and price_range > 0:
        # Find swing points using vectorized rolling max/min
        min_swing = max(5, n // 50)
        # Subsample prices for swing detection if too many ticks
        if n > 5000:
            step_sw = n // 2500
            p_sw = prices[::step_sw]
            min_swing_sw = max(3, min_swing // step_sw)
        else:
            p_sw = prices
            step_sw = 1
            min_swing_sw = min_swing
        n_sw = len(p_sw)
        swing_prices = []
        swing_types = []
        if n_sw > 2 * min_swing_sw:
            # Vectorized: compute rolling max and min
            win_sz = 2 * min_swing_sw + 1
            if n_sw >= win_sz:
                windows = sliding_window_view(p_sw, win_sz)
                center_idx = min_swing_sw  # center of each window
                center_vals = p_sw[min_swing_sw:min_swing_sw + len(windows)]
                is_max = (center_vals == windows.max(axis=1)) & (center_vals > p_sw[0:len(windows)])
                is_min = (center_vals == windows.min(axis=1)) & (center_vals < p_sw[0:len(windows)])
                # Enforce minimum separation
                last_idx = -min_swing_sw * 2
                for i_sw in range(len(is_max)):
                    if i_sw - last_idx < min_swing_sw:
                        continue
                    if is_max[i_sw]:
                        swing_prices.append(float(center_vals[i_sw]))
                        swing_types.append(1)
                        last_idx = i_sw
                    elif is_min[i_sw]:
                        swing_prices.append(float(center_vals[i_sw]))
                        swing_types.append(-1)
                        last_idx = i_sw

        if len(swing_prices) >= 3:
            # Wave amplitudes: absolute price changes between consecutive swings
            wave_amps = np.abs(np.diff(swing_prices))

            # Consecutive wave amplitude ratios
            if len(wave_amps) >= 2:
                wave_ratios = wave_amps[1:] / np.maximum(wave_amps[:-1], 1e-12)
                # How many wave ratios are near Fibonacci ratios?
                fib_wave_count = 0
                for wr in wave_ratios:
                    for fr in [phi, inv_phi, 0.382, 0.236, 1.0]:
                        if abs(wr - fr) < 0.15 * max(fr, 0.1):
                            fib_wave_count += 1
                            break
                features["fib_wave_ratio_count"] = fib_wave_count
                features["fib_wave_ratio_pct"] = fib_wave_count / len(wave_ratios)

                # Average distance of wave ratios from nearest Fib ratio
                avg_fib_dist = 0.0
                for wr in wave_ratios:
                    min_d = min(abs(wr - fr) for fr in [phi, inv_phi, 0.382, 0.236, 1.0, 2.618])
                    avg_fib_dist += min_d
                features["fib_wave_avg_dist"] = avg_fib_dist / len(wave_ratios)
            else:
                features["fib_wave_ratio_count"] = 0
                features["fib_wave_ratio_pct"] = 0.0
                features["fib_wave_avg_dist"] = 1.0

            features["fib_swing_count"] = len(swing_prices)

            # Up-waves vs down-waves ratio: is it near φ?
            up_waves = sum(1 for i in range(len(wave_amps)) if swing_types[i+1] == 1)
            down_waves = sum(1 for i in range(len(wave_amps)) if swing_types[i+1] == -1)
            if down_waves > 0:
                ud_ratio = up_waves / down_waves
                features["fib_updown_wave_ratio"] = ud_ratio
                features["fib_updown_golden_dist"] = min(abs(ud_ratio - phi), abs(ud_ratio - inv_phi))
            else:
                features["fib_updown_wave_ratio"] = float(up_waves)
                features["fib_updown_golden_dist"] = 1.0
        else:
            features["fib_wave_ratio_count"] = 0
            features["fib_wave_ratio_pct"] = 0.0
            features["fib_wave_avg_dist"] = 1.0
            features["fib_swing_count"] = len(swing_prices)
            features["fib_updown_wave_ratio"] = 1.0
            features["fib_updown_golden_dist"] = 1.0
    else:
        features["fib_wave_ratio_count"] = 0
        features["fib_wave_ratio_pct"] = 0.0
        features["fib_wave_avg_dist"] = 1.0
        features["fib_swing_count"] = 0
        features["fib_updown_wave_ratio"] = 1.0
        features["fib_updown_golden_dist"] = 1.0

    # --- 45g. Golden Angle Distribution (Phyllotaxis) ---
    if n >= 30 and candle_duration_s > 0:
        # Map each trade to an angle on a circle using golden angle (137.508°)
        golden_angle_rad = 2 * np.pi * (1 - 1 / phi)  # ≈ 2.399 rad ≈ 137.508°

        # Ideal golden angle positions for n trades
        ideal_angles = np.mod(np.arange(n) * golden_angle_rad, 2 * np.pi)
        # Actual trade positions mapped to circle by time
        actual_angles = rel_time * 2 * np.pi

        # Circular variance: how well do actual trades follow golden angle spacing?
        # Low variance = good golden angle distribution
        angle_diffs = np.mod(actual_angles - ideal_angles, 2 * np.pi)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
        features["golden_angle_deviation"] = float(np.mean(angle_diffs))
        features["golden_angle_uniformity"] = 1.0 - features["golden_angle_deviation"] / np.pi

        # Sunflower score: how "sunflower-like" is the trade distribution?
        # Perfect sunflower = trades at golden angle intervals
        # Compare actual inter-trade angles to golden angle
        if n >= 10:
            actual_inter = np.diff(actual_angles)
            golden_inter = golden_angle_rad
            inter_diffs = np.abs(np.mod(actual_inter, 2 * np.pi) - np.mod(golden_inter, 2 * np.pi))
            inter_diffs = np.minimum(inter_diffs, 2 * np.pi - inter_diffs)
            features["sunflower_score"] = 1.0 - float(np.mean(inter_diffs)) / np.pi
        else:
            features["sunflower_score"] = 0.0
    else:
        features["golden_angle_deviation"] = np.pi
        features["golden_angle_uniformity"] = 0.0
        features["sunflower_score"] = 0.0

    # --- 45h. Fibonacci Inter-Trade Time Analysis ---
    if n >= 20:
        it_fib = np.diff(timestamps_s)
        it_fib = it_fib[it_fib > 0]
        if len(it_fib) >= 10:
            # Ratio of consecutive inter-trade times
            it_ratios = it_fib[1:] / np.maximum(it_fib[:-1], 1e-12)
            # How many are near golden ratio?
            golden_it_count = 0
            for itr in it_ratios:
                if min(abs(itr - phi), abs(itr - inv_phi)) < 0.15:
                    golden_it_count += 1
            features["fib_intertime_ratio_pct"] = golden_it_count / len(it_ratios)

            # Distribution of inter-trade times: does it follow Fibonacci scaling?
            # Sort inter-trade times, check if quantile ratios are near φ
            sorted_it = np.sort(it_fib)
            q25 = np.percentile(sorted_it, 25)
            q50 = np.percentile(sorted_it, 50)
            q75 = np.percentile(sorted_it, 75)
            q90 = np.percentile(sorted_it, 90)
            if q25 > 0 and q50 > 0:
                r1 = q50 / q25
                r2 = q75 / q50
                r3 = q90 / q75 if q75 > 0 else 0.0
                fib_scale_score = 0
                for r in [r1, r2, r3]:
                    if r > 0 and min(abs(r - phi), abs(r - inv_phi), abs(r - phi**2)) < 0.3:
                        fib_scale_score += 1
                features["fib_time_scaling_score"] = fib_scale_score / 3
            else:
                features["fib_time_scaling_score"] = 0.0
        else:
            features["fib_intertime_ratio_pct"] = 0.0
            features["fib_time_scaling_score"] = 0.0
    else:
        features["fib_intertime_ratio_pct"] = 0.0
        features["fib_time_scaling_score"] = 0.0

    # -----------------------------------------------------------------------
    # 46. Elliott Wave Principle Analysis
    # -----------------------------------------------------------------------
    # Detect impulse (5-wave) and corrective (3-wave) patterns in tick data
    # using the swing points already found in section 45f.
    # Re-use swing_prices and swing_types if available, else recompute.
    ew_swings = swing_prices if 'swing_prices' in dir() and len(swing_prices) >= 5 else []
    ew_types = swing_types if 'swing_types' in dir() and len(swing_types) >= 5 else []

    if len(ew_swings) < 5 and n >= 50 and price_range > 0:
        # Recompute swings — reuse the same vectorized approach from 45f
        ew_min_swing = max(5, n // 30)
        if n > 5000:
            step_ew = n // 2500
            p_ew = prices[::step_ew]
            ew_ms = max(3, ew_min_swing // step_ew)
        else:
            p_ew = prices
            ew_ms = ew_min_swing
        n_ew = len(p_ew)
        ew_swings = []
        ew_types = []
        win_ew = 2 * ew_ms + 1
        if n_ew >= win_ew:
            windows_ew = sliding_window_view(p_ew, win_ew)
            cv_ew = p_ew[ew_ms:ew_ms + len(windows_ew)]
            is_max_ew = (cv_ew == windows_ew.max(axis=1)) & (cv_ew > p_ew[0:len(windows_ew)])
            is_min_ew = (cv_ew == windows_ew.min(axis=1)) & (cv_ew < p_ew[0:len(windows_ew)])
            last_ew = -ew_ms * 2
            for i_ew in range(len(is_max_ew)):
                if i_ew - last_ew < ew_ms:
                    continue
                if is_max_ew[i_ew]:
                    ew_swings.append(float(cv_ew[i_ew]))
                    ew_types.append(1)
                    last_ew = i_ew
                elif is_min_ew[i_ew]:
                    ew_swings.append(float(cv_ew[i_ew]))
                    ew_types.append(-1)
                    last_ew = i_ew

    if len(ew_swings) >= 5:
        ew_waves = np.diff(ew_swings)  # signed wave moves
        ew_abs = np.abs(ew_waves)
        n_waves = len(ew_waves)

        # --- EW Rule: 5-wave impulse detection ---
        # Look for 5 consecutive waves where waves 1,3,5 are same direction
        # and waves 2,4 are opposite (corrections)
        impulse_score = 0.0
        best_impulse_quality = 0.0

        for start_w in range(n_waves - 4):
            w = ew_waves[start_w:start_w + 5]
            wa = ew_abs[start_w:start_w + 5]

            # Check alternating direction: w1,w3,w5 same sign; w2,w4 opposite
            if np.sign(w[0]) == np.sign(w[2]) == np.sign(w[4]) and \
               np.sign(w[1]) == np.sign(w[3]) and \
               np.sign(w[0]) != np.sign(w[1]):

                quality = 0.0

                # Rule: Wave 3 is never the shortest of 1, 3, 5
                impulse_waves = [wa[0], wa[2], wa[4]]
                if wa[2] != min(impulse_waves):
                    quality += 0.2

                # Rule: Wave 3 is typically the longest
                if wa[2] == max(impulse_waves):
                    quality += 0.15

                # Rule: Wave 2 retraces less than 100% of Wave 1
                if wa[1] < wa[0]:
                    quality += 0.15

                # Rule: Wave 4 doesn't overlap Wave 1 territory
                # (Wave 4 end doesn't go past Wave 1 start)
                w1_start = ew_swings[start_w]
                w1_end = ew_swings[start_w + 1]
                w4_end = ew_swings[start_w + 4]
                if np.sign(w[0]) > 0:  # uptrend impulse
                    if w4_end > w1_end:  # Wave 4 low above Wave 1 high
                        quality += 0.15
                else:  # downtrend impulse
                    if w4_end < w1_end:
                        quality += 0.15

                # Fibonacci ratios in wave relationships
                # Wave 3 ≈ 1.618 × Wave 1
                if wa[0] > 0:
                    w3_w1_ratio = wa[2] / wa[0]
                    if abs(w3_w1_ratio - phi) < 0.3:
                        quality += 0.1
                    # Wave 5 ≈ Wave 1 or 0.618 × Wave 3
                    w5_w1_ratio = wa[4] / wa[0]
                    if abs(w5_w1_ratio - 1.0) < 0.2 or abs(w5_w1_ratio - inv_phi) < 0.2:
                        quality += 0.1

                # Wave 2 retracement near Fib level (38.2%, 50%, 61.8%)
                if wa[0] > 0:
                    w2_retrace = wa[1] / wa[0]
                    fib_retrace_dists = [abs(w2_retrace - fl) for fl in [0.382, 0.5, 0.618]]
                    if min(fib_retrace_dists) < 0.08:
                        quality += 0.075

                # Alternation: Wave 2 and Wave 4 should differ in character
                # Sharp (>61.8% retrace) vs flat (<38.2% retrace)
                if wa[0] > 0 and wa[2] > 0:
                    w2_depth = wa[1] / wa[0]
                    w4_depth = wa[3] / wa[2]
                    # One deep, one shallow = alternation
                    if (w2_depth > 0.5 and w4_depth < 0.5) or \
                       (w2_depth < 0.5 and w4_depth > 0.5):
                        quality += 0.075

                impulse_score = max(impulse_score, 1.0)
                best_impulse_quality = max(best_impulse_quality, quality)

        features["ew_impulse_detected"] = impulse_score
        features["ew_impulse_quality"] = best_impulse_quality

        # --- EW: 3-wave correction (ABC) detection ---
        correction_score = 0.0
        best_correction_quality = 0.0

        for start_w in range(n_waves - 2):
            w = ew_waves[start_w:start_w + 3]
            wa = ew_abs[start_w:start_w + 3]

            # ABC: A and C same direction, B opposite
            if np.sign(w[0]) == np.sign(w[2]) and np.sign(w[0]) != np.sign(w[1]):
                quality = 0.0

                # Wave C ≈ Wave A (common)
                if wa[0] > 0:
                    c_a_ratio = wa[2] / wa[0]
                    if abs(c_a_ratio - 1.0) < 0.2:
                        quality += 0.25
                    # Wave C ≈ 1.618 × Wave A (extended)
                    elif abs(c_a_ratio - phi) < 0.3:
                        quality += 0.2
                    # Wave C ≈ 0.618 × Wave A (truncated)
                    elif abs(c_a_ratio - inv_phi) < 0.2:
                        quality += 0.15

                # Wave B retracement of A near Fib level
                if wa[0] > 0:
                    b_retrace = wa[1] / wa[0]
                    fib_b_dists = [abs(b_retrace - fl) for fl in [0.382, 0.5, 0.618, 0.786]]
                    if min(fib_b_dists) < 0.08:
                        quality += 0.25

                # Flat correction: B retraces >90% of A, C ≈ A
                if wa[0] > 0:
                    if wa[1] / wa[0] > 0.9 and abs(wa[2] / wa[0] - 1.0) < 0.15:
                        quality += 0.25  # flat pattern

                # Zigzag: B retraces 38-62% of A, C > A
                if wa[0] > 0:
                    b_r = wa[1] / wa[0]
                    if 0.35 < b_r < 0.65 and wa[2] > wa[0]:
                        quality += 0.25  # zigzag pattern

                correction_score = max(correction_score, 1.0)
                best_correction_quality = max(best_correction_quality, quality)

        features["ew_correction_detected"] = correction_score
        features["ew_correction_quality"] = best_correction_quality

        # --- EW: Wave personality signatures ---
        # In a 5-wave impulse, each wave has characteristic behavior:
        # Wave 1: low volume, hesitant
        # Wave 3: highest volume, strongest momentum
        # Wave 5: declining volume, divergence

        # Volume distribution across waves (using all waves)
        if n_waves >= 3:
            # Approximate volume per wave using time-proportional allocation
            wave_vols = []
            cumulative_swings = 0
            for wi in range(n_waves):
                # Rough: allocate volume proportional to wave amplitude
                wave_vols.append(ew_abs[wi])

            wave_vols = np.array(wave_vols)
            total_wave_vol = wave_vols.sum()
            if total_wave_vol > 0:
                wave_vol_pcts = wave_vols / total_wave_vol

                # Is volume declining in later waves? (exhaustion signal)
                if n_waves >= 4:
                    first_half_vol = wave_vols[:n_waves // 2].mean()
                    second_half_vol = wave_vols[n_waves // 2:].mean()
                    features["ew_vol_exhaustion"] = second_half_vol / first_half_vol if first_half_vol > 0 else 1.0
                else:
                    features["ew_vol_exhaustion"] = 1.0

                # Wave amplitude trend: are waves getting smaller? (completion signal)
                if n_waves >= 3:
                    amp_slope = np.polyfit(range(n_waves), ew_abs, 1)[0]
                    features["ew_amplitude_trend"] = float(amp_slope)
                    features["ew_amplitude_trend_norm"] = float(amp_slope / np.mean(ew_abs)) if np.mean(ew_abs) > 0 else 0.0
                else:
                    features["ew_amplitude_trend"] = 0.0
                    features["ew_amplitude_trend_norm"] = 0.0
            else:
                features["ew_vol_exhaustion"] = 1.0
                features["ew_amplitude_trend"] = 0.0
                features["ew_amplitude_trend_norm"] = 0.0
        else:
            features["ew_vol_exhaustion"] = 1.0
            features["ew_amplitude_trend"] = 0.0
            features["ew_amplitude_trend_norm"] = 0.0

        # --- EW: Wave count and structure ---
        features["ew_wave_count"] = n_waves
        features["ew_swing_count"] = len(ew_swings)

        # Ratio of impulse waves (odd) to corrective waves (even)
        # In perfect EW: 3 impulse + 2 corrective in a 5-wave
        odd_waves = ew_abs[::2]   # waves 1, 3, 5, ...
        even_waves = ew_abs[1::2]  # waves 2, 4, ...
        features["ew_impulse_correction_ratio"] = odd_waves.sum() / even_waves.sum() if even_waves.sum() > 0 else 1.0

        # Alternation score: do consecutive correction waves differ in depth?
        if len(even_waves) >= 2:
            alt_diffs = np.abs(np.diff(even_waves))
            features["ew_alternation_score"] = float(np.mean(alt_diffs) / np.mean(even_waves)) if np.mean(even_waves) > 0 else 0.0
        else:
            features["ew_alternation_score"] = 0.0

        # Wave symmetry: how symmetric is the wave structure?
        if n_waves >= 4:
            first_half_waves = ew_abs[:n_waves // 2]
            second_half_waves = ew_abs[n_waves // 2:]
            min_len_ew = min(len(first_half_waves), len(second_half_waves))
            if min_len_ew > 0:
                sym_corr = _safe_corr(first_half_waves[:min_len_ew], second_half_waves[:min_len_ew])
                features["ew_wave_symmetry"] = sym_corr
            else:
                features["ew_wave_symmetry"] = 0.0
        else:
            features["ew_wave_symmetry"] = 0.0

        # Net wave direction: sum of all waves / sum of absolute waves
        features["ew_net_direction"] = float(np.sum(ew_waves) / np.sum(ew_abs)) if np.sum(ew_abs) > 0 else 0.0

        # Wave completion estimate: where are we in the cycle?
        # Based on cumulative wave amplitude vs total range
        cum_amp = np.cumsum(ew_abs)
        total_amp = cum_amp[-1]
        if total_amp > 0:
            # What fraction of total wave energy is complete?
            features["ew_completion_pct"] = float(cum_amp[-1] / total_amp)  # always 1.0 for current candle
            # But the SHAPE tells us: if last wave is small vs first, we're near end
            features["ew_last_wave_pct"] = float(ew_abs[-1] / total_amp)
            features["ew_first_wave_pct"] = float(ew_abs[0] / total_amp)
        else:
            features["ew_completion_pct"] = 0.0
            features["ew_last_wave_pct"] = 0.0
            features["ew_first_wave_pct"] = 0.0

    else:
        features["ew_impulse_detected"] = 0.0
        features["ew_impulse_quality"] = 0.0
        features["ew_correction_detected"] = 0.0
        features["ew_correction_quality"] = 0.0
        features["ew_vol_exhaustion"] = 1.0
        features["ew_amplitude_trend"] = 0.0
        features["ew_amplitude_trend_norm"] = 0.0
        features["ew_wave_count"] = 0
        features["ew_swing_count"] = 0
        features["ew_impulse_correction_ratio"] = 1.0
        features["ew_alternation_score"] = 0.0
        features["ew_wave_symmetry"] = 0.0
        features["ew_net_direction"] = 0.0
        features["ew_last_wave_pct"] = 0.0
        features["ew_first_wave_pct"] = 0.0

    # -----------------------------------------------------------------------
    # 47. Math — Spectral / Fourier Analysis
    # -----------------------------------------------------------------------
    if n >= 64:
        # FFT of tick returns (detrended)
        ret_series = valid_returns - valid_returns.mean()
        n_fft = len(ret_series)
        fft_vals = np.fft.rfft(ret_series)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n_fft)

        if len(power) > 2:
            # Skip DC component (index 0)
            power_no_dc = power[1:]
            freqs_no_dc = freqs[1:]
            total_power = power_no_dc.sum()

            if total_power > 0:
                # Dominant frequency
                dom_idx = np.argmax(power_no_dc)
                features["dominant_freq"] = float(freqs_no_dc[dom_idx])
                features["dominant_freq_power_pct"] = float(power_no_dc[dom_idx] / total_power)

                # Spectral energy ratio: low freq (bottom 25%) vs high freq (top 25%)
                n_freq = len(power_no_dc)
                low_cutoff = n_freq // 4
                high_cutoff = 3 * n_freq // 4
                low_power = power_no_dc[:low_cutoff].sum()
                high_power = power_no_dc[high_cutoff:].sum()
                features["spectral_energy_ratio"] = low_power / high_power if high_power > 0 else 1.0

                # Spectral entropy
                spec_probs = power_no_dc / total_power
                spec_probs = spec_probs[spec_probs > 0]
                features["spectral_entropy"] = float(-np.sum(spec_probs * np.log2(spec_probs)))
                max_spec_entropy = np.log2(len(power_no_dc)) if len(power_no_dc) > 1 else 1.0
                features["spectral_flatness"] = features["spectral_entropy"] / max_spec_entropy if max_spec_entropy > 0 else 1.0

                # Spectral centroid: "center of mass" of the spectrum
                features["spectral_centroid"] = float(np.average(freqs_no_dc, weights=power_no_dc))
            else:
                features["dominant_freq"] = 0.0
                features["dominant_freq_power_pct"] = 0.0
                features["spectral_energy_ratio"] = 1.0
                features["spectral_entropy"] = 0.0
                features["spectral_flatness"] = 1.0
                features["spectral_centroid"] = 0.0
        else:
            features["dominant_freq"] = 0.0
            features["dominant_freq_power_pct"] = 0.0
            features["spectral_energy_ratio"] = 1.0
            features["spectral_entropy"] = 0.0
            features["spectral_flatness"] = 1.0
            features["spectral_centroid"] = 0.0
    else:
        features["dominant_freq"] = 0.0
        features["dominant_freq_power_pct"] = 0.0
        features["spectral_energy_ratio"] = 1.0
        features["spectral_entropy"] = 0.0
        features["spectral_flatness"] = 1.0
        features["spectral_centroid"] = 0.0

    # -----------------------------------------------------------------------
    # 48. Taylor Series Decomposition of Price/Volume/OFI Paths
    # -----------------------------------------------------------------------
    # Fit polynomial to normalized price path: p(t) = a0 + a1*t + a2*t² + ...
    # Coefficients capture the SHAPE of the path at different orders.
    # We fit up to degree 5 for price, degree 3 for volume and OFI.

    if n >= 30 and candle_duration_s > 0 and price_range > 0:
        # Normalize: t ∈ [0,1], price ∈ [0,1] (relative to range)
        t_norm = rel_time
        p_norm = (prices - l) / price_range

        # --- Price Taylor coefficients (degree 5) ---
        # Subsample if too many ticks (polyfit is O(n*d²), keep fast)
        max_pts = 2000
        if n > max_pts:
            idx_sub = np.linspace(0, n - 1, max_pts, dtype=int)
            t_sub = t_norm[idx_sub]
            p_sub = p_norm[idx_sub]
        else:
            t_sub = t_norm
            p_sub = p_norm

        try:
            price_poly = np.polyfit(t_sub, p_sub, 5)
            # polyfit returns [a5, a4, a3, a2, a1, a0] (highest degree first)
            features["taylor_price_a0"] = float(price_poly[5])  # intercept (≈ open position)
            features["taylor_price_a1"] = float(price_poly[4])  # linear trend
            features["taylor_price_a2"] = float(price_poly[3])  # curvature
            features["taylor_price_a3"] = float(price_poly[2])  # asymmetry/skew
            features["taylor_price_a4"] = float(price_poly[1])  # kurtosis-like
            features["taylor_price_a5"] = float(price_poly[0])  # 5th order complexity

            # Residual: how well does the polynomial fit?
            p_fitted = np.polyval(price_poly, t_sub)
            residuals = p_sub - p_fitted
            features["taylor_price_r2"] = 1.0 - (np.var(residuals) / np.var(p_sub)) if np.var(p_sub) > 0 else 1.0
            features["taylor_price_rmse"] = float(np.sqrt(np.mean(residuals ** 2)))

            # Coefficient ratios: curvature relative to trend
            if abs(price_poly[4]) > 1e-10:
                features["taylor_curvature_trend_ratio"] = float(price_poly[3] / price_poly[4])
            else:
                features["taylor_curvature_trend_ratio"] = 0.0

            # Asymmetry relative to curvature
            if abs(price_poly[3]) > 1e-10:
                features["taylor_asymmetry_curvature_ratio"] = float(price_poly[2] / price_poly[3])
            else:
                features["taylor_asymmetry_curvature_ratio"] = 0.0

            # Dominant order: which coefficient has the most influence?
            # Weight by typical t^k contribution at t=0.5
            coeff_influence = [abs(price_poly[5-k]) * (0.5 ** k) for k in range(6)]
            features["taylor_dominant_order"] = int(np.argmax(coeff_influence))

            # Complexity: how many terms needed for 90% of R²?
            complexity = 0
            for deg in range(1, 6):
                try:
                    partial_poly = np.polyfit(t_sub, p_sub, deg)
                    partial_fit = np.polyval(partial_poly, t_sub)
                    partial_r2 = 1.0 - (np.var(p_sub - partial_fit) / np.var(p_sub)) if np.var(p_sub) > 0 else 1.0
                    if partial_r2 >= 0.9:
                        complexity = deg
                        break
                except Exception:
                    pass
            if complexity == 0:
                complexity = 5
            features["taylor_complexity"] = complexity

        except Exception:
            features["taylor_price_a0"] = 0.0
            features["taylor_price_a1"] = 0.0
            features["taylor_price_a2"] = 0.0
            features["taylor_price_a3"] = 0.0
            features["taylor_price_a4"] = 0.0
            features["taylor_price_a5"] = 0.0
            features["taylor_price_r2"] = 0.0
            features["taylor_price_rmse"] = 0.0
            features["taylor_curvature_trend_ratio"] = 0.0
            features["taylor_asymmetry_curvature_ratio"] = 0.0
            features["taylor_dominant_order"] = 0
            features["taylor_complexity"] = 0

        # --- Volume path Taylor coefficients (degree 3) ---
        # Cumulative volume over time → shape of volume accumulation
        cum_vol = np.cumsum(sizes) / total_vol if total_vol > 0 else np.linspace(0, 1, n)
        if n > max_pts:
            cv_sub = cum_vol[idx_sub]
        else:
            cv_sub = cum_vol

        try:
            vol_poly = np.polyfit(t_sub, cv_sub, 3)
            features["taylor_vol_a1"] = float(vol_poly[2])  # linear rate
            features["taylor_vol_a2"] = float(vol_poly[1])  # acceleration
            features["taylor_vol_a3"] = float(vol_poly[0])  # jerk

            vol_fitted = np.polyval(vol_poly, t_sub)
            vol_resid = cv_sub - vol_fitted
            features["taylor_vol_r2"] = 1.0 - (np.var(vol_resid) / np.var(cv_sub)) if np.var(cv_sub) > 0 else 1.0

            # Volume acceleration relative to rate
            if abs(vol_poly[2]) > 1e-10:
                features["taylor_vol_accel_ratio"] = float(vol_poly[1] / vol_poly[2])
            else:
                features["taylor_vol_accel_ratio"] = 0.0
        except Exception:
            features["taylor_vol_a1"] = 0.0
            features["taylor_vol_a2"] = 0.0
            features["taylor_vol_a3"] = 0.0
            features["taylor_vol_r2"] = 0.0
            features["taylor_vol_accel_ratio"] = 0.0

        # --- OFI (Order Flow Imbalance) path Taylor coefficients (degree 3) ---
        # Cumulative signed volume over time → shape of order flow
        signed_vol = sizes * sides
        cum_ofi = np.cumsum(signed_vol)
        ofi_range = cum_ofi.max() - cum_ofi.min()
        if ofi_range > 0:
            ofi_norm = (cum_ofi - cum_ofi.min()) / ofi_range
        else:
            ofi_norm = np.zeros(n)

        if n > max_pts:
            ofi_sub = ofi_norm[idx_sub]
        else:
            ofi_sub = ofi_norm

        try:
            ofi_poly = np.polyfit(t_sub, ofi_sub, 3)
            features["taylor_ofi_a1"] = float(ofi_poly[2])  # OFI trend
            features["taylor_ofi_a2"] = float(ofi_poly[1])  # OFI acceleration
            features["taylor_ofi_a3"] = float(ofi_poly[0])  # OFI jerk

            ofi_fitted = np.polyval(ofi_poly, t_sub)
            ofi_resid = ofi_sub - ofi_fitted
            features["taylor_ofi_r2"] = 1.0 - (np.var(ofi_resid) / np.var(ofi_sub)) if np.var(ofi_sub) > 0 else 1.0

            # OFI-price alignment: do their Taylor trends agree?
            features["taylor_price_ofi_alignment"] = float(
                np.sign(features.get("taylor_price_a1", 0)) * np.sign(features.get("taylor_ofi_a1", 0))
            )
        except Exception:
            features["taylor_ofi_a1"] = 0.0
            features["taylor_ofi_a2"] = 0.0
            features["taylor_ofi_a3"] = 0.0
            features["taylor_ofi_r2"] = 0.0
            features["taylor_price_ofi_alignment"] = 0.0

        # --- Trade rate path Taylor coefficients (degree 3) ---
        # Cumulative trade count over time → shape of activity
        cum_trades = np.arange(1, n + 1, dtype=float) / n
        if n > max_pts:
            ct_sub = cum_trades[idx_sub]
        else:
            ct_sub = cum_trades

        try:
            rate_poly = np.polyfit(t_sub, ct_sub, 3)
            features["taylor_rate_a1"] = float(rate_poly[2])  # activity rate
            features["taylor_rate_a2"] = float(rate_poly[1])  # activity acceleration
            features["taylor_rate_a3"] = float(rate_poly[0])  # activity jerk

            rate_fitted = np.polyval(rate_poly, t_sub)
            rate_resid = ct_sub - rate_fitted
            features["taylor_rate_r2"] = 1.0 - (np.var(rate_resid) / np.var(ct_sub)) if np.var(ct_sub) > 0 else 1.0
        except Exception:
            features["taylor_rate_a1"] = 0.0
            features["taylor_rate_a2"] = 0.0
            features["taylor_rate_a3"] = 0.0
            features["taylor_rate_r2"] = 0.0

        # --- Cross-series Taylor divergence ---
        # Do price and volume paths diverge? (different shapes = divergence signal)
        features["taylor_price_vol_divergence"] = abs(
            features.get("taylor_price_a2", 0) - features.get("taylor_vol_a2", 0)
        )
        features["taylor_price_rate_divergence"] = abs(
            features.get("taylor_price_a2", 0) - features.get("taylor_rate_a2", 0)
        )

    else:
        for suffix in ["a0", "a1", "a2", "a3", "a4", "a5"]:
            features[f"taylor_price_{suffix}"] = 0.0
        features["taylor_price_r2"] = 0.0
        features["taylor_price_rmse"] = 0.0
        features["taylor_curvature_trend_ratio"] = 0.0
        features["taylor_asymmetry_curvature_ratio"] = 0.0
        features["taylor_dominant_order"] = 0
        features["taylor_complexity"] = 0
        for prefix in ["taylor_vol", "taylor_ofi", "taylor_rate"]:
            features[f"{prefix}_a1"] = 0.0
            features[f"{prefix}_a2"] = 0.0
            features[f"{prefix}_a3"] = 0.0
            features[f"{prefix}_r2"] = 0.0
        features["taylor_vol_accel_ratio"] = 0.0
        features["taylor_price_ofi_alignment"] = 0.0
        features["taylor_price_vol_divergence"] = 0.0
        features["taylor_price_rate_divergence"] = 0.0

    # -----------------------------------------------------------------------
    # 49. Information Theory — Beyond Shannon Entropy
    # -----------------------------------------------------------------------
    if n >= 30:
        # --- Transfer entropy: does price predict volume or vice versa? ---
        # Discretize into up/down/flat bins
        ret_disc = np.sign(tick_returns[1:])  # skip first NaN
        size_disc = np.where(sizes[1:] > np.median(sizes), 1, 0)
        valid_te = ~np.isnan(ret_disc)
        ret_d = ret_disc[valid_te].astype(int)
        sz_d = size_disc[valid_te].astype(int)

        if len(ret_d) >= 20:
            # Simplified transfer entropy using histogram-based approach
            def _simple_te(source, target):
                """Transfer entropy from source to target (vectorized)."""
                n_te = min(len(source), len(target)) - 1
                if n_te < 10:
                    return 0.0
                tp = target[:n_te]
                sp = source[:n_te]
                tf = target[1:n_te + 1]
                # Encode triplets as single integers for fast counting
                key_joint = tp * 100 + sp * 10 + tf
                key_tp_sp = tp * 10 + sp
                key_tp_tf = tp * 10 + tf
                c_joint = Counter(key_joint)
                c_tp_sp = Counter(key_tp_sp)
                c_tp_tf = Counter(key_tp_tf)
                c_tp = Counter(tp)
                te = 0.0
                for k, cnt in c_joint.items():
                    p_j = cnt / n_te
                    p_ts = c_tp_sp.get(k // 10 * 10 + (k % 100) // 10, 0) / n_te
                    # Decode: tp = k // 100, sp = (k % 100) // 10, tf = k % 10
                    tp_v = k // 100
                    tf_v = k % 10
                    p_ttf = c_tp_tf.get(tp_v * 10 + tf_v, 0) / n_te
                    p_t = c_tp.get(tp_v, 0) / n_te
                    if p_j > 0 and p_ts > 0 and p_ttf > 0 and p_t > 0:
                        te += p_j * np.log2(p_j * p_t / (p_ts * p_ttf))
                return max(te, 0.0)

            features["te_price_to_volume"] = _simple_te(ret_d, sz_d)
            features["te_volume_to_price"] = _simple_te(sz_d, ret_d)
            features["te_net_direction"] = features["te_price_to_volume"] - features["te_volume_to_price"]

            # --- Mutual information between buy/sell sequence and returns ---
            side_seq = sides[1:][valid_te].astype(int)
            if len(side_seq) >= 10:
                # MI(side, return_sign)
                mi = 0.0
                for vs in np.unique(side_seq):
                    for vr in np.unique(ret_d):
                        p_joint = np.mean((side_seq == vs) & (ret_d == vr))
                        p_s = np.mean(side_seq == vs)
                        p_r = np.mean(ret_d == vr)
                        if p_joint > 0 and p_s > 0 and p_r > 0:
                            mi += p_joint * np.log2(p_joint / (p_s * p_r))
                features["mi_side_return"] = max(mi, 0.0)
            else:
                features["mi_side_return"] = 0.0
        else:
            features["te_price_to_volume"] = 0.0
            features["te_volume_to_price"] = 0.0
            features["te_net_direction"] = 0.0
            features["mi_side_return"] = 0.0

        # --- Lempel-Ziv complexity of price direction sequence ---
        dir_seq = np.sign(np.diff(prices))
        dir_str = ''.join(['1' if d > 0 else '0' if d < 0 else '2' for d in dir_seq[:2000]])
        # LZ76 complexity
        def _lz_complexity(s):
            """Lempel-Ziv complexity (number of distinct substrings)."""
            i, k, l = 0, 1, 1
            c = 1
            n_s = len(s)
            while True:
                if s[i + k - 1] == s[l + k - 1] if (l + k - 1) < n_s else False:
                    k += 1
                    if l + k > n_s:
                        c += 1
                        break
                else:
                    if k > 1:
                        c += 1
                        l += k
                        if l > n_s:
                            break
                        i = 0
                        k = 1
                    else:
                        i += 1
                        if i == l:
                            c += 1
                            l += 1
                            if l > n_s:
                                break
                            i = 0
            return c

        if len(dir_str) >= 10:
            lz_c = _lz_complexity(dir_str)
            # Normalize by theoretical max for random sequence: n / log2(n)
            n_dir = len(dir_str)
            lz_max = n_dir / max(np.log2(n_dir), 1)
            features["lz_complexity"] = lz_c
            features["lz_complexity_norm"] = lz_c / lz_max if lz_max > 0 else 0.0
        else:
            features["lz_complexity"] = 0
            features["lz_complexity_norm"] = 0.0

        # --- Self-information / surprise per trade ---
        # Average surprise: -log2(P(event)) for each trade direction
        dir_counts = np.array([np.sum(dir_seq > 0), np.sum(dir_seq == 0), np.sum(dir_seq < 0)])
        dir_total = dir_counts.sum()
        if dir_total > 0:
            dir_probs = dir_counts / dir_total
            dir_probs = dir_probs[dir_probs > 0]
            features["avg_surprise"] = float(-np.mean(np.log2(dir_probs)))
        else:
            features["avg_surprise"] = 0.0
    else:
        features["te_price_to_volume"] = 0.0
        features["te_volume_to_price"] = 0.0
        features["te_net_direction"] = 0.0
        features["mi_side_return"] = 0.0
        features["lz_complexity"] = 0
        features["lz_complexity_norm"] = 0.0
        features["avg_surprise"] = 0.0

    # -----------------------------------------------------------------------
    # 50. Game Theory / Auction Theory
    # -----------------------------------------------------------------------
    if n >= 20:
        # --- Nash equilibrium proximity ---
        # Measure how balanced buy/sell forces are (equilibrium = balanced)
        buy_force = buy_vol * (buy_count / n if n > 0 else 0)
        sell_force = sell_vol * (sell_count / n if n > 0 else 0)
        total_force = buy_force + sell_force
        if total_force > 0:
            features["nash_balance"] = 1.0 - abs(buy_force - sell_force) / total_force
        else:
            features["nash_balance"] = 0.5

        # --- Stackelberg leader detection ---
        # Does one side consistently move first after price changes?
        # Look at who initiates after each direction change
        if len(tick_returns) >= 10:
            ret_signs = np.sign(tick_returns[1:])
            valid_rs = ~np.isnan(ret_signs)
            if valid_rs.sum() >= 10:
                rs = ret_signs[valid_rs]
                sd = sides[1:][valid_rs]
                # After up-tick, who trades next?
                up_ticks = np.where(rs > 0)[0]
                down_ticks = np.where(rs < 0)[0]
                leader_buy = 0
                leader_sell = 0
                for ut in up_ticks:
                    if ut + 1 < len(sd):
                        if sd[ut + 1] > 0:
                            leader_buy += 1
                        else:
                            leader_sell += 1
                total_leaders = leader_buy + leader_sell
                features["stackelberg_buy_leader"] = leader_buy / total_leaders if total_leaders > 0 else 0.5
            else:
                features["stackelberg_buy_leader"] = 0.5
        else:
            features["stackelberg_buy_leader"] = 0.5

        # --- Auction clearing speed ---
        # After large trades (>2× median), how quickly does price stabilize?
        med_size = np.median(sizes)
        large_mask_gt = sizes > 2 * med_size
        large_indices = np.where(large_mask_gt)[0]
        if len(large_indices) >= 3:
            # Vectorized: only process first 50 large trades for speed
            li_sample = large_indices[:50]
            li_valid = li_sample[(li_sample >= 5) & (li_sample + 5 < n)]
            if len(li_valid) >= 2:
                # Compute pre/post volatility in one pass
                pre_vols = np.array([prices[li-5:li].std() for li in li_valid])
                post_vols = np.array([prices[li:li+5].std() for li in li_valid])
                valid_mask = pre_vols > 0
                if valid_mask.sum() > 0:
                    features["auction_clearing_speed"] = float(np.mean(post_vols[valid_mask] / pre_vols[valid_mask]))
                else:
                    features["auction_clearing_speed"] = 1.0
            else:
                features["auction_clearing_speed"] = 1.0
        else:
            features["auction_clearing_speed"] = 1.0

        # --- Strategic timing: do large trades cluster at specific candle positions? ---
        if large_mask_gt.sum() >= 3:
            large_times = rel_time[large_mask_gt]
            features["large_trade_centroid"] = float(np.mean(large_times))
            features["large_trade_time_std"] = float(np.std(large_times))
        else:
            features["large_trade_centroid"] = 0.5
            features["large_trade_time_std"] = 0.3
    else:
        features["nash_balance"] = 0.5
        features["stackelberg_buy_leader"] = 0.5
        features["auction_clearing_speed"] = 1.0
        features["large_trade_centroid"] = 0.5
        features["large_trade_time_std"] = 0.3

    # -----------------------------------------------------------------------
    # 51. Network / Graph Theory
    # -----------------------------------------------------------------------
    if n >= 30 and price_range > 0:
        # --- Price level transition graph ---
        # Discretize price into 10 levels, build transition matrix
        n_levels = 10
        price_levels = np.clip(
            ((prices - l) / price_range * (n_levels - 1)).astype(int), 0, n_levels - 1
        )
        # Transition matrix
        trans_mat = np.zeros((n_levels, n_levels))
        for i_tr in range(len(price_levels) - 1):
            trans_mat[price_levels[i_tr], price_levels[i_tr + 1]] += 1

        total_trans = trans_mat.sum()
        if total_trans > 0:
            trans_prob = trans_mat / total_trans

            # Graph entropy: how predictable are transitions?
            tp_flat = trans_prob.flatten()
            tp_pos = tp_flat[tp_flat > 0]
            features["graph_transition_entropy"] = float(-np.sum(tp_pos * np.log2(tp_pos)))
            max_graph_entropy = np.log2(n_levels * n_levels)
            features["graph_transition_uniformity"] = features["graph_transition_entropy"] / max_graph_entropy if max_graph_entropy > 0 else 1.0

            # Self-loop ratio: how often does price stay at same level?
            self_loops = np.trace(trans_mat)
            features["graph_self_loop_ratio"] = self_loops / total_trans

            # Number of unique transitions (graph edges)
            features["graph_edge_count"] = int(np.count_nonzero(trans_mat))
            features["graph_edge_density"] = features["graph_edge_count"] / (n_levels * n_levels)

            # Asymmetry: is the transition matrix symmetric? (directional bias)
            asym = np.abs(trans_mat - trans_mat.T).sum()
            features["graph_asymmetry"] = asym / (2 * total_trans) if total_trans > 0 else 0.0
        else:
            features["graph_transition_entropy"] = 0.0
            features["graph_transition_uniformity"] = 1.0
            features["graph_self_loop_ratio"] = 0.0
            features["graph_edge_count"] = 0
            features["graph_edge_density"] = 0.0
            features["graph_asymmetry"] = 0.0

        # --- Recurrence: how often does price revisit levels? ---
        unique_levels = len(np.unique(price_levels))
        features["price_level_recurrence"] = 1.0 - unique_levels / n_levels
        # Average visits per level
        level_counts = np.bincount(price_levels, minlength=n_levels)
        features["avg_visits_per_level"] = float(np.mean(level_counts[level_counts > 0]))
        features["max_visits_level"] = int(np.max(level_counts))
    else:
        features["graph_transition_entropy"] = 0.0
        features["graph_transition_uniformity"] = 1.0
        features["graph_self_loop_ratio"] = 0.0
        features["graph_edge_count"] = 0
        features["graph_edge_density"] = 0.0
        features["graph_asymmetry"] = 0.0
        features["price_level_recurrence"] = 0.0
        features["avg_visits_per_level"] = 0.0
        features["max_visits_level"] = 0

    # -----------------------------------------------------------------------
    # 52. Chaos Theory / Nonlinear Dynamics
    # -----------------------------------------------------------------------
    if n >= 50 and len(valid_returns) >= 30:
        vr = valid_returns

        # --- Hurst exponent (uses top-level _hurst_rs) ---
        # H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk
        features["hurst_regime"] = features.get("hurst_exponent", 0.5) - 0.5

        # --- Approximate Lyapunov exponent (subsampled for speed) ---
        embedding_dim = 3
        tau = 1
        # Subsample to 500 points max before embedding
        vr_lyap = vr[::max(1, len(vr) // 500)][:500]
        if len(vr_lyap) >= embedding_dim + 10:
            m_ly = len(vr_lyap) - (embedding_dim - 1) * tau
            embedded = np.array([vr_lyap[i*tau:i*tau + m_ly] for i in range(embedding_dim)]).T
            n_emb = len(embedded)
            if n_emb >= 20:
                # Vectorized: pick 30 reference points, find nearest neighbors
                ref_idx = np.linspace(2, n_emb - 3, min(30, n_emb - 4), dtype=int)
                lyap_vals = []
                for i_ly in ref_idx:
                    diffs_ly = np.sum((embedded - embedded[i_ly]) ** 2, axis=1)
                    diffs_ly[max(0, i_ly-2):min(n_emb, i_ly+3)] = np.inf
                    nn = np.argmin(diffs_ly)
                    d0 = np.sqrt(diffs_ly[nn])
                    if d0 > 0 and nn + 1 < n_emb and i_ly + 1 < n_emb:
                        d1 = np.sqrt(np.sum((embedded[i_ly + 1] - embedded[nn + 1]) ** 2))
                        if d1 > 0:
                            lyap_vals.append(np.log(d1 / d0))
                features["lyapunov_exponent"] = float(np.mean(lyap_vals)) if lyap_vals else 0.0
            else:
                features["lyapunov_exponent"] = 0.0
        else:
            features["lyapunov_exponent"] = 0.0

        # --- Recurrence Quantification Analysis (fully vectorized) ---
        n_rqa = min(len(vr), 150)
        rqa_data = vr[::max(1, len(vr) // n_rqa)][:n_rqa]
        rqa_std = np.std(rqa_data)
        if rqa_std > 0 and len(rqa_data) >= 10:
            threshold = rqa_std * 0.5
            # Full distance matrix via broadcasting (150×150 = tiny)
            dist_mat = np.abs(rqa_data[:, None] - rqa_data[None, :])
            rec_mat = dist_mat < threshold
            np.fill_diagonal(rec_mat, False)
            n_rq = len(rqa_data)
            total_pairs = n_rq * (n_rq - 1)
            features["rqa_recurrence_rate"] = float(rec_mat.sum()) / total_pairs if total_pairs > 0 else 0.0
            # Determinism: diagonal lines in recurrence matrix
            diag_match = rec_mat[:-1, :-1] & rec_mat[1:, 1:]
            features["rqa_determinism"] = float(diag_match.sum()) / max(float(rec_mat.sum()), 1)
        else:
            features["rqa_recurrence_rate"] = 0.0
            features["rqa_determinism"] = 0.0
    else:
        features["hurst_exponent"] = 0.5
        features["hurst_regime"] = 0.0
        features["lyapunov_exponent"] = 0.0
        features["rqa_recurrence_rate"] = 0.0
        features["rqa_determinism"] = 0.0

    # -----------------------------------------------------------------------
    # 53. Signal Processing — Wavelet & Hilbert
    # -----------------------------------------------------------------------
    if n >= 64 and len(valid_returns) >= 32:
        vr_sp = valid_returns

        # --- Wavelet energy decomposition (Haar wavelet, manual) ---
        # Decompose into scales: each level halves the resolution
        def _haar_wavelet_energy(signal, max_levels=5):
            """Compute energy at each Haar wavelet scale."""
            energies = []
            approx = signal.copy()
            for lev in range(max_levels):
                if len(approx) < 4:
                    break
                n_a = len(approx) // 2 * 2  # make even
                approx_trunc = approx[:n_a]
                # Detail coefficients (high-pass)
                detail = (approx_trunc[::2] - approx_trunc[1::2]) / np.sqrt(2)
                # Approximation (low-pass)
                approx = (approx_trunc[::2] + approx_trunc[1::2]) / np.sqrt(2)
                energies.append(float(np.sum(detail ** 2)))
            return energies

        wave_energies = _haar_wavelet_energy(vr_sp)
        total_wave_energy = sum(wave_energies) if wave_energies else 1.0

        if total_wave_energy > 0 and len(wave_energies) >= 2:
            # Energy at each scale (normalized)
            for wi, we in enumerate(wave_energies[:5]):
                features[f"wavelet_energy_s{wi+1}"] = we / total_wave_energy

            # Pad if fewer than 5 levels
            for wi in range(len(wave_energies), 5):
                features[f"wavelet_energy_s{wi+1}"] = 0.0

            # High-freq vs low-freq energy ratio
            hf_energy = sum(wave_energies[:len(wave_energies)//2])
            lf_energy = sum(wave_energies[len(wave_energies)//2:])
            features["wavelet_hf_lf_ratio"] = hf_energy / lf_energy if lf_energy > 0 else 1.0

            # Wavelet entropy
            we_probs = np.array(wave_energies) / total_wave_energy
            we_probs = we_probs[we_probs > 0]
            features["wavelet_entropy"] = float(-np.sum(we_probs * np.log2(we_probs)))
        else:
            for wi in range(5):
                features[f"wavelet_energy_s{wi+1}"] = 0.0
            features["wavelet_hf_lf_ratio"] = 1.0
            features["wavelet_entropy"] = 0.0

        # --- Hilbert transform: instantaneous frequency & amplitude ---
        try:
            analytic = scipy_hilbert(vr_sp[:min(len(vr_sp), 2000)])
            amplitude_env = np.abs(analytic)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) / (2 * np.pi)

            features["hilbert_mean_amplitude"] = float(np.mean(amplitude_env))
            features["hilbert_std_amplitude"] = float(np.std(amplitude_env))
            features["hilbert_mean_freq"] = float(np.mean(inst_freq))
            features["hilbert_std_freq"] = float(np.std(inst_freq))
            # Amplitude-frequency correlation
            if len(inst_freq) >= 3:
                features["hilbert_amp_freq_corr"] = _safe_corr(
                    amplitude_env[:-1], inst_freq
                )
            else:
                features["hilbert_amp_freq_corr"] = 0.0
        except Exception:
            features["hilbert_mean_amplitude"] = 0.0
            features["hilbert_std_amplitude"] = 0.0
            features["hilbert_mean_freq"] = 0.0
            features["hilbert_std_freq"] = 0.0
            features["hilbert_amp_freq_corr"] = 0.0
    else:
        for wi in range(5):
            features[f"wavelet_energy_s{wi+1}"] = 0.0
        features["wavelet_hf_lf_ratio"] = 1.0
        features["wavelet_entropy"] = 0.0
        features["hilbert_mean_amplitude"] = 0.0
        features["hilbert_std_amplitude"] = 0.0
        features["hilbert_mean_freq"] = 0.0
        features["hilbert_std_freq"] = 0.0
        features["hilbert_amp_freq_corr"] = 0.0

    # -----------------------------------------------------------------------
    # 54. Biological / Ecological Analogies
    # -----------------------------------------------------------------------
    if n >= 30:
        # --- Predator-prey dynamics (Lotka-Volterra) ---
        # "Predators" = large trades (>2× median), "Prey" = small trades
        med_sz = np.median(sizes)
        predator_mask = sizes > 2 * med_sz
        prey_mask = ~predator_mask
        n_pred = predator_mask.sum()
        n_prey = prey_mask.sum()

        features["predator_prey_ratio"] = n_pred / n_prey if n_prey > 0 else 0.0
        features["predator_vol_share"] = sizes[predator_mask].sum() / total_vol if total_vol > 0 else 0.0

        # Do predators follow prey? (large trades after clusters of small trades)
        if n >= 50:
            # Split into 10 time slices
            slice_sz = n // 10
            pred_counts = []
            prey_counts = []
            for sl in range(10):
                s_start = sl * slice_sz
                s_end = (sl + 1) * slice_sz if sl < 9 else n
                pred_counts.append(predator_mask[s_start:s_end].sum())
                prey_counts.append(prey_mask[s_start:s_end].sum())
            pred_counts = np.array(pred_counts, dtype=float)
            prey_counts = np.array(prey_counts, dtype=float)
            # Correlation with lag: do predators follow prey?
            if len(pred_counts) >= 3 and len(prey_counts) >= 3:
                features["predator_follows_prey"] = _safe_corr(prey_counts[:-1], pred_counts[1:])
            else:
                features["predator_follows_prey"] = 0.0
        else:
            features["predator_follows_prey"] = 0.0

        # --- Population dynamics ---
        # Trade rate as "population" — estimate growth rate
        if candle_duration_s > 0:
            # Split into 5 epochs, measure trade rate in each
            epoch_sz = n // 5
            if epoch_sz >= 3:
                epoch_rates = []
                for ep in range(5):
                    ep_start = ep * epoch_sz
                    ep_end = (ep + 1) * epoch_sz if ep < 4 else n
                    ep_duration = (timestamps_s[min(ep_end-1, n-1)] - timestamps_s[ep_start])
                    if ep_duration > 0:
                        epoch_rates.append((ep_end - ep_start) / ep_duration)
                    else:
                        epoch_rates.append(0.0)
                epoch_rates = np.array(epoch_rates)
                if epoch_rates[0] > 0:
                    features["population_growth_rate"] = (epoch_rates[-1] - epoch_rates[0]) / epoch_rates[0]
                else:
                    features["population_growth_rate"] = 0.0
                # Carrying capacity: max sustainable rate
                features["carrying_capacity_ratio"] = epoch_rates.max() / np.mean(epoch_rates) if np.mean(epoch_rates) > 0 else 1.0
            else:
                features["population_growth_rate"] = 0.0
                features["carrying_capacity_ratio"] = 1.0
        else:
            features["population_growth_rate"] = 0.0
            features["carrying_capacity_ratio"] = 1.0

        # --- Ecosystem diversity (Simpson's index) ---
        # Diversity of trade sizes (binned into 10 size categories)
        if total_vol > 0:
            size_bins = np.clip(
                (np.log1p(sizes) / np.log1p(sizes.max()) * 9).astype(int), 0, 9
            ) if sizes.max() > 0 else np.zeros(n, dtype=int)
            bin_counts = np.bincount(size_bins, minlength=10).astype(float)
            bin_probs = bin_counts / bin_counts.sum() if bin_counts.sum() > 0 else np.ones(10) / 10
            # Simpson's diversity: 1 - Σ(p²)
            features["simpson_diversity"] = 1.0 - float(np.sum(bin_probs ** 2))
            # Shannon diversity (already have entropy, but this is for sizes)
            bp_pos = bin_probs[bin_probs > 0]
            features["shannon_size_diversity"] = float(-np.sum(bp_pos * np.log2(bp_pos)))
        else:
            features["simpson_diversity"] = 0.0
            features["shannon_size_diversity"] = 0.0
    else:
        features["predator_prey_ratio"] = 0.0
        features["predator_vol_share"] = 0.0
        features["predator_follows_prey"] = 0.0
        features["population_growth_rate"] = 0.0
        features["carrying_capacity_ratio"] = 1.0
        features["simpson_diversity"] = 0.0
        features["shannon_size_diversity"] = 0.0

    # -----------------------------------------------------------------------
    # 55. Compression / Complexity
    # -----------------------------------------------------------------------
    if n >= 30:
        # --- Approximate Entropy (ApEn) & Sample Entropy — fully vectorized ---
        # Subsample to 100 points max for O(n²) broadcasting
        n_sub_ae = 100
        vr_apen = valid_returns[::max(1, len(valid_returns) // n_sub_ae)][:n_sub_ae]
        n_ae = len(vr_apen)
        r_ae = 0.2 * np.std(vr_apen)
        if n_ae >= 10 and r_ae > 0:
            try:
                # Build template matrices via broadcasting (no Python loops)
                def _phi_broadcast(data, m_val, r_val):
                    n_d = len(data) - m_val + 1
                    if n_d < 2:
                        return 0.0
                    tmpl = np.array([data[i:i + m_val] for i in range(n_d)])
                    # Pairwise Chebyshev distance matrix: (n_d, n_d)
                    dists = np.max(np.abs(tmpl[:, None, :] - tmpl[None, :, :]), axis=2)
                    counts = np.sum(dists <= r_val, axis=1).astype(float) / n_d
                    counts = counts[counts > 0]
                    return float(np.mean(np.log(counts)))

                phi2 = _phi_broadcast(vr_apen, 2, r_ae)
                phi3 = _phi_broadcast(vr_apen, 3, r_ae)
                features["approx_entropy"] = abs(phi2 - phi3)

                # SampEn: same but exclude self-matches, use upper triangle
                def _sampen_broadcast(data, m_val, r_val):
                    n_d = len(data) - m_val
                    if n_d < 2:
                        return 0
                    tmpl = np.array([data[i:i + m_val] for i in range(n_d)])
                    dists = np.max(np.abs(tmpl[:, None, :] - tmpl[None, :, :]), axis=2)
                    # Upper triangle only (exclude self and double-counting)
                    return int(np.sum(np.triu(dists <= r_val, k=1)))

                b = _sampen_broadcast(vr_apen, 2, r_ae)
                a = _sampen_broadcast(vr_apen, 3, r_ae)
                features["sample_entropy"] = -np.log(a / b) if b > 0 and a > 0 else 0.0
            except Exception:
                features["approx_entropy"] = 0.0
                features["sample_entropy"] = 0.0
        else:
            features["approx_entropy"] = 0.0
            features["sample_entropy"] = 0.0

        # --- Permutation entropy (order 3, subsample for speed) ---
        price_sub = prices[:min(n, 1000)]
        n_pe = len(price_sub)
        if n_pe >= 10:
            patterns = {}
            for i_pe in range(n_pe - 2):
                pattern = tuple(np.argsort(price_sub[i_pe:i_pe + 3]))
                patterns[pattern] = patterns.get(pattern, 0) + 1
            total_patterns = sum(patterns.values())
            if total_patterns > 0:
                probs = np.array(list(patterns.values())) / total_patterns
                probs = probs[probs > 0]
                pe = float(-np.sum(probs * np.log2(probs)))
                features["permutation_entropy"] = pe / np.log2(6)  # 3! = 6
            else:
                features["permutation_entropy"] = 0.0
        else:
            features["permutation_entropy"] = 0.0

        # --- Compression ratio (run-length encoding efficiency) ---
        dir_seq_comp = np.sign(np.diff(prices))
        if len(dir_seq_comp) >= 5:
            # RLE: count runs
            runs = 1
            for i_c in range(1, len(dir_seq_comp)):
                if dir_seq_comp[i_c] != dir_seq_comp[i_c - 1]:
                    runs += 1
            features["rle_compression_ratio"] = runs / len(dir_seq_comp)
            # Low ratio = highly compressible = trending
            # High ratio = incompressible = noisy
        else:
            features["rle_compression_ratio"] = 1.0
    else:
        features["approx_entropy"] = 0.0
        features["sample_entropy"] = 0.0
        features["permutation_entropy"] = 0.0
        features["rle_compression_ratio"] = 1.0

    return features


# ---------------------------------------------------------------------------
# Streaming aggregation — one day at a time
# ---------------------------------------------------------------------------


def process_day_into_candles(
    day_df: pd.DataFrame,
    freq: str,
) -> list[dict]:
    """Slice one day of ticks into candle bins, compute features, return rows."""
    if len(day_df) == 0:
        return []
    day_df = day_df.set_index("ts")
    groups = day_df.groupby(pd.Grouper(freq=freq))
    rows = []
    for ts, grp in groups:
        if len(grp) == 0:
            continue
        feats = compute_features(grp.reset_index())
        feats["candle_time"] = ts
        rows.append(feats)
    return rows


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward return columns for predictability analysis."""
    for periods, label in [(1, "fwd_ret_1"), (2, "fwd_ret_2"), (3, "fwd_ret_3"),
                           (5, "fwd_ret_5"), (10, "fwd_ret_10")]:
        df[label] = df["return"].shift(-periods)
    return df


def add_cross_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that require cross-candle context (lagged values)."""
    # Return reversal score: current return × previous return
    df["return_reversal"] = df["return"] * df["return"].shift(1)

    # Consecutive same-direction candle count
    direction = np.sign(df["return"])
    changed = direction != direction.shift(1)
    groups = changed.cumsum()
    df["consecutive_direction"] = groups.groupby(groups).cumcount() + 1

    # Volume surprise: current / EMA(5)
    vol_ema = df["total_volume"].ewm(span=5, min_periods=3).mean()
    df["volume_surprise"] = df["total_volume"] / vol_ema.replace(0, np.nan)

    # Range surprise: current range / EMA(10) of range
    range_ema = df["range"].ewm(span=10, min_periods=5).mean()
    df["range_surprise"] = df["range"] / range_ema.replace(0, np.nan)

    # OFI persistence: correlation of OFI with previous 5 candles' OFI
    df["ofi_persistence"] = df["order_flow_imbalance"].rolling(5, min_periods=3).apply(
        lambda x: _safe_corr(x, np.arange(len(x))),
        raw=True
    )

    # -------------------------------------------------------------------
    # Fair Value Gaps (FVG) — 3-candle pattern
    # Bullish FVG: candle[i-2].high < candle[i].low (gap up not filled)
    # Bearish FVG: candle[i-2].low > candle[i].high (gap down not filled)
    # -------------------------------------------------------------------
    prev2_high = df["high"].shift(2)
    prev2_low = df["low"].shift(2)
    cur_low = df["low"]
    cur_high = df["high"]

    # Bullish FVG: gap between candle[-2] high and current low
    bull_fvg_size = cur_low - prev2_high
    bear_fvg_size = prev2_low - cur_high

    df["fvg_bullish"] = (bull_fvg_size > 0).astype(float)
    df["fvg_bearish"] = (bear_fvg_size > 0).astype(float)
    df["fvg_bullish_size_bps"] = np.where(
        bull_fvg_size > 0,
        bull_fvg_size / df["close"] * 10000,
        0.0
    )
    df["fvg_bearish_size_bps"] = np.where(
        bear_fvg_size > 0,
        bear_fvg_size / df["close"] * 10000,
        0.0
    )
    df["fvg_net"] = df["fvg_bullish"].astype(float) - df["fvg_bearish"].astype(float)

    # Cumulative unfilled FVG count (rolling 10 candles)
    df["fvg_bullish_count_10"] = df["fvg_bullish"].rolling(10, min_periods=1).sum()
    df["fvg_bearish_count_10"] = df["fvg_bearish"].rolling(10, min_periods=1).sum()

    # -------------------------------------------------------------------
    # Cross-candle Fair Value features
    # -------------------------------------------------------------------
    # Distance from current close to previous candle's fair value
    prev_fv = df["fair_value"].shift(1)
    df["close_vs_prev_fair_value_bps"] = np.where(
        prev_fv > 0,
        (df["close"] - prev_fv) / prev_fv * 10000,
        0.0
    )

    # Distance from current close to previous POC
    prev_poc = df["poc_price"].shift(1)
    df["close_vs_prev_poc_bps"] = np.where(
        prev_poc > 0,
        (df["close"] - prev_poc) / prev_poc * 10000,
        0.0
    )

    # Fair value trend: is fair value drifting up or down?
    df["fair_value_change_bps"] = np.where(
        prev_fv > 0,
        (df["fair_value"] - prev_fv) / prev_fv * 10000,
        0.0
    )

    # POC migration: how far did POC move between candles?
    df["poc_migration_bps"] = np.where(
        prev_poc > 0,
        (df["poc_price"] - prev_poc) / prev_poc * 10000,
        0.0
    )

    # Value area overlap with previous candle
    prev_va_low = df["value_area_low"].shift(1)
    prev_va_high = df["value_area_high"].shift(1)
    overlap_low = np.maximum(df["value_area_low"], prev_va_low)
    overlap_high = np.minimum(df["value_area_high"], prev_va_high)
    overlap_width = np.maximum(0, overlap_high - overlap_low)
    cur_va_width = df["value_area_high"] - df["value_area_low"]
    df["value_area_overlap_pct"] = np.where(
        cur_va_width > 0,
        overlap_width / cur_va_width,
        0.0
    )

    return df


# ---------------------------------------------------------------------------
# Rolling z-score features
# ---------------------------------------------------------------------------

# Features where z-score is meaningful (scale-dependent, time-varying magnitude)
ZSCORE_FEATURES = [
    # Volume / notional
    "total_volume", "total_notional", "buy_volume", "sell_volume",
    "net_volume_delta", "total_trades", "buy_trades", "sell_trades",
    # Trade size stats
    "avg_trade_size", "median_trade_size", "max_trade_size", "std_trade_size",
    "avg_buy_size", "avg_sell_size", "large_trade_count",
    "trade_size_p25", "trade_size_p75", "trade_size_iqr",
    # VWAP/TWAP spreads (not the price levels themselves)
    "vwap_spread", "vwap_spread_bps", "twap_vwap_spread_bps",
    "close_to_vwap",
    # Volatility
    "realized_vol", "garman_klass_vol", "parkinson_vol",
    "up_volatility", "down_volatility", "vol_asymmetry",
    "return_skewness", "return_kurtosis",
    # Trade arrival / intensity
    "trades_per_second", "median_inter_trade_ms", "inter_trade_std",
    "inter_trade_skew", "max_trades_per_second", "trade_rate_std",
    "volume_per_second_std",
    # Consecutive runs
    "side_switches", "max_buy_run", "max_sell_run",
    # CVD (absolute values, not the bounded ratio)
    "cvd_final", "cvd_max", "cvd_min", "cvd_range",
    # Price path
    "range", "price_path_length", "path_length_over_range",
    # Liquidity
    "kyle_lambda", "amihud_illiquidity", "roll_spread",
    "avg_notional_per_trade",
    # Size distribution
    "volume_skewness", "volume_kurtosis",
    # Intra-candle momentum
    "first_half_return", "second_half_return", "intracandle_reversal",
    "ofi_shift",
    # Order flow (these are bounded but z-score captures regime shifts)
    "order_flow_imbalance", "trade_imbalance",
    # Return
    "return", "return_autocorr_1",
    # --- NEW: Entropy ---
    "volume_entropy", "inter_trade_entropy", "side_sequence_entropy",
    "price_tick_entropy",
    # --- NEW: Toxicity ---
    "vpin", "toxic_flow_ratio", "effective_spread_bps",
    "price_impact_asymmetry",
    # --- NEW: Clustering ---
    "size_clustering", "price_clustering", "temporal_clustering",
    "volume_autocorr",
    # --- NEW: Acceleration ---
    "volume_acceleration", "price_acceleration", "price_curvature",
    "vol_of_vol",
    # --- NEW: Cross-side interaction ---
    "absorption_ratio", "aggression_imbalance", "response_asymmetry",
    "max_up_sweep", "max_down_sweep", "sweep_count",
    # --- NEW: Fractal ---
    "hurst_exponent", "fractal_dimension",
    # --- NEW: Tail / extreme ---
    "max_drawdown", "max_drawup", "drawdown_drawup_asymmetry",
    "tail_volume_ratio", "flash_event_count", "flash_event_pct",
    # --- NEW: Cross-candle context ---
    "volume_surprise", "range_surprise", "consecutive_direction",
    "return_reversal", "ofi_persistence",
    # --- NEW: Volume Profile & Fair Value ---
    "close_to_poc_bps", "value_area_width_bps",
    "low_volume_node_pct", "high_volume_node_pct", "volume_profile_skew",
    "close_to_fair_price_bps", "close_to_fair_value_bps",
    "absorption_ratio",
    # --- NEW: Fair Value Gaps & cross-candle FV ---
    "fvg_bullish_size_bps", "fvg_bearish_size_bps",
    "fvg_bullish_count_10", "fvg_bearish_count_10",
    "close_vs_prev_fair_value_bps", "close_vs_prev_poc_bps",
    "fair_value_change_bps", "poc_migration_bps",
    "value_area_overlap_pct",
    # --- NEW: Physics — Newtonian Mechanics ---
    "market_momentum", "market_force", "market_impulse",
    "kinetic_energy", "potential_energy", "total_energy",
    "market_inertia",
    # --- NEW: Physics — Thermodynamics ---
    "market_temperature", "market_pressure", "pv_work",
    "heat_capacity", "boltzmann_entropy",
    "temperature_change", "heating_rate",
    # --- NEW: Physics — Electromagnetism ---
    "ofi_field_gradient", "temporal_dipole", "dipole_strength",
    "price_dipole", "vwap_flux", "vwap_flux_normalized",
    # --- NEW: Physics — Fluid Dynamics ---
    "market_viscosity", "reynolds_number", "turbulence",
    "flow_velocity", "bernoulli",
    # --- NEW: Physics — Wave Physics ---
    "wave_amplitude", "wave_frequency", "zero_crossing_rate",
    "wave_wavelength", "standing_wave_ratio", "wave_energy",
    # --- NEW: Physics — Gravity ---
    "vwap_gravity", "escape_velocity", "orbital_energy",
    "binding_energy", "centripetal_accel",
    # --- NEW: Psychology — Anchoring ---
    "anchor_dist_100_bps", "anchor_dist_500_bps", "anchor_dist_1k_bps",
    "round_level_magnet_pct", "drift_from_open_bps",
    # --- NEW: Psychology — Loss Aversion ---
    "sell_urgency_ratio", "panic_sell_ratio", "fomo_buy_ratio",
    "contrarian_ratio", "loss_aversion_ratio", "down_move_vol_pct",
    # --- NEW: Psychology — Herding ---
    "vol_price_feedback", "herding_ratio", "attention_effect",
    # --- NEW: Psychology — Disposition ---
    "disposition_sell", "disposition_buy", "disposition_effect",
    # --- NEW: Psychology — Regret / Hesitation ---
    "post_shock_size_ratio", "trade_rate_decay",
    "post_shock_pause_ratio",
    # --- NEW: Psychology — Overconfidence ---
    "size_escalation", "size_gini", "extreme_price_pct",
    # --- NEW: Psychology — Fear & Greed ---
    "micro_fear_greed", "micro_fear_score", "micro_greed_score",
    # --- NEW: Psychology — Attention / Surprise ---
    "shock_attention_ratio", "rubberneck_ratio", "surprise_magnitude",
    # --- NEW: Temporal Distribution ---
    "activity_centroid", "activity_centroid_vol",
    "buy_centroid", "sell_centroid", "centroid_buy_sell_gap",
    "activity_peak_to_mean", "volume_peak_to_mean",
    "arrival_time_kurtosis", "burstiness",
    "trade_pct_q1", "trade_pct_q2", "trade_pct_q3", "trade_pct_q4",
    "volume_pct_q1", "volume_pct_q2", "volume_pct_q3", "volume_pct_q4",
    "quartile_count_ratio", "quartile_vol_ratio",
    "front_back_vol_ratio", "middle_edge_vol_ratio",
    "trade_time_entropy", "trade_time_uniformity",
    "volume_time_entropy", "volume_time_uniformity",
    "activity_ramp", "volume_ramp", "activity_curvature",
    # --- NEW: Math — Linear Algebra ---
    "eigen_ratio", "eigen_sum", "pv_principal_angle",
    # --- NEW: Math — Geometry ---
    "convex_hull_area", "path_hull_ratio",
    "ohlc_triangle_area", "candle_aspect_ratio",
    # --- NEW: Math — Angles & Slopes ---
    "price_slope_angle", "price_volume_angle_cos",
    # --- NEW: Math — Distance Metrics ---
    "manhattan_distance", "chebyshev_distance", "half_cosine_similarity",
    # --- NEW: Math — Calculus ---
    "signed_area_bps", "area_above_vwap_pct", "price_jerk",
    # --- NEW: Math — Topology ---
    "num_extrema", "extrema_density",
    "max_monotonic_run", "monotonicity_score", "tortuosity",
    # --- NEW: Deep Fibonacci ---
    "fib_nearest_dist", "fib_total_crosses", "fib_crosses_per_level",
    "fib_vol_concentration", "fib_vol_ratio",
    "fib_bounce_count", "fib_respect_score",
    "fib_time_vol_pct", "fib_time_trade_pct", "fib_time_vol_surprise",
    "golden_ratio_bs_dist", "golden_ratio_half_dist",
    "golden_quartile_pairs", "golden_ratio_composite",
    "fib_size_ratio_count", "fib_size_ratio_pct", "fib_size_max_med_dist",
    "fib_wave_ratio_count", "fib_wave_ratio_pct", "fib_wave_avg_dist",
    "fib_swing_count", "fib_updown_wave_ratio", "fib_updown_golden_dist",
    "golden_angle_deviation", "golden_angle_uniformity", "sunflower_score",
    "fib_intertime_ratio_pct", "fib_time_scaling_score",
    # --- NEW: Elliott Wave ---
    "ew_impulse_quality", "ew_correction_quality",
    "ew_vol_exhaustion", "ew_amplitude_trend", "ew_amplitude_trend_norm",
    "ew_wave_count", "ew_swing_count",
    "ew_impulse_correction_ratio", "ew_alternation_score",
    "ew_wave_symmetry", "ew_net_direction",
    "ew_last_wave_pct", "ew_first_wave_pct",
    # --- NEW: Math — Spectral / Fourier ---
    "dominant_freq", "dominant_freq_power_pct",
    "spectral_energy_ratio", "spectral_entropy",
    "spectral_flatness", "spectral_centroid",
    # --- NEW: Taylor Series ---
    "taylor_price_a1", "taylor_price_a2", "taylor_price_a3",
    "taylor_price_a4", "taylor_price_a5",
    "taylor_price_r2", "taylor_price_rmse",
    "taylor_curvature_trend_ratio", "taylor_asymmetry_curvature_ratio",
    "taylor_complexity",
    "taylor_vol_a1", "taylor_vol_a2", "taylor_vol_a3", "taylor_vol_r2",
    "taylor_vol_accel_ratio",
    "taylor_ofi_a1", "taylor_ofi_a2", "taylor_ofi_a3", "taylor_ofi_r2",
    "taylor_rate_a1", "taylor_rate_a2", "taylor_rate_a3", "taylor_rate_r2",
    "taylor_price_vol_divergence", "taylor_price_rate_divergence",
    # --- NEW: Information Theory ---
    "te_price_to_volume", "te_volume_to_price", "te_net_direction",
    "mi_side_return", "lz_complexity", "lz_complexity_norm", "avg_surprise",
    # --- NEW: Game Theory ---
    "nash_balance", "stackelberg_buy_leader", "auction_clearing_speed",
    "large_trade_centroid", "large_trade_time_std",
    # --- NEW: Network / Graph Theory ---
    "graph_transition_entropy", "graph_transition_uniformity",
    "graph_self_loop_ratio", "graph_edge_count", "graph_edge_density",
    "graph_asymmetry", "price_level_recurrence",
    "avg_visits_per_level", "max_visits_level",
    # --- NEW: Chaos Theory ---
    "hurst_exponent", "hurst_regime", "lyapunov_exponent",
    "rqa_recurrence_rate", "rqa_determinism",
    # --- NEW: Signal Processing ---
    "wavelet_energy_s1", "wavelet_energy_s2", "wavelet_energy_s3",
    "wavelet_energy_s4", "wavelet_energy_s5",
    "wavelet_hf_lf_ratio", "wavelet_entropy",
    "hilbert_mean_amplitude", "hilbert_std_amplitude",
    "hilbert_mean_freq", "hilbert_std_freq", "hilbert_amp_freq_corr",
    # --- NEW: Biological / Ecological ---
    "predator_prey_ratio", "predator_vol_share", "predator_follows_prey",
    "population_growth_rate", "carrying_capacity_ratio",
    "simpson_diversity", "shannon_size_diversity",
    # --- NEW: Compression / Complexity ---
    "approx_entropy", "sample_entropy", "permutation_entropy",
    "rle_compression_ratio",
]

ZSCORE_WINDOW = 20
ZSCORE_MIN_PERIODS = 10


def add_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling z-score columns for scale-dependent features.

    Z-score = (x - rolling_mean) / rolling_std, using a lookback window
    so the feature captures *deviation from recent norm* rather than
    raw magnitude.  This makes features comparable across regimes.
    """
    cols_present = [c for c in ZSCORE_FEATURES if c in df.columns]
    z_cols = {}
    for col in cols_present:
        rm = df[col].rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
        rs = df[col].rolling(window=ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std()
        z_cols[f"{col}_z"] = (df[col] - rm) / rs.replace(0, np.nan)
    return pd.concat([df, pd.DataFrame(z_cols, index=df.index)], axis=1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def correlation_analysis(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Compute correlation of each feature with forward returns."""
    fwd_cols = [c for c in df.columns if c.startswith("fwd_ret_")]
    feature_cols = [c for c in df.columns
                    if c not in fwd_cols
                    and c not in ("open", "high", "low", "close", "vwap", "twap",
                                  "vwap_buy", "vwap_sell", "candle_time",
                                  "session_asia", "session_europe", "session_us",
                                  "high_before_low", "poc_price", "fair_price",
                                  "fair_value", "value_area_low", "value_area_high",
                                  "close_above_value_area", "close_below_value_area",
                                  "overlap_asia_europe", "overlap_europe_us",
                                  "fvg_bullish", "fvg_bearish",
                                  "fib_nearest_level", "fib_proximity",
                                  "busiest_quartile", "busiest_vol_quartile")
                    and not c.startswith("fwd_")]

    rows = []
    for feat in feature_cols:
        row = {"feature": feat, "timeframe": timeframe}
        for fwd in fwd_cols:
            valid = df[[feat, fwd]].dropna()
            if len(valid) > 30:
                corr, pval = scipy_stats.pearsonr(valid[feat], valid[fwd])
                row[f"{fwd}_corr"] = corr
                row[f"{fwd}_pval"] = pval
                scorr, spval = scipy_stats.spearmanr(valid[feat], valid[fwd])
                row[f"{fwd}_scorr"] = scorr
                row[f"{fwd}_spval"] = spval
            else:
                row[f"{fwd}_corr"] = np.nan
                row[f"{fwd}_pval"] = np.nan
                row[f"{fwd}_scorr"] = np.nan
                row[f"{fwd}_spval"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Tick-to-candle microstructure features")
    parser.add_argument("symbol", help="e.g. BTCUSDT")
    parser.add_argument("start_date", help="YYYY-MM-DD")
    parser.add_argument("end_date", help="YYYY-MM-DD")
    parser.add_argument("--data-dir", default="./data", help="Root data directory")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES.keys()),
                        choices=list(TIMEFRAMES.keys()), help="Timeframes to compute")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol = args.symbol.upper()
    s = datetime.strptime(args.start_date, "%Y-%m-%d")
    e = datetime.strptime(args.end_date, "%Y-%m-%d")
    total_days = (e - s).days + 1

    print("=" * 70)
    print(f"Microstructure Feature Aggregation (streaming, low-RAM)")
    print(f"  Symbol:     {symbol}")
    print(f"  Period:     {args.start_date} -> {args.end_date} ({total_days} days)")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Pass 1: stream through days, compute candle features per timeframe
    # -----------------------------------------------------------------------
    # Accumulate feature rows per timeframe (tiny: ~100 rows/day for 15m)
    tf_rows: dict[str, list[dict]] = {tf: [] for tf in args.timeframes}

    print(f"\n[1/2] Processing tick data day-by-day...")
    t0_all = time.time()
    loaded = 0

    d = s
    while d <= e:
        date_str = d.strftime("%Y-%m-%d")
        fname = f"{symbol}{date_str}.csv.gz"
        path = data_dir / symbol / "bybit" / "futures" / fname

        if path.exists():
            day_t0 = time.time()
            day_df = load_day(path)
            n_trades = len(day_df)
            load_time = time.time() - day_t0

            for tf_label in args.timeframes:
                freq = TIMEFRAMES[tf_label]
                rows = process_day_into_candles(day_df, freq)
                tf_rows[tf_label].extend(rows)

            # Free memory immediately
            del day_df
            gc.collect()

            loaded += 1
            elapsed = time.time() - t0_all
            rate = loaded / elapsed if elapsed > 0 else 0
            remaining = (total_days - loaded) / rate if rate > 0 else 0
            print(f"  [{loaded}/{total_days}] {date_str}: {n_trades:,} trades, "
                  f"load {load_time:.1f}s "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s ETA]")
        else:
            print(f"  [?/{total_days}] {date_str}: MISSING")

        d += timedelta(days=1)

    if loaded == 0:
        print("ERROR: No data files found!")
        sys.exit(1)

    total_elapsed = time.time() - t0_all
    print(f"\n  Processed {loaded} days in {total_elapsed:.0f}s")

    # -----------------------------------------------------------------------
    # Pass 2: build DataFrames, add forward returns, correlations
    # -----------------------------------------------------------------------
    print(f"\n[2/2] Building feature tables & correlation analysis...")
    all_corr = []

    for tf_label in args.timeframes:
        rows = tf_rows[tf_label]
        if not rows:
            print(f"  {tf_label}: no candles, skipping")
            continue

        features_df = pd.DataFrame(rows)
        features_df.set_index("candle_time", inplace=True)
        features_df.sort_index(inplace=True)

        # Merge partial candles at day boundaries for multi-hour timeframes
        # (groupby already handles this since we use pd.Grouper with UTC)
        # But day-boundary candles may be split — deduplicate by taking last
        features_df = features_df[~features_df.index.duplicated(keep="last")]

        features_df = add_cross_candle_features(features_df)
        features_df = add_zscore_features(features_df)
        features_df = add_forward_returns(features_df)

        out_path = output_dir / f"microstructure_{symbol}_{tf_label}_{args.start_date}_{args.end_date}.csv"
        features_df.to_csv(out_path)
        print(f"  {tf_label}: {len(features_df)} candles, {len(features_df.columns)} features -> {out_path.name}")

        corr_df = correlation_analysis(features_df, tf_label)
        all_corr.append(corr_df)

        # Free
        del features_df, rows
        tf_rows[tf_label] = []
        gc.collect()

    # Combine correlation results
    corr_all = pd.concat(all_corr, ignore_index=True)
    corr_path = output_dir / f"microstructure_correlations_{symbol}_{args.start_date}_{args.end_date}.csv"
    corr_all.to_csv(corr_path, index=False)
    print(f"\n  Saved correlations -> {corr_path.name}")

    # Print top features by absolute Spearman correlation with fwd_ret_1
    print(f"\n{'=' * 70}")
    print(f"TOP PREDICTIVE FEATURES (|Spearman corr| with 1-candle forward return)")
    print(f"{'=' * 70}")

    for tf_label in args.timeframes:
        tf_corr = corr_all[corr_all["timeframe"] == tf_label].copy()
        if "fwd_ret_1_scorr" not in tf_corr.columns:
            continue
        tf_corr["abs_scorr"] = tf_corr["fwd_ret_1_scorr"].abs()
        tf_corr = tf_corr.sort_values("abs_scorr", ascending=False)
        print(f"\n  --- {tf_label} ---")
        print(f"  {'Feature':<35} {'Spearman':>10} {'p-value':>12} {'Pearson':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*10}")
        for _, row in tf_corr.head(20).iterrows():
            feat = row["feature"]
            scorr = row.get("fwd_ret_1_scorr", np.nan)
            spval = row.get("fwd_ret_1_spval", np.nan)
            pcorr = row.get("fwd_ret_1_corr", np.nan)
            sig = "***" if spval < 0.001 else "**" if spval < 0.01 else "*" if spval < 0.05 else ""
            print(f"  {feat:<35} {scorr:>9.4f} {spval:>11.2e} {pcorr:>9.4f} {sig}")

    print(f"\nDone! Results in {output_dir}/")


if __name__ == "__main__":
    main()
