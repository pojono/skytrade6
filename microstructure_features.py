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
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=scipy_stats.ConstantInputWarning)


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
            features["return_autocorr_1"] = float(np.corrcoef(valid_returns[:-1], valid_returns[1:])[0, 1])
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
            features["volume_autocorr"] = float(np.corrcoef(s1, s2)[0, 1])
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
    candle_hour = ts.iloc[0].hour if hasattr(ts.iloc[0], 'hour') else 0
    candle_minute = ts.iloc[0].minute if hasattr(ts.iloc[0], 'minute') else 0
    hour_frac = candle_hour + candle_minute / 60.0

    # Cyclical encoding
    features["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    # Day of week (0=Mon, 6=Sun)
    dow = ts.iloc[0].dayofweek if hasattr(ts.iloc[0], 'dayofweek') else 0
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Distance to next funding rate (8h cycle: 00:00, 08:00, 16:00 UTC)
    funding_hours = [0, 8, 16, 24]
    dist_to_funding = min(abs(fh - hour_frac) for fh in funding_hours)
    features["dist_to_funding_hrs"] = dist_to_funding

    # Session flags (approximate)
    features["session_asia"] = 1.0 if 0 <= candle_hour < 8 else 0.0
    features["session_europe"] = 1.0 if 7 <= candle_hour < 16 else 0.0
    features["session_us"] = 1.0 if 13 <= candle_hour < 22 else 0.0

    # -----------------------------------------------------------------------
    # 23. Relative / cross-candle context features
    #     (computed here as raw values; cross-candle versions added post-hoc)
    # -----------------------------------------------------------------------
    # These are single-candle building blocks; the actual cross-candle
    # features (return_reversal, consecutive_direction, etc.) are computed
    # after the DataFrame is assembled in add_cross_candle_features().

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
        lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1] if len(x) >= 3 and np.std(x) > 0 else 0.0,
        raw=True
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
                                  "high_before_low")
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
