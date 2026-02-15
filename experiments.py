#!/usr/bin/env python3
"""
Multi-experiment edge finder on Bybit futures (BTC, ETH, SOL).
Tests many different signal hypotheses on 7 days first.
Winners get validated on 30 days.

All experiments use Bybit VIP0 fees: 7 bps round-trip.

Experiments:
  1. Contrarian imbalance (baseline from v1)
  2. Momentum after large trades
  3. Volatility breakout (trade range expansion)
  4. Trade acceleration momentum
  5. VWAP reversion
  6. Volume surge + direction
  7. Spread between buy/sell aggression
  8. Candle pattern (wick rejection)
  9. Multi-timeframe: 5m signal, 15m confirmation
 10. Mean-reversion on returns (pure price)
"""

import sys
import time
import psutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SOURCE = "bybit_futures"
PARQUET_DIR = Path("./parquet")
ROUND_TRIP_FEE_BPS = 7.0

# Test periods
PERIOD_7D = ("2025-12-01", "2025-12-07")
PERIOD_30D = ("2025-12-01", "2025-12-30")


# ---------------------------------------------------------------------------
# Data loading (5m features from tick data)
# ---------------------------------------------------------------------------

def compute_features_5m(trades):
    """Full feature set from tick data at 5m intervals."""
    INTERVAL_US = 300_000_000
    bucket = (trades["timestamp_us"].values // INTERVAL_US) * INTERVAL_US
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

        q90 = np.percentile(q, 90)
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)
        large_vol_pct = q[large_mask].sum() / max(total_vol, 1e-10)

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

        price_mid = (p.max() + p.min()) / 2
        vol_above = q[p >= price_mid].sum()
        vol_below = q[p < price_mid].sum()
        vol_profile_skew = (vol_above - vol_below) / max(total_vol, 1e-10)

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        full_range = high_p - low_p
        if full_range > 0:
            upper_wick = (high_p - max(open_p, close_p)) / full_range
            lower_wick = (min(open_p, close_p) - low_p) / full_range
            body_pct = abs(close_p - open_p) / full_range
        else:
            upper_wick = 0.0; lower_wick = 0.0; body_pct = 0.0

        # Buy/sell aggression: avg trade size by side
        avg_buy_size = buy_vol / max(buy_count, 1)
        avg_sell_size = sell_vol / max(sell_count, 1)
        size_imbalance = (avg_buy_size - avg_sell_size) / max(avg_buy_size + avg_sell_size, 1e-10)

        features.append({
            "timestamp_us": bkt,
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "large_imbalance": large_imbalance,
            "large_vol_pct": large_vol_pct,
            "count_imbalance": count_imbalance,
            "arrival_rate": arrival_rate,
            "iti_cv": iti_cv,
            "trade_acceleration": trade_acceleration,
            "price_range": price_range,
            "close_vs_vwap": close_vs_vwap,
            "kyle_lambda": kyle_lambda,
            "vol_profile_skew": vol_profile_skew,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "body_pct": body_pct,
            "size_imbalance": size_imbalance,
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "buy_volume": buy_vol, "sell_volume": sell_vol,
            "quote_volume": buy_quote + sell_quote, "trade_count": n,
        })

    return pd.DataFrame(features)


def load_features(symbol, start_date, end_date):
    """Load trades day-by-day, compute features."""
    dates = pd.date_range(start_date, end_date)
    all_feat = []
    t0 = time.time()

    for i, date in enumerate(dates, 1):
        date_str = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{date_str}.parquet"
        if not path.exists():
            continue

        trades = pd.read_parquet(path)
        feat = compute_features_5m(trades)
        del trades
        all_feat.append(feat)

        elapsed = time.time() - t0
        eta = (len(dates) - i) / (i / elapsed) if i > 0 else 0
        mem_gb = psutil.virtual_memory().used / (1024**3)
        if i % 5 == 0 or i == len(dates):
            print(f"    [{i:2d}/{len(dates)}] {date_str}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  RAM={mem_gb:.1f}GB", flush=True)

    if not all_feat:
        return pd.DataFrame()

    df = pd.concat(all_feat, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df


# ---------------------------------------------------------------------------
# Derived features (computed on the feature DataFrame)
# ---------------------------------------------------------------------------

def add_derived_features(df):
    """Add rolling/derived features needed by experiments."""
    # Rolling volatility
    df["rvol_12"] = df["returns"].rolling(12).std()     # 1h
    df["rvol_60"] = df["returns"].rolling(60).std()      # 5h
    df["rvol_288"] = df["returns"].rolling(288).std()    # 1d
    df["vol_ratio"] = df["rvol_12"] / df["rvol_288"].clip(lower=1e-10)

    # Rolling volume z-score
    df["vol_zscore"] = (df["volume"] - df["volume"].rolling(288).mean()) / df["volume"].rolling(288).std().clip(lower=1e-10)

    # Rolling arrival rate z-score
    df["rate_zscore"] = (df["arrival_rate"] - df["arrival_rate"].rolling(288).mean()) / df["arrival_rate"].rolling(288).std().clip(lower=1e-10)

    # Price momentum
    df["mom_12"] = df["close"].pct_change(12)   # 1h momentum
    df["mom_60"] = df["close"].pct_change(60)   # 5h momentum

    # Mean-reversion: z-score of close vs rolling mean
    df["price_zscore_60"] = (df["close"] - df["close"].rolling(60).mean()) / df["close"].rolling(60).std().clip(lower=1e-10)
    df["price_zscore_288"] = (df["close"] - df["close"].rolling(288).mean()) / df["close"].rolling(288).std().clip(lower=1e-10)

    # Range expansion: current range vs rolling avg range
    df["range_zscore"] = (df["price_range"] - df["price_range"].rolling(288).mean()) / df["price_range"].rolling(288).std().clip(lower=1e-10)

    # Cumulative imbalance (rolling sum)
    df["cum_imbalance_6"] = df["vol_imbalance"].rolling(6).sum()    # 30m
    df["cum_imbalance_12"] = df["vol_imbalance"].rolling(12).sum()  # 1h

    # VWAP deviation z-score
    df["vwap_zscore"] = (df["close_vs_vwap"] - df["close_vs_vwap"].rolling(288).mean()) / df["close_vs_vwap"].rolling(288).std().clip(lower=1e-10)

    return df


# ---------------------------------------------------------------------------
# Generic backtest
# ---------------------------------------------------------------------------

def backtest_signal(df, signal_col, entry_threshold, holding_bars, fee_bps,
                    direction="contrarian"):
    """
    Generic backtest.
    direction='contrarian': short when signal high, long when signal low
    direction='momentum': long when signal high, short when signal low
    """
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
            else:  # momentum
                if signals[i] > entry_threshold:
                    in_trade = True; entry_idx = i; trade_dir = 1
                elif signals[i] < -entry_threshold:
                    in_trade = True; entry_idx = i; trade_dir = -1

    return np.array(pnls) if pnls else np.array([])


def zscore_signal(df, col, window=864):
    """Rolling z-score of a column."""
    return (df[col] - df[col].rolling(window, min_periods=288).mean()) / \
           df[col].rolling(window, min_periods=288).std().clip(lower=1e-10)


def rank_composite(df, cols, window=864):
    """Rank-based composite of multiple columns."""
    for col in cols:
        df[f"_rank_{col}"] = df[col].rolling(window, min_periods=288).rank(pct=True)
    rank_cols = [f"_rank_{col}" for col in cols]
    composite = df[rank_cols].mean(axis=1)
    signal = (composite - composite.rolling(window, min_periods=288).mean()) / \
             composite.rolling(window, min_periods=288).std().clip(lower=1e-10)
    # Clean up temp columns
    df.drop(columns=rank_cols, inplace=True)
    return signal


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = []

def register(name, desc):
    """Decorator to register an experiment."""
    def decorator(func):
        EXPERIMENTS.append((name, desc, func))
        return func
    return decorator


@register("E01_contrarian_imbalance", "Contrarian composite imbalance (baseline)")
def exp_contrarian_imbalance(df):
    cols = ["vol_imbalance", "dollar_imbalance", "large_imbalance",
            "count_imbalance", "close_vs_vwap"]
    df["sig"] = rank_composite(df, cols)
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            results.append((thresh, hl, pnls))
    return results


@register("E02_momentum_large_trades", "Momentum: follow large trade direction")
def exp_momentum_large(df):
    df["sig"] = zscore_signal(df, "large_imbalance")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@register("E03_vol_breakout", "Volatility breakout: trade range expansion direction")
def exp_vol_breakout(df):
    # Signal: range expansion + direction (close vs open)
    df["range_dir"] = df["range_zscore"] * np.sign(df["returns"])
    df["sig"] = zscore_signal(df, "range_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@register("E04_trade_acceleration", "Trade acceleration: rising trade rate = momentum")
def exp_trade_accel(df):
    df["accel_dir"] = df["trade_acceleration"] * np.sign(df["returns"])
    df["sig"] = zscore_signal(df, "accel_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@register("E05_vwap_reversion", "VWAP reversion: price far from VWAP reverts")
def exp_vwap_reversion(df):
    df["sig"] = df["vwap_zscore"]
    results = []
    for thresh in [1.0, 1.5, 2.0]:
        for hb, hl in [(6, "30m"), (12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            results.append((thresh, hl, pnls))
    return results


@register("E06_volume_surge_direction", "Volume surge + direction: high volume confirms move")
def exp_vol_surge(df):
    # High volume + positive return = momentum signal
    df["vol_dir"] = df["vol_zscore"] * np.sign(df["returns"])
    df["sig"] = zscore_signal(df, "vol_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@register("E07_size_aggression", "Size aggression: large buys vs large sells")
def exp_size_aggression(df):
    df["sig"] = zscore_signal(df, "size_imbalance")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            # Test both directions
            pnls_c = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            pnls_m = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl + "_c", pnls_c))
            results.append((thresh, hl + "_m", pnls_m))
    return results


@register("E08_wick_rejection", "Wick rejection: long lower wick = buy, long upper wick = sell")
def exp_wick_rejection(df):
    # Net wick signal: lower_wick - upper_wick (positive = rejection of lows = bullish)
    df["wick_signal"] = df["lower_wick"] - df["upper_wick"]
    df["sig"] = zscore_signal(df, "wick_signal")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@register("E09_cumulative_imbalance", "Cumulative imbalance: sustained pressure over 1h")
def exp_cum_imbalance(df):
    df["sig"] = zscore_signal(df, "cum_imbalance_12")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls_c = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            pnls_m = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl + "_c", pnls_c))
            results.append((thresh, hl + "_m", pnls_m))
    return results


@register("E10_price_mean_reversion", "Price mean-reversion: z-score of price vs 5h MA")
def exp_price_mr(df):
    df["sig"] = df["price_zscore_60"]
    results = []
    for thresh in [1.0, 1.5, 2.0]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            results.append((thresh, hl, pnls))
    return results


@register("E11_kyle_lambda_momentum", "Kyle's lambda: high price impact = informed flow, follow it")
def exp_kyle(df):
    df["kyle_dir"] = df["kyle_lambda"] * np.sign(df["returns"])
    df["sig"] = zscore_signal(df, "kyle_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@register("E12_vol_regime_contrarian", "Vol regime contrarian: low vol + imbalance (strongest from nb02)")
def exp_vol_regime(df):
    # Only trade when vol is low (bottom 40%)
    cols = ["vol_imbalance", "count_imbalance", "close_vs_vwap"]
    df["sig"] = rank_composite(df, cols)
    # Mask: set signal to 0 in high vol
    vol_q60 = df["vol_ratio"].quantile(0.4)
    df.loc[df["vol_ratio"] > vol_q60, "sig"] = 0
    results = []
    for thresh in [0.5, 1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            results.append((thresh, hl, pnls))
    return results


@register("E13_momentum_1h", "1h momentum: follow the trend")
def exp_momentum_1h(df):
    df["sig"] = zscore_signal(df, "mom_12")
    results = []
    for thresh in [1.0, 1.5, 2.0]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@register("E14_reversal_after_momentum", "Reversal: fade strong 1h moves")
def exp_reversal(df):
    df["sig"] = zscore_signal(df, "mom_12")
    results = []
    for thresh in [1.5, 2.0, 2.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            results.append((thresh, hl, pnls))
    return results


@register("E15_composite_momentum", "Composite momentum: imbalance + price + volume all aligned")
def exp_composite_mom(df):
    # Everything pointing same direction = strong momentum
    df["aligned"] = (np.sign(df["vol_imbalance"]) + np.sign(df["returns"]) + np.sign(df["vol_zscore"].fillna(0))) / 3
    df["sig"] = zscore_signal(df, "aligned")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest_signal(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiments(symbol, start_date, end_date, label):
    """Run all experiments on one symbol/period."""
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    print(f"\n{'='*70}")
    print(f"  {symbol} — {label} ({days} days: {start_date} → {end_date})")
    print(f"{'='*70}")

    print(f"  Loading features...", flush=True)
    df = load_features(symbol, start_date, end_date)
    if df.empty:
        print(f"  ❌ No data!")
        return []

    print(f"  {len(df):,} bars, price {df['close'].min():.2f}–{df['close'].max():.2f}")
    print(f"  Adding derived features...", flush=True)
    df = add_derived_features(df)

    winners = []

    for exp_name, exp_desc, exp_func in EXPERIMENTS:
        print(f"\n  📋 {exp_name}: {exp_desc}", flush=True)
        df_copy = df.copy()

        try:
            results = exp_func(df_copy)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

        best_avg = -999
        best_cfg = None

        for thresh, hl, pnls in results:
            if len(pnls) < 5:
                continue
            avg = pnls.mean()
            total = pnls.sum()
            wr = (pnls > 0).mean()

            if avg > best_avg:
                best_avg = avg
                best_cfg = (thresh, hl, len(pnls), avg, total, wr)

        if best_cfg:
            thresh, hl, n_trades, avg, total, wr = best_cfg
            marker = "✅" if avg > 0 and n_trades >= 10 else "  "
            print(f"    {marker} Best: thresh={thresh}, hold={hl}, "
                  f"trades={n_trades}, avg={avg:+.2f} bps, total={total:+.1f}, WR={wr:.0%}")

            if avg > 0 and n_trades >= 10:
                winners.append({
                    "experiment": exp_name,
                    "symbol": symbol,
                    "period": label,
                    "threshold": thresh,
                    "holding": hl,
                    "n_trades": n_trades,
                    "avg_pnl_bps": avg,
                    "total_pnl_bps": total,
                    "win_rate": wr,
                })
        else:
            print(f"    — No viable config")

    return winners


def main():
    t_start = time.time()
    print("=" * 70)
    print("  EXPERIMENT SUITE: Edge Discovery on Bybit Futures")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Fees: {ROUND_TRIP_FEE_BPS} bps round-trip (Bybit VIP0)")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print("=" * 70)

    all_winners = []

    # Phase 1: 7-day test on all symbols
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: 7-DAY SCREENING")
    print(f"{'#'*70}")

    for symbol in SYMBOLS:
        winners = run_experiments(symbol, *PERIOD_7D, "7d")
        all_winners.extend(winners)

    # Summary of 7-day winners
    print(f"\n\n{'='*70}")
    print(f"  PHASE 1 RESULTS: 7-Day Winners (avg PnL > 0, trades >= 10)")
    print(f"{'='*70}")

    if not all_winners:
        print("  ❌ No winners found!")
        return

    print(f"  {'Experiment':35s} {'Symbol':>10s} {'Thresh':>7s} {'Hold':>6s} "
          f"{'Trades':>7s} {'Avg':>8s} {'Total':>9s} {'WR':>5s}")
    print(f"  {'-'*90}")

    for w in sorted(all_winners, key=lambda x: -x["avg_pnl_bps"]):
        print(f"  {w['experiment']:35s} {w['symbol']:>10s} {w['threshold']:>7.1f} "
              f"{w['holding']:>6s} {w['n_trades']:>7d} {w['avg_pnl_bps']:>+8.2f} "
              f"{w['total_pnl_bps']:>+9.1f} {w['win_rate']:>5.0%}")

    # Phase 2: 30-day validation of experiments that won on 2+ symbols
    print(f"\n\n{'#'*70}")
    print(f"  PHASE 2: 30-DAY VALIDATION (experiments winning on 7d)")
    print(f"{'#'*70}")

    # Find experiments that won on at least 1 symbol
    winning_experiments = set(w["experiment"] for w in all_winners)
    print(f"  Validating {len(winning_experiments)} experiments: {', '.join(sorted(winning_experiments))}")

    validated_winners = []
    for symbol in SYMBOLS:
        winners_30d = run_experiments(symbol, *PERIOD_30D, "30d")
        # Only keep experiments that also won on 7d
        for w in winners_30d:
            if w["experiment"] in winning_experiments:
                validated_winners.append(w)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS: 30-Day Validated Winners")
    print(f"{'='*70}")

    if not validated_winners:
        print("  ❌ No experiments survived 30-day validation!")
    else:
        print(f"  {'Experiment':35s} {'Symbol':>10s} {'Thresh':>7s} {'Hold':>6s} "
              f"{'Trades':>7s} {'Avg':>8s} {'Total':>9s} {'WR':>5s}")
        print(f"  {'-'*90}")
        for w in sorted(validated_winners, key=lambda x: -x["avg_pnl_bps"]):
            print(f"  {w['experiment']:35s} {w['symbol']:>10s} {w['threshold']:>7.1f} "
                  f"{w['holding']:>6s} {w['n_trades']:>7d} {w['avg_pnl_bps']:>+8.2f} "
                  f"{w['total_pnl_bps']:>+9.1f} {w['win_rate']:>5.0%}")

    elapsed = time.time() - t_start
    print(f"\n✅ All experiments complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
