#!/usr/bin/env python3
"""
Novel edge experiments — ideas from academic microstructure research
applied to crypto tick data on Bybit futures.

Sources of inspiration:
  - Easley, Lopez de Prado, O'Hara (2012): "Flow Toxicity and Liquidity"
    → VPIN (Volume-Synchronized Probability of Informed Trading)
  - Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
    → Order flow imbalance persistence and autocorrelation
  - Bouchaud et al. (2004): "Fluctuations and Response in Financial Markets"
    → Long-memory in order flow, Hurst exponent of signed trades
  - Kyle (1985) + Hasbrouck (1991): Information share / price discovery
    → Trade informativeness decay curve
  - Mandelbrot, multifractal models:
    → Multifractal spectrum of returns for regime detection
  - Entropy-based approaches (Shannon, Transfer entropy):
    → Predictability of trade flow using information theory
  - Amihud (2002) illiquidity + Pastor-Stambaugh (2003):
    → Liquidity-adjusted returns, liquidity shocks
  - Biais, Hillion, Spatt (1995): "An Empirical Analysis of the Limit Order Book"
    → Aggressive vs passive flow ratio changes
  - Lillo & Farmer (2004): "The Long Memory of the Efficient Market"
    → Autocorrelation structure of order signs

All experiments: Bybit VIP0 fees (7 bps RT), 7d screen → 30d validation.
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

PERIOD_7D = ("2025-12-01", "2025-12-07")
PERIOD_30D = ("2025-12-01", "2025-12-30")


# ---------------------------------------------------------------------------
# Tick-level feature computation (richer than v1)
# ---------------------------------------------------------------------------

def compute_rich_features(trades, interval_us=300_000_000):
    """Compute extended feature set including novel academic-inspired features."""
    bucket = (trades["timestamp_us"].values // interval_us) * interval_us
    trades = trades.copy()
    trades["bucket"] = bucket

    features = []
    for bkt, grp in trades.groupby("bucket"):
        p = grp["price"].values
        q = grp["quantity"].values
        qq = grp["quote_quantity"].values
        s = grp["side"].values  # +1 buy, -1 sell
        t = grp["timestamp_us"].values
        n = len(grp)
        if n < 10:
            continue

        buy_mask = s == 1
        sell_mask = s == -1
        buy_vol = q[buy_mask].sum()
        sell_vol = q[sell_mask].sum()
        total_vol = q.sum()
        buy_quote = qq[buy_mask].sum()
        sell_quote = qq[sell_mask].sum()
        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())

        # ===== STANDARD FEATURES =====
        vol_imbalance = (buy_vol - sell_vol) / max(total_vol, 1e-10)
        dollar_imbalance = (buy_quote - sell_quote) / max(buy_quote + sell_quote, 1e-10)
        count_imbalance = (buy_count - sell_count) / max(n, 1)

        q90 = np.percentile(q, 90)
        large_mask = q >= q90
        large_buy = q[large_mask & buy_mask].sum()
        large_sell = q[large_mask & sell_mask].sum()
        large_imbalance = (large_buy - large_sell) / max(large_buy + large_sell, 1e-10)

        vwap = qq.sum() / max(total_vol, 1e-10)
        close_vs_vwap = (p[-1] - vwap) / max(vwap, 1e-10)
        price_range = (p.max() - p.min()) / max(vwap, 1e-10)

        open_p, close_p, high_p, low_p = p[0], p[-1], p.max(), p.min()
        ret = (close_p - open_p) / max(open_p, 1e-10)

        # ===== NOVEL FEATURE 1: VPIN (Volume-Synchronized Probability of Informed Trading) =====
        # Easley, Lopez de Prado, O'Hara (2012)
        # Split interval into volume buckets, measure imbalance in each
        cum_vol = np.cumsum(q)
        vol_per_bucket = total_vol / 5  # 5 sub-buckets
        vpin_imbalances = []
        for vb in range(5):
            lo = vb * vol_per_bucket
            hi = (vb + 1) * vol_per_bucket
            mask = (cum_vol > lo) & (cum_vol <= hi)
            if mask.sum() > 0:
                bv = q[mask & buy_mask].sum()
                sv = q[mask & sell_mask].sum()
                vpin_imbalances.append(abs(bv - sv) / max(bv + sv, 1e-10))
        vpin = np.mean(vpin_imbalances) if vpin_imbalances else 0.0

        # ===== NOVEL FEATURE 2: Order Flow Hurst Exponent =====
        # Lillo & Farmer (2004) — long memory in order signs
        # Estimate via R/S method on trade signs
        signs = s.copy().astype(float)
        if len(signs) > 50:
            # Simplified R/S on chunks
            chunk_size = len(signs) // 5
            rs_values = []
            for ci in range(5):
                chunk = signs[ci*chunk_size:(ci+1)*chunk_size]
                if len(chunk) < 10:
                    continue
                cum_dev = np.cumsum(chunk - chunk.mean())
                R = cum_dev.max() - cum_dev.min()
                S = chunk.std()
                if S > 0:
                    rs_values.append(R / S)
            hurst_proxy = np.mean(rs_values) / max(chunk_size**0.5, 1) if rs_values else 0.5
        else:
            hurst_proxy = 0.5

        # ===== NOVEL FEATURE 3: Trade Sign Autocorrelation (lag 1-5) =====
        # Bouchaud et al. — order flow persistence
        if len(signs) > 20:
            sign_ac1 = np.corrcoef(signs[:-1], signs[1:])[0, 1]
            sign_ac5 = np.corrcoef(signs[:-5], signs[5:])[0, 1] if len(signs) > 10 else 0
        else:
            sign_ac1 = 0.0; sign_ac5 = 0.0

        # ===== NOVEL FEATURE 4: Trade Informativeness Decay =====
        # Hasbrouck (1991) — how quickly does trade impact decay?
        # Measure: correlation between signed volume in first half vs price change in second half
        mid = n // 2
        if mid > 5:
            first_signed_vol = (q[:mid] * s[:mid]).sum()
            second_price_change = p[-1] - p[mid]
            # Normalize
            info_persistence = np.sign(first_signed_vol) * np.sign(second_price_change)
        else:
            info_persistence = 0.0

        # ===== NOVEL FEATURE 5: Entropy of Trade Sizes =====
        # Information theory — low entropy = concentrated (informed), high = dispersed (noise)
        q_nonzero = q[q > 0]
        if len(q_nonzero) > 1:
            # Discretize into 10 bins
            try:
                hist, _ = np.histogram(q_nonzero, bins=10)
                hist = hist[hist > 0]
                probs = hist / hist.sum()
                size_entropy = -np.sum(probs * np.log2(probs))
            except:
                size_entropy = 0.0
        else:
            size_entropy = 0.0

        # ===== NOVEL FEATURE 6: Entropy of Inter-Trade Times =====
        # Low entropy = regular (algorithmic), high = random (organic)
        if n > 5:
            iti = np.diff(t).astype(float)
            iti_pos = iti[iti > 0]
            if len(iti_pos) > 5:
                try:
                    hist, _ = np.histogram(iti_pos, bins=10)
                    hist = hist[hist > 0]
                    probs = hist / hist.sum()
                    time_entropy = -np.sum(probs * np.log2(probs))
                except:
                    time_entropy = 0.0
            else:
                time_entropy = 0.0
        else:
            time_entropy = 0.0

        # ===== NOVEL FEATURE 7: Toxic Flow Indicator =====
        # Inspired by VPIN — when volume-weighted price moves against the
        # majority of volume, it signals toxic (informed) flow
        # Toxic = price moved up but most volume was selling (or vice versa)
        toxic_flow = -vol_imbalance * np.sign(ret) if abs(ret) > 1e-10 else 0.0

        # ===== NOVEL FEATURE 8: Amihud Illiquidity Asymmetry =====
        # Amihud (2002) — but split by buy vs sell side
        if buy_vol > 0 and sell_vol > 0:
            buy_prices = p[buy_mask]
            sell_prices = p[sell_mask]
            buy_impact = abs(buy_prices.max() - buy_prices.min()) / max(buy_vol, 1e-10)
            sell_impact = abs(sell_prices.max() - sell_prices.min()) / max(sell_vol, 1e-10)
            illiq_asymmetry = (buy_impact - sell_impact) / max(buy_impact + sell_impact, 1e-10)
        else:
            illiq_asymmetry = 0.0

        # ===== NOVEL FEATURE 9: Volume Clock Speed =====
        # How fast is volume arriving relative to normal?
        # Faster = more activity = potential informed trading
        duration_s = max((t[-1] - t[0]) / 1e6, 0.001)
        vol_speed = total_vol / duration_s

        # ===== NOVEL FEATURE 10: Price Efficiency Ratio =====
        # Kaufman efficiency: |net move| / sum of |individual moves|
        # High = trending, Low = noisy/mean-reverting
        price_changes = np.abs(np.diff(p))
        total_path = price_changes.sum()
        net_move = abs(p[-1] - p[0])
        efficiency = net_move / max(total_path, 1e-10)

        # ===== NOVEL FEATURE 11: Aggressive Volume Ratio =====
        # Biais et al. — ratio of volume from trades that moved the price
        # vs trades at the same price (aggressive vs passive)
        price_changed = np.diff(p) != 0
        if len(price_changed) > 0:
            aggressive_vol = q[1:][price_changed].sum()
            passive_vol = q[1:][~price_changed].sum()
            aggressive_ratio = aggressive_vol / max(aggressive_vol + passive_vol, 1e-10)
        else:
            aggressive_ratio = 0.5

        # ===== NOVEL FEATURE 12: Signed Volume Runs =====
        # Length of consecutive same-side trades — long runs = herding
        if n > 2:
            sign_changes = np.diff(s) != 0
            runs = np.split(np.arange(len(s) - 1), np.where(sign_changes)[0] + 1)
            run_lengths = [len(r) for r in runs if len(r) > 0]
            avg_run_length = np.mean(run_lengths) if run_lengths else 1
            max_run_length = max(run_lengths) if run_lengths else 1
            # Directional: are buy runs or sell runs longer?
            buy_runs = []
            sell_runs = []
            for r in runs:
                if len(r) > 0 and s[r[0]] == 1:
                    buy_runs.append(len(r))
                elif len(r) > 0:
                    sell_runs.append(len(r))
            avg_buy_run = np.mean(buy_runs) if buy_runs else 0
            avg_sell_run = np.mean(sell_runs) if sell_runs else 0
            run_imbalance = (avg_buy_run - avg_sell_run) / max(avg_buy_run + avg_sell_run, 1e-10)
        else:
            avg_run_length = 1; max_run_length = 1; run_imbalance = 0

        # ===== NOVEL FEATURE 13: Multifractal Volatility =====
        # Mandelbrot — compare realized vol at different sub-scales
        # If vol(small scale) >> vol(large scale), market is in turbulent regime
        if n > 20:
            # 1-trade returns vs 10-trade returns
            ret_1 = np.diff(p) / p[:-1]
            ret_10 = (p[10:] - p[:-10]) / p[:-10] if n > 20 else ret_1
            vol_1 = np.std(ret_1) if len(ret_1) > 1 else 0
            vol_10 = np.std(ret_10) / max(10**0.5, 1) if len(ret_10) > 1 else 0
            # Ratio > 1 = multifractal (clustered vol), < 1 = smooth
            multifractal_ratio = vol_1 / max(vol_10, 1e-10)
        else:
            multifractal_ratio = 1.0

        # ===== NOVEL FEATURE 14: Volume-Weighted Momentum =====
        # Weight each trade's return by its volume — large trades matter more
        if n > 2:
            trade_returns = np.diff(p) / p[:-1]
            trade_vols = q[1:]
            vw_momentum = np.sum(trade_returns * trade_vols) / max(trade_vols.sum(), 1e-10)
        else:
            vw_momentum = 0.0

        # ===== NOVEL FEATURE 15: Surprise Index =====
        # How unexpected is the current bar's activity?
        # Combine: unusual volume + unusual range + unusual imbalance
        # (will be z-scored later at the rolling level)
        surprise_raw = abs(vol_imbalance) * price_range * n

        features.append({
            "timestamp_us": bkt,
            # Standard
            "vol_imbalance": vol_imbalance,
            "dollar_imbalance": dollar_imbalance,
            "count_imbalance": count_imbalance,
            "large_imbalance": large_imbalance,
            "close_vs_vwap": close_vs_vwap,
            "price_range": price_range,
            "returns": ret,
            # Novel
            "vpin": vpin,
            "hurst_proxy": hurst_proxy,
            "sign_ac1": sign_ac1,
            "sign_ac5": sign_ac5,
            "info_persistence": info_persistence,
            "size_entropy": size_entropy,
            "time_entropy": time_entropy,
            "toxic_flow": toxic_flow,
            "illiq_asymmetry": illiq_asymmetry,
            "vol_speed": vol_speed,
            "efficiency": efficiency,
            "aggressive_ratio": aggressive_ratio,
            "avg_run_length": avg_run_length,
            "max_run_length": max_run_length,
            "run_imbalance": run_imbalance,
            "multifractal_ratio": multifractal_ratio,
            "vw_momentum": vw_momentum,
            "surprise_raw": surprise_raw,
            # OHLCV
            "open": open_p, "close": close_p, "high": high_p, "low": low_p,
            "volume": total_vol, "trade_count": n,
        })

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(symbol, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    all_feat = []
    t0 = time.time()

    for i, date in enumerate(dates, 1):
        date_str = date.strftime("%Y-%m-%d")
        path = PARQUET_DIR / symbol / "trades" / SOURCE / f"{date_str}.parquet"
        if not path.exists():
            continue

        trades = pd.read_parquet(path)
        feat = compute_rich_features(trades)
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
    return df


def add_derived(df):
    """Add rolling derived features."""
    # Rolling vol
    df["rvol_12"] = df["returns"].rolling(12).std()
    df["rvol_288"] = df["returns"].rolling(288).std()
    df["vol_ratio"] = df["rvol_12"] / df["rvol_288"].clip(lower=1e-10)

    # Cumulative features
    df["cum_imbalance_12"] = df["vol_imbalance"].rolling(12).sum()
    df["cum_toxic_12"] = df["toxic_flow"].rolling(12).sum()
    df["cum_info_persist_12"] = df["info_persistence"].rolling(12).sum()

    # Rolling means for z-scoring
    for col in ["vpin", "hurst_proxy", "sign_ac1", "size_entropy", "time_entropy",
                "efficiency", "aggressive_ratio", "avg_run_length", "run_imbalance",
                "multifractal_ratio", "vw_momentum", "surprise_raw", "vol_speed",
                "illiq_asymmetry"]:
        mean = df[col].rolling(288, min_periods=60).mean()
        std = df[col].rolling(288, min_periods=60).std().clip(lower=1e-10)
        df[f"{col}_z"] = (df[col] - mean) / std

    # Momentum
    df["mom_12"] = df["close"].pct_change(12)
    df["mom_60"] = df["close"].pct_change(60)

    return df


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(df, signal_col, thresh, hold_bars, fee_bps, direction="contrarian"):
    data = df.dropna(subset=[signal_col]).copy()
    signals = data[signal_col].values
    closes = data["close"].values
    n = len(data)

    pnls = []
    in_trade = False
    entry_idx = 0
    trade_dir = 0

    for i in range(n - hold_bars):
        if in_trade and i - entry_idx >= hold_bars:
            raw = (closes[i] / closes[entry_idx] - 1) * 10000 * trade_dir
            pnls.append(raw - fee_bps)
            in_trade = False

        if not in_trade:
            if direction == "contrarian":
                if signals[i] > thresh:
                    in_trade = True; entry_idx = i; trade_dir = -1
                elif signals[i] < -thresh:
                    in_trade = True; entry_idx = i; trade_dir = 1
            else:
                if signals[i] > thresh:
                    in_trade = True; entry_idx = i; trade_dir = 1
                elif signals[i] < -thresh:
                    in_trade = True; entry_idx = i; trade_dir = -1

    return np.array(pnls) if pnls else np.array([])


def zscore(df, col, window=864):
    return (df[col] - df[col].rolling(window, min_periods=288).mean()) / \
           df[col].rolling(window, min_periods=288).std().clip(lower=1e-10)


def rank_composite(df, cols, window=864):
    for col in cols:
        df[f"_r_{col}"] = df[col].rolling(window, min_periods=288).rank(pct=True)
    rc = [f"_r_{col}" for col in cols]
    comp = df[rc].mean(axis=1)
    sig = (comp - comp.rolling(window, min_periods=288).mean()) / \
          comp.rolling(window, min_periods=288).std().clip(lower=1e-10)
    df.drop(columns=rc, inplace=True)
    return sig


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = []

def reg(name, desc):
    def decorator(func):
        EXPERIMENTS.append((name, desc, func))
        return func
    return decorator


@reg("N01_vpin_toxicity", "VPIN: high toxicity predicts adverse selection → fade it")
def exp_vpin(df):
    df["sig"] = df["vpin_z"]
    results = []
    for thresh in [1.0, 1.5, 2.0]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            for d in ["contrarian", "momentum"]:
                pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, d)
                results.append((thresh, f"{hl}_{d[0]}", pnls))
    return results


@reg("N02_hurst_regime", "Hurst exponent: H>0.5 = trending, H<0.5 = mean-reverting → adapt")
def exp_hurst(df):
    # When Hurst is high, use momentum; when low, use contrarian
    df["hurst_mom"] = df["hurst_proxy_z"] * np.sign(df["returns"])
    df["sig"] = zscore(df, "hurst_mom")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N03_sign_persistence", "Order flow autocorrelation: persistent flow = informed → follow")
def exp_sign_persist(df):
    # High sign_ac1 + direction = informed flow continuing
    df["flow_persist"] = df["sign_ac1"] * np.sign(df["vol_imbalance"])
    df["sig"] = zscore(df, "flow_persist")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N04_info_decay", "Trade informativeness: first-half flow predicts second-half price")
def exp_info_decay(df):
    df["sig"] = zscore(df, "cum_info_persist_12")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N05_entropy_regime", "Low size entropy = concentrated (informed) trading → follow direction")
def exp_entropy(df):
    # Low entropy + directional = informed. Invert entropy, multiply by direction
    df["informed_flow"] = -df["size_entropy_z"] * np.sign(df["vol_imbalance"])
    df["sig"] = zscore(df, "informed_flow")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N06_toxic_flow_cumulative", "Cumulative toxic flow: sustained toxicity = big move coming")
def exp_toxic(df):
    df["sig"] = zscore(df, "cum_toxic_12")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            for d in ["contrarian", "momentum"]:
                pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, d)
                results.append((thresh, f"{hl}_{d[0]}", pnls))
    return results


@reg("N07_illiquidity_shock", "Illiquidity asymmetry: one side dries up → price moves to that side")
def exp_illiq(df):
    df["sig"] = df["illiq_asymmetry_z"]
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            for d in ["contrarian", "momentum"]:
                pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, d)
                results.append((thresh, f"{hl}_{d[0]}", pnls))
    return results


@reg("N08_efficiency_regime", "Price efficiency: low efficiency = noisy → mean-revert; high = trending → follow")
def exp_efficiency(df):
    # High efficiency + direction = trend continuation
    df["eff_dir"] = df["efficiency_z"] * np.sign(df["returns"])
    df["sig"] = zscore(df, "eff_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N09_aggressive_flow", "Aggressive volume ratio: more price-moving trades = informed")
def exp_aggressive(df):
    df["agg_dir"] = df["aggressive_ratio_z"] * np.sign(df["vol_imbalance"])
    df["sig"] = zscore(df, "agg_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N10_herding_runs", "Herding: long same-side runs = herding behavior → contrarian fade")
def exp_herding(df):
    df["sig"] = zscore(df, "run_imbalance")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            for d in ["contrarian", "momentum"]:
                pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, d)
                results.append((thresh, f"{hl}_{d[0]}", pnls))
    return results


@reg("N11_multifractal_vol", "Multifractal: high ratio = turbulent regime → expect big move, follow direction")
def exp_multifractal(df):
    df["mf_dir"] = df["multifractal_ratio_z"] * np.sign(df["returns"])
    df["sig"] = zscore(df, "mf_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N12_vw_momentum", "Volume-weighted momentum: large-trade-weighted price direction")
def exp_vw_mom(df):
    df["sig"] = zscore(df, "vw_momentum")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            for d in ["contrarian", "momentum"]:
                pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, d)
                results.append((thresh, f"{hl}_{d[0]}", pnls))
    return results


@reg("N13_surprise_contrarian", "Surprise index: unusual activity bars tend to revert")
def exp_surprise(df):
    # High surprise + direction → fade it
    df["surprise_dir"] = df["surprise_raw_z"] * np.sign(df["returns"])
    df["sig"] = zscore(df, "surprise_dir")
    results = []
    for thresh in [1.0, 1.5, 2.0]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "contrarian")
            results.append((thresh, hl, pnls))
    return results


@reg("N14_vol_speed_informed", "Volume clock speed: fast volume = informed activity → follow direction")
def exp_vol_speed(df):
    df["speed_dir"] = df["vol_speed_z"] * np.sign(df["vol_imbalance"])
    df["sig"] = zscore(df, "speed_dir")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


@reg("N15_composite_informed", "Composite informed flow: VPIN + entropy + persistence + aggression")
def exp_composite_informed(df):
    # Combine multiple informed-flow indicators
    cols = ["vpin", "sign_ac1", "aggressive_ratio", "avg_run_length"]
    df["sig"] = rank_composite(df, cols)
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            for d in ["contrarian", "momentum"]:
                pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, d)
                results.append((thresh, f"{hl}_{d[0]}", pnls))
    return results


@reg("N16_time_entropy_regime", "Time entropy: low = algorithmic (regular), high = organic → different edges")
def exp_time_entropy(df):
    # Low time entropy + imbalance = algo-driven flow → follow
    df["algo_flow"] = -df["time_entropy_z"] * np.sign(df["vol_imbalance"])
    df["sig"] = zscore(df, "algo_flow")
    results = []
    for thresh in [1.0, 1.5]:
        for hb, hl in [(12, "1h"), (24, "2h"), (48, "4h")]:
            pnls = backtest(df, "sig", thresh, hb, ROUND_TRIP_FEE_BPS, "momentum")
            results.append((thresh, hl, pnls))
    return results


# ---------------------------------------------------------------------------
# Runner (same structure as experiments.py)
# ---------------------------------------------------------------------------

def run_experiments(symbol, start_date, end_date, label):
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    print(f"\n{'='*70}")
    print(f"  {symbol} — {label} ({days} days: {start_date} → {end_date})")
    print(f"{'='*70}")

    print(f"  Loading & computing features...", flush=True)
    df = load_features(symbol, start_date, end_date)
    if df.empty:
        print(f"  ❌ No data!")
        return []

    print(f"  {len(df):,} bars, adding derived features...", flush=True)
    df = add_derived(df)

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
            if avg > best_avg:
                best_avg = avg
                best_cfg = (thresh, hl, len(pnls), avg, pnls.sum(), (pnls > 0).mean())

        if best_cfg:
            thresh, hl, nt, avg, total, wr = best_cfg
            marker = "✅" if avg > 0 and nt >= 10 else "  "
            print(f"    {marker} Best: thresh={thresh}, hold={hl}, "
                  f"trades={nt}, avg={avg:+.2f} bps, total={total:+.1f}, WR={wr:.0%}")

            if avg > 0 and nt >= 10:
                winners.append({
                    "experiment": exp_name, "symbol": symbol, "period": label,
                    "threshold": thresh, "holding": hl, "n_trades": nt,
                    "avg_pnl_bps": avg, "total_pnl_bps": total, "win_rate": wr,
                })
        else:
            print(f"    — No viable config")

    return winners


def main():
    t_start = time.time()
    print("=" * 70)
    print("  NOVEL EXPERIMENTS: Academic Microstructure Research Applied to Crypto")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Fees: {ROUND_TRIP_FEE_BPS} bps RT (Bybit VIP0)")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print("=" * 70)

    all_winners = []

    # Phase 1: 7-day screen
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: 7-DAY SCREENING")
    print(f"{'#'*70}")

    for symbol in SYMBOLS:
        winners = run_experiments(symbol, *PERIOD_7D, "7d")
        all_winners.extend(winners)

    print(f"\n\n{'='*70}")
    print(f"  PHASE 1 RESULTS: 7-Day Winners")
    print(f"{'='*70}")

    if not all_winners:
        print("  ❌ No winners!")
        return

    print(f"  {'Experiment':35s} {'Symbol':>10s} {'Thresh':>7s} {'Hold':>8s} "
          f"{'Trades':>7s} {'Avg':>8s} {'Total':>9s} {'WR':>5s}")
    print(f"  {'-'*95}")
    for w in sorted(all_winners, key=lambda x: -x["avg_pnl_bps"]):
        print(f"  {w['experiment']:35s} {w['symbol']:>10s} {w['threshold']:>7.1f} "
              f"{w['holding']:>8s} {w['n_trades']:>7d} {w['avg_pnl_bps']:>+8.2f} "
              f"{w['total_pnl_bps']:>+9.1f} {w['win_rate']:>5.0%}")

    # Phase 2: 30-day validation
    winning_experiments = set(w["experiment"] for w in all_winners)
    print(f"\n\n{'#'*70}")
    print(f"  PHASE 2: 30-DAY VALIDATION ({len(winning_experiments)} experiments)")
    print(f"{'#'*70}")

    validated = []
    for symbol in SYMBOLS:
        w30 = run_experiments(symbol, *PERIOD_30D, "30d")
        for w in w30:
            if w["experiment"] in winning_experiments:
                validated.append(w)

    print(f"\n\n{'='*70}")
    print(f"  FINAL: 30-Day Validated Winners")
    print(f"{'='*70}")

    if not validated:
        print("  ❌ No experiments survived 30-day validation!")
    else:
        print(f"  {'Experiment':35s} {'Symbol':>10s} {'Thresh':>7s} {'Hold':>8s} "
              f"{'Trades':>7s} {'Avg':>8s} {'Total':>9s} {'WR':>5s}")
        print(f"  {'-'*95}")
        for w in sorted(validated, key=lambda x: -x["avg_pnl_bps"]):
            print(f"  {w['experiment']:35s} {w['symbol']:>10s} {w['threshold']:>7.1f} "
                  f"{w['holding']:>8s} {w['n_trades']:>7d} {w['avg_pnl_bps']:>+8.2f} "
                  f"{w['total_pnl_bps']:>+9.1f} {w['win_rate']:>5.0%}")

    elapsed = time.time() - t_start
    print(f"\n✅ All novel experiments complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
