#!/usr/bin/env python3
"""
Regime Detection Research — Market-level trade/no-trade filter.

Instead of per-trade features, detect the MARKET REGIME and decide
whether the cross-exchange mean-reversion strategy should be active.

Approach:
1. Use BTC as the market regime proxy (drives all crypto)
2. Classify regimes: low-vol, normal, high-vol, trending, mean-reverting
3. Map each regime to strategy performance
4. Build a real-time regime signal that can gate trading

Key insight from trade filter research:
- Per-trade ML failed (AUC ~0.53, no better than coin flip)
- The strategy is a volatility harvester — works in high-vol, bleeds in low-vol
- We need a REGIME switch, not a per-trade filter
"""

import sys, time, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

DATALAKE = Path(__file__).resolve().parent.parent / "datalake"


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# =============================================================================
# 1. Load BTC data for regime classification (use Bybit — longest history)
# =============================================================================

def load_btc_regime_data():
    """Load BTC 5m bars from Bybit for regime classification."""
    import glob

    pattern = str(DATALAKE / "bybit" / "BTCUSDT" / "[0-9]*_kline_1m.csv")
    files = sorted(glob.glob(pattern))
    log(f"Loading BTC klines: {len(files)} daily files")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["startTime", "open", "high", "low", "close", "volume", "turnover"])
            if len(df) > 0:
                dfs.append(df)
        except:
            continue

    raw = pd.concat(dfs, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["startTime"], unit="ms", utc=True)
    raw = raw.sort_values("ts").drop_duplicates(subset=["ts"]).set_index("ts")

    # Resample to 5m
    bars = raw.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "turnover": "sum",
    }).dropna(subset=["close"])

    log(f"BTC 5m bars: {len(bars)} rows, {bars.index.min()} to {bars.index.max()}")
    return bars


# =============================================================================
# 2. Compute regime features
# =============================================================================

def compute_regime_features(bars):
    """
    Compute regime classification features from BTC 5m bars.
    All features are causal (use only past data).
    """
    close = bars["close"]
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=bars.index)

    # --- Realized volatility at multiple scales ---
    for window, label in [(12, "1h"), (48, "4h"), (144, "12h"), (288, "24h"),
                           (288*3, "3d"), (288*7, "7d"), (288*30, "30d")]:
        feat[f"rvol_{label}"] = log_ret.rolling(window).std() * np.sqrt(288 * 365) * 100  # annualized %

    # Vol regime ratios
    feat["rvol_1h_24h"] = feat["rvol_1h"] / feat["rvol_24h"].replace(0, np.nan)
    feat["rvol_4h_7d"] = feat["rvol_4h"] / feat["rvol_7d"].replace(0, np.nan)
    feat["rvol_24h_30d"] = feat["rvol_24h"] / feat["rvol_30d"].replace(0, np.nan)

    # --- Trend strength ---
    for window, label in [(12, "1h"), (48, "4h"), (288, "24h"), (288*3, "3d"), (288*7, "7d")]:
        feat[f"ret_{label}"] = close.pct_change(window) * 100  # percent
        feat[f"abs_ret_{label}"] = feat[f"ret_{label}"].abs()

    # ADX-style trend: abs(directional move) / total volatility
    feat["trend_ratio_24h"] = feat["abs_ret_24h"] / (feat["rvol_24h"] / np.sqrt(365) * 100).replace(0, np.nan)

    # --- Mean reversion tendency ---
    # Rolling autocorrelation of 5m returns (negative = mean-reverting, positive = trending)
    # Use vectorized approach: corr(ret_t, ret_{t-1}) over rolling window
    ret_lag = ret.shift(1)
    feat["autocorr_1h"] = ret.rolling(12).corr(ret_lag)
    feat["autocorr_6h"] = ret.rolling(72).corr(ret_lag)
    feat["autocorr_24h"] = ret.rolling(288).corr(ret_lag)

    # --- Range / consolidation ---
    # ATR-based: (high-low)/close averaged
    tr = (bars["high"] - bars["low"]) / close * 10000  # bps
    feat["atr_1h"] = tr.rolling(12).mean()
    feat["atr_24h"] = tr.rolling(288).mean()
    feat["atr_ratio"] = feat["atr_1h"] / feat["atr_24h"].replace(0, np.nan)

    # --- Volume regime ---
    vol = bars["turnover"]
    feat["vol_ma_1h"] = vol.rolling(12).mean()
    feat["vol_ma_24h"] = vol.rolling(288).mean()
    feat["vol_ma_7d"] = vol.rolling(288 * 7).mean()
    feat["vol_ratio_1h_24h"] = feat["vol_ma_1h"] / feat["vol_ma_24h"].replace(0, np.nan)
    feat["vol_ratio_24h_7d"] = feat["vol_ma_24h"] / feat["vol_ma_7d"].replace(0, np.nan)

    # --- Funding rate from Bybit (if available) ---
    import glob
    fr_files = sorted(glob.glob(str(DATALAKE / "bybit" / "BTCUSDT" / "*_funding_rate.csv")))
    if fr_files:
        fr_dfs = []
        for f in fr_files:
            try:
                df = pd.read_csv(f, usecols=["timestamp", "fundingRate"])
                fr_dfs.append(df)
            except:
                continue
        if fr_dfs:
            fr = pd.concat(fr_dfs, ignore_index=True)
            fr["ts"] = pd.to_datetime(fr["timestamp"], utc=True)
            fr = fr.sort_values("ts").drop_duplicates(subset=["ts"]).set_index("ts")
            fr["fundingRate"] = pd.to_numeric(fr["fundingRate"], errors="coerce")
            fr_5m = fr["fundingRate"].reindex(feat.index, method="ffill")
            feat["funding_rate"] = fr_5m
            feat["funding_abs"] = fr_5m.abs()
            feat["funding_ma_3d"] = feat["funding_abs"].rolling(288 * 3).mean()

    # --- OI regime ---
    oi_files = sorted(glob.glob(str(DATALAKE / "bybit" / "BTCUSDT" / "*_open_interest_5min.csv")))
    if oi_files:
        oi_dfs = []
        for f in oi_files:
            try:
                df = pd.read_csv(f, usecols=["timestamp", "openInterest"])
                oi_dfs.append(df)
            except:
                continue
        if oi_dfs:
            oi = pd.concat(oi_dfs, ignore_index=True)
            oi["ts"] = pd.to_datetime(oi["timestamp"], utc=True)
            oi = oi.sort_values("ts").drop_duplicates(subset=["ts"]).set_index("ts")
            oi["openInterest"] = pd.to_numeric(oi["openInterest"], errors="coerce")
            oi_5m = oi["openInterest"].reindex(feat.index, method="ffill")
            feat["oi"] = oi_5m
            feat["oi_change_24h"] = oi_5m.pct_change(288) * 100
            feat["oi_change_7d"] = oi_5m.pct_change(288 * 7) * 100

    # --- Time features ---
    feat["hour"] = feat.index.hour
    feat["dow"] = feat.index.dayofweek

    log(f"Computed {len(feat.columns)} regime features, {len(feat)} rows")
    return feat


# =============================================================================
# 3. Classify regimes using multiple methods
# =============================================================================

def classify_regimes(feat):
    """
    Classify market regimes using multiple approaches:
    A) Simple threshold-based (vol quantiles)
    B) Hidden Markov Model
    C) K-means clustering
    """
    results = {}

    # --- A) Threshold-based regime ---
    log("\n  A) Threshold-based regime (BTC 24h realized vol quantiles)")
    rvol = feat["rvol_24h"].dropna()

    # Use expanding quantiles (causal — only past data), aligned to feat index
    expanding_q25 = rvol.expanding(min_periods=288*30).quantile(0.25).reindex(feat.index)
    expanding_q75 = rvol.expanding(min_periods=288*30).quantile(0.75).reindex(feat.index)

    regime_vol = pd.Series("normal", index=feat.index)
    regime_vol[feat["rvol_24h"] < expanding_q25] = "low_vol"
    regime_vol[feat["rvol_24h"] > expanding_q75] = "high_vol"
    regime_vol[(feat["rvol_24h"] > expanding_q75) & (feat["rvol_1h_24h"] > 1.5)] = "vol_spike"

    results["vol_regime"] = regime_vol
    log(f"  Vol regime distribution:")
    for r in ["low_vol", "normal", "high_vol", "vol_spike"]:
        n = (regime_vol == r).sum()
        pct = n / len(regime_vol) * 100
        log(f"    {r:12s}: {n:8d} bars ({pct:.1f}%)")

    # --- B) Trend vs Mean-Reversion regime ---
    log("\n  B) Trend/MR regime (autocorrelation-based)")
    ac = feat["autocorr_24h"].dropna()
    regime_mr = pd.Series("neutral", index=feat.index)
    regime_mr[feat["autocorr_24h"] < -0.05] = "mean_revert"
    regime_mr[feat["autocorr_24h"] > 0.05] = "trending"
    results["mr_regime"] = regime_mr

    for r in ["mean_revert", "neutral", "trending"]:
        n = (regime_mr == r).sum()
        pct = n / len(regime_mr) * 100
        log(f"    {r:12s}: {n:8d} bars ({pct:.1f}%)")

    # --- C) Combined regime ---
    log("\n  C) Combined regime (vol × trend)")
    regime_combined = pd.Series("neutral", index=feat.index)

    # High vol + mean reverting = BEST for our strategy
    regime_combined[(regime_vol == "high_vol") & (regime_mr == "mean_revert")] = "ideal"
    regime_combined[(regime_vol == "high_vol") & (regime_mr != "mean_revert")] = "high_vol_trend"
    regime_combined[(regime_vol == "vol_spike")] = "vol_spike"
    regime_combined[(regime_vol == "low_vol")] = "quiet"
    regime_combined[(regime_vol == "normal") & (regime_mr == "mean_revert")] = "normal_mr"
    regime_combined[(regime_vol == "normal") & (regime_mr == "trending")] = "normal_trend"

    results["combined_regime"] = regime_combined

    for r in sorted(regime_combined.unique()):
        n = (regime_combined == r).sum()
        pct = n / len(regime_combined) * 100
        log(f"    {r:20s}: {n:8d} bars ({pct:.1f}%)")

    # --- D) HMM (if sklearn available) ---
    try:
        from sklearn.mixture import GaussianMixture

        log("\n  D) Gaussian Mixture Model (unsupervised)")
        # Use vol + trend features
        hmm_feats = feat[["rvol_24h", "autocorr_24h", "atr_ratio", "vol_ratio_1h_24h"]].dropna()
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(hmm_feats)

        # Try 3 and 4 states
        for n_states in [3, 4]:
            gmm = GaussianMixture(n_components=n_states, covariance_type="full",
                                   n_init=5, random_state=42)
            labels = gmm.fit_predict(X)
            regime_gmm = pd.Series("unknown", index=feat.index)
            regime_gmm.loc[hmm_feats.index] = [f"state_{l}" for l in labels]
            results[f"gmm_{n_states}"] = regime_gmm

            # Characterize each state
            log(f"  GMM {n_states} states:")
            for s in range(n_states):
                mask = labels == s
                sub = hmm_feats[mask]
                n = mask.sum()
                pct = n / len(labels) * 100
                log(f"    state_{s}: {n:6d} ({pct:.1f}%)  "
                    f"rvol={sub['rvol_24h'].mean():.1f}  "
                    f"ac={sub['autocorr_24h'].mean():.3f}  "
                    f"atr_r={sub['atr_ratio'].mean():.2f}  "
                    f"vol_r={sub['vol_ratio_1h_24h'].mean():.2f}")
    except Exception as e:
        log(f"  GMM failed: {e}")

    return results


# =============================================================================
# 4. Map regimes to strategy performance
# =============================================================================

def map_regime_to_performance(feat, regimes, trades_file="production_best_trades.csv"):
    """
    For each trade, look up the BTC regime at entry time.
    Compute performance stats per regime.
    """
    trades = pd.read_csv(trades_file)
    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_dt"] = pd.to_datetime(trades["exit_time"], utc=True)
    log(f"\n  Mapping {len(trades)} trades to BTC regimes...")

    BASE = 10000

    for regime_name, regime_series in regimes.items():
        log(f"\n  {'=' * 80}")
        log(f"  Regime: {regime_name}")
        log(f"  {'=' * 80}")

        # Look up regime at each trade entry
        trade_regimes = []
        for _, t in trades.iterrows():
            entry = t["entry_dt"]
            # Find nearest 5m bar
            idx = regime_series.index.searchsorted(entry)
            if idx >= len(regime_series):
                idx = len(regime_series) - 1
            trade_regimes.append(regime_series.iloc[idx])

        trades[f"regime_{regime_name}"] = trade_regimes

        # Stats per regime
        hdr = f"  {'Regime':>20s} {'Trades':>7s} {'WR':>5s} {'Avg bps':>8s} {'Total bps':>10s} {'Est USD':>10s} {'Avg Hold':>9s}"
        log(hdr)
        log("  " + "-" * 80)

        regime_stats = {}
        for r in sorted(trades[f"regime_{regime_name}"].unique()):
            sub = trades[trades[f"regime_{regime_name}"] == r]
            n = len(sub)
            wr = (sub["net_bps"] > 0).mean()
            avg = sub["net_bps"].mean()
            total = sub["net_bps"].sum()
            usd = (sub["net_bps"] / 10000 * sub["position_size"] * BASE).sum()
            avg_hold = sub["hold_bars"].mean() * 5

            regime_stats[r] = {"n": n, "wr": wr, "avg": avg, "total": total, "usd": usd}
            log(f"  {r:>20s} {n:>7d} {wr:>4.0%} {avg:>+8.0f} {total:>+10.0f} {usd:>+10.0f} {avg_hold:>8.0f}m")

        # What if we only trade in favorable regimes?
        log(f"\n  Filter analysis:")
        all_total = trades["net_bps"].sum()
        all_usd = (trades["net_bps"] / 10000 * trades["position_size"] * BASE).sum()

        # For each regime, compute: if we SKIP this regime, what happens?
        for r, stats in sorted(regime_stats.items(), key=lambda x: x[1]["avg"]):
            remaining = trades[trades[f"regime_{regime_name}"] != r]
            rem_n = len(remaining)
            rem_total = remaining["net_bps"].sum()
            rem_usd = (remaining["net_bps"] / 10000 * remaining["position_size"] * BASE).sum()
            rem_wr = (remaining["net_bps"] > 0).mean() if rem_n > 0 else 0
            log(f"    Skip {r:>15s}: {rem_n:>4d} trades, WR={rem_wr:.0%}, "
                f"total={rem_total:+.0f} bps, USD={rem_usd:+,.0f}")

    return trades


# =============================================================================
# 5. Walk-forward regime filter backtest
# =============================================================================

def walk_forward_regime_backtest(feat, regimes, trades):
    """
    Walk-forward test: at each month, use past data to decide which
    regimes to trade in, then test on current month.
    """
    log(f"\n  {'=' * 80}")
    log(f"  Walk-Forward Regime Filter Backtest")
    log(f"  {'=' * 80}")

    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["month"] = trades["entry_dt"].dt.to_period("M").astype(str)
    months = sorted(trades["month"].unique())
    BASE = 10000

    for regime_name in ["vol_regime", "combined_regime"]:
        if regime_name not in regimes:
            continue

        log(f"\n  --- {regime_name} ---")

        # Ensure regime column exists
        regime_col = f"regime_{regime_name}"
        if regime_col not in trades.columns:
            continue

        cumulative_base = 0
        cumulative_filtered = 0

        hdr = (f"  {'Month':>10s} {'Base N':>7s} {'Base$':>10s} "
               f"{'Filt N':>7s} {'Filt$':>10s} {'Regime':>20s} {'Skip%':>6s}")
        log(hdr)
        log("  " + "-" * 85)

        for m_idx, test_month in enumerate(months):
            # Get all past trades for training
            past_months = set(months[:m_idx])
            if len(past_months) < 1:
                # Not enough history — take all trades
                good_regimes = set(trades[regime_col].unique())
            else:
                past = trades[trades["month"].isin(past_months)]
                # For each regime, compute past performance
                good_regimes = set()
                for r in past[regime_col].unique():
                    sub = past[past[regime_col] == r]
                    if len(sub) >= 3 and sub["net_bps"].mean() > 0:
                        good_regimes.add(r)
                # If all regimes are bad, take all (fallback)
                if not good_regimes:
                    good_regimes = set(trades[regime_col].unique())

            # Test month trades
            test = trades[trades["month"] == test_month]
            filtered = test[test[regime_col].isin(good_regimes)]

            base_usd = (test["net_bps"] / 10000 * test["position_size"] * BASE).sum()
            filt_usd = (filtered["net_bps"] / 10000 * filtered["position_size"] * BASE).sum()
            cumulative_base += base_usd
            cumulative_filtered += filt_usd

            skip_pct = 1 - len(filtered) / max(len(test), 1)
            skipped_regimes = set(test[regime_col].unique()) - good_regimes

            log(f"  {test_month:>10s} {len(test):>7d} {base_usd:>+10,.0f} "
                f"{len(filtered):>7d} {filt_usd:>+10,.0f} "
                f"{'skip:' + ','.join(sorted(skipped_regimes)) if skipped_regimes else 'all':>20s} "
                f"{skip_pct:>5.0%}")

        log(f"\n  Cumulative: Base ${cumulative_base:+,.0f} | Filtered ${cumulative_filtered:+,.0f}")
        improvement = (cumulative_filtered - cumulative_base) / abs(cumulative_base) * 100 if cumulative_base != 0 else 0
        log(f"  Improvement: {improvement:+.1f}%")


# =============================================================================
# 6. Visualize regimes over time
# =============================================================================

def plot_regimes(feat, regimes, trades):
    """Plot BTC price with regime coloring and trade markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], utc=True)
    dollar = FuncFormatter(lambda x, p: f"${x:,.0f}")

    # Only plot the strategy period (2025-01 onwards for overlap with trades)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = feat.index.max()
    mask = (feat.index >= start) & (feat.index <= end)
    feat_plot = feat[mask]

    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True)
    fig.suptitle("BTC Regime Detection — Claude-Exp-1 Strategy Overlay\n"
                 "Regime = market-level trade/no-trade signal",
                 fontsize=14, fontweight="bold")

    # 1) BTC price with vol regime coloring
    ax = axes[0]
    close = feat_plot.get("rvol_24h", pd.Series(index=feat_plot.index))
    # Get BTC close from bars
    import glob
    pattern = str(DATALAKE / "bybit" / "BTCUSDT" / "[0-9]*_kline_1m.csv")
    # Just use the features index to recreate price from returns
    # Actually let's reload BTC price quickly
    btc_bars = load_btc_regime_data()
    btc_close = btc_bars["close"].reindex(feat_plot.index, method="ffill")

    ax.plot(feat_plot.index, btc_close, color="gray", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("BTC Price")
    ax.yaxis.set_major_formatter(dollar)
    ax.set_title("BTC Price + Volatility Regime")

    # Color background by vol regime
    if "vol_regime" in regimes:
        vol_r = regimes["vol_regime"].reindex(feat_plot.index)
        regime_colors = {"low_vol": "#E8F5E9", "normal": "#FFF9C4",
                         "high_vol": "#FFECB3", "vol_spike": "#FFCDD2"}
        prev_regime = None
        start_idx = feat_plot.index[0]
        for i, (ts, r) in enumerate(vol_r.items()):
            if r != prev_regime and prev_regime is not None:
                ax.axvspan(start_idx, ts, alpha=0.3,
                          color=regime_colors.get(prev_regime, "white"), label=None)
                start_idx = ts
            prev_regime = r
        # Legend
        for r, c in regime_colors.items():
            ax.axvspan(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"),
                      alpha=0.3, color=c, label=r)
        ax.legend(loc="upper left", fontsize=8)

    # Add trade markers
    wins = trades[trades["net_bps"] > 0]
    losses = trades[trades["net_bps"] <= 0]
    win_ts = pd.to_datetime(wins["entry_time"], utc=True)
    loss_ts = pd.to_datetime(losses["entry_time"], utc=True)

    # Map trade times to BTC price
    for ts_series, color, label, marker in [
        (win_ts, "green", "Win", "^"),
        (loss_ts, "red", "Loss", "v"),
    ]:
        prices = btc_close.reindex(ts_series, method="nearest")
        ax.scatter(ts_series, prices, c=color, marker=marker, s=15, alpha=0.6, label=label, zorder=5)
    ax.legend(loc="upper left", fontsize=8)

    # 2) Realized volatility
    ax = axes[1]
    ax.plot(feat_plot.index, feat_plot["rvol_24h"], label="24h rvol", color="#FF9800", linewidth=0.8)
    ax.plot(feat_plot.index, feat_plot["rvol_7d"], label="7d rvol", color="#2196F3", linewidth=0.8)
    ax.set_ylabel("Annualized Vol %")
    ax.set_title("Realized Volatility")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) Autocorrelation (mean-reversion tendency)
    ax = axes[2]
    ac = feat_plot["autocorr_24h"]
    ax.plot(feat_plot.index, ac, color="#9C27B0", linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=-0.05, color="green", linestyle="--", alpha=0.3, label="MR threshold")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.3, label="Trend threshold")
    ax.fill_between(feat_plot.index, ac, 0, where=ac < -0.05, alpha=0.2, color="green")
    ax.fill_between(feat_plot.index, ac, 0, where=ac > 0.05, alpha=0.2, color="red")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Return Autocorrelation (24h) — Green=Mean-Reverting, Red=Trending")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Cumulative PnL with regime overlay
    ax = axes[3]
    trades_sorted = trades.sort_values("entry_dt")
    cum_pnl = (trades_sorted["net_bps"] / 10000 * trades_sorted["position_size"] * 10000).cumsum()
    ax.plot(trades_sorted["entry_dt"].values, cum_pnl.values, color="#4CAF50", linewidth=1.5)
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Strategy Cumulative PnL with Trade Timestamps")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(dollar)

    plt.tight_layout()
    plt.savefig("regime_analysis.png", dpi=150, bbox_inches="tight")
    log("\n  Saved regime_analysis.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    log("=" * 100)
    log("  REGIME DETECTION RESEARCH — Claude-Exp-1")
    log("  Market-level trade/no-trade filter using BTC as proxy")
    log("=" * 100)

    # Phase 1: Load BTC data
    log("\n  Phase 1: Loading BTC data...")
    bars = load_btc_regime_data()

    # Phase 2: Compute features
    log("\n  Phase 2: Computing regime features...")
    feat = compute_regime_features(bars)

    # Phase 3: Classify regimes
    log("\n  Phase 3: Classifying regimes...")
    regimes = classify_regimes(feat)

    # Phase 4: Map to strategy performance
    log("\n  Phase 4: Mapping regimes to strategy performance...")
    trades = map_regime_to_performance(feat, regimes)

    # Phase 5: Walk-forward backtest
    log("\n  Phase 5: Walk-forward regime filter backtest...")
    walk_forward_regime_backtest(feat, regimes, trades)

    # Phase 6: Visualize
    log("\n  Phase 6: Plotting...")
    plot_regimes(feat, regimes, trades)

    # Save enriched trades
    trades.to_csv("trades_with_regimes.csv", index=False)
    log(f"\n  Saved trades_with_regimes.csv")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
