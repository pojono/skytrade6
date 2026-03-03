#!/usr/bin/env python3
"""
OI + FR Squeeze Strategy Research

Loads 6-channel REST API data for multiple coins, engineers features,
computes forward return labels, and runs signal exploration.

Hypothesis: when positioning is extreme (high OI + extreme FR + tilted LS ratio),
the market is vulnerable to a squeeze in the opposite direction.

Fee constraint: taker 10bps/leg = 20bps RT, maker 4bps/leg = 8bps RT.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "squeeze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT",  # 8h FR
    "TRUMPUSDT", "FARTCOINUSDT", "VIRTUALUSDT",  # 4h FR
    "POWERUSDT",  # 1h FR
]

# Forward return horizons (minutes)
FWD_HORIZONS_M = [5, 15, 30, 60, 120, 240]

# Fee assumptions (bps)
TAKER_RT_BPS = 20.0
MAKER_RT_BPS = 8.0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_symbol_data(symbol: str) -> dict[str, pd.DataFrame]:
    """Load all CSV data for a symbol, return dict of DataFrames."""
    sym_dir = DATA_DIR / symbol
    if not sym_dir.exists():
        print(f"  SKIP {symbol}: no data directory")
        return {}

    dfs = {}

    # --- Kline 1m ---
    kline_files = sorted(f for f in sym_dir.glob("*_kline_1m.csv")
                         if "mark_price" not in f.name and "premium_index" not in f.name)
    if kline_files:
        frames = []
        for f in kline_files:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        if frames:
            kline = pd.concat(frames, ignore_index=True)
            kline["ts"] = pd.to_datetime(kline["startTime"].astype(int), unit="ms", utc=True)
            kline = kline.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                if col in kline.columns:
                    kline[col] = kline[col].astype(float)
            dfs["kline"] = kline

    # --- Open Interest 5min ---
    oi_files = sorted(sym_dir.glob("*_open_interest_5min.csv"))
    if oi_files:
        frames = []
        for f in oi_files:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        if frames:
            oi = pd.concat(frames, ignore_index=True)
            oi["ts"] = pd.to_datetime(oi["timestamp"].astype(int), unit="ms", utc=True)
            oi["openInterest"] = oi["openInterest"].astype(float)
            oi = oi.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            dfs["oi"] = oi

    # --- Funding Rate ---
    fr_files = sorted(sym_dir.glob("*_funding_rate.csv"))
    if fr_files:
        frames = []
        for f in fr_files:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        if frames:
            fr = pd.concat(frames, ignore_index=True)
            fr["ts"] = pd.to_datetime(fr["timestamp"].astype(int), unit="ms", utc=True)
            fr["fundingRate"] = fr["fundingRate"].astype(float)
            fr = fr.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            dfs["fr"] = fr

    # --- Long/Short Ratio 5min ---
    ls_files = sorted(sym_dir.glob("*_long_short_ratio_5min.csv"))
    if ls_files:
        frames = []
        for f in ls_files:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        if frames:
            ls = pd.concat(frames, ignore_index=True)
            ls["ts"] = pd.to_datetime(ls["timestamp"].astype(int), unit="ms", utc=True)
            ls["buyRatio"] = ls["buyRatio"].astype(float)
            ls["sellRatio"] = ls["sellRatio"].astype(float)
            ls = ls.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            dfs["ls"] = ls

    # --- Premium Index 1m ---
    pi_files = sorted(sym_dir.glob("*_premium_index_kline_1m.csv"))
    if pi_files:
        frames = []
        for f in pi_files:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        if frames:
            pi = pd.concat(frames, ignore_index=True)
            pi["ts"] = pd.to_datetime(pi["startTime"].astype(int), unit="ms", utc=True)
            # Rename to avoid conflict with kline columns
            pi = pi.rename(columns={"open": "pi_open", "high": "pi_high",
                                     "low": "pi_low", "close": "pi_close"})
            for col in ["pi_open", "pi_high", "pi_low", "pi_close"]:
                pi[col] = pi[col].astype(float)
            pi = pi.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
            dfs["premium"] = pi

    return dfs


def build_5min_df(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all data onto a 5-minute grid aligned to kline data."""
    kline = data.get("kline")
    if kline is None or len(kline) == 0:
        return pd.DataFrame()

    # Resample kline to 5min OHLCV
    kline = kline.set_index("ts")
    k5 = kline.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "turnover": "sum",
    }).dropna(subset=["close"])
    k5 = k5.reset_index()

    # Merge OI (already 5min)
    if "oi" in data:
        oi = data["oi"][["ts", "openInterest"]].copy()
        k5 = pd.merge_asof(k5.sort_values("ts"), oi.sort_values("ts"),
                           on="ts", direction="backward", tolerance=pd.Timedelta("5min"))

    # Merge LS ratio (already 5min)
    if "ls" in data:
        ls = data["ls"][["ts", "buyRatio", "sellRatio"]].copy()
        k5 = pd.merge_asof(k5.sort_values("ts"), ls.sort_values("ts"),
                           on="ts", direction="backward", tolerance=pd.Timedelta("5min"))

    # Merge FR (forward-fill, sparse)
    if "fr" in data:
        fr = data["fr"][["ts", "fundingRate"]].copy()
        k5 = pd.merge_asof(k5.sort_values("ts"), fr.sort_values("ts"),
                           on="ts", direction="backward", tolerance=pd.Timedelta("12h"))

    # Merge premium index (resample to 5min close)
    if "premium" in data:
        pi = data["premium"].set_index("ts")[["pi_close"]].resample("5min").last().dropna()
        pi = pi.reset_index().rename(columns={"pi_close": "premium_index"})
        k5 = pd.merge_asof(k5.sort_values("ts"), pi.sort_values("ts"),
                           on="ts", direction="backward", tolerance=pd.Timedelta("5min"))

    k5 = k5.sort_values("ts").reset_index(drop=True)
    return k5


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all features to the 5-min DataFrame. All features are causal (no lookahead)."""
    df = df.copy()

    # --- Price features ---
    df["ret_5m"] = df["close"].pct_change() * 10000  # bps
    df["ret_15m"] = df["close"].pct_change(3) * 10000
    df["ret_1h"] = df["close"].pct_change(12) * 10000
    df["ret_4h"] = df["close"].pct_change(48) * 10000
    df["ret_24h"] = df["close"].pct_change(288) * 10000

    # Volatility (rolling std of 5m returns)
    df["vol_1h"] = df["ret_5m"].rolling(12).std()
    df["vol_4h"] = df["ret_5m"].rolling(48).std()
    df["vol_24h"] = df["ret_5m"].rolling(288).std()

    # --- Volume features ---
    df["vol_ma_1h"] = df["turnover"].rolling(12).mean()
    df["vol_ma_4h"] = df["turnover"].rolling(48).mean()
    df["vol_ratio_1h"] = df["turnover"] / df["vol_ma_1h"].replace(0, np.nan)

    # --- OI features ---
    if "openInterest" in df.columns:
        df["oi"] = df["openInterest"]
        df["oi_pct_1h"] = df["oi"].pct_change(12) * 100  # % change over 1h
        df["oi_pct_4h"] = df["oi"].pct_change(48) * 100
        df["oi_pct_24h"] = df["oi"].pct_change(288) * 100
        df["oi_ma_24h"] = df["oi"].rolling(288).mean()
        df["oi_zscore"] = (df["oi"] - df["oi_ma_24h"]) / df["oi"].rolling(288).std().replace(0, np.nan)

        # OI-Price divergence: OI rising but price falling (or vice versa)
        df["oi_price_div_1h"] = df["oi_pct_1h"] - df["ret_1h"] / 100
        df["oi_price_div_4h"] = df["oi_pct_4h"] - df["ret_4h"] / 100

    # --- Funding Rate features ---
    if "fundingRate" in df.columns:
        df["fr_bps"] = df["fundingRate"] * 10000
        df["fr_abs_bps"] = df["fr_bps"].abs()
        # Rolling FR stats (last N settlements)
        # FR is sparse — forward-filled, so rolling on 5min grid gives us the current value repeated
        # Better: compute z-score vs recent history
        df["fr_ma_24h"] = df["fr_bps"].rolling(288, min_periods=12).mean()
        df["fr_std_24h"] = df["fr_bps"].rolling(288, min_periods=12).std().replace(0, np.nan)
        df["fr_zscore"] = (df["fr_bps"] - df["fr_ma_24h"]) / df["fr_std_24h"]
        # Extreme FR flag
        df["fr_extreme_pos"] = (df["fr_bps"] > 5).astype(int)  # longs paying > 5bps
        df["fr_extreme_neg"] = (df["fr_bps"] < -5).astype(int)  # shorts paying > 5bps

    # --- Long/Short Ratio features ---
    if "buyRatio" in df.columns:
        df["ls_ratio"] = df["buyRatio"] / df["sellRatio"].replace(0, np.nan)
        df["ls_ratio_ma_1h"] = df["ls_ratio"].rolling(12).mean()
        df["ls_ratio_ma_4h"] = df["ls_ratio"].rolling(48).mean()
        df["ls_tilt"] = df["ls_ratio"] - df["ls_ratio_ma_4h"]  # deviation from norm
        df["ls_zscore"] = ((df["ls_ratio"] - df["ls_ratio"].rolling(288).mean())
                           / df["ls_ratio"].rolling(288).std().replace(0, np.nan))

    # --- Premium Index features ---
    if "premium_index" in df.columns:
        df["pi_bps"] = df["premium_index"] * 10000
        df["pi_abs_bps"] = df["pi_bps"].abs()
        df["pi_ma_1h"] = df["pi_bps"].rolling(12).mean()
        df["pi_ma_4h"] = df["pi_bps"].rolling(48).mean()
        df["pi_zscore"] = ((df["pi_bps"] - df["pi_bps"].rolling(288).mean())
                           / df["pi_bps"].rolling(288).std().replace(0, np.nan))

    # --- Composite squeeze features ---
    # Squeeze score: high OI + extreme FR + tilted LS = vulnerable
    has_oi = "oi_zscore" in df.columns
    has_fr = "fr_zscore" in df.columns
    has_ls = "ls_zscore" in df.columns

    if has_oi and has_fr and has_ls:
        # Long squeeze score: OI high + FR positive (longs paying) + LS tilted long
        df["long_squeeze_score"] = (
            df["oi_zscore"].clip(0, 3) / 3 * 0.4 +
            df["fr_zscore"].clip(0, 3) / 3 * 0.4 +
            df["ls_zscore"].clip(0, 3) / 3 * 0.2
        )
        # Short squeeze score: OI high + FR negative (shorts paying) + LS tilted short
        df["short_squeeze_score"] = (
            df["oi_zscore"].clip(0, 3) / 3 * 0.4 +
            (-df["fr_zscore"]).clip(0, 3) / 3 * 0.4 +
            (-df["ls_zscore"]).clip(0, 3) / 3 * 0.2
        )
        # Combined (directional): positive = short squeeze (go long), negative = long squeeze (go short)
        df["squeeze_dir"] = df["short_squeeze_score"] - df["long_squeeze_score"]

    return df


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------


def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward return labels at multiple horizons."""
    df = df.copy()
    for h_min in FWD_HORIZONS_M:
        periods = h_min // 5  # convert minutes to 5-min bars
        df[f"fwd_ret_{h_min}m"] = (df["close"].shift(-periods) / df["close"] - 1) * 10000  # bps

    # Filter out rows with bad close prices
    df = df[df["close"] > 0].copy()

    # MFE / MAE within 1h and 4h windows
    for window_bars, label in [(12, "1h"), (48, "4h")]:
        mfe = np.full(len(df), np.nan)
        mae = np.full(len(df), np.nan)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        for i in range(len(df) - window_bars):
            entry = close[i]
            if entry <= 0 or np.isnan(entry):
                continue
            future_high = np.nanmax(high[i+1:i+1+window_bars])
            future_low = np.nanmin(low[i+1:i+1+window_bars])
            mfe[i] = (future_high / entry - 1) * 10000
            mae[i] = (future_low / entry - 1) * 10000

        df[f"mfe_{label}"] = mfe
        df[f"mae_{label}"] = mae

    return df


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------


def univariate_sort(df: pd.DataFrame, feature: str, target: str, n_bins: int = 5) -> pd.DataFrame:
    """Quintile sort of target by feature. Returns summary per bin."""
    valid = df[[feature, target]].dropna()
    if len(valid) < n_bins * 20:
        return pd.DataFrame()

    valid["bin"] = pd.qcut(valid[feature], n_bins, labels=False, duplicates="drop")
    summary = valid.groupby("bin").agg(
        n=(target, "count"),
        mean=(target, "mean"),
        median=(target, "median"),
        std=(target, "std"),
        wr=(target, lambda x: (x > 0).mean()),
        p25=(target, lambda x: x.quantile(0.25)),
        p75=(target, lambda x: x.quantile(0.75)),
        feat_mean=(feature, "mean"),
        feat_median=(feature, "median"),
    ).reset_index()
    summary["feature"] = feature
    summary["target"] = target
    return summary


def run_signal_scan(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Run univariate sort for all features vs all forward return horizons."""
    features = [c for c in df.columns if any(c.startswith(p) for p in [
        "oi_pct", "oi_zscore", "oi_price_div",
        "fr_bps", "fr_abs_bps", "fr_zscore", "fr_extreme",
        "ls_ratio", "ls_tilt", "ls_zscore",
        "pi_bps", "pi_abs_bps", "pi_zscore",
        "vol_ratio", "vol_1h", "vol_4h",
        "ret_5m", "ret_15m", "ret_1h", "ret_4h", "ret_24h",
        "long_squeeze", "short_squeeze", "squeeze_dir",
    ])]

    targets = [f"fwd_ret_{h}m" for h in FWD_HORIZONS_M]

    results = []
    for feat in features:
        for tgt in targets:
            r = univariate_sort(df, feat, tgt)
            if len(r) > 0:
                results.append(r)

    if not results:
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True)
    all_results["symbol"] = symbol
    return all_results


def compute_monotonicity(sort_df: pd.DataFrame) -> pd.DataFrame:
    """For each (feature, target), compute monotonicity score and spread."""
    rows = []
    for (feat, tgt), grp in sort_df.groupby(["feature", "target"]):
        grp = grp.sort_values("bin")
        means = grp["mean"].values
        n_bins = len(means)
        if n_bins < 3:
            continue

        # Spread: Q5 mean - Q1 mean (bps)
        spread = means[-1] - means[0]

        # Monotonicity: Spearman correlation of bin rank vs mean return
        from scipy.stats import spearmanr
        rho, pval = spearmanr(range(n_bins), means)

        # WR spread
        wrs = grp["wr"].values
        wr_spread = wrs[-1] - wrs[0]

        # Total N
        total_n = grp["n"].sum()

        rows.append({
            "feature": feat,
            "target": tgt,
            "spread_bps": spread,
            "wr_spread": wr_spread,
            "spearman_rho": rho,
            "spearman_pval": pval,
            "n_total": total_n,
            "q1_mean": means[0],
            "q5_mean": means[-1],
            "q1_wr": wrs[0],
            "q5_wr": wrs[-1],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()

    print("=" * 80)
    print("OI + FR SQUEEZE STRATEGY — SIGNAL EXPLORATION")
    print("=" * 80)

    all_features = []
    all_sorts = []
    symbol_stats = []

    for sym in SYMBOLS:
        print(f"\n{'─'*60}")
        print(f"Loading {sym}...")
        data = load_symbol_data(sym)
        if not data or "kline" not in data:
            print(f"  SKIP: no kline data")
            continue

        print(f"  Kline rows: {len(data['kline']):,}")
        for k, v in data.items():
            if k != "kline":
                print(f"  {k} rows: {len(v):,}")

        # Build 5-min DataFrame
        df = build_5min_df(data)
        print(f"  5min grid: {len(df):,} rows ({df['ts'].min()} to {df['ts'].max()})")

        # Detect FR interval from data
        if "fr" in data:
            fr = data["fr"]
            if len(fr) > 1:
                fr_diffs = fr["ts"].diff().dt.total_seconds().dropna()
                median_interval_h = fr_diffs.median() / 3600
                fr_per_day = 24 / median_interval_h if median_interval_h > 0 else 0
                print(f"  FR interval: {median_interval_h:.1f}h ({fr_per_day:.0f}x/day)")

        # Engineer features
        df = engineer_features(df)
        print(f"  Features: {len([c for c in df.columns if c not in ['ts','open','high','low','close','volume','turnover']])}")

        # Add forward returns
        df = add_forward_returns(df)
        df["symbol"] = sym
        all_features.append(df)

        # Run signal scan
        print(f"  Running signal scan...")
        sorts = run_signal_scan(df, sym)
        if len(sorts) > 0:
            all_sorts.append(sorts)
            print(f"  Sort results: {len(sorts):,} rows")

        # Quick stats
        valid_rows = df.dropna(subset=["fwd_ret_60m"])
        symbol_stats.append({
            "symbol": sym,
            "days": (df["ts"].max() - df["ts"].min()).days,
            "rows_5min": len(df),
            "valid_rows": len(valid_rows),
            "fr_interval_h": median_interval_h if "fr" in data and len(data["fr"]) > 1 else None,
        })

    # ===================================================================
    # Cross-symbol analysis
    # ===================================================================
    print(f"\n{'='*80}")
    print("CROSS-SYMBOL ANALYSIS")
    print("=" * 80)

    stats_df = pd.DataFrame(symbol_stats)
    print("\nSymbol summary:")
    print(stats_df.to_string(index=False))

    if not all_sorts:
        print("\nNo sort results. Exiting.")
        return

    # Combine all sort results
    all_sorts_df = pd.concat(all_sorts, ignore_index=True)

    # Compute monotonicity per symbol
    all_mono = []
    for sym in all_sorts_df["symbol"].unique():
        sym_sorts = all_sorts_df[all_sorts_df["symbol"] == sym]
        mono = compute_monotonicity(sym_sorts)
        mono["symbol"] = sym
        all_mono.append(mono)

    mono_df = pd.concat(all_mono, ignore_index=True)

    # ===================================================================
    # Report: best signals
    # ===================================================================
    print(f"\n{'='*80}")
    print("TOP SIGNALS BY |SPREAD| (per symbol)")
    print("=" * 80)

    for sym in mono_df["symbol"].unique():
        sym_mono = mono_df[mono_df["symbol"] == sym].copy()
        sym_mono["abs_spread"] = sym_mono["spread_bps"].abs()
        top = sym_mono.sort_values("abs_spread", ascending=False).head(15)
        print(f"\n--- {sym} ---")
        print(f"{'Feature':<25} {'Target':<15} {'Spread':>8} {'WR_spr':>7} {'rho':>6} {'pval':>8} {'Q1_wr':>6} {'Q5_wr':>6} {'N':>7}")
        for _, r in top.iterrows():
            sig = "*" if r["spearman_pval"] < 0.05 else " "
            print(f"{r['feature']:<25} {r['target']:<15} {r['spread_bps']:>+8.1f} {r['wr_spread']:>+7.3f} {r['spearman_rho']:>+6.3f} {r['spearman_pval']:>8.4f} {r['q1_wr']:>6.3f} {r['q5_wr']:>6.3f} {r['n_total']:>7.0f}{sig}")

    # ===================================================================
    # Report: signals that work across multiple coins
    # ===================================================================
    print(f"\n{'='*80}")
    print("ROBUST SIGNALS (consistent across coins)")
    print("=" * 80)

    # For each (feature, target), count how many symbols show significant monotonicity
    robust = mono_df[mono_df["spearman_pval"] < 0.05].copy()
    if len(robust) > 0:
        cross = robust.groupby(["feature", "target"]).agg(
            n_symbols=("symbol", "nunique"),
            mean_spread=("spread_bps", "mean"),
            mean_rho=("spearman_rho", "mean"),
            mean_wr_spread=("wr_spread", "mean"),
            symbols=("symbol", lambda x: ",".join(sorted(x.unique()))),
        ).reset_index()
        cross = cross.sort_values(["n_symbols", "mean_spread"], ascending=[False, False])

        print(f"\nSignals significant (p<0.05) on multiple coins:")
        print(f"{'Feature':<25} {'Target':<15} {'#Coins':>6} {'AvgSpread':>10} {'AvgRho':>7} {'AvgWR_sp':>8} Symbols")
        for _, r in cross.head(30).iterrows():
            print(f"{r['feature']:<25} {r['target']:<15} {r['n_symbols']:>6} {r['mean_spread']:>+10.1f} {r['mean_rho']:>+7.3f} {r['mean_wr_spread']:>+8.3f} {r['symbols']}")
    else:
        print("\nNo signals with p < 0.05 across any coin.")

    # ===================================================================
    # Report: fee-viable signals (spread > 20bps for taker)
    # ===================================================================
    print(f"\n{'='*80}")
    print("FEE-VIABLE SIGNALS (|spread| > 20 bps, i.e. could survive taker fees)")
    print("=" * 80)

    viable = mono_df[mono_df["spread_bps"].abs() > TAKER_RT_BPS].copy()
    if len(viable) > 0:
        viable = viable.sort_values("spread_bps", key=abs, ascending=False)
        print(f"\n{'Symbol':<16} {'Feature':<25} {'Target':<15} {'Spread':>8} {'rho':>6} {'pval':>8} {'Q1_wr':>6} {'Q5_wr':>6}")
        for _, r in viable.head(40).iterrows():
            sig = "***" if r["spearman_pval"] < 0.01 else "**" if r["spearman_pval"] < 0.05 else "*" if r["spearman_pval"] < 0.1 else ""
            print(f"{r['symbol']:<16} {r['feature']:<25} {r['target']:<15} {r['spread_bps']:>+8.1f} {r['spearman_rho']:>+6.3f} {r['spearman_pval']:>8.4f} {r['q1_wr']:>6.3f} {r['q5_wr']:>6.3f} {sig}")
    else:
        print("\nNo signals with |spread| > 20 bps.")

    # ===================================================================
    # Save outputs
    # ===================================================================
    mono_df.to_csv(OUTPUT_DIR / "signal_monotonicity.csv", index=False)
    all_sorts_df.to_csv(OUTPUT_DIR / "quintile_sorts.csv", index=False)

    # Save combined features for later backtesting
    combined = pd.concat(all_features, ignore_index=True)
    combined.to_parquet(OUTPUT_DIR / "features_5min.parquet", index=False)
    print(f"\nSaved {len(combined):,} rows to {OUTPUT_DIR / 'features_5min.parquet'}")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
