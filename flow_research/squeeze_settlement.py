#!/usr/bin/env python3
"""
Settlement-Aligned Squeeze Analysis

Instead of arbitrary 5-min grid, anchor on FR settlement times and measure:
1. Pre-settlement conditions (OI buildup, FR extreme, LS tilt, premium)
2. Post-settlement price moves at various horizons
3. Whether extreme pre-conditions predict large directional moves

Hypothesis: squeezes happen AROUND settlements because that's when FR charges
create forced position adjustments.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "squeeze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT",
    "TRUMPUSDT", "FARTCOINUSDT", "VIRTUALUSDT",
    "POWERUSDT",
]

# ---------------------------------------------------------------------------
# Data loading (reuse from squeeze_research.py)
# ---------------------------------------------------------------------------


def load_kline_1m(sym_dir: Path) -> pd.DataFrame:
    kline_files = sorted(f for f in sym_dir.glob("*_kline_1m.csv")
                         if "mark_price" not in f.name and "premium_index" not in f.name)
    frames = []
    for f in kline_files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    kline = pd.concat(frames, ignore_index=True)
    kline["ts"] = pd.to_datetime(kline["startTime"].astype(int), unit="ms", utc=True)
    kline = kline.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        if col in kline.columns:
            kline[col] = kline[col].astype(float)
    return kline


def load_csv_ts(sym_dir: Path, pattern: str, ts_col: str, val_cols: list) -> pd.DataFrame:
    files = sorted(sym_dir.glob(pattern))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out[ts_col].astype(int), unit="ms", utc=True)
    for c in val_cols:
        out[c] = out[c].astype(float)
    out = out.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Settlement detection from FR data
# ---------------------------------------------------------------------------


def detect_settlements(fr_df: pd.DataFrame) -> pd.DataFrame:
    """Detect settlement events from funding rate timestamps.

    Each row in FR data IS a settlement. The timestamp is when FR was charged.
    We return settlement times with the FR that was charged.
    """
    if fr_df is None or len(fr_df) == 0:
        return pd.DataFrame()

    settlements = fr_df[["ts", "fundingRate"]].copy()
    settlements = settlements.rename(columns={"fundingRate": "fr_settled"})
    settlements["fr_settled_bps"] = settlements["fr_settled"] * 10000

    # Compute time between settlements to detect interval
    settlements["dt_h"] = settlements["ts"].diff().dt.total_seconds() / 3600
    return settlements


# ---------------------------------------------------------------------------
# Pre-settlement feature computation
# ---------------------------------------------------------------------------


def compute_settlement_features(
    settlements: pd.DataFrame,
    kline: pd.DataFrame,
    oi_df: pd.DataFrame,
    ls_df: pd.DataFrame,
    pi_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each settlement, compute pre-settlement features and post-settlement returns."""

    kline_idx = kline.set_index("ts").sort_index()
    rows = []

    for _, settle in settlements.iterrows():
        t_settle = settle["ts"]
        fr_bps = settle["fr_settled_bps"]

        # --- Pre-settlement price features (lookback windows before settlement) ---
        # 1h, 4h, 24h price returns leading into settlement
        for lookback_min, label in [(60, "1h"), (240, "4h"), (1440, "24h")]:
            t_start = t_settle - pd.Timedelta(minutes=lookback_min)
            window = kline_idx.loc[t_start:t_settle]
            if len(window) >= 2:
                ret = (window["close"].iloc[-1] / window["close"].iloc[0] - 1) * 10000
                vol = window["close"].pct_change().std() * 10000 if len(window) > 5 else np.nan
                volume_total = window["turnover"].sum()
            else:
                ret = vol = volume_total = np.nan
            rows_dict = rows[-1] if rows and rows[-1].get("_t") == t_settle else None
            if rows_dict is None:
                rows_dict = {"ts": t_settle, "fr_settled_bps": fr_bps, "_t": t_settle}
                rows.append(rows_dict)
            rows_dict[f"pre_ret_{label}"] = ret
            rows_dict[f"pre_vol_{label}"] = vol
            rows_dict[f"pre_volume_{label}"] = volume_total

        r = rows[-1]

        # --- OI features (pre-settlement) ---
        if oi_df is not None and len(oi_df) > 0:
            oi_pre = oi_df[oi_df["ts"] <= t_settle]
            if len(oi_pre) >= 2:
                oi_now = oi_pre.iloc[-1]["openInterest"]
                r["oi_at_settle"] = oi_now

                # OI change over various windows
                for lookback, label in [(12, "1h"), (48, "4h"), (288, "24h")]:
                    if len(oi_pre) > lookback:
                        oi_past = oi_pre.iloc[-1 - lookback]["openInterest"]
                        r[f"oi_pct_{label}"] = (oi_now / oi_past - 1) * 100 if oi_past > 0 else np.nan
                    else:
                        r[f"oi_pct_{label}"] = np.nan

                # OI z-score vs 24h
                oi_24h = oi_pre.tail(288)["openInterest"]
                if len(oi_24h) > 10:
                    r["oi_zscore"] = (oi_now - oi_24h.mean()) / oi_24h.std() if oi_24h.std() > 0 else 0
                else:
                    r["oi_zscore"] = np.nan

        # --- LS ratio features ---
        if ls_df is not None and len(ls_df) > 0:
            ls_pre = ls_df[ls_df["ts"] <= t_settle]
            if len(ls_pre) >= 1:
                r["ls_buy_ratio"] = ls_pre.iloc[-1]["buyRatio"]
                r["ls_sell_ratio"] = ls_pre.iloc[-1]["sellRatio"]
                r["ls_ratio"] = r["ls_buy_ratio"] / r["ls_sell_ratio"] if r["ls_sell_ratio"] > 0 else np.nan

                # LS tilt vs 4h average
                ls_4h = ls_pre.tail(48)
                if len(ls_4h) > 5:
                    ratio_series = ls_4h["buyRatio"] / ls_4h["sellRatio"].replace(0, np.nan)
                    r["ls_tilt_4h"] = r["ls_ratio"] - ratio_series.mean()
                    r["ls_zscore"] = ((r["ls_ratio"] - ratio_series.mean()) /
                                      ratio_series.std()) if ratio_series.std() > 0 else 0
                else:
                    r["ls_tilt_4h"] = r["ls_zscore"] = np.nan

        # --- Premium index at settlement ---
        if pi_df is not None and len(pi_df) > 0:
            pi_pre = pi_df[pi_df["ts"] <= t_settle]
            if len(pi_pre) >= 1:
                r["premium_bps"] = pi_pre.iloc[-1]["pi_close"] * 10000

        # --- Post-settlement returns ---
        for fwd_min, label in [(5, "5m"), (15, "15m"), (30, "30m"),
                                (60, "1h"), (120, "2h"), (240, "4h")]:
            t_end = t_settle + pd.Timedelta(minutes=fwd_min)
            window = kline_idx.loc[t_settle:t_end]
            if len(window) >= 2:
                entry_price = window["close"].iloc[0]
                exit_price = window["close"].iloc[-1]
                r[f"fwd_ret_{label}"] = (exit_price / entry_price - 1) * 10000

                # MFE/MAE
                r[f"mfe_{label}"] = (window["high"].max() / entry_price - 1) * 10000
                r[f"mae_{label}"] = (window["low"].min() / entry_price - 1) * 10000
            else:
                r[f"fwd_ret_{label}"] = r[f"mfe_{label}"] = r[f"mae_{label}"] = np.nan

        # --- Signed features (relative to FR direction) ---
        # If FR is negative (shorts pay), squeeze direction is UP (shorts get squeezed)
        # If FR is positive (longs pay), squeeze direction is DOWN (longs get squeezed)
        squeeze_sign = -np.sign(fr_bps) if fr_bps != 0 else 0
        for label in ["5m", "15m", "30m", "1h", "2h", "4h"]:
            if f"fwd_ret_{label}" in r and not np.isnan(r.get(f"fwd_ret_{label}", np.nan)):
                r[f"fwd_ret_signed_{label}"] = r[f"fwd_ret_{label}"] * squeeze_sign

    # Build DataFrame
    result = pd.DataFrame(rows)
    if "_t" in result.columns:
        result = result.drop(columns=["_t"])
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_symbol(symbol: str) -> pd.DataFrame:
    """Run full settlement-aligned analysis for one symbol."""
    sym_dir = DATA_DIR / symbol
    if not sym_dir.exists():
        return pd.DataFrame()

    print(f"\n{'─'*60}")
    print(f"Loading {symbol}...")

    kline = load_kline_1m(sym_dir)
    if len(kline) == 0:
        print(f"  SKIP: no kline data")
        return pd.DataFrame()
    print(f"  Kline: {len(kline):,} rows")

    fr_df = load_csv_ts(sym_dir, "*_funding_rate.csv", "timestamp", ["fundingRate"])
    oi_df = load_csv_ts(sym_dir, "*_open_interest_5min.csv", "timestamp", ["openInterest"])
    ls_df = load_csv_ts(sym_dir, "*_long_short_ratio_5min.csv", "timestamp", ["buyRatio", "sellRatio"])

    pi_files = sorted(sym_dir.glob("*_premium_index_kline_1m.csv"))
    pi_frames = []
    for f in pi_files:
        df = pd.read_csv(f)
        if len(df) > 0:
            df = df.rename(columns={"close": "pi_close"})
            df["pi_close"] = df["pi_close"].astype(float)
            df["ts"] = pd.to_datetime(df["startTime"].astype(int), unit="ms", utc=True)
            pi_frames.append(df)
    pi_df = pd.concat(pi_frames, ignore_index=True).sort_values("ts").drop_duplicates("ts") if pi_frames else None

    print(f"  FR: {len(fr_df):,}, OI: {len(oi_df):,}, LS: {len(ls_df):,}, PI: {len(pi_df) if pi_df is not None else 0:,}")

    # Detect settlements
    settlements = detect_settlements(fr_df)
    if len(settlements) == 0:
        print(f"  SKIP: no settlements")
        return pd.DataFrame()

    median_dt = settlements["dt_h"].median()
    print(f"  Settlements: {len(settlements):,} (interval ~{median_dt:.1f}h, {24/median_dt:.0f}x/day)")

    # Compute features
    print(f"  Computing settlement features...")
    t0 = time.monotonic()
    features = compute_settlement_features(settlements, kline, oi_df, ls_df, pi_df)
    elapsed = time.monotonic() - t0
    print(f"  Done: {len(features)} settlements with features ({elapsed:.1f}s)")

    features["symbol"] = symbol
    features["fr_interval_h"] = median_dt
    return features


def quintile_analysis(df: pd.DataFrame, feature: str, target: str, n_bins: int = 5):
    """Quick quintile sort."""
    valid = df[[feature, target]].dropna()
    if len(valid) < n_bins * 10:
        return None
    try:
        valid["bin"] = pd.qcut(valid[feature], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    summary = valid.groupby("bin")[target].agg(["count", "mean", "median", "std"]).reset_index()
    summary["wr"] = valid.groupby("bin")[target].apply(lambda x: (x > 0).mean()).values
    return summary


def main():
    t0_global = time.monotonic()
    print("=" * 80)
    print("SETTLEMENT-ALIGNED SQUEEZE ANALYSIS")
    print("=" * 80)

    all_features = []
    for sym in SYMBOLS:
        features = analyze_symbol(sym)
        if len(features) > 0:
            all_features.append(features)

    if not all_features:
        print("\nNo data. Exiting.")
        return

    combined = pd.concat(all_features, ignore_index=True)
    combined.to_parquet(OUTPUT_DIR / "settlement_features.parquet", index=False)
    print(f"\nTotal settlements: {len(combined):,}")

    # ===================================================================
    # Analysis 1: FR magnitude → post-settlement returns
    # ===================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 1: FR MAGNITUDE → POST-SETTLEMENT RETURNS")
    print("=" * 80)

    for sym in combined["symbol"].unique():
        sdf = combined[combined["symbol"] == sym].copy()
        sdf["fr_abs_bps"] = sdf["fr_settled_bps"].abs()
        n_total = len(sdf)

        print(f"\n--- {sym} (n={n_total}, interval={sdf['fr_interval_h'].iloc[0]:.1f}h) ---")

        # Signed returns: does price move in squeeze direction after extreme FR?
        for target in ["fwd_ret_signed_30m", "fwd_ret_signed_1h", "fwd_ret_signed_2h", "fwd_ret_signed_4h"]:
            result = quintile_analysis(sdf, "fr_abs_bps", target)
            if result is not None:
                print(f"\n  |FR| → {target}:")
                print(f"  {'Bin':>4} {'N':>5} {'Mean':>8} {'Med':>8} {'WR':>6}")
                for _, r in result.iterrows():
                    print(f"  Q{int(r['bin']):>3} {int(r['count']):>5} {r['mean']:>+8.1f} {r['median']:>+8.1f} {r['wr']:>6.1%}")
                # Top quintile stats
                top = result.iloc[-1]
                bot = result.iloc[0]
                spread = top["mean"] - bot["mean"]
                wr_spread = top["wr"] - bot["wr"]
                print(f"  Spread: {spread:>+8.1f} bps, WR spread: {wr_spread:>+6.1%}")

    # ===================================================================
    # Analysis 2: Extreme FR buckets
    # ===================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 2: EXTREME FR BUCKETS (|FR| thresholds)")
    print("=" * 80)

    for sym in combined["symbol"].unique():
        sdf = combined[combined["symbol"] == sym].copy()
        sdf["fr_abs_bps"] = sdf["fr_settled_bps"].abs()

        print(f"\n--- {sym} ---")
        print(f"  {'FR bucket':<20} {'N':>5} {'signed_30m':>10} {'signed_1h':>10} {'signed_2h':>10} {'signed_4h':>10} {'WR_1h':>6} {'WR_4h':>6}")

        for lo, hi, label in [(0, 2, "|FR|<2"), (2, 5, "|FR|2-5"),
                               (5, 10, "|FR|5-10"), (10, 20, "|FR|10-20"),
                               (20, 50, "|FR|20-50"), (50, 999, "|FR|>50")]:
            bucket = sdf[(sdf["fr_abs_bps"] >= lo) & (sdf["fr_abs_bps"] < hi)]
            if len(bucket) < 5:
                continue
            vals = {}
            for target in ["fwd_ret_signed_30m", "fwd_ret_signed_1h", "fwd_ret_signed_2h", "fwd_ret_signed_4h"]:
                v = bucket[target].dropna()
                vals[target] = v.median() if len(v) > 0 else np.nan

            wr_1h = (bucket["fwd_ret_signed_1h"].dropna() > 0).mean() if len(bucket["fwd_ret_signed_1h"].dropna()) > 0 else np.nan
            wr_4h = (bucket["fwd_ret_signed_4h"].dropna() > 0).mean() if len(bucket["fwd_ret_signed_4h"].dropna()) > 0 else np.nan

            print(f"  {label:<20} {len(bucket):>5} {vals.get('fwd_ret_signed_30m', np.nan):>+10.1f} "
                  f"{vals.get('fwd_ret_signed_1h', np.nan):>+10.1f} {vals.get('fwd_ret_signed_2h', np.nan):>+10.1f} "
                  f"{vals.get('fwd_ret_signed_4h', np.nan):>+10.1f} {wr_1h:>6.1%} {wr_4h:>6.1%}")

    # ===================================================================
    # Analysis 3: Multi-factor conditioning
    # ===================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 3: MULTI-FACTOR (|FR|>5 AND OI/LS/Premium conditions)")
    print("=" * 80)

    for sym in combined["symbol"].unique():
        sdf = combined[combined["symbol"] == sym].copy()
        sdf["fr_abs_bps"] = sdf["fr_settled_bps"].abs()

        # Base: extreme FR
        extreme = sdf[sdf["fr_abs_bps"] >= 5].copy()
        if len(extreme) < 10:
            print(f"\n--- {sym}: only {len(extreme)} extreme FR events, skipping ---")
            continue

        print(f"\n--- {sym} (|FR|>=5: n={len(extreme)}) ---")
        print(f"  {'Condition':<40} {'N':>5} {'signed_1h':>10} {'signed_4h':>10} {'WR_1h':>6} {'WR_4h':>6} {'MFE_1h':>8} {'MAE_1h':>8}")

        conditions = [
            ("All |FR|>=5", extreme),
        ]

        # Add OI conditions
        if "oi_pct_1h" in extreme.columns and extreme["oi_pct_1h"].notna().sum() > 10:
            oi_med = extreme["oi_pct_1h"].median()
            conditions.append(("+ OI rising (1h>med)", extreme[extreme["oi_pct_1h"] > oi_med]))
            conditions.append(("+ OI falling (1h<med)", extreme[extreme["oi_pct_1h"] <= oi_med]))

        if "oi_zscore" in extreme.columns and extreme["oi_zscore"].notna().sum() > 10:
            conditions.append(("+ OI zscore > 1", extreme[extreme["oi_zscore"] > 1]))
            conditions.append(("+ OI zscore > 2", extreme[extreme["oi_zscore"] > 2]))

        # Add LS conditions
        if "ls_zscore" in extreme.columns and extreme["ls_zscore"].notna().sum() > 10:
            conditions.append(("+ LS tilted WITH crowd (z>0.5)", extreme[extreme["ls_zscore"] > 0.5]))
            conditions.append(("+ LS tilted AGAINST crowd (z<-0.5)", extreme[extreme["ls_zscore"] < -0.5]))

        # Combined: extreme FR + OI high + LS tilted
        if all(c in extreme.columns for c in ["oi_zscore", "ls_zscore"]):
            combo = extreme[(extreme["oi_zscore"] > 1) & (extreme["ls_zscore"].abs() > 0.5)]
            conditions.append(("+ OI z>1 AND |LS z|>0.5", combo))

        for label, subset in conditions:
            if len(subset) < 3:
                continue
            s1h = subset["fwd_ret_signed_1h"].dropna()
            s4h = subset["fwd_ret_signed_4h"].dropna()
            mfe = subset["mfe_1h"].dropna()
            mae = subset["mae_1h"].dropna()

            print(f"  {label:<40} {len(subset):>5} "
                  f"{s1h.median() if len(s1h)>0 else np.nan:>+10.1f} "
                  f"{s4h.median() if len(s4h)>0 else np.nan:>+10.1f} "
                  f"{(s1h>0).mean() if len(s1h)>0 else np.nan:>6.1%} "
                  f"{(s4h>0).mean() if len(s4h)>0 else np.nan:>6.1%} "
                  f"{mfe.median() if len(mfe)>0 else np.nan:>+8.1f} "
                  f"{mae.median() if len(mae)>0 else np.nan:>+8.1f}")

    # ===================================================================
    # Analysis 4: Spearman correlations across all symbols
    # ===================================================================
    print(f"\n{'='*80}")
    print("ANALYSIS 4: SPEARMAN CORRELATIONS (feature → signed returns)")
    print("=" * 80)

    features_to_test = [
        "fr_settled_bps", "fr_abs_bps",
        "pre_ret_1h", "pre_ret_4h", "pre_ret_24h",
        "pre_vol_1h", "pre_vol_4h",
        "oi_pct_1h", "oi_pct_4h", "oi_pct_24h", "oi_zscore",
        "ls_ratio", "ls_tilt_4h", "ls_zscore",
        "premium_bps",
    ]
    targets = ["fwd_ret_signed_30m", "fwd_ret_signed_1h", "fwd_ret_signed_4h"]

    # Add fr_abs_bps to combined
    combined["fr_abs_bps"] = combined["fr_settled_bps"].abs()

    corr_rows = []
    for sym in combined["symbol"].unique():
        sdf = combined[combined["symbol"] == sym]
        for feat in features_to_test:
            if feat not in sdf.columns:
                continue
            for tgt in targets:
                if tgt not in sdf.columns:
                    continue
                valid = sdf[[feat, tgt]].dropna()
                if len(valid) < 20:
                    continue
                rho, pval = spearmanr(valid[feat], valid[tgt])
                corr_rows.append({
                    "symbol": sym, "feature": feat, "target": tgt,
                    "rho": rho, "pval": pval, "n": len(valid),
                })

    corr_df = pd.DataFrame(corr_rows)

    # Show features with consistent direction across coins
    print(f"\n{'Feature':<20} {'Target':<25} {'#Sig':>5} {'AvgRho':>7} {'Coins'}")
    for (feat, tgt), grp in corr_df.groupby(["feature", "target"]):
        sig = grp[grp["pval"] < 0.1]
        if len(sig) >= 2:
            syms = ",".join(sig["symbol"].values)
            avg_rho = sig["rho"].mean()
            print(f"{feat:<20} {tgt:<25} {len(sig):>5} {avg_rho:>+7.3f} {syms}")

    # Show all correlations for extreme FR analysis
    print(f"\n--- All correlations (p<0.1) ---")
    sig_corrs = corr_df[corr_df["pval"] < 0.1].sort_values("pval")
    print(f"{'Symbol':<16} {'Feature':<20} {'Target':<25} {'rho':>6} {'pval':>8} {'N':>5}")
    for _, r in sig_corrs.head(40).iterrows():
        print(f"{r['symbol']:<16} {r['feature']:<20} {r['target']:<25} {r['rho']:>+6.3f} {r['pval']:>8.4f} {r['n']:>5}")

    elapsed = time.monotonic() - t0_global
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
