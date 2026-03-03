#!/usr/bin/env python3
"""
Broad Universe FR Squeeze Analysis

Pools ALL settlement events across 100+ coins to get enough extreme FR events
for statistically meaningful analysis.

Key question: when FR is extreme (|FR| > X bps), does price move in the
squeeze direction (against the crowded side) after settlement?
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "squeeze"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAKER_RT_BPS = 20.0
MAKER_RT_BPS = 8.0

# ---------------------------------------------------------------------------
# Data loading (lightweight — only FR + kline needed for pooled analysis)
# ---------------------------------------------------------------------------


def load_symbol_lite(symbol: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Load FR + kline + OI + LS for a symbol. Returns (fr, kline, oi, ls) or Nones."""
    sym_dir = DATA_DIR / symbol
    if not sym_dir.exists():
        return None, None, None, None

    # FR
    fr_files = sorted(sym_dir.glob("*_funding_rate.csv"))
    if not fr_files:
        return None, None, None, None
    fr_frames = []
    for f in fr_files:
        df = pd.read_csv(f)
        if len(df) > 0:
            fr_frames.append(df)
    if not fr_frames:
        return None, None, None, None
    fr = pd.concat(fr_frames, ignore_index=True)
    fr["ts"] = pd.to_datetime(fr["timestamp"].astype(int), unit="ms", utc=True)
    fr["fundingRate"] = fr["fundingRate"].astype(float)
    fr = fr.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    # Kline 1m
    kline_files = sorted(f for f in sym_dir.glob("*_kline_1m.csv")
                         if "mark_price" not in f.name and "premium_index" not in f.name)
    if not kline_files:
        return None, None, None, None
    kl_frames = []
    for f in kline_files:
        df = pd.read_csv(f)
        if len(df) > 0:
            kl_frames.append(df)
    if not kl_frames:
        return None, None, None, None
    kline = pd.concat(kl_frames, ignore_index=True)
    kline["ts"] = pd.to_datetime(kline["startTime"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        kline[col] = kline[col].astype(float)
    kline = kline.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    # Filter bad rows
    kline = kline[kline["close"] > 0].copy()

    # OI (optional)
    oi = None
    oi_files = sorted(sym_dir.glob("*_open_interest_5min.csv"))
    if oi_files:
        oi_frames = []
        for f in oi_files:
            df = pd.read_csv(f)
            if len(df) > 0:
                oi_frames.append(df)
        if oi_frames:
            oi = pd.concat(oi_frames, ignore_index=True)
            oi["ts"] = pd.to_datetime(oi["timestamp"].astype(int), unit="ms", utc=True)
            oi["openInterest"] = oi["openInterest"].astype(float)
            oi = oi.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    # LS (optional)
    ls = None
    ls_files = sorted(sym_dir.glob("*_long_short_ratio_5min.csv"))
    if ls_files:
        ls_frames = []
        for f in ls_files:
            df = pd.read_csv(f)
            if len(df) > 0:
                ls_frames.append(df)
        if ls_frames:
            ls = pd.concat(ls_frames, ignore_index=True)
            ls["ts"] = pd.to_datetime(ls["timestamp"].astype(int), unit="ms", utc=True)
            ls["buyRatio"] = ls["buyRatio"].astype(float)
            ls["sellRatio"] = ls["sellRatio"].astype(float)
            ls = ls.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    return fr, kline, oi, ls


def compute_settlement_returns(
    symbol: str, fr: pd.DataFrame, kline: pd.DataFrame,
    oi: pd.DataFrame | None, ls: pd.DataFrame | None,
) -> pd.DataFrame:
    """For each settlement, compute post-settlement returns in squeeze direction.

    Vectorized approach: resample kline to 5min, then use merge_asof for speed.
    """
    # Resample kline to 5min for faster lookups
    kl = kline.set_index("ts").sort_index()
    k5 = kl.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "turnover": "sum",
    }).dropna(subset=["close"])

    # Pre-compute rolling returns on 5min bars for pre-settlement lookback
    k5["ret_12"] = k5["close"].pct_change(12) * 10000   # 1h
    k5["ret_48"] = k5["close"].pct_change(48) * 10000   # 4h
    k5["vol_12"] = k5["close"].pct_change().rolling(12).std() * 10000

    # Pre-compute forward returns for each horizon
    for bars, label in [(1, "5m"), (3, "15m"), (6, "30m"), (12, "1h"), (24, "2h"), (48, "4h")]:
        k5[f"fwd_{label}"] = (k5["close"].shift(-bars) / k5["close"] - 1) * 10000
        # Rolling max/min for MFE/MAE
        fwd_high = k5["high"].shift(-1).rolling(bars, min_periods=1).max().shift(-bars+1)
        fwd_low = k5["low"].shift(-1).rolling(bars, min_periods=1).min().shift(-bars+1)
        # Proper MFE/MAE: max high / min low in next N bars
        mfe_vals = np.full(len(k5), np.nan)
        mae_vals = np.full(len(k5), np.nan)
        close_arr = k5["close"].values
        high_arr = k5["high"].values
        low_arr = k5["low"].values
        for i in range(len(k5) - bars):
            if close_arr[i] > 0:
                mfe_vals[i] = (np.max(high_arr[i+1:i+1+bars]) / close_arr[i] - 1) * 10000
                mae_vals[i] = (np.min(low_arr[i+1:i+1+bars]) / close_arr[i] - 1) * 10000
        k5[f"mfe_{label}"] = mfe_vals
        k5[f"mae_{label}"] = mae_vals

    k5_reset = k5.reset_index()

    # Build settlement DataFrame
    settle = fr[["ts", "fundingRate"]].copy()
    settle["fr_bps"] = settle["fundingRate"] * 10000
    settle["fr_abs_bps"] = settle["fr_bps"].abs()
    settle["squeeze_sign"] = -np.sign(settle["fr_bps"])
    settle["symbol"] = symbol

    # merge_asof to nearest 5min bar at or before settlement
    settle = pd.merge_asof(
        settle.sort_values("ts"),
        k5_reset.sort_values("ts"),
        on="ts", direction="backward", tolerance=pd.Timedelta("5min"),
    )

    # OI at settlement
    if oi is not None and len(oi) > 0:
        oi_copy = oi[["ts", "openInterest"]].copy()
        oi_copy["oi_lag12"] = oi_copy["openInterest"].shift(12)
        oi_copy["oi_lag48"] = oi_copy["openInterest"].shift(48)
        settle = pd.merge_asof(
            settle.sort_values("ts"), oi_copy.sort_values("ts"),
            on="ts", direction="backward", tolerance=pd.Timedelta("5min"),
        )
        settle["oi"] = settle["openInterest"]
        settle["oi_pct_1h"] = np.where(
            settle["oi_lag12"] > 0,
            (settle["openInterest"] / settle["oi_lag12"] - 1) * 100, np.nan)
        settle["oi_pct_4h"] = np.where(
            settle["oi_lag48"] > 0,
            (settle["openInterest"] / settle["oi_lag48"] - 1) * 100, np.nan)
        settle.drop(columns=["openInterest", "oi_lag12", "oi_lag48"], inplace=True, errors="ignore")

    # LS at settlement
    if ls is not None and len(ls) > 0:
        settle = pd.merge_asof(
            settle.sort_values("ts"), ls[["ts", "buyRatio"]].sort_values("ts"),
            on="ts", direction="backward", tolerance=pd.Timedelta("5min"),
        )
        settle.rename(columns={"buyRatio": "ls_buy_ratio"}, inplace=True)

    # Rename pre-settlement features
    settle.rename(columns={"ret_12": "pre_ret_1h", "ret_48": "pre_ret_4h",
                            "vol_12": "pre_vol_1h"}, inplace=True)

    # Compute signed returns
    for label in ["5m", "15m", "30m", "1h", "2h", "4h"]:
        col = f"fwd_{label}"
        if col in settle.columns:
            settle[f"{col}_signed"] = settle[col] * settle["squeeze_sign"]

    # Drop merge artifacts
    settle.drop(columns=["fundingRate", "open", "high", "low", "close",
                          "volume", "turnover"], inplace=True, errors="ignore")

    return settle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()
    print("=" * 80)
    print("BROAD UNIVERSE FR SQUEEZE ANALYSIS")
    print("=" * 80)

    # Discover all symbols with data
    all_syms = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(all_syms)} symbol directories")

    # Load and process
    all_settle = []
    sym_stats = []
    skipped = 0

    for i, sym in enumerate(all_syms, 1):
        fr, kline, oi, ls = load_symbol_lite(sym)
        if fr is None or kline is None:
            skipped += 1
            continue
        if len(fr) < 20:
            skipped += 1
            continue

        # Detect FR interval
        dt_h = fr["ts"].diff().dt.total_seconds().median() / 3600

        settle_df = compute_settlement_returns(sym, fr, kline, oi, ls)
        settle_df["fr_interval_h"] = dt_h
        all_settle.append(settle_df)

        n_extreme = (settle_df["fr_abs_bps"] >= 5).sum()
        sym_stats.append({
            "symbol": sym,
            "n_settle": len(settle_df),
            "fr_interval_h": dt_h,
            "n_extreme_5": n_extreme,
            "fr_abs_median": settle_df["fr_abs_bps"].median(),
            "fr_abs_p95": settle_df["fr_abs_bps"].quantile(0.95),
            "fr_abs_max": settle_df["fr_abs_bps"].max(),
        })

        if i % 20 == 0 or i == len(all_syms):
            print(f"  [{i}/{len(all_syms)}] processed, skipped={skipped}")

    if not all_settle:
        print("No settlement data. Exiting.")
        return

    combined = pd.concat(all_settle, ignore_index=True)
    stats_df = pd.DataFrame(sym_stats).sort_values("n_extreme_5", ascending=False)

    print(f"\nLoaded {len(combined):,} settlements from {len(sym_stats)} coins (skipped {skipped})")
    print(f"  Total extreme FR (|FR|>=5): {(combined['fr_abs_bps'] >= 5).sum():,}")
    print(f"  Total extreme FR (|FR|>=10): {(combined['fr_abs_bps'] >= 10).sum():,}")
    print(f"  Total extreme FR (|FR|>=20): {(combined['fr_abs_bps'] >= 20).sum():,}")
    print(f"  Total extreme FR (|FR|>=50): {(combined['fr_abs_bps'] >= 50).sum():,}")

    # Save
    combined.to_parquet(OUTPUT_DIR / "broad_settlements.parquet", index=False)
    stats_df.to_csv(OUTPUT_DIR / "broad_symbol_stats.csv", index=False)

    # ===================================================================
    # Top extreme-FR coins
    # ===================================================================
    print(f"\n{'='*80}")
    print("TOP COINS BY EXTREME FR EVENTS (|FR|>=5)")
    print("=" * 80)
    print(f"{'Symbol':<20} {'N_set':>6} {'Int':>4} {'N_ext5':>7} {'Med|FR|':>8} {'P95|FR|':>8} {'Max|FR|':>8}")
    for _, r in stats_df.head(30).iterrows():
        print(f"{r['symbol']:<20} {r['n_settle']:>6.0f} {r['fr_interval_h']:>4.0f}h {r['n_extreme_5']:>7.0f} "
              f"{r['fr_abs_median']:>8.1f} {r['fr_abs_p95']:>8.1f} {r['fr_abs_max']:>8.1f}")

    # ===================================================================
    # POOLED ANALYSIS: all coins, all settlements
    # ===================================================================
    print(f"\n{'='*80}")
    print("POOLED ANALYSIS: FR MAGNITUDE → SIGNED POST-SETTLEMENT RETURNS")
    print("=" * 80)

    targets = ["fwd_5m_signed", "fwd_15m_signed", "fwd_30m_signed",
               "fwd_1h_signed", "fwd_2h_signed", "fwd_4h_signed"]

    # FR buckets
    buckets = [
        (0, 2, "|FR|<2"),
        (2, 5, "|FR|2-5"),
        (5, 10, "|FR|5-10"),
        (10, 20, "|FR|10-20"),
        (20, 50, "|FR|20-50"),
        (50, 100, "|FR|50-100"),
        (100, 9999, "|FR|>100"),
    ]

    print(f"\n{'Bucket':<15} {'N':>6}  ", end="")
    for t in targets:
        label = t.replace("fwd_", "").replace("_signed", "")
        print(f"{'med_'+label:>9} {'wr_'+label:>7}  ", end="")
    print()
    print("-" * 130)

    for lo, hi, label in buckets:
        subset = combined[(combined["fr_abs_bps"] >= lo) & (combined["fr_abs_bps"] < hi)]
        if len(subset) < 5:
            continue
        print(f"{label:<15} {len(subset):>6}  ", end="")
        for tgt in targets:
            vals = subset[tgt].dropna()
            if len(vals) > 0:
                med = vals.median()
                wr = (vals > 0).mean()
                print(f"{med:>+9.1f} {wr:>7.1%}  ", end="")
            else:
                print(f"{'N/A':>9} {'N/A':>7}  ", end="")
        print()

    # ===================================================================
    # Statistical test: is the signed return significantly > 0 for extreme FR?
    # ===================================================================
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS: signed return > 0 for extreme FR?")
    print("=" * 80)

    for lo, hi, label in buckets:
        if lo < 5:
            continue
        subset = combined[(combined["fr_abs_bps"] >= lo) & (combined["fr_abs_bps"] < hi)]
        if len(subset) < 10:
            continue
        print(f"\n--- {label} (n={len(subset)}) ---")
        for tgt in targets:
            vals = subset[tgt].dropna()
            if len(vals) < 10:
                continue
            mean = vals.mean()
            median = vals.median()
            wr = (vals > 0).mean()
            t_stat, p_val = ttest_1samp(vals, 0)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            # Is it profitable after fees?
            net_taker = mean - TAKER_RT_BPS
            net_maker = mean - MAKER_RT_BPS
            print(f"  {tgt:<20} mean={mean:>+7.1f} med={median:>+7.1f} WR={wr:.1%} "
                  f"t={t_stat:>+5.2f} p={p_val:.4f} net_taker={net_taker:>+7.1f} net_maker={net_maker:>+7.1f} {sig}")

    # ===================================================================
    # Conditioning: extreme FR + OI rising vs falling
    # ===================================================================
    print(f"\n{'='*80}")
    print("CONDITIONING: extreme FR (|FR|>=5) + OI/LS conditions")
    print("=" * 80)

    extreme = combined[combined["fr_abs_bps"] >= 5].copy()
    if len(extreme) > 20:
        conditions = [
            ("All |FR|>=5", extreme),
        ]
        if "oi_pct_1h" in extreme.columns and extreme["oi_pct_1h"].notna().sum() > 20:
            med = extreme["oi_pct_1h"].median()
            conditions.append(("+ OI rising 1h", extreme[extreme["oi_pct_1h"] > med]))
            conditions.append(("+ OI falling 1h", extreme[extreme["oi_pct_1h"] <= med]))
            conditions.append(("+ OI surge (>2%)", extreme[extreme["oi_pct_1h"] > 2]))
            conditions.append(("+ OI drop (<-2%)", extreme[extreme["oi_pct_1h"] < -2]))

        if "oi_pct_4h" in extreme.columns and extreme["oi_pct_4h"].notna().sum() > 20:
            conditions.append(("+ OI surge 4h (>5%)", extreme[extreme["oi_pct_4h"] > 5]))

        if "ls_buy_ratio" in extreme.columns and extreme["ls_buy_ratio"].notna().sum() > 20:
            # For positive FR (longs pay): high buy ratio = more crowded longs = bigger squeeze potential
            pos_fr = extreme[extreme["fr_bps"] > 5]
            neg_fr = extreme[extreme["fr_bps"] < -5]
            if len(pos_fr) > 10:
                med_buy = pos_fr["ls_buy_ratio"].median()
                conditions.append(("FR>5 + high buy ratio", pos_fr[pos_fr["ls_buy_ratio"] > med_buy]))
                conditions.append(("FR>5 + low buy ratio", pos_fr[pos_fr["ls_buy_ratio"] <= med_buy]))
            if len(neg_fr) > 10:
                med_buy = neg_fr["ls_buy_ratio"].median()
                conditions.append(("FR<-5 + high buy ratio", neg_fr[neg_fr["ls_buy_ratio"] > med_buy]))
                conditions.append(("FR<-5 + low buy ratio", neg_fr[neg_fr["ls_buy_ratio"] <= med_buy]))

        # Pre-momentum alignment
        if "pre_ret_1h" in extreme.columns and extreme["pre_ret_1h"].notna().sum() > 20:
            # Price already moving in squeeze direction?
            extreme_with_pre = extreme.dropna(subset=["pre_ret_1h"])
            aligned = extreme_with_pre[extreme_with_pre["pre_ret_1h"] * extreme_with_pre["squeeze_sign"] > 0]
            counter = extreme_with_pre[extreme_with_pre["pre_ret_1h"] * extreme_with_pre["squeeze_sign"] <= 0]
            conditions.append(("+ pre-1h aligned w/ squeeze", aligned))
            conditions.append(("+ pre-1h counter to squeeze", counter))

        print(f"\n{'Condition':<35} {'N':>5}  {'med_30m':>8} {'wr_30m':>6}  {'med_1h':>8} {'wr_1h':>6}  {'med_4h':>8} {'wr_4h':>6}")
        print("-" * 100)
        for label, subset in conditions:
            if len(subset) < 5:
                continue
            r30 = subset["fwd_30m_signed"].dropna()
            r1h = subset["fwd_1h_signed"].dropna()
            r4h = subset["fwd_4h_signed"].dropna()
            print(f"{label:<35} {len(subset):>5}  "
                  f"{r30.median() if len(r30)>0 else np.nan:>+8.1f} {(r30>0).mean() if len(r30)>0 else 0:>6.1%}  "
                  f"{r1h.median() if len(r1h)>0 else np.nan:>+8.1f} {(r1h>0).mean() if len(r1h)>0 else 0:>6.1%}  "
                  f"{r4h.median() if len(r4h)>0 else np.nan:>+8.1f} {(r4h>0).mean() if len(r4h)>0 else 0:>6.1%}")

    # ===================================================================
    # By FR interval: 1h vs 4h vs 8h
    # ===================================================================
    print(f"\n{'='*80}")
    print("BY FR INTERVAL")
    print("=" * 80)

    for interval in [1, 4, 8]:
        subset = combined[combined["fr_interval_h"].round() == interval]
        if len(subset) < 20:
            continue
        extreme_sub = subset[subset["fr_abs_bps"] >= 5]
        print(f"\n--- {interval}h interval ({len(subset):,} settlements, {len(extreme_sub):,} extreme) ---")
        if len(extreme_sub) < 10:
            print("  Not enough extreme events")
            continue

        for lo, hi, label in [(5, 20, "|FR|5-20"), (20, 100, "|FR|20-100"), (100, 9999, "|FR|>100")]:
            bucket = extreme_sub[(extreme_sub["fr_abs_bps"] >= lo) & (extreme_sub["fr_abs_bps"] < hi)]
            if len(bucket) < 5:
                continue
            for tgt_label, tgt in [("30m", "fwd_30m_signed"), ("1h", "fwd_1h_signed"), ("4h", "fwd_4h_signed")]:
                vals = bucket[tgt].dropna()
                if len(vals) < 5:
                    continue
                mean = vals.mean()
                wr = (vals > 0).mean()
                _, p = ttest_1samp(vals, 0) if len(vals) > 2 else (0, 1)
                sig = "**" if p < 0.05 else "*" if p < 0.1 else ""
                print(f"  {label:<15} {tgt_label:>4} n={len(vals):>4} mean={mean:>+7.1f} med={vals.median():>+7.1f} WR={wr:.1%} p={p:.3f} {sig}")

    # ===================================================================
    # Trade-level profitability simulation
    # ===================================================================
    print(f"\n{'='*80}")
    print("TRADE-LEVEL PROFITABILITY (simple: enter at settlement, exit at horizon)")
    print("=" * 80)

    for min_fr in [5, 10, 20, 50]:
        trades = combined[combined["fr_abs_bps"] >= min_fr].copy()
        if len(trades) < 10:
            continue
        print(f"\n--- Entry threshold: |FR| >= {min_fr} bps (n={len(trades)}) ---")
        for horizon_label, col in [("30m", "fwd_30m_signed"), ("1h", "fwd_1h_signed"),
                                     ("2h", "fwd_2h_signed"), ("4h", "fwd_4h_signed")]:
            vals = trades[col].dropna()
            if len(vals) < 5:
                continue
            gross = vals.mean()
            net_taker = gross - TAKER_RT_BPS
            net_maker = gross - MAKER_RT_BPS
            wr = (vals > 0).mean()
            # Per-trade PnL distribution
            p25 = vals.quantile(0.25)
            p75 = vals.quantile(0.75)

            # Annual estimate (rough)
            trades_per_day = len(vals) / 90  # ~90 days of data
            annual_net_taker = net_taker * trades_per_day * 365

            status = "✓" if net_taker > 0 else "✗"
            print(f"  {status} exit={horizon_label:>3} gross={gross:>+7.1f} net_taker={net_taker:>+7.1f} "
                  f"net_maker={net_maker:>+7.1f} WR={wr:.1%} P25/P75={p25:>+7.1f}/{p75:>+7.1f} "
                  f"trades/day={trades_per_day:.1f} ann_bps={annual_net_taker:>+8.0f}")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
