#!/usr/bin/env python3
"""
LS Ratio EWMA Momentum — 3-Year Validation (SOLUSDT)

Tests the simple adaptive EWMA z-score momentum signal from v24d
on the full 3-year period: Jan 2023 – Jan 2026.

This is the only signal that survived OOS in v24d:
  SOL EWMA(4h) z>1.5 momentum: +5.5 bps OOS, +23.0 bps IS
  SOL EWMA(4h) z>2.0 momentum: +8.1 bps OOS, +10.5 bps IS

Now we test it across 3 years covering multiple market regimes
(bear market, recovery, bull run, consolidation).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PARQUET_DIR = Path("./parquet")
SYMBOL = "SOLUSDT"

START_DATE = "2023-01-01"
END_DATE = "2026-01-31"

FEE_BPS = 7  # round-trip fee
FWD_HORIZON = 48  # 4h = 48 bars of 5m

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_binance_klines_5m(start_date, end_date):
    kline_dir = PARQUET_DIR / SYMBOL / "binance" / "klines_futures" / "5m"
    dates = pd.date_range(start_date, end_date)
    dfs = []
    loaded = 0
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = kline_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
            loaded += 1
    if not dfs:
        print(f"  No klines found!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Klines: {len(df)} bars from {loaded} days")
    return df


def load_binance_metrics(start_date, end_date):
    metrics_dir = PARQUET_DIR / SYMBOL / "binance" / "metrics"
    dates = pd.date_range(start_date, end_date)
    dfs = []
    loaded = 0
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = metrics_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
            loaded += 1
    if not dfs:
        print(f"  No metrics found!")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    print(f"  Metrics: {len(df)} rows from {loaded} days")
    return df


# ---------------------------------------------------------------------------
# Merge and compute EWMA z-scores
# ---------------------------------------------------------------------------

def build_data(klines_df, metrics_df):
    merged = pd.merge_asof(
        klines_df.sort_values("timestamp_us"),
        metrics_df.sort_values("timestamp_us"),
        on="timestamp_us",
        tolerance=300_000_000,
        direction="nearest",
    )
    n_matched = merged["top_trader_ls_ratio_accounts"].notna().sum()
    print(f"  Merged: {len(merged)} bars, {n_matched} with LS ratio ({100*n_matched/len(merged):.1f}%)")

    ls = merged["top_trader_ls_ratio_accounts"]

    # EWMA z-scores with multiple halflifes
    for hl_bars, hl_name in [(48, "4h"), (96, "8h"), (288, "24h"), (576, "48h")]:
        ewm_mean = ls.ewm(halflife=hl_bars, min_periods=hl_bars//2).mean()
        ewm_std = ls.ewm(halflife=hl_bars, min_periods=hl_bars//2).std()
        merged[f"ls_z_ewma_{hl_name}"] = (ls - ewm_mean) / (ewm_std + 1e-10)

    # Fixed rolling z-scores for comparison
    for win, name in [(288, "24h_fixed"), (576, "48h_fixed")]:
        roll_mean = ls.rolling(win, min_periods=win//2).mean()
        roll_std = ls.rolling(win, min_periods=win//2).std()
        merged[f"ls_z_{name}"] = (ls - roll_mean) / (roll_std + 1e-10)

    # Forward returns
    merged["fwd_ret_4h"] = merged["close"].pct_change(FWD_HORIZON).shift(-FWD_HORIZON) * 10000

    # Datetime for analysis
    merged["dt"] = pd.to_datetime(merged["timestamp_us"], unit="us")

    return merged


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------

def eval_signal(df, z_col, thresh, direction="momentum", label=""):
    """
    Evaluate a z-score threshold signal.
    momentum: z > thresh → long, z < -thresh → short
    contrarian: z > thresh → short, z < -thresh → long
    """
    z = df[z_col].values
    fwd = df["fwd_ret_4h"].values
    ts = df["dt"].values

    if direction == "momentum":
        long_mask = (z > thresh) & np.isfinite(fwd)
        short_mask = (z < -thresh) & np.isfinite(fwd)
    else:
        long_mask = (z < -thresh) & np.isfinite(fwd)
        short_mask = (z > thresh) & np.isfinite(fwd)

    trade_mask = long_mask | short_mask
    n_trades = trade_mask.sum()

    if n_trades < 50:
        return None

    trade_rets = np.where(
        long_mask[trade_mask], fwd[trade_mask], -fwd[trade_mask]
    ) - FEE_BPS

    avg = np.mean(trade_rets)
    wr = (trade_rets > 0).mean() * 100
    sharpe = np.mean(trade_rets) / (np.std(trade_rets) + 1e-10) * np.sqrt(252 * 288 / FWD_HORIZON)
    # Annualized: 288 bars/day, each trade holds 48 bars → 6 trades/day equivalent

    # Cumulative PnL
    cum_pnl = np.cumsum(trade_rets)
    max_dd = 0
    peak = 0
    for v in cum_pnl:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    return {
        "n": n_trades,
        "avg_bps": avg,
        "wr": wr,
        "sharpe_ann": sharpe,
        "total_pnl_bps": cum_pnl[-1],
        "max_dd_bps": max_dd,
        "trade_rets": trade_rets,
        "trade_ts": ts[trade_mask],
        "long_mask": long_mask,
        "short_mask": short_mask,
    }


def eval_by_period(df, z_col, thresh, direction="momentum", periods=None):
    """Evaluate signal across multiple time periods."""
    z = df[z_col].values
    fwd = df["fwd_ret_4h"].values
    dt = df["dt"].values

    results = {}
    for pname, (pstart, pend) in periods.items():
        ps = np.datetime64(pstart)
        pe = np.datetime64(pend)
        mask = (dt >= ps) & (dt < pe)

        if direction == "momentum":
            long_m = (z > thresh) & np.isfinite(fwd) & mask
            short_m = (z < -thresh) & np.isfinite(fwd) & mask
        else:
            long_m = (z < -thresh) & np.isfinite(fwd) & mask
            short_m = (z > thresh) & np.isfinite(fwd) & mask

        trade_m = long_m | short_m
        n = trade_m.sum()
        if n < 20:
            results[pname] = {"n": n, "avg_bps": float("nan"), "wr": float("nan")}
            continue

        rets = np.where(long_m[trade_m], fwd[trade_m], -fwd[trade_m]) - FEE_BPS
        results[pname] = {
            "n": n,
            "avg_bps": np.mean(rets),
            "wr": (rets > 0).mean() * 100,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  LS RATIO EWMA MOMENTUM — 3-YEAR VALIDATION")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Fee: {FEE_BPS} bps round-trip")
    print(f"{'='*70}")

    # Load data
    print(f"\nLoading data...")
    klines = load_binance_klines_5m(START_DATE, END_DATE)
    metrics = load_binance_metrics(START_DATE, END_DATE)

    if klines.empty or metrics.empty:
        print("FATAL: missing data")
        return

    # Build features
    print(f"\nBuilding features...")
    df = build_data(klines, metrics)

    # LS ratio distribution by quarter
    print(f"\n{'='*70}")
    print(f"  LS RATIO DISTRIBUTION BY QUARTER")
    print(f"{'='*70}")
    df["_quarter"] = df["dt"].dt.to_period("Q")
    for q, grp in df.groupby("_quarter"):
        ls_vals = grp["top_trader_ls_ratio_accounts"].dropna()
        price = grp["close"].iloc[-1] if len(grp) > 0 else float("nan")
        if len(ls_vals) > 0:
            print(f"  {q}: LS mean={ls_vals.mean():.3f} std={ls_vals.std():.3f} "
                  f"min={ls_vals.min():.3f} max={ls_vals.max():.3f} "
                  f"n={len(ls_vals)}  price≈${price:.1f}")

    # Price context
    print(f"\n  Price range: ${df['close'].min():.2f} – ${df['close'].max():.2f}")
    print(f"  Start: ${df['close'].iloc[288]:.2f}  End: ${df['close'].iloc[-1]:.2f}")

    # Define time periods for breakdown
    periods = {
        "2023-H1": ("2023-01-01", "2023-07-01"),
        "2023-H2": ("2023-07-01", "2024-01-01"),
        "2024-H1": ("2024-01-01", "2024-07-01"),
        "2024-H2": ("2024-07-01", "2025-01-01"),
        "2025-H1": ("2025-01-01", "2025-07-01"),
        "2025-H2": ("2025-07-01", "2026-02-01"),
    }

    # Also define yearly periods
    yearly_periods = {
        "2023": ("2023-01-01", "2024-01-01"),
        "2024": ("2024-01-01", "2025-01-01"),
        "2025": ("2025-01-01", "2026-01-01"),
        "2026-Jan": ("2026-01-01", "2026-02-01"),
    }

    # =====================================================================
    # Test all EWMA z-score configs
    # =====================================================================
    z_cols = [
        "ls_z_ewma_4h", "ls_z_ewma_8h", "ls_z_ewma_24h", "ls_z_ewma_48h",
        "ls_z_24h_fixed", "ls_z_48h_fixed",
    ]
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]

    print(f"\n{'='*70}")
    print(f"  MOMENTUM SIGNAL SCAN (z>T → long, z<-T → short)")
    print(f"{'='*70}")
    print(f"\n  {'Z-score':18s} {'Thresh':>6s} {'N':>8s} {'Avg':>8s} {'WR':>6s} "
          f"{'Sharpe':>8s} {'TotalPnL':>10s} {'MaxDD':>8s}")
    print(f"  {'-'*76}")

    best_results = []

    for z_col in z_cols:
        for thresh in thresholds:
            res = eval_signal(df, z_col, thresh, "momentum")
            if res is None:
                continue

            flag = "✅" if res["avg_bps"] > 0 and res["sharpe_ann"] > 0.5 else "  "
            print(f"{flag} {z_col:18s} {thresh:>6.1f} {res['n']:>8d} "
                  f"{res['avg_bps']:>+8.1f} {res['wr']:>5.1f}% "
                  f"{res['sharpe_ann']:>+8.2f} {res['total_pnl_bps']:>+10.0f} "
                  f"{res['max_dd_bps']:>8.0f}")

            if res["avg_bps"] > 0:
                best_results.append((z_col, thresh, res))

    # =====================================================================
    # Contrarian for comparison
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  CONTRARIAN SIGNAL SCAN (z>T → short, z<-T → long)")
    print(f"{'='*70}")
    print(f"\n  {'Z-score':18s} {'Thresh':>6s} {'N':>8s} {'Avg':>8s} {'WR':>6s} "
          f"{'Sharpe':>8s} {'TotalPnL':>10s} {'MaxDD':>8s}")
    print(f"  {'-'*76}")

    for z_col in z_cols:
        for thresh in thresholds:
            res = eval_signal(df, z_col, thresh, "contrarian")
            if res is None:
                continue

            flag = "✅" if res["avg_bps"] > 0 and res["sharpe_ann"] > 0.5 else "  "
            print(f"{flag} {z_col:18s} {thresh:>6.1f} {res['n']:>8d} "
                  f"{res['avg_bps']:>+8.1f} {res['wr']:>5.1f}% "
                  f"{res['sharpe_ann']:>+8.2f} {res['total_pnl_bps']:>+10.0f} "
                  f"{res['max_dd_bps']:>8.0f}")

    # =====================================================================
    # Detailed breakdown for best signals
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  HALF-YEARLY BREAKDOWN — TOP MOMENTUM SIGNALS")
    print(f"{'='*70}")

    # Sort by Sharpe
    best_results.sort(key=lambda x: x[2]["sharpe_ann"], reverse=True)

    for z_col, thresh, full_res in best_results[:10]:
        print(f"\n  {z_col} z>{thresh:.1f} (full: avg={full_res['avg_bps']:+.1f}bps "
              f"sharpe={full_res['sharpe_ann']:+.2f} n={full_res['n']})")

        period_res = eval_by_period(df, z_col, thresh, "momentum", periods)
        n_positive = 0
        n_periods = 0
        for pname, pr in sorted(period_res.items()):
            if pr["n"] < 20:
                print(f"    {pname}: n={pr['n']} (too few)")
                continue
            n_periods += 1
            flag = "✅" if pr["avg_bps"] > 0 else "  "
            if pr["avg_bps"] > 0:
                n_positive += 1
            print(f"  {flag} {pname}: avg={pr['avg_bps']:+.1f}bps wr={pr['wr']:.1f}% n={pr['n']}")

        if n_periods > 0:
            print(f"    → Positive in {n_positive}/{n_periods} half-years "
                  f"({100*n_positive/n_periods:.0f}%)")

    # =====================================================================
    # Yearly breakdown for the best signal
    # =====================================================================
    if best_results:
        z_col, thresh, full_res = best_results[0]
        print(f"\n{'='*70}")
        print(f"  YEARLY BREAKDOWN — BEST SIGNAL: {z_col} z>{thresh:.1f}")
        print(f"{'='*70}")

        yearly_res = eval_by_period(df, z_col, thresh, "momentum", yearly_periods)
        for pname, pr in sorted(yearly_res.items()):
            if pr["n"] < 20:
                print(f"  {pname}: n={pr['n']} (too few)")
                continue
            flag = "✅" if pr["avg_bps"] > 0 else "  "
            print(f"{flag} {pname}: avg={pr['avg_bps']:+.1f}bps wr={pr['wr']:.1f}% n={pr['n']}")

    # =====================================================================
    # Monthly IC of raw LS ratio vs 4h returns
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  MONTHLY IC: LS ratio (raw) vs fwd_ret_4h")
    print(f"{'='*70}")

    df["_month"] = df["dt"].dt.to_period("M")
    monthly_ics = []
    for m, grp in df.groupby("_month"):
        valid = grp[["top_trader_ls_ratio_accounts", "fwd_ret_4h"]].dropna()
        if len(valid) < 100:
            continue
        ic = valid["top_trader_ls_ratio_accounts"].corr(valid["fwd_ret_4h"])
        monthly_ics.append((str(m), ic, len(valid)))

    pos_months = sum(1 for _, ic, _ in monthly_ics if ic > 0)
    neg_months = sum(1 for _, ic, _ in monthly_ics if ic <= 0)

    for m, ic, n in monthly_ics:
        flag = "✅" if ic > 0.02 else "  "
        print(f"  {flag} {m}: IC={ic:+.4f} (n={n})")

    print(f"\n  Positive IC months: {pos_months}/{len(monthly_ics)} "
          f"({100*pos_months/len(monthly_ics):.0f}%)")
    avg_ic = np.mean([ic for _, ic, _ in monthly_ics])
    print(f"  Average monthly IC: {avg_ic:+.4f}")

    # =====================================================================
    # Non-overlapping trade simulation for best signal
    # =====================================================================
    if best_results:
        z_col, thresh, _ = best_results[0]
        print(f"\n{'='*70}")
        print(f"  NON-OVERLAPPING TRADE SIM — {z_col} z>{thresh:.1f}")
        print(f"  (Enter trade, hold 4h, then wait for next signal)")
        print(f"{'='*70}")

        z = df[z_col].values
        fwd = df["fwd_ret_4h"].values
        dt = df["dt"].values
        close = df["close"].values

        trades = []
        i = 0
        while i < len(df) - FWD_HORIZON:
            if not np.isfinite(z[i]) or not np.isfinite(fwd[i]):
                i += 1
                continue

            if z[i] > thresh:
                # Long
                ret = fwd[i] - FEE_BPS
                trades.append({
                    "dt": dt[i], "dir": "LONG", "z": z[i],
                    "entry_price": close[i], "ret_bps": ret,
                })
                i += FWD_HORIZON  # skip holding period
            elif z[i] < -thresh:
                # Short
                ret = -fwd[i] - FEE_BPS
                trades.append({
                    "dt": dt[i], "dir": "SHORT", "z": z[i],
                    "entry_price": close[i], "ret_bps": ret,
                })
                i += FWD_HORIZON
            else:
                i += 1

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df["cum_pnl"] = trades_df["ret_bps"].cumsum()

            n_trades = len(trades_df)
            avg_ret = trades_df["ret_bps"].mean()
            wr = (trades_df["ret_bps"] > 0).mean() * 100
            total_pnl = trades_df["ret_bps"].sum()

            # Max drawdown
            peak = 0
            max_dd = 0
            for v in trades_df["cum_pnl"]:
                if v > peak:
                    peak = v
                dd = peak - v
                if dd > max_dd:
                    max_dd = dd

            # Trades per day
            days = (trades_df["dt"].iloc[-1] - trades_df["dt"].iloc[0]) / np.timedelta64(1, "D")
            trades_per_day = n_trades / max(days, 1)

            print(f"\n  Total trades: {n_trades} over {days:.0f} days ({trades_per_day:.1f}/day)")
            print(f"  Avg return: {avg_ret:+.1f} bps")
            print(f"  Win rate: {wr:.1f}%")
            print(f"  Total PnL: {total_pnl:+.0f} bps")
            print(f"  Max drawdown: {max_dd:.0f} bps")

            # Long vs short breakdown
            longs = trades_df[trades_df["dir"] == "LONG"]
            shorts = trades_df[trades_df["dir"] == "SHORT"]
            print(f"\n  Longs:  n={len(longs)} avg={longs['ret_bps'].mean():+.1f}bps "
                  f"wr={(longs['ret_bps']>0).mean()*100:.1f}%")
            print(f"  Shorts: n={len(shorts)} avg={shorts['ret_bps'].mean():+.1f}bps "
                  f"wr={(shorts['ret_bps']>0).mean()*100:.1f}%")

            # Yearly breakdown
            trades_df["year"] = pd.to_datetime(trades_df["dt"]).dt.year
            print(f"\n  Yearly breakdown (non-overlapping):")
            for yr, ygrp in trades_df.groupby("year"):
                yr_avg = ygrp["ret_bps"].mean()
                yr_wr = (ygrp["ret_bps"] > 0).mean() * 100
                yr_pnl = ygrp["ret_bps"].sum()
                flag = "✅" if yr_avg > 0 else "  "
                print(f"  {flag} {yr}: n={len(ygrp)} avg={yr_avg:+.1f}bps "
                      f"wr={yr_wr:.1f}% pnl={yr_pnl:+.0f}bps")

            # Quarterly breakdown
            trades_df["quarter"] = pd.to_datetime(trades_df["dt"]).dt.to_period("Q")
            print(f"\n  Quarterly breakdown (non-overlapping):")
            n_pos_q = 0
            n_q = 0
            for q, qgrp in trades_df.groupby("quarter"):
                q_avg = qgrp["ret_bps"].mean()
                q_wr = (qgrp["ret_bps"] > 0).mean() * 100
                q_pnl = qgrp["ret_bps"].sum()
                n_q += 1
                if q_avg > 0:
                    n_pos_q += 1
                flag = "✅" if q_avg > 0 else "  "
                print(f"  {flag} {q}: n={len(qgrp)} avg={q_avg:+.1f}bps "
                      f"wr={q_wr:.1f}% pnl={q_pnl:+.0f}bps")
            print(f"\n  Positive quarters: {n_pos_q}/{n_q} ({100*n_pos_q/n_q:.0f}%)")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
