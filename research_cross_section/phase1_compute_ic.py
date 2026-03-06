"""
Phase 1 - Step 2: Cross-sectional IC analysis.

Loads all per-symbol signal parquets built by phase1_build_signals.py,
constructs wide panels (timestamp x symbol), and computes:

  - Mean IC, IC std, ICIR, t-stat for every (signal, forward_horizon) pair
  - Rolling 90-day IC to check stability over time
  - Signal autocorrelation (proxy for turnover — high autocorr = low turnover)
  - Fee-adjusted expected return assuming full-universe L/S decile portfolio

Output:
  results/ic_summary.csv       — main IC table
  results/ic_rolling.csv       — rolling 90d mean IC per (signal, fwd)
  results/ic_autocorr.csv      — signal rank autocorrelation at 1h,4h,8h,24h
  results/decile_spreads.csv   — top/bottom decile return spreads

Printed to stdout: formatted IC table with pass/fail vs fee hurdles.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

SIGNALS_DIR = "/home/ubuntu/Projects/skytrade6/research_cross_section/signals"
RESULTS_DIR = "/home/ubuntu/Projects/skytrade6/research_cross_section/results"

# Signals to evaluate
SIGNAL_COLS = {
    "mom_1h":   "Momentum 1h",
    "mom_2h":   "Momentum 2h",
    "mom_4h":   "Momentum 4h",
    "mom_8h":   "Momentum 8h",
    "mom_24h":  "Momentum 24h",
    "mom_48h":  "Momentum 48h",
    "funding":  "Funding carry",
    "prem_z":   "Premium z-score",
    "oi_div":   "OI-price divergence",
    "ls_z":     "L/S ratio z-score",
}

FWD_COLS = ["fwd_1h", "fwd_4h", "fwd_8h", "fwd_24h"]

# Fee hurdles (round-trip bps, one side entry one side exit)
FEE_MAKER = 8    # bps round-trip maker
FEE_TAKER = 20   # bps round-trip taker
FEE_MIXED = 14   # bps (maker entry, taker exit)

# Rolling IC window (hours)
ROLLING_WINDOW = 90 * 24


def load_panels(signals_dir):
    """Load all parquet files and build wide panels indexed by UTC timestamp."""
    files = sorted(glob.glob(os.path.join(signals_dir, "*.parquet")))
    print(f"Loading {len(files)} signal files...")

    all_cols = list(SIGNAL_COLS.keys()) + FWD_COLS + ["close"]
    panels = {c: {} for c in all_cols}

    for i, fpath in enumerate(files):
        symbol = os.path.basename(fpath).replace(".parquet", "")
        try:
            df = pd.read_parquet(fpath, columns=[c for c in all_cols if c != "close"] + ["close"])
            for col in all_cols:
                if col in df.columns:
                    panels[col][symbol] = df[col]
        except Exception as e:
            print(f"  WARNING: could not load {symbol}: {e}")
        if (i + 1) % 30 == 0 or i == len(files) - 1:
            print(f"  loaded {i+1}/{len(files)}")

    # Convert to DataFrames
    print("Building panels...")
    result = {}
    for col in all_cols:
        if panels[col]:
            result[col] = pd.DataFrame(panels[col])
            result[col].index = pd.to_datetime(result[col].index, utc=True)
            result[col] = result[col].sort_index()

    n_symbols = len(panels["close"]) if "close" in panels else 0
    n_bars = len(result.get("close", pd.DataFrame()))
    print(f"Panels built: {n_symbols} symbols x {n_bars} timestamps")
    print()
    return result


def cross_sectional_ic(sig_panel, fwd_panel):
    """
    Compute per-timestamp cross-sectional Spearman IC between signal and forward return.
    Returns a pd.Series of IC values indexed by timestamp.
    Uses rank-then-pearson for speed.
    """
    # Align on common timestamps and symbols
    common_ts = sig_panel.index.intersection(fwd_panel.index)
    common_syms = sig_panel.columns.intersection(fwd_panel.columns)
    S = sig_panel.loc[common_ts, common_syms]
    F = fwd_panel.loc[common_ts, common_syms]

    # Require at least 10 valid pairs per bar
    valid_mask = S.notna() & F.notna()
    n_valid = valid_mask.sum(axis=1)
    good = n_valid >= 10

    S = S.loc[good]
    F = F.loc[good]

    if S.empty:
        return pd.Series(dtype=float)

    # Rank cross-sectionally (axis=1)
    S_r = S.rank(axis=1)
    F_r = F.rank(axis=1)

    # Demean
    S_dm = S_r.sub(S_r.mean(axis=1), axis=0)
    F_dm = F_r.sub(F_r.mean(axis=1), axis=0)

    num = (S_dm * F_dm).sum(axis=1)
    denom = np.sqrt((S_dm ** 2).sum(axis=1) * (F_dm ** 2).sum(axis=1))
    ic = (num / denom.replace(0, np.nan)).dropna()
    return ic


def decile_spread(sig_panel, fwd_panel, n_decile=10):
    """
    At each bar: rank coins into deciles by signal, compute top-decile minus
    bottom-decile mean forward return. Returns a Series of spreads.
    """
    common_ts = sig_panel.index.intersection(fwd_panel.index)
    common_syms = sig_panel.columns.intersection(fwd_panel.columns)
    S = sig_panel.loc[common_ts, common_syms]
    F = fwd_panel.loc[common_ts, common_syms]
    valid = S.notna() & F.notna()
    n_valid = valid.sum(axis=1)
    good = n_valid >= 20

    spreads = []
    idx = []
    for ts in S.loc[good].index:
        sv = S.loc[ts].dropna()
        fv = F.loc[ts, sv.index].dropna()
        sv = sv.loc[fv.index]
        if len(sv) < 20:
            continue
        cutoff_lo = sv.quantile(1 / n_decile)
        cutoff_hi = sv.quantile((n_decile - 1) / n_decile)
        top = fv[sv >= cutoff_hi].mean()
        bot = fv[sv <= cutoff_lo].mean()
        spreads.append(top - bot)
        idx.append(ts)

    return pd.Series(spreads, index=idx)


def signal_autocorr(sig_panel, lags=[1, 4, 8, 24]):
    """
    Compute cross-sectional rank autocorrelation at given lags.
    High autocorr at lag 1h = slow signal = low turnover.
    """
    results = {}
    ranks = sig_panel.rank(axis=1)
    for lag in lags:
        shifted = ranks.shift(lag)
        # Per-row rank correlation (Pearson of ranks = Spearman of original)
        r_dm = ranks.sub(ranks.mean(axis=1), axis=0)
        s_dm = shifted.sub(shifted.mean(axis=1), axis=0)
        num = (r_dm * s_dm).sum(axis=1)
        denom = np.sqrt((r_dm ** 2).sum(axis=1) * (s_dm ** 2).sum(axis=1))
        ac = (num / denom.replace(0, np.nan)).dropna()
        results[f"lag_{lag}h"] = ac.mean()
    return results


def fmt_ic(val):
    if abs(val) >= 0.05:
        return f"\033[1;32m{val:+.4f}\033[0m"   # bold green
    elif abs(val) >= 0.02:
        return f"\033[33m{val:+.4f}\033[0m"      # yellow
    else:
        return f"\033[2m{val:+.4f}\033[0m"        # dim


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    panels = load_panels(SIGNALS_DIR)

    summary_rows = []
    rolling_rows = []
    autocorr_rows = []
    decile_rows = []

    print("=" * 90)
    print(f"{'Signal':<22} {'Fwd':>6}  {'IC mean':>8}  {'IC std':>8}  {'ICIR':>7}  {'t-stat':>7}  {'N bars':>7}  {'Maker?':>7}")
    print("=" * 90)

    for sig_col, sig_label in SIGNAL_COLS.items():
        if sig_col not in panels:
            print(f"  {sig_label}: no data, skipping")
            continue

        sig_panel = panels[sig_col]

        # Autocorrelation (once per signal)
        ac = signal_autocorr(sig_panel)
        ac["signal"] = sig_col
        autocorr_rows.append(ac)

        for fwd_col in FWD_COLS:
            if fwd_col not in panels:
                continue
            fwd_panel = panels[fwd_col]

            ic_series = cross_sectional_ic(sig_panel, fwd_panel)
            if ic_series.empty:
                continue

            ic_mean = ic_series.mean()
            ic_std  = ic_series.std()
            icir    = ic_mean / ic_std if ic_std > 0 else 0
            n       = len(ic_series)
            t_stat  = icir * np.sqrt(n)
            p_val   = 2 * (1 - stats.norm.cdf(abs(t_stat)))

            # Decile spread (bps)
            spread_series = decile_spread(sig_panel, fwd_panel)
            spread_mean_bps = spread_series.mean() * 10000
            spread_std_bps  = spread_series.std() * 10000

            # Fee assessment
            hurdle = FEE_MAKER  # most achievable
            maker_ok = abs(spread_mean_bps) > hurdle
            taker_ok = abs(spread_mean_bps) > FEE_TAKER

            marker = ""
            if taker_ok:   marker = "TAKER OK"
            elif maker_ok: marker = "maker ok"

            print(
                f"  {sig_label:<22} {fwd_col:>6}  {fmt_ic(ic_mean):>18}  "
                f"{ic_std:>8.4f}  {icir:>7.3f}  {t_stat:>7.2f}  {n:>7d}  {marker:>8}"
            )

            summary_rows.append({
                "signal": sig_col,
                "signal_label": sig_label,
                "fwd": fwd_col,
                "ic_mean": round(ic_mean, 5),
                "ic_std": round(ic_std, 5),
                "icir": round(icir, 4),
                "t_stat": round(t_stat, 3),
                "p_val": round(p_val, 5),
                "n_bars": n,
                "spread_mean_bps": round(spread_mean_bps, 2),
                "spread_std_bps": round(spread_std_bps, 2),
                "maker_ok": maker_ok,
                "taker_ok": taker_ok,
            })

            # Decile spread rows
            for ts, val in spread_series.items():
                decile_rows.append({"timestamp": ts, "signal": sig_col, "fwd": fwd_col, "spread_bps": val * 10000})

            # Rolling IC
            rolling_ic = ic_series.rolling(ROLLING_WINDOW, min_periods=30 * 24).mean()
            for ts, val in rolling_ic.items():
                rolling_rows.append({"timestamp": ts, "signal": sig_col, "fwd": fwd_col, "rolling_ic": val})

        print()

    print("=" * 90)

    # --- Save results ---
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "ic_summary.csv"), index=False)

    if rolling_rows:
        df_rolling = pd.DataFrame(rolling_rows)
        df_rolling.to_csv(os.path.join(RESULTS_DIR, "ic_rolling.csv"), index=False)

    if autocorr_rows:
        df_autocorr = pd.DataFrame(autocorr_rows).set_index("signal")
        df_autocorr.to_csv(os.path.join(RESULTS_DIR, "ic_autocorr.csv"))

    if decile_rows:
        df_decile = pd.DataFrame(decile_rows)
        df_decile.to_csv(os.path.join(RESULTS_DIR, "decile_spreads.csv"), index=False)

    # --- Pretty summary ---
    print()
    print("TOP SIGNALS BY |ICIR| (all horizons):")
    if not df_summary.empty:
        top = df_summary.reindex(df_summary["icir"].abs().sort_values(ascending=False).index).head(15)
        for _, row in top.iterrows():
            flag = " << TAKER OK" if row["taker_ok"] else (" < maker ok" if row["maker_ok"] else "")
            print(f"  {row['signal_label']:<22} {row['fwd']:>6}  IC={row['ic_mean']:+.4f}  ICIR={row['icir']:+.3f}  "
                  f"spread={row['spread_mean_bps']:+.1f}bps{flag}")

    print()
    print("SIGNAL AUTOCORRELATION (rank persistence = proxy for low turnover):")
    if autocorr_rows:
        df_ac = pd.DataFrame(autocorr_rows).set_index("signal")
        for sig, row in df_ac.iterrows():
            label = SIGNAL_COLS.get(sig, sig)
            lags = "  ".join([f"lag{k.split('_')[1]}={v:.3f}" for k, v in row.items()])
            print(f"  {label:<22}  {lags}")

    print()
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
