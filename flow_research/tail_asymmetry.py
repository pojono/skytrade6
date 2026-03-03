#!/usr/bin/env python3
"""
Stage A: Tail-Asymmetry Analysis on REG_OI_FUND

Tests whether REG_OI_FUND increases the probability of extreme moves
(tails of the return distribution), normalized by ATR.

Compares regime vs volatility-matched baseline.

Output:
  flow_research/output/regime/tail_uplift.csv
  flow_research/output/regime/tail_summary.csv
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIME_PARQUET = OUTPUT_DIR / "regime_dataset.parquet"

SYMBOLS = [
    "1000BONKUSDT", "ARBUSDT", "APTUSDT", "ATOMUSDT",
    "AIXBTUSDT", "1000RATSUSDT", "ARCUSDT", "1000TURBOUSDT",
]

START_TS = pd.Timestamp("2026-01-01", tz="UTC")
END_TS = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

COOLDOWN_MIN = 60
ATR_PERIOD = 15

HORIZONS = [15, 30, 60]  # minutes
TAIL_THRESHOLDS = [1, 2, 3, 4]  # ATR units

# Vol-matched baseline: sample N_BASELINE_MULT times more baseline points,
# then match by ATR quintile and hour-of-day
N_BASELINE_MULT = 10
N_BOOTSTRAP = 2000

SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mark_1m(symbol: str) -> pd.DataFrame:
    sym_dir = DATA_DIR / symbol
    files = sorted(sym_dir.glob("*_mark_price_kline_1m.csv"))
    if not files:
        files = sorted(
            f for f in sym_dir.glob("*_kline_1m.csv")
            if "mark_price" not in f.name and "premium_index" not in f.name
        )
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["startTime"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df = df[(df["ts"] >= START_TS) & (df["ts"] <= END_TS)].reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close"]]


def compute_atr_series(df_1m: pd.DataFrame) -> pd.Series:
    prev_close = df_1m["close"].shift(1)
    tr = pd.concat([
        df_1m["high"] - df_1m["low"],
        (df_1m["high"] - prev_close).abs(),
        (df_1m["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()


# ---------------------------------------------------------------------------
# Compute forward metrics for a set of sample points
# ---------------------------------------------------------------------------


def compute_forward_metrics(
    bars_1m: pd.DataFrame,
    sample_indices: np.ndarray,
    sample_atrs: np.ndarray,
    horizons: list[int],
) -> dict[int, pd.DataFrame]:
    """For each horizon H, compute ret_H_atr, MFE_H_atr, MAE_H_atr."""
    n_bars = len(bars_1m)
    closes = bars_1m["close"].values
    highs = bars_1m["high"].values
    lows = bars_1m["low"].values
    n = len(sample_indices)

    results = {}

    for H in horizons:
        ret_atr = np.full(n, np.nan)
        mfe_atr = np.full(n, np.nan)
        mae_atr = np.full(n, np.nan)

        for i in range(n):
            idx = sample_indices[i]
            atr = sample_atrs[i]
            end_idx = idx + H

            if end_idx >= n_bars or atr <= 0:
                continue

            p0 = closes[idx]
            p_end = closes[end_idx]

            # Slice bars in [idx+1, end_idx] for MFE/MAE
            h_slice = highs[idx + 1 : end_idx + 1]
            l_slice = lows[idx + 1 : end_idx + 1]

            if len(h_slice) == 0:
                continue

            ret_atr[i] = (p_end - p0) / atr
            mfe_atr[i] = (h_slice.max() - p0) / atr
            mae_atr[i] = (l_slice.min() - p0) / atr

        results[H] = pd.DataFrame({
            "ret_atr": ret_atr,
            "mfe_atr": mfe_atr,
            "mae_atr": mae_atr,
            "abs_ret_atr": np.abs(ret_atr),
            "range_atr": mfe_atr - mae_atr,  # total excursion in ATR
        })

    return results


# ---------------------------------------------------------------------------
# Volatility-matched baseline sampling
# ---------------------------------------------------------------------------


def sample_vol_matched_baseline(
    regime_indices: np.ndarray,
    regime_atrs: np.ndarray,
    regime_hours: np.ndarray,
    all_5m_indices: np.ndarray,
    all_5m_atrs: np.ndarray,
    all_5m_hours: np.ndarray,
    all_5m_regime: np.ndarray,
    rng: np.random.Generator,
    mult: int = N_BASELINE_MULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample baseline points matched by ATR quintile and hour-of-day."""
    # Non-regime mask
    non_regime_mask = all_5m_regime == 0
    non_regime_idx = np.where(non_regime_mask)[0]

    if len(non_regime_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    nr_bar_indices = all_5m_indices[non_regime_idx]
    nr_atrs = all_5m_atrs[non_regime_idx]
    nr_hours = all_5m_hours[non_regime_idx]

    # Compute ATR quintile edges from regime points
    atr_q = np.quantile(regime_atrs, [0.2, 0.4, 0.6, 0.8])

    def atr_bin(a):
        return np.searchsorted(atr_q, a)

    regime_bins = np.array([atr_bin(a) for a in regime_atrs])
    nr_bins = np.array([atr_bin(a) for a in nr_atrs])

    # For each regime point, sample mult baseline points from same (atr_bin, hour)
    sampled_bar_indices = []
    sampled_atrs = []

    for i in range(len(regime_indices)):
        ab = regime_bins[i]
        hr = regime_hours[i]
        candidates = np.where((nr_bins == ab) & (nr_hours == hr))[0]
        if len(candidates) == 0:
            # Relax hour constraint
            candidates = np.where(nr_bins == ab)[0]
        if len(candidates) == 0:
            continue
        k = min(mult, len(candidates))
        chosen = rng.choice(candidates, size=k, replace=False)
        sampled_bar_indices.extend(nr_bar_indices[chosen])
        sampled_atrs.extend(nr_atrs[chosen])

    return np.array(sampled_bar_indices, dtype=int), np.array(sampled_atrs)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def binomial_test_uplift(n_regime, k_regime, n_base, k_base):
    """One-sided binomial test: is regime tail prob > baseline tail prob?"""
    p_base = k_base / n_base if n_base > 0 else 0
    if n_regime == 0 or p_base <= 0:
        return 1.0
    # Exact binomial test
    pval = sp_stats.binomtest(k_regime, n_regime, p_base, alternative="greater").pvalue
    return pval


def bootstrap_uplift_ci(
    regime_vals: np.ndarray,
    base_vals: np.ndarray,
    threshold: float,
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator = None,
) -> tuple[float, float]:
    """Bootstrap 90% CI for the uplift ratio."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    n_r = len(regime_vals)
    n_b = len(base_vals)
    if n_r == 0 or n_b == 0:
        return np.nan, np.nan

    uplifts = np.empty(n_boot)
    for b in range(n_boot):
        r_boot = rng.choice(regime_vals, size=n_r, replace=True)
        b_boot = rng.choice(base_vals, size=n_b, replace=True)
        p_r = (np.abs(r_boot) > threshold).mean()
        p_b = (np.abs(b_boot) > threshold).mean()
        uplifts[b] = p_r / p_b if p_b > 0 else np.nan

    valid = uplifts[~np.isnan(uplifts)]
    if len(valid) < 100:
        return np.nan, np.nan
    return float(np.percentile(valid, 5)), float(np.percentile(valid, 95))


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    q = np.empty(n)
    q[sorted_idx] = sorted_p * n / (np.arange(1, n + 1))
    # Enforce monotonicity from the back
    q_out = np.minimum.accumulate(q[::-1])[::-1]
    return np.clip(q_out, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    t0 = time.monotonic()

    print("=" * 80)
    print("STAGE A: TAIL-ASYMMETRY ANALYSIS on REG_OI_FUND")
    print(f"Horizons: {HORIZONS}m   Thresholds: {TAIL_THRESHOLDS} ATR")
    print("=" * 80)

    regime_df = pd.read_parquet(REGIME_PARQUET)
    regime_df["ts"] = pd.to_datetime(regime_df["ts"], utc=True)

    rng = np.random.default_rng(SEED)

    all_uplift_rows = []
    all_summary_rows = []

    for sym_i, sym in enumerate(SYMBOLS, 1):
        sym_t0 = time.monotonic()
        print(f"\n{'─'*70}")
        print(f"[{sym_i}/{len(SYMBOLS)}] {sym}")
        print(f"{'─'*70}")

        bars_1m = load_mark_1m(sym)
        if len(bars_1m) == 0:
            print("  SKIP: no 1m data")
            continue
        print(f"  1m bars: {len(bars_1m):,}")

        atr_series = compute_atr_series(bars_1m)
        ts_1m = bars_1m["ts"].values.astype("int64")

        # Get all 5m points from regime dataset
        sym_regime = regime_df[regime_df["symbol"] == sym].sort_values("ts").reset_index(drop=True)
        sym_regime_ts = sym_regime["ts"].values
        sym_regime_flag = sym_regime["REG_OI_FUND"].values.astype(int)

        # Map 5m points → 1m bar indices
        all_5m_bar_idx = np.searchsorted(ts_1m, sym_regime_ts.astype("int64"), side="right") - 1
        valid_mask = (all_5m_bar_idx >= 0) & (all_5m_bar_idx < len(bars_1m))
        all_5m_bar_idx = all_5m_bar_idx[valid_mask]
        sym_regime_flag = sym_regime_flag[valid_mask]
        sym_regime_ts = sym_regime_ts[valid_mask]

        all_5m_atrs = np.array([atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else 0
                                for i in all_5m_bar_idx])
        all_5m_hours = np.array([pd.Timestamp(t, tz="UTC").hour for t in sym_regime_ts])

        # Extract regime signals with cooldown
        regime_mask = sym_regime_flag == 1
        regime_ts_raw = sym_regime_ts[regime_mask]
        regime_bar_raw = all_5m_bar_idx[regime_mask]
        regime_atr_raw = all_5m_atrs[regime_mask]

        cooldown_ns = int(COOLDOWN_MIN * 60 * 1e9)
        regime_indices = []
        regime_atrs = []
        regime_hours = []
        last_ts = -cooldown_ns * 2

        for j in range(len(regime_ts_raw)):
            t_ns = int(regime_ts_raw[j].astype("int64"))
            if t_ns - last_ts < cooldown_ns:
                continue
            atr_val = regime_atr_raw[j]
            if atr_val <= 0 or np.isnan(atr_val):
                continue
            regime_indices.append(regime_bar_raw[j])
            regime_atrs.append(atr_val)
            regime_hours.append(pd.Timestamp(regime_ts_raw[j], tz="UTC").hour)
            last_ts = t_ns

        regime_indices = np.array(regime_indices, dtype=int)
        regime_atrs = np.array(regime_atrs)
        regime_hours = np.array(regime_hours)
        n_regime = len(regime_indices)
        print(f"  Regime signals (cooldown): {n_regime}")

        if n_regime < 5:
            print("  SKIP: too few signals")
            continue

        # Vol-matched baseline
        base_indices, base_atrs = sample_vol_matched_baseline(
            regime_indices, regime_atrs, regime_hours,
            all_5m_bar_idx, all_5m_atrs, all_5m_hours,
            sym_regime_flag, rng,
        )
        n_base = len(base_indices)
        print(f"  Baseline samples (vol-matched): {n_base}")

        # Compute forward metrics
        regime_metrics = compute_forward_metrics(bars_1m, regime_indices, regime_atrs, HORIZONS)
        base_metrics = compute_forward_metrics(bars_1m, base_indices, base_atrs, HORIZONS)

        # Tail analysis per horizon × threshold
        best_uplift = 0
        best_h = None
        best_k = None

        for H in HORIZONS:
            rm = regime_metrics[H].dropna(subset=["ret_atr"])
            bm = base_metrics[H].dropna(subset=["ret_atr"])

            r_abs = rm["abs_ret_atr"].values
            b_abs = bm["abs_ret_atr"].values
            r_mfe = rm["mfe_atr"].values
            b_mfe = bm["mfe_atr"].values
            r_ret = rm["ret_atr"].values
            b_ret = bm["ret_atr"].values

            nr = len(r_abs)
            nb = len(b_abs)

            for k_thr in TAIL_THRESHOLDS:
                # |ret| > k
                k_r = int((r_abs > k_thr).sum())
                k_b = int((b_abs > k_thr).sum())
                p_r = k_r / nr if nr > 0 else 0
                p_b = k_b / nb if nb > 0 else 0
                uplift = p_r / p_b if p_b > 0 else (999 if p_r > 0 else 1.0)

                pval = binomial_test_uplift(nr, k_r, nb, k_b)
                ci_lo, ci_hi = bootstrap_uplift_ci(r_abs, b_abs, k_thr, rng=rng)

                # MFE > k
                k_r_mfe = int((r_mfe > k_thr).sum())
                k_b_mfe = int((b_mfe > k_thr).sum())
                p_r_mfe = k_r_mfe / nr if nr > 0 else 0
                p_b_mfe = k_b_mfe / nb if nb > 0 else 0
                uplift_mfe = p_r_mfe / p_b_mfe if p_b_mfe > 0 else (999 if p_r_mfe > 0 else 1.0)

                # Directional split (§8)
                k_r_up = int((r_ret > k_thr).sum())
                k_r_dn = int((r_ret < -k_thr).sum())
                k_b_up = int((b_ret > k_thr).sum())
                k_b_dn = int((b_ret < -k_thr).sum())
                p_r_up = k_r_up / nr if nr > 0 else 0
                p_r_dn = k_r_dn / nr if nr > 0 else 0
                p_b_up = k_b_up / nb if nb > 0 else 0
                p_b_dn = k_b_dn / nb if nb > 0 else 0
                uplift_up = p_r_up / p_b_up if p_b_up > 0 else (999 if p_r_up > 0 else 1.0)
                uplift_dn = p_r_dn / p_b_dn if p_b_dn > 0 else (999 if p_r_dn > 0 else 1.0)

                # Conditional expectation (§9)
                r_tail = r_abs[r_abs > k_thr]
                b_tail = b_abs[b_abs > k_thr]
                cond_exp_regime = float(r_tail.mean()) if len(r_tail) > 0 else np.nan
                cond_exp_base = float(b_tail.mean()) if len(b_tail) > 0 else np.nan

                row = {
                    "symbol": sym,
                    "horizon": H,
                    "threshold": k_thr,
                    "n_regime": nr,
                    "n_base": nb,
                    "regime_prob": p_r,
                    "base_prob": p_b,
                    "uplift": uplift,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_value": pval,
                    "regime_prob_mfe": p_r_mfe,
                    "base_prob_mfe": p_b_mfe,
                    "uplift_mfe": uplift_mfe,
                    "regime_prob_up": p_r_up,
                    "regime_prob_dn": p_r_dn,
                    "base_prob_up": p_b_up,
                    "base_prob_dn": p_b_dn,
                    "uplift_up": uplift_up,
                    "uplift_dn": uplift_dn,
                    "cond_exp_regime": cond_exp_regime,
                    "cond_exp_base": cond_exp_base,
                }
                all_uplift_rows.append(row)

                if uplift > best_uplift and p_r >= 0.05:
                    best_uplift = uplift
                    best_h = H
                    best_k = k_thr

        # Per-symbol summary
        verdict = "PASS" if best_uplift >= 2.0 else ("WEAK" if best_uplift >= 1.3 else "FAIL")
        all_summary_rows.append({
            "symbol": sym,
            "n_regime": n_regime,
            "n_base": n_base,
            "best_uplift": best_uplift,
            "best_horizon": best_h,
            "best_threshold": best_k,
            "verdict": verdict,
        })

        print(f"  Best uplift: {best_uplift:.2f}x at H={best_h}m, k={best_k} ATR → {verdict}")
        print(f"  Done in {time.monotonic()-sym_t0:.1f}s")

    if not all_uplift_rows:
        print("\nNo data. Exiting.")
        return

    # Apply FDR
    uplift_df = pd.DataFrame(all_uplift_rows)
    pvals = uplift_df["p_value"].values
    uplift_df["q_fdr"] = benjamini_hochberg(pvals)
    uplift_df.to_csv(OUTPUT_DIR / "tail_uplift.csv", index=False)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "tail_summary.csv", index=False)

    # ===================================================================
    # Print results
    # ===================================================================
    print(f"\n{'='*80}")
    print("TAIL UPLIFT RESULTS (|ret| > k ATR)")
    print("=" * 80)

    print(f"\n{'Sym':<16} {'H':>3} {'k':>2} | {'RegP':>6} {'BasP':>6} {'Uplft':>6} "
          f"{'CI':>12} {'p':>7} {'q':>7} | {'MFE_up':>6} | "
          f"{'P_up':>5} {'P_dn':>5} {'U_up':>5} {'U_dn':>5} | {'CE_r':>5} {'CE_b':>5}")
    print("-" * 120)

    for _, r in uplift_df.iterrows():
        sig = ""
        if r["q_fdr"] < 0.01:
            sig = "***"
        elif r["q_fdr"] < 0.05:
            sig = "**"
        elif r["q_fdr"] < 0.10:
            sig = "*"

        ci_str = f"[{r['ci_lo']:.1f},{r['ci_hi']:.1f}]" if not np.isnan(r["ci_lo"]) else "  [nan]  "

        print(f"{r['symbol']:<16} {r['horizon']:>3} {r['threshold']:>2} | "
              f"{r['regime_prob']:>6.1%} {r['base_prob']:>6.1%} {r['uplift']:>6.2f} "
              f"{ci_str:>12} {r['p_value']:>7.4f} {r['q_fdr']:>7.4f}{sig:>3} | "
              f"{r['uplift_mfe']:>6.2f} | "
              f"{r['regime_prob_up']:>5.1%} {r['regime_prob_dn']:>5.1%} "
              f"{r['uplift_up']:>5.1f} {r['uplift_dn']:>5.1f} | "
              f"{r['cond_exp_regime']:>5.2f} {r['cond_exp_base']:>5.2f}")

    # ===================================================================
    # Cross-symbol summary
    # ===================================================================
    print(f"\n{'='*80}")
    print("CROSS-SYMBOL SUMMARY")
    print("=" * 80)

    # Average uplift by horizon × threshold
    print(f"\n  Average uplift (|ret| > k ATR) across all symbols:")
    print(f"  {'H':>4} {'k':>3} {'AvgUplft':>8} {'AvgRegP':>8} {'AvgBasP':>8} {'N_sig':>6}")
    for H in HORIZONS:
        for k in TAIL_THRESHOLDS:
            sub = uplift_df[(uplift_df["horizon"] == H) & (uplift_df["threshold"] == k)]
            if len(sub) == 0:
                continue
            # Cap extreme uplifts for averaging
            u = sub["uplift"].clip(upper=20)
            print(f"  {H:>4} {k:>3} {u.mean():>8.2f} {sub['regime_prob'].mean():>8.1%} "
                  f"{sub['base_prob'].mean():>8.1%} {sub['n_regime'].mean():>6.0f}")

    # Significant results
    sig_df = uplift_df[(uplift_df["q_fdr"] < 0.10) & (uplift_df["regime_prob"] >= 0.05) & (uplift_df["uplift"] >= 2.0)]
    print(f"\n  Significant tail uplift (q<0.10, P_regime≥5%, uplift≥2x): {len(sig_df)} hits")
    if len(sig_df) > 0:
        for _, r in sig_df.sort_values("uplift", ascending=False).head(20).iterrows():
            print(f"    {r['symbol']:<16} H={r['horizon']:>2}m k={r['threshold']} ATR  "
                  f"regime={r['regime_prob']:.1%} base={r['base_prob']:.1%}  "
                  f"uplift={r['uplift']:.2f}x  q={r['q_fdr']:.4f}")

    # Directional asymmetry
    print(f"\n{'='*80}")
    print("DIRECTIONAL TAIL ASYMMETRY (§8)")
    print("=" * 80)

    for sym in sorted(uplift_df["symbol"].unique()):
        sym_df = uplift_df[(uplift_df["symbol"] == sym) & (uplift_df["horizon"] == 60)]
        print(f"\n  {sym} (H=60m):")
        print(f"  {'k':>3} | {'P(ret>k)':>8} {'P(ret<-k)':>9} {'U_up':>5} {'U_dn':>5} | {'Bias':>10}")
        for _, r in sym_df.iterrows():
            bias = "UP" if r["uplift_up"] > r["uplift_dn"] * 1.3 else \
                   ("DOWN" if r["uplift_dn"] > r["uplift_up"] * 1.3 else "NEUTRAL")
            print(f"  {r['threshold']:>3} | {r['regime_prob_up']:>8.1%} {r['regime_prob_dn']:>9.1%} "
                  f"{r['uplift_up']:>5.1f} {r['uplift_dn']:>5.1f} | {bias:>10}")

    # Per-symbol verdict
    print(f"\n{'='*80}")
    print("PER-SYMBOL VERDICT")
    print("=" * 80)

    for _, r in summary_df.iterrows():
        print(f"  {r['symbol']:<18} {r['verdict']:<6} best_uplift={r['best_uplift']:.2f}x "
              f"H={r['best_horizon']}m k={r['best_threshold']}ATR  N={r['n_regime']}")

    # Overall verdict
    n_pass = (summary_df["verdict"] == "PASS").sum()
    n_weak = (summary_df["verdict"] == "WEAK").sum()
    n_fail = (summary_df["verdict"] == "FAIL").sum()

    print(f"\n  PASS={n_pass}, WEAK={n_weak}, FAIL={n_fail}")

    if n_pass >= 4:
        print("\n  → SCENARIO 3: Strong uplift on majority of coins → tail-capture viable")
    elif n_pass + n_weak >= 4:
        print("\n  → SCENARIO 2: Moderate uplift → possible with additional filters")
    else:
        print("\n  → SCENARIO 1: No tail uplift → REG_OI_FUND does not amplify extremes")

    elapsed = time.monotonic() - t0
    print(f"\n{'='*80}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Outputs: {OUTPUT_DIR}/tail_uplift.csv, tail_summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
