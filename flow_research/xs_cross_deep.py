#!/usr/bin/env python3
"""
Deep dive into the 3 genuine cross-sectional signals:
  1. Market Compression (Spec 7) — median rv_6h <= P20
  2. Network Fragmentation (Spec 8) — correlation density drops before big moves
  3. Entropy Concentration (Spec 4) — low/high entropy effects

Analysis dimensions:
  A. Threshold sweep — find optimal cutoffs
  B. Monthly stability — does the signal work every month?
  C. Direction skew — does the signal predict UP or DOWN?
  D. Time-to-move — how quickly does the big move happen?
  E. Combined signals — multi-condition filters
  F. Per-coin uplift — which coins benefit most?
  G. Shuffle validation — are combined signals genuine?

Reuses market_features.csv and network_metrics.csv from xs_cross_sectional.py.
Reloads per-coin panels for fwd_ret and S07.
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
# Config (same as xs_cross_sectional.py)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "xs_cross"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2025-07-01", tz="UTC")
END = pd.Timestamp("2026-03-02 23:59:59", tz="UTC")
MIN_DAYS = 100

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")

RV_6H_WINDOW = 360
OI_Z_WINDOW = 7 * 24 * 60
FR_Z_WINDOW = 7 * 24 * 60
RV_2H_WINDOW = 120
ATR_14H = 14 * 60
ATR_K = 3.0
HORIZONS_MIN = {"12h": 720, "24h": 1440}
S07_RV6H_PCTL = 0.20
S07_OI_Z_THRESH = 1.5

SEED = 42
N_SHUFFLE = 1000

# ---------------------------------------------------------------------------
# Data loading (import from xs_cross_sectional)
# ---------------------------------------------------------------------------

# Import shared functions
from xs_cross_sectional import (
    discover_symbols, load_symbol, build_sym_1m,
    compute_coin_features, compute_big_move_targets,
)


def build_panels(symbols: list[str]) -> dict:
    """Build panels — same as xs_cross_sectional but also keeps fwd_ret direction."""
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    grid_5m = pd.date_range(START, END, freq="5min", tz="UTC")

    ret_5m = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    ret_1h = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    rv_6h = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    oi_z = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    s07 = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
    big_move = {}
    fwd_ret_panels = {}
    for h_label in HORIZONS_MIN:
        big_move[f"big_A_{h_label}"] = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)
        fwd_ret_panels[f"fwd_ret_{h_label}"] = pd.DataFrame(index=grid_5m, columns=symbols, dtype=float)

    t0 = time.monotonic()
    for i, sym in enumerate(symbols, 1):
        t1 = time.monotonic()
        raw = load_symbol(sym)
        df = build_sym_1m(sym, raw, grid_1m)
        df = compute_coin_features(df)
        df = compute_big_move_targets(df)

        df_5m = df.reindex(grid_5m)
        ret_5m[sym] = df_5m["ret_5m"].values
        ret_1h[sym] = df_5m["ret_1h"].values
        rv_6h[sym] = df_5m["rv_6h"].values
        oi_z[sym] = df_5m["oi_z"].values
        s07[sym] = df_5m["S07"].values

        for h_label in HORIZONS_MIN:
            big_move[f"big_A_{h_label}"][sym] = df_5m[f"big_A_{h_label}"].values
            fwd_ret_panels[f"fwd_ret_{h_label}"][sym] = df_5m[f"fwd_ret_{h_label}"].values

        elapsed = time.monotonic() - t0
        dt = time.monotonic() - t1
        eta = (len(symbols) - i) * elapsed / i
        print(f"  [{i}/{len(symbols)}] {sym:<20s} {dt:.1f}s  (total {elapsed:.0f}s, ETA {eta:.0f}s)")

    return {
        "ret_5m": ret_5m, "ret_1h": ret_1h, "rv_6h": rv_6h, "oi_z": oi_z,
        "s07": s07, "big_move": big_move, "fwd_ret": fwd_ret_panels,
    }


# ---------------------------------------------------------------------------
# A. Threshold Sweep
# ---------------------------------------------------------------------------

def sweep_compression_thresholds(mf: pd.DataFrame, panels: dict):
    """Sweep rv_6h percentile thresholds for market compression."""
    print("\n" + "=" * 70)
    print("A1. COMPRESSION THRESHOLD SWEEP")
    print("=" * 70)

    for target_key, bm_panel in panels["big_move"].items():
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                    ("test", mf.index >= TEST_START)]:
            data = pd.DataFrame({
                "rv_pctl": mf.loc[pmask, "market_rv_6h_pctl"],
                "bm_any": bm_any[pmask],
            }).dropna()
            if len(data) < 100:
                continue

            baseline = data["bm_any"].mean()
            print(f"\n  {target_key} — {period_name} (baseline={baseline:.4f})")
            print(f"  {'Threshold':<12} {'N':>7} {'BM Rate':>9} {'Uplift':>8} {'Freq%':>7}")
            print(f"  {'-'*48}")

            for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                mask = data["rv_pctl"] <= thresh
                n = mask.sum()
                if n < 20:
                    continue
                rate = data.loc[mask, "bm_any"].mean()
                uplift = rate / baseline if baseline > 0 else np.nan
                freq = n / len(data) * 100
                print(f"  P{thresh*100:<9.0f} {n:>7} {rate:>9.4f} {uplift:>8.2f}x {freq:>6.1f}%")


def sweep_entropy_thresholds(mf: pd.DataFrame, panels: dict):
    """Sweep entropy percentile thresholds."""
    print("\n" + "=" * 70)
    print("A2. ENTROPY THRESHOLD SWEEP")
    print("=" * 70)

    for target_key, bm_panel in panels["big_move"].items():
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                    ("test", mf.index >= TEST_START)]:
            data = pd.DataFrame({
                "ent_pctl": mf.loc[pmask, "norm_entropy_1h_pctl"],
                "bm_any": bm_any[pmask],
            }).dropna()
            if len(data) < 100:
                continue

            baseline = data["bm_any"].mean()
            print(f"\n  {target_key} — {period_name} (baseline={baseline:.4f})")

            # Low entropy (trigger)
            print(f"\n  LOW ENTROPY (trigger):")
            print(f"  {'Threshold':<12} {'N':>7} {'BM Rate':>9} {'Uplift':>8}")
            print(f"  {'-'*40}")
            for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                mask = data["ent_pctl"] <= thresh
                n = mask.sum()
                if n < 20:
                    continue
                rate = data.loc[mask, "bm_any"].mean()
                uplift = rate / baseline if baseline > 0 else np.nan
                print(f"  ≤P{thresh*100:<8.0f} {n:>7} {rate:>9.4f} {uplift:>8.2f}x")

            # High entropy (suppressor)
            print(f"\n  HIGH ENTROPY (suppressor):")
            print(f"  {'Threshold':<12} {'N':>7} {'BM Rate':>9} {'Uplift':>8}")
            print(f"  {'-'*40}")
            for thresh in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                mask = data["ent_pctl"] >= thresh
                n = mask.sum()
                if n < 20:
                    continue
                rate = data.loc[mask, "bm_any"].mean()
                uplift = rate / baseline if baseline > 0 else np.nan
                print(f"  ≥P{thresh*100:<8.0f} {n:>7} {rate:>9.4f} {uplift:>8.2f}x")


def sweep_network_thresholds(net_df: pd.DataFrame, panels: dict):
    """Sweep network density thresholds."""
    print("\n" + "=" * 70)
    print("A3. NETWORK DENSITY THRESHOLD SWEEP")
    print("=" * 70)

    # Compute density percentile on the full 5m grid (net_df already reindexed)
    PCTL_WIN = 30 * 24 * 12
    min_p = PCTL_WIN // 4
    density_pctl = net_df["density"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)

    for target_key, bm_panel in panels["big_move"].items():
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan

        idx = bm_any.index  # 5m grid

        for period_name, pmask in [("train", idx <= TRAIN_END),
                                    ("test", idx >= TEST_START)]:
            data = pd.DataFrame({
                "density_pctl": density_pctl.reindex(idx)[pmask],
                "density": net_df["density"].reindex(idx)[pmask],
                "bm_any": bm_any[pmask],
            }).dropna()
            if len(data) < 100:
                continue

            baseline = data["bm_any"].mean()
            print(f"\n  {target_key} — {period_name} (baseline={baseline:.4f})")
            print(f"  LOW DENSITY (fragmented → more big moves):")
            print(f"  {'Threshold':<12} {'N':>7} {'BM Rate':>9} {'Uplift':>8} {'Freq%':>7}")
            print(f"  {'-'*48}")
            for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                mask = data["density_pctl"] <= thresh
                n = mask.sum()
                if n < 20:
                    continue
                rate = data.loc[mask, "bm_any"].mean()
                uplift = rate / baseline if baseline > 0 else np.nan
                freq = n / len(data) * 100
                print(f"  ≤P{thresh*100:<8.0f} {n:>7} {rate:>9.4f} {uplift:>8.2f}x {freq:>6.1f}%")


# ---------------------------------------------------------------------------
# B. Monthly Stability
# ---------------------------------------------------------------------------

def monthly_stability(mf: pd.DataFrame, net_df: pd.DataFrame, panels: dict):
    """Check if signals are stable month-over-month."""
    print("\n" + "=" * 70)
    print("B. MONTHLY STABILITY")
    print("=" * 70)

    target_key = "big_A_24h"
    bm_panel = panels["big_move"][target_key]
    bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
    bm_any[bm_panel.isna().all(axis=1)] = np.nan

    # Add month column
    months = pd.Series(mf.index.to_period("M"), index=mf.index)
    unique_months = sorted(months.dropna().unique())

    # Signal definitions
    compressed = mf["market_rv_6h_pctl"] <= 0.20
    PCTL_WIN = 30 * 24 * 12
    min_p = PCTL_WIN // 4
    net_aligned = net_df
    density_pctl = net_aligned["density"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)
    low_density = density_pctl <= 0.25
    low_entropy = mf["norm_entropy_1h_pctl"] <= 0.20
    high_entropy = mf["norm_entropy_1h_pctl"] >= 0.80

    signals = {
        "compressed": compressed,
        "low_density": low_density,
        "low_entropy": low_entropy,
        "high_entropy": high_entropy,
    }

    for sig_name, sig_mask in signals.items():
        print(f"\n  Signal: {sig_name} ({target_key})")
        print(f"  {'Month':<10} {'N_sig':>6} {'BM_rate':>8} {'Baseline':>9} {'Uplift':>8}")
        print(f"  {'-'*46}")
        uplifts = []
        for m in unique_months:
            m_mask = months == m
            m_data = pd.DataFrame({
                "sig": sig_mask[m_mask].astype(float),
                "bm": bm_any[m_mask],
            }).dropna()
            if len(m_data) < 50:
                continue
            baseline_m = m_data["bm"].mean()
            sig_m = m_data["sig"] == 1
            n_sig = sig_m.sum()
            if n_sig < 10:
                print(f"  {str(m):<10} {n_sig:>6}  (too few)")
                continue
            rate_m = m_data.loc[sig_m, "bm"].mean()
            uplift_m = rate_m / baseline_m if baseline_m > 0 else np.nan
            uplifts.append(uplift_m)
            marker = " ✓" if uplift_m >= 1.0 else " ✗"
            print(f"  {str(m):<10} {n_sig:>6} {rate_m:>8.4f} {baseline_m:>9.4f} {uplift_m:>8.2f}x{marker}")

        if uplifts:
            n_pos = sum(1 for u in uplifts if u >= 1.0)
            print(f"  → {n_pos}/{len(uplifts)} months with uplift ≥ 1.0x  "
                  f"(mean={np.mean(uplifts):.2f}x, median={np.median(uplifts):.2f}x)")


# ---------------------------------------------------------------------------
# C. Direction Skew
# ---------------------------------------------------------------------------

def direction_skew(mf: pd.DataFrame, net_df: pd.DataFrame, panels: dict):
    """Does the signal predict UP or DOWN big moves?"""
    print("\n" + "=" * 70)
    print("C. DIRECTION SKEW ANALYSIS")
    print("=" * 70)

    PCTL_WIN = 30 * 24 * 12
    min_p = PCTL_WIN // 4
    net_aligned = net_df
    density_pctl = net_aligned["density"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)

    signal_defs = {
        "compressed": mf["market_rv_6h_pctl"] <= 0.20,
        "low_density": density_pctl <= 0.25,
        "low_entropy": mf["norm_entropy_1h_pctl"] <= 0.20,
        "high_entropy": mf["norm_entropy_1h_pctl"] >= 0.80,
    }

    for h_label in ["12h", "24h"]:
        fwd_ret_panel = panels["fwd_ret"][f"fwd_ret_{h_label}"]

        # For each signal, check direction of big moves
        for sig_name, sig_mask in signal_defs.items():
            for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                        ("test", mf.index >= TEST_START)]:
                sig_active = sig_mask[pmask].fillna(False)
                if sig_active.sum() < 20:
                    continue

                # Get forward returns for all coins when signal is active
                fwd_active = fwd_ret_panel[pmask][sig_active]
                fwd_baseline = fwd_ret_panel[pmask][~sig_active]

                # Flatten to get all coin-timestamp forward returns
                active_vals = fwd_active.values.flatten()
                active_vals = active_vals[~np.isnan(active_vals)]
                baseline_vals = fwd_baseline.values.flatten()
                baseline_vals = baseline_vals[~np.isnan(baseline_vals)]

                if len(active_vals) < 100:
                    continue

                # Direction stats
                pct_up_active = (active_vals > 0).mean()
                pct_up_baseline = (baseline_vals > 0).mean()

                # Big up/down rates (using 2% as threshold)
                big_up_active = (active_vals > 0.02).mean()
                big_down_active = (active_vals < -0.02).mean()
                big_up_baseline = (baseline_vals > 0.02).mean()
                big_down_baseline = (baseline_vals < -0.02).mean()

                up_uplift = big_up_active / big_up_baseline if big_up_baseline > 0 else np.nan
                down_uplift = big_down_active / big_down_baseline if big_down_baseline > 0 else np.nan

                # Mean return
                mean_active = active_vals.mean() * 10000  # bps
                mean_baseline = baseline_vals.mean() * 10000

                # Skewness
                skew_active = sp_stats.skew(active_vals)
                skew_baseline = sp_stats.skew(baseline_vals)

                print(f"\n  {sig_name} | {h_label} | {period_name} (n_sig={sig_active.sum()}, "
                      f"n_ret={len(active_vals)})")
                print(f"    %up:       signal={pct_up_active:.1%}  baseline={pct_up_baseline:.1%}")
                print(f"    Big up:    signal={big_up_active:.4f}  baseline={big_up_baseline:.4f}  "
                      f"uplift={up_uplift:.2f}x")
                print(f"    Big down:  signal={big_down_active:.4f}  baseline={big_down_baseline:.4f}  "
                      f"uplift={down_uplift:.2f}x")
                print(f"    Mean ret:  signal={mean_active:+.1f}bp  baseline={mean_baseline:+.1f}bp")
                print(f"    Skew:      signal={skew_active:.3f}  baseline={skew_baseline:.3f}")


# ---------------------------------------------------------------------------
# D. Time-to-Move
# ---------------------------------------------------------------------------

def time_to_move(mf: pd.DataFrame, net_df: pd.DataFrame, panels: dict):
    """When does the big move happen after signal fires?"""
    print("\n" + "=" * 70)
    print("D. TIME-TO-MOVE ANALYSIS")
    print("=" * 70)

    PCTL_WIN = 30 * 24 * 12
    min_p = PCTL_WIN // 4
    net_aligned = net_df
    density_pctl = net_aligned["density"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)

    signal_defs = {
        "compressed": mf["market_rv_6h_pctl"] <= 0.20,
        "low_density": density_pctl <= 0.25,
        "combined": (mf["market_rv_6h_pctl"] <= 0.20) & (density_pctl <= 0.25),
    }

    # Use 24h target; check when the 3×ATR threshold is first breached
    # We need to check at multiple sub-horizons
    sub_horizons = [60, 120, 180, 360, 720, 1440]  # 1h, 2h, 3h, 6h, 12h, 24h in minutes
    sub_labels = ["1h", "2h", "3h", "6h", "12h", "24h"]

    # We need the underlying 1m close and ATR data to compute this
    # For efficiency, use the fwd_ret panels at different horizons
    # We only have 12h and 24h, so let's compute breach times from the per-coin data

    # Simplified: for each signal timestamp, check if big_A_12h fires (breach within 12h)
    # and big_A_24h fires (breach within 24h). The difference tells us timing.
    bm_12h = panels["big_move"]["big_A_12h"]
    bm_24h = panels["big_move"]["big_A_24h"]

    for sig_name, sig_mask in signal_defs.items():
        for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                    ("test", mf.index >= TEST_START)]:
            sig_active = sig_mask[pmask].fillna(False)
            n_sig = sig_active.sum()
            if n_sig < 20:
                continue

            # For each coin at signal timestamps, check big move at 12h and 24h
            bm_12h_active = bm_12h[pmask][sig_active]
            bm_24h_active = bm_24h[pmask][sig_active]

            # Rate at each horizon
            rate_12h = bm_12h_active.values.flatten()
            rate_12h = rate_12h[~np.isnan(rate_12h)].mean()
            rate_24h = bm_24h_active.values.flatten()
            rate_24h = rate_24h[~np.isnan(rate_24h)].mean()

            # Also compute "any coin" rates
            any_12h = (bm_12h_active.sum(axis=1) > 0).mean()
            any_24h = (bm_24h_active.sum(axis=1) > 0).mean()

            # Baseline
            sig_inactive = ~sig_active
            base_any_12h = (bm_12h[pmask][sig_inactive].sum(axis=1) > 0).astype(float)
            base_any_12h = base_any_12h.dropna().mean()
            base_any_24h = (bm_24h[pmask][sig_inactive].sum(axis=1) > 0).astype(float)
            base_any_24h = base_any_24h.dropna().mean()

            print(f"\n  {sig_name} | {period_name} (n_sig={n_sig})")
            print(f"    Any coin BM within 12h: {any_12h:.4f} (baseline {base_any_12h:.4f}, "
                  f"uplift {any_12h/base_any_12h:.2f}x)")
            print(f"    Any coin BM within 24h: {any_24h:.4f} (baseline {base_any_24h:.4f}, "
                  f"uplift {any_24h/base_any_24h:.2f}x)")
            print(f"    Per-coin rate 12h: {rate_12h:.4f}")
            print(f"    Per-coin rate 24h: {rate_24h:.4f}")
            if rate_24h > 0 and rate_12h > 0:
                print(f"    % of 24h moves that happen within 12h: {rate_12h/rate_24h:.1%}")


# ---------------------------------------------------------------------------
# E. Combined Signals
# ---------------------------------------------------------------------------

def combined_signals(mf: pd.DataFrame, net_df: pd.DataFrame, panels: dict):
    """Test multi-condition filters for maximum uplift."""
    print("\n" + "=" * 70)
    print("E. COMBINED SIGNAL ANALYSIS")
    print("=" * 70)

    PCTL_WIN = 30 * 24 * 12
    min_p = PCTL_WIN // 4
    net_aligned = net_df
    density_pctl = net_aligned["density"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)

    # Build signal components
    compressed = mf["market_rv_6h_pctl"] <= 0.20
    v_compressed = mf["market_rv_6h_pctl"] <= 0.10  # very compressed
    low_density = density_pctl <= 0.25
    v_low_density = density_pctl <= 0.15
    low_entropy = mf["norm_entropy_1h_pctl"] <= 0.20
    not_high_entropy = mf["norm_entropy_1h_pctl"] <= 0.80

    # S07 (any coin)
    s07_any = (panels["s07"].sum(axis=1) > 0).astype(float)

    combos = {
        "baseline (no filter)":                  pd.Series(True, index=mf.index),
        "compressed_P20":                         compressed,
        "compressed_P10":                         v_compressed,
        "low_density_P25":                        low_density,
        "low_density_P15":                        v_low_density,
        "low_entropy_P20":                        low_entropy,
        "NOT_high_entropy":                       not_high_entropy,
        "compressed + low_density":               compressed & low_density,
        "compressed + NOT_high_ent":              compressed & not_high_entropy,
        "compressed + low_entropy":               compressed & low_entropy,
        "low_density + low_entropy":              low_density & low_entropy,
        "compressed + low_density + low_ent":     compressed & low_density & low_entropy,
        "compressed + low_density + NOT_hi_ent":  compressed & low_density & not_high_entropy,
        "S07_any":                                s07_any == 1,
        "S07 + compressed":                       (s07_any == 1) & compressed,
        "S07 + compressed + low_density":         (s07_any == 1) & compressed & low_density,
        "S07 + compressed + NOT_hi_ent":          (s07_any == 1) & compressed & not_high_entropy,
        "S07 + all_3":                            (s07_any == 1) & compressed & low_density & not_high_entropy,
    }

    rng = np.random.default_rng(SEED)

    for target_key, bm_panel in panels["big_move"].items():
        bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
        bm_any[bm_panel.isna().all(axis=1)] = np.nan
        bm_rate = bm_panel.mean(axis=1)  # per-coin rate

        for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                    ("test", mf.index >= TEST_START)]:
            print(f"\n  {target_key} — {period_name}")
            print(f"  {'Combo':<42} {'N':>7} {'Any%':>7} {'PerCoin%':>9} {'AnyUplift':>10} {'Freq%':>6}")
            print(f"  {'-'*85}")

            baseline_any = bm_any[pmask].dropna().mean()
            baseline_rate = bm_rate[pmask].dropna().mean()

            for combo_name, combo_mask in combos.items():
                active = combo_mask[pmask].fillna(False)
                n = active.sum()
                if n < 10:
                    continue

                any_rate = bm_any[pmask][active].dropna().mean()
                coin_rate = bm_rate[pmask][active].dropna().mean()
                any_uplift = any_rate / baseline_any if baseline_any > 0 else np.nan
                freq = n / pmask.sum() * 100

                print(f"  {combo_name:<42} {n:>7} {any_rate:>6.1%} {coin_rate:>8.4f} "
                      f"{any_uplift:>10.2f}x {freq:>5.1f}%")

        # Shuffle validation for best combos (test period only)
        print(f"\n  SHUFFLE VALIDATION ({target_key}, test):")
        test_mask = mf.index >= TEST_START
        bm_test = bm_any[test_mask].dropna()
        day_labels = bm_test.index.date

        for combo_name in ["compressed + low_density", "S07 + compressed + NOT_hi_ent",
                           "S07 + all_3", "compressed + low_density + NOT_hi_ent"]:
            if combo_name not in combos:
                continue
            active = combos[combo_name][test_mask].reindex(bm_test.index).fillna(False)
            n_active = active.sum()
            if n_active < 10:
                print(f"    {combo_name}: too few events ({n_active})")
                continue

            observed = bm_test[active].mean()

            # Shuffle within days
            unique_days = np.unique(day_labels)
            day_idx = {d: np.where(day_labels == d)[0] for d in unique_days}
            day_ns = {d: active.values[day_idx[d]].sum() for d in unique_days}

            count_ge = 0
            bm_arr = bm_test.values
            for _ in range(N_SHUFFLE):
                perm_sum = 0.0
                perm_n = 0
                for d in unique_days:
                    nd = day_ns[d]
                    if nd == 0:
                        continue
                    idx = day_idx[d]
                    perm_idx = rng.choice(idx, size=int(nd), replace=False)
                    perm_sum += bm_arr[perm_idx].sum()
                    perm_n += nd
                if perm_n > 0 and perm_sum / perm_n >= observed:
                    count_ge += 1

            pval = (count_ge + 1) / (N_SHUFFLE + 1)
            print(f"    {combo_name}: observed={observed:.4f}, p={pval:.4f} "
                  f"(n={n_active}, {'✓ GENUINE' if pval < 0.05 else '✗ noise'})")


# ---------------------------------------------------------------------------
# F. Per-Coin Uplift
# ---------------------------------------------------------------------------

def per_coin_uplift(mf: pd.DataFrame, net_df: pd.DataFrame, panels: dict):
    """Which coins benefit most from combined signals?"""
    print("\n" + "=" * 70)
    print("F. PER-COIN UPLIFT (top signals)")
    print("=" * 70)

    PCTL_WIN = 30 * 24 * 12
    min_p = PCTL_WIN // 4
    net_aligned = net_df
    density_pctl = net_aligned["density"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)

    compressed = mf["market_rv_6h_pctl"] <= 0.20
    low_density = density_pctl <= 0.25
    not_high_ent = mf["norm_entropy_1h_pctl"] <= 0.80

    # Best combined signal
    combined = compressed & low_density & not_high_ent

    target_key = "big_A_24h"
    bm_panel = panels["big_move"][target_key]

    test_mask = mf.index >= TEST_START
    active_test = combined[test_mask].fillna(False)

    symbols = bm_panel.columns.tolist()
    coin_results = []

    for sym in symbols:
        bm_sym = bm_panel.loc[test_mask, sym].dropna()
        if len(bm_sym) < 100:
            continue
        baseline = bm_sym.mean()
        active_idx = active_test.reindex(bm_sym.index).fillna(False)
        n_active = active_idx.sum()
        if n_active < 5:
            continue
        rate = bm_sym[active_idx].mean()
        uplift = rate / baseline if baseline > 0 else np.nan
        coin_results.append({
            "symbol": sym, "baseline": baseline, "signal_rate": rate,
            "uplift": uplift, "n_signal": n_active, "n_total": len(bm_sym),
        })

    cr = pd.DataFrame(coin_results).sort_values("uplift", ascending=False)
    cr.to_csv(OUTPUT_DIR / "per_coin_uplift.csv", index=False)

    print(f"\n  Combined signal: compressed + low_density + NOT_high_entropy")
    print(f"  Target: {target_key}, Period: test")
    print(f"  {'Symbol':<20} {'Baseline':>9} {'Signal':>8} {'Uplift':>8} {'N_sig':>6}")
    print(f"  {'-'*55}")
    for _, row in cr.head(20).iterrows():
        marker = " ★" if row["uplift"] >= 1.5 else ""
        print(f"  {row['symbol']:<20} {row['baseline']:>9.4f} {row['signal_rate']:>8.4f} "
              f"{row['uplift']:>8.2f}x {row['n_signal']:>6.0f}{marker}")

    print(f"\n  Summary: {len(cr)} coins, "
          f"{(cr['uplift'] >= 1.0).sum()}/{len(cr)} with uplift ≥ 1.0x, "
          f"mean uplift = {cr['uplift'].mean():.2f}x, "
          f"median uplift = {cr['uplift'].median():.2f}x")


# ---------------------------------------------------------------------------
# G. Heatmap: Signal Frequency vs Market Conditions
# ---------------------------------------------------------------------------

def signal_interaction_matrix(mf: pd.DataFrame, net_df: pd.DataFrame, panels: dict):
    """2D heatmap: compression × density → big move uplift."""
    print("\n" + "=" * 70)
    print("G. SIGNAL INTERACTION MATRIX (compression × density)")
    print("=" * 70)

    PCTL_WIN = 30 * 24 * 12
    min_p = PCTL_WIN // 4
    net_aligned = net_df
    density_pctl = net_aligned["density"].rolling(PCTL_WIN, min_periods=min_p).rank(pct=True)

    target_key = "big_A_24h"
    bm_panel = panels["big_move"][target_key]
    bm_any = (bm_panel.sum(axis=1) > 0).astype(float)
    bm_any[bm_panel.isna().all(axis=1)] = np.nan

    rv_bins = [0, 0.10, 0.20, 0.30, 0.50, 1.0]
    rv_labels = ["RV≤P10", "P10-20", "P20-30", "P30-50", "P50+"]
    dens_bins = [0, 0.15, 0.25, 0.35, 0.50, 1.0]
    dens_labels = ["D≤P15", "P15-25", "P25-35", "P35-50", "D>P50"]

    for period_name, pmask in [("train", mf.index <= TRAIN_END),
                                ("test", mf.index >= TEST_START)]:
        data = pd.DataFrame({
            "rv_bin": pd.cut(mf.loc[pmask, "market_rv_6h_pctl"], bins=rv_bins, labels=rv_labels, include_lowest=True),
            "dens_bin": pd.cut(density_pctl[pmask], bins=dens_bins, labels=dens_labels, include_lowest=True),
            "bm_any": bm_any[pmask],
        }).dropna()

        if len(data) < 100:
            continue

        baseline = data["bm_any"].mean()
        print(f"\n  {target_key} — {period_name} (baseline={baseline:.4f})")
        print(f"\n  {'':>12}", end="")
        for dl in dens_labels:
            print(f" {dl:>8}", end="")
        print()

        for rl in rv_labels:
            print(f"  {rl:>10}", end="")
            for dl in dens_labels:
                mask = (data["rv_bin"] == rl) & (data["dens_bin"] == dl)
                n = mask.sum()
                if n < 10:
                    print(f" {'--':>8}", end="")
                else:
                    rate = data.loc[mask, "bm_any"].mean()
                    uplift = rate / baseline
                    print(f" {uplift:>7.2f}x", end="")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("DEEP DIVE: 3 Genuine Cross-Sectional Signals")
    print("=" * 70)

    # Load cached market features
    mf_path = OUTPUT_DIR / "market_features.csv"
    print(f"\nLoading cached market features from {mf_path}...")
    mf = pd.read_csv(mf_path, index_col=0, parse_dates=True)
    if mf.index.tz is None:
        mf.index = mf.index.tz_localize("UTC")
    print(f"  Loaded {len(mf)} rows, {len(mf.columns)} cols")

    # Load cached network metrics
    net_path = OUTPUT_DIR / "network_metrics.csv"
    print(f"Loading cached network metrics from {net_path}...")
    net_df = pd.read_csv(net_path, index_col=0, parse_dates=True)
    if net_df.index.tz is None:
        net_df.index = net_df.index.tz_localize("UTC")
    print(f"  Loaded {len(net_df)} rows")

    # Reindex network metrics to 5m grid (net_df is hourly, panels are 5m)
    net_df = net_df.reindex(mf.index).ffill()
    print(f"  Reindexed to 5m grid: {len(net_df)} rows")

    # Rebuild panels (need per-coin big move + fwd_ret)
    symbols = discover_symbols()
    print(f"\nRebuilding per-coin panels ({len(symbols)} symbols)...")
    panels = build_panels(symbols)

    # Run all analyses
    t0 = time.monotonic()

    sweep_compression_thresholds(mf, panels)
    sweep_entropy_thresholds(mf, panels)
    sweep_network_thresholds(net_df, panels)
    monthly_stability(mf, net_df, panels)
    direction_skew(mf, net_df, panels)
    time_to_move(mf, net_df, panels)
    combined_signals(mf, net_df, panels)
    per_coin_uplift(mf, net_df, panels)
    signal_interaction_matrix(mf, net_df, panels)

    total = time.monotonic() - t0
    print(f"\n{'='*70}")
    print(f"DEEP DIVE COMPLETE ({total:.0f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
