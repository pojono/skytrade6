#!/usr/bin/env python3
"""
XS-6 Deep Analysis — Time-to-move, Direction Skew, Sanity Tests

Runs on top of xs6_bigmove_uplift.py infrastructure.
Focuses on top states (S06/S07 compression, S01/S03 positioning).

Answers:
  Q1: Time-to-move distribution (when does the big move happen after signal?)
  Q2: Direction skew (P(ret>+T) vs P(ret<-T) within state)
  Q3: Sanity — shuffle test (permute states → PASS should vanish)
  Q4: Sanity — OI leakage trap (shift OI +1 bar forward → uplift should spike if OI matters)
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(line_buffering=True)

# Reuse xs6 infrastructure
from xs6_bigmove_uplift import (
    DATA_DIR, OUTPUT_DIR, START, END, MIN_DAYS,
    SIGNAL_STEP_MIN, HORIZONS, ATR_K_VALUES, RAW_BP_THRESHOLDS,
    TRAIN_END, TEST_START, PURGE_HOURS, SEED,
    STATE_DEFS,
    discover_symbols, load_symbol, build_sym_1m, compute_features,
    compute_targets, compute_states,
)

OUTPUT_DEEP = OUTPUT_DIR / "deep"
OUTPUT_DEEP.mkdir(parents=True, exist_ok=True)

# Focus states for deep analysis
FOCUS_STATES = ["S06_compress_vol", "S07_compress_oi", "S01_fund_high",
                "S03_oi_surge", "S09_stall_fund", "S10_thin_move"]

# Time buckets for time-to-move analysis (in minutes from signal)
TIME_BUCKETS = [
    ("0-1h", 0, 60),
    ("1-3h", 60, 180),
    ("3-6h", 180, 360),
    ("6-12h", 360, 720),
    ("12-24h", 720, 1440),
]

# ATR multiples for direction skew
SKEW_K = 3.0  # |ret| >= k * ATR_1h


# ---------------------------------------------------------------------------
# Q1: Time-to-move analysis
# ---------------------------------------------------------------------------

def time_to_move_analysis(df_5m: pd.DataFrame, df_1m: pd.DataFrame,
                          sym: str, state_name: str) -> list[dict]:
    """For each signal where state=1 and a big move eventually happens in 24h,
    find WHEN the move threshold is first breached."""
    close_1m = df_1m["close"]
    atr_1h = df_1m["atr_1h_raw"]
    invalid_1m = df_1m["is_invalid"]

    st = df_5m[state_name].values == 1.0
    state_times = df_5m.index[st]

    if len(state_times) == 0:
        return []

    # Use test period only
    purge_end = TEST_START + pd.Timedelta(hours=PURGE_HOURS)
    state_times = state_times[state_times >= purge_end]

    results = []
    for t in state_times:
        if t not in close_1m.index:
            continue
        c0 = close_1m.loc[t]
        atr_val = atr_1h.loc[t] if t in atr_1h.index else np.nan
        if pd.isna(c0) or pd.isna(atr_val) or atr_val <= 0:
            continue

        # ATR threshold as return
        atr_thresh = SKEW_K * atr_val / c0
        # Fixed bp thresholds
        bp_thresh_12h = RAW_BP_THRESHOLDS["12h"] / 10000.0
        bp_thresh_24h = RAW_BP_THRESHOLDS["24h"] / 10000.0

        # Look at 1m returns from t+1 to t+1440 (24h)
        fwd_window = close_1m.loc[t:].iloc[1:1441]  # next 1440 bars
        if len(fwd_window) < 60:
            continue

        fwd_ret = (fwd_window / c0) - 1.0

        # Find first breach for each definition
        first_atr = np.nan
        first_bp = np.nan
        first_atr_dir = 0  # +1 up, -1 down
        first_bp_dir = 0

        abs_ret = fwd_ret.abs()
        for i, (ts_i, ret_i) in enumerate(fwd_ret.items()):
            minutes_from_signal = i + 1
            if pd.isna(ret_i):
                continue
            if np.isnan(first_atr) and abs(ret_i) >= atr_thresh:
                first_atr = minutes_from_signal
                first_atr_dir = 1 if ret_i > 0 else -1
            if np.isnan(first_bp) and abs(ret_i) >= bp_thresh_24h:
                first_bp = minutes_from_signal
                first_bp_dir = 1 if ret_i > 0 else -1
            if not np.isnan(first_atr) and not np.isnan(first_bp):
                break

        # Max adverse excursion before first favorable move
        # (for bracket order analysis)
        max_up = fwd_ret.max()
        max_down = fwd_ret.min()
        ret_12h = fwd_ret.iloc[719] if len(fwd_ret) > 719 else np.nan
        ret_24h = fwd_ret.iloc[-1] if len(fwd_ret) >= 1440 else np.nan

        results.append({
            "symbol": sym,
            "state": state_name,
            "signal_time": t,
            "first_breach_atr_min": first_atr,
            "first_breach_atr_dir": first_atr_dir,
            "first_breach_bp_min": first_bp,
            "first_breach_bp_dir": first_bp_dir,
            "max_up_24h": max_up,
            "max_down_24h": max_down,
            "ret_12h": ret_12h,
            "ret_24h": ret_24h,
            "atr_thresh": atr_thresh,
        })

    return results


def summarize_time_to_move(ttm_rows: list[dict]) -> pd.DataFrame:
    """Aggregate time-to-move into bucket distribution."""
    if not ttm_rows:
        return pd.DataFrame()

    df = pd.DataFrame(ttm_rows)

    summaries = []
    for state in df["state"].unique():
        sd = df[df["state"] == state]

        # ATR-based breaches
        breached_atr = sd[~sd["first_breach_atr_min"].isna()]
        total = len(sd)

        row = {"state": state, "n_signals": total}

        # Breach rate
        row["breach_rate_atr"] = len(breached_atr) / total if total > 0 else 0

        # Time distribution (ATR)
        if len(breached_atr) > 0:
            t_arr = breached_atr["first_breach_atr_min"].values
            for label, lo, hi in TIME_BUCKETS:
                cnt = ((t_arr >= lo) & (t_arr < hi)).sum()
                row[f"pct_{label}"] = cnt / len(breached_atr)
            row["median_min_atr"] = np.median(t_arr)
            row["p25_min_atr"] = np.percentile(t_arr, 25)
            row["p75_min_atr"] = np.percentile(t_arr, 75)

        # Direction at breach
        if len(breached_atr) > 0:
            dirs = breached_atr["first_breach_atr_dir"].values
            row["pct_up_at_breach"] = (dirs == 1).mean()
            row["pct_down_at_breach"] = (dirs == -1).mean()

        summaries.append(row)

    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Q2: Direction skew
# ---------------------------------------------------------------------------

def direction_skew_analysis(df_5m: pd.DataFrame, state_name: str,
                            sym: str) -> list[dict]:
    """For each (state, horizon), compute P(ret>+T) vs P(ret<-T)."""
    purge_end = TEST_START + pd.Timedelta(hours=PURGE_HOURS)
    mask_test = np.array(df_5m.index >= purge_end)

    st = df_5m[state_name].values == 1.0
    results = []

    for h_label in HORIZONS:
        fwd_col = f"fwd_ret_{h_label}"
        if fwd_col not in df_5m.columns:
            continue

        atr_col = "atr_1h_raw" if "atr_1h_raw" in df_5m.columns else None

        fwd = df_5m[fwd_col].values
        valid = ~np.isnan(fwd) & mask_test

        # State subset
        state_valid = valid & st
        n_state = state_valid.sum()
        if n_state < 10:
            continue

        fwd_state = fwd[state_valid]

        # Fixed thresholds
        for bp_label, bp in [("100bp", 0.01), ("200bp", 0.02), ("300bp", 0.03)]:
            p_up = (fwd_state >= bp).mean()
            p_down = (fwd_state <= -bp).mean()
            p_big = (np.abs(fwd_state) >= bp).mean()

            # Baseline (all valid, not just state)
            fwd_all = fwd[valid]
            p_up_base = (fwd_all >= bp).mean()
            p_down_base = (fwd_all <= -bp).mean()

            results.append({
                "symbol": sym,
                "state": state_name,
                "horizon": h_label,
                "threshold": bp_label,
                "n_state": n_state,
                "p_up": p_up,
                "p_down": p_down,
                "p_big": p_big,
                "skew_ratio": p_up / p_down if p_down > 0.001 else np.nan,
                "p_up_base": p_up_base,
                "p_down_base": p_down_base,
                "uplift_up": p_up / max(p_up_base, 1e-6),
                "uplift_down": p_down / max(p_down_base, 1e-6),
            })

    return results


# ---------------------------------------------------------------------------
# Q3: Shuffle sanity test
# ---------------------------------------------------------------------------

def shuffle_sanity_test(df_5m: pd.DataFrame, state_name: str, target_col: str,
                        n_shuffles: int = 500, rng=None) -> dict:
    """Shuffle state labels within each day and compute uplift distribution.
    If the real uplift is genuine, shuffled uplifts should be ~1.0."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    purge_end = TEST_START + pd.Timedelta(hours=PURGE_HOURS)
    mask_test = np.array(df_5m.index >= purge_end)

    valid = (~df_5m[target_col].isna()).values & mask_test
    bm = df_5m[target_col].values
    st = df_5m[state_name].values
    days = (df_5m.index - pd.Timestamp("2020-01-01", tz="UTC")).days

    ns = int((st[valid] == 1.0).sum())
    if ns < 5:
        return {"state": state_name, "target": target_col,
                "real_uplift": np.nan, "shuffle_mean": np.nan, "shuffle_p95": np.nan}

    p0 = np.nanmean(bm[valid])
    ps_real = np.nanmean(bm[valid & (st == 1.0)])
    real_uplift = ps_real / max(p0, 1e-8)

    # Shuffle
    unique_days = np.unique(days[valid])
    day_indices = {d: np.where((days == d) & valid)[0] for d in unique_days}
    day_ns = {d: int((st[day_indices[d]] == 1.0).sum()) for d in unique_days}

    shuffled_uplifts = []
    for _ in range(n_shuffles):
        perm_sum = 0.0
        perm_n = 0
        for d in unique_days:
            nd = day_ns[d]
            if nd == 0:
                continue
            idx = day_indices[d]
            perm_idx = rng.choice(idx, size=nd, replace=False)
            perm_sum += bm[perm_idx].sum()
            perm_n += nd
        if perm_n > 0:
            ps_shuf = perm_sum / perm_n
            shuffled_uplifts.append(ps_shuf / max(p0, 1e-8))

    if not shuffled_uplifts:
        return {"state": state_name, "target": target_col,
                "real_uplift": real_uplift, "shuffle_mean": np.nan, "shuffle_p95": np.nan}

    arr = np.array(shuffled_uplifts)
    return {
        "state": state_name,
        "target": target_col,
        "real_uplift": real_uplift,
        "shuffle_mean": arr.mean(),
        "shuffle_std": arr.std(),
        "shuffle_p95": np.percentile(arr, 95),
        "shuffle_p99": np.percentile(arr, 99),
        "real_exceeds_p99": int(real_uplift > np.percentile(arr, 99)),
    }


# ---------------------------------------------------------------------------
# Q4: OI leakage trap
# ---------------------------------------------------------------------------

def oi_leakage_test(sym: str, raw: dict, grid_1m: pd.DatetimeIndex) -> list[dict]:
    """Shift OI +1 bar into the future (leak) and check if uplift spikes.
    If OI is genuinely driving the signal, leaking it should boost uplift."""

    # Normal run
    df_1m_normal = build_sym_1m(sym, raw, grid_1m)
    valid_pct = 1 - df_1m_normal["is_invalid"].mean()
    if valid_pct < 0.5:
        return []
    df_1m_normal = compute_features(df_1m_normal)
    grid_5m = grid_1m[::SIGNAL_STEP_MIN]
    df_5m_normal = df_1m_normal.loc[grid_5m].copy()
    df_5m_normal = compute_targets(df_5m_normal, df_1m_normal)
    df_5m_normal = compute_states(df_5m_normal)

    # Leaked run: shift OI forward by 1 bar (5min) — this is future information
    raw_leaked = dict(raw)
    oi_leaked = raw["oi"].copy()
    if len(oi_leaked) > 0:
        # Remove the +5min causal shift by shifting back 5min
        # This effectively gives us OI 5min early (leak)
        oi_leaked["ts"] = oi_leaked["ts"] - pd.Timedelta(minutes=5)
    raw_leaked["oi"] = oi_leaked

    df_1m_leak = build_sym_1m(sym, raw_leaked, grid_1m)
    df_1m_leak = compute_features(df_1m_leak)
    df_5m_leak = df_1m_leak.loc[grid_5m].copy()
    df_5m_leak = compute_targets(df_5m_leak, df_1m_leak)
    df_5m_leak = compute_states(df_5m_leak)

    purge_end = TEST_START + pd.Timedelta(hours=PURGE_HOURS)

    results = []
    for state_name in ["S03_oi_surge", "S07_compress_oi", "S08_stall_oi"]:
        for target_col in [f"big_A_k3.0_24h", f"big_B_24h"]:
            if target_col not in df_5m_normal.columns:
                continue

            for label, df_5m in [("normal", df_5m_normal), ("leaked", df_5m_leak)]:
                mask_test = df_5m.index >= purge_end
                valid = ~df_5m[target_col].isna() & mask_test
                bm = df_5m[target_col].values
                st = df_5m[state_name].values

                ns = int((st[valid] == 1.0).sum())
                if ns < 5:
                    p0 = pS = uplift = np.nan
                else:
                    p0 = np.nanmean(bm[valid])
                    pS = np.nanmean(bm[valid & (st == 1.0)])
                    uplift = pS / max(p0, 1e-8)

                results.append({
                    "symbol": sym,
                    "state": state_name,
                    "target": target_col,
                    "variant": label,
                    "nS_test": ns,
                    "p0_test": p0,
                    "pS_test": pS,
                    "uplift_test": uplift,
                })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.monotonic()
    print("=" * 70)
    print("XS-6 Deep Analysis — Time-to-move, Skew, Sanity")
    print("=" * 70)
    print()

    symbols = discover_symbols()
    grid_1m = pd.date_range(START, END, freq="1min", tz="UTC")
    print(f"Symbols: {len(symbols)}, Grid: {len(grid_1m)} 1m bars")
    print()

    rng = np.random.default_rng(SEED)

    all_ttm = []
    all_skew = []
    all_shuffle = []
    all_leakage = []

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}...")
        t0 = time.monotonic()

        raw = load_symbol(sym)
        df_1m = build_sym_1m(sym, raw, grid_1m)
        valid_pct = 1 - df_1m["is_invalid"].mean()
        if valid_pct < 0.5:
            print(f"  skipped ({valid_pct:.0%} valid)")
            continue

        df_1m = compute_features(df_1m)
        grid_5m = grid_1m[::SIGNAL_STEP_MIN]
        df_5m = df_1m.loc[grid_5m].copy()
        df_5m = compute_targets(df_5m, df_1m)
        df_5m = compute_states(df_5m)

        for state_name in FOCUS_STATES:
            # Q1: Time-to-move
            ttm = time_to_move_analysis(df_5m, df_1m, sym, state_name)
            all_ttm.extend(ttm)

            # Q2: Direction skew
            skew = direction_skew_analysis(df_5m, state_name, sym)
            all_skew.extend(skew)

            # Q3: Shuffle sanity (on key targets only)
            for tgt in ["big_A_k3.0_24h", "big_B_24h"]:
                if tgt in df_5m.columns:
                    shuf = shuffle_sanity_test(df_5m, state_name, tgt, n_shuffles=500, rng=rng)
                    shuf["symbol"] = sym
                    all_shuffle.append(shuf)

        # Q4: OI leakage (only on OI-dependent states)
        leak = oi_leakage_test(sym, raw, grid_1m)
        all_leakage.extend(leak)

        elapsed = time.monotonic() - t0
        print(f"  done in {elapsed:.1f}s")

    # --- Save and summarize ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Q1: Time-to-move
    if all_ttm:
        df_ttm = pd.DataFrame(all_ttm)
        df_ttm.to_csv(OUTPUT_DEEP / "time_to_move_raw.csv", index=False)
        ttm_summary = summarize_time_to_move(all_ttm)
        ttm_summary.to_csv(OUTPUT_DEEP / "time_to_move_summary.csv", index=False)
        print("\n## Q1: Time-to-move (ATR-based breach, test period)")
        print(ttm_summary.to_string(index=False, float_format="%.3f"))
    else:
        print("\n## Q1: No time-to-move data")

    # Q2: Direction skew
    if all_skew:
        df_skew = pd.DataFrame(all_skew)
        df_skew.to_csv(OUTPUT_DEEP / "direction_skew.csv", index=False)

        # Aggregate by state
        print("\n## Q2: Direction skew (test period, 200bp threshold)")
        skew_200 = df_skew[df_skew["threshold"] == "200bp"]
        if len(skew_200) > 0:
            agg = skew_200.groupby(["state", "horizon"]).agg(
                mean_p_up=("p_up", "mean"),
                mean_p_down=("p_down", "mean"),
                mean_skew_ratio=("skew_ratio", "mean"),
                mean_uplift_up=("uplift_up", "mean"),
                mean_uplift_down=("uplift_down", "mean"),
                n_syms=("symbol", "nunique"),
            ).reset_index()
            print(agg.to_string(index=False, float_format="%.3f"))
    else:
        print("\n## Q2: No direction skew data")

    # Q3: Shuffle sanity
    if all_shuffle:
        df_shuf = pd.DataFrame(all_shuffle)
        df_shuf.to_csv(OUTPUT_DEEP / "shuffle_sanity.csv", index=False)

        print("\n## Q3: Shuffle sanity test")
        # Aggregate: for each state × target, how many symbols have real > p99?
        agg_shuf = df_shuf.groupby(["state", "target"]).agg(
            n_syms=("symbol", "nunique"),
            mean_real_uplift=("real_uplift", "mean"),
            mean_shuffle_mean=("shuffle_mean", "mean"),
            mean_shuffle_p95=("shuffle_p95", "mean"),
            pct_exceeds_p99=("real_exceeds_p99", "mean"),
        ).reset_index()
        print(agg_shuf.to_string(index=False, float_format="%.3f"))
    else:
        print("\n## Q3: No shuffle data")

    # Q4: OI leakage
    if all_leakage:
        df_leak = pd.DataFrame(all_leakage)
        df_leak.to_csv(OUTPUT_DEEP / "oi_leakage.csv", index=False)

        print("\n## Q4: OI leakage trap (normal vs leaked OI)")
        # Compare normal vs leaked uplift
        pivot = df_leak.pivot_table(
            index=["state", "target"],
            columns="variant",
            values="uplift_test",
            aggfunc="mean",
        )
        if "normal" in pivot.columns and "leaked" in pivot.columns:
            pivot["leak_boost"] = pivot["leaked"] / pivot["normal"].clip(lower=1e-4)
            print(pivot.to_string(float_format="%.3f"))
            print()
            print("leak_boost >> 1 = OI causality confirmed (leak helps)")
            print("leak_boost ~ 1  = OI not driving signal (possible bug)")
    else:
        print("\n## Q4: No leakage data")

    # --- Generate findings ---
    generate_deep_findings(all_ttm, all_skew, all_shuffle, all_leakage)

    elapsed = time.monotonic() - t_start
    print(f"\nDeep analysis done in {elapsed:.0f}s ({elapsed/60:.1f}min)")


def generate_deep_findings(all_ttm, all_skew, all_shuffle, all_leakage):
    """Write FINDINGS_xs6_deep.md."""
    lines = [
        "# XS-6 Deep Analysis — Time-to-move, Skew, Sanity",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # Q1 summary
    lines.append("## Q1: Time-to-move Distribution")
    lines.append("")
    if all_ttm:
        df = pd.DataFrame(all_ttm)
        for state in FOCUS_STATES:
            sd = df[df["state"] == state]
            breached = sd[~sd["first_breach_atr_min"].isna()]
            if len(sd) < 5:
                continue
            lines.append(f"### {state}")
            lines.append(f"- Signals (test): {len(sd)}")
            lines.append(f"- Breach rate (ATR k={SKEW_K}, 24h): {len(breached)/len(sd):.1%}")
            if len(breached) > 0:
                t_arr = breached["first_breach_atr_min"].values
                lines.append(f"- Median time to breach: {np.median(t_arr):.0f}min ({np.median(t_arr)/60:.1f}h)")
                lines.append(f"- P25/P75: {np.percentile(t_arr, 25):.0f}min / {np.percentile(t_arr, 75):.0f}min")
                for label, lo, hi in TIME_BUCKETS:
                    pct = ((t_arr >= lo) & (t_arr < hi)).mean()
                    lines.append(f"  - {label}: {pct:.1%}")
                dirs = breached["first_breach_atr_dir"].values
                lines.append(f"- Direction at breach: {(dirs==1).mean():.0%} up, {(dirs==-1).mean():.0%} down")
            lines.append("")
    else:
        lines.append("No data.")
        lines.append("")

    # Q2 summary
    lines.append("## Q2: Direction Skew")
    lines.append("")
    if all_skew:
        df = pd.DataFrame(all_skew)
        sk200 = df[df["threshold"] == "200bp"]
        if len(sk200) > 0:
            for state in FOCUS_STATES:
                sd = sk200[sk200["state"] == state]
                if len(sd) < 3:
                    continue
                lines.append(f"### {state}")
                for h in ["12h", "24h"]:
                    sh = sd[sd["horizon"] == h]
                    if len(sh) == 0:
                        continue
                    lines.append(f"  - {h}: P(up>200bp)={sh['p_up'].mean():.3f}, "
                                 f"P(down>200bp)={sh['p_down'].mean():.3f}, "
                                 f"skew_ratio={sh['skew_ratio'].mean():.2f}, "
                                 f"uplift_up={sh['uplift_up'].mean():.2f}x, "
                                 f"uplift_down={sh['uplift_down'].mean():.2f}x")
                lines.append("")
    else:
        lines.append("No data.")
        lines.append("")

    # Q3 summary
    lines.append("## Q3: Shuffle Sanity Test")
    lines.append("")
    lines.append("If real uplift is genuine, it should exceed p99 of shuffled distribution for most symbols.")
    lines.append("")
    if all_shuffle:
        df = pd.DataFrame(all_shuffle)
        for state in FOCUS_STATES:
            sd = df[df["state"] == state]
            if len(sd) == 0:
                continue
            for tgt in sd["target"].unique():
                st = sd[sd["target"] == tgt]
                valid = st.dropna(subset=["real_uplift"])
                if len(valid) == 0:
                    continue
                pct99 = valid["real_exceeds_p99"].mean() if "real_exceeds_p99" in valid.columns else 0
                lines.append(f"- **{state}** / {tgt}: "
                             f"real_uplift={valid['real_uplift'].mean():.2f}, "
                             f"shuffle_mean={valid['shuffle_mean'].mean():.2f}, "
                             f"shuffle_p99={valid['shuffle_p99'].mean():.2f}, "
                             f"exceeds_p99={pct99:.0%} of symbols")
        lines.append("")
    else:
        lines.append("No data.")
        lines.append("")

    # Q4 summary
    lines.append("## Q4: OI Leakage Trap")
    lines.append("")
    lines.append("If OI is causal, leaking future OI should boost uplift significantly.")
    lines.append("")
    if all_leakage:
        df = pd.DataFrame(all_leakage)
        pivot = df.pivot_table(
            index=["state", "target"],
            columns="variant",
            values="uplift_test",
            aggfunc="mean",
        )
        if "normal" in pivot.columns and "leaked" in pivot.columns:
            pivot["leak_boost"] = pivot["leaked"] / pivot["normal"].clip(lower=1e-4)
            for (state, target), row in pivot.iterrows():
                lb = row.get("leak_boost", np.nan)
                lines.append(f"- **{state}** / {target}: "
                             f"normal={row.get('normal', np.nan):.2f}, "
                             f"leaked={row.get('leaked', np.nan):.2f}, "
                             f"boost={lb:.2f}x")
        lines.append("")
    else:
        lines.append("No data.")
        lines.append("")

    md_path = OUTPUT_DEEP / "FINDINGS_xs6_deep.md"
    md_path.write_text("\n".join(lines))
    print(f"\nDeep findings written to {md_path}")


if __name__ == "__main__":
    main()
