#!/usr/bin/env python3
"""
OI Regime Lead/Lag Analysis — TRUE 5-SECOND RESOLUTION

Previous test (v24c) was flawed: it aggregated to 5-min bars first,
so OI and OHLCV always appeared to spike at the same bar.

This test works at raw 5-second ticker resolution:
1. Identify regime transitions from 5-min GMM labels
2. Load raw 5-second ticker data around each transition
3. Compute 5-second OI velocity and 5-second price volatility
4. Check which one spikes first — at SECOND-level precision

The question: does OI velocity rise 10-60 seconds BEFORE price starts moving?
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path("./parquet")
SOURCE = "bybit_futures"
INTERVAL_5M_US = 300_000_000

SYMBOL = "BTCUSDT"
START_DATE = "2025-05-12"
END_DATE = "2025-08-08"


def load_ohlcv_and_regimes():
    """Load 5-min bars and fit GMM for regime labels."""
    from regime_detection import load_bars, compute_regime_features
    print("Loading OHLCV bars and fitting GMM...")
    df = load_bars(SYMBOL, START_DATE, END_DATE)
    if df.empty:
        return df, []
    df = compute_regime_features(df)

    regime_cols = [c for c in [
        "rvol_1h", "rvol_4h", "rvol_24h", "parkvol_1h",
        "vol_ratio_1h_24h", "efficiency_1h", "efficiency_4h",
        "adx_4h", "bar_eff_4h", "trade_intensity_ratio",
    ] if c in df.columns]

    valid = df[regime_cols].notna().all(axis=1)
    X = df.loc[valid, regime_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2, covariance_type="diag",
                           n_init=10, random_state=42, max_iter=300)
    labels = gmm.fit_predict(X_scaled)

    if np.mean(df.loc[valid, "rvol_1h"].values[labels == 0]) > \
       np.mean(df.loc[valid, "rvol_1h"].values[labels == 1]):
        labels = 1 - labels

    regime = np.full(len(df), -1, dtype=np.int8)
    regime[valid.values] = labels
    df["regime"] = regime

    # Find quiet→volatile transitions
    transitions = []
    for i in range(1, len(regime)):
        if regime[i-1] == 0 and regime[i] == 1:
            transitions.append({
                "bar_idx": i,
                "timestamp_us": int(df.iloc[i]["timestamp_us"]),
            })

    n_quiet = (regime == 0).sum()
    n_vol = (regime == 1).sum()
    print(f"  {len(df)} bars, quiet={n_quiet} ({n_quiet/len(df)*100:.0f}%), "
          f"volatile={n_vol} ({n_vol/len(df)*100:.0f}%)")
    print(f"  Transitions quiet→volatile: {len(transitions)}")
    return df, transitions


def load_raw_ticker():
    """Load ALL raw 5-second ticker data."""
    print("Loading raw 5-second ticker data...")
    ticker_dir = PARQUET_DIR / SYMBOL / "ticker"
    dates = pd.date_range(START_DATE, END_DATE)
    dfs = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        path = ticker_dir / f"{ds}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    print(f"  {len(df):,} rows, {len(dfs)} days")
    return df


def analyze_transitions_at_5s(ticker_df, transitions):
    """
    For each quiet→volatile transition:
    - Extract 5-second data in a window around the transition timestamp
    - Compute rolling OI velocity and price volatility at 5-second resolution
    - Determine which one spikes first
    """
    print(f"\n{'='*70}")
    print(f"  5-SECOND LEAD/LAG ANALYSIS")
    print(f"  Window: 5 min before to 5 min after each transition")
    print(f"  Resolution: 5 seconds")
    print(f"{'='*70}")

    ts_array = ticker_df["timestamp_us"].values
    oi_array = ticker_df["open_interest"].values
    price_array = ticker_df["last_price"].values

    # Pre-compute 5-second changes
    oi_pct_change = np.zeros(len(ticker_df))
    oi_pct_change[1:] = np.diff(oi_array) / np.maximum(oi_array[:-1], 1) * 100

    price_pct_change = np.zeros(len(ticker_df))
    price_pct_change[1:] = np.diff(price_array) / np.maximum(price_array[:-1], 1e-10) * 100

    # Window: 5 min = 300s = 60 ticks of 5s
    window_ticks = 60  # 5 min before and after
    tick_interval_us = 5_000_000  # 5 seconds in microseconds

    # For each transition, collect OI velocity and price volatility profiles
    # at 5-second resolution
    n_profile_ticks = window_ticks * 2 + 1  # -60 to +60
    oi_vel_profiles = []  # rolling |OI change| (6-tick = 30s rolling)
    price_vol_profiles = []  # rolling |price change| (6-tick = 30s rolling)
    oi_spike_profiles = []  # is OI change > threshold?
    price_spike_profiles = []  # is price change > threshold?

    roll_window = 6  # 30 seconds rolling

    valid_count = 0
    skipped_count = 0

    for trans in transitions:
        trans_ts = trans["timestamp_us"]

        # Find the ticker index closest to transition timestamp
        idx = np.searchsorted(ts_array, trans_ts)
        if idx < window_ticks + roll_window or idx + window_ticks >= len(ts_array):
            skipped_count += 1
            continue

        # Check that data is continuous (no gaps > 15s)
        window_ts = ts_array[idx - window_ticks: idx + window_ticks + 1]
        gaps = np.diff(window_ts) / 1_000_000  # in seconds
        if np.max(gaps) > 15_000:  # gap > 15 seconds (in ms... wait, ts is in us)
            skipped_count += 1
            continue

        # Extract window
        start = idx - window_ticks
        end = idx + window_ticks + 1

        oi_vel_window = np.abs(oi_pct_change[start:end])
        price_vol_window = np.abs(price_pct_change[start:end])

        # Rolling mean (30-second window)
        oi_vel_rolling = pd.Series(oi_vel_window).rolling(roll_window, min_periods=1).mean().values
        price_vol_rolling = pd.Series(price_vol_window).rolling(roll_window, min_periods=1).mean().values

        oi_vel_profiles.append(oi_vel_rolling)
        price_vol_profiles.append(price_vol_rolling)

        # Spike detection: is the value > 2x the pre-transition baseline?
        # Baseline: ticks -60 to -30 (5 min to 2.5 min before)
        oi_baseline = np.mean(oi_vel_rolling[:30]) if np.mean(oi_vel_rolling[:30]) > 0 else 1e-10
        price_baseline = np.mean(price_vol_rolling[:30]) if np.mean(price_vol_rolling[:30]) > 0 else 1e-10

        oi_spike_profiles.append(oi_vel_rolling / oi_baseline)
        price_spike_profiles.append(price_vol_rolling / price_baseline)

        valid_count += 1

    print(f"  Valid transitions: {valid_count} (skipped {skipped_count} due to gaps/edges)")

    if valid_count < 10:
        print("  Not enough valid transitions for analysis!")
        return

    # Average profiles
    oi_vel_avg = np.mean(oi_vel_profiles, axis=0)
    price_vol_avg = np.mean(price_vol_profiles, axis=0)
    oi_spike_avg = np.mean(oi_spike_profiles, axis=0)
    price_spike_avg = np.mean(price_spike_profiles, axis=0)

    # Time axis: seconds relative to transition
    time_seconds = np.arange(-window_ticks, window_ticks + 1) * 5

    # Print profiles at key timepoints
    print(f"\n  Average OI velocity and Price volatility around transitions:")
    print(f"  (Values are 30-second rolling mean of |5s change|)")
    print(f"  {'Time':>8s} {'OI vel':>12s} {'Price vol':>12s} {'OI ratio':>10s} {'Price ratio':>12s}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    key_times = [-300, -240, -180, -120, -90, -60, -45, -30, -20, -15, -10, -5,
                 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300]
    for t in key_times:
        tick_idx = (t // 5) + window_ticks
        if 0 <= tick_idx < n_profile_ticks:
            oi_v = oi_vel_avg[tick_idx]
            pr_v = price_vol_avg[tick_idx]
            oi_r = oi_spike_avg[tick_idx]
            pr_r = price_spike_avg[tick_idx]
            marker = " <<<" if t == 0 else ""
            print(f"  {t:>+7d}s {oi_v:>12.6f} {pr_v:>12.6f} {oi_r:>10.2f}x {pr_r:>11.2f}x{marker}")

    # Find first time each metric exceeds 2x baseline
    print(f"\n  First time metric exceeds threshold (relative to transition at t=0):")
    print(f"  {'Metric':>20s} {'> 1.5x':>10s} {'> 2.0x':>10s} {'> 3.0x':>10s} {'> 5.0x':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, profile in [("OI velocity", oi_spike_avg), ("Price volatility", price_spike_avg)]:
        thresholds = {}
        for thresh in [1.5, 2.0, 3.0, 5.0]:
            first_idx = None
            for i in range(len(profile)):
                if profile[i] > thresh:
                    # Require sustained (2+ consecutive ticks)
                    if i + 1 < len(profile) and profile[i+1] > thresh * 0.8:
                        first_idx = i
                        break
            if first_idx is not None:
                t_seconds = (first_idx - window_ticks) * 5
                thresholds[thresh] = f"{t_seconds:+d}s"
            else:
                thresholds[thresh] = "never"

        print(f"  {name:>20s} {thresholds[1.5]:>10s} {thresholds[2.0]:>10s} "
              f"{thresholds[3.0]:>10s} {thresholds[5.0]:>10s}")

    # Per-transition analysis: which spikes first?
    print(f"\n  Per-transition: which metric spikes first (> 2x baseline)?")

    oi_first_count = 0
    price_first_count = 0
    simultaneous_count = 0
    neither_count = 0
    oi_lead_seconds = []

    for i in range(valid_count):
        oi_prof = oi_spike_profiles[i]
        pr_prof = price_spike_profiles[i]

        # Find first tick > 2x baseline in the window [-60, +60] ticks
        # Only look from tick 20 onwards (from -200s = -3.3min, to avoid noise)
        search_start = 20  # -200 seconds
        oi_first = None
        pr_first = None

        for t in range(search_start, n_profile_ticks):
            if oi_first is None and oi_prof[t] > 2.0:
                if t + 1 < n_profile_ticks and oi_prof[t+1] > 1.5:
                    oi_first = t
            if pr_first is None and pr_prof[t] > 2.0:
                if t + 1 < n_profile_ticks and pr_prof[t+1] > 1.5:
                    pr_first = t

        if oi_first is None and pr_first is None:
            neither_count += 1
        elif oi_first is not None and pr_first is None:
            oi_first_count += 1
        elif oi_first is None and pr_first is not None:
            price_first_count += 1
        elif oi_first < pr_first:
            oi_first_count += 1
            oi_lead_seconds.append((pr_first - oi_first) * 5)
        elif pr_first < oi_first:
            price_first_count += 1
            oi_lead_seconds.append(-(oi_first - pr_first) * 5)
        else:
            simultaneous_count += 1
            oi_lead_seconds.append(0)

    total_detected = oi_first_count + price_first_count + simultaneous_count
    print(f"    OI spikes first:    {oi_first_count:>5d} ({oi_first_count/max(valid_count,1)*100:.1f}%)")
    print(f"    Price spikes first: {price_first_count:>5d} ({price_first_count/max(valid_count,1)*100:.1f}%)")
    print(f"    Simultaneous:       {simultaneous_count:>5d} ({simultaneous_count/max(valid_count,1)*100:.1f}%)")
    print(f"    Neither spikes:     {neither_count:>5d} ({neither_count/max(valid_count,1)*100:.1f}%)")

    if oi_lead_seconds:
        arr = np.array(oi_lead_seconds)
        print(f"\n    OI lead time (positive = OI first):")
        print(f"      Mean:   {np.mean(arr):+.1f} seconds")
        print(f"      Median: {np.median(arr):+.1f} seconds")
        print(f"      P25:    {np.percentile(arr, 25):+.1f} seconds")
        print(f"      P75:    {np.percentile(arr, 75):+.1f} seconds")

        # Breakdown
        leads = arr[arr > 0]
        lags = arr[arr < 0]
        if len(leads) > 0:
            print(f"\n    When OI leads ({len(leads)} cases):")
            print(f"      Mean lead: {np.mean(leads):.0f}s, Median: {np.median(leads):.0f}s, "
                  f"Max: {np.max(leads):.0f}s")
        if len(lags) > 0:
            print(f"    When Price leads ({len(lags)} cases):")
            print(f"      Mean lead: {-np.mean(lags):.0f}s, Median: {-np.median(lags):.0f}s, "
                  f"Max: {-np.min(lags):.0f}s")


def analyze_oi_as_5s_early_warning(ticker_df, transitions):
    """
    Build a 5-second OI velocity detector and measure how many seconds
    before the 5-min regime transition it fires.
    """
    print(f"\n{'='*70}")
    print(f"  5-SECOND OI EARLY WARNING DETECTOR")
    print(f"  How many seconds before the 5-min bar transition can OI detect it?")
    print(f"{'='*70}")

    ts_array = ticker_df["timestamp_us"].values
    oi_array = ticker_df["open_interest"].values
    price_array = ticker_df["last_price"].values

    # Pre-compute
    oi_pct_change = np.zeros(len(ticker_df))
    oi_pct_change[1:] = np.abs(np.diff(oi_array) / np.maximum(oi_array[:-1], 1) * 100)

    price_pct_change = np.zeros(len(ticker_df))
    price_pct_change[1:] = np.abs(np.diff(price_array) / np.maximum(price_array[:-1], 1e-10) * 100)

    # Rolling baselines (288 ticks = 24 min)
    oi_vel_rolling = pd.Series(oi_pct_change).rolling(6, min_periods=1).mean().values
    price_vol_rolling = pd.Series(price_pct_change).rolling(6, min_periods=1).mean().values

    # Longer baseline for z-score (720 ticks = 1 hour)
    oi_baseline_mean = pd.Series(oi_vel_rolling).rolling(720, min_periods=120).mean().values
    oi_baseline_std = pd.Series(oi_vel_rolling).rolling(720, min_periods=120).std().values
    price_baseline_mean = pd.Series(price_vol_rolling).rolling(720, min_periods=120).mean().values
    price_baseline_std = pd.Series(price_vol_rolling).rolling(720, min_periods=120).std().values

    oi_zscore = np.where(oi_baseline_std > 0,
                         (oi_vel_rolling - oi_baseline_mean) / oi_baseline_std, 0)
    price_zscore = np.where(price_baseline_std > 0,
                            (price_vol_rolling - price_baseline_mean) / price_baseline_std, 0)

    # For each transition, look backwards to find when OI z-score first exceeded threshold
    lookback_ticks = 120  # 10 minutes back

    thresholds = [1.0, 1.5, 2.0, 3.0]

    print(f"\n  Detection lead time: seconds BEFORE the 5-min bar transition")
    print(f"  {'Detector':>25s} {'Detected':>10s} {'Med Lead':>10s} {'Mean Lead':>10s} {'P75 Lead':>10s} {'P90 Lead':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for thresh in thresholds:
        for name, zscore in [("OI vel", oi_zscore), ("Price vol", price_zscore)]:
            leads = []
            detected = 0

            for trans in transitions:
                trans_ts = trans["timestamp_us"]
                idx = np.searchsorted(ts_array, trans_ts)
                if idx < lookback_ticks + 720:
                    continue

                # Look backwards from transition
                lead_seconds = None
                for offset in range(0, lookback_ticks):
                    check_idx = idx - offset
                    if zscore[check_idx] >= thresh:
                        lead_seconds = offset * 5  # convert ticks to seconds
                    else:
                        if lead_seconds is not None:
                            break  # found the start of the spike

                if lead_seconds is not None and lead_seconds > 0:
                    leads.append(lead_seconds)
                    detected += 1

            n_valid = sum(1 for t in transitions
                         if np.searchsorted(ts_array, t["timestamp_us"]) >= lookback_ticks + 720)
            det_pct = detected / max(n_valid, 1) * 100

            if leads:
                med = np.median(leads)
                mean = np.mean(leads)
                p75 = np.percentile(leads, 75)
                p90 = np.percentile(leads, 90)
                print(f"  {name} z>{thresh:.1f}            {det_pct:>9.0f}% {med:>9.0f}s {mean:>9.0f}s {p75:>9.0f}s {p90:>9.0f}s")
            else:
                print(f"  {name} z>{thresh:.1f}            {det_pct:>9.0f}% {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")

    # Direct comparison at z > 1.5
    print(f"\n  Head-to-head at z > 1.5 threshold:")
    oi_leads_per_trans = {}
    price_leads_per_trans = {}

    for trans in transitions:
        trans_ts = trans["timestamp_us"]
        idx = np.searchsorted(ts_array, trans_ts)
        if idx < lookback_ticks + 720:
            continue

        for name, zscore, store in [("OI", oi_zscore, oi_leads_per_trans),
                                     ("Price", price_zscore, price_leads_per_trans)]:
            lead_seconds = None
            for offset in range(0, lookback_ticks):
                check_idx = idx - offset
                if zscore[check_idx] >= 1.5:
                    lead_seconds = offset * 5
                else:
                    if lead_seconds is not None:
                        break
            store[trans_ts] = lead_seconds if lead_seconds is not None and lead_seconds > 0 else 0

    # Compare
    common_keys = set(oi_leads_per_trans.keys()) & set(price_leads_per_trans.keys())
    oi_wins = 0
    price_wins = 0
    ties = 0
    advantages = []

    for k in common_keys:
        oi_lead = oi_leads_per_trans[k]
        pr_lead = price_leads_per_trans[k]
        if oi_lead > pr_lead:
            oi_wins += 1
            advantages.append(oi_lead - pr_lead)
        elif pr_lead > oi_lead:
            price_wins += 1
            advantages.append(-(pr_lead - oi_lead))
        else:
            ties += 1
            advantages.append(0)

    total = len(common_keys)
    print(f"    OI fires first:    {oi_wins:>5d} ({oi_wins/max(total,1)*100:.1f}%)")
    print(f"    Price fires first: {price_wins:>5d} ({price_wins/max(total,1)*100:.1f}%)")
    print(f"    Tie (same time):   {ties:>5d} ({ties/max(total,1)*100:.1f}%)")

    if advantages:
        arr = np.array(advantages)
        nonzero = arr[arr != 0]
        print(f"\n    OI advantage (positive = OI earlier):")
        print(f"      Mean:   {np.mean(arr):+.1f}s")
        print(f"      Median: {np.median(arr):+.1f}s")
        if len(nonzero) > 0:
            pos = nonzero[nonzero > 0]
            neg = nonzero[nonzero < 0]
            if len(pos) > 0:
                print(f"      When OI leads: avg {np.mean(pos):.0f}s ({len(pos)} cases)")
            if len(neg) > 0:
                print(f"      When Price leads: avg {-np.mean(neg):.0f}s ({len(neg)} cases)")


def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  OI REGIME LEAD/LAG — 5-SECOND RESOLUTION — {SYMBOL}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"{'='*70}")

    ohlcv_df, transitions = load_ohlcv_and_regimes()
    if not transitions:
        print("ERROR: No transitions found")
        return

    ticker_df = load_raw_ticker()
    if ticker_df.empty:
        print("ERROR: No ticker data")
        return

    analyze_transitions_at_5s(ticker_df, transitions)
    analyze_oi_as_5s_early_warning(ticker_df, transitions)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()
    SYMBOL = args.symbol
    main()
