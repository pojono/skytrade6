#!/usr/bin/env python3
"""
v33c: DST Analysis — Does the volatility peak shift with clock changes?

US DST transitions (second Sunday Mar → first Sunday Nov):
  Summer (EDT, UTC-4): NYSE opens 13:30 UTC, regular 14:30 UTC
  Winter (EST, UTC-5): NYSE opens 14:30 UTC, regular 15:30 UTC

London DST transitions (last Sunday Mar → last Sunday Oct):
  Summer (BST, UTC+1): LSE opens 07:00 UTC
  Winter (GMT, UTC+0): LSE opens 08:00 UTC

If the 14:00 UTC peak is truly driven by equity market open,
it should shift to ~15:00 UTC in winter.

Uses 3+ years of 5-min OHLCV parquet (all 5 symbols).
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

PARQUET_DIR = Path("parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
SOURCE = "bybit_futures"

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# US DST transitions (second Sunday of March → first Sunday of November)
# We define the exact dates for each year
US_DST_TRANSITIONS = {
    2023: (datetime(2023, 3, 12), datetime(2023, 11, 5)),
    2024: (datetime(2024, 3, 10), datetime(2024, 11, 3)),
    2025: (datetime(2025, 3, 9),  datetime(2025, 11, 2)),
    2026: (datetime(2026, 3, 8),  datetime(2026, 11, 1)),
}

# London DST transitions (last Sunday of March → last Sunday of October)
UK_DST_TRANSITIONS = {
    2023: (datetime(2023, 3, 26), datetime(2023, 10, 29)),
    2024: (datetime(2024, 3, 31), datetime(2024, 10, 27)),
    2025: (datetime(2025, 3, 30), datetime(2025, 10, 26)),
    2026: (datetime(2026, 3, 29), datetime(2026, 10, 25)),
}

# Actual session hours by DST state (UTC)
# NYSE: regular trading hours
# LSE: regular trading hours
SESSION_HOURS = {
    'us_summer': {'nyse_open': 13, 'nyse_close': 20, 'nyse_regular': 14},  # EDT
    'us_winter': {'nyse_open': 14, 'nyse_close': 21, 'nyse_regular': 15},  # EST
    'uk_summer': {'lse_open': 7, 'lse_close': 15},   # BST
    'uk_winter': {'lse_open': 8, 'lse_close': 16},   # GMT
}


def mem_gb():
    import psutil
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label):
    u, a = mem_gb()
    print(f"  [RAM] used={u:.1f}GB avail={a:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv(symbol):
    """Load all 5-min OHLCV parquet files for a symbol."""
    t0 = time.time()
    ohlcv_dir = PARQUET_DIR / symbol / "ohlcv" / "5m" / SOURCE
    files = sorted(ohlcv_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)

    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year

    # Classify DST state
    df['us_dst'] = 'winter'  # default
    df['uk_dst'] = 'winter'
    for year, (start, end) in US_DST_TRANSITIONS.items():
        mask = (df['date'] >= start.date()) & (df['date'] < end.date())
        df.loc[mask, 'us_dst'] = 'summer'
    for year, (start, end) in UK_DST_TRANSITIONS.items():
        mask = (df['date'] >= start.date()) & (df['date'] < end.date())
        df.loc[mask, 'uk_dst'] = 'summer'

    # Combined DST state
    df['dst_state'] = df['us_dst'] + '_' + df['uk_dst']

    elapsed = time.time() - t0
    print(f"  {symbol}: {len(df):,} bars ({elapsed:.1f}s)")

    # Print DST distribution
    for state in sorted(df['dst_state'].unique()):
        n = (df['dst_state'] == state).sum()
        pct = n / len(df) * 100
        print(f"    {state}: {n:,} bars ({pct:.1f}%)")

    return df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_dst_hourly(df, symbol):
    """Compare hourly volatility profiles between US summer and winter."""
    print(f"\n{'='*70}")
    print(f"DST HOURLY ANALYSIS — {symbol}")
    print(f"{'='*70}")

    # Weekday only for cleaner signal
    wdf = df[df['dow'] < 5].copy()

    summer = wdf[wdf['us_dst'] == 'summer']
    winter = wdf[wdf['us_dst'] == 'winter']

    print(f"\n  Weekday bars: summer={len(summer):,}, winter={len(winter):,}")

    # Hourly profiles
    s_hourly = summer.groupby('hour')['range_bps'].mean()
    w_hourly = winter.groupby('hour')['range_bps'].mean()

    print(f"\n  Hour   Summer(EDT)  Winter(EST)   Ratio   Summer Session          Winter Session")
    print(f"  {'─'*95}")

    for h in range(24):
        sv = s_hourly.get(h, 0)
        wv = w_hourly.get(h, 0)
        ratio = sv / wv if wv > 0 else 0

        # What's happening at this hour in each season
        s_sess = _session_at_hour(h, 'summer')
        w_sess = _session_at_hour(h, 'winter')

        marker = ''
        if sv == s_hourly.max():
            marker = ' ◄ SUMMER PEAK'
        if wv == w_hourly.max():
            marker += ' ◄ WINTER PEAK'

        print(f"  {h:4d}   {sv:10.2f}   {wv:10.2f}   {ratio:5.2f}x  {s_sess:<22s}  {w_sess:<22s}{marker}")

    # Peak hours
    s_peak = s_hourly.idxmax()
    w_peak = w_hourly.idxmax()
    print(f"\n  Summer peak hour: {s_peak:02d}:00 UTC ({s_hourly[s_peak]:.2f} bps)")
    print(f"  Winter peak hour: {w_peak:02d}:00 UTC ({w_hourly[w_peak]:.2f} bps)")
    print(f"  Peak shift: {w_peak - s_peak:+d} hour(s)")

    # Top-3 hours
    s_top3 = s_hourly.nlargest(3)
    w_top3 = w_hourly.nlargest(3)
    print(f"\n  Summer top-3: {', '.join(f'{h:02d}:00({v:.1f})' for h, v in s_top3.items())}")
    print(f"  Winter top-3: {', '.join(f'{h:02d}:00({v:.1f})' for h, v in w_top3.items())}")

    # Correlation between profiles
    common = sorted(set(s_hourly.index) & set(w_hourly.index))
    rho, p = stats.spearmanr(s_hourly[common], w_hourly[common])
    print(f"\n  Summer vs Winter hourly profile: ρ={rho:+.3f}, p={p:.4f}")

    return s_hourly, w_hourly, s_peak, w_peak


def _session_at_hour(h, season):
    """Describe what's happening at hour h UTC in given season."""
    parts = []

    # Tokyo (no DST)
    if 0 <= h < 6:
        parts.append('Tokyo')
    elif 6 <= h < 9:
        parts.append('Tokyo(late)')

    # London
    if season == 'summer':
        if 7 <= h < 15:
            parts.append('London')
    else:
        if 8 <= h < 16:
            parts.append('London')

    # NYSE
    if season == 'summer':
        if h == 13:
            parts.append('NYSE(pre)')
        elif 14 <= h < 20:
            parts.append('NYSE')
    else:
        if h == 14:
            parts.append('NYSE(pre)')
        elif 15 <= h < 21:
            parts.append('NYSE')

    if not parts:
        return 'Quiet'
    return ' + '.join(parts)


def analyze_4state_dst(df, symbol):
    """Analyze all 4 DST states: US×UK summer/winter."""
    print(f"\n{'='*70}")
    print(f"4-STATE DST ANALYSIS — {symbol} (weekdays only)")
    print(f"{'='*70}")

    wdf = df[df['dow'] < 5].copy()

    states = ['summer_summer', 'summer_winter', 'winter_summer', 'winter_winter']
    state_labels = {
        'summer_summer': 'US-EDT + UK-BST (Apr-Oct)',
        'summer_winter': 'US-EDT + UK-GMT (Mar, Nov)',
        'winter_summer': 'US-EST + UK-BST (never/rare)',
        'winter_winter': 'US-EST + UK-GMT (Nov-Mar)',
    }

    profiles = {}
    peaks = {}

    for state in states:
        sub = wdf[wdf['dst_state'] == state]
        if len(sub) < 1000:
            continue
        hourly = sub.groupby('hour')['range_bps'].mean()
        profiles[state] = hourly
        peaks[state] = hourly.idxmax()

    print(f"\n  {'State':<35s}  {'N bars':>8s}  {'Peak Hour':>10s}  {'Peak Range':>11s}")
    print(f"  {'─'*70}")
    for state in states:
        if state not in profiles:
            continue
        n = (wdf['dst_state'] == state).sum()
        ph = peaks[state]
        pv = profiles[state][ph]
        label = state_labels.get(state, state)
        print(f"  {label:<35s}  {n:>8,}  {ph:02d}:00 UTC  {pv:>10.2f}")

    # Print the key comparison
    if 'summer_summer' in profiles and 'winter_winter' in profiles:
        print(f"\n  Key comparison (most common states):")
        print(f"    Full summer (EDT+BST): peak at {peaks['summer_summer']:02d}:00 UTC")
        print(f"    Full winter (EST+GMT): peak at {peaks['winter_winter']:02d}:00 UTC")
        shift = peaks['winter_winter'] - peaks['summer_summer']
        print(f"    Shift: {shift:+d} hour(s)")

        if shift == 1:
            print(f"    ✓ CONFIRMED: Peak shifts +1h in winter, consistent with NYSE open moving from 14:30→15:30 UTC")
        elif shift == 0:
            print(f"    ✗ No shift detected — peak may not be purely equity-driven")

    return profiles, peaks


def save_dst_csv(all_profiles, out_path):
    """Save summer/winter hourly profiles for all symbols as CSV."""
    rows = []
    for h in range(24):
        row = {'hour_utc': f'{h:02d}:00',
               'summer_session': _session_at_hour(h, 'summer'),
               'winter_session': _session_at_hour(h, 'winter')}
        for symbol in SYMBOLS:
            key_s = f'{symbol}_summer'
            key_w = f'{symbol}_winter'
            if key_s in all_profiles:
                row[f'{symbol}_summer_range'] = round(all_profiles[key_s].get(h, 0), 2)
            if key_w in all_profiles:
                row[f'{symbol}_winter_range'] = round(all_profiles[key_w].get(h, 0), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()

    print(f"{'='*70}")
    print(f"v33c: DST Analysis — Does the Volatility Peak Shift?")
    print(f"{'='*70}")
    print_mem("start")

    all_profiles = {}
    all_peaks = {}
    summary_rows = []

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)
        if df.empty:
            continue

        s_hourly, w_hourly, s_peak, w_peak = analyze_dst_hourly(df, symbol)
        all_profiles[f'{symbol}_summer'] = s_hourly
        all_profiles[f'{symbol}_winter'] = w_hourly

        profiles_4, peaks_4 = analyze_4state_dst(df, symbol)

        summary_rows.append({
            'symbol': symbol,
            'summer_peak_hour': s_peak,
            'winter_peak_hour': w_peak,
            'shift': w_peak - s_peak,
        })

        del df

    # Summary table
    print(f"\n{'='*70}")
    print(f"CROSS-SYMBOL DST SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Symbol':>8s}  {'Summer Peak':>12s}  {'Winter Peak':>12s}  {'Shift':>6s}  {'Verdict':>30s}")
    print(f"  {'─'*75}")
    for row in summary_rows:
        shift = row['shift']
        if shift == 1:
            verdict = '✓ +1h shift (equity-driven)'
        elif shift == 0:
            verdict = '— no shift'
        else:
            verdict = f'? {shift:+d}h shift'
        print(f"  {row['symbol']:>8s}  {row['summer_peak_hour']:02d}:00 UTC   "
              f"{row['winter_peak_hour']:02d}:00 UTC   {shift:+4d}h  {verdict}")

    # Save CSV
    save_dst_csv(all_profiles, RESULTS_DIR / 'v33c_dst_hourly_profiles.csv')

    # Save summary
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / 'v33c_dst_peak_shifts.csv', index=False)
    print(f"  Saved: {RESULTS_DIR / 'v33c_dst_peak_shifts.csv'}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print_mem("final")


if __name__ == '__main__':
    main()
