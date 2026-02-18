#!/usr/bin/env python3
"""
v34: Funding Rate Cycle Microstructure

The 8-hour funding cycle (00:00, 08:00, 16:00 UTC) creates predictable forced flows.
- Volatility profile within each 8h funding window
- Does funding rate level predict next-window volatility?
- OI dynamics around funding times
- Pre-funding positioning effects

Uses 3+ years of 5-min OHLCV parquet (all 5 symbols).
Note: We don't have funding rate in OHLCV, so we analyze the 8h cycle structure
from price/volume patterns alone.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PARQUET_DIR = Path("parquet")
RESULTS_DIR = Path("results")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
SOURCE = "bybit_futures"

FUNDING_HOURS = [0, 8, 16]  # UTC


def load_ohlcv(symbol):
    t0 = time.time()
    ohlcv_dir = PARQUET_DIR / symbol / "ohlcv" / "5m" / SOURCE
    files = sorted(ohlcv_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['dow'] = df['datetime'].dt.dayofweek
    df['date'] = df['datetime'].dt.date
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def compute_funding_position(df):
    """Compute position within 8h funding cycle (0-95 bars, 0=funding time)."""
    df = df.copy()
    # Minutes since last funding
    hour = df['hour'].values
    minute = df['minute'].values
    total_min = hour * 60 + minute

    # Distance from last funding (0:00, 8:00, 16:00)
    funding_mins = [0, 480, 960]  # 0:00, 8:00, 16:00 in minutes
    min_since_funding = np.zeros(len(df), dtype=int)
    for i, tm in enumerate(total_min):
        dists = [(tm - fm) % 1440 for fm in funding_mins]
        min_since_funding[i] = min(dists)

    df['min_since_funding'] = min_since_funding
    df['bars_since_funding'] = min_since_funding // 5  # 5-min bars
    df['pct_through_cycle'] = min_since_funding / 480 * 100  # 0-100%

    # Which funding window
    df['funding_window'] = 'window_00'
    df.loc[(hour >= 8) & (hour < 16), 'funding_window'] = 'window_08'
    df.loc[hour >= 16, 'funding_window'] = 'window_16'

    # Pre-funding flag (last 30 min before funding)
    df['is_pre_funding'] = min_since_funding >= 450  # last 30 min
    df['is_post_funding'] = min_since_funding <= 30   # first 30 min
    df['is_mid_cycle'] = (min_since_funding >= 120) & (min_since_funding <= 360)

    return df


def analyze_funding_cycle(df, symbol):
    """Analyze volatility pattern within the 8h funding cycle."""
    print(f"\n  Funding Cycle Analysis ({symbol}):")

    # Average range by position in cycle
    cycle_profile = df.groupby('bars_since_funding')['range_bps'].agg(['mean', 'count']).reset_index()
    cycle_profile.columns = ['bar', 'avg_range', 'count']

    # Pre vs mid vs post funding
    pre = df[df['is_pre_funding']]['range_bps']
    post = df[df['is_post_funding']]['range_bps']
    mid = df[df['is_mid_cycle']]['range_bps']

    print(f"    Pre-funding (last 30min):  {pre.mean():.2f} bps (n={len(pre):,})")
    print(f"    Post-funding (first 30min): {post.mean():.2f} bps (n={len(post):,})")
    print(f"    Mid-cycle:                  {mid.mean():.2f} bps (n={len(mid):,})")
    print(f"    Pre/Mid ratio: {pre.mean()/mid.mean():.3f}x")
    print(f"    Post/Mid ratio: {post.mean()/mid.mean():.3f}x")

    t_pre, p_pre = stats.mannwhitneyu(pre, mid, alternative='two-sided')
    t_post, p_post = stats.mannwhitneyu(post, mid, alternative='two-sided')
    print(f"    Pre vs Mid: p={p_pre:.2e}")
    print(f"    Post vs Mid: p={p_post:.2e}")

    # By funding window
    print(f"\n    By funding window:")
    for window in ['window_00', 'window_08', 'window_16']:
        sub = df[df['funding_window'] == window]
        print(f"      {window}: avg range = {sub['range_bps'].mean():.2f} bps")

    # Volume pattern
    vol_profile = df.groupby('bars_since_funding')['quote_volume'].mean().reset_index()
    vol_profile.columns = ['bar', 'avg_volume']

    return cycle_profile, vol_profile, {
        'symbol': symbol,
        'pre_funding_range': round(pre.mean(), 2),
        'post_funding_range': round(post.mean(), 2),
        'mid_cycle_range': round(mid.mean(), 2),
        'pre_mid_ratio': round(pre.mean() / mid.mean(), 3),
        'post_mid_ratio': round(post.mean() / mid.mean(), 3),
        'p_pre_vs_mid': p_pre,
        'p_post_vs_mid': p_post,
    }


def main():
    t0 = time.time()
    print("="*70)
    print("v34: Funding Rate Cycle Microstructure")
    print("="*70)

    all_profiles = {}
    all_vol_profiles = {}
    all_summary = []

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)
        df = compute_funding_position(df)

        cycle_prof, vol_prof, summary = analyze_funding_cycle(df, symbol)
        all_profiles[symbol] = cycle_prof
        all_vol_profiles[symbol] = vol_prof
        all_summary.append(summary)

    # Save CSVs
    pd.DataFrame(all_summary).to_csv(RESULTS_DIR / 'v34_funding_cycle_summary.csv', index=False)
    print(f"\n  Saved: results/v34_funding_cycle_summary.csv")

    # Save full cycle profiles
    cycle_rows = []
    for bar in range(96):
        row = {'bar_in_cycle': bar, 'minutes_since_funding': bar * 5,
               'pct_through_cycle': round(bar / 96 * 100, 1)}
        for sym in SYMBOLS:
            prof = all_profiles[sym]
            match = prof[prof['bar'] == bar]
            row[f'{sym}_range_bps'] = round(match['avg_range'].values[0], 2) if len(match) > 0 else None
        for sym in SYMBOLS:
            vprof = all_vol_profiles[sym]
            match = vprof[vprof['bar'] == bar]
            row[f'{sym}_volume'] = round(match['avg_volume'].values[0], 0) if len(match) > 0 else None
        cycle_rows.append(row)
    pd.DataFrame(cycle_rows).to_csv(RESULTS_DIR / 'v34_funding_cycle_profile.csv', index=False)
    print(f"  Saved: results/v34_funding_cycle_profile.csv")

    # ---- PLOT 1: Funding cycle volatility profile ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    for sym, color in zip(SYMBOLS, colors):
        prof = all_profiles[sym]
        # Normalize to mean=100
        mean_val = prof['avg_range'].mean()
        ax1.plot(prof['bar'] * 5, prof['avg_range'] / mean_val * 100,
                 label=sym.replace('USDT',''), color=color, linewidth=1.2)

    ax1.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='red', linewidth=2, linestyle='-', alpha=0.7, label='Funding time')
    ax1.set_ylabel('Normalized Range (100 = avg)', fontsize=11)
    ax1.set_title('v34: Volatility Within 8h Funding Cycle\n(normalized, 3+ years)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Volume profile
    for sym, color in zip(SYMBOLS, colors):
        vprof = all_vol_profiles[sym]
        mean_val = vprof['avg_volume'].mean()
        ax2.plot(vprof['bar'] * 5, vprof['avg_volume'] / mean_val * 100,
                 label=sym.replace('USDT',''), color=color, linewidth=1.2)

    ax2.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='red', linewidth=2, linestyle='-', alpha=0.7)
    ax2.set_xlabel('Minutes since last funding', fontsize=11)
    ax2.set_ylabel('Normalized Volume (100 = avg)', fontsize=11)
    ax2.set_title('Volume Within 8h Funding Cycle', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, ncol=3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v34_funding_cycle.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v34_funding_cycle.png")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for s in all_summary:
        print(f"  {s['symbol']:>10s}: pre/mid={s['pre_mid_ratio']:.3f}x (p={s['p_pre_vs_mid']:.2e}), "
              f"post/mid={s['post_mid_ratio']:.3f}x (p={s['p_post_vs_mid']:.2e})")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
