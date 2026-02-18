#!/usr/bin/env python3
"""
v39: Weekend Gap / Monday Open Effect

Crypto trades 24/7 but institutional flow doesn't. Is there a "Monday open" effect?
- Compare day transitions: Sun→Mon vs other day transitions
- Gap-fill behavior: does Monday price revert to Friday close?
- Vol spike at Monday 00:00 UTC as algos re-engage?
- Friday close → Monday open return distribution

Uses 3+ years of 5-min OHLCV parquet (all 5 symbols).
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


def load_ohlcv(symbol):
    t0 = time.time()
    ohlcv_dir = PARQUET_DIR / symbol / "ohlcv" / "5m" / SOURCE
    files = sorted(ohlcv_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['ret_bps'] = (df['close'] / df['close'].shift(1) - 1) * 10000
    df['dow'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def analyze_day_transitions(df, symbol):
    """Compare the first hours of each day vs the last hours of previous day."""
    print(f"\n  Day transition analysis ({symbol}):")

    results = []
    # For each day-of-week, look at 00:00-02:00 range
    for dow in range(7):
        sub = df[(df['dow'] == dow) & (df['hour'] < 3)]
        avg_range = sub['range_bps'].mean()
        results.append({
            'symbol': symbol,
            'dow': dow,
            'day_name': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow],
            'first_3h_range': round(avg_range, 2),
        })
        print(f"    {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]} 00-03 UTC: {avg_range:.2f} bps")

    # Monday 00:00 vs Sunday 21:00-23:00
    mon_start = df[(df['dow'] == 0) & (df['hour'] < 2)]['range_bps']
    sun_end = df[(df['dow'] == 6) & (df['hour'] >= 22)]['range_bps']
    if len(mon_start) > 100 and len(sun_end) > 100:
        t, p = stats.mannwhitneyu(mon_start, sun_end, alternative='two-sided')
        ratio = mon_start.mean() / sun_end.mean()
        print(f"    Mon 00-02 vs Sun 22-24: ratio={ratio:.3f}x, p={p:.4f}")

    return results


def analyze_weekend_gap(df, symbol):
    """Analyze the 'gap' between Friday close and Monday open."""
    print(f"\n  Weekend gap analysis ({symbol}):")

    # Get daily OHLC
    daily = df.groupby('date').agg(
        open=('open', 'first'),
        close=('close', 'last'),
        high=('high', 'max'),
        low=('low', 'min'),
        range_bps=('range_bps', 'mean'),
        dow=('dow', 'first'),
    ).reset_index()

    # Weekend gap: Monday open vs Friday close
    gaps = []
    for i in range(1, len(daily)):
        if daily.iloc[i]['dow'] == 0 and daily.iloc[i-1]['dow'] >= 4:
            # Find the last Friday
            fri_idx = i - 1
            while fri_idx >= 0 and daily.iloc[fri_idx]['dow'] != 4:
                fri_idx -= 1
            if fri_idx >= 0:
                fri_close = daily.iloc[fri_idx]['close']
                mon_open = daily.iloc[i]['open']
                gap_bps = (mon_open / fri_close - 1) * 10000
                gaps.append({
                    'date': daily.iloc[i]['date'],
                    'gap_bps': gap_bps,
                    'mon_range': daily.iloc[i]['range_bps'],
                })

    if gaps:
        gap_df = pd.DataFrame(gaps)
        print(f"    N weekends: {len(gap_df)}")
        print(f"    Avg gap: {gap_df['gap_bps'].mean():+.2f} bps")
        print(f"    Std gap: {gap_df['gap_bps'].std():.2f} bps")
        print(f"    Abs gap: {gap_df['gap_bps'].abs().mean():.2f} bps")

        # Does gap size predict Monday vol?
        rho, p = stats.spearmanr(gap_df['gap_bps'].abs(), gap_df['mon_range'])
        print(f"    |Gap| vs Monday range: ρ={rho:+.3f}, p={p:.4f}")

        # Gap-fill: does Monday close revert toward Friday close?
        # (negative correlation between gap and Monday return)
        return gap_df

    return pd.DataFrame()


def analyze_hourly_transitions(df, symbol):
    """Hourly vol profile around day boundaries for each transition."""
    results = []
    transitions = [
        ('Fri→Sat', 4, 5), ('Sat→Sun', 5, 6), ('Sun→Mon', 6, 0),
        ('Mon→Tue', 0, 1), ('Tue→Wed', 1, 2), ('Wed→Thu', 2, 3), ('Thu→Fri', 3, 4),
    ]

    for name, dow_before, dow_after in transitions:
        # Last 3 hours of day_before + first 3 hours of day_after
        before = df[(df['dow'] == dow_before) & (df['hour'] >= 21)]
        after = df[(df['dow'] == dow_after) & (df['hour'] < 3)]

        for h in range(21, 27):
            actual_h = h % 24
            if h < 24:
                sub = df[(df['dow'] == dow_before) & (df['hour'] == actual_h)]
            else:
                sub = df[(df['dow'] == dow_after) & (df['hour'] == actual_h)]

            if len(sub) > 0:
                results.append({
                    'symbol': symbol,
                    'transition': name,
                    'relative_hour': h - 24,  # -3 to +2
                    'actual_hour': actual_h,
                    'avg_range_bps': round(sub['range_bps'].mean(), 2),
                })

    return results


def main():
    t0 = time.time()
    print("="*70)
    print("v39: Weekend Gap / Monday Open Effect")
    print("="*70)

    all_transitions = []
    all_gaps = []
    all_hourly_trans = []

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)

        trans = analyze_day_transitions(df, symbol)
        all_transitions.extend(trans)

        gap_df = analyze_weekend_gap(df, symbol)
        if not gap_df.empty:
            gap_df['symbol'] = symbol
            all_gaps.append(gap_df)

        hourly = analyze_hourly_transitions(df, symbol)
        all_hourly_trans.extend(hourly)

    # Save CSVs
    pd.DataFrame(all_transitions).to_csv(RESULTS_DIR / 'v39_day_transitions.csv', index=False)
    print(f"\n  Saved: results/v39_day_transitions.csv")

    if all_gaps:
        gaps_df = pd.concat(all_gaps, ignore_index=True)
        gaps_df.to_csv(RESULTS_DIR / 'v39_weekend_gaps.csv', index=False)
        print(f"  Saved: results/v39_weekend_gaps.csv")

    pd.DataFrame(all_hourly_trans).to_csv(RESULTS_DIR / 'v39_hourly_transitions.csv', index=False)
    print(f"  Saved: results/v39_hourly_transitions.csv")

    # ---- PLOT 1: First 3h range by day of week ----
    fig, ax = plt.subplots(figsize=(10, 5))
    trans_df = pd.DataFrame(all_transitions)
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    x = np.arange(7)
    width = 0.15
    for i, (sym, color) in enumerate(zip(SYMBOLS, colors)):
        vals = trans_df[trans_df['symbol'] == sym].sort_values('dow')['first_3h_range'].values
        ax.bar(x + i*width - 0.3, vals, width, label=sym.replace('USDT',''), color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_ylabel('Avg Range (bps) — first 3 hours', fontsize=11)
    ax.set_title('v39: "Opening" Volatility by Day (00:00-03:00 UTC)\n3+ years, 5 symbols', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v39_opening_vol_by_day.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v39_opening_vol_by_day.png")

    # ---- PLOT 2: Weekend gap distribution (BTC) ----
    if all_gaps:
        btc_gaps = gaps_df[gaps_df['symbol'] == 'BTCUSDT']['gap_bps']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(btc_gaps, bins=50, color='#1E88E5', alpha=0.7, edgecolor='white')
        ax.axvline(x=0, color='red', linewidth=1.5, linestyle='--')
        ax.axvline(x=btc_gaps.mean(), color='green', linewidth=1.5, linestyle='-',
                   label=f'Mean: {btc_gaps.mean():+.1f} bps')
        ax.set_xlabel('Weekend Gap (bps): Monday Open - Friday Close', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('v39: BTC Weekend Gap Distribution\n(3+ years)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'v39_weekend_gap_dist.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: results/v39_weekend_gap_dist.png")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
