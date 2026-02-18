#!/usr/bin/env python3
"""
v40: Calendar Anomalies

- Turn-of-month effect (last 3 days + first 3 days vs mid-month)
- Quarter-end rebalancing (last week of Mar/Jun/Sep/Dec)
- Day-of-month profile (1-31)
- Week-of-month effect

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
import calendar

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
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dow'] = df['datetime'].dt.dayofweek

    # Days from month end
    df['days_in_month'] = df['datetime'].apply(lambda x: calendar.monthrange(x.year, x.month)[1])
    df['days_from_end'] = df['days_in_month'] - df['day_of_month']

    # Turn of month: last 3 days or first 3 days
    df['is_turn_of_month'] = (df['day_of_month'] <= 3) | (df['days_from_end'] < 3)
    df['is_mid_month'] = (df['day_of_month'] > 7) & (df['days_from_end'] >= 7)

    # Quarter end: last 5 trading days of Mar/Jun/Sep/Dec
    df['is_quarter_end_month'] = df['month'].isin([3, 6, 9, 12])
    df['is_quarter_end_week'] = df['is_quarter_end_month'] & (df['days_from_end'] < 5)

    # Week of month (1-5)
    df['week_of_month'] = ((df['day_of_month'] - 1) // 7) + 1

    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def main():
    t0 = time.time()
    print("="*70)
    print("v40: Calendar Anomalies")
    print("="*70)

    all_dom = []  # day of month
    all_tom = []  # turn of month
    all_wom = []  # week of month
    all_qe = []   # quarter end

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)

        # Day of month profile
        print(f"  Day-of-month profile:")
        for d in range(1, 32):
            sub = df[df['day_of_month'] == d]
            if len(sub) > 100:
                all_dom.append({
                    'symbol': symbol, 'day_of_month': d,
                    'avg_range_bps': round(sub['range_bps'].mean(), 2),
                    'n_bars': len(sub),
                })

        # Turn of month vs mid-month
        tom = df[df['is_turn_of_month']]['range_bps']
        mid = df[df['is_mid_month']]['range_bps']
        t_stat, p_val = stats.mannwhitneyu(tom, mid, alternative='two-sided')
        ratio = tom.mean() / mid.mean()
        print(f"  Turn-of-month vs mid-month: ratio={ratio:.3f}x, p={p_val:.4f}")
        all_tom.append({
            'symbol': symbol,
            'turn_of_month_range': round(tom.mean(), 2),
            'mid_month_range': round(mid.mean(), 2),
            'ratio': round(ratio, 3),
            'p_value': p_val,
        })

        # Week of month
        print(f"  Week-of-month profile:")
        for w in range(1, 6):
            sub = df[df['week_of_month'] == w]
            if len(sub) > 100:
                all_wom.append({
                    'symbol': symbol, 'week': w,
                    'avg_range_bps': round(sub['range_bps'].mean(), 2),
                    'n_bars': len(sub),
                })
                print(f"    Week {w}: {sub['range_bps'].mean():.2f} bps (n={len(sub):,})")

        # Quarter end
        qe = df[df['is_quarter_end_week']]['range_bps']
        non_qe = df[~df['is_quarter_end_week'] & (df['dow'] < 5)]['range_bps']
        if len(qe) > 100:
            t_stat, p_val = stats.mannwhitneyu(qe, non_qe, alternative='two-sided')
            ratio = qe.mean() / non_qe.mean()
            print(f"  Quarter-end week vs normal: ratio={ratio:.3f}x, p={p_val:.4f}")
            all_qe.append({
                'symbol': symbol,
                'quarter_end_range': round(qe.mean(), 2),
                'normal_range': round(non_qe.mean(), 2),
                'ratio': round(ratio, 3),
                'p_value': p_val,
            })

    # Save CSVs
    pd.DataFrame(all_dom).to_csv(RESULTS_DIR / 'v40_day_of_month.csv', index=False)
    pd.DataFrame(all_tom).to_csv(RESULTS_DIR / 'v40_turn_of_month.csv', index=False)
    pd.DataFrame(all_wom).to_csv(RESULTS_DIR / 'v40_week_of_month.csv', index=False)
    pd.DataFrame(all_qe).to_csv(RESULTS_DIR / 'v40_quarter_end.csv', index=False)
    print(f"\n  Saved: 4 CSVs")

    # ---- PLOT 1: Day of month profile ----
    fig, ax = plt.subplots(figsize=(14, 5))
    dom_df = pd.DataFrame(all_dom)
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    for sym, color in zip(SYMBOLS, colors):
        sub = dom_df[dom_df['symbol'] == sym].sort_values('day_of_month')
        # Normalize to mean=100
        mean_val = sub['avg_range_bps'].mean()
        ax.plot(sub['day_of_month'], sub['avg_range_bps'] / mean_val * 100,
                label=sym.replace('USDT',''), color=color, linewidth=1.5, marker='o', markersize=3)

    ax.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvspan(0.5, 3.5, alpha=0.1, color='green', label='Turn of month')
    ax.axvspan(28.5, 31.5, alpha=0.1, color='green')
    ax.set_xlabel('Day of Month', fontsize=11)
    ax.set_ylabel('Normalized Range (100 = avg)', fontsize=11)
    ax.set_title('v40: Volatility by Day of Month\n(normalized per symbol, 3+ years)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 32))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v40_day_of_month.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v40_day_of_month.png")

    # ---- PLOT 2: Week of month ----
    fig, ax = plt.subplots(figsize=(8, 5))
    wom_df = pd.DataFrame(all_wom)

    x = np.arange(1, 6)
    width = 0.15
    for i, (sym, color) in enumerate(zip(SYMBOLS, colors)):
        sub = wom_df[wom_df['symbol'] == sym].sort_values('week')
        if len(sub) == 5:
            ax.bar(x + i*width - 0.3, sub['avg_range_bps'].values, width,
                   label=sym.replace('USDT',''), color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Week 1\n(1-7)', 'Week 2\n(8-14)', 'Week 3\n(15-21)',
                         'Week 4\n(22-28)', 'Week 5\n(29-31)'])
    ax.set_ylabel('Avg Range (bps)', fontsize=11)
    ax.set_title('v40: Volatility by Week of Month\n(3+ years, 5 symbols)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v40_week_of_month.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v40_week_of_month.png")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
