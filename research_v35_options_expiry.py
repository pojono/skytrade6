#!/usr/bin/env python3
"""
v35: Options Expiry / Derivatives Calendar Effects

Crypto options expire weekly (Fridays) and monthly (last Friday of month) on Deribit.
- Is Friday volatility different from other weekdays?
- Is last-Friday-of-month different from regular Fridays?
- Pre-expiry (Thursday) vs post-expiry (Monday) vol comparison
- Quarterly expiry effects (last Friday of Mar/Jun/Sep/Dec)

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
from datetime import timedelta

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
    df['date'] = df['datetime'].dt.date
    df['dow'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def tag_expiry_dates(df):
    """Tag each bar with expiry type."""
    dates = pd.Series(df['date'].unique())
    dates_dt = pd.to_datetime(dates)

    # Find last Friday of each month
    monthly_expiries = set()
    for year in dates_dt.dt.year.unique():
        for month in range(1, 13):
            # Last day of month
            if month == 12:
                last_day = pd.Timestamp(year=year+1, month=1, day=1) - timedelta(days=1)
            else:
                last_day = pd.Timestamp(year=year, month=month+1, day=1) - timedelta(days=1)
            # Walk back to Friday (dow=4)
            while last_day.dayofweek != 4:
                last_day -= timedelta(days=1)
            monthly_expiries.add(last_day.date())

    # Quarterly expiries (subset of monthly: Mar, Jun, Sep, Dec)
    quarterly_expiries = {d for d in monthly_expiries
                          if pd.Timestamp(d).month in [3, 6, 9, 12]}

    # All Fridays
    all_fridays = set(dates[dates_dt.dt.dayofweek == 4].values)

    # Tag
    df['is_friday'] = df['dow'] == 4
    df['is_monthly_expiry'] = df['date'].isin(monthly_expiries)
    df['is_quarterly_expiry'] = df['date'].isin(quarterly_expiries)
    df['is_regular_friday'] = df['is_friday'] & ~df['is_monthly_expiry']

    # Pre-expiry (Thursday before Friday expiry)
    pre_expiry = {d - timedelta(days=1) for d in monthly_expiries}
    df['is_pre_monthly_expiry'] = df['date'].isin(pre_expiry)

    # Post-expiry (Monday after Friday expiry)
    post_expiry = {d + timedelta(days=3) for d in monthly_expiries}
    df['is_post_monthly_expiry'] = df['date'].isin(post_expiry)

    # Day type
    df['day_type'] = 'other_weekday'
    df.loc[df['dow'] >= 5, 'day_type'] = 'weekend'
    df.loc[df['is_regular_friday'], 'day_type'] = 'regular_friday'
    df.loc[df['is_monthly_expiry'] & ~df['is_quarterly_expiry'], 'day_type'] = 'monthly_expiry_fri'
    df.loc[df['is_quarterly_expiry'], 'day_type'] = 'quarterly_expiry_fri'
    df.loc[df['is_pre_monthly_expiry'], 'day_type'] = 'pre_monthly_expiry'
    df.loc[df['is_post_monthly_expiry'], 'day_type'] = 'post_monthly_expiry'

    return df


def main():
    t0 = time.time()
    print("="*70)
    print("v35: Options Expiry / Derivatives Calendar Effects")
    print("="*70)

    all_results = []
    all_hourly = []

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)
        df = tag_expiry_dates(df)

        print(f"\n  Day type distribution:")
        for dt in ['other_weekday', 'regular_friday', 'monthly_expiry_fri',
                    'quarterly_expiry_fri', 'pre_monthly_expiry', 'post_monthly_expiry', 'weekend']:
            n = (df['day_type'] == dt).sum()
            avg = df.loc[df['day_type'] == dt, 'range_bps'].mean()
            print(f"    {dt:>25s}: {n:>7,} bars, avg range = {avg:.2f} bps")

        # Statistical tests
        weekday_non_fri = df[(df['dow'] < 4)]['range_bps']
        regular_fri = df[df['day_type'] == 'regular_friday']['range_bps']
        monthly_fri = df[df['day_type'] == 'monthly_expiry_fri']['range_bps']
        quarterly_fri = df[df['day_type'] == 'quarterly_expiry_fri']['range_bps']

        # Friday vs other weekdays
        t_fri, p_fri = stats.mannwhitneyu(regular_fri, weekday_non_fri, alternative='two-sided')
        print(f"\n  Regular Friday vs Mon-Thu: U={t_fri:.0f}, p={p_fri:.4f}, "
              f"ratio={regular_fri.mean()/weekday_non_fri.mean():.3f}x")

        # Monthly expiry vs regular Friday
        if len(monthly_fri) > 100:
            t_me, p_me = stats.mannwhitneyu(monthly_fri, regular_fri, alternative='two-sided')
            print(f"  Monthly expiry Fri vs regular Fri: U={t_me:.0f}, p={p_me:.4f}, "
                  f"ratio={monthly_fri.mean()/regular_fri.mean():.3f}x")

        # Quarterly expiry vs regular Friday
        if len(quarterly_fri) > 100:
            t_qe, p_qe = stats.mannwhitneyu(quarterly_fri, regular_fri, alternative='two-sided')
            print(f"  Quarterly expiry Fri vs regular Fri: U={t_qe:.0f}, p={p_qe:.4f}, "
                  f"ratio={quarterly_fri.mean()/regular_fri.mean():.3f}x")

        # Pre vs post monthly expiry
        pre = df[df['day_type'] == 'pre_monthly_expiry']['range_bps']
        post = df[df['day_type'] == 'post_monthly_expiry']['range_bps']
        if len(pre) > 100 and len(post) > 100:
            t_pp, p_pp = stats.mannwhitneyu(pre, post, alternative='two-sided')
            print(f"  Pre-expiry Thu vs Post-expiry Mon: U={t_pp:.0f}, p={p_pp:.4f}, "
                  f"ratio={pre.mean()/post.mean():.3f}x")

        # Collect results
        for dt in df['day_type'].unique():
            sub = df[df['day_type'] == dt]
            all_results.append({
                'symbol': symbol, 'day_type': dt,
                'n_bars': len(sub),
                'avg_range_bps': round(sub['range_bps'].mean(), 2),
                'median_range_bps': round(sub['range_bps'].median(), 2),
                'avg_volume': round(sub['quote_volume'].mean(), 0),
            })

        # Hourly profile on expiry vs non-expiry Fridays
        for dt_name, mask in [('regular_friday', df['day_type'] == 'regular_friday'),
                               ('monthly_expiry', df['is_monthly_expiry']),
                               ('other_weekday', df['dow'] < 4)]:
            sub = df[mask]
            for h in range(24):
                hsub = sub[sub['hour'] == h]
                if len(hsub) > 0:
                    all_hourly.append({
                        'symbol': symbol, 'day_type': dt_name, 'hour': h,
                        'avg_range_bps': round(hsub['range_bps'].mean(), 2),
                    })

    # Save CSVs
    pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'v35_expiry_effects.csv', index=False)
    print(f"\n  Saved: results/v35_expiry_effects.csv")

    pd.DataFrame(all_hourly).to_csv(RESULTS_DIR / 'v35_expiry_hourly.csv', index=False)
    print(f"  Saved: results/v35_expiry_hourly.csv")

    # ---- PLOT 1: Day type comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    res_df = pd.DataFrame(all_results)
    order = ['other_weekday', 'regular_friday', 'pre_monthly_expiry',
             'monthly_expiry_fri', 'quarterly_expiry_fri', 'post_monthly_expiry', 'weekend']
    colors_sym = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    x = np.arange(len(order))
    width = 0.15
    for i, (sym, color) in enumerate(zip(SYMBOLS, colors_sym)):
        vals = []
        for dt in order:
            sub = res_df[(res_df['symbol'] == sym) & (res_df['day_type'] == dt)]
            vals.append(sub['avg_range_bps'].values[0] if len(sub) > 0 else 0)
        ax.bar(x + i*width - 0.3, vals, width, label=sym.replace('USDT',''), color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([o.replace('_', '\n') for o in order], fontsize=8)
    ax.set_ylabel('Avg 5-min Range (bps)', fontsize=11)
    ax.set_title('v35: Volatility by Day Type (Options Expiry Effects)\n3+ years, 5 symbols', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v35_expiry_day_types.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v35_expiry_day_types.png")

    # ---- PLOT 2: Hourly profile on expiry vs non-expiry (BTC) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_df = pd.DataFrame(all_hourly)
    btc_h = hourly_df[hourly_df['symbol'] == 'BTCUSDT']

    for dt_name, color, ls in [('other_weekday', '#666666', '-'),
                                ('regular_friday', '#1E88E5', '-'),
                                ('monthly_expiry', '#E53935', '--')]:
        sub = btc_h[btc_h['day_type'] == dt_name].sort_values('hour')
        ax.plot(sub['hour'], sub['avg_range_bps'], label=dt_name.replace('_',' '),
                color=color, linewidth=2, linestyle=ls)

    ax.set_xlabel('Hour (UTC)', fontsize=11)
    ax.set_ylabel('Avg Range (bps)', fontsize=11)
    ax.set_title('v35: BTC Hourly Vol Profile — Expiry vs Non-Expiry Fridays\n(3+ years)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(24))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v35_expiry_hourly_btc.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v35_expiry_hourly_btc.png")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
