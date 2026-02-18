#!/usr/bin/env python3
"""
v41: Intraday VWAP Reversion

Does price tend to revert to intraday VWAP?
- Compute cumulative VWAP within each day
- Measure deviation from VWAP at each bar
- Test: does VWAP deviation predict next-bar return direction?
- Analyze VWAP deviation by hour and session

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
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def compute_intraday_vwap(df):
    """Compute cumulative intraday VWAP (resets daily)."""
    df = df.copy()
    df['tp_x_vol'] = df['typical_price'] * df['quote_volume']

    # Cumulative within each day
    df['cum_tp_vol'] = df.groupby('date')['tp_x_vol'].cumsum()
    df['cum_vol'] = df.groupby('date')['quote_volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol'].clip(lower=1e-10)

    # Deviation from VWAP in bps
    df['vwap_dev_bps'] = (df['close'] - df['vwap']) / df['vwap'] * 10000

    # Forward return (next bar)
    df['fwd_ret_bps'] = (df['close'].shift(-1) / df['close'] - 1) * 10000

    # Forward 6-bar return (30 min)
    df['fwd_ret_30m_bps'] = (df['close'].shift(-6) / df['close'] - 1) * 10000

    return df


def analyze_vwap_reversion(df, symbol):
    """Test if VWAP deviation predicts mean reversion."""
    print(f"\n  VWAP Reversion Analysis ({symbol}):")

    # Drop first bar of each day (VWAP = price, no deviation)
    valid = df.dropna(subset=['vwap_dev_bps', 'fwd_ret_bps']).copy()
    valid = valid[valid.groupby('date').cumcount() > 0]

    # Correlation: VWAP deviation vs forward return
    rho_1, p_1 = stats.spearmanr(valid['vwap_dev_bps'], valid['fwd_ret_bps'])
    valid_30m = valid[['vwap_dev_bps', 'fwd_ret_30m_bps']].dropna()
    rho_6, p_6 = stats.spearmanr(valid_30m['vwap_dev_bps'], valid_30m['fwd_ret_30m_bps'])

    print(f"    VWAP dev vs next-bar return: ρ={rho_1:+.4f}, p={p_1:.2e}")
    print(f"    VWAP dev vs 30min return:    ρ={rho_6:+.4f}, p={p_6:.2e}")

    if rho_1 < 0:
        print(f"    → MEAN REVERSION confirmed (negative correlation)")
    else:
        print(f"    → MOMENTUM (positive correlation)")

    # Quintile analysis
    valid['dev_quintile'] = pd.qcut(valid['vwap_dev_bps'], 5, labels=[1,2,3,4,5])
    quintile_stats = valid.groupby('dev_quintile').agg(
        avg_dev=('vwap_dev_bps', 'mean'),
        avg_fwd_ret=('fwd_ret_bps', 'mean'),
        avg_fwd_30m=('fwd_ret_30m_bps', 'mean'),
        n=('fwd_ret_bps', 'count'),
    ).reset_index()

    print(f"\n    Quintile analysis:")
    print(f"    {'Q':>3s}  {'Avg Dev':>10s}  {'Fwd 5m':>10s}  {'Fwd 30m':>10s}  {'N':>8s}")
    for _, row in quintile_stats.iterrows():
        print(f"    {int(row['dev_quintile']):>3d}  {row['avg_dev']:>+10.2f}  "
              f"{row['avg_fwd_ret']:>+10.3f}  {row['avg_fwd_30m']:>+10.3f}  {int(row['n']):>8,}")

    # Spread: Q1 - Q5 return
    q1_ret = quintile_stats[quintile_stats['dev_quintile'] == 1]['avg_fwd_ret'].values[0]
    q5_ret = quintile_stats[quintile_stats['dev_quintile'] == 5]['avg_fwd_ret'].values[0]
    spread = q1_ret - q5_ret
    print(f"    Q1-Q5 spread (5min): {spread:+.3f} bps")

    return {
        'symbol': symbol,
        'rho_5min': round(rho_1, 4), 'p_5min': p_1,
        'rho_30min': round(rho_6, 4), 'p_30min': p_6,
        'q1_q5_spread_5min': round(spread, 3),
    }, quintile_stats


def analyze_vwap_by_hour(df, symbol):
    """How does VWAP deviation magnitude vary by hour?"""
    valid = df.dropna(subset=['vwap_dev_bps']).copy()
    valid = valid[valid.groupby('date').cumcount() > 0]

    results = []
    for h in range(24):
        sub = valid[valid['hour'] == h]
        if len(sub) > 100:
            results.append({
                'symbol': symbol, 'hour': h,
                'avg_abs_dev_bps': round(sub['vwap_dev_bps'].abs().mean(), 2),
                'std_dev_bps': round(sub['vwap_dev_bps'].std(), 2),
                'reversion_corr': round(stats.spearmanr(sub['vwap_dev_bps'], sub['fwd_ret_bps'])[0], 4)
                    if len(sub) > 100 else np.nan,
            })
    return results


def main():
    t0 = time.time()
    print("="*70)
    print("v41: Intraday VWAP Reversion")
    print("="*70)

    all_summary = []
    all_quintiles = []
    all_hourly = []

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)
        df = compute_intraday_vwap(df)

        summary, quintiles = analyze_vwap_reversion(df, symbol)
        all_summary.append(summary)
        quintiles['symbol'] = symbol
        all_quintiles.append(quintiles)

        hourly = analyze_vwap_by_hour(df, symbol)
        all_hourly.extend(hourly)

    # Save CSVs
    pd.DataFrame(all_summary).to_csv(RESULTS_DIR / 'v41_vwap_reversion_summary.csv', index=False)
    pd.concat(all_quintiles).to_csv(RESULTS_DIR / 'v41_vwap_quintiles.csv', index=False)
    pd.DataFrame(all_hourly).to_csv(RESULTS_DIR / 'v41_vwap_hourly.csv', index=False)
    print(f"\n  Saved: 3 CSVs")

    # ---- PLOT 1: Quintile returns ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    quint_df = pd.concat(all_quintiles)

    for sym, color in zip(SYMBOLS, colors):
        sub = quint_df[quint_df['symbol'] == sym].sort_values('dev_quintile')
        ax1.plot(sub['dev_quintile'].astype(int), sub['avg_fwd_ret'],
                 label=sym.replace('USDT',''), color=color, linewidth=2, marker='o')
        ax2.plot(sub['dev_quintile'].astype(int), sub['avg_fwd_30m'],
                 label=sym.replace('USDT',''), color=color, linewidth=2, marker='o')

    for ax, title, ylabel in [(ax1, 'Next 5-min Return by VWAP Deviation Quintile', 'Avg Forward Return (bps)'),
                               (ax2, 'Next 30-min Return by VWAP Deviation Quintile', 'Avg Forward Return (bps)')]:
        ax.axhline(y=0, color='gray', linewidth=1, linestyle='--')
        ax.set_xlabel('VWAP Deviation Quintile (1=below, 5=above)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1,2,3,4,5])

    plt.suptitle('v41: VWAP Mean Reversion Analysis (3+ years)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v41_vwap_quintile_returns.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v41_vwap_quintile_returns.png")

    # ---- PLOT 2: Hourly reversion strength ----
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_df = pd.DataFrame(all_hourly)

    for sym, color in zip(SYMBOLS, colors):
        sub = hourly_df[hourly_df['symbol'] == sym].sort_values('hour')
        ax.plot(sub['hour'], sub['reversion_corr'], label=sym.replace('USDT',''),
                color=color, linewidth=1.5, marker='o', markersize=3)

    ax.axhline(y=0, color='gray', linewidth=1, linestyle='--')
    ax.set_xlabel('Hour (UTC)', fontsize=11)
    ax.set_ylabel('VWAP Dev vs Fwd Return Correlation', fontsize=11)
    ax.set_title('v41: VWAP Reversion Strength by Hour\n(negative = mean reversion)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(24))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v41_vwap_reversion_by_hour.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v41_vwap_reversion_by_hour.png")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for s in all_summary:
        print(f"  {s['symbol']:>10s}: ρ(5m)={s['rho_5min']:+.4f}, ρ(30m)={s['rho_30min']:+.4f}, "
              f"Q1-Q5={s['q1_q5_spread_5min']:+.3f} bps")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
