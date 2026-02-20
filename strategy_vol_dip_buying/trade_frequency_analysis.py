#!/usr/bin/env python3
"""
Trade Frequency Analysis for Vol Dip-Buying Strategy.

Computes trades per day, week, and month for all 20 symbols.
Shows min/max/avg/median and distribution.
"""

import sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path('parquet')
OUT_DIR = Path('strategy_vol_dip_buying')
CHART_DIR = OUT_DIR / 'charts'

RT_FEE_BPS = 4.0
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4

ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
]

TIER_A = ["SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "UNIUSDT", "SUIUSDT"]


def load_1h(symbol):
    d = PARQUET_DIR / symbol / 'ohlcv' / '1h' / 'bybit_futures'
    if not d.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    return raw[~raw.index.duplicated(keep='first')]


def compute_signals(df):
    c = df['close'].values.astype(np.float64)
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret, index=df.index)

    rvol = ret_s.rolling(24, min_periods=8).std()
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    df['rvol_z'] = ((rvol - rvol_mean) / rvol_std).values

    r4 = ret_s.rolling(4).sum()
    r4_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    r4_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    df['mr_4h'] = -((r4 - r4_mean) / r4_std).values

    df['combined'] = (df['rvol_z'].values + df['mr_4h'].values) / 2
    return df


def get_trades(df):
    sig = df['combined'].values
    c = df['close'].values.astype(np.float64)
    n = len(c)
    trades = []
    last_exit = 0

    for i in range(0, n - HOLD_BARS):
        if i < last_exit + COOLDOWN_BARS:
            continue
        if np.isnan(sig[i]) or abs(sig[i]) < THRESHOLD:
            continue

        trade_dir = 'long' if sig[i] > 0 else 'short'
        entry = c[i]
        exit_p = c[i + HOLD_BARS]
        raw_bps = ((exit_p - entry) / entry * 10000) if trade_dir == 'long' else \
                  ((entry - exit_p) / entry * 10000)

        trades.append({
            'entry_time': df.index[i],
            'dir': trade_dir,
            'net_bps': raw_bps - RT_FEE_BPS,
        })
        last_exit = i + HOLD_BARS

    return trades


def main():
    t0 = time.time()
    print("=" * 100)
    print("TRADE FREQUENCY ANALYSIS — Vol Dip-Buying, All 20 Symbols")
    print("=" * 100)

    all_results = []

    for si, symbol in enumerate(ALL_SYMBOLS, 1):
        df = load_1h(symbol)
        if df.empty or len(df) < 2000:
            print(f"[{si:2d}/20] {symbol:12s} — SKIP")
            continue

        df = compute_signals(df)
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())
        warmup_end = months[5] if len(months) > 5 else months[-1]
        trade_df = df[df['month'] > warmup_end]
        trades = get_trades(trade_df)

        if not trades:
            print(f"[{si:2d}/20] {symbol:12s} — NO TRADES")
            continue

        tdf = pd.DataFrame(trades)
        tdf['date'] = tdf['entry_time'].dt.date
        tdf['week'] = tdf['entry_time'].dt.isocalendar().week.values.astype(int)
        tdf['year_week'] = tdf['entry_time'].dt.strftime('%Y-W%U')
        tdf['month'] = tdf['entry_time'].dt.to_period('M')

        # Trades per day (only days with trades)
        daily = tdf.groupby('date').size()
        # But we also need total calendar days to get avg including zero-trade days
        first_trade = tdf['entry_time'].min()
        last_trade = tdf['entry_time'].max()
        total_days = (last_trade - first_trade).days + 1
        total_weeks = total_days / 7
        total_months = len(tdf['month'].unique())

        # Weekly counts (including zero weeks)
        weekly = tdf.groupby('year_week').size()

        # Monthly counts
        monthly = tdf.groupby('month').size()

        # Per-day stats (including zero-trade days)
        avg_per_day = len(trades) / total_days
        avg_per_week = len(trades) / total_weeks
        avg_per_month = len(trades) / total_months

        # Gap analysis: days between consecutive trades
        trade_times = sorted(tdf['entry_time'].values)
        gaps_hours = []
        for i in range(1, len(trade_times)):
            gap = (trade_times[i] - trade_times[i-1]) / np.timedelta64(1, 'h')
            gaps_hours.append(gap)
        gaps_days = [g / 24 for g in gaps_hours]

        tier = 'A' if symbol in TIER_A else 'B/C'

        result = {
            'symbol': symbol,
            'tier': tier,
            'total_trades': len(trades),
            'total_days': total_days,
            'total_months': total_months,
            # Daily (including zero days)
            'avg_per_day': avg_per_day,
            'max_per_day': daily.max(),
            # Weekly
            'avg_per_week': avg_per_week,
            'min_per_week': weekly.min(),
            'max_per_week': weekly.max(),
            'median_per_week': weekly.median(),
            # Monthly
            'avg_per_month': avg_per_month,
            'min_per_month': monthly.min(),
            'max_per_month': monthly.max(),
            'median_per_month': monthly.median(),
            # Gaps
            'avg_gap_days': np.mean(gaps_days) if gaps_days else 0,
            'min_gap_hours': min(gaps_hours) if gaps_hours else 0,
            'max_gap_days': max(gaps_days) if gaps_days else 0,
            'median_gap_days': np.median(gaps_days) if gaps_days else 0,
            # Zero-trade months
            'zero_trade_months': total_months - len(monthly),
            # Long %
            'long_pct': sum(1 for t in trades if t['dir'] == 'long') / len(trades) * 100,
        }
        all_results.append(result)

        print(f"[{si:2d}/20] {symbol:12s} ({tier:3s}) | "
              f"{len(trades):4d} trades over {total_days} days | "
              f"day: {avg_per_day:.2f} avg, {daily.max()} max | "
              f"week: {avg_per_week:.1f} avg | "
              f"month: {avg_per_month:.1f} avg ({monthly.min()}-{monthly.max()})")

    # ================================================================
    # SUMMARY TABLES
    # ================================================================
    print(f"\n{'='*100}")
    print("  TRADES PER DAY")
    print(f"{'='*100}")
    print(f"  {'Symbol':12s} {'Tier':>4s}  {'Total':>6s}  {'Avg/day':>8s}  {'Max/day':>8s}  "
          f"{'Days w/ trades':>14s}  {'% days active':>13s}")
    for r in sorted(all_results, key=lambda x: x['avg_per_day'], reverse=True):
        days_active = r['total_trades']  # approx, since max 1 trade per 8h
        pct_active = r['avg_per_day'] * 100  # rough
        print(f"  {r['symbol']:12s} {r['tier']:>4s}  {r['total_trades']:>5d}  "
              f"{r['avg_per_day']:>7.2f}  {r['max_per_day']:>7d}  "
              f"{r['avg_per_day']*r['total_days']:>13.0f}  "
              f"{r['avg_per_day']*100:>12.1f}%")

    print(f"\n{'='*100}")
    print("  TRADES PER WEEK")
    print(f"{'='*100}")
    print(f"  {'Symbol':12s} {'Tier':>4s}  {'Avg':>6s}  {'Min':>4s}  {'Med':>4s}  {'Max':>4s}")
    for r in sorted(all_results, key=lambda x: x['avg_per_week'], reverse=True):
        print(f"  {r['symbol']:12s} {r['tier']:>4s}  {r['avg_per_week']:>5.1f}  "
              f"{r['min_per_week']:>3.0f}  {r['median_per_week']:>3.0f}  "
              f"{r['max_per_week']:>3.0f}")

    print(f"\n{'='*100}")
    print("  TRADES PER MONTH")
    print(f"{'='*100}")
    print(f"  {'Symbol':12s} {'Tier':>4s}  {'Avg':>6s}  {'Min':>4s}  {'Med':>4s}  {'Max':>4s}  "
          f"{'Zero months':>11s}")
    for r in sorted(all_results, key=lambda x: x['avg_per_month'], reverse=True):
        print(f"  {r['symbol']:12s} {r['tier']:>4s}  {r['avg_per_month']:>5.1f}  "
              f"{r['min_per_month']:>3.0f}  {r['median_per_month']:>3.0f}  "
              f"{r['max_per_month']:>3.0f}  "
              f"{r['zero_trade_months']:>10d}")

    print(f"\n{'='*100}")
    print("  GAP BETWEEN TRADES (DAYS)")
    print(f"{'='*100}")
    print(f"  {'Symbol':12s} {'Tier':>4s}  {'Avg gap':>8s}  {'Med gap':>8s}  {'Min gap':>8s}  {'Max gap':>8s}")
    for r in sorted(all_results, key=lambda x: x['avg_gap_days']):
        print(f"  {r['symbol']:12s} {r['tier']:>4s}  {r['avg_gap_days']:>7.1f}d  "
              f"{r['median_gap_days']:>7.1f}d  {r['min_gap_hours']:>6.0f}h  "
              f"{r['max_gap_days']:>7.1f}d")

    # ================================================================
    # PORTFOLIO-LEVEL FREQUENCY
    # ================================================================
    print(f"\n{'='*100}")
    print("  PORTFOLIO-LEVEL TRADE FREQUENCY")
    print(f"{'='*100}")

    # Tier A portfolio
    tier_a_results = [r for r in all_results if r['tier'] == 'A']
    all_20_results = all_results

    for label, subset in [("Tier A (6 coins)", tier_a_results), ("All 20 coins", all_20_results)]:
        total_trades = sum(r['total_trades'] for r in subset)
        avg_days = np.mean([r['total_days'] for r in subset])
        portfolio_per_day = sum(r['avg_per_day'] for r in subset)
        portfolio_per_week = sum(r['avg_per_week'] for r in subset)
        portfolio_per_month = sum(r['avg_per_month'] for r in subset)

        print(f"\n  --- {label} ---")
        print(f"  Total trades (all symbols): {total_trades:,}")
        print(f"  Portfolio trades/day:        {portfolio_per_day:.2f}")
        print(f"  Portfolio trades/week:       {portfolio_per_week:.1f}")
        print(f"  Portfolio trades/month:      {portfolio_per_month:.1f}")
        print(f"  Avg time between trades:     {1/portfolio_per_day:.1f} days" if portfolio_per_day > 0 else "")

    # ================================================================
    # CHART: Monthly trade counts heatmap
    # ================================================================
    print("\nGenerating charts...")

    # Bar chart: avg trades per month by symbol
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: trades per month
    ax = axes[0]
    symbols = [r['symbol'].replace('USDT', '') for r in sorted(all_results, key=lambda x: x['avg_per_month'], reverse=True)]
    avgs = [r['avg_per_month'] for r in sorted(all_results, key=lambda x: x['avg_per_month'], reverse=True)]
    mins = [r['min_per_month'] for r in sorted(all_results, key=lambda x: x['avg_per_month'], reverse=True)]
    maxs = [r['max_per_month'] for r in sorted(all_results, key=lambda x: x['avg_per_month'], reverse=True)]
    tiers = [r['tier'] for r in sorted(all_results, key=lambda x: x['avg_per_month'], reverse=True)]

    colors = ['#2ecc71' if t == 'A' else '#95a5a6' for t in tiers]
    bars = ax.barh(range(len(symbols)), avgs, color=colors, alpha=0.8)

    # Error bars for min/max
    for i, (mn, mx, avg) in enumerate(zip(mins, maxs, avgs)):
        ax.plot([mn, mx], [i, i], color='#2c3e50', linewidth=1.5)
        ax.plot([mn], [i], 'o', color='#e74c3c', markersize=4)
        ax.plot([mx], [i], 'o', color='#27ae60', markersize=4)

    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_xlabel('Trades per Month')
    ax.set_title('Avg Trades/Month (green=Tier A)\nDots: min/max range', fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Right: gap between trades
    ax = axes[1]
    symbols2 = [r['symbol'].replace('USDT', '') for r in sorted(all_results, key=lambda x: x['avg_gap_days'])]
    avg_gaps = [r['avg_gap_days'] for r in sorted(all_results, key=lambda x: x['avg_gap_days'])]
    med_gaps = [r['median_gap_days'] for r in sorted(all_results, key=lambda x: x['avg_gap_days'])]
    max_gaps = [r['max_gap_days'] for r in sorted(all_results, key=lambda x: x['avg_gap_days'])]
    tiers2 = [r['tier'] for r in sorted(all_results, key=lambda x: x['avg_gap_days'])]

    colors2 = ['#2ecc71' if t == 'A' else '#95a5a6' for t in tiers2]
    ax.barh(range(len(symbols2)), avg_gaps, color=colors2, alpha=0.8, label='Avg gap')
    ax.barh(range(len(symbols2)), med_gaps, color=colors2, alpha=0.4, label='Median gap')

    ax.set_yticks(range(len(symbols2)))
    ax.set_yticklabels(symbols2, fontsize=9)
    ax.set_xlabel('Days Between Trades')
    ax.set_title('Avg Gap Between Trades (days)\nLighter = median', fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'trade_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'trade_frequency.png'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
