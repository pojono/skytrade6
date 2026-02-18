#!/usr/bin/env python3
"""
v33b: Temporal Patterns — Large-Sample Confirmation from 3-Year OHLCV Parquet

Confirms v33 findings (hour-of-day, day-of-week, session, funding cycle)
using 3+ years of 5-min OHLCV bars built from Bybit futures trades.

Available metrics (no OI/FR/liqs — trades-only parquet):
  - Realized volatility (log-return std)
  - Price range (high-low / close)
  - Absolute return
  - Quote volume ($ turnover)
  - Trade count
  - Buy ratio (buy_volume / total_volume)

NEW vs v33:
  - Month-of-year analysis
  - Year-over-year stability check
  - 5 symbols: BTC, ETH, SOL, DOGE, XRP
  - 1,143 days × 288 bars/day ≈ 329K bars per symbol
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

PARQUET_DIR = Path("parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
SOURCE = "bybit_futures"

SESSIONS = {
    'Asia':   (0, 8),
    'Europe': (8, 16),
    'US':     (16, 24),
}

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


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
        print(f"  No files found for {symbol}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('timestamp_us').reset_index(drop=True)

    # Compute derived metrics
    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'].iloc[0] = 0.0

    # Volatility: |log return| as proxy for 5-min realized vol
    df['abs_ret_bps'] = np.abs(df['log_ret']) * 10000

    # Range in bps
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000

    # Buy ratio
    total_vol = df['buy_volume'] + df['sell_volume']
    df['buy_ratio'] = np.where(total_vol > 0, df['buy_volume'] / total_vol, 0.5)

    # Temporal labels
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek  # 0=Mon
    df['day_name'] = df['dow'].map(lambda x: DAY_NAMES[x])
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['month'].map(lambda x: MONTH_NAMES[x - 1])
    df['year'] = df['datetime'].dt.year
    df['is_weekend'] = df['dow'] >= 5

    # Session
    df['session'] = pd.cut(df['hour'], bins=[0, 8, 16, 24],
                           labels=['Asia', 'Europe', 'US'],
                           right=False, include_lowest=True)

    elapsed = time.time() - t0
    print(f"  {symbol}: {len(df):,} bars, {len(files)} days "
          f"({df['datetime'].min().strftime('%Y-%m-%d')} to "
          f"{df['datetime'].max().strftime('%Y-%m-%d')}), {elapsed:.1f}s")
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_hour_of_day(df, symbol):
    print(f"\n{'='*70}")
    print(f"HOUR OF DAY ANALYSIS — {symbol}")
    print(f"{'='*70}")

    metrics = ['abs_ret_bps', 'range_bps', 'quote_volume', 'trade_count', 'buy_ratio']
    labels = ['|Ret| (bps)', 'Range (bps)', 'Volume ($)', 'Trade Count', 'Buy Ratio']

    # Hourly means
    hourly = df.groupby('hour')[metrics].mean()

    print(f"\n  Hour  |Ret| (bps)  Range (bps)    Volume ($)  Trade Count  Buy Ratio     N")
    print(f"  {'─'*80}")
    for h in range(24):
        row = hourly.loc[h]
        n = (df['hour'] == h).sum()
        print(f"  {h:4d}  {row['abs_ret_bps']:10.3f}  {row['range_bps']:11.3f}  "
              f"{row['quote_volume']:12,.0f}  {row['trade_count']:11,.0f}  "
              f"{row['buy_ratio']:9.4f}  {n:5d}")

    # Kruskal-Wallis
    print(f"\n  Kruskal-Wallis (H0: no difference across hours):")
    print(f"  {'Metric':>20s}  {'H-stat':>8s}  {'p-value':>12s}  Significant")
    print(f"  {'─'*60}")
    for m, lab in zip(metrics, labels):
        groups = [df[df['hour'] == h][m].dropna().values for h in range(24)]
        groups = [g for g in groups if len(g) > 0]
        H, p = stats.kruskal(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'no'
        print(f"  {lab:>20s}  {H:8.1f}  {p:12.6f}  {sig:>11s}")

    # Peak/trough
    vol_hourly = df.groupby('hour')['range_bps'].mean()
    peak_h = vol_hourly.idxmax()
    trough_h = vol_hourly.idxmin()
    ratio = vol_hourly[peak_h] / vol_hourly[trough_h]
    print(f"\n  Range peak={peak_h:02d}:00 ({vol_hourly[peak_h]:.2f}), "
          f"trough={trough_h:02d}:00 ({vol_hourly[trough_h]:.2f}), ratio={ratio:.2f}x")

    vol_hourly2 = df.groupby('hour')['quote_volume'].mean()
    peak_v = vol_hourly2.idxmax()
    trough_v = vol_hourly2.idxmin()
    ratio_v = vol_hourly2[peak_v] / vol_hourly2[trough_v]
    print(f"  Volume peak={peak_v:02d}:00 ({vol_hourly2[peak_v]:,.0f}), "
          f"trough={trough_v:02d}:00 ({vol_hourly2[trough_v]:,.0f}), ratio={ratio_v:.2f}x")

    return hourly


def analyze_day_of_week(df, symbol):
    print(f"\n{'='*70}")
    print(f"DAY OF WEEK ANALYSIS — {symbol}")
    print(f"{'='*70}")

    metrics = ['abs_ret_bps', 'range_bps', 'quote_volume', 'trade_count']
    labels = ['|Ret| (bps)', 'Range (bps)', 'Volume ($)', 'Trade Count']

    daily = df.groupby('dow')[metrics].mean()

    print(f"\n   Day  |Ret| (bps)  Range (bps)    Volume ($)  Trade Count     N")
    print(f"  {'─'*70}")
    for d in range(7):
        row = daily.loc[d]
        n = (df['dow'] == d).sum()
        print(f"  {DAY_NAMES[d]:>4s}  {row['abs_ret_bps']:10.3f}  {row['range_bps']:11.3f}  "
              f"{row['quote_volume']:12,.0f}  {row['trade_count']:11,.0f}  {n:5d}")

    # Weekday vs weekend
    wd = df[~df['is_weekend']]
    we = df[df['is_weekend']]

    print(f"\n  Weekday vs Weekend:")
    print(f"  {'':>20s}  {'Weekday':>12s}  {'Weekend':>12s}  {'Ratio':>7s}  {'t-stat':>7s}  {'p-value':>10s}")
    print(f"  {'─'*75}")
    for m, lab in zip(metrics, labels):
        wd_mean = wd[m].mean()
        we_mean = we[m].mean()
        ratio = wd_mean / we_mean if we_mean > 0 else float('inf')
        t, p = stats.ttest_ind(wd[m].dropna(), we[m].dropna())
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if m == 'quote_volume':
            print(f"  {lab:>20s}  {wd_mean:12,.0f}  {we_mean:12,.0f}  {ratio:6.2f}x  {t:7.2f}  {p:9.4f} {sig}")
        else:
            print(f"  {lab:>20s}  {wd_mean:12.3f}  {we_mean:12.3f}  {ratio:6.2f}x  {t:7.2f}  {p:9.4f} {sig}")

    # KW across 7 days
    print(f"\n  Kruskal-Wallis across 7 days:")
    print(f"  {'Metric':>20s}  {'H-stat':>8s}  {'p-value':>12s}  Significant")
    print(f"  {'─'*60}")
    for m, lab in zip(metrics, labels):
        groups = [df[df['dow'] == d][m].dropna().values for d in range(7)]
        H, p = stats.kruskal(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'no'
        print(f"  {lab:>20s}  {H:8.1f}  {p:12.6f}  {sig:>11s}")


def analyze_sessions(df, symbol):
    print(f"\n{'='*70}")
    print(f"TRADING SESSION ANALYSIS — {symbol}")
    print(f"{'='*70}")

    metrics = ['abs_ret_bps', 'range_bps', 'quote_volume', 'trade_count', 'buy_ratio']
    labels = ['|Ret| (bps)', 'Range (bps)', 'Volume ($)', 'Trade Count', 'Buy Ratio']

    sess = df.groupby('session', observed=True)[metrics].mean()

    print(f"\n  {'Session':>8s}  |Ret| (bps)  Range (bps)    Volume ($)  Trade Count  Buy Ratio     N")
    print(f"  {'─'*85}")
    for s in ['Asia', 'Europe', 'US']:
        row = sess.loc[s]
        n = (df['session'] == s).sum()
        print(f"  {s:>8s}  {row['abs_ret_bps']:10.3f}  {row['range_bps']:11.3f}  "
              f"{row['quote_volume']:12,.0f}  {row['trade_count']:11,.0f}  "
              f"{row['buy_ratio']:9.4f}  {n:5d}")

    # Pairwise t-tests on range_bps
    print(f"\n  Pairwise t-tests (range_bps):")
    for s1, s2 in [('Asia', 'Europe'), ('Asia', 'US'), ('Europe', 'US')]:
        a = df[df['session'] == s1]['range_bps'].dropna()
        b = df[df['session'] == s2]['range_bps'].dropna()
        t, p = stats.ttest_ind(a, b)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    {s1} vs {s2}: t={t:+.2f}, p={p:.4f} {sig}")


def analyze_month(df, symbol):
    print(f"\n{'='*70}")
    print(f"MONTH OF YEAR ANALYSIS — {symbol}")
    print(f"{'='*70}")

    metrics = ['abs_ret_bps', 'range_bps', 'quote_volume', 'trade_count']
    labels = ['|Ret| (bps)', 'Range (bps)', 'Volume ($)', 'Trade Count']

    monthly = df.groupby('month')[metrics].mean()

    print(f"\n  Month  |Ret| (bps)  Range (bps)    Volume ($)  Trade Count     N")
    print(f"  {'─'*70}")
    for m in range(1, 13):
        if m not in monthly.index:
            continue
        row = monthly.loc[m]
        n = (df['month'] == m).sum()
        print(f"  {MONTH_NAMES[m-1]:>5s}  {row['abs_ret_bps']:10.3f}  {row['range_bps']:11.3f}  "
              f"{row['quote_volume']:12,.0f}  {row['trade_count']:11,.0f}  {n:5d}")

    # KW across months
    print(f"\n  Kruskal-Wallis across months:")
    print(f"  {'Metric':>20s}  {'H-stat':>8s}  {'p-value':>12s}  Significant")
    print(f"  {'─'*60}")
    for met, lab in zip(metrics, labels):
        groups = [df[df['month'] == m][met].dropna().values for m in range(1, 13) if (df['month'] == m).sum() > 0]
        H, p = stats.kruskal(*groups)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'no'
        print(f"  {lab:>20s}  {H:8.1f}  {p:12.6f}  {sig:>11s}")

    # Peak/trough months
    range_monthly = df.groupby('month')['range_bps'].mean()
    peak_m = range_monthly.idxmax()
    trough_m = range_monthly.idxmin()
    ratio = range_monthly[peak_m] / range_monthly[trough_m]
    print(f"\n  Range peak={MONTH_NAMES[peak_m-1]} ({range_monthly[peak_m]:.2f}), "
          f"trough={MONTH_NAMES[trough_m-1]} ({range_monthly[trough_m]:.2f}), ratio={ratio:.2f}x")


def analyze_year_stability(df, symbol):
    """Check if hourly patterns are stable year-over-year."""
    print(f"\n{'='*70}")
    print(f"YEAR-OVER-YEAR STABILITY — {symbol}")
    print(f"{'='*70}")

    years = sorted(df['year'].unique())
    print(f"\n  Range (bps) by hour × year:")
    header = "  Hour  " + "  ".join(f"{y:>8d}" for y in years)
    print(header)
    print(f"  {'─'*len(header)}")

    for h in range(24):
        vals = []
        for y in years:
            mask = (df['hour'] == h) & (df['year'] == y)
            v = df.loc[mask, 'range_bps'].mean()
            vals.append(v)
        line = f"  {h:4d}  " + "  ".join(f"{v:8.2f}" for v in vals)
        print(line)

    # Correlation of hourly profiles across years
    print(f"\n  Hourly profile correlation (Spearman) between years:")
    profiles = {}
    for y in years:
        profiles[y] = df[df['year'] == y].groupby('hour')['range_bps'].mean()

    for i, y1 in enumerate(years):
        for y2 in years[i+1:]:
            common = profiles[y1].index.intersection(profiles[y2].index)
            if len(common) < 10:
                continue
            rho, p = stats.spearmanr(profiles[y1][common], profiles[y2][common])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"    {y1} vs {y2}: ρ={rho:+.3f}, p={p:.4f} {sig}")

    # Weekday/weekend ratio stability
    print(f"\n  Weekday/Weekend range ratio by year:")
    for y in years:
        ydf = df[df['year'] == y]
        wd = ydf[~ydf['is_weekend']]['range_bps'].mean()
        we = ydf[ydf['is_weekend']]['range_bps'].mean()
        ratio = wd / we if we > 0 else float('inf')
        print(f"    {y}: weekday={wd:.2f}, weekend={we:.2f}, ratio={ratio:.2f}x")


def cross_symbol_summary(all_results):
    """Print cross-symbol comparison of key patterns."""
    print(f"\n{'='*70}")
    print(f"CROSS-SYMBOL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Range peak/trough hour:")
    print(f"  {'Symbol':>8s}  {'Peak Hour':>10s}  {'Peak Range':>11s}  {'Trough Hour':>12s}  {'Trough Range':>13s}  {'Ratio':>6s}")
    print(f"  {'─'*70}")
    for sym, hourly in all_results.items():
        peak_h = hourly['range_bps'].idxmax()
        trough_h = hourly['range_bps'].idxmin()
        ratio = hourly['range_bps'][peak_h] / hourly['range_bps'][trough_h]
        print(f"  {sym:>8s}  {peak_h:02d}:00{'':<5s}  {hourly['range_bps'][peak_h]:11.2f}  "
              f"{trough_h:02d}:00{'':<7s}  {hourly['range_bps'][trough_h]:13.2f}  {ratio:5.2f}x")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_total = time.time()

    print(f"{'='*70}")
    print(f"v33b: Temporal Patterns — 3-Year OHLCV Confirmation")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Source:  {SOURCE}")
    print(f"Data:    parquet/*/ohlcv/5m/{SOURCE}/")
    print_mem("start")

    all_hourly = {}

    for symbol in SYMBOLS:
        print(f"\n{'─'*70}")
        print(f"Loading {symbol}...")
        df = load_ohlcv(symbol)
        if df.empty:
            continue

        hourly = analyze_hour_of_day(df, symbol)
        all_hourly[symbol] = hourly
        analyze_day_of_week(df, symbol)
        analyze_sessions(df, symbol)
        analyze_month(df, symbol)
        analyze_year_stability(df, symbol)

        del df

    if all_hourly:
        cross_symbol_summary(all_hourly)

    elapsed = time.time() - t0_total
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print_mem("final")

    out_path = RESULTS_DIR / "v33b_temporal_ohlcv_3yr.txt"
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
