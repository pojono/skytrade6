#!/usr/bin/env python3
"""
v33 Heatmap Visualizations — Temporal Patterns in Volatility

Generates 12 charts:
  For each of 5 symbols + ALL combined:
    1. Month × Year heatmap (12 cols × 3 rows): avg range by month per year
    2. Hour × Day-of-week heatmap (7 cols × 24 rows): avg range with session boundaries

Uses 3+ years of 5-min OHLCV parquet data from Bybit futures.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

PARQUET_DIR = Path("parquet")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
SOURCE = "bybit_futures"

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Realistic overlapping market sessions (UTC)
# Tokyo:  00:00–09:00 UTC  (09:00–18:00 JST)
# London: 07:00–16:00 UTC  (07:00–16:00 GMT)
# New York: 12:00–21:00 UTC (08:00–17:00 ET, summer; 07:00–16:00 ET winter)
#
# Overlaps:
#   Tokyo-London:  07:00–09:00 UTC
#   London-NY:     12:00–16:00 UTC  ← peak volatility zone

SESSIONS = [
    # (name, start_h, end_h, color, bar_x_offset)
    ('Tokyo',   0,  9,  '#4FC3F7', 0),     # light blue
    ('London',  7,  16, '#FFB74D', 1),     # orange
    ('New York', 12, 21, '#EF5350', 2),    # red
]

# For labeling each hour with active sessions
def get_session_label(h):
    """Return which sessions are active at hour h UTC."""
    active = []
    for name, start, end, _, _ in SESSIONS:
        if start <= h < end:
            active.append(name)
    if not active:
        return 'Quiet'
    return ' + '.join(active)


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

    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000

    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek  # 0=Mon
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    elapsed = time.time() - t0
    print(f"  {symbol}: {len(df):,} bars ({elapsed:.1f}s)")
    return df


# ============================================================================
# HEATMAP 1: Month × Year
# ============================================================================

def plot_month_year_heatmap(df, symbol, out_path):
    """Heatmap: 12 months (cols) × years (rows), value = mean range_bps."""
    years = sorted(df['year'].unique())
    # Only keep full or near-full years (at least 2 months of data)
    years = [y for y in years if df[df['year'] == y]['month'].nunique() >= 2]

    matrix = np.full((len(years), 12), np.nan)
    for i, y in enumerate(years):
        for m in range(1, 13):
            mask = (df['year'] == y) & (df['month'] == m)
            if mask.sum() > 0:
                matrix[i, m - 1] = df.loc[mask, 'range_bps'].mean()

    fig, ax = plt.subplots(figsize=(14, 3.5))

    # Use a diverging colormap centered on median
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    vmid = np.nanmedian(matrix)

    # Custom norm to center colormap
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', norm=norm,
                   interpolation='nearest')

    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_NAMES, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=11, fontweight='bold')

    # Annotate cells
    for i in range(len(years)):
        for j in range(12):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > (vmid + (vmax - vmid) * 0.5) or val < (vmid - (vmid - vmin) * 0.5) else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Range (bps)', fontsize=10)

    title = f'{symbol} — Avg 5-min Range (bps) by Month × Year'
    if symbol == 'ALL':
        title = 'ALL SYMBOLS — Avg 5-min Range (bps) by Month × Year (normalized)'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ============================================================================
# HEATMAP 2: Hour × Day-of-Week
# ============================================================================

def plot_hour_dow_heatmap(df, symbol, out_path):
    """Heatmap: 7 days (cols) × 24 hours (rows), value = mean range_bps.
    Session boundaries marked with colored bars on the left."""
    matrix = np.full((24, 7), np.nan)
    for h in range(24):
        for d in range(7):
            mask = (df['hour'] == h) & (df['dow'] == d)
            if mask.sum() > 0:
                matrix[h, d] = df.loc[mask, 'range_bps'].mean()

    fig, ax = plt.subplots(figsize=(7, 10))

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    vmid = np.nanmedian(matrix)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vmid, vmax=vmax)

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', norm=norm,
                   interpolation='nearest')

    # Labels
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_NAMES, fontsize=11, fontweight='bold')
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(24)], fontsize=9)

    # Annotate cells
    for h in range(24):
        for d in range(7):
            val = matrix[h, d]
            if not np.isnan(val):
                color = 'white' if val > (vmid + (vmax - vmid) * 0.5) or val < (vmid - (vmid - vmin) * 0.5) else 'black'
                ax.text(d, h, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

    # Session bars on the right side (overlapping)
    bar_width = 0.22
    bar_gap = 0.24
    bar_x_start = 6.65

    for name, start, end, color, x_off in SESSIONS:
        x = bar_x_start + x_off * bar_gap

        # Colored bar
        rect = Rectangle((x, start - 0.5), bar_width, end - start,
                         linewidth=0.5, edgecolor=color, facecolor=color, alpha=0.5,
                         clip_on=False)
        ax.add_patch(rect)

        # Session label
        mid_y = (start + end) / 2 - 0.5
        label_x = bar_x_start + 3 * bar_gap + 0.1
        ax.annotate(name, xy=(label_x, mid_y), fontsize=8, fontweight='bold',
                    color=color, ha='left', va='center',
                    annotation_clip=False)

    # Horizontal lines at key boundaries
    for h_line, style, lw in [(7, '--', 1.5), (9, '--', 1.5),   # Tokyo-London overlap
                               (12, '-', 2.0), (16, '-', 2.0),  # London-NY overlap
                               (21, '--', 1.5)]:
        ax.axhline(y=h_line - 0.5, color='#666666', linewidth=lw,
                   linestyle=style, alpha=0.7)

    # Overlap zone annotations on the far right
    overlap_x = bar_x_start + 3 * bar_gap + 0.1
    ax.annotate('Tokyo-London\noverlap', xy=(overlap_x, 7.5),
                fontsize=6.5, color='#666666', ha='left', va='center',
                fontstyle='italic', annotation_clip=False)
    ax.annotate('London-NY\noverlap ★', xy=(overlap_x, 13.5),
                fontsize=6.5, color='#333333', ha='left', va='center',
                fontweight='bold', fontstyle='italic', annotation_clip=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.18)
    cbar.set_label('Range (bps)', fontsize=10)

    title = f'{symbol} — Avg 5-min Range (bps)\nHour (UTC) × Day of Week'
    if symbol == 'ALL':
        title = 'ALL SYMBOLS — Avg 5-min Range (bps, normalized)\nHour (UTC) × Day of Week'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ============================================================================
# CSV EXPORT
# ============================================================================

def save_month_year_csv(df, symbol, out_path):
    """Save Month × Year matrix as CSV."""
    years = sorted(df['year'].unique())
    years = [y for y in years if df[df['year'] == y]['month'].nunique() >= 2]
    rows = []
    for y in years:
        row = {'year': y}
        for m in range(1, 13):
            mask = (df['year'] == y) & (df['month'] == m)
            row[MONTH_NAMES[m - 1]] = round(df.loc[mask, 'range_bps'].mean(), 2) if mask.sum() > 0 else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")


def save_hour_dow_csv(df, symbol, out_path):
    """Save Hour × Day-of-week matrix as CSV."""
    rows = []
    for h in range(24):
        row = {'hour_utc': f'{h:02d}:00'}
        for d in range(7):
            mask = (df['hour'] == h) & (df['dow'] == d)
            row[DAY_NAMES[d]] = round(df.loc[mask, 'range_bps'].mean(), 2) if mask.sum() > 0 else None
        row['sessions_active'] = get_session_label(h)
        row['tokyo'] = 1 if 0 <= h < 9 else 0
        row['london'] = 1 if 7 <= h < 16 else 0
        row['new_york'] = 1 if 12 <= h < 21 else 0
        row['n_sessions'] = row['tokyo'] + row['london'] + row['new_york']
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")


def save_hourly_csv(all_dfs):
    """Save per-hour summary across all symbols as a single CSV."""
    rows = []
    for h in range(24):
        row = {'hour_utc': f'{h:02d}:00',
               'sessions_active': get_session_label(h),
               'n_sessions': sum([1 for _, s, e, _, _ in SESSIONS if s <= h < e])}
        for symbol, df in all_dfs.items():
            mask = df['hour'] == h
            row[f'{symbol}_range_bps'] = round(df.loc[mask, 'range_bps'].mean(), 2)
            row[f'{symbol}_volume'] = round(df.loc[mask, 'quote_volume'].mean(), 0)
            row[f'{symbol}_trades'] = round(df.loc[mask, 'trade_count'].mean(), 0)
        rows.append(row)
    out = RESULTS_DIR / 'v33_hourly_all_symbols.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"    Saved: {out}")


def save_daily_csv(all_dfs):
    """Save per-day-of-week summary across all symbols as a single CSV."""
    rows = []
    for d in range(7):
        row = {'day': DAY_NAMES[d], 'is_weekend': d >= 5}
        for symbol, df in all_dfs.items():
            mask = df['dow'] == d
            row[f'{symbol}_range_bps'] = round(df.loc[mask, 'range_bps'].mean(), 2)
            row[f'{symbol}_volume'] = round(df.loc[mask, 'quote_volume'].mean(), 0)
        rows.append(row)
    out = RESULTS_DIR / 'v33_daily_all_symbols.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"    Saved: {out}")


def save_monthly_csv(all_dfs):
    """Save per-month summary across all symbols as a single CSV."""
    rows = []
    for m in range(1, 13):
        row = {'month': MONTH_NAMES[m - 1]}
        for symbol, df in all_dfs.items():
            mask = df['month'] == m
            row[f'{symbol}_range_bps'] = round(df.loc[mask, 'range_bps'].mean(), 2)
            row[f'{symbol}_volume'] = round(df.loc[mask, 'quote_volume'].mean(), 0)
        rows.append(row)
    out = RESULTS_DIR / 'v33_monthly_all_symbols.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"    Saved: {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("Loading data for all symbols...")

    all_dfs = {}
    for symbol in SYMBOLS:
        df = load_ohlcv(symbol)
        if not df.empty:
            all_dfs[symbol] = df

    # Build "ALL" combined — normalize each symbol's range to z-score
    # so they contribute equally despite different absolute levels
    print("\n  Building ALL (normalized) dataset...")
    combined_parts = []
    for symbol, df in all_dfs.items():
        sub = df[['hour', 'dow', 'month', 'year', 'range_bps']].copy()
        mu = sub['range_bps'].mean()
        sigma = sub['range_bps'].std()
        sub['range_bps'] = (sub['range_bps'] - mu) / sigma * 100 + 100  # normalized to ~100 mean
        combined_parts.append(sub)
    all_combined = pd.concat(combined_parts, ignore_index=True)
    print(f"  ALL: {len(all_combined):,} bars (normalized)")

    # Generate charts + CSVs
    print("\nGenerating heatmaps + CSVs...")
    for symbol, df in all_dfs.items():
        print(f"\n  {symbol}:")
        plot_month_year_heatmap(df, symbol,
                                RESULTS_DIR / f"v33_heatmap_month_{symbol}.png")
        plot_hour_dow_heatmap(df, symbol,
                              RESULTS_DIR / f"v33_heatmap_hour_dow_{symbol}.png")
        save_month_year_csv(df, symbol,
                            RESULTS_DIR / f"v33_month_year_{symbol}.csv")
        save_hour_dow_csv(df, symbol,
                          RESULTS_DIR / f"v33_hour_dow_{symbol}.csv")

    print(f"\n  ALL (combined):")
    plot_month_year_heatmap(all_combined, 'ALL',
                            RESULTS_DIR / "v33_heatmap_month_ALL.png")
    plot_hour_dow_heatmap(all_combined, 'ALL',
                          RESULTS_DIR / "v33_heatmap_hour_dow_ALL.png")
    save_month_year_csv(all_combined, 'ALL',
                        RESULTS_DIR / "v33_month_year_ALL.csv")
    save_hour_dow_csv(all_combined, 'ALL',
                      RESULTS_DIR / "v33_hour_dow_ALL.csv")

    # Cross-symbol summary CSVs
    print(f"\n  Cross-symbol summaries:")
    save_hourly_csv(all_dfs)
    save_daily_csv(all_dfs)
    save_monthly_csv(all_dfs)

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total. Charts + CSVs saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
