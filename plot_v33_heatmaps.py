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

SESSION_BOUNDS = {
    'Asia':   (0, 8,  '#4FC3F7'),   # light blue
    'Europe': (8, 16, '#FFB74D'),   # orange
    'US':     (16, 24, '#EF5350'),  # red
}


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

    # Session boundary lines and labels on the left
    for sess_name, (start, end, color) in SESSION_BOUNDS.items():
        # Horizontal lines at session boundaries
        if start > 0:
            ax.axhline(y=start - 0.5, color=color, linewidth=2.5, linestyle='-', alpha=0.9)

        # Session label bar on the right side
        mid_y = (start + end) / 2 - 0.5
        ax.annotate(sess_name, xy=(7.3, mid_y), fontsize=9, fontweight='bold',
                    color=color, ha='left', va='center',
                    annotation_clip=False)

        # Colored bar on the right edge
        rect = Rectangle((6.6, start - 0.5), 0.25, end - start,
                         linewidth=0, facecolor=color, alpha=0.6,
                         clip_on=False)
        ax.add_patch(rect)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
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

    # Generate charts
    print("\nGenerating heatmaps...")
    for symbol, df in all_dfs.items():
        print(f"\n  {symbol}:")
        plot_month_year_heatmap(df, symbol,
                                RESULTS_DIR / f"v33_heatmap_month_{symbol}.png")
        plot_hour_dow_heatmap(df, symbol,
                              RESULTS_DIR / f"v33_heatmap_hour_dow_{symbol}.png")

    print(f"\n  ALL (combined):")
    plot_month_year_heatmap(all_combined, 'ALL',
                            RESULTS_DIR / "v33_heatmap_month_ALL.png")
    plot_hour_dow_heatmap(all_combined, 'ALL',
                          RESULTS_DIR / "v33_heatmap_hour_dow_ALL.png")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total. 12 charts saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
