#!/usr/bin/env python3
"""
v42: Moon Phase Cycle Effect on Volatility

Calculate moon phase as percent of cycle (0%=new moon, 50%=full moon)
for each 5-min bar and test for any correlation with volatility.

Moon cycle = ~29.53 days (synodic month).
We compute phase from a known new moon reference date.

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

# Synodic month (new moon to new moon)
SYNODIC_MONTH = 29.530588853  # days

# Known new moon reference: January 21, 2023 20:53 UTC
# (any known new moon works as reference)
NEW_MOON_REF = pd.Timestamp('2023-01-21 20:53:00', tz='UTC')

PHASE_NAMES = {
    0: 'New Moon',
    1: 'Waxing Crescent',
    2: 'First Quarter',
    3: 'Waxing Gibbous',
    4: 'Full Moon',
    5: 'Waning Gibbous',
    6: 'Last Quarter',
    7: 'Waning Crescent',
}


def load_ohlcv(symbol):
    t0 = time.time()
    ohlcv_dir = PARQUET_DIR / symbol / "ohlcv" / "5m" / SOURCE
    files = sorted(ohlcv_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['dow'] = df['datetime'].dt.dayofweek
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def compute_moon_phase(dt_series):
    """Compute moon phase as percentage of cycle (0-100%).
    0% = new moon, ~50% = full moon, 100% = next new moon.
    Also returns phase octant (0-7) for binning."""
    days_since_ref = (dt_series - NEW_MOON_REF).dt.total_seconds() / 86400.0
    phase_pct = (days_since_ref % SYNODIC_MONTH) / SYNODIC_MONTH * 100
    phase_octant = ((phase_pct / 100 * 8) % 8).astype(int)
    return phase_pct, phase_octant


def main():
    t0 = time.time()
    print("="*70)
    print("v42: Moon Phase Cycle Effect on Volatility")
    print("="*70)

    all_octant = []
    all_pct = []
    all_summary = []

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)

        # Compute moon phase
        df['moon_pct'], df['moon_octant'] = compute_moon_phase(df['datetime'])

        # Weekday only (remove weekend noise)
        wdf = df[df['dow'] < 5].copy()

        print(f"  Moon phase analysis ({symbol}, weekdays only, n={len(wdf):,}):")

        # By octant (8 phases)
        print(f"\n    Phase octant analysis:")
        octant_vals = []
        for oct_id in range(8):
            sub = wdf[wdf['moon_octant'] == oct_id]
            avg = sub['range_bps'].mean()
            octant_vals.append(avg)
            all_octant.append({
                'symbol': symbol, 'octant': oct_id,
                'phase_name': PHASE_NAMES[oct_id],
                'avg_range_bps': round(avg, 2),
                'n_bars': len(sub),
            })
            print(f"      {oct_id} ({PHASE_NAMES[oct_id]:>18s}): {avg:.2f} bps (n={len(sub):,})")

        # Kruskal-Wallis across 8 phases
        groups = [wdf[wdf['moon_octant'] == i]['range_bps'].values for i in range(8)]
        H, p_kw = stats.kruskal(*groups)
        print(f"\n    Kruskal-Wallis H={H:.2f}, p={p_kw:.4f}")

        # Continuous correlation: moon_pct vs range
        # Use sin/cos to capture circular relationship
        wdf_clean = wdf.dropna(subset=['range_bps']).copy()
        moon_rad = wdf_clean['moon_pct'] / 100 * 2 * np.pi
        rho_sin, p_sin = stats.spearmanr(np.sin(moon_rad), wdf_clean['range_bps'])
        rho_cos, p_cos = stats.spearmanr(np.cos(moon_rad), wdf_clean['range_bps'])
        print(f"    sin(moon) vs range: ρ={rho_sin:+.4f}, p={p_sin:.4f}")
        print(f"    cos(moon) vs range: ρ={rho_cos:+.4f}, p={p_cos:.4f}")

        # New moon vs full moon
        new_moon = wdf[(wdf['moon_pct'] < 6.25) | (wdf['moon_pct'] > 93.75)]['range_bps']
        full_moon = wdf[(wdf['moon_pct'] > 43.75) & (wdf['moon_pct'] < 56.25)]['range_bps']
        if len(new_moon) > 100 and len(full_moon) > 100:
            t_nf, p_nf = stats.mannwhitneyu(new_moon, full_moon, alternative='two-sided')
            ratio = new_moon.mean() / full_moon.mean()
            print(f"    New moon vs Full moon: ratio={ratio:.3f}x, p={p_nf:.4f}")

        max_oct = max(range(8), key=lambda i: octant_vals[i])
        min_oct = min(range(8), key=lambda i: octant_vals[i])
        spread = octant_vals[max_oct] / octant_vals[min_oct]

        all_summary.append({
            'symbol': symbol,
            'kw_H': round(H, 2), 'kw_p': p_kw,
            'rho_sin': round(rho_sin, 4), 'p_sin': p_sin,
            'rho_cos': round(rho_cos, 4), 'p_cos': p_cos,
            'max_phase': PHASE_NAMES[max_oct],
            'min_phase': PHASE_NAMES[min_oct],
            'max_min_ratio': round(spread, 3),
            'significant': p_kw < 0.05,
        })

        # Fine-grained: 100 bins
        wdf_clean['moon_bin'] = (wdf_clean['moon_pct'] // 1).astype(int).clip(0, 99)
        for b in range(100):
            sub = wdf_clean[wdf_clean['moon_bin'] == b]
            if len(sub) > 0:
                all_pct.append({
                    'symbol': symbol, 'moon_pct': b,
                    'avg_range_bps': round(sub['range_bps'].mean(), 2),
                    'n_bars': len(sub),
                })

    # Save CSVs
    pd.DataFrame(all_octant).to_csv(RESULTS_DIR / 'v42_moon_octant.csv', index=False)
    pd.DataFrame(all_pct).to_csv(RESULTS_DIR / 'v42_moon_pct_profile.csv', index=False)
    pd.DataFrame(all_summary).to_csv(RESULTS_DIR / 'v42_moon_summary.csv', index=False)
    print(f"\n  Saved: 3 CSVs")

    # ---- PLOT 1: Moon phase octant bar chart ----
    fig, ax = plt.subplots(figsize=(12, 6))
    oct_df = pd.DataFrame(all_octant)
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    x = np.arange(8)
    width = 0.15
    for i, (sym, color) in enumerate(zip(SYMBOLS, colors)):
        sub = oct_df[oct_df['symbol'] == sym].sort_values('octant')
        # Normalize to mean=100
        mean_val = sub['avg_range_bps'].mean()
        ax.bar(x + i*width - 0.3, sub['avg_range_bps'] / mean_val * 100, width,
               label=sym.replace('USDT',''), color=color, alpha=0.8)

    ax.axhline(y=100, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{PHASE_NAMES[i]}\n({i})' for i in range(8)], fontsize=8)
    ax.set_ylabel('Normalized Range (100 = avg)', fontsize=11)
    ax.set_title('v42: Volatility by Moon Phase\n(normalized, weekdays only, 3+ years)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v42_moon_phase_octant.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v42_moon_phase_octant.png")

    # ---- PLOT 2: Continuous moon phase profile (polar) ----
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    pct_df = pd.DataFrame(all_pct)

    for sym, color in zip(SYMBOLS, colors):
        sub = pct_df[pct_df['symbol'] == sym].sort_values('moon_pct')
        # Normalize
        mean_val = sub['avg_range_bps'].mean()
        theta = sub['moon_pct'] / 100 * 2 * np.pi
        r = sub['avg_range_bps'] / mean_val * 100
        # Smooth with rolling mean
        r_smooth = pd.Series(r.values).rolling(5, center=True, min_periods=1).mean()
        ax.plot(theta, r_smooth, label=sym.replace('USDT',''), color=color, linewidth=1.2)

    # Mark new moon (0°) and full moon (180°)
    ax.annotate('New\nMoon', xy=(0, 105), fontsize=9, fontweight='bold', ha='center', color='#333')
    ax.annotate('Full\nMoon', xy=(np.pi, 105), fontsize=9, fontweight='bold', ha='center', color='#333')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('v42: Volatility by Moon Phase (polar)\n(normalized, weekdays, 3+ years)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v42_moon_phase_polar.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v42_moon_phase_polar.png")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for s in all_summary:
        sig = '***' if s['kw_p'] < 0.001 else ('**' if s['kw_p'] < 0.01 else ('*' if s['kw_p'] < 0.05 else 'ns'))
        print(f"  {s['symbol']:>10s}: KW H={s['kw_H']:.1f} ({sig}), "
              f"max/min={s['max_min_ratio']:.3f}x, "
              f"peak={s['max_phase']}, trough={s['min_phase']}")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
