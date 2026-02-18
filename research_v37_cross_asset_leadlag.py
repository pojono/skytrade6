#!/usr/bin/env python3
"""
v37: Cross-Asset Lead-Lag in Volatility

Does BTC volatility lead altcoin volatility?
- Cross-correlation of 5-min range between BTC and ETH/SOL/DOGE/XRP at lags 0-60 bars
- If BTC leads by 1-5 bars (5-25 min), that's tradeable for altcoin vol timing
- Also test: does any altcoin lead BTC?

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
RESULTS_DIR.mkdir(exist_ok=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
SOURCE = "bybit_futures"
MAX_LAG = 60  # 60 bars = 5 hours at 5-min


def load_ohlcv(symbol):
    t0 = time.time()
    ohlcv_dir = PARQUET_DIR / symbol / "ohlcv" / "5m" / SOURCE
    files = sorted(ohlcv_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['log_range'] = np.log1p(df['range_bps'])
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def cross_corr(x, y, max_lag):
    """Cross-correlation at lags -max_lag to +max_lag.
    Positive lag means x leads y."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    result = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            result[lag] = np.mean(x[:n-lag] * y[lag:]) if lag < n else 0
        else:
            result[lag] = np.mean(x[-lag:] * y[:n+lag]) if -lag < n else 0
    return result


def main():
    t0 = time.time()
    print("="*70)
    print("v37: Cross-Asset Lead-Lag in Volatility")
    print("="*70)

    # Load all symbols
    data = {}
    for sym in SYMBOLS:
        data[sym] = load_ohlcv(sym)

    # Align all to same timestamps
    print("\nAligning timestamps...")
    base = data['BTCUSDT'][['datetime', 'range_bps', 'log_range']].rename(
        columns={'range_bps': 'BTCUSDT_range', 'log_range': 'BTCUSDT_log_range'})

    for sym in SYMBOLS[1:]:
        sub = data[sym][['datetime', 'range_bps', 'log_range']].rename(
            columns={'range_bps': f'{sym}_range', 'log_range': f'{sym}_log_range'})
        base = base.merge(sub, on='datetime', how='inner')

    print(f"  Aligned: {len(base):,} bars")

    # Cross-correlations: BTC vs each altcoin
    print("\nComputing cross-correlations (BTC vs alts)...")
    all_cc = {}
    for sym in SYMBOLS[1:]:
        print(f"  BTC vs {sym}...", flush=True)
        cc = cross_corr(base['BTCUSDT_log_range'].values,
                        base[f'{sym}_log_range'].values, MAX_LAG)
        all_cc[sym] = cc

        # Find peak
        peak_lag = max(cc, key=cc.get)
        peak_val = cc[peak_lag]
        print(f"    Peak: lag={peak_lag} ({peak_lag*5}min), ρ={peak_val:.4f}")
        print(f"    At lag=0: ρ={cc[0]:.4f}")
        if peak_lag > 0:
            print(f"    → BTC LEADS {sym} by {peak_lag} bars ({peak_lag*5} min)")
        elif peak_lag < 0:
            print(f"    → {sym} LEADS BTC by {-peak_lag} bars ({-peak_lag*5} min)")

    # Also: all pairwise
    print("\nAll pairwise cross-correlations...")
    pair_results = []
    for i, sym1 in enumerate(SYMBOLS):
        for sym2 in SYMBOLS[i+1:]:
            cc = cross_corr(base[f'{sym1}_log_range'].values,
                            base[f'{sym2}_log_range'].values, MAX_LAG)
            peak_lag = max(cc, key=cc.get)
            peak_val = cc[peak_lag]
            lag0 = cc[0]

            leader = sym1 if peak_lag > 0 else (sym2 if peak_lag < 0 else 'simultaneous')
            pair_results.append({
                'pair': f'{sym1}-{sym2}',
                'lag0_corr': round(lag0, 4),
                'peak_lag': peak_lag,
                'peak_lag_min': peak_lag * 5,
                'peak_corr': round(peak_val, 4),
                'leader': leader,
                'lead_minutes': abs(peak_lag) * 5,
            })
            print(f"  {sym1} vs {sym2}: lag0={lag0:.4f}, peak_lag={peak_lag} ({peak_lag*5}min), peak_ρ={peak_val:.4f}, leader={leader}")

    # Save pair results CSV
    pair_df = pd.DataFrame(pair_results)
    pair_df.to_csv(RESULTS_DIR / 'v37_leadlag_pairs.csv', index=False)
    print(f"\n  Saved: results/v37_leadlag_pairs.csv")

    # Save full cross-correlation curves CSV
    cc_rows = []
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        row = {'lag_bars': lag, 'lag_minutes': lag * 5}
        for sym in SYMBOLS[1:]:
            row[f'BTC_vs_{sym}'] = round(all_cc[sym][lag], 6)
        cc_rows.append(row)
    pd.DataFrame(cc_rows).to_csv(RESULTS_DIR / 'v37_leadlag_curves.csv', index=False)
    print(f"  Saved: results/v37_leadlag_curves.csv")

    # ---- PLOT 1: BTC vs all alts cross-correlation ----
    fig, ax = plt.subplots(figsize=(12, 6))
    lags = list(range(-MAX_LAG, MAX_LAG + 1))
    colors = ['#E53935', '#43A047', '#1E88E5', '#FB8C00']

    for sym, color in zip(SYMBOLS[1:], colors):
        vals = [all_cc[sym][l] for l in lags]
        ax.plot([l*5 for l in lags], vals, label=f'BTC → {sym}', color=color, linewidth=1.5)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Lag (minutes) — positive = BTC leads', fontsize=11)
    ax.set_ylabel('Cross-correlation (log range)', fontsize=11)
    ax.set_title('v37: BTC Volatility Lead-Lag vs Altcoins\n(5-min range, 3+ years)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate peaks
    for sym, color in zip(SYMBOLS[1:], colors):
        peak_lag = max(all_cc[sym], key=all_cc[sym].get)
        peak_val = all_cc[sym][peak_lag]
        ax.annotate(f'peak: {peak_lag*5}min', xy=(peak_lag*5, peak_val),
                    fontsize=8, color=color, fontweight='bold',
                    textcoords="offset points", xytext=(10, 5))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v37_leadlag_btc_vs_alts.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v37_leadlag_btc_vs_alts.png")

    # ---- PLOT 2: Heatmap of lag-0 correlations ----
    fig, ax = plt.subplots(figsize=(7, 6))
    n = len(SYMBOLS)
    corr_matrix = np.zeros((n, n))
    for i, sym1 in enumerate(SYMBOLS):
        for j, sym2 in enumerate(SYMBOLS):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                cc = cross_corr(base[f'{sym1}_log_range'].values,
                                base[f'{sym2}_log_range'].values, 0)
                corr_matrix[i, j] = cc[0]

    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels([s.replace('USDT','') for s in SYMBOLS], fontsize=10, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels([s.replace('USDT','') for s in SYMBOLS], fontsize=10, fontweight='bold')

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr_matrix[i,j]:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if corr_matrix[i,j] > 0.8 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Contemporaneous correlation (log range)', fontsize=10)
    ax.set_title('v37: Cross-Asset Volatility Correlation Matrix\n(5-min, 3+ years)', fontsize=13, fontweight='bold')
    ax.xaxis.set_ticks_position('top')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v37_vol_correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v37_vol_correlation_matrix.png")

    # ---- Rolling lead-lag stability ----
    print("\nRolling lead-lag stability (yearly)...")
    base['year'] = base['datetime'].dt.year
    yearly_rows = []
    for year in sorted(base['year'].unique()):
        ydf = base[base['year'] == year]
        if len(ydf) < 10000:
            continue
        for sym in SYMBOLS[1:]:
            cc = cross_corr(ydf['BTCUSDT_log_range'].values,
                            ydf[f'{sym}_log_range'].values, 12)
            peak_lag = max(cc, key=cc.get)
            yearly_rows.append({
                'year': year, 'pair': f'BTC-{sym}',
                'lag0': round(cc[0], 4),
                'peak_lag': peak_lag,
                'peak_lag_min': peak_lag * 5,
                'peak_corr': round(cc[peak_lag], 4),
            })
            print(f"  {year} BTC vs {sym}: lag0={cc[0]:.4f}, peak_lag={peak_lag} ({peak_lag*5}min)")

    pd.DataFrame(yearly_rows).to_csv(RESULTS_DIR / 'v37_leadlag_yearly.csv', index=False)
    print(f"  Saved: results/v37_leadlag_yearly.csv")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
