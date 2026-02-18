#!/usr/bin/env python3
"""
v38: Volume-Volatility Causality

Does volume lead volatility, or vice versa?
- Granger causality tests at 5-min resolution
- Cross-correlation of volume vs range at various lags
- Buy/sell volume ratio as directional vol predictor
- Volume surge → subsequent vol analysis

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
MAX_LAG = 24  # 24 bars = 2 hours


def load_ohlcv(symbol):
    t0 = time.time()
    ohlcv_dir = PARQUET_DIR / symbol / "ohlcv" / "5m" / SOURCE
    files = sorted(ohlcv_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['log_range'] = np.log1p(df['range_bps'])
    df['log_volume'] = np.log1p(df['quote_volume'])
    df['log_trades'] = np.log1p(df['trade_count'])
    # Buy ratio from taker_buy_volume
    if 'taker_buy_volume' in df.columns:
        df['buy_ratio'] = df['taker_buy_volume'] / df['volume'].clip(lower=1e-10)
    elif 'taker_buy_quote_volume' in df.columns:
        df['buy_ratio'] = df['taker_buy_quote_volume'] / df['quote_volume'].clip(lower=1e-10)
    else:
        df['buy_ratio'] = 0.5
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def cross_corr_simple(x, y, max_lag):
    """Cross-correlation. Positive lag = x leads y."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    result = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            result[lag] = np.mean(x[:n-lag] * y[lag:])
        else:
            result[lag] = np.mean(x[-lag:] * y[:n+lag])
    return result


def granger_test(x, y, max_lag=6):
    """Simple Granger causality: does adding lagged x improve prediction of y?
    Returns F-stat and p-value for each lag."""
    n = len(x)
    results = []
    for lag in range(1, max_lag + 1):
        # Restricted model: y_t = a + b1*y_{t-1} + ... + bL*y_{t-L}
        # Unrestricted: y_t = a + b1*y_{t-1} + ... + bL*y_{t-L} + c1*x_{t-1} + ... + cL*x_{t-L}
        Y = y[lag:]
        n_obs = len(Y)

        # Build lagged matrices
        Y_lags = np.column_stack([y[lag-i-1:n-i-1] for i in range(lag)])
        X_lags = np.column_stack([x[lag-i-1:n-i-1] for i in range(lag)])

        # Restricted
        X_r = np.column_stack([np.ones(n_obs), Y_lags])
        beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
        resid_r = Y - X_r @ beta_r
        ssr_r = np.sum(resid_r**2)

        # Unrestricted
        X_u = np.column_stack([np.ones(n_obs), Y_lags, X_lags])
        beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
        resid_u = Y - X_u @ beta_u
        ssr_u = np.sum(resid_u**2)

        # F-test
        df1 = lag
        df2 = n_obs - 2 * lag - 1
        if ssr_u > 0 and df2 > 0:
            F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            p = 1 - stats.f.cdf(F, df1, df2)
        else:
            F, p = 0, 1

        results.append({'lag': lag, 'F_stat': F, 'p_value': p})

    return results


def volume_surge_analysis(df, symbol):
    """What happens to vol after a volume surge?"""
    log_vol = df['log_volume'].values
    log_range = df['log_range'].values

    mu_v = log_vol.mean()
    sigma_v = log_vol.std()

    results = []
    for threshold_name, threshold in [('1σ', mu_v + sigma_v),
                                       ('2σ', mu_v + 2*sigma_v),
                                       ('3σ', mu_v + 3*sigma_v)]:
        surge_idx = np.where(log_vol > threshold)[0]
        surge_idx = surge_idx[surge_idx < len(log_range) - 12]

        if len(surge_idx) < 10:
            continue

        # Average range in next 1,2,3,...,12 bars after surge
        fwd_ranges = {}
        for fwd in [1, 2, 3, 6, 12]:
            fwd_vals = log_range[surge_idx + fwd]
            baseline = log_range.mean()
            fwd_ranges[f'fwd_{fwd}_ratio'] = np.exp(fwd_vals.mean()) / np.exp(baseline)

        results.append({
            'symbol': symbol,
            'threshold': threshold_name,
            'n_surges': len(surge_idx),
            **fwd_ranges,
        })

    return results


def main():
    t0 = time.time()
    print("="*70)
    print("v38: Volume-Volatility Causality")
    print("="*70)

    all_cc = {}
    all_granger = []
    all_surge = []

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)

        log_vol = df['log_volume'].values
        log_range = df['log_range'].values
        buy_ratio = df['buy_ratio'].values

        # Cross-correlation: volume vs range
        print(f"  Cross-correlation (volume → range)...", flush=True)
        cc_vr = cross_corr_simple(log_vol, log_range, MAX_LAG)
        cc_rv = cross_corr_simple(log_range, log_vol, MAX_LAG)
        all_cc[symbol] = {'vol_leads_range': cc_vr, 'range_leads_vol': cc_rv}

        # Find peak
        peak_lag_vr = max(cc_vr, key=cc_vr.get)
        peak_lag_rv = max(cc_rv, key=cc_rv.get)
        print(f"    Volume→Range: peak at lag={peak_lag_vr} ({peak_lag_vr*5}min), ρ={cc_vr[peak_lag_vr]:.4f}")
        print(f"    Range→Volume: peak at lag={peak_lag_rv} ({peak_lag_rv*5}min), ρ={cc_rv[peak_lag_rv]:.4f}")
        print(f"    Contemporaneous: ρ={cc_vr[0]:.4f}")

        # Granger causality
        print(f"  Granger causality tests...", flush=True)
        gc_v2r = granger_test(log_vol, log_range, max_lag=6)
        gc_r2v = granger_test(log_range, log_vol, max_lag=6)

        for g in gc_v2r:
            all_granger.append({
                'symbol': symbol, 'direction': 'Volume→Range',
                'lag': g['lag'], 'F_stat': round(g['F_stat'], 2), 'p_value': g['p_value'],
                'significant': g['p_value'] < 0.001,
            })
        for g in gc_r2v:
            all_granger.append({
                'symbol': symbol, 'direction': 'Range→Volume',
                'lag': g['lag'], 'F_stat': round(g['F_stat'], 2), 'p_value': g['p_value'],
                'significant': g['p_value'] < 0.001,
            })

        best_v2r = max(gc_v2r, key=lambda x: x['F_stat'])
        best_r2v = max(gc_r2v, key=lambda x: x['F_stat'])
        print(f"    Vol→Range: best F={best_v2r['F_stat']:.1f} at lag={best_v2r['lag']}, p={best_v2r['p_value']:.2e}")
        print(f"    Range→Vol: best F={best_r2v['F_stat']:.1f} at lag={best_r2v['lag']}, p={best_r2v['p_value']:.2e}")

        if best_v2r['F_stat'] > best_r2v['F_stat']:
            print(f"    → VOLUME leads RANGE (stronger Granger signal)")
        else:
            print(f"    → RANGE leads VOLUME (stronger Granger signal)")

        # Buy ratio analysis
        print(f"  Buy ratio vs forward range...", flush=True)
        # Extreme buy ratio (>0.6 or <0.4) → next bar range
        high_buy = buy_ratio > 0.6
        low_buy = buy_ratio < 0.4
        neutral = (buy_ratio >= 0.4) & (buy_ratio <= 0.6)

        if high_buy.sum() > 100 and low_buy.sum() > 100:
            fwd_range = np.roll(log_range, -1)
            high_fwd = fwd_range[high_buy].mean()
            low_fwd = fwd_range[low_buy].mean()
            neutral_fwd = fwd_range[neutral].mean()
            print(f"    High buy ratio (>0.6): next range = {np.exp(high_fwd):.2f} bps (n={high_buy.sum():,})")
            print(f"    Low buy ratio (<0.4):  next range = {np.exp(low_fwd):.2f} bps (n={low_buy.sum():,})")
            print(f"    Neutral (0.4-0.6):     next range = {np.exp(neutral_fwd):.2f} bps (n={neutral.sum():,})")

        # Volume surge analysis
        surge = volume_surge_analysis(df, symbol)
        all_surge.extend(surge)
        for s in surge:
            print(f"    {s['threshold']} surge (n={s['n_surges']}): "
                  f"next bar {s.get('fwd_1_ratio',0):.2f}x, "
                  f"+30min {s.get('fwd_6_ratio',0):.2f}x, "
                  f"+60min {s.get('fwd_12_ratio',0):.2f}x baseline")

    # Save CSVs
    pd.DataFrame(all_granger).to_csv(RESULTS_DIR / 'v38_granger_causality.csv', index=False)
    print(f"\n  Saved: results/v38_granger_causality.csv")

    pd.DataFrame(all_surge).to_csv(RESULTS_DIR / 'v38_volume_surge.csv', index=False)
    print(f"  Saved: results/v38_volume_surge.csv")

    # Save cross-correlation curves
    cc_rows = []
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        row = {'lag_bars': lag, 'lag_minutes': lag * 5}
        for sym in SYMBOLS:
            row[f'{sym}_vol_leads_range'] = round(all_cc[sym]['vol_leads_range'][lag], 6)
        cc_rows.append(row)
    pd.DataFrame(cc_rows).to_csv(RESULTS_DIR / 'v38_vol_range_xcorr.csv', index=False)
    print(f"  Saved: results/v38_vol_range_xcorr.csv")

    # ---- PLOT 1: Volume→Range cross-correlation ----
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    for sym, color in zip(SYMBOLS, colors):
        lags = list(range(-MAX_LAG, MAX_LAG + 1))
        vals = [all_cc[sym]['vol_leads_range'][l] for l in lags]
        ax.plot([l*5 for l in lags], vals, label=sym.replace('USDT',''), color=color, linewidth=1.5)

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Lag (minutes) — positive = volume leads range', fontsize=11)
    ax.set_ylabel('Cross-correlation', fontsize=11)
    ax.set_title('v38: Volume → Range Cross-Correlation\n(5-min, 3+ years)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v38_vol_range_xcorr.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v38_vol_range_xcorr.png")

    # ---- PLOT 2: Granger F-stats comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    granger_df = pd.DataFrame(all_granger)

    for ax, direction, title in zip(axes,
                                     ['Volume→Range', 'Range→Volume'],
                                     ['Volume Granger-causes Range?', 'Range Granger-causes Volume?']):
        sub = granger_df[granger_df['direction'] == direction]
        for sym, color in zip(SYMBOLS, colors):
            ssub = sub[sub['symbol'] == sym]
            ax.bar(ssub['lag'] + SYMBOLS.index(sym)*0.15 - 0.3, ssub['F_stat'],
                   width=0.14, label=sym.replace('USDT',''), color=color, alpha=0.8)
        ax.set_xlabel('Lag (bars)', fontsize=10)
        ax.set_ylabel('F-statistic', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v38_granger_fstats.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v38_granger_fstats.png")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
