#!/usr/bin/env python3
"""
v36: Volatility Clustering Decay Function

How fast does volatility decay after a spike?
- Autocorrelation of 5-min range at lags 1-288 (5min to 24h)
- Fit exponential and power-law decay models → extract half-life
- Compare decay by asset and by session (Asia vs US shock)
- Conditional analysis: decay after 3σ+ events

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
from scipy.optimize import curve_fit
from scipy import stats

PARQUET_DIR = Path("parquet")
RESULTS_DIR = Path("results")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
SOURCE = "bybit_futures"
MAX_LAG = 288  # 288 bars × 5min = 24 hours


def load_ohlcv(symbol):
    t0 = time.time()
    ohlcv_dir = PARQUET_DIR / symbol / "ohlcv" / "5m" / SOURCE
    files = sorted(ohlcv_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp_us').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['timestamp_us'], unit='us', utc=True)
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['log_range'] = np.log1p(df['range_bps'])
    df['hour'] = df['datetime'].dt.hour
    print(f"  {symbol}: {len(df):,} bars ({time.time()-t0:.1f}s)", flush=True)
    return df


def compute_acf(x, max_lag):
    """Compute autocorrelation function."""
    x = x - x.mean()
    var = np.mean(x**2)
    n = len(x)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.mean(x[:n-lag] * x[lag:]) / var
    return acf


def exp_decay(x, a, b):
    return a * np.exp(-b * x)

def power_decay(x, a, b):
    return a * x**(-b)


def fit_decay(lags, acf_vals):
    """Fit exponential and power-law decay, return half-lives."""
    lags = np.array(lags, dtype=float)
    acf_vals = np.array(acf_vals, dtype=float)

    # Exponential fit
    try:
        popt_exp, _ = curve_fit(exp_decay, lags, acf_vals, p0=[0.5, 0.01], maxfev=5000)
        exp_halflife = np.log(2) / popt_exp[1]  # in bars
        exp_r2 = 1 - np.sum((acf_vals - exp_decay(lags, *popt_exp))**2) / np.sum((acf_vals - acf_vals.mean())**2)
    except:
        popt_exp = [0, 0]
        exp_halflife = np.nan
        exp_r2 = 0

    # Power-law fit
    try:
        popt_pow, _ = curve_fit(power_decay, lags, acf_vals, p0=[0.5, 0.5], maxfev=5000)
        pow_r2 = 1 - np.sum((acf_vals - power_decay(lags, *popt_pow))**2) / np.sum((acf_vals - acf_vals.mean())**2)
        # Half-life for power law: find lag where acf = acf[1]/2
        pow_halflife = (2**(1/popt_pow[1])) * lags[0] if popt_pow[1] > 0 else np.nan
    except:
        popt_pow = [0, 0]
        pow_halflife = np.nan
        pow_r2 = 0

    return {
        'exp_a': popt_exp[0], 'exp_b': popt_exp[1],
        'exp_halflife_bars': exp_halflife, 'exp_halflife_min': exp_halflife * 5,
        'exp_r2': exp_r2,
        'pow_a': popt_pow[0], 'pow_b': popt_pow[1],
        'pow_r2': pow_r2,
        'best_model': 'exponential' if exp_r2 > pow_r2 else 'power-law',
    }


def conditional_decay(df, threshold_sigma=3):
    """Compute average range trajectory after a vol spike (>threshold σ)."""
    log_range = df['log_range'].values
    mu = log_range.mean()
    sigma = log_range.std()
    threshold = mu + threshold_sigma * sigma

    spike_indices = np.where(log_range > threshold)[0]
    # Filter: at least 100 bars after spike
    spike_indices = spike_indices[spike_indices < len(log_range) - 100]

    if len(spike_indices) < 10:
        return None, 0

    # Average trajectory after spike
    trajectories = np.zeros((len(spike_indices), 100))
    for i, idx in enumerate(spike_indices):
        trajectories[i] = log_range[idx:idx+100]

    avg_trajectory = trajectories.mean(axis=0)
    # Normalize: express as ratio to bar-0
    norm_trajectory = avg_trajectory / avg_trajectory[0]

    return norm_trajectory, len(spike_indices)


def session_decay(df):
    """Compare vol decay after spikes in different sessions."""
    log_range = df['log_range'].values
    hours = df['hour'].values
    mu = log_range.mean()
    sigma = log_range.std()
    threshold = mu + 2 * sigma  # 2σ for more events

    results = {}
    for session_name, hour_range in [('Asia(0-7)', range(0, 8)),
                                      ('London(8-15)', range(8, 16)),
                                      ('NY(16-20)', range(16, 21))]:
        mask = np.isin(hours, list(hour_range))
        spike_idx = np.where((log_range > threshold) & mask)[0]
        spike_idx = spike_idx[spike_idx < len(log_range) - 60]

        if len(spike_idx) < 20:
            continue

        trajs = np.zeros((len(spike_idx), 60))
        for i, idx in enumerate(spike_idx):
            trajs[i] = log_range[idx:idx+60]

        avg = trajs.mean(axis=0)
        results[session_name] = {
            'trajectory': avg / avg[0],
            'n_spikes': len(spike_idx),
        }

    return results


def main():
    t0 = time.time()
    print("="*70)
    print("v36: Volatility Clustering Decay Function")
    print("="*70)

    all_acf = {}
    all_fits = []
    all_cond = {}
    all_session = {}

    for symbol in SYMBOLS:
        print(f"\nLoading {symbol}...")
        df = load_ohlcv(symbol)

        # Full ACF
        print(f"  Computing ACF (lags 1-{MAX_LAG})...", flush=True)
        acf = compute_acf(df['log_range'].values, MAX_LAG)
        all_acf[symbol] = acf

        # Fit decay
        lags_fit = np.arange(1, MAX_LAG + 1)
        fit = fit_decay(lags_fit, acf[1:])
        fit['symbol'] = symbol
        all_fits.append(fit)

        print(f"    Exponential: half-life = {fit['exp_halflife_min']:.0f} min, R² = {fit['exp_r2']:.3f}")
        print(f"    Power-law: R² = {fit['pow_r2']:.3f}")
        print(f"    Best model: {fit['best_model']}")

        # Conditional decay after 3σ spike
        cond_traj, n_spikes = conditional_decay(df, threshold_sigma=3)
        if cond_traj is not None:
            all_cond[symbol] = (cond_traj, n_spikes)
            # Find when it drops to 50% of initial
            half_idx = np.argmax(cond_traj < 0.5) if np.any(cond_traj < 0.5) else len(cond_traj)
            print(f"    3σ spikes: {n_spikes}, decay to 50% in {half_idx*5} min")

        # Session-specific decay
        sess = session_decay(df)
        all_session[symbol] = sess

    # ---- Save ACF CSV ----
    acf_rows = []
    for lag in range(MAX_LAG + 1):
        row = {'lag_bars': lag, 'lag_minutes': lag * 5}
        for sym in SYMBOLS:
            row[f'{sym}_acf'] = round(all_acf[sym][lag], 6)
        acf_rows.append(row)
    pd.DataFrame(acf_rows).to_csv(RESULTS_DIR / 'v36_vol_acf.csv', index=False)
    print(f"\n  Saved: results/v36_vol_acf.csv")

    # Save fits CSV
    pd.DataFrame(all_fits).to_csv(RESULTS_DIR / 'v36_vol_decay_fits.csv', index=False)
    print(f"  Saved: results/v36_vol_decay_fits.csv")

    # ---- PLOT 1: ACF curves ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    for sym, color in zip(SYMBOLS, colors):
        lags_min = np.arange(MAX_LAG + 1) * 5
        ax1.plot(lags_min, all_acf[sym], label=sym.replace('USDT',''), color=color, linewidth=1.2)

    ax1.set_xlabel('Lag (minutes)', fontsize=11)
    ax1.set_ylabel('Autocorrelation', fontsize=11)
    ax1.set_title('Volatility Autocorrelation (full 24h)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # Zoomed first 2 hours
    for sym, color in zip(SYMBOLS, colors):
        lags_min = np.arange(25) * 5
        ax2.plot(lags_min, all_acf[sym][:25], label=sym.replace('USDT',''), color=color, linewidth=1.5)

    ax2.set_xlabel('Lag (minutes)', fontsize=11)
    ax2.set_ylabel('Autocorrelation', fontsize=11)
    ax2.set_title('Zoomed: First 2 Hours', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v36_vol_acf_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v36_vol_acf_curves.png")

    # ---- PLOT 2: Conditional decay after 3σ spike ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for sym, color in zip(SYMBOLS, colors):
        if sym in all_cond:
            traj, n = all_cond[sym]
            ax.plot(np.arange(100)*5, traj, label=f'{sym.replace("USDT","")} (n={n})',
                    color=color, linewidth=1.5)

    ax.axhline(y=0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='50% decay')
    ax.set_xlabel('Minutes after 3σ vol spike', fontsize=11)
    ax.set_ylabel('Normalized range (ratio to spike bar)', fontsize=11)
    ax.set_title('v36: Volatility Decay After 3σ Spike\n(avg trajectory, 3+ years)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'v36_vol_spike_decay.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: results/v36_vol_spike_decay.png")

    # ---- PLOT 3: Session-specific decay (BTC) ----
    if 'BTCUSDT' in all_session and all_session['BTCUSDT']:
        fig, ax = plt.subplots(figsize=(10, 5))
        sess_colors = {'Asia(0-7)': '#4FC3F7', 'London(8-15)': '#FFB74D', 'NY(16-20)': '#EF5350'}
        for sess_name, data in all_session['BTCUSDT'].items():
            ax.plot(np.arange(60)*5, data['trajectory'],
                    label=f'{sess_name} (n={data["n_spikes"]})',
                    color=sess_colors.get(sess_name, 'gray'), linewidth=1.5)

        ax.axhline(y=0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel('Minutes after 2σ vol spike', fontsize=11)
        ax.set_ylabel('Normalized range', fontsize=11)
        ax.set_title('v36: BTC Vol Decay by Session of Origin\n(2σ+ spikes, 3+ years)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'v36_vol_decay_by_session.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: results/v36_vol_decay_by_session.png")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for fit in all_fits:
        print(f"  {fit['symbol']:>10s}: half-life={fit['exp_halflife_min']:.0f}min, "
              f"exp_R²={fit['exp_r2']:.3f}, pow_R²={fit['pow_r2']:.3f}, best={fit['best_model']}")

    elapsed = time.time() - t0
    print(f"\nDone! {elapsed:.1f}s total")


if __name__ == '__main__':
    main()
