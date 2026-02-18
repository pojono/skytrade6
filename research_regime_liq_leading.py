#!/usr/bin/env python3
"""
Deep dive: Are liquidations truly LEADING regime switches?

The v27 profile shows liquidations ramping up 30min before the GMM-labeled switch.
But is this:
  (a) Genuine leading signal — liquidations precede volatility increase
  (b) GMM labeling artifact — volatility already increased, GMM just labels it late

To distinguish, we compare:
  1. Liquidation profile vs ACTUAL volatility profile around switches
  2. Does liq spike BEFORE vol spikes, or simultaneously?
  3. If we use liq threshold as early warning, how much lead time do we get?
  4. Precision/recall of a simple liq-based early warning system
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import json
import gzip
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")


def load_ticker_to_5m_bars(symbol, data_dir='data'):
    """Load 5-second ticker data and aggregate to 5-minute OHLCV bars."""
    symbol_dir = Path(data_dir) / symbol
    ticker_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 200 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    r = data['result']['list'][0]
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'price': float(r['lastPrice']),
                    })
                except:
                    pass
    print(f" done ({len(records):,} ticks)")
    df = pd.DataFrame(records).sort_values('timestamp').drop_duplicates('timestamp', keep='last')
    df.set_index('timestamp', inplace=True)
    bars = df['price'].resample('5min').agg(['first', 'max', 'min', 'last', 'count'])
    bars.columns = ['open', 'high', 'low', 'close', 'tick_count']
    bars = bars.dropna(subset=['open'])
    bars = bars[bars['tick_count'] >= 5]
    bars['returns'] = bars['close'].pct_change()
    print(f"  Built {len(bars):,} 5-min bars")
    return bars


def load_liquidations_to_5m(symbol, data_dir='data'):
    """Load liquidation events and aggregate to 5-minute bars."""
    symbol_dir = Path(data_dir) / symbol
    liq_files = sorted(symbol_dir.glob("liquidation_*.jsonl.gz"))
    print(f"  Loading {len(liq_files)} liquidation files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 200 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for ev in data['result']['data']:
                            records.append({
                                'timestamp': pd.to_datetime(ev['T'], unit='ms'),
                                'side': ev['S'],
                                'volume': float(ev['v']),
                                'price': float(ev['p']),
                            })
                except:
                    continue
    print(f" done ({len(records):,} events)")
    df = pd.DataFrame(records).sort_values('timestamp')
    df['notional'] = df['volume'] * df['price']
    df.set_index('timestamp', inplace=True)
    liq_bars = pd.DataFrame()
    liq_bars['liq_count'] = df.resample('5min').size()
    liq_bars['liq_notional'] = df['notional'].resample('5min').sum()
    liq_bars = liq_bars.fillna(0)
    return liq_bars


def main():
    symbol = 'BTCUSDT'
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  DEEP DIVE: Are liquidations leading regime switches? ({symbol})")
    print(f"{'='*70}")

    bars = load_ticker_to_5m_bars(symbol)
    liq_bars = load_liquidations_to_5m(symbol)

    # Compute volatility
    ret = bars['returns']
    bars['rvol_1h'] = ret.rolling(12).std()
    bars['rvol_2h'] = ret.rolling(24).std()
    bars['rvol_4h'] = ret.rolling(48).std()
    log_hl = np.log(bars['high'].clip(lower=1e-10) / bars['low'].clip(lower=1e-10))
    bars['parkvol_1h'] = log_hl.rolling(12).mean() * np.sqrt(1.0 / (4 * np.log(2)))
    bars['parkvol_4h'] = log_hl.rolling(48).mean() * np.sqrt(1.0 / (4 * np.log(2)))

    # Efficiency
    abs_ret_sum = ret.abs().rolling(12).sum()
    net_ret = ret.rolling(12).sum().abs()
    bars['efficiency_1h'] = net_ret / abs_ret_sum.clip(lower=1e-10)
    abs_ret_sum2 = ret.abs().rolling(24).sum()
    net_ret2 = ret.rolling(24).sum().abs()
    bars['efficiency_2h'] = net_ret2 / abs_ret_sum2.clip(lower=1e-10)

    # Vol ratios
    bars['vol_ratio_1h_24h'] = bars['rvol_1h'] / bars['returns'].rolling(288).std().clip(lower=1e-10)
    bars['vol_ratio_1h_8h'] = bars['rvol_1h'] / bars['returns'].rolling(96).std().clip(lower=1e-10)
    bars['vol_accel_1h'] = bars['rvol_1h'].pct_change(12)

    # Merge
    merged = bars.join(liq_bars, how='inner').fillna(0)
    merged = merged.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # Smoothed liq rate (1h rolling)
    merged['liq_rate_1h'] = merged['liq_count'].rolling(12, min_periods=1).mean()

    # GMM K=2 regime labeling
    cluster_features = ['rvol_1h', 'rvol_2h', 'rvol_4h', 'parkvol_1h', 'parkvol_4h',
                        'vol_ratio_1h_24h', 'vol_ratio_1h_8h', 'vol_accel_1h',
                        'efficiency_1h', 'efficiency_2h']
    X = merged[cluster_features].dropna()
    valid_idx = X.index
    means = X.mean()
    stds = X.std().clip(lower=1e-10)
    X_scaled = (X - means) / stds
    gmm = GaussianMixture(n_components=2, covariance_type='diag', n_init=5, random_state=42)
    gmm.fit(X_scaled.values)
    labels = gmm.predict(X_scaled.values)

    # Align: 0 = quiet, 1 = volatile
    if merged.loc[valid_idx, 'rvol_1h'].values[labels == 0].mean() > \
       merged.loc[valid_idx, 'rvol_1h'].values[labels == 1].mean():
        labels = 1 - labels

    merged.loc[valid_idx, 'regime'] = labels
    merged['regime'] = merged['regime'].ffill().fillna(0).astype(int)

    # Find transitions
    regime_arr = merged['regime'].values
    transitions = []
    for i in range(1, len(regime_arr)):
        if regime_arr[i] != regime_arr[i-1]:
            transitions.append(i)

    print(f"\n  Regime transitions: {len(transitions)}")
    n_quiet = (regime_arr == 0).sum()
    n_vol = (regime_arr == 1).sum()
    print(f"  Quiet: {n_quiet} ({n_quiet/len(regime_arr)*100:.1f}%), Volatile: {n_vol} ({n_vol/len(regime_arr)*100:.1f}%)")

    # ================================================================
    # ANALYSIS 1: Compare volatility and liquidation profiles
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: Volatility vs Liquidation Profile Around Switches")
    print(f"{'='*70}")

    window = 24  # ±2h
    rvol_profiles = []
    liq_count_profiles = []
    liq_notional_profiles = []

    rvol_vals = merged['rvol_1h'].values
    liq_count_vals = merged['liq_count'].values
    liq_notional_vals = merged['liq_notional'].values

    for t in transitions:
        if t >= window and t + window < len(regime_arr):
            rvol_profiles.append(rvol_vals[t-window:t+window])
            liq_count_profiles.append(liq_count_vals[t-window:t+window])
            liq_notional_profiles.append(liq_notional_vals[t-window:t+window])

    if rvol_profiles:
        avg_rvol = np.nanmean(rvol_profiles, axis=0)
        avg_liq_count = np.mean(liq_count_profiles, axis=0)
        avg_liq_not = np.mean(liq_notional_profiles, axis=0)

        # Normalize to make comparison easier
        rvol_norm = (avg_rvol - avg_rvol.min()) / (avg_rvol.max() - avg_rvol.min() + 1e-10)
        liq_norm = (avg_liq_count - avg_liq_count.min()) / (avg_liq_count.max() - avg_liq_count.min() + 1e-10)

        print(f"\n  Profiles around {len(rvol_profiles)} transitions (±2h):")
        print(f"  {'Bar':>5s}  {'Time':>8s}  {'Avg rvol':>10s}  {'rvol_norm':>10s}  {'Avg LiqCnt':>10s}  {'liq_norm':>10s}  {'Avg LiqNot':>12s}")
        print(f"  {'-'*75}")
        for i in range(len(avg_rvol)):
            offset = (i - window) * 5
            marker = " ← SWITCH" if i == window else ""
            print(f"  {i-window:>+5d}  {offset:>+6d}min  {avg_rvol[i]:>10.6f}  {rvol_norm[i]:>10.3f}  "
                  f"{avg_liq_count[i]:>10.2f}  {liq_norm[i]:>10.3f}  ${avg_liq_not[i]:>11,.0f}{marker}")

    # ================================================================
    # ANALYSIS 2: When does each signal first cross its threshold?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: First Threshold Crossing — Liq vs Vol")
    print(f"{'='*70}")
    print(f"  For each transition, find when liq and vol first exceed their")
    print(f"  'elevated' threshold (75th percentile of their own distribution)")

    rvol_p75 = np.nanpercentile(rvol_vals[~np.isnan(rvol_vals)], 75)
    liq_p75 = np.percentile(liq_count_vals, 75)
    liq_rate_vals = merged['liq_rate_1h'].values
    liq_rate_p75 = np.percentile(liq_rate_vals, 75)

    print(f"  rvol_1h 75th percentile: {rvol_p75:.6f}")
    print(f"  liq_count 75th percentile: {liq_p75:.1f}")
    print(f"  liq_rate_1h 75th percentile: {liq_rate_p75:.1f}")

    # Only look at transitions INTO volatile regime
    to_volatile = [t for t in transitions if regime_arr[t] == 1]
    print(f"  Transitions to volatile: {len(to_volatile)}")

    lookback = 24  # 2h before switch
    rvol_first_cross = []
    liq_first_cross = []
    liq_rate_first_cross = []

    for t in to_volatile:
        if t < lookback:
            continue

        # Find first bar where rvol exceeds threshold (scanning backward from switch)
        rvol_cross = None
        for j in range(t, t - lookback - 1, -1):
            if j >= 0 and not np.isnan(rvol_vals[j]) and rvol_vals[j] >= rvol_p75:
                rvol_cross = j
            else:
                break
        if rvol_cross is not None:
            rvol_first_cross.append(t - rvol_cross)

        # Find first bar where liq_count exceeds threshold
        liq_cross = None
        for j in range(t, t - lookback - 1, -1):
            if j >= 0 and liq_count_vals[j] >= liq_p75:
                liq_cross = j
            else:
                break
        if liq_cross is not None:
            liq_first_cross.append(t - liq_cross)

        # Find first bar where liq_rate_1h exceeds threshold
        liq_rate_cross = None
        for j in range(t, t - lookback - 1, -1):
            if j >= 0 and liq_rate_vals[j] >= liq_rate_p75:
                liq_rate_cross = j
            else:
                break
        if liq_rate_cross is not None:
            liq_rate_first_cross.append(t - liq_rate_cross)

    print(f"\n  Lead time before GMM switch label (bars before, positive = earlier):")
    for name, arr in [("rvol_1h", rvol_first_cross),
                       ("liq_count", liq_first_cross),
                       ("liq_rate_1h", liq_rate_first_cross)]:
        if arr:
            arr = np.array(arr)
            print(f"    {name:>15s}: median={np.median(arr):>5.1f} bars ({np.median(arr)*5:.0f}min), "
                  f"mean={np.mean(arr):>5.1f}, p25={np.percentile(arr,25):.0f}, p75={np.percentile(arr,75):.0f}, "
                  f"n={len(arr)}")

    # ================================================================
    # ANALYSIS 3: Cross-correlation — does liq lead vol?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 3: Cross-Correlation — Does Liq Lead Vol?")
    print(f"{'='*70}")

    # Compute cross-correlation at different lags
    # Positive lag = liq leads vol
    rvol_clean = merged['rvol_1h'].fillna(method='ffill').fillna(0).values
    liq_clean = merged['liq_rate_1h'].fillna(0).values

    # Standardize
    rvol_z = (rvol_clean - np.mean(rvol_clean)) / (np.std(rvol_clean) + 1e-10)
    liq_z = (liq_clean - np.mean(liq_clean)) / (np.std(liq_clean) + 1e-10)

    n = len(rvol_z)
    lags = range(-24, 25)  # -2h to +2h
    xcorr = []
    for lag in lags:
        if lag >= 0:
            corr = np.mean(liq_z[:n-lag] * rvol_z[lag:])
        else:
            corr = np.mean(liq_z[-lag:] * rvol_z[:n+lag])
        xcorr.append(corr)

    print(f"\n  Cross-correlation: liq_rate_1h vs rvol_1h")
    print(f"  Positive lag = liq leads vol by N bars")
    print(f"  {'Lag':>5s}  {'Time':>8s}  {'XCorr':>8s}  {'Bar':>40s}")
    print(f"  {'-'*65}")

    max_corr = max(xcorr)
    for lag, xc in zip(lags, xcorr):
        bar_len = int(xc / max_corr * 30) if max_corr > 0 else 0
        bar = '█' * bar_len
        marker = " ← PEAK" if xc == max_corr else ""
        if lag % 3 == 0 or xc == max_corr:
            print(f"  {lag:>+5d}  {lag*5:>+6d}min  {xc:>+8.4f}  {bar}{marker}")

    peak_lag = list(lags)[xcorr.index(max_corr)]
    print(f"\n  Peak cross-correlation at lag={peak_lag} ({peak_lag*5}min)")
    if peak_lag > 0:
        print(f"  → Liquidations LEAD volatility by ~{peak_lag*5}min")
    elif peak_lag < 0:
        print(f"  → Volatility LEADS liquidations by ~{abs(peak_lag)*5}min")
    else:
        print(f"  → Simultaneous (no lead/lag)")

    # ================================================================
    # ANALYSIS 4: Granger-like causality test
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 4: Predictive Power — Liq Rate → Future Vol Change")
    print(f"{'='*70}")

    # Can current liq_rate predict FUTURE vol increase?
    # Compare: does high liq_rate now predict higher vol in 3/6/12 bars?
    for horizon in [3, 6, 12, 24]:
        future_vol = merged['rvol_1h'].shift(-horizon)
        current_vol = merged['rvol_1h']
        vol_change = (future_vol - current_vol) / current_vol.clip(lower=1e-10)

        liq_rate = merged['liq_rate_1h']

        # Split into quintiles of liq_rate
        valid = pd.DataFrame({'liq_rate': liq_rate, 'vol_change': vol_change}).dropna()
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

        if len(valid) < 1000:
            continue

        quintiles = pd.qcut(valid['liq_rate'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                           duplicates='drop')
        result = valid.groupby(quintiles)['vol_change'].agg(['mean', 'std', 'count'])

        print(f"\n  Horizon: {horizon} bars ({horizon*5}min)")
        print(f"  Future vol change by current liq_rate quintile:")
        print(f"  {'Quintile':>12s}  {'Mean Δvol':>10s}  {'Std':>10s}  {'Count':>8s}")
        print(f"  {'-'*45}")
        for q, row in result.iterrows():
            print(f"  {str(q):>12s}  {row['mean']:>+10.4f}  {row['std']:>10.4f}  {int(row['count']):>8d}")

        # Is Q5 significantly different from Q1?
        q1_vals = valid[quintiles == 'Q1(low)']['vol_change'].values
        q5_vals = valid[quintiles == 'Q5(high)']['vol_change'].values
        if len(q1_vals) > 50 and len(q5_vals) > 50:
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(q5_vals, q1_vals, alternative='two-sided')
            diff = np.mean(q5_vals) - np.mean(q1_vals)
            print(f"  Q5-Q1 diff: {diff:+.4f}, p-value: {pval:.6f} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''}")

    # ================================================================
    # ANALYSIS 5: Simple early warning system
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 5: Simple Liq-Based Early Warning System")
    print(f"{'='*70}")

    # If liq_rate_1h > threshold, predict regime switch within next N bars
    # Evaluate precision and recall

    liq_rate = merged['liq_rate_1h'].values
    regime = merged['regime'].values

    for threshold_pct in [75, 80, 85, 90, 95]:
        threshold = np.percentile(liq_rate, threshold_pct)

        for horizon in [6, 12, 24]:
            # Signal: liq_rate > threshold
            signal = liq_rate > threshold

            # Target: regime switches to volatile within next `horizon` bars
            target = np.zeros(len(regime), dtype=int)
            for t in to_volatile:
                for j in range(max(0, t - horizon), t):
                    target[j] = 1

            # Only evaluate on test set (last 30%)
            split = int(len(regime) * 0.7)
            sig_test = signal[split:]
            tgt_test = target[split:]

            tp = np.sum(sig_test & (tgt_test == 1))
            fp = np.sum(sig_test & (tgt_test == 0))
            fn = np.sum(~sig_test & (tgt_test == 1))
            tn = np.sum(~sig_test & (tgt_test == 0))

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            base_rate = tgt_test.mean()

            if threshold_pct in [75, 90]:  # only print selected
                print(f"  P{threshold_pct} thr={threshold:.1f}, horizon={horizon*5}min: "
                      f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} "
                      f"(base={base_rate:.3f}, lift={prec/max(base_rate,1e-3):.1f}x)")

    # ================================================================
    # ANALYSIS 6: Incremental value — what does liq add BEYOND vol?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 6: Incremental Value — Liq After Controlling for Vol")
    print(f"{'='*70}")

    # Bin by rvol_1h quintile, then within each bin, does liq_rate predict transitions?
    valid = merged[['rvol_1h', 'liq_rate_1h', 'regime']].dropna().copy()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

    # Create transition target (within next 12 bars)
    regime_arr_v = valid['regime'].values
    target = np.zeros(len(regime_arr_v), dtype=int)
    for i in range(1, len(regime_arr_v)):
        if regime_arr_v[i] != regime_arr_v[i-1]:
            for j in range(max(0, i-12), i):
                target[j] = 1
    valid['target'] = target

    vol_quintiles = pd.qcut(valid['rvol_1h'], 5, labels=['V1(low)', 'V2', 'V3', 'V4', 'V5(high)'],
                            duplicates='drop')

    print(f"\n  Within each vol quintile, does high liq_rate predict transitions?")
    print(f"  {'Vol Q':>10s}  {'Low Liq Trans%':>15s}  {'High Liq Trans%':>16s}  {'Diff':>8s}  {'p-val':>8s}")
    print(f"  {'-'*65}")

    from scipy.stats import mannwhitneyu

    for vq in vol_quintiles.cat.categories:
        subset = valid[vol_quintiles == vq]
        if len(subset) < 200:
            continue

        liq_median = subset['liq_rate_1h'].median()
        low_liq = subset[subset['liq_rate_1h'] <= liq_median]
        high_liq = subset[subset['liq_rate_1h'] > liq_median]

        low_trans = low_liq['target'].mean()
        high_trans = high_liq['target'].mean()
        diff = high_trans - low_trans

        try:
            stat, pval = mannwhitneyu(
                high_liq['target'].values, low_liq['target'].values, alternative='two-sided')
        except:
            pval = 1.0

        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"  {str(vq):>10s}  {low_trans:>14.3f}  {high_trans:>15.3f}  {diff:>+8.3f}  {pval:>7.4f}{sig}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Complete in {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
