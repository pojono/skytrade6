#!/usr/bin/env python3
"""
Regime Detection + Liquidation Features Research (v27)

Can liquidation data improve regime detection and prediction?

Hypothesis: Liquidation cascades, imbalance spikes, and rate changes
are LEADING indicators of regime switches. They represent forced
exits that precede volatility regime changes.

Approach:
1. Build 5-min OHLCV bars from ticker data
2. Compute standard regime features (vol, efficiency, etc.)
3. Label regimes using GMM K=2 (quiet vs volatile)
4. Compute liquidation features per 5-min bar
5. Test if liquidation features predict regime switches
6. Compare prediction with and without liquidation features

Data: May 11 - Aug 10, 2025 + Feb 9-17, 2026 (100 days)
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, silhouette_score)

warnings.filterwarnings("ignore")


def sanitize_features(X):
    """Replace inf/nan and clip extreme values for sklearn compatibility."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e10, 1e10)
    return X


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ticker_to_5m_bars(symbol, data_dir='data'):
    """Load 5-second ticker data and aggregate to 5-minute OHLCV bars."""
    symbol_dir = Path(data_dir) / symbol
    ticker_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    if not ticker_files:
        raise ValueError(f"No ticker files found for {symbol}")

    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
    records = []
    errors = 0
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
                        'volume24h': float(r.get('volume24h', 0)),
                        'turnover24h': float(r.get('turnover24h', 0)),
                    })
                except:
                    errors += 1
    print(f" done ({len(records):,} ticks, {errors} errors)")

    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp', keep='last')

    # Aggregate to 5-minute bars
    df.set_index('timestamp', inplace=True)
    bars = df['price'].resample('5min').agg(['first', 'max', 'min', 'last', 'count'])
    bars.columns = ['open', 'high', 'low', 'close', 'tick_count']
    bars = bars.dropna(subset=['open'])
    bars = bars[bars['tick_count'] >= 5]  # at least 5 ticks per bar

    # Compute returns
    bars['returns'] = bars['close'].pct_change()

    print(f"  Built {len(bars):,} 5-min bars from {bars.index.min()} to {bars.index.max()}")
    return bars


def load_liquidations_to_5m(symbol, data_dir='data'):
    """Load liquidation events and aggregate to 5-minute bars."""
    symbol_dir = Path(data_dir) / symbol
    liq_files = sorted(symbol_dir.glob("liquidation_*.jsonl.gz"))
    if not liq_files:
        raise ValueError(f"No liquidation files found for {symbol}")

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

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    df['is_buy'] = (df['side'] == 'Buy').astype(int)

    # Aggregate to 5-minute bars
    df.set_index('timestamp', inplace=True)

    liq_bars = pd.DataFrame(index=df.resample('5min').size().index)
    liq_bars['liq_count'] = df.resample('5min').size()
    liq_bars['liq_notional'] = df['notional'].resample('5min').sum()
    liq_bars['liq_buy_notional'] = df[df['is_buy'] == 1]['notional'].resample('5min').sum()
    liq_bars['liq_sell_notional'] = df[df['is_buy'] == 0]['notional'].resample('5min').sum()
    liq_bars['liq_buy_count'] = df['is_buy'].resample('5min').sum()
    liq_bars['liq_sell_count'] = (1 - df['is_buy']).resample('5min').sum()
    liq_bars['liq_max_notional'] = df['notional'].resample('5min').max()

    liq_bars = liq_bars.fillna(0)

    # Derived features
    total = liq_bars['liq_buy_notional'] + liq_bars['liq_sell_notional']
    liq_bars['liq_imbalance'] = (liq_bars['liq_buy_notional'] - liq_bars['liq_sell_notional']) / total.clip(lower=1)
    liq_bars['liq_imbalance'] = liq_bars['liq_imbalance'].fillna(0)

    print(f"  Built {len(liq_bars):,} 5-min liquidation bars")
    return liq_bars


# ============================================================================
# REGIME FEATURE ENGINEERING
# ============================================================================

def compute_regime_features(bars):
    """Compute backward-looking regime features from 5-min OHLCV bars."""
    c = bars['close'].values
    h = bars['high'].values
    l = bars['low'].values
    ret = bars['returns'].values

    # Volatility features
    for w, label in [(12, '1h'), (24, '2h'), (48, '4h'), (96, '8h'), (288, '24h')]:
        bars[f'rvol_{label}'] = pd.Series(ret).rolling(w).std().values
        log_hl = np.log(np.maximum(h, 1e-10) / np.maximum(l, 1e-10))
        bars[f'parkvol_{label}'] = pd.Series(log_hl).rolling(w).mean().values * np.sqrt(1.0 / (4 * np.log(2)))

    # Vol ratios
    bars['vol_ratio_1h_24h'] = bars['rvol_1h'] / bars['rvol_24h'].clip(lower=1e-10)
    bars['vol_ratio_1h_8h'] = bars['rvol_1h'] / bars['rvol_8h'].clip(lower=1e-10)

    # Vol acceleration
    bars['vol_accel_1h'] = bars['rvol_1h'].pct_change(12)

    # Efficiency ratio
    for w, label in [(12, '1h'), (24, '2h'), (48, '4h')]:
        abs_ret_sum = pd.Series(np.abs(ret)).rolling(w).sum()
        net_ret = pd.Series(ret).rolling(w).sum().abs()
        bars[f'efficiency_{label}'] = (net_ret / abs_ret_sum.clip(lower=1e-10)).values

    # Momentum
    for w, label in [(12, '1h'), (24, '2h'), (48, '4h')]:
        bars[f'momentum_{label}'] = pd.Series(c).pct_change(w).values

    return bars


def compute_liquidation_features(liq_bars):
    """Compute rolling liquidation features for regime prediction."""
    lb = liq_bars.copy()

    # Rolling liquidation rate (count per 5-min bar, smoothed)
    for w, label in [(12, '1h'), (24, '2h'), (48, '4h'), (96, '8h')]:
        lb[f'liq_rate_{label}'] = lb['liq_count'].rolling(w, min_periods=1).mean()
        lb[f'liq_notional_{label}'] = lb['liq_notional'].rolling(w, min_periods=1).mean()

    # Liquidation rate z-score (vs 24h rolling)
    mean_24h = lb['liq_count'].rolling(288, min_periods=12).mean()
    std_24h = lb['liq_count'].rolling(288, min_periods=12).std()
    lb['liq_rate_zscore'] = (lb['liq_count'] - mean_24h) / std_24h.clip(lower=0.1)

    # Notional z-score
    not_mean = lb['liq_notional'].rolling(288, min_periods=12).mean()
    not_std = lb['liq_notional'].rolling(288, min_periods=12).std()
    lb['liq_notional_zscore'] = (lb['liq_notional'] - not_mean) / not_std.clip(lower=1)

    # Rolling imbalance (smoothed)
    for w, label in [(6, '30m'), (12, '1h'), (24, '2h')]:
        lb[f'liq_imb_{label}'] = lb['liq_imbalance'].rolling(w, min_periods=1).mean()

    # Imbalance absolute value (extreme one-sided liquidations)
    lb['liq_imb_abs_1h'] = lb['liq_imbalance'].abs().rolling(12, min_periods=1).mean()

    # Rate of change of liquidation rate
    for w, label in [(3, '15m'), (6, '30m'), (12, '1h')]:
        lb[f'liq_rate_roc_{label}'] = lb['liq_rate_1h'].pct_change(w).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Cascade indicator: count of bars with >2 liquidations in last hour
    lb['liq_active_bars_1h'] = (lb['liq_count'] > 2).rolling(12, min_periods=1).sum()

    # Buy/sell ratio
    total = lb['liq_buy_count'] + lb['liq_sell_count']
    lb['liq_buy_ratio'] = lb['liq_buy_count'] / total.clip(lower=1)

    # Max single liquidation relative to mean
    denom = lb['liq_notional'].rolling(288, min_periods=12).mean().clip(lower=1)
    lb['liq_max_ratio'] = (lb['liq_max_notional'] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Liquidation acceleration (is rate increasing?)
    lb['liq_accel'] = lb['liq_rate_1h'].diff(6)

    return lb


# ============================================================================
# REGIME LABELING (GMM K=2)
# ============================================================================

def label_regimes_gmm(bars, features_for_clustering):
    """Label regimes using GMM K=2 (quiet vs volatile)."""
    X = bars[features_for_clustering].dropna()
    valid_idx = X.index

    # Standardize
    means = X.mean()
    stds = X.std().clip(lower=1e-10)
    X_scaled = (X - means) / stds

    # Fit GMM
    gmm = GaussianMixture(n_components=2, covariance_type='diag',
                           n_init=5, random_state=42)
    gmm.fit(X_scaled.values)
    labels = gmm.predict(X_scaled.values)

    # Align: regime 0 = quiet (lower volatility)
    r0_vol = bars.loc[valid_idx, 'rvol_1h'].values[labels == 0].mean()
    r1_vol = bars.loc[valid_idx, 'rvol_1h'].values[labels == 1].mean()
    if r0_vol > r1_vol:
        labels = 1 - labels

    sil = silhouette_score(X_scaled.values, labels, sample_size=min(10000, len(X_scaled)))

    # Assign to bars
    bars.loc[valid_idx, 'regime'] = labels
    bars['regime'] = bars['regime'].ffill().fillna(0).astype(int)

    # Stats
    n_volatile = (labels == 1).sum()
    n_quiet = (labels == 0).sum()
    transitions = np.sum(np.diff(labels) != 0)

    print(f"  GMM K=2: quiet={n_quiet} ({n_quiet/len(labels)*100:.1f}%), "
          f"volatile={n_volatile} ({n_volatile/len(labels)*100:.1f}%), "
          f"transitions={transitions}, silhouette={sil:.3f}")

    return bars, labels, valid_idx, gmm


def find_transitions(labels):
    """Find indices where regime changes."""
    trans = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            trans.append(i)
    return trans


def episode_stats(labels):
    """Compute episode duration statistics."""
    episodes = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            episodes.append({'regime': current, 'length': i - start})
            current = labels[i]
            start = i
    episodes.append({'regime': current, 'length': len(labels) - start})
    return pd.DataFrame(episodes)


# ============================================================================
# EXPERIMENT 1: Liquidation Feature Correlation with Regime
# ============================================================================

def exp1_correlation(merged, liq_features, symbol):
    """How well do liquidation features correlate with current regime?"""
    print(f"\n{'='*70}")
    print(f"  EXP 1: LIQUIDATION FEATURE CORRELATION WITH REGIME — {symbol}")
    print(f"{'='*70}")

    valid = merged.dropna(subset=['regime'] + liq_features)
    if len(valid) < 100:
        print("  Not enough data")
        return {}

    results = {}
    print(f"\n  {'Feature':>35s}  {'Corr':>8s}  {'Quiet mean':>12s}  {'Volatile mean':>14s}  {'Ratio':>8s}")
    print(f"  {'-'*85}")

    for feat in liq_features:
        vals = valid[feat].values
        regime = valid['regime'].values

        corr = np.corrcoef(vals, regime)[0, 1]
        quiet_mean = vals[regime == 0].mean()
        vol_mean = vals[regime == 1].mean()
        ratio = vol_mean / max(abs(quiet_mean), 1e-10)

        results[feat] = {'corr': corr, 'quiet_mean': quiet_mean, 'vol_mean': vol_mean, 'ratio': ratio}
        print(f"  {feat:>35s}  {corr:>+8.3f}  {quiet_mean:>12.4f}  {vol_mean:>14.4f}  {ratio:>8.2f}x")

    # Top 5 by absolute correlation
    top = sorted(results.items(), key=lambda x: abs(x[1]['corr']), reverse=True)[:5]
    print(f"\n  Top 5 by |correlation|:")
    for feat, r in top:
        print(f"    {feat:>35s}  corr={r['corr']:+.3f}  ratio={r['ratio']:.2f}x")

    return results


# ============================================================================
# EXPERIMENT 2: Liquidation Features as Leading Indicators
# ============================================================================

def exp2_leading_indicators(merged, liq_features, symbol):
    """Do liquidation features LEAD regime switches?"""
    print(f"\n{'='*70}")
    print(f"  EXP 2: LIQUIDATION AS LEADING INDICATORS — {symbol}")
    print(f"{'='*70}")

    labels = merged['regime'].values
    transitions = find_transitions(labels)
    print(f"  Total regime transitions: {len(transitions)}")

    if len(transitions) < 20:
        print("  Not enough transitions")
        return {}

    # For each transition, look at liquidation features in the window BEFORE
    windows = [3, 6, 12, 24]  # bars before transition (15m, 30m, 1h, 2h)
    results = {}

    from scipy.stats import mannwhitneyu

    # Pre-compute exclusion zones around transitions
    trans_set = set()
    for t in transitions:
        for offset in range(-48, 49):  # ±4h exclusion zone
            trans_set.add(t + offset)

    for feat in liq_features[:15]:  # top features only
        feat_vals = merged[feat].values
        feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)

        lead_scores = {}
        for w in windows:
            # Average feature value in window before transition
            pre_trans_vals = []
            for t in transitions:
                if t >= w:
                    pre_trans_vals.append(np.nanmean(feat_vals[t-w:t]))

            # Random non-transition windows for comparison
            non_trans_indices = [i for i in range(w, len(labels) - w)
                                if i not in trans_set]
            np.random.seed(42)
            n_sample = min(len(non_trans_indices), len(pre_trans_vals) * 3)
            if n_sample > 10 and pre_trans_vals:
                sample = np.random.choice(non_trans_indices, size=n_sample, replace=False)
                non_trans_vals = [np.nanmean(feat_vals[i-w:i]) for i in sample]

                try:
                    stat, pval = mannwhitneyu(pre_trans_vals, non_trans_vals, alternative='two-sided')
                    effect = np.mean(pre_trans_vals) - np.mean(non_trans_vals)
                    lead_scores[w] = {'effect': effect, 'pval': pval,
                                      'pre_mean': np.mean(pre_trans_vals),
                                      'non_mean': np.mean(non_trans_vals)}
                except:
                    pass

        if lead_scores:
            results[feat] = lead_scores

    # Print results
    print(f"\n  {'Feature':>35s}", end='')
    for w in windows:
        print(f"  {'effect':>8s} {'p-val':>7s}", end='')
    print()
    print(f"  {'-'*35}", end='')
    for w in windows:
        print(f"  {'-'*8} {'-'*7}", end='')
    print()

    for feat, scores in results.items():
        print(f"  {feat:>35s}", end='')
        for w in windows:
            if w in scores:
                s = scores[w]
                sig = '***' if s['pval'] < 0.001 else '**' if s['pval'] < 0.01 else '*' if s['pval'] < 0.05 else ''
                print(f"  {s['effect']:>+8.4f} {s['pval']:>6.4f}{sig}", end='')
            else:
                print(f"  {'':>8s} {'':>7s}", end='')
        print()

    return results


# ============================================================================
# EXPERIMENT 3: Regime Switch Prediction (with vs without liquidation features)
# ============================================================================

def exp3_prediction(merged, regime_features, liq_features, symbol):
    """Can liquidation features improve regime switch prediction?"""
    print(f"\n{'='*70}")
    print(f"  EXP 3: REGIME SWITCH PREDICTION — {symbol}")
    print(f"{'='*70}")

    labels = merged['regime'].values
    transitions = set(find_transitions(labels))
    n = len(merged)

    # Feature sets
    base_features = [f for f in regime_features if f in merged.columns and not merged[f].isna().all()]
    liq_feats = [f for f in liq_features if f in merged.columns and not merged[f].isna().all()]
    all_features = base_features + liq_feats

    print(f"  Base features: {len(base_features)}")
    print(f"  Liquidation features: {len(liq_feats)}")
    print(f"  Total features: {len(all_features)}")

    # Train/test split (70/30 time-based)
    split = int(n * 0.7)
    warmup = 288  # 1 day

    horizons = [6, 12, 24, 48]  # 30m, 1h, 2h, 4h

    all_results = {}

    for horizon in horizons:
        # Target: will regime switch in next `horizon` bars?
        target = np.zeros(n, dtype=int)
        for t in transitions:
            start = max(0, t - horizon)
            for j in range(start, t):
                target[j] = 1

        base_rate = target[split:].mean()

        print(f"\n  Horizon: {horizon} bars ({horizon*5}min = {horizon*5/60:.1f}h)")
        print(f"  Base rate: {base_rate:.3f} ({base_rate*100:.1f}%)")
        print(f"  Transitions in test: {sum(1 for t in transitions if t >= split)}")

        print(f"\n  {'Features':>15s}  {'Model':>16s}  {'Acc':>8s}  {'Prec':>8s}  "
              f"{'Recall':>8s}  {'F1':>8s}  {'AUC':>8s}")
        print(f"  {'-'*80}")

        horizon_results = {}

        for feat_name, feat_cols in [("Base only", base_features),
                                      ("Liq only", liq_feats),
                                      ("Base + Liq", all_features)]:
            X = sanitize_features(merged[feat_cols].values)
            X_train, X_test = X[warmup:split], X[split:]
            y_train, y_test = target[warmup:split], target[split:]

            for model_name, model in [
                ("Random Forest", RandomForestClassifier(
                    n_estimators=100, max_depth=8, min_samples_leaf=50,
                    class_weight='balanced', random_state=42, n_jobs=-1)),
                ("Gradient Boost", GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, min_samples_leaf=50,
                    subsample=0.8, random_state=42)),
            ]:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, zero_division=0)
                rec = recall_score(y_test, pred, zero_division=0)
                f1 = f1_score(y_test, pred, zero_division=0)
                try:
                    auc = roc_auc_score(y_test, prob)
                except:
                    auc = 0.5

                key = f"{feat_name}_{model_name}"
                horizon_results[key] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}

                print(f"  {feat_name:>15s}  {model_name:>16s}  {acc:>8.3f}  {prec:>8.3f}  "
                      f"{rec:>8.3f}  {f1:>8.3f}  {auc:>8.3f}")

                # Feature importance for combined model at 1h horizon
                if horizon == 12 and feat_name == "Base + Liq" and model_name == "Random Forest":
                    importances = model.feature_importances_
                    top_idx = np.argsort(importances)[::-1][:20]
                    print(f"\n  Top 20 features (Base+Liq RF, 1h horizon):")
                    for rank, fi in enumerate(top_idx):
                        marker = " ★" if feat_cols[fi].startswith('liq_') else ""
                        print(f"    {rank+1:>2d}. {feat_cols[fi]:>35s}  imp={importances[fi]:.4f}{marker}")

        all_results[horizon] = horizon_results

    return all_results


# ============================================================================
# EXPERIMENT 4: Regime-Conditional Liquidation Patterns
# ============================================================================

def exp4_regime_conditional(merged, symbol):
    """How do liquidation patterns differ between regimes?"""
    print(f"\n{'='*70}")
    print(f"  EXP 4: REGIME-CONDITIONAL LIQUIDATION PATTERNS — {symbol}")
    print(f"{'='*70}")

    valid = merged.dropna(subset=['regime'])
    quiet = valid[valid['regime'] == 0]
    volatile = valid[valid['regime'] == 1]

    metrics = [
        ('liq_count', 'Liquidations per 5min'),
        ('liq_notional', 'Notional per 5min ($)'),
        ('liq_imbalance', 'Imbalance (buy-sell)'),
        ('liq_max_notional', 'Max single liq ($)'),
        ('liq_buy_ratio', 'Buy ratio'),
        ('liq_rate_zscore', 'Rate z-score'),
        ('liq_notional_zscore', 'Notional z-score'),
        ('liq_imb_abs_1h', 'Abs imbalance 1h'),
        ('liq_active_bars_1h', 'Active bars in 1h'),
        ('liq_accel', 'Rate acceleration'),
    ]

    print(f"\n  {'Metric':>30s}  {'Quiet mean':>12s}  {'Volatile mean':>14s}  {'Ratio':>8s}  {'Quiet std':>12s}  {'Vol std':>12s}")
    print(f"  {'-'*95}")

    for col, name in metrics:
        if col not in valid.columns:
            continue
        q_mean = quiet[col].mean()
        v_mean = volatile[col].mean()
        q_std = quiet[col].std()
        v_std = volatile[col].std()
        ratio = v_mean / max(abs(q_mean), 1e-10)
        print(f"  {name:>30s}  {q_mean:>12.4f}  {v_mean:>14.4f}  {ratio:>8.2f}x  {q_std:>12.4f}  {v_std:>12.4f}")

    # Transition analysis: what happens to liquidations around regime switches?
    print(f"\n  --- Liquidation Behavior Around Regime Switches ---")
    labels = valid['regime'].values
    transitions = find_transitions(labels)
    liq_counts = valid['liq_count'].values
    liq_notionals = valid['liq_notional'].values

    if len(transitions) > 20:
        # Average liquidation profile around transitions
        window = 12  # 1 hour before and after
        profiles_count = []
        profiles_notional = []

        for t in transitions:
            if t >= window and t + window < len(labels):
                profiles_count.append(liq_counts[t-window:t+window])
                profiles_notional.append(liq_notionals[t-window:t+window])

        if profiles_count:
            avg_count = np.mean(profiles_count, axis=0)
            avg_notional = np.mean(profiles_notional, axis=0)

            print(f"\n  Average liq count around transitions (±1h, {len(profiles_count)} transitions):")
            print(f"  {'Bar':>5s}  {'Time':>8s}  {'Avg Count':>10s}  {'Avg Notional':>14s}")
            print(f"  {'-'*42}")
            for i, (c, n) in enumerate(zip(avg_count, avg_notional)):
                offset = (i - window) * 5
                marker = " ← SWITCH" if i == window else ""
                print(f"  {i-window:>+5d}  {offset:>+6d}min  {c:>10.2f}  ${n:>13,.0f}{marker}")

    return {}


# ============================================================================
# EXPERIMENT 5: Liquidation-Enhanced Regime Detection Speed
# ============================================================================

def exp5_detection_speed(merged, regime_features, liq_features, symbol):
    """Can liquidation features detect regime switches faster?"""
    print(f"\n{'='*70}")
    print(f"  EXP 5: REGIME DETECTION SPEED — {symbol}")
    print(f"{'='*70}")

    labels = merged['regime'].values
    transitions = find_transitions(labels)
    n = len(merged)

    if len(transitions) < 20:
        print("  Not enough transitions")
        return {}

    # Build classifiers for current regime (not prediction — detection)
    base_feats = [f for f in regime_features if f in merged.columns and not merged[f].isna().all()]
    liq_feats = [f for f in liq_features if f in merged.columns and not merged[f].isna().all()]
    all_feats = base_feats + liq_feats

    split = int(n * 0.7)
    warmup = 288

    results = {}

    for feat_name, feat_cols in [("Base only", base_feats),
                                  ("Base + Liq", all_feats)]:
        X = sanitize_features(merged[feat_cols].values)
        X_train, X_test = X[warmup:split], X[split:]
        y_train, y_test = labels[warmup:split], labels[split:]

        rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     min_samples_leaf=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        probs = rf.predict_proba(X_test)[:, 1]  # P(volatile)
        pred = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)

        # Detection lag at transitions in test set
        test_transitions = [t - split for t in transitions if t >= split and t < n]
        lags = []
        for t in test_transitions:
            if t < 0 or t >= len(pred):
                continue
            new_regime = y_test[t]
            for lag in range(0, min(50, len(pred) - t)):
                if pred[t + lag] == new_regime:
                    lags.append(lag)
                    break
            else:
                lags.append(50)

        if lags:
            lags = np.array(lags)
            med_lag = np.median(lags)
            mean_lag = np.mean(lags)
            pct_lt3 = (lags < 3).mean() * 100

            # False switches
            pred_trans = np.sum(np.diff(pred) != 0)
            gt_trans = len(test_transitions)
            false_sw = max(0, pred_trans - gt_trans)
            days = len(y_test) / 288
            false_per_day = false_sw / days

            results[feat_name] = {
                'acc': acc, 'med_lag': med_lag, 'mean_lag': mean_lag,
                'pct_lt3': pct_lt3, 'false_per_day': false_per_day
            }

            print(f"\n  {feat_name}:")
            print(f"    Accuracy:        {acc:.3f}")
            print(f"    Median lag:      {med_lag:.0f} bars ({med_lag*5:.0f}min)")
            print(f"    Mean lag:        {mean_lag:.1f} bars ({mean_lag*5:.0f}min)")
            print(f"    % detected <3b:  {pct_lt3:.1f}%")
            print(f"    False sw/day:    {false_per_day:.1f}")

    # Compare
    if 'Base only' in results and 'Base + Liq' in results:
        b = results['Base only']
        bl = results['Base + Liq']
        print(f"\n  --- Improvement from adding liquidation features ---")
        print(f"    Accuracy:     {b['acc']:.3f} → {bl['acc']:.3f} ({(bl['acc']-b['acc'])*100:+.1f}pp)")
        print(f"    Median lag:   {b['med_lag']:.0f} → {bl['med_lag']:.0f} bars")
        print(f"    Mean lag:     {b['mean_lag']:.1f} → {bl['mean_lag']:.1f} bars")
        print(f"    False sw/day: {b['false_per_day']:.1f} → {bl['false_per_day']:.1f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    """Run all experiments for one symbol."""
    print(f"\n{'='*70}")
    print(f"  {symbol} — REGIME + LIQUIDATION RESEARCH")
    print(f"{'='*70}")
    t0 = time.time()

    # Load data
    print(f"\n  --- Loading Data ---")
    bars = load_ticker_to_5m_bars(symbol, data_dir)
    liq_bars = load_liquidations_to_5m(symbol, data_dir)

    # Compute regime features
    print(f"\n  --- Computing Regime Features ---")
    bars = compute_regime_features(bars)

    # Compute liquidation features
    print(f"\n  --- Computing Liquidation Features ---")
    liq_bars = compute_liquidation_features(liq_bars)

    # Merge on 5-min index
    print(f"\n  --- Merging Data ---")
    merged = bars.join(liq_bars, how='inner')
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.fillna(method='ffill').fillna(0)
    print(f"  Merged: {len(merged):,} bars ({merged.index.min()} to {merged.index.max()})")

    # Label regimes
    print(f"\n  --- Labeling Regimes (GMM K=2) ---")
    cluster_features = ['rvol_1h', 'rvol_2h', 'rvol_4h', 'parkvol_1h', 'parkvol_4h',
                        'vol_ratio_1h_24h', 'vol_ratio_1h_8h', 'vol_accel_1h',
                        'efficiency_1h', 'efficiency_2h']
    cluster_features = [f for f in cluster_features if f in merged.columns]
    merged, labels, valid_idx, gmm = label_regimes_gmm(merged, cluster_features)

    # Episode stats
    ep = episode_stats(labels)
    for r in [0, 1]:
        rname = 'quiet' if r == 0 else 'volatile'
        subset = ep[ep['regime'] == r]['length']
        if len(subset) > 0:
            print(f"    {rname:>10s}: n={len(subset):>5d}, median={subset.median():>5.0f} bars ({subset.median()*5:.0f}m), "
                  f"mean={subset.mean():>6.1f}")

    # Define feature lists
    regime_features = ['rvol_1h', 'rvol_2h', 'rvol_4h', 'rvol_8h', 'rvol_24h',
                       'parkvol_1h', 'parkvol_4h',
                       'vol_ratio_1h_24h', 'vol_ratio_1h_8h', 'vol_accel_1h',
                       'efficiency_1h', 'efficiency_2h', 'efficiency_4h',
                       'momentum_1h', 'momentum_2h', 'momentum_4h']

    liq_features = ['liq_count', 'liq_notional', 'liq_imbalance',
                    'liq_rate_1h', 'liq_rate_2h', 'liq_rate_4h', 'liq_rate_8h',
                    'liq_notional_1h', 'liq_notional_2h', 'liq_notional_4h',
                    'liq_rate_zscore', 'liq_notional_zscore',
                    'liq_imb_30m', 'liq_imb_1h', 'liq_imb_2h',
                    'liq_imb_abs_1h', 'liq_rate_roc_15m', 'liq_rate_roc_30m', 'liq_rate_roc_1h',
                    'liq_active_bars_1h', 'liq_buy_ratio', 'liq_max_ratio',
                    'liq_accel', 'liq_max_notional', 'liq_buy_count', 'liq_sell_count']

    # Filter to existing columns
    regime_features = [f for f in regime_features if f in merged.columns]
    liq_features = [f for f in liq_features if f in merged.columns]

    # Run experiments
    r1 = exp1_correlation(merged, liq_features, symbol)
    r2 = exp2_leading_indicators(merged, liq_features, symbol)
    r3 = exp3_prediction(merged, regime_features, liq_features, symbol)
    r4 = exp4_regime_conditional(merged, symbol)
    r5 = exp5_detection_speed(merged, regime_features, liq_features, symbol)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  {symbol} complete in {elapsed:.1f}s")
    print(f"{'='*70}")

    return {'correlation': r1, 'leading': r2, 'prediction': r3,
            'conditional': r4, 'detection': r5}


def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    t_start = time.time()

    all_results = {}
    for symbol in symbols:
        try:
            all_results[symbol] = run_symbol(symbol)
        except Exception as e:
            print(f"\n  ✗ {symbol} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Grand summary
    print(f"\n\n{'='*70}")
    print(f"  GRAND SUMMARY — REGIME + LIQUIDATION RESEARCH")
    print(f"{'='*70}")

    # Prediction improvement summary
    print(f"\n  --- Regime Switch Prediction: AUC Improvement from Liquidation Features ---")
    print(f"  {'Symbol':>10s}", end='')
    for h in [6, 12, 24, 48]:
        print(f"  {'Base':>6s} {'+ Liq':>6s} {'Δ':>6s}", end='')
    print()
    print(f"  {'-'*10}", end='')
    for h in [6, 12, 24, 48]:
        print(f"  {'-'*6} {'-'*6} {'-'*6}", end='')
    print()

    for symbol, res in all_results.items():
        if 'prediction' not in res:
            continue
        print(f"  {symbol:>10s}", end='')
        for h in [6, 12, 24, 48]:
            if h in res['prediction']:
                hr = res['prediction'][h]
                base_auc = hr.get('Base only_Gradient Boost', {}).get('auc', 0)
                liq_auc = hr.get('Base + Liq_Gradient Boost', {}).get('auc', 0)
                delta = liq_auc - base_auc
                print(f"  {base_auc:>6.3f} {liq_auc:>6.3f} {delta:>+6.3f}", end='')
            else:
                print(f"  {'':>6s} {'':>6s} {'':>6s}", end='')
        print()

    # Detection speed summary
    print(f"\n  --- Detection Speed Improvement ---")
    print(f"  {'Symbol':>10s}  {'Base Acc':>8s}  {'+Liq Acc':>8s}  {'Base Lag':>8s}  {'+Liq Lag':>8s}  {'Base FalsSw':>11s}  {'+Liq FalsSw':>11s}")
    print(f"  {'-'*75}")
    for symbol, res in all_results.items():
        if 'detection' not in res or not res['detection']:
            continue
        d = res['detection']
        b = d.get('Base only', {})
        bl = d.get('Base + Liq', {})
        if b and bl:
            print(f"  {symbol:>10s}  {b['acc']:>8.3f}  {bl['acc']:>8.3f}  "
                  f"{b['mean_lag']:>7.1f}b  {bl['mean_lag']:>7.1f}b  "
                  f"{b['false_per_day']:>10.1f}  {bl['false_per_day']:>10.1f}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
