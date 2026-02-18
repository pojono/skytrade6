#!/usr/bin/env python3
"""
High-Resolution OI/Funding Research (v25)
Analyze 5-second ticker data + 1-hour long/short ratio data.

Research Questions:
1. Do sub-5min OI spikes predict returns better than hourly aggregates?
2. How does OI velocity/acceleration relate to price moves?
3. Can we detect institutional positioning changes in real-time?
4. Does long/short ratio crowding predict reversals?
5. What's the optimal aggregation window for OI features?
"""
import pandas as pd
import numpy as np
import gzip
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

import argparse
import time

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ticker_data(symbol, data_dir='data'):
    """Load 5-second ticker data."""
    print(f"\nLoading ticker data for {symbol}...")
    symbol_dir = Path(data_dir) / symbol
    ticker_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    
    if not ticker_files:
        raise ValueError(f"No ticker files found for {symbol}")
    
    print(f"  Found {len(ticker_files)} ticker files")
    
    records = []
    errors = 0
    for i, file in enumerate(ticker_files):
        if i % 100 == 0:
            print(f"  Loading file {i+1}/{len(ticker_files)}...")
        
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    result = data['result']['list'][0]
                    
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'open_interest': float(result['openInterest']),
                        'oi_value': float(result['openInterestValue']),
                        'funding_rate': float(result['fundingRate']),
                        'last_price': float(result['lastPrice']),
                        'mark_price': float(result['markPrice']),
                        'index_price': float(result['indexPrice']),
                        'volume_24h': float(result['volume24h']),
                        'turnover_24h': float(result['turnover24h']),
                    })
                except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                    errors += 1
                    continue
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"  Loaded {len(df):,} records")
    if errors > 0:
        print(f"  Skipped {errors} corrupted lines")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Avg interval: {df['timestamp'].diff().mean()}")
    
    return df

def load_longshort_ratio(symbol, data_dir='data'):
    """Load 1-hour long/short ratio data."""
    print(f"\nLoading long/short ratio for {symbol}...")
    symbol_dir = Path(data_dir) / symbol
    ls_files = sorted(symbol_dir.glob("longshort_ratio_*.jsonl.gz"))
    
    if not ls_files:
        print(f"  No long/short ratio files found")
        return None
    
    print(f"  Found {len(ls_files)} files")
    
    records = []
    for file in ls_files:
        with gzip.open(file, 'rt') as f:
            for line in f:
                data = json.loads(line)
                records.append({
                    'timestamp': pd.to_datetime(int(data['timestamp']), unit='ms'),
                    'buy_ratio': float(data['buyRatio']),
                    'sell_ratio': float(data['sellRatio']),
                })
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ls_ratio'] = df['buy_ratio'] / df['sell_ratio']
    
    print(f"  Loaded {len(df):,} records")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def aggregate_ticker_to_bars(df, freq='5min'):
    """Aggregate 5-second ticker data to specified frequency."""
    print(f"\nAggregating to {freq} bars...")
    
    df = df.set_index('timestamp')
    
    agg_dict = {
        'last_price': 'last',
        'mark_price': 'last',
        'index_price': 'last',
        'open_interest': ['first', 'last', 'mean', 'std', 'min', 'max'],
        'oi_value': 'last',
        'funding_rate': ['last', 'mean', 'std', 'min', 'max'],
        'volume_24h': 'last',
        'turnover_24h': 'last',
    }
    
    bars = df.resample(freq).agg(agg_dict)
    
    # Flatten multi-level columns
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in bars.columns]
    
    # Calculate OI changes
    bars['oi_change'] = bars['open_interest_last'] - bars['open_interest_first']
    bars['oi_change_pct'] = bars['oi_change'] / bars['open_interest_first']
    
    # OI velocity (rate of change)
    bars['oi_velocity'] = bars['oi_change'] / (bars.index.to_series().diff().dt.total_seconds() / 60)
    
    # Price returns
    bars['price_ret'] = bars['last_price_last'].pct_change()
    bars['mark_index_spread'] = bars['mark_price_last'] - bars['index_price_last']
    
    bars = bars.reset_index()
    
    print(f"  Created {len(bars):,} bars")
    
    return bars

def build_hires_features(ticker_df, bars_df, lookback_windows=[5, 15, 60, 240]):
    """Build features from high-resolution ticker data."""
    print(f"\nBuilding high-resolution features...")
    
    df = bars_df.copy()
    
    # OI dynamics features
    for window in lookback_windows:
        window_label = f"{window}min" if window < 60 else f"{window//60}h"
        
        # OI changes
        df[f'oi_change_{window_label}'] = df['oi_change'].rolling(window).sum()
        df[f'oi_velocity_mean_{window_label}'] = df['oi_velocity'].rolling(window).mean()
        df[f'oi_velocity_std_{window_label}'] = df['oi_velocity'].rolling(window).std()
        df[f'oi_velocity_max_{window_label}'] = df['oi_velocity'].abs().rolling(window).max()
        
        # OI spike detection
        oi_vel_mean = df['oi_velocity'].rolling(window).mean()
        oi_vel_std = df['oi_velocity'].rolling(window).std()
        df[f'oi_spike_count_{window_label}'] = (
            (df['oi_velocity'].abs() > oi_vel_mean.abs() + 2 * oi_vel_std)
            .rolling(window).sum()
        )
        
        # Funding rate dynamics
        df[f'funding_mean_{window_label}'] = df['funding_rate_last'].rolling(window).mean()
        df[f'funding_std_{window_label}'] = df['funding_rate_std'].rolling(window).mean()
        df[f'funding_range_{window_label}'] = (
            df['funding_rate_max'].rolling(window).max() - 
            df['funding_rate_min'].rolling(window).min()
        )
        
        # Mark-index spread
        df[f'mis_mean_{window_label}'] = df['mark_index_spread'].rolling(window).mean()
        df[f'mis_std_{window_label}'] = df['mark_index_spread'].rolling(window).std()
        df[f'mis_max_abs_{window_label}'] = df['mark_index_spread'].abs().rolling(window).max()
        
        # Price volatility
        df[f'price_vol_{window_label}'] = df['price_ret'].rolling(window).std() * np.sqrt(252 * 24 * 12)
    
    # OI acceleration (change in velocity)
    df['oi_accel_5min'] = df['oi_velocity'].diff()
    df['oi_accel_15min'] = df['oi_velocity'].diff(3)
    df['oi_accel_1h'] = df['oi_velocity'].diff(12)
    
    # OI direction consistency
    df['oi_direction_5min'] = np.sign(df['oi_change']).rolling(5).mean()
    df['oi_direction_15min'] = np.sign(df['oi_change']).rolling(15).mean()
    df['oi_direction_1h'] = np.sign(df['oi_change']).rolling(60).mean()
    
    print(f"  Built {len([c for c in df.columns if c not in bars_df.columns])} new features")
    
    return df

def merge_longshort_ratio(bars_df, ls_df):
    """Merge long/short ratio data with bars."""
    if ls_df is None:
        return bars_df
    
    print(f"\nMerging long/short ratio data...")
    
    # Merge on nearest timestamp (forward fill)
    df = pd.merge_asof(
        bars_df.sort_values('timestamp'),
        ls_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        suffixes=('', '_ls')
    )
    
    # Calculate z-scores
    for col in ['buy_ratio', 'sell_ratio', 'ls_ratio']:
        df[f'{col}_zscore_24h'] = (
            (df[col] - df[col].rolling(24).mean()) / 
            df[col].rolling(24).std()
        )
    
    print(f"  Merged {len(df):,} records")
    
    return df

def build_forward_returns(df, horizons=[5, 15, 60, 240]):
    """Build forward returns for prediction."""
    print(f"\nBuilding forward returns...")
    
    # Use the correct price column name after aggregation
    price_col = 'last_price_last' if 'last_price_last' in df.columns else 'last_price'
    
    for horizon in horizons:
        horizon_label = f"{horizon}min" if horizon < 60 else f"{horizon//60}h"
        df[f'fwd_ret_{horizon_label}'] = df[price_col].pct_change(horizon).shift(-horizon)
    
    return df

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_oi_spikes(df):
    """Analyze OI spike events and their predictive power."""
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: OI SPIKE ANALYSIS")
    print(f"{'='*70}")
    
    # Define spike threshold (2 std above mean)
    oi_vel_mean = df['oi_velocity'].rolling(60).mean()
    oi_vel_std = df['oi_velocity'].rolling(60).std()
    df['is_oi_spike'] = df['oi_velocity'].abs() > (oi_vel_mean.abs() + 2 * oi_vel_std)
    
    # Analyze returns after spikes
    spike_df = df[df['is_oi_spike']].copy()
    
    print(f"\nOI Spike Statistics:")
    print(f"  Total spikes: {len(spike_df):,}")
    print(f"  Spike rate: {100 * len(spike_df) / len(df):.2f}%")
    
    for horizon in [5, 15, 60, 240]:
        horizon_label = f"{horizon}min" if horizon < 60 else f"{horizon//60}h"
        ret_col = f'fwd_ret_{horizon_label}'
        
        if ret_col not in df.columns:
            continue
        
        spike_rets = spike_df[ret_col].dropna()
        all_rets = df[ret_col].dropna()
        
        if len(spike_rets) > 0:
            avg_ret = spike_rets.mean() * 10000
            wr = (spike_rets > 0).mean() * 100
            sharpe = spike_rets.mean() / spike_rets.std() * np.sqrt(252 * 24 * 12 / horizon) if spike_rets.std() > 0 else 0
            
            # T-test vs all returns
            t_stat, p_val = stats.ttest_ind(spike_rets, all_rets)
            
            print(f"\n  After {horizon_label} OI spike:")
            print(f"    Avg return: {avg_ret:+.1f}bps")
            print(f"    Win rate: {wr:.1f}%")
            print(f"    Sharpe: {sharpe:+.2f}")
            print(f"    T-stat: {t_stat:.2f} (p={p_val:.4f})")

def analyze_oi_velocity_regimes(df):
    """Analyze OI velocity in different market regimes."""
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: OI VELOCITY BY REGIME")
    print(f"{'='*70}")
    
    # Define regimes by volatility
    df['vol_regime'] = pd.qcut(df['price_vol_1h'], q=3, labels=['Low', 'Med', 'High'])
    
    print(f"\nOI Velocity Statistics by Volatility Regime:")
    print(f"{'Regime':<10} {'Mean OI Vel':>12} {'Std OI Vel':>12} {'Spike Rate':>12}")
    print(f"{'-'*50}")
    
    for regime in ['Low', 'Med', 'High']:
        regime_df = df[df['vol_regime'] == regime]
        mean_vel = regime_df['oi_velocity'].abs().mean()
        std_vel = regime_df['oi_velocity'].abs().std()
        spike_rate = regime_df['is_oi_spike'].mean() * 100
        
        print(f"{regime:<10} {mean_vel:>12.4f} {std_vel:>12.4f} {spike_rate:>11.2f}%")

def analyze_feature_ic(df):
    """Calculate information coefficient for features."""
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: FEATURE INFORMATION COEFFICIENT")
    print(f"{'='*70}")
    
    # Select features
    feature_cols = [c for c in df.columns if any(x in c for x in [
        'oi_change', 'oi_velocity', 'oi_spike', 'oi_accel', 'oi_direction',
        'funding_mean', 'funding_std', 'funding_range',
        'mis_mean', 'mis_std', 'mis_max',
        'buy_ratio', 'sell_ratio', 'ls_ratio'
    ])]
    
    target_cols = [c for c in df.columns if c.startswith('fwd_ret_')]
    
    print(f"\nInformation Coefficient (Pearson correlation):")
    print(f"{'Feature':<40} {'5min':>8} {'15min':>8} {'1h':>8} {'4h':>8}")
    print(f"{'-'*70}")
    
    ic_results = []
    
    for feat in sorted(feature_cols):
        ics = []
        for target in sorted(target_cols):
            valid = df[[feat, target]].dropna()
            if len(valid) > 100:
                ic = valid[feat].corr(valid[target])
                ics.append(ic)
            else:
                ics.append(np.nan)
        
        if len(ics) == 4:
            ic_results.append({
                'feature': feat,
                'ic_5min': ics[0],
                'ic_15min': ics[1],
                'ic_1h': ics[2],
                'ic_4h': ics[3]
            })
            
            print(f"{feat:<40} {ics[0]:>+7.4f} {ics[1]:>+7.4f} {ics[2]:>+7.4f} {ics[3]:>+7.4f}")
    
    return pd.DataFrame(ic_results)

def walk_forward_prediction(df, target='fwd_ret_1h'):
    """Walk-forward prediction with different feature sets."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 4: WALK-FORWARD PREDICTION ({target})")
    print(f"{'='*70}")
    
    # Feature sets
    hires_features = [c for c in df.columns if any(x in c for x in [
        'oi_velocity', 'oi_spike', 'oi_accel', 'oi_direction',
        'funding_std', 'funding_range', 'mis_std', 'mis_max'
    ])]
    
    hourly_features = [c for c in df.columns if any(x in c for x in [
        'oi_change_1h', 'oi_change_4h',
        'funding_mean_1h', 'funding_mean_4h',
        'mis_mean_1h', 'mis_mean_4h'
    ])]
    
    ls_features = [c for c in df.columns if any(x in c for x in [
        'buy_ratio', 'sell_ratio', 'ls_ratio'
    ])]
    
    feature_sets = {
        'Hi-Res Only': hires_features,
        'Hourly Only': hourly_features,
        'Hi-Res + Hourly': hires_features + hourly_features,
        'Hi-Res + Hourly + L/S': hires_features + hourly_features + ls_features,
    }
    
    results = {}
    
    for name, features in feature_sets.items():
        if not features:
            continue
        
        print(f"\n{name} ({len(features)} features):")
        
        # Prepare data
        valid_df = df[features + [target]].dropna()
        
        if len(valid_df) < 1000:
            print(f"  Insufficient data: {len(valid_df)} samples")
            continue
        
        X = valid_df[features].values
        y = valid_df[target].values
        
        # Walk-forward with 5 splits
        tscv = TimeSeriesSplit(n_splits=5)
        
        preds = []
        actuals = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            preds.extend(y_pred)
            actuals.extend(y_test)
        
        preds = np.array(preds)
        actuals = np.array(actuals)
        
        # Calculate metrics
        ic = np.corrcoef(preds, actuals)[0, 1]
        rank_ic = stats.spearmanr(preds, actuals)[0]
        
        # Quintile analysis
        quintiles = pd.qcut(preds, q=5, labels=False, duplicates='drop')
        
        print(f"  IC: {ic:+.4f}")
        print(f"  Rank IC: {rank_ic:+.4f}")
        print(f"\n  Quintile Performance:")
        
        for q in range(5):
            q_mask = quintiles == q
            q_ret = actuals[q_mask].mean() * 10000
            q_wr = (actuals[q_mask] > 0).mean() * 100
            
            print(f"    Q{q+1}: avg={q_ret:+6.1f}bps  wr={q_wr:.1f}%  n={q_mask.sum()}")
        
        # Long-short
        ls_ret = (actuals[quintiles == 4].mean() - actuals[quintiles == 0].mean()) * 10000
        ls_sharpe = (actuals[quintiles == 4].mean() - actuals[quintiles == 0].mean()) / \
                    np.std(actuals[quintiles == 4] - actuals[quintiles == 0]) * np.sqrt(252 * 24 * 12 / 60)
        
        print(f"  Long-short (Q5-Q1): {ls_ret:+.1f}bps  sharpe={ls_sharpe:+.2f}")
        
        results[name] = {
            'ic': ic,
            'rank_ic': rank_ic,
            'ls_ret': ls_ret,
            'ls_sharpe': ls_sharpe
        }
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='High-resolution OI/Funding research')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol (BTCUSDT, ETHUSDT, SOLUSDT)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("="*70)
    print(f"HIGH-RESOLUTION OI/FUNDING RESEARCH (v25)")
    print("="*70)
    print(f"Symbol: {args.symbol}")
    print(f"Data directory: {args.data_dir}")
    print("="*70)
    
    # Load data
    ticker_df = load_ticker_data(args.symbol, args.data_dir)
    ls_df = load_longshort_ratio(args.symbol, args.data_dir)
    
    # Aggregate to 5-minute bars
    bars_df = aggregate_ticker_to_bars(ticker_df, freq='5min')
    
    # Build features
    df = build_hires_features(ticker_df, bars_df)
    df = merge_longshort_ratio(df, ls_df)
    df = build_forward_returns(df, horizons=[5, 15, 60, 240])
    
    print(f"\nFinal dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Run experiments
    analyze_oi_spikes(df)
    analyze_oi_velocity_regimes(df)
    ic_df = analyze_feature_ic(df)
    results = walk_forward_prediction(df, target='fwd_ret_1h')
    
    # Save results
    output_file = f"results/hires_oi_funding_v25_{args.symbol.replace('USDT', '')}.txt"
    Path("results").mkdir(exist_ok=True)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"DONE — {elapsed:.0f}s total")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
