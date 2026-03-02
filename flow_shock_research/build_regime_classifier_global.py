#!/usr/bin/env python3
"""
Regime classifier with GLOBAL thresholds (fix stability issue).

Uses thresholds calculated from ALL samples combined, not per-sample quantiles.
This ensures consistent classification across different market regimes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_global_thresholds(samples, features):
    """Calculate global quantile thresholds from all samples combined."""
    all_data = pd.concat([df for df in samples.values()], ignore_index=True)
    
    thresholds = {}
    for feature in features:
        if feature in all_data.columns:
            thresholds[f'{feature}_q30'] = all_data[feature].quantile(0.30)
            thresholds[f'{feature}_q70'] = all_data[feature].quantile(0.70)
            thresholds[f'{feature}_q80'] = all_data[feature].quantile(0.80)
            thresholds[f'{feature}_q85'] = all_data[feature].quantile(0.85)
            thresholds[f'{feature}_q95'] = all_data[feature].quantile(0.95)
    
    return thresholds

def classify_regime(row, thresholds):
    """
    Classify regime using global thresholds.
    
    FOLLOW only (FADE abandoned due to insufficient gross edge).
    """
    vol_15m = row.get('15m_vol', np.nan)
    range_10s = row.get('10s_range', np.nan)
    drift_2m = row.get('2m_drift', np.nan)
    
    if pd.isna(vol_15m) or pd.isna(range_10s) or pd.isna(drift_2m):
        return 'NO_TRADE'
    
    # FOLLOW conditions
    vol_15m_low = vol_15m < thresholds.get('15m_vol_q30', 0)
    range_10s_high = range_10s > thresholds.get('10s_range_q80', 0)
    drift_2m_strong = abs(drift_2m) > thresholds.get('2m_drift_q70', 0)
    
    if vol_15m_low and range_10s_high and drift_2m_strong:
        return 'FOLLOW'
    
    return 'NO_TRADE'

def calculate_fee_aware_ev(df, regime_class, fees_bps=8):
    """Calculate fee-aware EV."""
    subset = df[df['regime_pred'] == regime_class].copy()
    
    if len(subset) == 0:
        return {
            'count': 0,
            'mean_ret_gross': 0,
            'mean_ret_net': 0,
            'win_rate_gross': 0,
            'win_rate_net': 0,
            'ev_per_trade': 0
        }
    
    # FOLLOW: return WITH flow
    subset['ret_directional'] = subset['ret_30s']
    subset['ret_net'] = subset['ret_directional'] - fees_bps
    
    return {
        'count': len(subset),
        'mean_ret_gross': subset['ret_directional'].mean(),
        'mean_ret_net': subset['ret_net'].mean(),
        'win_rate_gross': (subset['ret_directional'] > 0).mean(),
        'win_rate_net': (subset['ret_net'] > 0).mean(),
        'ev_per_trade': subset['ret_net'].mean()
    }

def main():
    print("="*80, flush=True)
    print("🔬 REGIME CLASSIFIER - GLOBAL THRESHOLDS", flush=True)
    print("="*80, flush=True)
    print("\nFOLLOW strategy only (FADE abandoned)", flush=True)
    print("Global thresholds from all samples combined", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Load all samples
    samples = {}
    for name, file in [
        ('Sample 1 (May)', "results/sample1_multiscale.csv"),
        ('Sample 2 (Jul-Aug)', "results/sample2_multiscale.csv"),
        ('Sample 3 (Jun)', "results/sample3_multiscale.csv")
    ]:
        if Path(file).exists():
            samples[name] = pd.read_csv(file)
    
    print(f"Loaded {len(samples)} samples", flush=True)
    for name, df in samples.items():
        print(f"   {name}: {len(df)} events", flush=True)
    
    # Calculate GLOBAL thresholds
    features = ['15m_vol', '10s_range', '2m_drift']
    global_thresholds = calculate_global_thresholds(samples, features)
    
    print(f"\n{'='*80}", flush=True)
    print(f"GLOBAL THRESHOLDS (from all samples)", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    for key, val in sorted(global_thresholds.items()):
        print(f"   {key}: {val:.2f}", flush=True)
    
    # Apply classifier to each sample
    print(f"\n{'='*80}", flush=True)
    print(f"CLASSIFICATION RESULTS", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    results = {}
    all_follow = []
    
    for sample_name, df in samples.items():
        df_copy = df.copy()
        df_copy['regime_pred'] = df_copy.apply(lambda row: classify_regime(row, global_thresholds), axis=1)
        
        # Distribution
        print(f"{sample_name}:", flush=True)
        for regime, count in df_copy['regime_pred'].value_counts().items():
            pct = count / len(df_copy) * 100
            print(f"   {regime:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
        
        # FOLLOW metrics (maker fees)
        follow_ev = calculate_fee_aware_ev(df_copy, 'FOLLOW', 8)
        all_follow.append(follow_ev)
        
        print(f"\n   FOLLOW Performance (maker 8bps):", flush=True)
        print(f"      Trades: {follow_ev['count']}", flush=True)
        print(f"      Gross: {follow_ev['mean_ret_gross']:+.2f} bps", flush=True)
        print(f"      Net: {follow_ev['ev_per_trade']:+.2f} bps", flush=True)
        print(f"      Win rate (net): {follow_ev['win_rate_net']:.1%}", flush=True)
        print(flush=True)
        
        results[sample_name] = df_copy
    
    # Summary
    print(f"{'='*80}", flush=True)
    print(f"SUMMARY - GLOBAL THRESHOLDS", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    total_trades = sum(ev['count'] for ev in all_follow)
    
    if total_trades > 0:
        avg_gross = sum(ev['mean_ret_gross'] * ev['count'] for ev in all_follow) / total_trades
        avg_net = sum(ev['ev_per_trade'] * ev['count'] for ev in all_follow) / total_trades
        positive_samples = sum(1 for ev in all_follow if ev['ev_per_trade'] > 0)
        
        print(f"FOLLOW Strategy (all samples):", flush=True)
        print(f"   Total trades: {total_trades}", flush=True)
        print(f"   Avg gross: {avg_gross:+.2f} bps", flush=True)
        print(f"   Avg net (maker 8bps): {avg_net:+.2f} bps", flush=True)
        print(f"   Positive samples: {positive_samples}/3", flush=True)
        print(f"   EV sign stable: {'✅ YES' if positive_samples == 3 else '❌ NO'}", flush=True)
        print(f"\n   Per-sample breakdown:", flush=True)
        for i, (sample_name, ev) in enumerate(zip(samples.keys(), all_follow)):
            status = "✅" if ev['ev_per_trade'] > 0 else "❌"
            print(f"      {sample_name:20s}: {ev['count']:3d} trades, {ev['ev_per_trade']:+7.2f} bps {status}", flush=True)
        
        # Frequency analysis
        total_days = 21  # 7 days per sample × 3 samples
        events_per_day = total_trades / total_days
        
        print(f"\n   Frequency:", flush=True)
        print(f"      Total days: {total_days}", flush=True)
        print(f"      Events/day: {events_per_day:.1f}", flush=True)
        print(f"      Daily EV: {events_per_day * avg_net:.2f} bps", flush=True)
        
        # Fee sensitivity
        print(f"\n   Fee sensitivity:", flush=True)
        for fee_name, fees in [('Maker (8bps)', 8), ('Mixed (14bps)', 14), ('Taker (20bps)', 20)]:
            net = avg_gross - fees
            print(f"      {fee_name}: {net:+.2f} bps net", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"✅ GLOBAL THRESHOLD CLASSIFIER COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)
    
    # Save results
    for sample_name, df in results.items():
        output_name = sample_name.split()[0].lower() + sample_name.split()[1].replace('(', '').replace(')', '')
        df.to_csv(f"results/{output_name}_global.csv", index=False)
        print(f"💾 Saved: results/{output_name}_global.csv", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
