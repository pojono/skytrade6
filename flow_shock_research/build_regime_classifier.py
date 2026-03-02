#!/usr/bin/env python3
"""
Rule-based regime classifier using top multi-scale features.

Binary classification:
- FOLLOW: Trade WITH flow (continuation/initiation regime)
- FADE: Trade AGAINST flow (exhaustion/reversal regime)
- NO TRADE: Mixed/unclear regime

Uses quantile-based thresholds (adaptive to market conditions).

Fee structure:
- Taker: 10 bps per leg
- Maker: 4 bps per leg
- Round-trip taker: 20 bps
- Round-trip maker: 8 bps
- Mixed (taker entry, maker exit): 14 bps
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_quantile_thresholds(df, features):
    """Calculate quantile thresholds for each feature."""
    thresholds = {}
    
    for feature in features:
        if feature in df.columns:
            thresholds[f'{feature}_q30'] = df[feature].quantile(0.30)
            thresholds[f'{feature}_q70'] = df[feature].quantile(0.70)
            thresholds[f'{feature}_q80'] = df[feature].quantile(0.80)
            thresholds[f'{feature}_q85'] = df[feature].quantile(0.85)
            thresholds[f'{feature}_q95'] = df[feature].quantile(0.95)
    
    return thresholds

def classify_regime(row, thresholds, version='v1'):
    """
    Classify regime using rule-based logic.
    
    FOLLOW (initiation/breakout):
    - Low long-term vol (stable base)
    - High short-term range (expansion)
    - Strong drift (momentum)
    
    FADE (exhaustion):
    - High long-term vol (elevated base)
    - Extreme short-term drift (overstretched)
    - High imbalance (panic)
    
    Returns: 'FOLLOW', 'FADE', or 'NO_TRADE'
    """
    # Extract features
    vol_15m = row.get('15m_vol', np.nan)
    range_10s = row.get('10s_range', np.nan)
    drift_2m = row.get('2m_drift', np.nan)
    imbalance_30s = row.get('30s_imbalance', np.nan)
    vol_30s = row.get('30s_vol', np.nan)
    
    # Check for missing data
    if pd.isna(vol_15m) or pd.isna(range_10s) or pd.isna(drift_2m):
        return 'NO_TRADE'
    
    # FOLLOW conditions (initiation/breakout)
    vol_15m_low = vol_15m < thresholds.get('15m_vol_q30', 0)
    range_10s_high = range_10s > thresholds.get('10s_range_q80', 0)
    drift_2m_strong = abs(drift_2m) > thresholds.get('2m_drift_q70', 0)
    
    if vol_15m_low and range_10s_high and drift_2m_strong:
        return 'FOLLOW'
    
    # FADE conditions - version dependent
    if version == 'v1':
        # Original (loose)
        vol_15m_high = vol_15m > thresholds.get('15m_vol_q70', 0)
        drift_2m_extreme = abs(drift_2m) > thresholds.get('2m_drift_q85', 0)
        imbalance_high = abs(imbalance_30s) > 0.5 if not pd.isna(imbalance_30s) else False
        
        if vol_15m_high and drift_2m_extreme and imbalance_high:
            return 'FADE'
    
    elif version == 'v2':
        # Stricter (better gross edge)
        vol_15m_high = vol_15m > thresholds.get('15m_vol_q85', 0)
        drift_2m_extreme = abs(drift_2m) > thresholds.get('2m_drift_q95', 0) if '2m_drift_q95' in thresholds else abs(drift_2m) > thresholds.get('2m_drift_q85', 0)
        imbalance_high = abs(imbalance_30s) > 0.7 if not pd.isna(imbalance_30s) else False
        vol_30s_high = vol_30s > thresholds.get('30s_vol_q80', 0) if not pd.isna(vol_30s) else False
        
        if vol_15m_high and drift_2m_extreme and imbalance_high and vol_30s_high:
            return 'FADE'
    
    # Default: unclear regime
    return 'NO_TRADE'

def calculate_fee_aware_ev(df, regime_class, fees_bps=20):
    """
    Calculate fee-aware expected value for a regime class.
    
    Args:
        df: DataFrame with events
        regime_class: 'FOLLOW' or 'FADE'
        fees_bps: Round-trip fees in bps (default 20 = taker both legs)
    
    Returns:
        dict with EV metrics
    """
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
    
    # Calculate directional returns
    if regime_class == 'FOLLOW':
        # Return WITH flow (positive if price continues in flow direction)
        subset['ret_directional'] = subset['ret_30s']
    else:  # FADE
        # Return AGAINST flow (positive if price reverses)
        subset['ret_directional'] = -subset['ret_30s']
    
    # Net return after fees
    subset['ret_net'] = subset['ret_directional'] - fees_bps
    
    # Metrics
    mean_ret_gross = subset['ret_directional'].mean()
    mean_ret_net = subset['ret_net'].mean()
    win_rate = (subset['ret_directional'] > 0).mean()
    win_rate_net = (subset['ret_net'] > 0).mean()
    
    return {
        'count': len(subset),
        'mean_ret_gross': mean_ret_gross,
        'mean_ret_net': mean_ret_net,
        'win_rate_gross': win_rate,
        'win_rate_net': win_rate_net,
        'ev_per_trade': mean_ret_net
    }

def analyze_classifier_performance(df, sample_name, version='v1'):
    """Analyze classifier performance on a sample."""
    print(f"\n{'='*80}", flush=True)
    print(f"📊 CLASSIFIER PERFORMANCE: {sample_name} ({version})", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Calculate quantile thresholds
    features = ['15m_vol', '10s_range', '2m_drift', '30s_imbalance', '30s_vol']
    thresholds = calculate_quantile_thresholds(df, features)
    
    print(f"Quantile thresholds (key ones):", flush=True)
    for key in ['15m_vol_q30', '15m_vol_q70', '15m_vol_q85', '10s_range_q80', '2m_drift_q85', '2m_drift_q95']:
        if key in thresholds:
            print(f"   {key}: {thresholds[key]:.2f}", flush=True)
    
    # Classify each event
    df['regime_pred'] = df.apply(lambda row: classify_regime(row, thresholds, version), axis=1)
    
    # Distribution
    print(f"\n📋 Regime Classification:", flush=True)
    for regime, count in df['regime_pred'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {regime:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
    
    # Fee scenarios
    fee_scenarios = {
        'Taker both legs': 20,  # 10 bps entry + 10 bps exit
        'Maker both legs': 8,   # 4 bps entry + 4 bps exit
        'Mixed (taker entry, maker exit)': 14  # 10 bps + 4 bps
    }
    
    for fee_name, fees_bps in fee_scenarios.items():
        print(f"\n{'='*80}", flush=True)
        print(f"FEE SCENARIO: {fee_name} ({fees_bps} bps)", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        # FOLLOW metrics
        follow_metrics = calculate_fee_aware_ev(df, 'FOLLOW', fees_bps)
        print(f"FOLLOW Strategy:", flush=True)
        print(f"   Events: {follow_metrics['count']}", flush=True)
        print(f"   Mean return (gross): {follow_metrics['mean_ret_gross']:+.2f} bps", flush=True)
        print(f"   Mean return (net): {follow_metrics['mean_ret_net']:+.2f} bps", flush=True)
        print(f"   Win rate (gross): {follow_metrics['win_rate_gross']:.1%}", flush=True)
        print(f"   Win rate (net): {follow_metrics['win_rate_net']:.1%}", flush=True)
        print(f"   EV per trade: {follow_metrics['ev_per_trade']:+.2f} bps", flush=True)
        
        # FADE metrics
        fade_metrics = calculate_fee_aware_ev(df, 'FADE', fees_bps)
        print(f"\nFADE Strategy:", flush=True)
        print(f"   Events: {fade_metrics['count']}", flush=True)
        print(f"   Mean return (gross): {fade_metrics['mean_ret_gross']:+.2f} bps", flush=True)
        print(f"   Mean return (net): {fade_metrics['mean_ret_net']:+.2f} bps", flush=True)
        print(f"   Win rate (gross): {fade_metrics['win_rate_gross']:.1%}", flush=True)
        print(f"   Win rate (net): {fade_metrics['win_rate_net']:.1%}", flush=True)
        print(f"   EV per trade: {fade_metrics['ev_per_trade']:+.2f} bps", flush=True)
        
        # Combined
        total_trades = follow_metrics['count'] + fade_metrics['count']
        if total_trades > 0:
            combined_ev = (follow_metrics['ev_per_trade'] * follow_metrics['count'] + 
                          fade_metrics['ev_per_trade'] * fade_metrics['count']) / total_trades
            print(f"\nCombined:", flush=True)
            print(f"   Total trades: {total_trades}", flush=True)
            print(f"   Blended EV: {combined_ev:+.2f} bps", flush=True)
    
    return df

def main():
    print("="*80, flush=True)
    print("🔬 RULE-BASED REGIME CLASSIFIER - FINAL VERSION", flush=True)
    print("="*80, flush=True)
    print("\nBinary classification: FOLLOW vs FADE vs NO_TRADE", flush=True)
    print("Fee-aware EV calculation", flush=True)
    print("Testing both v1 (original) and v2 (stricter FADE)", flush=True)
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
    
    print(f"Loaded {len(samples)} samples\n", flush=True)
    
    # Test both versions
    for version in ['v1', 'v2']:
        print(f"\n{'#'*80}", flush=True)
        print(f"# TESTING VERSION: {version}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        
        results = {}
        
        for sample_name, df in samples.items():
            df_copy = df.copy()
            df_copy = analyze_classifier_performance(df_copy, sample_name, version)
            results[sample_name] = df_copy
            
            # Save v2 results
            if version == 'v2':
                output_name = sample_name.split()[0].lower() + sample_name.split()[1].replace('(', '').replace(')', '')
                df_copy.to_csv(f"results/{output_name}_classified.csv", index=False)
                print(f"💾 Saved: results/{output_name}_classified.csv", flush=True)
        
        # Cross-sample stability check
        print(f"\n{'='*80}", flush=True)
        print(f"🎯 CROSS-SAMPLE EV STABILITY ({version})", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        print(f"Maker fees (8 bps) - Production scenario:\n", flush=True)
        
        all_follow = []
        all_fade = []
        
        for sample_name, df in results.items():
            follow_ev = calculate_fee_aware_ev(df, 'FOLLOW', 8)
            fade_ev = calculate_fee_aware_ev(df, 'FADE', 8)
            
            all_follow.append(follow_ev)
            all_fade.append(fade_ev)
            
            print(f"{sample_name:20s}:", flush=True)
            print(f"   FOLLOW: {follow_ev['count']:3d} trades, Gross: {follow_ev['mean_ret_gross']:+7.2f} bps, Net: {follow_ev['ev_per_trade']:+7.2f} bps, WR: {follow_ev['win_rate_net']:.1%}", flush=True)
            print(f"   FADE:   {fade_ev['count']:3d} trades, Gross: {fade_ev['mean_ret_gross']:+7.2f} bps, Net: {fade_ev['ev_per_trade']:+7.2f} bps, WR: {fade_ev['win_rate_net']:.1%}", flush=True)
        
        # Summary
        print(f"\n{'='*80}", flush=True)
        print(f"SUMMARY ({version}):", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        total_follow = sum(ev['count'] for ev in all_follow)
        total_fade = sum(ev['count'] for ev in all_fade)
        
        if total_follow > 0:
            avg_follow_gross = sum(ev['mean_ret_gross'] * ev['count'] for ev in all_follow) / total_follow
            avg_follow_net = sum(ev['ev_per_trade'] * ev['count'] for ev in all_follow) / total_follow
            follow_positive_samples = sum(1 for ev in all_follow if ev['ev_per_trade'] > 0)
            
            print(f"FOLLOW Strategy:", flush=True)
            print(f"   Total trades: {total_follow}", flush=True)
            print(f"   Avg gross: {avg_follow_gross:+.2f} bps", flush=True)
            print(f"   Avg net (maker): {avg_follow_net:+.2f} bps", flush=True)
            print(f"   Positive samples: {follow_positive_samples}/3", flush=True)
            print(f"   EV sign stable: {'✅ YES' if follow_positive_samples == 3 else '❌ NO'}", flush=True)
        
        if total_fade > 0:
            avg_fade_gross = sum(ev['mean_ret_gross'] * ev['count'] for ev in all_fade) / total_fade
            avg_fade_net = sum(ev['ev_per_trade'] * ev['count'] for ev in all_fade) / total_fade
            fade_positive_samples = sum(1 for ev in all_fade if ev['ev_per_trade'] > 0)
            
            print(f"\nFADE Strategy:", flush=True)
            print(f"   Total trades: {total_fade}", flush=True)
            print(f"   Avg gross: {avg_fade_gross:+.2f} bps", flush=True)
            print(f"   Avg net (maker): {avg_fade_net:+.2f} bps", flush=True)
            print(f"   Positive samples: {fade_positive_samples}/3", flush=True)
            print(f"   EV sign stable: {'✅ YES' if fade_positive_samples >= 2 else '❌ NO'}", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"✅ REGIME CLASSIFIER COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
