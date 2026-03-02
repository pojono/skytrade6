#!/usr/bin/env python3
"""
Hour-by-hour analysis with proper filters.
No session classification - use specific UTC hours.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
exhaustion_df = pd.read_csv("flow_shock_research/results/exhaustion_confirmation.csv")
exhaustion_df['datetime'] = pd.to_datetime(exhaustion_df['datetime'])
exhaustion_df['hour'] = exhaustion_df['datetime'].dt.hour

print("="*80)
print("HOUR-SPECIFIC FILTER ANALYSIS")
print("="*80)

# Filter combinations by hour
print("\n📊 Quality Events by Hour + Decay Filter")
print("="*80)

print(f"\n{'Hour':>6} | {'Total':>6} | {'Weak Decay':>11} | {'Mean Ret':>10} | {'Quality?':>8}")
print("-"*70)

hour_analysis = []

for hour in range(24):
    hour_events = exhaustion_df[exhaustion_df['hour'] == hour]
    
    if len(hour_events) == 0:
        continue
    
    # Apply weak decay filter
    weak_decay = hour_events[hour_events['vol_decay_ratio'] >= 0.5]
    
    if len(weak_decay) > 0 and 'ret_30s' in weak_decay.columns:
        mean_ret = weak_decay['ret_30s'].mean()
        quality = "✅" if len(weak_decay) >= 10 and abs(mean_ret) > 50 else "❌"
    else:
        mean_ret = 0
        quality = "❌"
    
    print(f"{hour:>6} | {len(hour_events):>6} | {len(weak_decay):>11} | {mean_ret:>9.1f} | {quality:>8}")
    
    hour_analysis.append({
        'hour': hour,
        'total_events': len(hour_events),
        'weak_decay_events': len(weak_decay),
        'mean_return': mean_ret
    })

# Focus on top hours
print("\n" + "="*80)
print("TOP HOURS DETAILED ANALYSIS")
print("="*80)

top_hours = [7, 18]  # From previous analysis

for hour in top_hours:
    print(f"\n{'='*80}")
    print(f"HOUR {hour:02d}:00 UTC")
    print(f"{'='*80}")
    
    hour_events = exhaustion_df[exhaustion_df['hour'] == hour].copy()
    
    print(f"\nTotal events: {len(hour_events)}")
    
    if len(hour_events) == 0:
        continue
    
    # Decay analysis
    print(f"\n📊 Decay Distribution:")
    print(f"   vol_decay_ratio < 0.5 (strong): {(hour_events['vol_decay_ratio'] < 0.5).sum()}")
    print(f"   vol_decay_ratio >= 0.5 (weak):  {(hour_events['vol_decay_ratio'] >= 0.5).sum()}")
    
    # Returns by decay
    if 'ret_30s' in hour_events.columns:
        strong_decay = hour_events[hour_events['vol_decay_ratio'] < 0.5]
        weak_decay = hour_events[hour_events['vol_decay_ratio'] >= 0.5]
        
        print(f"\n📈 Returns by Decay:")
        print(f"   Strong decay: {strong_decay['ret_30s'].mean():>7.2f} bps (n={len(strong_decay)})")
        print(f"   Weak decay:   {weak_decay['ret_30s'].mean():>7.2f} bps (n={len(weak_decay)})")
        
        # Classification
        if 'classification' in hour_events.columns:
            print(f"\n📋 Classification (weak decay only):")
            if len(weak_decay) > 0:
                class_counts = weak_decay['classification'].value_counts()
                for cls, count in class_counts.items():
                    pct = count / len(weak_decay) * 100
                    print(f"      {cls:15s}: {count:4d} ({pct:5.1f}%)")
    
    # FlowImpact distribution
    print(f"\n💥 FlowImpact (weak decay only):")
    weak_decay = hour_events[hour_events['vol_decay_ratio'] >= 0.5]
    if len(weak_decay) > 0:
        print(f"   Mean: {weak_decay['flow_impact'].mean():.1f}")
        print(f"   Median: {weak_decay['flow_impact'].median():.1f}")
        print(f"   Range: {weak_decay['flow_impact'].min():.1f} - {weak_decay['flow_impact'].max():.1f}")

# Final recommendation
print("\n" + "="*80)
print("RECOMMENDED HOUR-SPECIFIC FILTERS")
print("="*80)

print("\n✅ PRIMARY FILTER: Hour 7 UTC (Tokyo/London transition)")
h7_weak = exhaustion_df[(exhaustion_df['hour'] == 7) & (exhaustion_df['vol_decay_ratio'] >= 0.5)]
if len(h7_weak) > 0 and 'ret_30s' in h7_weak.columns:
    print(f"   Events: {len(h7_weak)}")
    print(f"   Mean return: {h7_weak['ret_30s'].mean():.1f} bps")
    print(f"   Events per day (est): {len(h7_weak) * 35 / len(exhaustion_df):.0f}")
    
    reversal_rate = (h7_weak['classification'] == 'Reversal').sum() / len(h7_weak) * 100 if 'classification' in h7_weak.columns else 0
    print(f"   Reversal rate: {reversal_rate:.1f}%")

print("\n⚠️  SECONDARY FILTER: Hour 18 UTC (London/NY transition)")
h18_weak = exhaustion_df[(exhaustion_df['hour'] == 18) & (exhaustion_df['vol_decay_ratio'] >= 0.5)]
if len(h18_weak) > 0 and 'ret_30s' in h18_weak.columns:
    print(f"   Events: {len(h18_weak)}")
    print(f"   Mean return: {h18_weak['ret_30s'].mean():.1f} bps")
    print(f"   Events per day (est): {len(h18_weak) * 35 / len(exhaustion_df):.0f}")
    
    reversal_rate = (h18_weak['classification'] == 'Reversal').sum() / len(h18_weak) * 100 if 'classification' in h18_weak.columns else 0
    print(f"   Reversal rate: {reversal_rate:.1f}%")

# Combined strategy
print("\n" + "="*80)
print("COMBINED HOUR-SPECIFIC STRATEGY")
print("="*80)

combined = exhaustion_df[
    ((exhaustion_df['hour'] == 7) | (exhaustion_df['hour'] == 18)) &
    (exhaustion_df['vol_decay_ratio'] >= 0.5)
]

print(f"\nTotal quality events: {len(combined)}")
print(f"Events per day (est): {len(combined) * 35 / len(exhaustion_df):.0f}")

if 'ret_30s' in combined.columns:
    print(f"Mean return: {combined['ret_30s'].mean():.1f} bps")
    print(f"Median return: {combined['ret_30s'].median():.1f} bps")
    
    # By hour breakdown
    print(f"\nBreakdown:")
    for hour in [7, 18]:
        subset = combined[combined['hour'] == hour]
        if len(subset) > 0:
            print(f"   Hour {hour:02d}: {len(subset):3d} events, {subset['ret_30s'].mean():>7.2f} bps mean")

# Save hour-specific results
output = combined.copy()
output_file = Path("flow_shock_research/results/hour_specific_quality_events.csv")
output.to_csv(output_file, index=False)
print(f"\n💾 Saved: {output_file}")

print("\n" + "="*80)
