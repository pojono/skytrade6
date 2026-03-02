#!/usr/bin/env python3
"""
Step 3: Filter FOLLOW by VacuumScore quantiles.

Goal: Find VacuumScore threshold that increases gross edge from ~14 bps to 30-60 bps.
Test different quantile thresholds and measure returns.
"""
import pandas as pd
import numpy as np
import sys

# Load dataset with vacuum scores (already has returns)
df = pd.read_parquet("results/production_dataset_vacuum.parquet")

# Add FOLLOW classification using global thresholds
CLASSIFIER_PARAMS = {
    'vol_15m_q30': 0.51,
    'range_10s_q80': 61.34,
    'drift_2m_q70': -41.87
}

def classify_follow(row):
    """Classify as FOLLOW using global thresholds."""
    vol_15m = row.get('pre_15m_vol', np.nan)
    range_10s = row.get('pre_10s_range', np.nan)
    drift_2m = row.get('pre_2m_drift', np.nan)
    
    if pd.isna(vol_15m) or pd.isna(range_10s) or pd.isna(drift_2m):
        return False
    
    vol_15m_low = vol_15m < CLASSIFIER_PARAMS['vol_15m_q30']
    range_10s_high = range_10s > CLASSIFIER_PARAMS['range_10s_q80']
    drift_2m_strong = abs(drift_2m) > CLASSIFIER_PARAMS['drift_2m_q70']
    
    return vol_15m_low and range_10s_high and drift_2m_strong

df['is_follow'] = df.apply(classify_follow, axis=1)

print("="*80)
print("📊 STEP 3: FILTER FOLLOW BY VACUUM SCORE")
print("="*80)
print(f"\nTotal events: {len(df)}")
print(f"FOLLOW events (baseline): {df['is_follow'].sum()}")
print(f"Events with vacuum_score: {df['vacuum_score'].notna().sum()}")
print("="*80 + "\n")

# Analyze FOLLOW events only
follow_df = df[df['is_follow'] & df['vacuum_score'].notna()].copy()

if len(follow_df) == 0:
    print("❌ No FOLLOW events with vacuum scores!")
    sys.exit(1)

print(f"FOLLOW events with vacuum scores: {len(follow_df)}")

# Calculate directional returns (WITH flow direction)
follow_df['ret_30s_directional'] = follow_df.apply(
    lambda row: row['ret_30s'] if row['direction'] == 'Buy' else -row['ret_30s'],
    axis=1
)

# Baseline FOLLOW performance
baseline_gross = follow_df['ret_30s_directional'].mean()
baseline_wr = (follow_df['ret_30s_directional'] > 0).mean()

print(f"\n{'='*80}")
print(f"BASELINE FOLLOW PERFORMANCE")
print(f"{'='*80}\n")
print(f"Events: {len(follow_df)}")
print(f"Gross return: {baseline_gross:+.2f} bps")
print(f"Win rate: {baseline_wr:.1%}")
print(f"Net (maker 8bps): {baseline_gross - 8:+.2f} bps")

# Test different VacuumScore quantile thresholds
print(f"\n{'='*80}")
print(f"VACUUM SCORE QUANTILE ANALYSIS")
print(f"{'='*80}\n")

quantiles = [0, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

results = []

for q in quantiles:
    threshold = follow_df['vacuum_score'].quantile(q)
    
    # Filter by vacuum score
    filtered = follow_df[follow_df['vacuum_score'] >= threshold]
    
    if len(filtered) == 0:
        continue
    
    gross = filtered['ret_30s_directional'].mean()
    wr = (filtered['ret_30s_directional'] > 0).mean()
    net_maker = gross - 8
    net_mixed = gross - 14
    net_taker = gross - 20
    
    results.append({
        'quantile': q,
        'threshold': threshold,
        'count': len(filtered),
        'gross': gross,
        'win_rate': wr,
        'net_maker': net_maker,
        'net_mixed': net_mixed,
        'net_taker': net_taker
    })
    
    status = "✅" if net_maker > 10 else "⚠️" if net_maker > 0 else "❌"
    
    print(f"Q{int(q*100):02d} (vacuum >= {threshold:.3f}): {len(filtered):3d} events, "
          f"gross {gross:+7.2f} bps, net(maker) {net_maker:+7.2f} bps {status}")

results_df = pd.DataFrame(results)

# Find best threshold
best_idx = results_df['net_maker'].idxmax()
best = results_df.iloc[best_idx]

print(f"\n{'='*80}")
print(f"BEST VACUUM THRESHOLD")
print(f"{'='*80}\n")
print(f"Quantile: Q{int(best['quantile']*100)}")
print(f"Threshold: vacuum_score >= {best['threshold']:.3f}")
print(f"Events: {int(best['count'])}")
print(f"Gross: {best['gross']:+.2f} bps")
print(f"Win rate: {best['win_rate']:.1%}")
print(f"Net (maker 8bps): {best['net_maker']:+.2f} bps")
print(f"Net (mixed 14bps): {best['net_mixed']:+.2f} bps")
print(f"Net (taker 20bps): {best['net_taker']:+.2f} bps")

# Improvement vs baseline
improvement = best['gross'] - baseline_gross
print(f"\nImprovement vs baseline: {improvement:+.2f} bps ({improvement/baseline_gross*100:+.1f}%)")

# Check if we hit target (30-60 bps gross)
if best['gross'] >= 30:
    print(f"\n✅ TARGET HIT: Gross edge {best['gross']:.2f} bps >= 30 bps target!")
else:
    print(f"\n⚠️ Target not hit: Gross edge {best['gross']:.2f} bps < 30 bps target")
    print(f"   May need to combine with other filters or accept lower frequency")

# Save filtered dataset
df['follow_vacuum_filtered'] = (df['is_follow']) & (df['vacuum_score'] >= best['threshold'])

df.to_parquet("results/production_dataset_filtered.parquet", index=False)
df.to_csv("results/production_dataset_filtered.csv", index=False)

print(f"\n💾 Saved: results/production_dataset_filtered.parquet")
print(f"   Column 'follow_vacuum_filtered' = FOLLOW + Vacuum filter")

# Show distribution of filtered events
print(f"\n{'='*80}")
print(f"FILTERED EVENT DISTRIBUTION")
print(f"{'='*80}\n")

filtered_events = df[df['follow_vacuum_filtered']]
print(f"Total filtered events: {len(filtered_events)}")
print(f"Events per day: {len(filtered_events) / 8:.1f}")  # 8 days in sample

if len(filtered_events) > 0:
    print(f"\nBy hour:")
    for hour, count in filtered_events['hour'].value_counts().sort_index().items():
        print(f"   Hour {hour:02d}: {count:3d} events")

print(f"\n{'='*80}")
print(f"✅ STEP 3 COMPLETE - Vacuum filter optimized")
print(f"{'='*80}")
