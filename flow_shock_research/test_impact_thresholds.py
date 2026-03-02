#!/usr/bin/env python3
"""
Quick threshold scan on existing results to find optimal impact threshold.
"""
import pandas as pd
from pathlib import Path

results_file = Path("flow_shock_research/results/flow_impact_minimal.csv")

if not results_file.exists():
    print("❌ Results file not found. Run research_flow_impact_minimal.py first.")
    exit(1)

print("="*80)
print("🔍 FLOW IMPACT THRESHOLD SCAN")
print("="*80)

df = pd.read_csv(results_file)
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

print(f"\nTotal events in dataset: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Days: {df['date'].nunique()}\n")

thresholds = [0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]

print("="*80)
print("THRESHOLD SCAN RESULTS")
print("="*80)
print(f"\n{'Threshold':>10} | {'Events':>8} | {'Events/Day':>11} | {'Target?':>8} | {'Mean Impact':>12}")
print("-"*80)

for threshold in thresholds:
    filtered = df[df['flow_impact'] >= threshold]
    
    if len(filtered) == 0:
        print(f"{threshold:>10.1f} | {'0':>8} | {'0.0':>11} | {'❌':>8} | {'N/A':>12}")
        continue
    
    events_per_day = filtered.groupby('date').size()
    avg_per_day = events_per_day.mean()
    median_per_day = events_per_day.median()
    
    # Check if in target range (1-5 events/day)
    in_target = "✅" if 1 <= avg_per_day <= 5 else "❌"
    
    mean_impact = filtered['flow_impact'].mean()
    
    print(f"{threshold:>10.1f} | {len(filtered):>8,} | {avg_per_day:>11.1f} | {in_target:>8} | {mean_impact:>12.1f}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Find optimal threshold
for threshold in thresholds:
    filtered = df[df['flow_impact'] >= threshold]
    if len(filtered) == 0:
        continue
    
    events_per_day = filtered.groupby('date').size()
    avg_per_day = events_per_day.mean()
    
    if 1 <= avg_per_day <= 5:
        print(f"\n✅ OPTIMAL: impact > {threshold}")
        print(f"   Events/day: {avg_per_day:.1f}")
        print(f"   Total events (10 days): {len(filtered)}")
        print(f"   Estimated (92 days): ~{len(filtered) * 92 / 10:.0f} events")
        print(f"   Mean impact: {filtered['flow_impact'].mean():.1f}")
        
        # Show distribution
        print(f"\n   Events per day distribution:")
        for date, count in events_per_day.items():
            print(f"      {date}: {count} events")
        
        break
else:
    print("\n⚠️  No threshold in 1-5 events/day range")
    print("   Need to test higher thresholds or adjust detection criteria")

print("\n" + "="*80)
