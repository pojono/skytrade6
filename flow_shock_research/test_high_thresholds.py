#!/usr/bin/env python3
"""Test very high impact thresholds."""
import pandas as pd
from pathlib import Path

df = pd.read_csv("flow_shock_research/results/flow_impact_minimal.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

print("="*80)
print("HIGH THRESHOLD SCAN")
print("="*80)

thresholds = [20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]

print(f"\n{'Threshold':>10} | {'Events':>8} | {'Events/Day':>11} | {'Target?':>8}")
print("-"*70)

for threshold in thresholds:
    filtered = df[df['flow_impact'] >= threshold]
    
    if len(filtered) == 0:
        continue
    
    epd = filtered.groupby('date').size()
    avg = epd.mean()
    in_target = "✅" if 1 <= avg <= 5 else "❌"
    
    print(f"{threshold:>10} | {len(filtered):>8,} | {avg:>11.1f} | {in_target:>8}")

print("\n" + "="*80)

# Find optimal
for threshold in thresholds:
    filtered = df[df['flow_impact'] >= threshold]
    if len(filtered) == 0:
        continue
    
    epd = filtered.groupby('date').size()
    avg = epd.mean()
    
    if 1 <= avg <= 5:
        print(f"\n✅ OPTIMAL: impact > {threshold}")
        print(f"   Events/day: {avg:.1f}")
        print(f"   Total (10d): {len(filtered)}")
        print(f"   Est. (92d): ~{len(filtered) * 92 / 10:.0f}")
        print(f"   Mean impact: {filtered['flow_impact'].mean():.0f}")
        print(f"   Max impact: {filtered['flow_impact'].max():.0f}")
        
        print(f"\n   Daily breakdown:")
        for date, count in epd.items():
            print(f"      {date}: {count} events")
        
        break

print("\n" + "="*80)
