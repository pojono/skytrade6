#!/usr/bin/env python3
"""Analyze v3 results to find optimal thresholds."""
import pandas as pd
import numpy as np

df = pd.read_csv("flow_shock_research/results/flow_pressure_v3.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

print("="*80)
print("FLOW PRESSURE v3 - THRESHOLD ANALYSIS")
print("="*80)
print(f"\nTotal events: {len(df)}")
print(f"Days: {df['date'].nunique()}")
print(f"Current: {len(df)/df['date'].nunique():.1f} events/day\n")

# Test different FlowImpact thresholds
print("="*80)
print("FLOW IMPACT THRESHOLD SCAN")
print("="*80)
print(f"\n{'Threshold':>10} | {'Events':>8} | {'Events/Day':>11} | {'Target?':>8} | {'Mean Imb':>9} | {'Mean Run':>9}")
print("-"*90)

thresholds = [0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]

for threshold in thresholds:
    filtered = df[df['flow_impact'] >= threshold]
    
    if len(filtered) == 0:
        continue
    
    epd = filtered.groupby('date').size()
    avg = epd.mean()
    in_target = "✅" if 10 <= avg <= 50 else "❌"
    
    mean_imb = filtered['imbalance'].mean()
    mean_run = filtered['max_run'].mean()
    
    print(f"{threshold:>10.1f} | {len(filtered):>8,} | {avg:>11.1f} | {in_target:>8} | {mean_imb:>8.1%} | {mean_run:>9.0f}")

# Test combined filters
print("\n" + "="*80)
print("COMBINED FILTER ANALYSIS")
print("="*80)

configs = [
    ("impact>1.0 + imb>0.7", (df['flow_impact'] >= 1.0) & (df['imbalance'] >= 0.7)),
    ("impact>2.0 + imb>0.7", (df['flow_impact'] >= 2.0) & (df['imbalance'] >= 0.7)),
    ("impact>3.0 + imb>0.7", (df['flow_impact'] >= 3.0) & (df['imbalance'] >= 0.7)),
    ("impact>5.0 + imb>0.7", (df['flow_impact'] >= 5.0) & (df['imbalance'] >= 0.7)),
    ("impact>2.0 + imb>0.8", (df['flow_impact'] >= 2.0) & (df['imbalance'] >= 0.8)),
    ("impact>3.0 + imb>0.8", (df['flow_impact'] >= 3.0) & (df['imbalance'] >= 0.8)),
    ("impact>5.0 + imb>0.8", (df['flow_impact'] >= 5.0) & (df['imbalance'] >= 0.8)),
    ("impact>2.0 + imb>0.8 + run>100", (df['flow_impact'] >= 2.0) & (df['imbalance'] >= 0.8) & (df['max_run'] >= 100)),
    ("impact>3.0 + imb>0.8 + run>100", (df['flow_impact'] >= 3.0) & (df['imbalance'] >= 0.8) & (df['max_run'] >= 100)),
    ("impact>5.0 + imb>0.8 + run>150", (df['flow_impact'] >= 5.0) & (df['imbalance'] >= 0.8) & (df['max_run'] >= 150)),
]

print(f"\n{'Config':>35} | {'Events':>8} | {'Events/Day':>11} | {'Target?':>8}")
print("-"*80)

for name, mask in configs:
    filtered = df[mask]
    
    if len(filtered) == 0:
        continue
    
    epd = filtered.groupby('date').size()
    avg = epd.mean()
    in_target = "✅" if 10 <= avg <= 50 else "❌"
    
    print(f"{name:>35} | {len(filtered):>8,} | {avg:>11.1f} | {in_target:>8}")

# Find optimal
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

for name, mask in configs:
    filtered = df[mask]
    if len(filtered) == 0:
        continue
    
    epd = filtered.groupby('date').size()
    avg = epd.mean()
    
    if 10 <= avg <= 50:
        print(f"\n✅ OPTIMAL: {name}")
        print(f"   Events/day: {avg:.1f}")
        print(f"   Total (10d): {len(filtered)}")
        print(f"   Est. (92d): ~{len(filtered) * 92 / 10:.0f}")
        print(f"   Mean impact: {filtered['flow_impact'].mean():.1f}")
        print(f"   Mean imbalance: {filtered['imbalance'].mean():.1%}")
        print(f"   Mean run: {filtered['max_run'].mean():.0f}")
        
        # Liquidity stress
        stressed = filtered[filtered['liquidity_stressed'] == True]
        print(f"   Liquidity stressed: {len(stressed)}/{len(filtered)} ({len(stressed)/len(filtered)*100:.0f}%)")
        
        print(f"\n   Daily breakdown:")
        for date, count in epd.items():
            print(f"      {date}: {count} events")
        
        break

print("\n" + "="*80)
