#!/usr/bin/env python3
"""Test extreme filter combinations."""
import pandas as pd

df = pd.read_csv("flow_shock_research/results/flow_pressure_v3.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date

print("="*80)
print("EXTREME FILTER COMBINATIONS")
print("="*80)

configs = [
    ("impact>50", df['flow_impact'] >= 50),
    ("impact>70", df['flow_impact'] >= 70),
    ("impact>100", df['flow_impact'] >= 100),
    ("impact>30 + imb>0.85", (df['flow_impact'] >= 30) & (df['imbalance'] >= 0.85)),
    ("impact>30 + imb>0.9", (df['flow_impact'] >= 30) & (df['imbalance'] >= 0.9)),
    ("impact>20 + imb>0.9 + run>200", (df['flow_impact'] >= 20) & (df['imbalance'] >= 0.9) & (df['max_run'] >= 200)),
    ("impact>30 + imb>0.85 + run>200", (df['flow_impact'] >= 30) & (df['imbalance'] >= 0.85) & (df['max_run'] >= 200)),
    ("impact>50 + imb>0.8", (df['flow_impact'] >= 50) & (df['imbalance'] >= 0.8)),
    ("impact>20 + imb>0.9 + stressed", (df['flow_impact'] >= 20) & (df['imbalance'] >= 0.9) & (df['liquidity_stressed'] == True)),
    ("impact>30 + imb>0.85 + stressed", (df['flow_impact'] >= 30) & (df['imbalance'] >= 0.85) & (df['liquidity_stressed'] == True)),
]

print(f"\n{'Config':>40} | {'Events':>8} | {'Events/Day':>11} | {'Target?':>8}")
print("-"*85)

optimal = []

for name, mask in configs:
    filtered = df[mask]
    
    if len(filtered) == 0:
        print(f"{name:>40} | {'0':>8} | {'0.0':>11} | {'❌':>8}")
        continue
    
    epd = filtered.groupby('date').size()
    avg = epd.mean()
    in_target = "✅" if 10 <= avg <= 50 else "❌"
    
    print(f"{name:>40} | {len(filtered):>8,} | {avg:>11.1f} | {in_target:>8}")
    
    if 10 <= avg <= 50:
        optimal.append((name, filtered, avg))

# Show optimal configs
if optimal:
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATIONS")
    print("="*80)
    
    for name, filtered, avg in optimal:
        print(f"\n✅ {name}")
        print(f"   Events/day: {avg:.1f}")
        print(f"   Total (10d): {len(filtered)}")
        print(f"   Est. (92d): ~{len(filtered) * 92 / 10:.0f}")
        print(f"   Mean impact: {filtered['flow_impact'].mean():.1f}")
        print(f"   Mean imbalance: {filtered['imbalance'].mean():.1%}")
        print(f"   Mean run: {filtered['max_run'].mean():.0f}")
        
        epd = filtered.groupby('date').size()
        print(f"   Daily: {', '.join([f'{d}:{c}' for d, c in epd.items()])}")
else:
    print("\n⚠️  No configuration in target range 10-50 events/day")
    print("   Need even stricter filters")

print("\n" + "="*80)
