#!/usr/bin/env python3
"""Analyze actual hourly distribution of events to understand session patterns."""
import pandas as pd
import numpy as np
from pathlib import Path

# Load exhaustion confirmation data
df = pd.read_csv("flow_shock_research/results/exhaustion_confirmation.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

print("="*80)
print("HOURLY DISTRIBUTION OF EVENTS")
print("="*80)

# Count by hour
hourly = df.groupby('hour').size().sort_index()

print("\nEvents by Hour (UTC):")
print(f"{'Hour':>6} | {'Count':>6} | {'%':>6} | Bar")
print("-"*50)

max_count = hourly.max()
for hour in range(24):
    count = hourly.get(hour, 0)
    pct = count / len(df) * 100
    bar = '█' * int(count / max_count * 40)
    print(f"{hour:>6} | {count:>6} | {pct:>5.1f}% | {bar}")

print(f"\nTotal: {len(df)} events")

# Traditional session definitions (for reference)
print("\n" + "="*80)
print("TRADITIONAL TRADING SESSIONS (UTC)")
print("="*80)

sessions_traditional = {
    'Tokyo': (0, 9),      # 00:00-09:00 UTC (09:00-18:00 JST)
    'London': (7, 16),    # 07:00-16:00 UTC (08:00-17:00 GMT+1)
    'New York': (12, 21), # 12:00-21:00 UTC (07:00-16:00 EST)
    'Sydney': (22, 7),    # 22:00-07:00 UTC (08:00-17:00 AEDT)
}

print("\nTraditional sessions:")
for name, (start, end) in sessions_traditional.items():
    if end < start:  # Wraps midnight
        mask = (df['hour'] >= start) | (df['hour'] < end)
    else:
        mask = (df['hour'] >= start) & (df['hour'] < end)
    
    count = mask.sum()
    pct = count / len(df) * 100
    
    if end < start:
        hours_str = f"{start:02d}:00-23:59, 00:00-{end:02d}:00"
    else:
        hours_str = f"{start:02d}:00-{end:02d}:00"
    
    print(f"  {name:10s} ({hours_str}): {count:4d} ({pct:5.1f}%)")

# Analyze returns by hour
if 'ret_30s' in df.columns:
    print("\n" + "="*80)
    print("RETURNS BY HOUR")
    print("="*80)
    
    print(f"\n{'Hour':>6} | {'Count':>6} | {'Mean Return':>12} | {'Median':>8}")
    print("-"*50)
    
    for hour in range(24):
        hour_data = df[df['hour'] == hour]
        if len(hour_data) > 0:
            mean_ret = hour_data['ret_30s'].mean()
            median_ret = hour_data['ret_30s'].median()
            print(f"{hour:>6} | {len(hour_data):>6} | {mean_ret:>11.2f} bps | {median_ret:>7.2f}")

# Identify peak hours
print("\n" + "="*80)
print("PEAK ACTIVITY HOURS")
print("="*80)

top_hours = hourly.nlargest(5)
print("\nTop 5 hours by event count:")
for hour, count in top_hours.items():
    pct = count / len(df) * 100
    print(f"  {hour:02d}:00 UTC: {count:4d} events ({pct:5.1f}%)")

# Suggest better session definitions
print("\n" + "="*80)
print("SUGGESTED SESSION DEFINITIONS (based on data)")
print("="*80)

# Find natural clusters
print("\nBased on hourly distribution, suggested sessions:")
print("  Asia-Pacific: 22:00-08:00 UTC (peak: 00:00-04:00)")
print("  Europe: 08:00-16:00 UTC")
print("  Americas: 16:00-22:00 UTC")

# Test these definitions
asia_pacific = ((df['hour'] >= 22) | (df['hour'] < 8)).sum()
europe = ((df['hour'] >= 8) & (df['hour'] < 16)).sum()
americas = ((df['hour'] >= 16) & (df['hour'] < 22)).sum()

print(f"\nEvent distribution with suggested sessions:")
print(f"  Asia-Pacific: {asia_pacific:4d} ({asia_pacific/len(df)*100:5.1f}%)")
print(f"  Europe:       {europe:4d} ({europe/len(df)*100:5.1f}%)")
print(f"  Americas:     {americas:4d} ({americas/len(df)*100:5.1f}%)")

print("\n" + "="*80)
