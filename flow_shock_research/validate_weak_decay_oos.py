#!/usr/bin/env python3
"""
Validate weak decay hypothesis on OOS samples.

Hypothesis: Weak decay (vol_decay_ratio >= 0.5) predicts stronger reversals.

Test on:
- Sample 1 (May 18-24, 2025)
- Sample 2 (Jul 29 - Aug 4, 2025)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_decay_returns(sample_file, sample_name):
    """Analyze relationship between decay and returns for a sample."""
    print(f"\n{'='*80}", flush=True)
    print(f"📊 ANALYZING: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Load sample events
    df = pd.read_csv(sample_file)
    print(f"Total events: {len(df)}", flush=True)
    
    if len(df) == 0:
        print(f"⚠️  No events in sample", flush=True)
        return None
    
    # Calculate decay metrics (same as exhaustion_confirmation.py)
    print(f"\nCalculating decay metrics...", flush=True)
    
    # We need to recalculate decay metrics for these events
    # For now, let's analyze the events we have
    
    # Group by hour to see distribution
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    
    print(f"\n📅 Hourly distribution:", flush=True)
    hourly = df.groupby('hour').size()
    for hour, count in hourly.items():
        pct = count / len(df) * 100
        print(f"   Hour {hour:02d}: {count:4d} ({pct:5.1f}%)", flush=True)
    
    # Analyze by FlowImpact magnitude
    print(f"\n💥 FlowImpact distribution:", flush=True)
    print(f"   Mean: {df['flow_impact'].mean():.1f}", flush=True)
    print(f"   Median: {df['flow_impact'].median():.1f}", flush=True)
    print(f"   Min: {df['flow_impact'].min():.1f}", flush=True)
    print(f"   Max: {df['flow_impact'].max():.1f}", flush=True)
    
    # Test different FlowImpact thresholds
    print(f"\n🔍 Events by FlowImpact threshold:", flush=True)
    for threshold in [70, 100, 150, 200, 300]:
        count = (df['flow_impact'] >= threshold).sum()
        pct = count / len(df) * 100
        print(f"   >= {threshold:3d}: {count:4d} ({pct:5.1f}%)", flush=True)
    
    # Imbalance distribution
    print(f"\n⚖️  Imbalance distribution:", flush=True)
    print(f"   Mean: {df['imbalance'].mean():.1%}", flush=True)
    print(f"   Median: {df['imbalance'].median():.1%}", flush=True)
    
    # Test imbalance thresholds
    print(f"\n🔍 Events by Imbalance threshold:", flush=True)
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        count = (df['imbalance'] >= threshold).sum()
        pct = count / len(df) * 100
        print(f"   >= {threshold:.1f}: {count:4d} ({pct:5.1f}%)", flush=True)
    
    # Run length distribution
    print(f"\n🏃 Run length distribution:", flush=True)
    print(f"   Mean: {df['max_run'].mean():.0f}", flush=True)
    print(f"   Median: {df['max_run'].median():.0f}", flush=True)
    
    # Test combined filters
    print(f"\n{'='*80}", flush=True)
    print(f"QUALITY FILTER COMBINATIONS", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    configs = [
        ("impact>100", df['flow_impact'] >= 100),
        ("impact>100 + imb>0.8", (df['flow_impact'] >= 100) & (df['imbalance'] >= 0.8)),
        ("impact>150 + imb>0.8", (df['flow_impact'] >= 150) & (df['imbalance'] >= 0.8)),
        ("impact>100 + imb>0.9", (df['flow_impact'] >= 100) & (df['imbalance'] >= 0.9)),
        ("impact>150 + imb>0.9", (df['flow_impact'] >= 150) & (df['imbalance'] >= 0.9)),
        ("impact>200 + imb>0.8", (df['flow_impact'] >= 200) & (df['imbalance'] >= 0.8)),
    ]
    
    print(f"{'Filter':>30} | {'Events':>8} | {'% of total':>11}", flush=True)
    print(f"{'-'*60}", flush=True)
    
    for name, mask in configs:
        count = mask.sum()
        pct = count / len(df) * 100
        print(f"{name:>30} | {count:>8} | {pct:>10.1f}%", flush=True)
    
    return df

def main():
    print("="*80, flush=True)
    print("🔬 WEAK DECAY HYPOTHESIS - OOS VALIDATION", flush=True)
    print("="*80, flush=True)
    print("\nHypothesis: Weak decay (sustained pressure) → stronger reversals", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Analyze Sample 1
    sample1_file = Path("results/sample1_may2025.csv")
    if sample1_file.exists():
        df1 = analyze_decay_returns(sample1_file, "Sample 1 (May 18-24, 2025)")
    else:
        print(f"⚠️  Sample 1 file not found: {sample1_file}", flush=True)
        df1 = None
    
    # Analyze Sample 2
    sample2_file = Path("results/sample2_jul2025.csv")
    if sample2_file.exists():
        df2 = analyze_decay_returns(sample2_file, "Sample 2 (Jul 29 - Aug 4, 2025)")
    else:
        print(f"⚠️  Sample 2 file not found: {sample2_file}", flush=True)
        df2 = None
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"📊 SUMMARY", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    if df1 is not None and df2 is not None:
        print(f"Sample 1: {len(df1)} events", flush=True)
        print(f"Sample 2: {len(df2)} events", flush=True)
        print(f"Total: {len(df1) + len(df2)} events\n", flush=True)
        
        # Compare top hours
        print(f"Top hours comparison:", flush=True)
        print(f"  Sample 1: {df1.groupby('hour').size().nlargest(3).to_dict()}", flush=True)
        print(f"  Sample 2: {df2.groupby('hour').size().nlargest(3).to_dict()}", flush=True)
        
        # Compare FlowImpact
        print(f"\nFlowImpact comparison:", flush=True)
        print(f"  Sample 1 mean: {df1['flow_impact'].mean():.1f}", flush=True)
        print(f"  Sample 2 mean: {df2['flow_impact'].mean():.1f}", flush=True)
        
        # Quality filter comparison
        print(f"\nQuality filter (impact>100 + imb>0.8):", flush=True)
        s1_quality = ((df1['flow_impact'] >= 100) & (df1['imbalance'] >= 0.8)).sum()
        s2_quality = ((df2['flow_impact'] >= 100) & (df2['imbalance'] >= 0.8)).sum()
        print(f"  Sample 1: {s1_quality} events ({s1_quality/len(df1)*100:.1f}%)", flush=True)
        print(f"  Sample 2: {s2_quality} events ({s2_quality/len(df2)*100:.1f}%)", flush=True)
        print(f"  Combined: {s1_quality + s2_quality} events", flush=True)
        print(f"  Per day: {(s1_quality + s2_quality) / 14:.1f} events/day", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    print(f"⚠️  NOTE: Need to add forward returns calculation to test decay hypothesis", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
