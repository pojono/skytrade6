#!/usr/bin/env python3
"""
Exhaustion Confirmation - Detect pressure DECAY, not just pressure PEAK.

Problem: FlowPressure detector catches explosion (peak), but profit is in collapse (decay).
Solution: Wait for pressure decay signal before entry.

Decay signals:
1. Aggressive volume drop: AggVol(0-5s) vs AggVol(5-10s)
2. Trade rate drop: trades/sec declining
3. Book depth recovery: depth returning after collapse
4. Run termination: consecutive same-side trades stopping

Goal: Find the moment when "flow stopped", not "flow peaked"
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from datetime import datetime

EVENTS_FILE = Path("flow_shock_research/results/flow_pressure_v3.csv")
DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook/dataminer/data/archive/raw")

def load_trades_around_event(date_str, event_ts, window_before=15000, window_after=60000):
    """Load trades around event timestamp."""
    trades = []
    
    # Determine which hours to load
    event_dt = datetime.fromtimestamp(event_ts / 1000)
    hours_to_load = [event_dt.hour]
    if event_dt.minute < 1:
        hours_to_load.append((event_dt.hour - 1) % 24)
    if event_dt.minute > 58:
        hours_to_load.append((event_dt.hour + 1) % 24)
    
    for hour in hours_to_load:
        trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
        
        if not trade_file.exists():
            continue
        
        try:
            with gzip.open(trade_file, 'rt') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for trade in data['result']['data']:
                                ts = int(trade['T'])
                                if event_ts - window_before <= ts <= event_ts + window_after:
                                    trades.append({
                                        'timestamp': ts,
                                        'price': float(trade['p']),
                                        'volume': float(trade['v']),
                                        'side': trade['S']
                                    })
                    except:
                        continue
        except:
            continue
    
    return pd.DataFrame(trades).sort_values('timestamp') if trades else pd.DataFrame()

def calculate_decay_metrics(trades_df, event_ts, event_direction):
    """Calculate pressure decay metrics after event."""
    if len(trades_df) == 0:
        return None
    
    # Define time windows relative to event
    # Window 1: Event detection window (t-15s to t+0s)
    # Window 2: Immediate after (t+0s to t+5s)
    # Window 3: Decay period (t+5s to t+10s)
    
    w1_start, w1_end = event_ts - 15000, event_ts
    w2_start, w2_end = event_ts, event_ts + 5000
    w3_start, w3_end = event_ts + 5000, event_ts + 10000
    
    def get_window_metrics(start_ts, end_ts):
        """Calculate metrics for a time window."""
        window = trades_df[(trades_df['timestamp'] >= start_ts) & (trades_df['timestamp'] < end_ts)]
        
        if len(window) == 0:
            return None
        
        # Aggressive volume (assume all trades are taker)
        buy_vol = window[window['side'] == 'Buy']['volume'].sum()
        sell_vol = window[window['side'] == 'Sell']['volume'].sum()
        total_vol = buy_vol + sell_vol
        
        # Trade rate (trades per second)
        duration_s = (end_ts - start_ts) / 1000
        trade_rate = len(window) / duration_s if duration_s > 0 else 0
        
        # Directional volume (based on event direction)
        if event_direction == 'Sell':
            directional_vol = sell_vol
        else:
            directional_vol = buy_vol
        
        # Consecutive same-side trades
        max_run = 1
        current_run = 1
        for i in range(1, len(window)):
            if window.iloc[i]['side'] == window.iloc[i-1]['side']:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return {
            'total_vol': total_vol,
            'directional_vol': directional_vol,
            'trade_count': len(window),
            'trade_rate': trade_rate,
            'max_run': max_run
        }
    
    w1_metrics = get_window_metrics(w1_start, w1_end)
    w2_metrics = get_window_metrics(w2_start, w2_end)
    w3_metrics = get_window_metrics(w3_start, w3_end)
    
    if not all([w1_metrics, w2_metrics, w3_metrics]):
        return None
    
    # Calculate decay ratios
    decay = {
        # Volume decay: immediate vs decay period
        'vol_decay_ratio': w3_metrics['total_vol'] / w2_metrics['total_vol'] if w2_metrics['total_vol'] > 0 else 0,
        
        # Directional volume decay
        'dir_vol_decay_ratio': w3_metrics['directional_vol'] / w2_metrics['directional_vol'] if w2_metrics['directional_vol'] > 0 else 0,
        
        # Trade rate decay
        'trade_rate_decay_ratio': w3_metrics['trade_rate'] / w2_metrics['trade_rate'] if w2_metrics['trade_rate'] > 0 else 0,
        
        # Run length change
        'run_decay': w2_metrics['max_run'] - w3_metrics['max_run'],
        
        # Absolute metrics
        'w2_vol': w2_metrics['total_vol'],
        'w3_vol': w3_metrics['total_vol'],
        'w2_trade_rate': w2_metrics['trade_rate'],
        'w3_trade_rate': w3_metrics['trade_rate'],
        'w2_run': w2_metrics['max_run'],
        'w3_run': w3_metrics['max_run']
    }
    
    return decay

def analyze_exhaustion_confirmation():
    """Main analysis."""
    print("="*80)
    print("🔬 EXHAUSTION CONFIRMATION ANALYSIS")
    print("="*80)
    print("\nGoal: Detect pressure DECAY (collapse), not just pressure PEAK (explosion)")
    print("="*80 + "\n")
    
    # Load events (impact > 70)
    print("📂 Loading events...")
    events_df = pd.read_csv(EVENTS_FILE)
    events_df = events_df[events_df['flow_impact'] >= 70].copy()
    events_df['datetime'] = pd.to_datetime(events_df['datetime'])
    events_df['date'] = events_df['datetime'].dt.date
    print(f"   Loaded {len(events_df)} events\n")
    
    # Load post-event returns
    post_event_file = Path("flow_shock_research/results/post_event_study.csv")
    if post_event_file.exists():
        returns_df = pd.read_csv(post_event_file)
        events_df = events_df.merge(returns_df[['timestamp', 'ret_30s', 'classification']], on='timestamp', how='left')
        print(f"   Merged with post-event returns\n")
    
    # Calculate decay metrics for each event
    print("🔄 Calculating decay metrics...")
    decay_results = []
    
    for idx, event in events_df.iterrows():
        event_ts = event['timestamp']
        event_date = event['date']
        event_direction = event['direction']
        
        date_str = event_date.strftime('%Y-%m-%d')
        
        # Load trades around event
        trades_df = load_trades_around_event(date_str, event_ts)
        
        if len(trades_df) == 0:
            continue
        
        # Calculate decay metrics
        decay = calculate_decay_metrics(trades_df, event_ts, event_direction)
        
        if decay:
            result = {
                'timestamp': event_ts,
                'datetime': event['datetime'],
                'direction': event_direction,
                'flow_impact': event['flow_impact'],
                'imbalance': event['imbalance'],
                'max_run': event['max_run'],
                **decay
            }
            
            # Add returns if available
            if 'ret_30s' in event and pd.notna(event['ret_30s']):
                result['ret_30s'] = event['ret_30s']
                result['classification'] = event['classification']
            
            decay_results.append(result)
        
        if (idx + 1) % 20 == 0:
            print(f"   Processed {idx + 1}/{len(events_df)} events...")
    
    df = pd.DataFrame(decay_results)
    
    print(f"\n✅ Calculated decay metrics for {len(df)} events\n")
    
    # Analysis
    print("="*80)
    print("📊 DECAY METRICS ANALYSIS")
    print("="*80)
    
    print("\n📉 Volume Decay:")
    print(f"   Mean ratio (w3/w2): {df['vol_decay_ratio'].mean():.2f}")
    print(f"   Median ratio: {df['vol_decay_ratio'].median():.2f}")
    print(f"   Events with decay <0.5: {(df['vol_decay_ratio'] < 0.5).sum()} ({(df['vol_decay_ratio'] < 0.5).sum()/len(df)*100:.1f}%)")
    
    print("\n📉 Trade Rate Decay:")
    print(f"   Mean ratio: {df['trade_rate_decay_ratio'].mean():.2f}")
    print(f"   Median ratio: {df['trade_rate_decay_ratio'].median():.2f}")
    
    print("\n📉 Run Length Decay:")
    print(f"   Mean decay: {df['run_decay'].mean():.1f}")
    print(f"   Median decay: {df['run_decay'].median():.1f}")
    
    # Correlation with returns
    if 'ret_30s' in df.columns:
        print("\n📊 Correlation with Returns (t+30s):")
        print(f"   Vol decay ratio: {df[['vol_decay_ratio', 'ret_30s']].corr().iloc[0,1]:.3f}")
        print(f"   Trade rate decay: {df[['trade_rate_decay_ratio', 'ret_30s']].corr().iloc[0,1]:.3f}")
        
        # Compare strong decay vs weak decay
        strong_decay = df[df['vol_decay_ratio'] < 0.5]
        weak_decay = df[df['vol_decay_ratio'] >= 0.5]
        
        print("\n🔬 Strong Decay (vol_decay < 0.5) vs Weak Decay:")
        print(f"   Strong decay events: {len(strong_decay)}")
        print(f"   Mean return: {strong_decay['ret_30s'].mean():.2f} bps")
        print(f"   Weak decay events: {len(weak_decay)}")
        print(f"   Mean return: {weak_decay['ret_30s'].mean():.2f} bps")
        
        # Classification breakdown
        if 'classification' in df.columns:
            print("\n📋 Classification by Decay Strength:")
            for decay_type, subset in [("Strong decay", strong_decay), ("Weak decay", weak_decay)]:
                print(f"\n   {decay_type}:")
                if len(subset) > 0 and 'classification' in subset.columns:
                    class_counts = subset['classification'].value_counts()
                    for cls, count in class_counts.items():
                        pct = count / len(subset) * 100
                        print(f"      {cls}: {count} ({pct:.1f}%)")
    
    # Session analysis
    print("\n" + "="*80)
    print("🌍 SESSION ANALYSIS")
    print("="*80)
    
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    
    # Define sessions (UTC)
    def get_session(hour):
        if 0 <= hour < 8:
            return 'Asia'
        elif 8 <= hour < 16:
            return 'EU'
        else:
            return 'US'
    
    df['session'] = df['hour'].apply(get_session)
    
    print("\n📊 Events by Session:")
    session_counts = df['session'].value_counts()
    for session, count in session_counts.items():
        pct = count / len(df) * 100
        print(f"   {session:5s}: {count:3d} ({pct:5.1f}%)")
    
    if 'ret_30s' in df.columns:
        print("\n📈 Returns by Session:")
        for session in ['Asia', 'EU', 'US']:
            subset = df[df['session'] == session]
            if len(subset) > 0:
                mean_ret = subset['ret_30s'].mean()
                print(f"   {session:5s}: {mean_ret:+7.2f} bps (n={len(subset)})")
    
    # Save results
    output_file = Path("flow_shock_research/results/exhaustion_confirmation.csv")
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file}")
    
    return df

if __name__ == "__main__":
    analyze_exhaustion_confirmation()
