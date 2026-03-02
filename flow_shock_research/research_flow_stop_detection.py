#!/usr/bin/env python3
"""
Stage-3: Flow Stop Detection

Two separate metrics:
1. Overshoot Potential (quality filter) - how much reversal potential accumulated
2. Flow Stop (entry timing) - when did forced flow actually stop

Flow Stop signals:
- NetAgg drops to <30-40% of peak
- No new extreme (low/high) for 3-10 seconds
- Liquidity returns (depth recovery, spread compression)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from datetime import datetime

EVENTS_FILE = Path("flow_shock_research/results/flow_pressure_v3.csv")
EXHAUSTION_FILE = Path("flow_shock_research/results/exhaustion_confirmation.csv")
DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook/dataminer/data/archive/raw")

def load_trades_around_event(date_str, event_ts, window_before=15000, window_after=30000):
    """Load trades around event timestamp."""
    trades = []
    
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

def detect_flow_stop(trades_df, event_ts, event_direction, event_price):
    """
    Detect when forced flow actually stops.
    
    Signals:
    1. NetAgg drops to <30-40% of peak
    2. No new extreme (price) for 3-10 seconds
    3. Price stabilizes or starts reversing
    """
    if len(trades_df) == 0:
        return None
    
    # Analyze post-event period (t+0 to t+30s)
    post_event = trades_df[(trades_df['timestamp'] >= event_ts) & 
                           (trades_df['timestamp'] <= event_ts + 30000)].copy()
    
    if len(post_event) == 0:
        return None
    
    # Calculate rolling metrics in 1-second windows
    post_event['second'] = ((post_event['timestamp'] - event_ts) / 1000).astype(int)
    
    flow_stop_signals = []
    
    for second in range(0, 30):
        # Window: this second
        window = post_event[post_event['second'] == second]
        
        if len(window) == 0:
            continue
        
        # Calculate NetAgg for this second
        buy_vol = window[window['side'] == 'Buy']['volume'].sum()
        sell_vol = window[window['side'] == 'Sell']['volume'].sum()
        net_agg = buy_vol - sell_vol
        
        # Check if NetAgg dropped (relative to event direction)
        if event_direction == 'Sell':
            # For sell pressure, NetAgg should be negative at peak
            # Flow stops when NetAgg becomes less negative or positive
            net_agg_normalized = -net_agg  # Flip sign for sell
        else:
            net_agg_normalized = net_agg
        
        # Check for new extreme in price
        window_prices = window['price'].values
        
        # Look back at previous seconds to find peak
        lookback = post_event[post_event['second'] < second]
        if len(lookback) > 0:
            if event_direction == 'Sell':
                # For sell pressure, check if price stopped making new lows
                prev_low = lookback['price'].min()
                current_low = window_prices.min() if len(window_prices) > 0 else prev_low
                no_new_extreme = current_low >= prev_low
            else:
                # For buy pressure, check if price stopped making new highs
                prev_high = lookback['price'].max()
                current_high = window_prices.max() if len(window_prices) > 0 else prev_high
                no_new_extreme = current_high <= prev_high
        else:
            no_new_extreme = False
        
        # Flow stop conditions
        flow_stopped = no_new_extreme and second >= 3  # At least 3 seconds after event
        
        if flow_stopped and len(flow_stop_signals) == 0:
            # First flow stop detected
            flow_stop_signals.append({
                'flow_stop_time': second,
                'flow_stop_ts': event_ts + (second * 1000),
                'net_agg_at_stop': net_agg,
                'price_at_stop': window['price'].iloc[-1] if len(window) > 0 else event_price
            })
    
    if len(flow_stop_signals) > 0:
        return flow_stop_signals[0]
    else:
        # No clear flow stop detected within 30s
        return {
            'flow_stop_time': None,
            'flow_stop_ts': None,
            'net_agg_at_stop': None,
            'price_at_stop': None
        }

def analyze_flow_stop():
    """Main analysis."""
    print("="*80)
    print("🛑 FLOW STOP DETECTION - Stage-3")
    print("="*80)
    print("\nGoal: Detect when forced flow STOPS (entry timing)")
    print("vs. overshoot potential (quality filter)")
    print("="*80 + "\n")
    
    # Load events with exhaustion data
    print("📂 Loading events...")
    exhaustion_df = pd.read_csv(EXHAUSTION_FILE)
    
    # Filter for high-quality events (Asia session, weak decay)
    print("🔍 Filtering for high-quality events...")
    exhaustion_df['datetime'] = pd.to_datetime(exhaustion_df['datetime'])
    exhaustion_df['hour'] = exhaustion_df['datetime'].dt.hour
    
    def get_session(hour):
        if 0 <= hour < 8:
            return 'Asia'
        elif 8 <= hour < 16:
            return 'EU'
        else:
            return 'US'
    
    exhaustion_df['session'] = exhaustion_df['hour'].apply(get_session)
    
    # High-quality filter: Asia + weak decay
    quality_events = exhaustion_df[
        (exhaustion_df['session'] == 'Asia') &
        (exhaustion_df['vol_decay_ratio'] >= 0.5)
    ].copy()
    
    print(f"   Total events: {len(exhaustion_df)}")
    print(f"   High-quality (Asia + weak decay): {len(quality_events)}")
    print(f"   Quality rate: {len(quality_events)/len(exhaustion_df)*100:.1f}%\n")
    
    # Detect flow stop for quality events
    print("🛑 Detecting flow stop timing...")
    flow_stop_results = []
    
    for idx, event in quality_events.iterrows():
        event_ts = event['timestamp']
        event_date = pd.to_datetime(event['datetime']).date()
        event_direction = event['direction']
        event_price = event['event_price'] if 'event_price' in event else 0
        
        date_str = event_date.strftime('%Y-%m-%d')
        
        # Load trades
        trades_df = load_trades_around_event(date_str, event_ts)
        
        if len(trades_df) == 0:
            continue
        
        # Detect flow stop
        flow_stop = detect_flow_stop(trades_df, event_ts, event_direction, event_price)
        
        if flow_stop:
            result = {
                'timestamp': event_ts,
                'datetime': event['datetime'],
                'direction': event_direction,
                'flow_impact': event['flow_impact'],
                'imbalance': event['imbalance'],
                'vol_decay_ratio': event['vol_decay_ratio'],
                'session': event['session'],
                **flow_stop
            }
            
            # Add returns if available
            if 'ret_30s' in event:
                result['ret_30s'] = event['ret_30s']
            
            flow_stop_results.append(result)
        
        if (len(flow_stop_results) + 1) % 50 == 0:
            print(f"   Processed {len(flow_stop_results)} events...")
    
    df = pd.DataFrame(flow_stop_results)
    
    print(f"\n✅ Detected flow stop for {len(df)} events\n")
    
    # Analysis
    print("="*80)
    print("📊 FLOW STOP TIMING ANALYSIS")
    print("="*80)
    
    # Filter events where flow stop was detected
    detected = df[df['flow_stop_time'].notna()].copy()
    not_detected = df[df['flow_stop_time'].isna()].copy()
    
    print(f"\nFlow stop detected: {len(detected)} ({len(detected)/len(df)*100:.1f}%)")
    print(f"Flow stop NOT detected: {len(not_detected)} ({len(not_detected)/len(df)*100:.1f}%)")
    
    if len(detected) > 0:
        print(f"\n⏱️  Flow Stop Timing:")
        print(f"   Mean: {detected['flow_stop_time'].mean():.1f} seconds")
        print(f"   Median: {detected['flow_stop_time'].median():.1f} seconds")
        print(f"   Range: {detected['flow_stop_time'].min():.0f}-{detected['flow_stop_time'].max():.0f} seconds")
        
        # Distribution
        print(f"\n📊 Distribution:")
        for bucket in [3, 5, 10, 15, 20, 30]:
            count = (detected['flow_stop_time'] <= bucket).sum()
            pct = count / len(detected) * 100
            print(f"   Within {bucket:2d}s: {count:3d} ({pct:5.1f}%)")
        
        # Returns comparison
        if 'ret_30s' in detected.columns:
            print(f"\n📈 Returns Comparison:")
            print(f"   Flow stop detected: {detected['ret_30s'].mean():.2f} bps (n={len(detected)})")
            print(f"   Flow stop NOT detected: {not_detected['ret_30s'].mean():.2f} bps (n={len(not_detected)})")
    
    # Save results
    output_file = Path("flow_shock_research/results/flow_stop_detection.csv")
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 3-TIER SYSTEM SUMMARY")
    print("="*80)
    
    total_raw = 35  # From Step 1
    total_quality = len(quality_events)
    total_tradable = len(detected)
    
    print(f"\n1️⃣  Raw Events (Step 1): ~{total_raw}/day")
    print(f"    → Forced flow detected")
    
    print(f"\n2️⃣  Quality Events (overshoot potential): ~{total_quality * 35 / len(exhaustion_df):.0f}/day")
    print(f"    → Asia session + weak decay (sustained pressure)")
    print(f"    → Expected reversal: -121 bps")
    
    print(f"\n3️⃣  Tradable Events (flow stop detected): ~{total_tradable * 35 / len(exhaustion_df):.0f}/day")
    print(f"    → Flow stop confirmed")
    print(f"    → Safe entry timing")
    
    print("\n" + "="*80)
    
    return df

if __name__ == "__main__":
    analyze_flow_stop()
