#!/usr/bin/env python3
"""
Build Regime Features dataset.

For each forced flow event, extract PRE-EVENT context (t-5m to t0):
- Volatility regime (expanding vs contracting)
- Trend context (ranging vs trending)
- Liquidity state (depth, spread)
- Price structure (distance from recent range)
- Session

Goal: Compare distributions between reversal vs continuation events.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
import sys

DATA_DIR_TRADE = Path("data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")

def load_trades_window(date_str, start_ts, end_ts):
    """Load trades in a time window."""
    trades = []
    
    start_dt = pd.to_datetime(start_ts, unit='ms')
    end_dt = pd.to_datetime(end_ts, unit='ms')
    
    # Determine which hours to load
    hours = set()
    current = start_dt
    while current <= end_dt:
        hours.add(current.hour)
        current += pd.Timedelta(hours=1)
    
    for hour in sorted(hours):
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
                                if start_ts <= ts <= end_ts:
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

def calculate_regime_features(trades_df, event_ts, event_price):
    """Calculate pre-event regime features."""
    if len(trades_df) == 0:
        return None
    
    features = {}
    
    # 1. Volatility Regime
    # Calculate returns in 1m windows
    trades_df['minute'] = (trades_df['timestamp'] // 60000) * 60000
    minute_prices = trades_df.groupby('minute')['price'].agg(['first', 'last', 'min', 'max'])
    
    if len(minute_prices) > 0:
        # Realized volatility (std of 1m returns)
        minute_prices['ret'] = (minute_prices['last'] / minute_prices['first'] - 1) * 10000  # bps
        features['vol_1m'] = minute_prices['ret'].std()
        
        # Range-based volatility
        minute_prices['range'] = (minute_prices['max'] - minute_prices['min']) / minute_prices['first'] * 10000
        features['vol_range'] = minute_prices['range'].mean()
        
        # Volatility trend (expanding or contracting)
        if len(minute_prices) >= 3:
            recent_vol = minute_prices['range'].iloc[-2:].mean()
            earlier_vol = minute_prices['range'].iloc[:-2].mean()
            features['vol_expanding'] = recent_vol > earlier_vol
            features['vol_ratio'] = recent_vol / earlier_vol if earlier_vol > 0 else 1.0
        else:
            features['vol_expanding'] = False
            features['vol_ratio'] = 1.0
    else:
        features['vol_1m'] = 0
        features['vol_range'] = 0
        features['vol_expanding'] = False
        features['vol_ratio'] = 1.0
    
    # 2. Trend Context
    # VWAP distance
    trades_df['notional'] = trades_df['price'] * trades_df['volume']
    vwap = trades_df['notional'].sum() / trades_df['volume'].sum()
    features['vwap_distance'] = (event_price - vwap) / vwap * 10000  # bps
    
    # Price drift (first to last)
    first_price = trades_df['price'].iloc[0]
    features['drift_5m'] = (event_price - first_price) / first_price * 10000  # bps
    
    # Trend strength (linear regression slope)
    trades_df['time_norm'] = (trades_df['timestamp'] - trades_df['timestamp'].min()) / 1000  # seconds
    if len(trades_df) > 10:
        slope = np.polyfit(trades_df['time_norm'], trades_df['price'], 1)[0]
        features['trend_slope'] = slope / event_price * 10000  # bps per second
    else:
        features['trend_slope'] = 0
    
    # 3. Price Structure
    # Distance from recent high/low
    high_5m = trades_df['price'].max()
    low_5m = trades_df['price'].min()
    range_5m = high_5m - low_5m
    
    features['high_5m'] = high_5m
    features['low_5m'] = low_5m
    features['range_5m'] = range_5m / event_price * 10000  # bps
    
    # Position in range
    if range_5m > 0:
        features['price_position'] = (event_price - low_5m) / range_5m  # 0 = low, 1 = high
    else:
        features['price_position'] = 0.5
    
    # 4. Liquidity/Activity
    # Trade rate
    duration_s = (trades_df['timestamp'].max() - trades_df['timestamp'].min()) / 1000
    features['trade_rate'] = len(trades_df) / duration_s if duration_s > 0 else 0
    
    # Volume
    features['volume_5m'] = trades_df['volume'].sum()
    
    # Buy/sell imbalance
    buy_vol = trades_df[trades_df['side'] == 'Buy']['volume'].sum()
    sell_vol = trades_df[trades_df['side'] == 'Sell']['volume'].sum()
    total_vol = buy_vol + sell_vol
    features['buy_imbalance'] = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
    
    return features

def build_regime_dataset(events_file, sample_name):
    """Build regime features for all events in a sample."""
    print(f"\n{'='*80}", flush=True)
    print(f"🔬 BUILDING REGIME FEATURES: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Load events with returns
    df = pd.read_csv(events_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    
    print(f"Total events: {len(df)}", flush=True)
    
    # Filter events with classification
    df = df[df['classification'].notna()].copy()
    print(f"Events with classification: {len(df)}", flush=True)
    
    # Classification breakdown
    print(f"\n📋 Classification:", flush=True)
    for cls, count in df['classification'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {cls:15s}: {count:4d} ({pct:5.1f}%)", flush=True)
    
    # Build regime features
    print(f"\n🔧 Extracting regime features (t-5m to t0)...", flush=True)
    
    regime_features = []
    
    for idx, event in df.iterrows():
        if (idx + 1) % 25 == 0:
            print(f"   Processed {idx + 1}/{len(df)} events...", flush=True)
        
        event_ts = event['timestamp']
        event_price = event['event_price']
        event_date = event['date']
        
        # Load trades from t-5m to t0
        start_ts = event_ts - (5 * 60 * 1000)  # 5 minutes before
        end_ts = event_ts
        
        date_str = event_date.strftime('%Y-%m-%d')
        trades_df = load_trades_window(date_str, start_ts, end_ts)
        
        if len(trades_df) < 10:
            continue
        
        # Calculate regime features
        features = calculate_regime_features(trades_df, event_ts, event_price)
        
        if features:
            # Add event info
            features['timestamp'] = event_ts
            features['datetime'] = event['datetime']
            features['classification'] = event['classification']
            features['ret_30s'] = event['ret_30s']
            features['flow_impact'] = event['flow_impact']
            features['imbalance'] = event['imbalance']
            features['direction'] = event['direction']
            features['hour'] = pd.to_datetime(event['datetime']).hour
            
            regime_features.append(features)
    
    print(f"\n✅ Extracted features for {len(regime_features)} events", flush=True)
    
    return pd.DataFrame(regime_features)

def compare_regimes(df, sample_name):
    """Compare regime features between reversal and continuation."""
    print(f"\n{'='*80}", flush=True)
    print(f"📊 REGIME COMPARISON: {sample_name}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Split by outcome
    reversal = df[df['classification'] == 'Reversal']
    continuation = df[df['classification'] == 'Continuation']
    exhaustion = df[df['classification'] == 'Exhaustion']
    
    print(f"Reversal events: {len(reversal)}", flush=True)
    print(f"Continuation events: {len(continuation)}", flush=True)
    print(f"Exhaustion events: {len(exhaustion)}", flush=True)
    
    if len(reversal) == 0 or len(continuation) == 0:
        print(f"\n⚠️  Not enough events for comparison", flush=True)
        return
    
    # Compare key features
    print(f"\n{'='*80}", flush=True)
    print(f"REGIME FEATURES COMPARISON", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    features_to_compare = [
        ('vol_1m', 'Volatility (1m std)'),
        ('vol_range', 'Volatility (range)'),
        ('vol_expanding', 'Vol Expanding'),
        ('vol_ratio', 'Vol Ratio (recent/earlier)'),
        ('drift_5m', 'Price Drift (5m, bps)'),
        ('trend_slope', 'Trend Slope (bps/s)'),
        ('vwap_distance', 'VWAP Distance (bps)'),
        ('range_5m', 'Range 5m (bps)'),
        ('price_position', 'Price Position (0-1)'),
        ('trade_rate', 'Trade Rate (trades/s)'),
        ('buy_imbalance', 'Buy Imbalance'),
    ]
    
    print(f"{'Feature':>30} | {'Reversal':>12} | {'Continuation':>12} | {'Diff':>8}", flush=True)
    print(f"{'-'*80}", flush=True)
    
    for col, label in features_to_compare:
        if col in df.columns:
            rev_mean = reversal[col].mean()
            cont_mean = continuation[col].mean()
            diff = cont_mean - rev_mean
            
            print(f"{label:>30} | {rev_mean:>12.2f} | {cont_mean:>12.2f} | {diff:>+8.2f}", flush=True)
    
    # Session analysis
    print(f"\n{'='*80}", flush=True)
    print(f"SESSION ANALYSIS", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    print(f"Reversal by hour:", flush=True)
    rev_hours = reversal.groupby('hour').size()
    for hour, count in rev_hours.items():
        pct = count / len(reversal) * 100
        print(f"   Hour {hour:02d}: {count:3d} ({pct:5.1f}%)", flush=True)
    
    print(f"\nContinuation by hour:", flush=True)
    cont_hours = continuation.groupby('hour').size()
    for hour, count in cont_hours.items():
        pct = count / len(continuation) * 100
        print(f"   Hour {hour:02d}: {count:3d} ({pct:5.1f}%)", flush=True)

def main():
    print("="*80, flush=True)
    print("🔬 REGIME FEATURES EXTRACTION", flush=True)
    print("="*80, flush=True)
    print("\nGoal: Extract pre-event context (t-5m to t0) for regime analysis", flush=True)
    print("="*80 + "\n", flush=True)
    
    # Sample 1
    sample1_file = Path("results/sample1_with_returns.csv")
    if sample1_file.exists():
        df1 = build_regime_dataset(sample1_file, "Sample 1 (May 18-24)")
        compare_regimes(df1, "Sample 1")
        df1.to_csv("results/sample1_regime_features.csv", index=False)
        print(f"\n💾 Saved: results/sample1_regime_features.csv", flush=True)
    else:
        print(f"⚠️  Sample 1 not found", flush=True)
        df1 = None
    
    # Sample 2
    sample2_file = Path("results/sample2_with_returns.csv")
    if sample2_file.exists():
        df2 = build_regime_dataset(sample2_file, "Sample 2 (Jul 29 - Aug 4)")
        compare_regimes(df2, "Sample 2")
        df2.to_csv("results/sample2_regime_features.csv", index=False)
        print(f"\n💾 Saved: results/sample2_regime_features.csv", flush=True)
    else:
        print(f"⚠️  Sample 2 not found", flush=True)
        df2 = None
    
    # Combined analysis
    if df1 is not None and df2 is not None:
        print(f"\n{'='*80}", flush=True)
        print(f"📊 COMBINED ANALYSIS", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        combined = pd.concat([df1, df2])
        compare_regimes(combined, "Combined OOS")
        combined.to_csv("results/combined_regime_features.csv", index=False)
        print(f"\n💾 Saved: results/combined_regime_features.csv", flush=True)
    
    print(f"\n{'='*80}", flush=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
