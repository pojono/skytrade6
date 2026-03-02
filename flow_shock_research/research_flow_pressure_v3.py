#!/usr/bin/env python3
"""
Flow Pressure Detector v3 - Complete Implementation

Measures market PRESSURE, not just volume:
1. FlowImpact = AggVol / TopDepth (pressure metric)
2. Direction: Imbalance > 0.6 (one-sided flow)
3. Robust baseline: median/MAD instead of mean/std
4. Liquidity stress: SpreadRatio, DepthDrop
5. Burst detection: runs of same-side trades

Target: 10-50 events/day (not thousands!)
"""
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd

DATA_DIR_TRADE = Path("/home/ubuntu/Projects/skytrade6/data_bybit/SOLUSDT/trade/dataminer/data/archive/raw")
DATA_DIR_OB = Path("/home/ubuntu/Projects/skytrade6/flow_shock_research/data_bybit/SOLUSDT/orderbook/dataminer/data/archive/raw")

# Configuration
WINDOW_SECONDS = 15  # 10-30s window
TOP_DEPTH_LEVELS = 5  # K=3-10 levels
FLOW_IMPACT_THRESHOLD = 0.6  # Start with 0.6 (strong stress)
IMBALANCE_THRESHOLD = 0.6  # 60%+ one direction
MIN_AGG_TRADES = 20  # Minimum aggressive trades in window
SAME_SIDE_SHARE = 0.75  # 75%+ same direction
RUN_LENGTH = 12  # Consecutive trades same side
RUN_TIME_SECONDS = 3  # Within 3 seconds

class FlowPressureDetector:
    """
    Detects forced flow using pressure metrics.
    """
    
    def __init__(self):
        self.window_ms = WINDOW_SECONDS * 1000
        
        # Trade window
        self.trades = deque()  # (ts, vol, side, price, is_taker)
        
        # Orderbook cache
        self.ob_cache = {}  # ts -> (bid_depth, ask_depth, spread)
        
        # Historical for robust baseline
        self.flow_impact_history = deque(maxlen=10000)  # Last 10k windows
        self.depth_history = deque(maxlen=3600)  # Last hour
        self.spread_history = deque(maxlen=3600)  # Last hour
        
        # Events
        self.events = []
        self.total_trades = 0
        
    def add_orderbook_snapshot(self, ts, bids, asks):
        """Add orderbook snapshot."""
        if not bids or not asks:
            return
        
        # Top K levels depth
        bid_depth = sum(float(b[1]) for b in bids[:TOP_DEPTH_LEVELS])
        ask_depth = sum(float(a[1]) for a in asks[:TOP_DEPTH_LEVELS])
        
        # Spread
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = (best_ask - best_bid) / best_bid
        
        ts_key = (ts // 1000) * 1000
        self.ob_cache[ts_key] = (bid_depth, ask_depth, spread)
        
        # Update history
        total_depth = bid_depth + ask_depth
        self.depth_history.append(total_depth)
        self.spread_history.append(spread)
    
    def get_orderbook_at_time(self, ts):
        """Get orderbook nearest to timestamp."""
        ts_key = (ts // 1000) * 1000
        for offset in range(-10000, 10001, 1000):
            if ts_key + offset in self.ob_cache:
                return self.ob_cache[ts_key + offset]
        return None
    
    def add_trade(self, ts, vol, side, price):
        """Add trade and check for pressure events."""
        self.total_trades += 1
        
        # Assume all trades are taker (we don't have maker/taker flag in data)
        # In production, filter only taker trades
        is_taker = True
        
        self.trades.append((ts, vol, side, price, is_taker))
        
        # Remove old trades
        cutoff = ts - self.window_ms
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()
        
        # Check every 50 trades to reduce overhead
        if len(self.trades) % 50 == 0 and len(self.trades) >= MIN_AGG_TRADES:
            self._check_for_event(ts)
    
    def _check_for_event(self, current_ts):
        """Check if current window has a pressure event."""
        if len(self.trades) < MIN_AGG_TRADES:
            return
        
        # 1. Calculate aggressive volumes
        agg_buy_vol = sum(v for ts, v, s, p, taker in self.trades if s == 'Buy' and taker)
        agg_sell_vol = sum(v for ts, v, s, p, taker in self.trades if s == 'Sell' and taker)
        agg_vol = agg_buy_vol + agg_sell_vol
        
        if agg_vol == 0:
            return
        
        # 2. Direction: Imbalance
        net_agg = agg_buy_vol - agg_sell_vol
        imbalance = abs(net_agg) / agg_vol
        
        if imbalance < IMBALANCE_THRESHOLD:
            return  # Not one-sided enough
        
        direction = 'Buy' if net_agg > 0 else 'Sell'
        
        # 3. Get orderbook
        ob_data = self.get_orderbook_at_time(current_ts)
        if not ob_data:
            return
        
        bid_depth, ask_depth, spread = ob_data
        top_depth = bid_depth + ask_depth
        
        if top_depth == 0:
            return
        
        # 4. FlowImpact = AggVol / TopDepth
        flow_impact = agg_vol / top_depth
        
        # Update history for robust baseline
        self.flow_impact_history.append(flow_impact)
        
        # 5. Robust z-score (optional, for dynamic threshold)
        robust_z = None
        if len(self.flow_impact_history) > 100:
            impacts = np.array(self.flow_impact_history)
            med = np.median(impacts)
            mad = np.median(np.abs(impacts - med))
            if mad > 0:
                robust_z = (flow_impact - med) / (1.4826 * mad)
        
        # 6. Liquidity stress filters
        spread_ratio = None
        depth_drop = None
        
        if len(self.spread_history) > 10:
            median_spread = np.median(list(self.spread_history))
            if median_spread > 0:
                spread_ratio = spread / median_spread
        
        if len(self.depth_history) > 10:
            median_depth = np.median(list(self.depth_history))
            if median_depth > 0:
                depth_drop = top_depth / median_depth
        
        # 7. Burst detection
        agg_trades_count = len([t for t in self.trades if t[4]])  # Count taker trades
        
        # Same-side share
        buy_count = sum(1 for ts, v, s, p, taker in self.trades if s == 'Buy' and taker)
        sell_count = sum(1 for ts, v, s, p, taker in self.trades if s == 'Sell' and taker)
        same_side_share = max(buy_count, sell_count) / max(agg_trades_count, 1)
        
        # Run detection: consecutive trades same direction
        max_run = self._detect_runs()
        
        # 8. DECISION LOGIC
        # Primary: FlowImpact + Imbalance
        passes_primary = flow_impact >= FLOW_IMPACT_THRESHOLD and imbalance >= IMBALANCE_THRESHOLD
        
        # Secondary: Burst
        passes_burst = agg_trades_count >= MIN_AGG_TRADES and same_side_share >= SAME_SIDE_SHARE
        
        # Tertiary: Liquidity stress (optional boost)
        liquidity_stressed = False
        if spread_ratio and spread_ratio > 1.5:
            liquidity_stressed = True
        if depth_drop and depth_drop < 0.7:
            liquidity_stressed = True
        
        # Final decision
        if passes_primary and passes_burst:
            event = {
                'timestamp': int(current_ts),
                'datetime': datetime.fromtimestamp(current_ts / 1000).isoformat(),
                
                # Core metrics
                'flow_impact': float(flow_impact),
                'agg_vol': float(agg_vol),
                'top_depth': float(top_depth),
                'imbalance': float(imbalance),
                'direction': direction,
                
                # Burst metrics
                'agg_trades_count': int(agg_trades_count),
                'same_side_share': float(same_side_share),
                'max_run': int(max_run),
                
                # Liquidity stress
                'spread': float(spread),
                'spread_ratio': float(spread_ratio) if spread_ratio else None,
                'depth_drop': float(depth_drop) if depth_drop else None,
                'liquidity_stressed': liquidity_stressed,
                
                # Robust baseline
                'robust_z': float(robust_z) if robust_z else None,
                
                # Price
                'price': float(self.trades[-1][3])
            }
            
            self.events.append(event)
            print(f"      🔥 EVENT: impact={flow_impact:.2f}, imb={imbalance:.1%}, {direction}, run={max_run}")
    
    def _detect_runs(self):
        """Detect maximum run of consecutive same-side trades within time window."""
        if len(self.trades) < RUN_LENGTH:
            return 0
        
        max_run = 0
        current_run = 1
        run_time_ms = RUN_TIME_SECONDS * 1000
        
        trades_list = list(self.trades)
        for i in range(1, len(trades_list)):
            prev_ts, prev_vol, prev_side, prev_price, prev_taker = trades_list[i-1]
            curr_ts, curr_vol, curr_side, curr_price, curr_taker = trades_list[i]
            
            # Check if same side and within time window
            if curr_side == prev_side and (curr_ts - prev_ts) <= run_time_ms:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return max_run

def parse_trade_file(filepath):
    """Parse trades."""
    trades = []
    try:
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for trade in data['result']['data']:
                            trades.append((
                                int(trade['T']),
                                float(trade['v']),
                                trade['S'],
                                float(trade['p'])
                            ))
                except:
                    continue
    except:
        pass
    return trades

def sample_orderbook_file(filepath, sample_rate=10):
    """Sample orderbook snapshots."""
    snapshots = []
    try:
        with gzip.open(filepath, 'rt') as f:
            for i, line in enumerate(f):
                if i % sample_rate != 0:
                    continue
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        ob_data = data['result']['data']
                        if 'b' in ob_data and 'a' in ob_data:
                            ts = int(data.get('ts', 0))
                            snapshots.append((ts, ob_data['b'], ob_data['a']))
                except:
                    continue
    except:
        pass
    return snapshots

def analyze_flow_pressure(num_days=10):
    """Analyze with flow pressure detector."""
    print("="*80)
    print("⚡ FLOW PRESSURE DETECTOR v3")
    print("="*80)
    print(f"Window: {WINDOW_SECONDS}s")
    print(f"FlowImpact threshold: {FLOW_IMPACT_THRESHOLD}")
    print(f"Imbalance threshold: {IMBALANCE_THRESHOLD}")
    print(f"Burst: {MIN_AGG_TRADES}+ trades, {SAME_SIDE_SHARE:.0%} same side")
    print(f"Run: {RUN_LENGTH}+ consecutive within {RUN_TIME_SECONDS}s")
    print("="*80 + "\n")
    
    start = datetime(2025, 5, 11)
    dates = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
    
    detector = FlowPressureDetector()
    
    # Load orderbook
    print("📂 Loading orderbook (sampled 1/10)...")
    for date_str in dates:
        date_snapshots = 0
        for hour in range(24):
            ob_file = DATA_DIR_OB / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=orderbook" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if ob_file.exists():
                snapshots = sample_orderbook_file(ob_file, sample_rate=10)
                for ts, bids, asks in snapshots:
                    detector.add_orderbook_snapshot(ts, bids, asks)
                date_snapshots += len(snapshots)
        print(f"   {date_str}: {date_snapshots} snapshots")
    
    print(f"\n   Total: {len(detector.ob_cache)} snapshots\n")
    
    # Process trades
    print("🔄 Processing trades...")
    total_trades = 0
    
    for date_str in dates:
        date_trades = []
        for hour in range(24):
            trade_file = DATA_DIR_TRADE / f"dt={date_str}" / f"hr={hour:02d}" / "exchange=bybit" / "source=ws" / "market=linear" / "stream=trade" / "symbol=SOLUSDT" / "data.jsonl.gz"
            if trade_file.exists():
                trades = parse_trade_file(trade_file)
                for ts, vol, side, price in trades:
                    detector.add_trade(ts, vol, side, price)
                date_trades.extend(trades)
        
        total_trades += len(date_trades)
        print(f"   {date_str}: {len(date_trades):,} trades, {len(detector.events)} total events")
    
    print(f"\n✅ Complete! {total_trades:,} trades, {len(detector.events)} events\n")
    
    if detector.events:
        df = pd.DataFrame(detector.events)
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
        df['date'] = df['datetime'].dt.date
        
        print("="*80)
        print("📊 RESULTS")
        print("="*80)
        
        epd = df.groupby('date').size()
        print(f"\nEvents/day: {epd.mean():.1f} avg, {epd.median():.0f} median (range: {epd.min()}-{epd.max()})")
        
        print(f"\nFlow Impact:")
        print(f"  Range: {df['flow_impact'].min():.2f} - {df['flow_impact'].max():.2f}")
        print(f"  Mean: {df['flow_impact'].mean():.2f}")
        print(f"  Median: {df['flow_impact'].median():.2f}")
        
        print(f"\nImbalance: {df['imbalance'].mean():.1%}")
        print(f"Same-side share: {df['same_side_share'].mean():.1%}")
        print(f"Max run: {df['max_run'].mean():.1f} avg")
        
        # Liquidity stress
        stressed = df[df['liquidity_stressed'] == True]
        print(f"\nLiquidity stressed: {len(stressed)}/{len(df)} ({len(stressed)/len(df)*100:.0f}%)")
        
        if 'spread_ratio' in df.columns:
            valid_sr = df[df['spread_ratio'].notna()]
            if len(valid_sr) > 0:
                print(f"Spread ratio: {valid_sr['spread_ratio'].mean():.2f} avg")
        
        if 'depth_drop' in df.columns:
            valid_dd = df[df['depth_drop'].notna()]
            if len(valid_dd) > 0:
                print(f"Depth drop: {valid_dd['depth_drop'].mean():.2f} avg")
        
        # Direction
        dir_counts = df['direction'].value_counts()
        print(f"\nDirection: Buy {dir_counts.get('Buy', 0)}, Sell {dir_counts.get('Sell', 0)}")
        
        # Extrapolate
        est_total = len(df) * (92 / num_days)
        est_per_day = est_total / 92
        print(f"\nExtrapolated (92 days): ~{est_total:.0f} events ({est_per_day:.1f}/day)")
        
        # Save with all metrics
        output = Path("flow_shock_research/results/flow_pressure_v3.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"\n💾 Saved: {output}")
        
        return df
    else:
        print("⚠️  No events detected")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=10)
    args = parser.parse_args()
    
    analyze_flow_pressure(args.days)
