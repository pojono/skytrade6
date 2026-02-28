#!/usr/bin/env python3
"""
Real-Time Settlement Sell Wave Predictor
=========================================
Calculates orderbook features in real-time (T-10s to T-0) and generates
a confidence score for expected post-settlement price drop.

Based on analysis of 64 settlements showing:
- Spread width: r = -0.52 (wider spread → more sell pressure)
- Bid/ask qty imbalance: r = +0.52 (bid-heavy → more sells)
- Total depth: r = -0.43 (thin orderbook → larger drops)

Usage:
    from settlement_predictor import SettlementPredictor
    
    predictor = SettlementPredictor()
    
    # Feed live data
    predictor.add_orderbook_snapshot(timestamp, bids, asks, depth_level)
    predictor.add_trade(timestamp, price, qty, side)
    
    # Get prediction at T-1s
    signal = predictor.get_signal()
    # Returns: {
    #   'confidence': 0-100,
    #   'expected_drop_bps': float,
    #   'position_size_multiplier': 0.0-2.0,
    #   'should_trade': bool,
    #   'features': {...},
    #   'reasons': [...]
    # }
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import numpy as np


@dataclass
class OrderbookSnapshot:
    """Single orderbook snapshot."""
    timestamp: float  # Unix timestamp in seconds
    bids: List[Tuple[float, float]]  # [(price, qty), ...]
    asks: List[Tuple[float, float]]
    depth_level: int  # 1, 50, or 200


@dataclass
class Trade:
    """Single trade."""
    timestamp: float
    price: float
    qty: float
    side: str  # "Buy" or "Sell"


class SettlementPredictor:
    """Real-time predictor for post-settlement sell waves."""
    
    # Thresholds from analysis (settlement_predictability_analysis.csv)
    THIN_ORDERBOOK_USD = 10000  # < this = thin
    DEEP_ORDERBOOK_USD = 30000  # > this = deep
    WIDE_SPREAD_BPS = 8.0       # > this = wide
    TIGHT_SPREAD_BPS = 3.0      # < this = tight
    HIGH_SPREAD_STD_BPS = 4.0   # > this = volatile
    BID_HEAVY_THRESHOLD = 0.2   # > this = bid-heavy
    HIGH_VOLATILITY_BPS = 4.0   # > this = high vol
    
    # Expected drops from analysis
    EXPECTED_DROP_THIN_OB = 120.0  # bps
    EXPECTED_DROP_WIDE_SPREAD = 100.0
    EXPECTED_DROP_BID_HEAVY = 90.0
    EXPECTED_DROP_BASELINE = 50.0  # median from analysis
    
    def __init__(self, window_seconds: float = 10.0):
        """
        Args:
            window_seconds: How many seconds before settlement to analyze (default 10s)
        """
        self.window_seconds = window_seconds
        
        # Rolling data buffers
        self.orderbooks_1: deque = deque(maxlen=1000)   # 10s @ 10ms = 1000 snapshots
        self.orderbooks_50: deque = deque(maxlen=500)   # 10s @ 20ms = 500 snapshots
        self.orderbooks_200: deque = deque(maxlen=100)  # 10s @ 100ms = 100 snapshots
        self.trades: deque = deque(maxlen=5000)         # ~500 trades/sec max
        
        self.settlement_time: Optional[float] = None
        self.ref_price: Optional[float] = None
    
    def set_settlement_time(self, timestamp: float):
        """Set the settlement timestamp (Unix time in seconds)."""
        self.settlement_time = timestamp
    
    def add_orderbook_snapshot(
        self,
        timestamp: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        depth_level: int = 200,
    ):
        """Add an orderbook snapshot.
        
        Args:
            timestamp: Unix timestamp in seconds
            bids: [(price, qty), ...] sorted descending
            asks: [(price, qty), ...] sorted ascending
            depth_level: 1, 50, or 200
        """
        snapshot = OrderbookSnapshot(timestamp, bids, asks, depth_level)
        
        if depth_level == 1:
            self.orderbooks_1.append(snapshot)
        elif depth_level == 50:
            self.orderbooks_50.append(snapshot)
        elif depth_level == 200:
            self.orderbooks_200.append(snapshot)
    
    def add_trade(self, timestamp: float, price: float, qty: float, side: str):
        """Add a trade."""
        self.trades.append(Trade(timestamp, price, qty, side))
        
        # Update reference price (most recent trade)
        self.ref_price = price
    
    def _get_recent_data(self, buffer: deque, cutoff_time: float) -> List:
        """Get data from buffer after cutoff_time."""
        return [item for item in buffer if item.timestamp >= cutoff_time]
    
    def _to_bps(self, price: float) -> float:
        """Convert price to bps relative to reference price."""
        if self.ref_price is None:
            return 0.0
        return (price - self.ref_price) / self.ref_price * 10000
    
    def calculate_features(self, current_time: Optional[float] = None) -> Dict:
        """Calculate all orderbook features for the window ending at current_time.
        
        Args:
            current_time: Unix timestamp. If None, uses time.time()
        
        Returns:
            Dict of features
        """
        if current_time is None:
            current_time = time.time()
        
        cutoff_time = current_time - self.window_seconds
        
        # Get recent data
        recent_ob1 = self._get_recent_data(self.orderbooks_1, cutoff_time)
        recent_ob50 = self._get_recent_data(self.orderbooks_50, cutoff_time)
        recent_ob200 = self._get_recent_data(self.orderbooks_200, cutoff_time)
        recent_trades = self._get_recent_data(self.trades, cutoff_time)
        
        features = {
            'timestamp': current_time,
            'data_quality': {
                'ob1_count': len(recent_ob1),
                'ob50_count': len(recent_ob50),
                'ob200_count': len(recent_ob200),
                'trade_count': len(recent_trades),
            }
        }
        
        # --- Spread features (from orderbook.1) ---
        if recent_ob1:
            spreads_bps = []
            qty_imbalances = []
            
            for ob in recent_ob1:
                if ob.bids and ob.asks:
                    bid = ob.bids[0][0]
                    ask = ob.asks[0][0]
                    spread_bps = self._to_bps(ask) - self._to_bps(bid)
                    spreads_bps.append(spread_bps)
                    
                    bid_qty = ob.bids[0][1]
                    ask_qty = ob.asks[0][1]
                    if (bid_qty + ask_qty) > 0:
                        qty_imb = (bid_qty - ask_qty) / (bid_qty + ask_qty)
                        qty_imbalances.append(qty_imb)
            
            if spreads_bps:
                features['spread_mean_bps'] = np.mean(spreads_bps)
                features['spread_std_bps'] = np.std(spreads_bps)
                features['spread_max_bps'] = np.max(spreads_bps)
                features['spread_min_bps'] = np.min(spreads_bps)
            else:
                features['spread_mean_bps'] = None
                features['spread_std_bps'] = None
                features['spread_max_bps'] = None
                features['spread_min_bps'] = None
            
            if qty_imbalances:
                features['qty_imb_mean'] = np.mean(qty_imbalances)
            else:
                features['qty_imb_mean'] = None
        else:
            features['spread_mean_bps'] = None
            features['spread_std_bps'] = None
            features['spread_max_bps'] = None
            features['spread_min_bps'] = None
            features['qty_imb_mean'] = None
        
        # --- Depth features (from orderbook.50) ---
        if recent_ob50:
            depth_imbalances = []
            bid10_notionals = []
            ask10_notionals = []
            
            for ob in recent_ob50:
                if ob.bids and ob.asks:
                    bid10 = sum(p * q for p, q in ob.bids[:10])
                    ask10 = sum(p * q for p, q in ob.asks[:10])
                    
                    bid10_notionals.append(bid10)
                    ask10_notionals.append(ask10)
                    
                    if (bid10 + ask10) > 0:
                        depth_imb = (bid10 - ask10) / (bid10 + ask10)
                        depth_imbalances.append(depth_imb)
            
            if depth_imbalances:
                features['depth_imb_mean'] = np.mean(depth_imbalances)
            else:
                features['depth_imb_mean'] = None
            
            if bid10_notionals:
                features['bid10_mean_usd'] = np.mean(bid10_notionals)
                features['ask10_mean_usd'] = np.mean(ask10_notionals)
            else:
                features['bid10_mean_usd'] = None
                features['ask10_mean_usd'] = None
        else:
            features['depth_imb_mean'] = None
            features['bid10_mean_usd'] = None
            features['ask10_mean_usd'] = None
        
        # --- Total depth features (from orderbook.200) ---
        if recent_ob200:
            total_bid_notionals = []
            total_ask_notionals = []
            
            for ob in recent_ob200:
                if ob.bids and ob.asks:
                    total_bid = sum(p * q for p, q in ob.bids)
                    total_ask = sum(p * q for p, q in ob.asks)
                    
                    total_bid_notionals.append(total_bid)
                    total_ask_notionals.append(total_ask)
            
            if total_bid_notionals:
                features['total_bid_mean_usd'] = np.mean(total_bid_notionals)
                features['total_ask_mean_usd'] = np.mean(total_ask_notionals)
                features['total_depth_usd'] = features['total_bid_mean_usd'] + features['total_ask_mean_usd']
            else:
                features['total_bid_mean_usd'] = None
                features['total_ask_mean_usd'] = None
                features['total_depth_usd'] = None
        else:
            features['total_bid_mean_usd'] = None
            features['total_ask_mean_usd'] = None
            features['total_depth_usd'] = None
        
        # --- Trade flow features ---
        if recent_trades:
            buy_vol = sum(t.price * t.qty for t in recent_trades if t.side == "Buy")
            sell_vol = sum(t.price * t.qty for t in recent_trades if t.side == "Sell")
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                features['trade_flow_imb'] = (buy_vol - sell_vol) / total_vol
            else:
                features['trade_flow_imb'] = 0.0
            
            # Price volatility
            prices = [t.price for t in sorted(recent_trades, key=lambda x: x.timestamp)]
            if len(prices) > 1:
                price_changes_bps = [self._to_bps(prices[i]) - self._to_bps(prices[i-1]) 
                                     for i in range(1, len(prices))]
                features['price_vol_bps'] = np.std(price_changes_bps)
            else:
                features['price_vol_bps'] = 0.0
        else:
            features['trade_flow_imb'] = 0.0
            features['price_vol_bps'] = 0.0
        
        return features
    
    def get_signal(self, current_time: Optional[float] = None) -> Dict:
        """Generate trading signal based on current orderbook state.
        
        Returns:
            {
                'confidence': 0-100,
                'expected_drop_bps': float,
                'position_size_multiplier': 0.0-2.0,
                'should_trade': bool,
                'features': {...},
                'reasons': [...],
                'warnings': [...]
            }
        """
        features = self.calculate_features(current_time)
        
        confidence_score = 0.0
        expected_drop = self.EXPECTED_DROP_BASELINE
        reasons = []
        warnings = []
        
        # --- Signal 1: Spread Width (strongest predictor, r = -0.52) ---
        spread_mean = features.get('spread_mean_bps')
        if spread_mean is not None:
            if spread_mean > self.WIDE_SPREAD_BPS:
                boost = min(30, (spread_mean - self.WIDE_SPREAD_BPS) * 2)
                confidence_score += boost
                expected_drop += self.EXPECTED_DROP_WIDE_SPREAD * (boost / 30)
                reasons.append(f"Wide spread ({spread_mean:.1f} bps) → +{boost:.0f} confidence")
            elif spread_mean < self.TIGHT_SPREAD_BPS:
                penalty = 20
                confidence_score -= penalty
                reasons.append(f"Tight spread ({spread_mean:.1f} bps) → -{penalty} confidence")
        else:
            warnings.append("No orderbook.1 data (spread unknown)")
        
        # --- Signal 2: Spread Volatility (r = -0.35 to -0.41) ---
        spread_std = features.get('spread_std_bps')
        if spread_std is not None:
            if spread_std > self.HIGH_SPREAD_STD_BPS:
                boost = min(20, (spread_std - self.HIGH_SPREAD_STD_BPS) * 3)
                confidence_score += boost
                expected_drop += 30 * (boost / 20)
                reasons.append(f"Volatile spread (std={spread_std:.1f} bps) → +{boost:.0f} confidence")
        
        # --- Signal 3: Bid/Ask Qty Imbalance (r = +0.52) ---
        qty_imb = features.get('qty_imb_mean')
        if qty_imb is not None:
            if qty_imb > self.BID_HEAVY_THRESHOLD:
                boost = min(25, qty_imb * 100)
                confidence_score += boost
                expected_drop += self.EXPECTED_DROP_BID_HEAVY * (boost / 25)
                reasons.append(f"Bid-heavy orderbook (imb={qty_imb:+.2f}) → +{boost:.0f} confidence")
            elif qty_imb < -self.BID_HEAVY_THRESHOLD:
                penalty = 15
                confidence_score -= penalty
                reasons.append(f"Ask-heavy orderbook (imb={qty_imb:+.2f}) → -{penalty} confidence")
        else:
            warnings.append("No orderbook.1 data (qty imbalance unknown)")
        
        # --- Signal 4: Total Orderbook Depth (r = -0.43) ---
        total_depth = features.get('total_depth_usd')
        if total_depth is not None:
            if total_depth < self.THIN_ORDERBOOK_USD:
                boost = min(25, (self.THIN_ORDERBOOK_USD - total_depth) / 400)
                confidence_score += boost
                expected_drop += self.EXPECTED_DROP_THIN_OB * (boost / 25)
                reasons.append(f"Thin orderbook (${total_depth:.0f}) → +{boost:.0f} confidence")
            elif total_depth > self.DEEP_ORDERBOOK_USD:
                penalty = min(25, (total_depth - self.DEEP_ORDERBOOK_USD) / 1000)
                confidence_score -= penalty
                expected_drop *= 0.6  # Deep orderbook reduces expected drop
                reasons.append(f"Deep orderbook (${total_depth:.0f}) → -{penalty:.0f} confidence")
        else:
            warnings.append("No orderbook.200 data (total depth unknown)")
        
        # --- Signal 5: Price Volatility (r = -0.39) ---
        price_vol = features.get('price_vol_bps')
        if price_vol is not None and price_vol > self.HIGH_VOLATILITY_BPS:
            boost = min(15, (price_vol - self.HIGH_VOLATILITY_BPS) * 2)
            confidence_score += boost
            expected_drop += 20 * (boost / 15)
            reasons.append(f"High volatility ({price_vol:.1f} bps) → +{boost:.0f} confidence")
        
        # --- Normalize confidence to 0-100 ---
        confidence_score = max(0, min(100, confidence_score + 50))  # Baseline at 50
        
        # --- Position sizing multiplier ---
        if confidence_score >= 75:
            size_mult = 2.0
        elif confidence_score >= 60:
            size_mult = 1.5
        elif confidence_score >= 50:
            size_mult = 1.0
        elif confidence_score >= 40:
            size_mult = 0.5
        else:
            size_mult = 0.0
        
        # --- Trading decision ---
        should_trade = confidence_score >= 40 and size_mult > 0
        
        if not should_trade and confidence_score < 40:
            reasons.append(f"⚠ Low confidence ({confidence_score:.0f}) - SKIP TRADE")
        
        return {
            'confidence': round(confidence_score, 1),
            'expected_drop_bps': round(expected_drop, 1),
            'position_size_multiplier': size_mult,
            'should_trade': should_trade,
            'features': features,
            'reasons': reasons,
            'warnings': warnings,
        }
    
    def reset(self):
        """Clear all buffers."""
        self.orderbooks_1.clear()
        self.orderbooks_50.clear()
        self.orderbooks_200.clear()
        self.trades.clear()
        self.ref_price = None


def format_signal(signal: Dict) -> str:
    """Format signal for human-readable output."""
    lines = []
    lines.append("=" * 80)
    lines.append("SETTLEMENT SELL WAVE PREDICTION")
    lines.append("=" * 80)
    lines.append(f"Confidence:       {signal['confidence']:.1f}/100")
    lines.append(f"Expected Drop:    {signal['expected_drop_bps']:.1f} bps")
    lines.append(f"Position Size:    {signal['position_size_multiplier']:.1f}x")
    lines.append(f"Trade Decision:   {'✅ TRADE' if signal['should_trade'] else '❌ SKIP'}")
    lines.append("")
    
    if signal['reasons']:
        lines.append("Reasoning:")
        for reason in signal['reasons']:
            lines.append(f"  • {reason}")
        lines.append("")
    
    if signal['warnings']:
        lines.append("⚠ Warnings:")
        for warning in signal['warnings']:
            lines.append(f"  • {warning}")
        lines.append("")
    
    lines.append("Key Features:")
    f = signal['features']
    if f.get('spread_mean_bps') is not None:
        lines.append(f"  Spread:           {f['spread_mean_bps']:.1f} ± {f.get('spread_std_bps', 0):.1f} bps")
    if f.get('qty_imb_mean') is not None:
        lines.append(f"  Qty Imbalance:    {f['qty_imb_mean']:+.2f}")
    if f.get('total_depth_usd') is not None:
        lines.append(f"  Total Depth:      ${f['total_depth_usd']:.0f}")
    if f.get('price_vol_bps') is not None:
        lines.append(f"  Price Volatility: {f['price_vol_bps']:.1f} bps")
    
    lines.append("")
    lines.append("Data Quality:")
    dq = f['data_quality']
    lines.append(f"  OB.1 snapshots:   {dq['ob1_count']}")
    lines.append(f"  OB.50 snapshots:  {dq['ob50_count']}")
    lines.append(f"  OB.200 snapshots: {dq['ob200_count']}")
    lines.append(f"  Trades:           {dq['trade_count']}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Settlement Predictor - Example Usage\n")
    
    predictor = SettlementPredictor(window_seconds=10.0)
    
    # Simulate some data
    current_time = time.time()
    
    # Add orderbook snapshots
    for i in range(100):
        t = current_time - 10 + i * 0.1
        bids = [(100.0 - i * 0.01, 100 + i) for i in range(10)]
        asks = [(100.0 + 0.5 + i * 0.01, 100 - i) for i in range(10)]
        predictor.add_orderbook_snapshot(t, bids, asks, depth_level=1)
    
    # Add some trades
    for i in range(50):
        t = current_time - 10 + i * 0.2
        predictor.add_trade(t, 100.0 + np.random.randn() * 0.1, 10, "Buy" if i % 2 == 0 else "Sell")
    
    # Get signal
    signal = predictor.get_signal(current_time)
    print(format_signal(signal))
