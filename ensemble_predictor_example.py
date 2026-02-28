#!/usr/bin/env python3
"""
Ensemble Settlement Predictor - Combining Rule-Based + ML
==========================================================
Combines the rule-based predictor with ML models for optimal predictions.

Usage:
    from ensemble_predictor_example import EnsemblePredictor
    
    predictor = EnsemblePredictor()
    
    # Feed live orderbook data
    predictor.add_orderbook_snapshot(timestamp, bids, asks, depth_level)
    predictor.add_trade(timestamp, price, qty, side)
    
    # Get ensemble prediction
    signal = predictor.get_ensemble_signal()
"""

from typing import Dict, Optional
import numpy as np
from settlement_predictor import SettlementPredictor
from ml_settlement_predictor import MLSettlementPredictor


class EnsemblePredictor:
    """Combines rule-based and ML predictions."""
    
    def __init__(self, ml_model_dir: str = 'ml_models/'):
        # Initialize both predictors
        self.rule_based = SettlementPredictor(window_seconds=10.0)
        self.ml_predictor = MLSettlementPredictor()
        
        # Try to load ML models
        try:
            self.ml_predictor.load_models(ml_model_dir)
            self.ml_available = True
            print(f"✓ ML models loaded from {ml_model_dir}")
        except Exception as e:
            self.ml_available = False
            print(f"⚠ ML models not available: {e}")
            print("  Falling back to rule-based predictor only")
    
    def add_orderbook_snapshot(self, timestamp, bids, asks, depth_level):
        """Add orderbook snapshot to rule-based predictor."""
        self.rule_based.add_orderbook_snapshot(timestamp, bids, asks, depth_level)
    
    def add_trade(self, timestamp, price, qty, side):
        """Add trade to rule-based predictor."""
        self.rule_based.add_trade(timestamp, price, qty, side)
    
    def get_ensemble_signal(self, current_time: Optional[float] = None) -> Dict:
        """Get ensemble prediction combining rule-based and ML.
        
        Returns:
            {
                'should_trade': bool,
                'position_size_multiplier': float,
                'exit_time_ms': int,
                'expected_drop_bps': float,
                'confidence': float,
                
                # ML predictions (if available)
                'ml_predictions': {
                    'time_to_bottom_ms': float,
                    'sell_volume_usd': float,
                    'price_100ms_bps': float,
                    'price_500ms_bps': float,
                    'price_1s_bps': float,
                    'price_5s_bps': float,
                },
                
                # Rule-based predictions
                'rule_based': {...},
                
                # Reasoning
                'reasons': [...],
                'strategy': str,
            }
        """
        # Get rule-based signal
        rule_signal = self.rule_based.get_signal(current_time)
        
        # Get features for ML
        features = self.rule_based.calculate_features(current_time)
        
        # Initialize ensemble signal
        ensemble = {
            'rule_based': rule_signal,
            'ml_predictions': None,
            'reasons': [],
            'strategy': 'rule_based_only',
        }
        
        # Try ML prediction if available and we have enough features
        ml_predictions = None
        if self.ml_available:
            try:
                # Convert features to dict format expected by ML
                feature_dict = {
                    'spread_mean_bps': features.get('spread_mean_bps'),
                    'spread_std_bps': features.get('spread_std_bps'),
                    'spread_max_bps': features.get('spread_max_bps'),
                    'qty_imb_mean': features.get('qty_imb_mean'),
                    'depth_imb_mean': features.get('depth_imb_mean'),
                    'bid10_mean_usd': features.get('bid10_mean_usd'),
                    'ask10_mean_usd': features.get('ask10_mean_usd'),
                    'total_depth_imb_mean': features.get('total_depth_imb_mean'),
                    'total_bid_mean_usd': features.get('total_bid_mean_usd'),
                    'total_ask_mean_usd': features.get('total_ask_mean_usd'),
                    'trade_flow_imb': features.get('trade_flow_imb'),
                    'pre_trade_count': features['data_quality']['trade_count'],
                    'pre_total_vol_usd': 0,  # Would need to calculate from trades
                    'pre_avg_trade_size_usd': 0,  # Would need to calculate
                    'pre_price_vol_bps': features.get('price_vol_bps'),
                }
                
                ml_predictions = self.ml_predictor.predict(feature_dict)
                ensemble['ml_predictions'] = ml_predictions
                ensemble['strategy'] = 'ensemble'
                
            except Exception as e:
                ensemble['reasons'].append(f"⚠ ML prediction failed: {e}")
        
        # Combine predictions
        if ml_predictions and ensemble['strategy'] == 'ensemble':
            # Ensemble logic
            rule_conf = rule_signal['confidence'] / 100
            ml_conf = ml_predictions['confidence']
            
            # Weighted average of expected drops
            rule_drop = rule_signal['expected_drop_bps']
            ml_drop = ml_predictions.get('price_500ms_bps', rule_drop)
            
            ensemble['expected_drop_bps'] = (
                rule_drop * rule_conf + ml_drop * ml_conf
            ) / (rule_conf + ml_conf)
            
            # Ensemble confidence
            ensemble['confidence'] = (rule_conf + ml_conf) / 2
            
            # Position sizing based on ensemble confidence
            if ensemble['confidence'] >= 0.75:
                ensemble['position_size_multiplier'] = 2.0
            elif ensemble['confidence'] >= 0.65:
                ensemble['position_size_multiplier'] = 1.5
            elif ensemble['confidence'] >= 0.50:
                ensemble['position_size_multiplier'] = 1.0
            elif ensemble['confidence'] >= 0.40:
                ensemble['position_size_multiplier'] = 0.5
            else:
                ensemble['position_size_multiplier'] = 0.0
            
            # Exit timing from ML prediction
            time_to_bottom = ml_predictions.get('time_to_bottom_ms', 1000)
            if time_to_bottom < 500:
                ensemble['exit_time_ms'] = 500
            elif time_to_bottom < 1500:
                ensemble['exit_time_ms'] = int(time_to_bottom + 200)
            else:
                ensemble['exit_time_ms'] = 2000
            
            # Trading decision
            ensemble['should_trade'] = (
                ensemble['confidence'] >= 0.40 and
                abs(ensemble['expected_drop_bps']) > 40  # Must beat fees
            )
            
            # Reasoning
            ensemble['reasons'].append(
                f"Ensemble confidence: {ensemble['confidence']:.1%} "
                f"(rule: {rule_conf:.1%}, ML: {ml_conf:.1%})"
            )
            ensemble['reasons'].append(
                f"Expected drop: {ensemble['expected_drop_bps']:.1f} bps "
                f"(rule: {rule_drop:.1f}, ML: {ml_drop:.1f})"
            )
            ensemble['reasons'].append(
                f"Exit timing: T+{ensemble['exit_time_ms']}ms "
                f"(predicted bottom: {time_to_bottom:.0f}ms)"
            )
            
            # Add ML-specific insights
            if ml_predictions.get('sell_volume_usd'):
                vol = ml_predictions['sell_volume_usd']
                ensemble['reasons'].append(f"Predicted sell volume: ${vol:,.0f}")
            
            # Multi-horizon analysis
            p100 = ml_predictions.get('price_100ms_bps', 0)
            p500 = ml_predictions.get('price_500ms_bps', 0)
            p1s = ml_predictions.get('price_1s_bps', 0)
            p5s = ml_predictions.get('price_5s_bps', 0)
            
            if p100 and p500 and p1s and p5s:
                recovery = p5s - p1s
                ensemble['reasons'].append(
                    f"Price trajectory: {p100:.0f} → {p500:.0f} → {p1s:.0f} → {p5s:.0f} bps"
                )
                if recovery > 20:
                    ensemble['reasons'].append(
                        f"⚠ Strong recovery expected (+{recovery:.0f} bps) - exit early!"
                    )
        
        else:
            # Fall back to rule-based only
            ensemble['confidence'] = rule_signal['confidence'] / 100
            ensemble['expected_drop_bps'] = rule_signal['expected_drop_bps']
            ensemble['position_size_multiplier'] = rule_signal['position_size_multiplier']
            ensemble['should_trade'] = rule_signal['should_trade']
            ensemble['exit_time_ms'] = 1000  # Default exit
            ensemble['reasons'] = rule_signal['reasons']
        
        return ensemble
    
    def format_signal(self, signal: Dict) -> str:
        """Format ensemble signal for display."""
        lines = []
        lines.append("=" * 80)
        lines.append("ENSEMBLE SETTLEMENT PREDICTION")
        lines.append("=" * 80)
        lines.append(f"Strategy:         {signal['strategy']}")
        lines.append(f"Confidence:       {signal['confidence']:.1%}")
        lines.append(f"Expected Drop:    {signal['expected_drop_bps']:.1f} bps")
        lines.append(f"Position Size:    {signal['position_size_multiplier']:.1f}x")
        lines.append(f"Exit Timing:      T+{signal.get('exit_time_ms', 'N/A')}ms")
        lines.append(f"Trade Decision:   {'✅ TRADE' if signal['should_trade'] else '❌ SKIP'}")
        lines.append("")
        
        if signal['reasons']:
            lines.append("Reasoning:")
            for reason in signal['reasons']:
                lines.append(f"  • {reason}")
            lines.append("")
        
        if signal.get('ml_predictions'):
            lines.append("ML Predictions:")
            ml = signal['ml_predictions']
            lines.append(f"  Time to bottom:   {ml.get('time_to_bottom_ms', 'N/A'):.0f} ms")
            lines.append(f"  Sell volume:      ${ml.get('sell_volume_usd', 0):,.0f}")
            lines.append(f"  Price @ T+100ms:  {ml.get('price_100ms_bps', 0):+.1f} bps")
            lines.append(f"  Price @ T+500ms:  {ml.get('price_500ms_bps', 0):+.1f} bps")
            lines.append(f"  Price @ T+1s:     {ml.get('price_1s_bps', 0):+.1f} bps")
            lines.append(f"  Price @ T+5s:     {ml.get('price_5s_bps', 0):+.1f} bps")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def reset(self):
        """Reset both predictors."""
        self.rule_based.reset()


# Example usage
if __name__ == "__main__":
    import time
    
    print("Ensemble Predictor - Example\n")
    
    # Initialize ensemble predictor
    predictor = EnsemblePredictor(ml_model_dir='ml_models/')
    
    # Simulate some orderbook data
    current_time = time.time()
    
    print("Feeding simulated data...")
    
    # Add orderbook snapshots
    for i in range(100):
        t = current_time - 10 + i * 0.1
        bids = [(100.0 - i * 0.01, 100 + i) for i in range(50)]
        asks = [(100.0 + 0.5 + i * 0.01, 100 - i) for i in range(50)]
        
        # Add all 3 depths
        predictor.add_orderbook_snapshot(t, bids[:1], asks[:1], depth_level=1)
        predictor.add_orderbook_snapshot(t, bids[:50], asks[:50], depth_level=50)
        predictor.add_orderbook_snapshot(t, bids, asks, depth_level=200)
    
    # Add trades
    for i in range(100):
        t = current_time - 10 + i * 0.1
        predictor.add_trade(
            t,
            100.0 + np.random.randn() * 0.1,
            10 + np.random.rand() * 5,
            "Buy" if i % 2 == 0 else "Sell"
        )
    
    print("Getting ensemble prediction...\n")
    
    # Get ensemble signal
    signal = predictor.get_ensemble_signal(current_time)
    
    # Display
    print(predictor.format_signal(signal))
    
    # Show decision logic
    print("\nTrading Decision Logic:")
    if signal['should_trade']:
        print(f"  ✅ EXECUTE TRADE")
        print(f"  Position size: ${1000 * signal['position_size_multiplier']:.0f}")
        print(f"  Entry: T+0ms (market order)")
        print(f"  Exit: T+{signal['exit_time_ms']}ms (market order)")
        print(f"  Expected profit: {abs(signal['expected_drop_bps']) - 20:.1f} bps (after fees)")
    else:
        print(f"  ❌ SKIP TRADE")
        print(f"  Reason: Confidence too low ({signal['confidence']:.1%})")
