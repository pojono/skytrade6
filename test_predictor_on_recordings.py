#!/usr/bin/env python3
"""
Test Settlement Predictor on Recorded JSONL Files
==================================================
Replays recorded market data through the predictor to validate signals.

Usage:
    python3 test_predictor_on_recordings.py charts_settlement/SAHARAUSDT_20260227_120000.jsonl
"""

import json
import sys
from pathlib import Path
from settlement_predictor import SettlementPredictor, format_signal


def replay_jsonl(filepath: Path):
    """Replay a JSONL recording through the predictor."""
    
    symbol = filepath.stem.split("_")[0]
    settle_label = "_".join(filepath.stem.split("_")[1:])
    
    print(f"\n{'='*80}")
    print(f"REPLAYING: {symbol} @ {settle_label}")
    print(f"{'='*80}\n")
    
    predictor = SettlementPredictor(window_seconds=10.0)
    
    # Parse messages and sort by timestamp
    messages = []
    with open(filepath) as f:
        for line in f:
            msg = json.loads(line)
            messages.append(msg)
    
    # Sort by _t_ms (relative to settlement)
    messages.sort(key=lambda m: m["_t_ms"])
    
    # Find settlement time (T=0)
    settlement_time = None
    for msg in messages:
        if msg["_t_ms"] >= 0:
            settlement_time = msg["_recv_us"] / 1_000_000 - msg["_t_ms"] / 1000
            break
    
    if settlement_time is None:
        print("⚠ Could not determine settlement time")
        return
    
    print(f"Settlement time: {settlement_time:.3f}")
    print(f"Processing {len(messages)} messages...\n")
    
    # Feed data to predictor
    trade_count = 0
    ob1_count = 0
    ob50_count = 0
    ob200_count = 0
    
    for msg in messages:
        t_sec = msg["_recv_us"] / 1_000_000
        topic = msg.get("topic", "")
        
        # Only process pre-settlement data
        if msg["_t_ms"] > 0:
            break
        
        if topic.startswith("publicTrade"):
            for tr in msg.get("data", []):
                predictor.add_trade(
                    t_sec,
                    float(tr["p"]),
                    float(tr["v"]),
                    tr["S"]
                )
                trade_count += 1
        
        elif topic.startswith("orderbook.1."):
            d = msg.get("data", {})
            bids = [(float(p), float(q)) for p, q in d.get("b", [])]
            asks = [(float(p), float(q)) for p, q in d.get("a", [])]
            if bids and asks:
                predictor.add_orderbook_snapshot(t_sec, bids, asks, depth_level=1)
                ob1_count += 1
        
        elif topic.startswith("orderbook.50."):
            d = msg.get("data", {})
            bids = [(float(p), float(q)) for p, q in d.get("b", [])]
            asks = [(float(p), float(q)) for p, q in d.get("a", [])]
            if bids and asks:
                predictor.add_orderbook_snapshot(t_sec, bids, asks, depth_level=50)
                ob50_count += 1
        
        elif topic.startswith("orderbook.200."):
            d = msg.get("data", {})
            bids = [(float(p), float(q)) for p, q in d.get("b", [])]
            asks = [(float(p), float(q)) for p, q in d.get("a", [])]
            if bids and asks:
                predictor.add_orderbook_snapshot(t_sec, bids, asks, depth_level=200)
                ob200_count += 1
    
    print(f"Data ingested:")
    print(f"  Trades: {trade_count}")
    print(f"  OB.1: {ob1_count}")
    print(f"  OB.50: {ob50_count}")
    print(f"  OB.200: {ob200_count}")
    print()
    
    # Generate signal at T-1s
    signal_time = settlement_time - 1.0
    signal = predictor.get_signal(signal_time)
    
    print(format_signal(signal))
    
    # Calculate actual outcome
    print("\n" + "="*80)
    print("ACTUAL OUTCOME (from recording)")
    print("="*80)
    
    post_trades = [msg for msg in messages if msg["_t_ms"] >= 0 and msg["_t_ms"] <= 5000 
                   and msg.get("topic", "").startswith("publicTrade")]
    
    if post_trades:
        all_prices = []
        for msg in post_trades:
            for tr in msg.get("data", []):
                all_prices.append((msg["_t_ms"], float(tr["p"])))
        
        if all_prices and predictor.ref_price:
            all_prices.sort()
            min_price = min(p for _, p in all_prices)
            min_time = next(t for t, p in all_prices if p == min_price)
            
            drop_bps = (min_price - predictor.ref_price) / predictor.ref_price * 10000
            
            print(f"Actual drop:      {drop_bps:.1f} bps")
            print(f"Time to bottom:   {min_time:.0f} ms")
            print(f"Predicted drop:   {signal['expected_drop_bps']:.1f} bps")
            print(f"Error:            {abs(drop_bps - signal['expected_drop_bps']):.1f} bps")
            print()
            
            # Evaluate prediction
            if signal['should_trade']:
                if abs(drop_bps) > 40:  # Profitable threshold (> 20 bps fees)
                    print("✅ CORRECT: Predicted trade, actual drop was profitable")
                else:
                    print("❌ FALSE POSITIVE: Predicted trade, but drop was too small")
            else:
                if abs(drop_bps) > 40:
                    print("❌ FALSE NEGATIVE: Skipped trade, but drop was profitable")
                else:
                    print("✅ CORRECT: Skipped trade, drop was too small")
    else:
        print("⚠ No post-settlement trades found")
    
    print("="*80)
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_predictor_on_recordings.py <file1.jsonl> [file2.jsonl ...]")
        print("\nExample:")
        print("  python3 test_predictor_on_recordings.py charts_settlement/SAHARAUSDT_20260227_120000.jsonl")
        print("  python3 test_predictor_on_recordings.py charts_settlement/*_20260227_1*.jsonl")
        sys.exit(1)
    
    files = [Path(f) for f in sys.argv[1:]]
    
    print(f"\n{'='*80}")
    print(f"TESTING PREDICTOR ON {len(files)} RECORDING(S)")
    print(f"{'='*80}")
    
    results = []
    
    for fp in files:
        if not fp.exists():
            print(f"\n⚠ File not found: {fp}")
            continue
        
        try:
            replay_jsonl(fp)
        except Exception as e:
            print(f"\n❌ Error processing {fp.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
