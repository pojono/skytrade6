#!/usr/bin/env python3
"""
Analyze Post-Settlement Sell Wave Predictability
=================================================
Research question: Can we predict the magnitude and timing of post-settlement
sell waves using pre-settlement orderbook state and FR magnitude?

For each settlement JSONL file, extract:

PRE-SETTLEMENT FEATURES (T-10s to T-0):
- Orderbook imbalance at multiple depths (1, 10, 50 levels)
- Orderbook depth/liquidity (bid/ask notional)
- Spread dynamics (mean, std, trend)
- Trade flow imbalance (buy vs sell volume)
- FR magnitude (from ticker)
- Price volatility

POST-SETTLEMENT TARGET (T+0 to T+5s):
- Price drop magnitude (bps)
- Time to bottom (ms)
- Sell wave intensity (volume, trade count)
- Recovery speed

Output: CSV with one row per settlement for correlation analysis.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime


def extract_features(filepath: Path):
    """Extract pre/post settlement features from a JSONL file."""
    
    symbol = filepath.stem.split("_")[0]
    settle_label = "_".join(filepath.stem.split("_")[1:])
    
    # Parse settlement time from filename
    try:
        dt_str = settle_label  # e.g., "20260227_120000"
        settle_dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    except:
        print(f"  ⚠ {symbol}: Cannot parse settlement time from {settle_label}")
        return None
    
    trades = []
    orderbooks_1 = []   # orderbook.1
    orderbooks_50 = []  # orderbook.50
    orderbooks_200 = [] # orderbook.200
    tickers = []
    
    with open(filepath) as f:
        for line in f:
            m = json.loads(line)
            t_ms = m["_t_ms"]  # milliseconds relative to settlement
            topic = m.get("topic", "")
            
            if topic.startswith("publicTrade"):
                for tr in m.get("data", []):
                    trades.append({
                        "t_ms": t_ms,
                        "price": float(tr["p"]),
                        "qty": float(tr["v"]),
                        "side": tr["S"],
                    })
            
            elif topic.startswith("orderbook.1."):
                d = m.get("data", {})
                bids = [(float(p), float(q)) for p, q in d.get("b", [])]
                asks = [(float(p), float(q)) for p, q in d.get("a", [])]
                if bids and asks:
                    orderbooks_1.append({
                        "t_ms": t_ms,
                        "bid": bids[0][0],
                        "ask": asks[0][0],
                        "bid_qty": bids[0][1],
                        "ask_qty": asks[0][1],
                    })
            
            elif topic.startswith("orderbook.50."):
                d = m.get("data", {})
                bids = [(float(p), float(q)) for p, q in d.get("b", [])]
                asks = [(float(p), float(q)) for p, q in d.get("a", [])]
                if bids and asks:
                    bid_notional = sum(p * q for p, q in bids[:10])
                    ask_notional = sum(p * q for p, q in asks[:10])
                    orderbooks_50.append({
                        "t_ms": t_ms,
                        "bid10_notional": bid_notional,
                        "ask10_notional": ask_notional,
                        "bid_levels": len(bids),
                        "ask_levels": len(asks),
                    })
            
            elif topic.startswith("orderbook.200."):
                d = m.get("data", {})
                bids = [(float(p), float(q)) for p, q in d.get("b", [])]
                asks = [(float(p), float(q)) for p, q in d.get("a", [])]
                if bids and asks:
                    total_bid = sum(p * q for p, q in bids)
                    total_ask = sum(p * q for p, q in asks)
                    orderbooks_200.append({
                        "t_ms": t_ms,
                        "total_bid_notional": total_bid,
                        "total_ask_notional": total_ask,
                    })
            
            elif topic.startswith("tickers"):
                d = m.get("data", {})
                fr = d.get("fundingRate")
                if fr is not None:
                    tickers.append({
                        "t_ms": t_ms,
                        "fr": float(fr),
                        "last_price": float(d.get("lastPrice", 0)),
                    })
    
    if not trades:
        print(f"  ⚠ {symbol}: No trades found")
        return None
    
    # Reference price = last trade before settlement
    pre_trades = [t for t in trades if t["t_ms"] < 0]
    if not pre_trades:
        print(f"  ⚠ {symbol}: No pre-settlement trades")
        return None
    
    ref_price = pre_trades[-1]["price"]
    
    def to_bps(p):
        return (p - ref_price) / ref_price * 10000
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRE-SETTLEMENT FEATURES (T-10s to T-0)
    # ═══════════════════════════════════════════════════════════════════════
    
    pre_window_start = -10000  # -10s
    pre_window_end = 0
    
    # --- Orderbook features ---
    pre_ob1 = [ob for ob in orderbooks_1 if pre_window_start <= ob["t_ms"] < pre_window_end]
    pre_ob50 = [ob for ob in orderbooks_50 if pre_window_start <= ob["t_ms"] < pre_window_end]
    pre_ob200 = [ob for ob in orderbooks_200 if pre_window_start <= ob["t_ms"] < pre_window_end]
    
    # Spread (from orderbook.1)
    if pre_ob1:
        spreads_bps = [to_bps(ob["ask"]) - to_bps(ob["bid"]) for ob in pre_ob1]
        spread_mean = np.mean(spreads_bps)
        spread_std = np.std(spreads_bps)
        spread_max = np.max(spreads_bps)
        
        # Bid/ask qty imbalance
        qty_imbalances = [(ob["bid_qty"] - ob["ask_qty"]) / (ob["bid_qty"] + ob["ask_qty"]) 
                          for ob in pre_ob1 if (ob["bid_qty"] + ob["ask_qty"]) > 0]
        qty_imb_mean = np.mean(qty_imbalances) if qty_imbalances else 0
    else:
        spread_mean = spread_std = spread_max = qty_imb_mean = None
    
    # Depth imbalance (from orderbook.50)
    if pre_ob50:
        depth_imbalances = [(ob["bid10_notional"] - ob["ask10_notional"]) / 
                            (ob["bid10_notional"] + ob["ask10_notional"])
                            for ob in pre_ob50 if (ob["bid10_notional"] + ob["ask10_notional"]) > 0]
        depth_imb_mean = np.mean(depth_imbalances) if depth_imbalances else 0
        bid10_mean = np.mean([ob["bid10_notional"] for ob in pre_ob50])
        ask10_mean = np.mean([ob["ask10_notional"] for ob in pre_ob50])
    else:
        depth_imb_mean = bid10_mean = ask10_mean = None
    
    # Total depth (from orderbook.200)
    if pre_ob200:
        total_depth_imbalances = [(ob["total_bid_notional"] - ob["total_ask_notional"]) / 
                                  (ob["total_bid_notional"] + ob["total_ask_notional"])
                                  for ob in pre_ob200 if (ob["total_bid_notional"] + ob["total_ask_notional"]) > 0]
        total_depth_imb_mean = np.mean(total_depth_imbalances) if total_depth_imbalances else 0
        total_bid_mean = np.mean([ob["total_bid_notional"] for ob in pre_ob200])
        total_ask_mean = np.mean([ob["total_ask_notional"] for ob in pre_ob200])
    else:
        total_depth_imb_mean = total_bid_mean = total_ask_mean = None
    
    # --- Trade flow features ---
    pre_trades_window = [t for t in trades if pre_window_start <= t["t_ms"] < pre_window_end]
    
    if pre_trades_window:
        buy_vol = sum(t["price"] * t["qty"] for t in pre_trades_window if t["side"] == "Buy")
        sell_vol = sum(t["price"] * t["qty"] for t in pre_trades_window if t["side"] == "Sell")
        total_vol = buy_vol + sell_vol
        
        trade_flow_imb = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
        trade_count = len(pre_trades_window)
        avg_trade_size = total_vol / trade_count if trade_count > 0 else 0
        
        # Price volatility (std of price changes)
        prices = [t["price"] for t in sorted(pre_trades_window, key=lambda x: x["t_ms"])]
        price_changes_bps = [to_bps(prices[i]) - to_bps(prices[i-1]) for i in range(1, len(prices))]
        price_vol = np.std(price_changes_bps) if len(price_changes_bps) > 1 else 0
    else:
        trade_flow_imb = trade_count = avg_trade_size = price_vol = 0
        buy_vol = sell_vol = total_vol = 0
    
    # --- FR magnitude ---
    # Use the most recent FR value before settlement (expand window to capture it)
    pre_tickers = [t for t in tickers if t["t_ms"] < 0]  # Any time before settlement
    if pre_tickers:
        # Use the most recent FR value
        most_recent = max(pre_tickers, key=lambda x: x["t_ms"])
        fr_bps = most_recent["fr"] * 10000
        fr_abs_bps = abs(fr_bps)
    else:
        fr_bps = fr_abs_bps = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # POST-SETTLEMENT TARGET (T+0 to T+5s)
    # ═══════════════════════════════════════════════════════════════════════
    
    post_window_start = 0
    post_window_end = 5000  # +5s
    
    post_trades_window = [t for t in trades if post_window_start <= t["t_ms"] <= post_window_end]
    
    if not post_trades_window:
        print(f"  ⚠ {symbol}: No post-settlement trades")
        return None
    
    # Price drop magnitude
    post_prices_bps = [to_bps(t["price"]) for t in sorted(post_trades_window, key=lambda x: x["t_ms"])]
    min_price_bps = min(post_prices_bps)
    max_price_bps = max(post_prices_bps)
    final_price_bps = post_prices_bps[-1]
    
    # Time to bottom
    min_idx = post_prices_bps.index(min_price_bps)
    time_to_bottom_ms = sorted(post_trades_window, key=lambda x: x["t_ms"])[min_idx]["t_ms"]
    
    # Sell wave intensity
    post_buy_vol = sum(t["price"] * t["qty"] for t in post_trades_window if t["side"] == "Buy")
    post_sell_vol = sum(t["price"] * t["qty"] for t in post_trades_window if t["side"] == "Sell")
    post_total_vol = post_buy_vol + post_sell_vol
    post_sell_ratio = post_sell_vol / post_total_vol if post_total_vol > 0 else 0
    
    post_sell_count = sum(1 for t in post_trades_window if t["side"] == "Sell")
    post_buy_count = sum(1 for t in post_trades_window if t["side"] == "Buy")
    
    # Recovery (from bottom to end)
    recovery_bps = final_price_bps - min_price_bps
    
    # First 100ms drop (immediate impact)
    first_100ms = [t for t in post_trades_window if t["t_ms"] <= 100]
    if first_100ms:
        first_100ms_prices = [to_bps(t["price"]) for t in sorted(first_100ms, key=lambda x: x["t_ms"])]
        drop_100ms = min(first_100ms_prices)
    else:
        drop_100ms = 0
    
    # First 500ms drop
    first_500ms = [t for t in post_trades_window if t["t_ms"] <= 500]
    if first_500ms:
        first_500ms_prices = [to_bps(t["price"]) for t in sorted(first_500ms, key=lambda x: x["t_ms"])]
        drop_500ms = min(first_500ms_prices)
    else:
        drop_500ms = 0
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPILE RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    
    result = {
        # Metadata
        "symbol": symbol,
        "settle_time": settle_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "ref_price": ref_price,
        
        # PRE-SETTLEMENT FEATURES
        "fr_bps": fr_bps,
        "fr_abs_bps": fr_abs_bps,
        
        # Orderbook.1 features
        "spread_mean_bps": spread_mean,
        "spread_std_bps": spread_std,
        "spread_max_bps": spread_max,
        "qty_imb_mean": qty_imb_mean,
        
        # Orderbook.50 features
        "depth_imb_mean": depth_imb_mean,
        "bid10_mean_usd": bid10_mean,
        "ask10_mean_usd": ask10_mean,
        
        # Orderbook.200 features
        "total_depth_imb_mean": total_depth_imb_mean,
        "total_bid_mean_usd": total_bid_mean,
        "total_ask_mean_usd": total_ask_mean,
        
        # Trade flow features
        "trade_flow_imb": trade_flow_imb,
        "pre_trade_count": trade_count,
        "pre_total_vol_usd": total_vol,
        "pre_avg_trade_size_usd": avg_trade_size,
        "pre_price_vol_bps": price_vol,
        
        # POST-SETTLEMENT TARGETS
        "drop_min_bps": min_price_bps,
        "drop_max_bps": max_price_bps,
        "drop_final_bps": final_price_bps,
        "time_to_bottom_ms": time_to_bottom_ms,
        "drop_100ms_bps": drop_100ms,
        "drop_500ms_bps": drop_500ms,
        "recovery_bps": recovery_bps,
        
        # Sell wave intensity
        "post_sell_ratio": post_sell_ratio,
        "post_sell_vol_usd": post_sell_vol,
        "post_buy_vol_usd": post_buy_vol,
        "post_sell_count": post_sell_count,
        "post_buy_count": post_buy_count,
        "post_total_vol_usd": post_total_vol,
        
        # Data quality
        "has_ob1": len(pre_ob1) > 0,
        "has_ob50": len(pre_ob50) > 0,
        "has_ob200": len(pre_ob200) > 0,
        "pre_trade_count_total": len(pre_trades_window),
        "post_trade_count_total": len(post_trades_window),
    }
    
    fr_str = f"{fr_bps:+.1f}bps" if fr_bps is not None else "N/A"
    print(f"  ✓ {symbol} {settle_label}: FR={fr_str}, drop={min_price_bps:.1f}bps @ T+{time_to_bottom_ms:.0f}ms")
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyse_settlement_predictability.py <file1.jsonl> [file2.jsonl ...]")
        print("\nExample:")
        print("  python3 analyse_settlement_predictability.py charts_settlement/*_20260227_*.jsonl")
        sys.exit(1)
    
    files = [Path(f) for f in sys.argv[1:]]
    print(f"Analyzing {len(files)} settlement file(s)...\n")
    
    results = []
    for fp in files:
        if not fp.exists():
            print(f"  ⚠ File not found: {fp}")
            continue
        
        result = extract_features(fp)
        if result:
            results.append(result)
    
    if not results:
        print("\n⚠ No valid results extracted")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    out_file = Path("settlement_predictability_analysis.csv")
    df.to_csv(out_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SAVED: {out_file}")
    print(f"{'='*80}")
    print(f"Total settlements analyzed: {len(df)}")
    print(f"Symbols: {df['symbol'].nunique()}")
    print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()
    
    # FR distribution
    print("Funding Rate Distribution:")
    print(f"  Mean: {df['fr_bps'].mean():+.1f} bps")
    print(f"  Median: {df['fr_bps'].median():+.1f} bps")
    print(f"  Min: {df['fr_bps'].min():+.1f} bps")
    print(f"  Max: {df['fr_bps'].max():+.1f} bps")
    print()
    
    # Price drop distribution
    print("Price Drop Distribution (minimum reached):")
    print(f"  Mean: {df['drop_min_bps'].mean():.1f} bps")
    print(f"  Median: {df['drop_min_bps'].median():.1f} bps")
    print(f"  Min: {df['drop_min_bps'].min():.1f} bps")
    print(f"  Max: {df['drop_min_bps'].max():.1f} bps")
    print()
    
    print("Time to Bottom:")
    print(f"  Mean: {df['time_to_bottom_ms'].mean():.0f} ms")
    print(f"  Median: {df['time_to_bottom_ms'].median():.0f} ms")
    print(f"  Min: {df['time_to_bottom_ms'].min():.0f} ms")
    print(f"  Max: {df['time_to_bottom_ms'].max():.0f} ms")
    print()
    
    # Correlations with price drop
    print("="*80)
    print("CORRELATIONS WITH PRICE DROP (drop_min_bps)")
    print("="*80)
    print()
    
    corr_cols = [
        "fr_bps", "fr_abs_bps",
        "spread_mean_bps", "spread_std_bps", "spread_max_bps",
        "qty_imb_mean", "depth_imb_mean", "total_depth_imb_mean",
        "trade_flow_imb", "pre_price_vol_bps",
        "bid10_mean_usd", "ask10_mean_usd",
        "total_bid_mean_usd", "total_ask_mean_usd",
    ]
    
    correlations = []
    for col in corr_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            corr = df[['drop_min_bps', col]].corr().iloc[0, 1]
            correlations.append((col, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for col, corr in correlations:
        print(f"  {col:30s}: {corr:+.3f}")
    
    print()
    
    # Sell wave intensity correlation
    print("="*80)
    print("CORRELATIONS WITH SELL WAVE INTENSITY (post_sell_ratio)")
    print("="*80)
    print()
    
    correlations_sell = []
    for col in corr_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            corr = df[['post_sell_ratio', col]].corr().iloc[0, 1]
            correlations_sell.append((col, corr))
    
    correlations_sell.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for col, corr in correlations_sell:
        print(f"  {col:30s}: {corr:+.3f}")
    
    print()
    
    # Data quality
    print("="*80)
    print("DATA QUALITY")
    print("="*80)
    print(f"  Files with orderbook.1:   {df['has_ob1'].sum()}/{len(df)}")
    print(f"  Files with orderbook.50:  {df['has_ob50'].sum()}/{len(df)}")
    print(f"  Files with orderbook.200: {df['has_ob200'].sum()}/{len(df)}")
    print()
    
    print(f"Analysis complete. Results saved to: {out_file.resolve()}")
    print()


if __name__ == "__main__":
    main()
