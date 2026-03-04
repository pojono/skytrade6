"""
Kimi-1 Quick Edge Detection - Optimized for speed
"""
import pandas as pd
import numpy as np
from framework import DataLoader, BacktestEngine
from strategies import FundingRateHoldStrategy
import warnings
warnings.filterwarnings('ignore')

FEE_MAKER = 0.0004
FEE_TAKER = 0.001
ROUND_TRIP_TAKER = 0.002

def quick_test(symbol, exchange, start, end, entry_bps, exit_bps, max_hold):
    """Quick test of a single configuration."""
    loader = DataLoader()
    
    klines = loader.load_klines(exchange, symbol, start, end)
    funding = loader.load_funding_rates(exchange, symbol, start, end)
    
    if len(klines) < 1000 or len(funding) < 10:
        return None
    
    strategy = FundingRateHoldStrategy(entry_bps, exit_bps, max_hold)
    data = {'klines': klines, 'funding': funding}
    
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(strategy, data, position_size=1.0)
    
    if result.total_trades < 5:
        return None
    
    net_pnl = sum(t.pnl_net for t in result.trades)
    total_fees = sum(t.fees for t in result.trades)
    
    return {
        'symbol': symbol,
        'entry': entry_bps,
        'exit': exit_bps,
        'hold': max_hold,
        'trades': result.total_trades,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'net_pnl': net_pnl,
        'fees': total_fees,
        'edge_after_fees': net_pnl - total_fees,
    }

if __name__ == '__main__':
    print("=" * 80)
    print("KIMI-1 QUICK EDGE DETECTION")
    print("=" * 80)
    
    # Focus on top liquid symbols with shorter date range
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    exchange = 'bybit'
    start = '2025-10-01'
    end = '2025-12-31'
    
    # Test realistic thresholds for 8h funding
    configs = [
        (0.5, 0.1, 3),
        (1.0, 0.3, 3),
        (1.5, 0.5, 3),
        (2.0, 0.5, 3),
        (1.0, 0.3, 2),
        (1.5, 0.5, 2),
    ]
    
    results = []
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        for entry_bps, exit_bps, max_hold in configs:
            result = quick_test(symbol, exchange, start, end, entry_bps, exit_bps, max_hold)
            if result:
                results.append(result)
                print(f"  E={entry_bps:.1f} X={exit_bps:.1f} H={max_hold}: "
                      f"{result['trades']} trades, WR={result['win_rate']:.1%}, "
                      f"Net=${result['net_pnl']:.2f}, Edge=${result['edge_after_fees']:.2f}")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if results:
        df = pd.DataFrame(results)
        
        # Show profitable configs (edge after fees > 0)
        profitable = df[df['edge_after_fees'] > 0].sort_values('edge_after_fees', ascending=False)
        
        if not profitable.empty:
            print(f"\nProfitable configs (edge after fees): {len(profitable)}")
            print("\nTop 10:")
            for idx, row in profitable.head(10).iterrows():
                print(f"  {row['symbol']:10s} | E={row['entry']:4.1f} X={row['exit']:4.1f} | "
                      f"Trades={row['trades']:3d} | WR={row['win_rate']:5.1%} | "
                      f"PF={row['profit_factor']:5.2f} | Edge=${row['edge_after_fees']:7.2f}")
        else:
            print("\nNo profitable configurations found.")
            print("8h funding rates may not provide sufficient edge alone.")
            print("\nRecommendations:")
            print("  1. Use 1h funding rate coins for higher volatility")
            print("  2. Cross-exchange arbitrage (FR differential)")
            print("  3. Combine FR with price momentum signals")
            print("  4. Post-settlement scalp (requires tick data)")
        
        df.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/quick_results.csv', index=False)
    else:
        print("No results generated.")
