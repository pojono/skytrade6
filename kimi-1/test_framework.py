"""
Quick validation test for kimi-1 framework.
Test single symbol with funding rate strategy.
"""
import pandas as pd
import numpy as np
from framework import DataLoader, BacktestEngine
from strategies import FundingRateHoldStrategy

print("=" * 80)
print("KIMI-1 FRAMEWORK VALIDATION TEST")
print("=" * 80)

loader = DataLoader()

# Test single symbol
symbol = 'BTCUSDT'
exchange = 'bybit'
start = '2025-07-01'
end = '2025-09-30'

print(f"\nLoading data for {symbol} from {exchange}...")
klines = loader.load_klines(exchange, symbol, start, end)
funding = loader.load_funding_rates(exchange, symbol, start, end)

print(f"Klines: {len(klines)} rows, columns: {list(klines.columns)[:5]}...")
print(f"Funding: {len(funding)} rows, columns: {list(funding.columns)}")

if klines.empty or funding.empty:
    print("ERROR: No data loaded!")
    exit(1)

print(f"\nData date range:")
print(f"  Klines: {klines['timestamp'].min()} to {klines['timestamp'].max()}")
print(f"  Funding: {funding['timestamp'].min()} to {funding['timestamp'].max()}")

# Test FR Hold strategy
print("\n" + "=" * 80)
print("TESTING FR HOLD STRATEGY")
print("=" * 80)

strategy = FundingRateHoldStrategy(entry_threshold_bps=20, exit_threshold_bps=8, max_hold_periods=3)
data = {'klines': klines, 'funding': funding}

engine = BacktestEngine(initial_capital=10000)
result = engine.run(strategy, data, position_size=1.0)

print(f"\nResults:")
print(f"  Total trades: {result.total_trades}")
print(f"  Win rate: {result.win_rate:.1%}")
print(f"  Profit factor: {result.profit_factor:.2f}")
print(f"  Total return: {result.total_return:.2%}")
print(f"  Max drawdown: {result.max_drawdown:.2%}")
print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")

if result.total_trades > 0:
    print(f"\n  Total fees paid: ${sum(t.fees for t in result.trades):.2f}")
    print(f"  Gross P&L: ${sum(t.pnl_gross for t in result.trades):.2f}")
    print(f"  Net P&L: ${sum(t.pnl_net for t in result.trades):.2f}")
    
    # Show first 5 trades
    print(f"\n  First 5 trades:")
    for i, trade in enumerate(result.trades[:5]):
        print(f"    {i+1}. {trade.entry_time.strftime('%Y-%m-%d %H:%M')} -> "
              f"{trade.exit_time.strftime('%Y-%m-%d %H:%M')} | "
              f"Net: ${trade.pnl_net:.2f} | Exit: {trade.exit_reason}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
