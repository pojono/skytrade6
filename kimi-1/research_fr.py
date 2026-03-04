"""
Kimi-1 Strategy Research - Funding Rate Edge Detection
Realistic thresholds based on actual data distribution.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime
from framework import DataLoader, BacktestEngine, Trade, BacktestResult
from strategies import FundingRateHoldStrategy
import warnings
warnings.filterwarnings('ignore')

FEE_MAKER = 0.0004  # 0.04%
FEE_TAKER = 0.001   # 0.10%
ROUND_TRIP_TAKER = 0.002  # 0.20%

def analyze_funding_distribution(exchange='bybit', symbols=None, n_samples=100):
    """Analyze funding rate distribution to find realistic thresholds."""
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 
                   'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT', 'DOTUSDT']
    
    loader = DataLoader()
    all_stats = []
    
    print(f"\nAnalyzing funding rate distribution for {len(symbols)} symbols on {exchange}...")
    
    for symbol in symbols:
        files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/*_funding_rate.csv'))
        if not files:
            continue
            
        # Load sample of files
        rates = []
        for f in files[:n_samples]:
            df = pd.read_csv(f)
            if 'fundingRate' in df.columns:
                rates.extend(df['fundingRate'].tolist())
        
        if not rates:
            continue
            
        rates = pd.Series(rates)
        
        stats = {
            'symbol': symbol,
            'n': len(rates),
            'mean_bps': rates.mean() * 10000,
            'median_bps': rates.median() * 10000,
            'std_bps': rates.std() * 10000,
            'p75_bps': rates.quantile(0.75) * 10000,
            'p90_bps': rates.quantile(0.90) * 10000,
            'p95_bps': rates.quantile(0.95) * 10000,
            'p99_bps': rates.quantile(0.99) * 10000,
            'max_bps': rates.max() * 10000,
            'pos_pct': (rates > 0).mean() * 100,
        }
        all_stats.append(stats)
    
    return pd.DataFrame(all_stats)


def test_fr_hold_params(symbol, exchange, start, end, thresholds):
    """Test FR Hold strategy with various threshold combinations."""
    loader = DataLoader()
    
    klines = loader.load_klines(exchange, symbol, start, end)
    funding = loader.load_funding_rates(exchange, symbol, start, end)
    
    if klines.empty or funding.empty:
        return []
    
    results = []
    data = {'klines': klines, 'funding': funding}
    
    for entry_bps, exit_bps, max_hold in thresholds:
        strategy = FundingRateHoldStrategy(
            entry_threshold_bps=entry_bps,
            exit_threshold_bps=exit_bps,
            max_hold_periods=max_hold
        )
        
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(strategy, data, position_size=1.0)
        
        if result.total_trades > 5:
            results.append({
                'symbol': symbol,
                'entry_bps': entry_bps,
                'exit_bps': exit_bps,
                'max_hold': max_hold,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_return': result.total_return,
                'max_dd': result.max_drawdown,
                'sharpe': result.sharpe_ratio,
                'net_pnl': sum(t.pnl_net for t in result.trades),
                'total_fees': sum(t.fees for t in result.trades),
                'avg_trade_bps': (sum(t.pnl_net for t in result.trades) / result.total_trades / 100) * 10000 if result.total_trades > 0 else 0,
            })
    
    return results


def scan_best_configs(symbols, exchange='bybit', start='2025-07-01', end='2025-12-31'):
    """Scan multiple symbols and threshold combinations to find best configs."""
    
    # Thresholds to test (realistic for 8h funding rates)
    thresholds = [
        (1.0, 0.3, 3),   # Entry 1 bps, exit 0.3 bps
        (1.5, 0.5, 3),   # Entry 1.5 bps, exit 0.5 bps
        (2.0, 0.5, 3),   # Entry 2 bps, exit 0.5 bps
        (1.5, 0.5, 2),   # Shorter hold
        (2.0, 1.0, 3),   # Higher exit threshold
        (1.0, 0.5, 4),   # Lower entry, longer hold
    ]
    
    all_results = []
    
    print(f"\nScanning {len(symbols)} symbols with {len(thresholds)} threshold combinations...")
    print("=" * 100)
    
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Testing {symbol}...")
        results = test_fr_hold_params(symbol, exchange, start, end, thresholds)
        all_results.extend(results)
        
        # Show best for this symbol
        if results:
            best = max(results, key=lambda x: x['net_pnl'])
            print(f"  Best: {best['trades']} trades, WR={best['win_rate']:.1%}, "
                  f"Net=${best['net_pnl']:.2f}, PF={best['profit_factor']:.2f}")
    
    return pd.DataFrame(all_results)


def plot_equity_curves(results_df, top_n=5):
    """Plot equity curves for top performing configurations."""
    if results_df.empty:
        print("No results to plot")
        return
    
    top = results_df.nlargest(top_n, 'net_pnl')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(top.iterrows()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        ax.set_title(f"{row['symbol']} | E={row['entry_bps']:.1f} X={row['exit_bps']:.1f}\n"
                    f"WR={row['win_rate']:.1%} PF={row['profit_factor']:.2f}")
        ax.text(0.5, 0.5, f"Net P&L: ${row['net_pnl']:.2f}\nTrades: {row['trades']}",
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/Projects/skytrade6/kimi-1/top_configs.png')
    print(f"\nPlot saved to top_configs.png")


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 FUNDING RATE STRATEGY RESEARCH")
    print("=" * 100)
    
    # Step 1: Analyze funding distribution
    print("\n" + "=" * 100)
    print("STEP 1: Funding Rate Distribution Analysis")
    print("=" * 100)
    
    dist_df = analyze_funding_distribution()
    print("\n" + dist_df.to_string())
    
    # Step 2: Scan for best configs
    print("\n" + "=" * 100)
    print("STEP 2: Parameter Scan")
    print("=" * 100)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 
               'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT']
    
    results_df = scan_best_configs(symbols, start='2025-07-01', end='2025-12-31')
    
    # Step 3: Show results
    print("\n" + "=" * 100)
    print("STEP 3: Results Summary")
    print("=" * 100)
    
    if not results_df.empty:
        print(f"\nTotal configurations tested: {len(results_df)}")
        print(f"Profitable configs: {(results_df['net_pnl'] > 0).sum()}")
        
        # Show top 10 by net P&L
        print("\n" + "-" * 100)
        print("TOP 10 CONFIGURATIONS BY NET P&L:")
        print("-" * 100)
        top10 = results_df.nlargest(10, 'net_pnl')
        for idx, row in top10.iterrows():
            print(f"{row['symbol']:12s} | E={row['entry_bps']:4.1f} X={row['exit_bps']:4.1f} H={row['max_hold']} | "
                  f"Trades={row['trades']:3d} | WR={row['win_rate']:5.1%} | "
                  f"PF={row['profit_factor']:5.2f} | Net=${row['net_pnl']:8.2f} | "
                  f"Fees=${row['total_fees']:6.2f}")
        
        # Show top 10 by Sharpe
        print("\n" + "-" * 100)
        print("TOP 10 CONFIGURATIONS BY SHARPE RATIO:")
        print("-" * 100)
        top_sharpe = results_df.nlargest(10, 'sharpe')
        for idx, row in top_sharpe.iterrows():
            print(f"{row['symbol']:12s} | E={row['entry_bps']:4.1f} X={row['exit_bps']:4.1f} H={row['max_hold']} | "
                  f"Trades={row['trades']:3d} | WR={row['win_rate']:5.1%} | "
                  f"Sharpe={row['sharpe']:5.2f} | Net=${row['net_pnl']:8.2f}")
        
        # Save results
        results_df.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/fr_scan_results.csv', index=False)
        print(f"\nResults saved to fr_scan_results.csv")
        
        # Check for real edge (profitable after fees)
        print("\n" + "=" * 100)
        print("STEP 4: Edge Validation")
        print("=" * 100)
        
        profitable = results_df[results_df['net_pnl'] > results_df['total_fees']]
        if not profitable.empty:
            print(f"\nConfigs profitable after 2x fees: {len(profitable)}")
            print("\nTop 5 by true edge (net_pnl - fees):")
            profitable['true_edge'] = profitable['net_pnl'] - profitable['total_fees']
            top_edge = profitable.nlargest(5, 'true_edge')
            for idx, row in top_edge.iterrows():
                print(f"  {row['symbol']:12s} | True Edge=${row['true_edge']:8.2f}")
        else:
            print("\nNo configurations found with clear edge after fees.")
            print("This suggests the FR hold strategy may not work with 8h rates alone.")
            print("Consider:")
            print("  - Using 1h funding rate coins (higher volatility)")
            print("  - Cross-exchange arbitrage")
            print("  - Combining with price momentum signals")
    else:
        print("No results generated - check data availability")
    
    print("\n" + "=" * 100)
    print("RESEARCH COMPLETE")
    print("=" * 100)
