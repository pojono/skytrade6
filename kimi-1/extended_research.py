"""
Kimi-1 Extended Research - Test longer time periods and combined signals
"""
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

ROUND_TRIP_FEE = 0.002  # 0.2%

def load_funding_range(symbol, start_date, end_date):
    """Load funding rates for a date range."""
    all_data = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        try:
            file = f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/{date_str}_funding_rate.csv'
            df = pd.read_csv(file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['fr_bps'] = df['fundingRate'] * 10000
            all_data.append(df[['timestamp', 'fr_bps']])
        except:
            pass
        current += timedelta(days=1)
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)


def fr_hold_backtest(symbol, start_date, end_date, entry_bps, exit_bps, max_periods=3):
    """
    Backtest FR hold strategy.
    Long when FR >= entry_bps, collect funding, exit when FR <= exit_bps.
    """
    funding = load_funding_range(symbol, start_date, end_date)
    if funding is None or len(funding) < 10:
        return None
    
    funding = funding.sort_values('timestamp').reset_index(drop=True)
    
    trades = []
    position = 0
    entry_idx = 0
    
    for i in range(len(funding)):
        fr = funding['fr_bps'].iloc[i]
        
        if position == 0 and fr >= entry_bps:
            position = 1
            entry_idx = i
        
        elif position == 1:
            periods_held = i - entry_idx
            exit_signal = (fr <= exit_bps) or (periods_held >= max_periods)
            
            if exit_signal:
                # Simulate P&L from funding collection
                # Assume we collect (entry_FR + exit_FR) / 2 per period held
                entry_fr = funding['fr_bps'].iloc[entry_idx]
                avg_fr = (entry_fr + fr) / 2 if fr < entry_fr else entry_fr
                
                # Gross P&L from funding (in bps)
                gross_bps = avg_fr * periods_held
                
                # Fees: 20 bps round trip
                fee_bps = 20
                
                # Net P&L (bps)
                net_bps = gross_bps - fee_bps
                
                # Convert to dollars ($10k position)
                pnl_net = net_bps * 10000 / 10000  # $ per bps for $10k position
                
                trades.append({
                    'entry_fr': entry_fr,
                    'exit_fr': fr,
                    'periods': periods_held,
                    'gross_bps': gross_bps,
                    'fee_bps': fee_bps,
                    'net_bps': net_bps,
                    'pnl_net': pnl_net,
                    'won': net_bps > 0
                })
                position = 0
    
    if len(trades) < 5:
        return None
    
    total_pnl = sum(t['pnl_net'] for t in trades)
    wins = sum(t['won'] for t in trades)
    avg_bps = np.mean([t['net_bps'] for t in trades])
    
    return {
        'symbol': symbol,
        'entry_bps': entry_bps,
        'exit_bps': exit_bps,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'avg_trade_bps': avg_bps,
        'profitable_pct': (avg_bps > 0) * 100
    }


def analyze_fr_distribution(symbol, start_date, end_date):
    """Analyze FR distribution for a symbol over a date range."""
    funding = load_funding_range(symbol, start_date, end_date)
    if funding is None:
        return None
    
    fr = funding['fr_bps']
    
    return {
        'symbol': symbol,
        'observations': len(fr),
        'mean': fr.mean(),
        'std': fr.std(),
        'min': fr.min(),
        'max': fr.max(),
        'p50': fr.median(),
        'p75': fr.quantile(0.75),
        'p90': fr.quantile(0.90),
        'p95': fr.quantile(0.95),
        'p99': fr.quantile(0.99),
        'pct_above_10bps': (fr >= 10).mean() * 100,
        'pct_above_20bps': (fr >= 20).mean() * 100,
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 EXTENDED RESEARCH (2024 Data)")
    print("=" * 100)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    # Step 1: Analyze FR distribution over full year
    print("\n" + "=" * 100)
    print("STEP 1: Funding Rate Distribution (Full Year 2024)")
    print("=" * 100)
    
    dist_results = []
    for symbol in symbols:
        result = analyze_fr_distribution(symbol, start_date, end_date)
        if result:
            dist_results.append(result)
            print(f"\n{symbol}:")
            print(f"  Observations: {result['observations']}")
            print(f"  Mean: {result['mean']:.2f} bps | Std: {result['std']:.2f} bps")
            print(f"  Range: {result['min']:.2f} to {result['max']:.2f} bps")
            print(f"  P95: {result['p95']:.2f} bps | P99: {result['p99']:.2f} bps")
            print(f"  Above 10 bps: {result['pct_above_10bps']:.1f}%")
            print(f"  Above 20 bps: {result['pct_above_20bps']:.1f}%")
    
    # Step 2: Test FR hold with various thresholds
    print("\n" + "=" * 100)
    print("STEP 2: FR Hold Strategy Backtest (Full Year 2024)")
    print("=" * 100)
    
    # Use realistic thresholds based on distribution
    configs = [
        (0.5, 0.1),   # Low threshold
        (1.0, 0.3),   # Medium
        (2.0, 0.5),   # Higher
        (5.0, 1.0),   # Very high (rare)
    ]
    
    all_results = []
    for symbol in symbols:
        print(f"\n{symbol}:")
        for entry_bps, exit_bps in configs:
            result = fr_hold_backtest(symbol, start_date, end_date, entry_bps, exit_bps)
            if result:
                all_results.append(result)
                status = "PROFIT" if result['avg_trade_bps'] > 0 else "LOSS"
                print(f"  E={entry_bps:.1f} X={exit_bps:.1f} | {result['trades']:3d} trades | "
                      f"WR={result['win_rate']:.0%} | Avg={result['avg_trade_bps']:+.1f}bps | {status}")
    
    # Step 3: Summary
    print("\n" + "=" * 100)
    print("STEP 3: Summary")
    print("=" * 100)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        profitable = df[df['avg_trade_bps'] > 0]
        print(f"\nTotal configurations tested: {len(df)}")
        print(f"Profitable configurations: {len(profitable)}")
        
        if len(profitable) > 0:
            print("\nTop 10 profitable configurations:")
            top = profitable.nlargest(10, 'avg_trade_bps')
            for _, r in top.iterrows():
                print(f"  {r['symbol']:12s} E={r['entry_bps']:4.1f} X={r['exit_bps']:4.1f} | "
                      f"{r['trades']:3d} trades | WR={r['win_rate']:.0%} | "
                      f"Avg={r['avg_trade_bps']:+.1f}bps | ${r['total_pnl']:+.2f}")
        
        # Save results
        df.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/extended_results_2024.csv', index=False)
        print(f"\nResults saved to extended_results_2024.csv")
    else:
        print("\nNo results generated.")
    
    print("\n" + "=" * 100)
