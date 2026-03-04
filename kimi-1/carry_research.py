"""
Kimi-1 Carry Trade Research - Hold for multiple days to reduce fee impact
Test across many altcoins to find any with edge.
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROUND_TRIP_FEE = 0.002  # 0.2%

def get_all_symbols():
    """Get all symbols available in bybit datalake."""
    path = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')
    return [d.name for d in path.iterdir() if d.is_dir()][:100]  # Top 100


def load_fr_full(symbol):
    """Load all available funding rate data for a symbol."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_funding_rate.csv'))
    if len(files) < 30:  # Need at least 30 days
        return None
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['fr_bps'] = df['fundingRate'] * 10000
            all_data.append(df[['timestamp', 'fr_bps']])
        except:
            pass
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)


def test_carry_strategy(symbol, entry_percentile=75, exit_percentile=25, max_holds=9):
    """
    Carry trade: Long when FR is high (above percentile), collect funding over multiple periods.
    Entry: FR > P75 (relatively high for this coin)
    Exit: FR < P25 (relatively low) or max holds reached
    """
    fr = load_fr_full(symbol)
    if fr is None or len(fr) < 100:
        return None
    
    fr = fr.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate rolling percentiles for relative thresholds
    window = 30  # 30 periods = ~10 days for 8h funding
    fr['p75'] = fr['fr_bps'].rolling(window=window, min_periods=10).quantile(0.75)
    fr['p25'] = fr['fr_bps'].rolling(window=window, min_periods=10).quantile(0.25)
    
    trades = []
    position = 0
    entry_idx = 0
    entry_fr = 0
    
    for i in range(window, len(fr)):
        fr_val = fr['fr_bps'].iloc[i]
        p75 = fr['p75'].iloc[i]
        p25 = fr['p25'].iloc[i]
        
        if pd.isna(p75) or pd.isna(p25):
            continue
        
        if position == 0 and fr_val >= p75:
            position = 1
            entry_idx = i
            entry_fr = fr_val
        
        elif position == 1:
            periods_held = i - entry_idx
            exit_signal = (fr_val <= p25) or (periods_held >= max_holds)
            
            if exit_signal:
                # Collect average FR over hold period
                hold_fr = fr['fr_bps'].iloc[entry_idx:i+1]
                total_fr = hold_fr.sum()
                
                # Net P&L
                fee_bps = 20
                net_bps = total_fr - fee_bps
                
                trades.append({
                    'periods': periods_held,
                    'total_fr': total_fr,
                    'net_bps': net_bps,
                    'won': net_bps > 0
                })
                position = 0
    
    if len(trades) < 10:
        return None
    
    wins = sum(t['won'] for t in trades)
    avg_bps = np.mean([t['net_bps'] for t in trades])
    total_bps = sum(t['net_bps'] for t in trades)
    
    return {
        'symbol': symbol,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'avg_trade_bps': avg_bps,
        'total_bps': total_bps,
        'profitable': avg_bps > 0
    }


def quick_fr_stats(symbol):
    """Quick FR statistics for a symbol."""
    fr = load_fr_full(symbol)
    if fr is None:
        return None
    
    fr_vals = fr['fr_bps']
    return {
        'symbol': symbol,
        'days': len(fr) / 3,  # 3 FR per day
        'mean': fr_vals.mean(),
        'std': fr_vals.std(),
        'max': fr_vals.max(),
        'p95': fr_vals.quantile(0.95),
        'p99': fr_vals.quantile(0.99),
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 CARRY TRADE RESEARCH - 100+ COINS")
    print("=" * 100)
    
    symbols = get_all_symbols()
    print(f"Found {len(symbols)} symbols")
    
    # First, find coins with highest FR volatility
    print("\n" + "=" * 100)
    print("STEP 1: Finding High FR Volatility Coins")
    print("=" * 100)
    
    stats = []
    for i, sym in enumerate(symbols):
        if i % 10 == 0:
            print(f"  [{i}/{len(symbols)}] Analyzing {sym}...")
        result = quick_fr_stats(sym)
        if result:
            stats.append(result)
    
    if stats:
        df_stats = pd.DataFrame(stats)
        df_stats = df_stats.sort_values('p99', ascending=False)
        
        print("\nTop 20 coins by P99 funding rate:")
        print(df_stats.head(20)[['symbol', 'days', 'mean', 'max', 'p95', 'p99']].to_string())
        
        # Test carry strategy on top candidates
        print("\n" + "=" * 100)
        print("STEP 2: Testing Carry Strategy on Top Candidates")
        print("=" * 100)
        
        top_symbols = df_stats.head(20)['symbol'].tolist()
        results = []
        
        for sym in top_symbols:
            result = test_carry_strategy(sym)
            if result:
                results.append(result)
                status = "PROFIT" if result['profitable'] else "LOSS"
                print(f"{result['symbol']:15s} | {result['trades']:3d} trades | "
                      f"WR={result['win_rate']:.0%} | Avg={result['avg_trade_bps']:+.1f}bps | {status}")
        
        # Summary
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        
        if results:
            df_results = pd.DataFrame(results)
            profitable = df_results[df_results['profitable']]
            
            print(f"\nCoins tested: {len(results)}")
            print(f"Profitable: {len(profitable)}")
            
            if len(profitable) > 0:
                print("\nProfitable carry strategies:")
                for _, r in profitable.iterrows():
                    print(f"  {r['symbol']:15s} | {r['trades']} trades | "
                          f"Total={r['total_bps']:+.1f}bps | Avg={r['avg_trade_bps']:+.1f}bps")
                
                df_results.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/carry_results.csv', index=False)
            else:
                print("\nNo profitable carry strategies found.")
                print("Even with longer holds and relative thresholds, 8h FR too small for fees.")
        
        df_stats.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/fr_volatility_ranking.csv', index=False)
        print(f"\nVolatility ranking saved to fr_volatility_ranking.csv")
    
    print("\n" + "=" * 100)
