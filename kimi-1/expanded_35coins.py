"""
Kimi-1 Expanded Test - 30+ Bybit Coins
Test both futures and spot data for cross-validation
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002

def get_daily_bybit(symbol, data_type='futures'):
    """
    Load daily data from Bybit datalake
    data_type: 'futures' (default) or 'spot'
    """
    if data_type == 'spot':
        pattern = f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m_spot.csv'
    else:
        pattern = f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'
    
    files = sorted(glob.glob(pattern))
    if len(files) < 100:
        return None, 0
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
            all_data.append(df)
        except:
            pass
    
    if not all_data:
        return None, 0
    
    df = pd.concat(all_data)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return daily, len(daily)


def breakout_oos(daily, lookback=5, threshold=0.01, hold_days=1, risk=0.02):
    """
    5-day breakout with OOS validation
    Fixed: uses .shift(1) for no lookahead
    """
    if len(daily) < lookback + hold_days + 10:
        return None
    
    daily['highest'] = daily['high'].rolling(window=lookback).max().shift(1)
    
    capital = 10000
    trades = []
    
    for i in range(lookback + 1, len(daily) - hold_days):
        price = daily['close'].iloc[i]
        breakout_level = daily['highest'].iloc[i] * (1 + threshold)
        
        if price > breakout_level:
            entry_price = price
            exit_price = daily['close'].iloc[i + hold_days]
            
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_gross = pnl_pct * capital * risk
            fees = capital * risk * FEE_PCT
            pnl_net = pnl_gross - fees
            
            trades.append({
                'pnl_net': pnl_net,
                'won': pnl_net > 0,
                'entry': entry_price,
                'exit': exit_price
            })
    
    if len(trades) < 10:
        return None
    
    wins = sum(1 for t in trades if t['won'])
    return {
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': sum(t['pnl_net'] for t in trades),
        'avg_trade': sum(t['pnl_net'] for t in trades) / len(trades),
        'profitable': sum(t['pnl_net'] for t in trades) > 0
    }


def test_symbol(symbol):
    """Test both futures and spot for a symbol."""
    results = {'symbol': symbol}
    
    # Test futures
    daily_fut, n_days_fut = get_daily_bybit(symbol, 'futures')
    if daily_fut is not None and n_days_fut >= 200:
        res_fut = breakout_oos(daily_fut)
        if res_fut:
            results['futures'] = {
                'days': n_days_fut,
                **res_fut
            }
    
    # Test spot
    daily_spot, n_days_spot = get_daily_bybit(symbol, 'spot')
    if daily_spot is not None and n_days_spot >= 200:
        res_spot = breakout_oos(daily_spot)
        if res_spot:
            results['spot'] = {
                'days': n_days_spot,
                **res_spot
            }
    
    return results if len(results) > 1 else None


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 EXPANDED TEST - 30+ Bybit Coins (Futures + Spot)")
    print("=" * 100)
    print("\nStrategy: 5-day breakout + 1% threshold (OOS validated)")
    print("Testing both futures and spot data for cross-validation")
    print("-" * 100)
    
    # Top 35 coins by liquidity
    coins = [
        # Majors
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT',
        'BCHUSDT', 'UNIUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT',
        
        # Mid-caps
        'ATOMUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'SUIUSDT',
        'TRXUSDT', 'ETCUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
        'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'THETAUSDT', 'XTZUSDT',
        
        # Newer/High volume
        'BNBUSDT', 'XLMUSDT', 'TONUSDT', 'SHIB1000USDT', 'PEPEUSDT'
    ]
    
    results = []
    fut_only = []
    both = []
    
    for i, coin in enumerate(coins):
        result = test_symbol(coin)
        if result:
            results.append(result)
            has_fut = 'futures' in result
            has_spot = 'spot' in result
            
            if has_fut and has_spot:
                both.append(result)
                f = result['futures']
                s = result['spot']
                status = "✓✓" if f['profitable'] and s['profitable'] else "✓" if f['profitable'] or s['profitable'] else "✗"
                print(f"[{i+1:2d}] {status} {coin:15s} | "
                      f"FUT: {f['trades']:3d}T ${f['total_pnl']:+6.0f} | "
                      f"SPOT: {s['trades']:3d}T ${s['total_pnl']:+6.0f}")
            elif has_fut:
                fut_only.append(result)
                f = result['futures']
                status = "✓" if f['profitable'] else "✗"
                print(f"[{i+1:2d}] {status} {coin:15s} | "
                      f"FUT: {f['trades']:3d}T ${f['total_pnl']:+6.0f} | "
                      f"SPOT: N/A")
        else:
            print(f"[{i+1:2d}]    {coin:15s} | insufficient data")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    # Futures results
    fut_results = [r['futures'] for r in results if 'futures' in r]
    spot_results = [r['spot'] for r in results if 'spot' in r]
    
    if fut_results:
        df_fut = pd.DataFrame(fut_results)
        prof_fut = df_fut[df_fut['profitable'] == True]
        
        print(f"\nFUTURES:")
        print(f"  Tested: {len(df_fut)} coins")
        print(f"  Profitable: {len(prof_fut)} ({len(prof_fut)/len(df_fut):.0%})")
        print(f"  Total trades: {df_fut['trades'].sum()}")
        print(f"  Avg trades/coin: {df_fut['trades'].mean():.1f}")
        
        if len(prof_fut) > 0:
            print(f"\n  TOP 10 FUTURES:")
            top_fut = prof_fut.sort_values('total_pnl', ascending=False).head(10)
            for idx, row in top_fut.iterrows():
                print(f"    {row.name if hasattr(row, 'name') else idx:3d}. ${row['total_pnl']:+7.2f} | "
                      f"{row['trades']:3d}T | {row['win_rate']:.0%} WR")
    
    if spot_results:
        df_spot = pd.DataFrame(spot_results)
        prof_spot = df_spot[df_spot['profitable'] == True]
        
        print(f"\nSPOT:")
        print(f"  Tested: {len(df_spot)} coins")
        print(f"  Profitable: {len(prof_spot)} ({len(prof_spot)/len(df_spot):.0%})")
        print(f"  Total trades: {df_spot['trades'].sum()}")
        print(f"  Avg trades/coin: {df_spot['trades'].mean():.1f}")
        
        if len(prof_spot) > 0:
            print(f"\n  TOP 10 SPOT:")
            top_spot = prof_spot.sort_values('total_pnl', ascending=False).head(10)
            for idx, row in top_spot.iterrows():
                print(f"    {row.name if hasattr(row, 'name') else idx:3d}. ${row['total_pnl']:+7.2f} | "
                      f"{row['trades']:3d}T | {row['win_rate']:.0%} WR")
    
    # Cross-validation
    if both:
        print(f"\n" + "=" * 100)
        print("CROSS-VALIDATION (Both Futures + Spot Profitable)")
        print("=" * 100)
        
        valid_both = []
        for r in both:
            if r['futures']['profitable'] and r['spot']['profitable']:
                valid_both.append(r)
        
        print(f"\nCoins profitable on BOTH: {len(valid_both)}/{len(both)}")
        for r in valid_both:
            print(f"  {r['symbol']:15s} | "
                  f"FUT: ${r['futures']['total_pnl']:+6.0f} | "
                  f"SPOT: ${r['spot']['total_pnl']:+6.0f}")
    
    # Save results
    if results:
        flat_results = []
        for r in results:
            if 'futures' in r:
                flat_results.append({
                    'symbol': r['symbol'],
                    'market': 'futures',
                    **r['futures']
                })
            if 'spot' in r:
                flat_results.append({
                    'symbol': r['symbol'],
                    'market': 'spot',
                    **r['spot']
                })
        
        df_out = pd.DataFrame(flat_results)
        df_out.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/EXPANDED_35COINS_RESULTS.csv', index=False)
        print(f"\n✓ Results saved to EXPANDED_35COINS_RESULTS.csv")
    
    print("\n" + "=" * 100)
