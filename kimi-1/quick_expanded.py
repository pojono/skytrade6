"""
Kimi-1 Quick Expanded Test - Test top 20 coins with optimized params
"""
import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002

def test_coin(symbol):
    """Test single coin with optimized breakout."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 100:
        return None
    
    # Load first 200 files for speed
    all_data = []
    for f in files[:200]:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
            all_data.append(df)
        except:
            pass
    
    if len(all_data) < 100:
        return None
    
    df = pd.concat(all_data)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    daily = df.resample('1D').agg({'high': 'max', 'close': 'last'}).dropna()
    if len(daily) < 150:
        return None
    
    # Optimized breakout: 3-day lookback, 0.5% threshold
    lookback, threshold, hold_days = 3, 0.005, 1
    
    daily['highest'] = daily['high'].rolling(window=lookback).max().shift(1)
    
    capital = 10000
    risk = 0.02
    trades = []
    
    for i in range(lookback + 1, len(daily) - hold_days):
        price = daily['close'].iloc[i]
        breakout_level = daily['highest'].iloc[i] * (1 + threshold)
        
        if price > breakout_level:
            pnl_pct = (daily['close'].iloc[i + hold_days] - price) / price
            pnl_net = pnl_pct * capital * risk - (capital * risk * FEE_PCT)
            trades.append(pnl_net)
    
    if len(trades) < 10:
        return None
    
    wins = sum(1 for p in trades if p > 0)
    return {
        'symbol': symbol,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': sum(trades),
        'avg_trade': sum(trades) / len(trades),
        'profitable': sum(trades) > 0
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 QUICK EXPANDED TEST (3-day breakout, 0.5% threshold)")
    print("=" * 100)
    
    coins = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT',
        'BCHUSDT', 'UNIUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT',
        'ATOMUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'SUIUSDT'
    ]
    
    results = []
    for i, coin in enumerate(coins):
        result = test_coin(coin)
        if result:
            results.append(result)
            status = "✓" if result['profitable'] else "✗"
            print(f"[{i+1:2d}] {status} {result['symbol']:12s} | "
                  f"{result['trades']:3d} trades | WR={result['win_rate']:.0%} | "
                  f"PnL=${result['total_pnl']:+.2f}")
        else:
            print(f"[{i+1:2d}]    {coin:12s} | insufficient data")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if results:
        df = pd.DataFrame(results)
        prof = df[df['profitable']]
        
        print(f"\nTested: {len(results)} coins")
        print(f"Profitable: {len(prof)} ({len(prof)/len(results):.0%})")
        print(f"Total trades: {df['trades'].sum()}")
        print(f"Avg trades/coin: {df['trades'].mean():.1f}")
        
        if len(prof) > 0:
            print("\n✓ PROFITABLE:")
            for _, r in prof.sort_values('total_pnl', ascending=False).iterrows():
                print(f"  {r['symbol']:12s} | {r['trades']:3d} trades | ${r['total_pnl']:+.2f}")
    
    print("\n" + "=" * 100)
