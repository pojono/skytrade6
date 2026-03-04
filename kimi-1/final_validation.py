"""
Kimi-1 Final Strategy - Simple Momentum with Validation
Simplified approach: Price momentum over 3-5 days
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002  # 0.2% round-trip

def get_daily(symbol):
    """Load daily data."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 60:
        return None
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
            all_data.append(df)
        except:
            pass
    
    if not all_data:
        return None
    
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
    
    return daily


def momentum_backtest(daily, lookback=3, hold_days=3, risk=0.02):
    """
    Simple momentum: Go long after 3 days of positive returns, hold for 3 days.
    """
    if len(daily) < lookback + hold_days + 10:
        return None
    
    # Calculate momentum (no lookahead)
    daily['momentum'] = daily['close'].pct_change(lookback).shift(1)
    
    capital = 10000
    trades = []
    
    for i in range(lookback + 1, len(daily) - hold_days):
        mom = daily['momentum'].iloc[i]
        
        # Entry: Positive momentum
        if mom > 0:
            entry_price = daily['close'].iloc[i]
            exit_price = daily['close'].iloc[i + hold_days]
            
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_gross = pnl_pct * capital * risk
            fees = capital * risk * FEE_PCT
            pnl_net = pnl_gross - fees
            
            trades.append({
                'pnl_net': pnl_net,
                'won': pnl_net > 0
            })
    
    if len(trades) < 10:
        return None
    
    wins = sum(t['won'] for t in trades)
    total_pnl = sum(t['pnl_net'] for t in trades)
    
    return {
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'avg_trade': total_pnl / len(trades),
        'profitable': total_pnl > 0
    }


def breakout_backtest(daily, lookback=5, risk=0.02):
    """
    Breakout: Go long when price breaks above N-day high.
    """
    if len(daily) < lookback + 10:
        return None
    
    daily['highest'] = daily['high'].rolling(window=lookback).max().shift(1)
    
    capital = 10000
    trades = []
    
    for i in range(lookback + 1, len(daily) - 1):
        price = daily['close'].iloc[i]
        prev_high = daily['highest'].iloc[i]
        
        # Entry: Break above N-day high
        if price > prev_high * 1.01:  # 1% breakout threshold
            entry_price = price
            exit_price = daily['close'].iloc[i + 1]  # Hold 1 day
            
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_gross = pnl_pct * capital * risk
            fees = capital * risk * FEE_PCT
            pnl_net = pnl_gross - fees
            
            trades.append({
                'pnl_net': pnl_net,
                'won': pnl_net > 0
            })
    
    if len(trades) < 10:
        return None
    
    wins = sum(t['won'] for t in trades)
    total_pnl = sum(t['pnl_net'] for t in trades)
    
    return {
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'profitable': total_pnl > 0
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 FINAL STRATEGY VALIDATION")
    print("=" * 100)
    
    # Top 30 liquid symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT',
               'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT',
               'BCHUSDT', 'UNIUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT',
               'ATOMUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'SUIUSDT',
               'TRXUSDT', 'ETCUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
               'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'THETAUSDT', 'XTZUSDT']
    
    print(f"\nTesting {len(symbols)} symbols...")
    print("-" * 100)
    
    # Test momentum
    print("\n1. Momentum Strategy (3-day lookback, 3-day hold):")
    mom_results = []
    for sym in symbols:
        daily = get_daily(sym)
        if daily is None:
            continue
        result = momentum_backtest(daily)
        if result:
            mom_results.append({'symbol': sym, **result})
            status = "✓" if result['profitable'] else "✗"
            print(f"  {status} {sym:12s} | {result['trades']:3d} trades | "
                  f"WR={result['win_rate']:.0%} | PnL=${result['total_pnl']:+.2f}")
    
    # Test breakout
    print("\n2. Breakout Strategy (5-day lookback, 1-day hold):")
    break_results = []
    for sym in symbols:
        daily = get_daily(sym)
        if daily is None:
            continue
        result = breakout_backtest(daily)
        if result:
            break_results.append({'symbol': sym, **result})
            status = "✓" if result['profitable'] else "✗"
            print(f"  {status} {sym:12s} | {result['trades']:3d} trades | "
                  f"WR={result['win_rate']:.0%} | PnL=${result['total_pnl']:+.2f}")
    
    # Summary
    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    
    if mom_results:
        df_mom = pd.DataFrame(mom_results)
        prof_mom = df_mom[df_mom['profitable']]
        print(f"\nMomentum: {len(prof_mom)}/{len(df_mom)} profitable ({len(prof_mom)/len(df_mom):.0%})")
        if len(prof_mom) > 0:
            print("Profitable symbols:", ', '.join(prof_mom['symbol'].tolist()))
            prof_mom.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/momentum_profitable.csv', index=False)
    
    if break_results:
        df_break = pd.DataFrame(break_results)
        prof_break = df_break[df_break['profitable']]
        print(f"\nBreakout: {len(prof_break)}/{len(df_break)} profitable ({len(prof_break)/len(df_break):.0%})")
        if len(prof_break) > 0:
            print("Profitable symbols:", ', '.join(prof_break['symbol'].tolist()))
            prof_break.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/breakout_profitable.csv', index=False)
    
    # Combined
    all_results = mom_results + break_results
    if all_results:
        df_all = pd.DataFrame(all_results)
        prof_all = df_all[df_all['profitable']]
        print(f"\n{'=' * 100}")
        print(f"TOTAL: {len(prof_all)}/{len(df_all)} configurations profitable ({len(prof_all)/len(df_all):.0%})")
        
        if len(prof_all) > 0:
            print(f"\nProfitable combinations:")
            for _, r in prof_all.nlargest(10, 'total_pnl').iterrows():
                print(f"  {r['symbol']:12s} | {r['trades']} trades | "
                      f"${r['total_pnl']:+.2f} | Avg=${r['avg_trade']:+.2f}")
    
    print("\n" + "=" * 100)
