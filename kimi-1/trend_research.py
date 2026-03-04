"""
Kimi-1 Trend Following - Daily bars, few trades, minimize fee impact
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002  # 0.2% round trip

def get_daily_ohlc(symbol, exchange='bybit'):
    """Aggregate 1m klines to daily OHLC."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/{exchange}/{symbol}/*_kline_1m.csv'))
    if len(files) < 30:
        return None
    
    all_data = []
    for f in files[:365]:  # Max 1 year
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
    
    # Resample to daily
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return daily


def test_trend_following(symbol, fast=10, slow=30, atr_period=14, risk_per_trade=0.02):
    """
    Trend following with ATR-based position sizing and stops.
    Entry: Fast EMA > Slow EMA and price > Fast EMA
    Exit: Stop loss (2x ATR) or trend reversal
    """
    daily = get_daily_ohlc(symbol)
    if daily is None or len(daily) < slow + 10:
        return None
    
    # Calculate indicators
    daily['ema_fast'] = daily['close'].ewm(span=fast, adjust=False).mean()
    daily['ema_slow'] = daily['close'].ewm(span=slow, adjust=False).mean()
    
    # ATR
    daily['tr1'] = daily['high'] - daily['low']
    daily['tr2'] = abs(daily['high'] - daily['close'].shift(1))
    daily['tr3'] = abs(daily['low'] - daily['close'].shift(1))
    daily['tr'] = daily[['tr1', 'tr2', 'tr3']].max(axis=1)
    daily['atr'] = daily['tr'].rolling(window=atr_period).mean()
    daily['atr_pct'] = daily['atr'] / daily['close']
    
    # Signals (no lookahead)
    daily['trend_up'] = (daily['ema_fast'] > daily['ema_slow']) & (daily['close'] > daily['ema_fast'])
    daily['trend_down'] = (daily['ema_fast'] < daily['ema_slow']) & (daily['close'] < daily['ema_fast'])
    
    # Backtest
    capital = 10000
    trades = []
    position = 0
    entry_price = 0
    stop_price = 0
    
    for i in range(slow + 1, len(daily)):
        price = daily['close'].iloc[i]
        atr = daily['atr'].iloc[i]
        trend_up = daily['trend_up'].iloc[i]
        trend_down = daily['trend_down'].iloc[i]
        
        if position == 0:
            if trend_up:
                position = 1
                entry_price = price
                stop_price = price - 2 * atr
            elif trend_down:
                position = -1
                entry_price = price
                stop_price = price + 2 * atr
        
        elif position == 1:
            # Check stop or trend reversal
            stop_hit = price <= stop_price
            trend_reversal = trend_down
            
            if stop_hit or trend_reversal:
                pnl_pct = (price - entry_price) / entry_price
                pnl_gross = pnl_pct * capital * risk_per_trade
                fees = capital * risk_per_trade * FEE_PCT
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'direction': 'long',
                    'pnl_net': pnl_net,
                    'pnl_pct': pnl_pct * 100,
                    'won': pnl_net > 0
                })
                position = 0
            else:
                # Trail stop
                new_stop = price - 2 * atr
                if new_stop > stop_price:
                    stop_price = new_stop
        
        elif position == -1:
            stop_hit = price >= stop_price
            trend_reversal = trend_up
            
            if stop_hit or trend_reversal:
                pnl_pct = (entry_price - price) / entry_price
                pnl_gross = pnl_pct * capital * risk_per_trade
                fees = capital * risk_per_trade * FEE_PCT
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'direction': 'short',
                    'pnl_net': pnl_net,
                    'pnl_pct': pnl_pct * 100,
                    'won': pnl_net > 0
                })
                position = 0
            else:
                new_stop = price + 2 * atr
                if new_stop < stop_price:
                    stop_price = new_stop
    
    if len(trades) < 5:
        return None
    
    wins = sum(t['won'] for t in trades)
    total_pnl = sum(t['pnl_net'] for t in trades)
    
    return {
        'symbol': symbol,
        'fast': fast,
        'slow': slow,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'avg_trade': total_pnl / len(trades),
        'profitable': total_pnl > 0
    }


def test_breakout(symbol, lookback=20, min_volatility=0.02):
    """
    Volatility breakout - enter on break of N-day high/low.
    """
    daily = get_daily_ohlc(symbol)
    if daily is None or len(daily) < lookback + 10:
        return None
    
    daily['highest'] = daily['high'].rolling(window=lookback).max().shift(1)
    daily['lowest'] = daily['low'].rolling(window=lookback).min().shift(1)
    daily['volatility'] = daily['close'].pct_change().rolling(window=20).std() * np.sqrt(365)
    
    capital = 10000
    risk = 0.02
    trades = []
    position = 0
    entry_price = 0
    
    for i in range(lookback + 1, len(daily)):
        price = daily['close'].iloc[i]
        high = daily['high'].iloc[i]
        low = daily['low'].iloc[i]
        vol = daily['volatility'].iloc[i]
        
        if vol < min_volatility:
            continue
        
        if position == 0:
            if high > daily['highest'].iloc[i]:
                position = 1
                entry_price = price
            elif low < daily['lowest'].iloc[i]:
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit after 5 days or on reversal
            bars_since_entry = i - lookback - 1
            if bars_since_entry >= 5 or low < daily['lowest'].iloc[i]:
                pnl_pct = (price - entry_price) / entry_price
                pnl_gross = pnl_pct * capital * risk
                fees = capital * risk * FEE_PCT
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'pnl_net': pnl_net,
                    'won': pnl_net > 0
                })
                position = 0
        
        elif position == -1:
            bars_since_entry = i - lookback - 1
            if bars_since_entry >= 5 or high > daily['highest'].iloc[i]:
                pnl_pct = (entry_price - price) / entry_price
                pnl_gross = pnl_pct * capital * risk
                fees = capital * risk * FEE_PCT
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'pnl_net': pnl_net,
                    'won': pnl_net > 0
                })
                position = 0
    
    if len(trades) < 5:
        return None
    
    wins = sum(t['won'] for t in trades)
    total_pnl = sum(t['pnl_net'] for t in trades)
    
    return {
        'symbol': symbol,
        'strategy': 'breakout',
        'lookback': lookback,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'profitable': total_pnl > 0
    }


if __name__ == '__main__':
    print("=" * 100)
    print("KIMI-1 TREND FOLLOWING (DAILY BARS)")
    print("=" * 100)
    
    # Test on top 20 liquid symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT',
               'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT',
               'BCHUSDT', 'UNIUSDT', 'AAVEUSDT', 'NEARUSDT', 'FILUSDT',
               'ATOMUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'SUIUSDT']
    
    print(f"\nTesting {len(symbols)} symbols...")
    print("-" * 100)
    
    # Test trend following
    print("\nTrend Following (EMA 10/30):")
    trend_results = []
    for sym in symbols:
        result = test_trend_following(sym, fast=10, slow=30)
        if result:
            trend_results.append(result)
            status = "PROFIT" if result['profitable'] else "LOSS"
            print(f"  {result['symbol']:12s} | {result['trades']:2d} trades | "
                  f"WR={result['win_rate']:.0%} | Total=${result['total_pnl']:+.2f} | {status}")
    
    # Test breakout
    print("\nBreakout (20-day):")
    breakout_results = []
    for sym in symbols:
        result = test_breakout(sym, lookback=20)
        if result:
            breakout_results.append(result)
            status = "PROFIT" if result['profitable'] else "LOSS"
            print(f"  {result['symbol']:12s} | {result['trades']:2d} trades | "
                  f"WR={result['win_rate']:.0%} | Total=${result['total_pnl']:+.2f} | {status}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if trend_results:
        df_trend = pd.DataFrame(trend_results)
        prof_trend = df_trend[df_trend['profitable']]
        print(f"\nTrend Following: {len(prof_trend)}/{len(df_trend)} profitable")
        if len(prof_trend) > 0:
            print("Profitable:", ', '.join(prof_trend['symbol'].tolist()))
    
    if breakout_results:
        df_break = pd.DataFrame(breakout_results)
        prof_break = df_break[df_break['profitable']]
        print(f"\nBreakout: {len(prof_break)}/{len(df_break)} profitable")
        if len(prof_break) > 0:
            print("Profitable:", ', '.join(prof_break['symbol'].tolist()))
    
    # Combined
    all_profitable = []
    if trend_results:
        all_profitable.extend([r['symbol'] for r in trend_results if r['profitable']])
    if breakout_results:
        all_profitable.extend([r['symbol'] for r in breakout_results if r['profitable']])
    
    if all_profitable:
        print(f"\nAny profitable edge found: {len(set(all_profitable))} symbols")
        print("Symbols with edge:", ', '.join(set(all_profitable)))
    else:
        print("\nNo profitable edge found in trend following or breakout strategies.")
    
    print("\n" + "=" * 100)
