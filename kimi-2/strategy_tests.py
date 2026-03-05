"""
Fresh Strategy Research - Test ALL hypotheses from scratch
No trust in prior research - verify everything with raw data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/kimi-2')
from framework import (
    load_klines, load_funding_rates, load_open_interest, load_long_short_ratio,
    BacktestResult, Trade, RT_TAKER, RT_MAKER, TAKER_FEE, MAKER_FEE
)

# Fee structures for user's account
FEE_TAKER_RT = 0.002  # 0.2% round-trip
FEE_MAKER_RT = 0.0008  # 0.08% round-trip


def analyze_funding_rate_hold(symbol: str, start: str, end: str, 
                               entry_threshold_bps: float = 20,
                               exit_threshold_bps: float = 8,
                               max_holds: int = 3,
                               use_maker: bool = False) -> BacktestResult:
    """
    Test: Go long when funding rate is extreme, collect funding, exit when it normalizes
    """
    fee_rate = FEE_MAKER_RT if use_maker else FEE_TAKER_RT
    
    # Load data
    klines = load_klines(symbol, start, end)
    funding = load_funding_rates(symbol, start, end)
    
    if len(klines) == 0 or len(funding) == 0:
        return BacktestResult(
            strategy_name='fr_hold',
            symbol=symbol,
            start_date=pd.to_datetime(start),
            end_date=pd.to_datetime(end),
            trades=[]
        )
    
    # Merge funding to klines (forward fill)
    klines = klines.set_index('timestamp')
    funding = funding.set_index('timestamp')
    df = klines.join(funding[['fundingRate']], how='left')
    df['fundingRate'] = df['fundingRate'].fillna(method='ffill')
    df = df.reset_index()
    
    trades = []
    in_position = False
    position_start = None
    position_entry_price = None
    holds_count = 0
    
    for i, row in df.iterrows():
        if pd.isna(row['fundingRate']):
            continue
            
        fr_bps = row['fundingRate'] * 10000
        
        # Entry condition: FR <= -entry_threshold (collect negative funding = get paid)
        if not in_position and fr_bps <= -entry_threshold_bps:
            in_position = True
            position_start = row['timestamp']
            position_entry_price = row['close']
            holds_count = 0
        
        # Exit conditions
        elif in_position:
            holds_count += 1
            
            # Exit if FR normalizes or max holds reached
            should_exit = (fr_bps >= -exit_threshold_bps) or (holds_count >= max_holds)
            
            if should_exit or i == len(df) - 1:
                # Calculate P&L
                exit_price = row['close']
                price_return = (exit_price - position_entry_price) / position_entry_price
                
                # Funding collected (negative FR = long gets paid)
                # Rough estimate: sum of FR during hold
                hold_period = df[(df['timestamp'] >= position_start) & (df['timestamp'] <= row['timestamp'])]
                fr_collected = -hold_period['fundingRate'].sum() * 10000  # in bps
                
                # For 8h funding, each period pays ~8h worth
                # Actual payment is (FR * position_value) every 8h
                position_value = 10000  # $10k notional
                funding_pnl = fr_collected / 10000 * position_value
                
                # Price P&L
                price_pnl = price_return * position_value
                
                # Fees
                fees = position_value * fee_rate
                
                gross_pnl = funding_pnl + price_pnl
                net_pnl = gross_pnl - fees
                
                trades.append(Trade(
                    entry_time=position_start,
                    exit_time=row['timestamp'],
                    symbol=symbol,
                    side='long',
                    entry_price=position_entry_price,
                    exit_price=exit_price,
                    size_usd=position_value,
                    fees=fees,
                    pnl_gross=gross_pnl,
                    pnl_net=net_pnl,
                    exit_reason='fr_normalized' if fr_bps >= -exit_threshold_bps else 'max_holds'
                ))
                
                in_position = False
                position_start = None
                position_entry_price = None
    
    result = BacktestResult(
        strategy_name=f'fr_hold_{entry_threshold_bps}in_{exit_threshold_bps}out',
        symbol=symbol,
        start_date=pd.to_datetime(start),
        end_date=pd.to_datetime(end),
        trades=trades
    )
    result.calculate_metrics()
    return result


def analyze_momentum(symbol: str, start: str, end: str,
                     lookback: int = 20,
                     threshold_std: float = 1.5,
                     hold_bars: int = 10,
                     use_maker: bool = False) -> BacktestResult:
    """
    Test: Volatility breakout - enter on large moves, exit after hold period
    """
    fee_rate = FEE_MAKER_RT if use_maker else FEE_TAKER_RT
    
    df = load_klines(symbol, start, end)
    if len(df) == 0:
        return BacktestResult('momentum', symbol, pd.to_datetime(start), pd.to_datetime(end), [])
    
    # Calculate returns and volatility
    df['return'] = df['close'].pct_change()
    df['vol'] = df['return'].rolling(lookback).std()
    df['mean_ret'] = df['return'].rolling(lookback).mean()
    
    trades = []
    
    for i in range(lookback + 1, len(df) - hold_bars):
        ret = df.iloc[i]['return']
        vol = df.iloc[i]['vol']
        mean_ret = df.iloc[i]['mean_ret']
        
        if pd.isna(vol) or vol == 0:
            continue
        
        # Entry: return > mean + threshold * vol (breakout)
        if ret > mean_ret + threshold_std * vol:
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[i + hold_bars]['close']
            
            position_value = 10000
            price_return = (exit_price - entry_price) / entry_price
            
            fees = position_value * fee_rate
            gross_pnl = price_return * position_value
            net_pnl = gross_pnl - fees
            
            trades.append(Trade(
                entry_time=df.iloc[i]['timestamp'],
                exit_time=df.iloc[i + hold_bars]['timestamp'],
                symbol=symbol,
                side='long',
                entry_price=entry_price,
                exit_price=exit_price,
                size_usd=position_value,
                fees=fees,
                pnl_gross=gross_pnl,
                pnl_net=net_pnl,
                exit_reason='time_exit'
            ))
    
    result = BacktestResult(
        strategy_name=f'momentum_{lookback}_{threshold_std}_{hold_bars}',
        symbol=symbol,
        start_date=pd.to_datetime(start),
        end_date=pd.to_datetime(end),
        trades=trades
    )
    result.calculate_metrics()
    return result


def analyze_mean_reversion(symbol: str, start: str, end: str,
                          lookback: int = 20,
                          zscore_threshold: float = 2.0,
                          hold_bars: int = 5,
                          use_maker: bool = False) -> BacktestResult:
    """
    Test: Mean reversion - enter when price deviates, exit when it reverts
    """
    fee_rate = FEE_MAKER_RT if use_maker else FEE_TAKER_RT
    
    df = load_klines(symbol, start, end)
    if len(df) == 0:
        return BacktestResult('mean_rev', symbol, pd.to_datetime(start), pd.to_datetime(end), [])
    
    # Calculate z-score of price relative to moving average
    df['ma'] = df['close'].rolling(lookback).mean()
    df['std'] = df['close'].rolling(lookback).std()
    df['zscore'] = (df['close'] - df['ma']) / df['std']
    
    trades = []
    
    for i in range(lookback, len(df) - hold_bars):
        z = df.iloc[i]['zscore']
        
        if pd.isna(z):
            continue
        
        # Entry: price > 2 std below mean (buy dip)
        if z < -zscore_threshold:
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[i + hold_bars]['close']
            
            position_value = 10000
            price_return = (exit_price - entry_price) / entry_price
            
            fees = position_value * fee_rate
            gross_pnl = price_return * position_value
            net_pnl = gross_pnl - fees
            
            trades.append(Trade(
                entry_time=df.iloc[i]['timestamp'],
                exit_time=df.iloc[i + hold_bars]['timestamp'],
                symbol=symbol,
                side='long',
                entry_price=entry_price,
                exit_price=exit_price,
                size_usd=position_value,
                fees=fees,
                pnl_gross=gross_pnl,
                pnl_net=net_pnl,
                exit_reason='time_exit'
            ))
    
    result = BacktestResult(
        strategy_name=f'meanrev_{lookback}_{zscore_threshold}_{hold_bars}',
        symbol=symbol,
        start_date=pd.to_datetime(start),
        end_date=pd.to_datetime(end),
        trades=trades
    )
    result.calculate_metrics()
    return result


def analyze_oi_divergence(symbol: str, start: str, end: str,
                          use_maker: bool = False) -> BacktestResult:
    """
    Test: Open Interest divergence with price
    When OI rises but price falls = potential short squeeze (long)
    When OI falls but price rises = potential distribution (short)
    """
    fee_rate = FEE_MAKER_RT if use_maker else FEE_TAKER_RT
    
    klines = load_klines(symbol, start, end)
    oi = load_open_interest(symbol, start, end)
    
    if len(klines) == 0 or len(oi) == 0:
        return BacktestResult('oi_div', symbol, pd.to_datetime(start), pd.to_datetime(end), [])
    
    # Resample OI to 1m (forward fill)
    klines = klines.set_index('timestamp')
    oi = oi.set_index('timestamp')
    df = klines.join(oi[['openInterest']], how='left')
    df['openInterest'] = df['openInterest'].fillna(method='ffill')
    df = df.reset_index()
    
    # Calculate changes
    df['price_chg'] = df['close'].pct_change(5)  # 5-bar change
    df['oi_chg'] = df['openInterest'].pct_change(1)
    
    trades = []
    
    for i in range(10, len(df) - 10):
        price_chg = df.iloc[i]['price_chg']
        oi_chg = df.iloc[i]['oi_chg']
        
        if pd.isna(price_chg) or pd.isna(oi_chg):
            continue
        
        # Signal: OI rising + price falling = short buildup, potential squeeze
        if oi_chg > 0.01 and price_chg < -0.005:
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[i + 10]['close']
            
            position_value = 10000
            price_return = (exit_price - entry_price) / entry_price
            
            fees = position_value * fee_rate
            gross_pnl = price_return * position_value
            net_pnl = gross_pnl - fees
            
            trades.append(Trade(
                entry_time=df.iloc[i]['timestamp'],
                exit_time=df.iloc[i + 10]['timestamp'],
                symbol=symbol,
                side='long',
                entry_price=entry_price,
                exit_price=exit_price,
                size_usd=position_value,
                fees=fees,
                pnl_gross=gross_pnl,
                pnl_net=net_pnl,
                exit_reason='time_exit'
            ))
    
    result = BacktestResult(
        strategy_name='oi_divergence_long',
        symbol=symbol,
        start_date=pd.to_datetime(start),
        end_date=pd.to_datetime(end),
        trades=trades
    )
    result.calculate_metrics()
    return result


if __name__ == '__main__':
    # Test all strategies
    symbols = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'DOGEUSDT']
    start = '2025-01-01'
    end = '2025-12-31'
    
    print('='*70)
    print('STRATEGY TESTING - FRESH ANALYSIS')
    print('='*70)
    print(f'Testing period: {start} to {end}')
    print(f'Symbols: {symbols}')
    print(f'Fee structure: RT taker = {FEE_TAKER_RT*100:.2f}%, RT maker = {FEE_MAKER_RT*100:.2f}%')
    print('='*70)
    
    all_results = []
    
    for symbol in symbols:
        print(f'\nTesting {symbol}...')
        
        # Test FR hold
        for entry in [10, 20, 30]:
            for exit in [5, 10]:
                result = analyze_funding_rate_hold(symbol, start, end, entry, exit, use_maker=False)
                if result.num_trades > 0:
                    all_results.append(result)
                    if result.total_pnl_net > 0:
                        print(f'  FR_HOLD {entry}/{exit}: {result.num_trades} trades, WR={result.win_rate:.1f}%, Net=${result.total_pnl_net:.2f}')
        
        # Test momentum
        for lookback in [10, 20]:
            for thresh in [1.0, 1.5, 2.0]:
                result = analyze_momentum(symbol, start, end, lookback, thresh, use_maker=False)
                if result.num_trades > 0:
                    all_results.append(result)
                    if result.total_pnl_net > 0:
                        print(f'  MOM {lookback}/{thresh}: {result.num_trades} trades, WR={result.win_rate:.1f}%, Net=${result.total_pnl_net:.2f}')
        
        # Test mean reversion
        for lookback in [20, 50]:
            for thresh in [1.5, 2.0, 2.5]:
                result = analyze_mean_reversion(symbol, start, end, lookback, thresh, use_maker=False)
                if result.num_trades > 0:
                    all_results.append(result)
                    if result.total_pnl_net > 0:
                        print(f'  MR {lookback}/{thresh}: {result.num_trades} trades, WR={result.win_rate:.1f}%, Net=${result.total_pnl_net:.2f}')
        
        # Test OI divergence
        result = analyze_oi_divergence(symbol, start, end, use_maker=False)
        if result.num_trades > 0:
            all_results.append(result)
            if result.total_pnl_net > 0:
                print(f'  OI_DIV: {result.num_trades} trades, WR={result.win_rate:.1f}%, Net=${result.total_pnl_net:.2f}')
    
    # Summary of profitable strategies
    print('\n' + '='*70)
    print('SUMMARY - ALL PROFITABLE CONFIGURATIONS')
    print('='*70)
    
    profitable = [r for r in all_results if r.total_pnl_net > 0 and r.num_trades >= 10]
    profitable.sort(key=lambda x: x.total_pnl_net, reverse=True)
    
    if profitable:
        for r in profitable[:10]:
            print(f'{r.strategy_name:30s} | {r.symbol:10s} | {r.num_trades:4d} trades | WR={r.win_rate:5.1f}% | Net=${r.total_pnl_net:8.2f}')
    else:
        print('NO PROFITABLE STRATEGIES FOUND with current configurations')
    
    print('='*70)
