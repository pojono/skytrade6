"""
Kimi-1 Strategy with Customizable Lookback and Threshold
Implements signal lookback period and threshold breakout customization
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002


def load_daily_data(symbol):
    """Load and resample data to daily timeframe."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 50:
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
    df.sort_index(inplace=True)
    
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return daily


def breakout_backtest(df, lookback=5, threshold=0.01, hold_bars=1, risk=0.02, initial_capital=1000):
    """
    Daily breakout backtest with customizable lookback and threshold.
    
    Parameters:
    -----------
    lookback : int
        Number of days to look back for high (default: 5)
    threshold : float
        Breakout threshold as decimal (default: 0.01 = 1%)
    hold_bars : int
        Number of bars to hold (default: 1)
    risk : float
        Risk per trade as decimal (default: 0.02 = 2%)
    initial_capital : float
        Starting capital (default: 1000)
    """
    if len(df) < lookback + hold_bars + 20:
        return None
    
    df = df.copy()
    df['highest'] = df['high'].rolling(window=lookback).max().shift(1)
    df['month'] = df.index.to_period('M')
    
    capital = initial_capital
    max_capital = capital
    max_drawdown = 0
    
    trades = []
    equity_curve = []
    monthly_data = {}
    
    for i in range(lookback + 1, len(df) - hold_bars):
        price = df['close'].iloc[i]
        breakout_level = df['highest'].iloc[i] * (1 + threshold)
        
        # Record equity
        equity_curve.append({
            'date': df.index[i],
            'equity': capital,
            'month': str(df['month'].iloc[i])
        })
        
        if price > breakout_level:
            entry = price
            exit_price = df['close'].iloc[i + hold_bars]
            
            # Calculate PnL
            pnl_pct = (exit_price - entry) / entry
            position_value = capital * risk
            pnl_gross = pnl_pct * position_value
            fees = position_value * FEE_PCT
            pnl_net = pnl_gross - fees
            
            capital += pnl_net
            
            month = str(df['month'].iloc[i])
            if month not in monthly_data:
                monthly_data[month] = {'trades': 0, 'pnl': 0, 'wins': 0, 'start': capital - pnl_net}
            monthly_data[month]['trades'] += 1
            monthly_data[month]['pnl'] += pnl_net
            if pnl_net > 0:
                monthly_data[month]['wins'] += 1
            monthly_data[month]['end'] = capital
            
            trades.append({
                'date': df.index[i],
                'month': month,
                'pnl_net': pnl_net,
                'entry': entry,
                'exit': exit_price
            })
            
            # Track drawdown
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    # Add final equity point
    if len(df) > 0:
        equity_curve.append({
            'date': df.index[-1],
            'equity': capital,
            'month': str(df['month'].iloc[-1])
        })
    
    winning_trades = [t for t in trades if t['pnl_net'] > 0]
    
    # Calculate monthly stats
    profitable_months = sum(1 for m in monthly_data.values() if m['pnl'] > 0)
    total_months = len(monthly_data)
    
    return {
        'trades': len(trades),
        'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
        'total_pnl': capital - initial_capital,
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'avg_trade': np.mean([t['pnl_net'] for t in trades]) if trades else 0,
        'final_capital': capital,
        'max_drawdown': max_drawdown,
        'profitable_months': profitable_months,
        'total_months': total_months,
        'monthly_consistency': profitable_months / total_months * 100 if total_months > 0 else 0,
        'best_month_pnl': max([m['pnl'] for m in monthly_data.values()]) if monthly_data else 0,
        'worst_month_pnl': min([m['pnl'] for m in monthly_data.values()]) if monthly_data else 0,
        'monthly_data': monthly_data,
        'equity_curve': equity_curve,
        'trades_list': trades
    }


def test_lookback_threshold(symbol, lookback=5, threshold=0.01, risk=0.02, initial=1000):
    """Test specific lookback and threshold combination."""
    df = load_daily_data(symbol)
    if df is None or len(df) < 100:
        return None
    
    result = breakout_backtest(df, lookback=lookback, threshold=threshold, risk=risk, initial_capital=initial)
    if result:
        result['symbol'] = symbol
        result['lookback'] = lookback
        result['threshold'] = threshold
        result['risk'] = risk
        return result
    return None


def run_parameter_sweep(symbol='DOGEUSDT'):
    """Test various lookback and threshold combinations."""
    print("=" * 100)
    print(f"PARAMETER SWEEP - {symbol}")
    print("=" * 100)
    print("Testing different lookback periods and threshold combinations")
    print("-" * 100)
    
    # Parameter grid
    lookbacks = [3, 5, 7, 10, 14]
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.03]
    
    results = []
    
    for lookback in lookbacks:
        for threshold in thresholds:
            result = test_lookback_threshold(symbol, lookback=lookback, threshold=threshold, risk=0.02, initial=1000)
            if result:
                results.append(result)
                status = "✓" if result['total_pnl'] > 0 else "✗"
                print(f"  {status} LB={lookback:2d} TH={threshold*100:4.1f}% | "
                      f"PnL: ${result['total_pnl']:+7.2f} | Return: {result['total_return']:+.1f}% | "
                      f"DD: {result['max_drawdown']:.1f}% | WR: {result['win_rate']:.0f}% | "
                      f"Trades: {result['trades']}")
    
    if results:
        df = pd.DataFrame(results)
        
        # Best by different metrics
        print(f"\n{'='*100}")
        print("  TOP PERFORMERS BY METRIC")
        print(f"{'='*100}")
        
        best_pnl = df.loc[df['total_pnl'].idxmax()]
        print(f"\n  Best Total PnL:")
        print(f"    {best_pnl['symbol']} | LB={best_pnl['lookback']}, TH={best_pnl['threshold']*100:.1f}%")
        print(f"    PnL: ${best_pnl['total_pnl']:.2f}, Return: {best_pnl['total_return']:.1f}%, DD: {best_pnl['max_drawdown']:.1f}%")
        
        best_return = df.loc[df['total_return'].idxmax()]
        print(f"\n  Best Return %:")
        print(f"    {best_return['symbol']} | LB={best_return['lookback']}, TH={best_return['threshold']*100:.1f}%")
        print(f"    Return: {best_return['total_return']:.1f}%, PnL: ${best_return['total_pnl']:.2f}, DD: {best_return['max_drawdown']:.1f}%")
        
        best_sharpe = df.loc[(df['total_return'] / df['max_drawdown']).idxmax()]
        print(f"\n  Best Return/DD Ratio:")
        print(f"    {best_sharpe['symbol']} | LB={best_sharpe['lookback']}, TH={best_sharpe['threshold']*100:.1f}%")
        print(f"    Return/DD: {best_sharpe['total_return']/best_sharpe['max_drawdown']:.2f}, Return: {best_sharpe['total_return']:.1f}%, DD: {best_sharpe['max_drawdown']:.1f}%")
        
        # Parameter sensitivity
        print(f"\n{'='*100}")
        print("  PARAMETER SENSITIVITY")
        print(f"{'='*100}")
        
        lookback_perf = df.groupby('lookback').agg({
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'trades': 'mean'
        }).round(2)
        print(f"\n  By Lookback Period:")
        print(lookback_perf)
        
        threshold_perf = df.groupby('threshold').agg({
            'total_return': 'mean',
            'max_drawdown': 'mean',
            'trades': 'mean'
        }).round(2)
        print(f"\n  By Threshold:")
        print(threshold_perf)
        
        # Save results
        df.to_csv(f'/home/ubuntu/Projects/skytrade6/kimi-1/param_sweep_{symbol}.csv', index=False)
        print(f"\n✓ Results saved to param_sweep_{symbol}.csv")
    
    print(f"{'='*100}")
    return results


def compare_to_baseline(symbol='DOGEUSDT'):
    """Compare custom parameters to baseline (5-day, 1%)."""
    print("=" * 100)
    print(f"BASELINE COMPARISON - {symbol}")
    print("=" * 100)
    
    # Baseline
    baseline = test_lookback_threshold(symbol, lookback=5, threshold=0.01)
    
    # Some promising alternatives
    alternatives = [
        {'lookback': 3, 'threshold': 0.01, 'name': 'Short Lookback (3-day)'},
        {'lookback': 7, 'threshold': 0.01, 'name': 'Long Lookback (7-day)'},
        {'lookback': 5, 'threshold': 0.005, 'name': 'Tight Threshold (0.5%)'},
        {'lookback': 5, 'threshold': 0.02, 'name': 'Wide Threshold (2%)'},
        {'lookback': 3, 'threshold': 0.02, 'name': 'Short+Wide (3-day, 2%)'},
        {'lookback': 7, 'threshold': 0.005, 'name': 'Long+Tight (7-day, 0.5%)'},
    ]
    
    print(f"\n  BASELINE (5-day, 1% threshold):")
    if baseline:
        print(f"    Return: {baseline['total_return']:+.1f}%, PnL: ${baseline['total_pnl']:.2f}, "
              f"DD: {baseline['max_drawdown']:.1f}%, Trades: {baseline['trades']}")
    
    print(f"\n  ALTERNATIVE CONFIGURATIONS:")
    for alt in alternatives:
        result = test_lookback_threshold(symbol, lookback=alt['lookback'], threshold=alt['threshold'])
        if result:
            comparison = "↑" if result['total_pnl'] > baseline['total_pnl'] else "↓" if result['total_pnl'] < baseline['total_pnl'] else "="
            print(f"\n  {alt['name']}:")
            print(f"    {comparison} Return: {result['total_return']:+.1f}%, PnL: ${result['total_pnl']:.2f}, "
                  f"DD: {result['max_drawdown']:.1f}%, Trades: {result['trades']}")
    
    print(f"\n{'='*100}")


if __name__ == '__main__':
    # Run parameter sweep on DOGE
    results = run_parameter_sweep('DOGEUSDT')
    
    # Compare alternatives
    compare_to_baseline('DOGEUSDT')
    
    # Also test on IP and SPX
    print("\n" + "="*100)
    print("Testing on IPUSDT and SPXUSDT...")
    print("="*100)
    run_parameter_sweep('IPUSDT')
    run_parameter_sweep('SPXUSDT')
