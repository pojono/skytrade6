"""
Monthly Breakdown Analysis - OOS Robust Coins
Verify that OOS robust daily breakout coins are not driven by single-month outliers
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002

# OOS Robust coins from broad universe test
OOS_ROBUST_COINS = [
    'DOGEUSDT', 'ENSOUSDT', 'IPUSDT', 'SPXUSDT', '1000TURBOUSDT',
    'GUNUSDT', 'USELESSUSDT', 'ETHUSDT', 'RESOLVUSDT', 'SNXUSDT',
    '1000PEPEUSDT', 'HBARUSDT', 'IMXUSDT', 'FORMUSDT', 'BTCUSDT'
]


def load_ohlcv_daily(symbol):
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
    
    # Resample to daily
    daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return daily


def breakout_backtest_monthly(df, lookback=5, threshold=0.01, hold_bars=1, risk=0.02):
    """
    Daily breakout backtest with monthly PnL tracking.
    """
    if len(df) < lookback + hold_bars + 20:
        return None
    
    df = df.copy()
    df['highest'] = df['high'].rolling(window=lookback).max().shift(1)
    df['month'] = df.index.to_period('M')
    
    capital = 10000
    trades = []
    monthly_pnls = {}
    
    for i in range(lookback + 1, len(df) - hold_bars):
        price = df['close'].iloc[i]
        breakout_level = df['highest'].iloc[i] * (1 + threshold)
        
        if price > breakout_level:
            entry = price
            exit_price = df['close'].iloc[i + hold_bars]
            month = df['month'].iloc[i]
            
            pnl_pct = (exit_price - entry) / entry
            pnl_gross = pnl_pct * capital * risk
            fees = capital * risk * FEE_PCT
            pnl_net = pnl_gross - fees
            
            trades.append({
                'pnl': pnl_net,
                'month': str(month),
                'date': df.index[i]
            })
            
            if month not in monthly_pnls:
                monthly_pnls[month] = []
            monthly_pnls[month].append(pnl_net)
    
    if len(trades) < 5:
        return None
    
    # Calculate monthly totals
    monthly_summary = {}
    for month, pnls in monthly_pnls.items():
        monthly_summary[str(month)] = {
            'trades': len(pnls),
            'pnl': sum(pnls),
            'avg': sum(pnls) / len(pnls)
        }
    
    all_pnls = [t['pnl'] for t in trades]
    wins = sum(1 for p in all_pnls if p > 0)
    
    # Find best month
    best_month = max(monthly_summary.items(), key=lambda x: x[1]['pnl'])
    worst_month = min(monthly_summary.items(), key=lambda x: x[1]['pnl'])
    total_pnl = sum(all_pnls)
    
    return {
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'total_pnl': total_pnl,
        'avg_trade': sum(all_pnls) / len(trades),
        'monthly_data': monthly_summary,
        'best_month': best_month[0],
        'best_month_pnl': best_month[1]['pnl'],
        'best_month_pct': best_month[1]['pnl'] / total_pnl * 100 if total_pnl != 0 else 0,
        'worst_month': worst_month[0],
        'worst_month_pnl': worst_month[1]['pnl'],
        'profitable_months': sum(1 for m in monthly_summary.values() if m['pnl'] > 0),
        'total_months': len(monthly_summary)
    }


def analyze_oos_robust_coins():
    """Run monthly breakdown on all OOS robust coins."""
    print("=" * 100)
    print("MONTHLY BREAKDOWN - OOS ROBUST COINS")
    print("=" * 100)
    print("Verifying that OOS robust coins are not driven by single-month outliers")
    print("-" * 100)
    
    results = []
    
    for i, symbol in enumerate(OOS_ROBUST_COINS):
        print(f"\n[{i+1}/{len(OOS_ROBUST_COINS)}] {symbol}")
        
        df = load_ohlcv_daily(symbol)
        if df is None:
            print("  - No data")
            continue
        
        result = breakout_backtest_monthly(df)
        if result is None:
            print("  - Insufficient trades")
            continue
        
        results.append({
            'symbol': symbol,
            **result
        })
        
        # Display summary
        status = "✓ CONSISTENT" if result['best_month_pct'] < 50 else "⚠️ OUTLIER RISK"
        print(f"  Total PnL: ${result['total_pnl']:+.2f} | Trades: {result['trades']}")
        print(f"  Profitable months: {result['profitable_months']}/{result['total_months']}")
        print(f"  Best month: {result['best_month']} ${result['best_month_pnl']:+.2f} ({result['best_month_pct']:.0f}% of total)")
        print(f"  Worst month: {result['worst_month']} ${result['worst_month_pnl']:+.2f}")
        print(f"  Status: {status}")
        
        # Show monthly detail if needed
        if result['best_month_pct'] >= 50 or result['profitable_months'] < result['total_months'] // 2:
            print(f"  Monthly breakdown:")
            for month, data in sorted(result['monthly_data'].items()):
                marker = "★" if month == result['best_month'] else ""
                print(f"    {month}: ${data['pnl']:+7.2f} ({data['trades']}T) {marker}")
    
    # Summary analysis
    print("\n" + "=" * 100)
    print("SUMMARY - MONTHLY CONSISTENCY CHECK")
    print("=" * 100)
    
    if results:
        df = pd.DataFrame(results)
        
        # Categorize
        true_robust = df[df['best_month_pct'] < 50]  # Best month < 50% of total
        outlier_driven = df[df['best_month_pct'] >= 50]
        
        print(f"\nTotal OOS robust coins analyzed: {len(results)}")
        print(f"\nTrue Robust (best month < 50%): {len(true_robust)}")
        print(f"Outlier Driven (best month >= 50%): {len(outlier_driven)}")
        
        if len(true_robust) > 0:
            print(f"\n--- TRUE ROBUST COINS (Consistent Monthly Performance) ---")
            for _, row in true_robust.sort_values('best_month_pct').iterrows():
                print(f"  {row['symbol']:15s} ${row['total_pnl']:+7.2f} | Best: {row['best_month_pct']:.0f}% | {row['profitable_months']}/{row['total_months']} months")
        
        if len(outlier_driven) > 0:
            print(f"\n--- OUTLIER DRIVEN (Single Month Dominates) ---")
            for _, row in outlier_driven.sort_values('best_month_pct', ascending=False).iterrows():
                print(f"  {row['symbol']:15s} ${row['total_pnl']:+7.2f} | Best: {row['best_month_pct']:.0f}% | {row['best_month']}")
        
        # Most consistent
        print(f"\n--- MOST CONSISTENT (Highest % Profitable Months) ---")
        df['profitable_pct'] = df['profitable_months'] / df['total_months'] * 100
        most_consistent = df.sort_values('profitable_pct', ascending=False).head(5)
        for _, row in most_consistent.iterrows():
            print(f"  {row['symbol']:15s} {row['profitable_months']}/{row['total_months']} months ({row['profitable_pct']:.0f}%) | ${row['total_pnl']:+7.2f}")
        
        # Save results
        df_export = df[['symbol', 'total_pnl', 'trades', 'win_rate', 'profitable_months', 
                        'total_months', 'best_month', 'best_month_pct', 'worst_month']].copy()
        df_export.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/OOS_ROBUST_MONTHLY_BREAKDOWN.csv', index=False)
        print(f"\n✓ Results saved to OOS_ROBUST_MONTHLY_BREAKDOWN.csv")
    
    print("\n" + "=" * 100)
    return results


if __name__ == '__main__':
    results = analyze_oos_robust_coins()
