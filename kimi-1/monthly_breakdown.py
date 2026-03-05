"""
Kimi-1 Monthly Breakdown Analysis
Verify no single month outliers in top performers
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002


def load_and_resample(symbol, timeframe='1H'):
    """Load data and resample to target timeframe."""
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 100:
        return None
    
    all_data = []
    for f in files[:300]:  # Load first 300 files
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
    
    # Resample
    resampled = df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    return resampled


def breakout_with_monthly(df, lookback=24, threshold=0.008, hold_bars=4):
    """
    Run breakout strategy and track monthly P&L.
    Returns dict with total results and monthly breakdown.
    """
    if len(df) < lookback + hold_bars + 20:
        return None
    
    df = df.copy()
    df['highest'] = df['high'].rolling(window=lookback).max().shift(1)
    df['month'] = df.index.to_period('M')
    
    capital = 10000
    risk = 0.02
    
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
                'month': month,
                'won': pnl_net > 0
            })
            
            # Track monthly
            month_str = str(month)
            if month_str not in monthly_pnls:
                monthly_pnls[month_str] = {'pnl': 0, 'trades': 0, 'wins': 0}
            monthly_pnls[month_str]['pnl'] += pnl_net
            monthly_pnls[month_str]['trades'] += 1
            if pnl_net > 0:
                monthly_pnls[month_str]['wins'] += 1
    
    if len(trades) < 10:
        return None
    
    total_pnl = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['won'])
    
    # Calculate monthly stats
    months = list(monthly_pnls.keys())
    month_pnls = [monthly_pnls[m]['pnl'] for m in months]
    
    return {
        'total_trades': len(trades),
        'total_pnl': total_pnl,
        'win_rate': wins / len(trades),
        'months': len(months),
        'monthly_data': monthly_pnls,
        'best_month': max(month_pnls) if month_pnls else 0,
        'worst_month': min(month_pnls) if month_pnls else 0,
        'avg_monthly': np.mean(month_pnls) if month_pnls else 0,
        'monthly_std': np.std(month_pnls) if month_pnls else 0,
        'profitable_months': sum(1 for p in month_pnls if p > 0),
        'consistency': sum(1 for p in month_pnls if p > 0) / len(months) if months else 0
    }


def analyze_top_performers():
    """Analyze monthly consistency for top multi-timeframe performers."""
    
    # Top performers from multi-TF test
    top_coins = [
        ('COAIUSDT', '1H', 24, 0.008, 4),
        ('COAIUSDT', '2H', 12, 0.01, 2),
        ('MYXUSDT', '2H', 12, 0.01, 2),
        ('MYXUSDT', '1H', 24, 0.008, 4),
        ('HUSDT', '1H', 24, 0.008, 4),
        ('HUSDT', '2H', 12, 0.01, 2),
        ('AVNTUSDT', '1H', 24, 0.008, 4),
        ('MEMEUSDT', '1H', 24, 0.008, 4),
        ('IPUSDT', '1H', 24, 0.008, 4),
    ]
    
    print("=" * 100)
    print("MONTHLY BREAKDOWN - TOP MULTI-TIMEFRAME PERFORMERS")
    print("=" * 100)
    print("\nChecking for single-month outliers...")
    print("-" * 100)
    
    results = []
    
    for symbol, tf, lookback, threshold, hold in top_coins:
        print(f"\n{symbol} on {tf}...")
        
        df = load_and_resample(symbol, tf)
        if df is None:
            print(f"  ✗ No data")
            continue
        
        result = breakout_with_monthly(df, lookback, threshold, hold)
        if result is None:
            print(f"  ✗ Not enough trades")
            continue
        
        results.append({
            'symbol': symbol,
            'timeframe': tf,
            **result
        })
        
        # Check for outlier
        outlier_ratio = result['best_month'] / result['total_pnl'] if result['total_pnl'] > 0 else 0
        outlier_warning = "⚠️ OUTLIER!" if outlier_ratio > 0.5 else ""
        
        print(f"  Total PnL: ${result['total_pnl']:+.2f} ({result['total_trades']} trades)")
        print(f"  Win rate: {result['win_rate']:.0%}")
        print(f"  Months: {result['months']} | Profitable: {result['profitable_months']} ({result['consistency']:.0%})")
        print(f"  Best month: ${result['best_month']:+.2f} ({outlier_ratio:.0%} of total) {outlier_warning}")
        print(f"  Worst month: ${result['worst_month']:+.2f}")
        print(f"  Avg monthly: ${result['avg_monthly']:+.2f} (±${result['monthly_std']:.2f})")
        
        # Monthly breakdown
        print(f"\n  Monthly breakdown:")
        for month, data in sorted(result['monthly_data'].items()):
            status = "✓" if data['pnl'] > 0 else "✗"
            print(f"    {status} {month}: ${data['pnl']:+7.2f} ({data['trades']:2d}T, {data['wins']}/{data['trades']} wins)")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY - MONTHLY CONSISTENCY")
    print("=" * 100)
    
    if results:
        df_results = pd.DataFrame(results)
        
        # Check for outliers
        outliers = df_results[df_results.apply(
            lambda r: r['best_month'] / r['total_pnl'] > 0.5 if r['total_pnl'] > 0 else False, axis=1
        )]
        
        consistent = df_results[df_results['consistency'] >= 0.6]
        
        print(f"\nTotal tested: {len(df_results)} coin-timeframe combinations")
        print(f"With single-month outliers (>50% from one month): {len(outliers)}")
        print(f"With good monthly consistency (≥60% profitable months): {len(consistent)}")
        
        if len(outliers) > 0:
            print(f"\n⚠️ OUTLIERS (need caution):")
            for _, r in outliers.iterrows():
                ratio = r['best_month'] / r['total_pnl'] if r['total_pnl'] > 0 else 0
                print(f"  {r['symbol']} {r['timeframe']}: {ratio:.0%} from best month")
        
        if len(consistent) > 0:
            print(f"\n✅ CONSISTENT (≥60% profitable months):")
            for _, r in consistent.iterrows():
                print(f"  {r['symbol']} {r['timeframe']}: {r['consistency']:.0%} months profitable")
        
        # Save results
        df_results.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/MONTHLY_BREAKDOWN_RESULTS.csv', index=False)
        print(f"\n✓ Results saved to MONTHLY_BREAKDOWN_RESULTS.csv")
    
    print("\n" + "=" * 100)


if __name__ == '__main__':
    analyze_top_performers()
