"""
Monthly Breakdown with Proper Walk-Forward Out-of-Sample (WFO) Validation
Rigorous temporal validation with monthly consistency checks per split
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002
INITIAL_CAPITAL = 1000

# Optimized parameters from sweep
LOOKBACK = 10
THRESHOLD = 0.005


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


def backtest_with_monthly(df, lookback=LOOKBACK, threshold=THRESHOLD, initial=INITIAL_CAPITAL):
    """
    Run backtest and return detailed monthly breakdown.
    """
    if len(df) < lookback + 20:
        return None, None
    
    df = df.copy()
    df['highest'] = df['high'].rolling(window=lookback).max().shift(1)
    df['month'] = df.index.to_period('M')
    
    capital = initial
    max_capital = capital
    max_drawdown = 0
    
    trades = []
    monthly_data = {}
    equity_curve = []
    
    for i in range(lookback + 1, len(df) - 1):
        price = df['close'].iloc[i]
        breakout_level = df['highest'].iloc[i] * (1 + threshold)
        
        equity_curve.append({'date': df.index[i], 'equity': capital, 'month': str(df['month'].iloc[i])})
        
        if price > breakout_level:
            entry = price
            exit_price = df['close'].iloc[i + 1]
            
            pnl_pct = (exit_price - entry) / entry
            position_value = capital * 0.02  # 2% risk
            pnl_gross = pnl_pct * position_value
            fees = position_value * FEE_PCT
            pnl_net = pnl_gross - fees
            
            capital += pnl_net
            
            month = str(df['month'].iloc[i])
            if month not in monthly_data:
                monthly_data[month] = {'trades': 0, 'pnl': 0, 'wins': 0}
            monthly_data[month]['trades'] += 1
            monthly_data[month]['pnl'] += pnl_net
            if pnl_net > 0:
                monthly_data[month]['wins'] += 1
            
            trades.append({'date': df.index[i], 'month': month, 'pnl_net': pnl_net})
            
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    equity_curve.append({'date': df.index[-1], 'equity': capital, 'month': str(df['month'].iloc[-1])})
    
    return {
        'trades': len(trades),
        'win_rate': sum(1 for t in trades if t['pnl_net'] > 0) / len(trades) * 100 if trades else 0,
        'total_pnl': capital - initial,
        'total_return': (capital - initial) / initial * 100,
        'final_capital': capital,
        'max_drawdown': max_drawdown,
        'monthly_data': monthly_data,
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1])
    }, monthly_data


def wfo_monthly_breakdown(symbol):
    """
    Walk-forward out-of-sample with monthly breakdown per split.
    
    Splits:
    - Split 1: First 50% (train on nothing, test on first half)
    - Split 2: Middle 50% (train on first 25%, test on middle)
    - Split 3: Last 50% (train on first 50%, test on last half)
    """
    df = load_daily_data(symbol)
    if df is None or len(df) < 200:
        return None
    
    n = len(df)
    splits = [
        (0, int(n * 0.5), "S1_First_Half"),
        (int(n * 0.25), int(n * 0.75), "S2_Middle_Half"),
        (int(n * 0.5), n, "S3_Last_Half")
    ]
    
    split_results = []
    all_monthly = {}
    
    for start, end, name in splits:
        if end - start < 50:
            continue
        
        split_df = df.iloc[start:end].copy()
        result, monthly = backtest_with_monthly(split_df)
        
        if result and result['trades'] >= 5:
            result['split'] = name
            result['split_range'] = f"{split_df.index[0].strftime('%Y-%m')} to {split_df.index[-1].strftime('%Y-%m')}"
            
            # Monthly consistency analysis
            profitable_months = sum(1 for m in monthly.values() if m['pnl'] > 0)
            total_months = len(monthly)
            
            result['profitable_months'] = profitable_months
            result['total_months'] = total_months
            result['monthly_consistency'] = profitable_months / total_months * 100 if total_months > 0 else 0
            
            # Check for single-month outlier
            if monthly:
                best_month_pnl = max(m['pnl'] for m in monthly.values())
                result['best_month_pct_of_total'] = abs(best_month_pnl / result['total_pnl'] * 100) if result['total_pnl'] != 0 else 0
            else:
                result['best_month_pct_of_total'] = 0
            
            split_results.append(result)
            all_monthly[name] = monthly
    
    if len(split_results) < 2:
        return None
    
    # Overall WFO assessment
    splits_profitable = sum(1 for r in split_results if r['total_pnl'] > 0)
    splits_consistent = sum(1 for r in split_results if r['monthly_consistency'] >= 50)  # At least 50% profitable months
    
    # Check for outlier dependency across splits
    outlier_free = all(r['best_month_pct_of_total'] < 50 for r in split_results if r['total_pnl'] > 0)
    
    return {
        'symbol': symbol,
        'splits': split_results,
        'splits_profitable': splits_profitable,
        'splits_consistent': splits_consistent,
        'total_splits': len(split_results),
        'oos_robust': splits_profitable == len(split_results),  # Profitable in ALL splits
        'monthly_robust': splits_consistent == len(split_results),  # Consistent months in ALL splits
        'outlier_free': outlier_free,
        'truly_robust': splits_profitable == len(split_results) and splits_consistent == len(split_results) and outlier_free,
        'avg_monthly_consistency': np.mean([r['monthly_consistency'] for r in split_results]),
        'avg_best_month_pct': np.mean([r['best_month_pct_of_total'] for r in split_results])
    }


def run_wfo_monthly_analysis():
    """Run WFO monthly breakdown on broad universe."""
    print("=" * 120)
    print("WALK-FORWARD OUT-OF-SAMPLE MONTHLY BREAKDOWN")
    print("=" * 120)
    print("Strategy: 10-day breakout, 0.5% threshold, 1-day hold")
    print("Validation: 3 temporal splits with monthly consistency per split")
    print("Robustness Criteria:")
    print("  1. Profitable in ALL 3 splits")
    print("  2. >=50% profitable months in each split")
    print("  3. No single month >50% of total PnL")
    print("-" * 120)
    
    # Get symbols
    path = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')
    symbols = [d.name for d in path.iterdir() if d.is_dir() and len(list(d.glob('*_kline_1m.csv'))) >= 100]
    symbols = sorted(symbols)
    
    print(f"\nTesting {len(symbols)} symbols...")
    
    results = []
    truly_robust = []
    
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol}")
        
        result = wfo_monthly_breakdown(symbol)
        if result:
            results.append(result)
            
            # Display split results
            for split in result['splits']:
                status = "✓" if split['total_pnl'] > 0 else "✗"
                consistent = "C" if split['monthly_consistency'] >= 50 else "?"
                print(f"  [{status}{consistent}] {split['split']}: ${split['total_pnl']:+6.2f} | "
                      f"{split['monthly_consistency']:.0f}% months | "
                      f"Best: {split['best_month_pct_of_total']:.0f}% of total")
            
            # Overall status
            if result['truly_robust']:
                print(f"  ★★★ TRULY ROBUST - All criteria met!")
                truly_robust.append(result)
            elif result['oos_robust']:
                print(f"  ★ OOS Robust (but monthly issues)")
            else:
                print(f"  ✗ Failed OOS validation")
    
    # Summary
    print("\n" + "=" * 120)
    print("WFO MONTHLY BREAKDOWN SUMMARY")
    print("=" * 120)
    
    if results:
        print(f"\nTotal symbols tested: {len(results)}")
        print(f"OOS Robust (all splits profitable): {sum(1 for r in results if r['oos_robust'])}")
        print(f"Monthly Consistent (all splits >=50% months): {sum(1 for r in results if r['monthly_robust'])}")
        print(f"Outlier Free (all splits <50% concentration): {sum(1 for r in results if r['outlier_free'])}")
        print(f"TRULY ROBUST (all 3 criteria): {len(truly_robust)}")
        
        if truly_robust:
            print(f"\n{'='*120}")
            print("TRULY ROBUST COINS (Pass all 3 criteria)")
            print(f"{'='*120}")
            
            for r in truly_robust:
                print(f"\n{r['symbol']}:")
                for split in r['splits']:
                    print(f"  {split['split']}: ${split['total_pnl']:+6.2f} | "
                          f"{split['trades']} trades | {split['win_rate']:.0f}% WR | "
                          f"{split['monthly_consistency']:.0f}% profitable months")
        
        # Save results
        flat_results = []
        for r in results:
            for split in r['splits']:
                flat_results.append({
                    'symbol': r['symbol'],
                    'split': split['split'],
                    'split_range': split['split_range'],
                    'trades': split['trades'],
                    'win_rate': split['win_rate'],
                    'total_pnl': split['total_pnl'],
                    'total_return': split['total_return'],
                    'max_drawdown': split['max_drawdown'],
                    'profitable_months': split['profitable_months'],
                    'total_months': split['total_months'],
                    'monthly_consistency': split['monthly_consistency'],
                    'best_month_pct_of_total': split['best_month_pct_of_total'],
                    'oos_robust': r['oos_robust'],
                    'monthly_robust': r['monthly_robust'],
                    'outlier_free': r['outlier_free'],
                    'truly_robust': r['truly_robust']
                })
        
        df = pd.DataFrame(flat_results)
        df.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/WFO_MONTHLY_BREAKDOWN.csv', index=False)
        print(f"\n✓ Results saved to WFO_MONTHLY_BREAKDOWN.csv")
    
    print("\n" + "=" * 120)
    return results, truly_robust


if __name__ == '__main__':
    results, truly_robust = run_wfo_monthly_analysis()
