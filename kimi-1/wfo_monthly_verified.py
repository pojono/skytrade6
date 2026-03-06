"""
WFO Monthly Breakdown - Verified Coins Focus
Fast analysis of DOGE, IP, SPX with proper walk-forward and monthly consistency
"""
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FEE_PCT = 0.002
INITIAL_CAPITAL = 1000
VERIFIED_COINS = ['DOGEUSDT', 'IPUSDT', 'SPXUSDT']

# Test both baseline and optimized parameters
PARAMS_SETS = [
    {'name': 'Baseline (5d, 1%)', 'lookback': 5, 'threshold': 0.01},
    {'name': 'Optimized (10d, 0.5%)', 'lookback': 10, 'threshold': 0.005}
]


def load_daily(symbol):
    files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv'))
    if len(files) < 50:
        return None
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['startTime'], unit='ms')
            dfs.append(df)
        except:
            pass
    
    if not dfs:
        return None
    
    df = pd.concat(dfs)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    
    return df.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


def backtest_split(df, lookback, threshold, start_idx, end_idx):
    """Backtest a single temporal split with monthly breakdown."""
    split_df = df.iloc[start_idx:end_idx].copy()
    if len(split_df) < lookback + 20:
        return None
    
    split_df['highest'] = split_df['high'].rolling(window=lookback).max().shift(1)
    split_df['month'] = split_df.index.to_period('M')
    
    capital = INITIAL_CAPITAL
    max_capital = capital
    max_dd = 0
    trades = []
    monthly = {}
    
    for i in range(lookback + 1, len(split_df) - 1):
        price = split_df['close'].iloc[i]
        breakout = split_df['highest'].iloc[i] * (1 + threshold)
        
        if price > breakout:
            entry = price
            exit_p = split_df['close'].iloc[i + 1]
            pnl_pct = (exit_p - entry) / entry
            pos_val = capital * 0.02
            pnl_net = pnl_pct * pos_val - pos_val * FEE_PCT
            capital += pnl_net
            
            month = str(split_df['month'].iloc[i])
            if month not in monthly:
                monthly[month] = {'trades': 0, 'pnl': 0, 'wins': 0}
            monthly[month]['trades'] += 1
            monthly[month]['pnl'] += pnl_net
            if pnl_net > 0:
                monthly[month]['wins'] += 1
            
            trades.append({'month': month, 'pnl': pnl_net})
            
            if capital > max_capital:
                max_capital = capital
            dd = (max_capital - capital) / max_capital * 100
            if dd > max_dd:
                max_dd = dd
    
    if not trades:
        return None
    
    prof_months = sum(1 for m in monthly.values() if m['pnl'] > 0)
    total_months = len(monthly)
    
    # Check outlier
    if capital - INITIAL_CAPITAL != 0:
        best_month_pct = max(abs(m['pnl']) for m in monthly.values()) / abs(capital - INITIAL_CAPITAL) * 100
    else:
        best_month_pct = 0
    
    return {
        'trades': len(trades),
        'win_rate': sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100,
        'pnl': capital - INITIAL_CAPITAL,
        'return': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        'max_dd': max_dd,
        'profitable_months': prof_months,
        'total_months': total_months,
        'monthly_pct': prof_months / total_months * 100 if total_months > 0 else 0,
        'best_month_pct': best_month_pct,
        'outlier': best_month_pct > 50,
        'range': f"{split_df.index[0].strftime('%Y-%m')} to {split_df.index[-1].strftime('%Y-%m')}"
    }


def wfo_analysis(symbol, params):
    """Run 3-split WFO analysis."""
    df = load_daily(symbol)
    if df is None or len(df) < 100:
        return None
    
    n = len(df)
    splits = [
        (0, int(n * 0.5), "S1_Early"),
        (int(n * 0.25), int(n * 0.75), "S2_Mid"),
        (int(n * 0.5), n, "S3_Late")
    ]
    
    results = []
    for start, end, name in splits:
        r = backtest_split(df, params['lookback'], params['threshold'], start, end)
        if r:
            r['split'] = name
            results.append(r)
    
    if len(results) < 2:
        return None
    
    all_profitable = all(r['pnl'] > 0 for r in results)
    all_consistent = all(r['monthly_pct'] >= 50 for r in results)
    all_outlier_free = all(not r['outlier'] for r in results)
    
    return {
        'symbol': symbol,
        'params': params['name'],
        'splits': results,
        'oos_robust': all_profitable,
        'monthly_robust': all_consistent,
        'outlier_free': all_outlier_free,
        'truly_robust': all_profitable and all_consistent and all_outlier_free,
        'avg_return': np.mean([r['return'] for r in results]),
        'avg_dd': np.mean([r['max_dd'] for r in results])
    }


def main():
    print("=" * 110)
    print("WFO MONTHLY BREAKDOWN - VERIFIED COINS")
    print("=" * 110)
    print("Walk-Forward: 3 temporal splits (Early 50%, Mid 50%, Late 50%)")
    print("Monthly Check: >=50% profitable months per split, no >50% outlier months")
    print("-" * 110)
    
    all_results = []
    
    for params in PARAMS_SETS:
        print(f"\n{'='*110}")
        print(f"  PARAMETERS: {params['name']}")
        print(f"{'='*110}")
        
        for coin in VERIFIED_COINS:
            result = wfo_analysis(coin, params)
            if not result:
                print(f"\n  {coin}: No data")
                continue
            
            print(f"\n  {coin}:")
            for s in result['splits']:
                status = "✓" if s['pnl'] > 0 else "✗"
                cons = "C" if s['monthly_pct'] >= 50 else "?"
                out = "O" if s['outlier'] else ""
                print(f"    [{status}{cons}{out}] {s['split']}: ${s['pnl']:+6.2f} ({s['return']:+.1f}%) | "
                      f"{s['trades']}T | {s['win_rate']:.0f}% WR | "
                      f"{s['monthly_pct']:.0f}% months | Best: {s['best_month_pct']:.0f}%")
            
            badge = "★★★ TRULY ROBUST" if result['truly_robust'] else \
                    "★★ OOS+Monthly" if result['oos_robust'] and result['monthly_robust'] else \
                    "★ OOS Only" if result['oos_robust'] else "✗ Failed"
            print(f"    >>> {badge} | Avg Return: {result['avg_return']:+.1f}% | Avg DD: {result['avg_dd']:.1f}%")
            
            all_results.append(result)
    
    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    
    for params in PARAMS_SETS:
        print(f"\n  {params['name']}:")
        for r in all_results:
            if r['params'] == params['name']:
                status = "✓ PASS" if r['truly_robust'] else "✗ FAIL"
                print(f"    {status} {r['symbol']}: OOS={r['oos_robust']}, Monthly={r['monthly_robust']}, OutlierFree={r['outlier_free']}")
    
    # Save
    flat = []
    for r in all_results:
        for s in r['splits']:
            flat.append({
                'symbol': r['symbol'],
                'params': r['params'],
                'split': s['split'],
                'range': s['range'],
                'pnl': s['pnl'],
                'return': s['return'],
                'trades': s['trades'],
                'win_rate': s['win_rate'],
                'max_dd': s['max_dd'],
                'profitable_months': s['profitable_months'],
                'total_months': s['total_months'],
                'monthly_pct': s['monthly_pct'],
                'best_month_pct': s['best_month_pct'],
                'outlier': s['outlier'],
                'oos_robust': r['oos_robust'],
                'monthly_robust': r['monthly_robust'],
                'truly_robust': r['truly_robust']
            })
    
    pd.DataFrame(flat).to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/WFO_MONTHLY_VERIFIED.csv', index=False)
    print("\n✓ Saved to WFO_MONTHLY_VERIFIED.csv")
    print("=" * 110)


if __name__ == '__main__':
    main()
