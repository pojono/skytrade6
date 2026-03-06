"""
Monthly Breakdown & Equity Curves - 3 Verified Coins
Starting capital: $1000 per coin
Strategy: Daily breakout (5-day high, 1% threshold, 1-day hold)
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


def load_daily_data(symbol):
    """Load 1-minute klines and resample to daily."""
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


def backtest_with_equity(symbol, initial_capital=1000):
    """
    Run backtest with detailed trade log and equity curve.
    """
    df = load_daily_data(symbol)
    if df is None or len(df) < 50:
        return None
    
    df['highest'] = df['high'].rolling(5).max().shift(1)
    df['month'] = df.index.to_period('M')
    
    capital = initial_capital
    equity_curve = []
    trades = []
    monthly_data = {}
    
    for i in range(6, len(df) - 1):
        price = df['close'].iloc[i]
        breakout_level = df['highest'].iloc[i] * 1.01
        
        # Record equity at start of day
        equity_curve.append({
            'date': df.index[i],
            'equity': capital,
            'month': str(df['month'].iloc[i])
        })
        
        if price > breakout_level:
            entry = price
            exit_price = df['close'].iloc[i + 1]
            
            # Calculate PnL
            pnl_pct = (exit_price - entry) / entry
            position_value = capital * 0.02  # 2% risk
            pnl_gross = pnl_pct * position_value
            fees = position_value * FEE_PCT
            pnl_net = pnl_gross - fees
            
            # Update capital
            capital += pnl_net
            
            month = str(df['month'].iloc[i])
            
            trade = {
                'date': df.index[i],
                'month': month,
                'entry': entry,
                'exit': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_net': pnl_net,
                'equity_after': capital,
                'fees': fees
            }
            trades.append(trade)
            
            if month not in monthly_data:
                monthly_data[month] = {'trades': 0, 'pnl': 0, 'wins': 0, 'equity_start': None}
            
            if monthly_data[month]['equity_start'] is None:
                monthly_data[month]['equity_start'] = capital - pnl_net
            
            monthly_data[month]['trades'] += 1
            monthly_data[month]['pnl'] += pnl_net
            if pnl_net > 0:
                monthly_data[month]['wins'] += 1
            monthly_data[month]['equity_end'] = capital
    
    # Add final equity point
    if len(df) > 0:
        equity_curve.append({
            'date': df.index[-1],
            'equity': capital,
            'month': str(df['month'].iloc[-1])
        })
    
    return {
        'symbol': symbol,
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'total_pnl': capital - initial_capital,
        'trades': trades,
        'equity_curve': equity_curve,
        'monthly_data': monthly_data,
        'total_trades': len(trades),
        'win_rate': sum(1 for t in trades if t['pnl_net'] > 0) / len(trades) if trades else 0
    }


def print_monthly_table(result):
    """Print detailed monthly breakdown table."""
    print(f"\n{'='*100}")
    print(f"  {result['symbol']} - Monthly Breakdown & Equity Progression")
    print(f"{'='*100}")
    print(f"  Initial Capital: ${result['initial_capital']:,.2f}")
    print(f"  Final Capital:   ${result['final_capital']:,.2f}")
    print(f"  Total Return:    {result['total_return']:+.2f}%")
    print(f"  Total Trades:    {result['total_trades']}")
    print(f"  Win Rate:        {result['win_rate']*100:.1f}%")
    
    print(f"\n  {'Month':<10} {'Trades':>8} {'Wins':>6} {'Monthly PnL':>14} {'End Equity':>14} {'Return':>10}")
    print(f"  {'-'*80}")
    
    cumulative = result['initial_capital']
    for month in sorted(result['monthly_data'].keys()):
        data = result['monthly_data'][month]
        if data['equity_start'] is None:
            data['equity_start'] = cumulative
        if 'equity_end' not in data or data['equity_end'] is None:
            data['equity_end'] = data['equity_start'] + data['pnl']
        
        monthly_return = (data['equity_end'] - data['equity_start']) / data['equity_start'] * 100
        win_pct = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        
        marker = " ★" if data['pnl'] == max(m['pnl'] for m in result['monthly_data'].values()) else ""
        
        print(f"  {month:<10} {data['trades']:>8} {data['wins']:>6}/{data['trades']:<3} "
              f"${data['pnl']:>+11.2f} ${data['equity_end']:>+11.2f} {monthly_return:>+8.2f}%{marker}")
        cumulative = data['equity_end']
    
    print(f"  {'-'*80}")
    print(f"  {'TOTAL':<10} {result['total_trades']:>8} "
          f"{sum(1 for t in result['trades'] if t['pnl_net'] > 0):>6}/{result['total_trades']:<3} "
          f"${result['total_pnl']:>+11.2f} ${result['final_capital']:>+11.2f} "
          f"{result['total_return']:>+8.2f}%")


def save_equity_curves(results):
    """Save equity curve data to CSV for plotting."""
    for result in results:
        # Daily equity curve
        df_equity = pd.DataFrame(result['equity_curve'])
        df_equity.to_csv(f"/home/ubuntu/Projects/skytrade6/kimi-1/equity_{result['symbol']}.csv", index=False)
        
        # Monthly summary
        monthly_rows = []
        for month, data in sorted(result['monthly_data'].items()):
            monthly_rows.append({
                'month': month,
                'trades': data['trades'],
                'wins': data['wins'],
                'win_rate': data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0,
                'pnl': data['pnl'],
                'equity_start': data.get('equity_start', 0),
                'equity_end': data.get('equity_end', 0)
            })
        df_monthly = pd.DataFrame(monthly_rows)
        df_monthly.to_csv(f"/home/ubuntu/Projects/skytrade6/kimi-1/monthly_{result['symbol']}.csv", index=False)
    
    # Combined portfolio equity
    print(f"\n{'='*100}")
    print("  PORTFOLIO COMBINATION (Equal Weight)")
    print(f"{'='*100}")
    
    # Align all equity curves by date
    all_dates = set()
    for result in results:
        all_dates.update([e['date'] for e in result['equity_curve']])
    all_dates = sorted(all_dates)
    
    portfolio_rows = []
    portfolio_capital = 3000  # $1000 x 3 coins
    
    for date in all_dates:
        daily_pnl = 0
        active_coins = 0
        for result in results:
            # Find equity for this date
            equity_points = [e['equity'] for e in result['equity_curve'] if e['date'] == date]
            if equity_points:
                active_coins += 1
        
        if active_coins > 0:
            portfolio_rows.append({
                'date': date,
                'portfolio_value': sum([e['equity'] for r in results for e in r['equity_curve'] if e['date'] == date]),
                'active_coins': active_coins
            })
    
    if portfolio_rows:
        df_portfolio = pd.DataFrame(portfolio_rows)
        df_portfolio.to_csv("/home/ubuntu/Projects/skytrade6/kimi-1/equity_portfolio.csv", index=False)
        
        final_value = portfolio_rows[-1]['portfolio_value'] if portfolio_rows else 3000
        total_return = (final_value - 3000) / 3000 * 100
        print(f"  Initial Portfolio: $3,000.00")
        print(f"  Final Portfolio:   ${final_value:,.2f}")
        print(f"  Total Return:      {total_return:+.2f}%")
        print(f"  Total PnL:         ${final_value - 3000:+.2f}")


def run_analysis():
    """Run complete analysis for all 3 verified coins."""
    print("=" * 100)
    print("MONTHLY BREAKDOWN & EQUITY CURVES - 3 VERIFIED COINS")
    print("=" * 100)
    print(f"Strategy: Daily Breakout (5-day high, 1% threshold, 1-day hold)")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,} per coin")
    print(f"Risk per Trade: 2%")
    print(f"Fees: 0.2% round-trip")
    print("-" * 100)
    
    results = []
    for coin in VERIFIED_COINS:
        print(f"\nProcessing {coin}...")
        result = backtest_with_equity(coin, INITIAL_CAPITAL)
        if result:
            results.append(result)
            print_monthly_table(result)
    
    # Summary comparison
    print(f"\n{'='*100}")
    print("  SUMMARY COMPARISON - 3 VERIFIED COINS")
    print(f"{'='*100}")
    print(f"  {'Coin':<15} {'Start':>12} {'End':>12} {'Return':>10} {'Trades':>8} {'Win%':>8} {'Max DD*':>10}")
    print(f"  {'-'*85}")
    
    for result in results:
        # Calculate max drawdown
        equity_values = [e['equity'] for e in result['equity_curve']]
        peak = equity_values[0]
        max_dd = 0
        for eq in equity_values:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        print(f"  {result['symbol']:<15} ${result['initial_capital']:>10,.0f} "
              f"${result['final_capital']:>+10,.2f} {result['total_return']:>+8.2f}% "
              f"{result['total_trades']:>8} {result['win_rate']*100:>7.1f}% {max_dd:>9.2f}%")
    
    # Portfolio total
    total_start = sum(r['initial_capital'] for r in results)
    total_end = sum(r['final_capital'] for r in results)
    total_return = (total_end - total_start) / total_start * 100
    total_trades = sum(r['total_trades'] for r in results)
    
    print(f"  {'-'*85}")
    print(f"  {'PORTFOLIO':<15} ${total_start:>10,.0f} ${total_end:>+10,.2f} "
          f"{total_return:>+8.2f}% {total_trades:>8}")
    
    print(f"\n  *Max Drawdown = Peak-to-trough decline during the period")
    
    # Save data
    save_equity_curves(results)
    
    print(f"\n{'='*100}")
    print("  Files saved:")
    for coin in VERIFIED_COINS:
        print(f"    - equity_{coin}.csv (daily equity curve)")
        print(f"    - monthly_{coin}.csv (monthly summary)")
    print(f"    - equity_portfolio.csv (combined portfolio)")
    print(f"{'='*100}")
    
    return results


if __name__ == '__main__':
    results = run_analysis()
