"""
Aggressive Growth Modeling - Position Sizing & Leverage Analysis
Compare conservative (2%) vs aggressive (5%, 10%, 20%) position sizing
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

# Risk levels to test
RISK_LEVELS = [0.02, 0.05, 0.10, 0.20]  # 2%, 5%, 10%, 20%
LEVERAGE_LEVELS = [1, 2, 3, 5]  # 1x, 2x, 3x, 5x leverage


def load_daily_data(symbol):
    """Load and resample to daily."""
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


def backtest_with_risk(symbol, risk_pct=0.02, leverage=1, initial=1000):
    """
    Run backtest with specific risk level and leverage.
    
    Leverage amplifies both gains and losses:
    - Position size = capital * risk_pct * leverage
    - PnL = (price_change * leverage) * position_value
    - Liquidation risk if loss > (1/leverage) of position
    """
    df = load_daily_data(symbol)
    if df is None or len(df) < 50:
        return None
    
    df['highest'] = df['high'].rolling(5).max().shift(1)
    df['month'] = df.index.to_period('M')
    
    capital = initial
    equity_curve = []
    trades = []
    
    max_capital = capital
    max_drawdown = 0
    liquidation = False
    
    for i in range(6, len(df) - 1):
        price = df['close'].iloc[i]
        breakout_level = df['highest'].iloc[i] * 1.01
        
        # Record equity
        equity_curve.append({
            'date': df.index[i],
            'equity': capital,
            'month': str(df['month'].iloc[i])
        })
        
        if price > breakout_level:
            entry = price
            exit_price = df['close'].iloc[i + 1]
            
            # Calculate position with leverage
            position_value = capital * risk_pct * leverage
            
            # Calculate PnL with leverage
            pnl_pct = (exit_price - entry) / entry * leverage
            pnl_gross = pnl_pct * position_value / leverage  # Base position value
            fees = position_value * FEE_PCT
            pnl_net = pnl_gross - fees
            
            # Check for liquidation (loss > margin)
            margin = position_value / leverage
            if pnl_net < -margin:
                liquidation = True
                capital = 0
                trades.append({
                    'date': df.index[i],
                    'pnl_net': -margin,
                    'liquidation': True
                })
                break
            
            capital += pnl_net
            
            trades.append({
                'date': df.index[i],
                'month': str(df['month'].iloc[i]),
                'entry': entry,
                'exit': exit_price,
                'pnl_net': pnl_net,
                'pnl_pct': pnl_net / capital * 100 if capital > 0 else 0,
                'liquidation': False
            })
            
            # Track drawdown
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    
    # Add final point
    if len(df) > 0 and not liquidation:
        equity_curve.append({
            'date': df.index[-1],
            'equity': capital,
            'month': str(df['month'].iloc[-1])
        })
    
    winning_trades = [t for t in trades if t['pnl_net'] > 0 and not t.get('liquidation', False)]
    
    return {
        'symbol': symbol,
        'risk_pct': risk_pct,
        'leverage': leverage,
        'initial': initial,
        'final': capital,
        'total_return': (capital - initial) / initial * 100 if not liquidation else -100,
        'total_trades': len([t for t in trades if not t.get('liquidation', False)]),
        'win_rate': len(winning_trades) / len([t for t in trades if not t.get('liquidation', False)]) * 100 if trades else 0,
        'max_drawdown': max_drawdown,
        'liquidation': liquidation,
        'equity_curve': equity_curve,
        'trades': trades,
        'avg_trade': np.mean([t['pnl_net'] for t in trades if not t.get('liquidation', False)]) if trades else 0,
        'worst_trade': min([t['pnl_net'] for t in trades if not t.get('liquidation', False)]) if trades else 0
    }


def run_sizing_analysis():
    """Run analysis across all risk levels and coins."""
    print("=" * 120)
    print("AGGRESSIVE GROWTH MODELING - POSITION SIZING ANALYSIS")
    print("=" * 120)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,} per coin")
    print(f"Strategy: Daily breakout (5-day high, 1% threshold, 1-day hold)")
    print(f"Fees: 0.2% round-trip")
    print(f"Leverage: 1x (spot/futures margin)")
    print("-" * 120)
    
    all_results = []
    
    for risk in RISK_LEVELS:
        print(f"\n{'='*120}")
        print(f"  RISK LEVEL: {risk*100:.0f}% per trade")
        print(f"{'='*120}")
        
        risk_results = []
        for coin in VERIFIED_COINS:
            result = backtest_with_risk(coin, risk_pct=risk, leverage=1, initial=INITIAL_CAPITAL)
            if result:
                risk_results.append(result)
                all_results.append(result)
                
                liq_marker = " 💥 LIQUIDATED" if result['liquidation'] else ""
                print(f"\n  {coin}{liq_marker}")
                print(f"    Initial: ${result['initial']:,.2f} -> Final: ${result['final']:,.2f} ({result['total_return']:+.1f}%)")
                print(f"    Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
                print(f"    Max Drawdown: {result['max_drawdown']:.1f}%")
                print(f"    Avg Trade: ${result['avg_trade']:+.2f}, Worst: ${result['worst_trade']:+.2f}")
        
        # Risk level summary
        if risk_results:
            avg_return = np.mean([r['total_return'] for r in risk_results])
            avg_dd = np.mean([r['max_drawdown'] for r in risk_results])
            total_trades = sum([r['total_trades'] for r in risk_results])
            
            print(f"\n  {'-'*120}")
            print(f"  SUMMARY for {risk*100:.0f}% risk:")
            print(f"    Avg Return: {avg_return:+.1f}%")
            print(f"    Avg Max DD: {avg_dd:.1f}%")
            print(f"    Total Trades: {total_trades}")
            print(f"    Return/DD Ratio: {avg_return/max(avg_dd, 0.1):.2f}")
    
    return all_results


def run_leverage_analysis():
    """Run analysis with different leverage levels (at 5% base risk)."""
    print(f"\n{'='*120}")
    print("LEVERAGE ANALYSIS - 5% base risk with varying leverage")
    print(f"{'='*120}")
    print("WARNING: High leverage amplifies both gains AND losses")
    print("Liquidation possible if single trade loss exceeds margin")
    print("-" * 120)
    
    base_risk = 0.05  # 5% base risk
    all_results = []
    
    for lev in LEVERAGE_LEVELS:
        print(f"\n{'='*120}")
        print(f"  LEVERAGE: {lev}x (Effective risk: {base_risk*lev*100:.0f}%)")
        print(f"{'='*120}")
        
        lev_results = []
        for coin in VERIFIED_COINS:
            # Effective risk = base_risk * leverage
            result = backtest_with_risk(coin, risk_pct=base_risk, leverage=lev, initial=INITIAL_CAPITAL)
            if result:
                lev_results.append(result)
                all_results.append(result)
                
                status = "✓" if not result['liquidation'] else "💥 LIQUIDATED"
                print(f"\n  {coin} - {status}")
                if result['liquidation']:
                    print(f"    💥 LIQUIDATED at trade #{len(result['trades'])}")
                    print(f"    Loss: -100% (${result['initial']:,.2f})")
                else:
                    print(f"    Initial: ${result['initial']:,.2f} -> Final: ${result['final']:,.2f} ({result['total_return']:+.1f}%)")
                    print(f"    Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
                    print(f"    Max Drawdown: {result['max_drawdown']:.1f}%")
        
        # Leverage level summary
        if lev_results:
            non_liq = [r for r in lev_results if not r['liquidation']]
            liq_count = len(lev_results) - len(non_liq)
            
            if non_liq:
                avg_return = np.mean([r['total_return'] for r in non_liq])
                avg_dd = np.mean([r['max_drawdown'] for r in non_liq])
                print(f"\n  {'-'*120}")
                print(f"  SUMMARY for {lev}x leverage:")
                print(f"    Coins Liquidated: {liq_count}/{len(lev_results)}")
                if non_liq:
                    print(f"    Avg Return (survivors): {avg_return:+.1f}%")
                    print(f"    Avg Max DD (survivors): {avg_dd:.1f}%")
    
    return all_results


def create_comparison_table(sizing_results, leverage_results):
    """Create comprehensive comparison table."""
    print(f"\n{'='*120}")
    print("COMPREHENSIVE COMPARISON - ALL SCENARIOS")
    print(f"{'='*120}")
    
    # Position sizing comparison
    print("\n  POSITION SIZING COMPARISON (1x leverage):")
    print(f"  {'Risk':>8} {'Coin':>12} {'Return':>10} {'Max DD':>10} {'Trades':>8} {'Win%':>8} {'Ret/DD':>8}")
    print(f"  {'-'*70}")
    
    for risk in RISK_LEVELS:
        risk_label = f"{risk*100:.0f}%"
        for r in sizing_results:
            if r['risk_pct'] == risk and r['leverage'] == 1:
                print(f"  {risk_label:>8} {r['symbol']:>12} {r['total_return']:>+9.1f}% {r['max_drawdown']:>9.1f}% "
                      f"{r['total_trades']:>8} {r['win_rate']:>7.1f}% {r['total_return']/max(r['max_drawdown'],0.1):>7.2f}")
                risk_label = ""  # Only show risk level once
    
    # Leverage comparison
    print(f"\n  LEVERAGE COMPARISON (5% base risk):")
    print(f"  {'Leverage':>10} {'Coin':>12} {'Status':>14} {'Return':>10} {'Max DD':>10}")
    print(f"  {'-'*70}")
    
    for lev in LEVERAGE_LEVELS:
        lev_label = f"{lev}x"
        for r in leverage_results:
            if r['leverage'] == lev and r['risk_pct'] == 0.05:
                status = "LIQUIDATED" if r['liquidation'] else "SURVIVED"
                ret = r['total_return'] if not r['liquidation'] else -100.0
                dd = r['max_drawdown'] if not r['liquidation'] else 100.0
                print(f"  {lev_label:>10} {r['symbol']:>12} {status:>14} {ret:>+9.1f}% {dd:>9.1f}%")
                lev_label = ""
    
    # Risk-adjusted summary
    print(f"\n  RISK-ADJUSTED METRICS SUMMARY:")
    print(f"  {'Scenario':>20} {'Avg Return':>12} {'Avg Max DD':>12} {'Ret/DD':>10} {'Sharpe Est':>12}")
    print(f"  {'-'*75}")
    
    # Group by scenario
    scenarios = {}
    for r in sizing_results:
        key = f"{r['risk_pct']*100:.0f}% risk"
        if key not in scenarios:
            scenarios[key] = []
        scenarios[key].append(r)
    
    for scenario, results in sorted(scenarios.items()):
        avg_ret = np.mean([r['total_return'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        ret_dd = avg_ret / max(avg_dd, 0.1)
        # Approximate Sharpe assuming 2% risk-free, monthly data
        sharpe = (avg_ret - 2) / (avg_dd * 0.5) if avg_dd > 0 else 0
        print(f"  {scenario:>20} {avg_ret:>+11.1f}% {avg_dd:>11.1f}% {ret_dd:>9.2f} {sharpe:>11.2f}")


def run_analysis():
    """Run complete analysis."""
    # Position sizing analysis
    sizing_results = run_sizing_analysis()
    
    # Leverage analysis
    leverage_results = run_leverage_analysis()
    
    # Comparison table
    create_comparison_table(sizing_results, leverage_results)
    
    # Save results
    df_sizing = pd.DataFrame([{
        'symbol': r['symbol'],
        'risk_pct': r['risk_pct'],
        'leverage': r['leverage'],
        'initial': r['initial'],
        'final': r['final'],
        'return': r['total_return'],
        'trades': r['total_trades'],
        'win_rate': r['win_rate'],
        'max_drawdown': r['max_drawdown'],
        'liquidation': r['liquidation']
    } for r in sizing_results + leverage_results])
    
    df_sizing.to_csv('/home/ubuntu/Projects/skytrade6/kimi-1/AGGRESSIVE_GROWTH_ANALYSIS.csv', index=False)
    
    print(f"\n{'='*120}")
    print("✓ Results saved to AGGRESSIVE_GROWTH_ANALYSIS.csv")
    print(f"{'='*120}")
    
    return sizing_results, leverage_results


if __name__ == '__main__':
    sizing_results, leverage_results = run_analysis()
