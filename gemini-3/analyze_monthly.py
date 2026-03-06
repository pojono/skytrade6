import pandas as pd
import numpy as np

def analyze_monthly_robustness():
    try:
        df = pd.read_csv('massive_hft_results.csv')
    except Exception as e:
        print("Could not load massive_hft_results.csv:", e)
        return
        
    # We decided to drop the shorts (signal == -1) and only run Longs (signal == 1)
    df = df[df['signal'] == 1].copy()
    print(f"Total Long-only ('Despair Pit') Trades: {len(df)}")
    
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['month'] = df['entry_time'].dt.to_period('M')
    
    monthly_stats = df.groupby('month').agg(
        trades=('net_ret_pct', 'count'),
        win_rate=('net_ret_pct', lambda x: (x > 0).mean()),
        total_ret_pct=('net_ret_pct', 'sum'),
        avg_ret_pct=('net_ret_pct', 'mean')
    )
    
    # Calculate per-month Sharpe (assume zero risk-free rate, daily equivalence roughly scaled by sqrt(trades))
    def calc_sharpe(group):
        rets = group['net_ret_pct']
        if len(rets) < 2 or rets.std() == 0:
            return 0.0
        return np.sqrt(len(rets)) * (rets.mean() / rets.std())
        
    monthly_stats['sharpe'] = df.groupby('month').apply(calc_sharpe)
    
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print("\n=== Monthly Breakdown (Long-Only HFT Sniper) ===")
    print(monthly_stats)
    
    # Summary of robustness
    winning_months = (monthly_stats['total_ret_pct'] > 0).sum()
    total_months = len(monthly_stats)
    
    print(f"\nMonthly Win Rate (Months Profitable): {winning_months}/{total_months} ({(winning_months/total_months)*100:.1f}%)")
    print(f"Average Monthly Return: {monthly_stats['total_ret_pct'].mean():.2f}%")
    print(f"Worst Month Drawdown: {monthly_stats['total_ret_pct'].min():.2f}%")

if __name__ == "__main__":
    analyze_monthly_robustness()
