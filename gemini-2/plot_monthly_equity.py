import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-2/portfolio_trades_optimized.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['month'] = df['entry_time'].dt.to_period('M')

# Directory for plots
os.makedirs('/home/ubuntu/Projects/skytrade6/gemini-2/monthly_plots', exist_ok=True)

months = sorted(df['month'].unique())
num_months = len(months)

# Determine grid size for subplots (e.g., 4x4 for 16 months)
cols = 4
rows = math.ceil(num_months / cols)

fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
axes = axes.flatten()

print(f"Generating equity curves for {num_months} months...")

monthly_stats = []

for i, month in enumerate(months):
    month_str = str(month)
    df_month = df[df['month'] == month].copy()
    
    if len(df_month) == 0:
        continue
        
    # Re-simulate the portfolio starting with a fresh $10k each month to see raw performance
    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    MAX_POSITIONS = 5
    
    events = []
    for idx, t in df_month.iterrows():
        events.append({'time': t['entry_time'], 'type': 'enter', 'trade_idx': idx})
        events.append({'time': t['exit_time'], 'type': 'exit', 'trade_idx': idx})
        
    events.sort(key=lambda x: x['time'])
    
    active_trades = {}
    eq_curve = [{'time': df_month['entry_time'].min() - pd.Timedelta(hours=1), 'capital': capital}]
    
    for ev in events:
        tid = ev['trade_idx']
        t = df_month.loc[tid]
        
        if ev['type'] == 'enter':
            if len(active_trades) < MAX_POSITIONS:
                symbols_active = [df_month.loc[at_id, 'symbol'] for at_id in active_trades.keys()]
                if t['symbol'] not in symbols_active:
                    pos_size = capital / MAX_POSITIONS
                    active_trades[tid] = pos_size
        elif ev['type'] == 'exit':
            if tid in active_trades:
                pos_size = active_trades.pop(tid)
                pnl = pos_size * t['net_ret']
                capital += pnl
                eq_curve.append({'time': ev['time'], 'capital': capital})
                
    # End of month point
    eq_curve.append({'time': df_month['exit_time'].max() + pd.Timedelta(hours=1), 'capital': capital})
    
    df_eq = pd.DataFrame(eq_curve)
    
    # Calculate stats
    ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
    if len(df_eq) > 1:
        peak = df_eq['capital'].cummax()
        max_dd = ((peak - df_eq['capital']) / peak).max() * 100
    else:
        max_dd = 0
        
    win_rate = (df_month['net_ret'] > 0).mean() * 100
    
    monthly_stats.append({
        'Month': month_str,
        'Trades': len(df_month),
        'WinRate': win_rate,
        'Return': ret_pct,
        'MaxDD': max_dd
    })
    
    # Plot
    ax = axes[i]
    ax.plot(df_eq['time'], df_eq['capital'], color='blue', linewidth=2)
    ax.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5)
    
    # Format the title with stats
    color = 'green' if ret_pct > 0 else 'red'
    ax.set_title(f"{month_str} | Ret: {ret_pct:+.1f}% | DD: {max_dd:.1f}%\nTrades: {len(df_month)} | WR: {win_rate:.0f}%", 
                color=color, fontsize=10)
    
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('/home/ubuntu/Projects/skytrade6/gemini-2/monthly_plots/monthly_equity_grid.png')
print("Saved grid plot to monthly_plots/monthly_equity_grid.png")

# Also save an individual file for each month for easy viewing
for i, month in enumerate(months):
    month_str = str(month)
    df_month = df[df['month'] == month]
    
    if len(df_month) == 0: continue
    
    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    MAX_POSITIONS = 5
    
    events = []
    for idx, t in df_month.iterrows():
        events.append({'time': t['entry_time'], 'type': 'enter', 'trade_idx': idx})
        events.append({'time': t['exit_time'], 'type': 'exit', 'trade_idx': idx})
        
    events.sort(key=lambda x: x['time'])
    
    active_trades = {}
    eq_curve = [{'time': df_month['entry_time'].min() - pd.Timedelta(hours=1), 'capital': capital}]
    
    for ev in events:
        tid = ev['trade_idx']
        t = df_month.loc[tid]
        if ev['type'] == 'enter':
            if len(active_trades) < MAX_POSITIONS:
                symbols_active = [df_month.loc[at_id, 'symbol'] for at_id in active_trades.keys()]
                if t['symbol'] not in symbols_active:
                    pos_size = capital / MAX_POSITIONS
                    active_trades[tid] = pos_size
        elif ev['type'] == 'exit':
            if tid in active_trades:
                pos_size = active_trades.pop(tid)
                pnl = pos_size * t['net_ret']
                capital += pnl
                eq_curve.append({'time': ev['time'], 'capital': capital})
                
    eq_curve.append({'time': df_month['exit_time'].max() + pd.Timedelta(hours=1), 'capital': capital})
    df_eq = pd.DataFrame(eq_curve)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_eq['time'], df_eq['capital'], color='blue', linewidth=2)
    plt.axhline(y=INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5)
    
    ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
    peak = df_eq['capital'].cummax()
    max_dd = ((peak - df_eq['capital']) / peak).max() * 100
    
    plt.title(f"Liquidation Flush: {month_str} Equity\nReturn: {ret_pct:+.2f}% | Max DD: {max_dd:.2f}% | Trades: {len(df_month)}")
    plt.grid(True, alpha=0.3)
    plt.ylabel('Account Balance ($)')
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/Projects/skytrade6/gemini-2/monthly_plots/{month_str}.png')
    plt.close()

# Print summary table
print("\n--- Isolated Monthly Performance ($10k Restart Each Month) ---")
df_stats = pd.DataFrame(monthly_stats)
print(df_stats.to_string(index=False))

# Update FINDINGS.md with new data
with open('/home/ubuntu/Projects/skytrade6/gemini-2/FINDINGS.md', 'a') as f:
    f.write("\n## Isolated Monthly Performance\n")
    f.write("Assuming a fresh $10,000 start at the beginning of each month to measure raw monthly alpha without the compounding effect of previous months:\n\n")
    f.write("```text\n")
    f.write(df_stats.to_string(index=False) + "\n")
    f.write("```\n")

