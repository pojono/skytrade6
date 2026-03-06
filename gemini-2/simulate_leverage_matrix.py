import pandas as pd
import numpy as np

# Load trades from the optimized liquidation flush strategy
df = pd.read_csv('/home/ubuntu/Projects/skytrade6/gemini-2/portfolio_trades_optimized.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

events = []
for idx, t in df.iterrows():
    events.append({'time': t['entry_time'], 'type': 'enter', 'trade_idx': idx})
    events.append({'time': t['exit_time'], 'type': 'exit', 'trade_idx': idx})
    
events.sort(key=lambda x: x['time'])

alloc_pcts = [0.05, 0.10, 0.15, 0.20]
leverages = [1, 2, 3, 5]

results = []

for alloc in alloc_pcts:
    for lev in leverages:
        INITIAL_CAPITAL = 10000
        capital = INITIAL_CAPITAL
        MAX_POSITIONS = 5
        
        active_trades = {}
        bankrupt = False
        
        for ev in events:
            if bankrupt: break
                
            tid = ev['trade_idx']
            t = df.loc[tid]
            
            if ev['type'] == 'enter':
                if len(active_trades) < MAX_POSITIONS:
                    symbols_active = [df.loc[at_id, 'symbol'] for at_id in active_trades.keys()]
                    if t['symbol'] not in symbols_active:
                        # Position size = capital * alloc * lev
                        # But margin tied up = capital * alloc
                        # We don't deduct margin, we just calculate the leveraged PnL
                        pos_size = capital * alloc * lev
                        active_trades[tid] = pos_size
            elif ev['type'] == 'exit':
                if tid in active_trades:
                    pos_size = active_trades.pop(tid)
                    # net_ret in df is already (exit - entry)/entry - 0.0020
                    # For a leveraged position, the return on position size is exactly net_ret.
                    # PnL = pos_size * net_ret
                    # Example: $1000 * 3x = $3000 pos_size. net_ret = 5% -> PnL = $150
                    pnl = pos_size * t['net_ret']
                    capital += pnl
                    
                    if capital <= 0:
                        capital = 0
                        bankrupt = True
                        break
                        
        ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
        
        results.append({
            'Alloc %': f"{alloc*100:.0f}%",
            'Leverage': f"{lev}x",
            'Final Capital': f"${capital:,.0f}",
            'Return %': f"{ret_pct:,.1f}%",
            'Bankrupt': bankrupt
        })

df_res = pd.DataFrame(results)
print("=== PORTFOLIO MATRIX: ALLOCATION vs LEVERAGE ===")
print(df_res.to_string(index=False))

# Now, generate monthly breakdowns for a few interesting combinations
combos = [
    (0.10, 1), # 10% alloc, 1x lev (Safe)
    (0.20, 1), # 20% alloc, 1x lev (Base)
    (0.10, 3), # 10% alloc, 3x lev (Aggressive but managed)
    (0.20, 2), # 20% alloc, 2x lev (High Risk)
]

print("\n\n=== MONTHLY BREAKDOWN FOR SELECTED PROFILES ===")

for alloc, lev in combos:
    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    MAX_POSITIONS = 5
    
    active_trades = {}
    bankrupt = False
    
    monthly_pnls = {}
    
    for ev in events:
        if bankrupt: break
            
        tid = ev['trade_idx']
        t = df.loc[tid]
        
        if ev['type'] == 'enter':
            if len(active_trades) < MAX_POSITIONS:
                symbols_active = [df.loc[at_id, 'symbol'] for at_id in active_trades.keys()]
                if t['symbol'] not in symbols_active:
                    pos_size = capital * alloc * lev
                    active_trades[tid] = pos_size
        elif ev['type'] == 'exit':
            if tid in active_trades:
                pos_size = active_trades.pop(tid)
                pnl = pos_size * t['net_ret']
                capital += pnl
                
                m = t['entry_time'].strftime('%Y-%m')
                if m not in monthly_pnls:
                    monthly_pnls[m] = 0
                monthly_pnls[m] += pnl
                
                if capital <= 0:
                    capital = 0
                    bankrupt = True
                    break
                    
    print(f"\nProfile: {alloc*100:.0f}% Capital/Trade | {lev}x Leverage")
    print(f"Final Capital: ${capital:,.0f} | Bankrupt: {bankrupt}")
    print("Month    | Net PnL ($)")
    print("----------------------")
    for m in sorted(monthly_pnls.keys()):
        print(f"{m}  | ${monthly_pnls[m]:>9,.2f}")
