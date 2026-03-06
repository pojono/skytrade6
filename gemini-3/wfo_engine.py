import pandas as pd
import numpy as np
import pickle

def simulate_path(fill_price, price_path, sl_pct, tp_pct):
    # vectorized fast simulation for a single path
    pnls = (price_path - fill_price) / fill_price
    
    sl_hits = np.where(pnls <= -sl_pct)[0]
    tp_hits = np.where(pnls >= tp_pct)[0]
    
    sl_idx = sl_hits[0] if len(sl_hits) > 0 else float('inf')
    tp_idx = tp_hits[0] if len(tp_hits) > 0 else float('inf')
    
    if sl_idx == float('inf') and tp_idx == float('inf'):
        exit_price = price_path[-1]
    elif sl_idx < tp_idx:
        exit_price = price_path[sl_idx]
    else:
        exit_price = price_path[tp_idx]
        
    # Include taker fees 10bps total
    return (exit_price - fill_price) / fill_price - 0.001

def run_wfo():
    with open("wfo_paths.pkl", "rb") as f:
        paths = pickle.load(f)
        
    df = pd.DataFrame([{
        'symbol': p['symbol'],
        'entry_time': p['entry_time'],
        'fill_price': p['fill_price'],
        'path': p['price_path']
    } for p in paths])
    
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df = df.sort_values('entry_time')
    df['month'] = df['entry_time'].dt.to_period('M')
    
    months = sorted(df['month'].unique())
    print(f"Running WFO over {len(months)} months...")
    
    sl_grid = np.arange(0.005, 0.035, 0.005) # 0.5% to 3.0%
    tp_grid = np.arange(0.010, 0.065, 0.005) # 1.0% to 6.0%
    
    # 3-month train, 1-month test
    train_window = 3
    
    oos_trades = []
    
    for i in range(len(months) - train_window):
        train_months = months[i : i+train_window]
        test_month = months[i+train_window]
        
        train_data = df[df['month'].isin(train_months)]
        test_data = df[df['month'] == test_month]
        
        if len(train_data) < 20 or len(test_data) == 0:
            continue
            
        best_sharpe = -999
        best_params = (0.01, 0.03) # fallback
        
        # Grid Search on Train Data
        for sl in sl_grid:
            for tp in tp_grid:
                train_rets = np.array([simulate_path(row['fill_price'], row['path'], sl, tp) for _, row in train_data.iterrows()])
                
                # We need stability, not just absolute return. Optimize for Sharpe.
                if len(train_rets) > 0 and train_rets.std() > 0:
                    sharpe = np.sqrt(len(train_rets)) * (train_rets.mean() / train_rets.std())
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = (sl, tp)
                        
        print(f"Train {train_months[0]} to {train_months[-1]} -> Best Params: SL={best_params[0]*100:.1f}%, TP={best_params[1]*100:.1f}% (Train Sharpe: {best_sharpe:.2f})")
        
        # Apply exactly those parameters to the blindly held Out-Of-Sample month
        for _, row in test_data.iterrows():
            oos_ret = simulate_path(row['fill_price'], row['path'], best_params[0], best_params[1])
            oos_trades.append({
                'month': test_month,
                'symbol': row['symbol'],
                'sl': best_params[0],
                'tp': best_params[1],
                'net_ret_pct': oos_ret * 100
            })
            
    oos_df = pd.DataFrame(oos_trades)
    
    print("\n=======================================================================")
    print("                HONEST OUT-OF-SAMPLE (OOS) RESULTS                     ")
    print("=======================================================================")
    print(f"Total OOS Trades: {len(oos_df)}")
    print(f"Win Rate: {(oos_df['net_ret_pct'] > 0).mean()*100:.1f}%")
    print(f"Total OOS Net Return: {oos_df['net_ret_pct'].sum():.2f}%")
    print(f"Average Return per Trade: {oos_df['net_ret_pct'].mean():.2f}%")
    print(f"OOS Sharpe Ratio: {np.sqrt(len(oos_df)) * (oos_df['net_ret_pct'].mean() / oos_df['net_ret_pct'].std()):.2f}")
    
    monthly = oos_df.groupby('month').agg(
        trades=('net_ret_pct', 'count'),
        win_rate=('net_ret_pct', lambda x: (x>0).mean()),
        ret=('net_ret_pct', 'sum')
    )
    print("\nOOS Monthly Breakdown:")
    print(monthly)

if __name__ == "__main__":
    run_wfo()
