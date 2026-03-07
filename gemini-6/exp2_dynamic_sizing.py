import pandas as pd
import numpy as np
import os

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
TOP_12 = ["BERAUSDT", "ZKUSDT", "ZECUSDT", "HUMAUSDT", "INITUSDT", 
          "NEARUSDT", "SAHARAUSDT", "AGLDUSDT", "XRPUSDT", "HBARUSDT", 
          "AAVEUSDT", "LINKUSDT"]

def run_experiment_2():
    print("--- Experiment 2: Volatility-Adjusted Sizing ---")
    all_events = []
    
    for sym in TOP_12:
        file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
        if not os.path.exists(file_path): continue
        
        df = pd.read_parquet(file_path)
        
        rolling_window = 4 * 60
        df['roll_fut_whale'] = df['fut_whale_cvd'].rolling(rolling_window).sum()
        df['roll_fut_retail'] = df['fut_retail_cvd'].rolling(rolling_window).sum()
        
        df['fwd_ret_4h'] = df['price'].shift(-240) / df['price'] - 1
        
        z_window = 3 * 24 * 60
        df['fut_whale_mean'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_whale_std'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_whale_z'] = (df['roll_fut_whale'] - df['fut_whale_mean']) / df['fut_whale_std']
        
        df['fut_retail_mean'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_retail_std'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_retail_z'] = (df['roll_fut_retail'] - df['fut_retail_mean']) / df['fut_retail_std']
        
        df['spot_whale_1h_avg'] = df['spot_whale_cvd'].abs().rolling(60).mean()
        
        # Volatility for sizing
        df['volatility_4h'] = df['price'].rolling(4*60).std() / df['price']
        
        bullish_div = (
            (df['fut_retail_z'] < -1.5) & 
            (df['fut_whale_z'] > 1.5) &
            (df['spot_whale_cvd'] > df['spot_whale_1h_avg'] * 3.0) &
            (df['spot_whale_cvd'] > 0)
        )
        
        def filter_signals(signals, wait_time=120):
            filtered = pd.Series(False, index=signals.index)
            last_sig_time = None
            for i, (idx, val) in enumerate(signals.items()):
                if val:
                    if last_sig_time is None or (idx - last_sig_time).total_seconds() / 60 > wait_time:
                        filtered[idx] = True
                        last_sig_time = idx
            return filtered

        df['bull_sig'] = filter_signals(bullish_div, 120)
        events = df[df['bull_sig']].dropna(subset=['fwd_ret_4h', 'volatility_4h']).copy()
        
        if len(events) > 0:
            events['symbol'] = sym
            all_events.append(events[['symbol', 'fwd_ret_4h', 'volatility_4h']])
            
    df_all = pd.concat(all_events).reset_index(drop=True)
    
    # Baseline: Fixed $10,000 per trade
    base_pnl = df_all['fwd_ret_4h'].sum() * 10000
    base_fees = len(df_all) * 0.0020 * 10000
    net_base_pnl = base_pnl - base_fees
    
    # Dynamic Sizing: Target a specific dollar volatility
    # Instead of $10k flat, we size so that 1 standard deviation move = $200 risk
    # Size = $200 / volatility_4h
    df_all['dynamic_size'] = 200 / df_all['volatility_4h']
    
    # Cap size at $50,000 to be realistic
    df_all['dynamic_size'] = df_all['dynamic_size'].clip(upper=50000)
    
    # Calculate Dynamic PnL
    df_all['gross_dyn_pnl'] = df_all['dynamic_size'] * df_all['fwd_ret_4h']
    df_all['dyn_fees'] = df_all['dynamic_size'] * 0.0020
    df_all['net_dyn_pnl'] = df_all['gross_dyn_pnl'] - df_all['dyn_fees']
    
    # Compare
    print(f"Total Trades: {len(df_all)}")
    print(f"\nBASELINE (Fixed $10k per trade):")
    print(f"Total Capital Deployed: ${len(df_all) * 10000:,.0f}")
    print(f"Net PnL: ${net_base_pnl:,.0f}")
    print(f"Return on Capital Deployed: {(net_base_pnl / (len(df_all) * 10000)) * 100:.2f}%")
    
    print(f"\nDYNAMIC (Volatility-Adjusted Sizing):")
    print(f"Avg Size Per Trade: ${df_all['dynamic_size'].mean():,.0f}")
    print(f"Total Capital Deployed: ${df_all['dynamic_size'].sum():,.0f}")
    print(f"Net PnL: ${df_all['net_dyn_pnl'].sum():,.0f}")
    print(f"Return on Capital Deployed: {(df_all['net_dyn_pnl'].sum() / df_all['dynamic_size'].sum()) * 100:.2f}%")
    
    # Show how sizing varies by coin
    grouped = df_all.groupby('symbol').agg({
        'volatility_4h': 'mean',
        'dynamic_size': 'mean',
        'net_dyn_pnl': 'sum'
    }).sort_values(by='volatility_4h')
    
    print("\nAvg Sizing by Asset (Least Volatile to Most):")
    grouped['volatility_4h'] = grouped['volatility_4h'].apply(lambda x: f"{x*100:.2f}%")
    grouped['dynamic_size'] = grouped['dynamic_size'].apply(lambda x: f"${x:,.0f}")
    grouped['net_dyn_pnl'] = grouped['net_dyn_pnl'].apply(lambda x: f"${x:,.0f}")
    print(grouped.to_string())

if __name__ == "__main__":
    run_experiment_2()
