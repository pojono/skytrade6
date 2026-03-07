import pandas as pd
import numpy as np
import os

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
TOP_12 = ["BERAUSDT", "ZKUSDT", "ZECUSDT", "HUMAUSDT", "INITUSDT", 
          "NEARUSDT", "SAHARAUSDT", "AGLDUSDT", "XRPUSDT", "HBARUSDT", 
          "AAVEUSDT", "LINKUSDT"]

def run_experiment_1():
    print("--- Experiment 1: Dynamic CVD Exit ---")
    results = []
    
    for sym in TOP_12:
        file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
        if not os.path.exists(file_path): continue
        
        df = pd.read_parquet(file_path)
        
        rolling_window = 4 * 60
        df['roll_fut_whale'] = df['fut_whale_cvd'].rolling(rolling_window).sum()
        df['roll_fut_retail'] = df['fut_retail_cvd'].rolling(rolling_window).sum()
        
        z_window = 3 * 24 * 60
        df['fut_whale_mean'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_whale_std'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_whale_z'] = (df['roll_fut_whale'] - df['fut_whale_mean']) / df['fut_whale_std']
        
        df['fut_retail_mean'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_retail_std'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_retail_z'] = (df['roll_fut_retail'] - df['fut_retail_mean']) / df['fut_retail_std']
        
        df['spot_whale_1h_avg'] = df['spot_whale_cvd'].abs().rolling(60).mean()
        
        # Fixed 4h Exit for Baseline
        df['fwd_ret_4h'] = df['price'].shift(-240) / df['price'] - 1
        
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
        events = df[df['bull_sig']].copy()
        
        if len(events) == 0: continue
        
        dynamic_returns = []
        hold_times = []
        
        # Calculate dynamic exits
        for idx in events.index:
            entry_price = df.loc[idx, 'price']
            
            # Look forward up to 12 hours (720 minutes)
            future_df = df.loc[idx:].iloc[1:720]
            
            # Exit condition: Retail CVD flips to positive (FOMO) OR max time reached
            exit_candidates = future_df[future_df['fut_retail_z'] > 0]
            
            if len(exit_candidates) > 0:
                exit_idx = exit_candidates.index[0]
                exit_price = df.loc[exit_idx, 'price']
                hold_time = (exit_idx - idx).total_seconds() / 60
            else:
                # Time stop at 12 hours
                if len(future_df) > 0:
                    exit_idx = future_df.index[-1]
                    exit_price = future_df['price'].iloc[-1]
                    hold_time = len(future_df)
                else:
                    # End of dataset
                    exit_price = entry_price
                    hold_time = 0
                    
            ret = (exit_price / entry_price) - 1
            dynamic_returns.append(ret)
            hold_times.append(hold_time)
            
        events['dyn_ret'] = dynamic_returns
        events['hold_time_mins'] = hold_times
        events = events.dropna(subset=['fwd_ret_4h'])
        
        if len(events) > 0:
            results.append({
                'Symbol': sym,
                'Trades': len(events),
                'Baseline 4h': events['fwd_ret_4h'].mean(),
                'Dynamic Exit': events['dyn_ret'].mean(),
                'Avg Hold (Hrs)': events['hold_time_mins'].mean() / 60,
                'Dyn Win Rate': (events['dyn_ret'] > 0).mean()
            })
            
    df_res = pd.DataFrame(results)
    df_res.loc['TOTAL/AVG'] = [
        'AVERAGE', 
        df_res['Trades'].sum(),
        (df_res['Baseline 4h'] * df_res['Trades']).sum() / df_res['Trades'].sum(),
        (df_res['Dynamic Exit'] * df_res['Trades']).sum() / df_res['Trades'].sum(),
        (df_res['Avg Hold (Hrs)'] * df_res['Trades']).sum() / df_res['Trades'].sum(),
        (df_res['Dyn Win Rate'] * df_res['Trades']).sum() / df_res['Trades'].sum()
    ]
    
    format_pct = lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x
    for col in ['Baseline 4h', 'Dynamic Exit', 'Dyn Win Rate']:
        df_res[col] = df_res[col].apply(format_pct)
        
    df_res['Avg Hold (Hrs)'] = df_res['Avg Hold (Hrs)'].apply(lambda x: f"{x:.1f}h" if isinstance(x, float) else x)
    
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_experiment_1()
