import pandas as pd
import numpy as np
import os
from datetime import timedelta

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
all_files = [f for f in os.listdir(FEAT_DIR) if f.endswith('_1m.parquet')]
ALL_SYMBOLS = [f.replace('_1m.parquet', '') for f in all_files]

def run_lookback_sweep_fixed():
    print("--- Experiment 6 (Fixed): Rolling Universe with Strict 1m Forward Execution Shift ---")
    print("Loading datasets and calculating triggers...")
    
    all_events = []
    daily_metrics_list = []
    
    for sym in ALL_SYMBOLS:
        file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
        if not os.path.exists(file_path): continue
        df = pd.read_parquet(file_path)
        if len(df) < 30 * 24 * 60: continue
        
        # Calculate daily selection metrics (Proxy Volume and Volatility)
        df['proxy_vol'] = df['spot_whale_cvd'].abs() + df['spot_retail_cvd'].abs()
        daily = df.resample('D').agg({
            'price': ['last', 'std'],
            'proxy_vol': 'sum'
        })
        daily.columns = ['close', 'price_std', 'proxy_volume']
        daily['volatility'] = daily['price_std'] / daily['close']
        daily['symbol'] = sym
        daily = daily.dropna()
        daily_metrics_list.append(daily)
        
        # Calculate engine features
        rolling_window = 4 * 60
        df['roll_fut_whale'] = df['fut_whale_cvd'].rolling(rolling_window).sum()
        df['roll_fut_retail'] = df['fut_retail_cvd'].rolling(rolling_window).sum()
        
        # Z-Scores are calculated purely on past data (including the current row)
        z_window = 3 * 24 * 60
        df['fut_whale_mean'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_whale_std'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_whale_z'] = (df['roll_fut_whale'] - df['fut_whale_mean']) / df['fut_whale_std']
        
        df['fut_retail_mean'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).mean()
        df['fut_retail_std'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).std()
        df['fut_retail_z'] = (df['roll_fut_retail'] - df['fut_retail_mean']) / df['fut_retail_std']
        
        df['spot_whale_1h_avg'] = df['spot_whale_cvd'].abs().rolling(60).mean()
        
        # The Trigger Condition at index T (which represents data from T to T+59s)
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
            
        dyn_rets = []
        for idx in events.index:
            # === THE CRITICAL FIX ===
            # The signal triggered based on data aggregated during minute T.
            # We cannot execute at minute T. We must execute at the open of minute T+1.
            # 'price' at index T is the CLOSE of minute T. We will use the OPEN of minute T+1.
            # Because our parquet only has 'price' (which is close), the close of T is exactly equal to the open of T+1.
            # HOWEVER, we must calculate forward returns starting from T+1 to ensure no 59-second leakage!
            
            # Shift execution index to T+1
            exec_idx = idx + timedelta(minutes=1)
            
            if exec_idx not in df.index:
                # End of dataset
                continue
                
            entry_price = df.loc[exec_idx, 'price'] # The close of T+1 (realistic execution delayed by 1m)
            # Actually, entering at the CLOSE of T+1 is even more conservative than open of T+1. We'll use Close of T+1 to be brutally safe.
            
            # Look forward up to 12 hours starting from exec_idx
            future_df = df.loc[exec_idx:].iloc[1:720]
            exit_candidates = future_df[future_df['fut_retail_z'] > 0]
            
            if len(exit_candidates) > 0:
                # Exit at the close of the minute it flips
                exit_price = df.loc[exit_candidates.index[0], 'price']
            else:
                exit_price = future_df['price'].iloc[-1] if len(future_df) > 0 else entry_price
            
            # Net Return = Gross - 24 bps (Bybit Execution)
            net_ret = (exit_price / entry_price) - 1 - 0.0024
            dyn_rets.append(net_ret)
            
        # We need to make sure we don't have mismatch in lengths if we skipped end of dataset
        if len(dyn_rets) < len(events):
            events = events.iloc[:len(dyn_rets)]
            
        events['net_ret'] = dyn_rets
        events['symbol'] = sym
        events['timestamp'] = events.index + timedelta(minutes=1) # The true timestamp of the trade execution
        all_events.append(events[['symbol', 'timestamp', 'net_ret']])
        
    df_metrics = pd.concat(daily_metrics_list)
    df_events = pd.concat(all_events).sort_values('timestamp')
    
    # Calculate optimal metric: Volume * Volatility
    df_metrics['score'] = df_metrics['proxy_volume'] * df_metrics['volatility']
    
    def simulate_lookback(lookback_days, update_freq_days=1):
        start_date = df_events['timestamp'].min().replace(hour=0, minute=0, second=0)
        end_date = df_events['timestamp'].max()
        
        # Start trading on the same day for all tests
        trading_start_date = start_date + timedelta(days=30)
        current_date = trading_start_date
        
        accepted_trades = []
        
        while current_date < end_date:
            next_update = current_date + timedelta(days=update_freq_days)
            
            lookback_start = current_date - timedelta(days=lookback_days)
            # STRICT '< current_date' ensures no same-day future leakage into selection metric
            mask = (df_metrics.index >= lookback_start) & (df_metrics.index < current_date)
            hist_metrics = df_metrics[mask]
            
            if len(hist_metrics) > 0:
                coin_scores = hist_metrics.groupby('symbol')['score'].mean().sort_values(ascending=False)
                allowed_coins = [c for c in coin_scores.index if c not in ['BTCUSDT', 'ETHUSDT']]
                top_universe = allowed_coins[:12]
                
                week_mask = (df_events['timestamp'] >= current_date) & (df_events['timestamp'] < next_update)
                week_trades = df_events[week_mask]
                
                for _, trade in week_trades.iterrows():
                    if trade['symbol'] in top_universe:
                        accepted_trades.append(trade)
                        
            current_date = next_update
            
        df_accepted = pd.DataFrame(accepted_trades)
        
        if len(df_accepted) > 0:
            pnl = df_accepted['net_ret'].sum() * 10000
            win = (df_accepted['net_ret'] > 0).mean() * 100
            edge = df_accepted['net_ret'].mean() * 100
            return len(df_accepted), win, edge, pnl
        else:
            return 0, 0.0, 0.0, 0.0
            
    print("\n--- Parameter Sweep: STRICT CAUSALITY (1m Execution Shift) ---")
    print("Execution happens at the CLOSE of Minute T+1, completely eliminating any possibility of 59-second lookahead bias.")
    print("If the edge survives this, the alpha is incredibly robust.")
    print(f"{'Lookback Window':<20} | {'Trades':<6} | {'Win Rate':<8} | {'Avg Edge':<8} | {'Net PnL'}")
    print("-" * 65)
    
    lookbacks = [1, 3, 7, 14, 21, 30]
    
    print("\n[Updated Weekly]")
    for lb in lookbacks:
        trades, win, edge, pnl = simulate_lookback(lookback_days=lb, update_freq_days=7)
        print(f"{str(lb)+' Days':<20} | {trades:<6} | {win:>5.1f}%   | {edge:>5.2f}%   | ${pnl:,.0f}")
        
    print("\n[Updated DAILY (The Hyper-Reactive Solution)]")
    for lb in lookbacks:
        trades, win, edge, pnl = simulate_lookback(lookback_days=lb, update_freq_days=1)
        print(f"{str(lb)+' Days (Daily Upd)':<20} | {trades:<6} | {win:>5.1f}%   | {edge:>5.2f}%   | ${pnl:,.0f}")

if __name__ == "__main__":
    run_lookback_sweep_fixed()
