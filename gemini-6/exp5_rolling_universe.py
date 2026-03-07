import pandas as pd
import numpy as np
import os
from datetime import timedelta
import sys

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
all_files = [f for f in os.listdir(FEAT_DIR) if f.endswith('_1m.parquet')]
ALL_SYMBOLS = [f.replace('_1m.parquet', '') for f in all_files]

def run_rolling_backtest():
    print("--- Experiment 5: Dynamic Rolling Universe Selection ---")
    print("Loading all feature datasets (this might take a minute)...")
    
    # We need to construct a daily dataframe of Volatility and Spot Volume Proxy for every coin
    # We will proxy Spot Volume by taking the absolute sum of Spot Whale + Spot Retail CVD
    
    all_events = []
    daily_metrics_list = []
    
    for sym in ALL_SYMBOLS:
        file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
        df = pd.read_parquet(file_path)
        if len(df) < 30 * 24 * 60: continue
        
        # Calculate engine features
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
        
        # Calculate daily selection metrics
        # Proxy Volume = Abs(Spot Whale CVD) + Abs(Spot Retail CVD)
        df['proxy_vol'] = df['spot_whale_cvd'].abs() + df['spot_retail_cvd'].abs()
        
        daily = df.resample('D').agg({
            'price': ['last', 'std'],
            'proxy_vol': 'sum'
        })
        daily.columns = ['close', 'price_std', 'volume_proxy']
        daily['volatility'] = daily['price_std'] / daily['close']
        daily['symbol'] = sym
        daily = daily.dropna()
        daily_metrics_list.append(daily)
        
        # Trigger logic
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
            entry_price = df.loc[idx, 'price']
            future_df = df.loc[idx:].iloc[1:720]
            exit_candidates = future_df[future_df['fut_retail_z'] > 0]
            
            if len(exit_candidates) > 0:
                exit_idx = exit_candidates.index[0]
                exit_price = df.loc[exit_idx, 'price']
            else:
                if len(future_df) > 0:
                    exit_price = future_df['price'].iloc[-1]
                else:
                    exit_price = entry_price
            
            # Using 24 bps drag (Bybit)
            gross_ret = (exit_price / entry_price) - 1
            net_ret = gross_ret - 0.0024
            dyn_rets.append(net_ret)
            
        events['net_ret'] = dyn_rets
        events['symbol'] = sym
        events['timestamp'] = events.index
        all_events.append(events[['symbol', 'timestamp', 'net_ret']])
        
    df_metrics = pd.concat(daily_metrics_list)
    df_events = pd.concat(all_events).sort_values('timestamp')
    
    # ---------------------------------------------------------
    # The Rolling Selection Algorithm
    # ---------------------------------------------------------
    # Every Monday, we look back 30 days.
    # We rank coins by Volume Proxy * Volatility (High Beta, High Liquidity)
    # We select the Top 15 coins to be our active universe for the next 7 days.
    
    print("\nRunning Weekly Rolling Selection Backtest...")
    
    df_metrics['score'] = df_metrics['volume_proxy'] * df_metrics['volatility']
    
    # Get all unique weeks
    start_date = df_events['timestamp'].min().replace(hour=0, minute=0, second=0)
    end_date = df_events['timestamp'].max()
    
    current_date = start_date + timedelta(days=30) # Need 30 days of burn-in
    
    accepted_trades = []
    rejected_trades = []
    
    while current_date < end_date:
        next_week = current_date + timedelta(days=7)
        
        # Lookback 30 days
        lookback_start = current_date - timedelta(days=30)
        mask = (df_metrics.index >= lookback_start) & (df_metrics.index < current_date)
        hist_metrics = df_metrics[mask]
        
        if len(hist_metrics) > 0:
            # Aggregate score over the 30 days
            coin_scores = hist_metrics.groupby('symbol')['score'].mean().sort_values(ascending=False)
            top_15_universe = coin_scores.head(15).index.tolist()
            
            # Filter trades in the NEXT 7 days against this dynamic universe
            week_mask = (df_events['timestamp'] >= current_date) & (df_events['timestamp'] < next_week)
            week_trades = df_events[week_mask]
            
            for _, trade in week_trades.iterrows():
                if trade['symbol'] in top_15_universe:
                    accepted_trades.append(trade)
                else:
                    rejected_trades.append(trade)
                    
        current_date = next_week
        
    df_accepted = pd.DataFrame(accepted_trades)
    df_rejected = pd.DataFrame(rejected_trades)
    
    print(f"\n--- Results: Dynamic Rolling Top 15 vs The Rest ---")
    print(f"(Using Bybit Execution: 11bps fee + 13bps slippage = 24bps total drag)")
    
    if len(df_accepted) > 0:
        acc_win = (df_accepted['net_ret'] > 0).mean()
        acc_edge = df_accepted['net_ret'].mean()
        acc_pnl = df_accepted['net_ret'].sum() * 10000
        print(f"\nACCEPTED TRADES (Dynamic Top 15 Cult/Liquid Universe):")
        print(f"Trades: {len(df_accepted)}")
        print(f"Win Rate: {acc_win*100:.1f}%")
        print(f"Avg Net Edge: {acc_edge*100:.2f}%")
        print(f"Net PnL ($10k size): ${acc_pnl:,.0f}")
    
    if len(df_rejected) > 0:
        rej_win = (df_rejected['net_ret'] > 0).mean()
        rej_edge = df_rejected['net_ret'].mean()
        rej_pnl = df_rejected['net_ret'].sum() * 10000
        print(f"\nREJECTED TRADES (Illiquid Mid-Caps / Dying Narratives):")
        print(f"Trades: {len(df_rejected)}")
        print(f"Win Rate: {rej_win*100:.1f}%")
        print(f"Avg Net Edge: {rej_edge*100:.2f}%")
        print(f"Net PnL ($10k size): ${rej_pnl:,.0f}")
        
    print("\n--- Compare against Hindsight Static Top 12 ---")
    # We calculated earlier that the Static Hindsight Top 12 generated ~$25,500
    print("Static Hindsight Top 12 PnL: ~$25,500")
    if acc_pnl > 20000:
        print("CONCLUSION: The Dynamic Rolling Filter successfully identifies the correct assets without lookahead bias. The strategy survives Out-Of-Sample.")
    else:
        print("CONCLUSION: Selection Bias was propping up the strategy. Without hindsight, it fails to identify the correct assets.")

if __name__ == "__main__":
    run_rolling_backtest()
