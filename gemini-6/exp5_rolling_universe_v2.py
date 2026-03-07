import pandas as pd
import numpy as np
import os
from datetime import timedelta

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"

all_files = [f for f in os.listdir(FEAT_DIR) if f.endswith('_1m.parquet')]
ALL_SYMBOLS = [f.replace('_1m.parquet', '') for f in all_files]

def load_klines(sym):
    kline_path = os.path.join(DATALAKE_DIR, "binance", sym, f"{sym}_kline_1m.csv")
    if not os.path.exists(kline_path):
        # fallback to bybit
        kline_path = os.path.join(DATALAKE_DIR, "bybit", sym, f"{sym}_kline_1m.csv")
    if not os.path.exists(kline_path):
        return None
    try:
        df_k = pd.read_csv(kline_path, usecols=['timestamp', 'close', 'volume'])
        df_k['timestamp'] = pd.to_datetime(df_k['timestamp'])
        df_k.set_index('timestamp', inplace=True)
        # Calculate daily volume in USD
        df_k['usd_vol'] = df_k['close'] * df_k['volume']
        daily = df_k.resample('D').agg({
            'close': ['last', 'std'],
            'usd_vol': 'sum'
        })
        daily.columns = ['close', 'price_std', 'usd_volume']
        daily['volatility'] = daily['price_std'] / daily['close']
        daily['symbol'] = sym
        return daily.dropna()
    except Exception as e:
        print(f"Error loading kline for {sym}: {e}")
        return None

def run_rolling_backtest():
    print("--- Experiment 5b: Dynamic Rolling Universe with Real Volume ---")
    print("Loading datasets...")
    
    all_events = []
    daily_metrics_list = []
    
    for sym in ALL_SYMBOLS:
        daily = load_klines(sym)
        if daily is not None:
            daily_metrics_list.append(daily)
            
        file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
        if not os.path.exists(file_path): continue
        df = pd.read_parquet(file_path)
        if len(df) < 30 * 24 * 60: continue
        
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
                exit_price = df.loc[exit_candidates.index[0], 'price']
            else:
                exit_price = future_df['price'].iloc[-1] if len(future_df) > 0 else entry_price
            
            # Gross Return minus 24 bps drag (Bybit)
            net_ret = (exit_price / entry_price) - 1 - 0.0024
            dyn_rets.append(net_ret)
            
        events['net_ret'] = dyn_rets
        events['symbol'] = sym
        events['timestamp'] = events.index
        all_events.append(events[['symbol', 'timestamp', 'net_ret']])
        
    df_metrics = pd.concat(daily_metrics_list)
    df_events = pd.concat(all_events).sort_values('timestamp')
    
    print("\nRunning Weekly Rolling Selection Backtest...")
    
    # We will try a few selection metrics:
    # 1. Highest Dollar Volume
    # 2. Highest Volatility
    # 3. Volume * Volatility
    
    def simulate_metric(metric_col, top_n=15):
        start_date = df_events['timestamp'].min().replace(hour=0, minute=0, second=0)
        end_date = df_events['timestamp'].max()
        current_date = start_date + timedelta(days=30)
        
        accepted_trades = []
        rejected_trades = []
        
        while current_date < end_date:
            next_week = current_date + timedelta(days=7)
            
            lookback_start = current_date - timedelta(days=30)
            mask = (df_metrics.index >= lookback_start) & (df_metrics.index < current_date)
            hist_metrics = df_metrics[mask]
            
            if len(hist_metrics) > 0:
                coin_scores = hist_metrics.groupby('symbol')[metric_col].mean().sort_values(ascending=False)
                # Exclude BTC/ETH if they are in the list, focus on alts
                allowed_coins = [c for c in coin_scores.index if c not in ['BTCUSDT', 'ETHUSDT']]
                top_universe = allowed_coins[:top_n]
                
                week_mask = (df_events['timestamp'] >= current_date) & (df_events['timestamp'] < next_week)
                week_trades = df_events[week_mask]
                
                for _, trade in week_trades.iterrows():
                    if trade['symbol'] in top_universe:
                        accepted_trades.append(trade)
                    else:
                        rejected_trades.append(trade)
                        
            current_date = next_week
            
        df_accepted = pd.DataFrame(accepted_trades)
        
        if len(df_accepted) > 0:
            pnl = df_accepted['net_ret'].sum() * 10000
            win = (df_accepted['net_ret'] > 0).mean() * 100
            edge = df_accepted['net_ret'].mean() * 100
            print(f"Metric: {metric_col:20} | Trades: {len(df_accepted):4} | Win: {win:5.1f}% | Edge: {edge:5.2f}% | PnL: ${pnl:,.0f}")
        else:
            print(f"Metric: {metric_col:20} | No trades")
            
    df_metrics['vol_times_volatility'] = df_metrics['usd_volume'] * df_metrics['volatility']
    
    print("\nComparing Rolling Selection Metrics (Top 15 Coins):")
    simulate_metric('usd_volume')
    simulate_metric('volatility')
    simulate_metric('vol_times_volatility')

if __name__ == "__main__":
    run_rolling_backtest()
