import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor
import time

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"
DATES = [f"2025-07-{str(i).zfill(2)}" for i in range(1, 32)]

CONFIG = {
    "SUIUSDT": {"whale_threshold": 10000, "retail_threshold": 500}
}

def process_day(args):
    symbol, date, whale_thresh, retail_thresh = args
    filepath = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades.csv.gz")
    
    if not os.path.exists(filepath):
        return None
        
    try:
        df = pd.read_csv(filepath)
        df['direction'] = np.where(df['is_buyer_maker'], -1, 1)
        df['signed_qty'] = df['quote_qty'] * df['direction']
        
        df['is_whale'] = df['quote_qty'] >= whale_thresh
        df['is_retail'] = df['quote_qty'] <= retail_thresh
        
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        ohlc = df['price'].resample('1min').ohlc()
        total_cvd = df['signed_qty'].resample('1min').sum().rename('total_cvd')
        whale_cvd = df.loc[df['is_whale'], 'signed_qty'].resample('1min').sum().rename('whale_cvd')
        retail_cvd = df.loc[df['is_retail'], 'signed_qty'].resample('1min').sum().rename('retail_cvd')
        
        res = pd.concat([ohlc, total_cvd, whale_cvd, retail_cvd], axis=1)
        res.fillna(0, inplace=True)
        
        res['close'] = res['close'].replace(0, np.nan).ffill()
        mask = res['open'] == 0
        res.loc[mask, 'open'] = res.loc[mask, 'close']
        res.loc[mask, 'high'] = res.loc[mask, 'close']
        res.loc[mask, 'low'] = res.loc[mask, 'close']
        
        res['open'] = res['open'].ffill()
        res['high'] = res['high'].ffill()
        res['low'] = res['low'].ffill()
        
        res['symbol'] = symbol
        res['date'] = date
        
        return res
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def plot_event(df, event_time, title, filename, window_before=240, window_after=120):
    start_time = event_time - pd.Timedelta(minutes=window_before)
    end_time = event_time + pd.Timedelta(minutes=window_after)
    
    mask = (df.index >= start_time) & (df.index <= end_time)
    plot_df = df[mask].copy()
    
    if len(plot_df) == 0:
        return
        
    # Reset CVDs to 0 at the start of the plot window
    plot_df['whale_cvd_cum'] = plot_df['whale_cvd'].cumsum()
    plot_df['retail_cvd_cum'] = plot_df['retail_cvd'].cumsum()
    plot_df['total_cvd_cum'] = plot_df['total_cvd'].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot Price
    ax1.plot(plot_df.index, plot_df['close'], color='black', label='Price')
    ax1.axvline(x=event_time, color='red', linestyle='--', alpha=0.7, label='Divergence Event')
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot CVDs
    ax2.plot(plot_df.index, plot_df['whale_cvd_cum'], color='blue', label='Whale CVD')
    ax2.plot(plot_df.index, plot_df['retail_cvd_cum'], color='orange', label='Retail CVD')
    # ax2.plot(plot_df.index, plot_df['total_cvd_cum'], color='gray', alpha=0.5, label='Total CVD')
    ax2.axvline(x=event_time, color='red', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Cumulative Volume Delta')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%00'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_and_plot(df, symbol):
    rolling_window = 4 * 60
    
    df['roll_whale_cvd'] = df['whale_cvd'].rolling(rolling_window).sum()
    df['roll_retail_cvd'] = df['retail_cvd'].rolling(rolling_window).sum()
    df['roll_high'] = df['high'].rolling(rolling_window).max()
    df['roll_low'] = df['low'].rolling(rolling_window).min()
    
    df['fwd_ret_2h'] = df['close'].shift(-120) / df['close'] - 1
    
    whale_cvd_thresh = df['roll_whale_cvd'].quantile(0.10)
    retail_cvd_thresh = df['roll_retail_cvd'].quantile(0.90)
    
    bearish_div = (
        (df['close'] >= df['roll_high'] * 0.998) & 
        (df['roll_retail_cvd'] > retail_cvd_thresh) & 
        (df['roll_whale_cvd'] < whale_cvd_thresh)
    )
    
    whale_cvd_bull_thresh = df['roll_whale_cvd'].quantile(0.90)
    retail_cvd_bull_thresh = df['roll_retail_cvd'].quantile(0.10)
    
    bullish_div = (
        (df['close'] <= df['roll_low'] * 1.002) & 
        (df['roll_retail_cvd'] < retail_cvd_bull_thresh) & 
        (df['roll_whale_cvd'] > whale_cvd_bull_thresh)
    )
    
    def filter_signals(signals, wait_time=60):
        filtered = pd.Series(False, index=signals.index)
        last_sig_time = None
        for i, (idx, val) in enumerate(signals.items()):
            if val:
                if last_sig_time is None or (idx - last_sig_time).total_seconds() / 60 > wait_time:
                    filtered[idx] = True
                    last_sig_time = idx
        return filtered

    df['bear_sig'] = filter_signals(bearish_div)
    df['bull_sig'] = filter_signals(bullish_div)
    
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot Bearish
    bear_events = df[df['bear_sig']]
    for idx, (time, row) in enumerate(bear_events.iterrows()):
        ret = row['fwd_ret_2h'] * 100
        title = f"{symbol} Bearish Div {time.strftime('%Y-%m-%d %H:%M')} (2h Ret: {ret:.2f}%)"
        filename = f"{out_dir}/{symbol}_bear_{idx}.png"
        plot_event(df, time, title, filename)
        
    # Plot Bullish
    bull_events = df[df['bull_sig']]
    for idx, (time, row) in enumerate(bull_events.iterrows()):
        ret = row['fwd_ret_2h'] * 100
        title = f"{symbol} Bullish Div {time.strftime('%Y-%m-%d %H:%M')} (2h Ret: {ret:.2f}%)"
        filename = f"{out_dir}/{symbol}_bull_{idx}.png"
        plot_event(df, time, title, filename)
        
    print(f"Saved {len(bear_events)} bearish and {len(bull_events)} bullish plots to {out_dir}/")

def main():
    start_time = time.time()
    for symbol, thresholds in CONFIG.items():
        args_list = [(symbol, d, thresholds['whale_threshold'], thresholds['retail_threshold']) for d in DATES]
        
        daily_dfs = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_day, args_list))
            
        for res in results:
            if res is not None:
                daily_dfs.append(res)
                
        if daily_dfs:
            combined_df = pd.concat(daily_dfs)
            combined_df.sort_index(inplace=True)
            analyze_and_plot(combined_df, symbol)
            
    print(f"Completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
