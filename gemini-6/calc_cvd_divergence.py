import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import time

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"
# Use a 31-day sample to get enough events (July 2025)
DATES = [f"2025-07-{str(i).zfill(2)}" for i in range(1, 32)]

CONFIG = {
    "SOLUSDT": {"whale_threshold": 25000, "retail_threshold": 1000},
    "SUIUSDT": {"whale_threshold": 10000, "retail_threshold": 500}
}

def process_day(args):
    symbol, date, whale_thresh, retail_thresh = args
    filepath = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades.csv.gz")
    
    if not os.path.exists(filepath):
        return None
        
    try:
        # Load trades
        # Columns: id, price, qty, quote_qty, time, is_buyer_maker
        df = pd.read_csv(filepath)
        
        # is_buyer_maker = True means the maker was the buyer, meaning the taker was the SELLER.
        # So is_buyer_maker == True -> Market Sell (-)
        # is_buyer_maker == False -> Market Buy (+)
        df['direction'] = np.where(df['is_buyer_maker'], -1, 1)
        df['signed_qty'] = df['quote_qty'] * df['direction']
        
        # Classify trades
        df['is_whale'] = df['quote_qty'] >= whale_thresh
        df['is_retail'] = df['quote_qty'] <= retail_thresh
        
        # Convert time to datetime minute bins
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Resample to 1-minute bars
        # Price: OHLC
        ohlc = df['price'].resample('1min').ohlc()
        
        # CVDs
        # Total CVD
        total_cvd = df['signed_qty'].resample('1min').sum().rename('total_cvd')
        
        # Whale CVD
        whale_cvd = df.loc[df['is_whale'], 'signed_qty'].resample('1min').sum().rename('whale_cvd')
        
        # Retail CVD
        retail_cvd = df.loc[df['is_retail'], 'signed_qty'].resample('1min').sum().rename('retail_cvd')
        
        # Combine
        res = pd.concat([ohlc, total_cvd, whale_cvd, retail_cvd], axis=1)
        res.fillna(0, inplace=True) # Fill missing CVDs with 0
        
        # Forward fill prices if there are empty minutes
        res['close'] = res['close'].replace(0, np.nan).ffill()
        
        # Manually fill OHLC with close where 0
        mask = res['open'] == 0
        res.loc[mask, 'open'] = res.loc[mask, 'close']
        res.loc[mask, 'high'] = res.loc[mask, 'close']
        res.loc[mask, 'low'] = res.loc[mask, 'close']
        
        # Then forward fill remaining NaNs
        res['open'] = res['open'].ffill()
        res['high'] = res['high'].ffill()
        res['low'] = res['low'].ffill()
        
        res['symbol'] = symbol
        res['date'] = date
        
        return res
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def analyze_divergences(df, symbol):
    """
    Finds divergence events and calculates forward returns.
    """
    # Calculate cumulative CVDs for the whole period (or rolling)
    # Let's use a rolling 4-hour window for CVD to see the local build-up
    rolling_window = 4 * 60 # 4 hours in minutes
    
    df['roll_whale_cvd'] = df['whale_cvd'].rolling(rolling_window).sum()
    df['roll_retail_cvd'] = df['retail_cvd'].rolling(rolling_window).sum()
    
    # Calculate rolling high/low for price context
    df['roll_high'] = df['high'].rolling(rolling_window).max()
    df['roll_low'] = df['low'].rolling(rolling_window).min()
    
    # Forward returns (1h, 2h, 4h)
    df['fwd_ret_1h'] = df['close'].shift(-60) / df['close'] - 1
    df['fwd_ret_2h'] = df['close'].shift(-120) / df['close'] - 1
    df['fwd_ret_4h'] = df['close'].shift(-240) / df['close'] - 1
    
    # --- FIX: STRICTLY BACKWARD-LOOKING THRESHOLDS ---
    # We use a 3-day (4320 mins) rolling window to calculate the mean and std dev 
    # of the rolling 4h CVD, creating a localized Z-score with ZERO lookahead bias.
    z_window = 3 * 24 * 60
    
    df['whale_cvd_mean'] = df['roll_whale_cvd'].rolling(z_window, min_periods=rolling_window).mean()
    df['whale_cvd_std'] = df['roll_whale_cvd'].rolling(z_window, min_periods=rolling_window).std()
    df['whale_cvd_z'] = (df['roll_whale_cvd'] - df['whale_cvd_mean']) / df['whale_cvd_std']
    
    df['retail_cvd_mean'] = df['roll_retail_cvd'].rolling(z_window, min_periods=rolling_window).mean()
    df['retail_cvd_std'] = df['roll_retail_cvd'].rolling(z_window, min_periods=rolling_window).std()
    df['retail_cvd_z'] = (df['roll_retail_cvd'] - df['retail_cvd_mean']) / df['retail_cvd_std']
    
    # Define Smart Money Bearish Divergence
    # 1. Price is near the 4h high (within 0.2%)
    # 2. Retail has been net buying aggressively (Z-score > 1.5)
    # 3. Whales have been net selling heavily (Z-score < -1.5)
    bearish_div = (
        (df['close'] >= df['roll_high'] * 0.998) & 
        (df['retail_cvd_z'] > 1.5) & 
        (df['whale_cvd_z'] < -1.5)
    )
    
    # Define Smart Money Bullish Divergence
    # 1. Price is near the 4h low
    # 2. Retail is net selling heavily (Z-score < -1.5)
    # 3. Whales are net buying heavily (Z-score > 1.5)
    bullish_div = (
        (df['close'] <= df['roll_low'] * 1.002) & 
        (df['retail_cvd_z'] < -1.5) & 
        (df['whale_cvd_z'] > 1.5)
    )
    
    # Clean up overlapping signals (only take the first signal in a 60 min window)
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
    
    # Results
    print(f"\n--- {symbol} Divergence Results (7 days) ---")
    
    # Bearish
    bear_events = df[df['bear_sig']]
    print(f"Bearish Divergences (Whale Sell, Retail Buy at Highs): {len(bear_events)}")
    if len(bear_events) > 0:
        print(f"  Avg 1h Fwd Return: {bear_events['fwd_ret_1h'].mean()*100:.2f}% (Win Rate: {(bear_events['fwd_ret_1h'] < 0).mean()*100:.1f}%)")
        print(f"  Avg 2h Fwd Return: {bear_events['fwd_ret_2h'].mean()*100:.2f}% (Win Rate: {(bear_events['fwd_ret_2h'] < 0).mean()*100:.1f}%)")
        print(f"  Avg 4h Fwd Return: {bear_events['fwd_ret_4h'].mean()*100:.2f}% (Win Rate: {(bear_events['fwd_ret_4h'] < 0).mean()*100:.1f}%)")
        
    # Bullish
    bull_events = df[df['bull_sig']]
    print(f"\nBullish Divergences (Whale Buy, Retail Sell at Lows): {len(bull_events)}")
    if len(bull_events) > 0:
        print(f"  Avg 1h Fwd Return: {bull_events['fwd_ret_1h'].mean()*100:.2f}% (Win Rate: {(bull_events['fwd_ret_1h'] > 0).mean()*100:.1f}%)")
        print(f"  Avg 2h Fwd Return: {bull_events['fwd_ret_2h'].mean()*100:.2f}% (Win Rate: {(bull_events['fwd_ret_2h'] > 0).mean()*100:.1f}%)")
        print(f"  Avg 4h Fwd Return: {bull_events['fwd_ret_4h'].mean()*100:.2f}% (Win Rate: {(bull_events['fwd_ret_4h'] > 0).mean()*100:.1f}%)")

def main():
    print("Starting CVD Divergence Analysis...")
    start_time = time.time()
    
    for symbol, thresholds in CONFIG.items():
        print(f"\nProcessing {symbol}...")
        
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
            
            analyze_divergences(combined_df, symbol)
            
    print(f"\nCompleted in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
