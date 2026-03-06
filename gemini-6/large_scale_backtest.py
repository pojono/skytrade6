import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime, timedelta

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"

# Top 15 liquid/volatile altcoins to test
SYMBOLS = [
    "SOLUSDT", "SUIUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT", 
    "WLDUSDT", "APTUSDT", "ARBUSDT", "AAVEUSDT", "INJUSDT",
    "RNDRUSDT", "TIAUSDT", "SEIUSDT", "OPUSDT", "FETUSDT"
]

# Generate date range: 2 full months (July - Aug 2025)
start_date = datetime(2025, 7, 1)
end_date = datetime(2025, 8, 31)
DATES = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]

def process_day(args):
    symbol, date = args
    filepath = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades.csv.gz")
    
    if not os.path.exists(filepath):
        return None
        
    try:
        # Load trades efficiently
        df = pd.read_csv(filepath, usecols=['price', 'quote_qty', 'time', 'is_buyer_maker'])
        
        # Calculate dynamic thresholds per day (approximating 7-day rolling by using daily quantiles to save RAM)
        # Using 98th percentile for Whales, 20th percentile for Retail
        whale_thresh = df['quote_qty'].quantile(0.98)
        retail_thresh = df['quote_qty'].quantile(0.20)
        
        df['direction'] = np.where(df['is_buyer_maker'], -1, 1)
        df['signed_qty'] = df['quote_qty'] * df['direction']
        
        df['is_whale'] = df['quote_qty'] >= whale_thresh
        df['is_retail'] = df['quote_qty'] <= retail_thresh
        
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Resample
        ohlc = df['price'].resample('1min').ohlc()
        total_cvd = df['signed_qty'].resample('1min').sum().rename('total_cvd')
        whale_cvd = df.loc[df['is_whale'], 'signed_qty'].resample('1min').sum().rename('whale_cvd')
        retail_cvd = df.loc[df['is_retail'], 'signed_qty'].resample('1min').sum().rename('retail_cvd')
        
        res = pd.concat([ohlc, total_cvd, whale_cvd, retail_cvd], axis=1)
        res.fillna(0, inplace=True)
        
        # Forward fill
        res['close'] = res['close'].replace(0, np.nan).ffill()
        mask = res['open'] == 0
        res.loc[mask, 'open'] = res.loc[mask, 'close']
        res.loc[mask, 'high'] = res.loc[mask, 'close']
        res.loc[mask, 'low'] = res.loc[mask, 'close']
        
        res['open'] = res['open'].ffill()
        res['high'] = res['high'].ffill()
        res['low'] = res['low'].ffill()
        
        # We don't return the raw dataframe to save RAM, we just return the minute bars
        return res
        
    except Exception as e:
        # Some coins might not have data for a specific day, ignore silently
        return None

def analyze_symbol(symbol):
    print(f"[{symbol}] Loading and aggregating 2 months of tick data...")
    
    args_list = [(symbol, d) for d in DATES]
    
    daily_dfs = []
    # Use max_workers=4 to not blow up RAM (each file is ~30MB, but expands in memory)
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_day, args_list))
        
    for res in results:
        if res is not None:
            daily_dfs.append(res)
            
    if not daily_dfs:
        print(f"[{symbol}] No data found.")
        return None
        
    df = pd.concat(daily_dfs)
    df.sort_index(inplace=True)
    
    print(f"[{symbol}] Analyzing divergences...")
    rolling_window = 4 * 60 # 4 hours
    
    df['roll_whale_cvd'] = df['whale_cvd'].rolling(rolling_window).sum()
    df['roll_retail_cvd'] = df['retail_cvd'].rolling(rolling_window).sum()
    df['roll_high'] = df['high'].rolling(rolling_window).max()
    df['roll_low'] = df['low'].rolling(rolling_window).min()
    
    df['fwd_ret_1h'] = df['close'].shift(-60) / df['close'] - 1
    df['fwd_ret_2h'] = df['close'].shift(-120) / df['close'] - 1
    df['fwd_ret_4h'] = df['close'].shift(-240) / df['close'] - 1
    
    z_window = 3 * 24 * 60
    
    df['whale_cvd_mean'] = df['roll_whale_cvd'].rolling(z_window, min_periods=rolling_window).mean()
    df['whale_cvd_std'] = df['roll_whale_cvd'].rolling(z_window, min_periods=rolling_window).std()
    df['whale_cvd_z'] = (df['roll_whale_cvd'] - df['whale_cvd_mean']) / df['whale_cvd_std']
    
    df['retail_cvd_mean'] = df['roll_retail_cvd'].rolling(z_window, min_periods=rolling_window).mean()
    df['retail_cvd_std'] = df['roll_retail_cvd'].rolling(z_window, min_periods=rolling_window).std()
    df['retail_cvd_z'] = (df['roll_retail_cvd'] - df['retail_cvd_mean']) / df['retail_cvd_std']
    
    bearish_div = (
        (df['close'] >= df['roll_high'] * 0.998) & 
        (df['retail_cvd_z'] > 1.5) & 
        (df['whale_cvd_z'] < -1.5)
    )
    
    bullish_div = (
        (df['close'] <= df['roll_low'] * 1.002) & 
        (df['retail_cvd_z'] < -1.5) & 
        (df['whale_cvd_z'] > 1.5)
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

    # Wait 2 hours between signals to avoid double counting the same cluster
    df['bear_sig'] = filter_signals(bearish_div, 120)
    df['bull_sig'] = filter_signals(bullish_div, 120)
    
    bear_events = df[df['bear_sig']].dropna(subset=['fwd_ret_4h'])
    bull_events = df[df['bull_sig']].dropna(subset=['fwd_ret_4h'])
    
    return {
        'symbol': symbol,
        'bear_count': len(bear_events),
        'bear_1h': bear_events['fwd_ret_1h'].mean() if len(bear_events) > 0 else 0,
        'bear_2h': bear_events['fwd_ret_2h'].mean() if len(bear_events) > 0 else 0,
        'bear_4h': bear_events['fwd_ret_4h'].mean() if len(bear_events) > 0 else 0,
        'bear_wr': (bear_events['fwd_ret_4h'] < 0).mean() if len(bear_events) > 0 else 0,
        'bull_count': len(bull_events),
        'bull_1h': bull_events['fwd_ret_1h'].mean() if len(bull_events) > 0 else 0,
        'bull_2h': bull_events['fwd_ret_2h'].mean() if len(bull_events) > 0 else 0,
        'bull_4h': bull_events['fwd_ret_4h'].mean() if len(bull_events) > 0 else 0,
        'bull_wr': (bull_events['fwd_ret_4h'] > 0).mean() if len(bull_events) > 0 else 0,
    }

def main():
    start_time = time.time()
    
    all_results = []
    for symbol in SYMBOLS:
        res = analyze_symbol(symbol)
        if res:
            all_results.append(res)
            
    print("\n" + "="*60)
    print("FINAL RESULTS (July-August 2025)")
    print("="*60)
    
    total_bear_trades = sum(r['bear_count'] for r in all_results)
    total_bull_trades = sum(r['bull_count'] for r in all_results)
    
    # Calculate weighted averages based on trade count
    if total_bear_trades > 0:
        avg_bear_4h = sum(r['bear_4h'] * r['bear_count'] for r in all_results) / total_bear_trades
        avg_bear_wr = sum(r['bear_wr'] * r['bear_count'] for r in all_results) / total_bear_trades
    else:
        avg_bear_4h = 0
        avg_bear_wr = 0
        
    if total_bull_trades > 0:
        avg_bull_4h = sum(r['bull_4h'] * r['bull_count'] for r in all_results) / total_bull_trades
        avg_bull_wr = sum(r['bull_wr'] * r['bull_count'] for r in all_results) / total_bull_trades
    else:
        avg_bull_4h = 0
        avg_bull_wr = 0
        
    df_res = pd.DataFrame(all_results)
    df_res.set_index('symbol', inplace=True)
    
    # Formatting
    format_pct = lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "0.00%"
    
    for col in ['bear_1h', 'bear_2h', 'bear_4h', 'bear_wr', 'bull_1h', 'bull_2h', 'bull_4h', 'bull_wr']:
        df_res[col] = df_res[col].apply(format_pct)
        
    print(df_res[['bear_count', 'bear_4h', 'bear_wr', 'bull_count', 'bull_4h', 'bull_wr']].to_string())
    
    print("\n--- AGGREGATE SUMMARY ---")
    print(f"Total Bearish Divergences (Whale Sell/Retail Buy): {total_bear_trades}")
    print(f"  Avg 4h Return: {avg_bear_4h*100:.2f}% (Expected Negative)")
    print(f"  Win Rate (4h): {avg_bear_wr*100:.1f}%")
    print()
    print(f"Total Bullish Divergences (Whale Buy/Retail Sell): {total_bull_trades}")
    print(f"  Avg 4h Return: {avg_bull_4h*100:.2f}% (Expected Positive)")
    print(f"  Win Rate (4h): {avg_bull_wr*100:.1f}%")
    print(f"\nExecution time: {(time.time() - start_time) / 60:.2f} minutes")
    
    # Save to MD report
    with open("OOS_RESULTS.md", "w") as f:
        f.write("# Tick-Level Smart Money Divergence\n")
        f.write("## Out-Of-Sample Validation (July - August 2025)\n\n")
        f.write(f"- **Universe:** {len(SYMBOLS)} highly volatile altcoins\n")
        f.write("- **Methodology:** Dynamic daily sizing (98th percentile = Whale, 20th percentile = Retail).\n")
        f.write("- **Signal:** Rolling 4h CVD Z-score > 1.5 for one cohort and < -1.5 for the other, coinciding with a 4h price extreme.\n\n")
        f.write("### Aggregate Results\n")
        f.write(f"- **Bearish Events (Short):** {total_bear_trades} trades | Win Rate: {avg_bear_wr*100:.1f}% | Avg 4h Edge: {avg_bear_4h*100:.2f}%\n")
        f.write(f"- **Bullish Events (Long):** {total_bull_trades} trades | Win Rate: {avg_bull_wr*100:.1f}% | Avg 4h Edge: {avg_bull_4h*100:.2f}%\n\n")
        f.write("### Per-Coin Breakdown\n")
        f.write("```text\n")
        f.write(df_res[['bear_count', 'bear_4h', 'bear_wr', 'bull_count', 'bull_4h', 'bull_wr']].to_string())
        f.write("\n```\n")

if __name__ == "__main__":
    main()
