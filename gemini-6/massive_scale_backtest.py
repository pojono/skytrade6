import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime, timedelta

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"

# Expanded universe to top 30 volatile altcoins based on directory sizes (skipping BTC/ETH as majors)
SYMBOLS = [
    "LINKUSDT", "LTCUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", 
    "DOTUSDT", "AVAXUSDT", "SOLUSDT", "AAVEUSDT", "NEARUSDT", 
    "ZECUSDT", "ARBUSDT", "SUIUSDT", "WIFUSDT", "ATOMUSDT", 
    "APTUSDT", "ENAUSDT", "TAOUSDT", "TONUSDT", "TIAUSDT", 
    "WLDUSDT", "FILUSDT", "CRVUSDT", "VIRTUALUSDT", "XLMUSDT", 
    "AGLDUSDT", "BCHUSDT", "GALAUSDT", "JTOUSDT", "ZROUSDT"
]

# Generate date range: 6 full months (July 1, 2025 - Dec 31, 2025)
start_date = datetime(2025, 7, 1)
end_date = datetime(2025, 12, 31)
DATES = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]

def process_day(args):
    """Processes a single day of raw tick data and returns 1-minute OHLCV + CVD stats."""
    symbol, date = args
    filepath = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades.csv.gz")
    
    if not os.path.exists(filepath):
        return None
        
    try:
        # Load trades efficiently
        df = pd.read_csv(filepath, usecols=['price', 'quote_qty', 'time', 'is_buyer_maker'])
        if len(df) == 0:
            return None
            
        # Dynamic daily thresholds
        whale_thresh = df['quote_qty'].quantile(0.98)
        retail_thresh = df['quote_qty'].quantile(0.20)
        
        df['direction'] = np.where(df['is_buyer_maker'], -1, 1)
        df['signed_qty'] = df['quote_qty'] * df['direction']
        
        df['is_whale'] = df['quote_qty'] >= whale_thresh
        df['is_retail'] = df['quote_qty'] <= retail_thresh
        
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Resample to 1-minute bars
        ohlc = df['price'].resample('1min').ohlc()
        total_cvd = df['signed_qty'].resample('1min').sum().rename('total_cvd')
        whale_cvd = df.loc[df['is_whale'], 'signed_qty'].resample('1min').sum().rename('whale_cvd')
        retail_cvd = df.loc[df['is_retail'], 'signed_qty'].resample('1min').sum().rename('retail_cvd')
        
        res = pd.concat([ohlc, total_cvd, whale_cvd, retail_cvd], axis=1)
        res.fillna(0, inplace=True)
        
        # Forward fill prices
        res['close'] = res['close'].replace(0, np.nan).ffill()
        mask = res['open'] == 0
        res.loc[mask, 'open'] = res.loc[mask, 'close']
        res.loc[mask, 'high'] = res.loc[mask, 'close']
        res.loc[mask, 'low'] = res.loc[mask, 'close']
        
        res['open'] = res['open'].ffill()
        res['high'] = res['high'].ffill()
        res['low'] = res['low'].ffill()
        
        return res
        
    except Exception as e:
        return None

def analyze_symbol(symbol):
    """Aggregates months of minute-bar data and finds divergences."""
    print(f"[{symbol}] Processing 6 months ({len(DATES)} days)...")
    
    args_list = [(symbol, d) for d in DATES]
    
    daily_dfs = []
    # Using 8 workers. We are only processing one symbol at a time in the main thread, 
    # so we can use multiple cores for the daily files of that symbol.
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_day, args_list))
        
    for res in results:
        if res is not None:
            daily_dfs.append(res)
            
    if not daily_dfs:
        print(f"[{symbol}] No data found.")
        return None
        
    df = pd.concat(daily_dfs)
    df.sort_index(inplace=True)
    
    rolling_window = 4 * 60 # 4 hours
    
    df['roll_whale_cvd'] = df['whale_cvd'].rolling(rolling_window).sum()
    df['roll_retail_cvd'] = df['retail_cvd'].rolling(rolling_window).sum()
    df['roll_high'] = df['high'].rolling(rolling_window).max()
    df['roll_low'] = df['low'].rolling(rolling_window).min()
    
    # Calculate 1h, 2h, 4h forward returns
    df['fwd_ret_1h'] = df['close'].shift(-60) / df['close'] - 1
    df['fwd_ret_2h'] = df['close'].shift(-120) / df['close'] - 1
    df['fwd_ret_4h'] = df['close'].shift(-240) / df['close'] - 1
    
    # Strictly backward-looking Z-scores (3-day window)
    z_window = 3 * 24 * 60
    
    df['whale_cvd_mean'] = df['roll_whale_cvd'].rolling(z_window, min_periods=rolling_window).mean()
    df['whale_cvd_std'] = df['roll_whale_cvd'].rolling(z_window, min_periods=rolling_window).std()
    df['whale_cvd_z'] = (df['roll_whale_cvd'] - df['whale_cvd_mean']) / df['whale_cvd_std']
    
    df['retail_cvd_mean'] = df['roll_retail_cvd'].rolling(z_window, min_periods=rolling_window).mean()
    df['retail_cvd_std'] = df['roll_retail_cvd'].rolling(z_window, min_periods=rolling_window).std()
    df['retail_cvd_z'] = (df['roll_retail_cvd'] - df['retail_cvd_mean']) / df['retail_cvd_std']
    
    # Define Smart Money Bullish Divergence ONLY (Long only edge)
    # 1. Price is near the 4h low
    # 2. Retail is net selling heavily (Z-score < -1.5)
    # 3. Whales are net buying heavily (Z-score > 1.5)
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
    df['bull_sig'] = filter_signals(bullish_div, 120)
    
    bull_events = df[df['bull_sig']].dropna(subset=['fwd_ret_4h'])
    
    # Get total trades (rows) for this coin during this period for context
    total_minutes = len(df)
    
    return {
        'symbol': symbol,
        'bull_count': len(bull_events),
        'bull_1h': bull_events['fwd_ret_1h'].mean() if len(bull_events) > 0 else 0,
        'bull_2h': bull_events['fwd_ret_2h'].mean() if len(bull_events) > 0 else 0,
        'bull_4h': bull_events['fwd_ret_4h'].mean() if len(bull_events) > 0 else 0,
        'bull_wr': (bull_events['fwd_ret_4h'] > 0).mean() if len(bull_events) > 0 else 0,
        'days_analyzed': total_minutes / (24 * 60)
    }

def main():
    start_time = time.time()
    
    all_results = []
    
    # Process symbols sequentially at the macro level to avoid RAM explosion, 
    # but use multiprocessing inside analyze_symbol for the daily files.
    for symbol in SYMBOLS:
        res = analyze_symbol(symbol)
        if res:
            all_results.append(res)
            
    print("\n" + "="*60)
    print("MASSIVE BACKTEST RESULTS (July-Dec 2025)")
    print("="*60)
    
    total_bull_trades = sum(r['bull_count'] for r in all_results)
    
    if total_bull_trades > 0:
        avg_bull_1h = sum(r['bull_1h'] * r['bull_count'] for r in all_results) / total_bull_trades
        avg_bull_2h = sum(r['bull_2h'] * r['bull_count'] for r in all_results) / total_bull_trades
        avg_bull_4h = sum(r['bull_4h'] * r['bull_count'] for r in all_results) / total_bull_trades
        avg_bull_wr = sum(r['bull_wr'] * r['bull_count'] for r in all_results) / total_bull_trades
    else:
        avg_bull_1h = avg_bull_2h = avg_bull_4h = avg_bull_wr = 0
        
    df_res = pd.DataFrame(all_results)
    df_res.set_index('symbol', inplace=True)
    
    format_pct = lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "0.00%"
    
    for col in ['bull_1h', 'bull_2h', 'bull_4h', 'bull_wr']:
        df_res[col] = df_res[col].apply(format_pct)
        
    print(df_res[['bull_count', 'bull_1h', 'bull_2h', 'bull_4h', 'bull_wr']].to_string())
    
    print("\n--- AGGREGATE SUMMARY ---")
    print(f"Total Symbols Tested: {len(SYMBOLS)}")
    print(f"Total Long Events (Whale Buy / Retail Sell): {total_bull_trades}")
    print(f"  Avg 1h Edge: {avg_bull_1h*100:.2f}%")
    print(f"  Avg 2h Edge: {avg_bull_2h*100:.2f}%")
    print(f"  Avg 4h Edge: {avg_bull_4h*100:.2f}%")
    print(f"  Win Rate (4h): {avg_bull_wr*100:.1f}%")
    print(f"\nExecution time: {(time.time() - start_time) / 60:.2f} minutes")
    
    with open("MASSIVE_OOS_RESULTS.md", "w") as f:
        f.write("# Massive 6-Month Scale OOS Validation\n")
        f.write("## July 2025 - December 2025\n\n")
        f.write(f"- **Universe:** {len(SYMBOLS)} volatile altcoins\n")
        f.write("- **Methodology:** Long-Only. Dynamic daily sizing. Rolling 4h CVD Z-score > 1.5 (Whales) & < -1.5 (Retail) at lows.\n\n")
        f.write("### Aggregate Results\n")
        f.write(f"- **Total Events:** {total_bull_trades} trades\n")
        f.write(f"- **Avg 1h Edge:** {avg_bull_1h*100:.2f}%\n")
        f.write(f"- **Avg 2h Edge:** {avg_bull_2h*100:.2f}%\n")
        f.write(f"- **Avg 4h Edge:** {avg_bull_4h*100:.2f}%\n")
        f.write(f"- **Win Rate (4h):** {avg_bull_wr*100:.1f}%\n\n")
        f.write("### Per-Coin Breakdown\n")
        f.write("```text\n")
        f.write(df_res[['bull_count', 'bull_1h', 'bull_2h', 'bull_4h', 'bull_wr']].to_string())
        f.write("\n```\n")

if __name__ == "__main__":
    main()
