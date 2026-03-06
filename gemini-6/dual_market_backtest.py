import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime, timedelta

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"
EXCHANGE = "binance"

# Let's test on a few highly volatile symbols where Spot data has the biggest impact
SYMBOLS = ["SOLUSDT", "SUIUSDT", "WLDUSDT", "NEARUSDT", "ARBUSDT"]

start_date = datetime(2025, 7, 1)
end_date = datetime(2025, 8, 31) # 2 months for rapid iteration
DATES = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]

def load_and_resample(symbol, date, market_type="futures"):
    filepath = os.path.join(DATALAKE_DIR, EXCHANGE, symbol, f"{date}_trades" + ("_spot" if market_type == "spot" else "") + ".csv.gz")
    if not os.path.exists(filepath): return None
    
    try:
        if market_type == "spot":
            try:
                df = pd.read_csv(filepath, names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match'])
            except:
                df = pd.read_csv(filepath)
            
            if len(df) == 0: return None
            if 'time' not in df.columns: return None
                
            if df['time'].iloc[0] > 1e14:
                df['time'] = pd.to_datetime(df['time'], unit='us')
            else:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
        else:
            df = pd.read_csv(filepath, usecols=['price', 'quote_qty', 'time', 'is_buyer_maker'])
            if len(df) == 0: return None
            df['time'] = pd.to_datetime(df['time'], unit='ms')

        df['direction'] = np.where(df['is_buyer_maker'] == True, -1, 1)
        df['signed_qty'] = df['quote_qty'] * df['direction']
        
        # Calculate percentiles directly
        whale_thresh = df['quote_qty'].quantile(0.98)
        retail_thresh = df['quote_qty'].quantile(0.20)
        
        df['is_whale'] = df['quote_qty'] >= whale_thresh
        df['is_retail'] = df['quote_qty'] <= retail_thresh
        
        df.set_index('time', inplace=True)
        
        if market_type == "futures":
            price = df['price'].resample('1min').last().ffill()
        else:
            price = None
            
        whale_cvd = df.loc[df['is_whale'], 'signed_qty'].resample('1min').sum().rename(f'{market_type}_whale_cvd')
        retail_cvd = df.loc[df['is_retail'], 'signed_qty'].resample('1min').sum().rename(f'{market_type}_retail_cvd')
        
        if price is not None:
            return pd.concat([price, whale_cvd, retail_cvd], axis=1).fillna(0)
        return pd.concat([whale_cvd, retail_cvd], axis=1).fillna(0)

    except Exception as e:
        return None

def process_day(args):
    symbol, date = args
    df_fut = load_and_resample(symbol, date, "futures")
    df_spot = load_and_resample(symbol, date, "spot")
    
    if df_fut is None or df_spot is None: return None
    
    res = pd.concat([df_fut, df_spot], axis=1).fillna(0)
    res['close'] = res['price'].replace(0, np.nan).ffill()
    
    return res

def analyze_symbol(symbol):
    print(f"[{symbol}] Processing dual-market data...")
    args_list = [(symbol, d) for d in DATES]
    
    daily_dfs = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_day, args_list))
        
    for res in results:
        if res is not None:
            daily_dfs.append(res)
            
    if not daily_dfs: return None
    
    df = pd.concat(daily_dfs)
    df.sort_index(inplace=True)
    
    rolling_window = 4 * 60
    df['roll_fut_whale'] = df['futures_whale_cvd'].rolling(rolling_window).sum()
    df['roll_fut_retail'] = df['futures_retail_cvd'].rolling(rolling_window).sum()
    
    # Forward Returns
    df['fwd_ret_1h'] = df['close'].shift(-60) / df['close'] - 1
    df['fwd_ret_2h'] = df['close'].shift(-120) / df['close'] - 1
    df['fwd_ret_4h'] = df['close'].shift(-240) / df['close'] - 1
    
    z_window = 3 * 24 * 60
    df['fut_whale_mean'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).mean()
    df['fut_whale_std'] = df['roll_fut_whale'].rolling(z_window, min_periods=rolling_window).std()
    df['fut_whale_z'] = (df['roll_fut_whale'] - df['fut_whale_mean']) / df['fut_whale_std']
    
    df['fut_retail_mean'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).mean()
    df['fut_retail_std'] = df['roll_fut_retail'].rolling(z_window, min_periods=rolling_window).std()
    df['fut_retail_z'] = (df['roll_fut_retail'] - df['fut_retail_mean']) / df['fut_retail_std']
    
    # Rolling 1-hour spot volume average to detect anomalous spikes
    df['spot_whale_1h_avg'] = df['spot_whale_cvd'].abs().rolling(60).mean()
    
    # --------------------------------------------------------------------------
    # STRATEGY: DUAL SPOT-FUTURES LEAD-LAG
    # 1. Structural setup: Futures Retail heavily selling (Z < -1.5)
    # 2. Structural setup: Futures Whales accumulating (Z > 1.5)
    # 3. Execution Trigger: Sudden 1-minute spike in Spot Whale Buying (>3x hourly average)
    # --------------------------------------------------------------------------
    
    bullish_div = (
        (df['fut_retail_z'] < -1.5) & 
        (df['fut_whale_z'] > 1.5) &
        (df['spot_whale_cvd'] > df['spot_whale_1h_avg'] * 3.0) & # The Trigger
        (df['spot_whale_cvd'] > 0) # Must be net buying
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
    bull_events = df[df['bull_sig']].dropna(subset=['fwd_ret_4h'])
    
    return {
        'symbol': symbol,
        'count': len(bull_events),
        '1h': bull_events['fwd_ret_1h'].mean() if len(bull_events) > 0 else 0,
        '2h': bull_events['fwd_ret_2h'].mean() if len(bull_events) > 0 else 0,
        '4h': bull_events['fwd_ret_4h'].mean() if len(bull_events) > 0 else 0,
        'wr': (bull_events['fwd_ret_4h'] > 0).mean() if len(bull_events) > 0 else 0,
    }

def main():
    start_time = time.time()
    all_results = []
    
    for symbol in SYMBOLS:
        res = analyze_symbol(symbol)
        if res: all_results.append(res)
            
    print("\n" + "="*60)
    print("DUAL-MARKET LEAD-LAG RESULTS (July-August 2025)")
    print("="*60)
    
    total_trades = sum(r['count'] for r in all_results)
    
    if total_trades > 0:
        avg_1h = sum(r['1h'] * r['count'] for r in all_results) / total_trades
        avg_2h = sum(r['2h'] * r['count'] for r in all_results) / total_trades
        avg_4h = sum(r['4h'] * r['count'] for r in all_results) / total_trades
        avg_wr = sum(r['wr'] * r['count'] for r in all_results) / total_trades
    else:
        avg_1h = avg_2h = avg_4h = avg_wr = 0
        
    df_res = pd.DataFrame(all_results)
    df_res.set_index('symbol', inplace=True)
    
    format_pct = lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "0.00%"
    for col in ['1h', '2h', '4h', 'wr']:
        df_res[col] = df_res[col].apply(format_pct)
        
    print(df_res.to_string())
    
    print("\n--- AGGREGATE SUMMARY ---")
    print(f"Total Symbols Tested: {len(SYMBOLS)}")
    print(f"Total Events (Fut Div + Spot Spike Trigger): {total_trades}")
    print(f"  Avg 1h Edge: {avg_1h*100:.2f}%")
    print(f"  Avg 2h Edge: {avg_2h*100:.2f}%")
    print(f"  Avg 4h Edge: {avg_4h*100:.2f}%")
    print(f"  Win Rate (4h): {avg_wr*100:.1f}%")
    print(f"\nExecution time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
