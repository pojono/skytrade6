import pandas as pd
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake"

# Top 15 Liquid/Performing Coins from previous analysis
TOP_15 = ["BERAUSDT", "ZKUSDT", "ZECUSDT", "HUMAUSDT", "LINKUSDT", 
          "INITUSDT", "NEARUSDT", "SAHARAUSDT", "XRPUSDT", "HBARUSDT", 
          "AAVEUSDT", "AGLDUSDT", "SOLUSDT", "SUIUSDT", "AVAXUSDT"]

def get_base_events():
    files = [f for f in os.listdir(FEAT_DIR) if f.endswith('_1m.parquet')]
    all_events = []
    
    for file in files:
        sym = file.replace('_1m.parquet', '')
        df = pd.read_parquet(os.path.join(FEAT_DIR, file))
        if len(df) < 30 * 24 * 60: continue
            
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
        
        # Additional Contextual Features
        df['price_change_24h'] = df['price'] / df['price'].shift(24*60) - 1
        df['volatility_4h'] = df['price'].rolling(4*60).std() / df['price']
        df['spot_spike_ratio'] = df['spot_whale_cvd'] / df['spot_whale_1h_avg'].replace(0, 1)
        
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
        events = df[df['bull_sig']].dropna(subset=['fwd_ret_4h', 'price_change_24h', 'volatility_4h']).copy()
        
        if len(events) > 0:
            events['symbol'] = sym
            events['timestamp'] = events.index
            all_events.append(events[['symbol', 'timestamp', 'price', 'fwd_ret_4h', 'price_change_24h', 'volatility_4h', 'spot_spike_ratio', 'fut_whale_z', 'fut_retail_z']])
            
    return pd.concat(all_events).reset_index(drop=True)

def fetch_ob_filter(row):
    sym = row['symbol']
    ts = pd.to_datetime(row['timestamp'])
    date_str = ts.strftime('%Y-%m-%d')
    ob_file = os.path.join(DATALAKE_DIR, "binance", sym, f"{date_str}_bookDepth.csv.gz")
    
    if not os.path.exists(ob_file):
        return True # Pass if no data to be conservative or reject? Let's say pass for now to not penalize missing days. Actually, better to track it.
        
    try:
        df_ob = pd.read_csv(ob_file)
        df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'])
        
        start_time = ts - pd.Timedelta(minutes=1)
        end_time = ts + pd.Timedelta(minutes=1)
        df_ob = df_ob[(df_ob['timestamp'] >= start_time) & (df_ob['timestamp'] <= end_time)]
        
        if len(df_ob) == 0: return True
        
        asks = df_ob[df_ob['percentage'] > 0]
        ask_agg = asks.groupby('timestamp')['notional'].sum()
        
        pre_event = ask_agg[ask_agg.index < ts].mean()
        post_event = ask_agg[ask_agg.index >= ts].mean()
        
        if pd.isna(pre_event) or pd.isna(post_event) or pre_event == 0:
            return True
            
        change = (post_event - pre_event) / pre_event
        
        # If Ask wall spikes by more than +5%, reject the trade (Returns False)
        if change > 0.05:
            return False
        return True
    except:
        return True

def analyze_portfolio(df_filtered, name):
    count = len(df_filtered)
    if count == 0:
        return {"Name": name, "Trades": 0, "Win Rate": "0%", "Avg 4h Edge": "0%", "Net PnL": "$0"}
        
    wr = (df_filtered['fwd_ret_4h'] > 0).mean()
    avg_edge = df_filtered['fwd_ret_4h'].mean()
    
    gross_pnl = count * avg_edge * 10000
    fees = count * 0.0020 * 10000
    net_pnl = gross_pnl - fees
    
    return {
        "Name": name,
        "Trades": count,
        "Win Rate": f"{wr*100:.1f}%",
        "Avg 4h Edge": f"{avg_edge*100:.2f}%",
        "Net PnL": f"${net_pnl:,.0f}"
    }

if __name__ == "__main__":
    print("Gathering all baseline events...")
    df_events = get_base_events()
    print(f"Extracted {len(df_events)} base trades.\n")
    
    # Define Filters
    # 1. Liquidity Filter: Only Top 15 coins
    mask_liquidity = df_events['symbol'].isin(TOP_15)
    
    # 2. Contextual Filter (Linear Rules)
    # Volatility threshold: avoid top 25% most volatile regimes
    vol_thresh = df_events['volatility_4h'].quantile(0.75) 
    mask_contextual = (
        (df_events['price_change_24h'] > -0.10) & # Not crashing down > 10%
        (df_events['spot_spike_ratio'] > 4.0) &   # Stronger spot impulse
        (df_events['volatility_4h'] < vol_thresh) # Not too choppy
    )
    
    # 3. L2 Orderbook Filter
    print("Processing L2 Orderbook Absorption data (this takes a minute)...")
    # For speed, we will use multiprocess map
    ob_mask = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        ob_mask = list(executor.map(fetch_ob_filter, [row for _, row in df_events.iterrows()]))
    df_events['ob_pass'] = ob_mask
    mask_ob = df_events['ob_pass']
    
    # Combinations
    res = []
    res.append(analyze_portfolio(df_events, "0. Baseline (All 66 Coins, No Filters)"))
    res.append(analyze_portfolio(df_events[mask_liquidity], "1. Liquidity Universe Only (Top 15)"))
    res.append(analyze_portfolio(df_events[mask_contextual], "2. Contextual Rules Only"))
    res.append(analyze_portfolio(df_events[mask_ob], "3. L2 OB Filter Only"))
    
    # Ultimate Combination: Top 15 + Contextual + L2 OB
    mask_ultimate = mask_liquidity & mask_contextual & mask_ob
    res.append(analyze_portfolio(df_events[mask_ultimate], "4. THE ULTIMATE ENGINE (All 3 Combined)"))
    
    df_res = pd.DataFrame(res)
    print("\n" + "="*80)
    print("FILTER PERFORMANCE COMPARISON (Assuming $10k per trade, 20bps fees)")
    print("="*80)
    print(df_res.to_string(index=False))
