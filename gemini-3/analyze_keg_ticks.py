import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def load_trades_around_event(symbol, event_time_str):
    event_time = pd.to_datetime(event_time_str)
    date_str = event_time.strftime('%Y-%m-%d')
    
    # Let's use Bybit tick data as we have 1.8TB of it and it's very granular
    tick_file = DATALAKE / f"bybit/{symbol}/{date_str}_trades.csv.gz"
    if not tick_file.exists():
        print(f"Missing tick data for {symbol} on {date_str}")
        return None
        
    print(f"Loading ticks for {symbol} {date_str}...")
    try:
        # Load columns: timestamp, side, size, price
        df = pd.read_csv(tick_file, usecols=['timestamp', 'side', 'size', 'price'])
        # Bybit timestamps are seconds.microseconds
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Extract window: Event time to +6 hours
        start_time = event_time
        end_time = event_time + pd.Timedelta(hours=6)
        
        window_df = df.loc[start_time:end_time].copy()
        if window_df.empty:
            return None
            
        return window_df
    except Exception as e:
        print(f"Error loading {tick_file}: {e}")
        return None

def analyze_ultra_high_freq(events_csv="keg_events.csv"):
    if not os.path.exists(events_csv):
        print("Run powder_keg_high_res.py first")
        return
        
    events = pd.read_csv(events_csv)
    
    # We will simulate a simple microstructure model:
    # After the Powder Keg is formed (hourly), we don't enter immediately.
    # Instead, we monitor the 10-second Rolling Order Flow Imbalance (CVD).
    # We wait for the exact moment when the market aggressively flips direction.
    
    results = []
    
    for _, row in events.iterrows():
        symbol = row['symbol']
        event_time = row['timestamp']
        signal = row['signal'] # -1 for Short (Keg), 1 for Long (Despair)
        
        print(f"--- Analyzing {symbol} at {event_time} (Signal: {signal}) ---")
        ticks = load_trades_around_event(symbol, event_time)
        if ticks is None or ticks.empty:
            continue
            
        # Calculate signed volume (positive for Buy, negative for Sell)
        ticks['signed_vol'] = np.where(ticks['side'] == 'Buy', ticks['size'], -ticks['size'])
        
        # Resample to 1-second bins to build our execution trigger
        sec_df = ticks.resample('1s').agg({
            'price': 'last',
            'size': 'sum',
            'signed_vol': 'sum'
        }).ffill()
        
        # Calculate 30-second rolling Volume Imbalance (CVD momentum)
        sec_df['vol_imb_30s'] = sec_df['signed_vol'].rolling(30).sum()
        sec_df['total_vol_30s'] = sec_df['size'].rolling(30).sum()
        
        # Calculate Imbalance Ratio (-1.0 to 1.0)
        # Avoid division by zero
        sec_df['imb_ratio'] = sec_df['vol_imb_30s'] / (sec_df['total_vol_30s'] + 1e-8)
        
        entry_idx = None
        entry_price = 0
        
        # Trigger Condition:
        # If Powder Keg (Short bias): Wait until the 30s imbalance drops below -0.6 (Massive selling starts)
        # If Despair Pit (Long bias): Wait until the 30s imbalance jumps above +0.6 (Massive buying starts)
        
        for idx, sec_row in sec_df.iterrows():
            if pd.isna(sec_row['imb_ratio']):
                continue
                
            if signal == -1 and sec_row['imb_ratio'] < -0.7: # Extremely aggressive selling
                entry_idx = idx
                entry_price = sec_row['price']
                break
            elif signal == 1 and sec_row['imb_ratio'] > 0.7: # Extremely aggressive buying
                entry_idx = idx
                entry_price = sec_row['price']
                break
                
        if entry_idx is None:
            print(f"No execution trigger found within 6 hours. Trade skipped.")
            continue
            
        # Simulate Network Latency
        # RTT = 2ms, Exchange Processing = 28ms -> Total = 30ms latency
        # We find the actual price we would get filled at exactly entry_time + 30ms
        fill_time = entry_idx + pd.Timedelta(milliseconds=30)
        
        # Get the first tick AFTER our fill_time
        post_latency_ticks = ticks.loc[fill_time:]
        if post_latency_ticks.empty:
            continue
            
        actual_fill_price = post_latency_ticks.iloc[0]['price']
        latency_slippage = (actual_fill_price - entry_price) / entry_price * signal
        
        print(f"Trigger hit at: {entry_idx} | Price: {entry_price:.4f}")
        print(f"Simulated Fill (+30ms): {fill_time} | Price: {actual_fill_price:.4f} | Slippage: {latency_slippage*10000:.1f} bps")
        
        # Now track the trade for exactly 1 hour (scalp the cascade)
        exit_time = fill_time + pd.Timedelta(hours=1)
        post_trade = ticks.loc[fill_time:exit_time]
        
        if not post_trade.empty:
            # Let's apply a 3% dynamic trailing stop and 6% TP
            max_pnl = 0
            min_pnl = 0
            exit_price = post_trade.iloc[-1]['price']
            exit_reason = "Time (1h)"
            
            for t_idx, t_row in post_trade.iterrows():
                pnl = (t_row['price'] - actual_fill_price) / actual_fill_price * signal
                
                if pnl > max_pnl: max_pnl = pnl
                if pnl < min_pnl: min_pnl = pnl
                
                if pnl <= -0.02: # 2% Stop Loss
                    exit_price = t_row['price']
                    exit_reason = "Stop Loss"
                    break
                if pnl >= 0.06: # 6% Take Profit
                    exit_price = t_row['price']
                    exit_reason = "Take Profit"
                    break
                    
            net_ret = ((exit_price - actual_fill_price) / actual_fill_price * signal) - 0.001 # 10 bps total roundtrip fee
            
            print(f"Exit Reason: {exit_reason} | Net Return: {net_ret*100:.2f}% | Max Drawdown: {min_pnl*100:.2f}% | Max Runup: {max_pnl*100:.2f}%\n")
            
            results.append({
                'symbol': symbol,
                'signal': signal,
                'entry_time': fill_time,
                'fill_price': actual_fill_price,
                'slippage_bps': latency_slippage * 10000,
                'exit_reason': exit_reason,
                'net_ret_pct': net_ret * 100,
                'max_runup_pct': max_pnl * 100,
                'max_dd_pct': min_pnl * 100
            })
            
    if results:
        res_df = pd.DataFrame(results)
        print("=== Ultra-High Frequency Execution Results ===")
        print(f"Total Trades Taken: {len(res_df)}")
        print(f"Average Slippage (30ms latency): {res_df['slippage_bps'].mean():.2f} bps")
        print(f"Win Rate: {(res_df['net_ret_pct'] > 0).mean()*100:.1f}%")
        print(f"Total Net Return: {res_df['net_ret_pct'].sum():.2f}%")
        print(f"Average Max Runup (Theoretical potential): {res_df['max_runup_pct'].mean():.2f}%")

if __name__ == "__main__":
    analyze_ultra_high_freq()
