import pandas as pd
import numpy as np
import os

FEAT_DIR = "/home/ubuntu/Projects/skytrade6/gemini-6/features"
TOP_12 = ["BERAUSDT", "ZKUSDT", "ZECUSDT", "HUMAUSDT", "INITUSDT", 
          "NEARUSDT", "SAHARAUSDT", "AGLDUSDT", "XRPUSDT", "HBARUSDT", 
          "AAVEUSDT", "LINKUSDT"]

def run_experiment_3():
    print("--- Experiment 3: Maker Execution vs Taker Execution ---")
    all_events = []
    
    for sym in TOP_12:
        file_path = os.path.join(FEAT_DIR, f"{sym}_1m.parquet")
        if not os.path.exists(file_path): continue
        
        df = pd.read_parquet(file_path)
        
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
        events = df[df['bull_sig']].dropna(subset=['fwd_ret_4h']).copy()
        
        if len(events) > 0:
            events['symbol'] = sym
            all_events.append(events[['symbol', 'fwd_ret_4h']])
            
    df_all = pd.concat(all_events).reset_index(drop=True)
    
    # 1. Baseline: 100% Fill Rate, 20 bps round trip Taker fees
    base_gross = df_all['fwd_ret_4h'].sum() * 10000
    base_fees = len(df_all) * 0.0020 * 10000
    base_net = base_gross - base_fees
    
    # 2. Perfect Maker: 100% Fill Rate, 0 bps fees (assuming VIP 0 maker fee)
    perfect_maker_net = base_gross - 0
    
    # 3. Realistic Maker: 80% Fill Rate (miss 20% of trades because price runs away)
    # To simulate missing 20% of trades, we randomly drop 20% of the dataset
    # We will do this 100 times and average the result to get a stable estimate
    realistic_maker_nets = []
    missed_gross_pnls = []
    for i in range(100):
        # Sample 80% of trades
        df_sample = df_all.sample(frac=0.8, random_state=i)
        gross = df_sample['fwd_ret_4h'].sum() * 10000
        realistic_maker_nets.append(gross)
        
        # What did we miss? (The 20% that didn't fill)
        missed = df_all.drop(df_sample.index)['fwd_ret_4h'].sum() * 10000
        missed_gross_pnls.append(missed)
        
    avg_realistic_maker_net = np.mean(realistic_maker_nets)
    avg_missed_gross = np.mean(missed_gross_pnls)
    
    print(f"Total Trades: {len(df_all)}")
    print(f"\n1. TAKER EXECUTION (Baseline)")
    print(f"  Fill Rate: 100%")
    print(f"  Fees Paid: ${base_fees:,.0f} (20 bps)")
    print(f"  Net PnL:   ${base_net:,.0f}")
    
    print(f"\n2. PERFECT MAKER EXECUTION (Unrealistic)")
    print(f"  Fill Rate: 100%")
    print(f"  Fees Paid: $0 (0 bps)")
    print(f"  Net PnL:   ${perfect_maker_net:,.0f} (+{((perfect_maker_net/base_net)-1)*100:.1f}%)")
    
    print(f"\n3. REALISTIC MAKER EXECUTION (Limit orders miss the fastest squeezes)")
    print(f"  Fill Rate: 80% (Missed 20% of trades due to runaway price)")
    print(f"  Fees Paid: $0")
    print(f"  Net PnL:   ${avg_realistic_maker_net:,.0f} ({((avg_realistic_maker_net/base_net)-1)*100:.1f}% vs Baseline)")
    print(f"  Missed Opportunity Cost: ${avg_missed_gross:,.0f} in PnL from trades that didn't fill.")
    
    print("\nCONCLUSION:")
    if avg_realistic_maker_net > base_net:
        print("  Even with a 20% miss rate, Maker Execution is MORE profitable than paying Taker fees.")
    else:
        print("  Missing 20% of trades destroys more PnL than the 20 bps fees save. Stick to Taker execution.")

if __name__ == "__main__":
    run_experiment_3()
