import pandas as pd
import numpy as np

def analyze_results():
    try:
        df = pd.read_csv('massive_hft_results.csv')
    except:
        print("Results file not found.")
        return
        
    print("--- Performance by Coin ---")
    coin_stats = df.groupby('symbol').agg(
        trades=('symbol', 'count'),
        win_rate=('net_ret_pct', lambda x: (x > 0).mean()),
        total_ret=('net_ret_pct', 'sum'),
        avg_ret=('net_ret_pct', 'mean')
    ).sort_values('total_ret', ascending=False)
    
    print("\nTop 10 Most Profitable Coins:")
    print(coin_stats.head(10).to_string())
    
    print("\nBottom 5 Least Profitable Coins:")
    print(coin_stats.tail(5).to_string())
    
    print("\n--- Performance by Signal Type ---")
    signal_stats = df.groupby('signal').agg(
        trades=('signal', 'count'),
        win_rate=('net_ret_pct', lambda x: (x > 0).mean()),
        total_ret=('net_ret_pct', 'sum'),
        avg_ret=('net_ret_pct', 'mean')
    )
    # Map signals
    signal_stats.index = signal_stats.index.map({-1: 'Powder Keg (Short)', 1: 'Despair Pit (Long)'})
    print(signal_stats.to_string())

if __name__ == "__main__":
    analyze_results()
