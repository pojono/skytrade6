"""
Kimi-1 Fast Edge Detection - Ultra-simplified for speed
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

FEE_TAKER = 0.001  # 0.1% per leg

def fast_backtest(symbol, start_date, end_date, min_fr_bps=1.0):
    """
    Ultra-simplified FR hold backtest - no fancy abstractions.
    """
    # Load klines
    kline_files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/{start_date}_*_kline_1m.csv'))
    if not kline_files:
        return None
    
    # Load first file only for speed
    klines = pd.read_csv(kline_files[0])
    klines['timestamp'] = pd.to_datetime(klines['startTime'], unit='ms')
    
    # Load funding
    fr_files = sorted(glob.glob(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/{start_date}_*_funding_rate.csv'))
    if not fr_files:
        return None
    
    funding = pd.read_csv(fr_files[0])
    funding['timestamp'] = pd.to_datetime(funding['timestamp'], unit='ms')
    funding['fr_bps'] = funding['fundingRate'] * 10000
    
    # Simple signal: FR > threshold
    trades = []
    capital = 10000
    position = 0
    entry_price = 0
    entry_time = None
    
    for _, fr_row in funding.iterrows():
        fr = fr_row['fr_bps']
        ts = fr_row['timestamp']
        
        # Find price at this time
        price_data = klines[klines['timestamp'] >= ts]
        if price_data.empty:
            continue
        price = price_data.iloc[0]['close']
        
        # Entry
        if position == 0 and fr >= min_fr_bps:
            position = 1
            entry_price = price
            entry_time = ts
        
        # Exit: 24h later or FR drops
        elif position == 1:
            exit_condition = fr < min_fr_bps * 0.5
            
            if exit_condition:
                # Calculate P&L
                pnl_pct = (price - entry_price) / entry_price
                pnl_gross = pnl_pct * capital
                fees = capital * FEE_TAKER * 2  # Round trip taker
                pnl_net = pnl_gross - fees
                
                trades.append({
                    'entry': entry_time,
                    'exit': ts,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'fr_entry': fr_row['fr_bps'],
                    'pnl_net': pnl_net,
                    'fees': fees
                })
                
                capital += pnl_net
                position = 0
    
    if len(trades) < 3:
        return None
    
    pnl_total = sum(t['pnl_net'] for t in trades)
    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    
    return {
        'symbol': symbol,
        'trades': len(trades),
        'win_rate': wins / len(trades),
        'net_pnl': pnl_total,
        'fees': sum(t['fees'] for t in trades),
        'avg_trade': pnl_total / len(trades)
    }


if __name__ == '__main__':
    print("=" * 80)
    print("KIMI-1 FAST EDGE SCAN")
    print("=" * 80)
    
    # Test one day across multiple symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'ADAUSDT', 
               'AVAXUSDT', 'LINKUSDT', 'LTCUSDT', 'DOTUSDT']
    
    test_date = '2025-10-15'  # Single day for speed
    thresholds = [0.5, 1.0, 1.5]
    
    all_results = []
    
    print(f"\nTesting {len(symbols)} symbols on {test_date}...")
    print("-" * 80)
    
    for symbol in symbols:
        for thresh in thresholds:
            result = fast_backtest(symbol, test_date, test_date, thresh)
            if result:
                all_results.append(result)
                print(f"{symbol:12s} | FR>={thresh:.1f}bps | {result['trades']} trades | "
                      f"WR={result['win_rate']:.0%} | Net=${result['net_pnl']:.2f}")
    
    print("\n" + "=" * 80)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df['edge'] = df['net_pnl'] - df['fees']
        
        profitable = df[df['edge'] > 0].sort_values('edge', ascending=False)
        print(f"\nProfitable configs: {len(profitable)}/{len(df)}")
        
        if len(profitable) > 0:
            print("\nTop performers:")
            print(profitable[['symbol', 'trades', 'win_rate', 'net_pnl', 'edge']].head().to_string())
        else:
            print("\nNo edge found on single-day test.")
            print("Need: (1) more data (2) better strategy (3) cross-exchange arb")
    
    print("\n" + "=" * 80)
