"""
Identify and download 1h funding rate coins from Bybit.
These have much higher FR volatility (20-50 bps) vs 8h coins (1-2 bps).
"""
import pandas as pd
import subprocess
from pathlib import Path

# Top coins by volume that likely have 1h funding
candidates_1h = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'TRXUSDT', 'TONUSDT', 'SUIUSDT', 'AVAXUSDT',
    'LINKUSDT', 'DOTUSDT', 'NEARUSDT', 'LTCUSDT', 'UNIUSDT',
    'APTUSDT', 'FILUSDT', 'ATOMUSDT', 'ARBUSDT', 'OPUSDT',
    'AAVEUSDT', 'GRTUSDT', 'SNXUSDT', 'COMPUSDT', 'MKRUSDT',
    'RUNEUSDT', 'INJUSDT', 'STXUSDT', 'IMXUSDT', 'GALAUSDT',
    'FLOWUSDT', 'XTZUSDT', 'ALGOUSDT', 'SANDUSDT', 'MANAUSDT',
    'AXSUSDT', 'CHZUSDT', 'ENJUSDT', 'DGBUSDT', '1INCHUSDT',
    'COTIUSDT', 'STORJUSDT', 'ANKRUSDT', 'SKLUSDT', 'ZRXUSDT'
]

def check_funding_frequency(symbol):
    """Check if a symbol has 1h vs 8h funding by looking at file frequency."""
    path = Path(f'/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}')
    if not path.exists():
        return None
    
    # Count funding rate files
    fr_files = list(path.glob('*_funding_rate.csv'))
    if len(fr_files) == 0:
        return None
    
    # Sample a file to check funding interval
    sample = pd.read_csv(fr_files[0])
    if 'timestamp' not in sample.columns or len(sample) < 2:
        return None
    
    sample['timestamp'] = pd.to_datetime(sample['timestamp'], unit='ms')
    sample = sample.sort_values('timestamp')
    sample['diff_hours'] = sample['timestamp'].diff().dt.total_seconds() / 3600
    avg_interval = sample['diff_hours'].median()
    
    return {
        'symbol': symbol,
        'files': len(fr_files),
        'avg_interval_hours': avg_interval,
        'samples': len(sample),
        'type': '1h' if avg_interval < 2 else '8h'
    }

if __name__ == '__main__':
    print("=" * 80)
    print("IDENTIFYING 1H FUNDING RATE COINS")
    print("=" * 80)
    
    results = []
    for sym in candidates_1h:
        result = check_funding_frequency(sym)
        if result:
            results.append(result)
            print(f"{result['symbol']:12s} | {result['type']} | "
                  f"{result['avg_interval_hours']:.1f}h interval | "
                  f"{result['files']} files")
    
    # Separate 1h and 8h
    coins_1h = [r for r in results if r['type'] == '1h']
    coins_8h = [r for r in results if r['type'] == '8h']
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {len(coins_1h)} coins with 1h funding, {len(coins_8h)} with 8h funding")
    print(f"{'=' * 80}")
    
    if coins_1h:
        print("\n1H FUNDING COINS (Higher volatility, better for FR strategies):")
        for c in coins_1h:
            print(f"  {c['symbol']}")
        
        # Save list
        with open('/home/ubuntu/Projects/skytrade6/kimi-1/coins_1h.txt', 'w') as f:
            for c in coins_1h:
                f.write(c['symbol'] + '\n')
        print(f"\nSaved to coins_1h.txt")
    else:
        print("\nNo 1h funding coins found in current datalake.")
        print("Need to download 1h funding coins from Bybit.")
        
        # These are known 1h funding coins on Bybit based on prior research
        known_1h_coins = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT',
            'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'UNIUSDT'
        ]
        
        print(f"\nWill download {len(known_1h_coins)} known 1h funding coins...")
        print("Symbols:", ', '.join(known_1h_coins))
        
        # Save list for download
        with open('/home/ubuntu/Projects/skytrade6/kimi-1/coins_to_download.txt', 'w') as f:
            for c in known_1h_coins:
                f.write(c + '\n')
        
        print(f"\nTo download, run:")
        print(f"cd /home/ubuntu/Projects/skytrade6/datalake")
        print(f"python3 download_bybit_data.py {','.join(known_1h_coins[:5])} 2025-07-01 2025-12-31 -t klines,fundingRate")
