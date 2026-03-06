import pandas as pd
import numpy as np
from pathlib import Path
import gzip

DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

def process_trades(symbol, target_date):
    print(f"Processing {symbol} for {target_date}...")
    
    # 1. Bybit Futures
    bb_fut_file = DATALAKE / f"bybit/{symbol}/{target_date}_trades.csv.gz"
    if bb_fut_file.exists():
        df = pd.read_csv(bb_fut_file, usecols=['timestamp', 'side', 'size', 'price', 'foreignNotional'], engine='c')
        # Bybit features 'foreignNotional' which is size * price (Quote Vol)
        # 'side' is Buy or Sell (Taker Side)
        vol = df['foreignNotional'].sum()
        trades = len(df)
        avg_trade = vol / trades
        buyer_vol = df[df['side'] == 'Buy']['foreignNotional'].sum()
        seller_vol = df[df['side'] == 'Sell']['foreignNotional'].sum()
        # Large trades (> $100k)
        large_trades = len(df[df['foreignNotional'] > 100000])
        bb_fut_metrics = {
            'market': 'Bybit Futures',
            'trade_count': trades,
            'total_vol': vol,
            'avg_trade_size': avg_trade,
            'taker_buy_pct': buyer_vol / vol if vol > 0 else 0,
            'large_trades_100k': large_trades,
            'trades_per_sec': trades / 86400
        }
    else:
        bb_fut_metrics = None

    # 2. Bybit Spot
    bb_spot_file = DATALAKE / f"bybit/{symbol}/{target_date}_trades_spot.csv.gz"
    if bb_spot_file.exists():
        df = pd.read_csv(bb_spot_file, usecols=['volume', 'price', 'side'], engine='c')
        df['quote_vol'] = df['volume'] * df['price']
        
        vol = df['quote_vol'].sum()
        trades = len(df)
        avg_trade = vol / trades if trades > 0 else 0
        buyer_vol = df[df['side'].str.lower() == 'buy']['quote_vol'].sum()
        large_trades = len(df[df['quote_vol'] > 100000])
        bb_spot_metrics = {
            'market': 'Bybit Spot',
            'trade_count': trades,
            'total_vol': vol,
            'avg_trade_size': avg_trade,
            'taker_buy_pct': buyer_vol / vol if vol > 0 else 0,
            'large_trades_100k': large_trades,
            'trades_per_sec': trades / 86400
        }
    else:
        bb_spot_metrics = None

    # 3. Binance Futures
    bn_fut_file = DATALAKE / f"binance/{symbol}/{target_date}_trades.csv.gz"
    if bn_fut_file.exists():
        df = pd.read_csv(bn_fut_file, usecols=['quote_qty', 'is_buyer_maker'], engine='c')
        # is_buyer_maker = True means Maker bought, Taker sold.
        # So if is_buyer_maker == False, Taker bought.
        
        vol = df['quote_qty'].sum()
        trades = len(df)
        avg_trade = vol / trades if trades > 0 else 0
        buyer_vol = df[df['is_buyer_maker'] == False]['quote_qty'].sum()
        large_trades = len(df[df['quote_qty'] > 100000])
        bn_fut_metrics = {
            'market': 'Binance Futures',
            'trade_count': trades,
            'total_vol': vol,
            'avg_trade_size': avg_trade,
            'taker_buy_pct': buyer_vol / vol if vol > 0 else 0,
            'large_trades_100k': large_trades,
            'trades_per_sec': trades / 86400
        }
    else:
        bn_fut_metrics = None

    # 4. Binance Spot
    bn_spot_file = DATALAKE / f"binance/{symbol}/{target_date}_trades_spot.csv.gz"
    if bn_spot_file.exists():
        # Columns: id, price, qty, quote_qty, time, is_buyer_maker, is_best_match
        df = pd.read_csv(bn_spot_file, header=None, usecols=[1, 2, 3, 5], engine='c')
        df.columns = ['price', 'qty', 'quote_qty', 'is_buyer_maker']
        
        vol = df['quote_qty'].sum()
        trades = len(df)
        avg_trade = vol / trades if trades > 0 else 0
        buyer_vol = df[df['is_buyer_maker'] == False]['quote_qty'].sum()
        large_trades = len(df[df['quote_qty'] > 100000])
        bn_spot_metrics = {
            'market': 'Binance Spot',
            'trade_count': trades,
            'total_vol': vol,
            'avg_trade_size': avg_trade,
            'taker_buy_pct': buyer_vol / vol if vol > 0 else 0,
            'large_trades_100k': large_trades,
            'trades_per_sec': trades / 86400
        }
    else:
        bn_spot_metrics = None

    results = [m for m in [bn_fut_metrics, bn_spot_metrics, bb_fut_metrics, bb_spot_metrics] if m is not None]
    return pd.DataFrame(results)

if __name__ == "__main__":
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
    target_date = "2026-02-24" # Pick a random recent day that we know has data
    
    all_res = []
    for sym in symbols:
        res = process_trades(sym, target_date)
        if len(res) > 0:
            res.insert(0, 'symbol', sym)
            all_res.append(res)
            
    final_df = pd.concat(all_res, ignore_index=True)
    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print("\n--- Microstructure Tick Analysis (Date: 2026-02-24) ---")
    print(final_df.to_string(index=False))
