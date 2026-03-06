#!/usr/bin/env python3
"""
PRODUCTION SCRIPT: Asymmetric Macro Trend Breakout
==================================================
Exchange: Bybit USDT Perpetual
Timeframe: 4H
Logic: 20-period Donchian Breakout + Volume Spike (2x) + 200 EMA + KER (>0.15)
Risk: 2% risk per trade, 10% Stop Loss, 20% Take Profit
"""
import ccxt
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import os

# --- CONFIGURATION ---
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT', 
           'XRP/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'ADA/USDT:USDT', 
           'DOT/USDT:USDT', 'NEAR/USDT:USDT']

TIMEFRAME = '4h'
RISK_PER_TRADE = 0.02
SL_PCT = 0.10
TP_PCT = 0.20
KER_THRESHOLD = 0.15

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)

# Initialize exchange
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    }
})

def fetch_4h_data(symbol, limit=250):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

def calc_ker(df, period=21):
    change = abs(df['close'] - df['close'].shift(period))
    volatility = abs(df['close'] - df['close'].shift(1)).rolling(period).sum()
    ker = change / volatility
    return ker.iloc[-1]

def analyze_symbol(symbol, btc_ker):
    df = fetch_4h_data(symbol)
    if df is None or len(df) < 200:
        return None
        
    current_close = df['close'].iloc[-1]
    
    # Exclude the currently forming (unclosed) candle for signal calculations
    df_closed = df.iloc[:-1].copy()
    
    # Metrics based on closed candles
    ema_200 = df_closed['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    high_20 = df_closed['high'].rolling(20).max().iloc[-1]
    low_20 = df_closed['low'].rolling(20).min().iloc[-1]
    
    vol_ma_20 = df_closed['volume'].rolling(20).mean().iloc[-1]
    last_closed_vol = df_closed['volume'].iloc[-1]
    
    local_ker = calc_ker(df_closed)
    
    # Logic
    macro_bull = current_close > ema_200
    vol_spike = last_closed_vol > (vol_ma_20 * 2.0)
    regime_ok = (btc_ker >= KER_THRESHOLD) or (local_ker >= KER_THRESHOLD)
    
    long_sig = (current_close > high_20) and macro_bull and vol_spike and regime_ok
    short_sig = (current_close < low_20) and (not macro_bull) and vol_spike and regime_ok
    
    signal = 0
    if long_sig: signal = 1
    elif short_sig: signal = -1
    
    return {
        'symbol': symbol,
        'signal': signal,
        'close': current_close,
        'high_20': high_20,
        'low_20': low_20,
        'ema_200': ema_200,
        'ker': local_ker
    }

def get_positions():
    try:
        positions = exchange.fetch_positions(SYMBOLS)
        active = [p for p in positions if float(p['contracts']) > 0]
        return active
    except Exception as e:
        logging.error(f"Error fetching positions: {e}")
        return []

def execute_trade(symbol, signal, price):
    try:
        balance = exchange.fetch_balance()
        usdt_equity = balance['total']['USDT']
        
        # Risk management logic
        pos_size_usd = usdt_equity * (RISK_PER_TRADE / SL_PCT)
        qty = pos_size_usd / price
        
        market = exchange.market(symbol)
        qty = float(exchange.amount_to_precision(symbol, qty))
        
        if qty < market['limits']['amount']['min']:
            logging.warning(f"Calculated qty {qty} is below exchange min limit for {symbol}")
            return False
            
        side = 'buy' if signal == 1 else 'sell'
        
        # 1. Enter Market
        logging.info(f"EXECUTING {side.upper()} on {symbol} | Qty: {qty} | Price: {price}")
        order = exchange.create_order(symbol, 'market', side, qty)
        
        # 2. Place SL / TP
        sl_price = price * (1 - SL_PCT) if signal == 1 else price * (1 + SL_PCT)
        tp_price = price * (1 + TP_PCT) if signal == 1 else price * (1 - TP_PCT)
        
        sl_price = float(exchange.price_to_precision(symbol, sl_price))
        tp_price = float(exchange.price_to_precision(symbol, tp_price))
        
        close_side = 'sell' if signal == 1 else 'buy'
        
        # Send Take Profit
        exchange.create_order(symbol, 'limit', close_side, qty, tp_price, params={'reduceOnly': True})
        
        # Send Stop Loss
        exchange.create_order(symbol, 'stop_market', close_side, qty, params={'stopPrice': sl_price, 'reduceOnly': True})
        
        logging.info(f"Orders placed successfully for {symbol}. TP: {tp_price}, SL: {sl_price}")
        return True
        
    except Exception as e:
        logging.error(f"CRITICAL ERROR executing {symbol}: {e}")
        return False

def main():
    logging.info("--- Starting Asymmetric Macro Trend Bot ---")
    
    # Wait until exactly the 4H open (e.g., 00:00, 04:00, 08:00 UTC)
    # For now, we just run immediately for the current block
    
    btc_df = fetch_4h_data('BTC/USDT:USDT')
    if btc_df is None:
        logging.error("Failed to load BTC data for regime filter.")
        return
        
    btc_ker = calc_ker(btc_df.iloc[:-1])
    logging.info(f"BTC KER (Regime): {btc_ker:.3f} | Trending: {btc_ker >= KER_THRESHOLD}")
    
    active_positions = get_positions()
    active_symbols = [p['symbol'] for p in active_positions]
    
    for symbol in SYMBOLS:
        if symbol in active_symbols:
            logging.info(f"Already in position for {symbol}. Skipping.")
            continue
            
        res = analyze_symbol(symbol, btc_ker)
        if not res: continue
        
        if res['signal'] != 0:
            logging.info(f"*** SIGNAL GENERATED *** | {res['symbol']} | Direction: {res['signal']}")
            execute_trade(symbol, res['signal'], res['close'])

if __name__ == "__main__":
    main()
