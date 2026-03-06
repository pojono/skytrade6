import time
import datetime
import numpy as np
import pandas as pd
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PowderKegLive")

GOLDEN_CLUSTER = ['BTCUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'WLDUSDT']

class PowderKegTrader:
    def __init__(self, portfolio_equity=10000.0):
        self.portfolio_equity = portfolio_equity
        self.active_trades = {}
        # Simulated rolling 7-day memory buffer
        self.data_buffer = {sym: pd.DataFrame(columns=['oi', 'funding_rate', 'close', 'high', 'low']) for sym in GOLDEN_CLUSTER}
        
    def _calculate_atr(self, symbol, current_price):
        """Calculate 24-hour ATR percentage for position sizing."""
        df = self.data_buffer[symbol].tail(24 * 60) # Last 24 hours (assuming 1m data in buffer)
        if len(df) < 2:
            return 0.05 # Default 5% ATR if not enough data
            
        df['prev_close'] = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['prev_close']).abs()
        tr3 = (df['low'] - df['prev_close']).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr_absolute = tr.mean() * 60 * 24 # Rough daily approximation from 1m 
        return atr_absolute / current_price

    def _get_z_scores(self, symbol, current_oi, current_fr):
        """Calculate 7-day rolling Z-scores for OI and Funding Rate."""
        df = self.data_buffer[symbol].copy()
        if len(df) < 168: # Need at least 168 hours of data ideally, but we will mock it for this script
            return 0.0, 0.0
            
        # Append current reading
        new_row = pd.DataFrame({'oi': [current_oi], 'funding_rate': [current_fr]})
        df = pd.concat([df, new_row], ignore_index=True)
        
        oi_mean = df['oi'].tail(168).mean() # Mocking hourly data points
        oi_std = df['oi'].tail(168).std()
        oi_z = (current_oi - oi_mean) / oi_std if oi_std > 0 else 0
        
        fr_mean = df['funding_rate'].tail(168).mean()
        fr_std = df['funding_rate'].tail(168).std()
        fr_z = (current_fr - fr_mean) / fr_std if fr_std > 0 else 0
        
        return oi_z, fr_z

    def execute_trade(self, symbol, side, entry_price, atr_pct):
        """Execute the trade using Inverse-ATR sizing and Wide Net rules."""
        if symbol in self.active_trades:
            logger.info(f"[{symbol}] Trade already active. Skipping.")
            return

        # Sizing: Risk exactly 1% of equity per 1x ATR move
        target_risk = self.portfolio_equity * 0.01
        notional_size = target_risk / atr_pct
        
        # Cap max leverage at 3x to prevent exchange liquidation
        notional_size = min(notional_size, self.portfolio_equity * 3.0)
        
        qty = notional_size / entry_price
        
        # Set exact 10% limit order Take Profit
        if side == 'SHORT':
            tp_price = entry_price * 0.90
        else:
            tp_price = entry_price * 1.10
            
        expire_time = datetime.datetime.now() + datetime.timedelta(hours=24)
        
        self.active_trades[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'qty': qty,
            'notional': notional_size,
            'tp_price': tp_price,
            'expire_time': expire_time
        }
        
        logger.warning(f"�� EXECUTED {side} on {symbol}! Entry: {entry_price:.4f} | Size: ${notional_size:.2f} | TP: {tp_price:.4f} | Time Stop: {expire_time}")

    def on_market_tick(self, symbol, current_price, current_oi, current_fr, high_1m, low_1m):
        """Called every minute by the websocket streams."""
        # Update Buffer (Simulated)
        # In production, this pushes to Redis and keeps max 7 days of 1-minute data
        if len(self.data_buffer[symbol]) > 10080: # 7 days * 24h * 60m
            self.data_buffer[symbol] = self.data_buffer[symbol].iloc[1:]
            
        new_row = pd.DataFrame({'oi': [current_oi], 'funding_rate': [current_fr], 'close': [current_price], 'high': [high_1m], 'low': [low_1m]})
        self.data_buffer[symbol] = pd.concat([self.data_buffer[symbol], new_row], ignore_index=True)

        # 1. Manage existing trades (Take Profit & Time Stop)
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            
            # Check Time Stop
            if datetime.datetime.now() >= trade['expire_time']:
                pnl = (current_price - trade['entry_price']) / trade['entry_price'] * (-1 if trade['side'] == 'SHORT' else 1)
                logger.info(f"⏰ TIME STOP: Closed {trade['side']} {symbol} at {current_price:.4f} (PnL: {pnl*100:.2f}%)")
                del self.active_trades[symbol]
                return

            # Check Take Profit
            if trade['side'] == 'SHORT' and low_1m <= trade['tp_price']:
                logger.info(f"💰 TAKE PROFIT: Closed {trade['side']} {symbol} at {trade['tp_price']:.4f} (PnL: +10.00%)")
                del self.active_trades[symbol]
                return
            elif trade['side'] == 'LONG' and high_1m >= trade['tp_price']:
                logger.info(f"💰 TAKE PROFIT: Closed {trade['side']} {symbol} at {trade['tp_price']:.4f} (PnL: +10.00%)")
                del self.active_trades[symbol]
                return
                
            return # Don't take a new trade if one is active
            
        # 2. Check for new signals
        # Only check signals once an hour (at the top of the hour) to avoid 1-minute noise
        if datetime.datetime.now().minute == 0:
            oi_z, fr_z = self.get_z_scores(symbol, current_oi, current_fr)
            
            if oi_z > 2.0 and fr_z > 2.0:
                atr_pct = self._calculate_atr(symbol, current_price)
                self.execute_trade(symbol, 'SHORT', current_price, atr_pct)
                
            elif oi_z > 2.0 and fr_z < -2.0:
                atr_pct = self._calculate_atr(symbol, current_price)
                self.execute_trade(symbol, 'LONG', current_price, atr_pct)

# Mock Runner
if __name__ == "__main__":
    logger.info("Starting Volatility-Adjusted Powder Keg execution engine...")
    logger.info(f"Tracking Golden Cluster: {GOLDEN_CLUSTER}")
    logger.info("Awaiting websocket signals...")
    # In production, this script would connect to ccxt or exchange websockets.
