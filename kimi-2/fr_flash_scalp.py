#!/usr/bin/env python3
"""FR Flash Scalp - Test settlement edge with tick data"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATALAKE = Path('/home/ubuntu/Projects/skytrade6/datalake/bybit')

print('='*70)
print('FR FLASH SCALP - Tick-level Test')
print('='*70)

sym = 'SOLUSDT'

# Load funding rates to find settlement times
fr_files = sorted((DATALAKE / sym).glob('*_funding_rate.csv'))
fr_data = []
for f in fr_files[-60:]:  # Last 60 days
    df = pd.read_csv(f)
    fr_data.append(df)

fr_df = pd.concat(fr_data, ignore_index=True)
fr_df['timestamp'] = pd.to_datetime(fr_df['timestamp'], unit='ms')
fr_df = fr_df.sort_values('timestamp').reset_index(drop=True)

print(f'Loaded {len(fr_df)} funding rate records')

# Get settlements from 2025 (where we have trade data)
settlements = fr_df[fr_df['timestamp'] >= '2025-01-01']['timestamp'].tolist()
print(f'Settlements from 2025: {len(settlements)}')

if len(settlements) == 0:
    print('No 2025 settlements found')
    exit()

# For each settlement, analyze trades around it
total_pnl = 0
total_trades = 0

for i, settlement_time in enumerate(settlements[:20]):  # Test first 20
    date_str = settlement_time.strftime('%Y-%m-%d')
    trade_file = DATALAKE / sym / f'{date_str}_trades.csv.gz'
    
    if not trade_file.exists():
        continue
    
    # Load trades for this day
    trades_df = pd.read_csv(trade_file, compression='gzip')
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='s')
    trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)
    
    # Find trades around settlement
    # Settlement is typically at 00:00, 08:00, 16:00 UTC
    window_start = settlement_time - timedelta(seconds=5)
    window_end = settlement_time + timedelta(seconds=10)
    
    around_settlement = trades_df[(trades_df['timestamp'] >= window_start) & 
                                  (trades_df['timestamp'] <= window_end)]
    
    if len(around_settlement) < 10:
        continue
    
    # Strategy: Long 1s before settlement, exit 1-3s after
    entry_time = settlement_time - timedelta(seconds=1)
    exit_time = settlement_time + timedelta(seconds=2)
    
    # Find entry price (closest trade to entry_time)
    entry_trades = around_settlement[around_settlement['timestamp'] <= entry_time]
    if len(entry_trades) == 0:
        continue
    entry_price = entry_trades.iloc[-1]['price']
    
    # Find exit price (closest trade to exit_time)
    exit_trades = around_settlement[around_settlement['timestamp'] >= exit_time]
    if len(exit_trades) == 0:
        continue
    exit_price = exit_trades.iloc[0]['price']
    
    # Calculate P&L
    pnl_bps = (exit_price - entry_price) / entry_price * 10000
    
    # Get funding rate for this period
    fr_row = fr_df[fr_df['timestamp'] <= settlement_time].iloc[-1] if len(fr_df[fr_df['timestamp'] <= settlement_time]) > 0 else None
    fr_payment = fr_row['fundingRate'] * 10000 if fr_row is not None else 0
    
    # Net P&L = Price change + FR payment - fees
    # For long: we receive FR if FR > 0
    total_pnl_trade = pnl_bps + fr_payment - 8  # 8 bps maker fees
    
    print(f'Settlement {i+1}: {settlement_time}')
    print(f'  Entry: {entry_price:.4f} at {entry_trades.iloc[-1]["timestamp"]}')
    print(f'  Exit: {exit_price:.4f} at {exit_trades.iloc[0]["timestamp"]}')
    print(f'  Price change: {pnl_bps:.2f}bps | FR: {fr_payment:.2f}bps | Net: {total_pnl_trade:.2f}bps')
    
    total_pnl += total_pnl_trade
    total_trades += 1

print('='*70)
print(f'Total trades: {total_trades}')
print(f'Total P&L: {total_pnl:.2f}bps')
print(f'Average per trade: {total_pnl/total_trades:.2f}bps' if total_trades > 0 else 'N/A')
print('='*70)
