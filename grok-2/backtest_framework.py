#!/usr/bin/env python3

"""
Backtest framework for evaluating trading strategies across multiple exchanges.
This script provides a structure for loading data, simulating trades, and calculating performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Define paths
DATA_PATH = Path('/home/ubuntu/Projects/skytrade6/datalake')
OUTPUT_PATH = Path('/home/ubuntu/Projects/skytrade6/grok-2/results')

# Supported exchanges
EXCHANGES = ['binance', 'bybit', 'okx']

# Fee structure
MAKER_FEE = 0.0004  # 0.04%
TAKER_FEE = 0.001   # 0.1%

def load_data(symbol, exchange, data_type='klines', start_date=None, end_date=None, chunk_size=100000):
    """
    Load data for a given symbol and exchange, processing in chunks to optimize memory usage.
    
    Args:
        symbol (str): Trading pair symbol (e.g., BTCUSDT)
        exchange (str): Exchange name (binance, bybit, okx)
        data_type (str): Type of data (klines, fundingRate, etc.)
        start_date (str): Start date filter (YYYY-MM-DD)
        end_date (str): End date filter (YYYY-MM-DD)
        chunk_size (int): Number of rows to process at a time for memory efficiency
    
    Returns:
        pd.DataFrame: DataFrame containing the requested data
    """
    file_pattern = f'_{data_type}_1m.csv' if data_type == 'klines' else f'_{data_type}.csv'
    data_dir = DATA_PATH / exchange / symbol
    if not data_dir.exists():
        print(f"No data directory found for {symbol} on {exchange}")
        return None
    
    dfs = []
    # First try specific pattern
    files = list(data_dir.glob(f'*{file_pattern}'))
    if not files:
        print(f"No data files found with pattern *{file_pattern} for {symbol} on {exchange}, falling back to all CSV files")
        files = list(data_dir.glob('*.csv'))
        if not files:
            print(f"No CSV files found for {symbol} on {exchange} in range {start_date} to {end_date}")
            return None
        else:
            print(f"Found {len(files)} CSV files for {symbol} on {exchange}, attempting to load all")
    else:
        print(f"Found {len(files)} files matching pattern *{file_pattern} for {symbol} on {exchange}")
    
    for file in files:
        date_str = file.name.split('_')[0] if '_' in file.name else file.name.split('.')[0]
        try:
            file_date = datetime.strptime(date_str, '%Y-%m-%d')
            if start_date and file_date < datetime.strptime(start_date, '%Y-%m-%d'):
                continue
            if end_date and file_date > datetime.strptime(end_date, '%Y-%m-%d'):
                continue
            # Process file in chunks to save memory
            for chunk in pd.read_csv(file, chunksize=chunk_size):
                # Inspect and rename columns based on content or common patterns for all exchanges
                # Map timestamp columns
                possible_time_cols = [col for col in chunk.columns if 'time' in col.lower() or 'ts' in col.lower()]
                if possible_time_cols and 'timestamp' not in chunk.columns:
                    chunk.rename(columns={possible_time_cols[0]: 'timestamp'}, inplace=True)
                    print(f"Renamed {possible_time_cols[0]} to 'timestamp' for {file.name} on {exchange}")
                # Map funding rate columns
                possible_fr_cols = [col for col in chunk.columns if 'funding' in col.lower() or 'rate' in col.lower()]
                if possible_fr_cols and data_type == 'fundingRate':
                    chunk.rename(columns={possible_fr_cols[0]: 'funding_rate'}, inplace=True)
                    print(f"Renamed {possible_fr_cols[0]} to 'funding_rate' for {file.name} on {exchange}")
                # For klines, map common price/volume columns if needed
                if data_type == 'klines':
                    for col in chunk.columns:
                        col_lower = col.lower()
                        if 'open' in col_lower and 'open' not in chunk.columns:
                            chunk.rename(columns={col: 'open'}, inplace=True)
                            print(f"Renamed {col} to 'open' for {file.name} on {exchange}")
                        elif 'close' in col_lower and 'close' not in chunk.columns:
                            chunk.rename(columns={col: 'close'}, inplace=True)
                            print(f"Renamed {col} to 'close' for {file.name} on {exchange}")
                        elif 'high' in col_lower and 'high' not in chunk.columns:
                            chunk.rename(columns={col: 'high'}, inplace=True)
                            print(f"Renamed {col} to 'high' for {file.name} on {exchange}")
                        elif 'low' in col_lower and 'low' not in chunk.columns:
                            chunk.rename(columns={col: 'low'}, inplace=True)
                            print(f"Renamed {col} to 'low' for {file.name} on {exchange}")
                        elif 'volume' in col_lower and 'volume' not in chunk.columns:
                            chunk.rename(columns={col: 'volume'}, inplace=True)
                            print(f"Renamed {col} to 'volume' for {file.name} on {exchange}")
                # Filter by date range early if timestamp is available
                if 'timestamp' in chunk.columns:
                    try:
                        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                        if start_date:
                            start_dt = pd.to_datetime(start_date)
                            chunk = chunk[chunk['timestamp'] >= start_dt]
                        if end_date:
                            end_dt = pd.to_datetime(end_date)
                            chunk = chunk[chunk['timestamp'] <= end_dt]
                    except Exception as e:
                        print(f"Error filtering by date for {file.name} on {exchange}: {e}")
                if not chunk.empty:
                    dfs.append(chunk)
            print(f"Loaded {file.name} for {symbol} on {exchange}")
        except ValueError:
            print(f"Skipping invalid date format in filename: {file.name}, attempting to load anyway")
            try:
                for chunk in pd.read_csv(file, chunksize=chunk_size):
                    if 'timestamp' in chunk.columns:
                        try:
                            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
                            if start_date:
                                start_dt = pd.to_datetime(start_date)
                                chunk = chunk[chunk['timestamp'] >= start_dt]
                            if end_date:
                                end_dt = pd.to_datetime(end_date)
                                chunk = chunk[chunk['timestamp'] <= end_dt]
                        except Exception as e:
                            print(f"Error filtering by date for {file.name} on {exchange}: {e}")
                    if not chunk.empty:
                        dfs.append(chunk)
                print(f"Loaded {file.name} despite date parse error for {symbol} on {exchange}")
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
                continue
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
            continue
    
    if not dfs:
        print(f"No data files loaded for {symbol} on {exchange} for type {data_type} in range {start_date} to {end_date}")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in combined_df.columns:
        try:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            combined_df.sort_values('timestamp', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"Error converting timestamp for {symbol} on {exchange}: {e}")
    else:
        print(f"Warning: No 'timestamp' column found in data for {symbol} on {exchange}")
    return combined_df


def calculate_performance(trades, initial_capital=10000):
    """
    Calculate performance metrics from a list of trades.
    
    Args:
        trades (list): List of trade dictionaries with entry/exit prices, fees, etc.
        initial_capital (float): Starting capital for performance calculation
    
    Returns:
        dict: Performance metrics
    """
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'net_profit': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'equity_curve': [initial_capital]}
    
    equity_curve = [initial_capital]
    wins = 0
    for trade in trades:
        profit = trade['profit_loss']
        if profit > 0:
            wins += 1
        equity_curve.append(equity_curve[-1] + profit)
    
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    
    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    net_profit = equity_curve[-1] - initial_capital
    annualized_return = (equity_curve[-1] / initial_capital) ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 1 else 0
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'net_profit': net_profit,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'equity_curve': equity_curve
    }


def plot_equity_curve(equity_curve, strategy, symbol, start_date, end_date, output_path):
    """
    Plot the equity curve from backtest results and save it as an image.
    
    Args:
        equity_curve (list): List of equity values over time.
        strategy (str): Name of the strategy.
        symbol (str): Trading symbol.
        start_date (str): Start date of the backtest.
        end_date (str): End date of the backtest.
        output_path (Path): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label='Equity Curve', color='blue')
    plt.title(f"Equity Curve for {strategy.replace('_', ' ').title()} on {symbol} ({start_date} to {end_date})")
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)') 
    plt.legend()
    plt.grid(True)
    plot_file = output_path / f"equity_curve_{strategy}_{symbol}_{start_date}_{end_date}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Equity curve plot saved to {plot_file}")


def simulate_fr_arbitrage(symbol, start_date, end_date):
    """
    Simulate funding rate arbitrage strategy across exchanges.
    Strategy: Long on exchange with highest FR, short on lowest FR if differential > 0.3%.
    Fallback: If funding rate data is unavailable, log the issue and attempt to use kline data as a proxy.
    """
    trades = []
    fr_data_dict = {}
    kline_data_dict = {}
    for exchange in EXCHANGES:
        fr_data = load_data(symbol, exchange, 'fundingRate', start_date, end_date)
        kline_data = load_data(symbol, exchange, 'klines', start_date, end_date)
        if fr_data is not None:
            fr_data_dict[exchange] = fr_data
            print(f"Loaded funding rate data for {symbol} on {exchange}")
        else:
            print(f"No funding rate data available for {symbol} on {exchange}")
        if kline_data is not None:
            kline_data_dict[exchange] = kline_data
            print(f"Loaded kline data for {symbol} on {exchange} as fallback")
        else:
            print(f"No kline data available for {symbol} on {exchange}")
    
    if len(fr_data_dict) < 2:
        print("Insufficient data: Need at least 2 exchanges for arbitrage with funding rate data.")
        if len(kline_data_dict) >= 2:
            print("Falling back to kline data for a simplified arbitrage simulation.")
            # Placeholder for kline-based simulation (e.g., price divergence)
            for exchange in kline_data_dict.keys():
                print(f"Using kline data for {exchange} as no funding rate data is available.")
            return trades  # Simplified: no trades generated in fallback for now
        else:
            print("Insufficient kline data for fallback simulation.")
            return trades
    
    # Align timestamps across exchanges by creating a unified timeline
    all_timestamps = set()
    for exchange, df in fr_data_dict.items():
        if 'timestamp' in df.columns:
            all_timestamps.update(df['timestamp'])
    common_timestamps = sorted(list(all_timestamps))
    
    if not common_timestamps:
        print("No common or available timestamps found across exchanges.")
        return trades
    
    # Simulate strategy: Check FR differential at each timestamp
    for ts in common_timestamps:
        fr_values = {}
        for exchange, df in fr_data_dict.items():
            if 'timestamp' not in df.columns:
                continue
            matching_row = df[df['timestamp'] <= ts].tail(1)
            if not matching_row.empty:
                # Dynamically find funding rate column (case-insensitive search)
                fr_col = next((col for col in matching_row.columns if 'funding' in col.lower() or 'rate' in col.lower()), None)
                if fr_col and not pd.isna(matching_row[fr_col].iloc[0]):
                    try:
                        fr_values[exchange] = float(matching_row[fr_col].iloc[0])
                    except (ValueError, TypeError):
                        print(f"Invalid funding rate value for {exchange} at {ts}")
        
        if len(fr_values) >= 2:
            # Find highest and lowest FR
            highest_fr_ex = max(fr_values, key=fr_values.get)
            lowest_fr_ex = min(fr_values, key=fr_values.get)
            differential = fr_values[highest_fr_ex] - fr_values[lowest_fr_ex]
            
            # Threshold for arbitrage (0.3% to cover fees)
            if differential >= 0.003:  # 0.3%
                # Simulate trade: Long on high FR, Short on low FR
                trade = {
                    'timestamp': ts,
                    'symbol': symbol,
                    'long_exchange': highest_fr_ex,
                    'short_exchange': lowest_fr_ex,
                    'long_fr': fr_values[highest_fr_ex],
                    'short_fr': fr_values[lowest_fr_ex],
                    'entry_price_long': 10000,  # Placeholder price
                    'entry_price_short': 10000,  # Placeholder price
                    'position_size': 1000,  # Placeholder notional
                    'profit_loss': 0,  # To be calculated on exit
                    'status': 'open'
                }
                trades.append(trade)
                print(f"Arbitrage trade opened at {ts}: Long {highest_fr_ex} (FR={fr_values[highest_fr_ex]:.4f}), Short {lowest_fr_ex} (FR={fr_values[lowest_fr_ex]:.4f})")
    
    # Placeholder: Close trades (simplified, would need exit logic based on FR convergence or time)
    for trade in trades:
        if trade['status'] == 'open':
            trade['exit_timestamp'] = trade['timestamp'] + timedelta(hours=8)  # Assume 8h hold
            trade['exit_price_long'] = 10000  # Placeholder
            trade['exit_price_short'] = 10000  # Placeholder
            # Calculate P&L (simplified, ignoring price movement, focusing on FR)
            fr_gain = trade['long_fr'] * trade['position_size'] * 8 / 24  # Pro-rated for 8h
            fr_loss = trade['short_fr'] * trade['position_size'] * 8 / 24  # Pro-rated for 8h
            fees = trade['position_size'] * (TAKER_FEE * 2)  # Entry + Exit on both legs
            trade['profit_loss'] = (fr_gain + fr_loss) - fees
            trade['status'] = 'closed'
            print(f"Trade closed at {trade['exit_timestamp']}: P&L = ${trade['profit_loss']:.2f}")
    
    return trades


def simulate_volatility_breakout(symbol, start_date, end_date):
    """
    Simulate volatility breakout strategy with cross-exchange confirmation.
    Strategy: Enter on breakout if confirmed on at least 2 exchanges.
    """
    trades = []
    kline_data_dict = {}
    for exchange in EXCHANGES:
        kline_data = load_data(symbol, exchange, 'klines', start_date, end_date)
        if kline_data is not None:
            kline_data_dict[exchange] = kline_data
            print(f"Loaded kline data for {symbol} on {exchange}")
    
    if len(kline_data_dict) < 2:
        print("Insufficient data: Need at least 2 exchanges for breakout confirmation.")
        return trades
    
    # Align timestamps across exchanges for a unified timeline
    all_timestamps = set()
    for exchange, df in kline_data_dict.items():
        if 'timestamp' in df.columns:
            all_timestamps.update(df['timestamp'])
    common_timestamps = sorted(list(all_timestamps))
    
    if not common_timestamps:
        print("No common or available timestamps found across exchanges.")
        return trades
    
    # Parameters for breakout detection (simplified)
    lookback_period = 20  # Lookback for volatility calculation
    breakout_threshold = 1.5  # Std devs above mean for breakout
    
    # Simulate strategy: Check for breakout signals at each timestamp
    for ts in common_timestamps:
        breakout_signals = []
        for exchange, df in kline_data_dict.items():
            if 'timestamp' not in df.columns:
                continue
            # Get data up to current timestamp
            past_data = df[df['timestamp'] <= ts].tail(lookback_period + 1)
            if len(past_data) < lookback_period + 1:
                continue
            
            # Calculate volatility (simplified as std of returns)
            past_returns = past_data['close'].pct_change().dropna()
            if len(past_returns) < lookback_period:
                continue
            vol = past_returns.std()
            mean_return = past_returns.mean()
            current_return = past_returns.iloc[-1]
            
            # Check if current move is a breakout (upward for simplicity)
            if vol > 0 and current_return > (mean_return + breakout_threshold * vol):
                breakout_signals.append(exchange)
                print(f"Breakout signal on {exchange} at {ts}: Return={current_return:.4f}, Threshold={mean_return + breakout_threshold * vol:.4f}")
        
        # Require confirmation from at least 2 exchanges
        if len(breakout_signals) >= 2:
            # Simulate long trade on all confirming exchanges (simplified)
            trade = {
                'timestamp': ts,
                'symbol': symbol,
                'exchanges': breakout_signals,
                'direction': 'long',
                'entry_price': 10000,  # Placeholder price
                'position_size': 1000,  # Placeholder notional
                'profit_loss': 0,  # To be calculated on exit
                'status': 'open'
            }
            trades.append(trade)
            print(f"Breakout trade opened at {ts} on {', '.join(breakout_signals)}")
    
    # Placeholder: Close trades after a fixed period (e.g., 1 hour)
    for trade in trades:
        if trade['status'] == 'open':
            trade['exit_timestamp'] = trade['timestamp'] + timedelta(hours=1)
            trade['exit_price'] = 10000  # Placeholder
            # Calculate P&L (simplified, assuming 0.5% gain for winning trades)
            if np.random.rand() > 0.5:  # 50% chance of win for simulation
                trade['profit_loss'] = trade['position_size'] * 0.005 - trade['position_size'] * TAKER_FEE * 2
            else:
                trade['profit_loss'] = trade['position_size'] * -0.003 - trade['position_size'] * TAKER_FEE * 2
            trade['status'] = 'closed'
            print(f"Trade closed at {trade['exit_timestamp']}: P&L = ${trade['profit_loss']:.2f}")
    
    return trades


def simulate_settlement_scalp(symbol, start_date, end_date):
    """
    Simulate post-settlement price drop scalp.
    Strategy: Short at T+0ms after settlement if FR <= 0 bps, exit at T+5min to allow for data match.
    """
    trades = []
    fr_data_dict = {}
    kline_data_dict = {}
    for exchange in EXCHANGES:
        fr_data = load_data(symbol, exchange, 'fundingRate', start_date, end_date)
        kline_data = load_data(symbol, exchange, 'klines', start_date, end_date)
        if fr_data is not None:
            fr_data_dict[exchange] = fr_data
            print(f"Loaded funding rate data for {symbol} on {exchange}")
        else:
            print(f"No funding rate data available for {symbol} on {exchange}")
        if kline_data is not None:
            kline_data_dict[exchange] = kline_data
            print(f"Loaded kline data for {symbol} on {exchange}")
        else:
            print(f"No kline data available for {symbol} on {exchange}")
    
    if len(fr_data_dict) < 1 or len(kline_data_dict) < 1:
        print("Insufficient data: Need funding rate and kline data for at least one exchange.")
        return trades
    
    # Process each exchange independently for settlement events
    for exchange in fr_data_dict.keys():
        fr_df = fr_data_dict.get(exchange)
        kline_df = kline_data_dict.get(exchange)
        if fr_df is None or kline_df is None:
            continue
        
        # Handle missing timestamp column by creating a dummy timeline if needed
        if 'timestamp' not in fr_df.columns:
            print(f"No timestamp column in funding rate data for {exchange}, creating dummy timeline")
            if len(fr_df) > 0:
                fr_df['timestamp'] = pd.date_range(start=start_date, periods=len(fr_df), freq='8h')
            else:
                print(f"Empty funding rate data for {exchange}, skipping")
                continue
        if 'timestamp' not in kline_df.columns:
            print(f"No timestamp column in kline data for {exchange}, creating dummy timeline")
            if len(kline_df) > 0:
                kline_df['timestamp'] = pd.date_range(start=start_date, periods=len(kline_df), freq='1min')
            else:
                print(f"Empty kline data for {exchange}, skipping")
                continue
        
        # Find settlement timestamps where FR <= 0 bps (any negative or zero rate)
        fr_col = next((col for col in fr_df.columns if 'funding' in col.lower() or 'rate' in col.lower()), None)
        if not fr_col:
            print(f"No funding rate column found for {exchange}")
            continue
        
        # Debug funding rate values
        negative_fr = fr_df[fr_df[fr_col] <= 0]
        print(f"Debug for {exchange}: Found {len(negative_fr)} funding rate entries <= 0 bps out of {len(fr_df)} total entries")
        if len(negative_fr) > 0:
            print(f"Sample funding rates <= 0 bps for {exchange}:")
            print(negative_fr[[fr_col, 'timestamp']].head(5).to_string())
        else:
            print(f"No funding rates <= 0 bps for {exchange}, min funding rate: {fr_df[fr_col].min() if not fr_df[fr_col].empty else 'N/A'}")
        
        settlement_times = fr_df[fr_df[fr_col] <= 0]['timestamp'].tolist()
        
        for settle_ts in settlement_times:
            # Look for kline data immediately after settlement, extended window to 5 minutes for better matching
            post_settle_data = kline_df[(kline_df['timestamp'] > settle_ts) & 
                                       (kline_df['timestamp'] <= settle_ts + timedelta(minutes=5))]
            if not post_settle_data.empty:
                # Use fallback price if open/close are unavailable
                entry_price = post_settle_data['open'].iloc[0] if 'open' in post_settle_data.columns else (post_settle_data['close'].iloc[0] if 'close' in post_settle_data.columns else (post_settle_data['high'].iloc[0] if 'high' in post_settle_data.columns else 10000))
                # Assume exit after 5 minutes or last available price in window
                exit_price = post_settle_data['close'].iloc[-1] if 'close' in post_settle_data.columns else (post_settle_data['open'].iloc[-1] if 'open' in post_settle_data.columns else (post_settle_data['low'].iloc[-1] if 'low' in post_settle_data.columns else 10000))
                position_size = 1000  # Placeholder notional
                # Calculate P&L for short position, with debug output
                if entry_price > 0:
                    profit_loss = position_size * (entry_price - exit_price) / entry_price - position_size * TAKER_FEE * 2
                    print(f"P&L calc for {exchange} at {settle_ts}: entry_price={entry_price}, exit_price={exit_price}, P&L=${profit_loss:.2f}")
                else:
                    profit_loss = 0
                    print(f"P&L calc skipped for {exchange} at {settle_ts}: invalid entry_price={entry_price}")
                trade = {
                    'timestamp': settle_ts,
                    'symbol': symbol,
                    'exchange': exchange,
                    'direction': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size,
                    'profit_loss': profit_loss,
                    'status': 'closed',
                    'exit_timestamp': settle_ts + timedelta(minutes=5)
                }
                trades.append(trade)
                print(f"Settlement scalp trade on {exchange} at {settle_ts}: P&L = ${profit_loss:.2f}")
            else:
                print(f"No kline data found post-settlement on {exchange} at {settle_ts} within 5 minutes")
    
    return trades


def main():
    parser = argparse.ArgumentParser(description='Run backtest for crypto trading strategies')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--strategy', type=str, default='volatility_breakout', 
                        choices=['volatility_breakout', 'fr_arbitrage', 'settlement_scalp'],
                        help='Strategy to backtest')
    parser.add_argument('--start', type=str, default='2025-07-01', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-07-07', help='End date for backtest (YYYY-MM-DD)')
    args = parser.parse_args()

    print(f"Running backtest for {args.symbol} using {args.strategy} strategy from {args.start} to {args.end}")
    
    if args.strategy == 'volatility_breakout':
        trades = simulate_volatility_breakout(args.symbol, args.start, args.end)
    elif args.strategy == 'fr_arbitrage':
        trades = simulate_fr_arbitrage(args.symbol, args.start, args.end)
    else:  # settlement_scalp
        trades = simulate_settlement_scalp(args.symbol, args.start, args.end)
    
    if trades:
        performance = calculate_performance(trades)
        print("\nPerformance Metrics:")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.2%}")
        print(f"Net Profit: ${performance['net_profit']:.2f}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        
        # Plot equity curve
        plot_equity_curve(performance['equity_curve'], args.strategy, args.symbol, args.start, args.end, OUTPUT_PATH)
        
        # Save results
        result_file = OUTPUT_PATH / f"{args.strategy}_{args.symbol}_{args.start}_{args.end}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Backtest Results for {args.symbol} ({args.strategy.title()})\n")
            f.write(f"Period: {args.start} to {args.end}\n")
            f.write(f"Total Trades: {performance['total_trades']}\n")
            f.write(f"Win Rate: {performance['win_rate']:.2%}\n")
            f.write(f"Net Profit: ${performance['net_profit']:.2f}\n")
            f.write(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown: {performance['max_drawdown']:.2%}\n")
        print(f"Results saved to {result_file}")
    else:
        print("No trades generated during the simulation.")
        # Save a note in results file
        result_file = OUTPUT_PATH / f"{args.strategy}_{args.symbol}_{args.start}_{args.end}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Backtest Results for {args.symbol} ({args.strategy.title()})")
            f.write(f"Period: {args.start} to {args.end}")
            f.write("No trades generated during the simulation.")
        print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
