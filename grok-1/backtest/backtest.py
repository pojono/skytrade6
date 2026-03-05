import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/grok-1')
from data.data_loader import load_bybit_data
from research.signals import add_ma_crossover_signals, add_combined_signals

def backtest_strategy(symbol='SOLUSDT', start_date='2025-07-01', end_date='2026-02-28', hold_period=2, consecutive=2):
    """
    Backtest the MA Crossover strategy.
    """
    # Load data
    df = load_bybit_data(symbol, start_date, end_date)

    # Add signals
    df = add_ma_crossover_signals(df)
    df = add_combined_signals(df)

    df = df.dropna()
    df = df.reset_index(drop=True)

    print("MA signal counts:", df['ma_signal'].value_counts())
    print("Combined signal counts:", df['combined_signal'].value_counts())
    print("FR regime counts:", df['fr_regime'].value_counts())

    # Simulate trades
    df['position'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['pnl'] = 0.0
    df['fees'] = 0.0

    position = 0
    entry_price = np.nan
    hold_counter = 0

    for i in range(len(df)):
        signal = df.iloc[i]['combined_signal']

        if position == 0 and signal != 0 and i < len(df) - 1:
            # Enter
            position = signal
            entry_price = df.iloc[i + 1]['open']
            hold_counter = 0
            df.iloc[i + 1, df.columns.get_loc('entry_price')] = entry_price
            df.iloc[i + 1, df.columns.get_loc('position')] = position

            print(f"Enter at i={i}, open={df.iloc[i]['open']}, signal={signal}")

        elif position != 0:
            hold_counter += 1
            df.iloc[i, df.columns.get_loc('position')] = position

            if hold_counter >= hold_period:
                # Exit
                exit_price = df.iloc[i]['close']
                df.iloc[i, df.columns.get_loc('exit_price')] = exit_price

                # P&L
                if position == 1:
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price

                # Fees: 20bps round-trip
                fees = 0.002  # 20bps
                net_pnl = pnl - fees

                df.iloc[i, df.columns.get_loc('pnl')] = net_pnl
                df.iloc[i, df.columns.get_loc('fees')] = fees

                # Reset
                position = 0
                entry_price = np.nan
                hold_counter = 0

    # Cumulative returns
    df['cum_pnl'] = df['pnl'].cumsum()

    # Metrics
    pnl_series = df['pnl'][df['pnl'] != 0]
    total_trades = len(pnl_series)
    if total_trades > 0:
        win_rate = (pnl_series > 0).sum() / total_trades
        avg_pnl = pnl_series.mean()
        total_return = pnl_series.sum()
        sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(252 / 4) if pnl_series.std() > 0 else 0
    else:
        win_rate = 0
        avg_pnl = 0
        total_return = 0
        sharpe = 0

    print(f"Symbol: {symbol}")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg P&L per Trade: {avg_pnl:.4f}")
    print(f"Total Return: {total_return:.4f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    return df

if __name__ == '__main__':
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'LTCUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'TRXUSDT', 'DOGEUSDT', 'BNBUSDT', 'MATICUSDT', 'ICPUSDT', 'FILUSDT', 'ETCUSDT', 'XLMUSDT', 'HBARUSDT', 'VETUSDT', 'THETAUSDT']
    for symbol in symbols:
        print(f"\n=== Testing {symbol} ===")
        try:
            result_df = backtest_strategy(symbol=symbol, start_date='2025-07-01', end_date='2026-02-28', hold_period=2, consecutive=2)
            # Results printed inside function
        except Exception as e:
            print(f"Skipped {symbol}: {e}")
