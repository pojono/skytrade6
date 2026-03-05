import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sys
sys.path.append('/home/ubuntu/Projects/skytrade6/grok-1')
from data.data_loader import load_bybit_data
from research.signals import add_momentum_continuation_signals, add_combined_signals

def backtest_strategy(symbol='SOLUSDT', start_date='2025-07-01', end_date='2026-02-28', hold_period=2, consecutive=2):
    try:
        df = load_bybit_data(symbol, start_date, end_date)
        df = add_momentum_continuation_signals(df, consecutive=consecutive)
        df = add_combined_signals(df)
        df = df.dropna()
        df = df.reset_index(drop=True)
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['position'] = 0
        df['pnl'] = np.nan
        df['fees'] = np.nan
        position = 0
        entry_price = np.nan
        hold_counter = 0
        for i in range(len(df)):
            signal = df.iloc[i]['combined_signal']
            if position == 0 and signal != 0:
                position = signal
                entry_price = df.iloc[i]['open']
                hold_counter = 0
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                df.iloc[i, df.columns.get_loc('position')] = position
            elif position != 0:
                hold_counter += 1
                df.iloc[i, df.columns.get_loc('position')] = position
                if hold_counter >= hold_period:
                    exit_price = df.iloc[i]['close']
                    df.iloc[i, df.columns.get_loc('exit_price')] = exit_price
                    pnl = (exit_price - entry_price) / entry_price if position == 1 else (entry_price - exit_price) / entry_price
                    fees = 0.002
                    net_pnl = pnl - fees
                    df.iloc[i, df.columns.get_loc('pnl')] = net_pnl
                    df.iloc[i, df.columns.get_loc('fees')] = fees
                    position = 0
                    entry_price = np.nan
                    hold_counter = 0
        # Calculate metrics
        pnl_df = df.dropna(subset=['pnl'])
        total_trades = len(pnl_df)
        if total_trades == 0:
            return {'symbol': symbol, 'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_return': 0, 'sharpe': 0}
        win_rate = (pnl_df['pnl'] > 0).mean()
        avg_pnl = pnl_df['pnl'].mean()
        total_return = pnl_df['pnl'].sum()
        # Proper annualized Sharpe based on daily returns
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_pnl = df.groupby('date')['pnl'].sum().dropna()
        if len(daily_pnl) > 1:
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365) if daily_pnl.std() > 0 else 0
        else:
            sharpe = 0
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'sharpe': sharpe
        }
    except Exception as e:
        print(f"Error for {symbol}: {e}")
        return {'symbol': symbol, 'total_trades': 0, 'win_rate': 0, 'avg_pnl': 0, 'total_return': 0, 'sharpe': 0}

def parse_volume(v):
    v = v.replace('$', '').replace(',', '')
    if 'B' in v:
        return float(v.replace('B', '')) * 1e9
    elif 'M' in v:
        return float(v.replace('M', '')) * 1e6
    elif 'K' in v:
        return float(v.replace('K', '')) * 1e3
    else:
        try:
            return float(v)
        except:
            return 0

symbols = []
with open('/home/ubuntu/Projects/skytrade6/datalake/symbols_list.txt', 'r') as f:
    lines = f.readlines()
in_common = False
for line in lines:
    if 'COMMON' in line and 'symbols' in line:
        in_common = True
        continue
    if in_common and line.strip() == '':
        continue
    if 'BYBIT-ONLY' in line:
        break
    if in_common and line.startswith('     '):
        parts = line.split()
        if len(parts) > 1 and 'USDT' in parts[1]:
            symbol = parts[1]
            volume = parts[2] if len(parts) > 2 else '0'
            symbols.append((symbol, volume))

results = []
for symbol, volume in symbols:
    print(f"Testing {symbol}")
    metrics = backtest_strategy(symbol)
    metrics['volume'] = volume
    results.append(metrics)

df_results = pd.DataFrame(results)
df_results.to_csv('/home/ubuntu/Projects/skytrade6/grok-1/results/broad_test_results.csv', index=False)

# Classification
df_results['volume_num'] = df_results['volume'].apply(parse_volume)
df_results['class'] = pd.cut(df_results['volume_num'], bins=[0, 1e7, 1e9, 1e12], labels=['small', 'mid', 'large'])

# Clustering
features = df_results[['win_rate', 'avg_pnl', 'sharpe']].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=0)
df_results['cluster'] = kmeans.fit_predict(features)

df_results.to_csv('/home/ubuntu/Projects/skytrade6/grok-1/results/broad_test_results_classified.csv', index=False)

# Summary
print("Broad test completed. Results saved.")
print(df_results.groupby('class')[['win_rate', 'avg_pnl', 'sharpe']].mean())
print(df_results.groupby('cluster')[['win_rate', 'avg_pnl', 'sharpe']].mean())
