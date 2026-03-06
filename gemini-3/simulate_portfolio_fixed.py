import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")
GOLDEN_CLUSTER = ['BTCUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'WLDUSDT']

def load_data(symbol, start_date="2025-01-01"):
    try:
        metrics_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_metrics.csv")))
        metrics_files = [f for f in metrics_files if f.name >= start_date]
        if not metrics_files: return None, None
        
        dfs = []
        for f in metrics_files:
            try: dfs.append(pd.read_csv(f, usecols=['create_time', 'sum_open_interest_value'], engine='c'))
            except: pass
        oi_df = pd.concat(dfs, ignore_index=True)
        oi_df.rename(columns={'create_time': 'timestamp', 'sum_open_interest_value': 'oi_usd'}, inplace=True)
        try: oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp']).astype(np.int64) // 10**6
        except: oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
        if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
        oi_df.set_index('timestamp', inplace=True)
        
        bb_fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
        bb_fr_files = [f for f in bb_fr_files if f.name >= start_date]
        if not bb_fr_files: return None, None
        
        dfs = []
        for f in bb_fr_files:
            try:
                df = pd.read_csv(f, engine='c')
                ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
                val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'funding_rate']
                dfs.append(df)
            except: pass
        fr_df = pd.concat(dfs, ignore_index=True)
        fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
        if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
        fr_df.set_index('timestamp', inplace=True)
        fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
        
        kline_files = sorted(list((DATALAKE / f"binance/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark_price" not in f.name and "index_price" not in f.name and "premium_index" not in f.name and f.name >= start_date]
        if not kline_files: return None, None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['open_time', 'high', 'low', 'close'], engine='c'))
            except: pass
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'open_time': 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        m1_df = kline_df.copy()
        m1_df.index = pd.to_datetime(m1_df.index, unit='ms')
        
        merged = kline_df.join(oi_df, how='left').join(fr_df, how='left')
        merged['oi_usd'] = merged['oi_usd'].ffill()
        merged['funding_rate'] = merged['funding_rate'].ffill()
        merged = merged.dropna(subset=['close', 'oi_usd', 'funding_rate'])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged.index = pd.to_datetime(merged.index, unit='ms')
        
        # Calculate daily ATR correctly
        # Resample to 1d first to get true range
        daily = merged.resample('1D').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        daily['prev_close'] = daily['close'].shift(1)
        tr1 = daily['high'] - daily['low']
        tr2 = (daily['high'] - daily['prev_close']).abs()
        tr3 = (daily['low'] - daily['prev_close']).abs()
        daily['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily['atr_14d'] = daily['tr'].rolling(14).mean()
        daily['atr_pct'] = daily['atr_14d'] / daily['close']
        
        # Safe Resampling for hourly
        hourly = merged.resample('1h', label='right', closed='left').agg({
            'close': 'last',
            'oi_usd': 'last',
            'funding_rate': 'last'
        }).dropna()
        
        # Forward fill the daily ATR percentage to the hourly dataframe
        hourly['date'] = hourly.index.normalize()
        daily['date'] = daily.index.normalize()
        
        hourly = hourly.reset_index().merge(daily[['date', 'atr_pct']], on='date', how='left').set_index('timestamp')
        hourly['atr_pct'] = hourly['atr_pct'].ffill()
        
        return hourly, m1_df
    except Exception as e: 
        return None, None

def get_symbol_trades(symbol):
    hourly, m1_df = load_data(symbol)
    if hourly is None or m1_df is None or len(hourly) < 500: return []
    
    hourly['oi_z'] = (hourly['oi_usd'] - hourly['oi_usd'].rolling(168).mean()) / hourly['oi_usd'].rolling(168).std()
    hourly['fr_z'] = (hourly['funding_rate'] - hourly['funding_rate'].rolling(168).mean()) / hourly['funding_rate'].rolling(168).std()
    
    hourly['signal'] = 0
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] > 2.0), 'signal'] = -1
    hourly.loc[(hourly['oi_z'] > 2.0) & (hourly['fr_z'] < -2.0), 'signal'] = 1
    
    hourly['signal_shifted'] = hourly['signal'].shift(1).fillna(0)
    hourly.loc[hourly['signal'] == hourly['signal_shifted'], 'signal'] = 0
    
    trades = hourly[hourly['signal'] != 0].copy()
    if len(trades) == 0: return []
    
    trade_list = []
    
    for entry_ts, row in trades.iterrows():
        entry_price = row['close']
        signal = row['signal']
        atr_pct = row['atr_pct']
        if pd.isna(atr_pct) or atr_pct == 0: atr_pct = 0.05
        
        start_time = entry_ts
        end_time = start_time + pd.Timedelta(hours=24)
        
        path = m1_df.loc[start_time:end_time]
        if len(path) == 0: continue
            
        exit_price = path.iloc[-1]['close'] 
        exit_ts = path.index[-1]
        
        for ts, m1_row in path.iterrows():
            high = m1_row['high']
            low = m1_row['low']
            
            if signal == 1:
                if ((high - entry_price) / entry_price) * 10000 >= 1000.0:
                    exit_price = entry_price * 1.10
                    exit_ts = ts
                    break
            elif signal == -1:
                if ((entry_price - low) / entry_price) * 10000 >= 1000.0:
                    exit_price = entry_price * 0.90
                    exit_ts = ts
                    break
                    
        gross_ret = (exit_price - entry_price) / entry_price * signal
        net_ret = gross_ret - (10.0 / 10000) 
        
        trade_list.append({
            'symbol': symbol,
            'entry_time': entry_ts,
            'exit_time': exit_ts,
            'signal': signal,
            'atr_pct': atr_pct,
            'net_ret': net_ret
        })
        
    return trade_list

if __name__ == "__main__":
    print("Running accurate ATR sizing simulation...")
    all_trades = []
    with Pool(min(6, os.cpu_count() or 6)) as p:
        for t_list in p.imap_unordered(get_symbol_trades, GOLDEN_CLUSTER):
            all_trades.extend(t_list)
            
    if not all_trades:
        print("No trades found.")
        exit(0)
        
    df_trades = pd.DataFrame(all_trades)
    df_trades = df_trades.sort_values('entry_time').reset_index(drop=True)
    
    STARTING_CAPITAL = 1000.0
    RISK_PER_TRADE = 0.02 # Risk 2% of equity per 1x ATR move (Aggressive compounding)
    
    equity = STARTING_CAPITAL
    equity_curve = [{'time': pd.to_datetime('2025-01-01'), 'equity': equity}]
    
    df_trades['trade_pnl'] = 0.0
    df_trades['equity_after'] = 0.0
    
    for idx, row in df_trades.iterrows():
        # Correct Inverse ATR sizing
        atr = row['atr_pct'] 
        # e.g., if ATR is 0.05 (5%), multiplier = 0.02 / 0.05 = 0.4x leverage
        size_multiplier = RISK_PER_TRADE / atr
        # Cap max position size at 3x leverage to avoid liquidation risks
        size_multiplier = min(size_multiplier, 3.0) 
        
        trade_return_pct = row['net_ret']
        pnl_usd = equity * size_multiplier * trade_return_pct
        
        equity += pnl_usd
        
        df_trades.at[idx, 'trade_pnl'] = pnl_usd
        df_trades.at[idx, 'equity_after'] = equity
        
        equity_curve.append({'time': row['exit_time'], 'equity': equity})
        
    df_trades['month'] = df_trades['entry_time'].dt.to_period('M')
    monthly_stats = df_trades.groupby('month').agg(
        trades=('symbol', 'count'),
        wins=('net_ret', lambda x: (x > 0).sum()),
        pnl_usd=('trade_pnl', 'sum')
    )
    monthly_stats['win_rate'] = (monthly_stats['wins'] / monthly_stats['trades'] * 100).round(1)
    
    print("\n=== Monthly Breakdown (2% Risk / 1x ATR) ===")
    print(monthly_stats[['trades', 'win_rate', 'pnl_usd']])
    
    print(f"\n=== Overall Performance ===")
    print(f"Starting Capital: ${STARTING_CAPITAL:.2f}")
    print(f"Ending Capital:   ${equity:.2f}")
    print(f"Total Net Profit: ${equity - STARTING_CAPITAL:.2f}")
    print(f"Total Return:     {((equity - STARTING_CAPITAL)/STARTING_CAPITAL * 100):.2f}%")
    print(f"Total Trades:     {len(df_trades)}")
    win_rate = (df_trades['net_ret'] > 0).mean() * 100
    print(f"Overall Win Rate: {win_rate:.1f}%")
    
    eq_df = pd.DataFrame(equity_curve)
    eq_df.set_index('time', inplace=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(eq_df.index, eq_df['equity'], drawstyle='steps-post', color='green', linewidth=2)
    plt.title('Golden Cluster - Powder Keg Compounding ($1000 Start, 2% Risk/ATR)')
    plt.ylabel('Portfolio Equity ($)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('powder_keg_equity_fixed.png')
    print("\nEquity curve saved to 'powder_keg_equity_fixed.png'")

