import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import os
import warnings

warnings.filterwarnings('ignore')
DATALAKE = Path("/home/ubuntu/Projects/skytrade6/datalake")

GOLDEN_CLUSTER = ['BTCUSDT', 'SOLUSDT', 'LINKUSDT', 'AVAXUSDT', 'NEARUSDT', 'WLDUSDT']
START_DATE = "2022-01-01"
END_DATE = "2026-03-04"

def extract_macro_state():
    print("Extracting BTC Macro State (2022-2026)...")
    kline_files = sorted(list((DATALAKE / f"bybit/BTCUSDT").glob("*_kline_1m.csv")))
    kline_files = [f for f in kline_files if START_DATE <= f.name[:10] <= END_DATE and "mark" not in f.name and "index" not in f.name and "premium" not in f.name]
    
    dfs = []
    for f in kline_files:
        try: dfs.append(pd.read_csv(f, usecols=['startTime', 'close'], engine='c'))
        except: pass
        
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    if df['timestamp'].max() < 1e11: df['timestamp'] *= 1000
    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    
    hourly = df.resample('1h').agg({'close': 'last'}).dropna()
    
    hourly['sma_200'] = hourly['close'].rolling(200).mean()
    hourly['sma_200_slope'] = (hourly['sma_200'] - hourly['sma_200'].shift(24)) / hourly['sma_200'].shift(24) * 100
    hourly['hv_7d'] = hourly['close'].pct_change().rolling(168).std() * np.sqrt(24 * 365) * 100
    
    return hourly[['close', 'sma_200', 'sma_200_slope', 'hv_7d']]

def load_symbol_data(symbol):
    try:
        oi_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_open_interest*.csv")))
        oi_files = [f for f in oi_files if START_DATE <= f.name[:10] <= END_DATE]
        if not oi_files: return None, None
        
        dfs = []
        for f in oi_files:
            try:
                df = pd.read_csv(f, engine='c')
                ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
                val_col = 'openInterest' if 'openInterest' in df.columns else df.columns[2]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'oi_coin']
                dfs.append(df)
            except: pass
        if not dfs: return None, None
        oi_df = pd.concat(dfs, ignore_index=True)
        oi_df['timestamp'] = pd.to_numeric(oi_df['timestamp'])
        if oi_df['timestamp'].max() < 1e11: oi_df['timestamp'] *= 1000
        oi_df.set_index('timestamp', inplace=True)
        oi_df = oi_df[~oi_df.index.duplicated(keep='last')]
        
        fr_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_funding_rate.csv")))
        fr_files = [f for f in fr_files if START_DATE <= f.name[:10] <= END_DATE]
        if not fr_files: return None, None
        
        dfs = []
        for f in fr_files:
            try:
                df = pd.read_csv(f, engine='c')
                ts_col = 'fundingTime' if 'fundingTime' in df.columns else 'calcTime' if 'calcTime' in df.columns else df.columns[0]
                val_col = 'fundingRate' if 'fundingRate' in df.columns else df.columns[2]
                df = df[[ts_col, val_col]]
                df.columns = ['timestamp', 'funding_rate']
                dfs.append(df)
            except: pass
        if not dfs: return None, None
        fr_df = pd.concat(dfs, ignore_index=True)
        fr_df['timestamp'] = pd.to_numeric(fr_df['timestamp'])
        if fr_df['timestamp'].max() < 1e11: fr_df['timestamp'] *= 1000
        fr_df.set_index('timestamp', inplace=True)
        fr_df = fr_df[~fr_df.index.duplicated(keep='last')]
        
        kline_files = sorted(list((DATALAKE / f"bybit/{symbol}").glob("*_kline_1m.csv")))
        kline_files = [f for f in kline_files if "mark" not in f.name and "index" not in f.name and "premium" not in f.name and START_DATE <= f.name[:10] <= END_DATE]
        if not kline_files: return None, None
        
        dfs = []
        for f in kline_files:
            try: dfs.append(pd.read_csv(f, usecols=['startTime', 'high', 'low', 'close'], engine='c'))
            except: pass
        if not dfs: return None, None
        kline_df = pd.concat(dfs, ignore_index=True)
        kline_df.rename(columns={'startTime': 'timestamp'}, inplace=True)
        kline_df['timestamp'] = pd.to_numeric(kline_df['timestamp'])
        if kline_df['timestamp'].max() < 1e11: kline_df['timestamp'] *= 1000
        kline_df.set_index('timestamp', inplace=True)
        
        m1_df = kline_df.copy()
        m1_df.index = pd.to_datetime(m1_df.index, unit='ms')
        
        merged = kline_df.join(oi_df, how='left').join(fr_df, how='left')
        merged['oi_coin'] = merged['oi_coin'].ffill()
        merged['funding_rate'] = merged['funding_rate'].ffill()
        merged = merged.dropna(subset=['close', 'oi_coin', 'funding_rate'])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged.index = pd.to_datetime(merged.index, unit='ms')
        
        merged['oi_usd'] = merged['oi_coin'] * merged['close']
        
        daily = merged.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'})
        daily['prev_close'] = daily['close'].shift(1)
        tr1 = daily['high'] - daily['low']
        tr2 = (daily['high'] - daily['prev_close']).abs()
        tr3 = (daily['low'] - daily['prev_close']).abs()
        daily['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        daily['atr_14d'] = daily['tr'].rolling(14).mean()
        daily['atr_pct'] = daily['atr_14d'] / daily['close']
        
        hourly = merged.resample('1h', label='right', closed='left').agg({'close': 'last', 'oi_usd': 'last', 'funding_rate': 'last'}).dropna()
        hourly['date'] = hourly.index.normalize()
        daily['date'] = daily.index.normalize()
        
        hourly = hourly.reset_index().merge(daily[['date', 'atr_pct']], on='date', how='left').set_index('timestamp')
        hourly['atr_pct'] = hourly['atr_pct'].ffill()
        
        return hourly, m1_df
    except Exception as e:
        return None, None

def analyze_symbol(symbol):
    print(f"Processing {symbol}...")
    hourly, m1_df = load_symbol_data(symbol)
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
    macro_state = extract_macro_state()
    
    all_trades = []
    with Pool(min(6, os.cpu_count() or 6)) as p:
        for t_list in p.imap_unordered(analyze_symbol, GOLDEN_CLUSTER):
            if t_list: all_trades.extend(t_list)
            
    df_trades = pd.DataFrame(all_trades)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades = df_trades.sort_values('entry_time')
    
    df_trades = pd.merge_asof(df_trades, macro_state, left_on='entry_time', right_index=True, direction='backward')
    
    # APPLY REGIME FILTER
    filtered_trades = df_trades[(df_trades['hv_7d'] >= 40.0) & (df_trades['sma_200_slope'].abs() >= 0.10)].copy()
    
    print(f"\nTotal Raw Trades: {len(df_trades)}")
    print(f"Total Filtered Trades: {len(filtered_trades)} (Win Rate: {(filtered_trades['net_ret'] > 0).mean() * 100:.1f}%)")
    
    filtered_trades.to_csv("full_history_filtered_trades.csv", index=False)
    
    print("\n--- Simulation Grid (Profit & DD) ---")
    risks = [0.01, 0.02, 0.03, 0.05]
    leverages = [1.0, 2.0, 3.0, 5.0, 10.0]
    
    results = []
    
    for r in risks:
        for lev in leverages:
            equity = 1000.0
            peak_equity = 1000.0
            max_dd = 0.0
            
            for idx, row in filtered_trades.iterrows():
                atr = row['atr_pct']
                if atr <= 0: atr = 0.05
                size_multiplier = r / atr
                size_multiplier = min(size_multiplier, lev)
                
                pnl = equity * size_multiplier * row['net_ret']
                equity += pnl
                
                if equity > peak_equity: peak_equity = equity
                dd = (peak_equity - equity) / peak_equity * 100
                if dd > max_dd: max_dd = dd
                
            total_ret = (equity - 1000.0) / 1000.0 * 100
            results.append({'Risk': f"{r*100:.0f}%", 'MaxLev': f"{lev}x", 'Net Profit': f"{total_ret:.1f}%", 'Max DD': f"{max_dd:.1f}%"})
            
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    print("\n--- Monthly Breakdown (2% Risk, 3x Max Lev) ---")
    equity = 1000.0
    filtered_trades['equity'] = 0.0
    
    for idx, row in filtered_trades.iterrows():
        atr = row['atr_pct']
        if atr <= 0: atr = 0.05
        size_multiplier = 0.02 / atr
        size_multiplier = min(size_multiplier, 3.0)
        
        pnl = equity * size_multiplier * row['net_ret']
        equity += pnl
        filtered_trades.loc[idx, 'equity'] = equity
        
    filtered_trades['month'] = filtered_trades['entry_time'].dt.to_period('M')
    monthly = filtered_trades.groupby('month').agg(
        trades=('symbol', 'count'),
        wins=('net_ret', lambda x: (x > 0).sum())
    )
    monthly['win_rate'] = (monthly['wins'] / monthly['trades'] * 100).round(1)
    
    filtered_trades['month_str'] = filtered_trades['entry_time'].dt.strftime('%Y-%m')
    end_equities = filtered_trades.groupby('month_str')['equity'].last()
    
    all_months = pd.period_range(start='2022-01', end='2026-03', freq='M').strftime('%Y-%m')
    full_eq = pd.Series(index=all_months, dtype=float)
    
    current_eq = 1000.0
    
    monthly_stats = []
    for m in all_months:
        if m in end_equities:
            current_eq = end_equities[m]
        full_eq[m] = current_eq
        
        prev_m = (pd.Period(m) - 1).strftime('%Y-%m')
        prev_eq = full_eq[prev_m] if prev_m in full_eq else 1000.0
        
        ret_pct = (current_eq - prev_eq) / prev_eq * 100
        
        tr_count = monthly.loc[m, 'trades'] if m in monthly.index.astype(str) else 0
        wr = monthly.loc[m, 'win_rate'] if m in monthly.index.astype(str) else 0.0
        
        monthly_stats.append({
            'Month': m,
            'Trades': tr_count,
            'Win Rate %': wr,
            'Return %': round(ret_pct, 2),
            'End Equity': round(current_eq, 2)
        })
        
    m_df = pd.DataFrame(monthly_stats)
    print(m_df.to_string(index=False))
