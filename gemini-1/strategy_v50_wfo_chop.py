import os
import glob
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import itertools
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

DATALAKE_DIR = "/home/ubuntu/Projects/skytrade6/datalake/bybit"
MAKER_FEE = 0.0004
TAKER_FEE = 0.0010

def load_symbol_data(symbol):
    files = glob.glob(f"{DATALAKE_DIR}/{symbol}/*_kline_1m.csv")
    if not files: return None
    df_list = []
    for f in files:
        try: df_list.append(pd.read_csv(f))
        except: pass
    if not df_list: return None
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
    df = df[df['high'] >= df['low']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ret'] = df['close'].pct_change()
    df = df[(df['ret'] > -0.15) & (df['ret'] < 0.15)]
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df.dropna(inplace=True)
    if df.empty or len(df) < (60 * 24 * 7): return None
    return symbol, df

def get_btc_regime():
    files = glob.glob(f"{DATALAKE_DIR}/BTCUSDT/*_kline_1m.csv")
    if not files: return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df.get('startTime', df.get('timestamp', None)), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df_4h.dropna(inplace=True)
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['btc_ker'] = change / volatility
    return df_4h[['btc_ker']]

def calc_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calc_choppiness(df, period=14):
    atr = calc_atr(df, 1)
    atr_sum = atr.rolling(period).sum()
    max_high = df['high'].rolling(period).max()
    min_low = df['low'].rolling(period).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(period)
    return chop

# Generate all parameter combinations including Dynamic Chop
CONFIGS = []
# Total combinations: 3 * 3 * 2 * 2 * 2 * 3 = 216
for idx, (tp, sl, ker, donch, vol, chop_limit) in enumerate(itertools.product(
    [0.15, 0.20, 0.25], 
    [0.08, 0.10, 0.12], 
    [0.12, 0.15], 
    [15, 20], 
    [1.5, 2.0],
    [50, 61.8, 100])): # 100 effectively means 'No Chop Filter'
    CONFIGS.append({
        'id': idx, 'tp': tp, 'sl': sl, 'ker': ker, 'donch': donch, 'vol': vol, 'chop_limit': chop_limit
    })

def process_symbol(args):
    symbol, btc_regime = args
    data = load_symbol_data(symbol)
    if data is None: return []
    _, df = data
    
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    df_4h.dropna(inplace=True)
    
    if btc_regime is not None:
        df_4h = df_4h.join(btc_regime, how='left').ffill().infer_objects(copy=False)
    else:
        df_4h['btc_ker'] = 1.0
        
    period = 21
    change = abs(df_4h['close'] - df_4h['close'].shift(period))
    volatility = abs(df_4h['close'] - df_4h['close'].shift(1)).rolling(period).sum()
    df_4h['local_ker'] = change / volatility
    
    df_4h['chop'] = calc_choppiness(df_4h, 14)
    
    df_4h['ema_200'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    df_4h['macro_bull'] = df_4h['close'] > df_4h['ema_200']
    df_4h['vol_ma'] = df_4h['volume'].rolling(20).mean().shift(1)
    
    opens = df_4h['open'].values
    highs = df_4h['high'].values
    lows = df_4h['low'].values
    closes = df_4h['close'].values
    macro_bull = df_4h['macro_bull'].values
    volumes = df_4h['volume'].values
    vol_mas = df_4h['vol_ma'].values
    local_kers = df_4h['local_ker'].fillna(0).values
    btc_kers = df_4h['btc_ker'].fillna(0).values
    chops = df_4h['chop'].fillna(100).values
    
    donchian_highs = {}
    donchian_lows = {}
    for d in [15, 20]:
        donchian_highs[d] = df_4h['high'].rolling(d).max().shift(1).values
        donchian_lows[d] = df_4h['low'].rolling(d).min().shift(1).values

    TIME_STOP_CANDLES = 6 * 14
    n = len(df_4h)
    
    all_trades = []
    
    for cfg in CONFIGS:
        high_d = donchian_highs[cfg['donch']]
        low_d = donchian_lows[cfg['donch']]
        
        regime_ok = (btc_kers >= cfg['ker']) | (local_kers >= cfg['ker'])
        chop_ok = chops <= cfg['chop_limit']
        vol_spike = volumes > (vol_mas * cfg['vol'])
        
        long_sigs = (closes > high_d) & macro_bull & vol_spike & regime_ok & chop_ok
        short_sigs = (closes < low_d) & (~macro_bull) & vol_spike & regime_ok & chop_ok
        
        in_position = False
        entry_price = 0.0
        entry_idx = 0
        cooldown_until = 0
        pos_type = 0
        
        for i in range(200, n - 1):
            if not in_position:
                if i > cooldown_until:
                    if long_sigs[i]:
                        in_position = True
                        pos_type = 1
                        entry_price = opens[i+1]
                        entry_idx = i + 1
                        entry_time = df_4h.index[i+1]
                    elif short_sigs[i]:
                        in_position = True
                        pos_type = -1
                        entry_price = opens[i+1]
                        entry_idx = i + 1
                        entry_time = df_4h.index[i+1]
            else:
                if pos_type == 1:
                    target_price = entry_price * (1 + cfg['tp'])
                    stop_price = entry_price * (1 - cfg['sl'])
                    
                    if lows[i] <= stop_price:
                        net_pnl = ((stop_price - entry_price) / entry_price) - (TAKER_FEE * 2)
                        all_trades.append({'symbol': symbol, 'config_id': cfg['id'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'sl': cfg['sl']})
                        in_position = False
                        cooldown_until = i + 6
                    elif highs[i] >= target_price:
                        net_pnl = ((target_price - entry_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        all_trades.append({'symbol': symbol, 'config_id': cfg['id'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'sl': cfg['sl']})
                        in_position = False
                        cooldown_until = i + 6
                    elif (i - entry_idx) >= TIME_STOP_CANDLES:
                        net_pnl = ((closes[i] - entry_price) / entry_price) - (TAKER_FEE * 2)
                        all_trades.append({'symbol': symbol, 'config_id': cfg['id'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'sl': cfg['sl']})
                        in_position = False
                        cooldown_until = i + 6
                        
                elif pos_type == -1:
                    target_price = entry_price * (1 - cfg['tp'])
                    stop_price = entry_price * (1 + cfg['sl'])
                    
                    if highs[i] >= stop_price:
                        net_pnl = ((entry_price - stop_price) / entry_price) - (TAKER_FEE * 2)
                        all_trades.append({'symbol': symbol, 'config_id': cfg['id'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'sl': cfg['sl']})
                        in_position = False
                        cooldown_until = i + 6
                    elif lows[i] <= target_price:
                        net_pnl = ((entry_price - target_price) / entry_price) - TAKER_FEE - MAKER_FEE
                        all_trades.append({'symbol': symbol, 'config_id': cfg['id'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'sl': cfg['sl']})
                        in_position = False
                        cooldown_until = i + 6
                    elif (i - entry_idx) >= TIME_STOP_CANDLES:
                        net_pnl = ((entry_price - closes[i]) / entry_price) - (TAKER_FEE * 2)
                        all_trades.append({'symbol': symbol, 'config_id': cfg['id'], 'entry_time': entry_time, 'exit_time': df_4h.index[i], 'raw_pnl': net_pnl, 'sl': cfg['sl']})
                        in_position = False
                        cooldown_until = i + 6

    return all_trades

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT', 'NEARUSDT']
    print(f"Generating WFO Matrix for {len(CONFIGS)} dynamic Chop Filter configs...")
    
    btc_regime = get_btc_regime()
    args_list = [(s, btc_regime) for s in symbols]
    
    all_trades = []
    pool = multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count()))
    
    for trades in tqdm(pool.imap_unordered(process_symbol, args_list), total=len(symbols), desc="Evaluating configs"):
        all_trades.extend(trades)
            
    pool.close()
    pool.join()
                
    if not all_trades:
        print("No trades generated.")
        return
        
    df_trades = pd.DataFrame(all_trades)
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    
    # Precompute equity PnL assuming 2% risk
    df_trades['equity_pnl'] = df_trades['raw_pnl'] * (0.02 / df_trades['sl'])
    
    print("\n--- Running Walk-Forward Optimization ---")
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2026-03-01')
    
    train_months = 12
    test_months = 3
    
    current_train_start = start_date
    oos_trades = []
    
    while True:
        train_end = current_train_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)
        
        if train_end >= end_date:
            break
            
        mask_train = (df_trades['exit_time'] >= current_train_start) & (df_trades['exit_time'] < train_end)
        df_train = df_trades[mask_train]
        
        if len(df_train) == 0:
            best_config_id = 0
        else:
            # We want to maximize total return while limiting drawdown. 
            # So we use a simple Sortino-like ratio: sum(return) / abs(sum(negative_returns))
            # Just straightforward total return is best for trend following to capture tail risk
            fitness = df_train.groupby('config_id')['equity_pnl'].sum()
            trade_counts = df_train.groupby('config_id').size()
            valid_configs = fitness[trade_counts >= 5]
            
            if len(valid_configs) > 0:
                best_config_id = valid_configs.idxmax()
            else:
                best_config_id = fitness.idxmax() if len(fitness) > 0 else 0
                
        mask_test = (df_trades['config_id'] == best_config_id) & \
                    (df_trades['entry_time'] >= train_end) & \
                    (df_trades['entry_time'] < test_end)
        
        test_period_trades = df_trades[mask_test].copy()
        oos_trades.append(test_period_trades)
        
        current_train_start += relativedelta(months=test_months)
        
    final_oos_df = pd.concat(oos_trades).sort_values('exit_time').reset_index(drop=True)
    
    INITIAL_CAPITAL = 1000.0
    RISK_PER_TRADE = 0.02
    
    capital = INITIAL_CAPITAL
    equity_curve = []
    
    final_oos_df['equity'] = INITIAL_CAPITAL
    
    for idx, row in final_oos_df.iterrows():
        sl_pct = row['sl']
        pos_size = RISK_PER_TRADE / sl_pct
        pos_usd = capital * pos_size
        capital += pos_usd * row['raw_pnl']
        final_oos_df.at[idx, 'equity'] = capital
        equity_curve.append(capital)
        
    final_balance = final_oos_df.iloc[-1]['equity'] if not final_oos_df.empty else INITIAL_CAPITAL
    total_return = (final_balance - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    wins = sum(final_oos_df['raw_pnl'] > 0)
    total_trades = len(final_oos_df)
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    peak = final_oos_df['equity'].expanding(min_periods=1).max()
    drawdown = (final_oos_df['equity'] - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0
    
    print("\n" + "="*85)
    print(f"--- DYNAMIC CHOP FILTER WFO OOS RESULTS (Jan 2022 - Mar 2026) ---")
    print("="*85)
    print(f"STARTING BALANCE: ${INITIAL_CAPITAL:.2f}")
    print(f"FINAL BALANCE:    ${final_balance:.2f}")
    print(f"TOTAL RETURN:     {total_return:.2%}")
    print(f"MAX DRAWDOWN:     {max_dd:.2%} (Improves DD without hurting Profit?)")
    print(f"TOTAL TRADES:     {total_trades}")
    print(f"WIN RATE:         {win_rate:.2%}")
    print(f"AVG EV/TRADE:     {final_oos_df['equity_pnl'].mean():.2%}")
    print("="*85)

if __name__ == "__main__":
    main()
