import pandas as pd
import numpy as np
import glob
import os
import warnings
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Tuple, Dict

warnings.filterwarnings('ignore')

def identify_swings(df: pd.DataFrame, deviation_pct: float = 1.0) -> List[Tuple]:
    highs = df['high'].values
    lows = df['low'].values
    
    swings = []
    last_high_idx = 0
    last_low_idx = 0
    last_high = highs[0]
    last_low = lows[0]
    trend = 0
    
    for i in range(1, len(df)):
        if trend == 0:
            if highs[i] > last_high * (1 + deviation_pct / 100):
                trend = 1
                last_high = highs[i]
                last_high_idx = i
                swings.append((last_low_idx, last_low, 'Low'))
            elif lows[i] < last_low * (1 - deviation_pct / 100):
                trend = -1
                last_low = lows[i]
                last_low_idx = i
                swings.append((last_high_idx, last_high, 'High'))
            else:
                if highs[i] > last_high:
                    last_high = highs[i]
                    last_high_idx = i
                if lows[i] < last_low:
                    last_low = lows[i]
                    last_low_idx = i
        elif trend == 1:
            if highs[i] > last_high:
                last_high = highs[i]
                last_high_idx = i
            elif lows[i] < last_high * (1 - deviation_pct / 100):
                swings.append((last_high_idx, last_high, 'High'))
                trend = -1
                last_low = lows[i]
                last_low_idx = i
        elif trend == -1:
            if lows[i] < last_low:
                last_low = lows[i]
                last_low_idx = i
            elif highs[i] > last_low * (1 + deviation_pct / 100):
                swings.append((last_low_idx, last_low, 'Low'))
                trend = 1
                last_high = highs[i]
                last_high_idx = i
                
    if trend == 1:
        swings.append((last_high_idx, last_high, 'High'))
    elif trend == -1:
        swings.append((last_low_idx, last_low, 'Low'))
        
    return swings

def find_elliott_wave_setups(swings: List[Tuple], is_bullish: bool = True) -> List[Dict]:
    setups = []
    if len(swings) < 5:
        return setups
        
    for i in range(len(swings) - 4):
        p0, p1, p2, p3, p4 = swings[i], swings[i+1], swings[i+2], swings[i+3], swings[i+4]
        
        if is_bullish:
            if p0[2] == 'Low' and p1[2] == 'High' and p2[2] == 'Low' and p3[2] == 'High' and p4[2] == 'Low':
                # W2 > W0
                if p2[1] <= p0[1]: continue
                # W3 > W1
                if p3[1] <= p1[1]: continue
                # W4 > W2
                if p4[1] <= p2[1]: continue
                
                strict = (p4[1] > p1[1]) # True Elliott Wave (no overlap)
                
                setups.append({
                    'start_idx': p0[0],
                    'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
                    'is_bullish': True,
                    'strict': strict
                })
        else: # Bearish
            if p0[2] == 'High' and p1[2] == 'Low' and p2[2] == 'High' and p3[2] == 'Low' and p4[2] == 'High':
                # W2 < W0
                if p2[1] >= p0[1]: continue
                # W3 < W1
                if p3[1] >= p1[1]: continue
                # W4 < W2
                if p4[1] >= p2[1]: continue
                
                strict = (p4[1] < p1[1]) # True Elliott Wave (no overlap)
                
                setups.append({
                    'start_idx': p0[0],
                    'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
                    'is_bullish': False,
                    'strict': strict
                })
                
    return setups

def test_wave_5_outcomes(df: pd.DataFrame, setups: List[Dict], lookforward_bars: int = 100) -> pd.DataFrame:
    results = []
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    for s in setups:
        idx_p4 = s['p4'][0]
        end_idx = min(len(df), idx_p4 + lookforward_bars)
        
        if end_idx <= idx_p4 + 1:
            continue
            
        future_highs = highs[idx_p4+1:end_idx]
        future_lows = lows[idx_p4+1:end_idx]
        
        if s['is_bullish']:
            w3_high = s['p3'][1]
            max_future = np.max(future_highs)
            
            success = max_future > w3_high
            
            w4_low = s['p4'][1]
            hit_target_idx = np.where(future_highs > w3_high)[0]
            hit_stop_idx = np.where(future_lows < w4_low)[0]
            
            first_target = hit_target_idx[0] if len(hit_target_idx) > 0 else np.inf
            first_stop = hit_stop_idx[0] if len(hit_stop_idx) > 0 else np.inf
            
            first_to_hit = 'Target' if first_target < first_stop else ('Stop' if first_stop < first_target else 'Neither')
            
            results.append({
                'is_bullish': True,
                'strict': s['strict'],
                'success_exceed_w3': success,
                'first_to_hit': first_to_hit
            })
        else:
            w3_low = s['p3'][1]
            min_future = np.min(future_lows)
            
            success = min_future < w3_low
            
            w4_high = s['p4'][1]
            hit_target_idx = np.where(future_lows < w3_low)[0]
            hit_stop_idx = np.where(future_highs > w4_high)[0]
            
            first_target = hit_target_idx[0] if len(hit_target_idx) > 0 else np.inf
            first_stop = hit_stop_idx[0] if len(hit_stop_idx) > 0 else np.inf
            
            first_to_hit = 'Target' if first_target < first_stop else ('Stop' if first_stop < first_target else 'Neither')
            
            results.append({
                'is_bullish': False,
                'strict': s['strict'],
                'success_exceed_w3': success,
                'first_to_hit': first_to_hit
            })
            
    return pd.DataFrame(results)

def load_data(symbol: str, timeframe: str = '15min', max_days: int = 365) -> pd.DataFrame:
    path = f"/home/ubuntu/Projects/skytrade6/datalake/bybit/{symbol}/*_kline_1m.csv"
    files = sorted(glob.glob(path))[-max_days:]
    if not files: return None
    
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df['startTime'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('startTime').set_index('startTime')
    
    if len(df) == 0: return None
    
    if timeframe not in ['1m', '1min']:
        df = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
    return df

def analyze_symbol(args):
    symbol, timeframe, dev_pct, lookfwd = args
    df = load_data(symbol, timeframe, max_days=365)
    if df is None: return None
    
    swings = identify_swings(df, dev_pct)
    bull_setups = find_elliott_wave_setups(swings, True)
    bear_setups = find_elliott_wave_setups(swings, False)
    
    all_setups = bull_setups + bear_setups
    if len(all_setups) == 0:
        return {'symbol': symbol, 'res_df': pd.DataFrame()}
        
    res_df = test_wave_5_outcomes(df, all_setups, lookfwd)
    return {'symbol': symbol, 'res_df': res_df}

if __name__ == "__main__":
    symbols = [d.split('/')[-1] for d in glob.glob('/home/ubuntu/Projects/skytrade6/datalake/bybit/*') if os.path.isdir(d)]
    print(f"Found {len(symbols)} symbols. Running analysis...")
    
    configs = [
        ('15min', 1.0, 100),
        ('15min', 2.0, 100),
        ('1h', 2.0, 50),
        ('1h', 5.0, 50),
        ('4h', 5.0, 30),
    ]
    
    summary_results = []
    
    for tf, dev, lf in configs:
        print(f"\n--- Config: TF={tf}, Swing={dev}%, LookFwd={lf} ---")
        
        args = [(sym, tf, dev, lf) for sym in symbols] # All symbols
        
        with Pool(os.cpu_count() or 4) as p:
            results = list(tqdm(p.imap(analyze_symbol, args), total=len(args)))
            
        all_dfs = [r['res_df'] for r in results if r is not None and not r['res_df'].empty]
        if not all_dfs:
            print("No valid setups found.")
            continue
            
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        total = len(final_df)
        loose_df = final_df[~final_df['strict']]
        strict_df = final_df[final_df['strict']]
        
        def print_stats(df, name):
            if len(df) == 0:
                print(f"  [{name}] No setups")
                return 0, 0, 0, 0
            t_rate = (df['first_to_hit'] == 'Target').mean()
            s_rate = (df['first_to_hit'] == 'Stop').mean()
            n_rate = (df['first_to_hit'] == 'Neither').mean()
            print(f"  [{name}] Count: {len(df)} | Target (Wave 5): {t_rate:.2%} | Stop (Wave 4 low): {s_rate:.2%} | Neither: {n_rate:.2%}")
            return len(df), t_rate, s_rate, n_rate
            
        n_l, t_l, s_l, nl_l = print_stats(loose_df, "Loose (W4 overlaps W1)")
        n_s, t_s, s_s, nl_s = print_stats(strict_df, "Strict (W4 no overlap)")
        
        summary_results.append({
            'Timeframe': tf,
            'Swing_Pct': dev,
            'Setups_Loose': n_l,
            'Loose_Hit_Target': t_l,
            'Loose_Hit_Stop': s_l,
            'Setups_Strict': n_s,
            'Strict_Hit_Target': t_s,
            'Strict_Hit_Stop': s_s,
        })
        
    pd.DataFrame(summary_results).to_csv('/home/ubuntu/Projects/skytrade6/gemini-5/elliott_summary.csv', index=False)
    print("\nSummary saved to gemini-5/elliott_summary.csv")

