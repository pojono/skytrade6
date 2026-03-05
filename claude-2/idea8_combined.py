#!/usr/bin/env python3
"""
IDEA 8: Combined Strategy — Stack Best Signals
================================================
Combine the top edges found in Ideas 1-7:
 - Idea 4: BTC pump → long alts (excess +1270 bps, 70% WR) — UP ONLY
 - Idea 5: Spot leads futures > 40 bps → long futures (+739 bps, 79% cross-sym)
 - Idea 1: L/S crowding (extreme short) + OI spike → long 4h (+370 bps)
 - Idea 6: Coiled spring (low vol + rising OI) → filter for bigger moves
 - Idea 3: High implied FR (>20 bps) → momentum confirmation

Test signal stacking:
 A) Each signal alone (baseline)
 B) Any 2 signals together (higher conviction)
 C) Signal + coiled spring filter (volatility gating)
 D) Walk-forward: train 2025-06→2025-12, test 2026-01→2026-03
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import (
    get_symbols, load_kline, load_oi, load_ls_ratio, load_premium,
    progress_bar, RT_TAKER_BPS
)

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'
START = '2025-06-01'
END = '2026-03-04'
OOS_SPLIT = '2026-01-01'


def build_signals(symbol, btc_data, start, end):
    """Build all signal columns for one symbol, aligned to 5m bars."""
    kf = load_kline(symbol, start, end, spot=False)
    ks = load_kline(symbol, start, end, spot=True)
    oi = load_oi(symbol, start, end)
    ls = load_ls_ratio(symbol, start, end)
    prem = load_premium(symbol, start, end)
    
    if kf.empty or len(kf) < 10000:
        return None
    
    # 1m aligned data
    kf = kf[['ts', 'close']].set_index('ts').sort_index()
    kf = kf[~kf.index.duplicated(keep='first')]
    kf.columns = ['close']
    
    # Merge BTC
    df = kf.join(btc_data, how='inner')
    if len(df) < 10000:
        return None
    
    # Forward returns
    for h in [15, 30, 60, 240]:
        df[f'fwd_{h}m'] = (df['close'].shift(-h) / df['close'] - 1) * 10000
    
    # === Signal 4: BTC pump ===
    df['sig4_btc_up'] = df['btc_ret_3m'] > 150
    
    # === Signal 5: Spot leads ===
    df['sig5_spot_lead'] = False
    if not ks.empty and len(ks) > 5000:
        ks_s = ks[['ts', 'close']].set_index('ts').sort_index()
        ks_s = ks_s[~ks_s.index.duplicated(keep='first')]
        ks_s.columns = ['spot_close']
        df = df.join(ks_s, how='left')
        if 'spot_close' in df.columns:
            df['spot_ret_3m'] = (df['spot_close'] / df['spot_close'].shift(3) - 1) * 10000
            df['fut_ret_3m'] = (df['close'] / df['close'].shift(3) - 1) * 10000
            df['spot_lead_bps'] = df['spot_ret_3m'] - df['fut_ret_3m']
            df['sig5_spot_lead'] = df['spot_lead_bps'] > 40
    
    # === Signal 1: L/S crowding ===
    df['sig1_ls_crowd'] = False
    if not ls.empty and len(ls) > 100:
        ls_s = ls[['ts', 'buyRatio']].set_index('ts').sort_index()
        ls_s = ls_s[~ls_s.index.duplicated(keep='first')]
        df = df.join(ls_s, how='left')
        df['buyRatio'] = df['buyRatio'].ffill()
        if 'buyRatio' in df.columns:
            # Extreme short positioning → long signal
            df['sig1_ls_crowd'] = df['buyRatio'] < 0.32
    
    # === Signal 6: Coiled spring (low vol + rising OI) ===
    df['sig6_coiled'] = False
    if not oi.empty and len(oi) > 100:
        oi_s = oi[['ts', 'openInterest']].set_index('ts').sort_index()
        oi_s = oi_s[~oi_s.index.duplicated(keep='first')]
        df = df.join(oi_s, how='left')
        df['openInterest'] = df['openInterest'].ffill()
        if 'openInterest' in df.columns:
            df['oi_pct_4h'] = df['openInterest'].pct_change(240) * 100
            # Realized vol percentile
            df['rvol'] = df['close'].pct_change().rolling(240).std()
            df['rvol_pctile'] = df['rvol'].rolling(1440).rank(pct=True)
            df['sig6_coiled'] = (df['rvol_pctile'] < 0.20) & (df['oi_pct_4h'] > 2.0)
    
    # === Signal 3: High implied FR ===
    df['sig3_high_fr'] = False
    if not prem.empty and len(prem) > 1000:
        prem_s = prem[['ts', 'close']].set_index('ts').sort_index()
        prem_s = prem_s[~prem_s.index.duplicated(keep='first')]
        prem_s.columns = ['premium']
        df = df.join(prem_s, how='left')
        df['premium'] = df['premium'].ffill()
        if 'premium' in df.columns:
            df['implied_fr_bps'] = df['premium'] * 10000
            df['sig3_high_fr'] = df['implied_fr_bps'] > 20
    
    df = df.dropna(subset=['close', 'fwd_30m'])
    return df


def evaluate_signals(df, period_label):
    """Evaluate all signal combinations."""
    results = []
    
    signal_cols = {
        'sig4_btc_up': 'BTC_pump',
        'sig5_spot_lead': 'Spot_leads',
        'sig1_ls_crowd': 'LS_crowd',
        'sig6_coiled': 'Coiled_spring',
        'sig3_high_fr': 'High_IFR',
    }
    
    for h in [30, 60, 240]:
        fwd_col = f'fwd_{h}m'
        if fwd_col not in df.columns:
            continue
        
        # Single signals
        for sig_col, sig_name in signal_cols.items():
            if sig_col not in df.columns:
                continue
            mask = df[sig_col] == True
            if mask.sum() < 10:
                continue
            rets = df.loc[mask, fwd_col].dropna()
            if len(rets) < 5:
                continue
            results.append({
                'period': period_label,
                'signal': sig_name,
                'horizon_m': h,
                'n_signals': len(rets),
                'mean_ret': rets.mean(),
                'win_rate': (rets > 0).mean() * 100,
                'net_fees': rets.mean() - RT_TAKER_BPS,
                'sharpe': rets.mean() / max(rets.std(), 0.01),
            })
        
        # Pairwise combinations (ANY two)
        sig_keys = list(signal_cols.keys())
        for i in range(len(sig_keys)):
            for j in range(i + 1, len(sig_keys)):
                s1, s2 = sig_keys[i], sig_keys[j]
                if s1 not in df.columns or s2 not in df.columns:
                    continue
                mask = (df[s1] == True) & (df[s2] == True)
                if mask.sum() < 5:
                    continue
                rets = df.loc[mask, fwd_col].dropna()
                if len(rets) < 3:
                    continue
                n1 = signal_cols[s1]
                n2 = signal_cols[s2]
                results.append({
                    'period': period_label,
                    'signal': f'{n1}+{n2}',
                    'horizon_m': h,
                    'n_signals': len(rets),
                    'mean_ret': rets.mean(),
                    'win_rate': (rets > 0).mean() * 100,
                    'net_fees': rets.mean() - RT_TAKER_BPS,
                    'sharpe': rets.mean() / max(rets.std(), 0.01),
                })
        
        # Triple: BTC pump + Spot leads + Coiled
        for extra in ['sig1_ls_crowd', 'sig6_coiled', 'sig3_high_fr']:
            if extra not in df.columns:
                continue
            mask = (df['sig4_btc_up'] == True) & (df['sig5_spot_lead'] == True) & (df[extra] == True)
            if mask.sum() >= 3:
                rets = df.loc[mask, fwd_col].dropna()
                if len(rets) >= 2:
                    results.append({
                        'period': period_label,
                        'signal': f'BTC+Spot+{signal_cols[extra]}',
                        'horizon_m': h,
                        'n_signals': len(rets),
                        'mean_ret': rets.mean(),
                        'win_rate': (rets > 0).mean() * 100,
                        'net_fees': rets.mean() - RT_TAKER_BPS,
                        'sharpe': rets.mean() / max(rets.std(), 0.01),
                    })
    
    return results


def main():
    print("=" * 80)
    print("IDEA 8: COMBINED STRATEGY — SIGNAL STACKING")
    print(f"Full period: {START} → {END}")
    print(f"In-sample: {START} → {OOS_SPLIT}")
    print(f"Out-of-sample: {OOS_SPLIT} → {END}")
    print("=" * 80)
    
    # Load BTC
    print("\nLoading BTC reference data...")
    btc = load_kline('BTCUSDT', START, END)
    btc = btc[['ts', 'close']].set_index('ts').sort_index()
    btc = btc[~btc.index.duplicated(keep='first')]
    btc.columns = ['btc_close']
    btc['btc_ret_3m'] = (btc['btc_close'] / btc['btc_close'].shift(3) - 1) * 10000
    
    symbols = get_symbols(min_days=60)
    # Focus on mid-cap + large-cap with spot data for signal 5
    print(f"Testing {len(symbols)} symbols")
    
    all_results_full = []
    all_results_is = []
    all_results_oos = []
    t0 = time.time()
    
    for i, sym in enumerate(symbols):
        if sym == 'BTCUSDT':
            continue
        progress_bar(i, len(symbols), prefix='Building', start_time=t0)
        
        try:
            df = build_signals(sym, btc, START, END)
            if df is None or len(df) < 5000:
                continue
            
            # Full period
            res_full = evaluate_signals(df, 'full')
            for r in res_full:
                r['symbol'] = sym
            all_results_full.extend(res_full)
            
            # In-sample
            df_is = df[df.index < OOS_SPLIT]
            if len(df_is) > 2000:
                res_is = evaluate_signals(df_is, 'in_sample')
                for r in res_is:
                    r['symbol'] = sym
                all_results_is.extend(res_is)
            
            # Out-of-sample
            df_oos = df[df.index >= OOS_SPLIT]
            if len(df_oos) > 2000:
                res_oos = evaluate_signals(df_oos, 'out_of_sample')
                for r in res_oos:
                    r['symbol'] = sym
                all_results_oos.extend(res_oos)
        except Exception as e:
            print(f"\n  ⚠ {sym}: {e}")
    
    progress_bar(len(symbols), len(symbols), prefix='Building', start_time=t0)
    
    # Combine all results
    all_results = all_results_full + all_results_is + all_results_oos
    if not all_results:
        print("\n❌ No results!")
        return
    
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(f'{OUT}/idea8_combined_raw.csv', index=False)
    
    # ==================== ANALYSIS ====================
    for period in ['full', 'in_sample', 'out_of_sample']:
        sub = df_all[df_all['period'] == period]
        if len(sub) == 0:
            continue
        
        print(f"\n{'='*80}")
        print(f"PERIOD: {period.upper()}")
        print(f"{'='*80}")
        
        agg = sub.groupby(['signal', 'horizon_m']).agg(
            n_symbols=('symbol', 'nunique'),
            total_sigs=('n_signals', 'sum'),
            mean_ret=('mean_ret', 'mean'),
            mean_wr=('win_rate', 'mean'),
            net_fees=('net_fees', 'mean'),
            mean_sharpe=('sharpe', 'mean'),
        ).reset_index().sort_values('net_fees', ascending=False)
        
        profitable = agg[agg['net_fees'] > 0]
        print(f"\n{len(profitable)} configs profitable after fees")
        
        # Show best at 30m and 60m
        for h in [30, 60, 240]:
            print(f"\n  --- {h}m horizon ---")
            h_sub = agg[agg['horizon_m'] == h].head(10)
            for _, r in h_sub.iterrows():
                marker = '✅' if r['net_fees'] > 0 else '  '
                print(f"  {marker} {r['signal']:35s} net={r['net_fees']:+7.1f}bps  "
                      f"WR={r['mean_wr']:5.1f}%  sharpe={r['mean_sharpe']:+.3f}  "
                      f"sigs={r['total_sigs']:>6.0f}  syms={r['n_symbols']:.0f}")
    
    # Walk-forward comparison
    print(f"\n{'='*80}")
    print("WALK-FORWARD COMPARISON: IS vs OOS")
    print(f"{'='*80}")
    
    is_df = df_all[df_all['period'] == 'in_sample']
    oos_df = df_all[df_all['period'] == 'out_of_sample']
    
    if len(is_df) > 0 and len(oos_df) > 0:
        is_agg = is_df.groupby(['signal', 'horizon_m']).agg(
            is_net=('net_fees', 'mean'), is_wr=('win_rate', 'mean')
        ).reset_index()
        oos_agg = oos_df.groupby(['signal', 'horizon_m']).agg(
            oos_net=('net_fees', 'mean'), oos_wr=('win_rate', 'mean')
        ).reset_index()
        
        wf = pd.merge(is_agg, oos_agg, on=['signal', 'horizon_m'], how='inner')
        wf['is_profitable'] = wf['is_net'] > 0
        wf['oos_profitable'] = wf['oos_net'] > 0
        wf['survived'] = wf['is_profitable'] & wf['oos_profitable']
        
        survived = wf[wf['survived']].sort_values('oos_net', ascending=False)
        print(f"\n{len(survived)} configs profitable in BOTH IS and OOS:")
        for _, r in survived.head(15).iterrows():
            print(f"  {r['signal']:35s} {r['horizon_m']:>4.0f}m  "
                  f"IS={r['is_net']:+7.1f}bps WR={r['is_wr']:5.1f}%  →  "
                  f"OOS={r['oos_net']:+7.1f}bps WR={r['oos_wr']:5.1f}%")
        
        total_is_profitable = wf['is_profitable'].sum()
        total_survived = wf['survived'].sum()
        if total_is_profitable > 0:
            survival_rate = total_survived / total_is_profitable * 100
            print(f"\n  Survival rate: {total_survived}/{total_is_profitable} = {survival_rate:.0f}%")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
