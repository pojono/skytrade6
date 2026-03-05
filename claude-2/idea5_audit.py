#!/usr/bin/env python3
"""
AUDIT: Are Idea 4 (lead-lag) and Idea 5 (spot-futures) the same signal?
Check signal overlap and independent alpha.
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import get_symbols, load_kline, progress_bar, RT_TAKER_BPS

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'
START = '2025-06-01'
END = '2026-03-04'


def main():
    print("=" * 80)
    print("AUDIT: Signal Overlap — Idea 4 (lead-lag) vs Idea 5 (spot-futures)")
    print("=" * 80)
    
    # Load BTC
    btc = load_kline('BTCUSDT', START, END)
    btc = btc[['ts', 'close']].set_index('ts').sort_index()
    btc = btc[~btc.index.duplicated(keep='first')]
    btc.columns = ['btc_close']
    btc['btc_ret_3m'] = (btc['btc_close'] / btc['btc_close'].shift(3) - 1) * 10000
    
    # Test on a sample of coins with spot data
    test_syms = ['SOLUSDT', 'ETHUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
                 'LINKUSDT', 'ADAUSDT', 'SUIUSDT', 'APTUSDT', 'ARBUSDT',
                 'NEARUSDT', 'ATOMUSDT', 'DOTUSDT', 'HYPEUSDT', 'WIFUSDT']
    
    all_results = []
    t0 = time.time()
    
    for i, sym in enumerate(test_syms):
        progress_bar(i, len(test_syms), prefix='Checking', start_time=t0)
        
        kf = load_kline(sym, START, END, spot=False)
        ks = load_kline(sym, START, END, spot=True)
        
        if kf.empty or ks.empty or len(kf) < 5000 or len(ks) < 5000:
            continue
        
        # Futures
        kf = kf[['ts', 'close']].set_index('ts').sort_index()
        kf = kf[~kf.index.duplicated(keep='first')]
        kf.columns = ['fut_close']
        
        # Spot
        ks = ks[['ts', 'close']].set_index('ts').sort_index()
        ks = ks[~ks.index.duplicated(keep='first')]
        ks.columns = ['spot_close']
        
        # Merge all
        merged = kf.join(ks, how='inner').join(btc, how='inner').dropna()
        if len(merged) < 5000:
            continue
        
        # Signal 4: BTC pumped > 150 bps in 3m
        sig4 = merged['btc_ret_3m'] > 150
        
        # Signal 5: spot leads futures by > 40 bps in 3m
        merged['spot_ret_3m'] = (merged['spot_close'] / merged['spot_close'].shift(3) - 1) * 10000
        merged['fut_ret_3m'] = (merged['fut_close'] / merged['fut_close'].shift(3) - 1) * 10000
        merged['spot_lead'] = merged['spot_ret_3m'] - merged['fut_ret_3m']
        sig5 = merged['spot_lead'] > 40
        
        # Forward return
        merged['fwd_30m'] = (merged['fut_close'].shift(-30) / merged['fut_close'] - 1) * 10000
        
        # Overlap analysis
        both = sig4 & sig5
        only4 = sig4 & ~sig5
        only5 = sig5 & ~sig4
        neither = ~sig4 & ~sig5
        
        n4 = sig4.sum()
        n5 = sig5.sum()
        n_both = both.sum()
        n_only4 = only4.sum()
        n_only5 = only5.sum()
        
        overlap_pct = n_both / max(min(n4, n5), 1) * 100
        
        for label, mask in [('sig4_only', only4), ('sig5_only', only5), 
                            ('both', both), ('sig4_all', sig4), ('sig5_all', sig5),
                            ('baseline', neither)]:
            if mask.sum() < 5:
                continue
            rets = merged.loc[mask, 'fwd_30m'].dropna()
            if len(rets) < 3:
                continue
            all_results.append({
                'symbol': sym,
                'signal_set': label,
                'n_signals': len(rets),
                'mean_ret': rets.mean(),
                'median_ret': rets.median(),
                'win_rate': (rets > 0).mean() * 100,
                'net_fees': rets.mean() - RT_TAKER_BPS,
                'overlap_pct': overlap_pct,
            })
    
    progress_bar(len(test_syms), len(test_syms), prefix='Checking', start_time=t0)
    
    if not all_results:
        print("\n❌ No results!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Aggregate across symbols
    print("\n" + "=" * 80)
    print("SIGNAL OVERLAP ANALYSIS (30m forward return, long)")
    print("=" * 80)
    
    agg = df.groupby('signal_set').agg(
        n_coins=('symbol', 'nunique'),
        total_sigs=('n_signals', 'sum'),
        mean_ret=('mean_ret', 'mean'),
        mean_wr=('win_rate', 'mean'),
        net_fees=('net_fees', 'mean'),
    ).reset_index()
    
    print(f"\n{'Signal Set':>15s}  {'Coins':>5s}  {'Signals':>8s}  {'Mean Ret':>10s}  {'WR':>6s}  {'Net Fees':>10s}")
    for _, r in agg.iterrows():
        print(f"{r['signal_set']:>15s}  {r['n_coins']:>5.0f}  {r['total_sigs']:>8.0f}  "
              f"{r['mean_ret']:>+10.1f}  {r['mean_wr']:>5.1f}%  {r['net_fees']:>+10.1f}")
    
    # Overlap percentage
    both_rows = df[df['signal_set'] == 'both']
    if len(both_rows) > 0:
        print(f"\nAvg overlap: {both_rows['overlap_pct'].mean():.1f}% of signals fire simultaneously")
    
    # Key question: does each signal add alpha beyond the other?
    print("\n\nVERDICT:")
    sig4_only = agg[agg['signal_set'] == 'sig4_only']
    sig5_only = agg[agg['signal_set'] == 'sig5_only']
    both_agg = agg[agg['signal_set'] == 'both']
    baseline = agg[agg['signal_set'] == 'baseline']
    
    if len(sig4_only) > 0 and len(sig5_only) > 0:
        s4_net = sig4_only['net_fees'].values[0]
        s5_net = sig5_only['net_fees'].values[0]
        both_net = both_agg['net_fees'].values[0] if len(both_agg) > 0 else 0
        base_net = baseline['net_fees'].values[0] if len(baseline) > 0 else 0
        
        print(f"  Baseline (no signal): {base_net:+.1f} bps")
        print(f"  Idea 4 alone (BTC pump, no spot-lead): {s4_net:+.1f} bps")
        print(f"  Idea 5 alone (spot-lead, no BTC pump): {s5_net:+.1f} bps")
        print(f"  Both signals together: {both_net:+.1f} bps")
        
        if s4_net > RT_TAKER_BPS and s5_net > RT_TAKER_BPS:
            print("\n  ✅ INDEPENDENT: Both signals profitable on their own → COMBINE for higher confidence")
        elif s4_net > RT_TAKER_BPS or s5_net > RT_TAKER_BPS:
            print(f"\n  ⚠️ ONE DOMINATES: Only {'Idea4' if s4_net > s5_net else 'Idea5'} has independent alpha")
        else:
            print("\n  ❌ NEITHER has independent alpha at 30m")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
