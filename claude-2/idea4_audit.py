#!/usr/bin/env python3
"""
AUDIT: Lead-Lag Signal — is it real or just correlated market moves?

Key question: When BTC pumps 150bps in 3min, alts are ALREADY moving too.
The "forward return" on alts after BTC moves may just be measuring the
continuation of the same correlated market event, not a predictive lag.

Tests:
1. EXCESS return: alt forward return MINUS BTC forward return over same window
   (if alt just tracks BTC, excess return = 0 → no edge)
2. BETA-ADJUSTED: alt return minus beta * BTC return
3. TIMING: does the edge decay with lag? (enter 0,1,2,3,5 min after BTC signal)
4. ALREADY-MOVED filter: only trade alts that HAVEN'T moved yet
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
    print("AUDIT: Lead-Lag Signal — Correlation Bias Check")
    print("=" * 80)
    
    # Load BTC as leader
    btc = load_kline('BTCUSDT', START, END)
    btc = btc[['ts', 'close']].set_index('ts').sort_index()
    btc = btc[~btc.index.duplicated(keep='first')]
    btc.columns = ['btc_close']
    
    # BTC returns
    btc['btc_ret_3m'] = (btc['btc_close'] / btc['btc_close'].shift(3) - 1) * 10000
    # BTC forward returns (same horizons we'll measure for alts)
    for h in [1, 3, 5, 10, 15, 30]:
        btc[f'btc_fwd_{h}m'] = (btc['btc_close'].shift(-h) / btc['btc_close'] - 1) * 10000
    
    symbols = get_symbols(min_days=60)
    followers = [s for s in symbols if s != 'BTCUSDT']
    
    print(f"Testing {len(followers)} followers vs BTC")
    print(f"Signal: BTC 3m return > 150 bps")
    print()
    
    all_results = []
    t0 = time.time()
    
    for i, sym in enumerate(followers):
        progress_bar(i, len(followers), prefix='Auditing', start_time=t0)
        
        fk = load_kline(sym, START, END)
        if fk.empty or len(fk) < 5000:
            continue
        
        fk = fk[['ts', 'close']].set_index('ts').sort_index()
        fk = fk[~fk.index.duplicated(keep='first')]
        fk.columns = ['alt_close']
        
        # Alt's own recent move
        fk['alt_ret_3m'] = (fk['alt_close'] / fk['alt_close'].shift(3) - 1) * 10000
        
        # Alt forward returns
        for h in [1, 3, 5, 10, 15, 30]:
            fk[f'alt_fwd_{h}m'] = (fk['alt_close'].shift(-h) / fk['alt_close'] - 1) * 10000
        
        # Merge
        merged = fk.join(btc, how='inner').dropna()
        if len(merged) < 2000:
            continue
        
        # Compute beta (rolling 1-day correlation * vol ratio)
        ret_1m_alt = merged['alt_close'].pct_change()
        ret_1m_btc = merged['btc_close'].pct_change()
        beta = ret_1m_alt.rolling(1440).corr(ret_1m_btc) * (ret_1m_alt.rolling(1440).std() / ret_1m_btc.rolling(1440).std().clip(lower=1e-8))
        merged['beta'] = beta
        
        # Signal: BTC pumped > 150 bps in 3 min
        btc_up = merged['btc_ret_3m'] > 150
        btc_down = merged['btc_ret_3m'] < -150
        
        for direction, sig, dir_mult in [('up', btc_up, 1), ('down', btc_down, -1)]:
            if sig.sum() < 10:
                continue
            
            for h in [1, 3, 5, 10, 15, 30]:
                subset = merged[sig]
                
                # Raw alt forward return
                alt_ret = dir_mult * subset[f'alt_fwd_{h}m']
                # BTC forward return over same window  
                btc_fwd = dir_mult * subset[f'btc_fwd_{h}m']
                # Excess return (alt - btc)
                excess = alt_ret - btc_fwd
                # Beta-adjusted excess
                beta_adj = alt_ret - subset['beta'] * btc_fwd
                
                # How much had alt already moved when signal fired?
                alt_already = dir_mult * subset['alt_ret_3m']
                
                # Lagged entry: what if we enter 1,2,3 min AFTER signal?
                lagged_rets = {}
                for lag in [1, 2, 3, 5]:
                    lag_col = f'alt_fwd_{h}m'
                    # Shift the index forward by lag minutes to simulate delayed entry
                    # Forward return from t+lag to t+lag+h
                    if h + lag <= 30:
                        lagged = dir_mult * (merged['alt_close'].shift(-(h+lag)) / merged['alt_close'].shift(-lag) - 1) * 10000
                        lagged_rets[f'lag_{lag}m'] = lagged[sig].mean()
                
                all_results.append({
                    'follower': sym,
                    'direction': direction,
                    'horizon_m': h,
                    'n_signals': int(sig.sum()),
                    'raw_alt_ret': alt_ret.mean(),
                    'raw_btc_fwd': btc_fwd.mean(),
                    'excess_ret': excess.mean(),
                    'beta_adj_ret': beta_adj.mean(),
                    'alt_already_moved': alt_already.mean(),
                    'raw_net_fees': alt_ret.mean() - RT_TAKER_BPS,
                    'excess_net_fees': excess.mean() - RT_TAKER_BPS,
                    'beta_adj_net_fees': beta_adj.mean() - RT_TAKER_BPS,
                    'raw_wr': (alt_ret > 0).mean() * 100,
                    'excess_wr': (excess > 0).mean() * 100,
                    **{k: v for k, v in lagged_rets.items()},
                })
    
    progress_bar(len(followers), len(followers), prefix='Auditing', start_time=t0)
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT}/idea4_audit_raw.csv', index=False)
    
    # ==================== ANALYSIS ====================
    print("\n" + "=" * 80)
    print("AUDIT RESULTS")
    print("=" * 80)
    
    # Aggregate across all followers
    for direction in ['up', 'down']:
        print(f"\n--- BTC 3m > 150bps {direction.upper()} ---")
        sub = df[df['direction'] == direction]
        if len(sub) == 0:
            continue
        
        agg = sub.groupby('horizon_m').agg(
            n_coins=('follower', 'nunique'),
            n_signals=('n_signals', 'mean'),
            raw_ret=('raw_alt_ret', 'mean'),
            btc_fwd=('raw_btc_fwd', 'mean'),
            excess=('excess_ret', 'mean'),
            beta_adj=('beta_adj_ret', 'mean'),
            alt_already=('alt_already_moved', 'mean'),
            raw_wr=('raw_wr', 'mean'),
            excess_wr=('excess_wr', 'mean'),
        ).reset_index()
        
        print(f"  {'H':>4s}  {'Raw':>8s}  {'BTC_fwd':>8s}  {'Excess':>8s}  {'BetaAdj':>8s}  {'AltAlready':>10s}  {'RawWR':>6s}  {'ExWR':>6s}  {'Coins':>5s}")
        for _, r in agg.iterrows():
            print(f"  {r['horizon_m']:>4.0f}m  {r['raw_ret']:>+8.1f}  {r['btc_fwd']:>+8.1f}  "
                  f"{r['excess']:>+8.1f}  {r['beta_adj']:>+8.1f}  {r['alt_already']:>+10.1f}  "
                  f"{r['raw_wr']:>5.1f}%  {r['excess_wr']:>5.1f}%  {r['n_coins']:>5.0f}")
        
        # Verdict
        excess_30m = agg[agg['horizon_m'] == 30]['excess'].values
        if len(excess_30m) > 0:
            ex = excess_30m[0]
            if ex > RT_TAKER_BPS:
                print(f"\n  ✅ REAL EDGE: Excess return {ex:+.1f} bps > {RT_TAKER_BPS} bps fees")
            elif ex > 0:
                print(f"\n  ⚠️ MARGINAL: Excess return {ex:+.1f} bps (positive but below fees)")
            else:
                print(f"\n  ❌ FAKE: Excess return {ex:+.1f} bps — just correlated market moves")
    
    # Lagged entry analysis  
    print("\n\n--- LAGGED ENTRY ANALYSIS (BTC up, 30m horizon) ---")
    sub = df[(df['direction'] == 'up') & (df['horizon_m'] == 30)]
    lag_cols = [c for c in df.columns if c.startswith('lag_')]
    if lag_cols:
        print(f"  {'Lag':>6s}  {'Mean Ret':>10s}  {'Net Fees':>10s}")
        print(f"  {'0m':>6s}  {sub['raw_alt_ret'].mean():>+10.1f}  {sub['raw_alt_ret'].mean()-RT_TAKER_BPS:>+10.1f}")
        for lc in sorted(lag_cols):
            if lc in sub.columns:
                vals = sub[lc].dropna()
                if len(vals) > 0:
                    print(f"  {lc.replace('lag_',''):>6s}  {vals.mean():>+10.1f}  {vals.mean()-RT_TAKER_BPS:>+10.1f}")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
