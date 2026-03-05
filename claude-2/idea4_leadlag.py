#!/usr/bin/env python3
"""
IDEA 4: Cross-Symbol Lead-Lag Momentum
=======================================
Hypothesis: BTC/ETH price moves propagate to altcoins with a 1-5 minute delay.
When BTC drops 0.5%+ in 1-3 minutes and alts haven't reacted, short the laggards.

Also test: does ANY large-cap move predict alt moves? (SOL, XRP, etc.)

Signals:
 A) BTC 1m return > threshold → trade alts in same direction (momentum propagation)
 B) BTC 3m return > threshold → trade alts that haven't moved yet
 C) ETH as leader instead of BTC
 D) Sector-based: meme coins follow each other, L1s follow each other
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import (
    get_symbols, load_kline, progress_bar, RT_TAKER_BPS, RT_MAKER_BPS
)

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'

START = '2025-06-01'
END = '2026-03-04'
LEADERS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
HORIZONS_FWD = [1, 3, 5, 10, 15, 30]  # forward minutes for alt return
LEADER_WINDOWS = [1, 3, 5]  # leader move window in minutes
LEADER_THRESHOLDS_BPS = [30, 50, 75, 100, 150]  # min leader move


def load_leader_returns(leaders, start, end):
    """Load 1m returns for leader coins."""
    leader_data = {}
    for sym in leaders:
        k = load_kline(sym, start, end)
        if k.empty:
            continue
        k = k[['ts', 'close']].set_index('ts').sort_index()
        k = k[~k.index.duplicated(keep='first')]
        for w in LEADER_WINDOWS:
            k[f'ret_{w}m_bps'] = (k['close'] / k['close'].shift(w) - 1) * 10000
        leader_data[sym] = k
    return leader_data


def analyze_follower(follower_sym, leader_data, start, end):
    """Analyze if a follower coin follows leader moves with delay."""
    fk = load_kline(follower_sym, start, end)
    if fk.empty or len(fk) < 5000:
        return None
    
    fk = fk[['ts', 'close']].set_index('ts').sort_index()
    fk = fk[~fk.index.duplicated(keep='first')]
    
    # Compute follower forward returns
    for h in HORIZONS_FWD:
        fk[f'fwd_{h}m_bps'] = (fk['close'].shift(-h) / fk['close'] - 1) * 10000
    
    # Also compute follower's own recent move (to check if it already reacted)
    for w in LEADER_WINDOWS:
        fk[f'own_ret_{w}m_bps'] = (fk['close'] / fk['close'].shift(w) - 1) * 10000
    
    results = []
    
    for leader_sym, lk in leader_data.items():
        if leader_sym == follower_sym:
            continue
        
        # Align on timestamp
        merged = fk.join(lk[['close'] + [c for c in lk.columns if 'ret_' in c]], 
                        rsuffix='_leader', how='inner')
        
        if len(merged) < 2000:
            continue
        
        for lw in LEADER_WINDOWS:
            leader_ret_col = f'ret_{lw}m_bps'
            if leader_ret_col not in merged.columns:
                continue
            
            for lt in LEADER_THRESHOLDS_BPS:
                # Leader moved UP strongly
                leader_up = merged[leader_ret_col] > lt
                # Leader moved DOWN strongly
                leader_down = merged[leader_ret_col] < -lt
                
                # Variant: only when follower hasn't moved yet (lagging)
                own_ret_col = f'own_ret_{lw}m_bps'
                lag_threshold = lt * 0.3  # follower moved < 30% of leader
                
                for require_lag in [False, True]:
                    if require_lag:
                        up_sig = leader_up & (merged[own_ret_col].abs() < lag_threshold)
                        down_sig = leader_down & (merged[own_ret_col].abs() < lag_threshold)
                        variant = f'{leader_sym[:3]}_{lw}m>{lt}bps_lag'
                    else:
                        up_sig = leader_up
                        down_sig = leader_down
                        variant = f'{leader_sym[:3]}_{lw}m>{lt}bps'
                    
                    for h in HORIZONS_FWD:
                        fwd_col = f'fwd_{h}m_bps'
                        if fwd_col not in merged.columns:
                            continue
                        
                        # When leader goes UP, does follower follow UP?
                        if up_sig.sum() > 10:
                            rets = merged.loc[up_sig, fwd_col].dropna()
                            if len(rets) > 5:
                                results.append({
                                    'follower': follower_sym,
                                    'leader': leader_sym,
                                    'variant': variant,
                                    'leader_dir': 'up',
                                    'fwd_horizon_m': h,
                                    'n_signals': int(len(rets)),
                                    'mean_ret_bps': rets.mean(),
                                    'median_ret_bps': rets.median(),
                                    'win_rate': (rets > 0).mean() * 100,
                                    'sharpe': rets.mean() / max(rets.std(), 0.01),
                                    'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                                })
                        
                        # When leader goes DOWN, does follower follow DOWN?
                        if down_sig.sum() > 10:
                            rets = -merged.loc[down_sig, fwd_col].dropna()  # short
                            if len(rets) > 5:
                                results.append({
                                    'follower': follower_sym,
                                    'leader': leader_sym,
                                    'variant': variant,
                                    'leader_dir': 'down',
                                    'fwd_horizon_m': h,
                                    'n_signals': int(len(rets)),
                                    'mean_ret_bps': rets.mean(),
                                    'median_ret_bps': rets.median(),
                                    'win_rate': (rets > 0).mean() * 100,
                                    'sharpe': rets.mean() / max(rets.std(), 0.01),
                                    'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                                })
    
    return results


def main():
    print("=" * 80)
    print("IDEA 4: CROSS-SYMBOL LEAD-LAG MOMENTUM")
    print(f"Period: {START} → {END}")
    print(f"Leaders: {LEADERS}")
    print(f"Fees: taker {RT_TAKER_BPS} bps RT")
    print("=" * 80)
    
    symbols = get_symbols(min_days=30)
    followers = [s for s in symbols if s not in LEADERS]
    print(f"\nLoading leader data...")
    leader_data = load_leader_returns(LEADERS, START, END)
    print(f"Loaded {len(leader_data)} leaders")
    print(f"Testing {len(followers)} followers × {len(LEADERS)} leaders × "
          f"{len(LEADER_WINDOWS)} windows × {len(LEADER_THRESHOLDS_BPS)} thresholds")
    print()
    
    all_results = []
    t0 = time.time()
    
    for i, sym in enumerate(followers):
        progress_bar(i, len(followers), prefix='Scanning', start_time=t0)
        try:
            res = analyze_follower(sym, leader_data, START, END)
            if res:
                all_results.extend(res)
        except Exception as e:
            print(f"\n  ⚠ {sym}: {e}")
    
    progress_bar(len(followers), len(followers), prefix='Scanning', start_time=t0)
    
    if not all_results:
        print("\n❌ No results!")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT}/idea4_leadlag_raw.csv', index=False)
    print(f"\n✅ {len(df)} raw results saved")
    
    # ==================== ANALYSIS ====================
    print("\n" + "=" * 80)
    print("ANALYSIS: Lead-lag momentum")
    print("=" * 80)
    
    # Aggregate by variant (across all followers)
    agg = df.groupby(['variant', 'leader_dir', 'fwd_horizon_m']).agg(
        n_followers=('follower', 'nunique'),
        total_signals=('n_signals', 'sum'),
        mean_ret_bps=('mean_ret_bps', 'mean'),
        median_ret_bps=('median_ret_bps', 'mean'),
        mean_wr=('win_rate', 'mean'),
        mean_sharpe=('sharpe', 'mean'),
        net_fees_bps=('net_fees_bps', 'mean'),
    ).reset_index()
    
    agg = agg.sort_values('net_fees_bps', ascending=False)
    agg.to_csv(f'{OUT}/idea4_leadlag_agg.csv', index=False)
    
    profitable = agg[agg['net_fees_bps'] > 0]
    print(f"\n🟢 {len(profitable)} configs profitable after fees (of {len(agg)}):")
    if len(profitable) > 0:
        for _, r in profitable.head(25).iterrows():
            print(f"  {r['variant']:30s} {r['leader_dir']:5s} fwd={r['fwd_horizon_m']:>3.0f}m "
                  f"net={r['net_fees_bps']:+6.1f}bps WR={r['mean_wr']:5.1f}% "
                  f"sigs={r['total_signals']:>8.0f} coins={r['n_followers']:.0f}")
    
    # Best per leader
    print("\n\nBest config per leader:")
    for leader in LEADERS:
        prefix = leader[:3]
        sub = agg[agg['variant'].str.startswith(prefix)].head(1)
        if len(sub) > 0:
            r = sub.iloc[0]
            print(f"  {leader}: {r['variant']} {r['leader_dir']} fwd={r['fwd_horizon_m']:.0f}m "
                  f"net={r['net_fees_bps']:+.1f}bps WR={r['mean_wr']:.0f}%")
    
    # Cross-follower consistency
    print("\n\nCross-follower consistency (>55% of followers profitable):")
    for _, row in profitable.head(15).iterrows():
        v, d, h = row['variant'], row['leader_dir'], row['fwd_horizon_m']
        sub = df[(df['variant'] == v) & (df['leader_dir'] == d) & (df['fwd_horizon_m'] == h)]
        pct = (sub['net_fees_bps'] > 0).mean() * 100
        if pct > 55:
            print(f"  ✅ {v} {d} fwd={h:.0f}m: {pct:.0f}% of {len(sub)} coins, "
                  f"net {row['net_fees_bps']:+.1f}bps")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
