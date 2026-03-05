#!/usr/bin/env python3
"""
IDEA 2: Premium Index Mean Reversion
=====================================
Hypothesis: Futures premium (basis) over spot mean-reverts.
When premium spikes to extremes, futures are overpriced/underpriced vs spot.
Fade the extreme → capture the reversion.

Premium index data is 1m resolution — very granular.
Premium = (futures_price - spot_index_price) / spot_index_price

Signals:
 A) Premium z-score > 2σ → short (futures overpriced, will revert down)
 B) Premium z-score < -2σ → long (futures underpriced, will revert up)
 C) Combined with OI/LS for confirmation

We test multiple lookback windows and z-score thresholds.
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import (
    get_symbols, load_kline, load_premium, load_oi, load_ls_ratio,
    progress_bar, RT_TAKER_BPS, RT_MAKER_BPS
)

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'

START = '2025-06-01'
END = '2026-03-04'
HORIZONS = [5, 15, 30, 60, 240]  # minutes forward
LOOKBACKS = [60, 120, 360, 720]  # minutes for z-score calculation
ZSCORE_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5]


def analyze_symbol(symbol, start, end):
    """Analyze premium mean-reversion for a single symbol."""
    kline = load_kline(symbol, start, end)
    prem = load_premium(symbol, start, end)
    
    if kline.empty or prem.empty:
        return None
    if len(prem) < 1000:
        return None
    
    # Premium close is the 1m premium index value
    prem = prem[['ts', 'close']].rename(columns={'close': 'premium'})
    kline_sub = kline[['ts', 'close']].rename(columns={'close': 'price'})
    
    # Merge on timestamp
    merged = pd.merge(kline_sub, prem, on='ts', how='inner')
    merged = merged.sort_values('ts').reset_index(drop=True)
    
    if len(merged) < max(LOOKBACKS) + max(HORIZONS) + 100:
        return None
    
    # Premium in bps
    merged['premium_bps'] = merged['premium'] * 10000
    
    # Forward returns on price
    for h in HORIZONS:
        merged[f'fwd_{h}m_bps'] = (merged['price'].shift(-h) / merged['price'] - 1) * 10000
    
    results = []
    
    for lb in LOOKBACKS:
        # Rolling z-score of premium
        roll_mean = merged['premium_bps'].rolling(lb).mean()
        roll_std = merged['premium_bps'].rolling(lb).std().clip(lower=0.01)
        merged['prem_zscore'] = (merged['premium_bps'] - roll_mean) / roll_std
        
        # Also track premium level stats
        merged['prem_abs'] = merged['premium_bps'].abs()
        
        for zt in ZSCORE_THRESHOLDS:
            # Short signal: premium too high (futures overpriced)
            short_sig = merged['prem_zscore'] > zt
            # Long signal: premium too low (futures underpriced)
            long_sig = merged['prem_zscore'] < -zt
            
            n_short = short_sig.sum()
            n_long = long_sig.sum()
            
            if n_short + n_long < 10:
                continue
            
            for h in HORIZONS:
                col = f'fwd_{h}m_bps'
                valid = merged[col].notna()
                
                # Short: we profit when price goes DOWN
                if n_short > 2:
                    mask = short_sig & valid
                    rets = -merged.loc[mask, col]  # negate for short
                    if len(rets) > 2:
                        # Premium reversion: also measure how much premium itself reverts
                        prem_at_sig = merged.loc[mask, 'premium_bps']
                        results.append({
                            'symbol': symbol,
                            'lookback': lb,
                            'zscore_thresh': zt,
                            'direction': 'short',
                            'horizon_m': h,
                            'n_signals': int(len(rets)),
                            'mean_ret_bps': rets.mean(),
                            'median_ret_bps': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'sharpe': rets.mean() / max(rets.std(), 0.01),
                            'net_of_fees_bps': rets.mean() - RT_TAKER_BPS,
                            'avg_premium_bps': prem_at_sig.mean(),
                        })
                
                # Long: we profit when price goes UP
                if n_long > 2:
                    mask = long_sig & valid
                    rets = merged.loc[mask, col]  # positive for long
                    if len(rets) > 2:
                        prem_at_sig = merged.loc[mask, 'premium_bps']
                        results.append({
                            'symbol': symbol,
                            'lookback': lb,
                            'zscore_thresh': zt,
                            'direction': 'long',
                            'horizon_m': h,
                            'n_signals': int(len(rets)),
                            'mean_ret_bps': rets.mean(),
                            'median_ret_bps': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'sharpe': rets.mean() / max(rets.std(), 0.01),
                            'net_of_fees_bps': rets.mean() - RT_TAKER_BPS,
                            'avg_premium_bps': prem_at_sig.mean(),
                        })
    
    return results


def main():
    print("=" * 80)
    print("IDEA 2: PREMIUM INDEX MEAN REVERSION")
    print(f"Period: {START} → {END}")
    print(f"Fees: taker {RT_TAKER_BPS} bps RT")
    print("=" * 80)
    
    symbols = get_symbols(min_days=30)
    print(f"\nFound {len(symbols)} symbols")
    print(f"Testing {len(LOOKBACKS)} lookbacks × {len(ZSCORE_THRESHOLDS)} z-thresholds × {len(HORIZONS)} horizons")
    print()
    
    all_results = []
    t0 = time.time()
    
    for i, sym in enumerate(symbols):
        progress_bar(i, len(symbols), prefix='Scanning', start_time=t0)
        try:
            res = analyze_symbol(sym, START, END)
            if res:
                all_results.extend(res)
        except Exception as e:
            print(f"\n  ⚠ {sym}: {e}")
    
    progress_bar(len(symbols), len(symbols), prefix='Scanning', start_time=t0)
    
    if not all_results:
        print("\n❌ No results!")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT}/idea2_premium_reversion_raw.csv', index=False)
    print(f"\n✅ {len(df)} raw results saved")
    
    # ==================== ANALYSIS ====================
    print("\n" + "=" * 80)
    print("ANALYSIS: Premium reversion signals")
    print("=" * 80)
    
    agg = df.groupby(['lookback', 'zscore_thresh', 'direction', 'horizon_m']).agg(
        n_symbols=('symbol', 'nunique'),
        total_signals=('n_signals', 'sum'),
        mean_ret_bps=('mean_ret_bps', 'mean'),
        median_ret_bps=('median_ret_bps', 'mean'),
        mean_wr=('win_rate', 'mean'),
        mean_sharpe=('sharpe', 'mean'),
        net_fees_bps=('net_of_fees_bps', 'mean'),
        avg_prem_bps=('avg_premium_bps', 'mean'),
    ).reset_index()
    
    agg = agg.sort_values('net_fees_bps', ascending=False)
    agg.to_csv(f'{OUT}/idea2_premium_reversion_agg.csv', index=False)
    
    profitable = agg[agg['net_fees_bps'] > 0]
    print(f"\n🟢 {len(profitable)} configs profitable after fees (out of {len(agg)}):")
    if len(profitable) > 0:
        print(profitable.head(30).to_string(index=False, float_format='%.1f'))
    
    # Best by horizon
    print("\n\nBest config per horizon:")
    for h in HORIZONS:
        sub = agg[agg['horizon_m'] == h].head(1)
        if len(sub) > 0:
            r = sub.iloc[0]
            print(f"  {h:>3}m: lb={r['lookback']:.0f} z>{r['zscore_thresh']:.1f} {r['direction']:5s} "
                  f"net={r['net_fees_bps']:+.1f}bps WR={r['mean_wr']:.0f}% "
                  f"n={r['total_signals']:.0f} syms={r['n_symbols']:.0f} "
                  f"prem={r['avg_prem_bps']:+.1f}bps")
    
    # Cross-symbol consistency
    print("\n\nCross-symbol consistency (>50% of symbols profitable):")
    for _, row in profitable.head(20).iterrows():
        lb, zt, d, h = row['lookback'], row['zscore_thresh'], row['direction'], row['horizon_m']
        sub = df[(df['lookback'] == lb) & (df['zscore_thresh'] == zt) & 
                 (df['direction'] == d) & (df['horizon_m'] == h)]
        pct = (sub['net_of_fees_bps'] > 0).mean() * 100
        if pct > 50:
            print(f"  ✅ lb={lb:.0f} z>{zt:.1f} {d} {h}m: {pct:.0f}% of {len(sub)} coins, "
                  f"net {row['net_fees_bps']:+.1f}bps")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
