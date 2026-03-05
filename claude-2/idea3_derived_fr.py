#!/usr/bin/env python3
"""
IDEA 3: Derived 1-Minute Funding Rate from Mark Price
======================================================
Bybit FR data is only at 1h/8h resolution. But we can derive a continuous
1-minute implied funding rate from mark_price vs spot_price (premium index).

funding_rate ≈ clamp(premium_index, -0.05%, +0.05%) + interest_rate_component
The interest rate component is ~0.01%/8h = negligible at 1m scale.

So: implied_fr_1m ≈ premium_index_1m (already have this data!)

This gives us 1-minute resolution FR signals instead of 1h/8h.
We can detect FR spikes BEFORE they settle and trade accordingly.

Signals:
 A) Implied FR spike → predict next actual FR settlement will be high → position early
 B) FR momentum: rising implied FR → trend continuation
 C) FR reversal: implied FR at extreme → mean reversion
 D) FR divergence: implied FR vs actual settled FR → detect regime changes
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import (
    get_symbols, load_kline, load_premium, load_funding_rate, load_mark,
    progress_bar, RT_TAKER_BPS, RT_MAKER_BPS
)

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'

START = '2025-06-01'
END = '2026-03-04'
HORIZONS = [5, 15, 30, 60, 240]


def analyze_symbol(symbol, start, end):
    """Analyze derived FR signals for one symbol."""
    prem = load_premium(symbol, start, end)
    kline = load_kline(symbol, start, end)
    fr = load_funding_rate(symbol, start, end)
    
    if prem.empty or kline.empty or fr.empty:
        return None
    if len(prem) < 2000:
        return None
    
    # Implied FR = premium index close (already in decimal form)
    prem = prem[['ts', 'close']].rename(columns={'close': 'implied_fr'})
    prem['implied_fr_bps'] = prem['implied_fr'] * 10000
    
    # Actual FR
    fr = fr[['ts', 'fundingRate']].copy()
    fr['actual_fr_bps'] = fr['fundingRate'] * 10000
    
    # Price
    kline_sub = kline[['ts', 'close']].rename(columns={'close': 'price'})
    
    # Merge
    merged = pd.merge(kline_sub, prem, on='ts', how='inner')
    merged = merged.sort_values('ts').reset_index(drop=True)
    
    # Merge actual FR (forward fill — latest known settled FR)
    merged = pd.merge_asof(merged.sort_values('ts'), fr.sort_values('ts'),
                           on='ts', direction='backward')
    
    if len(merged) < 2000:
        return None
    
    # Forward returns
    for h in HORIZONS:
        merged[f'fwd_{h}m_bps'] = (merged['price'].shift(-h) / merged['price'] - 1) * 10000
    
    # === Feature engineering ===
    # 1. Implied FR rolling stats
    for w in [15, 60, 240]:
        merged[f'ifr_ma_{w}'] = merged['implied_fr_bps'].rolling(w).mean()
        merged[f'ifr_std_{w}'] = merged['implied_fr_bps'].rolling(w).std().clip(lower=0.01)
        merged[f'ifr_zscore_{w}'] = (merged['implied_fr_bps'] - merged[f'ifr_ma_{w}']) / merged[f'ifr_std_{w}']
    
    # 2. Implied FR momentum (rate of change)
    merged['ifr_momentum_15'] = merged['implied_fr_bps'] - merged['implied_fr_bps'].shift(15)
    merged['ifr_momentum_60'] = merged['implied_fr_bps'] - merged['implied_fr_bps'].shift(60)
    
    # 3. Implied FR level (absolute)
    merged['ifr_abs'] = merged['implied_fr_bps'].abs()
    
    # 4. Divergence: implied FR vs last settled FR
    merged['fr_divergence'] = merged['implied_fr_bps'] - merged['actual_fr_bps']
    
    merged = merged.dropna()
    if len(merged) < 500:
        return None
    
    results = []
    
    # === Signal A: Implied FR at extremes (mean reversion) ===
    for w in [60, 240]:
        for zt in [2.0, 2.5, 3.0]:
            zscore = merged[f'ifr_zscore_{w}']
            
            # High implied FR → market will correct down (short)
            short_sig = zscore > zt
            # Low implied FR → market will correct up (long)
            long_sig = zscore < -zt
            
            for h in HORIZONS:
                col = f'fwd_{h}m_bps'
                
                if short_sig.sum() > 5:
                    rets = -merged.loc[short_sig, col]
                    results.append({
                        'symbol': symbol, 'signal': f'ifr_revert_z{zt}_w{w}',
                        'direction': 'short', 'horizon_m': h,
                        'n_signals': int(short_sig.sum()),
                        'mean_ret_bps': rets.mean(), 'median_ret_bps': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'sharpe': rets.mean() / max(rets.std(), 0.01),
                        'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                    })
                
                if long_sig.sum() > 5:
                    rets = merged.loc[long_sig, col]
                    results.append({
                        'symbol': symbol, 'signal': f'ifr_revert_z{zt}_w{w}',
                        'direction': 'long', 'horizon_m': h,
                        'n_signals': int(long_sig.sum()),
                        'mean_ret_bps': rets.mean(), 'median_ret_bps': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'sharpe': rets.mean() / max(rets.std(), 0.01),
                        'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                    })
    
    # === Signal B: FR Momentum (trend following) ===
    for mom_col, mom_name in [('ifr_momentum_15', 'mom15'), ('ifr_momentum_60', 'mom60')]:
        for thresh_bps in [1.0, 2.0, 5.0]:
            # Strong upward FR momentum → longs paying more → price likely UP short-term
            long_sig = merged[mom_col] > thresh_bps
            short_sig = merged[mom_col] < -thresh_bps
            
            for h in HORIZONS:
                col = f'fwd_{h}m_bps'
                
                if long_sig.sum() > 10:
                    rets = merged.loc[long_sig, col]
                    results.append({
                        'symbol': symbol, 'signal': f'ifr_{mom_name}_gt{thresh_bps}',
                        'direction': 'long', 'horizon_m': h,
                        'n_signals': int(long_sig.sum()),
                        'mean_ret_bps': rets.mean(), 'median_ret_bps': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'sharpe': rets.mean() / max(rets.std(), 0.01),
                        'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                    })
                
                if short_sig.sum() > 10:
                    rets = -merged.loc[short_sig, col]
                    results.append({
                        'symbol': symbol, 'signal': f'ifr_{mom_name}_lt{thresh_bps}',
                        'direction': 'short', 'horizon_m': h,
                        'n_signals': int(short_sig.sum()),
                        'mean_ret_bps': rets.mean(), 'median_ret_bps': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'sharpe': rets.mean() / max(rets.std(), 0.01),
                        'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                    })
    
    # === Signal C: FR Divergence (implied vs actual) ===
    div_std = merged['fr_divergence'].std()
    if div_std > 0.01:
        for zt in [1.5, 2.0, 3.0]:
            div_z = merged['fr_divergence'] / div_std
            
            # Implied FR >> actual FR → market getting frothy → short
            short_sig = div_z > zt
            long_sig = div_z < -zt
            
            for h in HORIZONS:
                col = f'fwd_{h}m_bps'
                
                if short_sig.sum() > 5:
                    rets = -merged.loc[short_sig, col]
                    results.append({
                        'symbol': symbol, 'signal': f'fr_diverge_z{zt}',
                        'direction': 'short', 'horizon_m': h,
                        'n_signals': int(short_sig.sum()),
                        'mean_ret_bps': rets.mean(), 'median_ret_bps': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'sharpe': rets.mean() / max(rets.std(), 0.01),
                        'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                    })
                
                if long_sig.sum() > 5:
                    rets = merged.loc[long_sig, col]
                    results.append({
                        'symbol': symbol, 'signal': f'fr_diverge_z{zt}',
                        'direction': 'long', 'horizon_m': h,
                        'n_signals': int(long_sig.sum()),
                        'mean_ret_bps': rets.mean(), 'median_ret_bps': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'sharpe': rets.mean() / max(rets.std(), 0.01),
                        'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                    })
    
    # === Signal D: Absolute FR level (high |FR| = volatile = directional) ===
    for thresh in [5.0, 10.0, 20.0]:
        high_pos_fr = merged['implied_fr_bps'] > thresh
        high_neg_fr = merged['implied_fr_bps'] < -thresh
        
        for h in HORIZONS:
            col = f'fwd_{h}m_bps'
            
            # High positive implied FR → longs paying → price might continue up short-term
            if high_pos_fr.sum() > 10:
                rets = merged.loc[high_pos_fr, col]
                results.append({
                    'symbol': symbol, 'signal': f'ifr_level_gt{thresh}',
                    'direction': 'long_momentum', 'horizon_m': h,
                    'n_signals': int(high_pos_fr.sum()),
                    'mean_ret_bps': rets.mean(), 'median_ret_bps': rets.median(),
                    'win_rate': (rets > 0).mean() * 100,
                    'sharpe': rets.mean() / max(rets.std(), 0.01),
                    'net_fees_bps': rets.mean() - RT_TAKER_BPS,
                })
                # Also test contrarian (fade)
                rets_short = -rets
                results.append({
                    'symbol': symbol, 'signal': f'ifr_level_gt{thresh}_fade',
                    'direction': 'short_contrarian', 'horizon_m': h,
                    'n_signals': int(high_pos_fr.sum()),
                    'mean_ret_bps': rets_short.mean(), 'median_ret_bps': rets_short.median(),
                    'win_rate': (rets_short > 0).mean() * 100,
                    'sharpe': rets_short.mean() / max(rets_short.std(), 0.01),
                    'net_fees_bps': rets_short.mean() - RT_TAKER_BPS,
                })
    
    return results


def main():
    print("=" * 80)
    print("IDEA 3: DERIVED 1-MINUTE FUNDING RATE FROM MARK/PREMIUM")
    print(f"Period: {START} → {END}")
    print(f"Fees: taker {RT_TAKER_BPS} bps RT")
    print("=" * 80)
    
    symbols = get_symbols(min_days=30)
    print(f"\nFound {len(symbols)} symbols")
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
    df.to_csv(f'{OUT}/idea3_derived_fr_raw.csv', index=False)
    print(f"\n✅ {len(df)} raw results saved")
    
    # ==================== ANALYSIS ====================
    print("\n" + "=" * 80)
    print("ANALYSIS: Derived FR signals")
    print("=" * 80)
    
    agg = df.groupby(['signal', 'direction', 'horizon_m']).agg(
        n_symbols=('symbol', 'nunique'),
        total_signals=('n_signals', 'sum'),
        mean_ret_bps=('mean_ret_bps', 'mean'),
        median_ret_bps=('median_ret_bps', 'mean'),
        mean_wr=('win_rate', 'mean'),
        mean_sharpe=('sharpe', 'mean'),
        net_fees_bps=('net_fees_bps', 'mean'),
    ).reset_index()
    
    agg = agg.sort_values('net_fees_bps', ascending=False)
    agg.to_csv(f'{OUT}/idea3_derived_fr_agg.csv', index=False)
    
    profitable = agg[agg['net_fees_bps'] > 0]
    print(f"\n🟢 {len(profitable)} configs profitable after {RT_TAKER_BPS}bps fees (of {len(agg)}):")
    if len(profitable) > 0:
        print(profitable.head(30).to_string(index=False, float_format='%.1f'))
    
    # Best by signal type
    print("\n\nBest signal type:")
    for sig_type in df['signal'].str.extract(r'^(ifr_revert|ifr_mom|fr_diverge|ifr_level)')[0].dropna().unique():
        sub = agg[agg['signal'].str.startswith(sig_type)].head(1)
        if len(sub) > 0:
            r = sub.iloc[0]
            print(f"  {sig_type:25s} {r['signal']:35s} {r['direction']:18s} "
                  f"net={r['net_fees_bps']:+.1f}bps WR={r['mean_wr']:.0f}% syms={r['n_symbols']:.0f}")
    
    # Cross-symbol consistency
    print("\n\nCross-symbol consistency (>50% symbols profitable):")
    for _, row in profitable.head(20).iterrows():
        s, d, h = row['signal'], row['direction'], row['horizon_m']
        sub = df[(df['signal'] == s) & (df['direction'] == d) & (df['horizon_m'] == h)]
        pct = (sub['net_fees_bps'] > 0).mean() * 100
        if pct > 50:
            print(f"  ✅ {s} {d} {h}m: {pct:.0f}% of {len(sub)} coins, net {row['net_fees_bps']:+.1f}bps")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
