#!/usr/bin/env python3
"""
IDEA 7: Cross-Symbol Relative Value / Pairs Trading
=====================================================
Hypothesis: Correlated crypto pairs diverge and converge.
When one outperforms its correlated peer by too much, short the outperformer
and long the underperformer.

Test pairs:
- Memes: DOGEUSDT/1000PEPEUSDT, WIFUSDT/1000BONKUSDT
- L1s: SOLUSDT/AVAXUSDT, NEARUSDT/APTUSDT, SUIUSDT/SEIUSDT
- ETH ecosystem: ETHUSDT/ARBUSDT, ETHUSDT/OPUSDT
- BTC-correlated: LTCUSDT/BCHUSDT

Also auto-detect highest-corr pairs from the data.
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

MANUAL_PAIRS = [
    ('DOGEUSDT', '1000PEPEUSDT'), ('WIFUSDT', '1000BONKUSDT'),
    ('SOLUSDT', 'AVAXUSDT'), ('NEARUSDT', 'APTUSDT'), ('SUIUSDT', 'SEIUSDT'),
    ('ETHUSDT', 'ARBUSDT'), ('ETHUSDT', 'OPUSDT'),
    ('LTCUSDT', 'BCHUSDT'), ('LINKUSDT', 'DOTUSDT'),
    ('XRPUSDT', 'ADAUSDT'), ('ATOMUSDT', 'INJUSDT'),
    ('SOLUSDT', 'SUIUSDT'), ('HYPEUSDT', 'VIRTUALUSDT'),
]


def analyze_pair(sym_a, sym_b, start, end):
    ka = load_kline(sym_a, start, end)
    kb = load_kline(sym_b, start, end)
    
    if ka.empty or kb.empty or len(ka) < 10000 or len(kb) < 10000:
        return None
    
    a = ka[['ts', 'close']].set_index('ts').sort_index()
    b = kb[['ts', 'close']].set_index('ts').sort_index()
    a = a[~a.index.duplicated(keep='first')]
    b = b[~b.index.duplicated(keep='first')]
    a.columns = ['a_close']
    b.columns = ['b_close']
    
    merged = a.join(b, how='inner').dropna()
    if len(merged) < 10000:
        return None
    
    # Log price ratio (spread)
    merged['log_ratio'] = np.log(merged['a_close'] / merged['b_close'])
    
    # Returns
    merged['a_ret_1m'] = merged['a_close'].pct_change() * 10000
    merged['b_ret_1m'] = merged['b_close'].pct_change() * 10000
    merged['spread_ret'] = merged['a_ret_1m'] - merged['b_ret_1m']
    
    # Forward returns for pair trade: long A short B (or reverse)
    for h in [15, 30, 60, 240]:
        merged[f'a_fwd_{h}m'] = (merged['a_close'].shift(-h) / merged['a_close'] - 1) * 10000
        merged[f'b_fwd_{h}m'] = (merged['b_close'].shift(-h) / merged['b_close'] - 1) * 10000
        # Pair P&L: long underperformer, short outperformer
        merged[f'pair_fwd_{h}m'] = merged[f'a_fwd_{h}m'] - merged[f'b_fwd_{h}m']
    
    results = []
    
    # Rolling correlation
    corr = merged['a_ret_1m'].rolling(1440).corr(merged['b_ret_1m'])
    avg_corr = corr.mean()
    
    # === Signal: Spread z-score extreme → mean reversion ===
    for lb in [60, 240, 720]:
        roll_mean = merged['log_ratio'].rolling(lb).mean()
        roll_std = merged['log_ratio'].rolling(lb).std().clip(lower=1e-8)
        zscore = (merged['log_ratio'] - roll_mean) / roll_std
        
        for zt in [1.5, 2.0, 2.5, 3.0]:
            # A overperformed → short A, long B (negative pair return expected)
            short_a = zscore > zt
            # A underperformed → long A, short B (positive pair return expected)
            long_a = zscore < -zt
            
            for h in [15, 30, 60, 240]:
                col = f'pair_fwd_{h}m'
                
                # When A overvalued → pair return should be negative (A drops vs B)
                if short_a.sum() > 10:
                    rets = -merged.loc[short_a, col].dropna()  # negate: short A, long B
                    if len(rets) > 5:
                        results.append({
                            'pair': f'{sym_a}/{sym_b}',
                            'signal': f'spread_z{zt}_lb{lb}',
                            'direction': f'short_{sym_a[:4]}',
                            'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(),
                            'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - 2 * RT_TAKER_BPS,  # 2 legs
                            'avg_corr': avg_corr,
                        })
                
                if long_a.sum() > 10:
                    rets = merged.loc[long_a, col].dropna()
                    if len(rets) > 5:
                        results.append({
                            'pair': f'{sym_a}/{sym_b}',
                            'signal': f'spread_z{zt}_lb{lb}',
                            'direction': f'long_{sym_a[:4]}',
                            'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(),
                            'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - 2 * RT_TAKER_BPS,  # 2 legs
                            'avg_corr': avg_corr,
                        })
    
    # === Signal: Momentum divergence → short-term reversal ===
    for w in [5, 15, 30]:
        merged[f'a_mom_{w}'] = (merged['a_close'] / merged['a_close'].shift(w) - 1) * 10000
        merged[f'b_mom_{w}'] = (merged['b_close'] / merged['b_close'].shift(w) - 1) * 10000
        merged[f'mom_div_{w}'] = merged[f'a_mom_{w}'] - merged[f'b_mom_{w}']
        
        for thresh in [50, 100, 150]:
            # A rallied much more than B → short A, long B
            short_a = merged[f'mom_div_{w}'] > thresh
            long_a = merged[f'mom_div_{w}'] < -thresh
            
            for h in [15, 30, 60, 240]:
                col = f'pair_fwd_{h}m'
                
                if short_a.sum() > 5:
                    rets = -merged.loc[short_a, col].dropna()
                    if len(rets) > 3:
                        results.append({
                            'pair': f'{sym_a}/{sym_b}',
                            'signal': f'momdiv_{w}m_gt{thresh}',
                            'direction': f'short_{sym_a[:4]}',
                            'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(),
                            'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - 2 * RT_TAKER_BPS,
                            'avg_corr': avg_corr,
                        })
                
                if long_a.sum() > 5:
                    rets = merged.loc[long_a, col].dropna()
                    if len(rets) > 3:
                        results.append({
                            'pair': f'{sym_a}/{sym_b}',
                            'signal': f'momdiv_{w}m_gt{thresh}',
                            'direction': f'long_{sym_a[:4]}',
                            'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(),
                            'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - 2 * RT_TAKER_BPS,
                            'avg_corr': avg_corr,
                        })
    
    return results


def find_top_pairs(symbols, start, end, n_pairs=15):
    """Find most correlated pairs from data."""
    print("  Finding top correlated pairs from data...")
    # Load 1h resampled returns for speed
    returns = {}
    for sym in symbols[:60]:  # limit for speed
        k = load_kline(sym, start, end)
        if k.empty or len(k) < 5000:
            continue
        k = k[['ts', 'close']].set_index('ts').sort_index()
        k = k[~k.index.duplicated(keep='first')]
        r = k['close'].resample('1h').last().pct_change().dropna()
        if len(r) > 100:
            returns[sym] = r
    
    if len(returns) < 10:
        return []
    
    ret_df = pd.DataFrame(returns).dropna()
    corr_matrix = ret_df.corr()
    
    # Extract top pairs (excluding self and BTC/ETH which correlate with everything)
    pairs = []
    for i, sym_a in enumerate(corr_matrix.columns):
        for j, sym_b in enumerate(corr_matrix.columns):
            if i >= j:
                continue
            if sym_a in ['BTCUSDT', 'ETHUSDT'] or sym_b in ['BTCUSDT', 'ETHUSDT']:
                continue
            c = corr_matrix.loc[sym_a, sym_b]
            if c > 0.7:
                pairs.append((sym_a, sym_b, c))
    
    pairs.sort(key=lambda x: -x[2])
    print(f"  Found {len(pairs)} pairs with corr > 0.7")
    for a, b, c in pairs[:10]:
        print(f"    {a}/{b}: {c:.3f}")
    
    return [(a, b) for a, b, c in pairs[:n_pairs]]


def main():
    print("=" * 80)
    print("IDEA 7: CROSS-SYMBOL PAIRS TRADING")
    print(f"Period: {START} → {END}")
    print(f"Note: fees are 2x (40 bps RT) for 2-leg pairs")
    print("=" * 80)
    
    symbols = get_symbols(min_days=60)
    
    # Find auto-detected pairs
    auto_pairs = find_top_pairs(symbols, START, END)
    
    # Combine manual + auto, deduplicate
    all_pairs = list(MANUAL_PAIRS)
    for p in auto_pairs:
        if p not in all_pairs and (p[1], p[0]) not in all_pairs:
            all_pairs.append(p)
    
    print(f"\nTesting {len(all_pairs)} pairs total")
    
    all_results = []
    t0 = time.time()
    
    for i, (sym_a, sym_b) in enumerate(all_pairs):
        progress_bar(i, len(all_pairs), prefix='Scanning', start_time=t0)
        try:
            res = analyze_pair(sym_a, sym_b, START, END)
            if res:
                all_results.extend(res)
        except Exception as e:
            print(f"\n  ⚠ {sym_a}/{sym_b}: {e}")
    
    progress_bar(len(all_pairs), len(all_pairs), prefix='Scanning', start_time=t0)
    
    if not all_results:
        print("\n❌ No results!")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT}/idea7_pairs_raw.csv', index=False)
    
    # Aggregate by signal type across all pairs
    agg = df.groupby(['signal', 'horizon_m']).agg(
        n_pairs=('pair', 'nunique'),
        total_sigs=('n_signals', 'sum'),
        mean_ret=('mean_ret', 'mean'),
        mean_wr=('win_rate', 'mean'),
        net_fees=('net_fees', 'mean'),
        avg_corr=('avg_corr', 'mean'),
    ).reset_index().sort_values('net_fees', ascending=False)
    
    agg.to_csv(f'{OUT}/idea7_pairs_agg.csv', index=False)
    
    profitable = agg[agg['net_fees'] > 0]
    print(f"\n✅ {len(df)} raw, {len(profitable)} configs profitable after 40bps fees (of {len(agg)})")
    
    if len(profitable) > 0:
        print("\nTop 20 profitable configs:")
        for _, r in profitable.head(20).iterrows():
            print(f"  {r['signal']:30s} {r['horizon_m']:>4.0f}m  net={r['net_fees']:+7.1f}bps  "
                  f"WR={r['mean_wr']:5.1f}%  pairs={r['n_pairs']:.0f}  corr={r['avg_corr']:.2f}")
    else:
        print("\n🔴 No configs profitable after 40bps RT fees (2 legs)")
    
    # Best individual pairs
    print("\n\nBest individual pairs (30m or 60m horizon):")
    best_pairs = df[(df['horizon_m'].isin([30, 60])) & (df['net_fees'] > 0)]
    if len(best_pairs) > 0:
        best_pairs = best_pairs.sort_values('net_fees', ascending=False)
        for _, r in best_pairs.head(15).iterrows():
            print(f"  {r['pair']:25s} {r['signal']:25s} {r['direction']:15s} {r['horizon_m']:>4.0f}m  "
                  f"net={r['net_fees']:+7.1f}bps  WR={r['win_rate']:5.1f}%  n={r['n_signals']}")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
