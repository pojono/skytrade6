#!/usr/bin/env python3
"""
IDEA 5: Spot-Futures Price Divergence
======================================
When futures price diverges from spot price, it mean-reverts.
We have both kline_1m (futures) and kline_1m_spot for ~91 symbols.

Signals:
 A) Futures-spot spread z-score extreme → fade
 B) Futures leading spot (futures move, spot hasn't) → trade spot direction
 C) Spot leading futures (real demand) → follow spot direction
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import (
    get_symbols, load_kline, progress_bar, RT_TAKER_BPS
)

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'
START = '2025-06-01'
END = '2026-03-04'
HORIZONS = [5, 15, 30, 60, 240]


def get_spot_symbols(min_days=30):
    """Get symbols that have both futures AND spot kline data."""
    from pathlib import Path
    import os, re
    datalake = Path("/home/ubuntu/Projects/skytrade6/datalake/bybit")
    pat = re.compile(r'^\d{4}-\d{2}-\d{2}_kline_1m_spot\.csv$')
    symbols = []
    for d in sorted(datalake.iterdir()):
        if not d.is_dir():
            continue
        spot_files = [f for f in os.listdir(d) if pat.match(f)]
        if len(spot_files) >= min_days:
            symbols.append(d.name)
    return symbols


def analyze_symbol(symbol, start, end):
    kf = load_kline(symbol, start, end, spot=False)
    ks = load_kline(symbol, start, end, spot=True)
    
    if kf.empty or ks.empty:
        return None
    if len(kf) < 5000 or len(ks) < 5000:
        return None
    
    # Align on timestamp
    kf = kf[['ts', 'close']].rename(columns={'close': 'fut_close'}).set_index('ts')
    ks = ks[['ts', 'close']].rename(columns={'close': 'spot_close'}).set_index('ts')
    kf = kf[~kf.index.duplicated(keep='first')]
    ks = ks[~ks.index.duplicated(keep='first')]
    
    merged = kf.join(ks, how='inner').dropna()
    if len(merged) < 5000:
        return None
    
    # Basis = (futures - spot) / spot in bps
    merged['basis_bps'] = (merged['fut_close'] / merged['spot_close'] - 1) * 10000
    
    # Futures and spot 1m returns
    merged['fut_ret_1m'] = merged['fut_close'].pct_change() * 10000
    merged['spot_ret_1m'] = merged['spot_close'].pct_change() * 10000
    
    # Return divergence: futures ret - spot ret (who moved more?)
    merged['ret_div_1m'] = merged['fut_ret_1m'] - merged['spot_ret_1m']
    
    # Futures forward returns
    for h in HORIZONS:
        merged[f'fwd_{h}m'] = (merged['fut_close'].shift(-h) / merged['fut_close'] - 1) * 10000
    
    results = []
    
    # === Signal A: Basis z-score extreme → fade ===
    for lb in [60, 240, 720]:
        roll_mean = merged['basis_bps'].rolling(lb).mean()
        roll_std = merged['basis_bps'].rolling(lb).std().clip(lower=0.01)
        zscore = (merged['basis_bps'] - roll_mean) / roll_std
        
        for zt in [2.0, 2.5, 3.0]:
            short_sig = zscore > zt   # futures overpriced → short futures
            long_sig = zscore < -zt   # futures underpriced → long futures
            
            for h in HORIZONS:
                col = f'fwd_{h}m'
                if short_sig.sum() > 10:
                    rets = -merged.loc[short_sig, col].dropna()
                    if len(rets) > 5:
                        results.append({
                            'symbol': symbol, 'signal': f'basis_z{zt}_lb{lb}',
                            'direction': 'short', 'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(), 'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - RT_TAKER_BPS,
                        })
                if long_sig.sum() > 10:
                    rets = merged.loc[long_sig, col].dropna()
                    if len(rets) > 5:
                        results.append({
                            'symbol': symbol, 'signal': f'basis_z{zt}_lb{lb}',
                            'direction': 'long', 'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(), 'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - RT_TAKER_BPS,
                        })
    
    # === Signal B: Return divergence — futures led, spot didn't follow ===
    for lb in [30, 60]:
        div_std = merged['ret_div_1m'].rolling(lb).std().clip(lower=0.01)
        div_z = merged['ret_div_1m'] / div_std
        
        for zt in [2.0, 3.0]:
            # Futures pumped way more than spot → overbought → short
            short_sig = div_z > zt
            # Futures dumped way more than spot → oversold → long
            long_sig = div_z < -zt
            
            for h in HORIZONS:
                col = f'fwd_{h}m'
                if short_sig.sum() > 10:
                    rets = -merged.loc[short_sig, col].dropna()
                    if len(rets) > 5:
                        results.append({
                            'symbol': symbol, 'signal': f'retdiv_z{zt}_lb{lb}',
                            'direction': 'short', 'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(), 'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - RT_TAKER_BPS,
                        })
                if long_sig.sum() > 10:
                    rets = merged.loc[long_sig, col].dropna()
                    if len(rets) > 5:
                        results.append({
                            'symbol': symbol, 'signal': f'retdiv_z{zt}_lb{lb}',
                            'direction': 'long', 'horizon_m': h,
                            'n_signals': len(rets),
                            'mean_ret': rets.mean(), 'median_ret': rets.median(),
                            'win_rate': (rets > 0).mean() * 100,
                            'net_fees': rets.mean() - RT_TAKER_BPS,
                        })
    
    # === Signal C: Spot leading — spot moved, futures didn't ===
    merged['spot_ret_3m'] = (merged['spot_close'] / merged['spot_close'].shift(3) - 1) * 10000
    merged['fut_ret_3m'] = (merged['fut_close'] / merged['fut_close'].shift(3) - 1) * 10000
    merged['spot_lead'] = merged['spot_ret_3m'] - merged['fut_ret_3m']
    
    for thresh in [20, 40, 60]:
        # Spot pumped more than futures → real demand → futures will catch up (long)
        long_sig = merged['spot_lead'] > thresh
        short_sig = merged['spot_lead'] < -thresh
        
        for h in HORIZONS:
            col = f'fwd_{h}m'
            if long_sig.sum() > 10:
                rets = merged.loc[long_sig, col].dropna()
                if len(rets) > 5:
                    results.append({
                        'symbol': symbol, 'signal': f'spotlead_gt{thresh}',
                        'direction': 'long', 'horizon_m': h,
                        'n_signals': len(rets),
                        'mean_ret': rets.mean(), 'median_ret': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'net_fees': rets.mean() - RT_TAKER_BPS,
                    })
            if short_sig.sum() > 10:
                rets = -merged.loc[short_sig, col].dropna()
                if len(rets) > 5:
                    results.append({
                        'symbol': symbol, 'signal': f'spotlead_lt{thresh}',
                        'direction': 'short', 'horizon_m': h,
                        'n_signals': len(rets),
                        'mean_ret': rets.mean(), 'median_ret': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'net_fees': rets.mean() - RT_TAKER_BPS,
                    })
    
    return results


def main():
    print("=" * 80)
    print("IDEA 5: SPOT-FUTURES PRICE DIVERGENCE")
    print(f"Period: {START} → {END}")
    print("=" * 80)
    
    symbols = get_spot_symbols(min_days=30)
    print(f"\nFound {len(symbols)} symbols with spot+futures data")
    
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
    df.to_csv(f'{OUT}/idea5_spot_futures_raw.csv', index=False)
    
    agg = df.groupby(['signal', 'direction', 'horizon_m']).agg(
        n_symbols=('symbol', 'nunique'),
        total_sigs=('n_signals', 'sum'),
        mean_ret=('mean_ret', 'mean'),
        mean_wr=('win_rate', 'mean'),
        net_fees=('net_fees', 'mean'),
    ).reset_index().sort_values('net_fees', ascending=False)
    
    agg.to_csv(f'{OUT}/idea5_spot_futures_agg.csv', index=False)
    
    profitable = agg[agg['net_fees'] > 0]
    print(f"\n✅ {len(df)} raw, {len(profitable)} configs profitable after fees")
    
    if len(profitable) > 0:
        print("\nTop 20 profitable configs:")
        for _, r in profitable.head(20).iterrows():
            print(f"  {r['signal']:25s} {r['direction']:6s} {r['horizon_m']:>4.0f}m  "
                  f"net={r['net_fees']:+7.1f}bps  WR={r['mean_wr']:5.1f}%  syms={r['n_symbols']:.0f}")
    
    # Cross-symbol consistency
    print("\n\nCross-symbol consistency (>50% profitable):")
    for _, row in profitable.head(15).iterrows():
        s, d, h = row['signal'], row['direction'], row['horizon_m']
        sub = df[(df['signal'] == s) & (df['direction'] == d) & (df['horizon_m'] == h)]
        pct = (sub['net_fees'] > 0).mean() * 100
        if pct > 50:
            print(f"  ✅ {s} {d} {h:.0f}m: {pct:.0f}% of {len(sub)} coins, net {row['net_fees']:+.1f}bps")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
