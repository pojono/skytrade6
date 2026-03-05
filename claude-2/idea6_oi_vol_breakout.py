#!/usr/bin/env python3
"""
IDEA 6: OI Regime + Volatility Breakout (Coiled Spring)
========================================================
Hypothesis: Low volatility + rising OI = market building positions quietly.
When vol finally breaks out, the move is explosive.

Signals:
 A) Vol compression (ATR or realized vol at N-period low) + OI rising → breakout coming
 B) Vol expansion after compression → momentum continuation  
 C) OI divergence from price → positioning vs price disagree → mean reversion
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import (
    get_symbols, load_kline, load_oi, progress_bar, RT_TAKER_BPS
)

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'
START = '2025-06-01'
END = '2026-03-04'
HORIZONS = [15, 30, 60, 240]


def analyze_symbol(symbol, start, end):
    kline = load_kline(symbol, start, end)
    oi = load_oi(symbol, start, end)
    
    if kline.empty or oi.empty or len(kline) < 10000:
        return None
    
    # 5-minute resample for alignment with OI
    k = kline[['ts', 'open', 'high', 'low', 'close']].set_index('ts').sort_index()
    k = k[~k.index.duplicated(keep='first')]
    k5 = k.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    if 'volume' in kline.columns:
        vol = kline[['ts', 'volume']].set_index('ts').sort_index()
        vol = vol[~vol.index.duplicated(keep='first')]
        k5['volume'] = vol['volume'].resample('5min').sum()
    
    # OI
    oi_s = oi[['ts', 'openInterest']].set_index('ts').sort_index()
    oi_s = oi_s[~oi_s.index.duplicated(keep='first')]
    
    merged = k5.join(oi_s, how='inner').dropna(subset=['close', 'openInterest'])
    if len(merged) < 2000:
        return None
    
    # === Features ===
    # ATR (5-min bars, so 12 bars = 1h)
    tr = pd.DataFrame({
        'hl': merged['high'] - merged['low'],
        'hc': (merged['high'] - merged['close'].shift(1)).abs(),
        'lc': (merged['low'] - merged['close'].shift(1)).abs(),
    }).max(axis=1)
    
    for w in [12, 48, 144]:  # 1h, 4h, 12h in 5min bars
        merged[f'atr_{w}'] = tr.rolling(w).mean()
        merged[f'atr_pctile_{w}'] = merged[f'atr_{w}'].rolling(288).rank(pct=True)  # 1-day rolling percentile
    
    # Realized vol (5-min returns, rolling)
    merged['ret_5m'] = merged['close'].pct_change()
    for w in [12, 48, 144]:
        merged[f'rvol_{w}'] = merged['ret_5m'].rolling(w).std() * np.sqrt(288) * 100  # annualized %
        merged[f'rvol_pctile_{w}'] = merged[f'rvol_{w}'].rolling(288).rank(pct=True)
    
    # OI changes
    merged['oi_pct_1h'] = merged['openInterest'].pct_change(12) * 100
    merged['oi_pct_4h'] = merged['openInterest'].pct_change(48) * 100
    
    # Price momentum
    merged['ret_1h_bps'] = (merged['close'] / merged['close'].shift(12) - 1) * 10000
    merged['ret_4h_bps'] = (merged['close'] / merged['close'].shift(48) - 1) * 10000
    
    # Forward returns
    for h in HORIZONS:
        bars = h // 5
        merged[f'fwd_{h}m_bps'] = (merged['close'].shift(-bars) / merged['close'] - 1) * 10000
    
    merged = merged.dropna()
    if len(merged) < 500:
        return None
    
    results = []
    
    # === Signal A: Vol compression + OI rising = coiled spring ===
    for vol_pctile_thresh in [0.10, 0.15, 0.20]:
        for oi_thresh in [1.0, 2.0, 3.0]:  # OI increased by X% in 4h
            # Low vol AND rising OI
            coiled = (merged['rvol_pctile_48'] < vol_pctile_thresh) & (merged['oi_pct_4h'] > oi_thresh)
            
            if coiled.sum() < 10:
                continue
            
            for h in HORIZONS:
                col = f'fwd_{h}m_bps'
                rets = merged.loc[coiled, col].dropna()
                if len(rets) < 5:
                    continue
                
                # Absolute return (we don't know direction, but we know a big move is coming)
                abs_rets = rets.abs()
                # Also test: does the move tend to continue the recent micro-trend?
                micro_up = coiled & (merged['ret_1h_bps'] > 10)
                micro_down = coiled & (merged['ret_1h_bps'] < -10)
                
                results.append({
                    'symbol': symbol,
                    'signal': f'coiled_v{vol_pctile_thresh}_oi{oi_thresh}',
                    'direction': 'abs_move',
                    'horizon_m': h,
                    'n_signals': len(rets),
                    'mean_abs_ret': abs_rets.mean(),
                    'mean_ret': rets.mean(),
                    'median_ret': rets.median(),
                    'win_rate': (rets > 0).mean() * 100,
                    'net_fees': abs_rets.mean() - RT_TAKER_BPS,
                })
                
                # Momentum continuation: trade in direction of micro-trend
                if micro_up.sum() > 5:
                    up_rets = merged.loc[micro_up, col].dropna()
                    results.append({
                        'symbol': symbol,
                        'signal': f'coiled_v{vol_pctile_thresh}_oi{oi_thresh}_momup',
                        'direction': 'long',
                        'horizon_m': h,
                        'n_signals': len(up_rets),
                        'mean_abs_ret': up_rets.abs().mean(),
                        'mean_ret': up_rets.mean(),
                        'median_ret': up_rets.median(),
                        'win_rate': (up_rets > 0).mean() * 100,
                        'net_fees': up_rets.mean() - RT_TAKER_BPS,
                    })
                if micro_down.sum() > 5:
                    dn_rets = -merged.loc[micro_down, col].dropna()
                    results.append({
                        'symbol': symbol,
                        'signal': f'coiled_v{vol_pctile_thresh}_oi{oi_thresh}_momdn',
                        'direction': 'short',
                        'horizon_m': h,
                        'n_signals': len(dn_rets),
                        'mean_abs_ret': dn_rets.abs().mean(),
                        'mean_ret': dn_rets.mean(),
                        'median_ret': dn_rets.median(),
                        'win_rate': (dn_rets > 0).mean() * 100,
                        'net_fees': dn_rets.mean() - RT_TAKER_BPS,
                    })
    
    # === Signal B: Vol expansion after compression → momentum ===
    for comp_pctile in [0.15, 0.20]:
        for expand_mult in [2.0, 3.0]:
            # Was compressed, now expanding
            was_compressed = merged['rvol_pctile_48'].shift(12) < comp_pctile
            now_expanding = merged[f'atr_12'] > expand_mult * merged[f'atr_48']
            breakout = was_compressed & now_expanding
            
            if breakout.sum() < 10:
                continue
            
            # Trade in direction of the breakout move
            breakout_up = breakout & (merged['ret_1h_bps'] > 20)
            breakout_down = breakout & (merged['ret_1h_bps'] < -20)
            
            for h in HORIZONS:
                col = f'fwd_{h}m_bps'
                if breakout_up.sum() > 5:
                    rets = merged.loc[breakout_up, col].dropna()
                    results.append({
                        'symbol': symbol,
                        'signal': f'volbreak_c{comp_pctile}_x{expand_mult}_up',
                        'direction': 'long',
                        'horizon_m': h,
                        'n_signals': len(rets),
                        'mean_abs_ret': rets.abs().mean(),
                        'mean_ret': rets.mean(),
                        'median_ret': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'net_fees': rets.mean() - RT_TAKER_BPS,
                    })
                if breakout_down.sum() > 5:
                    rets = -merged.loc[breakout_down, col].dropna()
                    results.append({
                        'symbol': symbol,
                        'signal': f'volbreak_c{comp_pctile}_x{expand_mult}_dn',
                        'direction': 'short',
                        'horizon_m': h,
                        'n_signals': len(rets),
                        'mean_abs_ret': rets.abs().mean(),
                        'mean_ret': rets.mean(),
                        'median_ret': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'net_fees': rets.mean() - RT_TAKER_BPS,
                    })
    
    # === Signal C: OI-Price divergence ===
    # Price going up but OI going down = weak rally → short
    # Price going down but OI going up = accumulation → long
    for price_thresh in [50, 100]:
        for oi_thresh in [2.0, 5.0]:
            # Weak rally: price up, OI down
            weak_rally = (merged['ret_4h_bps'] > price_thresh) & (merged['oi_pct_4h'] < -oi_thresh)
            # Accumulation: price down, OI up
            accum = (merged['ret_4h_bps'] < -price_thresh) & (merged['oi_pct_4h'] > oi_thresh)
            
            for h in HORIZONS:
                col = f'fwd_{h}m_bps'
                if weak_rally.sum() > 5:
                    rets = -merged.loc[weak_rally, col].dropna()
                    results.append({
                        'symbol': symbol,
                        'signal': f'oipricediv_p{price_thresh}_oi{oi_thresh}_weak',
                        'direction': 'short',
                        'horizon_m': h,
                        'n_signals': len(rets),
                        'mean_abs_ret': rets.abs().mean(),
                        'mean_ret': rets.mean(),
                        'median_ret': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'net_fees': rets.mean() - RT_TAKER_BPS,
                    })
                if accum.sum() > 5:
                    rets = merged.loc[accum, col].dropna()
                    results.append({
                        'symbol': symbol,
                        'signal': f'oipricediv_p{price_thresh}_oi{oi_thresh}_accum',
                        'direction': 'long',
                        'horizon_m': h,
                        'n_signals': len(rets),
                        'mean_abs_ret': rets.abs().mean(),
                        'mean_ret': rets.mean(),
                        'median_ret': rets.median(),
                        'win_rate': (rets > 0).mean() * 100,
                        'net_fees': rets.mean() - RT_TAKER_BPS,
                    })
    
    return results


def main():
    print("=" * 80)
    print("IDEA 6: OI REGIME + VOLATILITY BREAKOUT")
    print(f"Period: {START} → {END}")
    print("=" * 80)
    
    symbols = get_symbols(min_days=30)
    print(f"\nTesting {len(symbols)} symbols")
    
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
    df.to_csv(f'{OUT}/idea6_oi_vol_raw.csv', index=False)
    
    agg = df.groupby(['signal', 'direction', 'horizon_m']).agg(
        n_symbols=('symbol', 'nunique'),
        total_sigs=('n_signals', 'sum'),
        mean_ret=('mean_ret', 'mean'),
        mean_wr=('win_rate', 'mean'),
        net_fees=('net_fees', 'mean'),
    ).reset_index().sort_values('net_fees', ascending=False)
    
    agg.to_csv(f'{OUT}/idea6_oi_vol_agg.csv', index=False)
    
    profitable = agg[agg['net_fees'] > 0]
    print(f"\n✅ {len(df)} raw, {len(profitable)} configs profitable after fees (of {len(agg)})")
    
    if len(profitable) > 0:
        print("\nTop 25 profitable configs:")
        for _, r in profitable.head(25).iterrows():
            print(f"  {r['signal']:45s} {r['direction']:10s} {r['horizon_m']:>4.0f}m  "
                  f"net={r['net_fees']:+7.1f}bps  WR={r['mean_wr']:5.1f}%  syms={r['n_symbols']:.0f}")
    
    # Cross-symbol consistency
    print("\n\nCross-symbol consistency (>50% profitable):")
    cnt = 0
    for _, row in profitable.head(30).iterrows():
        s, d, h = row['signal'], row['direction'], row['horizon_m']
        sub = df[(df['signal'] == s) & (df['direction'] == d) & (df['horizon_m'] == h)]
        pct = (sub['net_fees'] > 0).mean() * 100
        if pct > 50:
            print(f"  ✅ {s} {d} {h:.0f}m: {pct:.0f}% of {len(sub)} coins, net {row['net_fees']:+.1f}bps")
            cnt += 1
            if cnt >= 10:
                break
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
