#!/usr/bin/env python3
"""
IDEA 1: OI + L/S Ratio Crowding → Liquidation Cascade
======================================================
Hypothesis: When OI builds up rapidly AND L/S ratio hits extremes,
the market is fragile. A small adverse move triggers cascading liquidations.
Trade: Fade the crowd when OI + L/S reach extremes.

We test multiple signal variants:
 A) L/S ratio extreme alone (buyRatio > 0.70 → short, < 0.30 → long)
 B) OI spike + L/S extreme (OI rising fast + crowded positioning)
 C) OI spike + L/S extreme + price already extended (triple confirmation)

We measure forward returns at 5m, 15m, 30m, 1h, 4h horizons.
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import (
    get_symbols, load_kline, load_oi, load_ls_ratio,
    progress_bar, RT_TAKER_BPS, RT_MAKER_BPS, backtest_signals
)

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'

# Test parameters
START = '2025-06-01'
END = '2026-03-04'
HORIZONS = [5, 15, 30, 60, 240]  # minutes
LS_THRESHOLDS = [0.65, 0.68, 0.70, 0.72, 0.75]
OI_LOOKBACK = 60  # 5min bars = 5 hours
OI_SPIKE_MULT = [1.5, 2.0, 2.5, 3.0]  # OI change > X std devs


def resample_kline_to_5m(kline_df):
    """Resample 1m kline to 5m to align with OI/LS data."""
    if kline_df.empty:
        return pd.DataFrame()
    df = kline_df.set_index('ts').sort_index()
    r = df.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    if 'volume' in df.columns:
        r['volume'] = df['volume'].resample('5min').sum()
    r = r.reset_index()
    return r


def compute_forward_returns(prices_5m, horizons_min):
    """Compute forward returns at various horizons (in 5m bars)."""
    ret = {}
    for h in horizons_min:
        bars = h // 5
        ret[f'fwd_{h}m_bps'] = (prices_5m.shift(-bars) / prices_5m - 1) * 10000
    return pd.DataFrame(ret, index=prices_5m.index)


def analyze_symbol(symbol, start, end):
    """Analyze OI+LS crowding signals for a single symbol."""
    kline = load_kline(symbol, start, end)
    oi = load_oi(symbol, start, end)
    ls = load_ls_ratio(symbol, start, end)
    
    if kline.empty or oi.empty or ls.empty:
        return None
    if len(oi) < OI_LOOKBACK + 10:
        return None
    
    # Resample kline to 5m
    k5 = resample_kline_to_5m(kline)
    if k5.empty:
        return None
    
    # Merge OI and LS on nearest timestamp
    oi = oi[['ts', 'openInterest']].copy()
    ls = ls[['ts', 'buyRatio', 'sellRatio']].copy()
    
    # Merge all on 5m timestamps
    merged = k5[['ts', 'close']].copy()
    if 'volume' in k5.columns:
        merged['volume'] = k5['volume']
    
    merged = pd.merge_asof(merged.sort_values('ts'), oi.sort_values('ts'),
                           on='ts', direction='backward')
    merged = pd.merge_asof(merged.sort_values('ts'), ls.sort_values('ts'),
                           on='ts', direction='backward')
    merged = merged.dropna(subset=['close', 'openInterest', 'buyRatio'])
    
    if len(merged) < OI_LOOKBACK + 50:
        return None
    
    # Compute OI features
    merged['oi_pct_change'] = merged['openInterest'].pct_change(12)  # 1h OI change
    merged['oi_roll_mean'] = merged['oi_pct_change'].rolling(OI_LOOKBACK).mean()
    merged['oi_roll_std'] = merged['oi_pct_change'].rolling(OI_LOOKBACK).std()
    merged['oi_zscore'] = (merged['oi_pct_change'] - merged['oi_roll_mean']) / merged['oi_roll_std'].clip(lower=1e-8)
    
    # Compute price momentum (5h = 60 bars of 5min)
    merged['price_ret_5h_bps'] = (merged['close'] / merged['close'].shift(60) - 1) * 10000
    
    # Forward returns
    fwd = compute_forward_returns(merged['close'], HORIZONS)
    merged = pd.concat([merged, fwd], axis=1)
    merged = merged.dropna()
    
    if len(merged) < 100:
        return None
    
    results = []
    
    for ls_thresh in LS_THRESHOLDS:
        # Signal A: LS extreme alone
        long_sig_a = merged['buyRatio'] < (1 - ls_thresh)
        short_sig_a = merged['buyRatio'] > ls_thresh
        
        for oi_mult in [None] + OI_SPIKE_MULT:
            for require_extended in [False, True]:
                if oi_mult is not None:
                    oi_cond = merged['oi_zscore'] > oi_mult
                    long_sig = long_sig_a & oi_cond
                    short_sig = short_sig_a & oi_cond
                    variant = f'LS>{ls_thresh}_OI>{oi_mult}σ'
                else:
                    long_sig = long_sig_a
                    short_sig = short_sig_a
                    variant = f'LS>{ls_thresh}_noOI'
                
                if require_extended:
                    # Price already extended in crowd direction (contrarian)
                    long_sig = long_sig & (merged['price_ret_5h_bps'] < -50)
                    short_sig = short_sig & (merged['price_ret_5h_bps'] > 50)
                    variant += '_extended'
                
                n_long = long_sig.sum()
                n_short = short_sig.sum()
                
                if n_long + n_short < 5:
                    continue
                
                for h in HORIZONS:
                    col = f'fwd_{h}m_bps'
                    
                    # Long signals: we expect price to go UP (fade shorts)
                    if n_long > 0:
                        long_rets = merged.loc[long_sig, col]
                        results.append({
                            'symbol': symbol,
                            'variant': variant,
                            'direction': 'long',
                            'horizon_m': h,
                            'n_signals': int(n_long),
                            'mean_ret_bps': long_rets.mean(),
                            'median_ret_bps': long_rets.median(),
                            'win_rate': (long_rets > 0).mean() * 100,
                            'sharpe': long_rets.mean() / max(long_rets.std(), 0.01),
                            'net_of_fees_bps': long_rets.mean() - RT_TAKER_BPS,
                        })
                    
                    # Short signals: we expect price to go DOWN (fade longs)
                    if n_short > 0:
                        short_rets = -merged.loc[short_sig, col]  # negate for short
                        results.append({
                            'symbol': symbol,
                            'variant': variant,
                            'direction': 'short',
                            'horizon_m': h,
                            'n_signals': int(n_short),
                            'mean_ret_bps': short_rets.mean(),
                            'median_ret_bps': short_rets.median(),
                            'win_rate': (short_rets > 0).mean() * 100,
                            'sharpe': short_rets.mean() / max(short_rets.std(), 0.01),
                            'net_of_fees_bps': short_rets.mean() - RT_TAKER_BPS,
                        })
    
    return results


def main():
    print("=" * 80)
    print("IDEA 1: OI + L/S CROWDING → LIQUIDATION CASCADE")
    print(f"Period: {START} → {END}")
    print(f"Fees: taker {RT_TAKER_BPS} bps RT, maker {RT_MAKER_BPS} bps RT")
    print("=" * 80)
    
    symbols = get_symbols(min_days=30)
    print(f"\nFound {len(symbols)} symbols with 30+ days data")
    print(f"Testing {len(LS_THRESHOLDS)} LS thresholds × {len(OI_SPIKE_MULT)+1} OI variants × 2 extended × {len(HORIZONS)} horizons")
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
        print("\n❌ No results found!")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT}/idea1_oi_crowding_raw.csv', index=False)
    print(f"\n✅ {len(df)} raw signal results saved")
    
    # ==================== ANALYSIS ====================
    print("\n" + "=" * 80)
    print("ANALYSIS: Best signal variants (net of fees)")
    print("=" * 80)
    
    # Aggregate across symbols
    agg = df.groupby(['variant', 'direction', 'horizon_m']).agg(
        n_symbols=('symbol', 'nunique'),
        total_signals=('n_signals', 'sum'),
        mean_ret_bps=('mean_ret_bps', 'mean'),
        median_ret_bps=('median_ret_bps', 'mean'),
        mean_wr=('win_rate', 'mean'),
        mean_sharpe=('sharpe', 'mean'),
        net_fees_bps=('net_of_fees_bps', 'mean'),
    ).reset_index()
    
    agg = agg.sort_values('net_fees_bps', ascending=False)
    agg.to_csv(f'{OUT}/idea1_oi_crowding_agg.csv', index=False)
    
    # Show top 30 profitable configs
    profitable = agg[agg['net_fees_bps'] > 0]
    if len(profitable) > 0:
        print(f"\n🟢 {len(profitable)} configs profitable after fees:")
        print(profitable.head(30).to_string(index=False, float_format='%.1f'))
    else:
        print("\n🔴 No configs profitable after fees")
    
    # Show best by horizon
    print("\n\nBest config per horizon:")
    for h in HORIZONS:
        sub = agg[agg['horizon_m'] == h].head(1)
        if len(sub) > 0:
            row = sub.iloc[0]
            print(f"  {h:>3}m: {row['variant']:40s} {row['direction']:5s} "
                  f"net={row['net_fees_bps']:+.1f}bps WR={row['mean_wr']:.0f}% "
                  f"n={row['total_signals']:.0f} syms={row['n_symbols']:.0f}")
    
    # Cross-symbol consistency
    print("\n\nCross-symbol consistency (configs where >60% of symbols profitable):")
    for _, row in profitable.head(20).iterrows():
        v, d, h = row['variant'], row['direction'], row['horizon_m']
        sub = df[(df['variant'] == v) & (df['direction'] == d) & (df['horizon_m'] == h)]
        pct_profitable = (sub['net_of_fees_bps'] > 0).mean() * 100
        if pct_profitable > 60:
            print(f"  ✅ {v} {d} {h}m: {pct_profitable:.0f}% of {len(sub)} symbols profitable, "
                  f"avg net {row['net_fees_bps']:+.1f} bps")
    
    elapsed = time.time() - t0
    print(f"\n⏱ Total time: {elapsed:.0f}s")
    print(f"Results saved to {OUT}/idea1_oi_crowding_*.csv")


if __name__ == '__main__':
    main()
