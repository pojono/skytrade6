#!/usr/bin/env python3
"""
ALL SIGNALS + REGIME FILTER: Comprehensive daily test
======================================================
Tests ALL signals across full 2024-01 → 2026-03 period, both exchanges.
Then applies walk-forward regime filters to each.

Signals tested:
 1. Idea 1: L/S crowding → fade crowd (long only, 5m bars)
 2. Idea 3: High implied FR (premium > threshold) → momentum
 3. Idea 6: Vol compression + OI rising → breakout with micro-trend
 4. Idea 4: BTC pump > 150 bps → long alts (confirmed real)
 5. Idea 5: Spot leads futures > 40 bps → long (regime-filtered)
 6. NEW: Cross-exchange price divergence (Bybit vs Binance)
"""
import sys, os
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from pathlib import Path
from data_loader import (load_kline, load_oi, load_ls_ratio, load_premium,
                         load_funding_rate, progress_bar, RT_TAKER_BPS)

BYBIT = Path("/home/ubuntu/Projects/skytrade6/datalake/bybit")
BINANCE = Path("/home/ubuntu/Projects/skytrade6/datalake/binance")
OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'
START = '2024-01-01'
END = '2026-03-04'
np.random.seed(42)

ALTS = [
    'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
    'LINKUSDT', 'ADAUSDT', 'APTUSDT', 'ARBUSDT', 'NEARUSDT',
    'ATOMUSDT', 'DOTUSDT', 'OPUSDT', 'INJUSDT', 'LTCUSDT',
    'BCHUSDT', 'FILUSDT', 'AAVEUSDT', 'BNBUSDT', 'MKRUSDT',
    'SUIUSDT', 'TIAUSDT', 'SEIUSDT', 'WIFUSDT', 'MATICUSDT',
]

MIN_GAP = 30  # 30 bars declustering (1m)
MIN_GAP_5M = 6  # 30 min in 5m bars


def decluster_idx(indices, gap):
    if len(indices) == 0:
        return []
    kept = [indices[0]]
    for idx in indices[1:]:
        if idx - kept[-1] >= gap:
            kept.append(idx)
    return kept


def _load_binance_kline(sym, start=START, end=END):
    """Load Binance futures kline."""
    import re
    sym_dir = BINANCE / sym
    if not sym_dir.exists():
        return pd.DataFrame()
    pat = re.compile(r'^\d{4}-\d{2}-\d{2}_kline_1m\.csv$')
    files = sorted(f for f in os.listdir(sym_dir) if pat.match(f))
    files = [str(sym_dir / f) for f in files if f[:10] >= start and f[:10] <= end]
    if not files:
        return pd.DataFrame()
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                chunks.append(df)
        except:
            continue
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    if 'open_time' in df.columns:
        df['ts'] = pd.to_datetime(df['open_time'], unit='ms')
    elif 'startTime' in df.columns:
        df['ts'] = pd.to_datetime(df['startTime'], unit='ms')
    df = df.sort_values('ts').drop_duplicates('ts', keep='first')
    return df[['ts', 'close']].set_index('ts') if 'close' in df.columns else pd.DataFrame()


# ============================================================
# SIGNAL GENERATORS
# ============================================================

def signals_idea1(sym, k1m, start, end):
    """Idea 1: L/S crowding → fade crowd (long only).
    Uses 5m bars for OI/LS alignment. Returns list of (timestamp, fwd_ret)."""
    oi = load_oi(sym, start, end)
    ls = load_ls_ratio(sym, start, end)
    if oi.empty or ls.empty or len(oi) < 100:
        return []

    # Resample to 5m
    k5 = k1m.resample('5min').agg({'close': 'last'}).dropna()
    k5['fwd_240m'] = (k5['close'].shift(-48) / k5['close'] - 1) * 10000  # 240m = 48 5m-bars
    k5['dfwd_240m'] = k5['fwd_240m'].shift(-1)  # T+1 entry

    oi_s = oi[['ts', 'openInterest']].set_index('ts').sort_index()
    oi_s = oi_s[~oi_s.index.duplicated(keep='first')]
    ls_s = ls[['ts', 'buyRatio']].set_index('ts').sort_index()
    ls_s = ls_s[~ls_s.index.duplicated(keep='first')]

    merged = k5.join(oi_s, how='left').join(ls_s, how='left').dropna()
    if len(merged) < 200:
        return []

    # OI z-score
    merged['oi_pct'] = merged['openInterest'].pct_change(12)
    merged['oi_zmean'] = merged['oi_pct'].rolling(60).mean()
    merged['oi_zstd'] = merged['oi_pct'].rolling(60).std()
    merged['oi_z'] = (merged['oi_pct'] - merged['oi_zmean']) / merged['oi_zstd'].clip(lower=1e-8)
    merged = merged.dropna()

    # Signal: crowd heavily short (buyRatio < 0.32) + OI spike > 2σ → LONG
    sig = (merged['buyRatio'] < 0.32) & (merged['oi_z'] > 2.0)
    indices = np.where(sig.values)[0]
    kept = decluster_idx(indices, MIN_GAP_5M)

    results = []
    for i in kept:
        ret = merged['dfwd_240m'].iloc[i]
        if not np.isnan(ret):
            results.append((merged.index[i], ret))
    return results


def signals_idea3(sym, k1m, start, end):
    """Idea 3: High implied FR → momentum. Uses premium index as proxy."""
    prem = load_premium(sym, start, end)
    if prem.empty or len(prem) < 5000:
        return []

    prem_s = prem[['ts', 'close']].set_index('ts').sort_index()
    prem_s = prem_s[~prem_s.index.duplicated(keep='first')]
    prem_s.columns = ['premium']
    prem_s['premium_bps'] = prem_s['premium'] * 10000

    merged = k1m[['close']].join(prem_s[['premium_bps']], how='inner')
    if len(merged) < 5000:
        return []

    merged['fwd_240m'] = (merged['close'].shift(-240) / merged['close'] - 1) * 10000
    merged['dfwd_240m'] = merged['fwd_240m'].shift(-1)

    # Signal: implied FR > 20 bps → long (momentum)
    sig = merged['premium_bps'] > 20
    indices = np.where(sig.values)[0]
    kept = decluster_idx(indices, MIN_GAP)

    results = []
    for i in kept:
        ret = merged['dfwd_240m'].iloc[i]
        if not np.isnan(ret):
            results.append((merged.index[i], ret))
    return results


def signals_idea6(sym, k1m, start, end):
    """Idea 6: Vol compression + OI rising → directional breakout with micro-trend."""
    oi = load_oi(sym, start, end)
    if oi.empty or len(oi) < 200:
        return []

    # 5m bars
    k5 = k1m.resample('5min').agg({
        'close': 'last', 'high': 'max', 'low': 'min'
    }).dropna()

    oi_s = oi[['ts', 'openInterest']].set_index('ts').sort_index()
    oi_s = oi_s[~oi_s.index.duplicated(keep='first')]
    merged = k5.join(oi_s, how='inner').dropna(subset=['close', 'openInterest'])
    if len(merged) < 1000:
        return []

    # Realized vol
    merged['ret_5m'] = merged['close'].pct_change()
    merged['rvol_48'] = merged['ret_5m'].rolling(48).std() * np.sqrt(288) * 100
    merged['rvol_pctile'] = merged['rvol_48'].rolling(288).rank(pct=True)

    # OI growth
    merged['oi_pct_4h'] = merged['openInterest'].pct_change(48) * 100

    # Micro-trend
    merged['ret_1h_bps'] = (merged['close'] / merged['close'].shift(12) - 1) * 10000

    # Forward return
    merged['fwd_240m'] = (merged['close'].shift(-48) / merged['close'] - 1) * 10000
    merged['dfwd_240m'] = merged['fwd_240m'].shift(-1)
    merged = merged.dropna()

    # Coiled spring + micro-trend up → LONG
    sig_long = (merged['rvol_pctile'] < 0.15) & (merged['oi_pct_4h'] > 2.0) & (merged['ret_1h_bps'] > 10)
    # Coiled spring + micro-trend down → SHORT (use negative return)
    sig_short = (merged['rvol_pctile'] < 0.15) & (merged['oi_pct_4h'] > 2.0) & (merged['ret_1h_bps'] < -10)

    results = []
    for sig, direction in [(sig_long, 1), (sig_short, -1)]:
        indices = np.where(sig.values)[0]
        kept = decluster_idx(indices, MIN_GAP_5M)
        for i in kept:
            ret = merged['dfwd_240m'].iloc[i] * direction
            if not np.isnan(ret):
                results.append((merged.index[i], ret))
    return results


def signals_idea4(sym, k1m, btc_1m):
    """Idea 4: BTC pump > 150 bps in 3m → long alt."""
    merged = k1m[['close']].join(btc_1m[['btc_ret_3m']], how='inner')
    if len(merged) < 5000:
        return []

    merged['fwd_240m'] = (merged['close'].shift(-240) / merged['close'] - 1) * 10000
    merged['dfwd_240m'] = merged['fwd_240m'].shift(-1)

    sig = merged['btc_ret_3m'] > 150
    indices = np.where(sig.values)[0]
    kept = decluster_idx(indices, MIN_GAP)

    results = []
    for i in kept:
        ret = merged['dfwd_240m'].iloc[i]
        if not np.isnan(ret):
            results.append((merged.index[i], ret))
    return results


def signals_idea5(sym, k1m, start, end):
    """Idea 5: Spot leads futures > 40 bps → long."""
    ks = load_kline(sym, start, end, spot=True)
    if ks.empty or len(ks) < 5000:
        return []

    ks_s = ks[['ts', 'close']].set_index('ts').sort_index()
    ks_s = ks_s[~ks_s.index.duplicated(keep='first')]
    ks_s.columns = ['spot_close']

    merged = k1m[['close']].join(ks_s, how='left')
    if 'spot_close' not in merged.columns:
        return []
    merged['spot_ret_3m'] = (merged['spot_close'] / merged['spot_close'].shift(3) - 1) * 10000
    merged['fut_ret_3m'] = (merged['close'] / merged['close'].shift(3) - 1) * 10000
    merged['fwd_240m'] = (merged['close'].shift(-240) / merged['close'] - 1) * 10000
    merged['dfwd_240m'] = merged['fwd_240m'].shift(-1)

    sig = ((merged['spot_ret_3m'] - merged['fut_ret_3m']) > 40).fillna(False)
    indices = np.where(sig.values)[0]
    kept = decluster_idx(indices, MIN_GAP)

    results = []
    for i in kept:
        ret = merged['dfwd_240m'].iloc[i]
        if not np.isnan(ret):
            results.append((merged.index[i], ret))
    return results


def signals_xexch(sym, k1m_bybit, start, end):
    """NEW: Cross-exchange divergence. Bybit price > Binance price → short Bybit (mean reversion)."""
    k1m_bn = _load_binance_kline(sym, start, end)
    if k1m_bn.empty or len(k1m_bn) < 5000:
        return []

    k1m_bn.columns = ['bn_close']
    merged = k1m_bybit[['close']].join(k1m_bn, how='inner')
    if len(merged) < 5000:
        return []

    # Spread in bps
    merged['spread_bps'] = (merged['close'] / merged['bn_close'] - 1) * 10000
    merged['spread_z'] = (merged['spread_bps'] - merged['spread_bps'].rolling(60).mean()) / \
                          merged['spread_bps'].rolling(60).std().clip(lower=0.01)

    merged['fwd_60m'] = (merged['close'].shift(-60) / merged['close'] - 1) * 10000
    merged['dfwd_60m'] = merged['fwd_60m'].shift(-1)

    # Bybit expensive → short (fade), Bybit cheap → long
    sig_short = merged['spread_z'] > 3.0  # Bybit expensive
    sig_long = merged['spread_z'] < -3.0  # Bybit cheap

    results = []
    for sig, direction in [(sig_long, 1), (sig_short, -1)]:
        indices = np.where(sig.values)[0]
        kept = decluster_idx(indices, MIN_GAP)
        for i in kept:
            ret = merged['dfwd_60m'].iloc[i] * direction
            if not np.isnan(ret):
                results.append((merged.index[i], ret))
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("  ALL SIGNALS + REGIME FILTER")
    print(f"  Period: {START} → {END} | Coins: {len(ALTS)}")
    print("=" * 75)

    # Load BTC
    print("\n  Loading BTC...")
    btc = load_kline('BTCUSDT', START, END)
    btc_1m = btc[['ts', 'close']].set_index('ts').sort_index()
    btc_1m = btc_1m[~btc_1m.index.duplicated(keep='first')]
    btc_1m['btc_ret_3m'] = (btc_1m['close'] / btc_1m['close'].shift(3) - 1) * 10000

    daily_results = {}  # (date, signal_name) → [rets]

    t0 = time.time()
    for ai, sym in enumerate(ALTS):
        progress_bar(ai, len(ALTS), prefix='  Processing', start_time=t0)

        kf = load_kline(sym, START, END)
        if kf.empty or len(kf) < 5000:
            continue
        k1m = kf[['ts', 'close', 'high', 'low']].set_index('ts').sort_index()
        k1m = k1m[~k1m.index.duplicated(keep='first')]

        # Run all signals
        for sig_name, sig_func in [
            ('idea1_crowding', lambda: signals_idea1(sym, k1m, START, END)),
            ('idea3_ifr', lambda: signals_idea3(sym, k1m, START, END)),
            ('idea6_coiled', lambda: signals_idea6(sym, k1m, START, END)),
            ('idea4_btcpump', lambda: signals_idea4(sym, k1m, btc_1m)),
            ('idea5_spotlead', lambda: signals_idea5(sym, k1m, START, END)),
            ('xexch_diverge', lambda: signals_xexch(sym, k1m, START, END)),
        ]:
            try:
                trades = sig_func()
            except Exception as e:
                continue

            for ts, ret in trades:
                dt = ts.date()
                key = (dt, sig_name)
                if key not in daily_results:
                    daily_results[key] = []
                daily_results[key].append(ret)

    progress_bar(len(ALTS), len(ALTS), prefix='  Processing', start_time=t0)

    # Aggregate to daily
    rows = []
    for (dt, sig), rets in daily_results.items():
        rows.append({
            'date': pd.Timestamp(dt),
            'signal': sig,
            'n_trades': len(rets),
            'mean_ret': np.mean(rets),
            'net_ret': np.mean(rets) - RT_TAKER_BPS,
            'win_rate': np.mean([r > 0 for r in rets]) * 100,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/all_signals_daily.csv', index=False)

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print(f"\n\n{'='*75}")
    print("  ALL SIGNALS SUMMARY (daily-aggregated, T+1 entry, declustered)")
    print(f"{'='*75}")
    print(f"\n  {'Signal':<20s} │ {'Days':>5s} │ {'Net bps':>8s} │ {'WR':>5s} │ {'Prof%':>6s} │ {'2024':>8s} │ {'2025':>8s} │ {'2026':>8s}")
    print(f"  {'─'*20}─┼─{'─'*5}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")

    signal_order = ['idea4_btcpump', 'idea5_spotlead', 'idea1_crowding',
                    'idea3_ifr', 'idea6_coiled', 'xexch_diverge']

    for sig in signal_order:
        sub = df[df['signal'] == sig]
        if len(sub) == 0:
            continue
        sub2 = sub.copy()
        sub2['year'] = sub2['date'].dt.year

        net = sub['net_ret'].mean()
        wr = sub['win_rate'].mean()
        prof = (sub['net_ret'] > 0).mean() * 100
        n = len(sub)

        yr_stats = {}
        for yr in [2024, 2025, 2026]:
            g = sub2[sub2['year'] == yr]
            yr_stats[yr] = f"{g['net_ret'].mean():+7.0f}" if len(g) > 0 else "    N/A"

        print(f"  {sig:<20s} │ {n:>5d} │ {net:>+7.0f}  │ {wr:>4.0f}% │ {prof:>5.0f}% │ {yr_stats[2024]} │ {yr_stats[2025]} │ {yr_stats[2026]}")

    # ============================================================
    # REGIME FILTER for each signal
    # ============================================================
    print(f"\n\n{'='*75}")
    print("  REGIME FILTER (walk-forward, expanding, min 180d)")
    print(f"{'='*75}")

    # Load pre-computed features
    feat_path = f'{OUT}/regime_v2_features.csv'
    if os.path.exists(feat_path):
        features = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    else:
        print("  ⚠️ No regime features found. Run regime_v2.py first.")
        return

    for sig in signal_order:
        sub = df[df['signal'] == sig].set_index('date').sort_index()
        if len(sub) < 30:
            print(f"\n  {sig}: only {len(sub)} days — skipping regime filter")
            continue

        merged = sub.join(features, how='inner')
        feat_cols = [c for c in features.columns if c in merged.columns]
        if len(merged) < 30:
            continue

        overall_net = merged['net_ret'].mean()
        print(f"\n  {'─'*70}")
        print(f"  {sig}: {len(merged)} days, overall avg={overall_net:+.0f} bps")

        # Test top features
        best_lift = -9999
        best_name = None
        best_stats = None
        min_train = 180

        # Score features by correlation
        scored = []
        for feat_name in feat_cols:
            valid = merged[['net_ret', feat_name]].dropna()
            if len(valid) < 30:
                continue
            corr = valid['net_ret'].corr(valid[feat_name])
            scored.append((feat_name, corr))
        scored.sort(key=lambda x: abs(x[1]), reverse=True)

        configs = [(f, c) for f, c in scored[:7]]
        # Add pairs of top 3
        for i in range(min(3, len(scored))):
            for j in range(i+1, min(5, len(scored))):
                configs.append((f"{scored[i][0]}+{scored[j][0]}", (scored[i], scored[j])))

        for cfg in configs:
            if isinstance(cfg[1], tuple):
                # Pair
                cfg_name = cfg[0]
                (f1, _), (f2, _) = cfg[1]
                if f1 not in merged.columns or f2 not in merged.columns:
                    continue
            else:
                cfg_name = cfg[0]
                f1 = cfg[0]
                f2 = None

            wf_trades = []
            wf_notrades = []

            for di in range(min_train, len(merged)):
                train = merged.iloc[:di]
                test_row = merged.iloc[di]

                if f2 is None:
                    if pd.isna(test_row.get(f1)):
                        continue
                    tv = train[[f1, 'net_ret']].dropna()
                    if len(tv) < 20:
                        continue
                    med = tv[f1].median()
                    above = tv[tv[f1] > med]['net_ret'].mean()
                    below = tv[tv[f1] <= med]['net_ret'].mean()
                    should_trade = test_row[f1] > med if above > below else test_row[f1] <= med
                else:
                    if pd.isna(test_row.get(f1)) or pd.isna(test_row.get(f2)):
                        continue
                    tv = train[[f1, f2, 'net_ret']].dropna()
                    if len(tv) < 20:
                        continue
                    m1, m2 = tv[f1].median(), tv[f2].median()
                    a1 = tv[tv[f1] > m1]['net_ret'].mean()
                    b1 = tv[tv[f1] <= m1]['net_ret'].mean()
                    a2 = tv[tv[f2] > m2]['net_ret'].mean()
                    b2 = tv[tv[f2] <= m2]['net_ret'].mean()
                    c1 = test_row[f1] > m1 if a1 > b1 else test_row[f1] <= m1
                    c2 = test_row[f2] > m2 if a2 > b2 else test_row[f2] <= m2
                    should_trade = c1 and c2

                actual = test_row['net_ret']
                if should_trade:
                    wf_trades.append(actual)
                else:
                    wf_notrades.append(actual)

            if len(wf_trades) < 5 or len(wf_notrades) < 3:
                continue

            t_avg = np.mean(wf_trades)
            nt_avg = np.mean(wf_notrades)
            lift = t_avg - nt_avg

            if lift > best_lift and t_avg > 0:
                best_lift = lift
                best_name = cfg_name
                best_stats = {
                    'trade_n': len(wf_trades), 'trade_avg': t_avg,
                    'trade_wr': np.mean([r > 0 for r in wf_trades]) * 100,
                    'notrade_n': len(wf_notrades), 'notrade_avg': nt_avg,
                    'lift': lift,
                }

        if best_stats and best_stats['trade_avg'] > overall_net:
            print(f"  ★ Best filter: {best_name}")
            print(f"    TRADE:    {best_stats['trade_n']:>4d} days, avg={best_stats['trade_avg']:>+7.0f} bps, WR={best_stats['trade_wr']:.0f}%")
            print(f"    NO-TRADE: {best_stats['notrade_n']:>4d} days, avg={best_stats['notrade_avg']:>+7.0f} bps")
            print(f"    Lift: {best_stats['lift']:+.0f} bps | Always: {overall_net:+.0f} → Filtered: {best_stats['trade_avg']:+.0f}")
        else:
            if overall_net > 0:
                print(f"  → No filter improves on always-trade ({overall_net:+.0f} bps). Use unconditionally.")
            else:
                print(f"  → No viable filter found. Signal avg={overall_net:+.0f} bps.")

    # ============================================================
    # FINAL TABLE
    # ============================================================
    print(f"\n\n{'='*75}")
    print("  FINAL RANKING")
    print(f"{'='*75}")
    for sig in signal_order:
        sub = df[df['signal'] == sig]
        if len(sub) == 0:
            continue
        net = sub['net_ret'].mean()
        n = len(sub)
        prof = (sub['net_ret'] > 0).mean() * 100
        status = '✅' if net > 20 and prof > 50 else '⚠️' if net > 0 else '❌'
        print(f"  {status} {sig:<20s}: {net:>+7.0f} bps avg, {n} days, {prof:.0f}% profitable")

    print(f"\n✅ Saved: all_signals_daily.csv")
    print(f"⏱ Total: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
