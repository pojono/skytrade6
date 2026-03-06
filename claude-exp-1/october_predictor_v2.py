#!/usr/bin/env python3
"""
October Predictor v2: Activation signal for vol-explosion regime
================================================================
From v1 analysis, we learned the strategy profits during:
  - High vol (rvol surging, not compressing)
  - OI deleveraging (liquidation cascades)
  - Recent price drop (mom_7d < 0)
  - Volume surging (vol_ratio > 1)
  - Positive autocorrelation (trending, catching the bounce)

This script:
1. Computes daily BTC macro features (2022-2026)
2. Builds activation signal based on correct conditions
3. Maps to actual strategy trades
4. Tests multiple thresholds
5. Scans for "pre-event" conditions that PRECEDE the activation regime
"""
import sys, os, warnings
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from pathlib import Path
from data_loader import load_csv_daterange

BYBIT = Path("/home/ubuntu/Projects/skytrade6/datalake/bybit")
OUT = '/home/ubuntu/Projects/skytrade6/claude-exp-1'


def load_btc_full():
    """Load BTC kline, OI, funding, LS ratio from 2022 onwards."""
    print("Loading BTC data...")
    t0 = time.time()

    kline = load_csv_daterange("BTCUSDT", "kline_1m", "2022-01-01", "2026-03-05")
    if 'timestamp' in kline.columns:
        kline['dt'] = pd.to_datetime(kline['timestamp'], unit='ms', utc=True)
    elif 'open_time' in kline.columns:
        kline['dt'] = pd.to_datetime(kline['open_time'], unit='ms', utc=True)
    else:
        kline['dt'] = pd.to_datetime(kline.iloc[:, 0], unit='ms', utc=True)
    kline = kline.set_index('dt').sort_index()

    ohlcv = kline.resample('1h').agg({
        kline.columns[1]: 'first',
        kline.columns[2]: 'max',
        kline.columns[3]: 'min',
        kline.columns[4]: 'last',
        kline.columns[5]: 'sum',
    }).dropna()
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    print(f"  1h bars: {len(ohlcv)} ({ohlcv.index.min().date()} to {ohlcv.index.max().date()})")

    oi = load_csv_daterange("BTCUSDT", "open_interest_5min", "2022-01-01", "2026-03-05")
    if not oi.empty:
        if 'timestamp' in oi.columns:
            oi['dt'] = pd.to_datetime(oi['timestamp'], unit='ms', utc=True)
        else:
            oi['dt'] = pd.to_datetime(oi.iloc[:, 0], unit='ms', utc=True)
        oi = oi.set_index('dt').sort_index()
        oi_col = [c for c in oi.columns if 'interest' in c.lower() or 'oi' in c.lower() or 'value' in c.lower()]
        oi_1h = (oi[oi_col[0]] if oi_col else oi.iloc[:, 0].astype(float)).resample('1h').last().dropna()
        print(f"  OI: {len(oi_1h)} hourly points")
    else:
        oi_1h = pd.Series(dtype=float)

    fr = load_csv_daterange("BTCUSDT", "funding_rate", "2022-01-01", "2026-03-05")
    if not fr.empty:
        if 'timestamp' in fr.columns:
            fr['dt'] = pd.to_datetime(fr['timestamp'], unit='ms', utc=True)
        elif 'funding_rate_timestamp' in fr.columns:
            fr['dt'] = pd.to_datetime(fr['funding_rate_timestamp'], unit='ms', utc=True)
        else:
            fr['dt'] = pd.to_datetime(fr.iloc[:, 0], unit='ms', utc=True)
        fr = fr.set_index('dt').sort_index()
        fr_col = [c for c in fr.columns if 'funding' in c.lower() and 'rate' in c.lower()]
        fr_series = (fr[fr_col[0]] if fr_col else fr.iloc[:, -1]).astype(float)
        print(f"  Funding: {len(fr_series)} points")
    else:
        fr_series = pd.Series(dtype=float)

    print(f"  Loaded in {time.time()-t0:.0f}s")
    return ohlcv, oi_1h, fr_series


def compute_features(ohlcv, oi_1h, fr_series):
    """Compute daily features focused on vol-explosion detection."""
    print("\nComputing features...")

    daily = ohlcv.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    feat = pd.DataFrame(index=daily.index)
    feat['price'] = daily['close']
    feat['ret_1d'] = daily['close'].pct_change()

    # --- VOLATILITY (key: is vol SURGING?) ---
    feat['rvol_7d'] = feat['ret_1d'].rolling(7).std() * np.sqrt(365) * 100
    feat['rvol_30d'] = feat['ret_1d'].rolling(30).std() * np.sqrt(365) * 100
    feat['rvol_90d'] = feat['ret_1d'].rolling(90).std() * np.sqrt(365) * 100
    feat['vol_ratio_7_30'] = feat['rvol_7d'] / feat['rvol_30d']  # >1 = vol expanding
    feat['vol_ratio_7_90'] = feat['rvol_7d'] / feat['rvol_90d']

    # Vol percentile (expanding, causal)
    feat['rvol_7d_pct'] = feat['rvol_7d'].expanding(90).rank(pct=True)

    # --- PRICE MOVEMENT ---
    feat['mom_3d'] = daily['close'].pct_change(3) * 100
    feat['mom_7d'] = daily['close'].pct_change(7) * 100
    feat['mom_14d'] = daily['close'].pct_change(14) * 100
    feat['drawdown_from_7d_high'] = (daily['close'] / daily['high'].rolling(7).max() - 1) * 100
    feat['drawdown_from_14d_high'] = (daily['close'] / daily['high'].rolling(14).max() - 1) * 100
    feat['drawdown_from_30d_high'] = (daily['close'] / daily['high'].rolling(30).max() - 1) * 100

    # --- VOLUME ---
    feat['vol_ma7'] = daily['volume'].rolling(7).mean()
    feat['vol_ma30'] = daily['volume'].rolling(30).mean()
    feat['volume_surge'] = feat['vol_ma7'] / feat['vol_ma30']

    # --- OI (key: deleveraging) ---
    if len(oi_1h) > 100:
        oi_daily = oi_1h.resample('1D').last().reindex(feat.index, method='ffill')
        feat['oi'] = oi_daily
        feat['oi_chg_3d'] = oi_daily.pct_change(3) * 100
        feat['oi_chg_7d'] = oi_daily.pct_change(7) * 100
        feat['oi_chg_14d'] = oi_daily.pct_change(14) * 100
        feat['oi_vs_ma30'] = (oi_daily / oi_daily.rolling(30).mean() - 1) * 100
    else:
        for c in ['oi', 'oi_chg_3d', 'oi_chg_7d', 'oi_chg_14d', 'oi_vs_ma30']:
            feat[c] = np.nan

    # --- AUTOCORRELATION ---
    feat['autocorr_7d'] = feat['ret_1d'].rolling(14).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False)

    # --- RANGE ---
    tr = pd.concat([
        daily['high'] - daily['low'],
        (daily['high'] - daily['close'].shift(1)).abs(),
        (daily['low'] - daily['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_pct_7d'] = tr.rolling(7).mean() / daily['close'] * 100
    feat['atr_pct_30d'] = tr.rolling(30).mean() / daily['close'] * 100
    feat['atr_expansion'] = feat['atr_pct_7d'] / feat['atr_pct_30d']

    # --- FUNDING ---
    if len(fr_series) > 100:
        fr_daily = fr_series.resample('1D').mean().reindex(feat.index, method='ffill')
        feat['fr_daily'] = fr_daily * 100
        feat['fr_cum_7d'] = fr_daily.rolling(7).sum() * 100
    else:
        feat['fr_daily'] = np.nan
        feat['fr_cum_7d'] = np.nan

    feat = feat.dropna(subset=['rvol_90d'])
    print(f"  {len(feat)} daily feature rows ({feat.index.min().date()} to {feat.index.max().date()})")
    return feat


def test_activation_signals(feat):
    """Test multiple activation signal definitions."""
    print("\n" + "=" * 120)
    print("  TESTING ACTIVATION SIGNALS — Which conditions predict profitable trading days?")
    print("=" * 120)

    # Load trades
    trades_csv = os.path.join(OUT, 'expanded_all_trades.csv')
    if not os.path.exists(trades_csv):
        print("  No expanded trades found!")
        return None

    trades = pd.read_csv(trades_csv)
    trades['entry_dt'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['entry_date'] = trades['entry_dt'].dt.normalize()
    trades['usd_pnl'] = trades['net_bps'] / 10000 * trades['position_size'] * 10000
    print(f"  Loaded {len(trades)} trades")

    # Build daily PnL from trades
    daily_pnl = trades.groupby('entry_date').agg(
        n=('net_bps', 'count'),
        total_bps=('net_bps', 'sum'),
        usd=('usd_pnl', 'sum'),
        wr=('net_bps', lambda x: (x > 0).mean()),
    )

    # Merge with features
    feat_norm = feat.copy()
    feat_norm.index = feat_norm.index.normalize()
    merged = daily_pnl.join(feat_norm, how='left')
    print(f"  {len(merged)} trading days with features")

    # =========================================================================
    # SIGNAL DEFINITIONS
    # =========================================================================
    signals = {}

    # Signal 1: Vol explosion (rvol_7d surging above 30d)
    for thr in [1.0, 1.2, 1.5, 2.0]:
        name = f"vol_expand_{thr}"
        signals[name] = feat_norm['vol_ratio_7_30'] > thr

    # Signal 2: OI deleveraging (7d OI change very negative)
    for thr in [-3, -5, -8, -10]:
        name = f"oi_delev_{abs(thr)}"
        signals[name] = feat_norm['oi_chg_7d'] < thr

    # Signal 3: Price drawdown from recent high
    for thr in [-3, -5, -8, -10]:
        name = f"dd_14d_{abs(thr)}"
        signals[name] = feat_norm['drawdown_from_14d_high'] < thr

    # Signal 4: Volume surge
    for thr in [1.2, 1.5, 2.0]:
        name = f"vol_surge_{thr}"
        signals[name] = feat_norm['volume_surge'] > thr

    # Signal 5: ATR expansion
    for thr in [1.2, 1.5, 2.0]:
        name = f"atr_exp_{thr}"
        signals[name] = feat_norm['atr_expansion'] > thr

    # Signal 6: High absolute rvol
    for thr in [40, 50, 60, 80]:
        name = f"rvol_7d_gt{thr}"
        signals[name] = feat_norm['rvol_7d'] > thr

    # Signal 7: Positive autocorrelation (trending)
    for thr in [0.0, 0.1, 0.2]:
        name = f"autocorr_gt{thr}"
        signals[name] = feat_norm['autocorr_7d'] > thr

    # Composite signals
    # C1: Vol explosion + OI deleveraging
    signals['combo_vol_oi'] = (feat_norm['vol_ratio_7_30'] > 1.2) & (feat_norm['oi_chg_7d'] < -3)
    # C2: Vol explosion + drawdown
    signals['combo_vol_dd'] = (feat_norm['vol_ratio_7_30'] > 1.2) & (feat_norm['drawdown_from_14d_high'] < -5)
    # C3: Full pattern: vol + OI delev + drawdown
    signals['combo_full_loose'] = ((feat_norm['vol_ratio_7_30'] > 1.0) &
                                    (feat_norm['oi_chg_7d'] < -3) &
                                    (feat_norm['drawdown_from_14d_high'] < -3))
    signals['combo_full_tight'] = ((feat_norm['vol_ratio_7_30'] > 1.2) &
                                    (feat_norm['oi_chg_7d'] < -5) &
                                    (feat_norm['drawdown_from_14d_high'] < -5))
    # C4: Vol + volume surge + OI delev
    signals['combo_vol_volsurge_oi'] = ((feat_norm['vol_ratio_7_30'] > 1.2) &
                                         (feat_norm['volume_surge'] > 1.2) &
                                         (feat_norm['oi_chg_7d'] < -3))
    # C5: High rvol + OI delev
    signals['combo_hirvol_oi'] = ((feat_norm['rvol_7d'] > 50) & (feat_norm['oi_chg_7d'] < -5))
    # C6: ATR expansion + drawdown
    signals['combo_atr_dd'] = ((feat_norm['atr_expansion'] > 1.3) &
                                (feat_norm['drawdown_from_14d_high'] < -5))
    # C7: Vol percentile + OI
    signals['combo_volpct_oi'] = ((feat_norm['rvol_7d_pct'] > 0.7) & (feat_norm['oi_chg_7d'] < -5))

    # =========================================================================
    # EVALUATE EACH SIGNAL
    # =========================================================================
    print(f"\n  {'Signal':35s} {'Days':>5s} {'Rate':>6s} {'Trades':>7s} {'WR':>6s} "
          f"{'AvgBps':>8s} {'TotUSD':>12s} {'noOct$':>12s} {'PF':>6s}")
    print("  " + "-" * 110)

    results = []
    for name, mask in sorted(signals.items()):
        # Days where signal is active
        active_dates = set(mask[mask].index)
        n_days = len(active_dates)

        if n_days == 0:
            continue

        rate = n_days / len(feat_norm) * 100

        # Filter trades to active days
        active_trades = trades[trades['entry_date'].isin(active_dates)]
        inactive_trades = trades[~trades['entry_date'].isin(active_dates)]

        if len(active_trades) == 0:
            continue

        n_trades = len(active_trades)
        wr = (active_trades['net_bps'] > 0).mean()
        avg_bps = active_trades['net_bps'].mean()
        total_usd = active_trades['usd_pnl'].sum()

        # Ex-October
        no_oct = active_trades[~active_trades['month'].str.contains('2025-10')]
        no_oct_usd = no_oct['usd_pnl'].sum() if len(no_oct) > 0 else 0

        # Profit factor
        wins = active_trades.loc[active_trades['net_bps'] > 0, 'net_bps'].sum()
        losses = abs(active_trades.loc[active_trades['net_bps'] <= 0, 'net_bps'].sum())
        pf = wins / losses if losses > 0 else 999

        results.append({
            'signal': name, 'n_days': n_days, 'rate': rate,
            'n_trades': n_trades, 'wr': wr, 'avg_bps': avg_bps,
            'total_usd': total_usd, 'no_oct_usd': no_oct_usd, 'pf': pf,
        })

        print(f"  {name:35s} {n_days:>5d} {rate:>5.1f}% {n_trades:>7d} {wr:>5.0%} "
              f"{avg_bps:>+8.0f} ${total_usd:>+11,.0f} ${no_oct_usd:>+11,.0f} {pf:>6.2f}")

    # =========================================================================
    # RANK BY BEST EX-OCTOBER PERFORMANCE
    # =========================================================================
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("  No results!")
        return None

    print(f"\n  === TOP 10 BY EX-OCTOBER USD (strategy works outside October too) ===")
    top = results_df.sort_values('no_oct_usd', ascending=False).head(10)
    for _, r in top.iterrows():
        capture = r['total_usd'] / trades['usd_pnl'].sum() * 100
        print(f"    {r['signal']:35s} {r['n_trades']:>4.0f}T {r['wr']:.0%} "
              f"avg={r['avg_bps']:+.0f} total=${r['total_usd']:+,.0f} "
              f"noOct=${r['no_oct_usd']:+,.0f} ({capture:.0f}% capture)")

    print(f"\n  === TOP 10 BY TOTAL CAPTURE (maximize total including Oct) ===")
    results_df['capture'] = results_df['total_usd'] / trades['usd_pnl'].sum() * 100
    top2 = results_df.sort_values('total_usd', ascending=False).head(10)
    for _, r in top2.iterrows():
        print(f"    {r['signal']:35s} {r['n_trades']:>4.0f}T {r['wr']:.0%} "
              f"avg={r['avg_bps']:+.0f} total=${r['total_usd']:+,.0f} "
              f"noOct=${r['no_oct_usd']:+,.0f} ({r['capture']:.0f}% capture)")

    print(f"\n  === TOP 10 BY PROFIT FACTOR (quality of signal) ===")
    top3 = results_df[results_df['n_trades'] >= 10].sort_values('pf', ascending=False).head(10)
    for _, r in top3.iterrows():
        print(f"    {r['signal']:35s} {r['n_trades']:>4.0f}T {r['wr']:.0%} PF={r['pf']:.2f} "
              f"avg={r['avg_bps']:+.0f} total=${r['total_usd']:+,.0f} noOct=${r['no_oct_usd']:+,.0f}")

    return results_df, signals, feat_norm, trades


def deep_dive_best_signal(results_df, signals, feat_norm, trades):
    """Deep dive on the best signal: monthly breakdown, false positives, lead time."""
    print("\n" + "=" * 120)
    print("  DEEP DIVE: BEST ACTIVATION SIGNALS")
    print("=" * 120)

    # Pick top signals by different criteria
    best_names = set()
    # Best ex-Oct
    best_names.add(results_df.sort_values('no_oct_usd', ascending=False).iloc[0]['signal'])
    # Best PF with >=10 trades
    pf_cands = results_df[results_df['n_trades'] >= 10]
    if len(pf_cands) > 0:
        best_names.add(pf_cands.sort_values('pf', ascending=False).iloc[0]['signal'])
    # Best capture
    best_names.add(results_df.sort_values('total_usd', ascending=False).iloc[0]['signal'])
    # Add combo signals
    for name in ['combo_vol_oi', 'combo_full_loose', 'combo_full_tight',
                 'combo_hirvol_oi', 'combo_volpct_oi']:
        if name in results_df['signal'].values:
            best_names.add(name)

    for signal_name in sorted(best_names):
        if signal_name not in signals:
            continue

        mask = signals[signal_name]
        active_dates = set(mask[mask].index)

        active_trades = trades[trades['entry_date'].isin(active_dates)].copy()
        if len(active_trades) == 0:
            continue

        active_trades['month_str'] = active_trades['entry_dt'].dt.to_period('M').astype(str)

        print(f"\n  --- {signal_name} ---")
        print(f"  Active {len(active_dates)} days ({len(active_dates)/len(feat_norm)*100:.1f}%)")

        # Monthly breakdown
        print(f"\n  {'Month':>10s} {'Trades':>7s} {'WR':>6s} {'Avg':>8s} {'USD':>12s}")
        print("  " + "-" * 50)

        for m in sorted(active_trades['month_str'].unique()):
            sub = active_trades[active_trades['month_str'] == m]
            usd = sub['usd_pnl'].sum()
            wr = (sub['net_bps'] > 0).mean()
            avg = sub['net_bps'].mean()
            status = "✓" if usd > 0 else "✗"
            oct = " ◄◄" if "2025-10" in m else ""
            print(f"  {m:>10s} {status} {len(sub):>5d} {wr:>5.0%} {avg:>+8.0f} ${usd:>+11,.0f}{oct}")

        no_oct = active_trades[~active_trades['month_str'].str.contains('2025-10')]
        if len(no_oct) > 0:
            no_oct_usd = no_oct['usd_pnl'].sum()
            print(f"\n  Ex-Oct: {len(no_oct)} trades, ${no_oct_usd:+,.0f}")

        # When does signal activate in the full history?
        activated_full = mask[mask]
        if len(activated_full) > 0:
            # Group into episodes
            episodes = []
            current_start = activated_full.index[0]
            current_end = activated_full.index[0]
            for dt in activated_full.index[1:]:
                if (dt - current_end).days <= 3:
                    current_end = dt
                else:
                    episodes.append((current_start, current_end))
                    current_start = dt
                    current_end = dt
            episodes.append((current_start, current_end))

            print(f"\n  Full history activations ({len(episodes)} episodes):")
            print(f"  {'#':>3s} {'Start':>12s} {'End':>12s} {'Days':>5s} {'Fwd7d':>8s} {'Fwd30d':>8s} {'StratUSD':>12s}")
            print("  " + "-" * 70)

            for i, (s, e) in enumerate(episodes):
                # Forward returns
                fwd_idx = feat_norm.index.searchsorted(e)
                fwd_7d = ((feat_norm['price'].iloc[fwd_idx + 7] / feat_norm['price'].iloc[fwd_idx] - 1) * 100
                          if fwd_idx + 7 < len(feat_norm) else np.nan)
                fwd_30d = ((feat_norm['price'].iloc[fwd_idx + 30] / feat_norm['price'].iloc[fwd_idx] - 1) * 100
                           if fwd_idx + 30 < len(feat_norm) else np.nan)

                # Strategy PnL during this episode
                ep_dates = set(pd.date_range(s, e, freq='D').normalize())
                ep_trades = trades[trades['entry_date'].isin(ep_dates)]
                ep_usd = ep_trades['usd_pnl'].sum() if len(ep_trades) > 0 else 0

                tag = ""
                if s.year == 2025 and s.month in [9, 10, 11]:
                    tag = " ◄◄"
                print(f"  {i+1:>3d} {s.strftime('%Y-%m-%d'):>12s} {e.strftime('%Y-%m-%d'):>12s} "
                      f"{(e-s).days+1:>5d} {fwd_7d:>+7.1f}% {fwd_30d:>+7.1f}% ${ep_usd:>+11,.0f}{tag}")


def precursor_analysis(feat_norm):
    """Look for what happens BEFORE the vol explosion — the actual 'activation trigger'."""
    print("\n" + "=" * 120)
    print("  PRE-CURSOR ANALYSIS: What happens in the weeks BEFORE a vol explosion?")
    print("  (This is what you'd monitor in shadow mode to know when to turn the strategy ON)")
    print("=" * 120)

    # Define vol explosion events: rvol_7d surging to top 20%
    rvol_pct = feat_norm['rvol_7d_pct']
    explosion_starts = []

    # Find transitions from low/normal vol to high vol
    in_explosion = False
    for i in range(1, len(feat_norm)):
        if not in_explosion and rvol_pct.iloc[i] > 0.75 and rvol_pct.iloc[i-1] <= 0.75:
            explosion_starts.append(feat_norm.index[i])
            in_explosion = True
        elif in_explosion and rvol_pct.iloc[i] < 0.5:
            in_explosion = False

    print(f"\n  Found {len(explosion_starts)} vol explosion onset events")

    # For each explosion, look at features 7d and 14d BEFORE
    cols = ['rvol_7d', 'rvol_30d', 'vol_ratio_7_30', 'mom_7d', 'mom_14d',
            'drawdown_from_14d_high', 'volume_surge', 'oi_chg_7d', 'oi_chg_14d',
            'oi_vs_ma30', 'autocorr_7d', 'atr_expansion', 'fr_cum_7d']
    cols = [c for c in cols if c in feat_norm.columns]

    pre_data_7d = []
    pre_data_14d = []
    for dt in explosion_starts:
        idx = feat_norm.index.searchsorted(dt)
        if idx >= 14:
            pre_data_7d.append(feat_norm.iloc[idx - 7][cols])
            pre_data_14d.append(feat_norm.iloc[idx - 14][cols])

    if not pre_data_7d:
        print("  No valid pre-explosion windows found")
        return

    pre_7d = pd.DataFrame(pre_data_7d)
    pre_14d = pd.DataFrame(pre_data_14d)
    all_days = feat_norm[cols]

    print(f"\n  Feature comparison: 7d before vol explosion vs all days")
    print(f"  {'Feature':25s} {'All Med':>10s} {'Pre-7d Med':>10s} {'z-score':>8s}")
    print("  " + "-" * 60)

    precursors = []
    for col in cols:
        all_med = all_days[col].median()
        all_std = all_days[col].std()
        pre_med = pre_7d[col].median()
        z = (pre_med - all_med) / all_std if all_std > 0 else 0
        sig = " ★★" if abs(z) > 1.0 else (" ★" if abs(z) > 0.5 else "")
        print(f"  {col:25s} {all_med:>10.2f} {pre_med:>10.2f} {z:>+7.2f}{sig}")
        if abs(z) > 0.5:
            precursors.append((col, z))

    print(f"\n  Feature comparison: 14d before vol explosion vs all days")
    print(f"  {'Feature':25s} {'All Med':>10s} {'Pre-14d Med':>10s} {'z-score':>8s}")
    print("  " + "-" * 60)

    for col in cols:
        all_med = all_days[col].median()
        all_std = all_days[col].std()
        pre_med = pre_14d[col].median()
        z = (pre_med - all_med) / all_std if all_std > 0 else 0
        sig = " ★★" if abs(z) > 1.0 else (" ★" if abs(z) > 0.5 else "")
        print(f"  {col:25s} {all_med:>10.2f} {pre_med:>10.2f} {z:>+7.2f}{sig}")

    # Build a "precursor" signal
    print(f"\n  === PRECURSOR SIGNAL (conditions that predict upcoming vol explosion) ===")
    print(f"  Key precursors found: {[(c, f'z={z:+.1f}') for c, z in precursors]}")

    # Specific to October 2025: what were features on Sep 15-30?
    sep_end = feat_norm.loc['2025-09-15':'2025-09-30']
    print(f"\n  September 15-30, 2025 (before October explosion):")
    for col in cols:
        val = sep_end[col].median()
        all_med = all_days[col].median()
        all_std = all_days[col].std()
        z = (val - all_med) / all_std if all_std > 0 else 0
        print(f"    {col:25s}: {val:>10.2f}  (z={z:+.2f})")


def main():
    t0 = time.time()

    ohlcv, oi_1h, fr_series = load_btc_full()
    feat = compute_features(ohlcv, oi_1h, fr_series)

    # Test all activation signals against actual trades
    result = test_activation_signals(feat)

    if result:
        results_df, signals, feat_norm, trades = result

        # Deep dive on best signals
        deep_dive_best_signal(results_df, signals, feat_norm, trades)

    # Precursor analysis — what happens BEFORE vol explosions?
    feat_norm = feat.copy()
    feat_norm.index = feat_norm.index.normalize()
    precursor_analysis(feat_norm)

    print(f"\n{'='*100}")
    print(f"  TOTAL TIME: {time.time()-t0:.0f}s")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
