#!/usr/bin/env python3
"""
October Predictor: What signals preceded the Oct 2025 vol event?
================================================================
Analyzes BTC macro features across the full Bybit history (2022-2026)
to find what was unique about the weeks BEFORE October 2025.

Goal: Build an "activation signal" that fires before Oct-like events
so the strategy can run in shadow mode until conditions are right.
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
    """Load BTC kline, OI, funding, LS ratio, premium from 2022 onwards."""
    print("Loading BTC data...")
    t0 = time.time()

    # Klines (futures)
    kline = load_csv_daterange("BTCUSDT", "kline_1m", "2022-01-01", "2026-03-05")
    if 'timestamp' in kline.columns:
        kline['dt'] = pd.to_datetime(kline['timestamp'], unit='ms', utc=True)
    elif 'open_time' in kline.columns:
        kline['dt'] = pd.to_datetime(kline['open_time'], unit='ms', utc=True)
    else:
        kline['dt'] = pd.to_datetime(kline.iloc[:, 0], unit='ms', utc=True)
    kline = kline.set_index('dt').sort_index()
    print(f"  Kline: {len(kline)} rows, {kline.index.min()} to {kline.index.max()}")

    # Resample to 1h bars for macro analysis
    ohlcv = kline.resample('1h').agg({
        kline.columns[1]: 'first',   # open
        kline.columns[2]: 'max',     # high
        kline.columns[3]: 'min',     # low
        kline.columns[4]: 'last',    # close
        kline.columns[5]: 'sum',     # volume
    }).dropna()
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    print(f"  1h bars: {len(ohlcv)}")

    # OI
    oi = load_csv_daterange("BTCUSDT", "open_interest_5min", "2022-01-01", "2026-03-05")
    if not oi.empty:
        if 'timestamp' in oi.columns:
            oi['dt'] = pd.to_datetime(oi['timestamp'], unit='ms', utc=True)
        else:
            oi['dt'] = pd.to_datetime(oi.iloc[:, 0], unit='ms', utc=True)
        oi = oi.set_index('dt').sort_index()
        # Get the OI value column
        oi_col = [c for c in oi.columns if 'interest' in c.lower() or 'oi' in c.lower() or 'value' in c.lower()]
        if oi_col:
            oi_1h = oi[oi_col[0]].resample('1h').last().dropna()
        else:
            oi_1h = oi.iloc[:, 0].astype(float).resample('1h').last().dropna()
        print(f"  OI: {len(oi_1h)} hourly points")
    else:
        oi_1h = pd.Series(dtype=float)

    # Funding rate
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
        if fr_col:
            fr_series = fr[fr_col[0]].astype(float)
        else:
            fr_series = fr.iloc[:, -1].astype(float)
        print(f"  Funding: {len(fr_series)} points")
    else:
        fr_series = pd.Series(dtype=float)

    # LS ratio
    ls = load_csv_daterange("BTCUSDT", "long_short_ratio_5min", "2022-01-01", "2026-03-05")
    if not ls.empty:
        if 'timestamp' in ls.columns:
            ls['dt'] = pd.to_datetime(ls['timestamp'], unit='ms', utc=True)
        else:
            ls['dt'] = pd.to_datetime(ls.iloc[:, 0], unit='ms', utc=True)
        ls = ls.set_index('dt').sort_index()
        ls_col = [c for c in ls.columns if 'ratio' in c.lower() or 'long' in c.lower()]
        if ls_col:
            ls_1h = ls[ls_col[0]].astype(float).resample('1h').last().dropna()
        else:
            ls_1h = ls.iloc[:, 0].astype(float).resample('1h').last().dropna()
        print(f"  LS ratio: {len(ls_1h)} hourly points")
    else:
        ls_1h = pd.Series(dtype=float)

    print(f"  Loaded in {time.time()-t0:.0f}s")
    return ohlcv, oi_1h, fr_series, ls_1h


def compute_macro_features(ohlcv, oi_1h, fr_series, ls_1h):
    """Compute daily macro features for regime detection."""
    print("\nComputing macro features...")

    # Work at daily level
    daily = ohlcv.resample('1D').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    feat = pd.DataFrame(index=daily.index)
    feat['price'] = daily['close']
    feat['ret_1d'] = daily['close'].pct_change()

    # ---- VOLATILITY FEATURES ----
    feat['rvol_7d'] = feat['ret_1d'].rolling(7).std() * np.sqrt(365) * 100
    feat['rvol_14d'] = feat['ret_1d'].rolling(14).std() * np.sqrt(365) * 100
    feat['rvol_30d'] = feat['ret_1d'].rolling(30).std() * np.sqrt(365) * 100
    feat['rvol_90d'] = feat['ret_1d'].rolling(90).std() * np.sqrt(365) * 100

    # Vol compression: short-term vol vs long-term vol
    feat['vol_compression'] = feat['rvol_7d'] / feat['rvol_30d']
    feat['vol_compression_14_90'] = feat['rvol_14d'] / feat['rvol_90d']

    # ATR
    tr = pd.concat([
        daily['high'] - daily['low'],
        (daily['high'] - daily['close'].shift(1)).abs(),
        (daily['low'] - daily['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['atr_7d'] = tr.rolling(7).mean() / daily['close'] * 100
    feat['atr_14d'] = tr.rolling(14).mean() / daily['close'] * 100
    feat['atr_ratio'] = feat['atr_7d'] / feat['atr_14d']

    # ---- TREND / MOMENTUM ----
    feat['mom_7d'] = daily['close'].pct_change(7) * 100
    feat['mom_14d'] = daily['close'].pct_change(14) * 100
    feat['mom_30d'] = daily['close'].pct_change(30) * 100

    # Price relative to moving averages
    feat['price_vs_sma20'] = (daily['close'] / daily['close'].rolling(20).mean() - 1) * 100
    feat['price_vs_sma50'] = (daily['close'] / daily['close'].rolling(50).mean() - 1) * 100

    # ---- RANGE / COMPRESSION ----
    feat['range_7d'] = (daily['high'].rolling(7).max() - daily['low'].rolling(7).min()) / daily['close'] * 100
    feat['range_30d'] = (daily['high'].rolling(30).max() - daily['low'].rolling(30).min()) / daily['close'] * 100
    feat['range_compression'] = feat['range_7d'] / feat['range_30d']

    # ---- VOLUME ----
    feat['vol_ma7'] = daily['volume'].rolling(7).mean()
    feat['vol_ma30'] = daily['volume'].rolling(30).mean()
    feat['vol_ratio'] = feat['vol_ma7'] / feat['vol_ma30']

    # ---- AUTOCORRELATION (mean-reversion vs trending) ----
    feat['autocorr_7d'] = feat['ret_1d'].rolling(14).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False)
    feat['autocorr_14d'] = feat['ret_1d'].rolling(30).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False)

    # ---- OI FEATURES ----
    if len(oi_1h) > 100:
        oi_daily = oi_1h.resample('1D').last().reindex(feat.index, method='ffill')
        feat['oi'] = oi_daily
        feat['oi_chg_7d'] = oi_daily.pct_change(7) * 100
        feat['oi_chg_14d'] = oi_daily.pct_change(14) * 100
        feat['oi_chg_30d'] = oi_daily.pct_change(30) * 100
        # OI relative to 30d mean
        feat['oi_vs_ma30'] = (oi_daily / oi_daily.rolling(30).mean() - 1) * 100
    else:
        for c in ['oi', 'oi_chg_7d', 'oi_chg_14d', 'oi_chg_30d', 'oi_vs_ma30']:
            feat[c] = np.nan

    # ---- FUNDING RATE ----
    if len(fr_series) > 100:
        fr_daily = fr_series.resample('1D').mean().reindex(feat.index, method='ffill')
        feat['fr_daily'] = fr_daily * 100  # in %
        feat['fr_7d_avg'] = fr_daily.rolling(7).mean() * 100
        feat['fr_cum_7d'] = fr_daily.rolling(7).sum() * 100
    else:
        for c in ['fr_daily', 'fr_7d_avg', 'fr_cum_7d']:
            feat[c] = np.nan

    # ---- LS RATIO ----
    if len(ls_1h) > 100:
        ls_daily = ls_1h.resample('1D').last().reindex(feat.index, method='ffill')
        feat['ls_ratio'] = ls_daily
        feat['ls_ratio_7d_avg'] = ls_daily.rolling(7).mean()
    else:
        feat['ls_ratio'] = np.nan
        feat['ls_ratio_7d_avg'] = np.nan

    feat = feat.dropna(subset=['rvol_30d'])
    print(f"  {len(feat)} daily feature rows ({feat.index.min().date()} to {feat.index.max().date()})")
    return feat


def analyze_october_preconditions(feat):
    """Compare features before/during October 2025 vs all other months."""
    print("\n" + "=" * 100)
    print("  WHAT WAS DIFFERENT ABOUT OCTOBER 2025?")
    print("=" * 100)

    # Define time windows
    oct_start = pd.Timestamp('2025-10-01', tz='UTC')
    oct_end = pd.Timestamp('2025-10-31', tz='UTC')

    # Pre-October: the 2-4 weeks before
    pre_oct_2w = feat.loc['2025-09-15':'2025-09-30']
    pre_oct_4w = feat.loc['2025-09-01':'2025-09-30']
    during_oct = feat.loc['2025-10-01':'2025-10-31']

    # All other months (non-Oct, non-Sep)
    other = feat.loc[(feat.index < '2025-09-01') | (feat.index > '2025-10-31')]

    # Key features to compare
    compare_cols = [
        'rvol_7d', 'rvol_14d', 'rvol_30d', 'rvol_90d',
        'vol_compression', 'vol_compression_14_90',
        'atr_7d', 'atr_ratio',
        'mom_7d', 'mom_14d', 'mom_30d',
        'price_vs_sma20', 'price_vs_sma50',
        'range_7d', 'range_30d', 'range_compression',
        'vol_ratio',
        'autocorr_7d', 'autocorr_14d',
        'oi_chg_7d', 'oi_chg_14d', 'oi_chg_30d', 'oi_vs_ma30',
        'fr_daily', 'fr_7d_avg',
        'ls_ratio',
    ]

    print(f"\n  {'Feature':30s} {'All Other':>10s} {'Pre-Oct 4w':>10s} {'Pre-Oct 2w':>10s} {'During Oct':>10s} {'Signal?':>10s}")
    print("  " + "-" * 90)

    signals = {}
    for col in compare_cols:
        if col not in feat.columns:
            continue
        other_med = other[col].median()
        pre4w_med = pre_oct_4w[col].median()
        pre2w_med = pre_oct_2w[col].median()
        during_med = during_oct[col].median()

        # Compute z-score of pre-oct vs all history
        all_std = other[col].std()
        if all_std > 0:
            z_pre4w = (pre4w_med - other_med) / all_std
            z_pre2w = (pre2w_med - other_med) / all_std
        else:
            z_pre4w = z_pre2w = 0

        signal = ""
        if abs(z_pre2w) > 1.5:
            signal = f"z={z_pre2w:+.1f} ★★"
        elif abs(z_pre2w) > 1.0:
            signal = f"z={z_pre2w:+.1f} ★"
        elif abs(z_pre4w) > 1.0:
            signal = f"z4w={z_pre4w:+.1f} ★"

        if signal:
            signals[col] = z_pre2w

        print(f"  {col:30s} {other_med:>10.2f} {pre4w_med:>10.2f} {pre2w_med:>10.2f} {during_med:>10.2f} {signal:>10s}")

    return signals


def find_similar_periods(feat, trades_df=None):
    """Find ALL periods in history that looked like pre-October 2025."""
    print("\n" + "=" * 100)
    print("  SCANNING FULL HISTORY FOR PRE-OCTOBER-LIKE CONDITIONS")
    print("=" * 100)

    # Define what made pre-October special based on the analysis
    # We'll use a rolling window approach

    # Compute the "October conditions" from Sep 2025 data
    pre_oct = feat.loc['2025-09-15':'2025-09-30']

    # Key discriminating features (will be identified from the analysis)
    # For now, use vol compression + OI buildup + momentum setup
    key_features = ['vol_compression', 'rvol_7d', 'rvol_30d', 'mom_30d',
                    'range_compression', 'autocorr_7d']
    key_features = [f for f in key_features if f in feat.columns and not feat[f].isna().all()]

    # Compute z-scores for each feature relative to expanding window
    z_scores = pd.DataFrame(index=feat.index)
    for col in key_features:
        expanding_mean = feat[col].expanding(min_periods=60).mean()
        expanding_std = feat[col].expanding(min_periods=60).std()
        z_scores[col] = (feat[col] - expanding_mean) / expanding_std.replace(0, np.nan)

    # Get the pre-Oct signature
    pre_oct_z = z_scores.loc['2025-09-15':'2025-09-30'].median()
    print(f"\n  Pre-October signature (z-scores):")
    for col in key_features:
        print(f"    {col:25s}: {pre_oct_z[col]:+.2f}")

    # Compute similarity to pre-Oct across all time (14-day rolling window)
    similarity = pd.Series(0.0, index=feat.index)
    for col in key_features:
        if not np.isnan(pre_oct_z[col]):
            # Higher similarity = closer z-score to the pre-Oct pattern
            diff = (z_scores[col].rolling(14).mean() - pre_oct_z[col]).abs()
            similarity += 1 / (1 + diff)  # 0 to 1 per feature

    similarity = similarity / len(key_features)  # normalize to 0-1
    feat['oct_similarity'] = similarity

    # Find peaks
    high_sim = similarity[similarity > similarity.quantile(0.95)]
    print(f"\n  Top 5% similarity dates (>{similarity.quantile(0.95):.3f}):")

    # Group into episodes
    if len(high_sim) > 0:
        episodes = []
        current_start = high_sim.index[0]
        current_end = high_sim.index[0]
        for dt in high_sim.index[1:]:
            if (dt - current_end).days <= 3:
                current_end = dt
            else:
                episodes.append((current_start, current_end))
                current_start = dt
                current_end = dt
        episodes.append((current_start, current_end))

        print(f"\n  {'Episode':>4s} {'Start':>12s} {'End':>12s} {'Days':>5s} {'Sim':>6s} {'Fwd 30d ret':>12s}")
        print("  " + "-" * 60)
        for i, (s, e) in enumerate(episodes):
            sim = similarity.loc[s:e].mean()
            # Forward 30d return
            fwd_idx = feat.index.searchsorted(e)
            if fwd_idx + 30 < len(feat):
                fwd_ret = (feat['price'].iloc[fwd_idx + 30] / feat['price'].iloc[fwd_idx] - 1) * 100
            else:
                fwd_ret = np.nan

            is_oct = "◄◄ OCT" if s.year == 2025 and s.month in [9, 10] else ""
            print(f"  {i+1:>4d} {s.strftime('%Y-%m-%d'):>12s} {e.strftime('%Y-%m-%d'):>12s} "
                  f"{(e-s).days+1:>5d} {sim:>6.3f} {fwd_ret:>+11.1f}% {is_oct}")

    return feat


def analyze_trade_conditions(feat, trades_csv):
    """Map actual trade results to daily features — what feature ranges produce winning trades?"""
    print("\n" + "=" * 100)
    print("  TRADE-LEVEL FEATURE ANALYSIS: Which conditions produce winners?")
    print("=" * 100)

    trades = pd.read_csv(trades_csv)
    trades['entry_dt'] = pd.to_datetime(trades['entry_time'], utc=True)
    trades['entry_date'] = trades['entry_dt'].dt.normalize()
    trades['win'] = trades['net_bps'] > 0

    # Merge with daily features
    feat_daily = feat.copy()
    feat_daily.index = feat_daily.index.normalize()

    merged = trades.merge(feat_daily, left_on='entry_date', right_index=True, how='left')
    print(f"  Merged {len(merged)} trades with daily features")

    # Key features
    analysis_cols = ['rvol_7d', 'rvol_30d', 'vol_compression', 'mom_7d', 'mom_30d',
                     'range_compression', 'autocorr_7d', 'vol_ratio',
                     'oi_chg_7d', 'oi_vs_ma30', 'fr_daily']
    analysis_cols = [c for c in analysis_cols if c in merged.columns]

    print(f"\n  {'Feature':25s} {'Win Med':>8s} {'Loss Med':>9s} {'Diff':>8s} {'p-val':>8s}")
    print("  " + "-" * 65)

    from scipy import stats as sp_stats

    useful_features = []
    for col in analysis_cols:
        wins = merged.loc[merged['win'], col].dropna()
        losses = merged.loc[~merged['win'], col].dropna()
        if len(wins) < 5 or len(losses) < 5:
            continue

        win_med = wins.median()
        loss_med = losses.median()
        diff = win_med - loss_med

        # Mann-Whitney U test
        try:
            _, pval = sp_stats.mannwhitneyu(wins, losses, alternative='two-sided')
        except:
            pval = 1.0

        sig = "★★" if pval < 0.01 else ("★" if pval < 0.05 else "")
        print(f"  {col:25s} {win_med:>8.2f} {loss_med:>9.2f} {diff:>+8.2f} {pval:>8.4f} {sig}")

        if pval < 0.10:
            useful_features.append((col, win_med, loss_med, pval))

    # Monthly feature evolution
    print(f"\n  Monthly feature evolution (key features):")
    feat_monthly = feat.resample('M').median()
    show_cols = ['rvol_7d', 'vol_compression', 'range_compression', 'oi_chg_30d', 'autocorr_7d']
    show_cols = [c for c in show_cols if c in feat_monthly.columns]

    hdr = f"  {'Month':>10s}" + "".join(f" {c:>18s}" for c in show_cols)
    print(hdr)
    print("  " + "-" * (12 + 19 * len(show_cols)))
    for m, r in feat_monthly.loc['2025-01':].iterrows():
        tag = " ◄◄" if m.month == 10 and m.year == 2025 else ""
        row = f"  {m.strftime('%Y-%m'):>10s}"
        for c in show_cols:
            row += f" {r[c]:>18.2f}"
        print(row + tag)

    return merged, useful_features


def build_activation_signal(feat):
    """Build a composite activation signal based on pre-October conditions."""
    print("\n" + "=" * 100)
    print("  BUILDING ACTIVATION SIGNAL")
    print("=" * 100)

    # Based on what we find, define activation conditions
    # These are hypothesis-driven, validated against the data

    # Condition 1: Vol compression (short vol << long vol) — coiled spring
    vol_comp = feat['vol_compression'] < feat['vol_compression'].expanding(60).quantile(0.25)

    # Condition 2: Rising OI (market loading up positions)
    oi_rising = feat.get('oi_chg_14d', pd.Series(0, index=feat.index)) > 5  # >5% OI growth

    # Condition 3: Positive momentum (bull setup)
    bull_mom = feat['mom_30d'] > 0

    # Condition 4: Range compression (tight range)
    range_comp = feat['range_compression'] < feat['range_compression'].expanding(60).quantile(0.30)

    # Condition 5: Negative autocorrelation (mean-reverting → about to trend)
    autocorr_neg = feat['autocorr_7d'] < -0.15

    # Composite score (0-5)
    score = vol_comp.astype(int) + oi_rising.astype(int) + bull_mom.astype(int) + \
            range_comp.astype(int) + autocorr_neg.astype(int)
    feat['activation_score'] = score

    # Activation: score >= 3 (majority of conditions met)
    feat['activated'] = score >= 3

    # Show when it fires
    activated_periods = feat[feat['activated']].copy()
    if len(activated_periods) > 0:
        # Group into episodes
        episodes = []
        current_start = activated_periods.index[0]
        current_end = activated_periods.index[0]
        for dt in activated_periods.index[1:]:
            if (dt - current_end).days <= 5:
                current_end = dt
            else:
                episodes.append((current_start, current_end))
                current_start = dt
                current_end = dt
        episodes.append((current_start, current_end))

        print(f"\n  Activation signal fires {len(activated_periods)} days across {len(episodes)} episodes")
        print(f"  Total days in dataset: {len(feat)}")
        print(f"  Activation rate: {len(activated_periods)/len(feat)*100:.1f}%")

        print(f"\n  {'#':>3s} {'Start':>12s} {'End':>12s} {'Days':>5s} {'Score':>6s} "
              f"{'Fwd 7d':>8s} {'Fwd 14d':>8s} {'Fwd 30d':>8s}")
        print("  " + "-" * 75)

        episode_returns = []
        for i, (s, e) in enumerate(episodes):
            avg_score = feat.loc[s:e, 'activation_score'].mean()
            fwd_rets = {}
            for horizon, label in [(7, 'Fwd 7d'), (14, 'Fwd 14d'), (30, 'Fwd 30d')]:
                fwd_idx = feat.index.searchsorted(e)
                if fwd_idx + horizon < len(feat):
                    fwd_rets[label] = (feat['price'].iloc[fwd_idx + horizon] / feat['price'].iloc[fwd_idx] - 1) * 100
                else:
                    fwd_rets[label] = np.nan

            is_oct = " ◄◄" if (s <= pd.Timestamp('2025-10-01', tz='UTC') and
                                e >= pd.Timestamp('2025-09-15', tz='UTC')) else ""

            print(f"  {i+1:>3d} {s.strftime('%Y-%m-%d'):>12s} {e.strftime('%Y-%m-%d'):>12s} "
                  f"{(e-s).days+1:>5d} {avg_score:>6.1f} "
                  f"{fwd_rets.get('Fwd 7d', np.nan):>+7.1f}% "
                  f"{fwd_rets.get('Fwd 14d', np.nan):>+7.1f}% "
                  f"{fwd_rets.get('Fwd 30d', np.nan):>+7.1f}%{is_oct}")

            episode_returns.append({
                'start': s, 'end': e,
                'fwd_7d': fwd_rets.get('Fwd 7d', np.nan),
                'fwd_14d': fwd_rets.get('Fwd 14d', np.nan),
                'fwd_30d': fwd_rets.get('Fwd 30d', np.nan),
            })

        # Summary stats
        er = pd.DataFrame(episode_returns)
        print(f"\n  Episode summary:")
        for h in ['fwd_7d', 'fwd_14d', 'fwd_30d']:
            vals = er[h].dropna()
            if len(vals) > 0:
                print(f"    {h}: mean={vals.mean():+.1f}%, median={vals.median():+.1f}%, "
                      f"positive={( vals > 0).sum()}/{len(vals)}")

    # Also test on strategy trades if available
    trades_csv = os.path.join(OUT, 'expanded_all_trades.csv')
    if os.path.exists(trades_csv):
        trades = pd.read_csv(trades_csv)
        trades['entry_dt'] = pd.to_datetime(trades['entry_time'], utc=True)
        trades['entry_date'] = trades['entry_dt'].dt.normalize()

        # Map activation to trades
        act_dates = set(activated_periods.index.normalize())
        trades['activated'] = trades['entry_date'].isin(act_dates)

        act_trades = trades[trades['activated']]
        inact_trades = trades[~trades['activated']]

        print(f"\n  Strategy trades with activation signal:")
        print(f"    Activated:   {len(act_trades):>4d} trades, "
              f"WR={(act_trades['net_bps']>0).mean():.0%}, "
              f"avg={act_trades['net_bps'].mean():+.0f} bps, "
              f"total=${(act_trades['net_bps']/10000*act_trades['position_size']*10000).sum():+,.0f}")
        print(f"    Deactivated: {len(inact_trades):>4d} trades, "
              f"WR={(inact_trades['net_bps']>0).mean():.0%}, "
              f"avg={inact_trades['net_bps'].mean():+.0f} bps, "
              f"total=${(inact_trades['net_bps']/10000*inact_trades['position_size']*10000).sum():+,.0f}")

        # Monthly breakdown for activated-only
        if len(act_trades) > 0:
            act_trades = act_trades.copy()
            act_trades['month'] = act_trades['entry_dt'].dt.to_period('M').astype(str)
            print(f"\n    Activated trades by month:")
            for m in sorted(act_trades['month'].unique()):
                sub = act_trades[act_trades['month'] == m]
                usd = (sub['net_bps'] / 10000 * sub['position_size'] * 10000).sum()
                status = "✓" if usd > 0 else "✗"
                print(f"      {m} {status} {len(sub):>3d}T WR={(sub['net_bps']>0).mean():.0%} "
                      f"avg={sub['net_bps'].mean():+.0f} ${usd:+,.0f}")

    return feat


def main():
    t0 = time.time()

    # Load data
    ohlcv, oi_1h, fr_series, ls_1h = load_btc_full()

    # Compute features
    feat = compute_macro_features(ohlcv, oi_1h, fr_series, ls_1h)

    # Analysis 1: What was different about October?
    signals = analyze_october_preconditions(feat)

    # Analysis 2: Find similar periods in history
    feat = find_similar_periods(feat)

    # Analysis 3: Trade-level feature analysis
    trades_csv = os.path.join(OUT, 'expanded_all_trades.csv')
    if os.path.exists(trades_csv):
        merged, useful = analyze_trade_conditions(feat, trades_csv)

    # Analysis 4: Build activation signal
    feat = build_activation_signal(feat)

    print(f"\n{'='*100}")
    print(f"  TOTAL TIME: {time.time()-t0:.0f}s")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
