#!/usr/bin/env python3
"""
REGIME FILTER: When do the signals work?
==========================================
The signals work in 2025-06 → 2026-03 but NOT in 2024-01 → 2025-05.
Find real-time measurable features that separate these regimes.

Approach:
1. Compute rolling regime features (no lookahead) for the FULL period 2024-01 → 2026-03
2. For each week, measure: did Idea 4 / Idea 5 produce positive returns?
3. Find features that predict "signal works this week" vs "signal fails this week"
4. Build a simple trade/no-trade filter
5. Validate with leave-one-out or rolling walk-forward

Candidate regime features (all backward-looking, no lookahead):
 - BTC 30d realized vol
 - BTC 30d trend (return)
 - BTC-alt correlation (30d rolling)
 - Average spot-futures basis level
 - Average funding rate level
 - Cross-alt dispersion (return spread across alts)
 - OI growth (30d)
 - Volume trend
"""
import sys
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from data_loader import load_kline, load_oi, load_funding_rate, load_premium, progress_bar, RT_TAKER_BPS

OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'
START = '2024-01-01'
END = '2026-03-04'

# Coins for regime analysis
ALT_SYMBOLS = [
    'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
    'LINKUSDT', 'ADAUSDT', 'APTUSDT', 'ARBUSDT',
    'NEARUSDT', 'ATOMUSDT', 'DOTUSDT',
    'OPUSDT', 'INJUSDT', 'LTCUSDT', 'BCHUSDT',
    'FILUSDT', 'AAVEUSDT', 'BNBUSDT', 'MKRUSDT',
]

HORIZONS = [30, 60, 240]
np.random.seed(42)


def build_regime_features(btc, alts_dict, spot_dict):
    """Build weekly regime features from daily/intraday data.
    All features use ONLY past data (no lookahead).
    Indexed by week start date.
    """
    # Resample BTC to hourly for efficiency
    btc_h = btc['close'].resample('1h').last().dropna()
    btc_d = btc['close'].resample('1D').last().dropna()

    features = pd.DataFrame(index=btc_d.index)

    # === BTC features ===
    # 30d realized vol (annualized)
    btc_ret_d = btc_d.pct_change()
    features['btc_vol_30d'] = btc_ret_d.rolling(30).std() * np.sqrt(365) * 100

    # BTC trend: 30d return
    features['btc_trend_30d'] = btc_d.pct_change(30) * 100

    # BTC trend: 7d return
    features['btc_trend_7d'] = btc_d.pct_change(7) * 100

    # BTC vol regime: vol of vol (30d rolling std of daily vol)
    features['btc_vol_of_vol'] = features['btc_vol_30d'].rolling(30).std()

    # BTC price level (log, normalized)
    features['btc_log_price'] = np.log(btc_d)

    # === Cross-alt features ===
    alt_rets_d = pd.DataFrame()
    for sym, df in alts_dict.items():
        if len(df) < 5000:
            continue
        ad = df['close'].resample('1D').last().dropna()
        alt_rets_d[sym] = ad.pct_change()

    if len(alt_rets_d.columns) > 3:
        # Alt dispersion: cross-sectional std of daily returns
        features['alt_dispersion'] = alt_rets_d.std(axis=1).rolling(30).mean() * 100

        # BTC-alt correlation: avg rolling 30d correlation
        btc_ret_aligned = btc_ret_d.reindex(alt_rets_d.index)
        corrs = []
        for sym in alt_rets_d.columns:
            c = btc_ret_aligned.rolling(30).corr(alt_rets_d[sym])
            corrs.append(c)
        if corrs:
            corr_df = pd.concat(corrs, axis=1)
            features['btc_alt_corr_30d'] = corr_df.mean(axis=1)

        # Alt momentum: avg 30d return across alts
        alt_mom = pd.DataFrame()
        for sym, df in alts_dict.items():
            ad = df['close'].resample('1D').last().dropna()
            alt_mom[sym] = ad.pct_change(30) * 100
        features['alt_avg_mom_30d'] = alt_mom.mean(axis=1)

    # === Spot-futures basis ===
    basis_list = []
    for sym in spot_dict:
        if sym not in alts_dict:
            continue
        fut_d = alts_dict[sym]['close'].resample('1D').last().dropna()
        spot_d = spot_dict[sym]['close'].resample('1D').last().dropna()
        aligned = pd.DataFrame({'fut': fut_d, 'spot': spot_d}).dropna()
        if len(aligned) > 30:
            basis = ((aligned['fut'] / aligned['spot'] - 1) * 10000)
            basis_list.append(basis)
    if basis_list:
        basis_all = pd.concat(basis_list, axis=1)
        features['avg_basis_bps'] = basis_all.mean(axis=1).rolling(7).mean()

    # === Funding rate level ===
    fr_list = []
    for sym in list(alts_dict.keys())[:10]:
        fr = load_funding_rate(sym, START, END)
        if not fr.empty and 'fundingRate' in fr.columns:
            fr_s = fr.set_index('ts')['fundingRate'].resample('1D').mean() * 10000  # bps
            fr_list.append(fr_s)
    if fr_list:
        fr_all = pd.concat(fr_list, axis=1)
        features['avg_fr_bps'] = fr_all.mean(axis=1).rolling(7).mean()

    features = features.dropna(how='all')
    return features


def compute_weekly_signal_returns(btc, alts_dict, spot_dict):
    """For each week, compute signal returns for Idea 4 and Idea 5."""
    results = []

    btc_1m = btc.copy()
    btc_1m['btc_ret_3m'] = (btc_1m['close'] / btc_1m['close'].shift(3) - 1) * 10000

    # Weekly periods
    weeks = pd.date_range(START, END, freq='W-MON')

    t0 = time.time()
    for wi, week_start in enumerate(weeks):
        progress_bar(wi, len(weeks), prefix='Weekly returns', start_time=t0)
        week_end = week_start + pd.Timedelta(days=7)

        btc_week = btc_1m[(btc_1m.index >= week_start) & (btc_1m.index < week_end)]
        if len(btc_week) < 1000:
            continue

        idea4_rets = []
        idea5_rets = []

        for sym in list(alts_dict.keys()):
            alt = alts_dict[sym]
            aw = alt[(alt.index >= week_start) & (alt.index < week_end)]
            if len(aw) < 1000:
                continue

            merged = aw[['close']].join(btc_week[['btc_ret_3m']], how='inner')
            if len(merged) < 500:
                continue

            # Forward return at 240m (best horizon from OOS)
            merged['fwd_240m'] = (merged['close'].shift(-240) / merged['close'] - 1) * 10000
            # T+1 entry
            merged['dfwd_240m'] = merged['fwd_240m'].shift(-1)

            # Idea 4: BTC pump > 150 bps
            sig4 = merged['btc_ret_3m'] > 150
            # Decluster
            indices = np.where(sig4.values)[0]
            if len(indices) > 0:
                kept = [indices[0]]
                for idx in indices[1:]:
                    if idx - kept[-1] >= 30:
                        kept.append(idx)
                sig4_dc = pd.Series(False, index=merged.index)
                sig4_dc.iloc[kept] = True
                r4 = merged.loc[sig4_dc, 'dfwd_240m'].dropna()
                if len(r4) > 0:
                    idea4_rets.extend(r4.values)

            # Idea 5: Spot leads
            if sym in spot_dict:
                sw = spot_dict[sym]
                sw_week = sw[(sw.index >= week_start) & (sw.index < week_end)]
                if len(sw_week) > 500:
                    merged2 = merged.join(sw_week[['close']].rename(columns={'close': 'spot_close'}), how='left')
                    if 'spot_close' in merged2.columns:
                        merged2['spot_ret_3m'] = (merged2['spot_close'] / merged2['spot_close'].shift(3) - 1) * 10000
                        merged2['fut_ret_3m'] = (merged2['close'] / merged2['close'].shift(3) - 1) * 10000
                        sig5 = ((merged2['spot_ret_3m'] - merged2['fut_ret_3m']) > 40).fillna(False)
                        indices5 = np.where(sig5.values)[0]
                        if len(indices5) > 0:
                            kept5 = [indices5[0]]
                            for idx in indices5[1:]:
                                if idx - kept5[-1] >= 30:
                                    kept5.append(idx)
                            sig5_dc = pd.Series(False, index=merged2.index)
                            sig5_dc.iloc[kept5] = True
                            r5 = merged2.loc[sig5_dc, 'dfwd_240m'].dropna()
                            if len(r5) > 0:
                                idea5_rets.extend(r5.values)

        row = {
            'week': week_start,
            'idea4_n': len(idea4_rets),
            'idea4_mean': np.mean(idea4_rets) if idea4_rets else np.nan,
            'idea4_net': np.mean(idea4_rets) - RT_TAKER_BPS if idea4_rets else np.nan,
            'idea5_n': len(idea5_rets),
            'idea5_mean': np.mean(idea5_rets) if idea5_rets else np.nan,
            'idea5_net': np.mean(idea5_rets) - RT_TAKER_BPS if idea5_rets else np.nan,
        }
        results.append(row)

    progress_bar(len(weeks), len(weeks), prefix='Weekly returns', start_time=t0)
    return pd.DataFrame(results)


def find_regime_filters(features, weekly_rets):
    """Find which features predict signal profitability."""
    # Align features (daily) to weekly by taking the value at week start
    feat_weekly = features.resample('W-MON').last()

    print(f"\n{'='*70}")
    print("  REGIME FEATURE ANALYSIS")
    print(f"{'='*70}")

    for signal in ['idea4', 'idea5']:
        net_col = f'{signal}_net'
        n_col = f'{signal}_n'

        wr = weekly_rets[weekly_rets[n_col] > 0].copy()
        if len(wr) < 10:
            print(f"\n  {signal}: too few weeks with signals ({len(wr)})")
            continue

        wr = wr.set_index('week')
        wr['profitable'] = wr[net_col] > 0

        # Merge with features
        merged = wr.join(feat_weekly, how='inner')
        if len(merged) < 10:
            continue

        print(f"\n{'─'*70}")
        print(f"  {signal.upper()}: {len(merged)} weeks with signals")
        print(f"  Profitable weeks: {merged['profitable'].sum()}/{len(merged)} = {merged['profitable'].mean()*100:.0f}%")
        print(f"{'─'*70}")

        feat_cols = [c for c in features.columns if c in merged.columns]

        print(f"\n  {'Feature':<25s} │ {'Corr w/ ret':>11s} │ {'Prof=True':>10s} │ {'Prof=False':>10s} │ {'Separation':>10s}")
        print(f"  {'─'*25}─┼─{'─'*11}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")

        best_features = []
        for feat in feat_cols:
            valid = merged[[net_col, 'profitable', feat]].dropna()
            if len(valid) < 10:
                continue

            corr = valid[net_col].corr(valid[feat])

            prof_true = valid.loc[valid['profitable'], feat].mean()
            prof_false = valid.loc[~valid['profitable'], feat].mean()
            pooled_std = valid[feat].std()
            if pooled_std > 0:
                separation = (prof_true - prof_false) / pooled_std
            else:
                separation = 0

            print(f"  {feat:<25s} │ {corr:>+10.3f}  │ {prof_true:>10.2f} │ {prof_false:>10.2f} │ {separation:>+10.3f}")
            best_features.append((feat, corr, separation, prof_true, prof_false))

        # Sort by absolute separation
        best_features.sort(key=lambda x: abs(x[2]), reverse=True)

        if best_features:
            print(f"\n  TOP 3 DISCRIMINATING FEATURES:")
            for feat, corr, sep, pt, pf in best_features[:3]:
                direction = "HIGH when profitable" if pt > pf else "LOW when profitable"
                print(f"    {feat}: separation={sep:+.3f}, corr={corr:+.3f} → {direction}")

            # === BUILD SIMPLE FILTER ===
            print(f"\n  FILTER CANDIDATES:")
            for feat, corr, sep, pt, pf in best_features[:5]:
                if abs(sep) < 0.2:
                    continue

                # Find threshold that maximizes Sharpe
                valid = merged[[net_col, feat]].dropna()
                thresholds = np.percentile(valid[feat].values, np.arange(20, 81, 10))

                best_sharpe = -999
                best_thresh = None
                best_dir = None
                best_stats = None

                for thresh in thresholds:
                    for direction in ['above', 'below']:
                        if direction == 'above':
                            mask = valid[feat] > thresh
                        else:
                            mask = valid[feat] < thresh

                        if mask.sum() < 5 or (~mask).sum() < 5:
                            continue

                        rets_in = valid.loc[mask, net_col]
                        rets_out = valid.loc[~mask, net_col]

                        sharpe = rets_in.mean() / max(rets_in.std(), 1) if len(rets_in) > 2 else 0

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_thresh = thresh
                            best_dir = direction
                            best_stats = {
                                'trade_weeks': mask.sum(),
                                'no_trade_weeks': (~mask).sum(),
                                'trade_mean': rets_in.mean(),
                                'no_trade_mean': rets_out.mean(),
                                'trade_wr': (rets_in > 0).mean() * 100,
                                'no_trade_wr': (rets_out > 0).mean() * 100,
                                'trade_sharpe': sharpe,
                            }

                if best_stats:
                    print(f"\n    Filter: {feat} {best_dir} {best_thresh:.2f}")
                    print(f"      TRADE:    {best_stats['trade_weeks']:.0f} weeks, "
                          f"avg={best_stats['trade_mean']:+.0f} bps, WR={best_stats['trade_wr']:.0f}%, "
                          f"Sharpe={best_stats['trade_sharpe']:.3f}")
                    print(f"      NO-TRADE: {best_stats['no_trade_weeks']:.0f} weeks, "
                          f"avg={best_stats['no_trade_mean']:+.0f} bps, WR={best_stats['no_trade_wr']:.0f}%")

        # === WALK-FORWARD FILTER VALIDATION ===
        print(f"\n  WALK-FORWARD FILTER TEST (expanding window):")
        if best_features and abs(best_features[0][2]) > 0.2:
            top_feat = best_features[0][0]
            valid = merged[[net_col, top_feat]].dropna().sort_index()

            # Use expanding window: train on first N weeks, test on week N+1
            min_train = 20
            wf_results = []
            for i in range(min_train, len(valid)):
                train = valid.iloc[:i]
                test = valid.iloc[i:i+1]

                # Find best threshold on training data
                train_med = train[top_feat].median()
                above_mean = train.loc[train[top_feat] > train_med, net_col].mean()
                below_mean = train.loc[train[top_feat] <= train_med, net_col].mean()

                # Trade when feature is on the profitable side
                if above_mean > below_mean:
                    should_trade = test[top_feat].values[0] > train_med
                else:
                    should_trade = test[top_feat].values[0] <= train_med

                wf_results.append({
                    'week': test.index[0],
                    'should_trade': should_trade,
                    'actual_ret': test[net_col].values[0],
                })

            wf_df = pd.DataFrame(wf_results)
            trade_weeks = wf_df[wf_df['should_trade']]
            notrade_weeks = wf_df[~wf_df['should_trade']]

            print(f"    Feature: {top_feat}")
            print(f"    Total WF weeks: {len(wf_df)}")
            if len(trade_weeks) > 0:
                print(f"    TRADE ({len(trade_weeks)} weeks): avg={trade_weeks['actual_ret'].mean():+.0f} bps, "
                      f"WR={(trade_weeks['actual_ret']>0).mean()*100:.0f}%")
            if len(notrade_weeks) > 0:
                print(f"    NO-TRADE ({len(notrade_weeks)} weeks): avg={notrade_weeks['actual_ret'].mean():+.0f} bps, "
                      f"WR={(notrade_weeks['actual_ret']>0).mean()*100:.0f}%")
            if len(trade_weeks) > 0 and len(notrade_weeks) > 0:
                lift = trade_weeks['actual_ret'].mean() - notrade_weeks['actual_ret'].mean()
                print(f"    LIFT: {lift:+.0f} bps (trade vs no-trade)")

                # Is this actually better than always trading?
                all_avg = wf_df['actual_ret'].mean()
                filtered_avg = trade_weeks['actual_ret'].mean()
                print(f"    Always trade: {all_avg:+.0f} bps | Filtered: {filtered_avg:+.0f} bps")
                if filtered_avg > all_avg and filtered_avg > 0:
                    print(f"    ✅ Filter IMPROVES returns by {filtered_avg - all_avg:+.0f} bps")
                elif filtered_avg > 0:
                    print(f"    ⚠️ Filter profitable but doesn't beat always-trade")
                else:
                    print(f"    ❌ Filter doesn't help")

    return best_features


def main():
    print("=" * 70)
    print("  REGIME FILTER: When do the signals work?")
    print(f"  Full period: {START} → {END}")
    print("=" * 70)

    # Load BTC
    print("\n  Loading BTC...")
    btc = load_kline('BTCUSDT', START, END)
    btc = btc[['ts', 'close']].set_index('ts').sort_index()
    btc = btc[~btc.index.duplicated(keep='first')]

    # Load alts
    print("  Loading alts...")
    alts_dict = {}
    spot_dict = {}
    t0 = time.time()
    for i, sym in enumerate(ALT_SYMBOLS):
        progress_bar(i, len(ALT_SYMBOLS), prefix='  Loading', start_time=t0)
        kf = load_kline(sym, START, END, spot=False)
        if not kf.empty and len(kf) > 5000:
            kf = kf[['ts', 'close']].set_index('ts').sort_index()
            kf = kf[~kf.index.duplicated(keep='first')]
            alts_dict[sym] = kf

        ks = load_kline(sym, START, END, spot=True)
        if not ks.empty and len(ks) > 5000:
            ks = ks[['ts', 'close']].set_index('ts').sort_index()
            ks = ks[~ks.index.duplicated(keep='first')]
            spot_dict[sym] = ks
    progress_bar(len(ALT_SYMBOLS), len(ALT_SYMBOLS), prefix='  Loading', start_time=t0)

    print(f"\n  Loaded {len(alts_dict)} alts, {len(spot_dict)} with spot data")

    # Step 1: Build regime features
    print("\n  Building regime features...")
    features = build_regime_features(btc, alts_dict, spot_dict)
    features.to_csv(f'{OUT}/regime_features.csv')
    print(f"  Built {len(features.columns)} features over {len(features)} days")
    print(f"  Features: {list(features.columns)}")

    # Step 2: Compute weekly signal returns
    print("\n  Computing weekly signal returns...")
    btc_1m = btc.copy()
    weekly_rets = compute_weekly_signal_returns(btc_1m, alts_dict, spot_dict)
    weekly_rets.to_csv(f'{OUT}/regime_weekly_rets.csv', index=False)

    idea4_weeks = weekly_rets[weekly_rets['idea4_n'] > 0]
    idea5_weeks = weekly_rets[weekly_rets['idea5_n'] > 0]
    print(f"\n  Idea 4: {len(idea4_weeks)} weeks with signals, "
          f"avg net={idea4_weeks['idea4_net'].mean():+.0f} bps")
    print(f"  Idea 5: {len(idea5_weeks)} weeks with signals, "
          f"avg net={idea5_weeks['idea5_net'].mean():+.0f} bps")

    # Step 3: Find regime filters
    best = find_regime_filters(features, weekly_rets)

    elapsed = time.time() - t0
    print(f"\n\n⏱ Total: {elapsed:.0f}s")
    print(f"✅ Saved: regime_features.csv, regime_weekly_rets.csv")


if __name__ == '__main__':
    main()
