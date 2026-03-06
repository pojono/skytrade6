#!/usr/bin/env python3
"""
REGIME FILTER v2: Daily resolution, dual-exchange, 700+ observations
=====================================================================
Uses FULL data: Bybit 2024-01→2026-03 + Binance 2025-01→2026-03.
Daily signal returns + daily regime features → walk-forward trade/no-trade filter.

Key improvements over v1:
 - Daily resolution (700+ obs vs 55 weekly)
 - Cross-exchange features (Bybit vs Binance divergence)
 - Binance metrics (taker ratio, OI)
 - Expanding-window walk-forward with 180-day minimum training
 - Multiple feature combinations tested
"""
import sys, os
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time, re
from pathlib import Path
from data_loader import progress_bar, RT_TAKER_BPS

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


# ============================================================
# DATA LOADING (dual-exchange)
# ============================================================

def _load_csv(sym_dir, dtype, start, end):
    """Load CSV from a symbol directory with date filtering."""
    pat = re.compile(r'^\d{4}-\d{2}-\d{2}_' + re.escape(dtype) + r'\.csv$')
    if not sym_dir.exists():
        return pd.DataFrame()
    files = sorted(f for f in os.listdir(sym_dir) if pat.match(f))
    files = [str(sym_dir / f) for f in files if (not start or f[:10] >= start) and (not end or f[:10] <= end)]
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
    for c in ['startTime', 'timestamp', 'open_time']:
        if c in df.columns:
            df['ts'] = pd.to_datetime(df[c], unit='ms')
            break
    if 'create_time' in df.columns:
        df['ts'] = pd.to_datetime(df['create_time'])
    if 'ts' in df.columns:
        df = df.sort_values('ts').drop_duplicates('ts', keep='first').reset_index(drop=True)
    return df


def load_bybit_kline(sym, start=START, end=END, spot=False):
    dtype = 'kline_1m_spot' if spot else 'kline_1m'
    df = _load_csv(BYBIT / sym, dtype, start, end)
    if not df.empty and 'close' in df.columns:
        df = df[['ts', 'close']].set_index('ts')
    return df


def load_binance_kline(sym, start=START, end=END, spot=False):
    dtype = 'kline_1m_spot' if spot else 'kline_1m'
    sym_dir = BINANCE / sym
    if not sym_dir.exists():
        return pd.DataFrame()

    if spot:
        # Spot klines have NO header, different timestamp format
        pat = re.compile(r'^\d{4}-\d{2}-\d{2}_kline_1m_spot\.csv$')
        files = sorted(f for f in os.listdir(sym_dir) if pat.match(f))
        files = [str(sym_dir / f) for f in files if (not start or f[:10] >= start) and (not end or f[:10] <= end)]
        if not files:
            return pd.DataFrame()
        chunks = []
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore']
        for f in files:
            try:
                df = pd.read_csv(f, header=None, names=cols)
                # Timestamp in microseconds for spot
                df['ts'] = pd.to_datetime(df['open_time'] // 1000, unit='ms')
                chunks.append(df[['ts', 'close']])
            except:
                continue
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks).sort_values('ts').drop_duplicates('ts', keep='first').set_index('ts')
        return df
    else:
        df = _load_csv(sym_dir, 'kline_1m', start, end)
        if not df.empty and 'close' in df.columns:
            df = df[['ts', 'close']].set_index('ts')
        return df


def load_binance_metrics(sym, start=START, end=END):
    df = _load_csv(BINANCE / sym, 'metrics', start, end)
    return df


# ============================================================
# DAILY SIGNAL RETURNS
# ============================================================

def compute_daily_signal_returns(alts, start, end):
    """Compute daily aggregated signal returns for Ideas 4 and 5.
    For each day: avg net return of declustered, T+1-delayed signals across all alts.
    """
    print("\n  Loading BTC from both exchanges...")
    btc_bybit = load_bybit_kline('BTCUSDT', start, end)
    btc_binance = load_binance_kline('BTCUSDT', start, end)

    if btc_bybit.empty:
        print("  ❌ No BTC Bybit data")
        return pd.DataFrame()

    btc = btc_bybit.copy()
    btc.columns = ['btc_close']
    btc['btc_ret_3m'] = (btc['btc_close'] / btc['btc_close'].shift(3) - 1) * 10000

    # Also track if Binance BTC agrees (cross-exchange confirmation)
    if not btc_binance.empty:
        btc_bn = btc_binance.copy()
        btc_bn.columns = ['btc_close_bn']
        btc_bn['btc_ret_3m_bn'] = (btc_bn['btc_close_bn'] / btc_bn['btc_close_bn'].shift(3) - 1) * 10000
        btc = btc.join(btc_bn, how='left')

    daily_results = {}  # date → {idea4_rets: [], idea5_rets: []}

    t0 = time.time()
    for ai, sym in enumerate(alts):
        progress_bar(ai, len(alts), prefix='  Signals', start_time=t0)

        # Load from both exchanges
        kf_bybit = load_bybit_kline(sym, start, end)
        kf_binance = load_binance_kline(sym, start, end)
        ks_bybit = load_bybit_kline(sym, start, end, spot=True)
        ks_binance = load_binance_kline(sym, start, end, spot=True)

        for exchange, kf, ks in [('bybit', kf_bybit, ks_bybit), ('binance', kf_binance, ks_binance)]:
            if kf.empty or len(kf) < 5000:
                continue

            k = kf.copy()
            k.columns = ['close']
            k = k.join(btc[['btc_ret_3m']], how='inner')
            if len(k) < 5000:
                continue

            # Forward returns at 240m (T+1 entry)
            k['fwd_240m'] = (k['close'].shift(-240) / k['close'] - 1) * 10000
            k['dfwd_240m'] = k['fwd_240m'].shift(-1)

            # === IDEA 4: BTC pump > 150 bps ===
            sig4 = k['btc_ret_3m'] > 150
            idx4 = np.where(sig4.values)[0]
            if len(idx4) > 0:
                kept4 = [idx4[0]]
                for idx in idx4[1:]:
                    if idx - kept4[-1] >= 30:
                        kept4.append(idx)
                for i in kept4:
                    if i < len(k) and not np.isnan(k['dfwd_240m'].iloc[i]):
                        dt = k.index[i].date()
                        key = (dt, 'idea4')
                        if key not in daily_results:
                            daily_results[key] = []
                        daily_results[key].append(k['dfwd_240m'].iloc[i])

            # === IDEA 5: Spot leads > 40 bps ===
            if not ks.empty and len(ks) > 5000:
                ks2 = ks.copy()
                ks2.columns = ['spot_close']
                k2 = k.join(ks2, how='left')
                if 'spot_close' in k2.columns:
                    k2['spot_ret_3m'] = (k2['spot_close'] / k2['spot_close'].shift(3) - 1) * 10000
                    k2['fut_ret_3m'] = (k2['close'] / k2['close'].shift(3) - 1) * 10000
                    sig5 = ((k2['spot_ret_3m'] - k2['fut_ret_3m']) > 40).fillna(False)
                    idx5 = np.where(sig5.values)[0]
                    if len(idx5) > 0:
                        kept5 = [idx5[0]]
                        for idx in idx5[1:]:
                            if idx - kept5[-1] >= 30:
                                kept5.append(idx)
                        for i in kept5:
                            if i < len(k2) and not np.isnan(k2['dfwd_240m'].iloc[i]):
                                dt = k2.index[i].date()
                                key = (dt, 'idea5')
                                if key not in daily_results:
                                    daily_results[key] = []
                                daily_results[key].append(k2['dfwd_240m'].iloc[i])

    progress_bar(len(alts), len(alts), prefix='  Signals', start_time=t0)

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

    return pd.DataFrame(rows)


# ============================================================
# REGIME FEATURES (daily, dual-exchange)
# ============================================================

def build_daily_regime_features(start, end):
    """Build daily regime features using both exchanges."""
    print("\n  Building regime features...")

    # BTC from Bybit (longer history)
    btc = load_bybit_kline('BTCUSDT', start, end)
    if btc.empty:
        return pd.DataFrame()
    btc.columns = ['btc_close']
    btc_d = btc['btc_close'].resample('1D').last().dropna()
    btc_ret = btc_d.pct_change()

    feat = pd.DataFrame(index=btc_d.index)

    # --- BTC features ---
    feat['btc_vol_7d'] = btc_ret.rolling(7).std() * np.sqrt(365) * 100
    feat['btc_vol_30d'] = btc_ret.rolling(30).std() * np.sqrt(365) * 100
    feat['btc_trend_7d'] = btc_d.pct_change(7) * 100
    feat['btc_trend_30d'] = btc_d.pct_change(30) * 100
    feat['btc_trend_90d'] = btc_d.pct_change(90) * 100
    feat['btc_vol_ratio'] = feat['btc_vol_7d'] / feat['btc_vol_30d'].clip(lower=1)
    # Vol regime: is vol expanding or contracting?
    feat['btc_vol_trend'] = feat['btc_vol_7d'] - feat['btc_vol_30d']

    # --- Alt features (cross-sectional, Bybit) ---
    alt_rets_d = pd.DataFrame()
    alt_close_d = pd.DataFrame()
    for sym in ALTS[:15]:
        kf = load_bybit_kline(sym, start, end)
        if kf.empty or len(kf) < 5000:
            continue
        ad = kf['close'].resample('1D').last().dropna()
        alt_rets_d[sym] = ad.pct_change()
        alt_close_d[sym] = ad

    if len(alt_rets_d.columns) > 3:
        # Cross-sectional dispersion
        feat['alt_dispersion_7d'] = alt_rets_d.rolling(7).std().mean(axis=1) * 100
        feat['alt_dispersion_30d'] = alt_rets_d.rolling(30).std().mean(axis=1) * 100

        # BTC-alt avg correlation
        corrs = []
        for sym in alt_rets_d.columns:
            c = btc_ret.rolling(30).corr(alt_rets_d[sym])
            corrs.append(c)
        feat['btc_alt_corr'] = pd.concat(corrs, axis=1).mean(axis=1)

        # Alt-alt avg correlation (herd behavior)
        from itertools import combinations
        pair_corrs = []
        cols = list(alt_rets_d.columns)[:10]
        for i, j in combinations(range(len(cols)), 2):
            c = alt_rets_d[cols[i]].rolling(30).corr(alt_rets_d[cols[j]])
            pair_corrs.append(c)
        if pair_corrs:
            feat['alt_alt_corr'] = pd.concat(pair_corrs, axis=1).mean(axis=1)

        # Alt avg momentum
        for sym in alt_close_d.columns:
            alt_close_d[f'{sym}_mom30'] = alt_close_d[sym].pct_change(30) * 100
        mom_cols = [c for c in alt_close_d.columns if '_mom30' in c]
        feat['alt_avg_mom_30d'] = alt_close_d[mom_cols].mean(axis=1)

    # --- Funding rate & OI (Bybit) ---
    fr_list = []
    oi_growth_list = []
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']:
        from data_loader import load_funding_rate, load_oi
        fr = load_funding_rate(sym, start, end)
        if not fr.empty and 'fundingRate' in fr.columns:
            fr_s = fr.set_index('ts')['fundingRate'].resample('1D').mean() * 10000
            fr_list.append(fr_s)
        oi = load_oi(sym, start, end)
        if not oi.empty and 'openInterest' in oi.columns:
            oi_d = oi.set_index('ts')['openInterest'].resample('1D').last().dropna()
            oi_growth_list.append(oi_d.pct_change(7) * 100)

    if fr_list:
        feat['avg_fr_bps'] = pd.concat(fr_list, axis=1).mean(axis=1).rolling(7).mean()
    if oi_growth_list:
        feat['avg_oi_growth_7d'] = pd.concat(oi_growth_list, axis=1).mean(axis=1)

    # --- Premium / basis (Bybit) ---
    from data_loader import load_premium
    prem_list = []
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        prem = load_premium(sym, start, end)
        if not prem.empty and 'close' in prem.columns:
            prem_d = prem.set_index('ts')['close'].resample('1D').mean() * 10000
            prem_list.append(prem_d)
    if prem_list:
        feat['avg_premium_bps'] = pd.concat(prem_list, axis=1).mean(axis=1).rolling(7).mean()

    # --- Cross-exchange features (Bybit vs Binance) ---
    btc_bn = load_binance_kline('BTCUSDT', start, end)
    if not btc_bn.empty:
        btc_bn.columns = ['btc_bn_close']
        btc_bn_d = btc_bn['btc_bn_close'].resample('1D').last().dropna()
        # Price spread between exchanges
        aligned = pd.DataFrame({'bybit': btc_d, 'binance': btc_bn_d}).dropna()
        if len(aligned) > 30:
            feat['xexch_spread_bps'] = ((aligned['bybit'] / aligned['binance'] - 1) * 10000).rolling(7).mean()

    # Binance taker buy/sell ratio
    bn_metrics = load_binance_metrics('BTCUSDT', start, end)
    if not bn_metrics.empty and 'sum_taker_long_short_vol_ratio' in bn_metrics.columns:
        taker = bn_metrics.set_index('ts')['sum_taker_long_short_vol_ratio'].resample('1D').mean()
        feat['taker_ratio'] = taker.rolling(7).mean()

    # Binance OI
    if not bn_metrics.empty and 'sum_open_interest_value' in bn_metrics.columns:
        bn_oi = bn_metrics.set_index('ts')['sum_open_interest_value'].resample('1D').last().dropna()
        feat['bn_oi_growth_7d'] = bn_oi.pct_change(7) * 100

    feat = feat.dropna(how='all')
    print(f"  Built {len(feat.columns)} features over {len(feat)} days")
    print(f"  Features: {list(feat.columns)}")
    return feat


# ============================================================
# REGIME FILTER ANALYSIS
# ============================================================

def analyze_regime_filters(feat, daily_rets, signal_name):
    """Find regime features that predict signal profitability, with walk-forward."""
    sig_daily = daily_rets[daily_rets['signal'] == signal_name].set_index('date').sort_index()
    if len(sig_daily) < 30:
        print(f"\n  {signal_name}: only {len(sig_daily)} days with signals — skipping")
        return

    merged = sig_daily.join(feat, how='inner')
    feat_cols = [c for c in feat.columns if c in merged.columns]

    print(f"\n{'─'*75}")
    print(f"  {signal_name.upper()}: {len(merged)} days with signals")
    print(f"  Overall: avg net={merged['net_ret'].mean():+.0f} bps, WR={merged['win_rate'].mean():.0f}%, "
          f"profitable days={(merged['net_ret']>0).mean()*100:.0f}%")
    print(f"{'─'*75}")

    # Correlation with returns
    print(f"\n  {'Feature':<22s} │ {'Corr':>6s} │ {'Prof.days':>9s} │ {'Unprof.days':>11s} │ {'Sep':>6s}")
    print(f"  {'─'*22}─┼─{'─'*6}─┼─{'─'*9}─┼─{'─'*11}─┼─{'─'*6}")

    scored = []
    for feat_name in feat_cols:
        valid = merged[['net_ret', feat_name]].dropna()
        if len(valid) < 30:
            continue
        corr = valid['net_ret'].corr(valid[feat_name])
        prof = valid[valid['net_ret'] > 0][feat_name].mean()
        unprof = valid[valid['net_ret'] <= 0][feat_name].mean()
        std = valid[feat_name].std()
        sep = (prof - unprof) / std if std > 0 else 0

        print(f"  {feat_name:<22s} │ {corr:>+5.3f} │ {prof:>9.2f} │ {unprof:>11.2f} │ {sep:>+5.3f}")
        scored.append((feat_name, corr, sep))

    scored.sort(key=lambda x: abs(x[2]), reverse=True)
    if not scored:
        return

    # ============================================================
    # WALK-FORWARD FILTER (expanding window, min 180 days training)
    # ============================================================
    print(f"\n  WALK-FORWARD REGIME FILTER TESTS (expanding, min 180d train):")

    # Test top 5 individual features + top 2 combinations
    test_configs = []

    # Individual features
    for feat_name, corr, sep in scored[:7]:
        if abs(sep) < 0.1:
            continue
        test_configs.append(('single', feat_name, corr))

    # Pairs of top features
    if len(scored) >= 2:
        for i in range(min(3, len(scored))):
            for j in range(i+1, min(5, len(scored))):
                f1, c1, s1 = scored[i]
                f2, c2, s2 = scored[j]
                if abs(s1) > 0.1 and abs(s2) > 0.1:
                    test_configs.append(('pair', f'{f1}+{f2}', (c1, c2, f1, f2)))

    min_train = 180
    best_lift = -9999
    best_config_name = None
    best_wf_stats = None

    for cfg_type, cfg_name, cfg_data in test_configs:
        # Walk-forward
        all_dates = merged.index.sort_values()
        wf_trades = []
        wf_notrades = []

        for di in range(min_train, len(merged)):
            train = merged.iloc[:di]
            test_row = merged.iloc[di]
            test_date = all_dates[di]

            if cfg_type == 'single':
                feat_name = cfg_name
                if feat_name not in train.columns or pd.isna(test_row.get(feat_name)):
                    continue
                # Find optimal threshold on training data
                train_valid = train[[feat_name, 'net_ret']].dropna()
                if len(train_valid) < 30:
                    continue
                # Use median split
                med = train_valid[feat_name].median()
                above_net = train_valid[train_valid[feat_name] > med]['net_ret'].mean()
                below_net = train_valid[train_valid[feat_name] <= med]['net_ret'].mean()

                if above_net > below_net:
                    should_trade = test_row[feat_name] > med
                else:
                    should_trade = test_row[feat_name] <= med

            elif cfg_type == 'pair':
                c1, c2, f1, f2 = cfg_data
                if f1 not in train.columns or f2 not in train.columns:
                    continue
                if pd.isna(test_row.get(f1)) or pd.isna(test_row.get(f2)):
                    continue
                train_valid = train[[f1, f2, 'net_ret']].dropna()
                if len(train_valid) < 30:
                    continue
                # Simple: both features on profitable side
                med1 = train_valid[f1].median()
                med2 = train_valid[f2].median()
                a1 = train_valid[train_valid[f1] > med1]['net_ret'].mean()
                b1 = train_valid[train_valid[f1] <= med1]['net_ret'].mean()
                a2 = train_valid[train_valid[f2] > med2]['net_ret'].mean()
                b2 = train_valid[train_valid[f2] <= med2]['net_ret'].mean()

                cond1 = test_row[f1] > med1 if a1 > b1 else test_row[f1] <= med1
                cond2 = test_row[f2] > med2 if a2 > b2 else test_row[f2] <= med2
                should_trade = cond1 and cond2

            actual_ret = test_row['net_ret']
            if should_trade:
                wf_trades.append(actual_ret)
            else:
                wf_notrades.append(actual_ret)

        if len(wf_trades) < 10 or len(wf_notrades) < 5:
            continue

        trade_avg = np.mean(wf_trades)
        notrade_avg = np.mean(wf_notrades)
        all_avg = np.mean(wf_trades + wf_notrades)
        lift = trade_avg - notrade_avg
        trade_wr = np.mean([r > 0 for r in wf_trades]) * 100
        notrade_wr = np.mean([r > 0 for r in wf_notrades]) * 100

        verdict = '✅' if trade_avg > 0 and trade_avg > all_avg else '⚠️' if trade_avg > 0 else '❌'

        print(f"\n    {verdict} Filter: {cfg_name}")
        print(f"      TRADE:    {len(wf_trades):>4d} days, avg={trade_avg:>+7.0f} bps, WR={trade_wr:.0f}%")
        print(f"      NO-TRADE: {len(wf_notrades):>4d} days, avg={notrade_avg:>+7.0f} bps, WR={notrade_wr:.0f}%")
        print(f"      Lift: {lift:+.0f} bps | Always: {all_avg:+.0f} | Filtered: {trade_avg:+.0f}")

        if lift > best_lift and trade_avg > 0:
            best_lift = lift
            best_config_name = cfg_name
            best_wf_stats = {
                'trade_n': len(wf_trades), 'trade_avg': trade_avg, 'trade_wr': trade_wr,
                'notrade_n': len(wf_notrades), 'notrade_avg': notrade_avg,
                'all_avg': all_avg, 'lift': lift,
                'wf_trades': wf_trades,
            }

    if best_wf_stats:
        print(f"\n  ★ BEST FILTER for {signal_name}: {best_config_name}")
        print(f"    Lift: {best_wf_stats['lift']:+.0f} bps, Trade avg: {best_wf_stats['trade_avg']:+.0f} bps")

        # Monthly breakdown of best filter
        # Reconstruct monthly from wf_trades
        trade_rets = best_wf_stats['wf_trades']
        if len(trade_rets) > 20:
            chunk = len(trade_rets) // 4
            for qi, qname in enumerate(['Q1 (oldest)', 'Q2', 'Q3', 'Q4 (newest)']):
                q_rets = trade_rets[qi*chunk:(qi+1)*chunk]
                print(f"    {qname}: avg={np.mean(q_rets):+.0f} bps, WR={np.mean([r>0 for r in q_rets])*100:.0f}% ({len(q_rets)} days)")

    return best_wf_stats


def main():
    print("=" * 75)
    print("  REGIME FILTER v2: Daily resolution, dual-exchange")
    print(f"  Period: {START} → {END} (700+ days)")
    print(f"  Exchanges: Bybit + Binance | Coins: {len(ALTS)}")
    print("=" * 75)

    # Step 1: Compute daily signal returns
    print("\n  STEP 1: Daily signal returns across all coins, both exchanges")
    daily_rets = compute_daily_signal_returns(ALTS, START, END)
    daily_rets.to_csv(f'{OUT}/regime_v2_daily_rets.csv', index=False)

    for sig in ['idea4', 'idea5']:
        sub = daily_rets[daily_rets['signal'] == sig]
        if len(sub) > 0:
            print(f"\n  {sig}: {len(sub)} days with signals, "
                  f"avg net={sub['net_ret'].mean():+.0f} bps, "
                  f"WR prof days={(sub['net_ret']>0).mean()*100:.0f}%")

    # Step 2: Build regime features
    print("\n  STEP 2: Build daily regime features")
    features = build_daily_regime_features(START, END)
    features.to_csv(f'{OUT}/regime_v2_features.csv')

    # Step 3: Analyze regime filters
    print("\n  STEP 3: Walk-forward regime filter analysis")

    results = {}
    for sig in ['idea4', 'idea5']:
        r = analyze_regime_filters(features, daily_rets, sig)
        if r:
            results[sig] = r

    # Summary
    print(f"\n\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    for sig, stats in results.items():
        print(f"\n  {sig}: TRADE days avg {stats['trade_avg']:+.0f} bps ({stats['trade_n']} days)")
        print(f"  {'':8s} NO-TRADE avg {stats['notrade_avg']:+.0f} bps ({stats['notrade_n']} days)")
        print(f"  {'':8s} Lift: {stats['lift']:+.0f} bps | Beats always-trade: {stats['trade_avg'] > stats['all_avg']}")

    print(f"\n✅ Saved: regime_v2_daily_rets.csv, regime_v2_features.csv")


if __name__ == '__main__':
    main()
