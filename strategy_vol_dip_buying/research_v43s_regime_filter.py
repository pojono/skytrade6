#!/usr/bin/env python3
"""
v43s: Regime-Filtered Volatility Dip-Buying

v43r showed the combined signal (rvol_z + mr_4h) is real but fails in
sustained downtrends. This script adds regime filters to skip trades
when the medium-term trend is strongly negative.

Regime filters tested:
  1. SMA trend: skip longs when price < SMA(N)
  2. Momentum filter: skip longs when 20d return < -X%
  3. Drawdown filter: skip longs when price is >X% below 30d high
  4. Combined: multiple filters together

Also tests: allowing SHORT trades when trend is negative (reverse the bias).

Walk-forward monthly validation on all 5 symbols.
"""

import sys, time, random
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path('parquet')
RT_FEE_BPS = 4.0


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_1h(symbol):
    d = PARQUET_DIR / symbol / 'ohlcv' / '1h' / 'bybit_futures'
    if not d.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    return raw[~raw.index.duplicated(keep='first')]


def add_all_features(df):
    c = df['close'].values.astype(np.float64)
    n = len(c)
    ret = np.zeros(n); ret[1:] = np.diff(c) / c[:-1] * 10000
    ret_s = pd.Series(ret, index=df.index)

    # Core signals
    rvol = ret_s.rolling(24, min_periods=8).std()
    df['rvol_z'] = ((rvol - rvol.rolling(168, min_periods=48).mean()) /
                     rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)).values
    r4 = ret_s.rolling(4).sum()
    df['mr_4h'] = -((r4 - ret_s.rolling(48, min_periods=12).mean()*4) /
                     (ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8)*2)).values
    df['combined'] = (df['rvol_z'].values + df['mr_4h'].values) / 2

    # Regime features
    c_s = pd.Series(c, index=df.index)

    # SMA filters
    for w in [120, 240, 480]:  # 5d, 10d, 20d in hours
        df[f'sma_{w}'] = c_s.rolling(w, min_periods=w//2).mean().values

    # Momentum (return over N hours)
    for w in [120, 240, 480]:
        mom = np.zeros(n)
        mom[w:] = (c[w:] - c[:-w]) / c[:-w] * 100
        df[f'mom_{w}h'] = mom

    # Drawdown from rolling high
    for w in [240, 720]:  # 10d, 30d
        rolling_high = c_s.rolling(w, min_periods=w//2).max().values
        dd = np.where(rolling_high > 0, (c - rolling_high) / rolling_high * 100, 0)
        df[f'dd_{w}h'] = dd

    return df


def simulate_filtered(df, threshold=2.0, hold_bars=4, fee_bps=4.0,
                       regime_filter=None, filter_params=None,
                       randomize=False, seed=42):
    """
    Simulate with optional regime filter.
    regime_filter: 'sma', 'momentum', 'drawdown', 'combined', None
    """
    sig = df['combined'].values
    c = df['close'].values.astype(np.float64)
    n = len(c)
    rng = random.Random(seed)
    trades = []
    last_exit = 0
    skipped = 0

    for i in range(480, n - hold_bars):  # warmup for regime features
        if i < last_exit + 4:
            continue
        if np.isnan(sig[i]) or abs(sig[i]) < threshold:
            continue

        if randomize:
            trade_dir = rng.choice(['long', 'short'])
        else:
            trade_dir = 'long' if sig[i] > 0 else 'short'

        # Apply regime filter
        if regime_filter and not randomize:
            skip = False

            if regime_filter == 'sma':
                sma_col = f'sma_{filter_params.get("window", 240)}'
                if sma_col in df.columns:
                    sma_val = df[sma_col].iloc[i]
                    if not np.isnan(sma_val):
                        if trade_dir == 'long' and c[i] < sma_val:
                            skip = True
                        elif trade_dir == 'short' and c[i] > sma_val:
                            skip = True

            elif regime_filter == 'momentum':
                mom_col = f'mom_{filter_params.get("window", 240)}h'
                thresh_pct = filter_params.get('thresh_pct', -10)
                if mom_col in df.columns:
                    mom_val = df[mom_col].iloc[i]
                    if trade_dir == 'long' and mom_val < thresh_pct:
                        skip = True
                    elif trade_dir == 'short' and mom_val > -thresh_pct:
                        skip = True

            elif regime_filter == 'drawdown':
                dd_col = f'dd_{filter_params.get("window", 720)}h'
                dd_thresh = filter_params.get('dd_thresh', -15)
                if dd_col in df.columns:
                    dd_val = df[dd_col].iloc[i]
                    if trade_dir == 'long' and dd_val < dd_thresh:
                        skip = True

            elif regime_filter == 'combined':
                # Skip longs when: price < SMA(240) AND 20d momentum < -10%
                sma_val = df['sma_240'].iloc[i] if 'sma_240' in df.columns else np.nan
                mom_val = df['mom_480h'].iloc[i] if 'mom_480h' in df.columns else 0
                if trade_dir == 'long':
                    if not np.isnan(sma_val) and c[i] < sma_val and mom_val < -10:
                        skip = True

            elif regime_filter == 'adaptive':
                # In downtrend: flip to short (contrarian becomes momentum)
                sma_val = df['sma_240'].iloc[i] if 'sma_240' in df.columns else np.nan
                if not np.isnan(sma_val):
                    if c[i] < sma_val and trade_dir == 'long':
                        trade_dir = 'short'  # flip direction in downtrend

            if skip:
                skipped += 1
                continue

        entry = c[i]
        exit_p = c[i + hold_bars]
        raw = ((exit_p - entry) / entry * 10000) if trade_dir == 'long' else \
              ((entry - exit_p) / entry * 10000)
        net = raw - fee_bps

        trades.append({
            'time': df.index[i], 'dir': trade_dir, 'net_bps': net,
        })
        last_exit = i + hold_bars

    return trades, skipped


def summarize(trades, label, skipped=0):
    if not trades:
        print(f"    {label}: NO TRADES (skipped={skipped})"); return None
    net = np.array([t['net_bps'] for t in trades])
    n_t = len(net)
    wr = (net > 0).sum() / n_t * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n_t > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0

    long_n = sum(1 for t in trades if t['dir'] == 'long')
    short_n = n_t - long_n

    # Max drawdown
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    maxdd = (peak - cum).max()

    print(f"    {label:40s} n={n_t:4d} WR={wr:5.1f}% avg={avg:+7.1f}bps "
          f"total={total:+7.2f}% Sharpe={sharpe:+5.2f} DD={maxdd:.0f}bps "
          f"L/S={long_n}/{short_n} skip={skipped}")
    return {'n': n_t, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe, 'maxdd': maxdd}


def walk_forward_monthly(df, regime_filter=None, filter_params=None,
                          threshold=2.0, hold_bars=4):
    """Walk-forward monthly test."""
    df['month'] = df.index.to_period('M')
    months = sorted(df['month'].unique())
    results = []
    for mi in range(6, len(months)):
        test_df = df[df['month'] == months[mi]]
        if len(test_df) < 100:
            continue
        trades, _ = simulate_filtered(test_df, threshold, hold_bars,
                                       regime_filter=regime_filter,
                                       filter_params=filter_params)
        if trades:
            net = np.array([t['net_bps'] for t in trades])
            results.append({'month': str(months[mi]), 'n': len(net),
                            'total': net.sum() / 100, 'avg': net.mean()})
        else:
            results.append({'month': str(months[mi]), 'n': 0, 'total': 0, 'avg': 0})
    return results


def main():
    t0 = time.time()
    print("=" * 90)
    print("v43s: Regime-Filtered Volatility Dip-Buying")
    print("=" * 90)

    symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'DOGEUSDT', 'XRPUSDT']

    filter_configs = [
        (None, None, 'No filter (baseline)'),
        ('sma', {'window': 240}, 'SMA(10d): skip long below SMA'),
        ('sma', {'window': 480}, 'SMA(20d): skip long below SMA'),
        ('momentum', {'window': 480, 'thresh_pct': -10}, 'Mom(20d): skip long if <-10%'),
        ('momentum', {'window': 480, 'thresh_pct': -5}, 'Mom(20d): skip long if <-5%'),
        ('momentum', {'window': 240, 'thresh_pct': -10}, 'Mom(10d): skip long if <-10%'),
        ('drawdown', {'window': 720, 'dd_thresh': -15}, 'DD(30d): skip long if DD>15%'),
        ('drawdown', {'window': 720, 'dd_thresh': -10}, 'DD(30d): skip long if DD>10%'),
        ('combined', None, 'Combined: below SMA(10d) AND mom(20d)<-10%'),
        ('adaptive', None, 'Adaptive: flip to short in downtrend'),
    ]

    for symbol in symbols:
        print(f"\n{'='*90}")
        print(f"  {symbol}")
        print(f"{'='*90}")

        df = load_1h(symbol)
        if df.empty or len(df) < 2000:
            print(f"  Too few bars"); continue

        df = add_all_features(df)
        print(f"  {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

        # IS/OOS split
        split = int(len(df) * 0.65)
        df_is = df.iloc[:split]
        df_oos = df.iloc[split:]

        for filt, params, label in filter_configs:
            # Full dataset
            trades, skipped = simulate_filtered(df, regime_filter=filt,
                                                 filter_params=params)
            summarize(trades, f"FULL {label}", skipped)

            # OOS only
            trades_oos, skip_oos = simulate_filtered(df_oos, regime_filter=filt,
                                                      filter_params=params)
            summarize(trades_oos, f"OOS  {label}", skip_oos)

            # Random baseline for OOS
            rand_trades, _ = simulate_filtered(df_oos, regime_filter=filt,
                                                filter_params=params,
                                                randomize=True, seed=42)
            summarize(rand_trades, f"RAND {label}")

        # Walk-forward for best configs
        print(f"\n  --- Walk-Forward Monthly ---")
        for filt, params, label in [(None, None, 'No filter'),
                                      ('sma', {'window': 240}, 'SMA(10d)'),
                                      ('momentum', {'window': 480, 'thresh_pct': -10}, 'Mom(20d)<-10%'),
                                      ('drawdown', {'window': 720, 'dd_thresh': -15}, 'DD(30d)>15%'),
                                      ('combined', None, 'Combined')]:
            wf = walk_forward_monthly(df, filt, params)
            active = [r for r in wf if r['n'] > 0]
            if active:
                pos = sum(1 for r in active if r['total'] > 0)
                total = sum(r['total'] for r in active)
                print(f"    {label:25s}: {pos}/{len(active)} pos months ({pos/len(active)*100:.0f}%) "
                      f"total={total:+.2f}%")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
