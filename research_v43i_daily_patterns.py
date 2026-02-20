#!/usr/bin/env python3
"""
v43i: Daily-Level Pattern Search — 3 Years, 5 Symbols

The most fundamental test: are there ANY predictable patterns in daily crypto returns?

At daily timeframe, fees (4 bps RT) are negligible vs daily moves (50-500 bps).
This eliminates the fee wall problem that killed all short-term strategies.

Tests:
  1. Momentum: yesterday up → today up (autocorrelation)
  2. Mean-reversion: yesterday up → today down
  3. Day-of-week effects (Monday, Friday, etc.)
  4. Monthly seasonality
  5. Volatility regime: low vol → momentum, high vol → MR
  6. Multi-day momentum (3d, 5d, 10d lookback)
  7. Volume-price divergence on daily bars
  8. Consecutive up/down day patterns

Data: 1h OHLCV aggregated to daily (1143 days × 5 symbols)
Holding: 1 day (24h) — enter at close, exit at next close
Fees: maker entry + maker exit = 4 bps RT
No trailing stop. No TP/SL. Pure directional bet.
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_BPS = 2.0
PARQUET_DIR = Path('parquet')
RT_FEE_BPS = MAKER_FEE_BPS * 2  # maker + maker = 4 bps


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_daily_bars(symbol, exchange='bybit_futures'):
    """Load 1h OHLCV and aggregate to daily bars."""
    ohlcv_dir = PARQUET_DIR / symbol / 'ohlcv' / '1h' / exchange
    if not ohlcv_dir.exists():
        return pd.DataFrame()

    files = sorted(ohlcv_dir.glob('*.parquet'))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    raw = raw[~raw.index.duplicated(keep='first')]

    # Aggregate to daily
    daily = pd.DataFrame()
    daily['open'] = raw['open'].resample('1D').first()
    daily['high'] = raw['high'].resample('1D').max()
    daily['low'] = raw['low'].resample('1D').min()
    daily['close'] = raw['close'].resample('1D').last()
    daily['volume'] = raw['volume'].resample('1D').sum()
    if 'buy_volume' in raw.columns:
        daily['buy_volume'] = raw['buy_volume'].resample('1D').sum()
        daily['sell_volume'] = raw['sell_volume'].resample('1D').sum()
    if 'trade_count' in raw.columns:
        daily['trade_count'] = raw['trade_count'].resample('1D').sum()

    daily = daily.dropna(subset=['close'])
    return daily


def add_daily_features(df):
    """Add features for daily pattern analysis."""
    c = df['close'].values.astype(np.float64)
    n = len(c)

    # Returns
    df['ret_1d'] = np.concatenate([[0], (c[1:] - c[:-1]) / c[:-1] * 10000])  # bps
    df['ret_1d_pct'] = df['ret_1d'] / 100  # %

    # Multi-day returns
    for d in [2, 3, 5, 10, 20]:
        r = np.zeros(n)
        r[d:] = (c[d:] - c[:-d]) / c[:-d] * 10000
        df[f'ret_{d}d'] = r

    # Realized volatility
    ret_s = pd.Series(df['ret_1d'].values, index=df.index)
    for w in [5, 10, 20]:
        df[f'rvol_{w}d'] = ret_s.rolling(w, min_periods=w//2).std().values

    # Volume features
    if 'buy_volume' in df.columns:
        tv = df['buy_volume'] + df['sell_volume']
        df['vol_imbalance'] = np.where(tv > 0,
            (df['buy_volume'] - df['sell_volume']) / tv, 0)

    # Day of week (0=Mon, 6=Sun)
    df['dow'] = df.index.dayofweek

    # Month
    df['month'] = df.index.month

    # Consecutive up/down days
    sign = np.sign(df['ret_1d'].values)
    consec = np.zeros(n)
    for i in range(1, n):
        if sign[i] == sign[i-1] and sign[i] != 0:
            consec[i] = consec[i-1] + sign[i]
        else:
            consec[i] = sign[i]
    df['consec_days'] = consec

    # Range (high-low) in bps
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000

    # Close position in range
    df['close_pos'] = np.where(df['high'] != df['low'],
        (df['close'] - df['low']) / (df['high'] - df['low']), 0.5)

    return df


def test_strategy(df, signal_col, direction='momentum', threshold=0,
                  hold_days=1, label=''):
    """
    Test a daily strategy.
    Signal: if signal_col > threshold → long (momentum) or short (contrarian)
    Hold: hold_days
    Fees: 4 bps RT (maker+maker)
    """
    sig = df[signal_col].values
    # Forward return (what we'd earn)
    fwd = np.zeros(len(df))
    c = df['close'].values
    for i in range(len(df) - hold_days):
        fwd[i] = (c[i + hold_days] - c[i]) / c[i] * 10000  # bps

    trades = []
    for i in range(20, len(df) - hold_days):
        if np.isnan(sig[i]):
            continue

        if direction == 'momentum':
            if sig[i] > threshold:
                d = 'long'
            elif sig[i] < -threshold:
                d = 'short'
            else:
                continue
        elif direction == 'contrarian':
            if sig[i] > threshold:
                d = 'short'
            elif sig[i] < -threshold:
                d = 'long'
            else:
                continue
        elif direction == 'long_only':
            if sig[i] > threshold:
                d = 'long'
            else:
                continue
        elif direction == 'short_only':
            if sig[i] > threshold:
                d = 'short'
            else:
                continue
        else:
            continue

        raw_bps = fwd[i] if d == 'long' else -fwd[i]
        net_bps = raw_bps - RT_FEE_BPS
        trades.append({'net': net_bps, 'raw': raw_bps, 'dir': d})

    if len(trades) < 20:
        return None

    net = np.array([t['net'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    avg = net.mean()
    total = net.sum() / 100  # bps → %
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0  # daily Sharpe

    return {'label': label, 'n': n, 'wr': wr, 'avg': avg,
            'total': total, 'sharpe': sharpe}


def print_result(r):
    if r is None:
        return
    print(f"    {r['label']:45s} n={r['n']:5d} WR={r['wr']:5.1f}% "
          f"avg={r['avg']:+7.1f}bps total={r['total']:+8.2f}% "
          f"Sharpe={r['sharpe']:+5.2f}")


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43i: Daily Pattern Search — 3 Years × 5 Symbols")
    print("=" * 80)

    symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        df = load_daily_bars(symbol)
        if df.empty or len(df) < 100:
            print(f"  Too few bars ({len(df)})")
            continue

        df = add_daily_features(df)
        print(f"  {len(df)} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
        print(f"  Avg daily return: {df['ret_1d'].mean():+.1f} bps")
        print(f"  Avg daily range: {df['range_bps'].mean():.0f} bps")

        results = []

        # ============================================================
        # TEST 1: Autocorrelation (momentum vs MR)
        # ============================================================
        print(f"\n  --- 1-Day Autocorrelation ---")
        for hold in [1, 2, 3, 5]:
            r = test_strategy(df, 'ret_1d', 'momentum', threshold=0, hold_days=hold,
                              label=f'1d momentum hold={hold}d')
            print_result(r)
            if r: results.append(r)

            r = test_strategy(df, 'ret_1d', 'contrarian', threshold=0, hold_days=hold,
                              label=f'1d contrarian hold={hold}d')
            print_result(r)
            if r: results.append(r)

        # With threshold
        print(f"\n  --- 1-Day with Threshold ---")
        for thresh_bps in [50, 100, 200]:
            for hold in [1, 3]:
                r = test_strategy(df, 'ret_1d', 'momentum', threshold=thresh_bps,
                                  hold_days=hold, label=f'1d mom >{thresh_bps}bps hold={hold}d')
                print_result(r)
                if r: results.append(r)

                r = test_strategy(df, 'ret_1d', 'contrarian', threshold=thresh_bps,
                                  hold_days=hold, label=f'1d MR >{thresh_bps}bps hold={hold}d')
                print_result(r)
                if r: results.append(r)

        # ============================================================
        # TEST 2: Multi-day momentum
        # ============================================================
        print(f"\n  --- Multi-Day Momentum ---")
        for lookback in [3, 5, 10, 20]:
            for hold in [1, 3, 5]:
                r = test_strategy(df, f'ret_{lookback}d', 'momentum', threshold=0,
                                  hold_days=hold, label=f'{lookback}d momentum hold={hold}d')
                print_result(r)
                if r: results.append(r)

        # ============================================================
        # TEST 3: Day-of-week
        # ============================================================
        print(f"\n  --- Day-of-Week Effects ---")
        for dow in range(7):
            dow_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]
            mask = df['dow'] == dow
            fwd_ret = df['ret_1d'].shift(-1)  # next day return
            dow_rets = fwd_ret[mask].dropna()
            if len(dow_rets) > 20:
                avg = dow_rets.mean()
                wr = (dow_rets > 0).sum() / len(dow_rets) * 100
                n = len(dow_rets)
                total = dow_rets.sum() / 100
                print(f"    {dow_name}: n={n:4d} avg={avg:+7.1f}bps WR={wr:5.1f}% "
                      f"total={total:+7.2f}%")

        # ============================================================
        # TEST 4: Consecutive days
        # ============================================================
        print(f"\n  --- Consecutive Day Patterns ---")
        for consec_thresh in [2, 3, 4]:
            # After N consecutive up days → ?
            r = test_strategy(df, 'consec_days', 'momentum', threshold=consec_thresh,
                              hold_days=1, label=f'After {consec_thresh}+ up days → long')
            print_result(r)
            if r: results.append(r)

            r = test_strategy(df, 'consec_days', 'contrarian', threshold=consec_thresh,
                              hold_days=1, label=f'After {consec_thresh}+ up days → short')
            print_result(r)
            if r: results.append(r)

        # ============================================================
        # TEST 5: Volatility regime
        # ============================================================
        print(f"\n  --- Volatility Regime ---")
        # Low vol → momentum, high vol → MR
        rvol = df['rvol_20d'].values
        rvol_median = np.nanmedian(rvol)
        df['low_vol'] = (rvol < rvol_median).astype(float)
        df['high_vol'] = (rvol >= rvol_median).astype(float)

        # In low vol: momentum
        df['low_vol_mom'] = df['ret_1d'] * df['low_vol']
        r = test_strategy(df, 'low_vol_mom', 'momentum', threshold=0,
                          hold_days=1, label='Low vol momentum')
        print_result(r)
        if r: results.append(r)

        # In high vol: MR
        df['high_vol_mr'] = df['ret_1d'] * df['high_vol']
        r = test_strategy(df, 'high_vol_mr', 'contrarian', threshold=0,
                          hold_days=1, label='High vol MR')
        print_result(r)
        if r: results.append(r)

        # ============================================================
        # TEST 6: Volume-price divergence
        # ============================================================
        if 'vol_imbalance' in df.columns:
            print(f"\n  --- Volume-Price Divergence ---")
            r = test_strategy(df, 'vol_imbalance', 'momentum', threshold=0,
                              hold_days=1, label='Vol imbalance momentum 1d')
            print_result(r)
            if r: results.append(r)

            r = test_strategy(df, 'vol_imbalance', 'momentum', threshold=0.05,
                              hold_days=1, label='Vol imbalance mom >0.05 1d')
            print_result(r)
            if r: results.append(r)

        # ============================================================
        # TEST 7: Close position in range
        # ============================================================
        print(f"\n  --- Close Position in Range ---")
        # Close near high → momentum (trend day) or MR?
        df['close_pos_signal'] = df['close_pos'] - 0.5  # center at 0
        r = test_strategy(df, 'close_pos_signal', 'momentum', threshold=0.2,
                          hold_days=1, label='Close near high → long')
        print_result(r)
        if r: results.append(r)

        r = test_strategy(df, 'close_pos_signal', 'contrarian', threshold=0.2,
                          hold_days=1, label='Close near high → short (MR)')
        print_result(r)
        if r: results.append(r)

        # ============================================================
        # SUMMARY for symbol
        # ============================================================
        if results:
            print(f"\n  TOP 5 by avg bps:")
            results.sort(key=lambda x: x['avg'], reverse=True)
            for r in results[:5]:
                print_result(r)

            print(f"\n  BOTTOM 3:")
            for r in results[-3:]:
                print_result(r)

            # Store for cross-symbol analysis
            for r in results:
                r['symbol'] = symbol
            all_results.extend(results)

    # ================================================================
    # CROSS-SYMBOL SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("CROSS-SYMBOL: Configs positive on ≥3 symbols")
    print(f"{'='*80}")

    # Group by label
    by_label = {}
    for r in all_results:
        by_label.setdefault(r['label'], []).append(r)

    consistent = []
    for label, rs in by_label.items():
        pos = sum(1 for r in rs if r['avg'] > 0)
        if pos >= 3 and len(rs) >= 3:
            avg_avg = np.mean([r['avg'] for r in rs])
            consistent.append({'label': label, 'pos': pos, 'total': len(rs),
                                'avg_avg': avg_avg})

    if consistent:
        consistent.sort(key=lambda x: x['avg_avg'], reverse=True)
        for c in consistent:
            print(f"  {c['label']:45s} pos={c['pos']}/{c['total']} "
                  f"avg_avg={c['avg_avg']:+.1f}bps")
    else:
        print("  NONE — no pattern is consistent across ≥3 symbols")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
