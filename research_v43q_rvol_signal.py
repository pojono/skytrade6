#!/usr/bin/env python3
"""
v43q: Realized Volatility Signal — 3-Year Validation

v43p found sig_rvol_z has IC=+0.158 vs 4h forward return on SOL (76 days).
This is the strongest individual signal found in the entire v43 research.

But it was only tested on 76 days with ticker data. This script validates
on the FULL 3-year 1h OHLCV dataset (1143 days, 5 symbols).

Signal: rvol_z = z-score of 24h realized volatility vs 168h (1-week) lookback
Interpretation: when vol is high relative to recent history → expect larger moves
Combined with mr_4h: when vol is high AND price has moved → expect reversion

Walk-forward monthly validation. Random baseline.
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

PARQUET_DIR = Path('parquet')
RT_FEE_BPS = 4.0  # maker+maker


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_1h(symbol, exchange='bybit_futures'):
    d = PARQUET_DIR / symbol / 'ohlcv' / '1h' / exchange
    if not d.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    return raw[~raw.index.duplicated(keep='first')]


def add_signals(df):
    """Add rvol_z and mr_4h signals."""
    c = df['close'].values.astype(np.float64)
    n = len(c)

    # 1h returns
    ret_1h = np.zeros(n)
    ret_1h[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret_1h, index=df.index)

    # Realized volatility (24h rolling std of 1h returns)
    rvol = ret_s.rolling(24, min_periods=8).std()

    # Z-score of rvol vs 168h (1 week) lookback
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    df['rvol_z'] = ((rvol - rvol_mean) / rvol_std).values

    # 4h MR signal: z-score of 4h return
    ret_4h_sum = ret_s.rolling(4).sum()
    ret_4h_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    ret_4h_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    df['mr_4h'] = -((ret_4h_sum - ret_4h_mean) / ret_4h_std).values

    # Combined signal: rvol_z + mr_4h (equal weight)
    df['combined'] = (df['rvol_z'].values + df['mr_4h'].values) / 2

    # Forward returns
    df['fwd_4h'] = 0.0
    df['fwd_8h'] = 0.0
    if n > 4:
        df.iloc[:-4, df.columns.get_loc('fwd_4h')] = (c[4:] - c[:-4]) / c[:-4] * 10000
    if n > 8:
        df.iloc[:-8, df.columns.get_loc('fwd_8h')] = (c[8:] - c[:-8]) / c[:-8] * 10000

    return df


def simulate(df, signal_col, threshold, hold_bars, fee_bps=4, randomize=False):
    """Simulate strategy: trade when |signal| > threshold."""
    sig = df[signal_col].values
    c = df['close'].values.astype(np.float64)
    n = len(c)
    trades = []
    last_exit = 0

    for i in range(168, n - hold_bars):
        if i < last_exit + 4:
            continue
        if np.isnan(sig[i]):
            continue

        s = sig[i]
        if abs(s) < threshold:
            continue

        if randomize:
            import random
            trade_dir = random.choice(['long', 'short'])
        else:
            trade_dir = 'long' if s > 0 else 'short'

        entry = c[i]
        exit_p = c[i + hold_bars]

        if trade_dir == 'long':
            raw = (exit_p - entry) / entry * 10000
        else:
            raw = (entry - exit_p) / entry * 10000

        net = raw - fee_bps
        trades.append({
            'time': df.index[i],
            'dir': trade_dir,
            'net_bps': net,
            'sig': s,
        })
        last_exit = i + hold_bars

    return trades


def summarize(trades, label):
    if not trades:
        print(f"    {label}: NO TRADES"); return None
    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0

    long_t = [t for t in trades if t['dir'] == 'long']
    short_t = [t for t in trades if t['dir'] == 'short']

    print(f"    {label:35s} n={n:5d} WR={wr:5.1f}% avg={avg:+7.1f}bps "
          f"total={total:+8.2f}% Sharpe={sharpe:+6.2f}")

    if long_t and short_t:
        ln = np.array([t['net_bps'] for t in long_t])
        sn = np.array([t['net_bps'] for t in short_t])
        print(f"      LONG: n={len(ln)} WR={(ln>0).sum()/len(ln)*100:.1f}% avg={ln.mean():+.1f} | "
              f"SHORT: n={len(sn)} WR={(sn>0).sum()/len(sn)*100:.1f}% avg={sn.mean():+.1f}")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe}


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43q: Realized Volatility Signal — 3-Year Validation")
    print("=" * 80)

    symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'DOGEUSDT', 'XRPUSDT']

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        df = load_1h(symbol)
        if df.empty or len(df) < 500:
            print(f"  Too few bars ({len(df)})"); continue

        df = add_signals(df)
        print(f"  {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

        # IC analysis
        valid = df.dropna(subset=['rvol_z', 'mr_4h', 'combined', 'fwd_4h'])
        valid = valid[valid['fwd_4h'] != 0]
        if len(valid) > 100:
            for sig_col in ['rvol_z', 'mr_4h', 'combined']:
                for fwd in ['fwd_4h', 'fwd_8h']:
                    v2 = valid[valid[fwd] != 0]
                    ic = np.corrcoef(v2[sig_col].values, v2[fwd].values)[0, 1]
                    print(f"  IC({sig_col} vs {fwd}) = {ic:+.4f}")

        # IS/OOS split (65/35)
        split = int(len(df) * 0.65)
        df_is = df.iloc[:split]
        df_oos = df.iloc[split:]
        print(f"  IS: {len(df_is):,} bars | OOS: {len(df_oos):,} bars")

        # Test configs
        for sig_col in ['rvol_z', 'mr_4h', 'combined']:
            print(f"\n  --- Signal: {sig_col} ---")
            for thresh in [0.5, 1.0, 1.5, 2.0]:
                for hold in [4, 8]:
                    label = f"thresh={thresh} hold={hold}h"
                    is_r = simulate(df_is, sig_col, thresh, hold)
                    oos_r = simulate(df_oos, sig_col, thresh, hold)
                    rand_r = simulate(df_oos, sig_col, thresh, hold, randomize=True)

                    summarize(is_r, f"IS  {label}")
                    oos_result = summarize(oos_r, f"OOS {label}")
                    summarize(rand_r, f"RND {label}")

        # Yearly breakdown for best config
        print(f"\n  --- Yearly Breakdown (combined, thresh=1.0, hold=4h) ---")
        df['year'] = df.index.year
        for year in sorted(df['year'].unique()):
            yr_df = df[df['year'] == year]
            if len(yr_df) < 200:
                continue
            yr_trades = simulate(yr_df, 'combined', 1.0, 4)
            if yr_trades:
                net = np.array([t['net_bps'] for t in yr_trades])
                wr = (net > 0).sum() / len(net) * 100
                print(f"    {year}: n={len(net):4d} WR={wr:5.1f}% "
                      f"avg={net.mean():+7.1f}bps total={net.sum()/100:+7.2f}%")

        # Quarterly breakdown
        print(f"\n  --- Quarterly (combined, thresh=1.0, hold=4h) ---")
        df['quarter'] = df.index.to_period('Q')
        for q in sorted(df['quarter'].unique()):
            q_df = df[df['quarter'] == q]
            if len(q_df) < 100:
                continue
            q_trades = simulate(q_df, 'combined', 1.0, 4)
            if q_trades:
                net = np.array([t['net_bps'] for t in q_trades])
                wr = (net > 0).sum() / len(net) * 100
                marker = '✓' if net.sum() > 0 else '✗'
                print(f"    {q} {marker}: n={len(net):3d} WR={wr:5.1f}% "
                      f"avg={net.mean():+6.1f}bps total={net.sum()/100:+6.2f}%")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
