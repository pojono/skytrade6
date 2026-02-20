#!/usr/bin/env python3
"""
Broad Validation: Vol Dip-Buying Strategy on 20 Coins

Walk-forward monthly test on all available symbols.
No lookahead bias — all signals use backward-looking rolling windows.
Fixed parameters: threshold=2.0, hold=4h, fees=4bps RT.
6-month warmup before first trade.

Output: summary table + per-symbol monthly breakdown.
"""

import sys, time, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path('parquet')
RT_FEE_BPS = 4.0
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4

ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
]


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


def compute_signals(df):
    c = df['close'].values.astype(np.float64)
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret, index=df.index)

    rvol = ret_s.rolling(24, min_periods=8).std()
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    df['rvol_z'] = ((rvol - rvol_mean) / rvol_std).values

    r4 = ret_s.rolling(4).sum()
    r4_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    r4_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    df['mr_4h'] = -((r4 - r4_mean) / r4_std).values

    df['combined'] = (df['rvol_z'].values + df['mr_4h'].values) / 2
    return df


def simulate_month(df_slice, randomize=False, seed=42):
    sig = df_slice['combined'].values
    c = df_slice['close'].values.astype(np.float64)
    n = len(c)
    rng = random.Random(seed)
    trades = []
    last_exit = 0

    for i in range(0, n - HOLD_BARS):
        if i < last_exit + COOLDOWN_BARS:
            continue
        if np.isnan(sig[i]) or abs(sig[i]) < THRESHOLD:
            continue

        if randomize:
            trade_dir = rng.choice(['long', 'short'])
        else:
            trade_dir = 'long' if sig[i] > 0 else 'short'

        entry = c[i]
        exit_p = c[i + HOLD_BARS]
        raw_bps = ((exit_p - entry) / entry * 10000) if trade_dir == 'long' else \
                  ((entry - exit_p) / entry * 10000)
        net_bps = raw_bps - RT_FEE_BPS

        trades.append({
            'time': df_slice.index[i], 'dir': trade_dir, 'net_bps': net_bps,
        })
        last_exit = i + HOLD_BARS

    return trades


def main():
    t0 = time.time()
    print("=" * 110)
    print("BROAD VALIDATION — Vol Dip-Buying on 20 Coins")
    print(f"Parameters: threshold={THRESHOLD}, hold={HOLD_BARS}h, fees={RT_FEE_BPS}bps RT, warmup=6 months")
    print("=" * 110)

    summary_rows = []

    for si, symbol in enumerate(ALL_SYMBOLS, 1):
        sym_t0 = time.time()
        df = load_1h(symbol)
        if df.empty or len(df) < 2000:
            print(f"[{si:2d}/20] {symbol:12s} — SKIP (only {len(df)} bars)")
            continue

        df = compute_signals(df)
        n_bars = len(df)
        date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"

        # B&H
        c = df['close'].values
        bah_total = (c[-1] / c[0] - 1) * 100

        # Walk-forward monthly
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())

        monthly_results = []
        for mi in range(6, len(months)):
            m = months[mi]
            mdf = df[df['month'] == m]
            if len(mdf) < 50:
                continue
            if mdf['combined'].notna().sum() < 10:
                continue

            trades = simulate_month(mdf)

            # Random baselines (5 seeds for speed)
            rand_avgs = []
            for seed in range(5):
                rt = simulate_month(mdf, randomize=True, seed=seed)
                if rt:
                    rand_avgs.append(np.mean([t['net_bps'] for t in rt]))

            bah_m = 0
            mc = mdf['close'].values
            if len(mc) > 1:
                bah_m = (mc[-1] / mc[0] - 1) * 100

            if trades:
                net = np.array([t['net_bps'] for t in trades])
                long_n = sum(1 for t in trades if t['dir'] == 'long')
                monthly_results.append({
                    'month': str(m), 'n': len(net), 'total_pct': net.sum() / 100,
                    'avg_bps': net.mean(), 'wr': (net > 0).sum() / len(net) * 100,
                    'long_pct': long_n / len(net) * 100,
                    'rand_avg': np.mean(rand_avgs) if rand_avgs else 0,
                    'bah': bah_m,
                })
            else:
                monthly_results.append({
                    'month': str(m), 'n': 0, 'total_pct': 0, 'avg_bps': 0,
                    'wr': 0, 'long_pct': 0, 'rand_avg': 0, 'bah': bah_m,
                })

        # Aggregate
        active = [r for r in monthly_results if r['n'] > 0]
        zero = [r for r in monthly_results if r['n'] == 0]

        if not active:
            print(f"[{si:2d}/20] {symbol:12s} — NO TRADES")
            continue

        total_pnl = sum(r['total_pct'] for r in active)
        n_trades = sum(r['n'] for r in active)
        pos_months = sum(1 for r in active if r['total_pct'] > 0)
        beat_rand = sum(1 for r in active if r['avg_bps'] > r['rand_avg'])
        n_active = len(active)
        avg_bps = np.mean([r['avg_bps'] for r in active])
        avg_wr = np.mean([r['wr'] for r in active])
        avg_long_pct = np.mean([r['long_pct'] for r in active])

        # Monthly Sharpe
        rets = [r['total_pct'] for r in active]
        m_sharpe = np.mean(rets) / max(np.std(rets), 0.001) * np.sqrt(12)

        # Max drawdown
        cum = np.cumsum(rets)
        peak = np.maximum.accumulate(cum)
        maxdd = (peak - cum).max()

        # Longest losing streak
        streak = 0; max_streak = 0
        for r in active:
            if r['total_pct'] < 0:
                streak += 1; max_streak = max(max_streak, streak)
            else:
                streak = 0

        # Annualized
        n_years = n_active / 12
        ann_ret = total_pnl / max(n_years, 0.5)

        elapsed = time.time() - sym_t0

        print(f"[{si:2d}/20] {symbol:12s} {n_bars:6,} bars | "
              f"WF: {pos_months:2d}/{n_active:2d} pos ({pos_months/n_active*100:4.0f}%) | "
              f"beat_rand: {beat_rand:2d}/{n_active:2d} ({beat_rand/n_active*100:4.0f}%) | "
              f"total: {total_pnl:+7.1f}% | ann: {ann_ret:+6.1f}%/yr | "
              f"mSharpe: {m_sharpe:+5.2f} | DD: {maxdd:5.1f}% | "
              f"trades: {n_trades:4d} | WR: {avg_wr:4.0f}% | "
              f"L%: {avg_long_pct:4.0f}% | B&H: {bah_total:+7.1f}% | {elapsed:.1f}s")

        summary_rows.append({
            'symbol': symbol, 'bars': n_bars, 'date_range': date_range,
            'wf_months': n_active, 'pos_months': pos_months,
            'pos_pct': pos_months / n_active * 100,
            'beat_rand': beat_rand, 'beat_rand_pct': beat_rand / n_active * 100,
            'total_pnl': total_pnl, 'ann_ret': ann_ret,
            'm_sharpe': m_sharpe, 'maxdd': maxdd, 'max_streak': max_streak,
            'n_trades': n_trades, 'avg_bps': avg_bps, 'avg_wr': avg_wr,
            'avg_long_pct': avg_long_pct, 'bah_total': bah_total,
        })

    # ============================================================
    # FINAL SUMMARY TABLE
    # ============================================================
    print(f"\n{'='*110}")
    print(f"  SUMMARY TABLE — Sorted by Monthly Sharpe")
    print(f"{'='*110}")
    print(f"  {'Symbol':12s} {'Months':>7s} {'Pos%':>5s} {'BeatR%':>6s} "
          f"{'Total':>8s} {'Ann':>7s} {'mSharpe':>8s} {'MaxDD':>6s} "
          f"{'Streak':>6s} {'Trades':>6s} {'AvgBps':>7s} {'WR':>5s} "
          f"{'Long%':>5s} {'B&H':>8s}")

    for r in sorted(summary_rows, key=lambda x: x['m_sharpe'], reverse=True):
        print(f"  {r['symbol']:12s} {r['pos_months']:3d}/{r['wf_months']:2d} "
              f"{r['pos_pct']:4.0f}% {r['beat_rand_pct']:5.0f}% "
              f"{r['total_pnl']:+7.1f}% {r['ann_ret']:+6.1f}% "
              f"{r['m_sharpe']:+7.2f} {r['maxdd']:5.1f}% "
              f"{r['max_streak']:5d} {r['n_trades']:5d} "
              f"{r['avg_bps']:+6.1f} {r['avg_wr']:4.0f}% "
              f"{r['avg_long_pct']:4.0f}% {r['bah_total']:+7.1f}%")

    # Aggregate stats
    n_positive = sum(1 for r in summary_rows if r['total_pnl'] > 0)
    n_total = len(summary_rows)
    n_strong = sum(1 for r in summary_rows if r['m_sharpe'] > 0.5)
    n_beat_rand_majority = sum(1 for r in summary_rows if r['beat_rand_pct'] > 50)

    print(f"\n  Symbols with positive total: {n_positive}/{n_total}")
    print(f"  Symbols with mSharpe > 0.5: {n_strong}/{n_total}")
    print(f"  Symbols beating random >50% of months: {n_beat_rand_majority}/{n_total}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
