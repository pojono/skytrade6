#!/usr/bin/env python3
"""
v43j: Multi-Day Momentum — Rigorous Validation

v43i found that 10d momentum (hold 3-5d) is positive on ALL 5 symbols over 3 years.
This script validates whether this is a REAL edge or just buy-and-hold bias.

Critical checks:
  1. Buy-and-hold comparison (is momentum better than just holding?)
  2. Long-only vs Short-only vs Long-Short (is short side profitable?)
  3. Yearly breakdown (does it work every year or just bull years?)
  4. Walk-forward (train on year 1, test on year 2-3)
  5. Bear market performance (2023-Q1, 2024-Q2, 2025-Q1 drawdowns)
  6. Risk-adjusted: Sharpe, Sortino, max drawdown
  7. Transaction cost sensitivity (what if fees are higher?)

Data: 1h OHLCV → daily bars, 1143 days, 5 symbols
Fees: 4 bps RT (maker+maker) baseline, also test 10 bps and 20 bps
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_daily(symbol, exchange='bybit_futures'):
    """Load 1h OHLCV → daily bars."""
    ohlcv_dir = PARQUET_DIR / symbol / 'ohlcv' / '1h' / exchange
    if not ohlcv_dir.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(ohlcv_dir.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    raw = raw[~raw.index.duplicated(keep='first')]

    daily = pd.DataFrame()
    daily['open'] = raw['open'].resample('1D').first()
    daily['high'] = raw['high'].resample('1D').max()
    daily['low'] = raw['low'].resample('1D').min()
    daily['close'] = raw['close'].resample('1D').last()
    daily['volume'] = raw['volume'].resample('1D').sum()
    return daily.dropna(subset=['close'])


def compute_strategy_returns(df, lookback=10, hold=3, fee_bps=4):
    """
    Compute daily strategy returns for momentum strategy.
    
    Signal: if close > close[lookback] → long, else → short
    Hold: hold days (rebalance every hold days)
    Returns a DataFrame with columns: long_ret, short_ret, ls_ret, bah_ret
    """
    close = df['close'].values.astype(np.float64)
    n = len(close)
    dates = df.index

    # Daily returns (for buy-and-hold)
    daily_ret = np.zeros(n)
    daily_ret[1:] = (close[1:] - close[:-1]) / close[:-1] * 10000  # bps

    # Momentum signal: lookback-day return
    mom_signal = np.zeros(n)
    for i in range(lookback, n):
        mom_signal[i] = (close[i] - close[i - lookback]) / close[i - lookback] * 10000

    # Strategy: rebalance every 'hold' days
    # Position: +1 (long) if mom_signal > 0, -1 (short) if < 0
    position = np.zeros(n)
    for i in range(lookback, n):
        if i % hold == 0 or i == lookback:
            position[i] = 1 if mom_signal[i] > 0 else -1
        else:
            position[i] = position[i-1]

    # Strategy returns (next-day return × position, minus fees on rebalance)
    strat_ret = np.zeros(n)
    long_ret = np.zeros(n)
    short_ret = np.zeros(n)

    for i in range(lookback + 1, n):
        ret = daily_ret[i]

        # Fee on rebalance days or position change
        fee = 0
        if position[i] != position[i-1]:
            fee = fee_bps  # round-trip fee

        if position[i-1] > 0:
            long_ret[i] = ret - fee
            strat_ret[i] = ret - fee
        elif position[i-1] < 0:
            short_ret[i] = -ret - fee
            strat_ret[i] = -ret - fee

    result = pd.DataFrame({
        'date': dates,
        'close': close,
        'daily_ret': daily_ret,
        'mom_signal': mom_signal,
        'position': position,
        'long_ret': long_ret,
        'short_ret': short_ret,
        'ls_ret': strat_ret,
        'bah_ret': daily_ret,  # buy-and-hold
    }, index=dates)

    return result


def analyze_returns(rets, label, col='ls_ret'):
    """Analyze a return series."""
    r = rets[col].values
    valid = r[r != 0] if col != 'bah_ret' else r[20:]  # skip warmup

    if len(valid) < 30:
        return None

    n = len(valid)
    total_bps = valid.sum()
    total_pct = total_bps / 100
    avg = valid.mean()
    std = valid.std()
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0

    # Sortino (downside deviation)
    downside = valid[valid < 0]
    down_std = downside.std() if len(downside) > 1 else std
    sortino = avg / down_std * np.sqrt(252) if down_std > 0 else 0

    # Max drawdown
    cum = np.cumsum(valid)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    maxdd = dd.max()

    # Win rate
    wr = (valid > 0).sum() / n * 100

    # Trading days
    n_trades = (np.diff(rets['position'].values) != 0).sum()

    return {
        'label': label, 'n': n, 'total_pct': total_pct, 'avg': avg,
        'sharpe': sharpe, 'sortino': sortino, 'maxdd': maxdd,
        'wr': wr, 'n_trades': n_trades,
    }


def print_analysis(a):
    if a is None:
        return
    print(f"    {a['label']:40s} n={a['n']:5d} total={a['total_pct']:+8.2f}% "
          f"avg={a['avg']:+7.1f}bps Sharpe={a['sharpe']:+5.2f} "
          f"Sortino={a['sortino']:+5.2f} maxDD={a['maxdd']:7.0f}bps "
          f"WR={a['wr']:5.1f}% trades={a['n_trades']}")


def main():
    t0 = time.time()
    print("=" * 90)
    print("v43j: Multi-Day Momentum — Rigorous Validation")
    print("=" * 90)

    symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'DOGEUSDT', 'XRPUSDT']

    # Best configs from v43i
    test_configs = [
        (10, 3, '10d-mom hold=3d'),
        (10, 5, '10d-mom hold=5d'),
        (5, 3,  '5d-mom hold=3d'),
        (3, 3,  '3d-mom hold=3d'),
        (20, 5, '20d-mom hold=5d'),
    ]

    grand_results = []

    for symbol in symbols:
        print(f"\n{'='*90}")
        print(f"  {symbol}")
        print(f"{'='*90}")

        df = load_daily(symbol)
        if df.empty or len(df) < 200:
            print(f"  Too few bars ({len(df)})")
            continue

        print(f"  {len(df)} daily bars ({df.index[0].date()} to {df.index[-1].date()})")
        bah_total = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        print(f"  Buy-and-hold: {bah_total:+.1f}%")

        for lookback, hold, cfg_label in test_configs:
            print(f"\n  --- {cfg_label} ---")

            # ============================================================
            # CHECK 1: Full period, fee sensitivity
            # ============================================================
            print(f"  Full period:")
            for fee in [4, 10, 20]:
                rets = compute_strategy_returns(df, lookback, hold, fee_bps=fee)

                a_ls = analyze_returns(rets, f'L/S fee={fee}bps', 'ls_ret')
                print_analysis(a_ls)

            # Baseline (4 bps)
            rets = compute_strategy_returns(df, lookback, hold, fee_bps=4)

            # Long-only vs Short-only
            a_long = analyze_returns(rets, 'Long-only', 'long_ret')
            a_short = analyze_returns(rets, 'Short-only', 'short_ret')
            a_bah = analyze_returns(rets, 'Buy-and-hold', 'bah_ret')
            print_analysis(a_long)
            print_analysis(a_short)
            print_analysis(a_bah)

            # ============================================================
            # CHECK 2: Yearly breakdown
            # ============================================================
            print(f"  Yearly breakdown:")
            rets['year'] = rets.index.year
            for year in sorted(rets['year'].unique()):
                yr_rets = rets[rets['year'] == year]
                if len(yr_rets) < 30:
                    continue
                yr_ls = yr_rets['ls_ret'].values
                yr_valid = yr_ls[yr_ls != 0]
                if len(yr_valid) < 10:
                    continue
                yr_total = yr_valid.sum() / 100
                yr_avg = yr_valid.mean()
                yr_wr = (yr_valid > 0).sum() / len(yr_valid) * 100
                yr_std = yr_valid.std()
                yr_sharpe = yr_avg / yr_std * np.sqrt(252) if yr_std > 0 else 0

                # Buy-and-hold for same year
                yr_bah = yr_rets['bah_ret'].values[20:]
                yr_bah_total = yr_bah.sum() / 100 if len(yr_bah) > 0 else 0

                print(f"    {year}: L/S total={yr_total:+7.2f}% avg={yr_avg:+6.1f}bps "
                      f"Sharpe={yr_sharpe:+5.2f} WR={yr_wr:5.1f}% | "
                      f"B&H={yr_bah_total:+7.2f}%")

            # ============================================================
            # CHECK 3: Quarterly breakdown
            # ============================================================
            print(f"  Quarterly breakdown:")
            rets['quarter'] = rets.index.to_period('Q')
            for q in sorted(rets['quarter'].unique()):
                q_rets = rets[rets['quarter'] == q]
                q_ls = q_rets['ls_ret'].values
                q_valid = q_ls[q_ls != 0]
                if len(q_valid) < 10:
                    continue
                q_total = q_valid.sum() / 100
                q_avg = q_valid.mean()
                q_wr = (q_valid > 0).sum() / len(q_valid) * 100

                q_bah = q_rets['bah_ret'].values
                q_bah_total = q_bah.sum() / 100

                marker = '✓' if q_total > 0 else '✗'
                print(f"    {q} {marker}: L/S={q_total:+6.2f}% avg={q_avg:+5.1f}bps "
                      f"WR={q_wr:4.0f}% | B&H={q_bah_total:+6.2f}%")

            # ============================================================
            # CHECK 4: Walk-forward (train Y1, test Y2-Y3)
            # ============================================================
            # Not applicable for fixed-param momentum, but we check
            # if the signal works in the SECOND HALF of data
            mid = len(df) // 2
            print(f"  Walk-forward (1st half vs 2nd half):")
            for half_label, half_df in [('1st half', df.iloc[:mid]),
                                         ('2nd half', df.iloc[mid:])]:
                h_rets = compute_strategy_returns(half_df, lookback, hold, fee_bps=4)
                h_a = analyze_returns(h_rets, half_label, 'ls_ret')
                print_analysis(h_a)

            # Store for cross-symbol
            a_full = analyze_returns(rets, f'{symbol} {cfg_label}', 'ls_ret')
            if a_full:
                a_full['symbol'] = symbol
                a_full['config'] = cfg_label
                a_full['bah_pct'] = bah_total
                grand_results.append(a_full)

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{'='*90}")
    print("GRAND SUMMARY")
    print(f"{'='*90}")

    print(f"\n{'Symbol':10s} {'Config':20s} | {'Total%':>8s} {'Avg':>8s} {'Sharpe':>7s} "
          f"{'Sortino':>8s} {'MaxDD':>8s} {'WR':>6s} {'B&H%':>8s} {'Alpha':>8s}")

    for r in grand_results:
        alpha = r['total_pct'] - r['bah_pct']
        print(f"{r['symbol']:10s} {r['config']:20s} | "
              f"{r['total_pct']:+7.2f}% {r['avg']:+7.1f} {r['sharpe']:+6.2f} "
              f"{r['sortino']:+7.2f} {r['maxdd']:7.0f} {r['wr']:5.1f}% "
              f"{r['bah_pct']:+7.1f}% {alpha:+7.2f}%")

    # Check: is alpha (strategy - B&H) positive?
    print(f"\nALPHA CHECK (Strategy total - Buy-and-Hold):")
    for cfg_label in [c[2] for c in test_configs]:
        cfg_results = [r for r in grand_results if r['config'] == cfg_label]
        if not cfg_results:
            continue
        alphas = [r['total_pct'] - r['bah_pct'] for r in cfg_results]
        pos_alpha = sum(1 for a in alphas if a > 0)
        avg_alpha = np.mean(alphas)
        print(f"  {cfg_label:20s}: alpha_pos={pos_alpha}/{len(cfg_results)} "
              f"avg_alpha={avg_alpha:+.2f}%")

    # Check: is SHORT side profitable?
    print(f"\nSHORT SIDE CHECK:")
    for symbol in symbols:
        df = load_daily(symbol)
        if df.empty:
            continue
        rets = compute_strategy_returns(df, 10, 3, fee_bps=4)
        short_valid = rets['short_ret'].values
        short_valid = short_valid[short_valid != 0]
        if len(short_valid) > 10:
            s_total = short_valid.sum() / 100
            s_avg = short_valid.mean()
            s_wr = (short_valid > 0).sum() / len(short_valid) * 100
            print(f"  {symbol:10s} 10d-mom short: n={len(short_valid)} "
                  f"total={s_total:+.2f}% avg={s_avg:+.1f}bps WR={s_wr:.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
