#!/usr/bin/env python3
"""
Walk-Forward Audit of Volatility Dip-Buying Strategy

CRITICAL: No lookahead bias. At each month t:
  - Signal z-scores use ONLY data up to month t (rolling windows)
  - No future data leaks into signal calculation
  - Parameters (threshold=2.0, hold=4h) are FIXED — not optimized per period

Walk-forward protocol:
  1. First 6 months (Jan-Jun 2023): warmup only, no trades
  2. From month 7 onward: compute signal using trailing data, trade, record PnL
  3. Report month-by-month returns for each symbol
  4. Multi-seed random baseline per month
  5. Comprehensive self-audit with pros/cons

Output: Monthly returns table, equity curves, drawdown analysis, audit report.
"""

import sys, time, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

PARQUET_DIR = Path('parquet')
RT_FEE_BPS = 4.0  # maker+maker
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4
WARMUP_BARS = 480  # 20 days for all rolling windows to stabilize


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


def compute_signals_no_lookahead(df):
    """
    Compute signals using ONLY past data at each point.
    All rolling windows are backward-looking by construction:
      - rolling(N) in pandas uses the N most recent values (no future)
      - This is inherently causal / no lookahead
    """
    c = df['close'].values.astype(np.float64)
    n = len(c)

    # 1h returns (backward-looking diff)
    ret = np.zeros(n)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret, index=df.index)

    # Realized volatility: std of last 24 1h returns
    rvol = ret_s.rolling(24, min_periods=8).std()

    # Z-score of rvol vs trailing 168h (1 week) — BACKWARD ONLY
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    df['rvol_z'] = ((rvol - rvol_mean) / rvol_std).values

    # 4h MR: z-score of trailing 4h return vs trailing 48h stats
    r4_sum = ret_s.rolling(4).sum()
    r4_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    r4_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    df['mr_4h'] = -((r4_sum - r4_mean) / r4_std).values

    # Combined signal
    df['combined'] = (df['rvol_z'].values + df['mr_4h'].values) / 2

    # Regime features (for filtered version)
    c_s = pd.Series(c, index=df.index)
    rolling_high_720 = c_s.rolling(720, min_periods=360).max()
    df['dd_30d'] = np.where(rolling_high_720 > 0,
                             (c - rolling_high_720.values) / rolling_high_720.values * 100, 0)

    return df


def simulate_month(df_slice, use_regime_filter=False, randomize=False, seed=42):
    """
    Simulate trades within a single month slice.
    df_slice must already have signals computed from full history.
    """
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

        # Regime filter: skip longs when DD > 15%
        if use_regime_filter and not randomize:
            dd = df_slice['dd_30d'].iloc[i]
            if trade_dir == 'long' and dd < -15:
                continue

        entry = c[i]
        exit_p = c[i + HOLD_BARS]

        if trade_dir == 'long':
            raw_bps = (exit_p - entry) / entry * 10000
        else:
            raw_bps = (entry - exit_p) / entry * 10000

        net_bps = raw_bps - RT_FEE_BPS

        trades.append({
            'entry_time': df_slice.index[i],
            'exit_time': df_slice.index[i + HOLD_BARS],
            'dir': trade_dir,
            'entry_price': entry,
            'exit_price': exit_p,
            'raw_bps': raw_bps,
            'net_bps': net_bps,
            'signal': sig[i],
        })
        last_exit = i + HOLD_BARS

    return trades


def main():
    t0 = time.time()
    print("=" * 100)
    print("WALK-FORWARD AUDIT — Volatility Dip-Buying Strategy")
    print("No lookahead bias: all signals use only past data")
    print(f"Parameters: threshold={THRESHOLD}, hold={HOLD_BARS}h, fees={RT_FEE_BPS}bps RT")
    print("=" * 100)

    symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*100}")
        print(f"  {symbol}")
        print(f"{'='*100}")

        df = load_1h(symbol)
        if df.empty or len(df) < 2000:
            print(f"  Too few bars ({len(df)})"); continue

        # Compute signals on FULL dataset — but this is NOT lookahead because
        # all rolling windows are backward-looking by construction
        df = compute_signals_no_lookahead(df)
        print(f"  {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")
        print(f"  Signal NaN count: {df['combined'].isna().sum()} "
              f"(first valid: {df['combined'].first_valid_index()})")

        # Split into months
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())
        print(f"  {len(months)} months total, first 6 = warmup")

        # B&H monthly returns for comparison
        monthly_bah = {}
        for m in months:
            mdf = df[df['month'] == m]
            mc = mdf['close'].values
            if len(mc) > 1:
                monthly_bah[str(m)] = (mc[-1] / mc[0] - 1) * 100

        # ============================================================
        # WALK-FORWARD: trade from month 7 onward
        # ============================================================
        monthly_results = []
        monthly_results_filtered = []

        for mi in range(6, len(months)):
            m = months[mi]
            mdf = df[df['month'] == m]

            if len(mdf) < 50:
                continue

            # Check that signals are valid (warmup complete)
            valid_sigs = mdf['combined'].notna().sum()
            if valid_sigs < 10:
                continue

            # Unfiltered
            trades = simulate_month(mdf, use_regime_filter=False)
            # Filtered (DD>15%)
            trades_filt = simulate_month(mdf, use_regime_filter=True)

            # Random baselines (10 seeds)
            rand_avgs = []
            for seed in range(10):
                rt = simulate_month(mdf, randomize=True, seed=seed)
                if rt:
                    rand_avgs.append(np.mean([t['net_bps'] for t in rt]))

            bah = monthly_bah.get(str(m), 0)

            if trades:
                net = np.array([t['net_bps'] for t in trades])
                long_n = sum(1 for t in trades if t['dir'] == 'long')
                short_n = len(trades) - long_n
                monthly_results.append({
                    'month': str(m), 'n': len(net), 'total_pct': net.sum() / 100,
                    'avg_bps': net.mean(), 'wr': (net > 0).sum() / len(net) * 100,
                    'long': long_n, 'short': short_n, 'bah': bah,
                    'rand_avg': np.mean(rand_avgs) if rand_avgs else 0,
                })
            else:
                monthly_results.append({
                    'month': str(m), 'n': 0, 'total_pct': 0, 'avg_bps': 0,
                    'wr': 0, 'long': 0, 'short': 0, 'bah': bah, 'rand_avg': 0,
                })

            if trades_filt:
                net_f = np.array([t['net_bps'] for t in trades_filt])
                monthly_results_filtered.append({
                    'month': str(m), 'n': len(net_f), 'total_pct': net_f.sum() / 100,
                    'avg_bps': net_f.mean(), 'wr': (net_f > 0).sum() / len(net_f) * 100,
                })
            else:
                monthly_results_filtered.append({
                    'month': str(m), 'n': 0, 'total_pct': 0, 'avg_bps': 0, 'wr': 0,
                })

        # ============================================================
        # PRINT MONTH-BY-MONTH TABLE
        # ============================================================
        print(f"\n  {'Month':10s} {'n':>4s} {'Total%':>8s} {'Avg bps':>8s} {'WR%':>6s} "
              f"{'L/S':>5s} {'B&H%':>8s} {'Rand':>6s} {'vs Rand':>8s} {'Filt%':>8s}")
        print(f"  {'-'*80}")

        cum_pnl = 0
        cum_filt = 0
        pos_months = 0
        neg_months = 0
        zero_months = 0
        beat_random = 0
        beat_bah = 0

        for i, r in enumerate(monthly_results):
            rf = monthly_results_filtered[i] if i < len(monthly_results_filtered) else {'total_pct': 0, 'n': 0}

            if r['n'] > 0:
                cum_pnl += r['total_pct']
                cum_filt += rf['total_pct']
                marker = '✓' if r['total_pct'] > 0 else '✗'
                vs_rand = r['avg_bps'] - r['rand_avg']
                if r['total_pct'] > 0: pos_months += 1
                else: neg_months += 1
                if r['avg_bps'] > r['rand_avg']: beat_random += 1
                if r['total_pct'] > r['bah']: beat_bah += 1

                print(f"  {r['month']:10s} {r['n']:4d} {r['total_pct']:+7.2f}% "
                      f"{r['avg_bps']:+7.1f} {r['wr']:5.1f}% "
                      f"{r['long']:2d}/{r['short']:<2d} {r['bah']:+7.1f}% "
                      f"{r['rand_avg']:+5.1f} {vs_rand:+7.1f} "
                      f"{rf['total_pct']:+7.2f}% {marker}")
            else:
                zero_months += 1
                print(f"  {r['month']:10s}    0    0.00%     0.0   0.0%  0/0  "
                      f"{r['bah']:+7.1f}%   0.0     0.0    0.00% -")

        total_months = pos_months + neg_months
        print(f"\n  {'SUMMARY':10s}")
        print(f"  Total months: {total_months} active, {zero_months} zero-trade")
        print(f"  Positive: {pos_months}/{total_months} ({pos_months/max(total_months,1)*100:.0f}%)")
        print(f"  Beat random: {beat_random}/{total_months} ({beat_random/max(total_months,1)*100:.0f}%)")
        print(f"  Cumulative PnL: {cum_pnl:+.2f}%")
        print(f"  Cumulative PnL (filtered): {cum_filt:+.2f}%")

        # ============================================================
        # EQUITY CURVE & DRAWDOWN
        # ============================================================
        active = [r for r in monthly_results if r['n'] > 0]
        if active:
            cum = np.cumsum([r['total_pct'] for r in active])
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            maxdd = dd.max()
            maxdd_idx = dd.argmax()
            maxdd_month = active[maxdd_idx]['month'] if maxdd_idx < len(active) else '?'

            # Longest losing streak
            streak = 0; max_streak = 0
            for r in active:
                if r['total_pct'] < 0:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0

            # Annualized return
            n_years = total_months / 12
            ann_ret = cum_pnl / max(n_years, 0.5)

            # Monthly Sharpe
            rets = [r['total_pct'] for r in active]
            monthly_sharpe = np.mean(rets) / max(np.std(rets), 0.001) * np.sqrt(12)

            print(f"\n  Max drawdown: {maxdd:.2f}% (at {maxdd_month})")
            print(f"  Longest losing streak: {max_streak} months")
            print(f"  Annualized return: {ann_ret:+.1f}%/yr")
            print(f"  Monthly Sharpe: {monthly_sharpe:+.2f}")

        # ============================================================
        # YEARLY SUMMARY
        # ============================================================
        print(f"\n  Yearly Summary:")
        yearly = defaultdict(lambda: {'pnl': 0, 'n': 0, 'months': 0, 'pos': 0})
        for r in monthly_results:
            yr = r['month'][:4]
            yearly[yr]['pnl'] += r['total_pct']
            yearly[yr]['n'] += r['n']
            if r['n'] > 0:
                yearly[yr]['months'] += 1
                if r['total_pct'] > 0:
                    yearly[yr]['pos'] += 1

        for yr in sorted(yearly.keys()):
            y = yearly[yr]
            if y['months'] > 0:
                print(f"    {yr}: {y['pnl']:+7.2f}% ({y['n']:3d} trades, "
                      f"{y['pos']}/{y['months']} pos months)")

        all_results[symbol] = monthly_results

    # ============================================================
    # CROSS-SYMBOL SUMMARY
    # ============================================================
    print(f"\n{'='*100}")
    print(f"  CROSS-SYMBOL SUMMARY")
    print(f"{'='*100}")
    print(f"  {'Symbol':10s} {'Months':>7s} {'Pos%':>6s} {'Total%':>8s} {'Ann%':>7s} "
          f"{'mSharpe':>8s} {'Trades':>7s}")

    for sym in symbols:
        if sym not in all_results:
            continue
        active = [r for r in all_results[sym] if r['n'] > 0]
        if not active:
            continue
        total = sum(r['total_pct'] for r in active)
        pos = sum(1 for r in active if r['total_pct'] > 0)
        n_trades = sum(r['n'] for r in active)
        n_months = len(active)
        rets = [r['total_pct'] for r in active]
        ms = np.mean(rets) / max(np.std(rets), 0.001) * np.sqrt(12)
        ann = total / max(n_months / 12, 0.5)
        print(f"  {sym:10s} {n_months:4d}/{len(all_results[sym]):2d} {pos/n_months*100:5.0f}% "
              f"{total:+7.2f}% {ann:+6.1f}% {ms:+7.2f} {n_trades:6d}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
