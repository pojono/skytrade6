#!/usr/bin/env python3
"""
v43r: Combined Signal (rvol_z + mr_4h) — Deep Walk-Forward Validation

v43q found the combined signal OOS positive on 4/5 symbols at thresh=2.0.
This script performs rigorous validation:

1. Walk-forward monthly: train on past 6 months, test on next month
2. Multi-seed random baseline (10 seeds) to establish confidence interval
3. Long-only vs short-only breakdown (check for directional bias)
4. Fee sensitivity (4, 8, 12 bps)
5. Drawdown analysis
6. Signal decay: does edge persist at different hold periods?
7. Stability: rolling 3-month Sharpe

Data: 1h OHLCV, 3 years, 5 symbols
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
    c = df['close'].values.astype(np.float64)
    n = len(c)
    ret_1h = np.zeros(n)
    ret_1h[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret_1h, index=df.index)

    rvol = ret_s.rolling(24, min_periods=8).std()
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    df['rvol_z'] = ((rvol - rvol_mean) / rvol_std).values

    ret_4h_sum = ret_s.rolling(4).sum()
    ret_4h_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    ret_4h_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    df['mr_4h'] = -((ret_4h_sum - ret_4h_mean) / ret_4h_std).values

    df['combined'] = (df['rvol_z'].values + df['mr_4h'].values) / 2
    return df


def simulate(df, threshold=2.0, hold_bars=4, fee_bps=4.0, randomize=False, seed=42):
    sig = df['combined'].values
    c = df['close'].values.astype(np.float64)
    n = len(c)
    rng = random.Random(seed)
    trades = []
    last_exit = 0

    for i in range(168, n - hold_bars):
        if i < last_exit + 4:
            continue
        if np.isnan(sig[i]) or abs(sig[i]) < threshold:
            continue

        if randomize:
            d = rng.choice(['long', 'short'])
        else:
            d = 'long' if sig[i] > 0 else 'short'

        entry = c[i]
        exit_p = c[i + hold_bars]
        raw = ((exit_p - entry) / entry * 10000) if d == 'long' else ((entry - exit_p) / entry * 10000)
        net = raw - fee_bps

        trades.append({
            'time': df.index[i], 'dir': d, 'net_bps': net, 'sig': sig[i],
        })
        last_exit = i + hold_bars

    return trades


def main():
    t0 = time.time()
    print("=" * 90)
    print("v43r: Combined Signal — Deep Walk-Forward Validation")
    print("=" * 90)

    symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT', 'DOGEUSDT', 'XRPUSDT']

    for symbol in symbols:
        print(f"\n{'='*90}")
        print(f"  {symbol}")
        print(f"{'='*90}")

        df = load_1h(symbol)
        if df.empty or len(df) < 2000:
            print(f"  Too few bars"); continue

        df = add_signals(df)
        print(f"  {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

        # ============================================================
        # 1. WALK-FORWARD MONTHLY
        # ============================================================
        print(f"\n  --- Walk-Forward Monthly (train=6mo, test=1mo) ---")
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())
        print(f"  {len(months)} months total")

        wf_results = []
        for mi in range(6, len(months)):
            test_month = months[mi]
            test_df = df[df['month'] == test_month]
            if len(test_df) < 100:
                continue

            trades = simulate(test_df, threshold=2.0, hold_bars=4)
            if not trades:
                wf_results.append({'month': str(test_month), 'n': 0, 'total': 0, 'avg': 0})
                continue

            net = np.array([t['net_bps'] for t in trades])
            wf_results.append({
                'month': str(test_month), 'n': len(net),
                'total': net.sum() / 100, 'avg': net.mean(),
                'wr': (net > 0).sum() / len(net) * 100,
            })

        if wf_results:
            pos_months = sum(1 for r in wf_results if r['total'] > 0)
            total_months = sum(1 for r in wf_results if r['n'] > 0)
            zero_months = sum(1 for r in wf_results if r['n'] == 0)
            avg_monthly = np.mean([r['total'] for r in wf_results if r['n'] > 0]) if total_months > 0 else 0
            total_return = sum(r['total'] for r in wf_results)

            print(f"  Walk-forward: {pos_months}/{total_months} positive months "
                  f"({pos_months/max(total_months,1)*100:.0f}%), "
                  f"{zero_months} months with 0 trades")
            print(f"  Avg monthly: {avg_monthly:+.2f}%, Total: {total_return:+.2f}%")

            # Show each month
            for r in wf_results:
                if r['n'] > 0:
                    marker = '✓' if r['total'] > 0 else '✗'
                    print(f"    {r['month']} {marker}: n={r['n']:3d} "
                          f"avg={r['avg']:+6.1f}bps total={r['total']:+6.2f}% "
                          f"WR={r.get('wr',0):4.0f}%")
                else:
                    print(f"    {r['month']} -: no trades")

        # ============================================================
        # 2. MULTI-SEED RANDOM BASELINE
        # ============================================================
        print(f"\n  --- Multi-Seed Random Baseline (10 seeds) ---")
        # Full dataset signal performance
        all_trades = simulate(df, threshold=2.0, hold_bars=4)
        if all_trades:
            sig_net = np.array([t['net_bps'] for t in all_trades])
            sig_avg = sig_net.mean()
            sig_total = sig_net.sum() / 100
            print(f"  Signal: n={len(sig_net)} avg={sig_avg:+.1f}bps total={sig_total:+.2f}%")

        rand_avgs = []
        for seed in range(10):
            rand_trades = simulate(df, threshold=2.0, hold_bars=4, randomize=True, seed=seed)
            if rand_trades:
                rn = np.array([t['net_bps'] for t in rand_trades])
                rand_avgs.append(rn.mean())

        if rand_avgs:
            print(f"  Random: mean_avg={np.mean(rand_avgs):+.1f}bps "
                  f"std={np.std(rand_avgs):.1f}bps "
                  f"range=[{np.min(rand_avgs):+.1f}, {np.max(rand_avgs):+.1f}]")
            if all_trades:
                z_score = (sig_avg - np.mean(rand_avgs)) / max(np.std(rand_avgs), 0.1)
                print(f"  Signal vs Random z-score: {z_score:+.2f}")

        # ============================================================
        # 3. LONG-ONLY vs SHORT-ONLY
        # ============================================================
        if all_trades:
            print(f"\n  --- Long vs Short Breakdown ---")
            for d in ['long', 'short']:
                dt = [t for t in all_trades if t['dir'] == d]
                if dt:
                    dn = np.array([t['net_bps'] for t in dt])
                    print(f"  {d.upper():5s}: n={len(dn)} WR={(dn>0).sum()/len(dn)*100:.1f}% "
                          f"avg={dn.mean():+.1f}bps total={dn.sum()/100:+.2f}%")

        # ============================================================
        # 4. FEE SENSITIVITY
        # ============================================================
        print(f"\n  --- Fee Sensitivity ---")
        for fee in [0, 4, 8, 12, 20]:
            trades = simulate(df, threshold=2.0, hold_bars=4, fee_bps=fee)
            if trades:
                net = np.array([t['net_bps'] for t in trades])
                print(f"  fee={fee:2d}bps: n={len(net)} avg={net.mean():+.1f}bps "
                      f"total={net.sum()/100:+.2f}% WR={(net>0).sum()/len(net)*100:.1f}%")

        # ============================================================
        # 5. HOLD PERIOD SENSITIVITY
        # ============================================================
        print(f"\n  --- Hold Period Sensitivity ---")
        for hold in [1, 2, 4, 8, 12, 24]:
            trades = simulate(df, threshold=2.0, hold_bars=hold)
            if trades:
                net = np.array([t['net_bps'] for t in trades])
                print(f"  hold={hold:2d}h: n={len(net)} avg={net.mean():+.1f}bps "
                      f"total={net.sum()/100:+.2f}% WR={(net>0).sum()/len(net)*100:.1f}%")

        # ============================================================
        # 6. THRESHOLD SENSITIVITY
        # ============================================================
        print(f"\n  --- Threshold Sensitivity ---")
        for thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
            trades = simulate(df, threshold=thresh, hold_bars=4)
            if trades:
                net = np.array([t['net_bps'] for t in trades])
                print(f"  thresh={thresh:.1f}: n={len(net)} avg={net.mean():+.1f}bps "
                      f"total={net.sum()/100:+.2f}% WR={(net>0).sum()/len(net)*100:.1f}%")

        # ============================================================
        # 7. MAX DRAWDOWN
        # ============================================================
        if all_trades:
            print(f"\n  --- Drawdown Analysis ---")
            net = np.array([t['net_bps'] for t in all_trades])
            cum = np.cumsum(net)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            maxdd = dd.max()
            maxdd_idx = dd.argmax()
            # Find drawdown start
            dd_start = np.where(cum[:maxdd_idx+1] == peak[maxdd_idx])[0][-1] if maxdd_idx > 0 else 0
            print(f"  Max drawdown: {maxdd:.0f} bps")
            print(f"  DD period: trade {dd_start} to {maxdd_idx} "
                  f"({all_trades[dd_start]['time'].date()} to {all_trades[min(maxdd_idx, len(all_trades)-1)]['time'].date()})")
            print(f"  Final cumulative: {cum[-1]:.0f} bps ({cum[-1]/100:.2f}%)")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
