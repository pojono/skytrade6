#!/usr/bin/env python3
"""
v43m: Cascade Momentum — Extended Validation

v43l found cascade momentum (follow forced flow) profitable on SOL 14 days:
  - 72% WR, +18 bps avg, Sharpe 27 (TP=50 SL=100, market entry)

This script validates on:
  1. ALL available dates (not just 14)
  2. All 3 symbols with liquidation data (SOL, ETH, DOGE)
  3. IS/OOS split (first 65% / last 35%)
  4. Rolling 7-day windows for stability
  5. Random baseline (same timing, random direction)

Processes data day-by-day to keep RAM low.
Best configs from v43l: TP=50 SL=100, TP=30 SL=60, TP=100 SL=200
"""

import sys, time, gzip, json, gc, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE = 0.0002
TAKER_FEE = 0.00055
DATA_DIR = Path('data')
PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_liquidations(symbol, date_str):
    liq_dir = DATA_DIR / symbol / 'bybit' / 'liquidations'
    files = sorted(liq_dir.glob(f'liquidation_{date_str}_*.jsonl.gz'))
    events = []
    for f in files:
        try:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    rec = json.loads(line)
                    for item in rec.get('result', {}).get('data', []):
                        events.append({
                            'ts_ms': int(item['T']),
                            'side': item['S'],
                            'qty': float(item['v']),
                            'price': float(item['p']),
                        })
        except Exception:
            continue
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values('ts_ms').reset_index(drop=True)


def detect_cascades(liq_df, window_ms=60000, min_count=5, min_usd=10000):
    if liq_df.empty:
        return []
    ts = liq_df['ts_ms'].values
    sides = liq_df['side'].values
    qtys = liq_df['qty'].values
    prices = liq_df['price'].values
    n = len(ts)
    cascades = []
    i = 0
    last_ts = 0
    while i < n:
        j = i
        while j < n and ts[j] - ts[i] <= window_ms:
            j += 1
        count = j - i
        if count >= min_count:
            total_usd = sum(qtys[k] * prices[k] for k in range(i, j))
            if total_usd >= min_usd:
                buy_count = sum(1 for k in range(i, j) if sides[k] == 'Buy')
                if buy_count > count - buy_count:
                    cascade_dir = 'up'
                else:
                    cascade_dir = 'down'
                c_ts = ts[j - 1]
                if c_ts - last_ts > 300000:
                    cascades.append({
                        'ts_ms': c_ts, 'direction': cascade_dir,
                        'count': count, 'total_usd': total_usd,
                        'price': prices[j - 1],
                    })
                    last_ts = c_ts
                i = j
                continue
        i += 1
    return cascades


def load_ticks(symbol, date_str):
    path = PARQUET_DIR / symbol / 'trades' / 'bybit_futures' / f'{date_str}.parquet'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=['timestamp_us', 'price'])
    df['ts_ms'] = df['timestamp_us'] // 1000
    return df.sort_values('ts_ms').reset_index(drop=True)


def simulate_day(cascades, ticks, tp_bps, sl_bps, timeout_ms=3600000,
                 randomize=False):
    """Simulate cascade momentum trades for one day's data."""
    if ticks.empty or not cascades:
        return []
    tick_ts = ticks['ts_ms'].values
    tick_price = ticks['price'].values
    n_ticks = len(tick_ts)
    trades = []
    last_exit_ts = 0

    for cascade in cascades:
        c_ts = cascade['ts_ms']
        c_dir = cascade['direction']
        if c_ts < last_exit_ts + 300000:
            continue

        # MOMENTUM: follow cascade direction
        if randomize:
            trade_dir = random.choice(['long', 'short'])
        else:
            trade_dir = 'long' if c_dir == 'up' else 'short'

        start_idx = np.searchsorted(tick_ts, c_ts)
        if start_idx + 1 >= n_ticks:
            continue

        # Market entry at next tick
        fill_idx = start_idx + 1
        entry = tick_price[fill_idx]
        fill_ts = tick_ts[fill_idx]
        entry_fee = TAKER_FEE

        # TP/SL
        if trade_dir == 'long':
            tp_p = entry * (1 + tp_bps / 10000)
            sl_p = entry * (1 - sl_bps / 10000)
        else:
            tp_p = entry * (1 - tp_bps / 10000)
            sl_p = entry * (1 + sl_bps / 10000)

        # Exit
        exit_deadline = fill_ts + timeout_ms
        xp = None; xr = None; xt = None

        for ti in range(fill_idx + 1, n_ticks):
            t = tick_ts[ti]; p = tick_price[ti]
            if t > exit_deadline:
                xp = p; xr = 'to'; xt = t; break
            if trade_dir == 'long':
                if p <= sl_p: xp = sl_p; xr = 'sl'; xt = t; break
                if p >= tp_p: xp = tp_p; xr = 'tp'; xt = t; break
            else:
                if p >= sl_p: xp = sl_p; xr = 'sl'; xt = t; break
                if p <= tp_p: xp = tp_p; xr = 'tp'; xt = t; break

        if xp is None:
            xp = tick_price[-1]; xr = 'eod'; xt = tick_ts[-1]

        if trade_dir == 'long':
            raw = (xp - entry) / entry
        else:
            raw = (entry - xp) / entry

        exit_fee = MAKER_FEE if xr == 'tp' else TAKER_FEE
        net = raw - entry_fee - exit_fee

        trades.append({
            'dir': trade_dir, 'xr': xr,
            'net_bps': net * 10000,
            'hold_sec': (xt - fill_ts) / 1000,
        })
        last_exit_ts = xt

    return trades


def summarize(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'avg': 0, 'total': 0}
    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    return {
        'n': n, 'wr': (net > 0).sum() / n * 100,
        'avg': net.mean(), 'total': net.sum() / 100,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43m: Cascade Momentum — Extended Validation")
    print("=" * 80)

    configs = [
        (30, 60,   'TP30/SL60'),
        (50, 100,  'TP50/SL100'),
        (100, 200, 'TP100/SL200'),
        (50, 25,   'TP50/SL25(2:1)'),
    ]

    for symbol in ['SOLUSDT', 'ETHUSDT', 'DOGEUSDT']:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        liq_dir = DATA_DIR / symbol / 'bybit' / 'liquidations'
        trade_dir = PARQUET_DIR / symbol / 'trades' / 'bybit_futures'
        if not liq_dir.exists() or not trade_dir.exists():
            print(f"  Missing data"); continue

        liq_dates = set()
        for f in liq_dir.glob('liquidation_*.jsonl.gz'):
            liq_dates.add(f.stem.replace('.jsonl', '').split('_')[1])
        trade_dates = set(f.stem for f in trade_dir.glob('*.parquet'))
        common = sorted(liq_dates & trade_dates)
        if len(common) < 10:
            print(f"  Only {len(common)} dates"); continue

        print(f"  {len(common)} dates: {common[0]} to {common[-1]}")

        # IS/OOS split
        split = int(len(common) * 0.65)
        is_dates = common[:split]
        oos_dates = common[split:]
        print(f"  IS: {len(is_dates)} days ({is_dates[0]} to {is_dates[-1]})")
        print(f"  OOS: {len(oos_dates)} days ({oos_dates[0]} to {oos_dates[-1]})")

        for tp, sl, cfg_label in configs:
            print(f"\n  --- {cfg_label} ---")

            # Process IS
            is_trades = []
            is_rand_trades = []
            for di, d in enumerate(is_dates):
                liq_df = load_liquidations(symbol, d)
                cascades = detect_cascades(liq_df)
                ticks = load_ticks(symbol, d)
                day_trades = simulate_day(cascades, ticks, tp, sl)
                day_rand = simulate_day(cascades, ticks, tp, sl, randomize=True)
                is_trades.extend(day_trades)
                is_rand_trades.extend(day_rand)
                del ticks
                if (di+1) % 10 == 0 or di == len(is_dates)-1:
                    elapsed = time.time() - t0
                    print(f"    IS [{di+1}/{len(is_dates)}] trades={len(is_trades)} "
                          f"{elapsed:.0f}s RAM={get_ram_mb():.0f}MB", flush=True)

            # Process OOS
            oos_trades = []
            oos_rand_trades = []
            for di, d in enumerate(oos_dates):
                liq_df = load_liquidations(symbol, d)
                cascades = detect_cascades(liq_df)
                ticks = load_ticks(symbol, d)
                day_trades = simulate_day(cascades, ticks, tp, sl)
                day_rand = simulate_day(cascades, ticks, tp, sl, randomize=True)
                oos_trades.extend(day_trades)
                oos_rand_trades.extend(day_rand)
                del ticks
                if (di+1) % 10 == 0 or di == len(oos_dates)-1:
                    elapsed = time.time() - t0
                    print(f"    OOS [{di+1}/{len(oos_dates)}] trades={len(oos_trades)} "
                          f"{elapsed:.0f}s RAM={get_ram_mb():.0f}MB", flush=True)

            # Results
            s_is = summarize(is_trades)
            s_oos = summarize(oos_trades)
            s_rand_is = summarize(is_rand_trades)
            s_rand_oos = summarize(oos_rand_trades)

            print(f"\n  {cfg_label} RESULTS:")
            print(f"    IS:       n={s_is['n']:4d} WR={s_is['wr']:5.1f}% "
                  f"avg={s_is['avg']:+7.1f}bps total={s_is['total']:+7.2f}%")
            print(f"    OOS:      n={s_oos['n']:4d} WR={s_oos['wr']:5.1f}% "
                  f"avg={s_oos['avg']:+7.1f}bps total={s_oos['total']:+7.2f}%")
            print(f"    RAND IS:  n={s_rand_is['n']:4d} WR={s_rand_is['wr']:5.1f}% "
                  f"avg={s_rand_is['avg']:+7.1f}bps total={s_rand_is['total']:+7.2f}%")
            print(f"    RAND OOS: n={s_rand_oos['n']:4d} WR={s_rand_oos['wr']:5.1f}% "
                  f"avg={s_rand_oos['avg']:+7.1f}bps total={s_rand_oos['total']:+7.2f}%")

            # Direction breakdown
            for label, trades in [('IS', is_trades), ('OOS', oos_trades)]:
                for d in ['long', 'short']:
                    dt = [t for t in trades if t['dir'] == d]
                    if dt:
                        dn = np.array([t['net_bps'] for t in dt])
                        print(f"    {label} {d.upper():5s}: n={len(dt)} "
                              f"WR={(dn>0).sum()/len(dn)*100:.1f}% avg={dn.mean():+.1f}bps")

            # Rolling 7-day windows
            window_totals = []
            for wi in range(0, len(common) - 6):
                w_dates = common[wi:wi+7]
                w_trades = []
                for d in w_dates:
                    liq_df = load_liquidations(symbol, d)
                    cascades = detect_cascades(liq_df)
                    ticks = load_ticks(symbol, d)
                    w_trades.extend(simulate_day(cascades, ticks, tp, sl))
                    del ticks
                ws = summarize(w_trades)
                window_totals.append(ws['total'])

            if window_totals:
                pos_w = sum(1 for w in window_totals if w > 0)
                print(f"    Rolling 7d: {pos_w}/{len(window_totals)} positive "
                      f"({pos_w/len(window_totals)*100:.0f}%) "
                      f"avg={np.mean(window_totals):+.2f}%")

            gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
