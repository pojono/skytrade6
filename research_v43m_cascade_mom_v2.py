#!/usr/bin/env python3
"""
v43m v2: Cascade Momentum — Optimized Validation

Optimized approach: 
  1. First pass: detect ALL cascades from lightweight liquidation data
  2. Second pass: load tick data ONLY for days with cascades
  3. Process one day at a time, simulate trades, free memory

Tests best configs from v43l on all available dates with IS/OOS split.
Also runs random baseline to verify signal is real.
"""

import sys, time, gzip, json, gc, random
import numpy as np
import pandas as pd
from pathlib import Path

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


# ============================================================================
# PHASE 1: Detect all cascades (lightweight — only liquidation data)
# ============================================================================

def load_day_liquidations(symbol, date_str):
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
    return events


def detect_cascades_from_events(events, window_ms=60000, min_count=5, min_usd=10000):
    if not events:
        return []
    events.sort(key=lambda x: x['ts_ms'])
    n = len(events)
    cascades = []
    i = 0
    last_ts = 0
    while i < n:
        j = i
        while j < n and events[j]['ts_ms'] - events[i]['ts_ms'] <= window_ms:
            j += 1
        count = j - i
        if count >= min_count:
            total_usd = sum(events[k]['qty'] * events[k]['price'] for k in range(i, j))
            if total_usd >= min_usd:
                buy_count = sum(1 for k in range(i, j) if events[k]['side'] == 'Buy')
                cascade_dir = 'up' if buy_count > count - buy_count else 'down'
                c_ts = events[j-1]['ts_ms']
                if c_ts - last_ts > 300000:
                    cascades.append({
                        'ts_ms': c_ts, 'direction': cascade_dir,
                        'count': count, 'total_usd': total_usd,
                        'price': events[j-1]['price'], 'date': None,
                    })
                    last_ts = c_ts
                i = j; continue
        i += 1
    return cascades


def get_all_cascades(symbol, dates):
    """Detect cascades across all dates. Very fast (liquidation data is tiny)."""
    all_cascades = {}  # date → list of cascades
    total = 0
    t0 = time.time()
    for di, d in enumerate(dates):
        events = load_day_liquidations(symbol, d)
        cascades = detect_cascades_from_events(events)
        for c in cascades:
            c['date'] = d
        if cascades:
            all_cascades[d] = cascades
            total += len(cascades)
        if (di+1) % 20 == 0 or di == len(dates)-1:
            print(f"    [{di+1}/{len(dates)}] {total} cascades {time.time()-t0:.1f}s", flush=True)
    return all_cascades


# ============================================================================
# PHASE 2: Simulate trades day-by-day (only days with cascades)
# ============================================================================

def simulate_day_trades(cascades, ticks_ts, ticks_price, tp_bps, sl_bps,
                        timeout_ms=3600000, randomize=False):
    """Simulate trades for one day. Takes numpy arrays for speed."""
    n = len(ticks_ts)
    trades = []
    last_exit_ts = 0

    for cascade in cascades:
        c_ts = cascade['ts_ms']
        if c_ts < last_exit_ts + 300000:
            continue

        if randomize:
            trade_dir = random.choice(['long', 'short'])
        else:
            trade_dir = 'long' if cascade['direction'] == 'up' else 'short'

        si = np.searchsorted(ticks_ts, c_ts)
        if si + 1 >= n:
            continue

        # Market entry
        fi = si + 1
        entry = ticks_price[fi]
        fill_ts = ticks_ts[fi]

        if trade_dir == 'long':
            tp_p = entry * (1 + tp_bps / 10000)
            sl_p = entry * (1 - sl_bps / 10000)
        else:
            tp_p = entry * (1 - tp_bps / 10000)
            sl_p = entry * (1 + sl_bps / 10000)

        deadline = fill_ts + timeout_ms
        xp = None; xr = None; xt = None

        for ti in range(fi + 1, n):
            t = ticks_ts[ti]; p = ticks_price[ti]
            if t > deadline:
                xp = p; xr = 'to'; xt = t; break
            if trade_dir == 'long':
                if p <= sl_p: xp = sl_p; xr = 'sl'; xt = t; break
                if p >= tp_p: xp = tp_p; xr = 'tp'; xt = t; break
            else:
                if p >= sl_p: xp = sl_p; xr = 'sl'; xt = t; break
                if p <= tp_p: xp = tp_p; xr = 'tp'; xt = t; break

        if xp is None:
            xp = ticks_price[-1]; xr = 'eod'; xt = ticks_ts[-1]

        raw = (xp - entry) / entry if trade_dir == 'long' else (entry - xp) / entry
        exit_fee = MAKER_FEE if xr == 'tp' else TAKER_FEE
        net = (raw - TAKER_FEE - exit_fee) * 10000

        trades.append({'dir': trade_dir, 'xr': xr, 'net_bps': net,
                        'hold_sec': (xt - fill_ts) / 1000})
        last_exit_ts = xt

    return trades


def run_validation(symbol, dates, cascade_dict, tp_bps, sl_bps, label, randomize=False):
    """Run backtest on a set of dates. Returns trades list."""
    trades = []
    t0 = time.time()
    days_with_cascades = [d for d in dates if d in cascade_dict]

    for di, d in enumerate(days_with_cascades):
        # Load only price column for speed
        path = PARQUET_DIR / symbol / 'trades' / 'bybit_futures' / f'{d}.parquet'
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=['timestamp_us', 'price'])
            ts = (df['timestamp_us'].values // 1000).astype(np.int64)
            px = df['price'].values.astype(np.float64)
            del df

            day_trades = simulate_day_trades(
                cascade_dict[d], ts, px, tp_bps, sl_bps, randomize=randomize)
            trades.extend(day_trades)
            del ts, px
        except Exception as e:
            continue

        if (di+1) % 10 == 0 or di == len(days_with_cascades)-1:
            elapsed = time.time() - t0
            eta = elapsed / (di+1) * (len(days_with_cascades) - di - 1)
            print(f"      [{di+1}/{len(days_with_cascades)}] n={len(trades)} "
                  f"{elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    return trades


def print_summary(trades, label):
    if not trades:
        print(f"    {label}: NO TRADES"); return
    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0

    reasons = {}
    for t in trades: reasons[t['xr']] = reasons.get(t['xr'], 0) + 1

    print(f"    {label:20s} n={n:5d} WR={wr:5.1f}% avg={avg:+7.1f}bps "
          f"total={total:+8.2f}% Sharpe={sharpe:+6.1f} exits={reasons}")

    for d in ['long', 'short']:
        dt = [t for t in trades if t['dir'] == d]
        if dt:
            dn = np.array([t['net_bps'] for t in dt])
            print(f"      {d.upper():5s}: n={len(dt)} WR={(dn>0).sum()/len(dn)*100:.1f}% "
                  f"avg={dn.mean():+.1f}bps")


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43m: Cascade Momentum — Extended Validation (Optimized)")
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

        # Phase 1: Detect all cascades (fast)
        print(f"  Phase 1: Detecting cascades...")
        cascade_dict = get_all_cascades(symbol, common)
        total_cascades = sum(len(v) for v in cascade_dict.values())
        days_with = len(cascade_dict)
        print(f"  Total: {total_cascades} cascades across {days_with}/{len(common)} days")

        if total_cascades < 20:
            print(f"  Too few cascades"); continue

        # Direction distribution
        all_c = [c for cs in cascade_dict.values() for c in cs]
        up_c = sum(1 for c in all_c if c['direction'] == 'up')
        print(f"  Directions: UP={up_c} DOWN={total_cascades - up_c}")

        # IS/OOS split
        split = int(len(common) * 0.65)
        is_dates = common[:split]
        oos_dates = common[split:]
        print(f"  IS: {len(is_dates)} days | OOS: {len(oos_dates)} days")

        # Phase 2: Test each config
        for tp, sl, cfg_label in configs:
            print(f"\n  --- {cfg_label} ---")

            print(f"    IS:")
            is_trades = run_validation(symbol, is_dates, cascade_dict, tp, sl, 'IS')
            print(f"    OOS:")
            oos_trades = run_validation(symbol, oos_dates, cascade_dict, tp, sl, 'OOS')

            # Random baseline (same cascade timing, random direction)
            print(f"    RANDOM IS:")
            rand_is = run_validation(symbol, is_dates, cascade_dict, tp, sl, 'RAND_IS', randomize=True)
            print(f"    RANDOM OOS:")
            rand_oos = run_validation(symbol, oos_dates, cascade_dict, tp, sl, 'RAND_OOS', randomize=True)

            print(f"\n  {cfg_label} RESULTS:")
            print_summary(is_trades, 'IS (signal)')
            print_summary(oos_trades, 'OOS (signal)')
            print_summary(rand_is, 'IS (random)')
            print_summary(rand_oos, 'OOS (random)')

            gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
