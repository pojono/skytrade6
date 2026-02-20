#!/usr/bin/env python3
"""
v43l: Cascade MOMENTUM — Follow Forced Flow (Tick-Level)

v43k showed that FADING cascades fails catastrophically (0% WR on tight configs).
This test does the OPPOSITE: FOLLOW the cascade direction.

When shorts get liquidated (price UP) → go LONG (follow the momentum)
When longs get liquidated (price DOWN) → go SHORT (follow the momentum)

Two entry modes:
  A) Limit order BEHIND current price (pullback entry) — maker fee
  B) Market order at current price — taker fee (more realistic for momentum)

Exit: fixed TP (limit) or SL (market) or timeout (market)
No trailing stop.

Also tests: waiting for a PULLBACK after cascade before entering (smarter entry).
"""

import sys, time, gzip, json, gc
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
                sell_count = count - buy_count
                if buy_count > sell_count:
                    cascade_dir = 'up'
                else:
                    cascade_dir = 'down'
                c_ts = ts[j - 1]
                if c_ts - last_ts > 300000:
                    cascades.append({
                        'ts_ms': c_ts,
                        'direction': cascade_dir,
                        'count': count,
                        'total_usd': total_usd,
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
    df = pd.read_parquet(path, columns=['timestamp_us', 'price', 'quantity', 'side'])
    df['ts_ms'] = df['timestamp_us'] // 1000
    return df.sort_values('ts_ms').reset_index(drop=True)


def simulate_momentum(cascades, ticks, tp_bps, sl_bps, timeout_ms=3600000,
                       entry_mode='market', pullback_bps=0, fill_timeout_ms=300000):
    """
    MOMENTUM: follow cascade direction.
    entry_mode='market': enter immediately at market (taker fee)
    entry_mode='limit': place limit order behind price for pullback entry (maker fee)
    pullback_bps: how far behind price to place limit (only for limit mode)
    """
    if ticks.empty or not cascades:
        return []

    tick_ts = ticks['ts_ms'].values
    tick_price = ticks['price'].values
    n_ticks = len(tick_ts)
    trades = []
    last_exit_ts = 0

    for cascade in cascades:
        c_ts = cascade['ts_ms']
        c_price = cascade['price']
        c_dir = cascade['direction']

        if c_ts < last_exit_ts + 300000:
            continue

        # FOLLOW direction (opposite of v43k)
        if c_dir == 'up':
            trade_dir = 'long'   # price going up, follow it
        else:
            trade_dir = 'short'  # price going down, follow it

        start_idx = np.searchsorted(tick_ts, c_ts)
        if start_idx >= n_ticks - 100:
            continue

        # Entry
        if entry_mode == 'market':
            # Market order: fill at next tick price
            if start_idx + 1 >= n_ticks:
                continue
            fill_idx = start_idx + 1
            actual_entry = tick_price[fill_idx]
            entry_fee = TAKER_FEE
        else:
            # Limit order: place behind current price (pullback entry)
            if trade_dir == 'long':
                entry_price = c_price * (1 - pullback_bps / 10000)
            else:
                entry_price = c_price * (1 + pullback_bps / 10000)

            fill_deadline = c_ts + fill_timeout_ms
            fill_idx = None
            for ti in range(start_idx, n_ticks):
                if tick_ts[ti] > fill_deadline:
                    break
                p = tick_price[ti]
                if trade_dir == 'long' and p <= entry_price:
                    fill_idx = ti; break
                elif trade_dir == 'short' and p >= entry_price:
                    fill_idx = ti; break

            if fill_idx is None:
                continue
            actual_entry = entry_price
            entry_fee = MAKER_FEE

        fill_ts = tick_ts[fill_idx]

        # TP/SL prices
        if trade_dir == 'long':
            tp_price = actual_entry * (1 + tp_bps / 10000)
            sl_price = actual_entry * (1 - sl_bps / 10000)
        else:
            tp_price = actual_entry * (1 - tp_bps / 10000)
            sl_price = actual_entry * (1 + sl_bps / 10000)

        # Exit simulation
        exit_deadline = fill_ts + timeout_ms
        exit_price = None
        exit_reason = None
        exit_ts = None

        for ti in range(fill_idx + 1, n_ticks):
            t = tick_ts[ti]
            p = tick_price[ti]

            if t > exit_deadline:
                exit_price = p; exit_reason = 'timeout'; exit_ts = t; break

            # SL first
            if trade_dir == 'long' and p <= sl_price:
                exit_price = sl_price; exit_reason = 'sl'; exit_ts = t; break
            elif trade_dir == 'short' and p >= sl_price:
                exit_price = sl_price; exit_reason = 'sl'; exit_ts = t; break

            # TP
            if trade_dir == 'long' and p >= tp_price:
                exit_price = tp_price; exit_reason = 'tp'; exit_ts = t; break
            elif trade_dir == 'short' and p <= tp_price:
                exit_price = tp_price; exit_reason = 'tp'; exit_ts = t; break

        if exit_price is None:
            exit_price = tick_price[-1]; exit_reason = 'eod'; exit_ts = tick_ts[-1]

        # PnL
        if trade_dir == 'long':
            raw_pnl = (exit_price - actual_entry) / actual_entry
        else:
            raw_pnl = (actual_entry - exit_price) / actual_entry

        exit_fee = MAKER_FEE if exit_reason == 'tp' else TAKER_FEE
        net_pnl = raw_pnl - entry_fee - exit_fee

        trades.append({
            'dir': trade_dir, 'xr': exit_reason,
            'raw_bps': raw_pnl * 10000, 'net_bps': net_pnl * 10000,
            'hold_sec': (exit_ts - fill_ts) / 1000,
            'cascade_usd': cascade['total_usd'],
        })
        last_exit_ts = exit_ts

    return trades


def analyze(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES"); return None
    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    total = net.sum() / 100
    avg = net.mean()
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 24) if std > 0 else 0
    reasons = defaultdict(int)
    for t in trades: reasons[t['xr']] += 1
    avg_hold = np.mean([t['hold_sec'] for t in trades])
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    maxdd = (peak - cum).max()

    print(f"  {label}")
    print(f"    n={n:4d} WR={wr:5.1f}% avg={avg:+7.1f}bps total={total:+7.2f}% "
          f"Sharpe={sharpe:+6.1f} maxDD={maxdd:.0f}bps "
          f"avgHold={avg_hold:.0f}s exits={dict(reasons)}")

    for d in ['long', 'short']:
        dt = [t for t in trades if t['dir'] == d]
        if dt:
            dn = np.array([t['net_bps'] for t in dt])
            print(f"    {d.upper():5s}: n={len(dt)} WR={(dn>0).sum()/len(dn)*100:.1f}% "
                  f"avg={dn.mean():+.1f}bps")
    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe}


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43l: Cascade MOMENTUM — Follow Forced Flow (Tick-Level)")
    print("=" * 80)

    configs = [
        (20, 40,   'TP=20 SL=40'),
        (30, 60,   'TP=30 SL=60'),
        (50, 100,  'TP=50 SL=100'),
        (50, 50,   'TP=50 SL=50'),
        (100, 200, 'TP=100 SL=200'),
        (30, 15,   'TP=30 SL=15 (2:1 RR)'),
        (50, 25,   'TP=50 SL=25 (2:1 RR)'),
        (100, 50,  'TP=100 SL=50 (2:1 RR)'),
    ]

    cascade_params = [
        (5, 10000,  'min5 $10k'),
        (10, 50000, 'min10 $50k'),
    ]

    for symbol in ['SOLUSDT', 'ETHUSDT', 'DOGEUSDT']:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        liq_dir = DATA_DIR / symbol / 'bybit' / 'liquidations'
        trade_dir = PARQUET_DIR / symbol / 'trades' / 'bybit_futures'
        if not liq_dir.exists():
            print(f"  No liquidation data"); continue

        liq_dates = set()
        for f in liq_dir.glob('liquidation_*.jsonl.gz'):
            liq_dates.add(f.stem.replace('.jsonl', '').split('_')[1])
        trade_dates = set(f.stem for f in trade_dir.glob('*.parquet')) if trade_dir.exists() else set()
        common = sorted(liq_dates & trade_dates)
        if len(common) < 7:
            print(f"  Too few dates ({len(common)})"); continue

        test_dates = common[:14]
        print(f"  Testing {len(test_dates)} days: {test_dates[0]} to {test_dates[-1]}")

        for cp_count, cp_usd, cp_label in cascade_params:
            print(f"\n  --- Cascade: {cp_label} ---")

            all_cascades = []
            all_ticks_list = []

            for di, d in enumerate(test_dates):
                liq_df = load_liquidations(symbol, d)
                day_cascades = detect_cascades(liq_df, min_count=cp_count, min_usd=cp_usd)
                ticks = load_ticks(symbol, d)
                if ticks.empty: continue
                all_cascades.extend(day_cascades)
                all_ticks_list.append(ticks)
                if (di+1) % 7 == 0 or di == len(test_dates)-1:
                    print(f"    [{di+1}/{len(test_dates)}] cascades={len(all_cascades)} "
                          f"RAM={get_ram_mb():.0f}MB", flush=True)

            if not all_ticks_list or not all_cascades:
                print(f"    No data!"); continue

            all_ticks = pd.concat(all_ticks_list, ignore_index=True).sort_values('ts_ms').reset_index(drop=True)
            print(f"    Total: {len(all_cascades)} cascades, {len(all_ticks):,} ticks")

            # Direction distribution
            up_c = sum(1 for c in all_cascades if c['direction'] == 'up')
            dn_c = len(all_cascades) - up_c
            print(f"    Cascade dirs: UP={up_c} DOWN={dn_c}")

            # === MODE A: Market entry (taker) ===
            print(f"\n    MODE A: Market Entry (taker fee)")
            for tp, sl, cfg_label in configs:
                trades = simulate_momentum(all_cascades, all_ticks, tp, sl,
                                            entry_mode='market')
                analyze(trades, f"{cp_label} {cfg_label} [market]")

            # === MODE B: Limit pullback entry (maker) ===
            print(f"\n    MODE B: Limit Pullback Entry (maker fee)")
            for pullback in [10, 20, 30]:
                for tp, sl, cfg_label in [(50, 100, 'TP=50 SL=100'),
                                           (30, 60, 'TP=30 SL=60'),
                                           (100, 200, 'TP=100 SL=200')]:
                    trades = simulate_momentum(all_cascades, all_ticks, tp, sl,
                                                entry_mode='limit', pullback_bps=pullback)
                    analyze(trades, f"{cp_label} {cfg_label} pullback={pullback}bps [limit]")

            del all_ticks, all_ticks_list
            gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
