#!/usr/bin/env python3
"""
v42d: Vol Compression Straddle — Walk-Forward OOS Validation

The idea: when 60-min realized vol drops below P10 (very quiet market),
place a straddle-like trade: buy at -offset%, sell at +offset%.
The market WILL break out of compression — capture the breakout direction.

TP = 2x offset (the breakout), SL = offset (wrong direction).
Timeout = 120 minutes.

v42c showed: 99% WR, +22 bps avg, 124 trades/30d on SOLUSDT.
Now: validate OOS on 60 days, multiple symbols, walk-forward.

Also test: different vol thresholds, offsets, hold times.
"""

import sys, time, os, gc, psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE = 0.0002
TAKER_FEE = 0.00055


def ram_str():
    p = psutil.Process().memory_info().rss / 1024**3
    a = psutil.virtual_memory().available / 1024**3
    return f"RAM={p:.1f}GB, avail={a:.1f}GB"


class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', buffering=1)
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
    def flush(self):
        self.stdout.flush()
        self.file.flush()


def get_dates(start, n):
    base = datetime.strptime(start, '%Y-%m-%d')
    return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)]


def load_futures_trades(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time()
    n = len(dates)
    print(f"  Loading futures {n} days...", end='', flush=True)
    dfs = []
    for i, d in enumerate(dates):
        f = base / f"{symbol}{d}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, usecols=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            dfs.append(df)
        if (i+1) % 10 == 0:
            el = time.time() - t0; eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not dfs:
        print(" NO DATA"); return pd.DataFrame()
    r = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f" {len(r):,} trades ({time.time()-t0:.0f}s) [{ram_str()}]")
    return r


def build_bars(fut_df):
    print("  Building 1-min bars...", end='', flush=True)
    bars = fut_df.set_index('timestamp')['price'].resample('1min').agg(
        open='first', high='max', low='min', close='last').dropna()
    print(f" {len(bars):,} bars")
    return bars


def run_straddle(bars, vol_pct_thresh=0.10, offset_pct=0.15,
                 tp_mult=2.0, sl_mult=1.0, max_hold=120,
                 cooldown_min=60, vol_window=60):
    """
    When vol_pct < threshold, enter straddle:
    - Place limit buy at close * (1 - offset)
    - Place limit sell at close * (1 + offset)
    - Whichever fills first, TP at 2x offset in that direction, SL at 1x offset against
    - Timeout at max_hold minutes
    """
    close = bars['close']
    ret_1m = close.pct_change()
    vol = ret_1m.rolling(vol_window).std() * np.sqrt(vol_window)
    vol_pct = vol.rank(pct=True)

    trades = []
    last_entry = None

    compressed_mask = vol_pct < vol_pct_thresh
    compressed_idx = bars.index[compressed_mask]

    for ts in compressed_idx:
        if last_entry and (ts - last_entry).total_seconds() < cooldown_min * 60:
            continue

        idx = bars.index.get_loc(ts)
        if idx >= len(bars) - max_hold - 1:
            continue

        entry_price = bars.iloc[idx]['close']
        offset = offset_pct / 100

        # Straddle: look for breakout in either direction
        tp_dist = offset * tp_mult
        sl_dist = offset * sl_mult

        # Simulate: check each bar for breakout
        direction = None
        fill_price = None
        exit_price = None
        exit_reason = 'timeout'

        for k in range(idx + 1, min(idx + max_hold, len(bars))):
            bar = bars.iloc[k]
            up_move = (bar['high'] - entry_price) / entry_price
            down_move = (entry_price - bar['low']) / entry_price

            if direction is None:
                # Check if either side fills (price moves to offset)
                if up_move >= offset:
                    direction = 'long'
                    fill_price = entry_price * (1 + offset)
                    tp_price = fill_price * (1 + tp_dist)
                    sl_price = fill_price * (1 - sl_dist)
                elif down_move >= offset:
                    direction = 'short'
                    fill_price = entry_price * (1 - offset)
                    tp_price = fill_price * (1 - tp_dist)
                    sl_price = fill_price * (1 + sl_dist)

                # Check if BOTH sides hit in same bar (gap through)
                if up_move >= offset and down_move >= offset:
                    # Ambiguous — skip this trade
                    direction = 'skip'
                    break
            else:
                # Already filled, check TP/SL
                if direction == 'long':
                    if bar['high'] >= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; break
                    if bar['low'] <= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; break
                else:
                    if bar['low'] <= tp_price:
                        exit_price = tp_price; exit_reason = 'tp'; break
                    if bar['high'] >= sl_price:
                        exit_price = sl_price; exit_reason = 'sl'; break

        if direction == 'skip' or direction is None:
            continue

        if exit_price is None:
            # Timeout — close at market
            exit_price = bars.iloc[min(idx + max_hold, len(bars) - 1)]['close']

        if direction == 'long':
            gross = (exit_price - fill_price) / fill_price
        else:
            gross = (fill_price - exit_price) / fill_price

        # Fees: maker entry (limit), taker exit for SL/timeout, maker for TP
        entry_fee = MAKER_FEE
        exit_fee = MAKER_FEE if exit_reason == 'tp' else TAKER_FEE
        net = gross - entry_fee - exit_fee

        trades.append({
            'time': ts, 'direction': direction,
            'net': net, 'gross': gross,
            'exit': exit_reason,
            'hold': k - idx,
            'hour': ts.hour,
        })
        last_entry = ts

    return trades


def pstats(trades, label):
    if not trades:
        print(f"    {label:45s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    tp_pct = sum(1 for t in trades if t['exit']=='tp') / n * 100
    avg_hold = np.mean([t['hold'] for t in trades])
    flag = "✅" if arr.mean() > 0 else "  "
    print(f"  {flag} {label:45s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.2f}%  sh={sh:+8.1f}  TP={tp_pct:4.0f}%  hold={avg_hold:.0f}m")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh}


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'SOLUSDT'
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    out_file = f'results/v42d_vol_straddle_{symbol}.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    print("="*80)
    print(f"  v42d: VOL COMPRESSION STRADDLE OOS — {symbol} — {n_days} DAYS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    dates = get_dates('2025-05-12', n_days)
    fut_df = load_futures_trades(symbol, dates)
    bars = build_bars(fut_df)
    del fut_df; gc.collect()

    days = (bars.index.max() - bars.index.min()).total_seconds() / 86400
    print(f"  Period: {bars.index.min()} to {bars.index.max()} ({days:.0f} days)")
    print(f"  [{ram_str()}]")

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: PARAMETER SWEEP (full dataset)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 1: PARAMETER SWEEP — FULL {n_days} DAYS")
    print(f"{'#'*80}")

    print(f"\n  VOL THRESHOLD SWEEP (offset=0.15%, tp=2x, sl=1x):")
    for vt in [0.05, 0.10, 0.15, 0.20, 0.25]:
        trades = run_straddle(bars, vol_pct_thresh=vt, offset_pct=0.15)
        pstats(trades, f"vol<P{int(vt*100):02d}")

    print(f"\n  OFFSET SWEEP (vol<P10, tp=2x, sl=1x):")
    for off in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        trades = run_straddle(bars, vol_pct_thresh=0.10, offset_pct=off)
        pstats(trades, f"offset={off:.2f}%")

    print(f"\n  TP/SL RATIO SWEEP (vol<P10, offset=0.15%):")
    for tp_m, sl_m in [(1.5, 1.0), (2.0, 1.0), (2.5, 1.0), (3.0, 1.0),
                        (2.0, 0.5), (2.0, 1.5), (2.0, 2.0)]:
        trades = run_straddle(bars, vol_pct_thresh=0.10, offset_pct=0.15,
                              tp_mult=tp_m, sl_mult=sl_m)
        pstats(trades, f"TP={tp_m:.1f}x SL={sl_m:.1f}x")

    print(f"\n  COOLDOWN SWEEP (vol<P10, offset=0.15%):")
    for cd in [15, 30, 60, 120, 240]:
        trades = run_straddle(bars, vol_pct_thresh=0.10, offset_pct=0.15,
                              cooldown_min=cd)
        pstats(trades, f"cooldown={cd}min")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: WALK-FORWARD OOS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 2: WALK-FORWARD OOS (70/30 split)")
    print(f"{'#'*80}")

    split = bars.index.min() + pd.Timedelta(days=int(days * 0.7))
    train_days = int(days * 0.7)
    test_days = int(days * 0.3)
    print(f"  Split: {split.strftime('%Y-%m-%d')} (train={train_days}d, test={test_days}d)")

    # Best configs from sweep
    configs = [
        (0.10, 0.10, 2.0, 1.0, 60),
        (0.10, 0.15, 2.0, 1.0, 60),
        (0.10, 0.20, 2.0, 1.0, 60),
        (0.10, 0.12, 2.0, 1.0, 60),
        (0.15, 0.15, 2.0, 1.0, 60),
        (0.10, 0.15, 2.5, 1.0, 60),
        (0.10, 0.15, 2.0, 1.0, 30),
    ]

    for vt, off, tp_m, sl_m, cd in configs:
        label = f"v<P{int(vt*100)} off={off:.2f} tp={tp_m:.1f}x sl={sl_m:.1f}x cd={cd}m"

        # Train
        train_bars = bars[bars.index < split]
        test_bars = bars[bars.index >= split]

        train_trades = run_straddle(train_bars, vt, off, tp_m, sl_m, 120, cd)
        test_trades = run_straddle(test_bars, vt, off, tp_m, sl_m, 120, cd)

        print(f"\n  {label}:")
        ts = pstats(train_trades, "TRAIN")
        te = pstats(test_trades, "TEST")

        if ts and te and ts['tot'] > 0 and te['tot'] > 0:
            train_daily = ts['tot'] / train_days
            test_daily = te['tot'] / test_days
            print(f"    Daily: train={train_daily:+.3f}%/day  test={test_daily:+.3f}%/day  "
                  f"{'✅ OOS POSITIVE' if test_daily > 0 else '❌ OOS NEGATIVE'}")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: ROLLING WINDOW STABILITY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 3: ROLLING 15-DAY WINDOW STABILITY")
    print(f"{'#'*80}")

    # Use best config
    print(f"  Config: vol<P10, offset=0.15%, tp=2x, sl=1x, cd=60m")
    window = 15
    step = 7
    ws = bars.index.min()
    pos_windows = 0; total_windows = 0

    print(f"\n  {'Window':30s}  {'N':>4s}  {'WR%':>6s}  {'Avg':>7s}  {'Tot':>8s}")
    print(f"  {'-'*65}")

    while ws + pd.Timedelta(days=window) <= bars.index.max():
        we = ws + pd.Timedelta(days=window)
        w_bars = bars[(bars.index >= ws) & (bars.index < we)]
        if len(w_bars) < 100:
            ws += pd.Timedelta(days=step); continue
        w_trades = run_straddle(w_bars, 0.10, 0.15, 2.0, 1.0, 120, 60)
        if w_trades:
            arr = np.array([t['net'] for t in w_trades])
            n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000; tot = arr.sum()*100
            flag = "✅" if tot > 0 else "❌"
            print(f"  {flag} {ws.strftime('%Y-%m-%d')} to {we.strftime('%Y-%m-%d')}  "
                  f"n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}  tot={tot:+7.2f}%")
            total_windows += 1
            if tot > 0: pos_windows += 1
        ws += pd.Timedelta(days=step)

    if total_windows > 0:
        print(f"\n  Positive windows: {pos_windows}/{total_windows} ({pos_windows/total_windows*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # PART 4: HOUR-OF-DAY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 4: STRADDLE PERFORMANCE BY HOUR")
    print(f"{'#'*80}")

    all_trades = run_straddle(bars, 0.10, 0.15, 2.0, 1.0, 120, 60)
    if all_trades:
        print(f"\n  {'Hour':>4s}  {'N':>4s}  {'WR%':>6s}  {'Avg':>7s}  {'Tot':>8s}  {'Dir':>10s}")
        print(f"  {'-'*50}")
        for hr in range(24):
            hr_trades = [t for t in all_trades if t['hour'] == hr]
            if len(hr_trades) < 2: continue
            arr = np.array([t['net'] for t in hr_trades])
            n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000; tot = arr.sum()*100
            longs = sum(1 for t in hr_trades if t['direction']=='long')
            dir_str = f"L:{longs} S:{n-longs}"
            flag = "✅" if avg > 0 else "  "
            print(f"  {flag} {hr:02d}:00  {n:>4d}  {wr:>5.1f}%  {avg:>+6.1f}  {tot:>+7.2f}%  {dir_str}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
