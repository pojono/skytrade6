#!/usr/bin/env python3
"""
TICK-LEVEL ENTRY OPTIMIZATION for Idea 4 (BTC pump → long alts)
================================================================
Questions:
1. How fast do alts respond after BTC pumps 150+ bps in 3 min?
2. What's the optimal entry delay? (immediate? +10s? +30s? +1m? +5m?)
3. Which alts are fastest/slowest — can we target slow reactors?
4. Does entry in first 30s capture more edge than waiting 1 bar?

Method:
- Find BTC pump events from 1m klines
- Load tick data for BTC + 10 alts around each event
- Track alt price evolution at 1s, 5s, 10s, 30s, 1m, 5m, 15m, 30m, 1h, 4h after BTC pump
"""
import sys, os, gzip
sys.path.insert(0, '/home/ubuntu/Projects/skytrade6/claude-2')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from pathlib import Path
from data_loader import load_kline, progress_bar, RT_TAKER_BPS

BYBIT = Path("/home/ubuntu/Projects/skytrade6/datalake/bybit")
OUT = '/home/ubuntu/Projects/skytrade6/claude-2/out'

# Use 2025 data (tick data available from 2025-01)
START = '2025-01-01'
END = '2026-03-04'

# Alts to analyze at tick level (liquid ones with lots of tick data)
TICK_ALTS = ['ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT',
             'LINKUSDT', 'SUIUSDT', 'APTUSDT', 'ARBUSDT', 'NEARUSDT']

# Time windows to measure (seconds after BTC pump detection)
WINDOWS_SEC = [1, 5, 10, 30, 60, 300, 900, 1800, 3600, 14400]

np.random.seed(42)


def find_btc_pump_events(start, end, threshold_bps=150):
    """Find all BTC 3m pump > threshold from 1m klines."""
    btc = load_kline('BTCUSDT', start, end)
    btc = btc[['ts', 'close']].set_index('ts').sort_index()
    btc = btc[~btc.index.duplicated(keep='first')]
    btc['ret_3m'] = (btc['close'] / btc['close'].shift(3) - 1) * 10000

    # Find pump events, decluster 30 min
    sig = btc['ret_3m'] > threshold_bps
    indices = np.where(sig.values)[0]
    kept = [indices[0]] if len(indices) > 0 else []
    for idx in indices[1:]:
        if idx - kept[-1] >= 30:
            kept.append(idx)

    events = []
    for i in kept:
        ts = btc.index[i]
        ret = btc['ret_3m'].iloc[i]
        price = btc['close'].iloc[i]
        events.append({'ts': ts, 'btc_ret_bps': ret, 'btc_price': price})

    print(f"  Found {len(events)} BTC pump events (>{threshold_bps} bps in 3m)")
    return events


def load_tick_window(sym, event_ts, window_before_sec=60, window_after_sec=14400):
    """Load tick data for a symbol around an event timestamp."""
    date_str = event_ts.strftime('%Y-%m-%d')
    # May need next day too if event is near midnight
    dates = [date_str]
    if event_ts.hour >= 20:
        next_day = (event_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        dates.append(next_day)

    all_ticks = []
    for d in dates:
        fpath = BYBIT / sym / f"{d}_trades.csv.gz"
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath, compression='gzip')
            if 'timestamp' in df.columns and 'price' in df.columns:
                df['ts'] = pd.to_datetime(df['timestamp'], unit='s')
                # Filter to window
                t_start = event_ts - pd.Timedelta(seconds=window_before_sec)
                t_end = event_ts + pd.Timedelta(seconds=window_after_sec)
                df = df[(df['ts'] >= t_start) & (df['ts'] <= t_end)]
                if len(df) > 0:
                    all_ticks.append(df[['ts', 'price', 'size', 'side']])
        except:
            continue

    if not all_ticks:
        return pd.DataFrame()
    return pd.concat(all_ticks).sort_values('ts').reset_index(drop=True)


def analyze_event(event, alts):
    """Analyze alt price reaction to a single BTC pump event at tick level."""
    event_ts = event['ts']
    results = []

    for sym in alts:
        ticks = load_tick_window(sym, event_ts)
        if ticks.empty or len(ticks) < 100:
            continue

        # Price at event time (or closest before)
        pre_ticks = ticks[ticks['ts'] <= event_ts]
        if pre_ticks.empty:
            continue
        entry_price = pre_ticks['price'].iloc[-1]

        # Price at each window
        for w in WINDOWS_SEC:
            target_ts = event_ts + pd.Timedelta(seconds=w)
            # Find closest tick to target
            post = ticks[(ticks['ts'] >= target_ts - pd.Timedelta(seconds=2)) &
                        (ticks['ts'] <= target_ts + pd.Timedelta(seconds=2))]
            if post.empty:
                # Use last available before target
                before_target = ticks[ticks['ts'] <= target_ts]
                if before_target.empty:
                    continue
                exit_price = before_target['price'].iloc[-1]
            else:
                exit_price = post['price'].iloc[0]

            ret_bps = (exit_price / entry_price - 1) * 10000

            results.append({
                'event_ts': event_ts,
                'symbol': sym,
                'window_sec': w,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'ret_bps': ret_bps,
                'net_bps': ret_bps - RT_TAKER_BPS,
            })

        # Also measure: how much had the alt ALREADY moved before the signal?
        pre_60s = ticks[ticks['ts'] <= event_ts - pd.Timedelta(seconds=180)]
        if not pre_60s.empty:
            pre_price = pre_60s['price'].iloc[0]
            already_moved = (entry_price / pre_price - 1) * 10000
            results.append({
                'event_ts': event_ts,
                'symbol': sym,
                'window_sec': -180,  # negative = already moved
                'entry_price': pre_price,
                'exit_price': entry_price,
                'ret_bps': already_moved,
                'net_bps': already_moved,
            })

    return results


def main():
    print("=" * 75)
    print("  TICK-LEVEL ENTRY OPTIMIZATION: Idea 4 (BTC pump → long alts)")
    print(f"  Period: {START} → {END}")
    print("=" * 75)

    # Step 1: Find BTC pump events
    events = find_btc_pump_events(START, END)
    if not events:
        print("  ❌ No events found")
        return

    # Step 2: Analyze each event at tick level
    all_results = []
    t0 = time.time()
    for ei, event in enumerate(events):
        progress_bar(ei, len(events), prefix='  Events', start_time=t0)
        try:
            res = analyze_event(event, TICK_ALTS)
            all_results.extend(res)
        except Exception as e:
            continue
    progress_bar(len(events), len(events), prefix='  Events', start_time=t0)

    if not all_results:
        print("  ❌ No tick results")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(f'{OUT}/tick_entry_optimize.csv', index=False)

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n\n{'='*75}")
    print("  ENTRY TIMING ANALYSIS")
    print(f"  {len(events)} BTC pump events, {len(TICK_ALTS)} alts")
    print(f"{'='*75}")

    # Average alt return at each time window
    print(f"\n  {'Window':<12s} │ {'Avg ret':>8s} │ {'Net ret':>8s} │ {'WR':>5s} │ {'N':>6s} │ Interpretation")
    print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*30}")

    for w in [-180] + WINDOWS_SEC:
        sub = df[df['window_sec'] == w]
        if len(sub) == 0:
            continue
        avg = sub['ret_bps'].mean()
        net = sub['net_bps'].mean()
        wr = (sub['ret_bps'] > 0).mean() * 100
        n = len(sub)

        if w == -180:
            label = "Already moved"
            interp = "Alt move BEFORE signal"
        elif w <= 5:
            label = f"+{w}s"
            interp = "Ultra-fast entry"
        elif w <= 60:
            label = f"+{w}s"
            interp = "Fast entry"
        elif w <= 300:
            label = f"+{w//60}m"
            interp = "Normal entry"
        else:
            label = f"+{w//60}m"
            interp = "Delayed entry"

        print(f"  {label:<12s} │ {avg:>+7.0f}  │ {net:>+7.0f}  │ {wr:>4.0f}% │ {n:>6d} │ {interp}")

    # Per-alt speed of reaction
    print(f"\n\n{'─'*75}")
    print(f"  PER-ALT REACTION SPEED")
    print(f"  (return already captured at +30s vs total at +4h)")
    print(f"{'─'*75}")
    print(f"  {'Symbol':<12s} │ {'Already':>8s} │ {'+10s':>8s} │ {'+30s':>8s} │ {'+1m':>8s} │ {'+4h':>8s} │ {'Speed':>6s}")
    print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}")

    sym_speeds = []
    for sym in TICK_ALTS:
        vals = {}
        for w in [-180, 10, 30, 60, 14400]:
            sub = df[(df['symbol'] == sym) & (df['window_sec'] == w)]
            vals[w] = sub['ret_bps'].mean() if len(sub) > 0 else np.nan

        speed = vals.get(30, 0) / max(vals.get(14400, 1), 1) * 100 if vals.get(14400, 0) > 0 else 0
        sym_speeds.append((sym, speed, vals))

        already = f"{vals.get(-180, np.nan):>+7.0f}" if not np.isnan(vals.get(-180, np.nan)) else "    N/A"
        s10 = f"{vals.get(10, np.nan):>+7.0f}" if not np.isnan(vals.get(10, np.nan)) else "    N/A"
        s30 = f"{vals.get(30, np.nan):>+7.0f}" if not np.isnan(vals.get(30, np.nan)) else "    N/A"
        s60 = f"{vals.get(60, np.nan):>+7.0f}" if not np.isnan(vals.get(60, np.nan)) else "    N/A"
        s4h = f"{vals.get(14400, np.nan):>+7.0f}" if not np.isnan(vals.get(14400, np.nan)) else "    N/A"

        print(f"  {sym:<12s} │ {already} │ {s10} │ {s30} │ {s60} │ {s4h} │ {speed:>5.0f}%")

    # Sort by speed (slowest reactors = best targets)
    sym_speeds.sort(key=lambda x: x[1])
    print(f"\n  SLOWEST REACTORS (best targets for entry):")
    for sym, speed, vals in sym_speeds[:5]:
        r4h = vals.get(14400, 0)
        print(f"    {sym}: {speed:.0f}% captured at +30s, {r4h:+.0f} bps remaining at +4h")

    # Optimal entry window
    print(f"\n\n{'─'*75}")
    print(f"  OPTIMAL ENTRY WINDOW")
    print(f"{'─'*75}")

    # Marginal return by window (incremental)
    prev_ret = 0
    print(f"  {'Window':<12s} │ {'Cumulative':>10s} │ {'Marginal':>9s} │ {'Net-fees':>9s} │ Verdict")
    for w in WINDOWS_SEC:
        sub = df[df['window_sec'] == w]
        if len(sub) == 0:
            continue
        cum_ret = sub['ret_bps'].mean()
        marginal = cum_ret - prev_ret
        net = cum_ret - RT_TAKER_BPS

        if w <= 10:
            label = f"+{w}s"
        elif w < 60:
            label = f"+{w}s"
        else:
            label = f"+{w//60}m"

        verdict = "✅ profitable" if net > 0 else "❌ not yet"

        print(f"  {label:<12s} │ {cum_ret:>+9.0f}  │ {marginal:>+8.0f}  │ {net:>+8.0f}  │ {verdict}")
        prev_ret = cum_ret

    elapsed = time.time() - t0
    print(f"\n⏱ Total: {elapsed:.0f}s")
    print(f"✅ Saved: tick_entry_optimize.csv")


if __name__ == '__main__':
    main()
