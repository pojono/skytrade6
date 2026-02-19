#!/usr/bin/env python3
"""
Liquidation Big-Move Strategy (v26h)

Goal: Overcome fees by catching BIGGER post-liquidation reversals.

Key differences from v26d cascade MM:
  - Filters for the LARGEST events only (P99, deep cascades, high notional)
  - Waits for price to displace X bps before entering (buys the dip deeper)
  - Wider TP targets (15-50 bps) to clear round-trip fees
  - No stop loss — time-based exit only
  - Sweeps across filter combinations to find fee-viable setups

Fee model:
  - Maker entry + Maker TP: 2 × 0.02% = 0.04% = 4 bps round-trip
  - Maker entry + Taker SL/timeout: 0.02% + 0.055% = 0.075% = 7.5 bps

Data: 5-second ticker bars, 282 days across 4 symbols
"""

import sys
import time
import json
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ["DOGEUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT"]
OUT_DIR = Path("results")

MAKER_FEE_BPS = 2.0   # 0.02%
TAKER_FEE_BPS = 5.5   # 0.055%


# ============================================================================
# DATA LOADING
# ============================================================================

def load_liquidations(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    if not liq_files:
        raise ValueError(f"No liquidation files for {symbol}")
    print(f"  Loading {len(liq_files)} liq files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 500 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for ev in data['result']['data']:
                            records.append({
                                'timestamp': pd.to_datetime(ev['T'], unit='ms'),
                                'side': ev['S'],
                                'volume': float(ev['v']),
                                'price': float(ev['p']),
                            })
                except Exception:
                    continue
    print(f" done ({len(records):,})")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_prices(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    ticker_dirs = [symbol_dir / "bybit" / "ticker", symbol_dir]
    ticker_files = []
    for d in ticker_dirs:
        ticker_files.extend(sorted(d.glob("ticker_*.jsonl.gz")))
    ticker_files = sorted(set(ticker_files))
    if not ticker_files:
        raise ValueError(f"No ticker files for {symbol}")
    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 500 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    r = data['result']['list'][0]
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'price': float(r['lastPrice']),
                    })
                except Exception:
                    continue
    print(f" done ({len(records):,})")
    return pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)


def build_price_bars(tick_df, freq='5s'):
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    return bars.dropna()


# ============================================================================
# EVENT DETECTION: find large liquidation events with context
# ============================================================================

def detect_events(liq_df, bar_close, bar_index):
    """
    For each liquidation, compute:
    - notional percentile
    - whether it's part of a cascade (2+ P95 events within 60s)
    - cascade depth (number of events in cluster)
    - total cascade notional
    - price displacement at the moment of the event vs 1min/5min ago
    """
    # Pre-compute percentile thresholds
    thresholds = {}
    for pct in [90, 95, 97, 99]:
        thresholds[pct] = liq_df['notional'].quantile(pct / 100)

    timestamps = liq_df['timestamp'].values
    sides = liq_df['side'].values
    notionals = liq_df['notional'].values
    prices = liq_df['price'].values

    n = len(liq_df)

    # For each event, find cascade context (look back 60s for other large events)
    # Use P95 threshold for cascade membership
    is_p95 = notionals >= thresholds[95]
    is_p97 = notionals >= thresholds[97]
    is_p99 = notionals >= thresholds[99]

    events = []
    last_event_ts = None

    for i in range(n):
        if not is_p95[i]:
            continue

        ts = timestamps[i]
        side = sides[i]
        notional = notionals[i]
        price = prices[i]

        # Count cascade: how many P95 events within ±60s?
        cascade_count = 0
        cascade_notional = 0
        cascade_buy_not = 0
        cascade_sell_not = 0
        j = i - 1
        while j >= 0:
            dt = (ts - timestamps[j]).astype('timedelta64[s]').astype(float)
            if dt > 60:
                break
            if is_p95[j]:
                cascade_count += 1
                cascade_notional += notionals[j]
                if sides[j] == 'Buy':
                    cascade_buy_not += notionals[j]
                else:
                    cascade_sell_not += notionals[j]
            j -= 1
        j = i + 1
        while j < n:
            dt = (timestamps[j] - ts).astype('timedelta64[s]').astype(float)
            if dt > 60:
                break
            if is_p95[j]:
                cascade_count += 1
                cascade_notional += notionals[j]
                if sides[j] == 'Buy':
                    cascade_buy_not += notionals[j]
                else:
                    cascade_sell_not += notionals[j]
            j += 1

        # Add self
        cascade_count += 1
        cascade_notional += notional
        if side == 'Buy':
            cascade_buy_not += notional
        else:
            cascade_sell_not += notional

        # Price displacement: how much has price moved in the last 1min and 5min?
        idx = bar_index.searchsorted(ts)
        if idx < 12 or idx >= len(bar_close) - 720:
            continue

        current_price = bar_close[idx]
        price_1min_ago = bar_close[max(0, idx - 12)]   # 12 × 5s = 60s
        price_5min_ago = bar_close[max(0, idx - 60)]   # 60 × 5s = 300s

        disp_1min_bps = (current_price - price_1min_ago) / price_1min_ago * 10000
        disp_5min_bps = (current_price - price_5min_ago) / price_5min_ago * 10000

        events.append({
            'timestamp': ts,
            'side': side,
            'notional': notional,
            'price': price,
            'bar_idx': idx,
            'current_price': current_price,
            'is_p97': bool(is_p97[i]),
            'is_p99': bool(is_p99[i]),
            'cascade_count': cascade_count,
            'cascade_notional': cascade_notional,
            'cascade_buy_not': cascade_buy_not,
            'cascade_sell_not': cascade_sell_not,
            'disp_1min_bps': disp_1min_bps,
            'disp_5min_bps': disp_5min_bps,
        })

    return events


# ============================================================================
# STRATEGY: FADE WITH FILTERS
# ============================================================================

def run_strategy(events, bar_close, bar_index,
                 # Entry filters
                 min_cascade_count=1,       # minimum events in cascade cluster
                 min_notional_pct=95,       # minimum notional percentile (95/97/99)
                 min_disp_bps=0,            # minimum adverse displacement before entry
                 entry_delay_bars=0,        # wait N bars after event before entering
                 # Exit parameters
                 tp_bps=15,                 # take profit in bps
                 sl_bps=None,               # stop loss in bps (None = no SL)
                 max_hold_bars=360,         # max hold time (360 bars = 30 min)
                 # Risk
                 cooldown_bars=12,          # minimum bars between entries (12 = 1 min)
                 buy_side_only=False,       # only trade buy-side liquidations
                 ):
    """
    Fade strategy with configurable filters.
    
    Entry: after a qualifying liquidation event, enter fade position
    - Long if longs were liquidated (buy-side), Short if shorts liquidated
    - Optionally wait for price to displace min_disp_bps in adverse direction
    
    Exit: TP (maker), SL (taker), or timeout (taker)
    """
    trades = []
    last_exit_bar = -cooldown_bars - 1

    for ev in events:
        idx = ev['bar_idx']

        # Cooldown
        if idx - last_exit_bar < cooldown_bars:
            continue

        # Filter: notional percentile
        if min_notional_pct >= 99 and not ev['is_p99']:
            continue
        if min_notional_pct >= 97 and not ev['is_p97'] and not ev['is_p99']:
            continue

        # Filter: cascade depth
        if ev['cascade_count'] < min_cascade_count:
            continue

        # Filter: buy-side only
        if buy_side_only and ev['side'] != 'Buy':
            continue

        side = ev['side']

        # Determine entry point
        entry_idx = idx + entry_delay_bars

        # If min_disp_bps > 0, wait for price to displace that much before entering
        if min_disp_bps > 0:
            found_entry = False
            search_end = min(idx + max_hold_bars, len(bar_close) - max_hold_bars)
            for b in range(idx + 1, search_end):
                if side == 'Buy':
                    # Longs liquidated → price dropping → wait for it to drop min_disp_bps
                    disp = (ev['current_price'] - bar_close[b]) / ev['current_price'] * 10000
                else:
                    # Shorts liquidated → price spiking → wait for it to spike min_disp_bps
                    disp = (bar_close[b] - ev['current_price']) / ev['current_price'] * 10000
                if disp >= min_disp_bps:
                    entry_idx = b
                    found_entry = True
                    break
            if not found_entry:
                continue

        if entry_idx >= len(bar_close) - max_hold_bars:
            continue

        entry_price = bar_close[entry_idx]
        if entry_price <= 0:
            continue

        # Compute TP and SL prices
        if side == 'Buy':
            # Fade = LONG → TP above, SL below
            tp_price = entry_price * (1 + tp_bps / 10000)
            sl_price = entry_price * (1 - sl_bps / 10000) if sl_bps else None
        else:
            # Fade = SHORT → TP below, SL above
            tp_price = entry_price * (1 - tp_bps / 10000)
            sl_price = entry_price * (1 + sl_bps / 10000) if sl_bps else None

        # Simulate exit
        exit_type = 'timeout'
        exit_bar = entry_idx + max_hold_bars
        exit_price = bar_close[min(exit_bar, len(bar_close) - 1)]

        for b in range(entry_idx + 1, min(entry_idx + max_hold_bars + 1, len(bar_close))):
            p = bar_close[b]
            if side == 'Buy':
                if p >= tp_price:
                    exit_type = 'tp'
                    exit_bar = b
                    exit_price = tp_price
                    break
                if sl_price and p <= sl_price:
                    exit_type = 'sl'
                    exit_bar = b
                    exit_price = sl_price
                    break
            else:
                if p <= tp_price:
                    exit_type = 'tp'
                    exit_bar = b
                    exit_price = tp_price
                    break
                if sl_price and p >= sl_price:
                    exit_type = 'sl'
                    exit_bar = b
                    exit_price = sl_price
                    break

        # Compute PnL
        if side == 'Buy':
            raw_pnl_bps = (exit_price - entry_price) / entry_price * 10000
        else:
            raw_pnl_bps = (entry_price - exit_price) / entry_price * 10000

        # Fee model
        entry_fee = MAKER_FEE_BPS  # limit order entry
        if exit_type == 'tp':
            exit_fee = MAKER_FEE_BPS  # limit order TP
        else:
            exit_fee = TAKER_FEE_BPS  # market order SL/timeout
        total_fee_bps = entry_fee + exit_fee
        net_pnl_bps = raw_pnl_bps - total_fee_bps

        hold_bars = exit_bar - entry_idx
        hold_min = hold_bars * 5 / 60

        trades.append({
            'entry_time': bar_index[entry_idx] if entry_idx < len(bar_index) else None,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'raw_pnl_bps': raw_pnl_bps,
            'net_pnl_bps': net_pnl_bps,
            'total_fee_bps': total_fee_bps,
            'hold_min': hold_min,
            'cascade_count': ev['cascade_count'],
            'cascade_notional': ev['cascade_notional'],
            'disp_1min_bps': ev['disp_1min_bps'],
        })

        last_exit_bar = exit_bar

    return trades


# ============================================================================
# ANALYSIS: SWEEP PARAMETERS
# ============================================================================

def analyze_trades(trades, label=""):
    """Compute summary stats for a set of trades."""
    if not trades:
        return None

    n = len(trades)
    net_pnls = np.array([t['net_pnl_bps'] for t in trades])
    raw_pnls = np.array([t['raw_pnl_bps'] for t in trades])
    exit_types = [t['exit_type'] for t in trades]

    n_tp = sum(1 for e in exit_types if e == 'tp')
    n_sl = sum(1 for e in exit_types if e == 'sl')
    n_to = sum(1 for e in exit_types if e == 'timeout')

    total_net = np.sum(net_pnls)
    avg_net = np.mean(net_pnls)
    win_rate = (net_pnls > 0).mean() * 100
    avg_hold = np.mean([t['hold_min'] for t in trades])

    # Sharpe (annualized, assuming 1 trade per day as baseline)
    if np.std(net_pnls) > 0 and n > 5:
        trades_per_day = n / 282  # 282 days of data
        daily_return = avg_net * trades_per_day
        daily_std = np.std(net_pnls) * np.sqrt(trades_per_day)
        sharpe = daily_return / daily_std * np.sqrt(365) if daily_std > 0 else 0
    else:
        sharpe = 0

    return {
        'label': label,
        'n': n,
        'total_net_bps': total_net,
        'avg_net_bps': avg_net,
        'avg_raw_bps': np.mean(raw_pnls),
        'win_rate': win_rate,
        'n_tp': n_tp,
        'n_sl': n_sl,
        'n_timeout': n_to,
        'tp_rate': n_tp / n * 100,
        'avg_hold_min': avg_hold,
        'sharpe': sharpe,
        'median_net': np.median(net_pnls),
        'p25_net': np.percentile(net_pnls, 25),
        'p75_net': np.percentile(net_pnls, 75),
    }


def print_results(stats, header=""):
    if stats is None:
        return
    if header:
        print(f"\n  {header}")
    print(f"    n={stats['n']:>5d}  net_total={stats['total_net_bps']:>+8.1f}bp  "
          f"avg_net={stats['avg_net_bps']:>+6.2f}bp  avg_raw={stats['avg_raw_bps']:>+6.2f}bp  "
          f"WR={stats['win_rate']:>5.1f}%  TP={stats['tp_rate']:>5.1f}%  "
          f"hold={stats['avg_hold_min']:>4.1f}m  Sharpe={stats['sharpe']:>5.2f}")


# ============================================================================
# MAIN SWEEP
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    print(f"\n{'='*90}")
    print(f"  {symbol} — BIG-MOVE LIQUIDATION STRATEGY")
    print(f"{'='*90}")

    t0 = time.time()

    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    print("  Building 5-second bars...", end='', flush=True)
    bars = build_price_bars(tick_df, '5s')
    bar_close = bars['close'].values
    bar_index = bars.index
    print(f" {len(bars):,} bars")

    days = (bar_index.max() - bar_index.min()).total_seconds() / 86400
    print(f"  Period: {days:.0f} days, {len(liq_df):,} liquidations")

    # Detect events with context
    print("  Detecting events...", end='', flush=True)
    events = detect_events(liq_df, bar_close, bar_index)
    print(f" {len(events):,} P95+ events")

    # ── SWEEP 1: Baseline — vary TP with no filters ──
    print(f"\n  ── SWEEP 1: BASELINE (P95, no cascade filter, no displacement filter) ──")
    print(f"  {'TP':>4s} {'SL':>5s} {'hold':>5s}  {'n':>5s}  {'total':>8s}  {'avg_net':>7s}  "
          f"{'avg_raw':>7s}  {'WR':>5s}  {'TP%':>5s}  {'hold':>5s}  {'Sharpe':>6s}")
    for tp in [10, 15, 20, 25, 30, 40, 50]:
        for max_hold in [60, 120, 360, 720]:  # 5, 10, 30, 60 min
            hold_label = f"{max_hold*5//60}m"
            trades = run_strategy(events, bar_close, bar_index,
                                  tp_bps=tp, sl_bps=None, max_hold_bars=max_hold,
                                  cooldown_bars=12)
            s = analyze_trades(trades)
            if s and s['n'] >= 20:
                print(f"  {tp:>4d} {'none':>5s} {hold_label:>5s}  {s['n']:>5d}  "
                      f"{s['total_net_bps']:>+8.1f}  {s['avg_net_bps']:>+7.2f}  "
                      f"{s['avg_raw_bps']:>+7.2f}  {s['win_rate']:>5.1f}  "
                      f"{s['tp_rate']:>5.1f}  {s['avg_hold_min']:>5.1f}  "
                      f"{s['sharpe']:>6.2f}")

    # ── SWEEP 2: Filter by cascade depth ──
    print(f"\n  ── SWEEP 2: CASCADE DEPTH FILTER (TP=20bp, no SL, 30min hold) ──")
    print(f"  {'cascade':>8s} {'pct':>4s}  {'n':>5s}  {'total':>8s}  {'avg_net':>7s}  "
          f"{'avg_raw':>7s}  {'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for min_cascade in [1, 2, 3, 5]:
        for min_pct in [95, 97, 99]:
            trades = run_strategy(events, bar_close, bar_index,
                                  min_cascade_count=min_cascade,
                                  min_notional_pct=min_pct,
                                  tp_bps=20, sl_bps=None, max_hold_bars=360,
                                  cooldown_bars=12)
            s = analyze_trades(trades)
            if s and s['n'] >= 10:
                print(f"  {min_cascade:>6d}+  P{min_pct:>2d}  {s['n']:>5d}  "
                      f"{s['total_net_bps']:>+8.1f}  {s['avg_net_bps']:>+7.2f}  "
                      f"{s['avg_raw_bps']:>+7.2f}  {s['win_rate']:>5.1f}  "
                      f"{s['tp_rate']:>5.1f}  {s['sharpe']:>6.2f}")

    # ── SWEEP 3: Displacement filter (enter AFTER price has moved) ──
    print(f"\n  ── SWEEP 3: DISPLACEMENT FILTER (wait for adverse move before entry) ──")
    print(f"  {'disp':>5s} {'TP':>4s} {'hold':>5s}  {'n':>5s}  {'total':>8s}  {'avg_net':>7s}  "
          f"{'avg_raw':>7s}  {'WR':>5s}  {'TP%':>5s}  {'hold':>5s}  {'Sharpe':>6s}")
    for min_disp in [5, 10, 15, 20, 30, 50]:
        for tp in [15, 20, 30, 50]:
            for max_hold in [120, 360, 720]:
                hold_label = f"{max_hold*5//60}m"
                trades = run_strategy(events, bar_close, bar_index,
                                      min_disp_bps=min_disp,
                                      tp_bps=tp, sl_bps=None, max_hold_bars=max_hold,
                                      cooldown_bars=12)
                s = analyze_trades(trades)
                if s and s['n'] >= 20:
                    print(f"  {min_disp:>4d}+ {tp:>4d} {hold_label:>5s}  {s['n']:>5d}  "
                          f"{s['total_net_bps']:>+8.1f}  {s['avg_net_bps']:>+7.2f}  "
                          f"{s['avg_raw_bps']:>+7.2f}  {s['win_rate']:>5.1f}  "
                          f"{s['tp_rate']:>5.1f}  {s['avg_hold_min']:>5.1f}  "
                          f"{s['sharpe']:>6.2f}")

    # ── SWEEP 4: Combined filters (best of cascade + displacement) ──
    print(f"\n  ── SWEEP 4: COMBINED FILTERS (cascade depth + displacement + TP) ──")
    print(f"  {'cas':>4s} {'pct':>4s} {'disp':>5s} {'TP':>4s} {'hold':>5s}  {'n':>5s}  "
          f"{'total':>8s}  {'avg_net':>7s}  {'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for min_cascade in [2, 3]:
        for min_pct in [95, 97]:
            for min_disp in [10, 20, 30]:
                for tp in [15, 20, 30, 50]:
                    for max_hold in [360, 720]:
                        hold_label = f"{max_hold*5//60}m"
                        trades = run_strategy(events, bar_close, bar_index,
                                              min_cascade_count=min_cascade,
                                              min_notional_pct=min_pct,
                                              min_disp_bps=min_disp,
                                              tp_bps=tp, sl_bps=None,
                                              max_hold_bars=max_hold,
                                              cooldown_bars=24)  # 2 min cooldown
                        s = analyze_trades(trades)
                        if s and s['n'] >= 10:
                            print(f"  {min_cascade:>3d}+ P{min_pct:>2d} {min_disp:>4d}+ "
                                  f"{tp:>4d} {hold_label:>5s}  {s['n']:>5d}  "
                                  f"{s['total_net_bps']:>+8.1f}  {s['avg_net_bps']:>+7.2f}  "
                                  f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  "
                                  f"{s['sharpe']:>6.2f}")

    # ── SWEEP 5: Entry delay (wait N seconds after event) ──
    print(f"\n  ── SWEEP 5: ENTRY DELAY (wait after event, TP=20, 30min hold) ──")
    print(f"  {'delay':>6s}  {'n':>5s}  {'total':>8s}  {'avg_net':>7s}  {'avg_raw':>7s}  "
          f"{'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for delay_sec in [0, 5, 10, 15, 30, 60, 120]:
        delay_bars = max(0, delay_sec // 5)
        trades = run_strategy(events, bar_close, bar_index,
                              entry_delay_bars=delay_bars,
                              tp_bps=20, sl_bps=None, max_hold_bars=360,
                              cooldown_bars=12)
        s = analyze_trades(trades)
        if s and s['n'] >= 20:
            print(f"  {delay_sec:>4d}s  {s['n']:>5d}  "
                  f"{s['total_net_bps']:>+8.1f}  {s['avg_net_bps']:>+7.2f}  "
                  f"{s['avg_raw_bps']:>+7.2f}  {s['win_rate']:>5.1f}  "
                  f"{s['tp_rate']:>5.1f}  {s['sharpe']:>6.2f}")

    # ── SWEEP 6: Buy-side only (longs liquidated → buy the dip) ──
    print(f"\n  ── SWEEP 6: BUY-SIDE ONLY (only fade long liquidations) ──")
    print(f"  {'disp':>5s} {'TP':>4s} {'hold':>5s}  {'n':>5s}  {'total':>8s}  {'avg_net':>7s}  "
          f"{'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for min_disp in [0, 10, 20]:
        for tp in [15, 20, 30]:
            for max_hold in [360, 720]:
                hold_label = f"{max_hold*5//60}m"
                trades = run_strategy(events, bar_close, bar_index,
                                      min_disp_bps=min_disp,
                                      tp_bps=tp, sl_bps=None, max_hold_bars=max_hold,
                                      cooldown_bars=12, buy_side_only=True)
                s = analyze_trades(trades)
                if s and s['n'] >= 20:
                    print(f"  {min_disp:>4d}+ {tp:>4d} {hold_label:>5s}  {s['n']:>5d}  "
                          f"{s['total_net_bps']:>+8.1f}  {s['avg_net_bps']:>+7.2f}  "
                          f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  "
                          f"{s['sharpe']:>6.2f}")

    elapsed = time.time() - t0
    print(f"\n  {symbol} done in {elapsed:.0f}s")


def main():
    t_start = time.time()

    print("=" * 90)
    print("  LIQUIDATION BIG-MOVE STRATEGY (v26h)")
    print("  Fee-aware sweep: finding setups that overcome maker+taker costs")
    print(f"  Fees: maker={MAKER_FEE_BPS:.1f}bp, taker={TAKER_FEE_BPS:.1f}bp")
    print(f"  Round-trip: maker+maker={2*MAKER_FEE_BPS:.1f}bp, maker+taker={MAKER_FEE_BPS+TAKER_FEE_BPS:.1f}bp")
    print("=" * 90)

    OUT_DIR.mkdir(exist_ok=True)

    for sym in SYMBOLS:
        try:
            run_symbol(sym)
        except Exception as e:
            print(f"\n  ✗ {sym} FAILED: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
