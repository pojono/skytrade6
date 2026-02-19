#!/usr/bin/env python3
"""
Integrated Liquidation Strategy (v26j)

Combines ALL validated research findings into one strategy:

FROM v33 (Temporal Patterns):
  - Exclude bad hours: 08, 09, 13, 16 UTC (worst cascade reversion)
  - Weekday-only option (weekends have 40-60% fewer liquidations)

FROM v42B (Cascade Size):
  - P97 threshold outperforms P95 by ~2 bps/trade
  - Monotonic: larger cascades → better reversion

FROM v42f (Cross-Symbol Contagion):
  - ETH cascade → also enter on SOL/DOGE (91% WR, +5.3 bps)
  - Combined triggers increase return by 70%+

FROM v42g (Direction Asymmetry):
  - LONG cascades (fading buy-liquidations) outperform SHORT by 2-3 bps
  - Clustered cascades (5-30 min gap) are best: +6.7 bps, 93.5% WR
  - Time-since-last: 10-30 min gap is sweet spot: +7.9 bps, 95.1% WR

FROM v41 (OOS Validation):
  - offset=0.15-0.25%, TP=0.15-0.25%, SL=0.50% survived walk-forward
  - All 12 rolling windows positive across 3 symbols

FROM v26g (Microstructure):
  - 94% of events see adverse move, median 62-78 bps
  - Only 6-7% bounce within 1min, ~65% within 60min
  - Deeper displacement = lower bounce probability

Fee model:
  - Maker entry + Maker TP: 0.02% + 0.02% = 0.04% round-trip
  - Maker entry + Taker SL/timeout: 0.02% + 0.055% = 0.075% round-trip
"""

import sys
import time
import json
import gzip
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ["DOGEUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT"]
OUT_DIR = Path("results")

MAKER_FEE_PCT = 0.02   # 0.02%
TAKER_FEE_PCT = 0.055  # 0.055%

# v33/v42H: Bad hours (UTC) — worst cascade reversion
BAD_HOURS = {8, 9, 13, 16}


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


def build_price_bars(tick_df, freq='1min'):
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    return bars.dropna()


# ============================================================================
# CASCADE DETECTION (enriched with all research context)
# ============================================================================

def detect_cascades_full(liq_df, price_bars, pct_thresh=95):
    """
    Detect cascades with full microstructure context.
    Returns list of cascade dicts enriched with:
    - cascade depth, notional, P97/P99 counts
    - direction (buy/sell dominant)
    - time since last cascade
    - cascade displacement
    - hour of day, day of week
    """
    thresh = liq_df['notional'].quantile(pct_thresh / 100)
    thresh97 = liq_df['notional'].quantile(0.97)
    thresh99 = liq_df['notional'].quantile(0.99)
    large = liq_df[liq_df['notional'] >= thresh].copy()

    bar_index = price_bars.index
    bar_close = price_bars['close'].values
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values

    timestamps = large['timestamp'].values
    sides = large['side'].values
    notionals = large['notional'].values
    prices = large['price'].values
    n = len(large)

    cascades = []
    i = 0
    while i < n:
        cluster = [i]
        j = i + 1
        while j < n:
            dt = (timestamps[j] - timestamps[cluster[-1]]).astype('timedelta64[s]').astype(float)
            if dt <= 60:
                cluster.append(j)
                j += 1
            else:
                break

        if len(cluster) >= 2:
            c_sides = sides[cluster]
            c_notionals = notionals[cluster]
            c_ts = timestamps[cluster]

            buy_not = c_notionals[c_sides == 'Buy'].sum()
            sell_not = c_notionals[c_sides == 'Sell'].sum()
            total_not = buy_not + sell_not
            buy_dominant = buy_not > sell_not

            n_p97 = (c_notionals >= thresh97).sum()
            n_p99 = (c_notionals >= thresh99).sum()

            end_ts = pd.Timestamp(c_ts[-1])
            end_idx = bar_index.searchsorted(end_ts)
            if end_idx >= len(bar_close) - 120 or end_idx < 10:
                i = cluster[-1] + 1
                continue

            current_price = bar_close[end_idx]

            # Cascade displacement
            start_idx = bar_index.searchsorted(pd.Timestamp(c_ts[0]))
            if start_idx > 0:
                pre_price = bar_close[max(0, start_idx - 1)]
                cascade_disp_bps = (current_price - pre_price) / pre_price * 10000
            else:
                cascade_disp_bps = 0

            # Time since last cascade
            if cascades:
                time_since_last = (end_ts - cascades[-1]['end']).total_seconds()
            else:
                time_since_last = 99999

            cascades.append({
                'start': pd.Timestamp(c_ts[0]),
                'end': end_ts,
                'n_events': len(cluster),
                'total_notional': total_not,
                'buy_notional': buy_not,
                'sell_notional': sell_not,
                'buy_dominant': buy_dominant,
                'n_p97': int(n_p97),
                'n_p99': int(n_p99),
                'duration_sec': (c_ts[-1] - c_ts[0]).astype('timedelta64[s]').astype(float),
                'end_bar_idx': end_idx,
                'current_price': current_price,
                'cascade_disp_bps': cascade_disp_bps,
                'hour_utc': end_ts.hour,
                'day_of_week': end_ts.dayofweek,  # 0=Mon, 6=Sun
                'time_since_last_sec': time_since_last,
            })

        i = cluster[-1] + 1 if len(cluster) >= 2 else i + 1

    return cascades


# ============================================================================
# STRATEGY: LIMIT ORDER FADE WITH ALL RESEARCH FILTERS
# ============================================================================

def run_strategy(cascades, price_bars,
                 # Entry
                 entry_offset_pct=0.15,
                 # Exit
                 tp_pct=0.15,
                 sl_pct=0.50,
                 max_hold_min=30,
                 # RESEARCH FILTERS
                 exclude_bad_hours=False,      # v33/v42H
                 weekday_only=False,           # v33
                 long_only=False,              # v42g: only fade buy-side liquidations
                 min_cascade_events=2,         # v42B: cascade depth
                 min_p97_events=0,             # v42B: require P97 events
                 min_cascade_disp_bps=0,       # v26g: minimum displacement
                 min_time_since_last=0,        # v42g: clustering filter (seconds)
                 max_time_since_last=99999,    # v42g: clustering filter (seconds)
                 # Risk
                 cooldown_min=5,
                 ):
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values
    bar_close = price_bars['close'].values

    trades = []
    last_trade_time = None

    for cascade in cascades:
        # Cooldown
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < cooldown_min * 60:
                continue

        # ── RESEARCH FILTERS ──

        # v33/v42H: Bad hours
        if exclude_bad_hours and cascade['hour_utc'] in BAD_HOURS:
            continue

        # v33: Weekday only
        if weekday_only and cascade['day_of_week'] >= 5:
            continue

        # v42g: Long only (fade buy-side liquidations only)
        if long_only and not cascade['buy_dominant']:
            continue

        # v42B: Cascade depth
        if cascade['n_events'] < min_cascade_events:
            continue

        # v42B: P97 events
        if cascade['n_p97'] < min_p97_events:
            continue

        # v26g: Displacement filter
        if abs(cascade['cascade_disp_bps']) < min_cascade_disp_bps:
            continue

        # v42g: Clustering filter
        tsl = cascade['time_since_last_sec']
        if tsl < min_time_since_last or tsl > max_time_since_last:
            continue

        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']

        # Direction
        if cascade['buy_dominant']:
            direction = 'long'
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100) if sl_pct else None
        else:
            direction = 'short'
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100) if sl_pct else None

        # Fill simulation
        filled = False
        fill_bar_idx = None
        end_bar_idx = min(idx + max_hold_min, len(bar_close) - 1)

        for j in range(idx, end_bar_idx + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True
                fill_bar_idx = j
                break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True
                fill_bar_idx = j
                break

        if not filled:
            trades.append({'filled': False, 'direction': direction})
            continue

        # TP/SL simulation
        exit_price = None
        exit_reason = 'timeout'
        exit_bar_idx = fill_bar_idx
        remaining = max_hold_min - (fill_bar_idx - idx)
        exit_end = min(fill_bar_idx + remaining, len(bar_close) - 1)

        for k in range(fill_bar_idx, exit_end + 1):
            if direction == 'long':
                if sl_price and bar_low[k] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar_idx = k
                    break
                if bar_high[k] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar_idx = k
                    break
            else:
                if sl_price and bar_high[k] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar_idx = k
                    break
                if bar_low[k] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar_idx = k
                    break

        if exit_price is None:
            exit_price = bar_close[exit_end]
            exit_bar_idx = exit_end

        # PnL with fees
        if direction == 'long':
            raw_pnl_pct = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl_pct = (limit_price - exit_price) / limit_price * 100

        entry_fee = MAKER_FEE_PCT
        exit_fee = MAKER_FEE_PCT if exit_reason == 'take_profit' else TAKER_FEE_PCT
        net_pnl_pct = raw_pnl_pct - entry_fee - exit_fee

        trades.append({
            'filled': True,
            'direction': direction,
            'exit_reason': exit_reason,
            'raw_pnl_pct': raw_pnl_pct,
            'net_pnl_pct': net_pnl_pct,
            'hold_min': exit_bar_idx - fill_bar_idx,
            'hour_utc': cascade['hour_utc'],
            'day_of_week': cascade['day_of_week'],
            'cascade_events': cascade['n_events'],
            'time_since_last': cascade['time_since_last_sec'],
            'entry_time': price_bars.index[fill_bar_idx] if fill_bar_idx < len(price_bars.index) else None,
        })
        last_trade_time = cascade['end']

    return trades


def summarize(trades, label=""):
    filled = [t for t in trades if t.get('filled', False)]
    n_total = len(trades)
    n_filled = len(filled)
    if n_filled < 5:
        return None

    net = np.array([t['net_pnl_pct'] for t in filled])
    raw = np.array([t['raw_pnl_pct'] for t in filled])
    exits = [t['exit_reason'] for t in filled]

    n_tp = sum(1 for e in exits if e == 'take_profit')
    n_sl = sum(1 for e in exits if e == 'stop_loss')
    n_to = sum(1 for e in exits if e == 'timeout')
    fill_rate = n_filled / n_total * 100 if n_total > 0 else 0

    total_net = np.sum(net)
    avg_net = np.mean(net)
    avg_raw = np.mean(raw)
    win_rate = (net > 0).mean() * 100
    avg_hold = np.mean([t['hold_min'] for t in filled])

    if np.std(net) > 0 and n_filled > 10:
        days = 282
        tpd = n_filled / days
        daily_ret = avg_net * tpd
        daily_std = np.std(net) * np.sqrt(tpd)
        sharpe = daily_ret / daily_std * np.sqrt(365) if daily_std > 0 else 0
    else:
        sharpe = 0

    return {
        'label': label, 'n_signals': n_total, 'n_filled': n_filled,
        'fill_rate': fill_rate, 'total_net_pct': total_net,
        'avg_net_pct': avg_net, 'avg_raw_pct': avg_raw,
        'win_rate': win_rate, 'n_tp': n_tp, 'n_sl': n_sl, 'n_timeout': n_to,
        'tp_rate': n_tp / n_filled * 100, 'avg_hold_min': avg_hold,
        'sharpe': sharpe,
    }


def print_row(s, label=""):
    if s is None:
        return
    print(f"  {label:>40s}  {s['n_filled']:>5d}  {s['total_net_pct']:>+8.2f}%  "
          f"{s['avg_net_pct']:>+7.4f}%  {s['avg_raw_pct']:>+7.4f}%  "
          f"{s['win_rate']:>5.1f}%  {s['tp_rate']:>5.1f}%  "
          f"{s['avg_hold_min']:>5.1f}m  {s['sharpe']:>6.2f}")


def monthly_breakdown(trades, label=""):
    filled = [t for t in trades if t.get('filled', False) and t.get('entry_time') is not None]
    if len(filled) < 10:
        return
    df = pd.DataFrame(filled)
    df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')
    print(f"\n  Monthly breakdown ({label}):")
    n_pos = 0
    for m, grp in df.groupby('month'):
        n = len(grp)
        avg = grp['net_pnl_pct'].mean()
        tot = grp['net_pnl_pct'].sum()
        wr = (grp['net_pnl_pct'] > 0).sum() / n * 100
        flag = "+" if tot > 0 else "-"
        if tot > 0:
            n_pos += 1
        print(f"    {flag} {m}: n={n:4d}  WR={wr:.0f}%  avg={avg:+.4f}%  total={tot:+.2f}%")
    n_months = df['month'].nunique()
    print(f"    Positive months: {n_pos}/{n_months}")


# ============================================================================
# MAIN
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    print(f"\n{'='*90}")
    print(f"  {symbol} — INTEGRATED LIQUIDATION STRATEGY")
    print(f"{'='*90}")

    t0 = time.time()

    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    print("  Building 1-min bars...", end='', flush=True)
    bars = build_price_bars(tick_df, '1min')
    print(f" {len(bars):,} bars")

    days = (bars.index.max() - bars.index.min()).total_seconds() / 86400
    print(f"  Period: {days:.0f} days, {len(liq_df):,} liquidations")

    # Detect cascades at P95 level (we filter further below)
    print("  Detecting P95 cascades...", end='', flush=True)
    cascades_p95 = detect_cascades_full(liq_df, bars, pct_thresh=95)
    print(f" {len(cascades_p95):,}")

    print("  Detecting P97 cascades...", end='', flush=True)
    cascades_p97 = detect_cascades_full(liq_df, bars, pct_thresh=97)
    print(f" {len(cascades_p97):,}")

    if len(cascades_p95) < 20:
        print("  Too few cascades, skipping")
        return

    # OOS-validated params from v41
    offsets = [0.15, 0.20, 0.25]
    tps = [0.15, 0.20, 0.25]
    sls = [0.50, None]  # SL=0.50 from v41, None from v26i R:R research

    # ── A: BASELINE (no research filters) ──
    print(f"\n  ── A: BASELINE (P95, no filters, 30min hold) ──")
    hdr = f"  {'config':>40s}  {'fills':>5s}  {'total':>8s}  {'avg_net':>7s}  {'avg_raw':>7s}  {'WR':>5s}  {'TP%':>5s}  {'hold':>5s}  {'Sharpe':>6s}"
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p95, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=30, cooldown_min=5)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 20:
                    print_row(s, label)

    # ── B: ADD BAD HOURS FILTER (v33/v42H) ──
    print(f"\n  ── B: + EXCLUDE BAD HOURS (08,09,13,16 UTC) ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p95, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=30, cooldown_min=5,
                                      exclude_bad_hours=True)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 20:
                    print_row(s, label)

    # ── C: ADD LONG-ONLY FILTER (v42g) ──
    print(f"\n  ── C: + LONG ONLY (fade buy-side liquidations) ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p95, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=30, cooldown_min=5,
                                      exclude_bad_hours=True, long_only=True)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 20:
                    print_row(s, label)

    # ── D: P97 CASCADE SIZE (v42B) ──
    print(f"\n  ── D: P97 CASCADE (larger events only) ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p97, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=30, cooldown_min=5,
                                      exclude_bad_hours=True)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 10:
                    print_row(s, label)

    # ── E: P97 + LONG ONLY ──
    print(f"\n  ── E: P97 + LONG ONLY + BAD HOURS ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p97, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=30, cooldown_min=5,
                                      exclude_bad_hours=True, long_only=True)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 10:
                    print_row(s, label)

    # ── F: CLUSTERING FILTER (v42g: 10-30 min since last cascade) ──
    print(f"\n  ── F: CLUSTERING (10-30 min since last cascade, P95, bad hours) ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            label = f"off={off:.2f} TP={tp:.2f} SL=0.50 cluster"
            trades = run_strategy(cascades_p95, bars,
                                  entry_offset_pct=off, tp_pct=tp,
                                  sl_pct=0.50, max_hold_min=30, cooldown_min=5,
                                  exclude_bad_hours=True,
                                  min_time_since_last=600, max_time_since_last=1800)
            s = summarize(trades, label)
            if s and s['n_filled'] >= 10:
                print_row(s, label)

    # ── G: DISPLACEMENT FILTER (v26g: cascade moved price 10+ bps) ──
    print(f"\n  ── G: DISPLACEMENT ≥10 bps + BAD HOURS ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label} disp10"
                trades = run_strategy(cascades_p95, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=30, cooldown_min=5,
                                      exclude_bad_hours=True,
                                      min_cascade_disp_bps=10)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 10:
                    print_row(s, label)

    # ── H: FULL COMBO (all filters that help) ──
    print(f"\n  ── H: FULL COMBO (P95, bad hours, long only, disp≥10, 30min hold) ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p95, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=30, cooldown_min=5,
                                      exclude_bad_hours=True, long_only=True,
                                      min_cascade_disp_bps=10)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 10:
                    print_row(s, label)

    # ── I: FULL COMBO with 60min hold ──
    print(f"\n  ── I: FULL COMBO + 60min hold ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p95, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=60, cooldown_min=5,
                                      exclude_bad_hours=True, long_only=True,
                                      min_cascade_disp_bps=10)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 10:
                    print_row(s, label)

    # ── J: FULL COMBO P97 + 60min hold ──
    print(f"\n  ── J: P97 + bad hours + long only + disp≥10 + 60min hold ──")
    print(hdr)
    for off in offsets:
        for tp in tps:
            for sl in sls:
                sl_label = f"SL={sl:.2f}" if sl else "noSL"
                label = f"off={off:.2f} TP={tp:.2f} {sl_label}"
                trades = run_strategy(cascades_p97, bars,
                                      entry_offset_pct=off, tp_pct=tp,
                                      sl_pct=sl, max_hold_min=60, cooldown_min=5,
                                      exclude_bad_hours=True, long_only=True,
                                      min_cascade_disp_bps=10)
                s = summarize(trades, label)
                if s and s['n_filled'] >= 10:
                    print_row(s, label)

    # ── BEST CONFIG: Monthly breakdown ──
    print(f"\n  ── MONTHLY BREAKDOWN: Best configs ──")

    # Baseline
    trades_base = run_strategy(cascades_p95, bars,
                               entry_offset_pct=0.15, tp_pct=0.15,
                               sl_pct=0.50, max_hold_min=30, cooldown_min=5)
    monthly_breakdown(trades_base, "BASELINE off=0.15 TP=0.15 SL=0.50")

    # Full combo
    trades_combo = run_strategy(cascades_p95, bars,
                                entry_offset_pct=0.15, tp_pct=0.15,
                                sl_pct=0.50, max_hold_min=60, cooldown_min=5,
                                exclude_bad_hours=True, long_only=True,
                                min_cascade_disp_bps=10)
    monthly_breakdown(trades_combo, "FULL COMBO off=0.15 TP=0.15 SL=0.50 60m")

    # No SL combo
    trades_nosl = run_strategy(cascades_p95, bars,
                               entry_offset_pct=0.20, tp_pct=0.15,
                               sl_pct=None, max_hold_min=60, cooldown_min=5,
                               exclude_bad_hours=True, long_only=True,
                               min_cascade_disp_bps=10)
    monthly_breakdown(trades_nosl, "FULL COMBO off=0.20 TP=0.15 noSL 60m")

    elapsed = time.time() - t0
    print(f"\n  {symbol} done in {elapsed:.0f}s")


def main():
    t_start = time.time()

    print("=" * 90)
    print("  INTEGRATED LIQUIDATION STRATEGY (v26j)")
    print("  Combining ALL validated research: temporal, cascade size, direction,")
    print("  displacement, clustering, contagion — with realistic fees")
    print(f"  Fees: maker={MAKER_FEE_PCT:.3f}%, taker={TAKER_FEE_PCT:.3f}%")
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
