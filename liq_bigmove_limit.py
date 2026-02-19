#!/usr/bin/env python3
"""
Liquidation Big-Move Strategy v2 — LIMIT ORDER ENTRY (v26i)

Key insight from v26h: market-order fade can't overcome fees (raw edge +1-4bp, fees 4-7.5bp).
The original cascade MM worked because limit orders placed INTO the cascade capture spread as alpha.

This script combines:
1. LIMIT ORDER entry (offset below market → better fill + maker fee)
2. MICROSTRUCTURE FILTERS (cascade depth, notional, recent displacement)
3. WIDER TP targets (15-50 bps) to catch bigger bounces
4. NO STOP LOSS (time-based exit only — validated by R:R research)

Fee model:
  - Entry: maker limit order → 0.02% = 2 bps
  - TP exit: maker limit order → 0.02% = 2 bps  (round-trip = 4 bps)
  - Timeout exit: taker market order → 0.055% = 5.5 bps  (round-trip = 7.5 bps)

Uses 1-minute OHLC bars for fill simulation (high/low for limit fills).
"""

import sys
import time
import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ["DOGEUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT"]
OUT_DIR = Path("results")

MAKER_FEE_PCT = 0.02   # 0.02%
TAKER_FEE_PCT = 0.055  # 0.055%


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
# CASCADE DETECTION WITH MICROSTRUCTURE CONTEXT
# ============================================================================

def detect_cascades_enriched(liq_df, price_bars):
    """
    Detect cascades and enrich with microstructure context.
    A cascade = 2+ P95 liquidations within 60 seconds.
    """
    thresh95 = liq_df['notional'].quantile(0.95)
    thresh97 = liq_df['notional'].quantile(0.97)
    thresh99 = liq_df['notional'].quantile(0.99)
    large = liq_df[liq_df['notional'] >= thresh95].copy()

    bar_index = price_bars.index
    bar_close = price_bars['close'].values
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values

    timestamps = large['timestamp'].values
    sides = large['side'].values
    notionals = large['notional'].values
    prices = large['price'].values
    n = len(large)

    # Cluster into cascades (greedy: extend cluster while events within 60s)
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
            # Build cascade info
            c_sides = sides[cluster]
            c_notionals = notionals[cluster]
            c_prices = prices[cluster]
            c_ts = timestamps[cluster]

            buy_not = c_notionals[c_sides == 'Buy'].sum()
            sell_not = c_notionals[c_sides == 'Sell'].sum()
            total_not = buy_not + sell_not
            buy_dominant = buy_not > sell_not

            # Count P97/P99 events in cluster
            n_p97 = (c_notionals >= thresh97).sum()
            n_p99 = (c_notionals >= thresh99).sum()
            max_single = c_notionals.max()

            # Price at cascade end
            end_ts = c_ts[-1]
            end_idx = bar_index.searchsorted(end_ts)
            if end_idx >= len(bar_close) - 120 or end_idx < 10:
                i = cluster[-1] + 1
                continue

            current_price = bar_close[end_idx]

            # Price displacement during cascade
            start_ts = c_ts[0]
            start_idx = bar_index.searchsorted(start_ts)
            if start_idx > 0 and start_idx < len(bar_close):
                pre_cascade_price = bar_close[max(0, start_idx - 1)]
                cascade_disp_bps = (current_price - pre_cascade_price) / pre_cascade_price * 10000
            else:
                cascade_disp_bps = 0

            cascades.append({
                'start': pd.Timestamp(c_ts[0]),
                'end': pd.Timestamp(c_ts[-1]),
                'n_events': len(cluster),
                'total_notional': total_not,
                'buy_notional': buy_not,
                'sell_notional': sell_not,
                'buy_dominant': buy_dominant,
                'n_p97': n_p97,
                'n_p99': n_p99,
                'max_single_notional': max_single,
                'duration_sec': (c_ts[-1] - c_ts[0]).astype('timedelta64[s]').astype(float),
                'end_bar_idx': end_idx,
                'current_price': current_price,
                'cascade_disp_bps': cascade_disp_bps,
            })

        i = cluster[-1] + 1 if len(cluster) >= 2 else i + 1

    return cascades


# ============================================================================
# STRATEGY: LIMIT ORDER FADE WITH MICROSTRUCTURE FILTERS
# ============================================================================

def run_limit_strategy(cascades, price_bars,
                       # Entry
                       entry_offset_pct=0.10,  # limit order offset from current price
                       # Exit
                       tp_pct=0.15,            # take profit %
                       sl_pct=None,            # stop loss % (None = no SL)
                       max_hold_min=30,        # max hold time in minutes
                       # Filters
                       min_cascade_events=2,   # minimum events in cascade
                       min_cascade_notional=0, # minimum total cascade notional
                       min_p97_events=0,       # minimum P97 events in cascade
                       min_cascade_disp_bps=0, # minimum cascade displacement (bps)
                       buy_side_only=False,
                       # Risk
                       cooldown_min=5,
                       ):
    """
    Limit order fade after cascade detection.
    
    Entry: place limit order at offset below/above market
    Fill: simulated using 1-min bar high/low
    Exit: TP (maker), SL (taker), or timeout (taker)
    """
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

        # Filters
        if cascade['n_events'] < min_cascade_events:
            continue
        if cascade['total_notional'] < min_cascade_notional:
            continue
        if cascade['n_p97'] < min_p97_events:
            continue
        if buy_side_only and not cascade['buy_dominant']:
            continue

        # Cascade displacement filter (absolute value)
        abs_disp = abs(cascade['cascade_disp_bps'])
        if abs_disp < min_cascade_disp_bps:
            continue

        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']

        # Determine direction and limit price
        if cascade['buy_dominant']:
            # Longs liquidated → price dropped → BUY limit below
            direction = 'long'
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100) if sl_pct else None
        else:
            # Shorts liquidated → price spiked → SELL limit above
            direction = 'short'
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100) if sl_pct else None

        # Fill simulation: check if limit order fills within max_hold_min
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
            trades.append({
                'filled': False,
                'direction': direction,
                'cascade_events': cascade['n_events'],
                'cascade_notional': cascade['total_notional'],
            })
            continue

        # TP/SL simulation after fill
        exit_price = None
        exit_reason = 'timeout'
        exit_bar_idx = fill_bar_idx
        remaining_hold = max_hold_min - (fill_bar_idx - idx)
        exit_end = min(fill_bar_idx + remaining_hold, len(bar_close) - 1)

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

        hold_min = exit_bar_idx - fill_bar_idx

        trades.append({
            'filled': True,
            'direction': direction,
            'entry_price': limit_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'raw_pnl_pct': raw_pnl_pct,
            'net_pnl_pct': net_pnl_pct,
            'hold_min': hold_min,
            'cascade_events': cascade['n_events'],
            'cascade_notional': cascade['total_notional'],
        })
        last_trade_time = cascade['end']

    return trades


def summarize(trades, label=""):
    """Compute summary stats for trades."""
    filled = [t for t in trades if t.get('filled', False)]
    n_total = len(trades)
    n_filled = len(filled)
    if n_filled < 5:
        return None

    net_pnls = np.array([t['net_pnl_pct'] for t in filled])
    raw_pnls = np.array([t['raw_pnl_pct'] for t in filled])
    exits = [t['exit_reason'] for t in filled]

    n_tp = sum(1 for e in exits if e == 'take_profit')
    n_sl = sum(1 for e in exits if e == 'stop_loss')
    n_to = sum(1 for e in exits if e == 'timeout')

    total_net = np.sum(net_pnls)
    avg_net = np.mean(net_pnls)
    avg_raw = np.mean(raw_pnls)
    win_rate = (net_pnls > 0).mean() * 100
    fill_rate = n_filled / n_total * 100 if n_total > 0 else 0
    avg_hold = np.mean([t['hold_min'] for t in filled])

    # Sharpe
    if np.std(net_pnls) > 0 and n_filled > 10:
        days = 282
        trades_per_day = n_filled / days
        daily_return = avg_net * trades_per_day
        daily_std = np.std(net_pnls) * np.sqrt(trades_per_day)
        sharpe = daily_return / daily_std * np.sqrt(365) if daily_std > 0 else 0
    else:
        sharpe = 0

    return {
        'label': label,
        'n_signals': n_total,
        'n_filled': n_filled,
        'fill_rate': fill_rate,
        'total_net_pct': total_net,
        'avg_net_pct': avg_net,
        'avg_raw_pct': avg_raw,
        'win_rate': win_rate,
        'n_tp': n_tp, 'n_sl': n_sl, 'n_timeout': n_to,
        'tp_rate': n_tp / n_filled * 100,
        'avg_hold_min': avg_hold,
        'sharpe': sharpe,
    }


# ============================================================================
# MAIN SWEEP
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    print(f"\n{'='*90}")
    print(f"  {symbol} — BIG-MOVE LIMIT ORDER STRATEGY")
    print(f"{'='*90}")

    t0 = time.time()

    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    print("  Building 1-min price bars...", end='', flush=True)
    bars = build_price_bars(tick_df, '1min')
    print(f" {len(bars):,} bars")

    days = (bars.index.max() - bars.index.min()).total_seconds() / 86400
    print(f"  Period: {days:.0f} days, {len(liq_df):,} liquidations")

    # Detect enriched cascades
    print("  Detecting cascades...", end='', flush=True)
    cascades = detect_cascades_enriched(liq_df, bars)
    print(f" {len(cascades):,} cascades")

    if len(cascades) < 10:
        print("  Too few cascades, skipping")
        return

    cas_df = pd.DataFrame(cascades)
    print(f"  Avg events: {cas_df['n_events'].mean():.1f}  "
          f"Avg notional: ${cas_df['total_notional'].mean():,.0f}  "
          f"Avg disp: {cas_df['cascade_disp_bps'].abs().mean():.1f} bps")

    # ── SWEEP 1: BASELINE — vary offset and TP (no microstructure filters) ──
    print(f"\n  ── SWEEP 1: OFFSET × TP (no SL, 30min hold, no filters) ──")
    print(f"  {'off':>5s} {'TP':>5s}  {'sig':>5s} {'fill':>5s} {'FR':>5s}  "
          f"{'total%':>8s}  {'avg_net':>7s}  {'avg_raw':>7s}  {'WR':>5s}  {'TP%':>5s}  "
          f"{'hold':>5s}  {'Sharpe':>6s}")
    for offset in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for tp in [0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
            trades = run_limit_strategy(cascades, bars,
                                        entry_offset_pct=offset, tp_pct=tp,
                                        sl_pct=None, max_hold_min=30,
                                        cooldown_min=5)
            s = summarize(trades)
            if s and s['n_filled'] >= 20:
                print(f"  {offset:>4.2f}% {tp:>4.2f}%  {s['n_signals']:>5d} {s['n_filled']:>5d} "
                      f"{s['fill_rate']:>4.0f}%  {s['total_net_pct']:>+8.3f}  "
                      f"{s['avg_net_pct']:>+7.4f}  {s['avg_raw_pct']:>+7.4f}  "
                      f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  "
                      f"{s['avg_hold_min']:>5.1f}  {s['sharpe']:>6.2f}")

    # ── SWEEP 2: OFFSET × TP with longer hold (60min) ──
    print(f"\n  ── SWEEP 2: OFFSET × TP (no SL, 60min hold) ──")
    print(f"  {'off':>5s} {'TP':>5s}  {'sig':>5s} {'fill':>5s} {'FR':>5s}  "
          f"{'total%':>8s}  {'avg_net':>7s}  {'avg_raw':>7s}  {'WR':>5s}  {'TP%':>5s}  "
          f"{'hold':>5s}  {'Sharpe':>6s}")
    for offset in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for tp in [0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
            trades = run_limit_strategy(cascades, bars,
                                        entry_offset_pct=offset, tp_pct=tp,
                                        sl_pct=None, max_hold_min=60,
                                        cooldown_min=5)
            s = summarize(trades)
            if s and s['n_filled'] >= 20:
                print(f"  {offset:>4.2f}% {tp:>4.2f}%  {s['n_signals']:>5d} {s['n_filled']:>5d} "
                      f"{s['fill_rate']:>4.0f}%  {s['total_net_pct']:>+8.3f}  "
                      f"{s['avg_net_pct']:>+7.4f}  {s['avg_raw_pct']:>+7.4f}  "
                      f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  "
                      f"{s['avg_hold_min']:>5.1f}  {s['sharpe']:>6.2f}")

    # ── SWEEP 3: CASCADE DEPTH FILTER ──
    print(f"\n  ── SWEEP 3: CASCADE DEPTH + OFFSET × TP (best offsets, 60min hold) ──")
    print(f"  {'cas':>4s} {'off':>5s} {'TP':>5s}  {'fill':>5s}  "
          f"{'total%':>8s}  {'avg_net':>7s}  {'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for min_events in [3, 5, 8]:
        for offset in [0.05, 0.10, 0.15, 0.20]:
            for tp in [0.12, 0.15, 0.20, 0.30]:
                trades = run_limit_strategy(cascades, bars,
                                            entry_offset_pct=offset, tp_pct=tp,
                                            sl_pct=None, max_hold_min=60,
                                            min_cascade_events=min_events,
                                            cooldown_min=5)
                s = summarize(trades)
                if s and s['n_filled'] >= 10:
                    print(f"  {min_events:>3d}+ {offset:>4.2f}% {tp:>4.2f}%  {s['n_filled']:>5d}  "
                          f"{s['total_net_pct']:>+8.3f}  {s['avg_net_pct']:>+7.4f}  "
                          f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  {s['sharpe']:>6.2f}")

    # ── SWEEP 4: P97 FILTER (at least 1 P97 event in cascade) ──
    print(f"\n  ── SWEEP 4: P97 FILTER + OFFSET × TP (60min hold) ──")
    print(f"  {'p97':>4s} {'off':>5s} {'TP':>5s}  {'fill':>5s}  "
          f"{'total%':>8s}  {'avg_net':>7s}  {'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for min_p97 in [1, 2]:
        for offset in [0.05, 0.10, 0.15, 0.20]:
            for tp in [0.12, 0.15, 0.20, 0.30]:
                trades = run_limit_strategy(cascades, bars,
                                            entry_offset_pct=offset, tp_pct=tp,
                                            sl_pct=None, max_hold_min=60,
                                            min_p97_events=min_p97,
                                            cooldown_min=5)
                s = summarize(trades)
                if s and s['n_filled'] >= 10:
                    print(f"  {min_p97:>3d}+ {offset:>4.2f}% {tp:>4.2f}%  {s['n_filled']:>5d}  "
                          f"{s['total_net_pct']:>+8.3f}  {s['avg_net_pct']:>+7.4f}  "
                          f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  {s['sharpe']:>6.2f}")

    # ── SWEEP 5: CASCADE DISPLACEMENT FILTER ──
    print(f"\n  ── SWEEP 5: CASCADE DISPLACEMENT FILTER (min abs disp, 60min hold) ──")
    print(f"  {'disp':>5s} {'off':>5s} {'TP':>5s}  {'fill':>5s}  "
          f"{'total%':>8s}  {'avg_net':>7s}  {'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for min_disp in [5, 10, 20, 30]:
        for offset in [0.05, 0.10, 0.15, 0.20]:
            for tp in [0.12, 0.15, 0.20, 0.30]:
                trades = run_limit_strategy(cascades, bars,
                                            entry_offset_pct=offset, tp_pct=tp,
                                            sl_pct=None, max_hold_min=60,
                                            min_cascade_disp_bps=min_disp,
                                            cooldown_min=5)
                s = summarize(trades)
                if s and s['n_filled'] >= 10:
                    print(f"  {min_disp:>4d}+ {offset:>4.2f}% {tp:>4.2f}%  {s['n_filled']:>5d}  "
                          f"{s['total_net_pct']:>+8.3f}  {s['avg_net_pct']:>+7.4f}  "
                          f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  {s['sharpe']:>6.2f}")

    # ── SWEEP 6: COMBINED BEST FILTERS ──
    print(f"\n  ── SWEEP 6: COMBINED (cascade 3+, P97 1+, disp 10+, 60min hold) ──")
    print(f"  {'off':>5s} {'TP':>5s}  {'fill':>5s}  "
          f"{'total%':>8s}  {'avg_net':>7s}  {'avg_raw':>7s}  {'WR':>5s}  {'TP%':>5s}  "
          f"{'hold':>5s}  {'Sharpe':>6s}")
    for offset in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
        for tp in [0.10, 0.12, 0.15, 0.20, 0.30, 0.50]:
            trades = run_limit_strategy(cascades, bars,
                                        entry_offset_pct=offset, tp_pct=tp,
                                        sl_pct=None, max_hold_min=60,
                                        min_cascade_events=3,
                                        min_p97_events=1,
                                        min_cascade_disp_bps=10,
                                        cooldown_min=5)
            s = summarize(trades)
            if s and s['n_filled'] >= 10:
                print(f"  {offset:>4.2f}% {tp:>4.2f}%  {s['n_filled']:>5d}  "
                      f"{s['total_net_pct']:>+8.3f}  {s['avg_net_pct']:>+7.4f}  "
                      f"{s['avg_raw_pct']:>+7.4f}  {s['win_rate']:>5.1f}  "
                      f"{s['tp_rate']:>5.1f}  {s['avg_hold_min']:>5.1f}  "
                      f"{s['sharpe']:>6.2f}")

    # ── SWEEP 7: BUY-SIDE ONLY (strongest signal) ──
    print(f"\n  ── SWEEP 7: BUY-SIDE ONLY + COMBINED FILTERS (60min hold) ──")
    print(f"  {'off':>5s} {'TP':>5s}  {'fill':>5s}  "
          f"{'total%':>8s}  {'avg_net':>7s}  {'WR':>5s}  {'TP%':>5s}  {'Sharpe':>6s}")
    for offset in [0.05, 0.10, 0.15, 0.20]:
        for tp in [0.12, 0.15, 0.20, 0.30]:
            trades = run_limit_strategy(cascades, bars,
                                        entry_offset_pct=offset, tp_pct=tp,
                                        sl_pct=None, max_hold_min=60,
                                        min_cascade_events=3,
                                        min_p97_events=1,
                                        min_cascade_disp_bps=10,
                                        buy_side_only=True,
                                        cooldown_min=5)
            s = summarize(trades)
            if s and s['n_filled'] >= 10:
                print(f"  {offset:>4.2f}% {tp:>4.2f}%  {s['n_filled']:>5d}  "
                      f"{s['total_net_pct']:>+8.3f}  {s['avg_net_pct']:>+7.4f}  "
                      f"{s['win_rate']:>5.1f}  {s['tp_rate']:>5.1f}  {s['sharpe']:>6.2f}")

    elapsed = time.time() - t0
    print(f"\n  {symbol} done in {elapsed:.0f}s")


def main():
    t_start = time.time()

    print("=" * 90)
    print("  LIQUIDATION BIG-MOVE LIMIT ORDER STRATEGY (v26i)")
    print("  Limit order entry + microstructure filters + realistic fees")
    print(f"  Fees: maker={MAKER_FEE_PCT:.3f}%, taker={TAKER_FEE_PCT:.3f}%")
    print(f"  Round-trip: TP={2*MAKER_FEE_PCT:.3f}%, timeout={MAKER_FEE_PCT+TAKER_FEE_PCT:.3f}%")
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
