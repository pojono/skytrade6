#!/usr/bin/env python3
"""
Liquidation Cascade Market-Making (v26d)

Key difference from v26b/c Cascade Fade:
  - v26b/c: Market order AFTER cascade ends (30s cooldown)
  - v26d:   Limit orders placed INTO the cascade (provide liquidity)

Market-making approach:
  1. Detect cascade start (2+ large liqs within 60s)
  2. Place limit orders on the opposite side at spread offsets
     (buy below market after long liqs, sell above after short liqs)
  3. Asymmetric TP/SL (2:1) exploits proven fat-tail edge
  4. Maker fees (0 bps) vs taker fees (2-4 bps)
  5. Fill only if price reaches the limit level during hold window

Data: Bybit liquidation + ticker streams, 5-second resolution
Period: May-Aug 2025 (92 days) + Feb 2026 (9 days)
"""

import sys
import time
import json
import gzip
import io
import contextlib
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# ============================================================================
# DATA LOADING (reuse from liquidation_strategies.py)
# ============================================================================

def load_liquidations(symbol, data_dir='data'):
    """Load all liquidation data for a symbol."""
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [
        symbol_dir / "bybit" / "liquidations",
        symbol_dir,
    ]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    if not liq_files:
        raise ValueError(f"No liquidation files found for {symbol}")

    print(f"  Loading {len(liq_files)} liquidation files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 200 == 0:
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
    print(f" done ({len(records):,} events)")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_prices(symbol, data_dir='data'):
    """Load 5-second ticker data and return price series."""
    symbol_dir = Path(data_dir) / symbol
    ticker_dirs = [
        symbol_dir / "bybit" / "ticker",
        symbol_dir,
    ]
    ticker_files = []
    for d in ticker_dirs:
        ticker_files.extend(sorted(d.glob("ticker_*.jsonl.gz")))
    ticker_files = sorted(set(ticker_files))
    if not ticker_files:
        raise ValueError(f"No ticker files found for {symbol}")

    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 200 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    r = data['result']['list'][0]
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'price': float(r['lastPrice']),
                        'bid': float(r.get('bid1Price', r['lastPrice'])),
                        'ask': float(r.get('ask1Price', r['lastPrice'])),
                    })
                except Exception:
                    continue
    print(f" done ({len(records):,} ticks)")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    return df


def build_price_bars(tick_df, freq='1min'):
    """Aggregate tick data to OHLC bars."""
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    bars = bars.dropna()
    return bars


# ============================================================================
# CASCADE DETECTION
# ============================================================================

def detect_cascades(liq_df, pct_thresh=95, time_window_sec=60, min_events=2):
    """
    Detect liquidation cascades: clusters of large liquidations.
    Returns list of cascade dicts with timing, direction, notional info.
    """
    vol_thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= vol_thresh].copy()

    cascades = []
    current = []
    for _, row in large.iterrows():
        if not current:
            current = [row]
        else:
            dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
            if dt <= time_window_sec:
                current.append(row)
            else:
                if len(current) >= min_events:
                    cdf = pd.DataFrame(current)
                    buy_not = cdf[cdf['side'] == 'Buy']['notional'].sum()
                    sell_not = cdf[cdf['side'] == 'Sell']['notional'].sum()
                    total_not = buy_not + sell_not
                    cascades.append({
                        'start': cdf['timestamp'].min(),
                        'end': cdf['timestamp'].max(),
                        'n_events': len(cdf),
                        'total_notional': total_not,
                        'buy_notional': buy_not,
                        'sell_notional': sell_not,
                        'buy_dominant': buy_not > sell_not,
                        'imbalance': (sell_not - buy_not) / (total_not + 1e-10),
                        'duration_sec': (cdf['timestamp'].max() - cdf['timestamp'].min()).total_seconds(),
                        'avg_price': (cdf['price'] * cdf['notional']).sum() / (total_not + 1e-10),
                    })
                current = [row]
    if len(current) >= min_events:
        cdf = pd.DataFrame(current)
        buy_not = cdf[cdf['side'] == 'Buy']['notional'].sum()
        sell_not = cdf[cdf['side'] == 'Sell']['notional'].sum()
        total_not = buy_not + sell_not
        cascades.append({
            'start': cdf['timestamp'].min(),
            'end': cdf['timestamp'].max(),
            'n_events': len(cdf),
            'total_notional': total_not,
            'buy_notional': buy_not,
            'sell_notional': sell_not,
            'buy_dominant': buy_not > sell_not,
            'imbalance': (sell_not - buy_not) / (total_not + 1e-10),
            'duration_sec': (cdf['timestamp'].max() - cdf['timestamp'].min()).total_seconds(),
            'avg_price': (cdf['price'] * cdf['notional']).sum() / (total_not + 1e-10),
        })

    return cascades


# ============================================================================
# MARKET-MAKING STRATEGIES
# ============================================================================

def strategy_mm_limit_fade(cascades, price_bars, tick_df,
                           entry_offset_pct=0.10, tp_pct=0.30, sl_pct=0.15,
                           max_hold_min=30, maker_fee_pct=0.00,
                           cooldown_min=5, min_notional=0,
                           us_hours_only=False, label=""):
    """
    Market-making limit order fade after cascade detection.

    Logic:
    1. Cascade detected (buy-dominant → longs liquidated → price dipped)
    2. Place limit BUY at (current_price - entry_offset)
       Or: sell-dominant → shorts liquidated → price spiked → limit SELL
    3. If filled (price reaches limit level within max_hold):
       - TP at entry ± tp_pct
       - SL at entry ∓ sl_pct
    4. Maker fee applied (0% or rebate)

    Fill model:
    - Limit buy fills if bar LOW <= limit price during hold window
    - Limit sell fills if bar HIGH >= limit price during hold window
    """
    trades = []
    last_trade_time = None

    for cascade in cascades:
        cascade_end = cascade['end']

        # Cooldown
        if last_trade_time is not None:
            if (cascade_end - last_trade_time).total_seconds() < cooldown_min * 60:
                continue

        # Minimum notional filter
        if cascade['total_notional'] < min_notional:
            continue

        # US hours filter
        if us_hours_only and not (13 <= cascade_end.hour < 18):
            continue

        # Find the price bar at cascade end
        entry_bar_idx = price_bars.index.searchsorted(cascade_end)
        if entry_bar_idx >= len(price_bars) - max_hold_min:
            continue
        if entry_bar_idx < 1:
            continue

        current_price = price_bars.iloc[entry_bar_idx]['close']

        # Determine direction: fade the dominant liquidation side
        if cascade['buy_dominant']:
            # Longs got liquidated → price dipped → BUY limit below
            direction = 'long'
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100)
        else:
            # Shorts got liquidated → price spiked → SELL limit above
            direction = 'short'
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100)

        # --- Fill simulation ---
        # Check if limit order gets filled within max_hold_min
        filled = False
        fill_bar_idx = None
        end_bar_idx = min(entry_bar_idx + max_hold_min, len(price_bars) - 1)

        for j in range(entry_bar_idx, end_bar_idx + 1):
            bar = price_bars.iloc[j]
            if direction == 'long' and bar['low'] <= limit_price:
                filled = True
                fill_bar_idx = j
                break
            elif direction == 'short' and bar['high'] >= limit_price:
                filled = True
                fill_bar_idx = j
                break

        if not filled:
            trades.append({
                'entry_time': price_bars.index[entry_bar_idx],
                'direction': direction,
                'cascade_notional': cascade['total_notional'],
                'cascade_imbalance': cascade['imbalance'],
                'cascade_events': cascade['n_events'],
                'current_price': current_price,
                'limit_price': limit_price,
                'filled': False,
                'pnl_pct': 0.0,
                'exit_reason': 'no_fill',
                'hold_minutes': 0,
            })
            continue

        # --- TP/SL simulation after fill ---
        exit_price = None
        exit_reason = 'timeout'
        exit_bar_idx = fill_bar_idx
        remaining_hold = max_hold_min - (fill_bar_idx - entry_bar_idx)
        exit_end = min(fill_bar_idx + remaining_hold, len(price_bars) - 1)

        for k in range(fill_bar_idx, exit_end + 1):
            bar = price_bars.iloc[k]
            if direction == 'long':
                # Check SL first (conservative)
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar_idx = k
                    break
                if bar['high'] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar_idx = k
                    break
            else:
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar_idx = k
                    break
                if bar['low'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar_idx = k
                    break

        if exit_price is None:
            # Timeout — exit at close of last bar
            exit_price = price_bars.iloc[exit_end]['close']
            exit_bar_idx = exit_end
            exit_reason = 'timeout'

        # PnL calculation
        if direction == 'long':
            pnl = (exit_price - limit_price) / limit_price - maker_fee_pct / 100
        else:
            pnl = (limit_price - exit_price) / limit_price - maker_fee_pct / 100

        hold_min = exit_bar_idx - fill_bar_idx

        trades.append({
            'entry_time': price_bars.index[fill_bar_idx],
            'exit_time': price_bars.index[exit_bar_idx],
            'direction': direction,
            'cascade_notional': cascade['total_notional'],
            'cascade_imbalance': cascade['imbalance'],
            'cascade_events': cascade['n_events'],
            'current_price': current_price,
            'limit_price': limit_price,
            'fill_price': limit_price,
            'exit_price': exit_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'filled': True,
            'pnl_pct': pnl,
            'exit_reason': exit_reason,
            'hold_minutes': hold_min,
        })
        last_trade_time = cascade_end

    return trades


def strategy_mm_layered(cascades, price_bars, tick_df,
                        offsets_pct=[0.05, 0.10, 0.20],
                        tp_pct=0.30, sl_pct=0.15,
                        max_hold_min=30, maker_fee_pct=0.00,
                        cooldown_min=5, label=""):
    """
    Layered market-making: place multiple limit orders at different offsets.
    Each layer is independent. More aggressive fills at tighter offsets,
    better fills at wider offsets.
    """
    all_trades = []
    for offset in offsets_pct:
        layer_trades = strategy_mm_limit_fade(
            cascades, price_bars, tick_df,
            entry_offset_pct=offset, tp_pct=tp_pct, sl_pct=sl_pct,
            max_hold_min=max_hold_min, maker_fee_pct=maker_fee_pct,
            cooldown_min=cooldown_min,
            label=f"layer_{offset}",
        )
        for t in layer_trades:
            t['layer_offset'] = offset
        all_trades.extend(layer_trades)
    return all_trades


# ============================================================================
# ANALYSIS
# ============================================================================

def compute_trade_stats(trades, label, show_detail=True):
    """Compute and print strategy statistics."""
    if not trades:
        print(f"\n  {label}: NO TRADES")
        return {}

    df = pd.DataFrame(trades)
    filled = df[df['filled'] == True]
    unfilled = df[df['filled'] == False]

    n_total = len(df)
    n_filled = len(filled)
    fill_rate = n_filled / n_total * 100 if n_total > 0 else 0

    if n_filled == 0:
        print(f"\n  {label}: {n_total} signals, 0 fills (0%)")
        return {'trades': 0, 'fill_rate': 0}

    wins = (filled['pnl_pct'] > 0).sum()
    wr = wins / n_filled * 100
    avg = filled['pnl_pct'].mean()
    med = filled['pnl_pct'].median()
    tot = filled['pnl_pct'].sum()
    std = filled['pnl_pct'].std() if n_filled > 1 else 0
    sharpe = avg / std * np.sqrt(252 * 24 * 60) if std > 0 else 0

    # Exit reason breakdown
    tp_count = (filled['exit_reason'] == 'take_profit').sum()
    sl_count = (filled['exit_reason'] == 'stop_loss').sum()
    to_count = (filled['exit_reason'] == 'timeout').sum()

    # Max drawdown
    cum_pnl = filled['pnl_pct'].cumsum()
    max_dd = (cum_pnl.cummax() - cum_pnl).max()

    print(f"\n  {label}:")
    print(f"    Signals: {n_total}  Fills: {n_filled} ({fill_rate:.1f}%)")
    print(f"    Win rate:    {wr:.1f}%")
    print(f"    Avg return:  {avg*100:+.3f}%")
    print(f"    Median ret:  {med*100:+.3f}%")
    print(f"    Total ret:   {tot*100:+.2f}%")
    print(f"    Sharpe:      {sharpe:+.1f}")
    print(f"    Max DD:      {max_dd*100:.2f}%")
    print(f"    Exits: TP={tp_count} ({tp_count/n_filled*100:.0f}%)  "
          f"SL={sl_count} ({sl_count/n_filled*100:.0f}%)  "
          f"Timeout={to_count} ({to_count/n_filled*100:.0f}%)")

    if show_detail and 'direction' in filled.columns:
        for d in ['long', 'short']:
            sub = filled[filled['direction'] == d]
            if len(sub) > 0:
                sub_wr = (sub['pnl_pct'] > 0).sum() / len(sub) * 100
                print(f"    {d.upper():5s}: n={len(sub):4d}  wr={sub_wr:.1f}%  "
                      f"avg={sub['pnl_pct'].mean()*100:+.3f}%  "
                      f"total={sub['pnl_pct'].sum()*100:+.2f}%")

    if show_detail and 'hold_minutes' in filled.columns:
        avg_hold = filled['hold_minutes'].mean()
        print(f"    Avg hold:    {avg_hold:.1f} min")

    return {
        'signals': n_total, 'fills': n_filled, 'fill_rate': fill_rate,
        'win_rate': wr, 'avg_ret': avg, 'total_ret': tot,
        'sharpe': sharpe, 'max_dd': max_dd,
        'tp_pct': tp_count / n_filled * 100,
        'sl_pct': sl_count / n_filled * 100,
    }


def analyze_by_period(trades, price_bars, label=""):
    """Break down results by month."""
    df = pd.DataFrame(trades)
    filled = df[df['filled'] == True].copy()
    if len(filled) == 0:
        return

    filled['month'] = pd.to_datetime(filled['entry_time']).dt.to_period('M')
    print(f"\n  Monthly breakdown ({label}):")
    n_pos = 0
    n_months = 0
    for m, grp in filled.groupby('month'):
        n = len(grp)
        avg = grp['pnl_pct'].mean()
        tot = grp['pnl_pct'].sum()
        wr = (grp['pnl_pct'] > 0).sum() / n * 100
        n_months += 1
        flag = "✅" if tot > 0 else "  "
        if tot > 0:
            n_pos += 1
        print(f"  {flag} {m}: n={n:4d} wr={wr:.1f}% avg={avg*100:+.3f}% total={tot*100:+.2f}%")

    if n_months > 0:
        print(f"    Positive months: {n_pos}/{n_months} ({100*n_pos/n_months:.0f}%)")


# ============================================================================
# MAIN
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    """Run all market-making strategies on a single symbol."""
    print(f"\n{'='*70}")
    print(f"  {symbol} — LIQUIDATION CASCADE MARKET-MAKING")
    print(f"{'='*70}")

    t0 = time.time()

    # Load data
    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    # Build 1-min price bars
    print("  Building 1-min price bars...", end='', flush=True)
    price_bars = build_price_bars(tick_df, '1min')
    print(f" {len(price_bars):,} bars")

    # Date range
    print(f"  Period: {price_bars.index.min()} to {price_bars.index.max()}")
    days = (price_bars.index.max() - price_bars.index.min()).total_seconds() / 86400
    print(f"  Duration: {days:.1f} days")

    # Detect cascades
    print(f"\n  Detecting cascades...", end='', flush=True)
    cascades = detect_cascades(liq_df, pct_thresh=95)
    print(f" {len(cascades)} cascades")

    if len(cascades) < 10:
        print(f"  Too few cascades, skipping")
        return {}

    # Cascade stats
    cas_df = pd.DataFrame(cascades)
    buy_dom = cas_df['buy_dominant'].sum()
    sell_dom = len(cas_df) - buy_dom
    print(f"  Buy-dominant: {buy_dom} ({100*buy_dom/len(cas_df):.0f}%)  "
          f"Sell-dominant: {sell_dom} ({100*sell_dom/len(cas_df):.0f}%)")
    print(f"  Avg notional: ${cas_df['total_notional'].mean():,.0f}  "
          f"Avg events: {cas_df['n_events'].mean():.1f}  "
          f"Avg duration: {cas_df['duration_sec'].mean():.0f}s")
    print(f"  Cascades/day: {len(cascades)/max(days,1):.1f}")

    results = {}

    # ──────────────────────────────────────────────────────────────────
    # STRATEGY A: Single-level limit fade (sweep offsets and TP/SL)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STRATEGY A: SINGLE-LEVEL LIMIT FADE")
    print(f"{'─'*70}")

    # Sweep: entry offset × TP × SL
    configs = []
    for offset in [0.00, 0.03, 0.05, 0.10, 0.15, 0.20]:
        for tp in [0.15, 0.20, 0.30, 0.50]:
            for sl in [0.10, 0.15, 0.25]:
                configs.append((offset, tp, sl))

    print(f"  Sweeping {len(configs)} configs...", flush=True)
    sweep_results = []
    for i, (offset, tp, sl) in enumerate(configs):
        if (i + 1) % 24 == 0:
            print(f"    [{i+1}/{len(configs)}]", flush=True)
        trades = strategy_mm_limit_fade(
            cascades, price_bars, tick_df,
            entry_offset_pct=offset, tp_pct=tp, sl_pct=sl,
            max_hold_min=30, maker_fee_pct=0.0, cooldown_min=5,
        )
        filled = [t for t in trades if t['filled']]
        if len(filled) < 5:
            continue
        fdf = pd.DataFrame(filled)
        n = len(fdf)
        wr = (fdf['pnl_pct'] > 0).sum() / n * 100
        avg = fdf['pnl_pct'].mean()
        tot = fdf['pnl_pct'].sum()
        std = fdf['pnl_pct'].std()
        sharpe = avg / (std + 1e-10) * np.sqrt(252 * 24 * 60)
        fill_rate = n / len(trades) * 100
        tp_rate = (fdf['exit_reason'] == 'take_profit').sum() / n * 100
        sl_rate = (fdf['exit_reason'] == 'stop_loss').sum() / n * 100

        sweep_results.append({
            'offset': offset, 'tp': tp, 'sl': sl,
            'n': n, 'fill_rate': fill_rate,
            'wr': wr, 'avg': avg, 'total': tot,
            'sharpe': sharpe, 'tp_rate': tp_rate, 'sl_rate': sl_rate,
        })

    if sweep_results:
        sdf = pd.DataFrame(sweep_results).sort_values('total', ascending=False)
        print(f"\n  Top 10 configs (by total return):")
        print(f"  {'Offset':>6s} {'TP':>5s} {'SL':>5s} {'Fills':>6s} {'FillR':>6s} "
              f"{'WR':>6s} {'Avg':>8s} {'Total':>9s} {'Sharpe':>8s} {'TP%':>5s} {'SL%':>5s}")
        print(f"  {'-'*76}")
        for _, r in sdf.head(10).iterrows():
            print(f"  {r['offset']:>5.2f}% {r['tp']:>4.2f}% {r['sl']:>4.2f}% "
                  f"{r['n']:>6.0f} {r['fill_rate']:>5.1f}% "
                  f"{r['wr']:>5.1f}% {r['avg']*100:>+7.3f}% "
                  f"{r['total']*100:>+8.2f}% {r['sharpe']:>+7.1f} "
                  f"{r['tp_rate']:>4.0f}% {r['sl_rate']:>4.0f}%")

        # Best config detailed analysis
        best = sdf.iloc[0]
        print(f"\n  Best config: offset={best['offset']:.2f}% TP={best['tp']:.2f}% SL={best['sl']:.2f}%")
        best_trades = strategy_mm_limit_fade(
            cascades, price_bars, tick_df,
            entry_offset_pct=best['offset'], tp_pct=best['tp'], sl_pct=best['sl'],
            max_hold_min=30, maker_fee_pct=0.0, cooldown_min=5,
        )
        r_best = compute_trade_stats(best_trades, f"Best Single-Level (0% maker fee)")
        results['best_single'] = r_best

        # Same with 2 bps maker fee
        best_trades_fee = strategy_mm_limit_fade(
            cascades, price_bars, tick_df,
            entry_offset_pct=best['offset'], tp_pct=best['tp'], sl_pct=best['sl'],
            max_hold_min=30, maker_fee_pct=0.02, cooldown_min=5,
        )
        r_best_fee = compute_trade_stats(best_trades_fee, f"Best Single-Level (2 bps maker fee)")
        results['best_single_fee'] = r_best_fee

        # Monthly breakdown
        analyze_by_period(best_trades, price_bars, "Best Single-Level")

    # ──────────────────────────────────────────────────────────────────
    # STRATEGY B: Comparison — market order fade (like v26b/c)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STRATEGY B: MARKET ORDER FADE (v26b/c baseline)")
    print(f"{'─'*70}")

    # Market order = offset 0, immediate fill
    for tp, sl in [(0.30, 0.15), (0.50, 0.25), (0.20, 0.10)]:
        trades_mkt = strategy_mm_limit_fade(
            cascades, price_bars, tick_df,
            entry_offset_pct=0.0, tp_pct=tp, sl_pct=sl,
            max_hold_min=30, maker_fee_pct=0.04, cooldown_min=1,
        )
        r_mkt = compute_trade_stats(trades_mkt,
            f"Market Order TP={tp:.2f}% SL={sl:.2f}% (4 bps taker)")
        results[f'market_{tp}_{sl}'] = r_mkt

    # ──────────────────────────────────────────────────────────────────
    # STRATEGY C: US-hours only limit fade
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STRATEGY C: US-HOURS LIMIT FADE (13-18 UTC)")
    print(f"{'─'*70}")

    if sweep_results:
        best = sdf.iloc[0]
        trades_us = strategy_mm_limit_fade(
            cascades, price_bars, tick_df,
            entry_offset_pct=best['offset'], tp_pct=best['tp'], sl_pct=best['sl'],
            max_hold_min=30, maker_fee_pct=0.0, cooldown_min=5,
            us_hours_only=True,
        )
        r_us = compute_trade_stats(trades_us, f"US-Hours Limit Fade (0% maker)")
        results['us_hours'] = r_us
        analyze_by_period(trades_us, price_bars, "US-Hours")

    # ──────────────────────────────────────────────────────────────────
    # STRATEGY D: Layered market-making
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STRATEGY D: LAYERED MARKET-MAKING (3 levels)")
    print(f"{'─'*70}")

    for tp, sl in [(0.30, 0.15), (0.50, 0.25)]:
        trades_layered = strategy_mm_layered(
            cascades, price_bars, tick_df,
            offsets_pct=[0.03, 0.08, 0.15],
            tp_pct=tp, sl_pct=sl,
            max_hold_min=30, maker_fee_pct=0.0, cooldown_min=3,
        )
        r_lay = compute_trade_stats(trades_layered,
            f"Layered [0.03/0.08/0.15] TP={tp:.2f}% SL={sl:.2f}%")
        results[f'layered_{tp}_{sl}'] = r_lay

    # ──────────────────────────────────────────────────────────────────
    # STRATEGY E: Large cascades only (higher conviction)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STRATEGY E: LARGE CASCADES ONLY (P99)")
    print(f"{'─'*70}")

    cascades_p99 = detect_cascades(liq_df, pct_thresh=99)
    print(f"  P99 cascades: {len(cascades_p99)}")

    if len(cascades_p99) >= 5 and sweep_results:
        best = sdf.iloc[0]
        trades_p99 = strategy_mm_limit_fade(
            cascades_p99, price_bars, tick_df,
            entry_offset_pct=best['offset'], tp_pct=best['tp'], sl_pct=best['sl'],
            max_hold_min=30, maker_fee_pct=0.0, cooldown_min=5,
        )
        r_p99 = compute_trade_stats(trades_p99, f"P99 Cascades (0% maker)")
        results['p99'] = r_p99

    elapsed = time.time() - t0
    print(f"\n{'─'*70}")
    print(f"  {symbol} complete in {elapsed:.0f}s")
    print(f"{'─'*70}")

    return results


def main():
    t_start = time.time()

    print("=" * 70)
    print("  LIQUIDATION CASCADE MARKET-MAKING (v26d)")
    print("  Limit orders INTO cascades, asymmetric TP/SL, maker fees")
    print("=" * 70)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Data: Bybit liquidation + ticker streams")
    print("=" * 70)

    all_results = {}
    for sym in SYMBOLS:
        try:
            all_results[sym] = run_symbol(sym)
        except Exception as e:
            print(f"\n  ✗ {sym} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[sym] = {}

    # ── GRAND SUMMARY ──
    print(f"\n\n{'='*70}")
    print(f"  GRAND SUMMARY — ALL SYMBOLS")
    print(f"{'='*70}")

    for sym, strats in all_results.items():
        if not strats:
            continue
        print(f"\n  {sym}:")
        for sname, stats in strats.items():
            if not stats or stats.get('fills', 0) == 0:
                continue
            print(f"    {sname:30s} fills={stats.get('fills',0):4d} "
                  f"wr={stats.get('win_rate',0):5.1f}% "
                  f"avg={stats.get('avg_ret',0)*100:+7.3f}% "
                  f"total={stats.get('total_ret',0)*100:+8.2f}% "
                  f"sharpe={stats.get('sharpe',0):+7.1f}")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
