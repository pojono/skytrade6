#!/usr/bin/env python3
"""
v43e: Grid Strategy — Extended Validation

Validates the grid strategy from v43d on:
  - All 76 available days (vs 14 in v43d)
  - 5 symbols: SOL, ETH, DOGE, XRP, BTC
  - Walk-forward: train on first 50 days, test on last 26 days
  - Rolling 14-day windows to check stability

Best configs from v43d to validate:
  A: spacing=200bps, hold=480min, rebal=30min  (highest avg)
  B: spacing=150bps, hold=480min, rebal=30min  (good balance)
  C: spacing=150bps, hold=120min, rebal=30min  (most trades)
  D: spacing=100bps, hold=240min, rebal=30min  (tighter grid)

All entries and TP exits via limit orders (maker 0.02%).
Timeout exits via market (taker 0.055%).
No trailing stop.
"""

import sys, time, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.055
PARQUET_DIR = Path('parquet')

SYMBOLS = ['SOLUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT', 'BTCUSDT']

# Configs to validate (from v43d top results)
CONFIGS = [
    {'name': 'A_wide',   'spacing': 200, 'hold': 480, 'rebal': 30},
    {'name': 'B_medium', 'spacing': 150, 'hold': 480, 'rebal': 30},
    {'name': 'C_fast',   'spacing': 150, 'hold': 120, 'rebal': 30},
    {'name': 'D_tight',  'spacing': 100, 'hold': 240, 'rebal': 30},
]


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


# ============================================================================
# DATA — load in chunks to save RAM
# ============================================================================

def load_bars_chunk(symbol, dates, freq='1min'):
    """Load ticker parquet for a chunk of dates → 1-min OHLC bars."""
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    all_bars = []
    for d in dates:
        path = ticker_dir / f'{d}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp_us'], unit='us')
        df = df.set_index('timestamp').sort_index()
        bars = df['last_price'].resample(freq).ohlc().dropna()
        all_bars.append(bars)
    if not all_bars:
        return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    return result[~result.index.duplicated(keep='first')]


# ============================================================================
# GRID SIMULATION (same as v43d but cleaner)
# ============================================================================

def simulate_grid(bars, spacing_bps, max_hold_bars, rebal_interval, vol_window=60):
    """Simulate symmetric grid. Returns list of trade dicts."""
    close = bars['close'].values
    high = bars['high'].values
    low = bars['low'].values
    n = len(close)

    if n < vol_window + max_hold_bars + 10:
        return []

    trades = []
    position = None

    for i in range(vol_window, n - max_hold_bars):
        # Manage open position
        if position is not None:
            hold = i - position['entry_bar']
            d = position['direction']
            tp = position['tp_price']

            # Check TP
            if d == 'long' and high[i] >= tp:
                pnl = (tp - position['entry_price']) / position['entry_price'] * 10000
                trades.append(_make_trade(position, i, tp, pnl, 'take_profit'))
                position = None
                continue
            elif d == 'short' and low[i] <= tp:
                pnl = (position['entry_price'] - tp) / position['entry_price'] * 10000
                trades.append(_make_trade(position, i, tp, pnl, 'take_profit'))
                position = None
                continue

            # Check timeout
            if hold >= max_hold_bars:
                ep = close[i]
                if d == 'long':
                    pnl = (ep - position['entry_price']) / position['entry_price'] * 10000
                else:
                    pnl = (position['entry_price'] - ep) / position['entry_price'] * 10000
                trades.append(_make_trade(position, i, ep, pnl, 'timeout'))
                position = None
                continue
            continue

        # No position — try to enter at rebalance points
        if i % rebal_interval != 0:
            continue

        mid = close[i]
        buy_limit = mid * (1 - spacing_bps / 2 / 10000)
        sell_limit = mid * (1 + spacing_bps / 2 / 10000)

        fill_end = min(i + rebal_interval, n - max_hold_bars)
        for j in range(i + 1, fill_end):
            if low[j] <= buy_limit:
                tp_price = buy_limit * (1 + spacing_bps / 10000)
                position = {'direction': 'long', 'entry_price': buy_limit,
                            'entry_bar': j, 'tp_price': tp_price}
                break
            elif high[j] >= sell_limit:
                tp_price = sell_limit * (1 - spacing_bps / 10000)
                position = {'direction': 'short', 'entry_price': sell_limit,
                            'entry_bar': j, 'tp_price': tp_price}
                break

    # Close remaining
    if position is not None:
        d = position['direction']
        ep = close[-1]
        if d == 'long':
            pnl = (ep - position['entry_price']) / position['entry_price'] * 10000
        else:
            pnl = (position['entry_price'] - ep) / position['entry_price'] * 10000
        trades.append(_make_trade(position, len(close)-1, ep, pnl, 'end'))

    return trades


def _make_trade(pos, exit_bar, exit_price, pnl_bps, reason):
    if reason == 'take_profit':
        fee_bps = MAKER_FEE_PCT * 100 * 2  # maker + maker
    else:
        fee_bps = MAKER_FEE_PCT * 100 + TAKER_FEE_PCT * 100  # maker + taker
    return {
        'entry_bar': pos['entry_bar'],
        'exit_bar': exit_bar,
        'direction': pos['direction'],
        'entry_price': pos['entry_price'],
        'exit_price': exit_price,
        'pnl_bps': pnl_bps,
        'net_bps': pnl_bps - fee_bps,
        'exit_reason': reason,
        'hold_bars': exit_bar - pos['entry_bar'],
    }


# ============================================================================
# ANALYSIS
# ============================================================================

def summarize(trades, label=''):
    if not trades:
        return {'n': 0, 'wr': 0, 'avg': 0, 'total': 0, 'sharpe': 0, 'tp_pct': 0}
    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    total = net.sum() / 100  # bps → %
    avg = net.mean()
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0
    tp_n = sum(1 for t in trades if t['exit_reason'] == 'take_profit')
    tp_pct = tp_n / n * 100

    # Max drawdown in bps
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    maxdd = (peak - cum).max() if len(cum) > 0 else 0

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe,
            'tp_pct': tp_pct, 'maxdd': maxdd}


def print_summary(s, label):
    print(f"  {label:40s} | n={s['n']:4d} WR={s['wr']:5.1f}% "
          f"avg={s['avg']:+7.1f}bps total={s['total']:+7.2f}% "
          f"Sharpe={s['sharpe']:+6.1f} TP%={s['tp_pct']:4.0f}% "
          f"maxDD={s['maxdd']:6.0f}bps")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("v43e: Grid Strategy — Extended Validation (76 days × 5 symbols)")
    print("=" * 80)

    all_results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        ticker_dir = PARQUET_DIR / symbol / 'ticker'
        available = sorted([f.stem for f in ticker_dir.glob('*.parquet')])
        if len(available) < 7:
            print(f"  Only {len(available)} days, skip")
            continue

        print(f"  Available: {len(available)} days ({available[0]} to {available[-1]})")

        # Split into IS (first ~65%) and OOS (last ~35%)
        split_idx = int(len(available) * 0.65)
        is_dates = available[:split_idx]
        oos_dates = available[split_idx:]
        print(f"  IS: {is_dates[0]} to {is_dates[-1]} ({len(is_dates)} days)")
        print(f"  OOS: {oos_dates[0]} to {oos_dates[-1]} ({len(oos_dates)} days)")

        # Load IS data
        print(f"\n  Loading IS data...", flush=True)
        t1 = time.time()
        bars_is = load_bars_chunk(symbol, is_dates)
        print(f"  IS: {len(bars_is):,} bars ({time.time()-t1:.1f}s, RAM={get_ram_mb():.0f}MB)")

        # Load OOS data
        print(f"  Loading OOS data...", flush=True)
        t1 = time.time()
        bars_oos = load_bars_chunk(symbol, oos_dates)
        print(f"  OOS: {len(bars_oos):,} bars ({time.time()-t1:.1f}s, RAM={get_ram_mb():.0f}MB)")

        # Test each config on IS and OOS
        print(f"\n  {'Config':12s} | {'--- IN-SAMPLE ---':60s} | {'--- OUT-OF-SAMPLE ---':60s}")

        for cfg in CONFIGS:
            # IS
            trades_is = simulate_grid(bars_is, cfg['spacing'], cfg['hold'], cfg['rebal'])
            s_is = summarize(trades_is)

            # OOS
            trades_oos = simulate_grid(bars_oos, cfg['spacing'], cfg['hold'], cfg['rebal'])
            s_oos = summarize(trades_oos)

            print(f"\n  Config {cfg['name']} (sp={cfg['spacing']} hold={cfg['hold']} rebal={cfg['rebal']})")
            print_summary(s_is, f"  IS  ({len(is_dates)}d)")
            print_summary(s_oos, f"  OOS ({len(oos_dates)}d)")

            # Rolling window analysis (14-day windows)
            print(f"  Rolling 14-day windows:")
            window_results = []
            for wi in range(0, len(available) - 13):
                w_dates = available[wi:wi+14]
                w_bars = load_bars_chunk(symbol, w_dates)
                if w_bars.empty:
                    continue
                w_trades = simulate_grid(w_bars, cfg['spacing'], cfg['hold'], cfg['rebal'])
                w_s = summarize(w_trades)
                window_results.append({
                    'start': w_dates[0], 'end': w_dates[-1],
                    'n': w_s['n'], 'avg': w_s['avg'], 'total': w_s['total'],
                })
                del w_bars
            
            if window_results:
                pos_windows = sum(1 for w in window_results if w['total'] > 0)
                total_windows = len(window_results)
                avg_window_total = np.mean([w['total'] for w in window_results])
                avg_window_n = np.mean([w['n'] for w in window_results])
                print(f"    {pos_windows}/{total_windows} positive windows "
                      f"({pos_windows/total_windows*100:.0f}%), "
                      f"avg_total={avg_window_total:+.2f}%, avg_n={avg_window_n:.0f}")

                # Show worst and best windows
                window_results.sort(key=lambda x: x['total'])
                worst = window_results[0]
                best = window_results[-1]
                print(f"    Worst: {worst['start']}–{worst['end']} "
                      f"total={worst['total']:+.2f}% n={worst['n']}")
                print(f"    Best:  {best['start']}–{best['end']} "
                      f"total={best['total']:+.2f}% n={best['n']}")

            all_results.append({
                'symbol': symbol, 'config': cfg['name'],
                'is_avg': s_is['avg'], 'is_total': s_is['total'],
                'is_n': s_is['n'], 'is_wr': s_is['wr'], 'is_sharpe': s_is['sharpe'],
                'oos_avg': s_oos['avg'], 'oos_total': s_oos['total'],
                'oos_n': s_oos['n'], 'oos_wr': s_oos['wr'], 'oos_sharpe': s_oos['sharpe'],
                'pos_windows': pos_windows if window_results else 0,
                'total_windows': total_windows if window_results else 0,
            })

        del bars_is, bars_oos
        gc.collect()

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("GRAND SUMMARY — All Symbols × All Configs")
    print(f"{'='*80}")
    print(f"{'Symbol':10s} {'Config':12s} | {'IS avg':>8s} {'IS tot':>8s} {'IS n':>5s} "
          f"{'IS WR':>6s} {'IS Sh':>6s} | {'OOS avg':>8s} {'OOS tot':>8s} {'OOS n':>5s} "
          f"{'OOS WR':>6s} {'OOS Sh':>6s} | {'Win%':>5s}")

    for r in all_results:
        wp = f"{r['pos_windows']}/{r['total_windows']}" if r['total_windows'] > 0 else "N/A"
        print(f"{r['symbol']:10s} {r['config']:12s} | "
              f"{r['is_avg']:+7.1f} {r['is_total']:+7.2f}% {r['is_n']:5d} "
              f"{r['is_wr']:5.1f}% {r['is_sharpe']:+5.1f} | "
              f"{r['oos_avg']:+7.1f} {r['oos_total']:+7.2f}% {r['oos_n']:5d} "
              f"{r['oos_wr']:5.1f}% {r['oos_sharpe']:+5.1f} | {wp}")

    # Highlight OOS-positive configs
    print(f"\n  OOS-POSITIVE configs:")
    oos_pos = [r for r in all_results if r['oos_total'] > 0]
    if oos_pos:
        for r in sorted(oos_pos, key=lambda x: x['oos_total'], reverse=True):
            print(f"    {r['symbol']:10s} {r['config']:12s} "
                  f"OOS: avg={r['oos_avg']:+.1f}bps total={r['oos_total']:+.2f}% "
                  f"n={r['oos_n']} WR={r['oos_wr']:.1f}% Sharpe={r['oos_sharpe']:+.1f}")
    else:
        print(f"    NONE — grid strategy does not survive OOS")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
