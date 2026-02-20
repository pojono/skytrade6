#!/usr/bin/env python3
"""
v43d: Adaptive Grid Strategy with Volatility Filter

Proven edges combined:
1. Grid bots earn cell_width - fees on every round-trip (v14: Sharpe 0.83-1.84)
2. Vol prediction R²=0.34 at 1h (v9, v10) — can predict when to widen/narrow grid
3. Markets are range-bound 85-94% of time (v8)

Strategy:
  - Place symmetric limit orders above and below current price
  - Grid spacing = f(predicted_volatility) — wider in high vol, narrower in low vol
  - When one side fills → place TP limit order at opposite grid level
  - Fees: maker+maker = 4 bps round-trip (both entry and exit are limit orders)
  - Vol filter: don't enter new positions when vol is extreme (trending)

Key difference from v14-v17 grid research:
  - Uses 5-sec ticker data (not just OHLCV)
  - Adaptive spacing based on realized vol
  - Explicit vol regime filter
  - Tests on fresh data period

Simulation on 1-min bars. No trailing stop. All limit orders.
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_PCT = 0.02  # 2 bps per side
PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


# ============================================================================
# DATA
# ============================================================================

def load_bars(symbol, dates, freq='1min'):
    """Load ticker parquet → OHLC bars + vol features."""
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    all_dfs = []
    t0 = time.time()

    for i, d in enumerate(dates, 1):
        path = ticker_dir / f'{d}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp_us'], unit='us')
        df = df.set_index('timestamp').sort_index()
        all_dfs.append(df)
        if i % 20 == 0 or i == len(dates):
            print(f"  [{i}/{len(dates)}] {time.time()-t0:.1f}s", flush=True)

    if not all_dfs:
        return pd.DataFrame()

    raw = pd.concat(all_dfs).sort_index()
    raw = raw[~raw.index.duplicated(keep='first')]

    bars = raw['last_price'].resample(freq).ohlc().dropna()
    bars['oi'] = raw['open_interest'].resample(freq).last()
    bars['funding_rate'] = raw['funding_rate'].resample(freq).last()
    print(f"  {len(bars):,} {freq} bars, RAM={get_ram_mb():.0f}MB")

    del raw
    return bars


# ============================================================================
# GRID SIMULATION
# ============================================================================

def simulate_grid(bars, grid_spacing_bps=50, max_hold_bars=240,
                  vol_window=60, vol_filter_mult=None,
                  rebalance_interval=60, max_inventory=1):
    """
    Simulate a simple symmetric grid strategy.

    Every rebalance_interval bars:
      - If no position: place buy limit at -grid_spacing/2 and sell limit at +grid_spacing/2
      - If position exists: place TP limit at entry ± grid_spacing (opposite side)

    Grid spacing can be fixed or adaptive (based on rolling vol).

    All orders are limit (maker fee).
    Max hold = max_hold_bars, exit at market (taker fee) if timeout.

    Returns list of completed round-trips.
    """
    close = bars['close'].values
    high = bars['high'].values
    low = bars['low'].values
    n = len(close)

    # Rolling volatility for adaptive spacing
    ret = np.zeros(n)
    ret[1:] = np.abs(close[1:] - close[:-1]) / close[:-1] * 10000  # abs return in bps
    roll_vol = pd.Series(ret).rolling(vol_window, min_periods=vol_window//2).mean().values

    trades = []
    position = None  # {'direction', 'entry_price', 'entry_bar', 'tp_price'}

    for i in range(vol_window, n - max_hold_bars):
        # Check if we have an open position
        if position is not None:
            entry_bar = position['entry_bar']
            hold = i - entry_bar

            # Check TP (limit order)
            tp = position['tp_price']
            direction = position['direction']

            if direction == 'long' and high[i] >= tp:
                # TP hit — exit at TP price (limit order, maker fee)
                pnl_bps = (tp - position['entry_price']) / position['entry_price'] * 10000
                fee_bps = MAKER_FEE_PCT * 100 * 2  # entry + exit both maker
                net_bps = pnl_bps - fee_bps
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': direction,
                    'entry_price': position['entry_price'],
                    'exit_price': tp,
                    'pnl_bps': pnl_bps,
                    'net_bps': net_bps,
                    'exit_reason': 'take_profit',
                    'hold_bars': hold,
                    'spacing': position.get('spacing', grid_spacing_bps),
                })
                position = None
                continue

            elif direction == 'short' and low[i] <= tp:
                pnl_bps = (position['entry_price'] - tp) / position['entry_price'] * 10000
                fee_bps = MAKER_FEE_PCT * 100 * 2
                net_bps = pnl_bps - fee_bps
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': direction,
                    'entry_price': position['entry_price'],
                    'exit_price': tp,
                    'pnl_bps': pnl_bps,
                    'net_bps': net_bps,
                    'exit_reason': 'take_profit',
                    'hold_bars': hold,
                    'spacing': position.get('spacing', grid_spacing_bps),
                })
                position = None
                continue

            # Check timeout
            if hold >= max_hold_bars:
                exit_price = close[i]
                if direction == 'long':
                    pnl_bps = (exit_price - position['entry_price']) / position['entry_price'] * 10000
                else:
                    pnl_bps = (position['entry_price'] - exit_price) / position['entry_price'] * 10000
                fee_bps = MAKER_FEE_PCT * 100 + 0.055 * 100  # entry maker + exit taker
                net_bps = pnl_bps - fee_bps
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': direction,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_bps': pnl_bps,
                    'net_bps': net_bps,
                    'exit_reason': 'timeout',
                    'hold_bars': hold,
                    'spacing': position.get('spacing', grid_spacing_bps),
                })
                position = None
                continue

            continue  # Position open, wait

        # No position — check if we should enter
        # Only enter at rebalance intervals
        if i % rebalance_interval != 0:
            continue

        # Vol filter: skip if vol is too high (trending market)
        current_vol = roll_vol[i]
        if np.isnan(current_vol) or current_vol < 0.1:
            continue

        if vol_filter_mult is not None:
            # Compute rolling vol median
            vol_window_long = min(i, vol_window * 4)
            vol_median = np.nanmedian(roll_vol[max(0, i-vol_window_long):i])
            if current_vol > vol_median * vol_filter_mult:
                continue  # Vol too high, skip

        # Adaptive spacing
        spacing = grid_spacing_bps
        # Could make adaptive: spacing = max(grid_spacing_bps, current_vol * K)
        # For now test fixed spacing first

        # Place symmetric limit orders
        mid = close[i]
        buy_limit = mid * (1 - spacing / 2 / 10000)
        sell_limit = mid * (1 + spacing / 2 / 10000)

        # Check if either fills in the next bar
        # (We place orders now, check fills on subsequent bars)
        # Look ahead up to rebalance_interval bars for a fill
        fill_window = min(i + rebalance_interval, n - max_hold_bars)

        for j in range(i + 1, fill_window):
            if low[j] <= buy_limit:
                # Buy limit filled
                tp_price = buy_limit + (spacing / 10000) * buy_limit
                position = {
                    'direction': 'long',
                    'entry_price': buy_limit,
                    'entry_bar': j,
                    'tp_price': tp_price,
                    'spacing': spacing,
                }
                break
            elif high[j] >= sell_limit:
                # Sell limit filled
                tp_price = sell_limit - (spacing / 10000) * sell_limit
                position = {
                    'direction': 'short',
                    'entry_price': sell_limit,
                    'entry_bar': j,
                    'tp_price': tp_price,
                    'spacing': spacing,
                }
                break

    # Close any remaining position
    if position is not None:
        direction = position['direction']
        exit_price = close[-1]
        if direction == 'long':
            pnl_bps = (exit_price - position['entry_price']) / position['entry_price'] * 10000
        else:
            pnl_bps = (position['entry_price'] - exit_price) / position['entry_price'] * 10000
        fee_bps = MAKER_FEE_PCT * 100 + 0.055 * 100
        trades.append({
            'entry_bar': position['entry_bar'],
            'exit_bar': len(close) - 1,
            'direction': direction,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_bps': pnl_bps,
            'net_bps': pnl_bps - fee_bps,
            'exit_reason': 'end',
            'hold_bars': len(close) - 1 - position['entry_bar'],
            'spacing': position.get('spacing', grid_spacing_bps),
        })

    return trades


def analyze(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES")
        return None

    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    total_pct = net.sum() / 100  # bps → %
    avg = net.mean()
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0

    reasons = {}
    for t in trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1

    avg_hold = np.mean([t['hold_bars'] for t in trades])

    # TP trades vs timeout trades
    tp_trades = [t for t in trades if t['exit_reason'] == 'take_profit']
    to_trades = [t for t in trades if t['exit_reason'] == 'timeout']

    tp_pct = len(tp_trades) / n * 100 if n > 0 else 0

    print(f"  {label}")
    print(f"    n={n:4d}  WR={wr:5.1f}%  avg={avg:+7.1f}bps  "
          f"total={total_pct:+7.2f}%  Sharpe={sharpe:+6.1f}  "
          f"TP%={tp_pct:.0f}%  avgHold={avg_hold:.0f}min  exits={reasons}")

    if tp_trades:
        tp_net = np.array([t['net_bps'] for t in tp_trades])
        print(f"    TP trades: avg={tp_net.mean():+.1f}bps")
    if to_trades:
        to_net = np.array([t['net_bps'] for t in to_trades])
        print(f"    Timeout trades: avg={to_net.mean():+.1f}bps")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total_pct, 'sharpe': sharpe,
            'tp_pct': tp_pct}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("v43d: Adaptive Grid Strategy with Vol Filter")
    print("=" * 70)

    for symbol in ['SOLUSDT', 'ETHUSDT']:
        print(f"\n{'='*70}")
        print(f"  {symbol}")
        print(f"{'='*70}")

        ticker_dir = PARQUET_DIR / symbol / 'ticker'
        available = sorted([f.stem for f in ticker_dir.glob('*.parquet')])
        if not available:
            continue

        # Use 14 days for grid test (need more data than MR)
        test_dates = available[:14]
        print(f"  Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

        bars = load_bars(symbol, test_dates)
        if bars.empty:
            continue

        # ============================================================
        # SWEEP: grid spacing × max hold × vol filter
        # ============================================================
        results = []

        print(f"\n  --- Fixed Grid (no vol filter) ---")
        for spacing in [30, 50, 80, 100, 150, 200]:
            for max_hold in [60, 120, 240, 480]:
                for rebal in [15, 30, 60]:
                    trades = simulate_grid(bars, grid_spacing_bps=spacing,
                                           max_hold_bars=max_hold,
                                           rebalance_interval=rebal,
                                           vol_filter_mult=None)
                    r = analyze(trades, f"spacing={spacing}bps hold={max_hold}m rebal={rebal}m")
                    if r and r['n'] >= 5:
                        r.update({'spacing': spacing, 'hold': max_hold,
                                  'rebal': rebal, 'vol_filter': None})
                        results.append(r)

        print(f"\n  --- Grid with Vol Filter ---")
        for spacing in [50, 80, 100, 150]:
            for max_hold in [120, 240]:
                for vol_mult in [1.5, 2.0, 3.0]:
                    trades = simulate_grid(bars, grid_spacing_bps=spacing,
                                           max_hold_bars=max_hold,
                                           rebalance_interval=30,
                                           vol_filter_mult=vol_mult)
                    r = analyze(trades, f"spacing={spacing} hold={max_hold} vol<{vol_mult}x")
                    if r and r['n'] >= 5:
                        r.update({'spacing': spacing, 'hold': max_hold,
                                  'rebal': 30, 'vol_filter': vol_mult})
                        results.append(r)

        # ============================================================
        # TOP RESULTS
        # ============================================================
        if results:
            print(f"\n  {'='*60}")
            print(f"  TOP 10 BY AVG NET BPS (min 5 trades)")
            print(f"  {'='*60}")

            results.sort(key=lambda x: x['avg'], reverse=True)
            for i, r in enumerate(results[:10], 1):
                vf = f"vol<{r['vol_filter']}x" if r['vol_filter'] else "no_vf"
                print(f"  #{i}: sp={r['spacing']:3d} hold={r['hold']:3d} "
                      f"rebal={r['rebal']:2d} {vf:10s} | "
                      f"n={r['n']:4d} WR={r['wr']:5.1f}% avg={r['avg']:+7.1f}bps "
                      f"total={r['total']:+7.2f}% Sharpe={r['sharpe']:+6.1f} "
                      f"TP%={r['tp_pct']:.0f}%")

            print(f"\n  BOTTOM 3:")
            for i, r in enumerate(results[-3:], 1):
                vf = f"vol<{r['vol_filter']}x" if r['vol_filter'] else "no_vf"
                print(f"  #{i}: sp={r['spacing']:3d} hold={r['hold']:3d} | "
                      f"n={r['n']:4d} avg={r['avg']:+7.1f}bps total={r['total']:+7.2f}%")

        import gc; gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
