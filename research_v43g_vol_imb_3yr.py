#!/usr/bin/env python3
"""
v43g: Volume Imbalance Momentum — 3-Year Walk-Forward Validation

Tests the promising signal from v43f on the full 3-year dataset (Jan 2023 – Feb 2026).
Processes data month-by-month to keep RAM under control.

Signal: vol_imbalance_z > 2.0 → go long (buy pressure momentum)
        vol_imbalance_z < -2.0 → go short (sell pressure momentum)
Holding: 4h (4 bars on 1h timeframe)
Entry: limit order at -5bps offset (maker fee)
Exit: timeout at 4h (market/taker fee) — no TP, no SL
Fees: entry maker 0.02% + exit taker 0.055% = 7.5 bps RT

Walk-forward: rolling 3-month train → 1-month test
Also tests z>1.5 and z>2.5 for robustness.
Also tests with TP=100 (limit exit = 4 bps RT).
Also tests on ETH and BTC for cross-symbol validation.
"""

import sys, time, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_BPS = 2.0    # 0.02%
TAKER_FEE_BPS = 5.5    # 0.055%
PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


# ============================================================================
# DATA: Build 1h bars from tick trades, one month at a time
# ============================================================================

def get_available_dates(symbol):
    """Get sorted list of available trade dates."""
    trade_dir = PARQUET_DIR / symbol / 'trades' / 'bybit_futures'
    if not trade_dir.exists():
        return []
    return sorted([f.stem for f in trade_dir.glob('*.parquet')])


def build_hourly_bars_for_dates(symbol, dates):
    """Build 1h bars from tick trades for given dates. Memory-efficient."""
    trade_dir = PARQUET_DIR / symbol / 'trades' / 'bybit_futures'
    all_bars = []

    for d in dates:
        path = trade_dir / f'{d}.parquet'
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=['timestamp_us', 'price', 'quantity', 'side'])
            ts = pd.to_datetime(df['timestamp_us'].values, unit='us')
            price = df['price'].values
            qty = df['quantity'].values
            side = df['side'].values
            del df

            # Manual hourly aggregation (faster than resample)
            hours = ts.floor('h')
            unique_hours = np.unique(hours)

            rows = []
            for hr in unique_hours:
                mask = hours == hr
                p = price[mask]
                q = qty[mask]
                s = side[mask]
                if len(p) == 0:
                    continue
                buy_mask = s == 1
                rows.append({
                    'timestamp': hr,
                    'open': p[0], 'high': p.max(), 'low': p.min(), 'close': p[-1],
                    'buy_vol': q[buy_mask].sum(), 'sell_vol': q[~buy_mask].sum(),
                    'total_vol': q.sum(), 'n_trades': len(p),
                })

            if rows:
                h = pd.DataFrame(rows).set_index('timestamp')
                all_bars.append(h)
        except Exception as e:
            continue

    if not all_bars:
        return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    return result[~result.index.duplicated(keep='first')]


def add_features(bars, window=24):
    """Add volume imbalance features. All backward-looking."""
    bars['vol_imbalance'] = ((bars['buy_vol'] - bars['sell_vol']) /
                              bars['total_vol'].clip(lower=1))

    roll_mean = bars['vol_imbalance'].rolling(window, min_periods=window//2).mean()
    roll_std = bars['vol_imbalance'].rolling(window, min_periods=window//2).std()
    bars['vol_imbalance_z'] = (bars['vol_imbalance'] - roll_mean) / roll_std.clip(lower=1e-8)

    bars['vol_ratio'] = (bars['total_vol'] /
                          bars['total_vol'].rolling(window, min_periods=window//2).mean().clip(lower=1))

    bars['ret_1h_bps'] = bars['close'].pct_change() * 10000
    return bars


# ============================================================================
# SIGNAL + SIMULATION
# ============================================================================

def run_backtest(bars, z_threshold=2.0, hold_bars=4, entry_offset_bps=5,
                 tp_bps=None, sl_bps=None, cooldown_bars=2):
    """Run backtest on pre-featured bars. Returns list of trades."""
    if 'vol_imbalance_z' not in bars.columns:
        return []

    z = bars['vol_imbalance_z'].values
    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values
    n = len(bars)
    times = bars.index

    trades = []
    last_exit = -cooldown_bars - 1

    for i in range(24, n - hold_bars - 3):
        if i <= last_exit + cooldown_bars:
            continue
        if np.isnan(z[i]) or abs(z[i]) < z_threshold:
            continue

        direction = 'long' if z[i] > 0 else 'short'
        price = close[i]

        # Entry limit
        if direction == 'long':
            entry_price = price * (1 - entry_offset_bps / 10000)
        else:
            entry_price = price * (1 + entry_offset_bps / 10000)

        # Fill within 2 bars
        fill_bar = None
        for j in range(i + 1, min(i + 3, n)):
            if direction == 'long' and low[j] <= entry_price:
                fill_bar = j
                break
            elif direction == 'short' and high[j] >= entry_price:
                fill_bar = j
                break
        if fill_bar is None:
            continue

        # TP/SL
        tp_price = None
        sl_price = None
        if tp_bps:
            tp_price = entry_price * (1 + tp_bps / 10000) if direction == 'long' else entry_price * (1 - tp_bps / 10000)
        if sl_bps:
            sl_price = entry_price * (1 - sl_bps / 10000) if direction == 'long' else entry_price * (1 + sl_bps / 10000)

        # Exit
        exit_end = min(fill_bar + hold_bars, n - 1)
        exit_price = None
        exit_reason = None
        exit_bar = exit_end

        for k in range(fill_bar + 1, exit_end + 1):
            if sl_price:
                if (direction == 'long' and low[k] <= sl_price) or \
                   (direction == 'short' and high[k] >= sl_price):
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break
            if tp_price:
                if (direction == 'long' and high[k] >= tp_price) or \
                   (direction == 'short' and low[k] <= tp_price):
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar = k
                    break

        if exit_price is None:
            exit_price = close[exit_end]
            exit_reason = 'timeout'

        # PnL
        if direction == 'long':
            raw_bps = (exit_price - entry_price) / entry_price * 10000
        else:
            raw_bps = (entry_price - exit_price) / entry_price * 10000

        entry_fee = MAKER_FEE_BPS
        exit_fee = MAKER_FEE_BPS if exit_reason == 'take_profit' else TAKER_FEE_BPS
        net_bps = raw_bps - entry_fee - exit_fee

        trades.append({
            'time': times[i],
            'direction': direction,
            'exit_reason': exit_reason,
            'raw_bps': raw_bps,
            'net_bps': net_bps,
            'hold': exit_bar - fill_bar,
        })
        last_exit = exit_bar

    return trades


def summarize(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'avg': 0, 'total': 0, 'sharpe': 0}
    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0
    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("v43g: Volume Imbalance Momentum — 3-Year Walk-Forward")
    print("=" * 80)

    configs = [
        {'label': 'z>2.0 noTP noSL',     'z': 2.0, 'tp': None, 'sl': None},
        {'label': 'z>1.5 noTP noSL',     'z': 1.5, 'tp': None, 'sl': None},
        {'label': 'z>2.5 noTP noSL',     'z': 2.5, 'tp': None, 'sl': None},
        {'label': 'z>2.0 TP=100 noSL',   'z': 2.0, 'tp': 100,  'sl': None},
        {'label': 'z>2.0 TP=50 noSL',    'z': 2.0, 'tp': 50,   'sl': None},
        {'label': 'z>2.0 TP=50 SL=100',  'z': 2.0, 'tp': 50,   'sl': 100},
        {'label': 'z>2.0 TP=100 SL=200', 'z': 2.0, 'tp': 100,  'sl': 200},
    ]

    for symbol in ['SOLUSDT', 'ETHUSDT', 'BTCUSDT']:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        all_dates = get_available_dates(symbol)
        if len(all_dates) < 90:
            print(f"  Only {len(all_dates)} days, skip")
            continue

        print(f"  Available: {len(all_dates)} days ({all_dates[0]} to {all_dates[-1]})")

        # Group dates by month
        months = {}
        for d in all_dates:
            m = d[:7]  # YYYY-MM
            months.setdefault(m, []).append(d)
        month_keys = sorted(months.keys())
        print(f"  Months: {len(month_keys)} ({month_keys[0]} to {month_keys[-1]})")

        # Walk-forward: 3-month train → 1-month test
        # But we don't actually need training (fixed params), so just test month by month
        # and report rolling results

        monthly_results = {cfg['label']: [] for cfg in configs}

        for mi, mk in enumerate(month_keys):
            t1 = time.time()
            dates = months[mk]

            # Need context from previous month for rolling features
            prev_dates = []
            if mi > 0:
                prev_dates = months[month_keys[mi-1]][-3:]  # last 3 days of prev month

            all_month_dates = prev_dates + dates
            bars = build_hourly_bars_for_dates(symbol, all_month_dates)
            if bars.empty or len(bars) < 48:
                print(f"  [{mi+1}/{len(month_keys)}] {mk}: skip (few bars)", flush=True)
                continue

            bars = add_features(bars)

            # Only count trades in the actual test month (not context days)
            cutoff = pd.Timestamp(dates[0]) if prev_dates else bars.index[0]

            n_first = 0
            for ci, cfg in enumerate(configs):
                trades = run_backtest(bars, z_threshold=cfg['z'],
                                       tp_bps=cfg['tp'], sl_bps=cfg['sl'])
                trades = [t for t in trades if t['time'] >= cutoff]
                s = summarize(trades)
                monthly_results[cfg['label']].append({
                    'month': mk, 'n': s['n'], 'avg': s['avg'],
                    'total': s['total'], 'wr': s['wr'],
                })
                if ci == 0:
                    n_first = s['n']

            elapsed = time.time() - t1
            total_elapsed = time.time() - t0
            eta = total_elapsed / (mi + 1) * (len(month_keys) - mi - 1)
            print(f"  [{mi+1}/{len(month_keys)}] {mk}: {len(bars)}bars "
                  f"n={n_first} {elapsed:.1f}s ETA={eta:.0f}s RAM={get_ram_mb():.0f}MB",
                  flush=True)

            del bars
            gc.collect()

        # ============================================================
        # SUMMARY for this symbol
        # ============================================================
        print(f"\n  {'='*70}")
        print(f"  SUMMARY: {symbol}")
        print(f"  {'='*70}")

        for cfg in configs:
            results = monthly_results[cfg['label']]
            if not results:
                continue

            total_n = sum(r['n'] for r in results)
            total_pct = sum(r['total'] for r in results)
            pos_months = sum(1 for r in results if r['total'] > 0)
            total_months = len(results)

            # Weighted average bps
            all_avg = [r['avg'] for r in results if r['n'] > 0]
            weighted_avg = np.mean(all_avg) if all_avg else 0

            # Monthly Sharpe
            monthly_totals = [r['total'] for r in results]
            m_avg = np.mean(monthly_totals)
            m_std = np.std(monthly_totals) if len(monthly_totals) > 1 else 1
            monthly_sharpe = m_avg / m_std * np.sqrt(12) if m_std > 0 else 0

            print(f"\n  {cfg['label']:25s}")
            print(f"    Total trades: {total_n:5d} over {total_months} months")
            print(f"    Total PnL:    {total_pct:+.2f}%")
            print(f"    Avg per trade: {weighted_avg:+.1f} bps")
            print(f"    Positive months: {pos_months}/{total_months} "
                  f"({pos_months/total_months*100:.0f}%)")
            print(f"    Monthly Sharpe: {monthly_sharpe:+.2f}")

            # Show monthly breakdown
            print(f"    Monthly: ", end='')
            for r in results:
                sign = '+' if r['total'] >= 0 else ''
                print(f"{r['month']}:{sign}{r['total']:.1f}%({r['n']})", end=' ')
            print()

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
