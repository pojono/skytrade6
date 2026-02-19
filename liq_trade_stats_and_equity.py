#!/usr/bin/env python3
"""
Trade frequency, duration stats, and equity curves for the liquidation cascade strategy.

Uses the best config from research: A1 (70% TP@12bps + 30% trail@3bps)
Also shows C1 (50% TP@10bps, trail 5→3bps) for comparison.

Loads both IS (May-Aug 2025) and OOS (Feb 2026) data.
Outputs:
  - Trade frequency stats (avg/min/max per day/week/month per symbol)
  - Trade duration stats (avg/min/max minutes per symbol)
  - Equity curve PNGs per symbol and combined
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, will skip PNG generation")

MAKER_FEE = 0.02
TAKER_FEE = 0.055
SYMBOLS = ['BTCUSDT', 'DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']
SYMBOL_SHORT = {'BTCUSDT': 'BTC', 'DOGEUSDT': 'DOGE', 'SOLUSDT': 'SOL',
                'ETHUSDT': 'ETH', 'XRPUSDT': 'XRP'}

# ============================================================================
# DATA LOADING (reused from OOS script)
# ============================================================================

def load_liquidations(symbol, data_dir='data', date_prefix=None):
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    if date_prefix:
        liq_files = [f for f in liq_files if date_prefix in f.name]
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
    if len(df) > 0:
        df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_csv(symbol, csv_name, data_dir='data'):
    csv_path = Path(data_dir) / symbol / csv_name
    if not csv_path.exists():
        return pd.DataFrame()
    print(f"  Loading {csv_name}...", end='', flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
    df['price'] = df['price'].astype(float)
    df = df[['timestamp', 'price']].sort_values('timestamp').reset_index(drop=True)
    print(f" done ({len(df):,})")
    return df


def build_bars(ticker_df, freq='1min'):
    ticker_df = ticker_df.set_index('timestamp')
    bars = ticker_df['price'].resample(freq).ohlc().dropna()
    return bars


# ============================================================================
# CASCADE DETECTION
# ============================================================================

def detect_signals(liq_df, price_bars, p95_threshold):
    large = liq_df[liq_df['notional'] >= p95_threshold].copy()
    if len(large) == 0:
        return []
    bar_index = price_bars.index
    bar_close = price_bars['close'].values
    timestamps = large['timestamp'].values
    sides = large['side'].values
    notionals = large['notional'].values
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
        c_sides = sides[cluster]
        c_notionals = notionals[cluster]
        c_ts = timestamps[cluster]
        buy_not = c_notionals[c_sides == 'Buy'].sum()
        sell_not = c_notionals[c_sides == 'Sell'].sum()
        buy_dominant = buy_not > sell_not
        end_ts = pd.Timestamp(c_ts[-1])
        end_idx = bar_index.searchsorted(end_ts)
        if end_idx >= len(bar_close) - 120 or end_idx < 10:
            i = cluster[-1] + 1
            continue
        current_price = bar_close[end_idx]
        start_idx = bar_index.searchsorted(pd.Timestamp(c_ts[0]))
        if start_idx > 0:
            pre_price = bar_close[max(0, start_idx - 1)]
            cascade_disp_bps = (current_price - pre_price) / pre_price * 10000
        else:
            cascade_disp_bps = 0
        cascades.append({
            'end': end_ts,
            'n_events': len(cluster),
            'buy_dominant': buy_dominant,
            'end_bar_idx': end_idx,
            'current_price': current_price,
            'cascade_disp_bps': cascade_disp_bps,
        })
        i = cluster[-1] + 1
    return cascades


# ============================================================================
# SIMULATION — returns trades with duration info
# ============================================================================

def find_fill(direction, limit_price, bar_high, bar_low, idx, end_bar):
    for j in range(idx, end_bar + 1):
        if direction == 'long' and bar_low[j] <= limit_price:
            return j
        elif direction == 'short' and bar_high[j] >= limit_price:
            return j
    return None


def run_trades_with_duration(cascades, price_bars, variant, params,
                             entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10):
    """Run simulation, returning trades with timing info."""
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values
    bar_close = price_bars['close'].values
    bar_index = price_bars.index
    n_bars = len(bar_close)
    trades = []
    last_trade_time = None

    for cascade in cascades:
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < 5 * 60:
                continue
        if abs(cascade['cascade_disp_bps']) < min_disp_bps:
            continue

        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']
        direction = 'long' if cascade['buy_dominant'] else 'short'

        if direction == 'long':
            limit_price = current_price * (1 - entry_offset_pct / 100)
        else:
            limit_price = current_price * (1 + entry_offset_pct / 100)

        end_bar = min(idx + max_hold_min, n_bars - 1)
        fill_bar = find_fill(direction, limit_price, bar_high, bar_low, idx, end_bar)
        if fill_bar is None:
            continue

        remaining = max_hold_min - (fill_bar - idx)
        exit_end = min(fill_bar + remaining, n_bars - 1)

        if variant == 'A':
            legs = _sim_variant_A(direction, limit_price, fill_bar, exit_end,
                                  bar_high, bar_low, bar_close, params)
        elif variant == 'C':
            legs = _sim_variant_C(direction, limit_price, fill_bar, exit_end,
                                  bar_high, bar_low, bar_close, params)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Find the last exit bar
        max_exit_bar = fill_bar
        total_net_pnl = 0.0
        exit_reasons = []
        for leg in legs:
            weight = leg['weight']
            ep = leg['exit_price']
            reason = leg['exit_reason']
            exit_bar = leg.get('exit_bar', exit_end)
            max_exit_bar = max(max_exit_bar, exit_bar)
            if direction == 'long':
                raw_pnl = (ep - limit_price) / limit_price * 100
            else:
                raw_pnl = (limit_price - ep) / limit_price * 100
            entry_fee = MAKER_FEE
            if reason in ('take_profit', 'trail_limit', 'partial_tp', 'milestone'):
                exit_fee = MAKER_FEE
            else:
                exit_fee = TAKER_FEE
            net = (raw_pnl - entry_fee - exit_fee) * weight
            total_net_pnl += net
            exit_reasons.append(reason)

        fill_time = bar_index[fill_bar] if fill_bar < len(bar_index) else cascade['end']
        exit_time = bar_index[min(max_exit_bar, len(bar_index) - 1)]
        # Duration from signal to last exit (full lifecycle)
        signal_to_exit_sec = (exit_time - cascade['end']).total_seconds()
        # Duration from fill to last exit (position held)
        fill_to_exit_sec = (exit_time - fill_time).total_seconds()
        # Duration from signal to fill (waiting for entry)
        signal_to_fill_sec = (fill_time - cascade['end']).total_seconds()

        trades.append({
            'net_pnl': total_net_pnl,
            'exit_reason': '+'.join(sorted(set(exit_reasons))),
            'time': cascade['end'],
            'fill_time': fill_time,
            'exit_time': exit_time,
            'direction': direction,
            'signal_to_exit_min': signal_to_exit_sec / 60,
            'fill_to_exit_min': fill_to_exit_sec / 60,
            'signal_to_fill_min': signal_to_fill_sec / 60,
            'fill_bar': fill_bar,
            'exit_bar': max_exit_bar,
        })
        last_trade_time = cascade['end']

    return trades


# ── Variant A ──

def _sim_variant_A(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    tp_frac = params['tp_frac']
    tp_bps = params['tp_bps']
    trail_bps = params['trail_bps']
    if direction == 'long':
        tp_price = fill_price * (1 + tp_bps / 10000)
    else:
        tp_price = fill_price * (1 - tp_bps / 10000)
    legs = []
    tp_filled = False
    peak = fill_price
    for k in range(fill_bar, exit_end + 1):
        if direction == 'long':
            peak = max(peak, bar_high[k])
        else:
            peak = min(peak, bar_low[k])
        if not tp_filled:
            if direction == 'long' and bar_high[k] >= tp_price:
                legs.append({'weight': tp_frac, 'exit_price': tp_price,
                             'exit_reason': 'partial_tp', 'exit_bar': k})
                tp_filled = True
            elif direction == 'short' and bar_low[k] <= tp_price:
                legs.append({'weight': tp_frac, 'exit_price': tp_price,
                             'exit_reason': 'partial_tp', 'exit_bar': k})
                tp_filled = True
        if direction == 'long':
            trail_level = peak * (1 - trail_bps / 10000)
            if bar_low[k] <= trail_level:
                legs.append({'weight': 1.0 - tp_frac, 'exit_price': max(trail_level, bar_low[k]),
                             'exit_reason': 'trail_limit', 'exit_bar': k})
                if not tp_filled:
                    legs.append({'weight': tp_frac, 'exit_price': max(trail_level, bar_low[k]),
                                 'exit_reason': 'trail_limit', 'exit_bar': k})
                return legs
        else:
            trail_level = peak * (1 + trail_bps / 10000)
            if bar_high[k] >= trail_level:
                legs.append({'weight': 1.0 - tp_frac, 'exit_price': min(trail_level, bar_high[k]),
                             'exit_reason': 'trail_limit', 'exit_bar': k})
                if not tp_filled:
                    legs.append({'weight': tp_frac, 'exit_price': min(trail_level, bar_high[k]),
                                 'exit_reason': 'trail_limit', 'exit_bar': k})
                return legs
    timeout_price = bar_close[exit_end]
    if not tp_filled:
        legs.append({'weight': tp_frac, 'exit_price': timeout_price,
                     'exit_reason': 'timeout', 'exit_bar': exit_end})
    legs.append({'weight': 1.0 - tp_frac, 'exit_price': timeout_price,
                 'exit_reason': 'timeout', 'exit_bar': exit_end})
    return legs


# ── Variant C ──

def _sim_variant_C(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    tp_frac = params['tp_frac']
    tp_bps = params['tp_bps']
    trail_bps = params['trail_bps']
    trail_tight_bps = params['trail_tight_bps']
    if direction == 'long':
        tp_price = fill_price * (1 + tp_bps / 10000)
    else:
        tp_price = fill_price * (1 - tp_bps / 10000)
    peak = fill_price
    tp_filled = False
    tp_bar = None
    current_trail_bps = trail_bps
    for k in range(fill_bar, exit_end + 1):
        if direction == 'long':
            peak = max(peak, bar_high[k])
        else:
            peak = min(peak, bar_low[k])
        if not tp_filled:
            if direction == 'long' and bar_high[k] >= tp_price:
                tp_filled = True
                tp_bar = k
                current_trail_bps = trail_tight_bps
                peak = bar_high[k] if direction == 'long' else bar_low[k]
                continue
            elif direction == 'short' and bar_low[k] <= tp_price:
                tp_filled = True
                tp_bar = k
                current_trail_bps = trail_tight_bps
                peak = bar_low[k] if direction == 'short' else bar_high[k]
                continue
        if direction == 'long':
            trail_level = peak * (1 - current_trail_bps / 10000)
            if bar_low[k] <= trail_level:
                trail_exit = max(trail_level, bar_low[k])
                if tp_filled:
                    return [
                        {'weight': tp_frac, 'exit_price': tp_price,
                         'exit_reason': 'partial_tp', 'exit_bar': tp_bar},
                        {'weight': 1.0 - tp_frac, 'exit_price': trail_exit,
                         'exit_reason': 'trail_limit', 'exit_bar': k},
                    ]
                else:
                    return [{'weight': 1.0, 'exit_price': trail_exit,
                             'exit_reason': 'trail_limit', 'exit_bar': k}]
        else:
            trail_level = peak * (1 + current_trail_bps / 10000)
            if bar_high[k] >= trail_level:
                trail_exit = min(trail_level, bar_high[k])
                if tp_filled:
                    return [
                        {'weight': tp_frac, 'exit_price': tp_price,
                         'exit_reason': 'partial_tp', 'exit_bar': tp_bar},
                        {'weight': 1.0 - tp_frac, 'exit_price': trail_exit,
                         'exit_reason': 'trail_limit', 'exit_bar': k},
                    ]
                else:
                    return [{'weight': 1.0, 'exit_price': trail_exit,
                             'exit_reason': 'trail_limit', 'exit_bar': k}]
    timeout_price = bar_close[exit_end]
    legs = []
    if tp_filled:
        legs.append({'weight': tp_frac, 'exit_price': tp_price,
                     'exit_reason': 'partial_tp', 'exit_bar': tp_bar})
        legs.append({'weight': 1.0 - tp_frac, 'exit_price': timeout_price,
                     'exit_reason': 'timeout', 'exit_bar': exit_end})
    else:
        legs.append({'weight': 1.0, 'exit_price': timeout_price,
                     'exit_reason': 'timeout', 'exit_bar': exit_end})
    return legs


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_global = time.time()

    print("=" * 100)
    print("  TRADE STATISTICS & EQUITY CURVES")
    print("  Config: A1 (70% TP@12bps + 30% trail@3bps) — best Sharpe")
    print("  Also: C1 (50% TP@10bps, trail 5→3bps) — highest return")
    print("=" * 100)

    A1_PARAMS = {'tp_frac': 0.7, 'tp_bps': 12, 'trail_bps': 3}
    C1_PARAMS = {'tp_frac': 0.5, 'tp_bps': 10, 'trail_bps': 5, 'trail_tight_bps': 3}

    # ── Load all data ──
    print(f"\n{'#' * 100}")
    print(f"  LOADING DATA")
    print(f"{'#' * 100}\n")

    sym_data = {}
    for sym in SYMBOLS:
        print(f"── {sym} ──")

        # IS liquidations (2025)
        print(f"  [IS 2025]")
        liq_is = load_liquidations(sym, date_prefix='2025')
        p95 = liq_is['notional'].quantile(0.95) if len(liq_is) > 0 else 0
        print(f"    P95: ${p95:,.0f}")

        # IS ticker
        ticker_is = load_ticker_csv(sym, 'ticker_prices.csv.gz')
        bars_is = build_bars(ticker_is) if len(ticker_is) > 0 else pd.DataFrame()
        print(f"    Bars: {len(bars_is):,}")

        # OOS liquidations (Feb 2026)
        print(f"  [OOS Feb 2026]")
        liq_oos = load_liquidations(sym, date_prefix='2026')

        # OOS ticker
        ticker_oos = load_ticker_csv(sym, 'ticker_prices_feb2026.csv.gz')
        bars_oos = build_bars(ticker_oos) if len(ticker_oos) > 0 else pd.DataFrame()
        print(f"    Bars: {len(bars_oos):,}")

        # Detect signals
        cascades_is = detect_signals(liq_is, bars_is, p95) if len(bars_is) > 0 else []
        cascades_oos = detect_signals(liq_oos, bars_oos, p95) if len(bars_oos) > 0 else []
        print(f"    Signals IS: {len(cascades_is)}, OOS: {len(cascades_oos)}")

        # Run A1 on both periods
        trades_is = run_trades_with_duration(cascades_is, bars_is, 'A', A1_PARAMS) if cascades_is else []
        trades_oos = run_trades_with_duration(cascades_oos, bars_oos, 'A', A1_PARAMS) if cascades_oos else []
        print(f"    Trades A1: IS={len(trades_is)}, OOS={len(trades_oos)}")

        # Run C1 on both periods
        trades_c1_is = run_trades_with_duration(cascades_is, bars_is, 'C', C1_PARAMS) if cascades_is else []
        trades_c1_oos = run_trades_with_duration(cascades_oos, bars_oos, 'C', C1_PARAMS) if cascades_oos else []
        print(f"    Trades C1: IS={len(trades_c1_is)}, OOS={len(trades_c1_oos)}")
        print()

        sym_data[sym] = {
            'trades_a1_is': trades_is,
            'trades_a1_oos': trades_oos,
            'trades_c1_is': trades_c1_is,
            'trades_c1_oos': trades_c1_oos,
            'p95': p95,
        }

    # ── PART 1: Trade Frequency Stats ──
    print(f"\n{'#' * 100}")
    print(f"  PART 1: TRADE FREQUENCY (A1: 70% TP@12bps + 30% trail@3bps)")
    print(f"{'#' * 100}\n")

    for period_label, trades_key in [("IN-SAMPLE (May-Aug 2025, 87 days)", 'trades_a1_is'),
                                      ("OUT-OF-SAMPLE (Feb 2026, 11 days)", 'trades_a1_oos')]:
        print(f"  ── {period_label} ──\n")
        print(f"    {'Symbol':<10s}  {'Total':>5s}  {'Days':>4s}  "
              f"{'Avg/day':>7s}  {'Min/day':>7s}  {'Max/day':>7s}  "
              f"{'Avg/wk':>7s}  {'Min/wk':>7s}  {'Max/wk':>7s}  "
              f"{'Avg/mo':>7s}")
        print(f"    {'─'*10}  {'─'*5}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

        for sym in SYMBOLS:
            trades = sym_data[sym][trades_key]
            if not trades:
                print(f"    {SYMBOL_SHORT[sym]:<10s}  {'0':>5s}  {'—':>4s}")
                continue

            df = pd.DataFrame(trades)
            df['date'] = df['time'].dt.date
            df['week'] = df['time'].dt.isocalendar().week
            df['month'] = df['time'].dt.to_period('M')

            daily = df.groupby('date').size()
            weekly = df.groupby('week').size()

            n_days = (df['time'].max() - df['time'].min()).days + 1
            n_months = max(1, n_days / 30)

            print(f"    {SYMBOL_SHORT[sym]:<10s}  {len(trades):>5d}  {n_days:>4d}  "
                  f"{daily.mean():>7.1f}  {daily.min():>7d}  {daily.max():>7d}  "
                  f"{weekly.mean():>7.1f}  {weekly.min():>7d}  {weekly.max():>7d}  "
                  f"{len(trades)/n_months:>7.1f}")
        print()

    # Combined across all symbols
    print(f"  ── COMBINED (all 5 symbols) ──\n")
    for period_label, trades_key in [("IS (87d)", 'trades_a1_is'),
                                      ("OOS (11d)", 'trades_a1_oos')]:
        all_trades = []
        for sym in SYMBOLS:
            all_trades.extend(sym_data[sym][trades_key])
        if not all_trades:
            continue
        df = pd.DataFrame(all_trades)
        df['date'] = df['time'].dt.date
        df['week'] = df['time'].dt.isocalendar().week
        daily = df.groupby('date').size()
        weekly = df.groupby('week').size()
        n_days = (df['time'].max() - df['time'].min()).days + 1
        n_months = max(1, n_days / 30)
        print(f"    {period_label:<10s}  n={len(all_trades):>5d}  "
              f"avg/day={daily.mean():.1f}  min/day={daily.min()}  max/day={daily.max()}  "
              f"avg/wk={weekly.mean():.1f}  min/wk={weekly.min()}  max/wk={weekly.max()}  "
              f"avg/mo={len(all_trades)/n_months:.0f}")
    print()

    # ── PART 2: Trade Duration Stats ──
    print(f"\n{'#' * 100}")
    print(f"  PART 2: TRADE DURATION (minutes)")
    print(f"{'#' * 100}\n")

    dur_sections = [
        ("Signal → Exit (full trade lifecycle, minutes)", 'signal_to_exit_min'),
        ("Signal → Fill (waiting for entry, minutes)", 'signal_to_fill_min'),
    ]
    for dur_label, dur_key in dur_sections:
        print(f"  ━━ {dur_label} ━━\n")
        for period_label, trades_key in [("IN-SAMPLE (May-Aug 2025)", 'trades_a1_is'),
                                          ("OUT-OF-SAMPLE (Feb 2026)", 'trades_a1_oos')]:
            print(f"    ── {period_label} ──")
            print(f"      {'Symbol':<10s}  {'n':>5s}  {'Avg':>7s}  {'Med':>7s}  {'Min':>7s}  {'Max':>7s}  "
                  f"{'P10':>7s}  {'P25':>7s}  {'P75':>7s}  {'P90':>7s}")
            print(f"      {'─'*10}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

            for sym in SYMBOLS:
                trades = sym_data[sym][trades_key]
                if not trades:
                    print(f"      {SYMBOL_SHORT[sym]:<10s}  {'0':>5s}")
                    continue
                durations = np.array([t[dur_key] for t in trades])
                print(f"      {SYMBOL_SHORT[sym]:<10s}  {len(trades):>5d}  "
                      f"{durations.mean():>7.1f}  {np.median(durations):>7.1f}  "
                      f"{durations.min():>7.1f}  {durations.max():>7.1f}  "
                      f"{np.percentile(durations, 10):>7.1f}  {np.percentile(durations, 25):>7.1f}  "
                      f"{np.percentile(durations, 75):>7.1f}  {np.percentile(durations, 90):>7.1f}")

            # Combined
            all_dur = []
            for sym in SYMBOLS:
                all_dur.extend([t[dur_key] for t in sym_data[sym][trades_key]])
            if all_dur:
                d = np.array(all_dur)
                print(f"      {'COMBINED':<10s}  {len(d):>5d}  "
                      f"{d.mean():>7.1f}  {np.median(d):>7.1f}  "
                      f"{d.min():>7.1f}  {d.max():>7.1f}  "
                      f"{np.percentile(d, 10):>7.1f}  {np.percentile(d, 25):>7.1f}  "
                      f"{np.percentile(d, 75):>7.1f}  {np.percentile(d, 90):>7.1f}")
            print()
        print()

    # Fill → Exit: report in bar counts since it's sub-minute
    print(f"  ━━ Fill → Exit (position held, in 1-min bars) ━━")
    print(f"  Note: With 3bps trail on 1-min bars, most trades exit on the same bar as fill.")
    print(f"  This means position is held for < 1 minute in most cases.\n")
    for period_label, trades_key in [("IN-SAMPLE", 'trades_a1_is'),
                                      ("OOS", 'trades_a1_oos')]:
        all_bars = []
        for sym in SYMBOLS:
            for t in sym_data[sym][trades_key]:
                all_bars.append(t['exit_bar'] - t['fill_bar'])
        if all_bars:
            d = np.array(all_bars)
            print(f"    {period_label}: n={len(d)}  avg={d.mean():.2f} bars  "
                  f"med={np.median(d):.0f}  max={d.max()}  "
                  f"same-bar={100*(d==0).mean():.1f}%  "
                  f"1-bar={100*(d==1).mean():.1f}%  "
                  f"2+bars={100*(d>=2).mean():.1f}%")
    print()

    # ── PART 3: Equity Curves ──
    if not HAS_MPL:
        print("  Skipping equity curves (matplotlib not available)")
        return

    print(f"\n{'#' * 100}")
    print(f"  PART 3: EQUITY CURVES (PNGs)")
    print(f"{'#' * 100}\n")

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)

    COLORS = {
        'BTCUSDT': '#F7931A',
        'DOGEUSDT': '#C2A633',
        'SOLUSDT': '#9945FF',
        'ETHUSDT': '#627EEA',
        'XRPUSDT': '#00AAE4',
    }

    # ── Per-symbol equity curves (IS + OOS combined) ──
    for config_label, is_key, oos_key in [
        ("A1: 70% TP@12bps + 30% trail@3bps", 'trades_a1_is', 'trades_a1_oos'),
        ("C1: 50% TP@10bps, trail 5→3bps", 'trades_c1_is', 'trades_c1_oos'),
    ]:
        config_short = is_key.replace('trades_', '').replace('_is', '')

        for sym in SYMBOLS:
            trades_is = sym_data[sym][is_key]
            trades_oos = sym_data[sym][oos_key]

            if not trades_is and not trades_oos:
                continue

            fig, ax = plt.subplots(figsize=(14, 5))

            # IS equity
            if trades_is:
                times_is = [t['time'] for t in trades_is]
                pnls_is = [t['net_pnl'] for t in trades_is]
                cum_is = np.cumsum(pnls_is)
                ax.plot(times_is, cum_is, color=COLORS[sym], linewidth=1.5, label=f'IS (May-Aug 2025)')
                ax.fill_between(times_is, 0, cum_is, alpha=0.15, color=COLORS[sym])

            # OOS equity
            if trades_oos:
                # Offset OOS to start from IS end
                offset = cum_is[-1] if trades_is else 0
                times_oos = [t['time'] for t in trades_oos]
                pnls_oos = [t['net_pnl'] for t in trades_oos]
                cum_oos = np.cumsum(pnls_oos) + offset
                ax.plot(times_oos, cum_oos, color=COLORS[sym], linewidth=1.5,
                        linestyle='--', label=f'OOS (Feb 2026)')
                ax.fill_between(times_oos, offset, cum_oos, alpha=0.25, color=COLORS[sym])

                # Vertical separator
                if trades_is:
                    sep_x = times_is[-1] + (times_oos[0] - times_is[-1]) / 2
                    ax.axvline(x=sep_x, color='gray', linestyle=':', alpha=0.5)
                    ax.text(sep_x, ax.get_ylim()[1] * 0.95, ' OOS →', fontsize=9,
                            color='gray', va='top')

            ax.set_title(f'{SYMBOL_SHORT[sym]} — {config_label}', fontsize=13, fontweight='bold')
            ax.set_ylabel('Cumulative PnL (%)', fontsize=11)
            ax.set_xlabel('Date', fontsize=11)
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            fig.autofmt_xdate()
            fig.tight_layout()

            fname = out_dir / f'equity_{SYMBOL_SHORT[sym].lower()}_{config_short}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")

    # ── Combined equity curve (all symbols stacked) ──
    for config_label, is_key, oos_key in [
        ("A1: 70% TP@12bps + 30% trail@3bps", 'trades_a1_is', 'trades_a1_oos'),
        ("C1: 50% TP@10bps, trail 5→3bps", 'trades_c1_is', 'trades_c1_oos'),
    ]:
        config_short = is_key.replace('trades_', '').replace('_is', '')

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Top: combined equity
        ax = axes[0]
        all_is = []
        all_oos = []
        for sym in SYMBOLS:
            all_is.extend(sym_data[sym][is_key])
            all_oos.extend(sym_data[sym][oos_key])

        all_is.sort(key=lambda t: t['time'])
        all_oos.sort(key=lambda t: t['time'])

        if all_is:
            times_is = [t['time'] for t in all_is]
            cum_is = np.cumsum([t['net_pnl'] for t in all_is])
            ax.plot(times_is, cum_is, color='#2196F3', linewidth=2, label='IS (May-Aug 2025)')
            ax.fill_between(times_is, 0, cum_is, alpha=0.1, color='#2196F3')

        if all_oos:
            offset = cum_is[-1] if all_is else 0
            times_oos = [t['time'] for t in all_oos]
            cum_oos = np.cumsum([t['net_pnl'] for t in all_oos]) + offset
            ax.plot(times_oos, cum_oos, color='#4CAF50', linewidth=2, label='OOS (Feb 2026)')
            ax.fill_between(times_oos, offset, cum_oos, alpha=0.15, color='#4CAF50')

            if all_is:
                sep_x = times_is[-1] + (times_oos[0] - times_is[-1]) / 2
                ax.axvline(x=sep_x, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
                ax.text(sep_x, ax.get_ylim()[1] * 0.9, '  OOS →', fontsize=11,
                        color='red', fontweight='bold', va='top')

        n_total = len(all_is) + len(all_oos)
        total_pnl = cum_oos[-1] if all_oos else (cum_is[-1] if all_is else 0)
        ax.set_title(f'Combined 5-Symbol Equity — {config_label}\n'
                     f'{n_total} trades, +{total_pnl:.1f}% total',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative PnL (%)', fontsize=12)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        # Bottom: per-symbol contribution
        ax2 = axes[1]
        for sym in SYMBOLS:
            trades_is = sym_data[sym][is_key]
            trades_oos = sym_data[sym][oos_key]
            all_sym = sorted(trades_is + trades_oos, key=lambda t: t['time'])
            if all_sym:
                times = [t['time'] for t in all_sym]
                cum = np.cumsum([t['net_pnl'] for t in all_sym])
                ax2.plot(times, cum, color=COLORS[sym], linewidth=1.2,
                         label=SYMBOL_SHORT[sym], alpha=0.8)

        ax2.set_ylabel('PnL (%)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.legend(loc='upper left', fontsize=9, ncol=5)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        fig.autofmt_xdate()
        fig.tight_layout()

        fname = out_dir / f'equity_combined_{config_short}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── OOS-only zoomed equity ──
    for config_label, oos_key in [
        ("A1: 70% TP@12bps + 30% trail@3bps", 'trades_a1_oos'),
        ("C1: 50% TP@10bps, trail 5→3bps", 'trades_c1_oos'),
    ]:
        config_short = oos_key.replace('trades_', '').replace('_oos', '')

        fig, ax = plt.subplots(figsize=(14, 6))
        for sym in SYMBOLS:
            trades = sym_data[sym][oos_key]
            if not trades:
                continue
            times = [t['time'] for t in trades]
            cum = np.cumsum([t['net_pnl'] for t in trades])
            ax.plot(times, cum, color=COLORS[sym], linewidth=1.5,
                    label=f'{SYMBOL_SHORT[sym]} (+{cum[-1]:.1f}%)', alpha=0.9)

        # Combined
        all_oos = []
        for sym in SYMBOLS:
            all_oos.extend(sym_data[sym][oos_key])
        all_oos.sort(key=lambda t: t['time'])
        if all_oos:
            times = [t['time'] for t in all_oos]
            cum = np.cumsum([t['net_pnl'] for t in all_oos])
            ax.plot(times, cum, color='black', linewidth=2.5,
                    label=f'COMBINED (+{cum[-1]:.1f}%)', alpha=0.9)

        ax.set_title(f'OOS Feb 2026 — {config_label}\n'
                     f'{len(all_oos)} trades in 11 days',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative PnL (%)', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        fig.autofmt_xdate()
        fig.tight_layout()

        fname = out_dir / f'equity_oos_{config_short}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")

    elapsed = time.time() - t0_global
    print(f"\n{'=' * 100}")
    print(f"  COMPLETE — {elapsed:.0f}s")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
