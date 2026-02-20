#!/usr/bin/env python3
"""
v43f: Volume Imbalance Momentum Strategy

Uses tick-level trade data (1.5M trades/day) to compute buy/sell volume
imbalance. When imbalance is extreme, enter in the direction of the flow.

Hypothesis: Large sustained buying/selling pressure predicts price continuation.
This is a MOMENTUM signal (follow the flow), not mean-reversion.

Data: Bybit futures tick trades (3 years: Jan 2023 – Feb 2026)
Timeframe: 1h signal bars, 4h holding period
Entry: limit order (maker 0.02%)
Exit: TP limit (maker 0.02%) or timeout market (taker 0.055%)
No trailing stop.

Also tests:
  - Trade intensity (volume spike) as filter
  - Large trade detection (whale flow)
  - Asymmetric TP/SL ratios

Quick test: 30 days on SOL, then expand.
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


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


# ============================================================================
# DATA: Build 1h feature bars from tick trades
# ============================================================================

def build_hourly_bars(symbol, dates):
    """Load tick trades day by day, aggregate to 1h bars with volume features."""
    trade_dir = PARQUET_DIR / symbol / 'trades' / 'bybit_futures'
    all_bars = []
    t0 = time.time()

    for i, d in enumerate(dates, 1):
        path = trade_dir / f'{d}.parquet'
        if not path.exists():
            continue

        df = pd.read_parquet(path, columns=['timestamp_us', 'price', 'quantity', 'side'])
        df['timestamp'] = pd.to_datetime(df['timestamp_us'], unit='us')
        df = df.set_index('timestamp').sort_index()

        # side: 1 = buy (taker buy), -1 = sell (taker sell)
        df['buy_vol'] = df['quantity'].where(df['side'] == 1, 0)
        df['sell_vol'] = df['quantity'].where(df['side'] == -1, 0)
        df['dollar_vol'] = df['quantity'] * df['price']
        df['buy_dollar'] = df['dollar_vol'].where(df['side'] == 1, 0)
        df['sell_dollar'] = df['dollar_vol'].where(df['side'] == -1, 0)

        # Large trades (top 1% by size within the day)
        q99 = df['quantity'].quantile(0.99)
        df['large_buy'] = df['buy_vol'].where(df['quantity'] >= q99, 0)
        df['large_sell'] = df['sell_vol'].where(df['quantity'] >= q99, 0)

        # Aggregate to 1h
        h = pd.DataFrame()
        h['open'] = df['price'].resample('1h').first()
        h['high'] = df['price'].resample('1h').max()
        h['low'] = df['price'].resample('1h').min()
        h['close'] = df['price'].resample('1h').last()
        h['n_trades'] = df['price'].resample('1h').count()
        h['buy_vol'] = df['buy_vol'].resample('1h').sum()
        h['sell_vol'] = df['sell_vol'].resample('1h').sum()
        h['total_vol'] = h['buy_vol'] + h['sell_vol']
        h['buy_dollar'] = df['buy_dollar'].resample('1h').sum()
        h['sell_dollar'] = df['sell_dollar'].resample('1h').sum()
        h['large_buy'] = df['large_buy'].resample('1h').sum()
        h['large_sell'] = df['large_sell'].resample('1h').sum()

        h = h.dropna(subset=['close'])
        all_bars.append(h)

        if i % 10 == 0 or i == len(dates):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(dates) - i)
            print(f"  [{i}/{len(dates)}] {elapsed:.0f}s ETA={eta:.0f}s RAM={get_ram_mb():.0f}MB",
                  flush=True)

        del df
        if i % 30 == 0:
            gc.collect()

    if not all_bars:
        return pd.DataFrame()

    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f"  Total: {len(result):,} 1h bars")
    return result


def add_features(bars, window=24):
    """Add volume imbalance and flow features. All backward-looking."""
    # Volume imbalance ratio: (buy - sell) / (buy + sell)
    bars['vol_imbalance'] = ((bars['buy_vol'] - bars['sell_vol']) /
                              bars['total_vol'].clip(lower=1))

    # Dollar imbalance
    total_dollar = bars['buy_dollar'] + bars['sell_dollar']
    bars['dollar_imbalance'] = ((bars['buy_dollar'] - bars['sell_dollar']) /
                                 total_dollar.clip(lower=1))

    # Large trade imbalance
    large_total = bars['large_buy'] + bars['large_sell']
    bars['large_imbalance'] = ((bars['large_buy'] - bars['large_sell']) /
                                large_total.clip(lower=1))

    # Rolling z-scores
    for col in ['vol_imbalance', 'dollar_imbalance', 'large_imbalance']:
        roll_mean = bars[col].rolling(window, min_periods=window//2).mean()
        roll_std = bars[col].rolling(window, min_periods=window//2).std()
        bars[f'{col}_z'] = (bars[col] - roll_mean) / roll_std.clip(lower=1e-8)

    # Cumulative imbalance (4h = 4 bars)
    bars['cum_imbalance_4h'] = bars['vol_imbalance'].rolling(4, min_periods=2).sum()
    bars['cum_dollar_imb_4h'] = bars['dollar_imbalance'].rolling(4, min_periods=2).sum()

    # Trade intensity (volume relative to rolling average)
    bars['vol_ratio'] = bars['total_vol'] / bars['total_vol'].rolling(window, min_periods=window//2).mean().clip(lower=1)

    # Returns
    bars['ret_1h'] = bars['close'].pct_change() * 10000  # bps
    bars['ret_4h'] = bars['close'].pct_change(4) * 10000

    # Realized vol
    bars['rvol_24h'] = bars['ret_1h'].rolling(window, min_periods=window//2).std()

    return bars


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(bars, signal_col='vol_imbalance_z', z_threshold=1.5,
                     direction_mode='momentum', vol_ratio_min=None):
    """
    Generate signals from volume imbalance z-score.
    Momentum: follow the imbalance (buy imbalance → long)
    Contrarian: fade the imbalance
    """
    signals = []
    z = bars[signal_col].values
    close = bars['close'].values
    vol_ratio = bars['vol_ratio'].values if 'vol_ratio' in bars.columns else None

    for i in range(24, len(bars) - 4):
        if np.isnan(z[i]) or abs(z[i]) < z_threshold:
            continue

        # Optional volume filter
        if vol_ratio_min is not None and vol_ratio is not None:
            if np.isnan(vol_ratio[i]) or vol_ratio[i] < vol_ratio_min:
                continue

        if direction_mode == 'momentum':
            direction = 'long' if z[i] > 0 else 'short'
        else:
            direction = 'short' if z[i] > 0 else 'long'

        signals.append({
            'bar_idx': i,
            'time': bars.index[i],
            'direction': direction,
            'z': z[i],
            'close': close[i],
        })

    return signals


# ============================================================================
# SIMULATION on 1h bars (simpler than 1-min, appropriate for 4h holding)
# ============================================================================

def simulate_trades(signals, bars, tp_bps=None, sl_bps=None,
                    hold_bars=4, entry_offset_bps=5, cooldown_bars=2):
    """
    Simulate on 1h bars.
    Entry: limit order at offset from signal bar close.
    Exit: TP (limit), SL (market), or timeout (market).
    """
    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values
    n = len(bars)

    trades = []
    last_exit_idx = -cooldown_bars - 1

    for sig in signals:
        idx = sig['bar_idx']
        if idx <= last_exit_idx + cooldown_bars:
            continue
        if idx + hold_bars + 2 >= n:
            continue

        direction = sig['direction']
        price = sig['close']

        # Entry limit order
        if direction == 'long':
            entry_price = price * (1 - entry_offset_bps / 10000)
        else:
            entry_price = price * (1 + entry_offset_bps / 10000)

        # Try to fill in next 2 bars
        fill_bar = None
        for j in range(idx + 1, min(idx + 3, n)):
            if direction == 'long' and low[j] <= entry_price:
                fill_bar = j
                break
            elif direction == 'short' and high[j] >= entry_price:
                fill_bar = j
                break

        if fill_bar is None:
            continue

        # TP/SL
        if tp_bps:
            if direction == 'long':
                tp_price = entry_price * (1 + tp_bps / 10000)
            else:
                tp_price = entry_price * (1 - tp_bps / 10000)
        else:
            tp_price = None

        if sl_bps:
            if direction == 'long':
                sl_price = entry_price * (1 - sl_bps / 10000)
            else:
                sl_price = entry_price * (1 + sl_bps / 10000)
        else:
            sl_price = None

        # Simulate exit
        exit_end = min(fill_bar + hold_bars, n - 1)
        exit_price = None
        exit_reason = None
        exit_bar = exit_end

        for k in range(fill_bar + 1, exit_end + 1):
            if sl_price:
                if direction == 'long' and low[k] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break
                elif direction == 'short' and high[k] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break
            if tp_price:
                if direction == 'long' and high[k] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar = k
                    break
                elif direction == 'short' and low[k] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar = k
                    break

        if exit_price is None:
            exit_price = close[exit_end]
            exit_reason = 'timeout'

        # PnL
        if direction == 'long':
            raw_pnl = (exit_price - entry_price) / entry_price * 10000
        else:
            raw_pnl = (entry_price - exit_price) / entry_price * 10000

        entry_fee = MAKER_FEE_PCT * 100
        exit_fee = MAKER_FEE_PCT * 100 if exit_reason == 'take_profit' else TAKER_FEE_PCT * 100
        net_bps = raw_pnl - entry_fee - exit_fee

        trades.append({
            'time': sig['time'],
            'direction': direction,
            'exit_reason': exit_reason,
            'raw_bps': raw_pnl,
            'net_bps': net_bps,
            'hold_bars': exit_bar - fill_bar,
            'z': sig['z'],
        })
        last_exit_idx = exit_bar

    return trades


def analyze(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES")
        return None

    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    total = net.sum() / 100
    avg = net.mean()
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0

    reasons = {}
    for t in trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1

    print(f"  {label}")
    print(f"    n={n:4d} WR={wr:5.1f}% avg={avg:+7.1f}bps total={total:+7.2f}% "
          f"Sharpe={sharpe:+6.1f} exits={reasons}")

    # Direction breakdown
    for d in ['long', 'short']:
        dt = [t for t in trades if t['direction'] == d]
        if dt:
            dn = np.array([t['net_bps'] for t in dt])
            print(f"    {d.upper():5s}: n={len(dt)} WR={(dn>0).sum()/len(dn)*100:.1f}% "
                  f"avg={dn.mean():+.1f}bps")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("v43f: Volume Imbalance Momentum — Tick-Level Features")
    print("=" * 70)

    symbol = 'SOLUSDT'
    trade_dir = PARQUET_DIR / symbol / 'trades' / 'bybit_futures'
    available = sorted([f.stem for f in trade_dir.glob('*.parquet')])
    print(f"Available: {len(available)} days ({available[0]} to {available[-1]})")

    # Use 2 separate periods for IS/OOS
    # IS: 30 days from mid-2025
    # OOS: 30 days from late 2025 (completely disjoint)
    is_dates = [d for d in available if '2025-06-01' <= d <= '2025-06-30']
    oos_dates = [d for d in available if '2025-11-01' <= d <= '2025-11-30']

    # Fallback if dates not available
    if len(is_dates) < 20:
        is_dates = available[400:430]  # ~mid-2024
    if len(oos_dates) < 20:
        oos_dates = available[700:730]  # ~late-2024

    print(f"IS:  {is_dates[0]} to {is_dates[-1]} ({len(is_dates)} days)")
    print(f"OOS: {oos_dates[0]} to {oos_dates[-1]} ({len(oos_dates)} days)")

    # Build 1h bars
    print(f"\nBuilding IS 1h bars from tick data...")
    bars_is = build_hourly_bars(symbol, is_dates)
    if bars_is.empty:
        print("ERROR: No IS data!")
        return

    print(f"\nBuilding OOS 1h bars from tick data...")
    bars_oos = build_hourly_bars(symbol, oos_dates)
    if bars_oos.empty:
        print("ERROR: No OOS data!")
        return

    # Add features
    bars_is = add_features(bars_is)
    bars_oos = add_features(bars_oos)
    print(f"\nIS: {len(bars_is):,} bars, OOS: {len(bars_oos):,} bars")
    print(f"RAM: {get_ram_mb():.0f}MB")

    # ================================================================
    # TEST MATRIX
    # ================================================================
    all_results = []

    signal_configs = [
        ('vol_imbalance_z', 'momentum'),
        ('vol_imbalance_z', 'contrarian'),
        ('dollar_imbalance_z', 'momentum'),
        ('large_imbalance_z', 'momentum'),
        ('large_imbalance_z', 'contrarian'),
    ]

    tp_sl_configs = [
        (None, None),       # pure timeout
        (50, None),         # TP only
        (100, None),        # wide TP
        (50, 100),          # TP + SL
        (100, 200),         # wide TP + wide SL
        (50, 150),          # asymmetric 1:3
    ]

    for signal_col, dir_mode in signal_configs:
        print(f"\n{'='*70}")
        print(f"  Signal: {signal_col} ({dir_mode})")
        print(f"{'='*70}")

        for z_thresh in [1.0, 1.5, 2.0, 2.5]:
            for vol_min in [None, 1.5]:
                # Generate signals
                sig_is = generate_signals(bars_is, signal_col, z_thresh,
                                           dir_mode, vol_ratio_min=vol_min)
                sig_oos = generate_signals(bars_oos, signal_col, z_thresh,
                                            dir_mode, vol_ratio_min=vol_min)

                if len(sig_is) < 5 and len(sig_oos) < 5:
                    continue

                vol_label = f"+vol>{vol_min}" if vol_min else ""
                print(f"\n  z>{z_thresh}{vol_label} (IS={len(sig_is)} OOS={len(sig_oos)} signals)")

                for tp, sl in tp_sl_configs:
                    tp_label = f"TP={tp}" if tp else "noTP"
                    sl_label = f"SL={sl}" if sl else "noSL"
                    label = f"{tp_label} {sl_label}"

                    trades_is = simulate_trades(sig_is, bars_is, tp_bps=tp, sl_bps=sl)
                    trades_oos = simulate_trades(sig_oos, bars_oos, tp_bps=tp, sl_bps=sl)

                    r_is = analyze(trades_is, f"IS  {label}")
                    r_oos = analyze(trades_oos, f"OOS {label}")

                    if r_is and r_oos:
                        all_results.append({
                            'signal': signal_col, 'mode': dir_mode,
                            'z': z_thresh, 'vol_min': vol_min,
                            'tp': tp, 'sl': sl,
                            'is_avg': r_is['avg'], 'is_total': r_is['total'],
                            'is_n': r_is['n'], 'is_sharpe': r_is['sharpe'],
                            'oos_avg': r_oos['avg'], 'oos_total': r_oos['total'],
                            'oos_n': r_oos['n'], 'oos_sharpe': r_oos['sharpe'],
                        })

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("OOS-POSITIVE CONFIGS (sorted by OOS avg)")
    print(f"{'='*70}")

    oos_pos = [r for r in all_results if r['oos_avg'] > 0 and r['oos_n'] >= 5]
    oos_pos.sort(key=lambda x: x['oos_avg'], reverse=True)

    if oos_pos:
        for r in oos_pos[:15]:
            tp_l = f"TP={r['tp']}" if r['tp'] else "noTP"
            sl_l = f"SL={r['sl']}" if r['sl'] else "noSL"
            vm = f"+vol>{r['vol_min']}" if r['vol_min'] else ""
            print(f"  {r['signal']:25s} {r['mode']:11s} z>{r['z']}{vm:8s} {tp_l:7s} {sl_l:7s} | "
                  f"IS: n={r['is_n']:3d} avg={r['is_avg']:+6.1f} Sh={r['is_sharpe']:+5.1f} | "
                  f"OOS: n={r['oos_n']:3d} avg={r['oos_avg']:+6.1f} Sh={r['oos_sharpe']:+5.1f}")
    else:
        print("  NONE — no OOS-positive configs found")

    print(f"\nALL OOS-NEGATIVE configs: {len(all_results) - len(oos_pos)}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
