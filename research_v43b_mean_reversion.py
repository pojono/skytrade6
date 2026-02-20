#!/usr/bin/env python3
"""
v43b: Mean-Reversion Strategy — Fixed TP/SL, No Trailing Stop

Core idea: Fade extreme short-term price moves on 1-min bars.
When price moves >N sigma of rolling volatility, enter a limit order
to fade the move. Exit via fixed TP (limit) or SL (market) or timeout.

Variants tested:
  1. Pure price MR (no confirmation) — baseline
  2. Price MR + OI confirmation (OI change in same direction = squeeze)
  3. Price MR + spread confirmation (spread widened = stress)
  4. Price MR + funding rate bias (fade moves aligned with funding pressure)

All entries via limit orders (maker 0.02%).
TP exit via limit order (maker 0.02%).
SL/timeout exit via market order (taker 0.055%).

Anti-overfitting: test on 7 days first, then expand. Fixed params, no optimization.
Anti-lookahead: all features use only past data.
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
# DATA LOADING
# ============================================================================

def load_ticker_bars(symbol, dates, bar_freq='1min'):
    """Load ticker parquet, build OHLC bars + features."""
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    all_bars = []
    t0 = time.time()

    for i, date_str in enumerate(dates, 1):
        path = ticker_dir / f'{date_str}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp_us'], unit='us')
        df = df.set_index('timestamp').sort_index()

        bars = df['last_price'].resample(bar_freq).ohlc().dropna()
        bars['oi'] = df['open_interest'].resample(bar_freq).last()
        bars['funding_rate'] = df['funding_rate'].resample(bar_freq).last()
        bars['bid1'] = df['bid1_price'].resample(bar_freq).mean()
        bars['ask1'] = df['ask1_price'].resample(bar_freq).mean()
        bars['spread_bps'] = (bars['ask1'] - bars['bid1']) / bars['close'] * 10000

        all_bars.append(bars)

        if i % 10 == 0 or i == len(dates):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(dates)}] loaded, {elapsed:.1f}s", flush=True)

    if not all_bars:
        return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f"  Total: {len(result):,} bars, RAM={get_ram_mb():.0f}MB")
    return result


# ============================================================================
# FEATURE ENGINEERING — all lookback only (no future data)
# ============================================================================

def add_features(bars, vol_window=60, oi_window=5, spread_window=60):
    """Add rolling features for signal generation. All backward-looking."""
    close = bars['close'].values.astype(np.float64)
    n = len(close)

    # 1-bar returns in bps
    ret_bps = np.zeros(n)
    ret_bps[1:] = (close[1:] - close[:-1]) / close[:-1] * 10000

    # Rolling volatility (std of returns over vol_window bars)
    roll_std = pd.Series(ret_bps).rolling(vol_window, min_periods=vol_window//2).std().values

    # Z-score of current return
    bars['ret_bps'] = ret_bps
    bars['roll_std'] = roll_std
    bars['ret_z'] = np.where(roll_std > 0.1, ret_bps / roll_std, 0)

    # Multi-bar returns (5-bar = 5 min cumulative move)
    for w in [3, 5, 10]:
        cum_ret = np.zeros(n)
        for i in range(w, n):
            cum_ret[i] = (close[i] - close[i-w]) / close[i-w] * 10000
        bars[f'ret_{w}bar_bps'] = cum_ret
        bars[f'ret_{w}bar_z'] = np.where(roll_std > 0.1, cum_ret / (roll_std * np.sqrt(w)), 0)

    # OI change (5-bar)
    oi = bars['oi'].values
    oi_change = np.zeros(n)
    for i in range(oi_window, n):
        if oi[i-oi_window] > 0 and not np.isnan(oi[i]) and not np.isnan(oi[i-oi_window]):
            oi_change[i] = (oi[i] - oi[i-oi_window]) / oi[i-oi_window] * 100
    bars['oi_change_pct'] = oi_change

    # Spread z-score
    spread = bars['spread_bps'].values
    spread_roll_mean = pd.Series(spread).rolling(spread_window, min_periods=spread_window//2).mean().values
    spread_roll_std = pd.Series(spread).rolling(spread_window, min_periods=spread_window//2).std().values
    bars['spread_z'] = np.where(spread_roll_std > 0.001,
                                 (spread - spread_roll_mean) / spread_roll_std, 0)

    # Funding rate sign (positive = longs pay shorts = bearish pressure)
    bars['funding_sign'] = np.sign(bars['funding_rate'].values)

    return bars


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(bars, z_threshold=2.0, multi_bar=5, cooldown_min=5,
                     require_oi=False, oi_threshold=-0.02,
                     require_spread=False, spread_z_min=1.0,
                     require_funding_align=False,
                     entry_offset_bps=3):
    """
    Generate mean-reversion signals.

    Entry logic:
    - When N-bar return z-score exceeds threshold, fade the move
    - Optional: require OI drop (squeeze confirmation)
    - Optional: require spread widening (stress confirmation)
    - Optional: require funding rate alignment (fade moves in funding direction)

    Entry: limit order at offset from current price (in fade direction)
    """
    signals = []
    bar_times = bars.index
    close = bars['close'].values
    z_col = f'ret_{multi_bar}bar_z'

    if z_col not in bars.columns:
        print(f"  ERROR: {z_col} not in bars")
        return signals

    z_vals = bars[z_col].values
    oi_change = bars['oi_change_pct'].values
    spread_z = bars['spread_z'].values
    funding_sign = bars['funding_sign'].values

    last_signal_idx = -cooldown_min - 1
    n = len(bars)

    for i in range(max(60, multi_bar), n - 60):
        # Check cooldown
        if i - last_signal_idx < cooldown_min:
            continue

        z = z_vals[i]
        if abs(z) < z_threshold:
            continue

        # Direction: fade the move
        if z > z_threshold:
            direction = 'short'  # price went up too much, fade
        elif z < -z_threshold:
            direction = 'long'   # price went down too much, fade
        else:
            continue

        # Optional confirmations
        if require_oi:
            # For mean reversion: OI drop = positions closing = squeeze
            if oi_change[i] > oi_threshold:
                continue

        if require_spread:
            if spread_z[i] < spread_z_min:
                continue

        if require_funding_align:
            # Fade moves aligned with funding pressure
            # If funding positive (longs pay) and price went up → short (aligned)
            # If funding negative (shorts pay) and price went down → long (aligned)
            if direction == 'short' and funding_sign[i] <= 0:
                continue
            if direction == 'long' and funding_sign[i] >= 0:
                continue

        # Entry price: limit order with offset
        price = close[i]
        if direction == 'long':
            entry_price = price * (1 - entry_offset_bps / 10000)
        else:
            entry_price = price * (1 + entry_offset_bps / 10000)

        signals.append({
            'time': bar_times[i],
            'direction': direction,
            'entry_price': entry_price,
            'z': z,
            'bar_idx': i,
        })
        last_signal_idx = i

    return signals


# ============================================================================
# SIMULATION — Fixed TP/SL, no trailing stop
# ============================================================================

def simulate_trades(signals, bars, tp_bps, sl_bps, timeout_min=60,
                    entry_fill_window=5):
    """
    Simulate with fixed TP (limit) + SL (market) + timeout.
    Entry via limit order with fill window.
    """
    bar_high = bars['high'].values
    bar_low = bars['low'].values
    bar_close = bars['close'].values
    bar_times = bars.index
    n = len(bars)

    trades = []
    last_exit_idx = -300

    for sig in signals:
        idx = sig['bar_idx']
        direction = sig['direction']
        entry_price = sig['entry_price']

        if idx <= last_exit_idx + 5:  # cooldown after last exit
            continue
        if idx + timeout_min + entry_fill_window >= n:
            continue

        # Try to fill entry limit order
        fill_bar = None
        for j in range(idx, min(idx + entry_fill_window, n)):
            if direction == 'long' and bar_low[j] <= entry_price:
                fill_bar = j
                break
            elif direction == 'short' and bar_high[j] >= entry_price:
                fill_bar = j
                break

        if fill_bar is None:
            continue

        # Compute TP/SL prices
        if direction == 'long':
            tp_price = entry_price * (1 + tp_bps / 10000)
            sl_price = entry_price * (1 - sl_bps / 10000)
        else:
            tp_price = entry_price * (1 - tp_bps / 10000)
            sl_price = entry_price * (1 + sl_bps / 10000)

        # Simulate exit (start from bar AFTER fill)
        exit_end = min(fill_bar + timeout_min, n - 1)
        exit_price = None
        exit_reason = None
        exit_bar = exit_end

        for k in range(fill_bar + 1, exit_end + 1):
            if direction == 'long':
                # SL first (conservative)
                if bar_low[k] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break
                if bar_high[k] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar = k
                    break
            else:
                if bar_high[k] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break
                if bar_low[k] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar = k
                    break

        if exit_price is None:
            exit_price = bar_close[exit_end]
            exit_reason = 'timeout'

        # PnL
        if direction == 'long':
            raw_pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            raw_pnl_pct = (entry_price - exit_price) / entry_price * 100

        entry_fee = MAKER_FEE_PCT
        exit_fee = MAKER_FEE_PCT if exit_reason == 'take_profit' else TAKER_FEE_PCT
        net_pnl_pct = raw_pnl_pct - entry_fee - exit_fee

        trades.append({
            'time': sig['time'],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'net_pnl_pct': net_pnl_pct,
            'net_pnl_bps': net_pnl_pct * 100,
            'hold_min': exit_bar - fill_bar,
            'z': sig['z'],
        })
        last_exit_idx = exit_bar

    return trades


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES")
        return None

    net = np.array([t['net_pnl_bps'] for t in trades])
    n = len(net)
    wins = (net > 0).sum()
    wr = wins / n * 100
    total_pct = sum(t['net_pnl_pct'] for t in trades)
    avg = net.mean()
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 24) if std > 0 else 0  # hourly trades

    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    maxdd = (peak - cum).max()

    print(f"  {label}")
    print(f"    n={n:4d}  WR={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"total={total_pct:+7.2f}%  Sharpe={sharpe:+6.1f}  "
          f"maxDD={maxdd:6.1f}bps  exits={reasons}")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total_pct,
            'sharpe': sharpe, 'maxdd': maxdd, 'reasons': reasons}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("v43b: Mean-Reversion Strategy — Fixed TP/SL, No Trailing Stop")
    print("=" * 70)

    symbol = 'ETHUSDT'
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    available = sorted([f.stem for f in ticker_dir.glob('*.parquet')])
    print(f"Available: {len(available)} days ({available[0]} to {available[-1]})")

    # 7-day test period
    test_dates = [d for d in available if '2025-08-04' <= d <= '2025-08-10']
    if len(test_dates) < 5:
        test_dates = available[-7:]
    print(f"Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Load and build features
    print(f"\nLoading {symbol}...")
    bars = load_ticker_bars(symbol, test_dates, bar_freq='1min')
    if bars.empty:
        print("ERROR: No data!")
        return

    print("Adding features...")
    bars = add_features(bars)
    print(f"Bars: {len(bars):,}, RAM={get_ram_mb():.0f}MB\n")

    # ================================================================
    # SWEEP: z-threshold × multi-bar × TP/SL
    # ================================================================
    results = []

    tp_sl_configs = [
        (8, 16),    # tight 1:2
        (10, 20),   # standard 1:2
        (10, 30),   # wide SL 1:3
        (15, 30),   # medium 1:2
        (15, 45),   # medium 1:3
        (20, 40),   # wide 1:2
        (20, 60),   # wide 1:3
    ]

    # ---- VARIANT 1: Pure price MR (baseline) ----
    print("=" * 70)
    print("VARIANT 1: Pure Price Mean-Reversion (no confirmation)")
    print("=" * 70)

    for z_thresh in [1.5, 2.0, 2.5, 3.0]:
        for multi_bar in [1, 3, 5, 10]:
            signals = generate_signals(bars, z_threshold=z_thresh,
                                        multi_bar=multi_bar, cooldown_min=5,
                                        entry_offset_bps=3)
            if len(signals) < 5:
                continue

            print(f"\n  z>{z_thresh} {multi_bar}-bar ({len(signals)} signals):")
            for tp, sl in tp_sl_configs:
                trades = simulate_trades(signals, bars, tp_bps=tp, sl_bps=sl)
                r = analyze(trades, f"TP={tp} SL={sl}")
                if r:
                    r.update({'variant': 'pure', 'z': z_thresh, 'bars': multi_bar,
                              'tp': tp, 'sl': sl})
                    results.append(r)

    # ---- VARIANT 2: Price MR + OI confirmation ----
    print(f"\n{'='*70}")
    print("VARIANT 2: Price MR + OI Drop Confirmation")
    print("=" * 70)

    for z_thresh in [1.5, 2.0, 2.5]:
        for multi_bar in [3, 5, 10]:
            signals = generate_signals(bars, z_threshold=z_thresh,
                                        multi_bar=multi_bar, cooldown_min=5,
                                        require_oi=True, oi_threshold=-0.02,
                                        entry_offset_bps=3)
            if len(signals) < 5:
                continue

            print(f"\n  z>{z_thresh} {multi_bar}-bar + OI drop ({len(signals)} signals):")
            for tp, sl in [(10, 20), (15, 30), (20, 40)]:
                trades = simulate_trades(signals, bars, tp_bps=tp, sl_bps=sl)
                r = analyze(trades, f"TP={tp} SL={sl}")
                if r:
                    r.update({'variant': 'oi', 'z': z_thresh, 'bars': multi_bar,
                              'tp': tp, 'sl': sl})
                    results.append(r)

    # ---- VARIANT 3: Price MR + Spread confirmation ----
    print(f"\n{'='*70}")
    print("VARIANT 3: Price MR + Spread Widening Confirmation")
    print("=" * 70)

    for z_thresh in [1.5, 2.0, 2.5]:
        for multi_bar in [3, 5, 10]:
            signals = generate_signals(bars, z_threshold=z_thresh,
                                        multi_bar=multi_bar, cooldown_min=5,
                                        require_spread=True, spread_z_min=1.0,
                                        entry_offset_bps=3)
            if len(signals) < 5:
                continue

            print(f"\n  z>{z_thresh} {multi_bar}-bar + spread ({len(signals)} signals):")
            for tp, sl in [(10, 20), (15, 30), (20, 40)]:
                trades = simulate_trades(signals, bars, tp_bps=tp, sl_bps=sl)
                r = analyze(trades, f"TP={tp} SL={sl}")
                if r:
                    r.update({'variant': 'spread', 'z': z_thresh, 'bars': multi_bar,
                              'tp': tp, 'sl': sl})
                    results.append(r)

    # ---- VARIANT 4: Price MR + Funding alignment ----
    print(f"\n{'='*70}")
    print("VARIANT 4: Price MR + Funding Rate Alignment")
    print("=" * 70)

    for z_thresh in [1.5, 2.0, 2.5]:
        for multi_bar in [3, 5, 10]:
            signals = generate_signals(bars, z_threshold=z_thresh,
                                        multi_bar=multi_bar, cooldown_min=5,
                                        require_funding_align=True,
                                        entry_offset_bps=3)
            if len(signals) < 5:
                continue

            print(f"\n  z>{z_thresh} {multi_bar}-bar + funding ({len(signals)} signals):")
            for tp, sl in [(10, 20), (15, 30), (20, 40)]:
                trades = simulate_trades(signals, bars, tp_bps=tp, sl_bps=sl)
                r = analyze(trades, f"TP={tp} SL={sl}")
                if r:
                    r.update({'variant': 'funding', 'z': z_thresh, 'bars': multi_bar,
                              'tp': tp, 'sl': sl})
                    results.append(r)

    # ================================================================
    # SUMMARY: Top 10 configs
    # ================================================================
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGS BY AVG NET BPS (min 10 trades)")
    print("=" * 70)

    valid = [r for r in results if r['n'] >= 10]
    valid.sort(key=lambda x: x['avg'], reverse=True)

    for i, r in enumerate(valid[:10], 1):
        print(f"  #{i}: {r['variant']:8s} z>{r['z']} {r['bars']:2d}-bar "
              f"TP={r['tp']:2d} SL={r['sl']:2d} | "
              f"n={r['n']:4d} WR={r['wr']:5.1f}% avg={r['avg']:+6.1f}bps "
              f"total={r['total']:+7.2f}% Sharpe={r['sharpe']:+6.1f}")

    print(f"\nBOTTOM 5 (worst):")
    for i, r in enumerate(valid[-5:], 1):
        print(f"  #{i}: {r['variant']:8s} z>{r['z']} {r['bars']:2d}-bar "
              f"TP={r['tp']:2d} SL={r['sl']:2d} | "
              f"n={r['n']:4d} WR={r['wr']:5.1f}% avg={r['avg']:+6.1f}bps "
              f"total={r['total']:+7.2f}%")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, {len(results)} configs tested, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
