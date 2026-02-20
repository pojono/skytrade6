#!/usr/bin/env python3
"""
v43c: Funding Rate Directional Strategy — 4h Holding Period

Proven edge: Funding rate IC=-0.12 on SOL at 4h (confirmed OOS in v24b).
When funding is positive (longs pay shorts) → price tends to fall → go SHORT.
When funding is negative (shorts pay longs) → price tends to rise → go LONG.

Strategy:
  - Every 4h bar, check funding rate z-score (24h rolling)
  - If |z| > threshold → enter in contrarian direction
  - Entry: limit order at offset from current price
  - Exit: fixed TP (limit) or timeout at 4h (market)
  - No SL (4h holding, let it play out — SL on 4h is too tight)
  - Alternative: with SL for risk management

Also tests:
  - Mark-index spread signal (IC=-0.06 confirmed OOS)
  - OI change signal (contrarian at extremes)
  - Combined signals

Fees: maker 0.02%, taker 0.055%
All entries via limit orders.
TP via limit order. Timeout via market order.
"""

import sys, time
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
# DATA LOADING — build 4h bars from 5-sec ticker
# ============================================================================

def load_multi_tf_bars(symbol, dates):
    """Load ticker data, build 1h and 4h bars with funding/OI features."""
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    all_dfs = []
    t0 = time.time()

    for i, date_str in enumerate(dates, 1):
        path = ticker_dir / f'{date_str}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp_us'], unit='us')
        df = df.set_index('timestamp').sort_index()
        all_dfs.append(df)

        if i % 20 == 0 or i == len(dates):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(dates)}] loaded, {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB",
                  flush=True)

    if not all_dfs:
        return {}, pd.DataFrame()

    raw = pd.concat(all_dfs).sort_index()
    raw = raw[~raw.index.duplicated(keep='first')]
    print(f"  Raw ticker: {len(raw):,} records")

    # Build 1-min bars for simulation
    bars_1m = raw['last_price'].resample('1min').ohlc().dropna()
    bars_1m['volume'] = raw['turnover_24h'].resample('1min').last()
    print(f"  1-min bars: {len(bars_1m):,}")

    # Build feature bars at different timeframes
    result = {}
    for tf, tf_label in [('1h', '1h'), ('4h', '4h')]:
        bars = raw['last_price'].resample(tf).ohlc().dropna()
        bars['funding_rate'] = raw['funding_rate'].resample(tf).mean()
        bars['oi'] = raw['open_interest'].resample(tf).last()
        bars['oi_start'] = raw['open_interest'].resample(tf).first()
        bars['mark_price'] = raw['mark_price'].resample(tf).mean()
        bars['index_price'] = raw['index_price'].resample(tf).mean()
        bars['bid1'] = raw['bid1_price'].resample(tf).mean()
        bars['ask1'] = raw['ask1_price'].resample(tf).mean()
        bars['spread_bps'] = (bars['ask1'] - bars['bid1']) / bars['close'] * 10000

        # Mark-index spread (basis)
        bars['mark_index_spread'] = (bars['mark_price'] - bars['index_price']) / bars['index_price'] * 10000

        result[tf_label] = bars
        print(f"  {tf_label} bars: {len(bars):,}")

    del raw
    return result, bars_1m


def add_features(bars, window=24):
    """Add rolling features. Window is in bars (e.g., 24 for 24h on 1h bars, 6 for 24h on 4h bars)."""
    # Funding rate z-score
    fr = bars['funding_rate']
    bars['fr_mean'] = fr.rolling(window, min_periods=window//2).mean()
    bars['fr_std'] = fr.rolling(window, min_periods=window//2).std()
    bars['fr_z'] = (fr - bars['fr_mean']) / bars['fr_std'].clip(lower=1e-8)

    # Cumulative funding (8h = sum of recent funding)
    bars['fr_cum_8h'] = fr.rolling(min(8, window), min_periods=1).sum()

    # OI change
    bars['oi_change_pct'] = bars['oi'].pct_change() * 100
    bars['oi_change_roll'] = bars['oi_change_pct'].rolling(window, min_periods=window//2).mean()
    oi_std = bars['oi_change_pct'].rolling(window, min_periods=window//2).std()
    bars['oi_z'] = (bars['oi_change_pct'] - bars['oi_change_roll']) / oi_std.clip(lower=1e-8)

    # Mark-index spread z-score
    mis = bars['mark_index_spread']
    bars['mis_mean'] = mis.rolling(window, min_periods=window//2).mean()
    bars['mis_std'] = mis.rolling(window, min_periods=window//2).std()
    bars['mis_z'] = (mis - bars['mis_mean']) / bars['mis_std'].clip(lower=1e-8)

    # Spread z-score
    sp = bars['spread_bps']
    bars['spread_mean'] = sp.rolling(window, min_periods=window//2).mean()
    bars['spread_std'] = sp.rolling(window, min_periods=window//2).std()
    bars['spread_z'] = (sp - bars['spread_mean']) / bars['spread_std'].clip(lower=1e-8)

    # Returns
    bars['ret_pct'] = bars['close'].pct_change() * 100
    bars['ret_bps'] = bars['ret_pct'] * 100

    # Realized volatility
    bars['rvol'] = bars['ret_pct'].rolling(window, min_periods=window//2).std()

    return bars


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_funding_signals(bars, fr_z_threshold=1.0, direction_mode='contrarian'):
    """
    Generate signals based on funding rate z-score.
    Contrarian: high funding → short (longs are crowded, will pay)
    Momentum: high funding → long (trend continues)
    """
    signals = []
    for i in range(1, len(bars)):
        z = bars['fr_z'].iloc[i]
        if np.isnan(z):
            continue

        if abs(z) < fr_z_threshold:
            continue

        if direction_mode == 'contrarian':
            # High funding (z>0) → short (fade the crowd)
            direction = 'short' if z > 0 else 'long'
        else:
            # Momentum: follow the crowd
            direction = 'long' if z > 0 else 'short'

        signals.append({
            'bar_idx': i,
            'time': bars.index[i],
            'direction': direction,
            'z': z,
            'signal_type': 'funding',
        })
    return signals


def generate_mis_signals(bars, mis_z_threshold=1.0):
    """Mark-index spread contrarian: high spread → short (premium too high)."""
    signals = []
    for i in range(1, len(bars)):
        z = bars['mis_z'].iloc[i]
        if np.isnan(z):
            continue
        if abs(z) < mis_z_threshold:
            continue
        # Contrarian: high mark-index → short
        direction = 'short' if z > 0 else 'long'
        signals.append({
            'bar_idx': i,
            'time': bars.index[i],
            'direction': direction,
            'z': z,
            'signal_type': 'mis',
        })
    return signals


def generate_combined_signals(bars, fr_z_thresh=0.5, mis_z_thresh=0.5):
    """Combined: both funding AND mark-index agree on direction."""
    signals = []
    for i in range(1, len(bars)):
        fr_z = bars['fr_z'].iloc[i]
        mis_z = bars['mis_z'].iloc[i]
        if np.isnan(fr_z) or np.isnan(mis_z):
            continue

        # Both must exceed threshold and agree on direction
        if fr_z > fr_z_thresh and mis_z > mis_z_thresh:
            direction = 'short'  # both say premium/funding too high
        elif fr_z < -fr_z_thresh and mis_z < -mis_z_thresh:
            direction = 'long'
        else:
            continue

        signals.append({
            'bar_idx': i,
            'time': bars.index[i],
            'direction': direction,
            'z': fr_z,
            'signal_type': 'combined',
        })
    return signals


# ============================================================================
# SIMULATION on 1-min bars
# ============================================================================

def simulate_4h_trades(signals, bars_1m, tp_bps=None, sl_bps=None,
                       hold_hours=4, entry_offset_bps=5):
    """
    Simulate trades on 1-min bars.
    Entry: limit order at offset from signal bar close.
    Exit: TP (limit), SL (market), or timeout at hold_hours (market).
    """
    bar_times = bars_1m.index
    bar_high = bars_1m['high'].values
    bar_low = bars_1m['low'].values
    bar_close = bars_1m['close'].values
    n = len(bars_1m)
    hold_bars = hold_hours * 60

    trades = []
    last_exit_time = None

    for sig in signals:
        sig_time = sig['time']

        # Cooldown: at least 1h between trades
        if last_exit_time is not None:
            dt = (sig_time - last_exit_time).total_seconds()
            if dt < 3600:
                continue

        # Find signal bar in 1-min data
        idx = bar_times.searchsorted(sig_time)
        if idx >= n - hold_bars - 10 or idx < 1:
            continue

        direction = sig['direction']
        current_price = bar_close[idx]

        # Entry limit order
        if direction == 'long':
            entry_price = current_price * (1 - entry_offset_bps / 10000)
        else:
            entry_price = current_price * (1 + entry_offset_bps / 10000)

        # Try to fill entry (within 30 min)
        fill_window = min(idx + 30, n - hold_bars)
        fill_bar = None
        for j in range(idx, fill_window):
            if direction == 'long' and bar_low[j] <= entry_price:
                fill_bar = j
                break
            elif direction == 'short' and bar_high[j] >= entry_price:
                fill_bar = j
                break

        if fill_bar is None:
            continue

        # TP/SL prices
        if tp_bps is not None:
            if direction == 'long':
                tp_price = entry_price * (1 + tp_bps / 10000)
            else:
                tp_price = entry_price * (1 - tp_bps / 10000)
        else:
            tp_price = None

        if sl_bps is not None:
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
            # Check SL first
            if sl_price is not None:
                if direction == 'long' and bar_low[k] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break
                elif direction == 'short' and bar_high[k] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break

            # Check TP
            if tp_price is not None:
                if direction == 'long' and bar_high[k] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar = k
                    break
                elif direction == 'short' and bar_low[k] <= tp_price:
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
            'time': sig_time,
            'direction': direction,
            'exit_reason': exit_reason,
            'raw_pnl_pct': raw_pnl_pct,
            'net_pnl_pct': net_pnl_pct,
            'net_pnl_bps': net_pnl_pct * 100,
            'hold_min': exit_bar - fill_bar,
            'z': sig['z'],
        })
        last_exit_time = bar_times[exit_bar]

    return trades


def analyze(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES")
        return None

    net = np.array([t['net_pnl_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    total_pct = sum(t['net_pnl_pct'] for t in trades)
    avg = net.mean()
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0  # ~6 trades/day at 4h

    reasons = {}
    for t in trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1

    print(f"  {label}")
    print(f"    n={n:4d}  WR={wr:5.1f}%  avg={avg:+7.1f}bps  "
          f"total={total_pct:+7.2f}%  Sharpe={sharpe:+6.1f}  exits={reasons}")

    # Direction breakdown
    for d in ['long', 'short']:
        dt = [t for t in trades if t['direction'] == d]
        if dt:
            dn = np.array([t['net_pnl_bps'] for t in dt])
            print(f"    {d.upper():5s}: n={len(dt)}, WR={(dn>0).sum()/len(dn)*100:.1f}%, "
                  f"avg={dn.mean():+.1f}bps")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total_pct, 'sharpe': sharpe}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("v43c: Funding Rate Directional — 4h Holding Period")
    print("=" * 70)

    # Test on SOL first (funding IC=-0.12 confirmed OOS)
    for symbol in ['SOLUSDT', 'ETHUSDT']:
        print(f"\n{'='*70}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'='*70}")

        ticker_dir = PARQUET_DIR / symbol / 'ticker'
        available = sorted([f.stem for f in ticker_dir.glob('*.parquet')])
        if not available:
            print(f"  No ticker data for {symbol}")
            continue

        print(f"  Available: {len(available)} days ({available[0]} to {available[-1]})")

        # Use all available data for now (will split IS/OOS later)
        # For quick test, use first 30 days
        test_dates = available[:30]
        print(f"  Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

        # Load data
        print(f"\n  Loading {symbol}...")
        tf_bars, bars_1m = load_multi_tf_bars(symbol, test_dates)
        if not tf_bars or bars_1m.empty:
            print("  ERROR: No data!")
            continue

        # Add features to 1h bars (window=24 = 24h)
        bars_1h = add_features(tf_bars['1h'], window=24)
        # Add features to 4h bars (window=6 = 24h)
        bars_4h = add_features(tf_bars['4h'], window=6)

        print(f"\n  RAM: {get_ram_mb():.0f}MB")

        # ============================================================
        # TEST 1: Funding Rate Contrarian on 1h bars
        # ============================================================
        print(f"\n  --- Funding Contrarian (1h bars, 4h hold) ---")
        for z_thresh in [0.5, 1.0, 1.5, 2.0]:
            signals = generate_funding_signals(bars_1h, fr_z_threshold=z_thresh,
                                                direction_mode='contrarian')
            if len(signals) < 3:
                print(f"  fr_z>{z_thresh}: only {len(signals)} signals, skip")
                continue

            # No TP, just timeout at 4h
            trades = simulate_4h_trades(signals, bars_1m, tp_bps=None, sl_bps=None,
                                         hold_hours=4, entry_offset_bps=5)
            analyze(trades, f"Funding contrarian z>{z_thresh} (no TP/SL, 4h timeout)")

            # With TP
            for tp in [30, 50, 80]:
                trades = simulate_4h_trades(signals, bars_1m, tp_bps=tp, sl_bps=None,
                                             hold_hours=4, entry_offset_bps=5)
                analyze(trades, f"Funding contrarian z>{z_thresh} TP={tp} (no SL)")

            # With TP + SL
            for tp, sl in [(30, 60), (50, 100), (80, 160)]:
                trades = simulate_4h_trades(signals, bars_1m, tp_bps=tp, sl_bps=sl,
                                             hold_hours=4, entry_offset_bps=5)
                analyze(trades, f"Funding contrarian z>{z_thresh} TP={tp} SL={sl}")

        # ============================================================
        # TEST 2: Mark-Index Spread Contrarian
        # ============================================================
        print(f"\n  --- Mark-Index Spread Contrarian (1h bars, 4h hold) ---")
        for z_thresh in [0.5, 1.0, 1.5, 2.0]:
            signals = generate_mis_signals(bars_1h, mis_z_threshold=z_thresh)
            if len(signals) < 3:
                continue
            trades = simulate_4h_trades(signals, bars_1m, tp_bps=None, sl_bps=None,
                                         hold_hours=4, entry_offset_bps=5)
            analyze(trades, f"MIS contrarian z>{z_thresh} (no TP/SL, 4h timeout)")

            for tp in [30, 50]:
                trades = simulate_4h_trades(signals, bars_1m, tp_bps=tp, sl_bps=None,
                                             hold_hours=4, entry_offset_bps=5)
                analyze(trades, f"MIS contrarian z>{z_thresh} TP={tp}")

        # ============================================================
        # TEST 3: Combined Funding + MIS
        # ============================================================
        print(f"\n  --- Combined Funding + MIS (1h bars, 4h hold) ---")
        for fr_z in [0.5, 1.0]:
            for mis_z in [0.5, 1.0]:
                signals = generate_combined_signals(bars_1h, fr_z_thresh=fr_z,
                                                     mis_z_thresh=mis_z)
                if len(signals) < 3:
                    continue
                trades = simulate_4h_trades(signals, bars_1m, tp_bps=None, sl_bps=None,
                                             hold_hours=4, entry_offset_bps=5)
                analyze(trades, f"Combined fr_z>{fr_z} mis_z>{mis_z} (4h timeout)")

                for tp in [30, 50]:
                    trades = simulate_4h_trades(signals, bars_1m, tp_bps=tp, sl_bps=None,
                                                 hold_hours=4, entry_offset_bps=5)
                    analyze(trades, f"Combined fr_z>{fr_z} mis_z>{mis_z} TP={tp}")

        # ============================================================
        # TEST 4: Funding on 4h bars (direct)
        # ============================================================
        print(f"\n  --- Funding Contrarian (4h bars, 4h hold) ---")
        for z_thresh in [0.5, 1.0, 1.5]:
            signals = generate_funding_signals(bars_4h, fr_z_threshold=z_thresh,
                                                direction_mode='contrarian')
            if len(signals) < 3:
                continue
            trades = simulate_4h_trades(signals, bars_1m, tp_bps=None, sl_bps=None,
                                         hold_hours=4, entry_offset_bps=5)
            analyze(trades, f"4h-bar Funding contrarian z>{z_thresh} (4h timeout)")

        # Cleanup
        del tf_bars, bars_1m, bars_1h, bars_4h
        import gc; gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
