#!/usr/bin/env python3
"""
v43: Three New Strategy Ideas — Quick 7-Day Prototype

Tests 3 independent strategy ideas on 1-min bars built from 5-sec ticker data.
ALL use fixed TP (limit) + fixed SL + timeout. NO trailing stop.
ALL entries via limit orders (maker fee 0.02%).

IDEA A: Funding Settlement Mean-Reversion
  - Every 8h (00:00, 08:00, 16:00 UTC) funding settles
  - Post-settlement vol is 14-21% higher (proven in v34)
  - If price moves >N bps in first M minutes after settlement,
    place limit order to fade the move
  - TP = fixed bps (limit), SL = fixed bps, timeout = 60 min

IDEA B: OI Squeeze Mean-Reversion
  - When OI drops sharply (positions closing = squeeze), price dislocates
  - Fade the price move when OI drop exceeds threshold
  - Uses 5-min OI change from ticker data

IDEA C: Spread Widening Mean-Reversion
  - When bid-ask spread widens (stress/illiquidity), price dislocates
  - Fade the price move when spread z-score exceeds threshold
  - Compensation for providing liquidity during stress

Fees: maker 0.02%, taker 0.055%
"""

import sys, time, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_PCT = 0.02    # entry (limit) + TP exit (limit)
TAKER_FEE_PCT = 0.055   # SL exit (market) or timeout exit (market)
PARQUET_DIR = Path('parquet')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ticker_days(symbol, dates):
    """Load ticker parquet for specific dates, build 1-min OHLC bars."""
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    all_bars = []
    for date_str in dates:
        path = ticker_dir / f'{date_str}.parquet'
        if not path.exists():
            print(f"  SKIP {date_str} (no file)")
            continue
        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp_us'], unit='us')
        df = df.set_index('timestamp').sort_index()

        # Build 1-min OHLC from last_price
        bars = df['last_price'].resample('1min').ohlc().dropna()
        # Add OI, funding, spread info per bar
        bars['oi'] = df['open_interest'].resample('1min').last()
        bars['oi_start'] = df['open_interest'].resample('1min').first()
        bars['funding_rate'] = df['funding_rate'].resample('1min').last()
        bars['next_funding_time'] = df['next_funding_time'].resample('1min').last()
        bars['bid1'] = df['bid1_price'].resample('1min').mean()
        bars['ask1'] = df['ask1_price'].resample('1min').mean()
        bars['spread_bps'] = (bars['ask1'] - bars['bid1']) / bars['close'] * 10000
        bars['volume_24h'] = df['volume_24h'].resample('1min').last()

        all_bars.append(bars)

    if not all_bars:
        return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f"  Loaded {len(result):,} 1-min bars for {symbol}")
    return result


# ============================================================================
# SIMULATION ENGINE — Fixed TP/SL, no trailing stop
# ============================================================================

def simulate_trades(signals, bars, tp_bps, sl_bps, timeout_min=60):
    """
    Simulate trades with fixed TP (limit) + SL (market) + timeout.

    signals: list of dicts with 'time' (index into bars), 'direction' ('long'/'short')
    bars: DataFrame with OHLC columns, DatetimeIndex
    tp_bps: take profit in basis points (limit order = maker fee)
    sl_bps: stop loss in basis points (market order = taker fee)
    timeout_min: max hold time in minutes

    Returns list of trade dicts.
    """
    bar_times = bars.index
    bar_high = bars['high'].values
    bar_low = bars['low'].values
    bar_close = bars['close'].values
    n_bars = len(bars)

    trades = []
    last_exit_time = None

    for sig in signals:
        sig_time = sig['time']
        direction = sig['direction']
        entry_price = sig['entry_price']

        # Find bar index for signal time
        idx = bar_times.searchsorted(sig_time)
        if idx >= n_bars - timeout_min - 1 or idx < 1:
            continue

        # Cooldown: skip if last trade exited less than 5 min ago
        if last_exit_time is not None:
            dt = (sig_time - last_exit_time).total_seconds()
            if dt < 300:
                continue

        # Compute TP and SL prices
        if direction == 'long':
            tp_price = entry_price * (1 + tp_bps / 10000)
            sl_price = entry_price * (1 - sl_bps / 10000)
        else:
            tp_price = entry_price * (1 - tp_bps / 10000)
            sl_price = entry_price * (1 + sl_bps / 10000)

        # First, check if entry limit order fills
        # Entry is a limit order placed at entry_price
        # For long: limit buy at entry_price, fills if bar_low <= entry_price
        # For short: limit sell at entry_price, fills if bar_high >= entry_price
        fill_bar = None
        fill_window = min(idx + 10, n_bars - timeout_min)  # 10 min to fill entry
        for j in range(idx, fill_window):
            if direction == 'long' and bar_low[j] <= entry_price:
                fill_bar = j
                break
            elif direction == 'short' and bar_high[j] >= entry_price:
                fill_bar = j
                break

        if fill_bar is None:
            continue  # Entry not filled

        # Now simulate exit
        exit_end = min(fill_bar + timeout_min, n_bars - 1)
        exit_price = None
        exit_reason = None

        for k in range(fill_bar + 1, exit_end + 1):
            if direction == 'long':
                # Check SL first (conservative)
                if bar_low[k] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar = k
                    break
                # Check TP
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
            exit_bar = exit_end

        # Calculate PnL
        if direction == 'long':
            raw_pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            raw_pnl_pct = (entry_price - exit_price) / entry_price * 100

        # Fees
        entry_fee = MAKER_FEE_PCT  # limit order entry
        if exit_reason == 'take_profit':
            exit_fee = MAKER_FEE_PCT  # limit order exit
        else:
            exit_fee = TAKER_FEE_PCT  # market order exit (SL or timeout)

        net_pnl_pct = raw_pnl_pct - entry_fee - exit_fee
        net_pnl_bps = net_pnl_pct * 100  # convert % to bps

        trades.append({
            'time': sig_time,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'raw_pnl_pct': raw_pnl_pct,
            'net_pnl_pct': net_pnl_pct,
            'net_pnl_bps': net_pnl_bps,
            'hold_min': exit_bar - fill_bar,
        })
        last_exit_time = bar_times[exit_bar]

    return trades


# ============================================================================
# IDEA A: Funding Settlement Mean-Reversion
# ============================================================================

def generate_signals_funding_mr(bars, move_threshold_bps=15, lookback_min=5,
                                 entry_offset_bps=5):
    """
    After funding settlement, if price moved >threshold in first lookback_min,
    place limit order to fade the move.

    Entry: limit order at (current_price - offset) for long,
           or (current_price + offset) for short.
    """
    signals = []
    bar_times = bars.index

    # Identify funding settlement times from next_funding_time changes
    nft = bars['next_funding_time'].values
    funding_times = set()
    for i in range(1, len(nft)):
        if nft[i] != nft[i-1] and not np.isnan(nft[i]) and not np.isnan(nft[i-1]):
            # Funding just settled — nft jumped to next period
            funding_times.add(bar_times[i])

    print(f"  [Funding MR] Found {len(funding_times)} funding settlements")

    for ft in sorted(funding_times):
        ft_idx = bar_times.searchsorted(ft)
        if ft_idx < 5 or ft_idx + lookback_min + 60 >= len(bars):
            continue

        # Price before settlement
        pre_price = bars['close'].iloc[ft_idx - 1]

        # Price after lookback_min minutes
        post_idx = ft_idx + lookback_min
        if post_idx >= len(bars):
            continue
        post_price = bars['close'].iloc[post_idx]

        # Move in bps
        move_bps = (post_price - pre_price) / pre_price * 10000

        if abs(move_bps) < move_threshold_bps:
            continue

        # Fade the move
        if move_bps > 0:
            # Price went up → fade → go short
            direction = 'short'
            entry_price = post_price * (1 + entry_offset_bps / 10000)
        else:
            # Price went down → fade → go long
            direction = 'long'
            entry_price = post_price * (1 - entry_offset_bps / 10000)

        signals.append({
            'time': bar_times[post_idx],
            'direction': direction,
            'entry_price': entry_price,
            'move_bps': move_bps,
        })

    print(f"  [Funding MR] Generated {len(signals)} signals")
    return signals


# ============================================================================
# IDEA B: OI Squeeze Mean-Reversion
# ============================================================================

def generate_signals_oi_squeeze(bars, oi_drop_pct=-0.1, lookback_min=5,
                                 entry_offset_bps=5):
    """
    When OI drops sharply (positions closing = squeeze), fade the price move.
    OI drop computed over lookback_min minutes.
    """
    signals = []
    bar_times = bars.index
    oi = bars['oi'].values
    close = bars['close'].values

    # Compute rolling OI change
    count = 0
    for i in range(lookback_min, len(bars) - 60):
        oi_now = oi[i]
        oi_prev = oi[i - lookback_min]
        if oi_prev <= 0 or np.isnan(oi_prev) or np.isnan(oi_now):
            continue

        oi_change_pct = (oi_now - oi_prev) / oi_prev * 100

        if oi_change_pct > oi_drop_pct:  # Not enough drop
            continue

        # Price move during the OI drop
        price_now = close[i]
        price_prev = close[i - lookback_min]
        price_move_bps = (price_now - price_prev) / price_prev * 10000

        if abs(price_move_bps) < 5:  # Need some price move to fade
            continue

        # Fade the price move
        if price_move_bps > 0:
            direction = 'short'
            entry_price = price_now * (1 + entry_offset_bps / 10000)
        else:
            direction = 'long'
            entry_price = price_now * (1 - entry_offset_bps / 10000)

        signals.append({
            'time': bar_times[i],
            'direction': direction,
            'entry_price': entry_price,
            'oi_change_pct': oi_change_pct,
            'price_move_bps': price_move_bps,
        })
        count += 1

    print(f"  [OI Squeeze] Generated {len(signals)} signals")
    return signals


# ============================================================================
# IDEA C: Spread Widening Mean-Reversion
# ============================================================================

def generate_signals_spread_mr(bars, spread_z_threshold=2.0, lookback_bars=60,
                                entry_offset_bps=5):
    """
    When bid-ask spread widens significantly (z-score > threshold),
    fade the concurrent price move.
    """
    signals = []
    bar_times = bars.index
    spread = bars['spread_bps'].values
    close = bars['close'].values

    # Rolling mean and std of spread
    for i in range(lookback_bars, len(bars) - 60):
        window = spread[i - lookback_bars:i]
        valid = window[~np.isnan(window)]
        if len(valid) < lookback_bars // 2:
            continue

        mean_spread = np.mean(valid)
        std_spread = np.std(valid)
        if std_spread < 0.01:
            continue

        current_spread = spread[i]
        if np.isnan(current_spread):
            continue

        z = (current_spread - mean_spread) / std_spread

        if z < spread_z_threshold:
            continue

        # Price move in last 5 bars
        if i < 5:
            continue
        price_move_bps = (close[i] - close[i - 5]) / close[i - 5] * 10000

        if abs(price_move_bps) < 3:  # Need some move
            continue

        # Fade the move
        if price_move_bps > 0:
            direction = 'short'
            entry_price = close[i] * (1 + entry_offset_bps / 10000)
        else:
            direction = 'long'
            entry_price = close[i] * (1 - entry_offset_bps / 10000)

        signals.append({
            'time': bar_times[i],
            'direction': direction,
            'entry_price': entry_price,
            'spread_z': z,
            'price_move_bps': price_move_bps,
        })

    print(f"  [Spread MR] Generated {len(signals)} signals")
    return signals


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_trades(trades, label):
    """Print trade statistics."""
    if not trades:
        print(f"  {label}: NO TRADES")
        return

    net = np.array([t['net_pnl_bps'] for t in trades])
    wins = (net > 0).sum()
    wr = wins / len(net) * 100
    total_pct = sum(t['net_pnl_pct'] for t in trades)
    avg_bps = net.mean()
    std_bps = net.std() if len(net) > 1 else 1
    sharpe = avg_bps / std_bps * np.sqrt(252 * 8) if std_bps > 0 else 0

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons[r] = reasons.get(r, 0) + 1

    # Max drawdown
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    maxdd = (peak - cum).max()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:     {len(trades)}")
    print(f"  Win Rate:   {wr:.1f}%")
    print(f"  Avg Net:    {avg_bps:+.1f} bps")
    print(f"  Total Net:  {total_pct:+.2f}%")
    print(f"  Sharpe:     {sharpe:.1f}")
    print(f"  Max DD:     {maxdd:.1f} bps")
    print(f"  Avg Hold:   {np.mean([t['hold_min'] for t in trades]):.1f} min")
    print(f"  Exits:      {reasons}")

    # Per-direction breakdown
    for d in ['long', 'short']:
        d_trades = [t for t in trades if t['direction'] == d]
        if d_trades:
            d_net = np.array([t['net_pnl_bps'] for t in d_trades])
            d_wr = (d_net > 0).sum() / len(d_net) * 100
            print(f"  {d.upper():5s}:  n={len(d_trades)}, WR={d_wr:.1f}%, avg={d_net.mean():+.1f} bps")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("v43: Three New Strategy Ideas — Quick Prototype")
    print("=" * 60)

    symbol = 'ETHUSDT'

    # Get available dates
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    available = sorted([f.stem for f in ticker_dir.glob('*.parquet')])
    print(f"\nAvailable ticker days: {len(available)} ({available[0]} to {available[-1]})")

    # Use 7 days for quick test — pick from available dates
    # Prefer Aug 2025 (edge of previous research) or Feb 2026 (fresh)
    test_dates = [d for d in available if '2025-08-04' <= d <= '2025-08-10']
    if len(test_dates) < 5:
        # Fallback: last 7 available days
        test_dates = available[-7:]

    test_dates = test_dates[:7]
    print(f"Test period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    # Load data
    print(f"\nLoading {symbol} ticker data...")
    bars = load_ticker_days(symbol, test_dates)
    if bars.empty:
        print("ERROR: No data loaded!")
        return

    print(f"RAM after load: {get_ram_mb():.0f} MB")

    # TP/SL configurations to test
    configs = [
        {'tp_bps': 10, 'sl_bps': 20},
        {'tp_bps': 15, 'sl_bps': 30},
        {'tp_bps': 20, 'sl_bps': 40},
        {'tp_bps': 10, 'sl_bps': 40},  # asymmetric: wide SL
        {'tp_bps': 8,  'sl_bps': 16},  # tight
    ]

    # ================================================================
    # IDEA A: Funding Settlement MR
    # ================================================================
    print(f"\n{'='*60}")
    print("IDEA A: Funding Settlement Mean-Reversion")
    print(f"{'='*60}")

    for move_thresh in [10, 15, 20, 30]:
        for lookback in [3, 5, 10]:
            signals = generate_signals_funding_mr(
                bars, move_threshold_bps=move_thresh, lookback_min=lookback,
                entry_offset_bps=3
            )
            if len(signals) < 3:
                continue
            for cfg in configs[:3]:  # Test top 3 configs
                trades = simulate_trades(signals, bars, **cfg)
                label = (f"Funding MR move>{move_thresh}bps lb={lookback}m "
                         f"TP={cfg['tp_bps']} SL={cfg['sl_bps']}")
                analyze_trades(trades, label)

    # ================================================================
    # IDEA B: OI Squeeze MR
    # ================================================================
    print(f"\n{'='*60}")
    print("IDEA B: OI Squeeze Mean-Reversion")
    print(f"{'='*60}")

    for oi_drop in [-0.05, -0.1, -0.2, -0.5]:
        for lookback in [5, 15, 30]:
            signals = generate_signals_oi_squeeze(
                bars, oi_drop_pct=oi_drop, lookback_min=lookback,
                entry_offset_bps=3
            )
            if len(signals) < 3:
                continue
            for cfg in configs[:3]:
                trades = simulate_trades(signals, bars, **cfg)
                label = (f"OI Squeeze drop<{oi_drop}% lb={lookback}m "
                         f"TP={cfg['tp_bps']} SL={cfg['sl_bps']}")
                analyze_trades(trades, label)

    # ================================================================
    # IDEA C: Spread Widening MR
    # ================================================================
    print(f"\n{'='*60}")
    print("IDEA C: Spread Widening Mean-Reversion")
    print(f"{'='*60}")

    for z_thresh in [1.5, 2.0, 3.0]:
        for lookback in [30, 60, 120]:
            signals = generate_signals_spread_mr(
                bars, spread_z_threshold=z_thresh, lookback_bars=lookback,
                entry_offset_bps=3
            )
            if len(signals) < 3:
                continue
            for cfg in configs[:3]:
                trades = simulate_trades(signals, bars, **cfg)
                label = (f"Spread MR z>{z_thresh} lb={lookback}m "
                         f"TP={cfg['tp_bps']} SL={cfg['sl_bps']}")
                analyze_trades(trades, label)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"RAM: {get_ram_mb():.0f} MB")


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


if __name__ == '__main__':
    main()
