#!/usr/bin/env python3
"""
v43k: Cascade MM with Fixed TP/SL — Tick-Level Simulation

The ONLY strategy with structural edge in this repo: liquidation cascades.
Previous tests used trailing stop (proven to have lookahead bias).
This test uses ONLY fixed TP/SL + timeout. No trailing stop.

Tick-level simulation:
  1. Load liquidation events → detect cascades (clusters of liqs)
  2. Load tick-level trades for same period
  3. On cascade: place limit order to fade the forced flow
  4. Process each tick sequentially — no intra-bar ambiguity
  5. Exit: fixed TP (limit) or SL (market) or timeout (market)

Data: Bybit liquidations + futures trades (May 2025 – Feb 2026)
Symbols: SOL, ETH, DOGE
Fees: maker 0.02%, taker 0.055%
"""

import sys, time, gzip, json, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE = 0.0002   # 2 bps
TAKER_FEE = 0.00055  # 5.5 bps
DATA_DIR = Path('data')
PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


# ============================================================================
# LIQUIDATION DATA
# ============================================================================

def load_liquidations(symbol, date_str):
    """Load liquidation events for a single day."""
    liq_dir = DATA_DIR / symbol / 'bybit' / 'liquidations'
    files = sorted(liq_dir.glob(f'liquidation_{date_str}_*.jsonl.gz'))
    events = []

    for f in files:
        try:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    for item in rec.get('result', {}).get('data', []):
                        events.append({
                            'ts_ms': int(item['T']),
                            'side': item['S'],  # 'Buy' = short liq, 'Sell' = long liq
                            'qty': float(item['v']),
                            'price': float(item['p']),
                        })
        except Exception:
            continue

    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events).sort_values('ts_ms').reset_index(drop=True)
    return df


def detect_cascades(liq_df, window_ms=60000, min_count=5, min_usd=10000):
    """
    Detect liquidation cascades: clusters of liquidations within window_ms.
    A cascade = min_count liquidations within window_ms, total value > min_usd.

    Returns list of cascade events with direction and timing.
    """
    if liq_df.empty:
        return []

    ts = liq_df['ts_ms'].values
    sides = liq_df['side'].values
    qtys = liq_df['qty'].values
    prices = liq_df['price'].values
    n = len(ts)

    cascades = []
    i = 0
    last_cascade_ts = 0

    while i < n:
        # Find window of liquidations
        j = i
        while j < n and ts[j] - ts[i] <= window_ms:
            j += 1

        count = j - i
        if count >= min_count:
            # Check total USD value
            total_usd = sum(qtys[k] * prices[k] for k in range(i, j))
            if total_usd >= min_usd:
                # Determine dominant side
                buy_count = sum(1 for k in range(i, j) if sides[k] == 'Buy')
                sell_count = count - buy_count
                buy_usd = sum(qtys[k] * prices[k] for k in range(i, j) if sides[k] == 'Buy')
                sell_usd = total_usd - buy_usd

                # 'Buy' liquidation = short position liquidated = price going UP
                # 'Sell' liquidation = long position liquidated = price going DOWN
                if buy_count > sell_count:
                    cascade_dir = 'up'  # shorts getting liquidated, price moving up
                else:
                    cascade_dir = 'down'  # longs getting liquidated, price moving down

                cascade_ts = ts[j - 1]  # end of cascade

                # Cooldown: skip if too close to last cascade
                if cascade_ts - last_cascade_ts > 300000:  # 5 min cooldown
                    cascades.append({
                        'ts_ms': cascade_ts,
                        'direction': cascade_dir,
                        'count': count,
                        'total_usd': total_usd,
                        'price': prices[j - 1],
                        'buy_count': buy_count,
                        'sell_count': sell_count,
                    })
                    last_cascade_ts = cascade_ts

                i = j  # skip past cascade
                continue

        i += 1

    return cascades


# ============================================================================
# TICK-LEVEL TRADE DATA
# ============================================================================

def load_ticks(symbol, date_str):
    """Load tick-level trades from parquet."""
    path = PARQUET_DIR / symbol / 'trades' / 'bybit_futures' / f'{date_str}.parquet'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=['timestamp_us', 'price', 'quantity', 'side'])
    df['ts_ms'] = df['timestamp_us'] // 1000  # us → ms
    return df.sort_values('ts_ms').reset_index(drop=True)


# ============================================================================
# TICK-LEVEL SIMULATION
# ============================================================================

def simulate_cascade_trades(cascades, ticks, tp_bps, sl_bps, timeout_ms=3600000,
                            entry_offset_bps=5, fill_timeout_ms=300000):
    """
    Tick-level simulation of cascade MM trades.

    For each cascade:
      1. Place limit order to FADE the cascade direction
         (cascade up → short limit above, cascade down → long limit below)
      2. Process ticks sequentially to check fill
      3. Once filled, check each tick for TP/SL/timeout
      4. No trailing stop

    All entries via limit (maker fee).
    TP exit via limit (maker fee).
    SL/timeout exit via market (taker fee).
    """
    if ticks.empty or not cascades:
        return []

    tick_ts = ticks['ts_ms'].values
    tick_price = ticks['price'].values
    n_ticks = len(tick_ts)

    trades = []
    last_exit_ts = 0

    for ci, cascade in enumerate(cascades):
        c_ts = cascade['ts_ms']
        c_price = cascade['price']
        c_dir = cascade['direction']

        # Cooldown
        if c_ts < last_exit_ts + 300000:  # 5 min after last exit
            continue

        # Fade direction
        if c_dir == 'up':
            trade_dir = 'short'  # price went up from liquidations, fade it
            entry_price = c_price * (1 + entry_offset_bps / 10000)
        else:
            trade_dir = 'long'  # price went down from liquidations, fade it
            entry_price = c_price * (1 - entry_offset_bps / 10000)

        # Find starting tick index
        start_idx = np.searchsorted(tick_ts, c_ts)
        if start_idx >= n_ticks - 100:
            continue

        # Phase 1: Try to fill entry limit order
        fill_deadline = c_ts + fill_timeout_ms
        fill_idx = None

        for ti in range(start_idx, n_ticks):
            if tick_ts[ti] > fill_deadline:
                break
            p = tick_price[ti]
            if trade_dir == 'long' and p <= entry_price:
                fill_idx = ti
                break
            elif trade_dir == 'short' and p >= entry_price:
                fill_idx = ti
                break

        if fill_idx is None:
            continue  # Entry not filled

        fill_ts = tick_ts[fill_idx]
        actual_entry = entry_price  # limit order fills at limit price

        # Compute TP/SL prices
        if trade_dir == 'long':
            tp_price = actual_entry * (1 + tp_bps / 10000)
            sl_price = actual_entry * (1 - sl_bps / 10000)
        else:
            tp_price = actual_entry * (1 - tp_bps / 10000)
            sl_price = actual_entry * (1 + sl_bps / 10000)

        # Phase 2: Process ticks for exit
        exit_deadline = fill_ts + timeout_ms
        exit_price = None
        exit_reason = None
        exit_ts = None

        for ti in range(fill_idx + 1, n_ticks):
            t = tick_ts[ti]
            p = tick_price[ti]

            if t > exit_deadline:
                # Timeout — exit at current price (market order)
                exit_price = p
                exit_reason = 'timeout'
                exit_ts = t
                break

            # Check SL first (conservative)
            if trade_dir == 'long' and p <= sl_price:
                exit_price = sl_price  # SL is a stop-market, fills at SL price
                exit_reason = 'sl'
                exit_ts = t
                break
            elif trade_dir == 'short' and p >= sl_price:
                exit_price = sl_price
                exit_reason = 'sl'
                exit_ts = t
                break

            # Check TP
            if trade_dir == 'long' and p >= tp_price:
                exit_price = tp_price  # TP is a limit order
                exit_reason = 'tp'
                exit_ts = t
                break
            elif trade_dir == 'short' and p <= tp_price:
                exit_price = tp_price
                exit_reason = 'tp'
                exit_ts = t
                break

        if exit_price is None:
            # End of data
            exit_price = tick_price[-1]
            exit_reason = 'eod'
            exit_ts = tick_ts[-1]

        # PnL
        if trade_dir == 'long':
            raw_pnl = (exit_price - actual_entry) / actual_entry
        else:
            raw_pnl = (actual_entry - exit_price) / actual_entry

        entry_fee = MAKER_FEE  # limit order
        exit_fee = MAKER_FEE if exit_reason == 'tp' else TAKER_FEE
        net_pnl = raw_pnl - entry_fee - exit_fee

        hold_ms = exit_ts - fill_ts

        trades.append({
            'cascade_ts': c_ts,
            'fill_ts': fill_ts,
            'exit_ts': exit_ts,
            'dir': trade_dir,
            'entry': actual_entry,
            'exit': exit_price,
            'exit_reason': exit_reason,
            'raw_bps': raw_pnl * 10000,
            'net_bps': net_pnl * 10000,
            'hold_sec': hold_ms / 1000,
            'cascade_count': cascade['count'],
            'cascade_usd': cascade['total_usd'],
        })
        last_exit_ts = exit_ts

    return trades


# ============================================================================
# ANALYSIS
# ============================================================================

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
    sharpe = avg / std * np.sqrt(252 * 24) if std > 0 else 0  # hourly

    reasons = defaultdict(int)
    for t in trades:
        reasons[t['exit_reason']] += 1

    avg_hold = np.mean([t['hold_sec'] for t in trades])

    # Max drawdown
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    maxdd = (peak - cum).max()

    print(f"  {label}")
    print(f"    n={n:4d} WR={wr:5.1f}% avg={avg:+7.1f}bps total={total:+7.2f}% "
          f"Sharpe={sharpe:+6.1f} maxDD={maxdd:.0f}bps "
          f"avgHold={avg_hold:.0f}s exits={dict(reasons)}")

    # Direction breakdown
    for d in ['long', 'short']:
        dt = [t for t in trades if t['dir'] == d]
        if dt:
            dn = np.array([t['net_bps'] for t in dt])
            print(f"    {d.upper():5s}: n={len(dt)} WR={(dn>0).sum()/len(dn)*100:.1f}% "
                  f"avg={dn.mean():+.1f}bps")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe,
            'maxdd': maxdd}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("v43k: Cascade MM — Tick-Level, Fixed TP/SL, No Trailing Stop")
    print("=" * 80)

    symbols = ['SOLUSDT', 'ETHUSDT', 'DOGEUSDT']

    # TP/SL configs to test
    configs = [
        (20, 40,   'TP=20 SL=40'),
        (30, 60,   'TP=30 SL=60'),
        (50, 100,  'TP=50 SL=100'),
        (50, 50,   'TP=50 SL=50 (1:1)'),
        (30, 90,   'TP=30 SL=90 (1:3)'),
        (20, 20,   'TP=20 SL=20 (1:1 tight)'),
        (100, 200, 'TP=100 SL=200'),
    ]

    # Cascade detection params to test
    cascade_params = [
        (5, 10000,  'min5 $10k'),
        (10, 50000, 'min10 $50k'),
        (3, 5000,   'min3 $5k'),
    ]

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        # Get available dates with both liquidation and trade data
        liq_dir = DATA_DIR / symbol / 'bybit' / 'liquidations'
        trade_dir = PARQUET_DIR / symbol / 'trades' / 'bybit_futures'

        if not liq_dir.exists():
            print(f"  No liquidation data")
            continue

        liq_dates = set()
        for f in liq_dir.glob('liquidation_*.jsonl.gz'):
            parts = f.stem.replace('.jsonl', '').split('_')
            liq_dates.add(parts[1])

        trade_dates = set()
        if trade_dir.exists():
            for f in trade_dir.glob('*.parquet'):
                trade_dates.add(f.stem)

        common_dates = sorted(liq_dates & trade_dates)
        print(f"  Common dates: {len(common_dates)} "
              f"({common_dates[0] if common_dates else 'N/A'} to "
              f"{common_dates[-1] if common_dates else 'N/A'})")

        if len(common_dates) < 7:
            print(f"  Too few dates, skip")
            continue

        # Use first 7 days for quick test, then expand
        test_dates = common_dates[:14]
        print(f"  Testing on {len(test_dates)} days: {test_dates[0]} to {test_dates[-1]}")

        # Load data day by day to save RAM
        for cp_count, cp_usd, cp_label in cascade_params:
            print(f"\n  --- Cascade params: {cp_label} ---")

            all_cascades = []
            all_ticks_list = []

            for di, date_str in enumerate(test_dates):
                t1 = time.time()

                # Load liquidations
                liq_df = load_liquidations(symbol, date_str)

                # Detect cascades
                day_cascades = detect_cascades(liq_df, window_ms=60000,
                                                min_count=cp_count, min_usd=cp_usd)

                # Load ticks
                ticks = load_ticks(symbol, date_str)

                if ticks.empty:
                    continue

                # Run simulation for each TP/SL config
                if di == 0:
                    # First day: show cascade stats
                    print(f"    Day {date_str}: {len(liq_df)} liqs, "
                          f"{len(day_cascades)} cascades, "
                          f"{len(ticks):,} ticks ({time.time()-t1:.1f}s)")

                all_cascades.extend(day_cascades)
                all_ticks_list.append(ticks)

                if (di + 1) % 5 == 0 or di == len(test_dates) - 1:
                    elapsed = time.time() - t0
                    print(f"    [{di+1}/{len(test_dates)}] "
                          f"cascades={len(all_cascades)} "
                          f"{elapsed:.0f}s RAM={get_ram_mb():.0f}MB", flush=True)

            if not all_ticks_list or not all_cascades:
                print(f"    No cascades or ticks!")
                continue

            # Combine all ticks
            all_ticks = pd.concat(all_ticks_list, ignore_index=True)
            all_ticks = all_ticks.sort_values('ts_ms').reset_index(drop=True)
            print(f"    Total: {len(all_cascades)} cascades, "
                  f"{len(all_ticks):,} ticks")

            # Test each TP/SL config
            for tp, sl, cfg_label in configs:
                trades = simulate_cascade_trades(
                    all_cascades, all_ticks,
                    tp_bps=tp, sl_bps=sl,
                    timeout_ms=3600000,  # 1h timeout
                    entry_offset_bps=5,
                    fill_timeout_ms=300000,  # 5 min to fill
                )
                analyze(trades, f"{cp_label} {cfg_label}")

            # Also test with longer timeout
            trades = simulate_cascade_trades(
                all_cascades, all_ticks,
                tp_bps=50, sl_bps=100,
                timeout_ms=7200000,  # 2h timeout
                entry_offset_bps=5,
            )
            analyze(trades, f"{cp_label} TP=50 SL=100 timeout=2h")

            del all_ticks, all_ticks_list
            gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
