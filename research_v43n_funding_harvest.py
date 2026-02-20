#!/usr/bin/env python3
"""
v43n: Funding Rate Harvesting — Delta-Neutral Carry Trade

Concept: Collect funding payments by positioning against the crowd.
When funding > threshold → go short (collect from longs)
When funding < -threshold → go long (collect from shorts)

Key insight: This is NOT price prediction. The edge is the funding payment itself.
But price risk during holding can overwhelm the carry.

Strategy variants:
  A) Simple: enter before funding, exit after funding (hold ~1h around settlement)
  B) Extended: hold through multiple funding periods (8h-24h)
  C) Threshold: only enter when |funding| > X bps (higher carry)
  D) Hedged: long spot + short futures when funding positive (true delta-neutral)
     — requires spot data, tested separately

Data: Ticker parquet (5-sec, 76 days) with funding_rate and next_funding_time
Fees: 4 bps RT (maker+maker) for entry/exit
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_BPS = 2.0
PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_ticker_data(symbol):
    """Load all ticker parquet data."""
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    files = sorted(ticker_dir.glob('*.parquet'))
    if not files:
        return pd.DataFrame()
    print(f"  Loading {len(files)} ticker files...", end='', flush=True)
    t0 = time.time()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp_us'], unit='us')
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    print(f" {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def identify_funding_settlements(df):
    """
    Identify funding settlement times from next_funding_time changes.
    When next_funding_time jumps forward, a settlement just happened.
    """
    nft = df['next_funding_time'].values
    settlements = []

    for i in range(1, len(nft)):
        if nft[i] > nft[i-1]:
            # Settlement happened between i-1 and i
            settlements.append({
                'idx': i,
                'time': df.index[i],
                'funding_rate': df['funding_rate'].iloc[i-1],  # rate that was just settled
                'price': df['last_price'].iloc[i],
                'next_funding_time': nft[i],
            })

    return settlements


def simulate_funding_harvest(df, settlements, entry_offset_min=30,
                              exit_offset_min=30, funding_threshold_bps=0,
                              hold_periods=1, direction_mode='contrarian'):
    """
    Simulate funding harvesting strategy.

    entry_offset_min: enter this many minutes BEFORE settlement
    exit_offset_min: exit this many minutes AFTER settlement
    funding_threshold_bps: only trade when |funding| > threshold
    hold_periods: hold through this many funding periods (1=8h, 3=24h)
    direction_mode:
      'contrarian': go against funding (short when funding positive = collect)
      'momentum': go with funding direction
    """
    trades = []
    last_exit_time = pd.Timestamp.min
    cooldown = pd.Timedelta(hours=1)

    i = 0
    while i < len(settlements):
        s = settlements[i]
        fr = s['funding_rate']
        fr_bps = fr * 10000

        # Threshold filter
        if abs(fr_bps) < funding_threshold_bps:
            i += 1
            continue

        # Cooldown
        if s['time'] < last_exit_time + cooldown:
            i += 1
            continue

        # Direction
        if direction_mode == 'contrarian':
            trade_dir = 'short' if fr > 0 else 'long'
        else:
            trade_dir = 'long' if fr > 0 else 'short'

        # Entry time: offset before settlement
        entry_time = s['time'] - pd.Timedelta(minutes=entry_offset_min)

        # Exit time: after hold_periods settlements
        exit_settlement_idx = i + hold_periods - 1
        if exit_settlement_idx >= len(settlements):
            break
        exit_settlement = settlements[exit_settlement_idx]
        exit_time = exit_settlement['time'] + pd.Timedelta(minutes=exit_offset_min)

        # Get entry and exit prices
        entry_pos = df.index.searchsorted(entry_time)
        if entry_pos >= len(df):
            i += 1
            continue
        entry_price = df['last_price'].iloc[entry_pos]

        exit_pos = df.index.searchsorted(exit_time)
        if exit_pos >= len(df):
            i += 1
            continue
        exit_price = df['last_price'].iloc[exit_pos]

        # Calculate funding collected
        total_funding_bps = 0
        for j in range(i, min(i + hold_periods, len(settlements))):
            collected_fr = settlements[j]['funding_rate']
            if trade_dir == 'short':
                total_funding_bps += collected_fr * 10000  # shorts collect when positive
            else:
                total_funding_bps -= collected_fr * 10000  # longs collect when negative

        # Price PnL
        if trade_dir == 'long':
            price_pnl_bps = (exit_price - entry_price) / entry_price * 10000
        else:
            price_pnl_bps = (entry_price - exit_price) / entry_price * 10000

        # Fees (maker entry + maker exit)
        fee_bps = MAKER_FEE_BPS * 2

        net_bps = price_pnl_bps + total_funding_bps - fee_bps

        trades.append({
            'time': s['time'],
            'dir': trade_dir,
            'funding_bps': total_funding_bps,
            'price_pnl_bps': price_pnl_bps,
            'fee_bps': fee_bps,
            'net_bps': net_bps,
            'hold_periods': hold_periods,
            'entry_fr_bps': fr_bps,
        })

        last_exit_time = exit_time
        i += hold_periods  # skip past held periods

    return trades


def analyze(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES"); return None
    net = np.array([t['net_bps'] for t in trades])
    funding = np.array([t['funding_bps'] for t in trades])
    price_pnl = np.array([t['price_pnl_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 3) if std > 0 else 0  # 3 trades/day

    print(f"  {label}")
    print(f"    n={n:4d} WR={wr:5.1f}% avg={avg:+7.1f}bps total={total:+7.2f}% Sharpe={sharpe:+5.2f}")
    print(f"    Funding: avg={funding.mean():+.2f}bps total={funding.sum()/100:+.2f}%")
    print(f"    Price:   avg={price_pnl.mean():+.1f}bps total={price_pnl.sum()/100:+.2f}%")
    print(f"    Fees:    avg={trades[0]['fee_bps']:.1f}bps total={sum(t['fee_bps'] for t in trades)/100:.2f}%")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe,
            'funding_avg': funding.mean(), 'price_avg': price_pnl.mean()}


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43n: Funding Rate Harvesting — Carry Trade")
    print("=" * 80)

    for symbol in ['SOLUSDT', 'ETHUSDT', 'BTCUSDT']:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        df = load_ticker_data(symbol)
        if df.empty:
            continue

        # Identify settlements
        settlements = identify_funding_settlements(df)
        print(f"  {len(settlements)} funding settlements detected")

        if len(settlements) < 20:
            print(f"  Too few settlements"); continue

        # Funding rate distribution
        frs = [s['funding_rate'] * 10000 for s in settlements]
        print(f"  Funding rate: mean={np.mean(frs):+.2f}bps "
              f"std={np.std(frs):.2f}bps "
              f"min={np.min(frs):+.2f}bps max={np.max(frs):+.2f}bps")
        print(f"  Positive: {sum(1 for f in frs if f > 0)}/{len(frs)} "
              f"({sum(1 for f in frs if f > 0)/len(frs)*100:.0f}%)")

        # IS/OOS split
        split = int(len(settlements) * 0.65)
        is_settlements = settlements[:split]
        oos_settlements = settlements[split:]
        print(f"  IS: {len(is_settlements)} settlements | OOS: {len(oos_settlements)} settlements")

        # ============================================================
        # TEST MATRIX
        # ============================================================

        test_configs = [
            # (entry_offset, exit_offset, threshold, hold_periods, direction, label)
            (30, 30, 0, 1, 'contrarian', 'Contrarian 1-period no-thresh'),
            (30, 30, 0.5, 1, 'contrarian', 'Contrarian 1-period >0.5bps'),
            (30, 30, 1.0, 1, 'contrarian', 'Contrarian 1-period >1.0bps'),
            (30, 30, 0, 3, 'contrarian', 'Contrarian 3-period (24h)'),
            (30, 30, 0.5, 3, 'contrarian', 'Contrarian 3-period >0.5bps'),
            (60, 60, 0, 1, 'contrarian', 'Contrarian 1-period wide entry'),
            (5, 5, 0, 1, 'contrarian', 'Contrarian 1-period tight entry'),
            (30, 30, 0, 1, 'momentum', 'Momentum 1-period no-thresh'),
            (30, 30, 0.5, 1, 'momentum', 'Momentum 1-period >0.5bps'),
        ]

        for entry_off, exit_off, thresh, hold, dir_mode, label in test_configs:
            print(f"\n  --- {label} ---")

            is_trades = simulate_funding_harvest(
                df, is_settlements, entry_off, exit_off, thresh, hold, dir_mode)
            oos_trades = simulate_funding_harvest(
                df, oos_settlements, exit_off, exit_off, thresh, hold, dir_mode)

            analyze(is_trades, f"IS")
            analyze(oos_trades, f"OOS")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
