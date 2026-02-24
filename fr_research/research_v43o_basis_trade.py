#!/usr/bin/env python3
"""
v43o: Spot-Futures Basis Trade — Delta-Neutral Carry

Concept: Exploit the premium/discount between Bybit futures and spot prices.
When futures trade at a premium → short futures + long spot (collect premium)
When futures trade at a discount → long futures + short spot (collect discount)

This is DELTA-NEUTRAL — no directional risk. Edge is structural:
the basis must converge (futures → spot at settlement).

Additionally, when basis is positive (contango), shorts collect funding.
So the trade earns: basis convergence + funding carry.

Data: Bybit spot + futures 1h OHLCV (1127 common days, Jan 2023 – Jan 2026)
Fees: 4 bps RT per leg × 2 legs = 8 bps total RT (maker+maker both sides)

Analysis:
  1. Basis distribution and dynamics over 3 years
  2. Mean-reversion of basis (does extreme basis revert?)
  3. Basis + funding combined carry
  4. Simulated basis trade with various entry/exit thresholds
  5. Cross-symbol validation (SOL, ETH, BTC)
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

PARQUET_DIR = Path('parquet')
# Fees: maker on both spot and futures, entry + exit
# Spot: maker 0.1% on Bybit spot (VIP0) — much higher than futures!
# Futures: maker 0.02%
# Total RT: spot entry 0.1% + spot exit 0.1% + futures entry 0.02% + futures exit 0.02% = 0.24%
# That's 24 bps RT — MUCH higher than futures-only strategies
SPOT_MAKER_FEE_BPS = 10.0   # 0.1% Bybit spot maker
FUTURES_MAKER_FEE_BPS = 2.0  # 0.02% Bybit futures maker
TOTAL_RT_FEE_BPS = (SPOT_MAKER_FEE_BPS + FUTURES_MAKER_FEE_BPS) * 2  # entry + exit both legs


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_1h_ohlcv(symbol, exchange):
    """Load all 1h OHLCV for a symbol/exchange."""
    d = PARQUET_DIR / symbol / 'ohlcv' / '1h' / exchange
    if not d.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    return raw[~raw.index.duplicated(keep='first')]


def compute_basis(spot, futures):
    """Compute basis = (futures - spot) / spot in bps."""
    # Align on common timestamps
    common = spot.index.intersection(futures.index)
    s = spot.loc[common, 'close'].values.astype(np.float64)
    f = futures.loc[common, 'close'].values.astype(np.float64)
    basis_bps = (f - s) / s * 10000
    return pd.Series(basis_bps, index=common, name='basis_bps')


def simulate_basis_trade(basis, entry_threshold_bps, exit_threshold_bps,
                          max_hold_bars=24*3, fee_bps=24):
    """
    Simulate basis mean-reversion trade.
    
    When basis > entry_threshold → short futures + long spot (expect basis to shrink)
    When basis < -entry_threshold → long futures + short spot
    Exit when |basis| < exit_threshold or timeout.
    
    PnL = basis_change (in bps) - fees
    """
    b = basis.values
    n = len(b)
    trades = []
    position = None  # None, 'short_basis', 'long_basis'
    entry_bar = 0
    entry_basis = 0

    for i in range(1, n):
        if position is not None:
            hold = i - entry_bar
            current_basis = b[i]

            # Exit conditions
            exit_reason = None
            if position == 'short_basis' and current_basis <= exit_threshold_bps:
                exit_reason = 'target'
            elif position == 'long_basis' and current_basis >= -exit_threshold_bps:
                exit_reason = 'target'
            elif hold >= max_hold_bars:
                exit_reason = 'timeout'

            if exit_reason:
                if position == 'short_basis':
                    pnl_bps = entry_basis - current_basis  # profit when basis shrinks
                else:
                    pnl_bps = current_basis - entry_basis  # profit when basis grows

                net_bps = pnl_bps - fee_bps
                trades.append({
                    'entry_time': basis.index[entry_bar],
                    'exit_time': basis.index[i],
                    'dir': position,
                    'entry_basis': entry_basis,
                    'exit_basis': current_basis,
                    'pnl_bps': pnl_bps,
                    'net_bps': net_bps,
                    'hold_bars': hold,
                    'exit_reason': exit_reason,
                })
                position = None
            continue

        # Entry conditions
        if b[i] > entry_threshold_bps:
            position = 'short_basis'
            entry_bar = i
            entry_basis = b[i]
        elif b[i] < -entry_threshold_bps:
            position = 'long_basis'
            entry_bar = i
            entry_basis = b[i]

    return trades


def analyze(trades, label):
    if not trades:
        print(f"  {label}: NO TRADES"); return None
    net = np.array([t['net_bps'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 3) if std > 0 else 0

    reasons = {}
    for t in trades: reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
    avg_hold = np.mean([t['hold_bars'] for t in trades])

    # Basis stats
    entry_bases = [abs(t['entry_basis']) for t in trades]
    pnl_bases = [t['pnl_bps'] for t in trades]

    print(f"  {label}")
    print(f"    n={n:4d} WR={wr:5.1f}% avg={avg:+7.1f}bps total={total:+7.2f}% "
          f"Sharpe={sharpe:+5.2f} avgHold={avg_hold:.0f}h exits={reasons}")
    print(f"    Avg entry |basis|={np.mean(entry_bases):.1f}bps "
          f"gross_pnl={np.mean(pnl_bases):+.1f}bps fees={trades[0].get('net_bps',0)-trades[0].get('pnl_bps',0):.0f}bps")

    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe}


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43o: Spot-Futures Basis Trade — Delta-Neutral Carry")
    print(f"Total RT fees: {TOTAL_RT_FEE_BPS:.0f} bps (spot maker 0.1% + futures maker 0.02%)")
    print("=" * 80)

    for symbol in ['SOLUSDT', 'ETHUSDT', 'BTCUSDT']:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        spot = load_1h_ohlcv(symbol, 'bybit_spot')
        futures = load_1h_ohlcv(symbol, 'bybit_futures')

        if spot.empty or futures.empty:
            print(f"  Missing data (spot={len(spot)}, futures={len(futures)})"); continue

        print(f"  Spot: {len(spot):,} bars ({spot.index[0]} to {spot.index[-1]})")
        print(f"  Futures: {len(futures):,} bars ({futures.index[0]} to {futures.index[-1]})")

        basis = compute_basis(spot, futures)
        print(f"  Basis: {len(basis):,} common bars")

        if len(basis) < 100:
            print(f"  Too few common bars"); continue

        # ============================================================
        # ANALYSIS 1: Basis distribution
        # ============================================================
        print(f"\n  --- Basis Distribution ---")
        b = basis.values
        print(f"    Mean: {b.mean():+.2f} bps")
        print(f"    Std:  {b.std():.2f} bps")
        print(f"    Min:  {b.min():+.1f} bps")
        print(f"    Max:  {b.max():+.1f} bps")
        print(f"    |basis| > 10 bps: {(np.abs(b) > 10).sum()}/{len(b)} "
              f"({(np.abs(b) > 10).sum()/len(b)*100:.1f}%)")
        print(f"    |basis| > 20 bps: {(np.abs(b) > 20).sum()}/{len(b)} "
              f"({(np.abs(b) > 20).sum()/len(b)*100:.1f}%)")
        print(f"    |basis| > 50 bps: {(np.abs(b) > 50).sum()}/{len(b)} "
              f"({(np.abs(b) > 50).sum()/len(b)*100:.1f}%)")

        # Percentiles
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"    P{p:2d}: {np.percentile(b, p):+.1f} bps")

        # ============================================================
        # ANALYSIS 2: Basis autocorrelation (does it mean-revert?)
        # ============================================================
        print(f"\n  --- Basis Autocorrelation ---")
        for lag in [1, 4, 8, 24, 48]:
            if lag < len(b):
                ac = np.corrcoef(b[:-lag], b[lag:])[0, 1]
                print(f"    Lag {lag:2d}h: AC={ac:.4f}")

        # Basis change autocorrelation (MR test)
        db = np.diff(b)
        for lag in [1, 4, 8, 24]:
            if lag < len(db):
                ac = np.corrcoef(db[:-lag], db[lag:])[0, 1]
                print(f"    Δbasis lag {lag:2d}h: AC={ac:.4f} "
                      f"{'(MR)' if ac < 0 else '(momentum)'}")

        # ============================================================
        # ANALYSIS 3: Basis trade simulation
        # ============================================================
        print(f"\n  --- Basis Trade Simulation (fees={TOTAL_RT_FEE_BPS:.0f}bps RT) ---")

        # IS/OOS split
        split = int(len(basis) * 0.65)
        basis_is = basis.iloc[:split]
        basis_oos = basis.iloc[split:]
        print(f"    IS: {len(basis_is):,} bars | OOS: {len(basis_oos):,} bars")

        configs = [
            (10, 2,  'entry>10 exit<2'),
            (15, 3,  'entry>15 exit<3'),
            (20, 5,  'entry>20 exit<5'),
            (30, 5,  'entry>30 exit<5'),
            (50, 10, 'entry>50 exit<10'),
            (10, 0,  'entry>10 exit<0 (cross zero)'),
            (20, 0,  'entry>20 exit<0'),
        ]

        for entry_t, exit_t, label in configs:
            is_trades = simulate_basis_trade(basis_is, entry_t, exit_t,
                                              fee_bps=TOTAL_RT_FEE_BPS)
            oos_trades = simulate_basis_trade(basis_oos, entry_t, exit_t,
                                               fee_bps=TOTAL_RT_FEE_BPS)
            print(f"\n    {label}:")
            analyze(is_trades, f"IS")
            analyze(oos_trades, f"OOS")

        # ============================================================
        # ANALYSIS 4: What if we only trade futures (no spot leg)?
        # ============================================================
        print(f"\n  --- Futures-Only Basis Signal (fees=4bps RT) ---")
        # Use basis as a signal for futures-only directional trade
        # When basis is high → short futures (expect convergence = price drop)
        # When basis is low → long futures (expect convergence = price rise)
        # Fees: 4 bps RT (futures maker+maker)

        futures_fee = FUTURES_MAKER_FEE_BPS * 2  # 4 bps

        for entry_t, exit_t, label in [(10, 2, 'entry>10 exit<2'),
                                         (20, 5, 'entry>20 exit<5'),
                                         (30, 5, 'entry>30 exit<5')]:
            is_trades = simulate_basis_trade(basis_is, entry_t, exit_t,
                                              fee_bps=futures_fee)
            oos_trades = simulate_basis_trade(basis_oos, entry_t, exit_t,
                                               fee_bps=futures_fee)
            print(f"\n    {label} (futures-only, 4bps RT):")
            analyze(is_trades, f"IS")
            analyze(oos_trades, f"OOS")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
