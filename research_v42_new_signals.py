#!/usr/bin/env python3
"""
v42: New Signal Research — Multiple Experiments

EXP A: Spot-Futures Basis Mean-Reversion
  - Compute tick-level basis (futures - spot) / spot
  - Test if extreme basis predicts reversion
  - If basis > P90 → short futures (expect basis to shrink)
  - If basis < P10 → long futures (expect basis to widen)

EXP B: Cascade Size Filtering
  - Do LARGE cascades (P99) revert more reliably than P95?
  - Does cascade notional predict edge magnitude?

EXP C: OI Divergence
  - When OI rises but price flat → breakout coming?
  - When OI drops but price flat → range-bound?

EXP D: Funding Rate Pre-Settlement
  - Predict FR direction, position 1h before 8h settlement

EXP E: Intraday Seasonality
  - Hour-of-day return patterns

Start with SOLUSDT, 7 days only. Expand if promising.
"""

import sys
import time
import json
import gzip
import os
import gc
import psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE = 0.0002
TAKER_FEE = 0.00055
OUT_FILE = 'results/v42_new_signals.txt'


def ram_str():
    p = psutil.Process().memory_info().rss / 1024**3
    a = psutil.virtual_memory().available / 1024**3
    return f"RAM={p:.1f}GB used, {a:.1f}GB avail"


class Tee:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', buffering=1)
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_futures_trades(symbol, dates, data_dir='data'):
    """Load futures tick trades for specific dates."""
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time()
    print(f"  Loading futures trades for {len(dates)} days...", end='', flush=True)
    dfs = []
    for i, date_str in enumerate(dates):
        f = base / f"{symbol}{date_str}.csv.gz"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=['timestamp', 'side', 'size', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        dfs.append(df)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f" [{i+1}/{len(dates)} {elapsed:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not dfs:
        print(" NO DATA")
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    elapsed = time.time() - t0
    print(f" done ({len(result):,} trades, {elapsed:.0f}s) [{ram_str()}]")
    return result


def load_spot_trades(symbol, dates, data_dir='data'):
    """Load spot tick trades for specific dates."""
    base = Path(data_dir) / symbol / "bybit" / "spot"
    t0 = time.time()
    print(f"  Loading spot trades for {len(dates)} days...", end='', flush=True)
    dfs = []
    for i, date_str in enumerate(dates):
        f = base / f"{symbol}_{date_str}.csv.gz"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        # Spot format: id, timestamp, price, volume, side, rpi
        if 'timestamp' in df.columns and 'price' in df.columns:
            # timestamp is in ms
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'price', 'volume', 'side']].copy()
            df.rename(columns={'volume': 'size'}, inplace=True)
        dfs.append(df)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f" [{i+1}/{len(dates)} {elapsed:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not dfs:
        print(" NO DATA")
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    elapsed = time.time() - t0
    print(f" done ({len(result):,} trades, {elapsed:.0f}s) [{ram_str()}]")
    return result


def load_ticker(symbol, dates, data_dir='data'):
    """Load ticker data (OI, FR, price) for specific dates."""
    base = Path(data_dir) / symbol / "bybit" / "ticker"
    t0 = time.time()
    print(f"  Loading ticker for {len(dates)} days...", end='', flush=True)
    records = []
    for i, date_str in enumerate(dates):
        # Ticker files are hourly
        for hr in range(24):
            f = base / f"ticker_{date_str}_hr{hr:02d}.jsonl.gz"
            if not f.exists():
                continue
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        ts = data.get('ts')
                        r = data.get('result', {})
                        d = r.get('data', {})
                        rec = {'timestamp': pd.to_datetime(ts, unit='ms')}
                        if 'lastPrice' in d:
                            rec['last_price'] = float(d['lastPrice'])
                        if 'openInterest' in d:
                            rec['oi'] = float(d['openInterest'])
                        if 'fundingRate' in d:
                            rec['fr'] = float(d['fundingRate'])
                        if 'nextFundingTime' in d:
                            rec['next_funding_ms'] = int(d['nextFundingTime'])
                        if len(rec) > 1:
                            records.append(rec)
                    except Exception:
                        continue
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f" [{i+1}/{len(dates)} {elapsed:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    elapsed = time.time() - t0
    if not records:
        print(" NO DATA")
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    print(f" done ({len(df):,} records, {elapsed:.0f}s) [{ram_str()}]")
    return df


def load_liquidations_dates(symbol, dates, data_dir='data'):
    """Load liquidation data for specific dates."""
    base = Path(data_dir) / symbol / "bybit" / "liquidations"
    t0 = time.time()
    print(f"  Loading liquidations for {len(dates)} days...", end='', flush=True)
    records = []
    for i, date_str in enumerate(dates):
        for hr in range(24):
            f = base / f"liquidation_{date_str}_hr{hr:02d}.jsonl.gz"
            if not f.exists():
                continue
            with gzip.open(f, 'rt') as fh:
                for line in fh:
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
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f" [{i+1}/{len(dates)} {elapsed:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    elapsed = time.time() - t0
    if not records:
        print(" NO DATA")
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    print(f" done ({len(df):,} records, {elapsed:.0f}s) [{ram_str()}]")
    return df


def get_date_range(start, n_days):
    """Generate list of date strings."""
    base = datetime.strptime(start, '%Y-%m-%d')
    return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days)]


# ============================================================================
# EXP A: SPOT-FUTURES BASIS MEAN-REVERSION
# ============================================================================

def exp_a_basis(symbol, dates):
    """
    Compute tick-level basis = (futures_price - spot_price) / spot_price
    Resample to 1-second bars, compute rolling z-score of basis.
    Test: when basis z > 2 → short futures (expect basis to shrink)
          when basis z < -2 → long futures (expect basis to widen)
    """
    print(f"\n{'='*80}")
    print(f"  EXP A: SPOT-FUTURES BASIS MEAN-REVERSION — {symbol}")
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"{'='*80}")

    fut = load_futures_trades(symbol, dates)
    spot = load_spot_trades(symbol, dates)

    if fut.empty or spot.empty:
        print("  ✗ Missing data, skipping")
        return None

    # Resample both to 1-second VWAP
    print("  Computing 1s VWAP bars...", end='', flush=True)
    fut['notional'] = fut['price'] * fut['size']
    fut_1s = fut.set_index('timestamp').resample('1s').agg({
        'notional': 'sum', 'size': 'sum', 'price': 'last'
    })
    fut_1s['vwap'] = fut_1s['notional'] / fut_1s['size'].replace(0, np.nan)
    fut_1s['vwap'] = fut_1s['vwap'].ffill()
    # Use last price where vwap is nan
    fut_1s['vwap'] = fut_1s['vwap'].fillna(fut_1s['price'].ffill())

    spot['notional_s'] = spot['price'] * spot['size']
    spot_1s = spot.set_index('timestamp').resample('1s').agg({
        'notional_s': 'sum', 'size': 'sum', 'price': 'last'
    })
    spot_1s['vwap'] = spot_1s['notional_s'] / spot_1s['size'].replace(0, np.nan)
    spot_1s['vwap'] = spot_1s['vwap'].ffill()
    spot_1s['vwap'] = spot_1s['vwap'].fillna(spot_1s['price'].ffill())
    print(f" done [{ram_str()}]")

    # Free raw data
    del fut, spot; gc.collect()

    # Compute basis
    print("  Computing basis...", end='', flush=True)
    basis = pd.DataFrame({
        'fut_price': fut_1s['vwap'],
        'spot_price': spot_1s['vwap'],
    }).dropna()
    basis['basis_bps'] = (basis['fut_price'] - basis['spot_price']) / basis['spot_price'] * 10000
    print(f" {len(basis):,} seconds")

    del fut_1s, spot_1s; gc.collect()

    # Rolling stats
    print("  Computing rolling z-scores...", end='', flush=True)
    window = 300  # 5-minute rolling window
    basis['basis_ma'] = basis['basis_bps'].rolling(window, min_periods=60).mean()
    basis['basis_std'] = basis['basis_bps'].rolling(window, min_periods=60).std()
    basis['basis_z'] = (basis['basis_bps'] - basis['basis_ma']) / basis['basis_std'].replace(0, np.nan)
    basis = basis.dropna(subset=['basis_z'])
    print(f" done ({len(basis):,} rows) [{ram_str()}]")

    # Basic stats
    print(f"\n  BASIS STATISTICS:")
    print(f"    Mean:   {basis['basis_bps'].mean():+.2f} bps")
    print(f"    Std:    {basis['basis_bps'].std():.2f} bps")
    print(f"    Min:    {basis['basis_bps'].min():+.2f} bps")
    print(f"    Max:    {basis['basis_bps'].max():+.2f} bps")
    print(f"    P5:     {basis['basis_bps'].quantile(0.05):+.2f} bps")
    print(f"    P95:    {basis['basis_bps'].quantile(0.95):+.2f} bps")

    # Forward returns on futures
    print("  Computing forward returns...", end='', flush=True)
    for horizon in [10, 30, 60, 300]:
        basis[f'fwd_{horizon}s'] = basis['fut_price'].shift(-horizon) / basis['fut_price'] - 1
    print(" done")

    # Test: does extreme basis z predict forward returns?
    print(f"\n  BASIS Z-SCORE → FORWARD RETURN (futures):")
    print(f"  {'Z bucket':15s}  {'Count':>7s}  {'fwd_10s':>10s}  {'fwd_30s':>10s}  {'fwd_60s':>10s}  {'fwd_300s':>10s}")
    print(f"  {'-'*70}")

    z_bins = [(-999, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 999)]
    z_labels = ['z<-3', '-3<z<-2', '-2<z<-1', '-1<z<0', '0<z<1', '1<z<2', '2<z<3', 'z>3']

    results_a = []
    for (lo, hi), label in zip(z_bins, z_labels):
        mask = (basis['basis_z'] >= lo) & (basis['basis_z'] < hi)
        sub = basis[mask]
        if len(sub) < 10:
            continue
        row = {'label': label, 'count': len(sub)}
        for h in [10, 30, 60, 300]:
            col = f'fwd_{h}s'
            row[col] = sub[col].mean() * 10000  # in bps
        results_a.append(row)
        print(f"  {label:15s}  {len(sub):>7,d}  {row['fwd_10s']:>+9.2f}  {row['fwd_30s']:>+9.2f}  "
              f"{row['fwd_60s']:>+9.2f}  {row['fwd_300s']:>+9.2f}")

    # Monotonicity check: does higher z predict lower future returns?
    if len(results_a) >= 4:
        fwd_60_vals = [r['fwd_60s'] for r in results_a]
        # Check if there's a negative relationship (high z → low return)
        if fwd_60_vals[0] > fwd_60_vals[-1]:
            spread = fwd_60_vals[0] - fwd_60_vals[-1]
            print(f"\n  ✅ MONOTONIC: extreme low z outperforms extreme high z by {spread:.2f} bps at 60s")
            print(f"     This suggests basis mean-reversion is REAL")
        else:
            print(f"\n  ❌ NOT MONOTONIC at 60s horizon")

    # Strategy simulation: trade when z crosses threshold
    print(f"\n  STRATEGY SIMULATION:")
    for z_thresh in [1.5, 2.0, 2.5, 3.0]:
        for hold_sec in [30, 60, 120, 300]:
            # Long futures when basis z < -thresh (basis too negative, expect it to widen)
            # Short futures when basis z > +thresh (basis too positive, expect it to shrink)
            long_mask = basis['basis_z'] < -z_thresh
            short_mask = basis['basis_z'] > z_thresh

            fwd_col = f'fwd_{hold_sec}s' if hold_sec in [10, 30, 60, 300] else None
            if fwd_col is None or fwd_col not in basis.columns:
                continue

            long_rets = basis.loc[long_mask, fwd_col].dropna()
            short_rets = -basis.loc[short_mask, fwd_col].dropna()  # negative because shorting

            all_rets = pd.concat([long_rets, short_rets])
            if len(all_rets) < 20:
                continue

            # Apply cooldown: only take 1 trade per hold_sec seconds
            # Simple: subsample every hold_sec rows
            all_rets_sampled = all_rets.iloc[::hold_sec]
            n = len(all_rets_sampled)
            if n < 10:
                continue

            avg_ret = all_rets_sampled.mean()
            wr = (all_rets_sampled > 0).mean() * 100
            total = all_rets_sampled.sum()
            std = all_rets_sampled.std()
            sharpe = avg_ret / (std + 1e-10) * np.sqrt(252 * 24 * 3600 / hold_sec)

            # Net of fees (maker entry + taker exit)
            fee = MAKER_FEE + TAKER_FEE
            avg_net = avg_ret - fee
            total_net = total - fee * n

            flag = "✅" if avg_net > 0 else "  "
            print(f"  {flag} z>{z_thresh:.1f} hold={hold_sec:3d}s  n={n:5d}  wr={wr:5.1f}%  "
                  f"gross={avg_ret*10000:+6.2f}bps  net={avg_net*10000:+6.2f}bps  "
                  f"total_net={total_net*100:+7.2f}%  sharpe={sharpe:+7.1f}")

    del basis; gc.collect()
    return results_a


# ============================================================================
# EXP B: CASCADE SIZE FILTERING
# ============================================================================

def exp_b_cascade_size(symbol, dates):
    """
    Test if larger cascades produce bigger mean-reversion.
    Compare P90, P95, P99 cascade thresholds.
    """
    print(f"\n{'='*80}")
    print(f"  EXP B: CASCADE SIZE FILTERING — {symbol}")
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"{'='*80}")

    liq_df = load_liquidations_dates(symbol, dates)
    fut = load_futures_trades(symbol, dates)

    if liq_df.empty or fut.empty:
        print("  ✗ Missing data, skipping")
        return None

    # Build 1-min bars from futures
    print("  Building 1-min bars...", end='', flush=True)
    fut_bars = fut.set_index('timestamp')['price'].resample('1min').agg(['first', 'max', 'min', 'last']).dropna()
    fut_bars.columns = ['open', 'high', 'low', 'close']
    print(f" {len(fut_bars):,} bars")

    del fut; gc.collect()

    # Test different cascade thresholds
    print(f"\n  CASCADE THRESHOLD COMPARISON:")
    print(f"  {'Threshold':12s}  {'Cascades':>8s}  {'Fills':>6s}  {'WR%':>6s}  {'AvgNet':>8s}  {'TotNet':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*70}")

    for pct in [90, 95, 97, 99]:
        vol_thresh = liq_df['notional'].quantile(pct / 100)
        large = liq_df[liq_df['notional'] >= vol_thresh]

        # Detect cascades
        cascades = []
        current = []
        for _, row in large.iterrows():
            if not current:
                current = [row]
            else:
                dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
                if dt <= 60:
                    current.append(row)
                else:
                    if len(current) >= 2:
                        cdf = pd.DataFrame(current)
                        buy_not = cdf[cdf['side'] == 'Buy']['notional'].sum()
                        sell_not = cdf[cdf['side'] == 'Sell']['notional'].sum()
                        cascades.append({
                            'end': cdf['timestamp'].max(),
                            'total_notional': buy_not + sell_not,
                            'buy_dominant': buy_not > sell_not,
                        })
                    current = [row]
            if len(current) >= 2:
                cdf_last = pd.DataFrame(current)
                # Don't finalize yet, wait for more
                pass

        # Run strategy with best known config
        trades = []
        last_time = None
        for c in cascades:
            if last_time and (c['end'] - last_time).total_seconds() < 300:
                continue
            idx = fut_bars.index.searchsorted(c['end'])
            if idx >= len(fut_bars) - 30 or idx < 1:
                continue
            price = fut_bars.iloc[idx]['close']
            is_long = c['buy_dominant']

            if is_long:
                limit = price * 0.998
                tp = limit * 1.002
                sl = limit * 0.995
            else:
                limit = price * 1.002
                tp = limit * 0.998
                sl = limit * 1.005

            # Fill check
            filled = False
            for j in range(idx, min(idx + 30, len(fut_bars))):
                bar = fut_bars.iloc[j]
                if is_long and bar['low'] <= limit:
                    filled = True
                    fill_idx = j
                    break
                elif not is_long and bar['high'] >= limit:
                    filled = True
                    fill_idx = j
                    break
            if not filled:
                continue

            # Exit
            exit_price = None
            exit_reason = 'timeout'
            for k in range(fill_idx, min(fill_idx + 30, len(fut_bars))):
                bar = fut_bars.iloc[k]
                if is_long:
                    if bar['low'] <= sl:
                        exit_price = sl; exit_reason = 'sl'; break
                    if bar['high'] >= tp:
                        exit_price = tp; exit_reason = 'tp'; break
                else:
                    if bar['high'] >= sl:
                        exit_price = sl; exit_reason = 'sl'; break
                    if bar['low'] <= tp:
                        exit_price = tp; exit_reason = 'tp'; break
            if exit_price is None:
                exit_price = fut_bars.iloc[min(fill_idx + 30, len(fut_bars) - 1)]['close']

            if is_long:
                gross = (exit_price - limit) / limit
            else:
                gross = (limit - exit_price) / limit

            fee = MAKER_FEE + (MAKER_FEE if exit_reason == 'tp' else TAKER_FEE)
            net = gross - fee
            trades.append(net)
            last_time = c['end']

        if len(trades) >= 5:
            trades_arr = np.array(trades)
            n = len(trades_arr)
            wr = (trades_arr > 0).mean() * 100
            avg = trades_arr.mean()
            tot = trades_arr.sum()
            std = trades_arr.std()
            sharpe = avg / (std + 1e-10) * np.sqrt(252 * 24 * 60)
            flag = "✅" if avg > 0 else "  "
            print(f"  {flag} P{pct:2d} (>{vol_thresh:,.0f})  {len(cascades):>8d}  {n:>6d}  {wr:>5.1f}%  "
                  f"{avg*10000:>+7.2f}bps  {tot*100:>+7.2f}%  {sharpe:>+7.1f}")
        else:
            print(f"     P{pct:2d} (>{vol_thresh:,.0f})  {len(cascades):>8d}  <5 fills")

    # Also test: does cascade NOTIONAL predict edge?
    print(f"\n  CASCADE NOTIONAL → EDGE (P95 threshold):")
    vol_thresh = liq_df['notional'].quantile(0.95)
    large = liq_df[liq_df['notional'] >= vol_thresh]
    cascades_full = []
    current = []
    for _, row in large.iterrows():
        if not current:
            current = [row]
        else:
            dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
            if dt <= 60:
                current.append(row)
            else:
                if len(current) >= 2:
                    cdf = pd.DataFrame(current)
                    buy_not = cdf[cdf['side'] == 'Buy']['notional'].sum()
                    sell_not = cdf[cdf['side'] == 'Sell']['notional'].sum()
                    cascades_full.append({
                        'end': cdf['timestamp'].max(),
                        'total_notional': buy_not + sell_not,
                        'buy_dominant': buy_not > sell_not,
                        'n_events': len(cdf),
                    })
                current = [row]

    if cascades_full:
        notionals = [c['total_notional'] for c in cascades_full]
        median_not = np.median(notionals)
        print(f"  Median cascade notional: ${median_not:,.0f}")
        print(f"  Testing: above-median vs below-median cascades")

        for label, filter_fn in [("SMALL (<median)", lambda c: c['total_notional'] < median_not),
                                  ("LARGE (>median)", lambda c: c['total_notional'] >= median_not)]:
            filtered = [c for c in cascades_full if filter_fn(c)]
            trades = []
            last_time = None
            for c in filtered:
                if last_time and (c['end'] - last_time).total_seconds() < 300:
                    continue
                idx = fut_bars.index.searchsorted(c['end'])
                if idx >= len(fut_bars) - 30 or idx < 1:
                    continue
                price = fut_bars.iloc[idx]['close']
                is_long = c['buy_dominant']
                if is_long:
                    limit = price * 0.998; tp = limit * 1.002; sl = limit * 0.995
                else:
                    limit = price * 1.002; tp = limit * 0.998; sl = limit * 1.005
                filled = False
                for j in range(idx, min(idx + 30, len(fut_bars))):
                    bar = fut_bars.iloc[j]
                    if is_long and bar['low'] <= limit:
                        filled = True; fill_idx = j; break
                    elif not is_long and bar['high'] >= limit:
                        filled = True; fill_idx = j; break
                if not filled:
                    continue
                exit_price = None; exit_reason = 'timeout'
                for k in range(fill_idx, min(fill_idx + 30, len(fut_bars))):
                    bar = fut_bars.iloc[k]
                    if is_long:
                        if bar['low'] <= sl: exit_price = sl; exit_reason = 'sl'; break
                        if bar['high'] >= tp: exit_price = tp; exit_reason = 'tp'; break
                    else:
                        if bar['high'] >= sl: exit_price = sl; exit_reason = 'sl'; break
                        if bar['low'] <= tp: exit_price = tp; exit_reason = 'tp'; break
                if exit_price is None:
                    exit_price = fut_bars.iloc[min(fill_idx + 30, len(fut_bars) - 1)]['close']
                if is_long: gross = (exit_price - limit) / limit
                else: gross = (limit - exit_price) / limit
                fee = MAKER_FEE + (MAKER_FEE if exit_reason == 'tp' else TAKER_FEE)
                trades.append(gross - fee)
                last_time = c['end']

            if len(trades) >= 5:
                arr = np.array(trades)
                print(f"    {label:20s}  n={len(arr):4d}  wr={((arr>0).mean()*100):5.1f}%  "
                      f"avg={arr.mean()*10000:+6.2f}bps  tot={arr.sum()*100:+6.2f}%")
            else:
                print(f"    {label:20s}  <5 trades")

    del liq_df, fut_bars; gc.collect()


# ============================================================================
# EXP C: OI DIVERGENCE
# ============================================================================

def exp_c_oi_divergence(symbol, dates):
    """
    When OI rises but price is flat → breakout coming?
    When OI drops but price is flat → range-bound?
    """
    print(f"\n{'='*80}")
    print(f"  EXP C: OI DIVERGENCE SIGNAL — {symbol}")
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"{'='*80}")

    ticker = load_ticker(symbol, dates)
    if ticker.empty or 'oi' not in ticker.columns:
        print("  ✗ Missing OI data, skipping")
        return None

    # Forward-fill OI and price to 1-minute bars
    print("  Building 1-min OI/price bars...", end='', flush=True)
    ticker_ts = ticker.set_index('timestamp')
    oi_1m = ticker_ts['oi'].resample('1min').last().ffill()
    price_1m = ticker_ts['last_price'].resample('1min').last().ffill()
    df = pd.DataFrame({'oi': oi_1m, 'price': price_1m}).dropna()
    print(f" {len(df):,} bars")

    if len(df) < 120:
        print("  ✗ Not enough data")
        return None

    # Compute 60-min changes
    df['oi_chg_60m'] = df['oi'].pct_change(60) * 100  # % change
    df['price_chg_60m'] = df['price'].pct_change(60) * 100
    df['price_vol_60m'] = df['price'].pct_change().rolling(60).std() * 100  # realized vol

    # Forward returns
    for h in [60, 120, 240]:
        df[f'fwd_ret_{h}m'] = df['price'].shift(-h) / df['price'] - 1
        df[f'fwd_vol_{h}m'] = df['price'].pct_change().shift(-1).rolling(h).std().shift(-h) * 100

    df = df.dropna()

    # Define regimes
    oi_thresh = df['oi_chg_60m'].quantile(0.75)
    oi_low = df['oi_chg_60m'].quantile(0.25)
    price_flat = df['price_chg_60m'].abs() < df['price_chg_60m'].abs().quantile(0.5)

    regimes = {
        'OI↑ Price flat': (df['oi_chg_60m'] > oi_thresh) & price_flat,
        'OI↓ Price flat': (df['oi_chg_60m'] < oi_low) & price_flat,
        'OI↑ Price↑': (df['oi_chg_60m'] > oi_thresh) & (df['price_chg_60m'] > df['price_chg_60m'].quantile(0.75)),
        'OI↓ Price↓': (df['oi_chg_60m'] < oi_low) & (df['price_chg_60m'] < df['price_chg_60m'].quantile(0.25)),
        'Baseline': pd.Series(True, index=df.index),
    }

    print(f"\n  OI-PRICE REGIME → FORWARD RETURNS & VOLATILITY:")
    print(f"  {'Regime':20s}  {'Count':>7s}  {'fwd_60m':>10s}  {'fwd_120m':>10s}  {'fwd_240m':>10s}  {'|fwd_60m|':>10s}")
    print(f"  {'-'*80}")

    for name, mask in regimes.items():
        sub = df[mask]
        if len(sub) < 20:
            continue
        fwd60 = sub['fwd_ret_60m'].mean() * 10000
        fwd120 = sub['fwd_ret_120m'].mean() * 10000
        fwd240 = sub['fwd_ret_240m'].mean() * 10000
        abs_fwd60 = sub['fwd_ret_60m'].abs().mean() * 10000
        print(f"  {name:20s}  {len(sub):>7,d}  {fwd60:>+9.2f}  {fwd120:>+9.2f}  {fwd240:>+9.2f}  {abs_fwd60:>9.2f}")

    # Key test: does OI↑ + Price flat predict HIGHER absolute returns (breakout)?
    if 'OI↑ Price flat' in regimes:
        oi_up_flat = df[regimes['OI↑ Price flat']]
        baseline = df[regimes['Baseline']]
        if len(oi_up_flat) >= 20:
            abs_oi = oi_up_flat['fwd_ret_60m'].abs().mean()
            abs_base = baseline['fwd_ret_60m'].abs().mean()
            ratio = abs_oi / abs_base if abs_base > 0 else 0
            print(f"\n  OI↑+flat → |fwd_60m| = {abs_oi*10000:.2f} bps vs baseline {abs_base*10000:.2f} bps (ratio={ratio:.2f}x)")
            if ratio > 1.2:
                print(f"  ✅ OI BUILDUP PREDICTS LARGER MOVES (breakout signal)")
            else:
                print(f"  ❌ OI buildup does NOT predict larger moves")

    del df, ticker; gc.collect()


# ============================================================================
# EXP D: FUNDING RATE PRE-SETTLEMENT
# ============================================================================

def exp_d_funding_rate(symbol, dates):
    """
    Test: position 1h before funding settlement based on current FR.
    If FR > 0 (longs pay shorts): go short → collect funding + expect price drop
    If FR < 0 (shorts pay longs): go long → collect funding + expect price rise
    """
    print(f"\n{'='*80}")
    print(f"  EXP D: FUNDING RATE PRE-SETTLEMENT — {symbol}")
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"{'='*80}")

    ticker = load_ticker(symbol, dates)
    if ticker.empty or 'fr' not in ticker.columns:
        print("  ✗ Missing FR data, skipping")
        return None

    # Get FR and price at 1-min resolution
    print("  Building FR/price bars...", end='', flush=True)
    ticker_ts = ticker.set_index('timestamp')
    fr_1m = ticker_ts['fr'].resample('1min').last().ffill()
    price_1m = ticker_ts['last_price'].resample('1min').last().ffill()
    df = pd.DataFrame({'fr': fr_1m, 'price': price_1m}).dropna()
    print(f" {len(df):,} bars")

    if len(df) < 120:
        print("  ✗ Not enough data")
        return None

    # Funding settlements happen at 00:00, 08:00, 16:00 UTC
    settlement_hours = [0, 8, 16]

    # For each settlement, look at FR 1h before and price change around settlement
    trades = []
    for ts in df.index:
        if ts.hour in settlement_hours and ts.minute == 0:
            # Look 60 minutes before
            entry_time = ts - pd.Timedelta(minutes=60)
            if entry_time not in df.index:
                continue
            # Look 60 minutes after
            exit_time = ts + pd.Timedelta(minutes=60)
            if exit_time not in df.index:
                continue

            fr_at_entry = df.loc[entry_time:ts, 'fr'].iloc[-1] if entry_time in df.index else np.nan
            if pd.isna(fr_at_entry):
                continue

            entry_price = df.loc[entry_time, 'price']
            exit_price = df.loc[exit_time, 'price']

            # Strategy: if FR > 0, go short (longs pay, expect price pressure down)
            if fr_at_entry > 0:
                direction = 'short'
                price_ret = (entry_price - exit_price) / entry_price
            else:
                direction = 'long'
                price_ret = (exit_price - entry_price) / entry_price

            # Funding income (collected if on the right side)
            funding_income = abs(fr_at_entry)  # simplified

            gross = price_ret + funding_income
            fee = MAKER_FEE + TAKER_FEE  # entry maker, exit taker
            net = gross - fee

            trades.append({
                'time': ts,
                'fr': fr_at_entry,
                'direction': direction,
                'price_ret': price_ret,
                'funding': funding_income,
                'gross': gross,
                'net': net,
            })

    if not trades:
        print("  ✗ No settlement events found")
        return None

    tdf = pd.DataFrame(trades)
    print(f"\n  FUNDING SETTLEMENT TRADES: {len(tdf)}")
    print(f"    Avg FR:        {tdf['fr'].mean()*100:.4f}%")
    print(f"    Avg price_ret: {tdf['price_ret'].mean()*10000:+.2f} bps")
    print(f"    Avg funding:   {tdf['funding'].mean()*10000:+.2f} bps")
    print(f"    Avg gross:     {tdf['gross'].mean()*10000:+.2f} bps")
    print(f"    Avg net:       {tdf['net'].mean()*10000:+.2f} bps")
    print(f"    Win rate:      {(tdf['net'] > 0).mean()*100:.1f}%")
    print(f"    Total net:     {tdf['net'].sum()*100:+.2f}%")

    if tdf['net'].mean() > 0:
        print(f"    ✅ FUNDING RATE STRATEGY IS PROFITABLE")
    else:
        print(f"    ❌ Funding rate strategy is NOT profitable")

    # Test with FR magnitude filter
    print(f"\n  FR MAGNITUDE FILTER:")
    for fr_thresh in [0.0001, 0.0003, 0.0005, 0.001]:
        filtered = tdf[tdf['fr'].abs() > fr_thresh]
        if len(filtered) < 5:
            continue
        avg_net = filtered['net'].mean()
        wr = (filtered['net'] > 0).mean() * 100
        tot = filtered['net'].sum()
        flag = "✅" if avg_net > 0 else "  "
        print(f"  {flag} |FR|>{fr_thresh*100:.2f}%  n={len(filtered):3d}  wr={wr:5.1f}%  "
              f"avg_net={avg_net*10000:+6.2f}bps  total={tot*100:+6.2f}%")

    del df, ticker; gc.collect()


# ============================================================================
# EXP E: INTRADAY SEASONALITY
# ============================================================================

def exp_e_seasonality(symbol, dates):
    """
    Test hour-of-day return patterns.
    """
    print(f"\n{'='*80}")
    print(f"  EXP E: INTRADAY SEASONALITY — {symbol}")
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"{'='*80}")

    fut = load_futures_trades(symbol, dates)
    if fut.empty:
        print("  ✗ Missing data, skipping")
        return None

    # Build 1-min bars
    print("  Building 1-min bars...", end='', flush=True)
    bars = fut.set_index('timestamp')['price'].resample('1min').last().ffill()
    bars = bars.to_frame('price').dropna()
    print(f" {len(bars):,} bars")

    del fut; gc.collect()

    # Hourly returns
    bars['ret_1h'] = bars['price'].pct_change(60)
    bars['hour'] = bars.index.hour
    bars['abs_ret_1h'] = bars['ret_1h'].abs()

    print(f"\n  HOUR-OF-DAY RETURNS:")
    print(f"  {'Hour':>4s}  {'Count':>7s}  {'MeanRet':>10s}  {'|MeanRet|':>10s}  {'StdRet':>10s}  {'Sharpe':>8s}  {'WR%':>6s}")
    print(f"  {'-'*65}")

    hourly_stats = []
    for hr in range(24):
        sub = bars[bars['hour'] == hr].dropna(subset=['ret_1h'])
        if len(sub) < 10:
            continue
        avg = sub['ret_1h'].mean()
        abs_avg = sub['abs_ret_1h'].mean()
        std = sub['ret_1h'].std()
        sharpe = avg / (std + 1e-10) * np.sqrt(365)
        wr = (sub['ret_1h'] > 0).mean() * 100
        hourly_stats.append({'hour': hr, 'avg': avg, 'abs_avg': abs_avg, 'std': std, 'sharpe': sharpe, 'wr': wr, 'n': len(sub)})
        flag = "✅" if abs(sharpe) > 0.5 else "  "
        print(f"  {flag} {hr:02d}:00  {len(sub):>7,d}  {avg*10000:>+9.2f}  {abs_avg*10000:>9.2f}  "
              f"{std*10000:>9.2f}  {sharpe:>+7.2f}  {wr:>5.1f}%")

    # Best/worst hours
    if hourly_stats:
        best = max(hourly_stats, key=lambda x: x['sharpe'])
        worst = min(hourly_stats, key=lambda x: x['sharpe'])
        print(f"\n  Best hour:  {best['hour']:02d}:00 (Sharpe={best['sharpe']:+.2f}, avg={best['avg']*10000:+.2f}bps)")
        print(f"  Worst hour: {worst['hour']:02d}:00 (Sharpe={worst['sharpe']:+.2f}, avg={worst['avg']*10000:+.2f}bps)")

        spread = best['avg'] - worst['avg']
        print(f"  Best-Worst spread: {spread*10000:.2f} bps/hour")
        if abs(spread) > 1:
            print(f"  ✅ SIGNIFICANT INTRADAY SEASONALITY")
        else:
            print(f"  ❌ No significant intraday seasonality")

    del bars; gc.collect()


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs('results', exist_ok=True)
    tee = Tee(OUT_FILE)
    sys.stdout = tee

    t_global = time.time()

    print("=" * 80)
    print(f"  v42: NEW SIGNAL RESEARCH — 5 EXPERIMENTS")
    print(f"  Symbol: SOLUSDT (start small)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  [{ram_str()}]")
    print("=" * 80)

    symbol = 'SOLUSDT'

    # Phase 1: 7 days only (quick iteration)
    dates_7d = get_date_range('2025-05-12', 7)
    print(f"\n  Phase 1: Quick test on 7 days ({dates_7d[0]} to {dates_7d[-1]})")

    exp_a_basis(symbol, dates_7d)
    gc.collect()
    print(f"\n  [{ram_str()}] after EXP A")

    exp_b_cascade_size(symbol, dates_7d)
    gc.collect()
    print(f"\n  [{ram_str()}] after EXP B")

    # For EXP C, D, E we need ticker data which starts 2025-05-11
    dates_14d = get_date_range('2025-05-12', 14)

    exp_c_oi_divergence(symbol, dates_14d)
    gc.collect()
    print(f"\n  [{ram_str()}] after EXP C")

    exp_d_funding_rate(symbol, dates_14d)
    gc.collect()
    print(f"\n  [{ram_str()}] after EXP D")

    exp_e_seasonality(symbol, dates_7d)
    gc.collect()
    print(f"\n  [{ram_str()}] after EXP E")

    elapsed = time.time() - t_global
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
