#!/usr/bin/env python3
"""
Liquidation-Based Trading Strategies (v26b)
4 strategies backtested on real liquidation + ticker data.

Strategy 1: Cascade Fade — mean reversion after large liquidation cascades
Strategy 2: Extreme Imbalance Reversal — fade >70% one-sided liquidations
Strategy 3: Liquidation Rate Spike — volatility breakout on liq rate spikes
Strategy 4: Time-of-Day Liquidation Fade — US-hours cascade fade
"""
import pandas as pd
import numpy as np
import json
import gzip
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_liquidations(symbol, data_dir='data'):
    """Load all liquidation data for a symbol."""
    symbol_dir = Path(data_dir) / symbol
    liq_files = sorted(symbol_dir.glob("liquidation_*.jsonl.gz"))
    if not liq_files:
        raise ValueError(f"No liquidation files found for {symbol}")

    print(f"  Loading {len(liq_files)} liquidation files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 50 == 0:
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
    print(f" done ({len(records):,} events)")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_prices(symbol, data_dir='data'):
    """Load 5-second ticker data and return price series."""
    symbol_dir = Path(data_dir) / symbol
    ticker_files = sorted(symbol_dir.glob("ticker_2026-*.jsonl.gz"))
    if not ticker_files:
        raise ValueError(f"No 2026 ticker files found for {symbol}")

    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 50 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    r = data['result']['list'][0]
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'price': float(r['lastPrice']),
                    })
                except Exception:
                    continue
    print(f" done ({len(records):,} ticks)")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    return df


def build_price_bars(tick_df, freq='1min'):
    """Aggregate tick data to OHLC bars."""
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    bars = bars.dropna()
    return bars


def build_liq_bars(liq_df, freq='1min'):
    """Aggregate liquidation events to bars with features."""
    df = liq_df.set_index('timestamp')
    buy = df[df['side'] == 'Buy']
    sell = df[df['side'] == 'Sell']

    idx = pd.date_range(df.index.min().floor(freq), df.index.max().ceil(freq), freq=freq)
    bars = pd.DataFrame(index=idx)

    bars['liq_count'] = df.resample(freq).size()
    bars['liq_notional'] = df['notional'].resample(freq).sum()
    bars['liq_buy_notional'] = buy['notional'].resample(freq).sum()
    bars['liq_sell_notional'] = sell['notional'].resample(freq).sum()
    bars = bars.fillna(0)

    total = bars['liq_notional'] + 1e-10
    bars['liq_imbalance'] = (bars['liq_sell_notional'] - bars['liq_buy_notional']) / total
    return bars

# ============================================================================
# STRATEGY HELPERS
# ============================================================================

def compute_trade_stats(trades, label):
    """Compute and print strategy statistics from a list of trade dicts."""
    if not trades:
        print(f"\n  {label}: NO TRADES")
        return {}

    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df['pnl_pct'] > 0).sum()
    wr = wins / n * 100
    avg = df['pnl_pct'].mean()
    med = df['pnl_pct'].median()
    tot = df['pnl_pct'].sum()
    std = df['pnl_pct'].std() if n > 1 else 0
    sharpe = avg / std * np.sqrt(252 * 24 * 60) if std > 0 else 0  # per-minute bars
    max_dd = df['pnl_pct'].cumsum().cummax() - df['pnl_pct'].cumsum()
    max_dd_val = max_dd.max()
    best = df['pnl_pct'].max()
    worst = df['pnl_pct'].min()

    # Avg hold time
    if 'hold_minutes' in df.columns:
        avg_hold = df['hold_minutes'].mean()
    else:
        avg_hold = 0

    print(f"\n  {label}:")
    print(f"    Trades:      {n}")
    print(f"    Win rate:    {wr:.1f}%")
    print(f"    Avg return:  {avg*100:.3f}%")
    print(f"    Median ret:  {med*100:.3f}%")
    print(f"    Total ret:   {tot*100:.2f}%")
    print(f"    Sharpe:      {sharpe:.2f}")
    print(f"    Best trade:  {best*100:.3f}%")
    print(f"    Worst trade: {worst*100:.3f}%")
    print(f"    Max drawdown:{max_dd_val*100:.3f}%")
    if avg_hold > 0:
        print(f"    Avg hold:    {avg_hold:.1f} min")

    # By side breakdown
    if 'direction' in df.columns:
        for d in ['long', 'short']:
            sub = df[df['direction'] == d]
            if len(sub) > 0:
                print(f"    {d.upper():5s} trades: {len(sub):4d}  wr={((sub['pnl_pct']>0).sum()/len(sub)*100):.1f}%  avg={sub['pnl_pct'].mean()*100:+.3f}%")

    return {
        'trades': n, 'win_rate': wr, 'avg_ret': avg, 'total_ret': tot,
        'sharpe': sharpe, 'max_dd': max_dd_val, 'avg_hold': avg_hold,
    }

# ============================================================================
# STRATEGY 1: CASCADE FADE
# ============================================================================

def strategy_cascade_fade(liq_df, price_bars, cascade_pct=95, cooldown_sec=30,
                          hold_minutes=15, stop_pct=0.005):
    """
    After a liquidation cascade ends, fade the move.
    Buy after buy-dominated cascade (longs got stopped → price dipped).
    Sell after sell-dominated cascade (shorts got stopped → price spiked).
    """
    print("\n  Detecting cascades for strategy...", end='', flush=True)
    vol_thresh = liq_df['notional'].quantile(cascade_pct / 100)

    large = liq_df[liq_df['notional'] >= vol_thresh].copy()
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
                    cascades.append(current)
                current = [row]
    if len(current) >= 2:
        cascades.append(current)
    print(f" {len(cascades)} cascades found")

    trades = []
    for cascade in cascades:
        cdf = pd.DataFrame(cascade)
        end_time = cdf['timestamp'].max()
        entry_time = end_time + timedelta(seconds=cooldown_sec)

        buy_not = cdf[cdf['side'] == 'Buy']['notional'].sum()
        sell_not = cdf[cdf['side'] == 'Sell']['notional'].sum()
        total_not = buy_not + sell_not

        if total_not < vol_thresh:
            continue

        # Determine direction: fade the dominant liquidation side
        if buy_not > sell_not:
            direction = 'long'   # longs got liquidated → buy the dip
        else:
            direction = 'short'  # shorts got liquidated → sell the spike

        # Find entry price
        entry_idx = price_bars.index.searchsorted(entry_time)
        if entry_idx >= len(price_bars) - hold_minutes:
            continue
        entry_price = price_bars.iloc[entry_idx]['close']

        # Find exit price (hold for N minutes)
        exit_idx = min(entry_idx + hold_minutes, len(price_bars) - 1)
        exit_price = price_bars.iloc[exit_idx]['close']

        # Check stop loss during hold period
        stopped = False
        for j in range(entry_idx, exit_idx + 1):
            bar = price_bars.iloc[j]
            if direction == 'long' and bar['low'] <= entry_price * (1 - stop_pct):
                exit_price = entry_price * (1 - stop_pct)
                stopped = True
                break
            elif direction == 'short' and bar['high'] >= entry_price * (1 + stop_pct):
                exit_price = entry_price * (1 + stop_pct)
                stopped = True
                break

        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': price_bars.index[entry_idx],
            'exit_time': price_bars.index[exit_idx],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'hold_minutes': hold_minutes if not stopped else (j - entry_idx),
            'stopped': stopped,
            'cascade_notional': total_not,
        })

    return trades

# ============================================================================
# STRATEGY 2: EXTREME IMBALANCE REVERSAL
# ============================================================================

def strategy_imbalance_reversal(liq_bars, price_bars, imbalance_thresh=0.7,
                                min_notional=1000, hold_minutes=15, stop_pct=0.005):
    """
    When liquidation imbalance is extreme (>70% one-sided), fade it.
    Extreme buy liquidations (imbalance < -0.7) → go long (buy the panic).
    Extreme sell liquidations (imbalance > +0.7) → go short (sell the euphoria).
    """
    # Rolling 5-min imbalance for smoother signal
    liq_bars = liq_bars.copy()
    liq_bars['imb_5m'] = liq_bars['liq_imbalance'].rolling(5, min_periods=1).mean()
    liq_bars['not_5m'] = liq_bars['liq_notional'].rolling(5, min_periods=1).sum()

    trades = []
    last_trade_time = None

    for ts, row in liq_bars.iterrows():
        # Cooldown: at least hold_minutes between trades
        if last_trade_time is not None and (ts - last_trade_time).total_seconds() < hold_minutes * 60:
            continue
        if row['not_5m'] < min_notional:
            continue

        direction = None
        if row['imb_5m'] < -imbalance_thresh:
            direction = 'long'
        elif row['imb_5m'] > imbalance_thresh:
            direction = 'short'

        if direction is None:
            continue

        entry_idx = price_bars.index.searchsorted(ts)
        if entry_idx >= len(price_bars) - hold_minutes:
            continue
        entry_price = price_bars.iloc[entry_idx]['close']

        exit_idx = min(entry_idx + hold_minutes, len(price_bars) - 1)
        exit_price = price_bars.iloc[exit_idx]['close']

        stopped = False
        for j in range(entry_idx, exit_idx + 1):
            bar = price_bars.iloc[j]
            if direction == 'long' and bar['low'] <= entry_price * (1 - stop_pct):
                exit_price = entry_price * (1 - stop_pct)
                stopped = True
                break
            elif direction == 'short' and bar['high'] >= entry_price * (1 + stop_pct):
                exit_price = entry_price * (1 + stop_pct)
                stopped = True
                break

        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': price_bars.index[entry_idx],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'hold_minutes': hold_minutes if not stopped else (j - entry_idx),
            'stopped': stopped,
            'imbalance': row['imb_5m'],
        })
        last_trade_time = ts

    return trades

# ============================================================================
# STRATEGY 3: LIQUIDATION RATE SPIKE
# ============================================================================

def strategy_liq_rate_spike(liq_bars, price_bars, zscore_thresh=3.0,
                            hold_minutes=30, stop_pct=0.008):
    """
    When liquidation rate spikes (>3σ), enter in the direction of the price move.
    This is a volatility breakout / trend-following strategy.
    """
    liq_bars = liq_bars.copy()
    # Rolling stats for rate
    liq_bars['rate_mean'] = liq_bars['liq_count'].rolling(60, min_periods=10).mean()
    liq_bars['rate_std'] = liq_bars['liq_count'].rolling(60, min_periods=10).std()
    liq_bars['rate_zscore'] = (liq_bars['liq_count'] - liq_bars['rate_mean']) / (liq_bars['rate_std'] + 1e-10)

    trades = []
    last_trade_time = None

    for ts, row in liq_bars.iterrows():
        if last_trade_time is not None and (ts - last_trade_time).total_seconds() < hold_minutes * 60:
            continue
        if row['rate_zscore'] < zscore_thresh:
            continue

        entry_idx = price_bars.index.searchsorted(ts)
        if entry_idx < 5 or entry_idx >= len(price_bars) - hold_minutes:
            continue

        # Determine direction from recent price move (last 5 bars)
        recent_ret = (price_bars.iloc[entry_idx]['close'] - price_bars.iloc[entry_idx - 5]['close']) / price_bars.iloc[entry_idx - 5]['close']
        direction = 'long' if recent_ret > 0 else 'short'

        entry_price = price_bars.iloc[entry_idx]['close']
        exit_idx = min(entry_idx + hold_minutes, len(price_bars) - 1)
        exit_price = price_bars.iloc[exit_idx]['close']

        stopped = False
        for j in range(entry_idx, exit_idx + 1):
            bar = price_bars.iloc[j]
            if direction == 'long' and bar['low'] <= entry_price * (1 - stop_pct):
                exit_price = entry_price * (1 - stop_pct)
                stopped = True
                break
            elif direction == 'short' and bar['high'] >= entry_price * (1 + stop_pct):
                exit_price = entry_price * (1 + stop_pct)
                stopped = True
                break

        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': price_bars.index[entry_idx],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'hold_minutes': hold_minutes if not stopped else (j - entry_idx),
            'stopped': stopped,
            'rate_zscore': row['rate_zscore'],
        })
        last_trade_time = ts

    return trades

# ============================================================================
# STRATEGY 4: TIME-OF-DAY LIQUIDATION FADE
# ============================================================================

def strategy_tod_liq_fade(liq_bars, price_bars, us_hours=(13, 17),
                          imbalance_thresh=0.5, min_notional=2000,
                          hold_minutes=10, stop_pct=0.004):
    """
    Same as imbalance reversal, but ONLY during US trading hours (13-17 UTC).
    Lower thresholds since US hours have more liquidity and tighter spreads.
    """
    liq_bars = liq_bars.copy()
    liq_bars['imb_5m'] = liq_bars['liq_imbalance'].rolling(5, min_periods=1).mean()
    liq_bars['not_5m'] = liq_bars['liq_notional'].rolling(5, min_periods=1).sum()

    trades = []
    last_trade_time = None

    for ts, row in liq_bars.iterrows():
        # Only trade during US hours
        if not (us_hours[0] <= ts.hour < us_hours[1]):
            continue
        if last_trade_time is not None and (ts - last_trade_time).total_seconds() < hold_minutes * 60:
            continue
        if row['not_5m'] < min_notional:
            continue

        direction = None
        if row['imb_5m'] < -imbalance_thresh:
            direction = 'long'
        elif row['imb_5m'] > imbalance_thresh:
            direction = 'short'

        if direction is None:
            continue

        entry_idx = price_bars.index.searchsorted(ts)
        if entry_idx >= len(price_bars) - hold_minutes:
            continue
        entry_price = price_bars.iloc[entry_idx]['close']

        exit_idx = min(entry_idx + hold_minutes, len(price_bars) - 1)
        exit_price = price_bars.iloc[exit_idx]['close']

        stopped = False
        for j in range(entry_idx, exit_idx + 1):
            bar = price_bars.iloc[j]
            if direction == 'long' and bar['low'] <= entry_price * (1 - stop_pct):
                exit_price = entry_price * (1 - stop_pct)
                stopped = True
                break
            elif direction == 'short' and bar['high'] >= entry_price * (1 + stop_pct):
                exit_price = entry_price * (1 + stop_pct)
                stopped = True
                break

        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': price_bars.index[entry_idx],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl,
            'hold_minutes': hold_minutes if not stopped else (j - entry_idx),
            'stopped': stopped,
            'imbalance': row['imb_5m'],
        })
        last_trade_time = ts

    return trades

# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def sweep_cascade_fade(liq_df, price_bars):
    """Sweep key parameters for cascade fade."""
    print("\n  Parameter sweep: Cascade Fade")
    results = []
    combos = [(p, h, s) for p in [90, 95, 99] for h in [5, 10, 15, 30] for s in [0.003, 0.005, 0.008]]
    for i, (pct, hold, stop) in enumerate(combos):
        if (i + 1) % 12 == 0:
            print(f"    Sweep progress: {i+1}/{len(combos)}", flush=True)
        # Suppress prints inside strategy
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            trades = strategy_cascade_fade(
                liq_df, price_bars, cascade_pct=pct,
                hold_minutes=hold, stop_pct=stop)
        if trades:
            df = pd.DataFrame(trades)
            n = len(df)
            wr = (df['pnl_pct'] > 0).sum() / n * 100
            avg = df['pnl_pct'].mean()
            tot = df['pnl_pct'].sum()
            results.append({
                'pct': pct, 'hold': hold, 'stop': stop,
                'n': n, 'wr': wr, 'avg': avg, 'total': tot,
            })
    if results:
        rdf = pd.DataFrame(results).sort_values('total', ascending=False)
        print(f"\n    Top 5 parameter combos:")
        for _, r in rdf.head(5).iterrows():
            print(f"      pct={r['pct']:.0f} hold={r['hold']:.0f}m stop={r['stop']:.3f}  →  "
                  f"total={r['total']*100:+.2f}%  n={r['n']:.0f}  wr={r['wr']:.1f}%  avg={r['avg']*100:+.3f}%")
        return rdf.iloc[0].to_dict()
    return {}

# ============================================================================
# MAIN
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    """Run all 4 strategies on a single symbol."""
    print(f"\n{'='*70}")
    print(f"  {symbol} — LIQUIDATION STRATEGIES BACKTEST")
    print(f"{'='*70}")

    t0 = datetime.now()

    # --- Load data ---
    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    # --- Build bars ---
    print("  Building 1-min price bars...", end='', flush=True)
    price_bars = build_price_bars(tick_df, '1min')
    print(f" {len(price_bars):,} bars")

    print("  Building 1-min liquidation bars...", end='', flush=True)
    liq_bars = build_liq_bars(liq_df, '1min')
    print(f" {len(liq_bars):,} bars")

    # Align indices
    common_start = max(price_bars.index.min(), liq_bars.index.min())
    common_end = min(price_bars.index.max(), liq_bars.index.max())
    price_bars = price_bars.loc[common_start:common_end]
    liq_bars = liq_bars.loc[common_start:common_end]
    print(f"  Overlapping period: {common_start} to {common_end}")
    print(f"  Price bars: {len(price_bars):,}  Liq bars: {len(liq_bars):,}")

    results = {}

    # --- Strategy 1: Cascade Fade ---
    print(f"\n{'─'*70}")
    print(f"  STRATEGY 1: CASCADE FADE")
    print(f"{'─'*70}")
    trades1 = strategy_cascade_fade(liq_df, price_bars)
    r1 = compute_trade_stats(trades1, "Cascade Fade (default)")

    # Sweep
    best1 = sweep_cascade_fade(liq_df, price_bars)
    results['cascade_fade'] = r1

    # --- Strategy 2: Extreme Imbalance Reversal ---
    print(f"\n{'─'*70}")
    print(f"  STRATEGY 2: EXTREME IMBALANCE REVERSAL")
    print(f"{'─'*70}")
    trades2 = strategy_imbalance_reversal(liq_bars, price_bars)
    r2 = compute_trade_stats(trades2, "Imbalance Reversal (default)")

    # Try different thresholds
    for thresh in [0.5, 0.6, 0.8, 0.9]:
        t = strategy_imbalance_reversal(liq_bars, price_bars, imbalance_thresh=thresh)
        compute_trade_stats(t, f"Imbalance Reversal (thresh={thresh})")
    results['imbalance_reversal'] = r2

    # --- Strategy 3: Liquidation Rate Spike ---
    print(f"\n{'─'*70}")
    print(f"  STRATEGY 3: LIQUIDATION RATE SPIKE")
    print(f"{'─'*70}")
    trades3 = strategy_liq_rate_spike(liq_bars, price_bars)
    r3 = compute_trade_stats(trades3, "Liq Rate Spike (default)")

    for z in [2.0, 2.5, 4.0]:
        t = strategy_liq_rate_spike(liq_bars, price_bars, zscore_thresh=z)
        compute_trade_stats(t, f"Liq Rate Spike (z={z})")
    results['liq_rate_spike'] = r3

    # --- Strategy 4: Time-of-Day Liquidation Fade ---
    print(f"\n{'─'*70}")
    print(f"  STRATEGY 4: TIME-OF-DAY LIQUIDATION FADE (US hours)")
    print(f"{'─'*70}")
    trades4 = strategy_tod_liq_fade(liq_bars, price_bars)
    r4 = compute_trade_stats(trades4, "ToD Liq Fade (default)")

    for thresh in [0.3, 0.4, 0.6, 0.7]:
        t = strategy_tod_liq_fade(liq_bars, price_bars, imbalance_thresh=thresh)
        compute_trade_stats(t, f"ToD Liq Fade (thresh={thresh})")
    results['tod_liq_fade'] = r4

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'─'*70}")
    print(f"  {symbol} complete in {elapsed:.1f}s")
    print(f"{'─'*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Liquidation Strategies Backtest (v26b)')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS,
                        help='Symbols to backtest')
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()

    print("="*70)
    print("  LIQUIDATION STRATEGIES BACKTEST (v26b)")
    print("="*70)
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"  Strategies: 4")
    print(f"    1. Cascade Fade (mean reversion)")
    print(f"    2. Extreme Imbalance Reversal")
    print(f"    3. Liquidation Rate Spike (vol breakout)")
    print(f"    4. Time-of-Day Liquidation Fade (US hours)")
    print("="*70)

    all_results = {}
    t_start = datetime.now()

    for sym in args.symbols:
        try:
            all_results[sym] = run_symbol(sym, args.data_dir)
        except Exception as e:
            print(f"\n  ✗ {sym} FAILED: {e}")
            all_results[sym] = {}

    # ── SUMMARY ──
    print(f"\n\n{'='*70}")
    print(f"  GRAND SUMMARY — ALL SYMBOLS × ALL STRATEGIES")
    print(f"{'='*70}")

    header = f"{'Symbol':<10} {'Strategy':<28} {'Trades':>6} {'WinRate':>8} {'AvgRet':>9} {'TotalRet':>10} {'Sharpe':>8} {'MaxDD':>8}"
    print(header)
    print("─" * len(header))

    for sym, strats in all_results.items():
        for sname, stats in strats.items():
            if not stats:
                continue
            print(f"{sym:<10} {sname:<28} {stats.get('trades',0):>6} "
                  f"{stats.get('win_rate',0):>7.1f}% "
                  f"{stats.get('avg_ret',0)*100:>+8.3f}% "
                  f"{stats.get('total_ret',0)*100:>+9.2f}% "
                  f"{stats.get('sharpe',0):>+7.2f} "
                  f"{stats.get('max_dd',0)*100:>7.3f}%")

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\n{'='*70}")
    print(f"  Total elapsed: {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
