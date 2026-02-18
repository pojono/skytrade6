#!/usr/bin/env python3
"""
Liquidation Cascade MM — Fee-Aware Sweep (v26e)

Proper double-fee model:
  - Entry: maker fee (0.02%)
  - TP exit: maker fee (0.02%) — limit order at TP price
  - SL exit: taker fee (0.055%) — stop-market order
  - Timeout exit: taker fee (0.055%) — market close

Sweep wider TP values (0.20-0.50%) to overcome round-trip fees.
Only run on the 4 best symbols (skip BTC).
"""

import sys
import time
import json
import gzip
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

# Your Bybit fee tier
MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.00055  # 0.055%

SYMBOLS = ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]

# ============================================================================
# DATA LOADING (same as liq_cascade_mm.py)
# ============================================================================

def load_liquidations(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    if not liq_files:
        raise ValueError(f"No liquidation files for {symbol}")

    print(f"  Loading {len(liq_files)} liq files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 200 == 0:
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
    df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_prices(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    ticker_dirs = [symbol_dir / "bybit" / "ticker", symbol_dir]
    ticker_files = []
    for d in ticker_dirs:
        ticker_files.extend(sorted(d.glob("ticker_*.jsonl.gz")))
    ticker_files = sorted(set(ticker_files))
    if not ticker_files:
        raise ValueError(f"No ticker files for {symbol}")

    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 200 == 0:
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
    print(f" done ({len(records):,})")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    return df


def build_price_bars(tick_df, freq='1min'):
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    return bars.dropna()


def detect_cascades(liq_df, pct_thresh=95, time_window_sec=60, min_events=2):
    vol_thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= vol_thresh].copy()
    cascades = []
    current = []
    for _, row in large.iterrows():
        if not current:
            current = [row]
        else:
            dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
            if dt <= time_window_sec:
                current.append(row)
            else:
                if len(current) >= min_events:
                    cdf = pd.DataFrame(current)
                    buy_not = cdf[cdf['side'] == 'Buy']['notional'].sum()
                    sell_not = cdf[cdf['side'] == 'Sell']['notional'].sum()
                    total_not = buy_not + sell_not
                    cascades.append({
                        'start': cdf['timestamp'].min(),
                        'end': cdf['timestamp'].max(),
                        'n_events': len(cdf),
                        'total_notional': total_not,
                        'buy_notional': buy_not,
                        'sell_notional': sell_not,
                        'buy_dominant': buy_not > sell_not,
                        'duration_sec': (cdf['timestamp'].max() - cdf['timestamp'].min()).total_seconds(),
                        'avg_price': (cdf['price'] * cdf['notional']).sum() / (total_not + 1e-10),
                    })
                current = [row]
    if len(current) >= min_events:
        cdf = pd.DataFrame(current)
        buy_not = cdf[cdf['side'] == 'Buy']['notional'].sum()
        sell_not = cdf[cdf['side'] == 'Sell']['notional'].sum()
        total_not = buy_not + sell_not
        cascades.append({
            'start': cdf['timestamp'].min(),
            'end': cdf['timestamp'].max(),
            'n_events': len(cdf),
            'total_notional': total_not,
            'buy_notional': buy_not,
            'sell_notional': sell_not,
            'buy_dominant': buy_not > sell_not,
            'duration_sec': (cdf['timestamp'].max() - cdf['timestamp'].min()).total_seconds(),
            'avg_price': (cdf['price'] * cdf['notional']).sum() / (total_not + 1e-10),
        })
    return cascades


# ============================================================================
# FEE-AWARE STRATEGY
# ============================================================================

def strategy_fee_aware(cascades, price_bars,
                       entry_offset_pct, tp_pct, sl_pct,
                       max_hold_min=30, cooldown_min=5,
                       maker_fee=MAKER_FEE, taker_fee=TAKER_FEE,
                       us_hours_only=False):
    """
    Proper double-fee model:
      Entry: always maker (limit order)
      TP: maker (limit order sitting at TP price)
      SL: taker (stop-market)
      Timeout: taker (market close)
    """
    trades = []
    last_trade_time = None

    for cascade in cascades:
        cascade_end = cascade['end']

        if last_trade_time is not None:
            if (cascade_end - last_trade_time).total_seconds() < cooldown_min * 60:
                continue

        if us_hours_only and not (13 <= cascade_end.hour < 18):
            continue

        entry_bar_idx = price_bars.index.searchsorted(cascade_end)
        if entry_bar_idx >= len(price_bars) - max_hold_min or entry_bar_idx < 1:
            continue

        current_price = price_bars.iloc[entry_bar_idx]['close']

        if cascade['buy_dominant']:
            direction = 'long'
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100)
        else:
            direction = 'short'
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100)

        # Fill simulation
        filled = False
        fill_bar_idx = None
        end_bar_idx = min(entry_bar_idx + max_hold_min, len(price_bars) - 1)

        for j in range(entry_bar_idx, end_bar_idx + 1):
            bar = price_bars.iloc[j]
            if direction == 'long' and bar['low'] <= limit_price:
                filled = True
                fill_bar_idx = j
                break
            elif direction == 'short' and bar['high'] >= limit_price:
                filled = True
                fill_bar_idx = j
                break

        if not filled:
            continue

        # TP/SL simulation
        exit_price = None
        exit_reason = 'timeout'
        exit_bar_idx = fill_bar_idx
        remaining_hold = max_hold_min - (fill_bar_idx - entry_bar_idx)
        exit_end = min(fill_bar_idx + remaining_hold, len(price_bars) - 1)

        for k in range(fill_bar_idx, exit_end + 1):
            bar = price_bars.iloc[k]
            if direction == 'long':
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar_idx = k
                    break
                if bar['high'] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar_idx = k
                    break
            else:
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    exit_bar_idx = k
                    break
                if bar['low'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    exit_bar_idx = k
                    break

        if exit_price is None:
            exit_price = price_bars.iloc[exit_end]['close']
            exit_bar_idx = exit_end
            exit_reason = 'timeout'

        # PnL with proper double fees
        if direction == 'long':
            gross_pnl = (exit_price - limit_price) / limit_price
        else:
            gross_pnl = (limit_price - exit_price) / limit_price

        # Entry is always maker
        entry_fee = maker_fee
        # Exit fee depends on exit type
        if exit_reason == 'take_profit':
            exit_fee = maker_fee    # limit order at TP
        else:
            exit_fee = taker_fee    # stop-market or market close

        net_pnl = gross_pnl - entry_fee - exit_fee

        trades.append({
            'direction': direction,
            'limit_price': limit_price,
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'total_fee': entry_fee + exit_fee,
            'exit_reason': exit_reason,
            'hold_minutes': exit_bar_idx - fill_bar_idx,
            'entry_time': price_bars.index[fill_bar_idx],
        })
        last_trade_time = cascade_end

    return trades


def summarize(trades, label):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df['net_pnl'] > 0).sum()
    wr = wins / n * 100
    avg_gross = df['gross_pnl'].mean()
    avg_net = df['net_pnl'].mean()
    avg_fee = df['total_fee'].mean()
    total_net = df['net_pnl'].sum()
    std = df['net_pnl'].std()
    sharpe = avg_net / (std + 1e-10) * np.sqrt(252 * 24 * 60)
    cum = df['net_pnl'].cumsum()
    max_dd = (cum.cummax() - cum).max()

    tp_n = (df['exit_reason'] == 'take_profit').sum()
    sl_n = (df['exit_reason'] == 'stop_loss').sum()
    to_n = (df['exit_reason'] == 'timeout').sum()

    # Monthly
    df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')
    monthly = df.groupby('month')['net_pnl'].sum()
    pos_months = (monthly > 0).sum()
    total_months = len(monthly)

    return {
        'label': label,
        'fills': n, 'wr': wr,
        'avg_gross': avg_gross, 'avg_net': avg_net, 'avg_fee': avg_fee,
        'total_net': total_net, 'sharpe': sharpe, 'max_dd': max_dd,
        'tp_pct': tp_n / n * 100, 'sl_pct': sl_n / n * 100, 'to_pct': to_n / n * 100,
        'pos_months': pos_months, 'total_months': total_months,
        'avg_hold': df['hold_minutes'].mean(),
    }


def print_result(r):
    if r is None:
        return
    print(f"  {r['label']:42s}  fills={r['fills']:4d}  wr={r['wr']:5.1f}%  "
          f"gross={r['avg_gross']*100:+6.3f}%  fee={r['avg_fee']*100:.3f}%  "
          f"net={r['avg_net']*100:+6.3f}%  total={r['total_net']*100:+8.2f}%  "
          f"sharpe={r['sharpe']:+7.1f}  DD={r['max_dd']*100:5.2f}%  "
          f"TP={r['tp_pct']:4.0f}% SL={r['sl_pct']:4.0f}%  "
          f"months={r['pos_months']}/{r['total_months']}  hold={r['avg_hold']:.1f}m")


# ============================================================================
# MAIN
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    print(f"\n{'='*90}")
    print(f"  {symbol} — FEE-AWARE SWEEP (maker={MAKER_FEE*100:.3f}% taker={TAKER_FEE*100:.3f}%)")
    print(f"{'='*90}")

    t0 = time.time()

    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    print("  Building 1-min bars...", end='', flush=True)
    price_bars = build_price_bars(tick_df, '1min')
    print(f" {len(price_bars):,} bars")

    days = (price_bars.index.max() - price_bars.index.min()).total_seconds() / 86400
    print(f"  Period: {price_bars.index.min()} to {price_bars.index.max()} ({days:.0f} days)")

    cascades = detect_cascades(liq_df, pct_thresh=95)
    print(f"  Cascades: {len(cascades)} ({len(cascades)/max(days,1):.1f}/day)")

    # ── SWEEP: offset × TP × SL ──
    print(f"\n{'─'*90}")
    print(f"  SWEEP: All hours, double-fee model")
    print(f"  {'Config':42s}  {'fills':>5s}  {'wr':>6s}  {'gross':>7s}  {'fee':>6s}  "
          f"{'net':>7s}  {'total':>9s}  {'sharpe':>8s}  {'DD':>6s}  "
          f"{'TP%':>4s} {'SL%':>4s}  {'months':>7s}  {'hold':>5s}")
    print(f"  {'─'*140}")

    all_results = []

    for offset in [0.10, 0.15, 0.20]:
        for tp in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            for sl in [0.15, 0.25, 0.35, 0.50]:
                trades = strategy_fee_aware(
                    cascades, price_bars,
                    entry_offset_pct=offset, tp_pct=tp, sl_pct=sl,
                    max_hold_min=30, cooldown_min=5,
                )
                label = f"off={offset:.2f}% TP={tp:.2f}% SL={sl:.2f}%"
                r = summarize(trades, label)
                if r and r['fills'] >= 10:
                    all_results.append(r)

    # Sort by total net return
    all_results.sort(key=lambda x: x['total_net'], reverse=True)

    print(f"\n  TOP 15 CONFIGS (by total net return after fees):")
    print(f"  {'Config':42s}  {'fills':>5s}  {'wr':>6s}  {'gross':>7s}  {'fee':>6s}  "
          f"{'net':>7s}  {'total':>9s}  {'sharpe':>8s}  {'DD':>6s}  "
          f"{'TP%':>4s} {'SL%':>4s}  {'months':>7s}  {'hold':>5s}")
    print(f"  {'─'*140}")
    for r in all_results[:15]:
        print_result(r)

    # ── BEST CONFIG: US hours only ──
    if all_results:
        best = all_results[0]
        # Parse best config
        parts = best['label'].split()
        b_off = float(parts[0].split('=')[1].rstrip('%'))
        b_tp = float(parts[1].split('=')[1].rstrip('%'))
        b_sl = float(parts[2].split('=')[1].rstrip('%'))

        print(f"\n{'─'*90}")
        print(f"  BEST CONFIG US-HOURS ONLY (13-18 UTC):")
        print(f"  {'─'*140}")
        trades_us = strategy_fee_aware(
            cascades, price_bars,
            entry_offset_pct=b_off, tp_pct=b_tp, sl_pct=b_sl,
            max_hold_min=30, cooldown_min=5,
            us_hours_only=True,
        )
        r_us = summarize(trades_us, f"US-hours off={b_off:.2f}% TP={b_tp:.2f}% SL={b_sl:.2f}%")
        if r_us:
            print_result(r_us)

        # Monthly breakdown for best config
        print(f"\n  Monthly breakdown (best config, all hours):")
        df_best = pd.DataFrame(strategy_fee_aware(
            cascades, price_bars,
            entry_offset_pct=b_off, tp_pct=b_tp, sl_pct=b_sl,
            max_hold_min=30, cooldown_min=5,
        ))
        if len(df_best) > 0:
            df_best['month'] = pd.to_datetime(df_best['entry_time']).dt.to_period('M')
            for m, grp in df_best.groupby('month'):
                n = len(grp)
                wr = (grp['net_pnl'] > 0).sum() / n * 100
                avg_g = grp['gross_pnl'].mean()
                avg_n = grp['net_pnl'].mean()
                tot = grp['net_pnl'].sum()
                flag = "✅" if tot > 0 else "  "
                print(f"    {flag} {m}: n={n:4d}  wr={wr:.1f}%  "
                      f"gross={avg_g*100:+.3f}%  net={avg_n*100:+.3f}%  "
                      f"total={tot*100:+.2f}%")

    # ── COMPARISON: What if 0% maker (VIP promo)? ──
    if all_results:
        print(f"\n{'─'*90}")
        print(f"  COMPARISON: Same best config with 0% maker fee:")
        print(f"  {'─'*140}")
        trades_0fee = strategy_fee_aware(
            cascades, price_bars,
            entry_offset_pct=b_off, tp_pct=b_tp, sl_pct=b_sl,
            max_hold_min=30, cooldown_min=5,
            maker_fee=0.0, taker_fee=TAKER_FEE,
        )
        r_0fee = summarize(trades_0fee, f"0% maker  off={b_off:.2f}% TP={b_tp:.2f}% SL={b_sl:.2f}%")
        if r_0fee:
            print_result(r_0fee)

    elapsed = time.time() - t0
    print(f"\n  {symbol} done in {elapsed:.0f}s")

    return all_results


def main():
    t_start = time.time()

    print("=" * 90)
    print(f"  LIQUIDATION CASCADE MM — FEE-AWARE SWEEP (v26e)")
    print(f"  Maker fee: {MAKER_FEE*100:.3f}%  Taker fee: {TAKER_FEE*100:.3f}%")
    print(f"  Double-fee model: entry(maker) + TP(maker) or SL/timeout(taker)")
    print("=" * 90)

    all_sym_results = {}
    for sym in SYMBOLS:
        try:
            all_sym_results[sym] = run_symbol(sym)
        except Exception as e:
            print(f"\n  ✗ {sym} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── GRAND SUMMARY ──
    print(f"\n\n{'='*90}")
    print(f"  GRAND SUMMARY — BEST CONFIG PER SYMBOL (after {MAKER_FEE*100:.3f}%+{TAKER_FEE*100:.3f}% fees)")
    print(f"{'='*90}")
    for sym, results in all_sym_results.items():
        if results:
            print(f"\n  {sym}:")
            for r in results[:5]:
                print_result(r)

    elapsed = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
