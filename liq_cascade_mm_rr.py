#!/usr/bin/env python3
"""
Liquidation Cascade MM — Risk/Reward Optimization (v26f)

Goal: Find R:R combos that minimize expensive SL exits (taker fee)
and maximize cheap TP exits (maker fee).

Key insight: SL costs 0.075% round-trip, TP costs 0.04% round-trip.
So we want very high TP% even if individual TP is small.

Sweep:
  - Very tight TP: 0.05%, 0.08%, 0.10%, 0.12%, 0.15%, 0.20%
  - Very wide SL: 0.30%, 0.50%, 0.75%, 1.00%, 1.50%, no-SL (timeout only)
  - Also test: no SL at all (just timeout after 30/60 min)
  - Offsets: 0.15%, 0.20%, 0.25%
  - Max hold: 30 min and 60 min

Run on DOGE + SOL (best symbols from v26e).
"""

import sys
import time
import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.00055  # 0.055%

SYMBOLS = ["DOGEUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT"]


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
        if i % 500 == 0:
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
        if i % 500 == 0:
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
    return pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)


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
                        'buy_dominant': buy_not > sell_not,
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
            'buy_dominant': buy_not > sell_not,
        })
    return cascades


def strategy_rr(cascades, price_bars,
                entry_offset_pct, tp_pct, sl_pct,
                max_hold_min=30, cooldown_min=5,
                maker_fee=MAKER_FEE, taker_fee=TAKER_FEE):
    """
    sl_pct=0 means NO stop loss — only exit via TP or timeout.
    """
    trades = []
    last_trade_time = None

    for cascade in cascades:
        cascade_end = cascade['end']
        if last_trade_time is not None:
            if (cascade_end - last_trade_time).total_seconds() < cooldown_min * 60:
                continue

        entry_bar_idx = price_bars.index.searchsorted(cascade_end)
        if entry_bar_idx >= len(price_bars) - max_hold_min or entry_bar_idx < 1:
            continue

        current_price = price_bars.iloc[entry_bar_idx]['close']

        if cascade['buy_dominant']:
            direction = 'long'
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100) if sl_pct > 0 else 0
        else:
            direction = 'short'
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100) if sl_pct > 0 else float('inf')

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
                if sl_pct > 0 and bar['low'] <= sl_price:
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
                if sl_pct > 0 and bar['high'] >= sl_price:
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

        # PnL with proper fees
        if direction == 'long':
            gross_pnl = (exit_price - limit_price) / limit_price
        else:
            gross_pnl = (limit_price - exit_price) / limit_price

        entry_fee = maker_fee
        if exit_reason == 'take_profit':
            exit_fee = maker_fee
        else:
            exit_fee = taker_fee

        net_pnl = gross_pnl - entry_fee - exit_fee

        trades.append({
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'exit_reason': exit_reason,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'hold_minutes': exit_bar_idx - fill_bar_idx,
            'entry_time': price_bars.index[fill_bar_idx],
        })
        last_trade_time = cascade_end

    return trades


def summarize(trades):
    if not trades or len(trades) < 5:
        return None
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df['net_pnl'] > 0).sum()
    wr = wins / n * 100
    avg_net = df['net_pnl'].mean()
    avg_gross = df['gross_pnl'].mean()
    avg_fee = (df['entry_fee'] + df['exit_fee']).mean()
    total_net = df['net_pnl'].sum()
    std = df['net_pnl'].std()
    sharpe = avg_net / (std + 1e-10) * np.sqrt(252 * 24 * 60)
    cum = df['net_pnl'].cumsum()
    max_dd = (cum.cummax() - cum).max()

    tp_n = (df['exit_reason'] == 'take_profit').sum()
    sl_n = (df['exit_reason'] == 'stop_loss').sum()
    to_n = (df['exit_reason'] == 'timeout').sum()

    df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')
    monthly = df.groupby('month')['net_pnl'].sum()
    pos_months = (monthly > 0).sum()
    total_months = len(monthly)

    return {
        'fills': n, 'wr': wr,
        'avg_gross': avg_gross, 'avg_net': avg_net, 'avg_fee': avg_fee,
        'total_net': total_net, 'sharpe': sharpe, 'max_dd': max_dd,
        'tp_pct': tp_n / n * 100, 'sl_pct': sl_n / n * 100, 'to_pct': to_n / n * 100,
        'pos_months': pos_months, 'total_months': total_months,
        'avg_hold': df['hold_minutes'].mean(),
    }


def print_header():
    print(f"  {'Config':48s} {'fills':>5s} {'wr':>5s} {'gross':>7s} {'fee':>6s} "
          f"{'net':>7s} {'total':>8s} {'shrp':>6s} {'DD':>5s} "
          f"{'TP%':>4s} {'SL%':>4s} {'TO%':>4s} {'mo+':>4s} {'hold':>5s}")
    print(f"  {'─'*145}")


def print_row(label, r):
    if r is None:
        return
    print(f"  {label:48s} {r['fills']:5d} {r['wr']:5.1f} {r['avg_gross']*100:+6.3f}% "
          f"{r['avg_fee']*100:5.3f}% {r['avg_net']*100:+6.3f}% "
          f"{r['total_net']*100:+7.2f}% {r['sharpe']:+6.1f} {r['max_dd']*100:5.2f}% "
          f"{r['tp_pct']:4.0f} {r['sl_pct']:4.0f} {r['to_pct']:4.0f} "
          f"{r['pos_months']}/{r['total_months']}  {r['avg_hold']:4.1f}m")


def run_symbol(symbol, data_dir='data'):
    print(f"\n{'='*90}")
    print(f"  {symbol} — R:R OPTIMIZATION (maker={MAKER_FEE*100:.3f}% taker={TAKER_FEE*100:.3f}%)")
    print(f"{'='*90}")

    t0 = time.time()

    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    print("  Building 1-min bars...", end='', flush=True)
    price_bars = build_price_bars(tick_df, '1min')
    print(f" {len(price_bars):,} bars")

    days = (price_bars.index.max() - price_bars.index.min()).total_seconds() / 86400
    print(f"  Period: {days:.0f} days")

    cascades = detect_cascades(liq_df, pct_thresh=95)
    print(f"  Cascades: {len(cascades)} ({len(cascades)/max(days,1):.1f}/day)")

    # ── MASSIVE SWEEP ──
    offsets = [0.15, 0.20, 0.25]
    tps = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    sls = [0, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50]  # 0 = no SL
    max_holds = [30, 60]

    total_configs = len(offsets) * len(tps) * len(sls) * len(max_holds)
    print(f"\n  Sweeping {total_configs} configs...", flush=True)

    all_results = []
    count = 0
    for offset in offsets:
        for tp in tps:
            for sl in sls:
                for mh in max_holds:
                    count += 1
                    if count % 100 == 0:
                        print(f"    [{count}/{total_configs}]", flush=True)

                    trades = strategy_rr(
                        cascades, price_bars,
                        entry_offset_pct=offset, tp_pct=tp, sl_pct=sl,
                        max_hold_min=mh, cooldown_min=5,
                    )
                    r = summarize(trades)
                    if r:
                        sl_label = f"{sl:.2f}%" if sl > 0 else "none"
                        label = f"off={offset:.2f} TP={tp:.2f} SL={sl_label} hold={mh}m"
                        r['label'] = label
                        r['offset'] = offset
                        r['tp'] = tp
                        r['sl'] = sl
                        r['max_hold'] = mh
                        all_results.append(r)

    print(f"    [{count}/{total_configs}] done")

    # Sort by total net
    all_results.sort(key=lambda x: x['total_net'], reverse=True)

    # ── TOP 25 by total return ──
    print(f"\n  TOP 25 BY TOTAL NET RETURN:")
    print_header()
    for r in all_results[:25]:
        print_row(r['label'], r)

    # ── TOP 25 by Sharpe ──
    by_sharpe = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  TOP 25 BY SHARPE RATIO:")
    print_header()
    for r in by_sharpe[:25]:
        print_row(r['label'], r)

    # ── TOP 25 by net per trade (efficiency) ──
    by_net = sorted(all_results, key=lambda x: x['avg_net'], reverse=True)
    print(f"\n  TOP 25 BY NET PER TRADE:")
    print_header()
    for r in by_net[:25]:
        print_row(r['label'], r)

    # ── BEST: filter for positive total + positive months ──
    viable = [r for r in all_results if r['total_net'] > 0 and r['pos_months'] >= 3]
    viable.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  VIABLE CONFIGS (net>0, 3+ positive months) sorted by Sharpe:")
    print_header()
    for r in viable[:30]:
        print_row(r['label'], r)

    # ── ANALYSIS: How does SL level affect fee drag? ──
    print(f"\n  FEE ANALYSIS BY SL LEVEL (best offset/TP for each SL):")
    print(f"  {'SL':>8s} {'best_config':>35s} {'TP%':>5s} {'SL%':>5s} {'TO%':>5s} "
          f"{'avg_fee':>8s} {'avg_net':>8s} {'total':>8s}")
    print(f"  {'─'*100}")
    for sl_val in sls:
        sl_group = [r for r in all_results if r['sl'] == sl_val]
        if sl_group:
            best = max(sl_group, key=lambda x: x['total_net'])
            sl_label = f"{sl_val:.2f}%" if sl_val > 0 else "none"
            print(f"  {sl_label:>8s} {best['label']:>35s} "
                  f"{best['tp_pct']:5.1f} {best['sl_pct']:5.1f} {best['to_pct']:5.1f} "
                  f"{best['avg_fee']*100:7.4f}% {best['avg_net']*100:+7.4f}% "
                  f"{best['total_net']*100:+7.2f}%")

    # ── Monthly breakdown for top 3 ──
    if all_results:
        for rank, r in enumerate(all_results[:3]):
            print(f"\n  #{rank+1} Monthly: {r['label']}")
            trades = strategy_rr(
                cascades, price_bars,
                entry_offset_pct=r['offset'], tp_pct=r['tp'], sl_pct=r['sl'],
                max_hold_min=r['max_hold'], cooldown_min=5,
            )
            df = pd.DataFrame(trades)
            df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')
            for m, grp in df.groupby('month'):
                n = len(grp)
                wr = (grp['net_pnl'] > 0).sum() / n * 100
                tot = grp['net_pnl'].sum()
                flag = "✅" if tot > 0 else "  "
                print(f"    {flag} {m}: n={n:4d}  wr={wr:.1f}%  total={tot*100:+.2f}%")

    elapsed = time.time() - t0
    print(f"\n  {symbol} done in {elapsed:.0f}s")
    return all_results


def main():
    t_start = time.time()

    print("=" * 90)
    print(f"  LIQUIDATION CASCADE MM — R:R OPTIMIZATION (v26f)")
    print(f"  Maker: {MAKER_FEE*100:.3f}%  Taker: {TAKER_FEE*100:.3f}%")
    print(f"  TP exit=maker, SL/timeout exit=taker")
    print(f"  Goal: minimize SL exits (expensive), maximize TP exits (cheap)")
    print("=" * 90)

    all_sym = {}
    for sym in SYMBOLS:
        try:
            all_sym[sym] = run_symbol(sym)
        except Exception as e:
            print(f"\n  ✗ {sym} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── CROSS-SYMBOL SUMMARY ──
    print(f"\n\n{'='*90}")
    print(f"  CROSS-SYMBOL: BEST CONFIG PER SYMBOL")
    print(f"{'='*90}")
    print_header()
    for sym, results in all_sym.items():
        if results:
            best = results[0]
            print_row(f"{sym} {best['label']}", best)

    # Find configs that work on ALL symbols
    print(f"\n  UNIVERSAL CONFIGS (positive on all symbols):")
    if len(all_sym) >= 2:
        # Get all config keys
        config_results = {}
        for sym, results in all_sym.items():
            for r in results:
                key = (r['offset'], r['tp'], r['sl'], r['max_hold'])
                if key not in config_results:
                    config_results[key] = {}
                config_results[key][sym] = r

        universal = []
        for key, sym_results in config_results.items():
            if len(sym_results) >= len(all_sym):
                all_positive = all(r['total_net'] > 0 for r in sym_results.values())
                if all_positive:
                    total_across = sum(r['total_net'] for r in sym_results.values())
                    min_sharpe = min(r['sharpe'] for r in sym_results.values())
                    universal.append({
                        'key': key,
                        'total_across': total_across,
                        'min_sharpe': min_sharpe,
                        'results': sym_results,
                    })

        universal.sort(key=lambda x: x['total_across'], reverse=True)
        print_header()
        for u in universal[:15]:
            off, tp, sl, mh = u['key']
            sl_label = f"{sl:.2f}%" if sl > 0 else "none"
            label = f"off={off:.2f} TP={tp:.2f} SL={sl_label} hold={mh}m"
            syms_str = " | ".join(f"{s}:{r['total_net']*100:+.1f}%" 
                                   for s, r in u['results'].items())
            print(f"  {label:48s} total_across={u['total_across']*100:+.1f}%  "
                  f"min_sharpe={u['min_sharpe']:+.1f}  [{syms_str}]")

    elapsed = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
