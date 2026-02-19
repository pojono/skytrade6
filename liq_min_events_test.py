#!/usr/bin/env python3
"""
Test min_events=1 vs min_events=2 with displacement ≥10 bps filter.
Quick comparison on all 4 symbols, Config 2 AGGR only.
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.055
SYMBOLS = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']


def load_liquidations(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
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


def detect_cascades(liq_df, price_bars, pct_thresh=95, min_events=2):
    thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= thresh].copy()
    bar_index = price_bars.index
    bar_close = price_bars['close'].values
    timestamps = large['timestamp'].values
    sides = large['side'].values
    notionals = large['notional'].values
    n = len(large)
    cascades = []
    i = 0
    while i < n:
        cluster = [i]
        j = i + 1
        while j < n:
            dt = (timestamps[j] - timestamps[cluster[-1]]).astype('timedelta64[s]').astype(float)
            if dt <= 60:
                cluster.append(j)
                j += 1
            else:
                break
        if len(cluster) >= min_events:
            c_sides = sides[cluster]
            c_notionals = notionals[cluster]
            c_ts = timestamps[cluster]
            buy_not = c_notionals[c_sides == 'Buy'].sum()
            sell_not = c_notionals[c_sides == 'Sell'].sum()
            buy_dominant = buy_not > sell_not
            end_ts = pd.Timestamp(c_ts[-1])
            end_idx = bar_index.searchsorted(end_ts)
            if end_idx >= len(bar_close) - 120 or end_idx < 10:
                i = cluster[-1] + 1
                continue
            current_price = bar_close[end_idx]
            start_idx = bar_index.searchsorted(pd.Timestamp(c_ts[0]))
            if start_idx > 0:
                pre_price = bar_close[max(0, start_idx - 1)]
                cascade_disp_bps = (current_price - pre_price) / pre_price * 10000
            else:
                cascade_disp_bps = 0
            cascades.append({
                'end': end_ts,
                'n_events': len(cluster),
                'buy_dominant': buy_dominant,
                'end_bar_idx': end_idx,
                'current_price': current_price,
                'cascade_disp_bps': cascade_disp_bps,
            })
        i = cluster[-1] + 1 if len(cluster) >= min_events else i + 1
    return cascades


def run_strategy(cascades, price_bars, tp_pct=0.12, sl_pct=None, 
                 entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10):
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values
    bar_close = price_bars['close'].values
    trades = []
    last_trade_time = None

    for cascade in cascades:
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < 5 * 60:
                continue
        if abs(cascade['cascade_disp_bps']) < min_disp_bps:
            continue

        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']
        if cascade['buy_dominant']:
            direction = 'long'
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100) if sl_pct else None
        else:
            direction = 'short'
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100) if sl_pct else None

        filled = False
        end_bar = min(idx + max_hold_min, len(bar_close) - 1)
        fill_bar = None
        for j in range(idx, end_bar + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True; fill_bar = j; break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True; fill_bar = j; break
        if not filled:
            continue

        exit_price = None; exit_reason = 'timeout'
        remaining = max_hold_min - (fill_bar - idx)
        exit_end = min(fill_bar + remaining, len(bar_close) - 1)
        for k in range(fill_bar, exit_end + 1):
            if direction == 'long':
                if sl_price and bar_low[k] <= sl_price:
                    exit_price = sl_price; exit_reason = 'stop_loss'; break
                if bar_high[k] >= tp_price:
                    exit_price = tp_price; exit_reason = 'take_profit'; break
            else:
                if sl_price and bar_high[k] >= sl_price:
                    exit_price = sl_price; exit_reason = 'stop_loss'; break
                if bar_low[k] <= tp_price:
                    exit_price = tp_price; exit_reason = 'take_profit'; break
        if exit_price is None:
            exit_price = bar_close[exit_end]

        if direction == 'long':
            raw_pnl = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl = (limit_price - exit_price) / limit_price * 100

        entry_fee = MAKER_FEE_PCT
        exit_fee = MAKER_FEE_PCT if exit_reason == 'take_profit' else TAKER_FEE_PCT
        net_pnl = raw_pnl - entry_fee - exit_fee
        trades.append({'net_pnl': net_pnl, 'exit_reason': exit_reason})
        last_trade_time = cascade['end']
    return trades


def summarize(trades, label):
    if not trades:
        print(f"  {label:55s}  n=    0  (no trades)")
        return
    net = np.array([t['net_pnl'] for t in trades])
    n_tp = sum(1 for t in trades if t['exit_reason'] == 'take_profit')
    n_sl = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
    n_to = sum(1 for t in trades if t['exit_reason'] == 'timeout')
    wr = (net > 0).mean() * 100
    total = net.sum()
    avg = net.mean()
    # Sharpe
    if net.std() > 0:
        sharpe = avg / net.std() * np.sqrt(252 * 8)
    else:
        sharpe = 0
    # Max DD
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    maxdd = dd.max()
    print(f"  {label:55s}  n={len(trades):>5d}  WR={wr:5.1f}%  avg={avg:+.4f}%  "
          f"tot={total:+7.2f}%  sh={sharpe:+6.1f}  dd={maxdd:5.2f}%  "
          f"TP={n_tp/len(trades)*100:4.0f}% SL={n_sl/len(trades)*100:4.0f}% TO={n_to/len(trades)*100:4.0f}%")


def main():
    t0 = time.time()
    print("=" * 100)
    print("  MIN_EVENTS COMPARISON: 1 vs 2 (with displacement ≥10 bps)")
    print("=" * 100)

    all_results = {}

    for symbol in SYMBOLS:
        print(f"\n{'─'*80}")
        print(f"  {symbol}")
        print(f"{'─'*80}")

        liq_df = load_liquidations(symbol)
        tick_df = load_ticker_prices(symbol)
        print("  Building bars...", end='', flush=True)
        bars = build_price_bars(tick_df, '1min')
        print(f" {len(bars):,} bars")

        results = {}
        for min_ev in [1, 2]:
            cascades = detect_cascades(liq_df, bars, pct_thresh=95, min_events=min_ev)
            n_with_disp = sum(1 for c in cascades if abs(c['cascade_disp_bps']) >= 10)
            print(f"\n  min_events={min_ev}: {len(cascades)} cascades, {n_with_disp} with disp≥10")

            # Config 2 AGGR: off=0.15, TP=0.12, no SL, 60min
            trades_aggr = run_strategy(cascades, bars, tp_pct=0.12, sl_pct=None,
                                       entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10)
            summarize(trades_aggr, f"Config2 AGGR (min_ev={min_ev})")

            # Config 1 SAFE: off=0.15, TP=0.15, SL=0.50, 60min
            trades_safe = run_strategy(cascades, bars, tp_pct=0.15, sl_pct=0.50,
                                       entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10)
            summarize(trades_safe, f"Config1 SAFE (min_ev={min_ev})")

            results[min_ev] = {
                'cascades': len(cascades),
                'with_disp': n_with_disp,
                'aggr_trades': trades_aggr,
                'safe_trades': trades_safe,
            }

        all_results[symbol] = results

    # Cross-symbol summary
    print(f"\n{'='*100}")
    print(f"  CROSS-SYMBOL SUMMARY")
    print(f"{'='*100}")

    for config_name, tp, sl in [("Config2 AGGR", 0.12, None), ("Config1 SAFE", 0.15, 0.50)]:
        key = 'aggr_trades' if 'AGGR' in config_name else 'safe_trades'
        print(f"\n  ── {config_name} ──")
        print(f"  {'':>12s}  {'min_ev=1':>40s}  {'min_ev=2':>40s}  {'delta':>10s}")

        total_1, total_2 = 0, 0
        n_1, n_2 = 0, 0
        for symbol in SYMBOLS:
            r1 = all_results[symbol][1][key]
            r2 = all_results[symbol][2][key]
            t1 = sum(t['net_pnl'] for t in r1) if r1 else 0
            t2 = sum(t['net_pnl'] for t in r2) if r2 else 0
            w1 = (np.array([t['net_pnl'] for t in r1]) > 0).mean() * 100 if r1 else 0
            w2 = (np.array([t['net_pnl'] for t in r2]) > 0).mean() * 100 if r2 else 0
            s = symbol[:4]
            print(f"  {s:>12s}  n={len(r1):>4d} WR={w1:5.1f}% tot={t1:+7.2f}%"
                  f"    n={len(r2):>4d} WR={w2:5.1f}% tot={t2:+7.2f}%"
                  f"    {t1-t2:+7.2f}%")
            total_1 += t1; total_2 += t2
            n_1 += len(r1); n_2 += len(r2)
        print(f"  {'COMBINED':>12s}  n={n_1:>4d}              tot={total_1:+7.2f}%"
              f"    n={n_2:>4d}              tot={total_2:+7.2f}%"
              f"    {total_1-total_2:+7.2f}%")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
