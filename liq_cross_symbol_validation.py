#!/usr/bin/env python3
"""
CROSS-SYMBOL VALIDATION — Test strategy on 7 NEW symbols never used in development.
Purpose: Confirm the liquidation cascade edge is structural, not symbol-specific.

Original symbols (used in development): DOGEUSDT, SOLUSDT, ETHUSDT, XRPUSDT
New symbols (out-of-sample):            ADAUSDT, BCHUSDT, LTCUSDT, NEARUSDT, POLUSDT, TONUSDT, XLMUSDT

Strategy: min_ev=1, displacement ≥10 bps, Config 2 AGGR (off=0.15%, TP=0.12%, no SL, 60min)
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.055

# All symbols: original (in-sample) + new (out-of-sample)
ORIGINAL_SYMBOLS = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']
NEW_SYMBOLS = ['ADAUSDT', 'BCHUSDT', 'LTCUSDT', 'NEARUSDT', 'POLUSDT', 'TONUSDT', 'XLMUSDT']
ALL_SYMBOLS = ORIGINAL_SYMBOLS + NEW_SYMBOLS


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
    # Priority 1: preprocessed CSV (fastest)
    csv_path = symbol_dir / "ticker_prices.csv.gz"
    if csv_path.exists():
        print(f"  Loading ticker CSV...", end='', flush=True)
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df['price'] = df['price'].astype(float)
        df = df[['timestamp', 'price']].sort_values('timestamp').reset_index(drop=True)
        print(f" done ({len(df):,})")
        return df
    # Priority 2: REST-format jsonl files
    rest_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    if rest_files:
        print(f"  Loading {len(rest_files)} ticker files (REST)...", end='', flush=True)
        records = []
        for i, file in enumerate(rest_files, 1):
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
    print(f"  No ticker files found!")
    return pd.DataFrame(columns=['timestamp', 'price'])


def build_price_bars(tick_df, freq='1min'):
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    return bars.dropna()


def detect_signals(liq_df, price_bars, pct_thresh=95):
    """Detect P95 liquidation signals (min_ev=1)."""
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
        i = cluster[-1] + 1
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

        end_bar = min(idx + max_hold_min, len(bar_close) - 1)
        filled = False
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
        trades.append({
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'time': cascade['end'],
            'direction': direction,
        })
        last_trade_time = cascade['end']
    return trades


def main():
    t0 = time.time()
    print("=" * 100)
    print("  CROSS-SYMBOL VALIDATION — 7 New Symbols (Out-of-Sample)")
    print("  Strategy: min_ev=1, disp≥10bps, off=0.15%, TP=0.12%, no SL, 60min")
    print("=" * 100)

    results = {}

    for symbol in ALL_SYMBOLS:
        print(f"\n{'─'*80}")
        print(f"  {symbol} {'(ORIGINAL — in-sample)' if symbol in ORIGINAL_SYMBOLS else '(NEW — out-of-sample)'}")
        print(f"{'─'*80}")

        liq_df = load_liquidations(symbol)
        if len(liq_df) < 100:
            print(f"  ⚠️  Only {len(liq_df)} liquidation events — skipping (too few)")
            results[symbol] = {'skip': True, 'reason': f'only {len(liq_df)} liq events'}
            continue

        tick_df = load_ticker_prices(symbol)
        if len(tick_df) < 1000:
            print(f"  ⚠️  Only {len(tick_df)} ticker records — skipping (too few)")
            results[symbol] = {'skip': True, 'reason': f'only {len(tick_df)} ticker records'}
            continue

        print("  Building bars...", end='', flush=True)
        bars = build_price_bars(tick_df, '1min')
        print(f" {len(bars):,} bars")

        # Detect signals
        signals = detect_signals(liq_df, bars, pct_thresh=95)
        n_with_disp = sum(1 for c in signals if abs(c['cascade_disp_bps']) >= 10)
        print(f"  P95 signals: {len(signals)}, with disp≥10: {n_with_disp}")

        if n_with_disp < 10:
            print(f"  ⚠️  Only {n_with_disp} signals with displacement — too few for meaningful test")
            results[symbol] = {'skip': True, 'reason': f'only {n_with_disp} disp signals'}
            continue

        # Config 2 AGGR
        trades_aggr = run_strategy(signals, bars, tp_pct=0.12, sl_pct=None,
                                   entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10)
        # Config 1 SAFE
        trades_safe = run_strategy(signals, bars, tp_pct=0.15, sl_pct=0.50,
                                   entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10)

        for config_name, trades in [("Config2 AGGR", trades_aggr), ("Config1 SAFE", trades_safe)]:
            if not trades:
                print(f"  {config_name}: no fills")
                continue
            net = np.array([t['net_pnl'] for t in trades])
            wr = (net > 0).mean() * 100
            total = net.sum()
            avg = net.mean()
            sharpe = avg / net.std() * np.sqrt(252 * 8) if net.std() > 0 else 0
            cum = np.cumsum(net)
            peak = np.maximum.accumulate(cum)
            maxdd = (peak - cum).max()
            n_tp = sum(1 for t in trades if t['exit_reason'] == 'take_profit')
            n_to = sum(1 for t in trades if t['exit_reason'] == 'timeout')
            n_sl = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
            flag = '✅' if total > 0 else '❌'
            print(f"  {flag} {config_name:15s}  n={len(trades):>4d}  WR={wr:5.1f}%  "
                  f"avg={avg:+.4f}%  tot={total:+7.2f}%  sh={sharpe:+6.1f}  dd={maxdd:5.2f}%  "
                  f"TP={n_tp/len(trades)*100:4.0f}% SL={n_sl/len(trades)*100:4.0f}% TO={n_to/len(trades)*100:4.0f}%")

        # Monthly breakdown for AGGR
        if trades_aggr:
            df = pd.DataFrame(trades_aggr)
            df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
            monthly = []
            for month, group in df.groupby('month'):
                n = len(group)
                tot = group['net_pnl'].sum()
                wr = (group['net_pnl'] > 0).mean() * 100
                flag = '✅' if tot > 0 else '❌'
                monthly.append((str(month), n, wr, tot, tot > 0))
                print(f"    {flag} {month}: n={n:>3d}  WR={wr:5.1f}%  tot={tot:+6.2f}%")
            
            pos_months = sum(1 for m in monthly if m[4])
            print(f"    Positive months: {pos_months}/{len(monthly)}")

        results[symbol] = {
            'skip': False,
            'liq_events': len(liq_df),
            'signals': len(signals),
            'signals_disp': n_with_disp,
            'aggr': trades_aggr,
            'safe': trades_safe,
        }

    # ========================================================================
    # CROSS-SYMBOL SUMMARY
    # ========================================================================
    print(f"\n{'='*100}")
    print(f"  CROSS-SYMBOL SUMMARY")
    print(f"{'='*100}")

    print(f"\n  {'Symbol':>10s}  {'Group':>12s}  {'Liq Events':>10s}  {'Signals':>8s}  "
          f"{'Fills':>6s}  {'WR':>6s}  {'Total':>8s}  {'Sharpe':>7s}  {'DD':>6s}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*6}")

    orig_total = 0; orig_n = 0
    new_total = 0; new_n = 0
    all_total = 0; all_n = 0
    orig_positive = 0; new_positive = 0
    orig_count = 0; new_count = 0

    for symbol in ALL_SYMBOLS:
        r = results[symbol]
        if r.get('skip'):
            group = "ORIGINAL" if symbol in ORIGINAL_SYMBOLS else "NEW"
            print(f"  {symbol:>10s}  {group:>12s}  {'SKIPPED':>10s}  ({r.get('reason', '')})")
            if symbol in NEW_SYMBOLS:
                new_count += 1
            else:
                orig_count += 1
            continue

        trades = r['aggr']
        if not trades:
            continue

        net = np.array([t['net_pnl'] for t in trades])
        wr = (net > 0).mean() * 100
        total = net.sum()
        sharpe = net.mean() / net.std() * np.sqrt(252 * 8) if net.std() > 0 else 0
        cum = np.cumsum(net)
        peak = np.maximum.accumulate(cum)
        maxdd = (peak - cum).max()

        group = "ORIGINAL" if symbol in ORIGINAL_SYMBOLS else "NEW"
        flag = '✅' if total > 0 else '❌'
        print(f"  {flag}{symbol:>9s}  {group:>12s}  {r['liq_events']:>10,d}  {r['signals_disp']:>8d}  "
              f"{len(trades):>6d}  {wr:5.1f}%  {total:+7.2f}%  {sharpe:+6.1f}  {maxdd:5.2f}%")

        all_total += total; all_n += len(trades)
        if symbol in ORIGINAL_SYMBOLS:
            orig_total += total; orig_n += len(trades); orig_count += 1
            if total > 0: orig_positive += 1
        else:
            new_total += total; new_n += len(trades); new_count += 1
            if total > 0: new_positive += 1

    print(f"\n  {'─'*80}")
    print(f"  ORIGINAL (in-sample):      n={orig_n:>5d}  tot={orig_total:+7.2f}%  "
          f"positive={orig_positive}/{orig_count}")
    print(f"  NEW (out-of-sample):       n={new_n:>5d}  tot={new_total:+7.2f}%  "
          f"positive={new_positive}/{new_count}")
    print(f"  ALL COMBINED:              n={all_n:>5d}  tot={all_total:+7.2f}%")

    # Per-trade quality comparison
    if orig_n > 0 and new_n > 0:
        orig_trades_all = []
        new_trades_all = []
        for symbol in ALL_SYMBOLS:
            r = results[symbol]
            if r.get('skip') or not r.get('aggr'):
                continue
            if symbol in ORIGINAL_SYMBOLS:
                orig_trades_all.extend(r['aggr'])
            else:
                new_trades_all.extend(r['aggr'])

        orig_net = np.array([t['net_pnl'] for t in orig_trades_all])
        new_net = np.array([t['net_pnl'] for t in new_trades_all])

        print(f"\n  PER-TRADE QUALITY:")
        print(f"    Original avg: {orig_net.mean():+.4f}%  WR: {(orig_net>0).mean()*100:.1f}%")
        print(f"    New avg:      {new_net.mean():+.4f}%  WR: {(new_net>0).mean()*100:.1f}%")
        print(f"    Difference:   {new_net.mean() - orig_net.mean():+.4f}%")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
