#!/usr/bin/env python3
"""
Verification script — spot-check the displacement filter results.
Prints individual trades for manual inspection.
Checks for look-ahead bias, double-counting, and other issues.
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.055


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


def detect_cascades(liq_df, price_bars, pct_thresh=95):
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
        if len(cluster) >= 2:
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
                'start': pd.Timestamp(c_ts[0]),
                'end': end_ts,
                'n_events': len(cluster),
                'buy_dominant': buy_dominant,
                'end_bar_idx': end_idx,
                'current_price': current_price,
                'pre_price': pre_price if start_idx > 0 else current_price,
                'cascade_disp_bps': cascade_disp_bps,
                'hour_utc': end_ts.hour,
                'day_of_week': end_ts.dayofweek,
            })
        i = cluster[-1] + 1 if len(cluster) >= 2 else i + 1
    return cascades


def main():
    symbol = "DOGEUSDT"
    print(f"{'='*100}")
    print(f"  VERIFICATION: {symbol}")
    print(f"{'='*100}")

    t0 = time.time()
    liq_df = load_liquidations(symbol)
    tick_df = load_ticker_prices(symbol)
    print("  Building bars...", end='', flush=True)
    bars = build_price_bars(tick_df, '1min')
    print(f" {len(bars):,} bars")

    cascades = detect_cascades(liq_df, bars, pct_thresh=95)
    print(f"  Total cascades: {len(cascades):,}")

    # ── CHECK 1: Displacement distribution ──
    disps = [abs(c['cascade_disp_bps']) for c in cascades]
    print(f"\n  ── DISPLACEMENT DISTRIBUTION ──")
    print(f"  Total cascades: {len(disps)}")
    print(f"  Mean: {np.mean(disps):.1f} bps")
    print(f"  Median: {np.median(disps):.1f} bps")
    print(f"  P25: {np.percentile(disps, 25):.1f} bps")
    print(f"  P75: {np.percentile(disps, 75):.1f} bps")
    print(f"  P90: {np.percentile(disps, 90):.1f} bps")
    print(f"  ≥10 bps: {sum(1 for d in disps if d >= 10)} ({sum(1 for d in disps if d >= 10)/len(disps)*100:.0f}%)")
    print(f"  <10 bps: {sum(1 for d in disps if d < 10)} ({sum(1 for d in disps if d < 10)/len(disps)*100:.0f}%)")

    # ── CHECK 2: Look-ahead bias check ──
    # The displacement is computed from bar BEFORE cascade start to bar AT cascade end
    # This should be known at cascade end time — no look-ahead
    print(f"\n  ── LOOK-AHEAD BIAS CHECK ──")
    for c in cascades[:5]:
        print(f"  Cascade: {c['start']} → {c['end']}")
        print(f"    pre_price={c['pre_price']:.6f}  end_price={c['current_price']:.6f}  disp={c['cascade_disp_bps']:+.1f} bps")
        print(f"    buy_dominant={c['buy_dominant']}  n_events={c['n_events']}")

    # ── CHECK 3: Run Config 2 AGGR with displacement, print first 20 trades ──
    print(f"\n  ── CONFIG 2 AGGR (off=0.15 TP=0.12 SL=none 60m) + DISPLACEMENT ≥10 ──")
    bar_high = bars['high'].values
    bar_low = bars['low'].values
    bar_close = bars['close'].values

    trades = []
    last_trade_time = None
    n_filtered_disp = 0
    n_filtered_cooldown = 0
    n_not_filled = 0

    for cascade in cascades:
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < 5 * 60:
                n_filtered_cooldown += 1
                continue

        if abs(cascade['cascade_disp_bps']) < 10:
            n_filtered_disp += 1
            continue

        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']

        if cascade['buy_dominant']:
            direction = 'long'
            limit_price = current_price * (1 - 0.15 / 100)
            tp_price = limit_price * (1 + 0.12 / 100)
        else:
            direction = 'short'
            limit_price = current_price * (1 + 0.15 / 100)
            tp_price = limit_price * (1 - 0.12 / 100)

        # Fill simulation
        filled = False
        fill_bar_idx = None
        end_bar_idx = min(idx + 60, len(bar_close) - 1)

        for j in range(idx, end_bar_idx + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True
                fill_bar_idx = j
                break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True
                fill_bar_idx = j
                break

        if not filled:
            n_not_filled += 1
            continue

        # TP simulation (no SL)
        exit_price = None
        exit_reason = 'timeout'
        remaining = 60 - (fill_bar_idx - idx)
        exit_end = min(fill_bar_idx + remaining, len(bar_close) - 1)

        for k in range(fill_bar_idx, exit_end + 1):
            if direction == 'long':
                if bar_high[k] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    break
            else:
                if bar_low[k] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    break

        if exit_price is None:
            exit_price = bar_close[exit_end]

        if direction == 'long':
            raw_pnl_pct = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl_pct = (limit_price - exit_price) / limit_price * 100

        entry_fee = MAKER_FEE_PCT
        exit_fee = MAKER_FEE_PCT if exit_reason == 'take_profit' else TAKER_FEE_PCT
        net_pnl_pct = raw_pnl_pct - entry_fee - exit_fee

        hold_min = (fill_bar_idx - idx) if fill_bar_idx else 0

        trades.append({
            'cascade_end': cascade['end'],
            'direction': direction,
            'disp_bps': cascade['cascade_disp_bps'],
            'current_price': current_price,
            'limit_price': limit_price,
            'tp_price': tp_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'raw_pnl_pct': raw_pnl_pct,
            'net_pnl_pct': net_pnl_pct,
            'fill_bar': fill_bar_idx,
            'hold_min': hold_min,
            'entry_time': bars.index[fill_bar_idx] if fill_bar_idx < len(bars.index) else None,
        })
        last_trade_time = cascade['end']

    print(f"  Cascades: {len(cascades)}")
    print(f"  Filtered by displacement: {n_filtered_disp}")
    print(f"  Filtered by cooldown: {n_filtered_cooldown}")
    print(f"  Not filled: {n_not_filled}")
    print(f"  Filled trades: {len(trades)}")

    # Print first 20 trades
    print(f"\n  FIRST 20 TRADES:")
    print(f"  {'#':>3s}  {'cascade_end':>20s}  {'dir':>5s}  {'disp':>6s}  {'mkt_price':>10s}  {'limit':>10s}  {'tp':>10s}  {'exit':>10s}  {'reason':>6s}  {'raw%':>7s}  {'net%':>7s}")
    for i, t in enumerate(trades[:20]):
        print(f"  {i+1:>3d}  {str(t['cascade_end']):>20s}  {t['direction']:>5s}  {t['disp_bps']:>+5.0f}  "
              f"{t['current_price']:>10.6f}  {t['limit_price']:>10.6f}  {t['tp_price']:>10.6f}  "
              f"{t['exit_price']:>10.6f}  {t['exit_reason'][:6]:>6s}  {t['raw_pnl_pct']:>+6.4f}%  {t['net_pnl_pct']:>+6.4f}%")

    # Print last 20 trades
    print(f"\n  LAST 20 TRADES:")
    print(f"  {'#':>3s}  {'cascade_end':>20s}  {'dir':>5s}  {'disp':>6s}  {'mkt_price':>10s}  {'limit':>10s}  {'tp':>10s}  {'exit':>10s}  {'reason':>6s}  {'raw%':>7s}  {'net%':>7s}")
    for i, t in enumerate(trades[-20:]):
        idx_real = len(trades) - 20 + i
        print(f"  {idx_real+1:>3d}  {str(t['cascade_end']):>20s}  {t['direction']:>5s}  {t['disp_bps']:>+5.0f}  "
              f"{t['current_price']:>10.6f}  {t['limit_price']:>10.6f}  {t['tp_price']:>10.6f}  "
              f"{t['exit_price']:>10.6f}  {t['exit_reason'][:6]:>6s}  {t['raw_pnl_pct']:>+6.4f}%  {t['net_pnl_pct']:>+6.4f}%")

    # Summary stats
    net = np.array([t['net_pnl_pct'] for t in trades])
    raw = np.array([t['raw_pnl_pct'] for t in trades])
    exits = [t['exit_reason'] for t in trades]
    n_tp = sum(1 for e in exits if e == 'take_profit')
    n_to = sum(1 for e in exits if e == 'timeout')

    print(f"\n  ── SUMMARY ──")
    print(f"  Trades: {len(trades)}")
    print(f"  TP exits: {n_tp} ({n_tp/len(trades)*100:.1f}%)")
    print(f"  Timeout exits: {n_to} ({n_to/len(trades)*100:.1f}%)")
    print(f"  Win rate: {(net > 0).mean()*100:.1f}%")
    print(f"  Avg raw PnL: {np.mean(raw):+.4f}%")
    print(f"  Avg net PnL: {np.mean(net):+.4f}%")
    print(f"  Total net: {np.sum(net):+.2f}%")
    print(f"  Avg entry fee: {MAKER_FEE_PCT:.3f}%")
    print(f"  Avg exit fee: {n_tp/len(trades)*MAKER_FEE_PCT + n_to/len(trades)*TAKER_FEE_PCT:.4f}%")

    # ── CHECK 4: Timeout trade PnL distribution ──
    timeout_pnls = [t['net_pnl_pct'] for t in trades if t['exit_reason'] == 'timeout']
    if timeout_pnls:
        print(f"\n  ── TIMEOUT TRADES (n={len(timeout_pnls)}) ──")
        print(f"  Mean net PnL: {np.mean(timeout_pnls):+.4f}%")
        print(f"  Min: {np.min(timeout_pnls):+.4f}%")
        print(f"  Max: {np.max(timeout_pnls):+.4f}%")
        print(f"  Win rate: {(np.array(timeout_pnls) > 0).mean()*100:.0f}%")

    # ── CHECK 5: Monthly breakdown ──
    print(f"\n  ── MONTHLY BREAKDOWN ──")
    times = [t['entry_time'] for t in trades if t.get('entry_time') is not None]
    s = pd.Series(net[:len(times)], index=pd.DatetimeIndex(times))
    monthly = s.resample('M').agg(['sum', 'count', lambda x: (x > 0).mean() * 100])
    monthly.columns = ['total', 'trades', 'wr']
    for idx_m, row in monthly.iterrows():
        flag = "✅" if row['total'] > 0 else "  "
        print(f"    {flag} {idx_m.strftime('%Y-%m')}: n={int(row['trades']):>4d}  WR={row['wr']:.0f}%  total={row['total']:>+.2f}%")
    n_pos = (monthly['total'] > 0).sum()
    print(f"    Positive months: {n_pos}/{len(monthly)}")

    # ── CHECK 6: Compare with and without displacement ──
    print(f"\n  ── SANITY CHECK: BASELINE (no displacement filter) ──")
    trades_base = []
    last_trade_time = None
    for cascade in cascades:
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < 5 * 60:
                continue
        # NO displacement filter
        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']
        if cascade['buy_dominant']:
            direction = 'long'
            limit_price = current_price * (1 - 0.15 / 100)
            tp_price = limit_price * (1 + 0.12 / 100)
        else:
            direction = 'short'
            limit_price = current_price * (1 + 0.15 / 100)
            tp_price = limit_price * (1 - 0.12 / 100)
        filled = False
        end_bar_idx = min(idx + 60, len(bar_close) - 1)
        for j in range(idx, end_bar_idx + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True; fill_bar_idx = j; break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True; fill_bar_idx = j; break
        if not filled:
            continue
        exit_price = None; exit_reason = 'timeout'
        remaining = 60 - (fill_bar_idx - idx)
        exit_end = min(fill_bar_idx + remaining, len(bar_close) - 1)
        for k in range(fill_bar_idx, exit_end + 1):
            if direction == 'long':
                if bar_high[k] >= tp_price:
                    exit_price = tp_price; exit_reason = 'take_profit'; break
            else:
                if bar_low[k] <= tp_price:
                    exit_price = tp_price; exit_reason = 'take_profit'; break
        if exit_price is None:
            exit_price = bar_close[exit_end]
        if direction == 'long':
            raw_pnl_pct = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl_pct = (limit_price - exit_price) / limit_price * 100
        entry_fee = MAKER_FEE_PCT
        exit_fee = MAKER_FEE_PCT if exit_reason == 'take_profit' else TAKER_FEE_PCT
        net_pnl_pct = raw_pnl_pct - entry_fee - exit_fee
        trades_base.append({'net_pnl_pct': net_pnl_pct, 'exit_reason': exit_reason})
        last_trade_time = cascade['end']

    net_base = np.array([t['net_pnl_pct'] for t in trades_base])
    n_tp_base = sum(1 for t in trades_base if t['exit_reason'] == 'take_profit')
    print(f"  Baseline trades: {len(trades_base)}")
    print(f"  Baseline TP rate: {n_tp_base/len(trades_base)*100:.1f}%")
    print(f"  Baseline WR: {(net_base > 0).mean()*100:.1f}%")
    print(f"  Baseline total: {np.sum(net_base):+.2f}%")
    print(f"  Baseline avg net: {np.mean(net_base):+.4f}%")

    print(f"\n  COMPARISON:")
    print(f"  {'':>20s}  {'Baseline':>10s}  {'+ Disp≥10':>10s}  {'Delta':>10s}")
    print(f"  {'Trades':>20s}  {len(trades_base):>10d}  {len(trades):>10d}  {len(trades)-len(trades_base):>+10d}")
    print(f"  {'WR':>20s}  {(net_base>0).mean()*100:>9.1f}%  {(net>0).mean()*100:>9.1f}%  {((net>0).mean()-(net_base>0).mean())*100:>+9.1f}%")
    print(f"  {'Total net':>20s}  {np.sum(net_base):>+9.2f}%  {np.sum(net):>+9.2f}%  {np.sum(net)-np.sum(net_base):>+9.2f}%")
    print(f"  {'Avg net/trade':>20s}  {np.mean(net_base):>+9.4f}%  {np.mean(net):>+9.4f}%  {np.mean(net)-np.mean(net_base):>+9.4f}%")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
