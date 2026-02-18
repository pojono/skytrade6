#!/usr/bin/env python3
"""
v41: Liquidation Cascade MM — Walk-Forward OOS Validation

Key questions:
  1. Does the best IS config survive OOS?
  2. Is cascade direction signal real? (vs random baseline)
  3. How stable is performance across rolling windows?

Approach: Start with SOLUSDT only (best all-months-positive).
  - 70/30 train/test split
  - Parameter sweep on train
  - Evaluate top configs on test
  - Random direction baseline
  - Rolling 30-day window analysis

RAM-conscious: single symbol, ~500MB expected.
"""

import sys
import time
import json
import gzip
import os
import psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.00055  # 0.055%

DEFAULT_SYMBOL = 'SOLUSDT'


def ram_mb():
    return psutil.Process().memory_info().rss / 1024**2


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

def load_liquidations(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol / "bybit" / "liquidations"
    liq_files = sorted(symbol_dir.glob("liquidation_*.jsonl.gz"))
    if not liq_files:
        raise ValueError(f"No liquidation files for {symbol}")

    n_files = len(liq_files)
    t0 = time.time()
    print(f"  Loading {n_files} liq files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 300 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_files - i)
            print(f" [{i}/{n_files} {elapsed:.0f}s ETA {eta:.0f}s]", end='', flush=True)
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
    elapsed = time.time() - t0
    print(f" done ({len(records):,} records, {elapsed:.0f}s) [{ram_str()}]")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_prices(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol / "bybit" / "ticker"
    ticker_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    if not ticker_files:
        raise ValueError(f"No ticker files for {symbol}")

    n_files = len(ticker_files)
    t0 = time.time()
    print(f"  Loading {n_files} ticker files...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 300 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (n_files - i)
            print(f" [{i}/{n_files} {elapsed:.0f}s ETA {eta:.0f}s]", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    ts = data.get('ts')
                    # Handle both formats
                    if 'result' in data:
                        r = data['result']
                        if 'list' in r:
                            # Old format
                            d = r['list'][0]
                        elif 'data' in r:
                            # New format (ws delta/snapshot)
                            d = r['data']
                        else:
                            continue
                    else:
                        continue
                    lp = d.get('lastPrice')
                    if lp is None:
                        continue
                    records.append({
                        'timestamp': pd.to_datetime(ts, unit='ms'),
                        'price': float(lp),
                    })
                except Exception:
                    continue
    elapsed = time.time() - t0
    print(f" done ({len(records):,} records, {elapsed:.0f}s) [{ram_str()}]")
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
# STRATEGY
# ============================================================================

def run_strategy(cascades, price_bars,
                 entry_offset_pct, tp_pct, sl_pct,
                 max_hold_min=30, cooldown_min=5,
                 maker_fee=MAKER_FEE, taker_fee=TAKER_FEE,
                 random_direction=False, rng_seed=42):
    """Run strategy. If random_direction=True, ignore cascade side and pick random."""
    trades = []
    last_trade_time = None
    rng = np.random.RandomState(rng_seed) if random_direction else None

    for cascade in cascades:
        cascade_end = cascade['end']

        if last_trade_time is not None:
            if (cascade_end - last_trade_time).total_seconds() < cooldown_min * 60:
                continue

        entry_bar_idx = price_bars.index.searchsorted(cascade_end)
        if entry_bar_idx >= len(price_bars) - max_hold_min or entry_bar_idx < 1:
            continue

        current_price = price_bars.iloc[entry_bar_idx]['close']

        if random_direction:
            is_long = rng.random() > 0.5
        else:
            is_long = cascade['buy_dominant']

        if is_long:
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

        # PnL
        if direction == 'long':
            gross_pnl = (exit_price - limit_price) / limit_price
        else:
            gross_pnl = (limit_price - exit_price) / limit_price

        entry_fee = maker_fee
        exit_fee = maker_fee if exit_reason == 'take_profit' else taker_fee
        net_pnl = gross_pnl - entry_fee - exit_fee

        trades.append({
            'direction': direction,
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'exit_reason': exit_reason,
            'hold_minutes': exit_bar_idx - fill_bar_idx,
            'entry_time': price_bars.index[fill_bar_idx],
            'total_fee': entry_fee + exit_fee,
        })
        last_trade_time = cascade_end

    return trades


def calc_stats(trades, label=''):
    if not trades:
        return None
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df['net_pnl'] > 0).sum()
    wr = wins / n * 100
    avg_net = df['net_pnl'].mean()
    total_net = df['net_pnl'].sum()
    std = df['net_pnl'].std()
    sharpe = avg_net / (std + 1e-10) * np.sqrt(252 * 24 * 60)
    cum = df['net_pnl'].cumsum()
    max_dd = (cum.cummax() - cum).max()
    tp_pct = (df['exit_reason'] == 'take_profit').sum() / n * 100
    pf = df.loc[df['net_pnl'] > 0, 'net_pnl'].sum() / max(abs(df.loc[df['net_pnl'] < 0, 'net_pnl'].sum()), 1e-10)
    return {
        'label': label, 'n': n, 'wr': wr,
        'avg_net_bps': avg_net * 10000,
        'total_ret_pct': total_net * 100,
        'sharpe': sharpe, 'max_dd_pct': max_dd * 100,
        'tp_pct': tp_pct, 'pf': pf,
        'avg_hold': df['hold_minutes'].mean(),
    }


def print_stats(s):
    if s is None:
        print("    (no trades)")
        return
    print(f"    {s['label']:40s}  n={s['n']:4d}  wr={s['wr']:5.1f}%  "
          f"avg={s['avg_net_bps']:+5.1f}bps  tot={s['total_ret_pct']:+7.2f}%  "
          f"sharpe={s['sharpe']:+7.1f}  DD={s['max_dd_pct']:5.2f}%  "
          f"TP={s['tp_pct']:4.0f}%  PF={s['pf']:5.2f}  hold={s['avg_hold']:.1f}m")


# ============================================================================
# MAIN
# ============================================================================

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SYMBOL
    out_file = f'results/v41_cascade_mm_oos_{symbol}.txt'

    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee

    t_global = time.time()

    print("=" * 90)
    print(f"  v41: CASCADE MM — WALK-FORWARD OOS VALIDATION — {symbol}")
    print(f"  Maker={MAKER_FEE*100:.3f}%  Taker={TAKER_FEE*100:.3f}%")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  [{ram_str()}]")
    print("=" * 90)
    print(f"\n{'#'*80}")
    print(f"  LOADING {symbol}")
    print(f"{'#'*80}")

    liq_df = load_liquidations(symbol)
    tick_df = load_ticker_prices(symbol)

    print("  Building 1-min bars...", end='', flush=True)
    price_bars = build_price_bars(tick_df, '1min')
    print(f" {len(price_bars):,} bars")

    # Free tick_df
    del tick_df
    import gc; gc.collect()
    print(f"  [{ram_str()}] (freed tick_df)")

    days_total = (price_bars.index.max() - price_bars.index.min()).total_seconds() / 86400
    print(f"  Period: {price_bars.index.min()} to {price_bars.index.max()} ({days_total:.0f} days)")

    # Detect cascades on FULL dataset (threshold from full data)
    print("  Detecting cascades...", end='', flush=True)
    all_cascades = detect_cascades(liq_df, pct_thresh=95)
    print(f" {len(all_cascades)} cascades ({len(all_cascades)/max(days_total,1):.1f}/day)")

    # Free liq_df
    del liq_df; gc.collect()
    print(f"  [{ram_str()}] (freed liq_df)")

    # ── TRAIN/TEST SPLIT ──
    split_date = price_bars.index.min() + pd.Timedelta(days=int(days_total * 0.7))
    print(f"\n  Train/Test split at {split_date.strftime('%Y-%m-%d')}")
    print(f"  Train: {price_bars.index.min().strftime('%Y-%m-%d')} to {split_date.strftime('%Y-%m-%d')} ({int(days_total*0.7)} days)")
    print(f"  Test:  {split_date.strftime('%Y-%m-%d')} to {price_bars.index.max().strftime('%Y-%m-%d')} ({int(days_total*0.3)} days)")

    train_cascades = [c for c in all_cascades if c['end'] < split_date]
    test_cascades = [c for c in all_cascades if c['end'] >= split_date]
    print(f"  Train cascades: {len(train_cascades)}  Test cascades: {len(test_cascades)}")

    # ── PART 1: PARAMETER SWEEP ON TRAIN ──
    print(f"\n{'='*80}")
    print(f"  PART 1: PARAMETER SWEEP ON TRAIN SET ({len(train_cascades)} cascades)")
    print(f"{'='*80}")

    offsets = [0.10, 0.15, 0.20, 0.25]
    tps = [0.10, 0.15, 0.20, 0.25, 0.30]
    sls = [0.15, 0.25, 0.35, 0.50]
    total_combos = len(offsets) * len(tps) * len(sls)

    print(f"  Sweeping {total_combos} configs...")
    print(f"  {'Config':40s}  {'n':>4s}  {'wr':>6s}  {'avg':>6s}  {'tot':>8s}  {'sharpe':>8s}  {'DD':>6s}  {'TP%':>4s}  {'PF':>5s}")
    print(f"  {'-'*100}")

    train_results = []
    t_sweep = time.time()
    done = 0

    for offset in offsets:
        for tp in tps:
            for sl in sls:
                trades = run_strategy(train_cascades, price_bars,
                                      entry_offset_pct=offset, tp_pct=tp, sl_pct=sl)
                label = f"off={offset:.2f} TP={tp:.2f} SL={sl:.2f}"
                s = calc_stats(trades, label)
                if s and s['n'] >= 10:
                    train_results.append({**s, 'offset': offset, 'tp': tp, 'sl': sl})
                done += 1
                if done % 20 == 0:
                    elapsed = time.time() - t_sweep
                    eta = elapsed / done * (total_combos - done)
                    print(f"  ... {done}/{total_combos} ({elapsed:.0f}s, ETA {eta:.0f}s) [{ram_str()}]")

    train_results.sort(key=lambda x: x['total_ret_pct'], reverse=True)

    print(f"\n  TOP 10 TRAIN CONFIGS (by total return):")
    print(f"  {'Config':40s}  {'n':>4s}  {'wr':>6s}  {'avg':>6s}  {'tot':>8s}  {'sharpe':>8s}  {'DD':>6s}  {'TP%':>4s}  {'PF':>5s}")
    print(f"  {'-'*100}")
    for s in train_results[:10]:
        print_stats(s)

    # ── PART 2: EVALUATE TOP CONFIGS ON TEST ──
    print(f"\n{'='*80}")
    print(f"  PART 2: TOP 5 TRAIN CONFIGS → TEST SET ({len(test_cascades)} cascades)")
    print(f"{'='*80}")

    top_n = min(5, len(train_results))
    for i, tr in enumerate(train_results[:top_n]):
        offset, tp, sl = tr['offset'], tr['tp'], tr['sl']
        label = f"off={offset:.2f} TP={tp:.2f} SL={sl:.2f}"

        # Test set
        test_trades = run_strategy(test_cascades, price_bars,
                                   entry_offset_pct=offset, tp_pct=tp, sl_pct=sl)
        ts = calc_stats(test_trades, f"TEST  {label}")

        print(f"\n  Config #{i+1}: {label}")
        print(f"    TRAIN: ", end='')
        print_stats(tr)
        print(f"    TEST:  ", end='')
        print_stats(ts)

        # Degradation
        if ts and tr['total_ret_pct'] != 0:
            deg = (1 - ts['total_ret_pct'] / tr['total_ret_pct']) * 100 if tr['total_ret_pct'] > 0 else 0
            # Normalize by days
            train_days = int(days_total * 0.7)
            test_days = int(days_total * 0.3)
            train_daily = tr['total_ret_pct'] / train_days
            test_daily = ts['total_ret_pct'] / test_days if test_days > 0 else 0
            print(f"    Daily return: train={train_daily:+.4f}%/day  test={test_daily:+.4f}%/day  "
                  f"{'✅ OOS POSITIVE' if test_daily > 0 else '❌ OOS NEGATIVE'}")

    # ── PART 3: RANDOM DIRECTION BASELINE ──
    print(f"\n{'='*80}")
    print(f"  PART 3: RANDOM DIRECTION BASELINE (is cascade side signal real?)")
    print(f"{'='*80}")

    # Use best train config
    if train_results:
        best = train_results[0]
        offset, tp, sl = best['offset'], best['tp'], best['sl']
        label_base = f"off={offset:.2f} TP={tp:.2f} SL={sl:.2f}"

        # Real direction on full data
        real_trades = run_strategy(all_cascades, price_bars,
                                   entry_offset_pct=offset, tp_pct=tp, sl_pct=sl)
        real_s = calc_stats(real_trades, f"REAL direction  {label_base}")

        # Random direction (10 seeds, average)
        random_rets = []
        random_wrs = []
        for seed in range(10):
            rand_trades = run_strategy(all_cascades, price_bars,
                                       entry_offset_pct=offset, tp_pct=tp, sl_pct=sl,
                                       random_direction=True, rng_seed=seed)
            rs = calc_stats(rand_trades, f"RANDOM seed={seed}")
            if rs:
                random_rets.append(rs['total_ret_pct'])
                random_wrs.append(rs['wr'])

        print(f"\n  Best config: {label_base}")
        print(f"    REAL direction:  ", end='')
        print_stats(real_s)
        if random_rets:
            print(f"    RANDOM direction (10 seeds):")
            print(f"      avg total_ret = {np.mean(random_rets):+.2f}%  "
                  f"(min={np.min(random_rets):+.2f}%, max={np.max(random_rets):+.2f}%)")
            print(f"      avg win_rate  = {np.mean(random_wrs):.1f}%  "
                  f"(min={np.min(random_wrs):.1f}%, max={np.max(random_wrs):.1f}%)")
            edge = real_s['total_ret_pct'] - np.mean(random_rets) if real_s else 0
            print(f"      Direction edge = {edge:+.2f}% over random")
            if edge > 2:
                print(f"      ✅ CASCADE DIRECTION ADDS REAL VALUE")
            elif edge > 0:
                print(f"      ⚠️ CASCADE DIRECTION ADDS MARGINAL VALUE")
            else:
                print(f"      ❌ CASCADE DIRECTION HAS NO VALUE — edge is from TP/SL structure")

    # ── PART 4: ROLLING 30-DAY WINDOWS ──
    print(f"\n{'='*80}")
    print(f"  PART 4: ROLLING 30-DAY WINDOW STABILITY")
    print(f"{'='*80}")

    if train_results:
        best = train_results[0]
        offset, tp, sl = best['offset'], best['tp'], best['sl']

        start_date = price_bars.index.min()
        end_date = price_bars.index.max()
        window_days = 30
        step_days = 15  # 50% overlap

        print(f"  Config: off={offset:.2f} TP={tp:.2f} SL={sl:.2f}")
        print(f"  Window: {window_days} days, step: {step_days} days")
        print(f"\n  {'Window':25s}  {'n':>4s}  {'wr':>6s}  {'avg':>6s}  {'tot':>8s}  {'sharpe':>8s}  {'DD':>6s}")
        print(f"  {'-'*80}")

        window_start = start_date
        pos_windows = 0
        total_windows = 0

        while window_start + pd.Timedelta(days=window_days) <= end_date:
            window_end = window_start + pd.Timedelta(days=window_days)
            window_cascades = [c for c in all_cascades
                               if c['end'] >= window_start and c['end'] < window_end]

            if len(window_cascades) >= 5:
                trades = run_strategy(window_cascades, price_bars,
                                      entry_offset_pct=offset, tp_pct=tp, sl_pct=sl)
                s = calc_stats(trades, f"{window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}")
                if s:
                    flag = "✅" if s['total_ret_pct'] > 0 else "  "
                    print(f"  {flag} {s['label']:40s}  n={s['n']:4d}  wr={s['wr']:5.1f}%  "
                          f"avg={s['avg_net_bps']:+5.1f}bps  tot={s['total_ret_pct']:+7.2f}%  "
                          f"sharpe={s['sharpe']:+7.1f}  DD={s['max_dd_pct']:5.2f}%")
                    total_windows += 1
                    if s['total_ret_pct'] > 0:
                        pos_windows += 1

            window_start += pd.Timedelta(days=step_days)

        if total_windows > 0:
            print(f"\n  Positive windows: {pos_windows}/{total_windows} ({pos_windows/total_windows*100:.0f}%)")

    # ── SUMMARY ──
    elapsed = time.time() - t_global
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {symbol} — {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
