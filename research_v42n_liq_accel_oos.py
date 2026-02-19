#!/usr/bin/env python3
"""
v42n: Liq Acceleration + Imbalance OOS Validation

Walk-forward: train=60d (May 12–Jul 10), test=28d (Jul 11–Aug 7)
Test on all 3 symbols.
Compare with cascade MM baseline.
"""

import sys, time, json, gzip, os, gc, psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE = 0.0002
TAKER_FEE = 0.00055


def ram_str():
    p = psutil.Process().memory_info().rss / 1024**3
    a = psutil.virtual_memory().available / 1024**3
    return f"RAM={p:.1f}GB, avail={a:.1f}GB"


class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', buffering=1)
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
    def flush(self):
        self.stdout.flush()
        self.file.flush()


def get_dates(start, n):
    base = datetime.strptime(start, '%Y-%m-%d')
    return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)]


def load_bars_chunked(symbol, dates, data_dir='data', chunk_days=10):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} bars...", end='', flush=True)
    all_bars = []
    for start in range(0, n, chunk_days):
        chunk_dates = dates[start:start+chunk_days]
        dfs = []
        for d in chunk_dates:
            f = base / f"{symbol}{d}.csv.gz"
            if f.exists():
                df = pd.read_csv(f, usecols=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                dfs.append(df)
        if dfs:
            chunk = pd.concat(dfs, ignore_index=True)
            del dfs
            b = chunk.set_index('timestamp')['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last').dropna()
            all_bars.append(b)
            del chunk; gc.collect()
        done = min(start+chunk_days, n)
        el = time.time()-t0
        print(f" [{done}/{n} {el:.0f}s]", end='', flush=True)
    if not all_bars: print(" NO DATA"); return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f" {len(result):,} bars ({time.time()-t0:.0f}s) [{ram_str()}]")
    return result


def load_liqs(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "liquidations"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} liqs...", end='', flush=True)
    recs = []
    for i, d in enumerate(dates):
        for hr in range(24):
            f = base / f"liquidation_{d}_hr{hr:02d}.jsonl.gz"
            if not f.exists(): continue
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for ev in data['result']['data']:
                                recs.append({
                                    'timestamp': pd.to_datetime(ev['T'], unit='ms'),
                                    'side': ev['S'], 'volume': float(ev['v']),
                                    'price': float(ev['p']),
                                })
                    except: continue
        if (i+1) % 15 == 0:
            el = time.time()-t0
            print(f" [{i+1}/{n} {el:.0f}s]", end='', flush=True)
    if not recs: print(" NO DATA"); return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    print(f" {len(df):,} ({time.time()-t0:.0f}s) [{ram_str()}]")
    return df


def sim_trade(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30,
              trail_act=3, trail_dist=2):
    price = bars.iloc[entry_idx]['close']
    if is_long:
        lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
    else:
        lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)
    filled = False
    for j in range(entry_idx, min(entry_idx+max_hold, len(bars))):
        b = bars.iloc[j]
        if is_long and b['low'] <= lim: filled=True; fi=j; break
        elif not is_long and b['high'] >= lim: filled=True; fi=j; break
    if not filled: return None
    ep = None; er = 'timeout'
    best_profit = 0; trailing_active = False; current_sl = sl_p
    for k in range(fi, min(fi+max_hold, len(bars))):
        b = bars.iloc[k]
        if is_long:
            cp = (b['high']-lim)/lim
            if cp > best_profit: best_profit = cp
            if best_profit >= trail_act/10000 and not trailing_active:
                trailing_active = True; current_sl = lim*(1+trail_dist/10000)
            if trailing_active:
                ns = b['high']*(1-trail_dist/10000)
                if ns > current_sl: current_sl = ns
            if b['low'] <= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; break
            if b['high'] >= tp_p: ep=tp_p; er='tp'; break
        else:
            cp = (lim-b['low'])/lim
            if cp > best_profit: best_profit = cp
            if best_profit >= trail_act/10000 and not trailing_active:
                trailing_active = True; current_sl = lim*(1-trail_dist/10000)
            if trailing_active:
                ns = b['low']*(1+trail_dist/10000)
                if ns < current_sl: current_sl = ns
            if b['high'] >= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; break
            if b['low'] <= tp_p: ep=tp_p; er='tp'; break
    if ep is None: ep = bars.iloc[min(fi+max_hold, len(bars)-1)]['close']
    if is_long: gross = (ep-lim)/lim
    else: gross = (lim-ep)/lim
    fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
    return {'net': gross-fee, 'exit': er, 'time': bars.index[fi]}


def pstats(trades, label):
    if not trades:
        print(f"    {label:55s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "  "
    print(f"  {flag} {label:55s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.2f}%  sh={sh:+8.1f}")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh}


def run_liq_accel(liq_df, bars, split_ts, window=15, thresh=5, cooldown=60):
    """Run liq acceleration strategy, return train and test trades separately."""
    liq_ts = liq_df.set_index('timestamp')
    buy_vol = liq_ts[liq_ts['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
    sell_vol = liq_ts[liq_ts['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
    total_vol = (buy_vol + sell_vol).reindex(bars.index, fill_value=0)
    buy_vol = buy_vol.reindex(bars.index, fill_value=0)
    sell_vol = sell_vol.reindex(bars.index, fill_value=0)

    roll_avg = total_vol.rolling(window, min_periods=1).mean()
    ratio = total_vol / (roll_avg + 1)
    signals = ratio[ratio > thresh].index

    train_trades = []; test_trades = []
    last_time = None
    for ts in signals:
        if last_time and (ts - last_time).total_seconds() < cooldown: continue
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars) - 30 or idx < 1: continue
        bv = buy_vol.iloc[idx] if idx < len(buy_vol) else 0
        sv = sell_vol.iloc[idx] if idx < len(sell_vol) else 0
        is_long = bv > sv
        t = sim_trade(bars, idx, is_long)
        if t:
            if ts < split_ts:
                train_trades.append(t)
            else:
                test_trades.append(t)
            last_time = ts
    return train_trades, test_trades


def run_liq_imbalance(liq_df, bars, split_ts, window=5, thresh=0.80, cooldown=300):
    """Run liq imbalance strategy, return train and test trades separately."""
    liq_ts = liq_df.set_index('timestamp')
    buy_vol = liq_ts[liq_ts['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
    sell_vol = liq_ts[liq_ts['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
    buy_vol = buy_vol.reindex(bars.index, fill_value=0)
    sell_vol = sell_vol.reindex(bars.index, fill_value=0)

    buy_roll = buy_vol.rolling(window, min_periods=1).sum()
    sell_roll = sell_vol.rolling(window, min_periods=1).sum()
    total_roll = buy_roll + sell_roll
    buy_ratio = buy_roll / (total_roll + 1)

    buy_signals = buy_ratio[buy_ratio > thresh].index
    sell_signals = buy_ratio[buy_ratio < (1-thresh)].index

    train_trades = []; test_trades = []
    last_time = None

    all_signals = [(ts, True) for ts in buy_signals] + [(ts, False) for ts in sell_signals]
    all_signals.sort(key=lambda x: x[0])

    for ts, is_long in all_signals:
        if last_time and (ts - last_time).total_seconds() < cooldown: continue
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars) - 30 or idx < 1: continue
        t = sim_trade(bars, idx, is_long)
        if t:
            if ts < split_ts:
                train_trades.append(t)
            else:
                test_trades.append(t)
            last_time = ts
    return train_trades, test_trades


def main():
    out_file = 'results/v42n_liq_accel_oos.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42n: LIQ ACCELERATION + IMBALANCE OOS VALIDATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print(f"  Train: May 12–Jul 10 (60d), Test: Jul 11–Aug 7 (28d)")
    print("="*80)

    for sym in symbols:
        liq = load_liqs(sym, all_dates)
        bars = load_bars_chunked(sym, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {sym}")
        print(f"{'#'*80}")

        # EXP GG OOS: Liq Acceleration
        print(f"\n  --- LIQ ACCELERATION ---")
        for w, th, cd in [(5, 3, 300), (15, 5, 60), (15, 5, 300), (30, 5, 300), (30, 10, 300)]:
            train, test = run_liq_accel(liq, bars, split_ts, window=w, thresh=th, cooldown=cd)
            print(f"\n  w={w}m thresh={th}x cd={cd}s:")
            ts_r = pstats(train, "TRAIN")
            te_r = pstats(test, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # EXP II OOS: Liq Imbalance
        print(f"\n  --- LIQ IMBALANCE ---")
        for w, th, cd in [(5, 0.80, 300), (5, 0.90, 300), (15, 0.80, 300), (15, 0.90, 300)]:
            train, test = run_liq_imbalance(liq, bars, split_ts, window=w, thresh=th, cooldown=cd)
            print(f"\n  w={w}m imb>{th:.0%} cd={cd}s:")
            ts_r = pstats(train, "TRAIN")
            te_r = pstats(test, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        del liq, bars; gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
