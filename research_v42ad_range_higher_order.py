#!/usr/bin/env python3
"""
v42ad: Range-Based Signals + Higher-Order Statistics

EXP ZZZ: Intraday Range Compression/Expansion
  - When 60-min range is compressed (low volatility) → breakout imminent
  - When range expands suddenly → fade the expansion (mean-reversion)
  - Bollinger Band squeeze analog at 1-min level

EXP AAAA: Skewness Signal
  - Rolling skewness of 1-min returns
  - Positive skew → more large up moves → fade (short)
  - Negative skew → more large down moves → fade (long)

EXP BBBB: Kurtosis Signal (Fat Tails)
  - High kurtosis → extreme moves happening → mean-reversion opportunity
  - Combine with direction for entry

EXP CCCC: High-Low Range Ratio
  - (High-Low)/Close ratio as volatility proxy
  - When range ratio spikes → fade the direction

SOLUSDT + DOGEUSDT, walk-forward, 88 days.
"""

import sys, time, os, gc, psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
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
    print(f"  Loading {symbol}...", end='', flush=True)
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


def sim_trade(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30,
              trail_act=3, trail_dist=2):
    if entry_idx >= len(bars) - max_hold or entry_idx < 1: return None
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


def main():
    out_file = 'results/v42ad_range_higher_order.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ad: RANGE-BASED SIGNALS + HIGHER-ORDER STATISTICS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in ['SOLUSDT', 'DOGEUSDT']:
        bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        ret_1m = bars['close'].pct_change()

        # ── EXP ZZZ: RANGE COMPRESSION/EXPANSION ──
        print(f"\n  --- EXP ZZZ: RANGE COMPRESSION/EXPANSION ---")

        hl_range = (bars['high'] - bars['low']) / bars['close'] * 10000  # in bps
        range_60m = hl_range.rolling(60, min_periods=30).mean()
        range_std = hl_range.rolling(60, min_periods=30).std()
        range_z = (hl_range - range_60m) / (range_std + 1e-10)

        print(f"  Range stats: mean={hl_range.mean():.1f}bps, P5={hl_range.quantile(0.05):.1f}bps, "
              f"P95={hl_range.quantile(0.95):.1f}bps")

        # Range expansion → fade
        for z_thresh in [2, 3, 5]:
            expanded = range_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[expanded]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Range expansion z>{z_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # Range compression → breakout follow
        for z_thresh in [-1, -1.5, -2]:
            compressed = range_z < z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[compressed]:
                if lt and (ts - lt).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts)
                # After compression, follow the next move
                if idx + 1 < len(bars):
                    next_ret = ret_1m.iloc[idx+1] if idx+1 < len(ret_1m) else 0
                    is_long = next_ret > 0  # follow breakout
                    t = sim_trade(bars, idx+1, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

            print(f"\n  Range compression z<{z_thresh} (follow breakout):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP AAAA: SKEWNESS SIGNAL ──
        print(f"\n  --- EXP AAAA: SKEWNESS SIGNAL ---")

        # Rolling skewness (30-min window)
        skew_30 = ret_1m.rolling(30, min_periods=15).skew()
        print(f"  Skewness stats: mean={skew_30.mean():.3f}, std={skew_30.std():.3f}")

        for skew_thresh in [1, 1.5, 2]:
            pos_skew = skew_30 > skew_thresh  # positive skew → short (fade)
            neg_skew = skew_30 < -skew_thresh  # negative skew → long (fade)

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[pos_skew]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade positive skew
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[neg_skew]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade negative skew
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Skewness |>{skew_thresh}| (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP BBBB: KURTOSIS SIGNAL ──
        print(f"\n  --- EXP BBBB: KURTOSIS SIGNAL ---")

        kurt_30 = ret_1m.rolling(30, min_periods=15).kurt()
        print(f"  Kurtosis stats: mean={kurt_30.mean():.2f}, P95={kurt_30.quantile(0.95):.2f}")

        for kurt_thresh in [3, 5, 10]:
            high_kurt = kurt_30 > kurt_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_kurt]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade in fat-tail regime
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Kurtosis >{kurt_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP CCCC: HIGH-LOW RANGE RATIO ──
        print(f"\n  --- EXP CCCC: HIGH-LOW RANGE RATIO ---")

        # Rolling high-low over 15-min window
        roll_high = bars['high'].rolling(15, min_periods=10).max()
        roll_low = bars['low'].rolling(15, min_periods=10).min()
        range_15m = (roll_high - roll_low) / bars['close'] * 10000
        range_15m_avg = range_15m.rolling(60, min_periods=30).mean()
        range_15m_std = range_15m.rolling(60, min_periods=30).std()
        range_15m_z = (range_15m - range_15m_avg) / (range_15m_std + 1e-10)

        for z_thresh in [2, 3]:
            wide_range = range_15m_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[wide_range]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                # Price near top of range → short, near bottom → long
                if idx < len(bars):
                    mid = (roll_high.iloc[idx] + roll_low.iloc[idx]) / 2
                    is_long = bars.iloc[idx]['close'] < mid
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

            print(f"\n  15m range z>{z_thresh} (fade to mid):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        del bars; gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
