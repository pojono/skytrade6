#!/usr/bin/env python3
"""
v42z: Autocorrelation + Momentum Persistence Patterns

EXP OOO: Return Autocorrelation Signal
  - When recent returns show strong negative autocorrelation → mean-reversion regime
  - When recent returns show strong positive autocorrelation → momentum regime
  - Trade accordingly: fade in MR regime, follow in momentum regime

EXP PPP: Consecutive Move Signal
  - N consecutive up/down 1-min bars → fade the streak
  - Higher N = stronger signal

EXP QQQ: Volatility Clustering Signal
  - After a cluster of high-vol bars, next bar tends to mean-revert
  - Combine with direction for entry

EXP RRR: Price-Volume Divergence
  - Price up but volume declining → weak move, fade it
  - Price up with volume increasing → strong move, follow it

All 4 symbols, walk-forward, 88 days.
"""

import sys, time, os, gc, psutil
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


def load_bars_with_volume(symbol, dates, data_dir='data', chunk_days=10):
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
                df = pd.read_csv(f, usecols=['timestamp', 'price', 'size', 'side'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                dfs.append(df)
        if dfs:
            chunk = pd.concat(dfs, ignore_index=True)
            del dfs
            chunk['notional'] = chunk['price'] * chunk['size']
            ts = chunk.set_index('timestamp')
            b = ts['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last')
            b['volume'] = ts['notional'].resample('1min').sum()
            b = b.dropna(subset=['close'])
            b['volume'] = b['volume'].fillna(0)
            all_bars.append(b)
            del chunk, ts; gc.collect()
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
    out_file = 'results/v42z_autocorr_patterns.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['SOLUSDT', 'DOGEUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42z: AUTOCORRELATION + MOMENTUM PERSISTENCE PATTERNS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in symbols:
        bars = load_bars_with_volume(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        ret_1m = bars['close'].pct_change()

        # ── EXP OOO: RETURN AUTOCORRELATION ──
        print(f"\n  --- EXP OOO: RETURN AUTOCORRELATION ---")

        # Rolling autocorrelation (lag-1) over 30-min window
        autocorr_30 = ret_1m.rolling(30, min_periods=15).apply(
            lambda x: x.autocorr(lag=1), raw=False)

        print(f"  Autocorr stats: mean={autocorr_30.mean():.4f}, "
              f"P5={autocorr_30.quantile(0.05):.4f}, P95={autocorr_30.quantile(0.95):.4f}")

        for pct in [90, 95]:
            # Strong negative autocorr → mean-reversion regime → fade
            neg_thresh = autocorr_30.quantile((100-pct)/100)
            neg_autocorr = autocorr_30 < neg_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[neg_autocorr]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade recent move in MR regime
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Neg autocorr P{100-pct} (fade in MR regime):")
            pstats(train_trades, "TRAIN")
            pstats(test_trades, "TEST")

        # ── EXP PPP: CONSECUTIVE MOVE ──
        print(f"\n  --- EXP PPP: CONSECUTIVE MOVE ---")

        up = (ret_1m > 0).astype(int)
        down = (ret_1m < 0).astype(int)

        # Count consecutive ups/downs
        consec_up = up.groupby((up != up.shift()).cumsum()).cumsum()
        consec_down = down.groupby((down != down.shift()).cumsum()).cumsum()

        for n_consec in [3, 5, 7]:
            streak_up = consec_up >= n_consec
            streak_down = consec_down >= n_consec

            train_trades = []; test_trades = []
            lt = None
            # Fade: after N consecutive ups, go short
            for ts in bars.index[streak_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # short after streak up
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            # Fade: after N consecutive downs, go long
            for ts in bars.index[streak_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # long after streak down
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Consecutive {n_consec}+ bars (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # ── EXP QQQ: VOLATILITY CLUSTERING ──
        print(f"\n  --- EXP QQQ: VOLATILITY CLUSTERING ---")

        abs_ret = ret_1m.abs()
        vol_5m = abs_ret.rolling(5, min_periods=3).mean()
        vol_60m = abs_ret.rolling(60, min_periods=30).mean()
        vol_ratio = vol_5m / (vol_60m + 1e-10)

        for vr_thresh in [2, 3, 5]:
            high_vol_cluster = vol_ratio > vr_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_vol_cluster]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade the vol cluster direction
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Vol cluster >{vr_thresh}x (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # ── EXP RRR: PRICE-VOLUME DIVERGENCE ──
        print(f"\n  --- EXP RRR: PRICE-VOLUME DIVERGENCE ---")

        price_ret_5m = bars['close'].pct_change(5)
        vol_ret_5m = bars['volume'].pct_change(5)

        # Price up + volume down = weak move → fade (short)
        # Price down + volume down = weak move → fade (long)
        for pr_thresh in [0.001, 0.002]:
            weak_up = (price_ret_5m > pr_thresh) & (vol_ret_5m < -0.2)
            weak_down = (price_ret_5m < -pr_thresh) & (vol_ret_5m < -0.2)

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[weak_up]:
                if lt and (ts - lt).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade weak up
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[weak_down]:
                if lt and (ts - lt).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade weak down
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Price-vol divergence (pr>{pr_thresh*100:.1f}%, vol down):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        del bars; gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
