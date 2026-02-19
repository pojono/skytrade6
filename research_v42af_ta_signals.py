#!/usr/bin/env python3
"""
v42af: Classic TA Signals Adapted for 1-Min Crypto

EXP HHHH: RSI Extreme Signal
  - 14-bar RSI at 1-min level
  - RSI < 20 → oversold → long (fade)
  - RSI > 80 → overbought → short (fade)

EXP IIII: Bollinger Band Squeeze + Touch
  - BB squeeze (bandwidth < P10) → breakout imminent
  - BB touch (price outside bands) → fade back to mean

EXP JJJJ: EMA Crossover Divergence
  - Fast EMA (5) vs Slow EMA (20) divergence
  - When divergence is extreme → fade

EXP KKKK: Stochastic Oscillator Extreme
  - %K/%D at extreme levels → fade

SOLUSDT + DOGEUSDT, walk-forward, 88 days.
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


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def main():
    out_file = 'results/v42af_ta_signals.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42af: CLASSIC TA SIGNALS FOR 1-MIN CRYPTO")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in ['SOLUSDT', 'DOGEUSDT']:
        bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        close = bars['close']

        # ── EXP HHHH: RSI EXTREME ──
        print(f"\n  --- EXP HHHH: RSI EXTREME ---")

        for period in [14, 30]:
            rsi = compute_rsi(close, period)
            print(f"  RSI({period}) stats: mean={rsi.mean():.1f}, P5={rsi.quantile(0.05):.1f}, "
                  f"P95={rsi.quantile(0.95):.1f}")

            for lo, hi in [(20, 80), (15, 85), (10, 90)]:
                oversold = rsi < lo
                overbought = rsi > hi

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[oversold]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)  # long when oversold
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[overbought]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)  # short when overbought
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  RSI({period}) <{lo}/>{ hi} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP IIII: BOLLINGER BAND ──
        print(f"\n  --- EXP IIII: BOLLINGER BAND ---")

        for bb_period in [20, 60]:
            bb_mid = close.rolling(bb_period, min_periods=bb_period).mean()
            bb_std = close.rolling(bb_period, min_periods=bb_period).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            bb_width = (bb_upper - bb_lower) / bb_mid * 10000  # in bps

            # BB touch: price outside bands → fade
            above_upper = close > bb_upper
            below_lower = close < bb_lower

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[above_upper]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # short when above upper band
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[below_lower]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # long when below lower band
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  BB({bb_period}) touch (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

            # BB squeeze: narrow bandwidth → breakout follow
            bw_roll = bb_width.rolling(120, min_periods=60).quantile(0.10)
            squeeze = bb_width < bw_roll

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[squeeze]:
                if lt and (ts - lt).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts)
                if idx + 1 < len(bars):
                    ret = close.iloc[idx+1] - close.iloc[idx]
                    is_long = ret > 0
                    t = sim_trade(bars, idx+1, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

            print(f"\n  BB({bb_period}) squeeze (follow breakout):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP JJJJ: EMA DIVERGENCE ──
        print(f"\n  --- EXP JJJJ: EMA DIVERGENCE ---")

        for fast, slow in [(5, 20), (10, 50)]:
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            ema_div = (ema_fast - ema_slow) / ema_slow * 10000  # in bps

            div_std = ema_div.rolling(60, min_periods=30).std()

            for z_thresh in [2, 3]:
                extreme_up = ema_div > z_thresh * div_std
                extreme_down = ema_div < -z_thresh * div_std

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[extreme_up]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)  # fade extreme up divergence
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[extreme_down]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)  # fade extreme down divergence
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  EMA({fast}/{slow}) div z>{z_thresh} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP KKKK: STOCHASTIC OSCILLATOR ──
        print(f"\n  --- EXP KKKK: STOCHASTIC OSCILLATOR ---")

        for k_period in [14, 30]:
            low_k = bars['low'].rolling(k_period, min_periods=k_period).min()
            high_k = bars['high'].rolling(k_period, min_periods=k_period).max()
            pct_k = (close - low_k) / (high_k - low_k + 1e-10) * 100

            for lo, hi in [(10, 90), (5, 95)]:
                oversold = pct_k < lo
                overbought = pct_k > hi

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[oversold]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[overbought]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  Stoch({k_period}) <{lo}/{hi} (fade):")
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
