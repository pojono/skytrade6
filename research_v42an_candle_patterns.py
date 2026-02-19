#!/usr/bin/env python3
"""
v42an: Candlestick Patterns + Price Action

EXP JJJJ2: Doji / Hammer / Engulfing Patterns
  - Doji: |close-open| < 10% of range → indecision → fade prior move
  - Hammer: long lower wick → reversal up
  - Engulfing: current bar fully engulfs prior bar → reversal

EXP KKKK2: Inside Bar Breakout
  - Inside bar: H < prev_H and L > prev_L
  - Breakout from inside bar → follow direction

EXP LLLL2: Pin Bar (Rejection Wick)
  - Upper wick > 2x body → rejection → short
  - Lower wick > 2x body → rejection → long

EXP MMMM2: Three-Bar Reversal
  - Three consecutive bars in same direction → fade
  - Combined with volatility filter for quality

ETHUSDT + SOLUSDT + DOGEUSDT + XRPUSDT, walk-forward, 88 days.
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


def main():
    out_file = 'results/v42an_candle_patterns.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42an: CANDLESTICK PATTERNS + PRICE ACTION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    grand = {}

    for symbol in symbols:
        bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        close = bars['close']
        opn = bars['open']
        high = bars['high']
        low = bars['low']
        ret_1m = close.pct_change()

        body = (close - opn).abs()
        bar_range = high - low + 1e-10
        upper_wick = high - pd.concat([close, opn], axis=1).max(axis=1)
        lower_wick = pd.concat([close, opn], axis=1).min(axis=1) - low

        # ── EXP JJJJ2: DOJI + HAMMER + ENGULFING ──
        print(f"\n  --- EXP JJJJ2: DOJI + HAMMER + ENGULFING ---")

        # Doji: body < 10% of range, after a move → fade
        doji = body / bar_range < 0.10
        prior_up = ret_1m.shift(1) > 0
        prior_down = ret_1m.shift(1) < 0

        # Doji after up move → short, after down move → long
        doji_short = doji & prior_up & (ret_1m.shift(1).abs() > ret_1m.abs().rolling(60).mean())
        doji_long = doji & prior_down & (ret_1m.shift(1).abs() > ret_1m.abs().rolling(60).mean())

        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[doji_long]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade(bars, idx, True)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        for ts in bars.index[doji_short]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade(bars, idx, False)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        train_trades.sort(key=lambda t: t['time'])
        test_trades.sort(key=lambda t: t['time'])

        print(f"\n  Doji after move (fade):")
        ts_r = pstats(train_trades, "TRAIN")
        te_r = pstats(test_trades, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
            grand[(symbol, 'doji')] = te_r

        # Hammer: lower wick > 2x body, small upper wick
        hammer = (lower_wick > 2 * body) & (upper_wick < body) & (bar_range > bar_range.rolling(60).mean())
        inv_hammer = (upper_wick > 2 * body) & (lower_wick < body) & (bar_range > bar_range.rolling(60).mean())

        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[hammer]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade(bars, idx, True)  # hammer → bullish
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        for ts in bars.index[inv_hammer]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade(bars, idx, False)  # inverted hammer → bearish
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        train_trades.sort(key=lambda t: t['time'])
        test_trades.sort(key=lambda t: t['time'])

        print(f"\n  Hammer/Inv Hammer:")
        ts_r = pstats(train_trades, "TRAIN")
        te_r = pstats(test_trades, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
            grand[(symbol, 'hammer')] = te_r

        # ── EXP KKKK2: INSIDE BAR BREAKOUT ──
        print(f"\n  --- EXP KKKK2: INSIDE BAR ---")

        prev_h = high.shift(1)
        prev_l = low.shift(1)
        inside_bar = (high < prev_h) & (low > prev_l)

        # Inside bar → next bar breakout direction
        # After inside bar, fade the breakout (mean-reversion)
        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[inside_bar]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            if idx + 1 >= len(bars): continue
            next_ret = ret_1m.iloc[idx + 1] if idx + 1 < len(ret_1m) else 0
            is_long = next_ret < 0  # fade breakout direction
            t = sim_trade(bars, idx + 1, is_long)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        print(f"\n  Inside bar breakout (fade):")
        ts_r = pstats(train_trades, "TRAIN")
        te_r = pstats(test_trades, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
            grand[(symbol, 'inside_bar')] = te_r

        # ── EXP LLLL2: PIN BAR (REJECTION WICK) ──
        print(f"\n  --- EXP LLLL2: PIN BAR ---")

        for wick_mult in [2, 3]:
            pin_bull = (lower_wick > wick_mult * body) & (bar_range > bar_range.rolling(60).mean() * 1.5)
            pin_bear = (upper_wick > wick_mult * body) & (bar_range > bar_range.rolling(60).mean() * 1.5)

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[pin_bull]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[pin_bear]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Pin bar wick>{wick_mult}x body (follow rejection):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'pin_bar_{wick_mult}x')] = te_r

        # ── EXP MMMM2: THREE-BAR REVERSAL ──
        print(f"\n  --- EXP MMMM2: THREE-BAR REVERSAL ---")

        three_up = (ret_1m > 0) & (ret_1m.shift(1) > 0) & (ret_1m.shift(2) > 0)
        three_down = (ret_1m < 0) & (ret_1m.shift(1) < 0) & (ret_1m.shift(2) < 0)

        # With volatility filter
        roll_std = ret_1m.rolling(60, min_periods=30).std()
        cum_3 = ret_1m + ret_1m.shift(1) + ret_1m.shift(2)

        for z_thresh in [1, 2]:
            strong_up = three_up & (cum_3 > z_thresh * roll_std * np.sqrt(3))
            strong_down = three_down & (cum_3 < -z_thresh * roll_std * np.sqrt(3))

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[strong_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade 3-bar up
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[strong_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade 3-bar down
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  3-bar reversal z>{z_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'3bar_z{z_thresh}')] = te_r

        del bars; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  GRAND SUMMARY (OOS only)")
    print(f"{'#'*80}")
    print(f"\n  {'Symbol':10s}  {'Signal':20s}  {'n':>5s}  {'WR':>6s}  {'Avg':>7s}  {'Total':>8s}  {'Sharpe':>7s}")
    print(f"  {'-'*65}")
    for (sym, sig), r in sorted(grand.items()):
        flag = "✅" if r['tot'] > 0 else "  "
        print(f"  {flag} {sym:10s}  {sig:20s}  {r['n']:5d}  {r['wr']:5.1f}%  "
              f"{r['avg']:+6.1f}bp  {r['tot']:+7.1f}%  {r['sharpe']:+7.0f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
