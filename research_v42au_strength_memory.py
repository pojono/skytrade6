#!/usr/bin/env python3
"""
v42au: Relative Strength + Price Memory + Vol Breakout + Trend Exhaustion

EXP LLLL4: Price Memory (Revisit Level)
  - Price returns to a level it was at N bars ago
  - "Memory" = close near close[N] → support/resistance
  - Fade the revisit

EXP MMMM4: Volatility Breakout (Squeeze Release)
  - Rolling vol at historical low → squeeze
  - First big move after squeeze → follow (not fade)
  - Opposite of most signals: momentum follow

EXP NNNN4: Trend Exhaustion Index
  - Count bars in same direction / total bars in window
  - Very high ratio → exhaustion → fade
  - Combines direction persistence with magnitude

EXP OOOO4: Range Contraction → Expansion
  - Rolling range (high-low) contracting
  - After contraction, first expansion bar → fade the expansion

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
    out_file = 'results/v42au_strength_memory.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42au: PRICE MEMORY + VOL BREAKOUT + TREND EXHAUST + RANGE")
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
        high = bars['high']
        low = bars['low']
        ret_1m = close.pct_change()

        # ── EXP LLLL4: PRICE MEMORY ──
        print(f"\n  --- EXP LLLL4: PRICE MEMORY ---")

        for lookback in [30, 60]:
            # Price near where it was N bars ago (within 0.05%)
            past_price = close.shift(lookback)
            price_diff = ((close - past_price) / past_price).abs()
            near_past = price_diff < 0.0005  # within 5 bps

            # Direction: was it above or below?
            came_from_above = close.shift(lookback // 2) > close  # price dropped to revisit
            came_from_below = close.shift(lookback // 2) < close  # price rose to revisit

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[near_past & came_from_above]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # support revisit → long
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[near_past & came_from_below]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # resistance revisit → short
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Price memory ({lookback}m) revisit:")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'memory_{lookback}')] = te_r

        # ── EXP MMMM4: VOL BREAKOUT (SQUEEZE RELEASE) ──
        print(f"\n  --- EXP MMMM4: VOL BREAKOUT ---")

        rvol = ret_1m.rolling(30, min_periods=15).std()
        rvol_pct = rvol.rolling(120, min_periods=60).rank(pct=True)

        # Low vol → squeeze
        squeeze = rvol_pct < 0.10
        squeeze_prev = squeeze.shift(1).fillna(False)

        # First big move after squeeze
        ret_z = ret_1m / (rvol + 1e-10)

        for z_thresh in [2, 3]:
            breakout_up = squeeze_prev & (ret_z > z_thresh)
            breakout_down = squeeze_prev & (ret_z < -z_thresh)

            train_trades = []; test_trades = []
            lt = None
            # Follow breakout (not fade)
            for ts in bars.index[breakout_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade breakout up
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[breakout_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade breakout down
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Vol breakout z>{z_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'vol_break_z{z_thresh}')] = te_r

        # ── EXP NNNN4: TREND EXHAUSTION INDEX ──
        print(f"\n  --- EXP NNNN4: TREND EXHAUSTION ---")

        up_bar = (ret_1m > 0).astype(int)
        down_bar = (ret_1m < 0).astype(int)

        for window in [15, 30]:
            up_ratio = up_bar.rolling(window, min_periods=window).mean()
            down_ratio = down_bar.rolling(window, min_periods=window).mean()

            # Extreme up ratio → exhaustion → fade
            for thresh in [0.75, 0.80]:
                exhaust_up = up_ratio > thresh
                exhaust_down = down_ratio > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[exhaust_up]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[exhaust_down]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  Trend exhaust({window}) >{thresh} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'exhaust_{window}_{thresh}')] = te_r

        # ── EXP OOOO4: RANGE CONTRACTION → EXPANSION ──
        print(f"\n  --- EXP OOOO4: RANGE CONTRACTION/EXPANSION ---")

        bar_range = high - low
        range_avg = bar_range.rolling(30, min_periods=15).mean()
        range_pct = bar_range.rolling(120, min_periods=60).rank(pct=True)

        # Range was contracting (low pct) → now expanding
        was_low = range_pct.shift(1) < 0.10
        now_high = bar_range > 2 * range_avg

        expansion = was_low & now_high

        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[expansion]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
            is_long = r < 0  # fade the expansion
            t = sim_trade(bars, idx, is_long)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        print(f"\n  Range contraction→expansion (fade):")
        ts_r = pstats(train_trades, "TRAIN")
        te_r = pstats(test_trades, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
            grand[(symbol, 'range_expand')] = te_r

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
