#!/usr/bin/env python3
"""
v42ao: Mean-Reversion Timing + Volatility Regime Detection

EXP NNNN2: Bollinger Band Width Squeeze → Breakout Fade
  - BB width at historical low → squeeze → expect breakout
  - After breakout from squeeze, fade it (MR)

EXP OOOO2: ATR Ratio (Short/Long)
  - ATR(5) / ATR(30) ratio
  - High ratio → recent vol spike → fade
  - Low ratio → calm before storm → wait

EXP PPPP2: Close Location Value (CLV)
  - CLV = (close - low) / (high - low) - 0.5
  - Extreme CLV → bar closed at extreme → fade

EXP QQQQ2: Consecutive Wick Direction
  - Multiple bars with same wick direction → exhaustion
  - 3+ bars with upper wick > body → selling pressure exhaustion → long

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
    out_file = 'results/v42ao_mr_timing.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ao: MR TIMING + VOL REGIME DETECTION")
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
        ret_1m = close.pct_change()
        high = bars['high']
        low = bars['low']
        opn = bars['open']

        # ── EXP NNNN2: BB WIDTH SQUEEZE → BREAKOUT FADE ──
        print(f"\n  --- EXP NNNN2: BB SQUEEZE BREAKOUT FADE ---")

        ma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std()
        bb_width = std20 / (ma20 + 1e-10)
        bb_width_pct = bb_width.rolling(120, min_periods=60).rank(pct=True)

        # Squeeze = BB width at bottom 10%
        squeeze = bb_width_pct < 0.10
        # After squeeze, price breaks out → fade
        squeeze_prev = squeeze.shift(1).fillna(False)
        breakout_up = squeeze_prev & (close > ma20 + 2 * std20)
        breakout_down = squeeze_prev & (close < ma20 - 2 * std20)

        train_trades = []; test_trades = []
        lt = None
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

        print(f"\n  BB squeeze breakout (fade):")
        ts_r = pstats(train_trades, "TRAIN")
        te_r = pstats(test_trades, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
            grand[(symbol, 'bb_squeeze')] = te_r

        # ── EXP OOOO2: ATR RATIO ──
        print(f"\n  --- EXP OOOO2: ATR RATIO ---")

        tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr5 = tr.rolling(5, min_periods=5).mean()
        atr30 = tr.rolling(30, min_periods=30).mean()
        atr_ratio = atr5 / (atr30 + 1e-10)

        atr_ratio_std = atr_ratio.rolling(60, min_periods=30).std()
        atr_ratio_avg = atr_ratio.rolling(60, min_periods=30).mean()

        for z_thresh in [2, 3]:
            high_ratio = (atr_ratio - atr_ratio_avg) > z_thresh * atr_ratio_std

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_ratio]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade vol spike
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  ATR ratio z>{z_thresh} (fade vol spike):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'atr_ratio_z{z_thresh}')] = te_r

        # ── EXP PPPP2: CLOSE LOCATION VALUE ──
        print(f"\n  --- EXP PPPP2: CLOSE LOCATION VALUE ---")

        bar_range = high - low + 1e-10
        clv = (close - low) / bar_range  # 0 = closed at low, 1 = closed at high

        # Rolling CLV
        clv_roll = clv.rolling(5, min_periods=3).mean()

        for thresh in [0.15, 0.10]:
            # CLV near 0 → closed at low → oversold → long
            # CLV near 1 → closed at high → overbought → short
            oversold = clv_roll < thresh
            overbought = clv_roll > (1 - thresh)

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

            print(f"\n  CLV(5) <{thresh}/{1-thresh:.2f} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'clv_{thresh}')] = te_r

        # ── EXP QQQQ2: CONSECUTIVE WICK DIRECTION ──
        print(f"\n  --- EXP QQQQ2: CONSECUTIVE WICK DIRECTION ---")

        body = (close - opn).abs()
        upper_wick = high - pd.concat([close, opn], axis=1).max(axis=1)
        lower_wick = pd.concat([close, opn], axis=1).min(axis=1) - low

        # Upper wick dominant
        uw_dom = upper_wick > body
        lw_dom = lower_wick > body

        for consec in [3, 5]:
            # N consecutive bars with upper wick > body → selling exhaustion → long
            uw_consec = uw_dom.rolling(consec, min_periods=consec).sum() == consec
            lw_consec = lw_dom.rolling(consec, min_periods=consec).sum() == consec

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[uw_consec]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # upper wick exhaustion → long
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[lw_consec]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # lower wick exhaustion → short
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  {consec} consec wick dominant (fade exhaustion):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'wick_consec_{consec}')] = te_r

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
