#!/usr/bin/env python3
"""
v42bk: CC Vol Ratio + Weighted Close Dev + Momentum Accel Ratio + Intrabar Reversal

EXP XXXX6: Close-to-Close Vol Ratio (short/long window)
  - Ratio of short-term CC vol to long-term CC vol
  - High ratio = vol expanding = overextended → fade
  - Low ratio = vol contracting = quiet → skip

EXP YYYY6: Weighted Close Deviation
  - Deviation of close from volume-weighted typical price
  - Typical price = (H+L+C)/3, weighted by range as proxy for volume
  - Extreme deviation → fade

EXP ZZZZ6: Momentum Acceleration Ratio
  - Ratio of recent momentum to older momentum (same window)
  - mom_ratio = mom(N) / mom(N).shift(N)
  - Extreme ratio = accelerating/decelerating → fade

EXP AAAA7: Intrabar Reversal Strength
  - How much price reversed within the bar: (high-close)/(high-low) for up bars
  - Rolling average of reversal strength
  - High reversal = rejection → MR

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
    out_file = 'results/v42bk_ccvol_wclose.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42bk: CC VOL RATIO + WEIGHTED CLOSE DEV + MOM ACCEL RATIO + INTRABAR REV")
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
        opn = bars['open']
        ret_1m = close.pct_change()

        # ── EXP XXXX6: CLOSE-TO-CLOSE VOL RATIO ──
        print(f"\n  --- EXP XXXX6: CC VOL RATIO ---")

        cc_vol_10 = ret_1m.rolling(10, min_periods=10).std()
        cc_vol_60 = ret_1m.rolling(60, min_periods=60).std()
        vol_ratio = cc_vol_10 / (cc_vol_60 + 1e-10)
        vr_pct = vol_ratio.rolling(120, min_periods=60).rank(pct=True)

        for thresh in [0.90, 0.95]:
            high_vr = vr_pct > thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_vr]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  CC vol ratio(10/60) >{thresh*100:.0f}th pct (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'ccvr_{thresh}')] = te_r

        cc_vol_20 = ret_1m.rolling(20, min_periods=20).std()
        vol_ratio2 = cc_vol_20 / (cc_vol_60 + 1e-10)
        vr2_pct = vol_ratio2.rolling(120, min_periods=60).rank(pct=True)

        for thresh in [0.90, 0.95]:
            high_vr2 = vr2_pct > thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_vr2]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  CC vol ratio(20/60) >{thresh*100:.0f}th pct (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'ccvr2_{thresh}')] = te_r

        # ── EXP YYYY6: WEIGHTED CLOSE DEVIATION ──
        print(f"\n  --- EXP YYYY6: WEIGHTED CLOSE DEVIATION ---")

        typical = (high + low + close) / 3
        bar_range = high - low

        for window in [20, 40]:
            wt_typical = (typical * bar_range).rolling(window, min_periods=window).sum() / \
                         (bar_range.rolling(window, min_periods=window).sum() + 1e-10)
            wc_dev = (close - wt_typical) / (close + 1e-10)
            wcd_pct = wc_dev.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.05, 0.10]:
                up_dev = wcd_pct > (1 - thresh)
                dn_dev = wcd_pct < thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[up_dev]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[dn_dev]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  Wt close dev({window}) <{thresh}/{1-thresh:.2f} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'wcd_{window}_{thresh}')] = te_r

        # ── EXP ZZZZ6: MOMENTUM ACCELERATION RATIO ──
        print(f"\n  --- EXP ZZZZ6: MOMENTUM ACCEL RATIO ---")

        for window in [10, 20]:
            mom_now = close - close.shift(window)
            mom_prev = close.shift(window) - close.shift(2*window)
            mom_ratio = mom_now / (mom_prev.abs() + 1e-10)
            mr_pct = mom_ratio.abs().rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                extreme_mr = mr_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[extreme_mr]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    mr_val = mom_ratio.iloc[idx]
                    is_long = mr_val < 0 or (mr_val > 0 and mom_now.iloc[idx] > 0)
                    is_long = mom_now.iloc[idx] < 0  # fade current momentum direction
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Mom accel({window}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'ma_{window}_{thresh}')] = te_r

        # ── EXP AAAA7: INTRABAR REVERSAL STRENGTH ──
        print(f"\n  --- EXP AAAA7: INTRABAR REVERSAL STRENGTH ---")

        # For up bars: reversal = (high - close) / range
        # For down bars: reversal = (close - low) / range
        up_bar = close > opn
        reversal = pd.Series(0.0, index=bars.index)
        reversal[up_bar] = (high[up_bar] - close[up_bar]) / (bar_range[up_bar] + 1e-10)
        reversal[~up_bar] = (close[~up_bar] - low[~up_bar]) / (bar_range[~up_bar] + 1e-10)

        for window in [20, 40]:
            rev_avg = reversal.rolling(window, min_periods=window).mean()
            rev_pct = rev_avg.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                high_rev = rev_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[high_rev]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Intrabar rev({window}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'ir_{window}_{thresh}')] = te_r

        del bars; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  GRAND SUMMARY (OOS only)")
    print(f"{'#'*80}")
    print(f"\n  {'Symbol':10s}  {'Signal':22s}  {'n':>5s}  {'WR':>6s}  {'Avg':>7s}  {'Total':>8s}  {'Sharpe':>7s}")
    print(f"  {'-'*68}")
    for (sym, sig), r in sorted(grand.items()):
        flag = "✅" if r['tot'] > 0 else "  "
        print(f"  {flag} {sym:10s}  {sig:22s}  {r['n']:5d}  {r['wr']:5.1f}%  "
              f"{r['avg']:+6.1f}bp  {r['tot']:+7.1f}%  {r['sharpe']:+7.0f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
