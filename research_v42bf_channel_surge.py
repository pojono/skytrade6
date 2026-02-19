#!/usr/bin/env python3
"""
v42bf: Price Channel Position + Volume Surge + Trend Strength Index + Close Location Value

EXP DDDD6: Price Channel Position (normalized within Donchian channel)
  - Position = (close - channel_low) / (channel_high - channel_low)
  - Near 1 = at top of channel → fade
  - Near 0 = at bottom of channel → fade

EXP EEEE6: Volume Surge Ratio (range spike as vol proxy)
  - Current range / avg range = surge ratio
  - Extreme surge → overextended → fade

EXP FFFF6: Trend Strength Index (ADX-like)
  - Ratio of net directional move to total path length
  - High TSI = strong trend → fade exhaustion
  - Low TSI = choppy → MR opportunity

EXP GGGG6: Close Location Value (CLV)
  - CLV = (2*close - high - low) / (high - low)
  - Rolling CLV extremes = persistent directional pressure → fade

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
    out_file = 'results/v42bf_channel_surge.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42bf: CHANNEL POS + VOL SURGE + TREND STRENGTH + CLV")
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

        # ── EXP DDDD6: PRICE CHANNEL POSITION ──
        print(f"\n  --- EXP DDDD6: PRICE CHANNEL POSITION ---")

        for window in [30, 60]:
            ch_high = high.rolling(window, min_periods=window).max()
            ch_low = low.rolling(window, min_periods=window).min()
            ch_pos = (close - ch_low) / (ch_high - ch_low + 1e-10)

            for thresh in [0.05, 0.10]:
                near_top = ch_pos > (1 - thresh)
                near_bot = ch_pos < thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[near_top]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[near_bot]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  Channel({window}) <{thresh}/{1-thresh:.2f} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'ch_{window}_{thresh}')] = te_r

        # ── EXP EEEE6: VOLUME SURGE RATIO ──
        print(f"\n  --- EXP EEEE6: VOLUME SURGE RATIO ---")

        bar_range = high - low
        for avg_w in [20, 40]:
            avg_range = bar_range.rolling(avg_w, min_periods=avg_w).mean()
            surge = bar_range / (avg_range + 1e-10)
            surge_pct = surge.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                big_surge = surge_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[big_surge]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Vol surge(avg {avg_w}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'surge_{avg_w}_{thresh}')] = te_r

        # ── EXP FFFF6: TREND STRENGTH INDEX ──
        print(f"\n  --- EXP FFFF6: TREND STRENGTH INDEX ---")

        for window in [20, 40]:
            net_move = (close - close.shift(window)).abs()
            path_len = ret_1m.abs().rolling(window, min_periods=window).sum() * close
            tsi = net_move / (path_len + 1e-10)
            tsi_pct = tsi.rolling(120, min_periods=60).rank(pct=True)

            # High TSI = strong trend → fade
            for thresh in [0.90, 0.95]:
                strong_trend = tsi_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[strong_trend]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    mom = close.iloc[idx] - close.iloc[idx - window] if idx >= window else 0
                    is_long = mom < 0  # fade the trend direction
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  TSI({window}) >{thresh*100:.0f}th pct (fade trend):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'tsi_{window}_{thresh}')] = te_r

        # ── EXP GGGG6: CLOSE LOCATION VALUE ──
        print(f"\n  --- EXP GGGG6: CLOSE LOCATION VALUE ---")

        clv = (2 * close - high - low) / (high - low + 1e-10)

        for window in [20, 40]:
            clv_avg = clv.rolling(window, min_periods=window).mean()
            clv_pct = clv_avg.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.05, 0.10]:
                clv_up = clv_pct > (1 - thresh)
                clv_dn = clv_pct < thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[clv_up]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)  # fade persistent up CLV
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[clv_dn]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)  # fade persistent down CLV
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  CLV({window}) <{thresh}/{1-thresh:.2f} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'clv_{window}_{thresh}')] = te_r

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
