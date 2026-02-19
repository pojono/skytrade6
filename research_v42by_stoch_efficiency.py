#!/usr/bin/env python3
"""
v42by: Stochastic Velocity + Price Efficiency Ratio + Weighted Mom Ratio + Return Distribution Width

EXP BBBB9: Stochastic Velocity (rate of change of stochastic %K)
  - %K = (close - low_N) / (high_N - low_N) * 100
  - stoch_vel = %K - %K.shift(N)
  - Extreme stoch velocity = rapid overbought/oversold → fade

EXP CCCC9: Price Efficiency Ratio (displacement / path length)
  - efficiency = abs(close - close.shift(N)) / sum(abs(ret), N)
  - Low efficiency = choppy/MR, high = trending
  - Extreme low efficiency → fade (MR environment)

EXP DDDD9: Weighted Momentum Ratio (volume-weighted vs equal-weighted)
  - Since we only have price, use range-weighted momentum
  - wt_mom = sum(ret * range_pct, N) / sum(range_pct, N)
  - Compare to simple mom: divergence = signal
  - Extreme divergence → fade

EXP EEEE9: Return Distribution Width (IQR of recent returns)
  - iqr = rolling Q75 - Q25 of returns
  - High IQR = wide distribution = volatile
  - Extreme IQR → fade

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
    out_file = 'results/v42by_stoch_efficiency.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42by: STOCH VEL + PRICE EFFICIENCY + WT MOM RATIO + RET DIST WIDTH")
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
        bar_range = high - low

        # ── EXP BBBB9: STOCHASTIC VELOCITY ──
        print(f"\n  --- EXP BBBB9: STOCHASTIC VELOCITY ---")

        for window in [14, 30]:
            roll_low = low.rolling(window, min_periods=window).min()
            roll_high = high.rolling(window, min_periods=window).max()
            pctk = (close - roll_low) / (roll_high - roll_low + 1e-10) * 100

            for vel_lag in [3, 5]:
                stoch_vel = pctk - pctk.shift(vel_lag)
                sv_pct = stoch_vel.abs().rolling(120, min_periods=60).rank(pct=True)

                for thresh in [0.90, 0.95]:
                    extreme_sv = sv_pct > thresh

                    train_trades = []; test_trades = []
                    lt = None
                    for ts in bars.index[extreme_sv]:
                        if lt and (ts - lt).total_seconds() < 60: continue
                        idx = bars.index.get_loc(ts)
                        is_long = stoch_vel.iloc[idx] < 0
                        t = sim_trade(bars, idx, is_long)
                        if t:
                            if ts < split_ts: train_trades.append(t)
                            else: test_trades.append(t)
                            lt = ts

                    print(f"\n  Stoch vel({window},{vel_lag}) >{thresh*100:.0f}th pct (fade):")
                    ts_r = pstats(train_trades, "TRAIN")
                    te_r = pstats(test_trades, "TEST")
                    if ts_r and te_r:
                        oos = "✅" if te_r['tot'] > 0 else "❌"
                        print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                        grand[(symbol, f'sv_{window}_{vel_lag}_{thresh}')] = te_r

        # ── EXP CCCC9: PRICE EFFICIENCY RATIO ──
        print(f"\n  --- EXP CCCC9: PRICE EFFICIENCY RATIO ---")

        abs_ret = ret_1m.abs()

        for window in [20, 40]:
            displacement = (close - close.shift(window)).abs()
            path_length = abs_ret.rolling(window, min_periods=window).sum() * close
            efficiency = displacement / (path_length + 1e-10)
            # Low efficiency = choppy → MR opportunity
            eff_pct = efficiency.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.05, 0.10]:
                low_eff = eff_pct < thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[low_eff]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Efficiency({window}) <{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'eff_{window}_{thresh}')] = te_r

        # ── EXP DDDD9: WEIGHTED MOMENTUM RATIO ──
        print(f"\n  --- EXP DDDD9: WEIGHTED MOMENTUM RATIO ---")

        range_pct = bar_range / (close + 1e-10)

        for window in [20, 40]:
            wt_mom = (ret_1m * range_pct).rolling(window, min_periods=window).sum() / \
                     (range_pct.rolling(window, min_periods=window).sum() + 1e-10)
            simple_mom = ret_1m.rolling(window, min_periods=window).mean()
            divergence = wt_mom - simple_mom
            div_pct = divergence.abs().rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                extreme_div = div_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[extreme_div]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    is_long = divergence.iloc[idx] < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Wt mom div({window}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'wm_{window}_{thresh}')] = te_r

        # ── EXP EEEE9: RETURN DISTRIBUTION WIDTH (IQR) ──
        print(f"\n  --- EXP EEEE9: RETURN DISTRIBUTION WIDTH ---")

        for window in [30, 60]:
            q75 = ret_1m.rolling(window, min_periods=window).quantile(0.75)
            q25 = ret_1m.rolling(window, min_periods=window).quantile(0.25)
            iqr = q75 - q25
            iqr_pct = iqr.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                high_iqr = iqr_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[high_iqr]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Ret IQR({window}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'iqr_{window}_{thresh}')] = te_r

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
