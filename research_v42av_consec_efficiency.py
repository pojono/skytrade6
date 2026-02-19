#!/usr/bin/env python3
"""
v42av: Consecutive Bar Patterns + Efficiency + Close-Close Vol

EXP PPPP4: Consecutive Same-Direction Bars
  - Count consecutive up/down bars
  - N consecutive up → exhaustion → fade
  - Different from trend exhaustion: strict consecutive, not ratio

EXP QQQQ4: Price Efficiency Ratio (Fractal Efficiency)
  - Net displacement / total path length over N bars
  - High efficiency = trending (straight line)
  - Low efficiency = choppy (random walk)
  - High efficiency + extreme move → fade

EXP RRRR4: Close-to-Close vs High-Low Volatility
  - CC vol = std of close-to-close returns
  - HL vol = Parkinson estimator
  - Ratio CC/HL: high = jumpy, low = smooth
  - Extreme ratio → regime change signal

EXP SSSS4: Cumulative Return Deviation
  - Cumulative return over N bars vs expected (0)
  - Z-score of cumulative return
  - Extreme cumulative deviation → fade

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


def count_consecutive(series):
    """Count consecutive same-sign values ending at each point."""
    signs = np.sign(series.values)
    counts = np.zeros(len(signs))
    for i in range(len(signs)):
        if i == 0 or signs[i] == 0:
            counts[i] = 0
        elif signs[i] == signs[i-1]:
            counts[i] = counts[i-1] + 1
        else:
            counts[i] = 1
    return pd.Series(counts, index=series.index)


def main():
    out_file = 'results/v42av_consec_efficiency.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42av: CONSECUTIVE BARS + EFFICIENCY + CC/HL VOL + CUM RETURN")
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

        # ── EXP PPPP4: CONSECUTIVE SAME-DIRECTION BARS ──
        print(f"\n  --- EXP PPPP4: CONSECUTIVE BARS ---")

        consec = count_consecutive(ret_1m)

        for n_consec in [4, 6]:
            consec_up = (consec >= n_consec) & (ret_1m > 0)
            consec_down = (consec >= n_consec) & (ret_1m < 0)

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[consec_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade consecutive up
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[consec_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade consecutive down
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Consecutive >={n_consec} bars (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'consec_{n_consec}')] = te_r

        # ── EXP QQQQ4: PRICE EFFICIENCY RATIO ──
        print(f"\n  --- EXP QQQQ4: PRICE EFFICIENCY ---")

        for window in [15, 30]:
            # Net displacement
            net_disp = (close - close.shift(window)).abs()
            # Total path length
            path_len = ret_1m.abs().rolling(window, min_periods=window).sum() * close
            efficiency = net_disp / (path_len + 1e-10)

            eff_pct = efficiency.rolling(120, min_periods=60).rank(pct=True)

            # High efficiency + extreme move → fade
            high_eff = eff_pct > 0.90
            price_up = close > close.shift(window)
            price_down = close < close.shift(window)

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_eff & price_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[high_eff & price_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Efficiency({window}) >90th + extreme (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'efficiency_{window}')] = te_r

        # ── EXP RRRR4: CC/HL VOL RATIO ──
        print(f"\n  --- EXP RRRR4: CC/HL VOL RATIO ---")

        cc_vol = ret_1m.rolling(30, min_periods=15).std()
        # Parkinson HL vol
        log_hl = np.log(high / low + 1e-10)
        hl_vol = (log_hl**2 / (4 * np.log(2))).rolling(30, min_periods=15).mean().apply(np.sqrt)

        vol_ratio = cc_vol / (hl_vol + 1e-10)
        vr_pct = vol_ratio.rolling(120, min_periods=60).rank(pct=True)

        for thresh in [0.90, 0.95]:
            # High CC/HL ratio → jumpy regime → fade
            high_ratio = vr_pct > thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_ratio]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  CC/HL vol ratio >{thresh*100:.0f}th pct (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'cchl_{thresh}')] = te_r

        # ── EXP SSSS4: CUMULATIVE RETURN DEVIATION ──
        print(f"\n  --- EXP SSSS4: CUMULATIVE RETURN DEVIATION ---")

        for window in [15, 30]:
            cum_ret = ret_1m.rolling(window, min_periods=window).sum()
            cr_avg = cum_ret.rolling(120, min_periods=60).mean()
            cr_std = cum_ret.rolling(120, min_periods=60).std()
            cr_z = (cum_ret - cr_avg) / (cr_std + 1e-10)

            for z_thresh in [2, 3]:
                extreme_up = cr_z > z_thresh
                extreme_down = cr_z < -z_thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[extreme_up]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, False)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[extreme_down]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade(bars, idx, True)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  Cum ret({window}) z>{z_thresh} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'cumret_{window}_z{z_thresh}')] = te_r

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
