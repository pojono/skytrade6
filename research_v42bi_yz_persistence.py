#!/usr/bin/env python3
"""
v42bi: Yang-Zhang Vol + Momentum Persistence + Return Dispersion + Bar Body Ratio

EXP PPPP6: Yang-Zhang Volatility
  - YZ = sqrt(overnight_var + open-close_var + Rogers-Satchell_var)
  - Most efficient OHLC vol estimator
  - Extreme YZ vol → fade

EXP QQQQ6: Momentum Persistence (streak length)
  - Count consecutive bars in same direction
  - Long streaks = persistent momentum → fade exhaustion

EXP RRRR6: Return Dispersion (std of sub-period returns)
  - Split N-bar window into sub-periods, compute std of sub-period returns
  - High dispersion = erratic → fade
  - Low dispersion = smooth trend → skip

EXP SSSS6: Bar Body Ratio (|close-open| / range)
  - High body ratio = strong conviction bars
  - Rolling average of body ratio extremes → fade

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
    out_file = 'results/v42bi_yz_persistence.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42bi: YANG-ZHANG VOL + MOM PERSISTENCE + RET DISPERSION + BODY RATIO")
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

        # ── EXP PPPP6: YANG-ZHANG VOLATILITY ──
        print(f"\n  --- EXP PPPP6: YANG-ZHANG VOLATILITY ---")

        log_oc = np.log(opn / (close.shift(1) + 1e-10))  # overnight
        log_co = np.log(close / (opn + 1e-10))  # open-to-close
        log_ho = np.log(high / (opn + 1e-10))
        log_lo = np.log(low / (opn + 1e-10))
        rs_component = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        for window in [30, 60]:
            oc_var = log_oc.rolling(window, min_periods=window).var()
            co_var = log_co.rolling(window, min_periods=window).var()
            rs_var = rs_component.rolling(window, min_periods=window).mean()
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            yz_vol = np.sqrt(oc_var + k * co_var + (1 - k) * rs_var.clip(lower=0))
            yz_pct = yz_vol.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                high_yz = yz_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[high_yz]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  YZ vol({window}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'yz_{window}_{thresh}')] = te_r

        # ── EXP QQQQ6: MOMENTUM PERSISTENCE ──
        print(f"\n  --- EXP QQQQ6: MOMENTUM PERSISTENCE ---")

        up_bar = (close > opn).astype(int)
        dn_bar = (close < opn).astype(int)

        # Count consecutive same-direction bars
        streak = pd.Series(0, index=bars.index, dtype=float)
        s = 0
        prev_dir = 0
        for i in range(len(bars)):
            if up_bar.iloc[i]:
                if prev_dir == 1: s += 1
                else: s = 1
                prev_dir = 1
            elif dn_bar.iloc[i]:
                if prev_dir == -1: s += 1
                else: s = 1
                prev_dir = -1
            else:
                s = 0; prev_dir = 0
            streak.iloc[i] = s * prev_dir

        streak_abs = streak.abs()
        streak_pct = streak_abs.rolling(120, min_periods=60).rank(pct=True)

        for thresh in [0.90, 0.95]:
            long_streak = streak_pct > thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[long_streak]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                is_long = streak.iloc[idx] < 0  # fade the streak direction
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Streak >{thresh*100:.0f}th pct (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'streak_{thresh}')] = te_r

        # ── EXP RRRR6: RETURN DISPERSION ──
        print(f"\n  --- EXP RRRR6: RETURN DISPERSION ---")

        for window, sub in [(20, 5), (40, 10)]:
            # Compute std of sub-period returns
            sub_rets = []
            for s_start in range(0, window, sub):
                sr = close.pct_change(sub).shift(window - s_start - sub)
                sub_rets.append(sr)
            sub_df = pd.concat(sub_rets, axis=1)
            ret_disp = sub_df.std(axis=1)
            rd_pct = ret_disp.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                high_disp = rd_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[high_disp]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Ret disp({window}/{sub}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'rd_{window}_{thresh}')] = te_r

        # ── EXP SSSS6: BAR BODY RATIO ──
        print(f"\n  --- EXP SSSS6: BAR BODY RATIO ---")

        body = (close - opn).abs()
        bar_range = high - low
        body_ratio = body / (bar_range + 1e-10)

        for window in [20, 40]:
            br_avg = body_ratio.rolling(window, min_periods=window).mean()
            br_pct = br_avg.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                high_body = br_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[high_body]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Body ratio({window}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'br_{window}_{thresh}')] = te_r

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
