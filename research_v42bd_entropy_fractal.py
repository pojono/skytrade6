#!/usr/bin/env python3
"""
v42bd: Return Entropy + Fractal Dimension Proxy + Momentum Quality + Range Expansion Rate

EXP VVVV5: Return Entropy (Shannon)
  - Discretize returns into bins, compute Shannon entropy
  - High entropy = random/unpredictable → less tradeable
  - Low entropy = structured/predictable → more MR opportunity

EXP WWWW5: Fractal Dimension Proxy (Higuchi-like)
  - FD = log(path length) / log(N)
  - High FD ≈ 2 = noisy/random
  - Low FD ≈ 1 = smooth/trending → fade extreme trends

EXP XXXX5: Momentum Quality (consistency)
  - What fraction of sub-periods agree with overall momentum?
  - High quality = consistent trend → fade exhaustion
  - Low quality = choppy → avoid

EXP YYYY5: Range Expansion Rate
  - How fast is the N-bar range expanding?
  - Fast expansion = breakout → fade after exhaustion
  - Slow expansion = consolidation

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


def shannon_entropy(x):
    """Fast Shannon entropy of discretized returns."""
    bins = np.linspace(-0.005, 0.005, 11)
    counts = np.histogram(x, bins=bins)[0]
    probs = counts / (counts.sum() + 1e-10)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def fractal_dim_proxy(prices):
    """Fast fractal dimension proxy using path length."""
    n = len(prices)
    if n < 5: return 1.5
    path = np.sum(np.abs(np.diff(prices)))
    span = np.abs(prices[-1] - prices[0]) + 1e-10
    return np.log(path / span + 1) / np.log(n)


def main():
    out_file = 'results/v42bd_entropy_fractal.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42bd: ENTROPY + FRACTAL DIM + MOM QUALITY + RANGE EXPANSION RATE")
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

        # ── EXP VVVV5: RETURN ENTROPY ──
        print(f"\n  --- EXP VVVV5: RETURN ENTROPY ---")

        for window in [30, 60]:
            entropy = ret_1m.rolling(window, min_periods=window).apply(shannon_entropy, raw=True)
            ent_pct = entropy.rolling(120, min_periods=60).rank(pct=True)

            # Low entropy = structured → fade with confidence
            for thresh in [0.10, 0.20]:
                low_ent = ent_pct < thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[low_ent]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Entropy({window}) <{thresh*100:.0f}th pct (fade structured):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'ent_{window}_{thresh}')] = te_r

        # ── EXP WWWW5: FRACTAL DIMENSION PROXY ──
        print(f"\n  --- EXP WWWW5: FRACTAL DIMENSION PROXY ---")

        for window in [20, 40]:
            fd = close.rolling(window, min_periods=window).apply(
                lambda x: fractal_dim_proxy(x.values), raw=False)
            fd_pct = fd.rolling(120, min_periods=60).rank(pct=True)

            # High FD = noisy → fade recent move
            for thresh in [0.90, 0.95]:
                high_fd = fd_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[high_fd]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Fractal({window}) >{thresh*100:.0f}th pct (fade noisy):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'fd_{window}_{thresh}')] = te_r

        # ── EXP XXXX5: MOMENTUM QUALITY ──
        print(f"\n  --- EXP XXXX5: MOMENTUM QUALITY ---")

        for total_w, sub_w in [(30, 5), (60, 10)]:
            overall_mom = close.pct_change(total_w)
            n_subs = total_w // sub_w
            # Count how many sub-periods agree with overall direction
            agree_count = pd.Series(0.0, index=close.index)
            for i in range(n_subs):
                sub_mom = close.pct_change(sub_w).shift(i * sub_w)
                agree = ((sub_mom > 0) == (overall_mom > 0)).astype(float)
                agree_count += agree
            mom_quality = agree_count / n_subs
            mq_pct = mom_quality.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                high_mq = mq_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[high_mq]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = overall_mom.iloc[idx] if idx < len(overall_mom) else 0
                    is_long = r < 0  # fade consistent trend
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Mom quality({total_w}/{sub_w}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'mq_{total_w}_{sub_w}_{thresh}')] = te_r

        # ── EXP YYYY5: RANGE EXPANSION RATE ──
        print(f"\n  --- EXP YYYY5: RANGE EXPANSION RATE ---")

        for window in [20, 40]:
            roll_range = high.rolling(window, min_periods=window).max() - \
                         low.rolling(window, min_periods=window).min()
            range_rate = roll_range.pct_change(5)  # 5-bar rate of range expansion
            rr_pct = range_rate.rolling(120, min_periods=60).rank(pct=True)

            for thresh in [0.90, 0.95]:
                fast_expand = rr_pct > thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[fast_expand]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                    is_long = r < 0
                    t = sim_trade(bars, idx, is_long)
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                print(f"\n  Range rate({window}) >{thresh*100:.0f}th pct (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'rrate_{window}_{thresh}')] = te_r

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
