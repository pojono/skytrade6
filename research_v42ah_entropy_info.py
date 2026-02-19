#!/usr/bin/env python3
"""
v42ah: Information-Theoretic Signals + Entropy Patterns

EXP LLLL: Return Entropy Signal
  - Rolling Shannon entropy of discretized returns
  - Low entropy → predictable regime → trade the pattern
  - High entropy → random regime → avoid or fade

EXP MMMM: Directional Persistence Score
  - Rolling ratio of |cumulative return| / sum(|individual returns|)
  - High ratio → persistent direction → follow
  - Low ratio → choppy → fade

EXP NNNN: Gap-Fill Signal
  - When open != prev close (gap) → fade the gap
  - Larger gaps = stronger signal

EXP OOOO: Momentum Exhaustion
  - When cumulative 30-min return is extreme but last 5 bars are flat
  - = momentum exhaustion → reversal imminent

SOLUSDT + DOGEUSDT + XRPUSDT, walk-forward, 88 days.
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


def rolling_entropy(series, window=30, bins=10):
    """Rolling Shannon entropy of discretized returns."""
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(window, len(vals)):
        chunk = vals[i-window:i]
        if np.isnan(chunk).any(): continue
        # Discretize into bins
        hist, _ = np.histogram(chunk, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        result.iloc[i] = entropy
    return result


def main():
    out_file = 'results/v42ah_entropy_info.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ah: INFORMATION-THEORETIC SIGNALS + ENTROPY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in ['SOLUSDT', 'DOGEUSDT', 'XRPUSDT']:
        bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        ret_1m = bars['close'].pct_change()

        # ── EXP LLLL: RETURN ENTROPY ──
        print(f"\n  --- EXP LLLL: RETURN ENTROPY ---")
        print(f"  Computing entropy (slow)...", flush=True)

        entropy = rolling_entropy(ret_1m, window=30, bins=8)
        valid = entropy.dropna()
        print(f"  Entropy stats: mean={valid.mean():.3f}, P10={valid.quantile(0.10):.3f}, "
              f"P90={valid.quantile(0.90):.3f}")

        # Low entropy → predictable → fade the current direction
        for pct in [10, 20]:
            thresh = valid.quantile(pct/100)
            low_ent = entropy < thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[low_ent]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade in predictable regime
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Low entropy P{pct} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP MMMM: DIRECTIONAL PERSISTENCE ──
        print(f"\n  --- EXP MMMM: DIRECTIONAL PERSISTENCE ---")

        cum_ret_30 = ret_1m.rolling(30, min_periods=15).sum()
        sum_abs_30 = ret_1m.abs().rolling(30, min_periods=15).sum()
        persistence = cum_ret_30.abs() / (sum_abs_30 + 1e-10)

        print(f"  Persistence stats: mean={persistence.mean():.3f}, P90={persistence.quantile(0.90):.3f}")

        # High persistence → trending → follow
        # Low persistence → choppy → fade
        for p_thresh in [0.3, 0.4, 0.5]:
            high_persist = persistence > p_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_persist]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = cum_ret_30.iloc[idx] if idx < len(cum_ret_30) else 0
                is_long = r < 0  # fade persistent move (mean-reversion)
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Persistence >{p_thresh} (fade persistent move):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP NNNN: GAP-FILL ──
        print(f"\n  --- EXP NNNN: GAP-FILL ---")

        gap = (bars['open'] - bars['close'].shift(1)) / bars['close'].shift(1) * 10000  # bps

        for gap_thresh in [5, 10, 20]:
            gap_up = gap > gap_thresh
            gap_down = gap < -gap_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[gap_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade gap up
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[gap_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade gap down
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Gap >{gap_thresh}bps (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP OOOO: MOMENTUM EXHAUSTION ──
        print(f"\n  --- EXP OOOO: MOMENTUM EXHAUSTION ---")

        cum_ret_30 = bars['close'].pct_change(30)
        recent_ret_5 = bars['close'].pct_change(5)
        cum_std = cum_ret_30.rolling(60, min_periods=30).std()
        recent_std = recent_ret_5.rolling(60, min_periods=30).std()

        # Big 30-min move but flat last 5 bars → exhaustion
        for z_thresh in [2, 3]:
            exhaustion_up = (cum_ret_30 > z_thresh * cum_std) & (recent_ret_5.abs() < 0.3 * recent_std)
            exhaustion_down = (cum_ret_30 < -z_thresh * cum_std) & (recent_ret_5.abs() < 0.3 * recent_std)

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[exhaustion_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade exhausted up move
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[exhaustion_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade exhausted down move
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Momentum exhaustion z>{z_thresh} (fade):")
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
