#!/usr/bin/env python3
"""
v42ab: Cross-Symbol Lead-Lag + Intraday Seasonality

EXP SSS: Cross-Symbol Lead-Lag
  - Does ETH lead SOL/DOGE/XRP by 1-5 minutes?
  - If ETH moves first, trade the lagging symbol
  - Test: ETH→SOL, ETH→DOGE, ETH→XRP, SOL→DOGE

EXP TTT: Intraday Seasonality (Hour-of-Day)
  - Some hours have consistent directional bias
  - Trade the bias: if hour X is historically bullish, go long at start of hour X
  - Walk-forward: use first 60d to learn patterns, test on last 28d

EXP UUU: Weekend/Weekday Effect
  - Crypto trades 24/7 — are weekends different?
  - Test: fade Friday close moves on Monday open

All 4 symbols, walk-forward, 88 days.
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
    out_file = 'results/v42ab_cross_sym_seasonality.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ab: CROSS-SYMBOL LEAD-LAG + INTRADAY SEASONALITY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load all bars
    all_bars = {}
    for sym in symbols:
        all_bars[sym] = load_bars_chunked(sym, all_dates, chunk_days=10)
        gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # EXP SSS: CROSS-SYMBOL LEAD-LAG
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP SSS: CROSS-SYMBOL LEAD-LAG")
    print(f"{'#'*80}")

    # Compute 1-min returns for all symbols
    rets = {}
    for sym in symbols:
        rets[sym] = all_bars[sym]['close'].pct_change()

    # Cross-correlation analysis
    print(f"\n  Cross-correlation (lag in minutes):")
    for leader in ['ETHUSDT', 'SOLUSDT']:
        for follower in symbols:
            if leader == follower: continue
            common = rets[leader].index.intersection(rets[follower].index)
            if len(common) < 1000: continue
            r_lead = rets[leader].reindex(common)
            r_follow = rets[follower].reindex(common)

            corrs = []
            for lag in range(0, 6):
                if lag == 0:
                    c = r_lead.corr(r_follow)
                else:
                    c = r_lead.iloc[:-lag].reset_index(drop=True).corr(
                        r_follow.iloc[lag:].reset_index(drop=True))
                corrs.append(c)
            print(f"    {leader:10s} → {follower:10s}  lag0={corrs[0]:.4f}  "
                  f"lag1={corrs[1]:.4f}  lag2={corrs[2]:.4f}  lag3={corrs[3]:.4f}  "
                  f"lag5={corrs[5]:.4f}")

    # Trade the lead-lag: when ETH moves big, trade follower in same direction
    for leader, follower in [('ETHUSDT', 'SOLUSDT'), ('ETHUSDT', 'DOGEUSDT'),
                              ('ETHUSDT', 'XRPUSDT'), ('SOLUSDT', 'DOGEUSDT')]:
        common = rets[leader].index.intersection(rets[follower].index)
        r_lead = rets[leader].reindex(common)
        bars_f = all_bars[follower]

        for sigma in [2, 3]:
            roll_std = r_lead.rolling(60, min_periods=30).std()
            big_up = r_lead > sigma * roll_std
            big_down = r_lead < -sigma * roll_std

            train_trades = []; test_trades = []
            lt = None

            # ETH big up → follower should follow → go long follower
            for ts in common[big_up.reindex(common, fill_value=False)]:
                if lt and (ts - lt).total_seconds() < 60: continue
                if ts not in bars_f.index: continue
                idx = bars_f.index.get_loc(ts)
                t = sim_trade(bars_f, idx+1, True)  # enter 1 min after leader signal
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            # ETH big down → follower should follow → go short follower
            for ts in common[big_down.reindex(common, fill_value=False)]:
                if lt and (ts - lt).total_seconds() < 60: continue
                if ts not in bars_f.index: continue
                idx = bars_f.index.get_loc(ts)
                t = sim_trade(bars_f, idx+1, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  {leader}→{follower} (sigma={sigma}, follow):")
            pstats(train_trades, "TRAIN")
            pstats(test_trades, "TEST")

    # ══════════════════════════════════════════════════════════════════════
    # EXP TTT: INTRADAY SEASONALITY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP TTT: INTRADAY SEASONALITY")
    print(f"{'#'*80}")

    for sym in ['SOLUSDT', 'DOGEUSDT']:
        bars = all_bars[sym]
        ret_1m = bars['close'].pct_change()

        # Learn hourly bias from training period
        train_mask = bars.index < split_ts
        train_ret = ret_1m[train_mask]

        hourly_bias = train_ret.groupby(train_ret.index.hour).mean()
        print(f"\n  {sym} hourly bias (train):")
        for h in range(24):
            bias = hourly_bias.get(h, 0) * 10000
            flag = "↑" if bias > 0 else "↓"
            print(f"    {h:02d}:00  {flag} {bias:+.2f} bps/min")

        # Trade: at start of each hour, go in direction of learned bias
        # Only trade hours with strong bias (top/bottom 6 hours)
        sorted_hours = hourly_bias.sort_values()
        bear_hours = set(sorted_hours.index[:6])  # most negative
        bull_hours = set(sorted_hours.index[-6:])  # most positive

        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index:
            h = ts.hour
            if ts.minute != 0: continue  # only at start of hour
            if lt and (ts - lt).total_seconds() < 3600: continue

            if h in bull_hours:
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            elif h in bear_hours:
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

        print(f"\n  {sym} hourly seasonality (top/bottom 6 hours):")
        pstats(train_trades, "TRAIN")
        pstats(test_trades, "TEST")

    # ══════════════════════════════════════════════════════════════════════
    # EXP UUU: CROSS-SYMBOL DIVERGENCE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP UUU: CROSS-SYMBOL DIVERGENCE")
    print(f"{'#'*80}")

    # When ETH and SOL diverge (one up, one down), fade the divergence
    for sym_a, sym_b in [('ETHUSDT', 'SOLUSDT'), ('ETHUSDT', 'DOGEUSDT'),
                          ('SOLUSDT', 'DOGEUSDT')]:
        bars_a = all_bars[sym_a]
        bars_b = all_bars[sym_b]
        common = bars_a.index.intersection(bars_b.index)

        ret_a = bars_a['close'].pct_change(5).reindex(common)
        ret_b = bars_b['close'].pct_change(5).reindex(common)
        spread = ret_a - ret_b  # positive = A outperforming B

        roll_std = spread.rolling(60, min_periods=30).std()

        for z in [2, 3]:
            # A outperforming B by z sigma → short A, long B (fade divergence)
            a_over = spread > z * roll_std
            b_over = spread < -z * roll_std

            train_trades = []; test_trades = []
            lt = None

            # A outperforming → short A
            for ts in common[a_over.reindex(common, fill_value=False)]:
                if lt and (ts - lt).total_seconds() < 300: continue
                idx_a = bars_a.index.get_loc(ts) if ts in bars_a.index else None
                if idx_a is None: continue
                t = sim_trade(bars_a, idx_a, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            # B outperforming → short B
            for ts in common[b_over.reindex(common, fill_value=False)]:
                if lt and (ts - lt).total_seconds() < 300: continue
                idx_b = bars_b.index.get_loc(ts) if ts in bars_b.index else None
                if idx_b is None: continue
                t = sim_trade(bars_b, idx_b, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  {sym_a} vs {sym_b} divergence (z>{z}):")
            ts_r = pstats(train_trades, "TRAIN (fade outperformer)")
            te_r = pstats(test_trades, "TEST (fade outperformer)")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
