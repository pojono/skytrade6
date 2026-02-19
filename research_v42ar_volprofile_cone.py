#!/usr/bin/env python3
"""
v42ar: Volume Profile + Realized Vol Cone + Momentum-Volume Divergence

EXP ZZZZ3: Volume-Weighted Price Level (VPOC proxy)
  - Rolling VWAP as value area center
  - Price far from VWAP → fade back to value
  - Different from VWAP divergence: uses distance percentile

EXP AAAA4: Realized Volatility Cone
  - Compare current realized vol to historical distribution
  - Vol at extreme low → expect expansion → avoid
  - Vol at extreme high → expect contraction → fade aggressively

EXP BBBB4: Momentum-Volume Confirmation
  - Price momentum + volume momentum agreement
  - When price moves but volume doesn't confirm → fade
  - When both agree → follow

EXP CCCC4: Rolling Sharpe Ratio Signal
  - Compute rolling Sharpe of returns
  - Extreme positive Sharpe → unsustainable → fade
  - Extreme negative Sharpe → oversold → buy

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


def load_bars_with_vol(symbol, dates, data_dir='data', chunk_days=10):
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
                df = pd.read_csv(f, usecols=['timestamp', 'price', 'size'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                dfs.append(df)
        if dfs:
            chunk = pd.concat(dfs, ignore_index=True)
            del dfs
            chunk['notional'] = chunk['price'] * chunk['size']
            ts = chunk.set_index('timestamp')
            b = ts['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last')
            b['volume'] = ts['notional'].resample('1min').sum()
            b = b.dropna(subset=['close'])
            b['volume'] = b['volume'].fillna(0)
            all_bars.append(b)
            del chunk, ts; gc.collect()
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
    out_file = 'results/v42ar_volprofile_cone.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ar: VOL PROFILE + REALIZED VOL CONE + MOM-VOL DIVERGENCE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    grand = {}

    for symbol in symbols:
        bars = load_bars_with_vol(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        close = bars['close']
        volume = bars['volume']
        ret_1m = close.pct_change()

        # ── EXP ZZZZ3: VWAP DISTANCE PERCENTILE ──
        print(f"\n  --- EXP ZZZZ3: VWAP DISTANCE PERCENTILE ---")

        cum_vol = volume.cumsum()
        cum_pv = (close * volume).cumsum()
        vwap = cum_pv / (cum_vol + 1)

        # Rolling VWAP (30m window)
        roll_pv = (close * volume).rolling(30, min_periods=15).sum()
        roll_vol = volume.rolling(30, min_periods=15).sum()
        roll_vwap = roll_pv / (roll_vol + 1)

        vwap_dist = (close - roll_vwap) / roll_vwap * 10000  # bps
        vwap_pct = vwap_dist.rolling(120, min_periods=60).rank(pct=True)

        for thresh in [0.05, 0.10]:
            far_above = vwap_pct > (1 - thresh)
            far_below = vwap_pct < thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[far_above]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[far_below]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  VWAP dist pct <{thresh}/{1-thresh:.2f} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'vwap_pct_{thresh}')] = te_r

        # ── EXP AAAA4: REALIZED VOL CONE ──
        print(f"\n  --- EXP AAAA4: REALIZED VOL CONE ---")

        rvol_30 = ret_1m.rolling(30, min_periods=30).std()
        rvol_pct = rvol_30.rolling(120, min_periods=60).rank(pct=True)

        # High vol percentile → expect contraction → fade aggressively
        for thresh in [0.90, 0.95]:
            high_vol = rvol_pct > thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_vol]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Vol cone >{thresh*100:.0f}th pct (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'vol_cone_{thresh}')] = te_r

        # ── EXP BBBB4: MOMENTUM-VOLUME DIVERGENCE ──
        print(f"\n  --- EXP BBBB4: MOM-VOL DIVERGENCE ---")

        price_mom = close.pct_change(15)
        vol_mom = volume.rolling(15, min_periods=10).mean().pct_change(15)

        # Price up but volume down → weak rally → fade
        # Price down but volume down → weak selloff → fade
        price_up = price_mom > price_mom.rolling(60).mean() + price_mom.rolling(60).std()
        vol_down = vol_mom < 0

        price_down = price_mom < price_mom.rolling(60).mean() - price_mom.rolling(60).std()
        vol_up = vol_mom > 0

        # Divergence: price extreme + volume not confirming
        div_bearish = price_up & vol_down
        div_bullish = price_down & vol_up  # actually this means selling on high vol, follow

        # Better: price extreme + volume declining
        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[div_bearish]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade(bars, idx, False)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        for ts in bars.index[price_down & vol_down]:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade(bars, idx, True)  # weak selloff → buy
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        train_trades.sort(key=lambda t: t['time'])
        test_trades.sort(key=lambda t: t['time'])

        print(f"\n  Mom-vol divergence (fade weak moves):")
        ts_r = pstats(train_trades, "TRAIN")
        te_r = pstats(test_trades, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
            grand[(symbol, 'mom_vol_div')] = te_r

        # ── EXP CCCC4: ROLLING SHARPE SIGNAL ──
        print(f"\n  --- EXP CCCC4: ROLLING SHARPE SIGNAL ---")

        roll_mean = ret_1m.rolling(30, min_periods=15).mean()
        roll_std = ret_1m.rolling(30, min_periods=15).std()
        roll_sharpe = roll_mean / (roll_std + 1e-10)

        sharpe_avg = roll_sharpe.rolling(120, min_periods=60).mean()
        sharpe_std = roll_sharpe.rolling(120, min_periods=60).std()

        for z_thresh in [2, 3]:
            extreme_pos = (roll_sharpe - sharpe_avg) > z_thresh * sharpe_std
            extreme_neg = (roll_sharpe - sharpe_avg) < -z_thresh * sharpe_std

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[extreme_pos]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade extreme positive sharpe
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[extreme_neg]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade extreme negative sharpe
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Rolling Sharpe z>{z_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'sharpe_z{z_thresh}')] = te_r

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
