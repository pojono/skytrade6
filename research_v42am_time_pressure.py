#!/usr/bin/env python3
"""
v42am: Time-of-Day Volatility + Buy/Sell Pressure Asymmetry

EXP FFFF2: Hour-of-Day Volatility Regime
  - Compute rolling vol by hour-of-day
  - Trade only during high-vol hours with MR signals
  - Avoid low-vol hours

EXP GGGG2: Buy/Sell Pressure Asymmetry
  - Rolling buy_vol / sell_vol ratio
  - Extreme imbalance → fade the dominant side
  - Different from VPIN (uses ratio not absolute imbalance)

EXP HHHH2: Acceleration of Volume
  - Volume acceleration = d(volume)/dt
  - Sudden volume surge → information event → fade

EXP IIII2: Price Momentum Quality
  - Momentum weighted by volume consistency
  - High-quality momentum (consistent volume) → follow
  - Low-quality momentum (volume spikes) → fade

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


def load_bars_with_sides(symbol, dates, data_dir='data', chunk_days=10):
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
                df = pd.read_csv(f, usecols=['timestamp', 'price', 'size', 'side'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                dfs.append(df)
        if dfs:
            chunk = pd.concat(dfs, ignore_index=True)
            del dfs
            chunk['notional'] = chunk['price'] * chunk['size']
            chunk['buy_vol'] = chunk['notional'].where(chunk['side'] == 'Buy', 0)
            chunk['sell_vol'] = chunk['notional'].where(chunk['side'] == 'Sell', 0)
            ts = chunk.set_index('timestamp')
            b = ts['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last')
            b['volume'] = ts['notional'].resample('1min').sum()
            b['buy_vol'] = ts['buy_vol'].resample('1min').sum()
            b['sell_vol'] = ts['sell_vol'].resample('1min').sum()
            b = b.dropna(subset=['close'])
            for c in ['volume', 'buy_vol', 'sell_vol']:
                b[c] = b[c].fillna(0)
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
    out_file = 'results/v42am_time_pressure.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42am: TIME-OF-DAY + BUY/SELL PRESSURE + VOL ACCELERATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    grand = {}

    for symbol in symbols:
        bars = load_bars_with_sides(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        close = bars['close']
        ret_1m = close.pct_change()

        # ── EXP FFFF2: HOUR-OF-DAY VOLATILITY ──
        print(f"\n  --- EXP FFFF2: HOUR-OF-DAY VOLATILITY ---")

        hour = bars.index.hour
        abs_ret = ret_1m.abs()

        # Compute hourly vol profile from training data
        train_mask = bars.index < split_ts
        hourly_vol = {}
        for h in range(24):
            mask = train_mask & (hour == h)
            hourly_vol[h] = abs_ret[mask].mean() if mask.sum() > 0 else 0

        # Top 8 volatile hours
        sorted_hours = sorted(hourly_vol.items(), key=lambda x: -x[1])
        top_hours = set([h for h, _ in sorted_hours[:8]])
        print(f"  Top 8 volatile hours (UTC): {sorted(top_hours)}")

        # Trade MR signal only during high-vol hours
        roll_std = ret_1m.rolling(60, min_periods=30).std()
        mr_signal = ret_1m.abs() > 2 * roll_std
        high_vol_hour = pd.Series(hour, index=bars.index).isin(top_hours)

        for use_filter in [False, True]:
            label = "MR + high-vol hours" if use_filter else "MR all hours"
            mask = mr_signal & high_vol_hour if use_filter else mr_signal

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[mask]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  {label}:")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, label)] = te_r

        # ── EXP GGGG2: BUY/SELL PRESSURE RATIO ──
        print(f"\n  --- EXP GGGG2: BUY/SELL PRESSURE RATIO ---")

        bs_ratio = bars['buy_vol'] / (bars['sell_vol'] + 1)
        bs_roll = bs_ratio.rolling(30, min_periods=15).mean()
        bs_std = bs_ratio.rolling(60, min_periods=30).std()
        bs_avg = bs_ratio.rolling(60, min_periods=30).mean()
        bs_z = (bs_roll - bs_avg) / (bs_std + 1e-10)

        for z_thresh in [2, 3]:
            buy_dom = bs_z > z_thresh
            sell_dom = bs_z < -z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[buy_dom]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)  # fade buy dominance
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[sell_dom]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)  # fade sell dominance
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  B/S ratio z>{z_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'bs_ratio_z{z_thresh}')] = te_r

        # ── EXP HHHH2: VOLUME ACCELERATION ──
        print(f"\n  --- EXP HHHH2: VOLUME ACCELERATION ---")

        vol_5m = bars['volume'].rolling(5, min_periods=3).mean()
        vol_15m = bars['volume'].rolling(15, min_periods=10).mean()
        vol_accel = (vol_5m - vol_15m) / (vol_15m + 1)
        accel_std = vol_accel.rolling(60, min_periods=30).std()

        for z_thresh in [2, 3]:
            high_accel = vol_accel > z_thresh * accel_std

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_accel]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade during volume surge
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Vol accel z>{z_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'vol_accel_z{z_thresh}')] = te_r

        # ── EXP IIII2: MOMENTUM QUALITY ──
        print(f"\n  --- EXP IIII2: MOMENTUM QUALITY ---")

        mom_30 = close.pct_change(30)
        vol_cv = bars['volume'].rolling(30, min_periods=15).std() / (bars['volume'].rolling(30, min_periods=15).mean() + 1)
        # Low CV = consistent volume = high quality momentum
        # High CV = spiky volume = low quality momentum

        mom_std = mom_30.rolling(60, min_periods=30).std()

        for cv_thresh in [0.5, 1.0]:
            # Low quality momentum (high CV) + extreme move → fade
            low_quality_up = (mom_30 > 2 * mom_std) & (vol_cv > cv_thresh)
            low_quality_down = (mom_30 < -2 * mom_std) & (vol_cv > cv_thresh)

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[low_quality_up]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[low_quality_down]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Low-quality momentum CV>{cv_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'mom_quality_cv{cv_thresh}')] = te_r

        del bars; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  GRAND SUMMARY (OOS only)")
    print(f"{'#'*80}")
    print(f"\n  {'Symbol':10s}  {'Signal':25s}  {'n':>5s}  {'WR':>6s}  {'Avg':>7s}  {'Total':>8s}  {'Sharpe':>7s}")
    print(f"  {'-'*70}")
    for (sym, sig), r in sorted(grand.items()):
        flag = "✅" if r['tot'] > 0 else "  "
        print(f"  {flag} {sym:10s}  {sig:25s}  {r['n']:5d}  {r['wr']:5.1f}%  "
              f"{r['avg']:+6.1f}bp  {r['tot']:+7.1f}%  {r['sharpe']:+7.0f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
