#!/usr/bin/env python3
"""
v42ac: Tick-Level Microstructure Patterns

EXP VVV: Trade Size Distribution Signal
  - When average trade size suddenly increases → institutional activity
  - Large trades = informed traders = follow their direction
  - Small trades = noise = fade

EXP WWW: Trade Arrival Rate Signal
  - When trades/minute spikes → information event
  - Combine with direction for entry
  - Different from volume spike (many small trades vs few large trades)

EXP XXX: Bid-Ask Bounce Signal (from ticker data)
  - When price bounces between bid and ask rapidly → noise
  - When price consistently hits one side → directional pressure
  - Fade the pressure (mean-reversion)

EXP YYY: Multi-Signal Ensemble Score
  - Combine top 5 signals into a single score
  - Trade only when score exceeds threshold
  - Does ensemble beat individual signals?

SOLUSDT + DOGEUSDT, walk-forward, 88 days.
"""

import sys, time, json, gzip, os, gc, psutil
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


def load_bars_with_trades(symbol, dates, data_dir='data', chunk_days=10):
    """Load bars with trade count and avg trade size."""
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
            ts = chunk.set_index('timestamp')
            b = ts['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last')
            b['volume'] = ts['notional'].resample('1min').sum()
            b['n_trades'] = ts['price'].resample('1min').count()
            b['avg_size'] = ts['notional'].resample('1min').mean()
            b['max_size'] = ts['notional'].resample('1min').max()
            b = b.dropna(subset=['close'])
            b['volume'] = b['volume'].fillna(0)
            b['n_trades'] = b['n_trades'].fillna(0)
            b['avg_size'] = b['avg_size'].fillna(0)
            b['max_size'] = b['max_size'].fillna(0)
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
    out_file = 'results/v42ac_tick_micro.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ac: TICK-LEVEL MICROSTRUCTURE PATTERNS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in ['SOLUSDT', 'DOGEUSDT']:
        bars = load_bars_with_trades(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        ret_1m = bars['close'].pct_change()
        price_ret_5m = bars['close'].pct_change(5)

        # ── EXP VVV: TRADE SIZE DISTRIBUTION ──
        print(f"\n  --- EXP VVV: TRADE SIZE DISTRIBUTION ---")

        avg_size_roll = bars['avg_size'].rolling(60, min_periods=30).mean()
        avg_size_std = bars['avg_size'].rolling(60, min_periods=30).std()
        size_z = (bars['avg_size'] - avg_size_roll) / (avg_size_std + 1e-10)

        max_size_roll = bars['max_size'].rolling(60, min_periods=30).mean()
        max_size_std = bars['max_size'].rolling(60, min_periods=30).std()
        max_z = (bars['max_size'] - max_size_roll) / (max_size_std + 1e-10)

        print(f"  Avg size z stats: mean={size_z.mean():.3f}, std={size_z.std():.3f}")
        print(f"  Max size z stats: mean={max_z.mean():.3f}, std={max_z.std():.3f}")

        for z_thresh in [2, 3]:
            # Large avg trade size → institutional → fade (they're taking liquidity)
            big_avg = size_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[big_avg]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Big avg trade size (z>{z_thresh}, fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        for z_thresh in [2, 3]:
            # Large max trade → whale → fade
            big_max = max_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[big_max]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Whale trade (max z>{z_thresh}, fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP WWW: TRADE ARRIVAL RATE ──
        print(f"\n  --- EXP WWW: TRADE ARRIVAL RATE ---")

        n_trades_roll = bars['n_trades'].rolling(60, min_periods=30).mean()
        n_trades_ratio = bars['n_trades'] / (n_trades_roll + 1)

        for ratio_thresh in [3, 5, 10]:
            high_rate = n_trades_ratio > ratio_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_rate]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = price_ret_5m.iloc[idx] if idx < len(price_ret_5m) else 0
                is_long = r < 0  # fade
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Trade rate >{ratio_thresh}x (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP YYY: MULTI-SIGNAL ENSEMBLE ──
        print(f"\n  --- EXP YYY: MULTI-SIGNAL ENSEMBLE ---")

        # Compute individual signal scores (z-scores)
        roll_std = ret_1m.rolling(60, min_periods=30).std()
        vol_avg = bars['volume'].rolling(60, min_periods=30).mean()
        vol_ratio = bars['volume'] / (vol_avg + 1)
        roll_pv = (bars['close'] * bars['volume']).rolling(60, min_periods=30).sum()
        roll_vol = bars['volume'].rolling(60, min_periods=30).sum()
        vwap_60 = roll_pv / (roll_vol + 1)
        vwap_dev = (bars['close'] - vwap_60) / vwap_60 * 10000
        abs_ret = ret_1m.abs()
        vol_5m = abs_ret.rolling(5, min_periods=3).mean()
        vol_60m = abs_ret.rolling(60, min_periods=30).mean()
        vol_cluster = vol_5m / (vol_60m + 1e-10)

        # Ensemble: count how many signals fire simultaneously
        # Each signal: 1 if "fade long" condition, -1 if "fade short", 0 otherwise
        sig_micro_mr = pd.Series(0, index=bars.index)
        sig_micro_mr[ret_1m < -2 * roll_std] = 1  # extreme down → long
        sig_micro_mr[ret_1m > 2 * roll_std] = -1  # extreme up → short

        sig_vwap = pd.Series(0, index=bars.index)
        sig_vwap[vwap_dev < -20] = 1  # below VWAP → long
        sig_vwap[vwap_dev > 20] = -1  # above VWAP → short

        sig_vol = pd.Series(0, index=bars.index)
        sig_vol[(vol_ratio > 3) & (price_ret_5m < 0)] = 1
        sig_vol[(vol_ratio > 3) & (price_ret_5m > 0)] = -1

        sig_vc = pd.Series(0, index=bars.index)
        sig_vc[(vol_cluster > 2) & (ret_1m < 0)] = 1
        sig_vc[(vol_cluster > 2) & (ret_1m > 0)] = -1

        sig_size = pd.Series(0, index=bars.index)
        sig_size[(size_z > 2) & (ret_1m < 0)] = 1
        sig_size[(size_z > 2) & (ret_1m > 0)] = -1

        ensemble = sig_micro_mr + sig_vwap + sig_vol + sig_vc + sig_size

        for min_score in [2, 3, 4]:
            strong_long = ensemble >= min_score
            strong_short = ensemble <= -min_score

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[strong_long]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            for ts in bars.index[strong_short]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Ensemble score >={min_score} (5 signals):")
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
