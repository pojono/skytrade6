#!/usr/bin/env python3
"""
v42ak: Order Flow Toxicity + Price Efficiency + Noise

EXP XXXX2: VPIN (Volume-Synchronized Probability of Informed Trading)
  - Approximate VPIN from buy/sell volume imbalance
  - High VPIN → toxic flow → avoid or fade

EXP YYYY2: Price Efficiency Ratio
  - |net move| / sum(|bar moves|) over N bars
  - Low efficiency → noisy/choppy → fade
  - High efficiency → trending → follow

EXP ZZZZ2: Noise Ratio
  - High-frequency noise = realized vol / (range-based vol)
  - When noise is extreme → microstructure opportunity

EXP AAAA2: Return Dispersion Signal
  - Rolling std of rolling means at different windows
  - High dispersion → regime change → opportunity

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
            b['volume'] = b['volume'].fillna(0)
            b['buy_vol'] = b['buy_vol'].fillna(0)
            b['sell_vol'] = b['sell_vol'].fillna(0)
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
    out_file = 'results/v42ak_toxicity_efficiency.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ak: ORDER FLOW TOXICITY + PRICE EFFICIENCY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in ['SOLUSDT', 'DOGEUSDT', 'XRPUSDT']:
        bars = load_bars_with_sides(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        close = bars['close']
        ret_1m = close.pct_change()

        # ── EXP XXXX2: VPIN (VOLUME-SYNC INFORMED TRADING) ──
        print(f"\n  --- EXP XXXX2: VPIN ---")

        total_vol = bars['buy_vol'] + bars['sell_vol'] + 1
        vol_imb = (bars['buy_vol'] - bars['sell_vol']).abs() / total_vol

        # Rolling VPIN (30-bar window)
        vpin_30 = vol_imb.rolling(30, min_periods=15).mean()
        vpin_std = vpin_30.rolling(60, min_periods=30).std()
        vpin_avg = vpin_30.rolling(60, min_periods=30).mean()
        vpin_z = (vpin_30 - vpin_avg) / (vpin_std + 1e-10)

        print(f"  VPIN stats: mean={vpin_30.mean():.4f}, P95={vpin_30.quantile(0.95):.4f}")

        for z_thresh in [2, 3]:
            high_vpin = vpin_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_vpin]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                # High VPIN + buy dominant → fade (short), sell dominant → fade (long)
                buy_dom = bars.iloc[idx]['buy_vol'] > bars.iloc[idx]['sell_vol']
                is_long = not buy_dom  # fade the dominant side
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  VPIN z>{z_thresh} (fade dominant side):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP YYYY2: PRICE EFFICIENCY RATIO ──
        print(f"\n  --- EXP YYYY2: PRICE EFFICIENCY RATIO ---")

        for window in [15, 30]:
            net_move = close.diff(window).abs()
            sum_moves = ret_1m.abs().rolling(window, min_periods=window).sum() * close
            efficiency = net_move / (sum_moves + 1e-10)

            eff_avg = efficiency.rolling(60, min_periods=30).mean()
            eff_std = efficiency.rolling(60, min_periods=30).std()

            # Low efficiency → choppy → fade
            for pct in [10, 20]:
                thresh = efficiency.rolling(120, min_periods=60).quantile(pct/100)
                low_eff = efficiency < thresh

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

                print(f"\n  Efficiency({window}) P{pct} (fade in choppy):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP ZZZZ2: NOISE RATIO ──
        print(f"\n  --- EXP ZZZZ2: NOISE RATIO ---")

        realized_vol = ret_1m.rolling(30, min_periods=15).std()
        hl_vol = ((bars['high'] - bars['low']) / close).rolling(30, min_periods=15).mean()
        noise_ratio = realized_vol / (hl_vol + 1e-10)

        nr_avg = noise_ratio.rolling(60, min_periods=30).mean()
        nr_std = noise_ratio.rolling(60, min_periods=30).std()
        nr_z = (noise_ratio - nr_avg) / (nr_std + 1e-10)

        for z_thresh in [2, 3]:
            high_noise = nr_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_noise]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Noise ratio z>{z_thresh} (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP AAAA2: RETURN DISPERSION ──
        print(f"\n  --- EXP AAAA2: RETURN DISPERSION ---")

        # Multiple rolling means at different windows
        ma5 = ret_1m.rolling(5, min_periods=3).mean()
        ma15 = ret_1m.rolling(15, min_periods=10).mean()
        ma30 = ret_1m.rolling(30, min_periods=15).mean()
        ma60 = ret_1m.rolling(60, min_periods=30).mean()

        # Dispersion = std of these means
        dispersion = pd.concat([ma5, ma15, ma30, ma60], axis=1).std(axis=1)
        disp_avg = dispersion.rolling(60, min_periods=30).mean()
        disp_std = dispersion.rolling(60, min_periods=30).std()
        disp_z = (dispersion - disp_avg) / (disp_std + 1e-10)

        for z_thresh in [2, 3]:
            high_disp = disp_z > z_thresh

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

            print(f"\n  Dispersion z>{z_thresh} (fade):")
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
