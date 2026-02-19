#!/usr/bin/env python3
"""
v42x: Order Flow Imbalance + Trade Intensity Signals

EXP III: Order Flow Imbalance
  - Compute buy vs sell volume ratio from futures trades
  - When ratio is extreme (>P95 or <P5), fade the imbalance
  - Aggressive buyers = overbought = short opportunity

EXP JJJ: Trade Intensity (Volume Spike)
  - When 1-min volume exceeds N times rolling average
  - High volume = information event = mean-reversion
  - Compare: fade vs follow the volume direction

EXP KKK: Volume-Weighted Price Deviation
  - VWAP deviation: when price is far from rolling VWAP
  - Fade extreme deviations (mean-reversion to VWAP)

SOLUSDT + ETHUSDT, 88 days, walk-forward.
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


def load_bars_with_volume(symbol, dates, data_dir='data', chunk_days=10):
    """Load bars with buy/sell volume breakdown."""
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} bars+vol...", end='', flush=True)
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
            chunk['buy_notional'] = chunk['notional'].where(chunk['side'] == 'Buy', 0)
            chunk['sell_notional'] = chunk['notional'].where(chunk['side'] == 'Sell', 0)
            ts = chunk.set_index('timestamp')
            b = ts['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last')
            b['volume'] = ts['notional'].resample('1min').sum()
            b['buy_vol'] = ts['buy_notional'].resample('1min').sum()
            b['sell_vol'] = ts['sell_notional'].resample('1min').sum()
            b['n_trades'] = ts['price'].resample('1min').count()
            b = b.dropna(subset=['close'])
            all_bars.append(b)
            del chunk, ts; gc.collect()
        done = min(start+chunk_days, n)
        el = time.time()-t0
        print(f" [{done}/{n} {el:.0f}s]", end='', flush=True)
    if not all_bars: print(" NO DATA"); return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    result['buy_vol'] = result['buy_vol'].fillna(0)
    result['sell_vol'] = result['sell_vol'].fillna(0)
    result['volume'] = result['volume'].fillna(0)
    print(f" {len(result):,} bars ({time.time()-t0:.0f}s) [{ram_str()}]")
    return result


def sim_trade_trail(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30,
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
    out_file = 'results/v42x_orderflow.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42x: ORDER FLOW IMBALANCE + TRADE INTENSITY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in ['SOLUSDT', 'ETHUSDT']:
        bars = load_bars_with_volume(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        # Order flow imbalance
        ofi = (bars['buy_vol'] - bars['sell_vol']) / (bars['buy_vol'] + bars['sell_vol'] + 1)
        ofi_5m = ofi.rolling(5, min_periods=1).mean()
        ofi_15m = ofi.rolling(15, min_periods=1).mean()

        # Volume features
        vol_avg = bars['volume'].rolling(60, min_periods=30).mean()
        vol_ratio = bars['volume'] / (vol_avg + 1)

        # VWAP
        cum_vol = bars['volume'].cumsum()
        cum_pv = (bars['close'] * bars['volume']).cumsum()
        # Rolling VWAP (60-min)
        roll_pv = (bars['close'] * bars['volume']).rolling(60, min_periods=30).sum()
        roll_vol = bars['volume'].rolling(60, min_periods=30).sum()
        vwap_60 = roll_pv / (roll_vol + 1)
        vwap_dev = (bars['close'] - vwap_60) / vwap_60 * 10000  # in bps

        print(f"  OFI stats: mean={ofi.mean():.4f}, std={ofi.std():.4f}")
        print(f"  Vol ratio stats: mean={vol_ratio.mean():.2f}, P95={vol_ratio.quantile(0.95):.2f}")
        print(f"  VWAP dev stats: mean={vwap_dev.mean():.1f}bps, std={vwap_dev.std():.1f}bps")

        # ══════════════════════════════════════════════════════════════════
        # EXP III: ORDER FLOW IMBALANCE
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  --- EXP III: ORDER FLOW IMBALANCE ---")

        for window, ofi_series, label in [(1, ofi, '1m'), (5, ofi_5m, '5m'), (15, ofi_15m, '15m')]:
            for pct in [90, 95]:
                high_thresh = ofi_series.quantile(pct/100)
                low_thresh = ofi_series.quantile((100-pct)/100)

                # Extreme buy imbalance → short (fade)
                # Extreme sell imbalance → long (fade)
                buy_extreme = ofi_series > high_thresh
                sell_extreme = ofi_series < low_thresh

                train_trades = []; test_trades = []
                lt = None
                for ts in bars.index[buy_extreme]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade_trail(bars, idx, False)  # short after buy extreme
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts
                for ts in bars.index[sell_extreme]:
                    if lt and (ts - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts)
                    t = sim_trade_trail(bars, idx, True)  # long after sell extreme
                    if t:
                        if ts < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts

                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  OFI {label} P{pct} fade:")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # ══════════════════════════════════════════════════════════════════
        # EXP JJJ: TRADE INTENSITY (VOLUME SPIKE)
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  --- EXP JJJ: TRADE INTENSITY ---")

        for vol_th in [3, 5, 10]:
            high_vol = vol_ratio > vol_th

            # Fade: go opposite to recent price move after volume spike
            price_ret = bars['close'].pct_change(5)
            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_vol]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                ret = price_ret.iloc[idx] if idx < len(price_ret) else 0
                is_long = ret < 0  # fade
                t = sim_trade_trail(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Vol spike >{vol_th}x (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # ══════════════════════════════════════════════════════════════════
        # EXP KKK: VWAP DEVIATION
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  --- EXP KKK: VWAP DEVIATION ---")

        for dev_thresh in [5, 10, 20]:
            above_vwap = vwap_dev > dev_thresh
            below_vwap = vwap_dev < -dev_thresh

            train_trades = []; test_trades = []
            lt = None
            # Above VWAP → short (fade to VWAP)
            for ts in bars.index[above_vwap]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade_trail(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts
            # Below VWAP → long (fade to VWAP)
            for ts in bars.index[below_vwap]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade_trail(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  VWAP dev >{dev_thresh}bps (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        del bars; gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
