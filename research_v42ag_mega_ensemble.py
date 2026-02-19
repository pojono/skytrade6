#!/usr/bin/env python3
"""
v42ag: MEGA ENSEMBLE — Top 10 Signals × 4 Symbols

Combines the highest-quality signals from all 30 types discovered:
1. Microstructure MR (sigma=2) — price only
2. VWAP Deviation (>20bps) — price only
3. Vol Clustering (>2x) — price only
4. 15m Range Fade (z>2) — price only
5. RSI(14) <20/>80 — price only
6. EMA(10/50) div z>2 — price only
7. Stochastic(30) <10/90 — price only
8. Trade Arrival Rate (>3x) — price+vol
9. Tick Imbalance (30m |>12|) — price only
10. Bollinger Band(60) touch — price only

Scoring: each signal contributes +1 (long) or -1 (short).
Trade when |score| >= threshold.

All 4 symbols, walk-forward 60d train / 28d test, RAM-safe.
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


def load_bars_with_trades(symbol, dates, data_dir='data', chunk_days=10):
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
            b = b.dropna(subset=['close'])
            b['volume'] = b['volume'].fillna(0)
            b['n_trades'] = b['n_trades'].fillna(0)
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


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


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


def main():
    out_file = 'results/v42ag_mega_ensemble.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ag: MEGA ENSEMBLE — TOP 10 SIGNALS × 4 SYMBOLS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    grand_summary = {}

    for sym in symbols:
        bars = load_bars_with_trades(sym, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {sym}")
        print(f"{'#'*80}")

        close = bars['close']
        ret_1m = close.pct_change()
        ret_5m = close.pct_change(5)

        # ── Compute all 10 signal scores ──
        # Each: +1 = long signal, -1 = short signal, 0 = no signal

        # 1. Micro MR (sigma=2)
        roll_std = ret_1m.rolling(60, min_periods=30).std()
        s1 = pd.Series(0, index=bars.index)
        s1[ret_1m < -2 * roll_std] = 1
        s1[ret_1m > 2 * roll_std] = -1

        # 2. VWAP Deviation (>20bps)
        roll_pv = (close * bars['volume']).rolling(60, min_periods=30).sum()
        roll_vol = bars['volume'].rolling(60, min_periods=30).sum()
        vwap_60 = roll_pv / (roll_vol + 1)
        vwap_dev = (close - vwap_60) / vwap_60 * 10000
        s2 = pd.Series(0, index=bars.index)
        s2[vwap_dev < -20] = 1
        s2[vwap_dev > 20] = -1

        # 3. Vol Clustering (>2x)
        abs_ret = ret_1m.abs()
        vol_5m = abs_ret.rolling(5, min_periods=3).mean()
        vol_60m = abs_ret.rolling(60, min_periods=30).mean()
        vol_cluster = vol_5m / (vol_60m + 1e-10)
        s3 = pd.Series(0, index=bars.index)
        s3[(vol_cluster > 2) & (ret_1m < 0)] = 1
        s3[(vol_cluster > 2) & (ret_1m > 0)] = -1

        # 4. 15m Range Fade (z>2)
        roll_high = bars['high'].rolling(15, min_periods=10).max()
        roll_low = bars['low'].rolling(15, min_periods=10).min()
        range_15m = (roll_high - roll_low) / close * 10000
        range_avg = range_15m.rolling(60, min_periods=30).mean()
        range_std = range_15m.rolling(60, min_periods=30).std()
        range_z = (range_15m - range_avg) / (range_std + 1e-10)
        s4 = pd.Series(0, index=bars.index)
        mid = (roll_high + roll_low) / 2
        s4[(range_z > 2) & (close < mid)] = 1
        s4[(range_z > 2) & (close >= mid)] = -1

        # 5. RSI(14) <20/>80
        rsi = compute_rsi(close, 14)
        s5 = pd.Series(0, index=bars.index)
        s5[rsi < 20] = 1
        s5[rsi > 80] = -1

        # 6. EMA(10/50) div z>2
        ema10 = close.ewm(span=10, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema_div = (ema10 - ema50) / ema50 * 10000
        div_std = ema_div.rolling(60, min_periods=30).std()
        s6 = pd.Series(0, index=bars.index)
        s6[ema_div > 2 * div_std] = -1  # fade extreme up
        s6[ema_div < -2 * div_std] = 1  # fade extreme down

        # 7. Stochastic(30) <10/90
        low_k = bars['low'].rolling(30, min_periods=30).min()
        high_k = bars['high'].rolling(30, min_periods=30).max()
        pct_k = (close - low_k) / (high_k - low_k + 1e-10) * 100
        s7 = pd.Series(0, index=bars.index)
        s7[pct_k < 10] = 1
        s7[pct_k > 90] = -1

        # 8. Trade Arrival Rate (>3x)
        n_trades_roll = bars['n_trades'].rolling(60, min_periods=30).mean()
        n_trades_ratio = bars['n_trades'] / (n_trades_roll + 1)
        s8 = pd.Series(0, index=bars.index)
        s8[(n_trades_ratio > 3) & (ret_5m < 0)] = 1
        s8[(n_trades_ratio > 3) & (ret_5m > 0)] = -1

        # 9. Tick Imbalance (30m |>12|)
        up_tick = (close > close.shift(1)).astype(int)
        down_tick = (close < close.shift(1)).astype(int)
        tick_imb = up_tick.rolling(30, min_periods=15).sum() - down_tick.rolling(30, min_periods=15).sum()
        s9 = pd.Series(0, index=bars.index)
        s9[tick_imb > 12] = -1  # fade up imbalance
        s9[tick_imb < -12] = 1  # fade down imbalance

        # 10. Bollinger Band(60) touch
        bb_mid = close.rolling(60, min_periods=60).mean()
        bb_std_val = close.rolling(60, min_periods=60).std()
        bb_upper = bb_mid + 2 * bb_std_val
        bb_lower = bb_mid - 2 * bb_std_val
        s10 = pd.Series(0, index=bars.index)
        s10[close > bb_upper] = -1
        s10[close < bb_lower] = 1

        # ── Ensemble score ──
        ensemble = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10

        print(f"\n  Ensemble score distribution:")
        for sc in range(-6, 7):
            cnt = (ensemble == sc).sum()
            if cnt > 0:
                print(f"    score={sc:+2d}: {cnt:6d} bars ({cnt/len(ensemble)*100:.1f}%)")

        # ── Trade at different thresholds ──
        for min_score in [2, 3, 4, 5]:
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

            if test_trades:
                arr = np.array([t['net'] for t in test_trades])
                n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
                tot = arr.sum()*100
                sh = arr.mean()/(arr.std()+1e-10)*np.sqrt(252*24*60)
                flag = "✅" if tot > 0 else "  "
                print(f"\n  {flag} score>={min_score:d}  OOS: n={n:5d}  wr={wr:5.1f}%  "
                      f"avg={avg:+6.1f}bps  tot={tot:+8.1f}%  sh={sh:+7.0f}")
                grand_summary[(sym, min_score)] = {
                    'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh
                }

                # Train stats
                tarr = np.array([t['net'] for t in train_trades])
                tn = len(tarr); twr = (tarr>0).mean()*100; tavg = tarr.mean()*10000
                ttot = tarr.sum()*100
                print(f"       TRAIN: n={tn:5d}  wr={twr:5.1f}%  avg={tavg:+6.1f}bps  tot={ttot:+8.1f}%")

                # Daily PnL
                daily = {}
                for t in test_trades:
                    d = t['time'].date()
                    daily[d] = daily.get(d, 0) + t['net']
                darr = np.array(list(daily.values()))
                pos = (darr > 0).sum()
                print(f"       OOS days: {pos}/{len(darr)} positive ({pos/len(darr)*100:.0f}%)")
                print(f"       OOS worst day: {darr.min()*100:+.3f}%")
            else:
                print(f"\n    score>={min_score:d}  NO OOS TRADES")

        del bars; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  GRAND SUMMARY: MEGA ENSEMBLE (OOS only)")
    print(f"{'#'*80}")
    print(f"\n  {'Symbol':10s}  {'Score':>6s}  {'n':>5s}  {'WR':>6s}  {'Avg':>7s}  {'Total':>8s}  {'Sharpe':>7s}")
    print(f"  {'-'*55}")
    for (sym, sc), r in sorted(grand_summary.items()):
        flag = "✅" if r['tot'] > 0 else "  "
        print(f"  {flag} {sym:10s}  >={sc:d}     {r['n']:5d}  {r['wr']:5.1f}%  "
              f"{r['avg']:+6.1f}bp  {r['tot']:+7.1f}%  {r['sharpe']:+7.0f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
