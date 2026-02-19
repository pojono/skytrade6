#!/usr/bin/env python3
"""
v42at: Price Acceleration + Volume Climax + Multi-TF Agreement

EXP HHHH4: Price Acceleration (2nd derivative)
  - 1st derivative = momentum = close.pct_change(N)
  - 2nd derivative = change in momentum
  - Extreme positive acceleration → parabolic → fade
  - Extreme negative acceleration → capitulation → fade

EXP IIII4: Volume Climax
  - Volume spike + price reversal bar = climax
  - Volume > 3x rolling avg + bar closes opposite to direction
  - Classic exhaustion signal

EXP JJJJ4: Multi-Timeframe Z-Score Agreement
  - Z-score on 5m, 15m, 30m timeframes
  - When all agree on extreme → stronger signal
  - When they disagree → avoid

EXP KKKK4: Price Oscillation Frequency
  - Count sign changes in returns over rolling window
  - High frequency = choppy/MR regime → fade aggressively
  - Low frequency = trending → avoid fading

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
    out_file = 'results/v42at_accel_climax.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42at: PRICE ACCEL + VOL CLIMAX + MULTI-TF + OSCILLATION")
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
        opn = bars['open']
        high = bars['high']
        low = bars['low']
        volume = bars['volume']
        ret_1m = close.pct_change()

        # ── EXP HHHH4: PRICE ACCELERATION ──
        print(f"\n  --- EXP HHHH4: PRICE ACCELERATION ---")

        for mom_period in [10, 20]:
            mom = close.pct_change(mom_period)
            accel = mom.diff()  # 2nd derivative

            accel_std = accel.rolling(60, min_periods=30).std()
            accel_avg = accel.rolling(60, min_periods=30).mean()

            for z_thresh in [2, 3]:
                accel_up = (accel - accel_avg) > z_thresh * accel_std
                accel_down = (accel - accel_avg) < -z_thresh * accel_std

                train_trades = []; test_trades = []
                lt = None
                for ts_idx in bars.index[accel_up]:
                    if lt and (ts_idx - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts_idx)
                    t = sim_trade(bars, idx, False)  # fade parabolic up
                    if t:
                        if ts_idx < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts_idx
                for ts_idx in bars.index[accel_down]:
                    if lt and (ts_idx - lt).total_seconds() < 60: continue
                    idx = bars.index.get_loc(ts_idx)
                    t = sim_trade(bars, idx, True)  # fade capitulation
                    if t:
                        if ts_idx < split_ts: train_trades.append(t)
                        else: test_trades.append(t)
                        lt = ts_idx
                train_trades.sort(key=lambda t: t['time'])
                test_trades.sort(key=lambda t: t['time'])

                print(f"\n  Accel({mom_period}) z>{z_thresh} (fade):")
                ts_r = pstats(train_trades, "TRAIN")
                te_r = pstats(test_trades, "TEST")
                if ts_r and te_r:
                    oos = "✅" if te_r['tot'] > 0 else "❌"
                    print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                    grand[(symbol, f'accel_{mom_period}_z{z_thresh}')] = te_r

        # ── EXP IIII4: VOLUME CLIMAX ──
        print(f"\n  --- EXP IIII4: VOLUME CLIMAX ---")

        vol_avg = volume.rolling(30, min_periods=15).mean()

        for vol_mult in [3, 5]:
            vol_spike = volume > vol_mult * vol_avg
            # Reversal bar: close opposite to open direction
            bull_bar = close > opn
            bear_bar = close < opn

            # Volume climax sell: huge volume + bearish bar → exhaustion → long
            climax_sell = vol_spike & bear_bar
            climax_buy = vol_spike & bull_bar

            train_trades = []; test_trades = []
            lt = None
            for ts_idx in bars.index[climax_sell]:
                if lt and (ts_idx - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts_idx)
                t = sim_trade(bars, idx, True)  # fade sell climax
                if t:
                    if ts_idx < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts_idx
            for ts_idx in bars.index[climax_buy]:
                if lt and (ts_idx - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts_idx)
                t = sim_trade(bars, idx, False)  # fade buy climax
                if t:
                    if ts_idx < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts_idx
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  Vol climax {vol_mult}x (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'vol_climax_{vol_mult}x')] = te_r

        # ── EXP JJJJ4: MULTI-TF Z-SCORE AGREEMENT ──
        print(f"\n  --- EXP JJJJ4: MULTI-TF Z-SCORE AGREEMENT ---")

        # Z-scores at different lookbacks
        for combo in [(5, 15, 30), (10, 30, 60)]:
            z_scores = []
            for w in combo:
                mu = close.rolling(w, min_periods=w).mean()
                sd = close.rolling(w, min_periods=w).std()
                z = (close - mu) / (sd + 1e-10)
                z_scores.append(z)

            # All z-scores agree on extreme
            all_high = (z_scores[0] > 2) & (z_scores[1] > 2) & (z_scores[2] > 2)
            all_low = (z_scores[0] < -2) & (z_scores[1] < -2) & (z_scores[2] < -2)

            train_trades = []; test_trades = []
            lt = None
            for ts_idx in bars.index[all_high]:
                if lt and (ts_idx - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts_idx)
                t = sim_trade(bars, idx, False)
                if t:
                    if ts_idx < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts_idx
            for ts_idx in bars.index[all_low]:
                if lt and (ts_idx - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts_idx)
                t = sim_trade(bars, idx, True)
                if t:
                    if ts_idx < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts_idx
            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            label = f"{combo[0]}/{combo[1]}/{combo[2]}"
            print(f"\n  Multi-TF z>2 ({label}) (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'mtf_{label}')] = te_r

        # ── EXP KKKK4: PRICE OSCILLATION FREQUENCY ──
        print(f"\n  --- EXP KKKK4: OSCILLATION FREQUENCY ---")

        sign_change = (ret_1m.shift(1) * ret_1m < 0).astype(int)

        for window in [15, 30]:
            osc_freq = sign_change.rolling(window, min_periods=window).sum() / window
            osc_pct = osc_freq.rolling(120, min_periods=60).rank(pct=True)

            # High oscillation = choppy → MR works well → fade
            high_osc = osc_pct > 0.90

            train_trades = []; test_trades = []
            lt = None
            for ts_idx in bars.index[high_osc]:
                if lt and (ts_idx - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts_idx)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts_idx < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts_idx

            print(f"\n  High oscillation({window}) >90th pct (fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")
                grand[(symbol, f'osc_freq_{window}')] = te_r

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
