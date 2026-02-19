#!/usr/bin/env python3
"""
v42ai: Price Impact + Microstructure Order Flow

EXP PPPP: Kyle's Lambda (Price Impact)
  - Rolling price impact = |return| / volume
  - High impact → illiquid → larger mean-reversion
  - Low impact → liquid → smaller moves

EXP QQQQ: Amihud Illiquidity
  - Rolling Amihud = mean(|return| / dollar_volume)
  - High Amihud → illiquid → fade extreme moves

EXP RRRR: Return Predictability (Variance Ratio)
  - VR = var(5-min returns) / (5 * var(1-min returns))
  - VR < 1 → mean-reverting → fade
  - VR > 1 → trending → follow

EXP SSSS: Close-to-Close vs High-Low Volatility Ratio
  - Parkinson vol (from H-L) vs close-to-close vol
  - Divergence indicates hidden information flow

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


def load_bars_with_volume(symbol, dates, data_dir='data', chunk_days=10):
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
    out_file = 'results/v42ai_price_impact.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42ai: PRICE IMPACT + MICROSTRUCTURE ORDER FLOW")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    for symbol in ['SOLUSDT', 'DOGEUSDT', 'XRPUSDT']:
        bars = load_bars_with_volume(symbol, all_dates, chunk_days=10)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {symbol}")
        print(f"{'#'*80}")

        ret_1m = bars['close'].pct_change()

        # ── EXP PPPP: KYLE'S LAMBDA (PRICE IMPACT) ──
        print(f"\n  --- EXP PPPP: PRICE IMPACT (KYLE'S LAMBDA) ---")

        price_impact = ret_1m.abs() / (bars['volume'] + 1) * 1e6
        pi_roll = price_impact.rolling(60, min_periods=30).mean()
        pi_std = price_impact.rolling(60, min_periods=30).std()
        pi_z = (price_impact - pi_roll) / (pi_std + 1e-10)

        print(f"  Price impact z stats: mean={pi_z.mean():.3f}, P95={pi_z.quantile(0.95):.3f}")

        for z_thresh in [2, 3]:
            high_impact = pi_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_impact]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0  # fade in high-impact (illiquid) regime
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Price impact z>{z_thresh} (fade in illiquid):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP QQQQ: AMIHUD ILLIQUIDITY ──
        print(f"\n  --- EXP QQQQ: AMIHUD ILLIQUIDITY ---")

        amihud = ret_1m.abs() / (bars['volume'] / 1e6 + 1e-10)
        amihud_roll = amihud.rolling(60, min_periods=30).mean()
        amihud_std = amihud.rolling(60, min_periods=30).std()
        amihud_z = (amihud - amihud_roll) / (amihud_std + 1e-10)

        for z_thresh in [2, 3]:
            high_amihud = amihud_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_amihud]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Amihud z>{z_thresh} (fade in illiquid):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP RRRR: VARIANCE RATIO ──
        print(f"\n  --- EXP RRRR: VARIANCE RATIO ---")

        var_1m = ret_1m.rolling(30, min_periods=15).var()
        ret_5m = bars['close'].pct_change(5)
        var_5m = ret_5m.rolling(6, min_periods=3).var()  # 6 non-overlapping 5-min periods
        vr = var_5m / (5 * var_1m + 1e-20)

        print(f"  VR stats: mean={vr.mean():.3f}, P10={vr.quantile(0.10):.3f}, P90={vr.quantile(0.90):.3f}")

        # VR < 1 → mean-reverting → fade
        for vr_thresh in [0.5, 0.7]:
            mr_regime = vr < vr_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[mr_regime]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  VR < {vr_thresh} (MR regime, fade):")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} test={te_r['tot']/28:+.3f}%/d")

        # ── EXP SSSS: PARKINSON VS CLOSE-TO-CLOSE VOL ──
        print(f"\n  --- EXP SSSS: PARKINSON VS CLOSE-TO-CLOSE VOL ---")

        # Parkinson volatility (from H-L)
        hl_ratio = np.log(bars['high'] / bars['low'])
        parkinson_vol = hl_ratio.rolling(30, min_periods=15).apply(
            lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))), raw=True)

        # Close-to-close volatility
        cc_vol = ret_1m.rolling(30, min_periods=15).std()

        # Ratio: Parkinson / CC
        vol_ratio = parkinson_vol / (cc_vol + 1e-10)
        vr_roll = vol_ratio.rolling(60, min_periods=30).mean()
        vr_std = vol_ratio.rolling(60, min_periods=30).std()
        vr_z = (vol_ratio - vr_roll) / (vr_std + 1e-10)

        for z_thresh in [2, 3]:
            # High ratio → range-based vol >> return-based vol → hidden info
            high_vr = vr_z > z_thresh

            train_trades = []; test_trades = []
            lt = None
            for ts in bars.index[high_vr]:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.get_loc(ts)
                r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
                is_long = r < 0
                t = sim_trade(bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  Parkinson/CC vol z>{z_thresh} (fade):")
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
