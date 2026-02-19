#!/usr/bin/env python3
"""
v42q: XRPUSDT Generalization + Cascade Predictor

EXP RR: Test on XRPUSDT (4th symbol) — does strategy generalize?
  - Same cascade MM with trailing stop
  - Same liq acceleration signal
  - Walk-forward: train=60d, test=28d
  - Cross-symbol contagion from ETH

EXP QQ: Real-Time Cascade Predictor
  - Use liq rate (count/min) + liq volume/min as features
  - When features exceed threshold, predict cascade is starting
  - Enter BEFORE cascade is officially detected
  - Compare: predictor entry vs standard cascade entry

88 days, RAM-safe.
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


def load_bars_chunked(symbol, dates, data_dir='data', chunk_days=10):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} bars...", end='', flush=True)
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


def load_liqs(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "liquidations"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} liqs...", end='', flush=True)
    recs = []
    for i, d in enumerate(dates):
        for hr in range(24):
            f = base / f"liquidation_{d}_hr{hr:02d}.jsonl.gz"
            if not f.exists(): continue
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for ev in data['result']['data']:
                                recs.append({
                                    'timestamp': pd.to_datetime(ev['T'], unit='ms'),
                                    'side': ev['S'], 'volume': float(ev['v']),
                                    'price': float(ev['p']),
                                })
                    except: continue
        if (i+1) % 15 == 0:
            el = time.time()-t0
            print(f" [{i+1}/{n} {el:.0f}s]", end='', flush=True)
    if not recs: print(" NO DATA"); return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    print(f" {len(df):,} ({time.time()-t0:.0f}s) [{ram_str()}]")
    return df


def detect_cascades(liq_df, pct_thresh=95, window=60, min_ev=2):
    if liq_df.empty: return []
    vol_thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= vol_thresh]
    cascades = []
    current = []
    for _, row in large.iterrows():
        if not current: current = [row]
        else:
            dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
            if dt <= window: current.append(row)
            else:
                if len(current) >= min_ev:
                    cdf = pd.DataFrame(current)
                    bn = cdf[cdf['side']=='Buy']['notional'].sum()
                    sn = cdf[cdf['side']=='Sell']['notional'].sum()
                    cascades.append({'start': cdf['timestamp'].min(), 'end': cdf['timestamp'].max(),
                                     'total_notional': bn+sn, 'buy_dominant': bn > sn})
                current = [row]
    if len(current) >= min_ev:
        cdf = pd.DataFrame(current)
        bn = cdf[cdf['side']=='Buy']['notional'].sum()
        sn = cdf[cdf['side']=='Sell']['notional'].sum()
        cascades.append({'start': cdf['timestamp'].min(), 'end': cdf['timestamp'].max(),
                         'total_notional': bn+sn, 'buy_dominant': bn > sn})
    return cascades


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
    out_file = 'results/v42q_xrp_generalize.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42q: XRPUSDT GENERALIZATION + CASCADE PREDICTOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # ══════════════════════════════════════════════════════════════════════
    # EXP RR: XRPUSDT GENERALIZATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP RR: XRPUSDT — DOES STRATEGY GENERALIZE?")
    print(f"{'#'*80}")

    xrp_liq = load_liqs('XRPUSDT', all_dates)
    xrp_bars = load_bars_chunked('XRPUSDT', all_dates, chunk_days=10)
    gc.collect()

    if xrp_liq.empty or xrp_bars.empty:
        print("  ⚠️ No XRP data available, skipping")
    else:
        xrp_cascades = detect_cascades(xrp_liq, pct_thresh=95)
        print(f"  XRP cascades: {len(xrp_cascades)}")

        # Same-symbol cascade MM with trail
        print(f"\n  SAME-SYMBOL CASCADE MM (trail 3/2):")
        train_c = [c for c in xrp_cascades if c['end'] < split_ts]
        test_c = [c for c in xrp_cascades if c['end'] >= split_ts]

        train_trades = []; test_trades = []
        lt = None
        for c in train_c:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = xrp_bars.index.searchsorted(c['end'])
            t = sim_trade(xrp_bars, idx, c['buy_dominant'])
            if t: train_trades.append(t); lt = c['end']
        lt = None
        for c in test_c:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = xrp_bars.index.searchsorted(c['end'])
            t = sim_trade(xrp_bars, idx, c['buy_dominant'])
            if t: test_trades.append(t); lt = c['end']

        ts_r = pstats(train_trades, "XRP TRAIN")
        te_r = pstats(test_trades, "XRP TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # Cross-symbol: ETH cascades → XRP
        print(f"\n  CROSS-SYMBOL: ETH cascades → XRP:")
        eth_liq = load_liqs('ETHUSDT', all_dates)
        eth_cascades = detect_cascades(eth_liq, pct_thresh=95)
        del eth_liq; gc.collect()

        eth_train = [c for c in eth_cascades if c['end'] < split_ts]
        eth_test = [c for c in eth_cascades if c['end'] >= split_ts]

        train_trades = []; test_trades = []
        lt = None
        for c in eth_train:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = xrp_bars.index.searchsorted(c['end'])
            t = sim_trade(xrp_bars, idx, c['buy_dominant'])
            if t: train_trades.append(t); lt = c['end']
        lt = None
        for c in eth_test:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = xrp_bars.index.searchsorted(c['end'])
            t = sim_trade(xrp_bars, idx, c['buy_dominant'])
            if t: test_trades.append(t); lt = c['end']

        ts_r = pstats(train_trades, "ETH→XRP TRAIN")
        te_r = pstats(test_trades, "ETH→XRP TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # Combined (ETH + XRP own cascades)
        print(f"\n  COMBINED (ETH + XRP own cascades):")
        combined = sorted(xrp_cascades + eth_cascades, key=lambda c: c['end'])
        train_c = [c for c in combined if c['end'] < split_ts]
        test_c = [c for c in combined if c['end'] >= split_ts]

        train_trades = []; test_trades = []
        lt = None
        for c in train_c:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = xrp_bars.index.searchsorted(c['end'])
            t = sim_trade(xrp_bars, idx, c['buy_dominant'])
            if t: train_trades.append(t); lt = c['end']
        lt = None
        for c in test_c:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = xrp_bars.index.searchsorted(c['end'])
            t = sim_trade(xrp_bars, idx, c['buy_dominant'])
            if t: test_trades.append(t); lt = c['end']

        ts_r = pstats(train_trades, "Combined TRAIN")
        te_r = pstats(test_trades, "Combined TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # Liq acceleration on XRP
        print(f"\n  LIQ ACCELERATION ON XRP:")
        liq_ts = xrp_liq.set_index('timestamp')
        buy_vol = liq_ts[liq_ts['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
        sell_vol = liq_ts[liq_ts['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
        total_vol = (buy_vol + sell_vol).reindex(xrp_bars.index, fill_value=0)
        buy_vol = buy_vol.reindex(xrp_bars.index, fill_value=0)
        sell_vol = sell_vol.reindex(xrp_bars.index, fill_value=0)

        for w, th, cd in [(15, 5, 60), (15, 5, 300)]:
            roll_avg = total_vol.rolling(w, min_periods=1).mean()
            ratio = total_vol / (roll_avg + 1)
            signals = ratio[ratio > th].index

            train_trades = []; test_trades = []
            lt = None
            for ts in signals:
                if lt and (ts - lt).total_seconds() < cd: continue
                idx = xrp_bars.index.searchsorted(ts)
                if idx >= len(xrp_bars) - 30 or idx < 1: continue
                bv = buy_vol.iloc[idx] if idx < len(buy_vol) else 0
                sv = sell_vol.iloc[idx] if idx < len(sell_vol) else 0
                is_long = bv > sv
                t = sim_trade(xrp_bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  w={w}m thresh={th}x cd={cd}s:")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    del xrp_liq, xrp_bars; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # EXP QQ: REAL-TIME CASCADE PREDICTOR
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP QQ: REAL-TIME CASCADE PREDICTOR — SOLUSDT")
    print(f"{'#'*80}")

    sol_liq = load_liqs('SOLUSDT', all_dates)
    sol_bars = load_bars_chunked('SOLUSDT', all_dates, chunk_days=10)
    gc.collect()

    # Build per-minute features
    liq_ts = sol_liq.set_index('timestamp')
    liq_vol_1m = liq_ts['notional'].resample('1min').sum().fillna(0).reindex(sol_bars.index, fill_value=0)
    liq_cnt_1m = liq_ts['notional'].resample('1min').count().fillna(0).reindex(sol_bars.index, fill_value=0)
    buy_vol = liq_ts[liq_ts['side']=='Buy']['notional'].resample('1min').sum().fillna(0).reindex(sol_bars.index, fill_value=0)
    sell_vol = liq_ts[liq_ts['side']=='Sell']['notional'].resample('1min').sum().fillna(0).reindex(sol_bars.index, fill_value=0)

    # Rolling averages for normalization
    liq_vol_avg = liq_vol_1m.rolling(15, min_periods=1).mean()
    liq_cnt_avg = liq_cnt_1m.rolling(15, min_periods=1).mean()

    # Predictor: when BOTH liq volume AND liq count spike simultaneously
    vol_ratio = liq_vol_1m / (liq_vol_avg + 1)
    cnt_ratio = liq_cnt_1m / (liq_cnt_avg + 1)

    print(f"\n  PREDICTOR: Combined liq volume + count spike")
    for vol_th in [3, 5]:
        for cnt_th in [2, 3]:
            signals = (vol_ratio > vol_th) & (cnt_ratio > cnt_th)
            signal_times = sol_bars.index[signals]

            train_trades = []; test_trades = []
            lt = None
            for ts in signal_times:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = sol_bars.index.get_loc(ts)
                bv = buy_vol.iloc[idx]; sv = sell_vol.iloc[idx]
                is_long = bv > sv
                t = sim_trade(sol_bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  vol>{vol_th}x AND cnt>{cnt_th}x:")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # Predictor with price velocity
    price_ret = sol_bars['close'].pct_change().abs()
    price_avg = price_ret.rolling(15, min_periods=1).mean()
    price_ratio = price_ret / (price_avg + 1e-10)

    print(f"\n  PREDICTOR: Liq volume + price velocity")
    for vol_th in [3, 5]:
        for price_th in [2, 3]:
            signals = (vol_ratio > vol_th) & (price_ratio > price_th)
            signal_times = sol_bars.index[signals]

            train_trades = []; test_trades = []
            lt = None
            for ts in signal_times:
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = sol_bars.index.get_loc(ts)
                bv = buy_vol.iloc[idx]; sv = sell_vol.iloc[idx]
                is_long = bv > sv
                t = sim_trade(sol_bars, idx, is_long)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            print(f"\n  vol>{vol_th}x AND price>{price_th}x:")
            ts_r = pstats(train_trades, "TRAIN")
            te_r = pstats(test_trades, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
