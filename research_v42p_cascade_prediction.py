#!/usr/bin/env python3
"""
v42p: Cascade Prediction + Regime Adaptation

EXP NN: Can we PREDICT cascades 1-5 minutes early?
  - Before each cascade, look at:
    1. Liquidation rate acceleration (small liqs before big ones)
    2. Price velocity (rapid price move precedes cascade)
    3. Volume spike (trade volume surges before cascade)
  - If predictable, enter BEFORE cascade detection → better fills
  - Compare: pre-cascade entry vs post-cascade entry

EXP OO: Regime Detection — Adapt Strategy to Market Conditions
  - Split data into high-vol and low-vol regimes
  - Does cascade MM work better in one regime?
  - Can we dynamically adjust parameters (offset, TP, SL)?

EXP PP: Cascade Duration as Real-Time Signal
  - As a cascade unfolds, does the DURATION so far predict total size?
  - If a cascade has been going for 30s, is it likely to continue?
  - Strategy: wait for cascade to reach certain duration before entering

88 days, SOL for initial testing, RAM-safe.
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
                                     'total_notional': bn+sn, 'buy_dominant': bn > sn,
                                     'n_events': len(cdf),
                                     'duration_s': (cdf['timestamp'].max()-cdf['timestamp'].min()).total_seconds()})
                current = [row]
    if len(current) >= min_ev:
        cdf = pd.DataFrame(current)
        bn = cdf[cdf['side']=='Buy']['notional'].sum()
        sn = cdf[cdf['side']=='Sell']['notional'].sum()
        cascades.append({'start': cdf['timestamp'].min(), 'end': cdf['timestamp'].max(),
                         'total_notional': bn+sn, 'buy_dominant': bn > sn,
                         'n_events': len(cdf),
                         'duration_s': (cdf['timestamp'].max()-cdf['timestamp'].min()).total_seconds()})
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
    out_file = 'results/v42p_cascade_prediction.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbol = 'SOLUSDT'
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42p: CASCADE PREDICTION + REGIME ADAPTATION — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    liq = load_liqs(symbol, all_dates)
    bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
    gc.collect()
    print(f"\n  [{ram_str()}] data loaded")

    cascades = detect_cascades(liq, pct_thresh=95)
    print(f"  Cascades: {len(cascades)}")

    # ══════════════════════════════════════════════════════════════════════
    # EXP NN: PRE-CASCADE SIGNALS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP NN: PRE-CASCADE SIGNALS")
    print(f"{'#'*80}")

    # Build 1-min liq volume bars
    liq_ts = liq.set_index('timestamp')
    liq_1m = liq_ts['notional'].resample('1min').sum().fillna(0).reindex(bars.index, fill_value=0)
    liq_count = liq_ts['notional'].resample('1min').count().fillna(0).reindex(bars.index, fill_value=0)

    # Price velocity (1-min return)
    price_ret = bars['close'].pct_change().abs()

    # For each cascade, look at the 1-5 minutes BEFORE
    print(f"\n  PRE-CASCADE PROFILE (avg of {len(cascades)} cascades):")
    for lookback in [1, 2, 3, 5, 10]:
        liq_before = []; price_before = []; count_before = []
        liq_normal = []; price_normal = []
        for c in cascades:
            idx = bars.index.searchsorted(c['start'])
            if idx < lookback + 1: continue
            # Pre-cascade
            liq_before.append(liq_1m.iloc[idx-lookback:idx].mean())
            price_before.append(price_ret.iloc[idx-lookback:idx].mean())
            count_before.append(liq_count.iloc[idx-lookback:idx].mean())
            # Normal (10 min before that)
            if idx >= lookback + 11:
                liq_normal.append(liq_1m.iloc[idx-lookback-10:idx-lookback].mean())
                price_normal.append(price_ret.iloc[idx-lookback-10:idx-lookback].mean())

        if liq_before and liq_normal:
            liq_ratio = np.mean(liq_before) / (np.mean(liq_normal) + 1)
            price_ratio = np.mean(price_before) / (np.mean(price_normal) + 1e-10)
            print(f"  {lookback}min before: liq_vol={np.mean(liq_before):.0f} "
                  f"(vs normal {np.mean(liq_normal):.0f}, ratio={liq_ratio:.1f}x)  "
                  f"price_move={np.mean(price_before)*10000:.1f}bps "
                  f"(vs normal {np.mean(price_normal)*10000:.1f}bps, ratio={price_ratio:.1f}x)  "
                  f"liq_count={np.mean(count_before):.1f}")

    # Strategy: enter when liq volume spikes (pre-cascade detection)
    print(f"\n  EARLY ENTRY: Enter at cascade START vs cascade END:")
    for entry_point, label in [('start', 'Cascade START'), ('end', 'Cascade END')]:
        trades = []
        last_time = None
        for c in cascades:
            ts = c[entry_point]
            if last_time and (ts - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(ts)
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t: trades.append(t); last_time = ts
        pstats(trades, label)

    # Enter N minutes BEFORE cascade start
    print(f"\n  PRE-ENTRY: Enter N minutes BEFORE cascade start:")
    for pre_min in [0, 1, 2, 3, 5]:
        trades = []
        last_time = None
        for c in cascades:
            ts = c['start'] - pd.Timedelta(minutes=pre_min)
            if last_time and (ts - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(ts)
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t: trades.append(t); last_time = ts
        pstats(trades, f"Entry {pre_min}min before start")

    # ══════════════════════════════════════════════════════════════════════
    # EXP OO: REGIME DETECTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP OO: REGIME DETECTION — VOLATILITY ADAPTATION")
    print(f"{'#'*80}")

    # Compute rolling volatility (60-min)
    vol_60 = bars['close'].pct_change().rolling(60).std() * np.sqrt(60) * 100
    vol_median = vol_60.median()
    print(f"  60-min rolling vol median: {vol_median:.3f}%")

    # Split cascades by regime
    high_vol_c = []; low_vol_c = []
    for c in cascades:
        idx = bars.index.searchsorted(c['end'])
        if idx < 60 or idx >= len(vol_60): continue
        v = vol_60.iloc[idx]
        if pd.isna(v): continue
        if v > vol_median:
            high_vol_c.append(c)
        else:
            low_vol_c.append(c)

    print(f"  High-vol cascades: {len(high_vol_c)}")
    print(f"  Low-vol cascades:  {len(low_vol_c)}")

    print(f"\n  BASELINE (all cascades):")
    all_trades = []
    last_time = None
    for c in cascades:
        if last_time and (c['end'] - last_time).total_seconds() < 60: continue
        idx = bars.index.searchsorted(c['end'])
        t = sim_trade(bars, idx, c['buy_dominant'])
        if t: all_trades.append(t); last_time = c['end']
    pstats(all_trades, "All cascades")

    print(f"\n  BY REGIME:")
    for regime_c, label in [(high_vol_c, "High-vol regime"), (low_vol_c, "Low-vol regime")]:
        trades = []
        last_time = None
        for c in regime_c:
            if last_time and (c['end'] - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(c['end'])
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t: trades.append(t); last_time = c['end']
        pstats(trades, label)

    # Adaptive parameters: wider offset/TP in high-vol, tighter in low-vol
    print(f"\n  ADAPTIVE PARAMETERS:")
    for regime_c, label, off, tp in [(high_vol_c, "High-vol (off=0.20 tp=0.20)", 0.20, 0.20),
                                      (low_vol_c, "Low-vol (off=0.10 tp=0.10)", 0.10, 0.10),
                                      (high_vol_c, "High-vol (off=0.15 tp=0.15)", 0.15, 0.15),
                                      (low_vol_c, "Low-vol (off=0.15 tp=0.15)", 0.15, 0.15)]:
        trades = []
        last_time = None
        for c in regime_c:
            if last_time and (c['end'] - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(c['end'])
            t = sim_trade(bars, idx, c['buy_dominant'], offset=off, tp=tp)
            if t: trades.append(t); last_time = c['end']
        pstats(trades, label)

    # ══════════════════════════════════════════════════════════════════════
    # EXP PP: CASCADE DURATION AS SIGNAL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP PP: CASCADE DURATION AS REAL-TIME SIGNAL")
    print(f"{'#'*80}")

    # Does longer cascade = better trade?
    print(f"\n  TRADE BY CASCADE DURATION:")
    dur_buckets = [(0, 5, '<5s'), (5, 15, '5-15s'), (15, 30, '15-30s'),
                   (30, 60, '30-60s'), (60, 300, '>60s')]
    for lo, hi, label in dur_buckets:
        sub = [c for c in cascades if lo <= c['duration_s'] < hi]
        trades = []
        last_time = None
        for c in sub:
            if last_time and (c['end'] - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(c['end'])
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t: trades.append(t); last_time = c['end']
        pstats(trades, f"dur {label} ({len(sub)} cascades)")

    # Does n_events predict trade quality?
    print(f"\n  TRADE BY N_EVENTS:")
    for n_min, n_max, label in [(2, 3, '2 events'), (3, 5, '3-4 events'),
                                 (5, 10, '5-9 events'), (10, 100, '10+ events')]:
        sub = [c for c in cascades if n_min <= c['n_events'] < n_max]
        trades = []
        last_time = None
        for c in sub:
            if last_time and (c['end'] - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(c['end'])
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t: trades.append(t); last_time = c['end']
        pstats(trades, f"n_ev={label} ({len(sub)} cascades)")

    # Walk-forward test: train on first 60d, test on last 28d
    print(f"\n  WALK-FORWARD: Duration filter")
    for max_dur in [15, 30, 60]:
        sub = [c for c in cascades if c['duration_s'] <= max_dur]
        train = [c for c in sub if c['end'] < split_ts]
        test = [c for c in sub if c['end'] >= split_ts]
        train_t = []; test_t = []
        lt = None
        for c in train:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = bars.index.searchsorted(c['end'])
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t: train_t.append(t); lt = c['end']
        lt = None
        for c in test:
            if lt and (c['end']-lt).total_seconds() < 60: continue
            idx = bars.index.searchsorted(c['end'])
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t: test_t.append(t); lt = c['end']
        print(f"\n  dur<={max_dur}s:")
        ts_r = pstats(train_t, "TRAIN")
        te_r = pstats(test_t, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
