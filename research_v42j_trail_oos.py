#!/usr/bin/env python3
"""
v42j: Trailing Stop OOS Validation + Combined Best Parameters

EXP Z2: Validate trailing stop on TRUE OOS (Jul 11–Aug 7, 28d)
  - Test best trailing configs from v42i on unseen data
  - All 3 symbols, cross-symbol contagion
  - Walk-forward on full 88d (May 12–Aug 7): train 60d, test 28d

EXP Z3: Combine ALL best parameters discovered so far
  - Trailing stop (act=3, dist=2)
  - Cascade window=180s (vs 60s baseline)
  - min_ev=3 (vs 2 baseline)
  - LONG direction filter
  - Exclude bad hours (08,09,13,16)
  - Cross-symbol contagion
  - Test each enhancement individually and combined

RAM-safe: chunked bar loading.
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
BAD_HOURS = {8, 9, 13, 16}


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
        if (i+1) % 10 == 0:
            el = time.time()-t0; eta = el/(i+1)*(n-i-1)
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


def run_strat(cascades, bars, offset=0.15, tp=0.15, sl=0.50, max_hold=30, cooldown=300,
              trail_activate=None, trail_dist=None,
              direction_filter=None, hour_filter=None):
    trades = []
    last_time = None
    use_trail = trail_activate is not None and trail_dist is not None

    for c in cascades:
        if last_time and (c['end'] - last_time).total_seconds() < cooldown: continue
        if hour_filter and c['end'].hour in hour_filter: continue

        is_long = c['buy_dominant']
        if direction_filter == 'long_only' and not is_long: continue
        if direction_filter == 'short_only' and is_long: continue

        idx = bars.index.searchsorted(c['end'])
        if idx >= len(bars) - max_hold or idx < 1: continue
        price = bars.iloc[idx]['close']

        if is_long:
            lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
        else:
            lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)

        filled = False
        for j in range(idx, min(idx+max_hold, len(bars))):
            b = bars.iloc[j]
            if is_long and b['low'] <= lim: filled=True; fi=j; break
            elif not is_long and b['high'] >= lim: filled=True; fi=j; break
        if not filled: continue

        ep = None; er = 'timeout'

        if use_trail:
            best_profit = 0; trailing_active = False; current_sl = sl_p
            for k in range(fi, min(fi+max_hold, len(bars))):
                b = bars.iloc[k]
                if is_long:
                    cp = (b['high'] - lim) / lim
                    if cp > best_profit: best_profit = cp
                    if best_profit >= trail_activate / 10000 and not trailing_active:
                        trailing_active = True
                        current_sl = lim * (1 + trail_dist / 10000)
                    if trailing_active:
                        new_sl = b['high'] * (1 - trail_dist / 10000)
                        if new_sl > current_sl: current_sl = new_sl
                    if b['low'] <= current_sl:
                        ep = current_sl; er = 'trail' if trailing_active else 'sl'; break
                    if b['high'] >= tp_p: ep = tp_p; er = 'tp'; break
                else:
                    cp = (lim - b['low']) / lim
                    if cp > best_profit: best_profit = cp
                    if best_profit >= trail_activate / 10000 and not trailing_active:
                        trailing_active = True
                        current_sl = lim * (1 - trail_dist / 10000)
                    if trailing_active:
                        new_sl = b['low'] * (1 + trail_dist / 10000)
                        if new_sl < current_sl: current_sl = new_sl
                    if b['high'] >= current_sl:
                        ep = current_sl; er = 'trail' if trailing_active else 'sl'; break
                    if b['low'] <= tp_p: ep = tp_p; er = 'tp'; break
        else:
            for k in range(fi, min(fi+max_hold, len(bars))):
                b = bars.iloc[k]
                if is_long:
                    if b['low'] <= sl_p: ep=sl_p; er='sl'; break
                    if b['high'] >= tp_p: ep=tp_p; er='tp'; break
                else:
                    if b['high'] >= sl_p: ep=sl_p; er='sl'; break
                    if b['low'] <= tp_p: ep=tp_p; er='tp'; break

        if ep is None: ep = bars.iloc[min(fi+max_hold, len(bars)-1)]['close']
        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
        trades.append({'net': gross-fee, 'exit': er, 'time': bars.index[fi]})
        last_time = c['end']
    return trades


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
    out_file = 'results/v42j_trail_oos.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']

    # Full period: May 12 – Aug 7 = 88 days
    all_dates = get_dates('2025-05-12', 88)
    train_dates = get_dates('2025-05-12', 60)
    test_dates = get_dates('2025-07-11', 28)

    print("="*80)
    print(f"  v42j: TRAILING STOP OOS + COMBINED BEST PARAMS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load all liqs for full period
    all_liq = {}
    for sym in symbols:
        all_liq[sym] = load_liqs(sym, all_dates)
        print(f"  {sym}: {len(all_liq[sym]):,} liquidations")
    gc.collect()

    # Load bars for full period
    all_bars = {}
    for sym in symbols:
        all_bars[sym] = load_bars_chunked(sym, all_dates, chunk_days=10)
        gc.collect()

    print(f"\n  [{ram_str()}] all data loaded")

    split_ts = pd.Timestamp('2025-07-11')

    # ══════════════════════════════════════════════════════════════════════
    # EXP Z2: TRAILING STOP ON TRUE OOS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP Z2: TRAILING STOP — WALK-FORWARD (train=60d, test=28d)")
    print(f"{'#'*80}")

    for sym in symbols:
        print(f"\n  === {sym} ===")
        # Detect cascades on full period
        cascades_full = detect_cascades(all_liq[sym], pct_thresh=95, window=60, min_ev=2)
        train_c = [c for c in cascades_full if c['end'] < split_ts]
        test_c = [c for c in cascades_full if c['end'] >= split_ts]
        print(f"  Cascades: train={len(train_c)}, test={len(test_c)}")

        # Baseline (no trail)
        for label, ta, td in [("Baseline (no trail)", None, None),
                                ("trail act=3 dist=2", 3, 2),
                                ("trail act=5 dist=3", 5, 3),
                                ("trail act=8 dist=3", 8, 3)]:
            tr = run_strat(train_c, all_bars[sym], trail_activate=ta, trail_dist=td)
            te = run_strat(test_c, all_bars[sym], trail_activate=ta, trail_dist=td)
            print(f"\n  {label}:")
            ts_r = pstats(tr, "TRAIN")
            te_r = pstats(te, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP Z2b: TRAILING + CROSS-SYMBOL on OOS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP Z2b: TRAILING + CROSS-SYMBOL CONTAGION (OOS)")
    print(f"{'#'*80}")

    # ETH cascades as triggers for all symbols
    eth_cascades = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95, window=60, min_ev=2)
    eth_train = [c for c in eth_cascades if c['end'] < split_ts]
    eth_test = [c for c in eth_cascades if c['end'] >= split_ts]

    for target in symbols:
        for label, ta, td in [("no trail", None, None), ("trail 3/2", 3, 2)]:
            tr = run_strat(eth_train, all_bars[target], trail_activate=ta, trail_dist=td)
            te = run_strat(eth_test, all_bars[target], trail_activate=ta, trail_dist=td)
            print(f"\n  ETH→{target[:3]} {label}:")
            ts_r = pstats(tr, "TRAIN")
            te_r = pstats(te, "TEST")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP Z3: COMBINE ALL BEST PARAMETERS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP Z3: COMBINED BEST PARAMS — ABLATION STUDY")
    print(f"{'#'*80}")

    # Use SOL for ablation (representative)
    print(f"\n  Using SOLUSDT, walk-forward train=60d test=28d")

    configs = [
        ("A: Baseline",                       95, 60,  2, None, None, None, None),
        ("B: + window=180s",                   95, 180, 2, None, None, None, None),
        ("C: + min_ev=3",                      95, 60,  3, None, None, None, None),
        ("D: + trail 3/2",                     95, 60,  2, 3,    2,    None, None),
        ("E: + LONG only",                     95, 60,  2, None, None, 'long_only', None),
        ("F: + no bad hours",                  95, 60,  2, None, None, None, BAD_HOURS),
        ("G: window=180 + min_ev=3",           95, 180, 3, None, None, None, None),
        ("H: trail 3/2 + LONG",               95, 60,  2, 3,    2,    'long_only', None),
        ("I: trail 3/2 + no bad hrs",          95, 60,  2, 3,    2,    None, BAD_HOURS),
        ("J: trail 3/2 + LONG + no bad",       95, 60,  2, 3,    2,    'long_only', BAD_HOURS),
        ("K: ALL (180s+ev3+trail+LONG+noBad)", 95, 180, 3, 3,    2,    'long_only', BAD_HOURS),
        ("L: ALL except LONG filter",          95, 180, 3, 3,    2,    None, BAD_HOURS),
        ("M: ALL except hour filter",          95, 180, 3, 3,    2,    'long_only', None),
    ]

    for label, pt, win, mev, ta, td, df, hf in configs:
        sol_c = detect_cascades(all_liq['SOLUSDT'], pct_thresh=pt, window=win, min_ev=mev)
        train_c = [c for c in sol_c if c['end'] < split_ts]
        test_c = [c for c in sol_c if c['end'] >= split_ts]

        tr = run_strat(train_c, all_bars['SOLUSDT'], trail_activate=ta, trail_dist=td,
                       direction_filter=df, hour_filter=hf)
        te = run_strat(test_c, all_bars['SOLUSDT'], trail_activate=ta, trail_dist=td,
                       direction_filter=df, hour_filter=hf)

        print(f"\n  {label}:")
        ts_r = pstats(tr, "TRAIN")
        te_r = pstats(te, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP Z3b: BEST CONFIG — FULL PORTFOLIO ON OOS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP Z3b: BEST CONFIG PORTFOLIO — OOS (Jul 11–Aug 7)")
    print(f"{'#'*80}")

    # Best config from ablation: trail 3/2, default cascade params
    # (avoid overfitting with too many filters)
    oos_trades = []
    for target in symbols:
        if target == 'ETHUSDT':
            combined = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95, window=60, min_ev=2)
        else:
            own = detect_cascades(all_liq[target], pct_thresh=95, window=60, min_ev=2)
            eth = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95, window=60, min_ev=2)
            combined = sorted(own + eth, key=lambda c: c['end'])

        test_c = [c for c in combined if c['end'] >= split_ts]
        trades = run_strat(test_c, all_bars[target], trail_activate=3, trail_dist=2)
        pstats(trades, f"OOS {target[:3]} (trail 3/2)")
        oos_trades.extend(trades)

    oos_trades.sort(key=lambda t: t['time'])
    if oos_trades:
        arr = np.array([t['net'] for t in oos_trades])
        eq = np.cumprod(1 + arr)
        tot = (eq[-1]-1)*100
        dd = min((eq[i]/eq[:i+1].max()-1) for i in range(len(eq)))*100
        wr = (arr>0).mean()*100
        print(f"\n  OOS PORTFOLIO (trail 3/2, combined triggers):")
        print(f"    Trades: {len(arr)}, WR: {wr:.1f}%, Avg: {arr.mean()*10000:+.1f}bps, "
              f"Total: {tot:+.2f}%, MaxDD: {dd:.2f}%")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
