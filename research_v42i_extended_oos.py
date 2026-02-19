#!/usr/bin/env python3
"""
v42i: Extended OOS + Cascade Params + Trailing Stop

EXP X: TRUE OUT-OF-SAMPLE on Jul 11 – Aug 7 (28 days)
  - Completely unseen period (all prior work used May 12 – Jul 10)
  - Test all 3 symbols with cross-symbol contagion
  - Rolling 7-day windows for stability

EXP Y: Cascade Detection Parameter Sensitivity
  - pct_thresh: 90, 93, 95, 97, 99
  - window: 30, 60, 90, 120 seconds
  - min_ev: 1, 2, 3, 4
  - On the original 60d period

EXP Z: Alternative Exit Structures
  - Trailing stop (move SL to breakeven after X bps profit)
  - Asymmetric TP/SL ratios
  - Time-decay exit (reduce SL over time)

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
    print(f"  Loading {symbol} bars (chunked)...", end='', flush=True)
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


def run_strat(cascades, bars, offset=0.15, tp=0.15, sl=0.50, max_hold=30, cooldown=300):
    trades = []
    last_time = None
    for c in cascades:
        if last_time and (c['end'] - last_time).total_seconds() < cooldown: continue
        idx = bars.index.searchsorted(c['end'])
        if idx >= len(bars) - max_hold or idx < 1: continue
        price = bars.iloc[idx]['close']
        is_long = c['buy_dominant']
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


def run_trailing(cascades, bars, offset=0.15, tp=0.15, sl=0.50,
                 trail_activate_bps=5, trail_dist_bps=3, max_hold=30, cooldown=300):
    """Trailing stop variant: after profit exceeds trail_activate, move SL to entry+trail_dist."""
    trades = []
    last_time = None
    for c in cascades:
        if last_time and (c['end'] - last_time).total_seconds() < cooldown: continue
        idx = bars.index.searchsorted(c['end'])
        if idx >= len(bars) - max_hold or idx < 1: continue
        price = bars.iloc[idx]['close']
        is_long = c['buy_dominant']
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

        # Trailing stop logic
        ep = None; er = 'timeout'
        best_profit = 0
        trailing_active = False
        current_sl = sl_p

        for k in range(fi, min(fi+max_hold, len(bars))):
            b = bars.iloc[k]
            if is_long:
                current_profit = (b['high'] - lim) / lim
                if current_profit > best_profit:
                    best_profit = current_profit
                # Activate trailing
                if best_profit >= trail_activate_bps / 10000 and not trailing_active:
                    trailing_active = True
                    current_sl = lim * (1 + trail_dist_bps / 10000)
                # Update trailing SL
                if trailing_active:
                    new_sl = b['high'] * (1 - trail_dist_bps / 10000)
                    if new_sl > current_sl:
                        current_sl = new_sl
                if b['low'] <= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; break
                if b['high'] >= tp_p: ep=tp_p; er='tp'; break
            else:
                current_profit = (lim - b['low']) / lim
                if current_profit > best_profit:
                    best_profit = current_profit
                if best_profit >= trail_activate_bps / 10000 and not trailing_active:
                    trailing_active = True
                    current_sl = lim * (1 - trail_dist_bps / 10000)
                if trailing_active:
                    new_sl = b['low'] * (1 + trail_dist_bps / 10000)
                    if new_sl < current_sl:
                        current_sl = new_sl
                if b['high'] >= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; break
                if b['low'] <= tp_p: ep=tp_p; er='tp'; break

        if ep is None: ep = bars.iloc[min(fi+max_hold, len(bars)-1)]['close']
        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (TAKER_FEE if er != 'tp' else MAKER_FEE)
        trades.append({'net': gross-fee, 'exit': er, 'time': bars.index[fi]})
        last_time = c['end']
    return trades


def pstats(trades, label):
    if not trades:
        print(f"    {label:50s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "  "
    print(f"  {flag} {label:50s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.2f}%  sh={sh:+8.1f}")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh}


def main():
    out_file = 'results/v42i_extended_oos.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']

    print("="*80)
    print(f"  v42i: EXTENDED OOS + PARAMS + TRAILING STOP")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # ══════════════════════════════════════════════════════════════════════
    # EXP X: TRUE OOS — Jul 11 to Aug 7 (28 days, completely unseen)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP X: TRUE OUT-OF-SAMPLE — Jul 11 to Aug 7 (28 days)")
    print(f"{'#'*80}")

    oos_dates = get_dates('2025-07-11', 28)

    oos_cascades = {}
    for sym in symbols:
        liq = load_liqs(sym, oos_dates)
        oos_cascades[sym] = detect_cascades(liq, pct_thresh=95)
        print(f"  {sym}: {len(oos_cascades[sym])} cascades")
        del liq
    gc.collect()

    oos_bars = {}
    for sym in symbols:
        oos_bars[sym] = load_bars_chunked(sym, oos_dates, chunk_days=10)
        gc.collect()

    print(f"\n  [{ram_str()}] OOS data loaded")

    # Test same-symbol and cross-symbol
    print(f"\n  SAME-SYMBOL CASCADE MM (OOS period):")
    for sym in symbols:
        trades = run_strat(oos_cascades[sym], oos_bars[sym])
        pstats(trades, f"{sym[:3]}→{sym[:3]}")

    print(f"\n  CROSS-SYMBOL (ETH triggers):")
    for target in symbols:
        trades = run_strat(oos_cascades['ETHUSDT'], oos_bars[target])
        pstats(trades, f"ETH→{target[:3]}")

    print(f"\n  COMBINED (ETH + same-symbol triggers):")
    oos_portfolio_trades = {}
    for target in symbols:
        if target == 'ETHUSDT':
            combined = oos_cascades['ETHUSDT']
        else:
            combined = sorted(oos_cascades[target] + oos_cascades['ETHUSDT'],
                              key=lambda c: c['end'])
        trades = run_strat(combined, oos_bars[target])
        oos_portfolio_trades[target] = trades
        pstats(trades, f"Combined→{target[:3]}")

    # Portfolio on OOS
    all_oos = []
    for sym in symbols:
        all_oos.extend(oos_portfolio_trades[sym])
    all_oos.sort(key=lambda t: t['time'])
    if all_oos:
        arr = np.array([t['net'] for t in all_oos])
        eq = np.cumprod(1 + arr)
        tot = (eq[-1]-1)*100
        dd = min((eq[i]/eq[:i+1].max()-1) for i in range(len(eq)))*100
        print(f"\n  OOS PORTFOLIO:")
        print(f"    Trades: {len(arr)}, WR: {(arr>0).mean()*100:.1f}%, "
              f"Avg: {arr.mean()*10000:+.1f}bps, Total: {tot:+.2f}%, MaxDD: {dd:.2f}%")

    # Rolling 7-day windows on OOS
    print(f"\n  ROLLING 7-DAY WINDOWS (OOS):")
    for sym in symbols:
        label = f"Combined→{sym[:3]}"
        if sym == 'ETHUSDT':
            comb = oos_cascades['ETHUSDT']
        else:
            comb = sorted(oos_cascades[sym] + oos_cascades['ETHUSDT'],
                          key=lambda c: c['end'])
        ws = oos_bars[sym].index.min()
        pos = 0; total = 0
        while ws + pd.Timedelta(days=7) <= oos_bars[sym].index.max():
            we = ws + pd.Timedelta(days=7)
            wc = [c for c in comb if ws <= c['end'] < we]
            if len(wc) >= 2:
                wt = run_strat(wc, oos_bars[sym])
                if wt:
                    a = np.array([t['net'] for t in wt])
                    tot = a.sum()*100; wr = (a>0).mean()*100
                    flag = "✅" if tot > 0 else "❌"
                    print(f"    {flag} {label} {ws.strftime('%m-%d')}→{we.strftime('%m-%d')}  "
                          f"n={len(a):3d}  wr={wr:5.1f}%  tot={tot:+6.2f}%")
                    total += 1
                    if tot > 0: pos += 1
            ws += pd.Timedelta(days=3)
        if total > 0:
            print(f"    {label} positive: {pos}/{total} ({pos/total*100:.0f}%)")

    del oos_bars, oos_cascades; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # EXP Y: CASCADE PARAMETER SENSITIVITY (on original 60d)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP Y: CASCADE PARAMETER SENSITIVITY (May 12 – Jul 10)")
    print(f"{'#'*80}")

    orig_dates = get_dates('2025-05-12', 60)

    # Load SOL only for param sweep (saves RAM)
    sol_liq = load_liqs('SOLUSDT', orig_dates)
    sol_bars = load_bars_chunked('SOLUSDT', orig_dates, chunk_days=10)
    gc.collect()

    print(f"\n  PCT_THRESH SWEEP (window=60s, min_ev=2):")
    for pt in [90, 93, 95, 97, 99]:
        c = detect_cascades(sol_liq, pct_thresh=pt)
        t = run_strat(c, sol_bars)
        pstats(t, f"P{pt} ({len(c)} cascades)")

    print(f"\n  WINDOW SWEEP (pct=95, min_ev=2):")
    for w in [15, 30, 60, 90, 120, 180]:
        c = detect_cascades(sol_liq, pct_thresh=95, window=w)
        t = run_strat(c, sol_bars)
        pstats(t, f"window={w}s ({len(c)} cascades)")

    print(f"\n  MIN_EVENTS SWEEP (pct=95, window=60s):")
    for me in [1, 2, 3, 4, 5]:
        c = detect_cascades(sol_liq, pct_thresh=95, window=60, min_ev=me)
        t = run_strat(c, sol_bars)
        pstats(t, f"min_ev={me} ({len(c)} cascades)")

    # ══════════════════════════════════════════════════════════════════════
    # EXP Z: ALTERNATIVE EXIT STRUCTURES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP Z: ALTERNATIVE EXIT STRUCTURES (SOL, 60d)")
    print(f"{'#'*80}")

    cascades_95 = detect_cascades(sol_liq, pct_thresh=95)

    # Baseline
    print(f"\n  BASELINE:")
    t_base = run_strat(cascades_95, sol_bars)
    pstats(t_base, "Fixed TP=15bps SL=50bps")

    # Asymmetric TP/SL
    print(f"\n  ASYMMETRIC TP/SL:")
    for tp, sl in [(0.10, 0.30), (0.10, 0.50), (0.15, 0.30), (0.15, 0.50),
                    (0.20, 0.30), (0.20, 0.50), (0.20, 0.75), (0.25, 0.50),
                    (0.30, 0.50), (0.10, 0.15)]:
        t = run_strat(cascades_95, sol_bars, tp=tp, sl=sl)
        pstats(t, f"TP={tp:.2f}% SL={sl:.2f}%")

    # Trailing stop
    print(f"\n  TRAILING STOP:")
    for act, dist in [(3, 2), (5, 3), (5, 5), (8, 3), (8, 5),
                       (10, 3), (10, 5), (10, 8)]:
        t = run_trailing(cascades_95, sol_bars, trail_activate_bps=act, trail_dist_bps=dist)
        pstats(t, f"trail act={act}bps dist={dist}bps")

    # Different max hold times
    print(f"\n  MAX HOLD TIME:")
    for mh in [10, 15, 20, 30, 45, 60]:
        t = run_strat(cascades_95, sol_bars, max_hold=mh)
        pstats(t, f"max_hold={mh}min")

    # Different offsets
    print(f"\n  ENTRY OFFSET:")
    for off in [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        t = run_strat(cascades_95, sol_bars, offset=off, tp=off, sl=0.50)
        pstats(t, f"offset=TP={off:.2f}% SL=0.50%")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
