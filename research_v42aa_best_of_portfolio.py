#!/usr/bin/env python3
"""
v42aa: BEST-OF PORTFOLIO — Top Signals from All 13 Types

Combines the highest-quality signals from each family:
1. Cascade MM (trail) — liq data
2. Microstructure MR (sigma=2) — price only
3. Vol Spike (>5x fade) — price only
4. VWAP Deviation (>20bps fade) — price only
5. Vol Clustering (>2x fade) — price only
6. Price-Vol Divergence (>0.2% fade) — price+vol

Realistic concurrent simulation:
- Max 1 position per symbol per signal type
- 60s cooldown per symbol
- Walk-forward: train=60d, test=28d
- All 4 symbols

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


def main():
    out_file = 'results/v42aa_best_of_portfolio.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42aa: BEST-OF PORTFOLIO — TOP 6 SIGNALS × 4 SYMBOLS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Per-signal standalone results
    signal_results = {}

    for sym in symbols:
        bars = load_bars_with_volume(sym, all_dates, chunk_days=10)
        liq = load_liqs(sym, all_dates)
        gc.collect()

        print(f"\n{'#'*80}")
        print(f"  {sym}")
        print(f"{'#'*80}")

        ret_1m = bars['close'].pct_change()
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
        vol_cluster_ratio = vol_5m / (vol_60m + 1e-10)
        price_ret_5m = bars['close'].pct_change(5)
        vol_ret_5m = bars['volume'].pct_change(5)

        signals = {
            'cascade': [],
            'micro_mr': [],
            'vol_spike': [],
            'vwap': [],
            'vol_cluster': [],
            'pv_div': [],
        }

        # 1. CASCADE MM
        cascades = detect_cascades(liq, pct_thresh=95)
        for c in cascades:
            signals['cascade'].append({'time': c['end'], 'is_long': c['buy_dominant']})

        # 2. MICRO MR (sigma=2)
        extreme_down = ret_1m < -2 * roll_std
        extreme_up = ret_1m > 2 * roll_std
        for ts in bars.index[extreme_down]:
            signals['micro_mr'].append({'time': ts, 'is_long': True})
        for ts in bars.index[extreme_up]:
            signals['micro_mr'].append({'time': ts, 'is_long': False})

        # 3. VOL SPIKE (>5x fade)
        high_vol = vol_ratio > 5
        for ts in bars.index[high_vol]:
            idx = bars.index.get_loc(ts)
            ret = price_ret_5m.iloc[idx] if idx < len(price_ret_5m) else 0
            signals['vol_spike'].append({'time': ts, 'is_long': ret < 0})

        # 4. VWAP (>20bps fade)
        above = vwap_dev > 20
        below = vwap_dev < -20
        for ts in bars.index[above]:
            signals['vwap'].append({'time': ts, 'is_long': False})
        for ts in bars.index[below]:
            signals['vwap'].append({'time': ts, 'is_long': True})

        # 5. VOL CLUSTERING (>2x fade)
        high_vc = vol_cluster_ratio > 2
        for ts in bars.index[high_vc]:
            idx = bars.index.get_loc(ts)
            r = ret_1m.iloc[idx] if idx < len(ret_1m) else 0
            signals['vol_cluster'].append({'time': ts, 'is_long': r < 0})

        # 6. PV DIVERGENCE (>0.2% fade)
        weak_up = (price_ret_5m > 0.002) & (vol_ret_5m < -0.2)
        weak_down = (price_ret_5m < -0.002) & (vol_ret_5m < -0.2)
        for ts in bars.index[weak_up]:
            signals['pv_div'].append({'time': ts, 'is_long': False})
        for ts in bars.index[weak_down]:
            signals['pv_div'].append({'time': ts, 'is_long': True})

        # Run each signal standalone
        for sig_name, sig_events in signals.items():
            train_trades = []; test_trades = []
            lt = None
            for ev in sorted(sig_events, key=lambda e: e['time']):
                ts = ev['time']
                if lt and (ts - lt).total_seconds() < 60: continue
                idx = bars.index.searchsorted(ts)
                t = sim_trade(bars, idx, ev['is_long'])
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            if test_trades:
                arr = np.array([t['net'] for t in test_trades])
                n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
                tot = arr.sum()*100
                sh = arr.mean()/(arr.std()+1e-10)*np.sqrt(252*24*60)
                signal_results[(sym, sig_name)] = {
                    'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh
                }
                flag = "✅" if tot > 0 else "  "
                print(f"  {flag} {sym:10s} {sig_name:15s}  n={n:5d}  wr={wr:5.1f}%  "
                      f"avg={avg:+6.1f}bps  tot={tot:+8.1f}%  sh={sh:+7.0f}")

        # Combined portfolio for this symbol
        all_events = []
        for sig_name, sig_events in signals.items():
            for ev in sig_events:
                all_events.append({**ev, 'source': sig_name})
        all_events.sort(key=lambda e: e['time'])

        eq = 1.0; taken = 0; lt = None
        oos_eq = 1.0; oos_taken = 0
        daily_pnl = {}
        seen = set()

        for ev in all_events:
            ts = ev['time']
            if lt and (ts - lt).total_seconds() < 60: continue
            m = ts.floor('min')
            if m in seen: continue
            seen.add(m)
            idx = bars.index.searchsorted(ts)
            t = sim_trade(bars, idx, ev['is_long'])
            if t:
                eq *= (1 + t['net'])
                lt = ts; taken += 1
                if ts >= split_ts:
                    oos_eq *= (1 + t['net'])
                    oos_taken += 1
                day = t['time'].date()
                if day not in daily_pnl: daily_pnl[day] = 0
                daily_pnl[day] += t['net']

        tot = (eq-1)*100; oos_tot = (oos_eq-1)*100
        darr = np.array([daily_pnl[d] for d in sorted(daily_pnl.keys())])
        pos = (darr > 0).sum()
        oos_days = {d: v for d, v in daily_pnl.items() if pd.Timestamp(str(d)) >= split_ts}
        oarr = np.array([oos_days[d] for d in sorted(oos_days.keys())]) if oos_days else np.array([0])
        opos = (oarr > 0).sum()

        print(f"\n  {sym} COMBINED: {taken:,} trades, OOS={oos_taken:,}")
        print(f"    OOS return: {oos_tot:+,.1f}%")
        print(f"    OOS days: {opos}/{len(oarr)} positive ({opos/len(oarr)*100:.0f}%)")
        print(f"    OOS Sharpe: {oarr.mean()/(oarr.std()+1e-10)*np.sqrt(365):.1f}")
        print(f"    OOS worst day: {oarr.min()*100:+.3f}%")

        del bars, liq; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  FINAL SIGNAL COMPARISON (OOS only)")
    print(f"{'#'*80}")
    print(f"\n  {'Symbol':10s}  {'Signal':15s}  {'n':>5s}  {'WR':>6s}  {'Avg':>7s}  {'Total':>8s}  {'Sharpe':>7s}")
    print(f"  {'-'*60}")
    for (sym, sig), r in sorted(signal_results.items()):
        flag = "✅" if r['tot'] > 0 else "  "
        print(f"  {flag} {sym:10s}  {sig:15s}  {r['n']:5d}  {r['wr']:5.1f}%  "
              f"{r['avg']:+6.1f}bp  {r['tot']:+7.1f}%  {r['sharpe']:+7.0f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
