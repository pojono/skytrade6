#!/usr/bin/env python3
"""
v42b: Expand promising signals from v42 to 30 days

1. CASCADE SIZE FILTERING on 30 days SOLUSDT
   - Does P97/P99 still dominate P90/P95?
   - Walk-forward: train on first 20 days, test on last 10

2. INTRADAY SEASONALITY on 30 days SOLUSDT
   - Are hour-of-day patterns stable across weeks?
   - Week-by-week consistency check

Start with SOLUSDT only. If promising, expand.
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


def load_futures_trades(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time()
    n = len(dates)
    print(f"  Loading futures {n} days...", end='', flush=True)
    dfs = []
    for i, d in enumerate(dates):
        f = base / f"{symbol}{d}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, usecols=['timestamp', 'side', 'size', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            dfs.append(df)
        if (i+1) % 10 == 0:
            el = time.time() - t0
            eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not dfs:
        print(" NO DATA"); return pd.DataFrame()
    r = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f" {len(r):,} trades ({time.time()-t0:.0f}s) [{ram_str()}]")
    return r


def load_liquidations_dates(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "liquidations"
    t0 = time.time()
    n = len(dates)
    print(f"  Loading liqs {n} days...", end='', flush=True)
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
                                    'side': ev['S'],
                                    'volume': float(ev['v']),
                                    'price': float(ev['p']),
                                })
                    except: continue
        if (i+1) % 10 == 0:
            el = time.time() - t0
            eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not recs:
        print(" NO DATA"); return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    print(f" {len(df):,} ({time.time()-t0:.0f}s) [{ram_str()}]")
    return df


def detect_cascades(liq_df, pct_thresh=95, window=60, min_ev=2):
    vol_thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= vol_thresh]
    cascades = []
    current = []
    for _, row in large.iterrows():
        if not current:
            current = [row]
        else:
            dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
            if dt <= window:
                current.append(row)
            else:
                if len(current) >= min_ev:
                    cdf = pd.DataFrame(current)
                    bn = cdf[cdf['side']=='Buy']['notional'].sum()
                    sn = cdf[cdf['side']=='Sell']['notional'].sum()
                    cascades.append({
                        'end': cdf['timestamp'].max(),
                        'total_notional': bn+sn,
                        'buy_dominant': bn > sn,
                        'n_events': len(cdf),
                    })
                current = [row]
    if len(current) >= min_ev:
        cdf = pd.DataFrame(current)
        bn = cdf[cdf['side']=='Buy']['notional'].sum()
        sn = cdf[cdf['side']=='Sell']['notional'].sum()
        cascades.append({
            'end': cdf['timestamp'].max(),
            'total_notional': bn+sn,
            'buy_dominant': bn > sn,
            'n_events': len(cdf),
        })
    return cascades


def run_cascade_strategy(cascades, bars, offset_pct=0.20, tp_pct=0.20, sl_pct=0.50,
                         max_hold=30, cooldown=300):
    trades = []
    last_time = None
    for c in cascades:
        if last_time and (c['end'] - last_time).total_seconds() < cooldown:
            continue
        idx = bars.index.searchsorted(c['end'])
        if idx >= len(bars) - max_hold or idx < 1:
            continue
        price = bars.iloc[idx]['close']
        is_long = c['buy_dominant']
        if is_long:
            lim = price * (1 - offset_pct/100)
            tp = lim * (1 + tp_pct/100)
            sl = lim * (1 - sl_pct/100)
        else:
            lim = price * (1 + offset_pct/100)
            tp = lim * (1 - tp_pct/100)
            sl = lim * (1 + sl_pct/100)
        # Fill
        filled = False
        for j in range(idx, min(idx+max_hold, len(bars))):
            b = bars.iloc[j]
            if is_long and b['low'] <= lim: filled=True; fi=j; break
            elif not is_long and b['high'] >= lim: filled=True; fi=j; break
        if not filled: continue
        # Exit
        ep = None; er = 'timeout'
        for k in range(fi, min(fi+max_hold, len(bars))):
            b = bars.iloc[k]
            if is_long:
                if b['low'] <= sl: ep=sl; er='sl'; break
                if b['high'] >= tp: ep=tp; er='tp'; break
            else:
                if b['high'] >= sl: ep=sl; er='sl'; break
                if b['low'] <= tp: ep=tp; er='tp'; break
        if ep is None:
            ep = bars.iloc[min(fi+max_hold, len(bars)-1)]['close']
        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
        trades.append({'net': gross-fee, 'gross': gross, 'exit': er,
                       'time': bars.index[fi], 'notional': c['total_notional']})
        last_time = c['end']
    return trades


def print_trade_stats(trades, label):
    if not trades:
        print(f"    {label:30s}  NO TRADES")
        return
    arr = np.array([t['net'] for t in trades])
    n = len(arr)
    wr = (arr>0).mean()*100
    avg = arr.mean()*10000
    tot = arr.sum()*100
    std = arr.std()
    sharpe = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "  "
    print(f"  {flag} {label:30s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.2f}%  sharpe={sharpe:+8.1f}")


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'SOLUSDT'
    out_file = f'results/v42b_cascade_size_{symbol}.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    print("="*80)
    print(f"  v42b: CASCADE SIZE + SEASONALITY — {symbol} — 30 DAYS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    dates_30d = get_dates('2025-05-12', 30)

    # ── LOAD DATA ──
    liq_df = load_liquidations_dates(symbol, dates_30d)
    fut = load_futures_trades(symbol, dates_30d)

    print("  Building 1-min bars...", end='', flush=True)
    bars = fut.set_index('timestamp')['price'].resample('1min').agg(
        open='first', high='max', low='min', close='last').dropna()
    print(f" {len(bars):,} bars")
    del fut; gc.collect()
    print(f"  [{ram_str()}] (freed futures)")

    days = (bars.index.max() - bars.index.min()).total_seconds() / 86400
    print(f"  Period: {bars.index.min()} to {bars.index.max()} ({days:.0f} days)")

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: CASCADE SIZE FILTERING — 30 DAYS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 1: CASCADE SIZE FILTERING — {symbol} — 30 DAYS")
    print(f"{'#'*80}")

    # Test different thresholds
    print(f"\n  THRESHOLD COMPARISON (full 30 days):")
    print(f"  {'Thresh':8s}  {'Cascades':>8s}  {'Fills':>5s}  {'WR%':>6s}  {'AvgNet':>8s}  {'TotNet':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*65}")

    for pct in [90, 93, 95, 97, 99]:
        cascades = detect_cascades(liq_df, pct_thresh=pct)
        trades = run_cascade_strategy(cascades, bars)
        print_trade_stats(trades, f"P{pct}")

    # Walk-forward: train on first 20 days, test on last 10
    print(f"\n  WALK-FORWARD: train=20d, test=10d")
    split = bars.index.min() + pd.Timedelta(days=20)
    print(f"  Split: {split.strftime('%Y-%m-%d')}")

    for pct in [90, 95, 97]:
        cascades = detect_cascades(liq_df, pct_thresh=pct)
        train_c = [c for c in cascades if c['end'] < split]
        test_c = [c for c in cascades if c['end'] >= split]

        # Best config from v41: offset=0.20-0.25, TP=0.15-0.25, SL=0.50
        for off, tp, sl in [(0.20, 0.20, 0.50), (0.25, 0.20, 0.50), (0.20, 0.15, 0.50)]:
            train_trades = run_cascade_strategy(train_c, bars, off, tp, sl)
            test_trades = run_cascade_strategy(test_c, bars, off, tp, sl)
            label = f"P{pct} off={off} tp={tp} sl={sl}"
            print(f"\n  {label}:")
            print_trade_stats(train_trades, "TRAIN")
            print_trade_stats(test_trades, "TEST")

    # Test: does cascade NOTIONAL predict edge? (30 days)
    print(f"\n  CASCADE NOTIONAL → EDGE (30 days, P95):")
    cascades_95 = detect_cascades(liq_df, pct_thresh=95)
    if cascades_95:
        notionals = [c['total_notional'] for c in cascades_95]
        p33 = np.percentile(notionals, 33)
        p66 = np.percentile(notionals, 66)
        for label, filt in [
            ("SMALL (bottom 1/3)", lambda c: c['total_notional'] < p33),
            ("MEDIUM (middle 1/3)", lambda c: p33 <= c['total_notional'] < p66),
            ("LARGE (top 1/3)", lambda c: c['total_notional'] >= p66),
        ]:
            filtered = [c for c in cascades_95 if filt(c)]
            trades = run_cascade_strategy(filtered, bars)
            print_trade_stats(trades, label)

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: INTRADAY SEASONALITY — 30 DAYS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 2: INTRADAY SEASONALITY — {symbol} — 30 DAYS")
    print(f"{'#'*80}")

    bars_close = bars['close'].copy()
    bars_close = bars_close.to_frame('price')
    bars_close['ret_1h'] = bars_close['price'].pct_change(60)
    bars_close['hour'] = bars_close.index.hour
    bars_close['week'] = bars_close.index.isocalendar().week.astype(int)

    # Overall hour stats
    print(f"\n  HOUR-OF-DAY RETURNS (30 days):")
    print(f"  {'Hr':>4s}  {'N':>6s}  {'AvgBps':>8s}  {'Sharpe':>8s}  {'WR%':>6s}  {'|Avg|':>8s}")
    print(f"  {'-'*50}")

    hourly = []
    for hr in range(24):
        sub = bars_close[bars_close['hour']==hr].dropna(subset=['ret_1h'])
        if len(sub) < 20: continue
        avg = sub['ret_1h'].mean()
        std = sub['ret_1h'].std()
        sh = avg/(std+1e-10)*np.sqrt(365)
        wr = (sub['ret_1h']>0).mean()*100
        hourly.append({'hr': hr, 'avg': avg, 'std': std, 'sharpe': sh, 'wr': wr, 'n': len(sub)})
        flag = "✅" if abs(sh) > 1.0 else "  "
        print(f"  {flag} {hr:02d}:00  {len(sub):>6d}  {avg*10000:>+7.2f}  {sh:>+7.2f}  {wr:>5.1f}%  {abs(avg)*10000:>7.2f}")

    # Week-by-week consistency
    if hourly:
        best_hr = max(hourly, key=lambda x: x['sharpe'])['hr']
        worst_hr = min(hourly, key=lambda x: x['sharpe'])['hr']

        print(f"\n  WEEK-BY-WEEK CONSISTENCY for best hour ({best_hr:02d}:00) and worst hour ({worst_hr:02d}:00):")
        print(f"  {'Week':>6s}  {'Best_hr avg':>12s}  {'Worst_hr avg':>12s}  {'Spread':>10s}")
        print(f"  {'-'*50}")

        weeks = sorted(bars_close['week'].unique())
        consistent_best = 0
        consistent_worst = 0
        for w in weeks:
            wk = bars_close[bars_close['week']==w]
            best_sub = wk[(wk['hour']==best_hr)].dropna(subset=['ret_1h'])
            worst_sub = wk[(wk['hour']==worst_hr)].dropna(subset=['ret_1h'])
            if len(best_sub) < 3 or len(worst_sub) < 3: continue
            b_avg = best_sub['ret_1h'].mean()*10000
            w_avg = worst_sub['ret_1h'].mean()*10000
            spread = b_avg - w_avg
            if b_avg > 0: consistent_best += 1
            if w_avg < 0: consistent_worst += 1
            flag = "✅" if spread > 0 else "  "
            print(f"  {flag} W{w:02d}    {b_avg:>+10.2f}  {w_avg:>+10.2f}  {spread:>+9.2f}")

        print(f"\n  Best hour positive: {consistent_best}/{len(weeks)} weeks")
        print(f"  Worst hour negative: {consistent_worst}/{len(weeks)} weeks")
        if consistent_best >= len(weeks) * 0.6 and consistent_worst >= len(weeks) * 0.6:
            print(f"  ✅ SEASONALITY IS CONSISTENT")
        else:
            print(f"  ❌ Seasonality is NOT consistent across weeks")

    del bars_close, liq_df; gc.collect()

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
