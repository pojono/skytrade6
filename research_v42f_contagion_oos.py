#!/usr/bin/env python3
"""
v42f: Cross-Symbol Cascade Contagion — Walk-Forward OOS

v42e showed: ETH cascade → SOL MM gives 91.5% WR, +5.3 bps, +19% in 30d.

Now validate:
1. Walk-forward OOS on 60 days (train 42d, test 18d)
2. Test ALL cross-symbol pairs (ETH→SOL, ETH→DOGE, SOL→ETH, etc.)
3. Compare contagion strategy vs same-symbol cascade MM
4. Rolling window stability
5. Test: does combining contagion + same-symbol improve results?

RAM-SAFE: load futures one symbol at a time → build bars → delete raw trades.
Only bars (~86k rows × 4 cols = ~3MB each) and cascades (lists) stay in memory.
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


def ram_check(label=""):
    avail = psutil.virtual_memory().available / 1024**3
    if avail < 1.0:
        print(f"\n  ⚠️ LOW RAM ({avail:.1f}GB avail) at {label} — forcing gc")
        gc.collect()
        avail = psutil.virtual_memory().available / 1024**3
        print(f"  After gc: {avail:.1f}GB avail")


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
    """Load futures trades in chunks, build bars incrementally. Never holds
    more than chunk_days of raw trades in memory at once."""
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time()
    n = len(dates)
    print(f"  Loading {symbol} bars (chunked, {chunk_days}d)...", end='', flush=True)

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
            bars = chunk.set_index('timestamp')['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last').dropna()
            all_bars.append(bars)
            del chunk
            gc.collect()

        done = min(start + chunk_days, n)
        el = time.time() - t0
        eta = el / done * (n - done) if done > 0 else 0
        print(f" [{done}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)

    if not all_bars:
        print(" NO DATA"); return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    # Remove any duplicate indices from chunk boundaries
    result = result[~result.index.duplicated(keep='first')]
    print(f" {len(result):,} bars ({time.time()-t0:.0f}s) [{ram_str()}]")
    return result


def load_liqs(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "liquidations"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} liqs {n}d...", end='', flush=True)
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
            el = time.time()-t0; eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
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


def run_contagion_strat(trigger_cascades, target_bars, offset=0.15, tp=0.15, sl=0.50,
                        max_hold=30, cooldown=300):
    """Use trigger symbol's cascades to trade target symbol."""
    trades = []
    last_time = None
    for c in trigger_cascades:
        if last_time and (c['end'] - last_time).total_seconds() < cooldown: continue
        idx = target_bars.index.searchsorted(c['end'])
        if idx >= len(target_bars) - max_hold or idx < 1: continue
        price = target_bars.iloc[idx]['close']
        is_long = c['buy_dominant']
        if is_long:
            lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
        else:
            lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)
        filled = False
        for j in range(idx, min(idx+max_hold, len(target_bars))):
            b = target_bars.iloc[j]
            if is_long and b['low'] <= lim: filled=True; fi=j; break
            elif not is_long and b['high'] >= lim: filled=True; fi=j; break
        if not filled: continue
        ep = None; er = 'timeout'
        for k in range(fi, min(fi+max_hold, len(target_bars))):
            b = target_bars.iloc[k]
            if is_long:
                if b['low'] <= sl_p: ep=sl_p; er='sl'; break
                if b['high'] >= tp_p: ep=tp_p; er='tp'; break
            else:
                if b['high'] >= sl_p: ep=sl_p; er='sl'; break
                if b['low'] <= tp_p: ep=tp_p; er='tp'; break
        if ep is None: ep = target_bars.iloc[min(fi+max_hold, len(target_bars)-1)]['close']
        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
        trades.append({'net': gross-fee, 'time': target_bars.index[fi], 'exit': er})
        last_time = c['end']
    return trades


def pstats(trades, label):
    if not trades:
        print(f"    {label:45s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "  "
    print(f"  {flag} {label:45s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.2f}%  sh={sh:+8.1f}")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh}


def main():
    out_file = 'results/v42f_contagion_oos.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    n_days = 60
    dates = get_dates('2025-05-12', n_days)

    print("="*80)
    print(f"  v42f: CROSS-SYMBOL CONTAGION OOS — {n_days} DAYS (RAM-SAFE)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']

    # ── STEP 1: Load all liquidations (small: ~200k rows each, ~50MB total) ──
    print(f"\n  --- Loading liquidations (small) ---")
    cascades = {}
    for sym in symbols:
        liq = load_liqs(sym, dates)
        cascades[sym] = detect_cascades(liq, pct_thresh=95)
        print(f"  {sym} cascades: {len(cascades[sym])}")
        del liq
    gc.collect()
    ram_check("after liqs")

    # ── STEP 2: Load bars ONE symbol at a time (chunked, ~3MB each) ──
    print(f"\n  --- Loading bars (chunked, RAM-safe) ---")
    bars = {}
    for sym in symbols:
        bars[sym] = load_bars_chunked(sym, dates, chunk_days=10)
        gc.collect()
        ram_check(f"after {sym} bars")

    days_actual = (bars['SOLUSDT'].index.max() - bars['SOLUSDT'].index.min()).total_seconds() / 86400
    print(f"\n  Period: ~{days_actual:.0f} days  [{ram_str()}]")

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: ALL CROSS-SYMBOL PAIRS (full 60 days)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 1: ALL CROSS-SYMBOL PAIRS — FULL {n_days} DAYS")
    print(f"{'#'*80}")

    for off, tp, sl in [(0.15, 0.15, 0.50), (0.20, 0.20, 0.50)]:
        print(f"\n  Config: off={off} tp={tp} sl={sl}")
        for trigger in symbols:
            for target in symbols:
                label = f"{trigger[:3]}→{target[:3]}"
                if trigger == target:
                    label += " (same-symbol)"
                trades = run_contagion_strat(cascades[trigger], bars[target], off, tp, sl)
                pstats(trades, label)

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: WALK-FORWARD OOS — ALL PAIRS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 2: WALK-FORWARD OOS (70/30 split)")
    print(f"{'#'*80}")

    split_day = int(days_actual * 0.7)
    split_ts = bars['SOLUSDT'].index.min() + pd.Timedelta(days=split_day)
    test_days = int(days_actual) - split_day
    print(f"  Split: {split_ts.strftime('%Y-%m-%d')} (train={split_day}d, test={test_days}d)")

    for off, tp, sl in [(0.15, 0.15, 0.50), (0.20, 0.20, 0.50)]:
        print(f"\n  Config: off={off} tp={tp} sl={sl}")
        for trigger in symbols:
            for target in symbols:
                label = f"{trigger[:3]}→{target[:3]}"
                train_c = [c for c in cascades[trigger] if c['end'] < split_ts]
                test_c = [c for c in cascades[trigger] if c['end'] >= split_ts]

                train_trades = run_contagion_strat(train_c, bars[target], off, tp, sl)
                test_trades = run_contagion_strat(test_c, bars[target], off, tp, sl)

                ts = pstats(train_trades, f"TRAIN {label}")
                te = pstats(test_trades, f"TEST  {label}")

                if ts and te:
                    train_daily = ts['tot'] / max(split_day, 1)
                    test_daily = te['tot'] / max(test_days, 1)
                    oos = "✅ OOS+" if te['tot'] > 0 else "❌ OOS-"
                    print(f"    {oos}  train={train_daily:+.3f}%/d  test={test_daily:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3: ROLLING WINDOW STABILITY — KEY PAIRS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 3: ROLLING 15-DAY WINDOWS")
    print(f"{'#'*80}")

    for trigger, target in [('ETHUSDT', 'SOLUSDT'), ('ETHUSDT', 'DOGEUSDT'),
                             ('SOLUSDT', 'SOLUSDT'), ('ETHUSDT', 'ETHUSDT')]:
        label = f"{trigger[:3]}→{target[:3]}"
        print(f"\n  {label} (off=0.15 tp=0.15 sl=0.50):")

        ws = bars[target].index.min()
        pos = 0; total = 0
        while ws + pd.Timedelta(days=15) <= bars[target].index.max():
            we = ws + pd.Timedelta(days=15)
            w_cascades = [c for c in cascades[trigger] if ws <= c['end'] < we]
            if len(w_cascades) >= 3:
                w_trades = run_contagion_strat(w_cascades, bars[target], 0.15, 0.15, 0.50)
                if w_trades:
                    arr = np.array([t['net'] for t in w_trades])
                    n = len(arr); wr = (arr>0).mean()*100; tot = arr.sum()*100
                    flag = "✅" if tot > 0 else "❌"
                    print(f"    {flag} {ws.strftime('%m-%d')}→{we.strftime('%m-%d')}  "
                          f"n={n:3d}  wr={wr:5.1f}%  tot={tot:+6.2f}%")
                    total += 1
                    if tot > 0: pos += 1
            ws += pd.Timedelta(days=7)

        if total > 0:
            print(f"    Positive: {pos}/{total} ({pos/total*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  PART 4: COMBINED SAME + CROSS SYMBOL")
    print(f"{'#'*80}")

    for target in ['SOLUSDT', 'DOGEUSDT']:
        print(f"\n  Target: {target}")
        same_trades = run_contagion_strat(cascades[target], bars[target], 0.15, 0.15, 0.50)
        pstats(same_trades, f"{target[:3]}→{target[:3]} (same only)")

        eth_trades = run_contagion_strat(cascades['ETHUSDT'], bars[target], 0.15, 0.15, 0.50)
        pstats(eth_trades, f"ETH→{target[:3]} (cross only)")

        combined_cascades = sorted(
            cascades[target] + cascades['ETHUSDT'],
            key=lambda c: c['end']
        )
        combined_trades = run_contagion_strat(combined_cascades, bars[target], 0.15, 0.15, 0.50)
        pstats(combined_trades, f"ETH+{target[:3]}→{target[:3]} (combined)")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
