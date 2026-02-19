#!/usr/bin/env python3
"""
v42w: Signal Combination & Quality Filters

EXP FFF: Use OI/spread as quality filters for cascade MM
  - Only take cascade trades when OI is dropping (squeeze)
  - Only take cascade trades when spread is wide (stressed)
  - Does filtering improve per-trade quality?

EXP GGG: Multi-timeframe confirmation
  - Cascade signal + micro MR signal within 5 min = high conviction
  - How often do signals from different families align?
  - Does alignment predict better trades?

EXP HHH: Anti-correlation exploitation
  - When cascade MM says long but micro MR says short = conflicting
  - Skip conflicting signals → improve quality?

SOLUSDT, 88 days, walk-forward.
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


def load_ticker_1m(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} ticker...", end='', flush=True)
    all_recs = []
    for i, d in enumerate(dates):
        for hr in range(24):
            f = base / f"ticker_{d}_hr{hr:02d}.jsonl.gz"
            if not f.exists(): continue
            try:
                with gzip.open(f, 'rt') as fh:
                    for line in fh:
                        try:
                            data = json.loads(line)
                            item = data['result']['list'][0]
                            all_recs.append({
                                'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                                'oi': float(item['openInterest']),
                                'bid1': float(item['bid1Price']),
                                'ask1': float(item['ask1Price']),
                                'last_price': float(item['lastPrice']),
                            })
                        except: continue
            except: continue
        if (i+1) % 15 == 0:
            el = time.time()-t0
            print(f" [{i+1}/{n} {el:.0f}s]", end='', flush=True)
    if not all_recs: print(" NO DATA"); return pd.DataFrame()
    df = pd.DataFrame(all_recs).sort_values('timestamp').reset_index(drop=True)
    df['spread_bps'] = (df['ask1'] - df['bid1']) / df['last_price'] * 10000
    df = df.set_index('timestamp')
    ticker_1m = df.resample('1min').agg({
        'oi': 'last', 'spread_bps': 'mean', 'last_price': 'last',
    }).dropna(subset=['oi'])
    print(f" {len(ticker_1m):,} 1m bars ({time.time()-t0:.0f}s) [{ram_str()}]")
    return ticker_1m


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
    out_file = 'results/v42w_signal_combos.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbol = 'SOLUSDT'
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42w: SIGNAL COMBINATIONS & QUALITY FILTERS — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    liq = load_liqs(symbol, all_dates)
    bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
    ticker = load_ticker_1m(symbol, all_dates)
    gc.collect()

    # Prepare signals
    cascades = detect_cascades(liq, pct_thresh=95)
    print(f"  Cascades: {len(cascades)}")

    # Ticker features aligned to bars
    oi = ticker['oi'].reindex(bars.index, method='ffill')
    spread = ticker['spread_bps'].reindex(bars.index, method='ffill')
    oi_pct_5m = oi.pct_change(5) * 100
    spread_roll = spread.rolling(60, min_periods=30).mean()
    spread_std = spread.rolling(60, min_periods=30).std()

    # Micro MR signals
    ret_1m = bars['close'].pct_change()
    roll_std = ret_1m.rolling(60, min_periods=30).std()
    extreme_down = ret_1m < -2 * roll_std
    extreme_up = ret_1m > 2 * roll_std

    # ══════════════════════════════════════════════════════════════════════
    # EXP FFF: OI/SPREAD AS QUALITY FILTERS FOR CASCADE MM
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP FFF: OI/SPREAD QUALITY FILTERS FOR CASCADE MM")
    print(f"{'#'*80}")

    # Baseline: all cascades
    train_all = []; test_all = []
    lt = None
    for c in cascades:
        if lt and (c['end']-lt).total_seconds() < 60: continue
        idx = bars.index.searchsorted(c['end'])
        t = sim_trade(bars, idx, c['buy_dominant'])
        if t:
            if c['end'] < split_ts: train_all.append(t)
            else: test_all.append(t)
            lt = c['end']

    print(f"\n  BASELINE (all cascades):")
    pstats(train_all, "TRAIN")
    pstats(test_all, "TEST")

    # Filter: only when OI is dropping (squeeze)
    oi_drop_thresh = oi_pct_5m.quantile(0.20)
    train_f = []; test_f = []
    lt = None
    for c in cascades:
        if lt and (c['end']-lt).total_seconds() < 60: continue
        idx = bars.index.searchsorted(c['end'])
        if idx >= len(oi_pct_5m): continue
        if oi_pct_5m.iloc[idx] > oi_drop_thresh: continue  # skip if OI not dropping
        t = sim_trade(bars, idx, c['buy_dominant'])
        if t:
            if c['end'] < split_ts: train_f.append(t)
            else: test_f.append(t)
            lt = c['end']

    print(f"\n  FILTER: OI dropping (P20):")
    ts_r = pstats(train_f, "TRAIN (OI filter)")
    te_r = pstats(test_f, "TEST (OI filter)")
    if ts_r and te_r:
        print(f"    Filtered: {ts_r['n']+te_r['n']} / {len(train_all)+len(test_all)} trades kept")

    # Filter: only when spread is wide
    train_f = []; test_f = []
    lt = None
    for c in cascades:
        if lt and (c['end']-lt).total_seconds() < 60: continue
        idx = bars.index.searchsorted(c['end'])
        if idx >= len(spread): continue
        s = spread.iloc[idx]; sr = spread_roll.iloc[idx]; ss = spread_std.iloc[idx]
        if pd.isna(sr) or pd.isna(ss) or s <= sr + ss: continue  # skip if spread not wide
        t = sim_trade(bars, idx, c['buy_dominant'])
        if t:
            if c['end'] < split_ts: train_f.append(t)
            else: test_f.append(t)
            lt = c['end']

    print(f"\n  FILTER: Wide spread (z>1):")
    ts_r = pstats(train_f, "TRAIN (spread filter)")
    te_r = pstats(test_f, "TEST (spread filter)")
    if ts_r and te_r:
        print(f"    Filtered: {ts_r['n']+te_r['n']} / {len(train_all)+len(test_all)} trades kept")

    # Filter: OI dropping AND spread wide
    train_f = []; test_f = []
    lt = None
    for c in cascades:
        if lt and (c['end']-lt).total_seconds() < 60: continue
        idx = bars.index.searchsorted(c['end'])
        if idx >= len(oi_pct_5m) or idx >= len(spread): continue
        oi_ok = oi_pct_5m.iloc[idx] <= oi_drop_thresh
        s = spread.iloc[idx]; sr = spread_roll.iloc[idx]; ss = spread_std.iloc[idx]
        spread_ok = not pd.isna(sr) and not pd.isna(ss) and s > sr + ss
        if not (oi_ok or spread_ok): continue
        t = sim_trade(bars, idx, c['buy_dominant'])
        if t:
            if c['end'] < split_ts: train_f.append(t)
            else: test_f.append(t)
            lt = c['end']

    print(f"\n  FILTER: OI drop OR wide spread:")
    ts_r = pstats(train_f, "TRAIN (OI OR spread)")
    te_r = pstats(test_f, "TEST (OI OR spread)")
    if ts_r and te_r:
        print(f"    Filtered: {ts_r['n']+te_r['n']} / {len(train_all)+len(test_all)} trades kept")

    # ══════════════════════════════════════════════════════════════════════
    # EXP GGG: MULTI-SIGNAL ALIGNMENT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP GGG: MULTI-SIGNAL ALIGNMENT")
    print(f"{'#'*80}")

    # For each cascade, check if micro MR also fires within ±5 min
    mr_long_times = set(bars.index[extreme_down])
    mr_short_times = set(bars.index[extreme_up])

    aligned = 0; not_aligned = 0
    train_aligned = []; test_aligned = []
    train_not = []; test_not = []
    lt = None
    for c in cascades:
        if lt and (c['end']-lt).total_seconds() < 60: continue
        idx = bars.index.searchsorted(c['end'])
        t = sim_trade(bars, idx, c['buy_dominant'])
        if not t: continue

        # Check if micro MR agrees within ±5 min
        has_mr = False
        for offset in range(-5, 6):
            check_idx = idx + offset
            if check_idx < 0 or check_idx >= len(bars): continue
            check_ts = bars.index[check_idx]
            if c['buy_dominant'] and check_ts in mr_long_times: has_mr = True; break
            if not c['buy_dominant'] and check_ts in mr_short_times: has_mr = True; break

        if has_mr:
            aligned += 1
            if c['end'] < split_ts: train_aligned.append(t)
            else: test_aligned.append(t)
        else:
            not_aligned += 1
            if c['end'] < split_ts: train_not.append(t)
            else: test_not.append(t)
        lt = c['end']

    print(f"\n  Cascade + Micro MR alignment: {aligned}/{aligned+not_aligned} ({100*aligned/(aligned+not_aligned):.0f}%)")
    print(f"\n  ALIGNED (cascade + micro MR agree):")
    pstats(train_aligned, "TRAIN")
    pstats(test_aligned, "TEST")
    print(f"\n  NOT ALIGNED (cascade only):")
    pstats(train_not, "TRAIN")
    pstats(test_not, "TEST")

    # ══════════════════════════════════════════════════════════════════════
    # EXP HHH: SIGNAL CONFLICT DETECTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP HHH: SIGNAL CONFLICT DETECTION")
    print(f"{'#'*80}")

    # For each cascade, check if micro MR fires in OPPOSITE direction
    conflicting = 0; non_conflicting = 0
    train_conf = []; test_conf = []
    train_noconf = []; test_noconf = []
    lt = None
    for c in cascades:
        if lt and (c['end']-lt).total_seconds() < 60: continue
        idx = bars.index.searchsorted(c['end'])
        t = sim_trade(bars, idx, c['buy_dominant'])
        if not t: continue

        has_conflict = False
        for offset in range(-5, 6):
            check_idx = idx + offset
            if check_idx < 0 or check_idx >= len(bars): continue
            check_ts = bars.index[check_idx]
            # Conflict: cascade says long but MR says short (or vice versa)
            if c['buy_dominant'] and check_ts in mr_short_times: has_conflict = True; break
            if not c['buy_dominant'] and check_ts in mr_long_times: has_conflict = True; break

        if has_conflict:
            conflicting += 1
            if c['end'] < split_ts: train_conf.append(t)
            else: test_conf.append(t)
        else:
            non_conflicting += 1
            if c['end'] < split_ts: train_noconf.append(t)
            else: test_noconf.append(t)
        lt = c['end']

    print(f"\n  Conflicting signals: {conflicting}/{conflicting+non_conflicting} ({100*conflicting/(conflicting+non_conflicting):.0f}%)")
    print(f"\n  CONFLICTING (cascade vs micro MR disagree):")
    pstats(train_conf, "TRAIN")
    pstats(test_conf, "TEST")
    print(f"\n  NON-CONFLICTING (no opposing signal):")
    pstats(train_noconf, "TRAIN")
    pstats(test_noconf, "TEST")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
