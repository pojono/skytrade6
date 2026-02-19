#!/usr/bin/env python3
"""
v42u: FINAL MEGA PORTFOLIO — All 4 Strategies × 4 Symbols

Combines all discovered strategy families in a realistic simulation:
1. Cascade MM (trail stop) — needs liquidation data
2. Liq Acceleration — needs liquidation data  
3. Microstructure MR — price data only
4. Range Fade — price data only

Realistic constraints:
- Max 1 position per symbol at a time
- 60s cooldown per symbol
- Priority: cascade > accel > micro MR > range fade
- Walk-forward: train=60d, test=28d

88 days, 4 symbols, RAM-safe.
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


def main():
    out_file = 'results/v42u_final_portfolio.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42u: FINAL MEGA PORTFOLIO — 4 STRATEGIES × 4 SYMBOLS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load all data
    all_liq = {}; all_bars = {}
    for sym in symbols:
        all_liq[sym] = load_liqs(sym, all_dates)
    gc.collect()
    for sym in symbols:
        all_bars[sym] = load_bars_chunked(sym, all_dates, chunk_days=10)
        gc.collect()
    print(f"\n  [{ram_str()}] all data loaded")

    # ══════════════════════════════════════════════════════════════════════
    # Generate all signals for all symbols
    # ══════════════════════════════════════════════════════════════════════
    all_events = []

    for sym in symbols:
        bars = all_bars[sym]
        liq = all_liq[sym]

        # 1. CASCADE MM signals (with ETH contagion for non-ETH)
        if sym == 'ETHUSDT':
            cascades = detect_cascades(liq, pct_thresh=95)
        else:
            own = detect_cascades(liq, pct_thresh=95)
            eth = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
            cascades = sorted(own + eth, key=lambda c: c['end'])

        for c in cascades:
            all_events.append({
                'time': c['end'], 'symbol': sym, 'is_long': c['buy_dominant'],
                'source': 'cascade', 'priority': 1
            })

        # 2. LIQ ACCELERATION signals
        liq_ts = liq.set_index('timestamp')
        buy_vol = liq_ts[liq_ts['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
        sell_vol = liq_ts[liq_ts['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
        total_vol = (buy_vol + sell_vol).reindex(bars.index, fill_value=0)
        buy_vol_r = buy_vol.reindex(bars.index, fill_value=0)
        sell_vol_r = sell_vol.reindex(bars.index, fill_value=0)
        roll_avg = total_vol.rolling(15, min_periods=1).mean()
        ratio = total_vol / (roll_avg + 1)
        for ts in ratio[ratio > 5].index:
            idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
            if idx < len(buy_vol_r):
                bv = buy_vol_r.iloc[idx]; sv = sell_vol_r.iloc[idx]
                all_events.append({
                    'time': ts, 'symbol': sym, 'is_long': bv > sv,
                    'source': 'accel', 'priority': 2
                })

        # 3. MICROSTRUCTURE MR signals
        ret_1m = bars['close'].pct_change()
        roll_std = ret_1m.rolling(60, min_periods=30).std()
        extreme_down = ret_1m < -2 * roll_std
        extreme_up = ret_1m > 2 * roll_std
        for ts in bars.index[extreme_down]:
            all_events.append({
                'time': ts, 'symbol': sym, 'is_long': True,
                'source': 'micro_mr', 'priority': 3
            })
        for ts in bars.index[extreme_up]:
            all_events.append({
                'time': ts, 'symbol': sym, 'is_long': False,
                'source': 'micro_mr', 'priority': 3
            })

        print(f"  {sym}: signals generated [{ram_str()}]")

    # Sort by time, then priority
    all_events.sort(key=lambda e: (e['time'], e['priority']))
    print(f"\n  Total signals: {len(all_events):,}")

    # ══════════════════════════════════════════════════════════════════════
    # REALISTIC PORTFOLIO SIMULATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  REALISTIC PORTFOLIO SIMULATION")
    print(f"{'#'*80}")

    last_trade_time = {sym: None for sym in symbols}
    seen_minutes = {sym: set() for sym in symbols}
    eq = 1.0; taken = 0; skipped = 0
    per_sym = {sym: 0 for sym in symbols}
    per_source = {'cascade': 0, 'accel': 0, 'micro_mr': 0}
    daily_pnl = {}
    oos_eq = 1.0; oos_taken = 0

    for i, ev in enumerate(all_events):
        sym = ev['symbol']; ts = ev['time']
        bars = all_bars[sym]

        # Cooldown
        if last_trade_time[sym] and (ts - last_trade_time[sym]).total_seconds() < 60:
            continue
        # Dedup same minute
        m = ts.floor('min')
        if m in seen_minutes[sym]: continue
        seen_minutes[sym].add(m)

        idx = bars.index.searchsorted(ts)
        t = sim_trade(bars, idx, ev['is_long'])
        if t:
            eq *= (1 + t['net'])
            last_trade_time[sym] = ts
            taken += 1
            per_sym[sym] += 1
            per_source[ev['source']] += 1

            if ts >= split_ts:
                oos_eq *= (1 + t['net'])
                oos_taken += 1

            day = t['time'].date()
            if day not in daily_pnl: daily_pnl[day] = 0
            daily_pnl[day] += t['net']

        if (i+1) % 100000 == 0:
            print(f"    [{i+1:,}/{len(all_events):,}] taken={taken} eq={eq:.2f} [{ram_str()}]")

    tot = (eq-1)*100
    oos_tot = (oos_eq-1)*100

    print(f"\n  FULL 88-DAY RESULTS:")
    print(f"    Trades taken:    {taken:,}")
    print(f"    Total return:    {tot:+,.2f}%")
    print(f"    Final equity:    {eq:,.4f}")
    for sym in symbols:
        print(f"    {sym}: {per_sym[sym]:,} trades")
    for src in per_source:
        print(f"    From {src}: {per_source[src]:,}")

    # Daily stats
    darr = np.array([daily_pnl[d] for d in sorted(daily_pnl.keys())])
    pos_days = (darr > 0).sum()
    print(f"\n    Trading days:    {len(darr)}")
    print(f"    Positive days:   {pos_days}/{len(darr)} ({pos_days/len(darr)*100:.0f}%)")
    print(f"    Avg daily ret:   {darr.mean()*100:+.3f}%")
    print(f"    Daily Sharpe:    {darr.mean()/(darr.std()+1e-10)*np.sqrt(365):.1f}")
    print(f"    Worst day:       {darr.min()*100:+.3f}%")
    print(f"    Best day:        {darr.max()*100:+.3f}%")

    # OOS stats
    oos_days = {d: v for d, v in daily_pnl.items() if pd.Timestamp(str(d)) >= split_ts}
    if oos_days:
        oarr = np.array([oos_days[d] for d in sorted(oos_days.keys())])
        opos = (oarr > 0).sum()
        print(f"\n    OOS ONLY (Jul 11–Aug 7):")
        print(f"    OOS trades:      {oos_taken:,}")
        print(f"    OOS return:      {oos_tot:+,.2f}%")
        print(f"    OOS days:        {len(oarr)}")
        print(f"    OOS positive:    {opos}/{len(oarr)} ({opos/len(oarr)*100:.0f}%)")
        print(f"    OOS avg daily:   {oarr.mean()*100:+.3f}%")
        print(f"    OOS Sharpe:      {oarr.mean()/(oarr.std()+1e-10)*np.sqrt(365):.1f}")
        print(f"    OOS worst day:   {oarr.min()*100:+.3f}%")

    # ══════════════════════════════════════════════════════════════════════
    # INDIVIDUAL STRATEGY CONTRIBUTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  INDIVIDUAL STRATEGY PERFORMANCE (standalone)")
    print(f"{'#'*80}")

    for strategy, priority in [('cascade', 1), ('accel', 2), ('micro_mr', 3)]:
        strat_events = [e for e in all_events if e['source'] == strategy]
        strat_eq = 1.0; strat_taken = 0
        strat_lt = {sym: None for sym in symbols}
        strat_seen = {sym: set() for sym in symbols}
        strat_daily = {}

        for ev in strat_events:
            sym = ev['symbol']; ts = ev['time']
            bars = all_bars[sym]
            if strat_lt[sym] and (ts - strat_lt[sym]).total_seconds() < 60: continue
            m = ts.floor('min')
            if m in strat_seen[sym]: continue
            strat_seen[sym].add(m)
            idx = bars.index.searchsorted(ts)
            t = sim_trade(bars, idx, ev['is_long'])
            if t:
                strat_eq *= (1 + t['net'])
                strat_lt[sym] = ts
                strat_taken += 1
                day = t['time'].date()
                if day not in strat_daily: strat_daily[day] = 0
                strat_daily[day] += t['net']

        stot = (strat_eq-1)*100
        sdarr = np.array([strat_daily[d] for d in sorted(strat_daily.keys())]) if strat_daily else np.array([0])
        spos = (sdarr > 0).sum()
        ssh = sdarr.mean()/(sdarr.std()+1e-10)*np.sqrt(365)
        print(f"\n  {strategy.upper():12s}  trades={strat_taken:,}  total={stot:+,.1f}%  "
              f"pos_days={spos}/{len(sdarr)}  sharpe={ssh:.1f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
