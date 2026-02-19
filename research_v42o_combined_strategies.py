#!/usr/bin/env python3
"""
v42o: Combined Strategies + Cross-Symbol Liq Acceleration

EXP KK: Combine Cascade MM + Liq Acceleration
  - Are the signals overlapping or independent?
  - Measure overlap: what % of cascade MM trades also triggered by liq accel?
  - Combined portfolio: run both strategies simultaneously
  - Realistic sim: max 1 position per symbol, cascade MM has priority

EXP LL: Ticker Spread/Volatility as Cascade Quality Filter
  - Load ticker data, compute bid-ask spread around cascade time
  - Wider spread = more volatile = potentially better mean-reversion
  - Or: wider spread = more slippage = worse fills

EXP MM: Cross-Symbol Liq Acceleration
  - ETH liq volume spike → trade SOL/DOGE (like cross-symbol contagion)
  - Does liq acceleration propagate across symbols?

88 days, walk-forward, RAM-safe.
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
    return {'net': gross-fee, 'exit': er, 'time': bars.index[fi], 'entry_idx': fi}


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


def get_liq_accel_signals(liq_df, bars, window=15, thresh=5):
    """Get liq acceleration signal timestamps and directions."""
    liq_ts = liq_df.set_index('timestamp')
    buy_vol = liq_ts[liq_ts['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
    sell_vol = liq_ts[liq_ts['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
    total_vol = (buy_vol + sell_vol).reindex(bars.index, fill_value=0)
    buy_vol = buy_vol.reindex(bars.index, fill_value=0)
    sell_vol = sell_vol.reindex(bars.index, fill_value=0)

    roll_avg = total_vol.rolling(window, min_periods=1).mean()
    ratio = total_vol / (roll_avg + 1)
    signal_idx = ratio[ratio > thresh].index

    signals = []
    for ts in signal_idx:
        idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
        if idx < len(buy_vol):
            bv = buy_vol.iloc[idx]; sv = sell_vol.iloc[idx]
            signals.append({'time': ts, 'is_long': bv > sv, 'idx': idx, 'source': 'accel'})
    return signals


def main():
    out_file = 'results/v42o_combined_strategies.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42o: COMBINED STRATEGIES + CROSS-SYMBOL LIQ ACCEL")
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
    # EXP KK: SIGNAL OVERLAP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP KK: SIGNAL OVERLAP — CASCADE MM vs LIQ ACCELERATION")
    print(f"{'#'*80}")

    for sym in symbols:
        print(f"\n  === {sym} ===")
        cascades = detect_cascades(all_liq[sym], pct_thresh=95)
        cascade_times = set()
        for c in cascades:
            # Mark the minute of each cascade end
            cascade_times.add(c['end'].floor('min'))

        accel_signals = get_liq_accel_signals(all_liq[sym], all_bars[sym])
        accel_times = set(s['time'].floor('min') for s in accel_signals)

        overlap = cascade_times & accel_times
        cascade_only = cascade_times - accel_times
        accel_only = accel_times - cascade_times

        print(f"  Cascade signals: {len(cascade_times)}")
        print(f"  Accel signals:   {len(accel_times)}")
        print(f"  Overlap:         {len(overlap)} ({len(overlap)/max(len(cascade_times),1)*100:.0f}% of cascades)")
        print(f"  Cascade-only:    {len(cascade_only)}")
        print(f"  Accel-only:      {len(accel_only)}")

    # ══════════════════════════════════════════════════════════════════════
    # EXP KK-b: COMBINED PORTFOLIO SIMULATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP KK-b: COMBINED PORTFOLIO (cascade MM + liq accel)")
    print(f"{'#'*80}")

    # Run both strategies independently, then combine
    for sym in symbols:
        print(f"\n  === {sym} ===")
        bars = all_bars[sym]

        # Cascade MM trades (with trail)
        if sym == 'ETHUSDT':
            combined_c = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
        else:
            own = detect_cascades(all_liq[sym], pct_thresh=95)
            eth = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
            combined_c = sorted(own + eth, key=lambda c: c['end'])

        cascade_trades = []
        last_time = None
        for c in combined_c:
            if last_time and (c['end'] - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(c['end'])
            t = sim_trade(bars, idx, c['buy_dominant'])
            if t:
                t['source'] = 'cascade'
                cascade_trades.append(t)
                last_time = c['end']

        # Liq accel trades
        accel_signals = get_liq_accel_signals(all_liq[sym], bars)
        accel_trades = []
        last_time = None
        for s in accel_signals:
            if last_time and (s['time'] - last_time).total_seconds() < 60: continue
            t = sim_trade(bars, s['idx'], s['is_long'])
            if t:
                t['source'] = 'accel'
                accel_trades.append(t)
                last_time = s['time']

        # Measure overlap in actual trades
        cascade_minutes = set(t['time'].floor('min') for t in cascade_trades)
        accel_minutes = set(t['time'].floor('min') for t in accel_trades)
        trade_overlap = cascade_minutes & accel_minutes

        print(f"  Cascade trades: {len(cascade_trades)}, Accel trades: {len(accel_trades)}")
        print(f"  Trade overlap (same minute): {len(trade_overlap)}")

        # Combined: merge, sort by time, skip if already in position
        all_t = cascade_trades + accel_trades
        all_t.sort(key=lambda t: t['time'])

        # Deduplicate: if two trades in same minute, keep cascade (higher quality)
        seen_minutes = set()
        deduped = []
        for t in all_t:
            m = t['time'].floor('min')
            if m not in seen_minutes:
                deduped.append(t)
                seen_minutes.add(m)

        # Split train/test
        train = [t for t in deduped if t['time'] < split_ts]
        test = [t for t in deduped if t['time'] >= split_ts]

        print(f"\n  INDIVIDUAL:")
        c_train = [t for t in cascade_trades if t['time'] < split_ts]
        c_test = [t for t in cascade_trades if t['time'] >= split_ts]
        a_train = [t for t in accel_trades if t['time'] < split_ts]
        a_test = [t for t in accel_trades if t['time'] >= split_ts]

        pstats(c_train, f"Cascade TRAIN")
        pstats(c_test, f"Cascade TEST")
        pstats(a_train, f"Accel TRAIN")
        pstats(a_test, f"Accel TEST")

        print(f"\n  COMBINED (deduped):")
        pstats(train, f"Combined TRAIN")
        tr = pstats(test, f"Combined TEST")
        if tr:
            print(f"    test={tr['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP MM: CROSS-SYMBOL LIQ ACCELERATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP MM: CROSS-SYMBOL LIQ ACCELERATION")
    print(f"{'#'*80}")

    # ETH liq acceleration → trade SOL/DOGE
    eth_signals = get_liq_accel_signals(all_liq['ETHUSDT'], all_bars['ETHUSDT'])
    print(f"  ETH accel signals: {len(eth_signals)}")

    for target in symbols:
        print(f"\n  ETH accel → {target}:")
        bars = all_bars[target]
        train_trades = []; test_trades = []
        last_time = None
        for s in eth_signals:
            if last_time and (s['time'] - last_time).total_seconds() < 60: continue
            idx = bars.index.searchsorted(s['time'])
            t = sim_trade(bars, idx, s['is_long'])
            if t:
                if s['time'] < split_ts:
                    train_trades.append(t)
                else:
                    test_trades.append(t)
                last_time = s['time']

        ts_r = pstats(train_trades, "TRAIN")
        te_r = pstats(test_trades, "TEST")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL: MEGA PORTFOLIO — ALL STRATEGIES COMBINED
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  MEGA PORTFOLIO: ALL STRATEGIES, ALL SYMBOLS, REALISTIC SIM")
    print(f"{'#'*80}")

    # Collect all signals for all symbols
    all_events = []
    for sym in symbols:
        bars = all_bars[sym]

        # Cascade MM signals (with cross-symbol)
        if sym == 'ETHUSDT':
            cascades = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
        else:
            own = detect_cascades(all_liq[sym], pct_thresh=95)
            eth = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
            cascades = sorted(own + eth, key=lambda c: c['end'])

        for c in cascades:
            all_events.append({
                'time': c['end'], 'symbol': sym, 'is_long': c['buy_dominant'],
                'source': 'cascade', 'priority': 1  # higher priority
            })

        # Liq accel signals
        accel = get_liq_accel_signals(all_liq[sym], bars)
        for s in accel:
            all_events.append({
                'time': s['time'], 'symbol': sym, 'is_long': s['is_long'],
                'source': 'accel', 'priority': 2
            })

    # Sort by time, then priority (cascade first)
    all_events.sort(key=lambda e: (e['time'], e['priority']))

    # Realistic sim: max 1 position per symbol, 60s cooldown per symbol
    position_end = {sym: None for sym in symbols}
    last_trade_time = {sym: None for sym in symbols}
    eq = 1.0; taken = 0; skipped = 0
    per_sym = {sym: 0 for sym in symbols}
    per_source = {'cascade': 0, 'accel': 0}
    seen_minutes = {sym: set() for sym in symbols}
    daily_pnl = {}

    for ev in all_events:
        sym = ev['symbol']; ts = ev['time']
        bars = all_bars[sym]

        # Cooldown
        if last_trade_time[sym] and (ts - last_trade_time[sym]).total_seconds() < 60:
            continue
        # Already in position
        if position_end[sym] and ts < position_end[sym]:
            skipped += 1; continue
        # Dedup same minute
        m = ts.floor('min')
        if m in seen_minutes[sym]: continue
        seen_minutes[sym].add(m)

        idx = bars.index.searchsorted(ts)
        t = sim_trade(bars, idx, ev['is_long'])
        if t:
            eq *= (1 + t['net'])
            position_end[sym] = t['time'] + pd.Timedelta(minutes=1)
            last_trade_time[sym] = ts
            taken += 1
            per_sym[sym] += 1
            per_source[ev['source']] += 1

            day = t['time'].date()
            if day not in daily_pnl: daily_pnl[day] = 0
            daily_pnl[day] += t['net']

    tot = (eq-1)*100
    print(f"\n  MEGA PORTFOLIO RESULTS (full 88 days):")
    print(f"    Trades taken:    {taken}")
    print(f"    Trades skipped:  {skipped}")
    print(f"    Total return:    {tot:+.2f}%")
    print(f"    Final equity:    {eq:.4f}")
    for sym in symbols:
        print(f"    {sym}: {per_sym[sym]} trades")
    print(f"    From cascade:    {per_source['cascade']}")
    print(f"    From accel:      {per_source['accel']}")

    # Daily stats
    darr = np.array([daily_pnl[d] for d in sorted(daily_pnl.keys())])
    pos_days = (darr > 0).sum()
    print(f"\n    Trading days:    {len(darr)}")
    print(f"    Positive days:   {pos_days}/{len(darr)} ({pos_days/len(darr)*100:.0f}%)")
    print(f"    Avg daily ret:   {darr.mean()*100:+.3f}%")
    print(f"    Daily Sharpe:    {darr.mean()/(darr.std()+1e-10)*np.sqrt(365):.1f}")
    print(f"    Worst day:       {darr.min()*100:+.3f}%")
    print(f"    Best day:        {darr.max()*100:+.3f}%")

    # OOS only
    oos_days = {d: v for d, v in daily_pnl.items() if pd.Timestamp(str(d)) >= split_ts}
    if oos_days:
        oarr = np.array([oos_days[d] for d in sorted(oos_days.keys())])
        opos = (oarr > 0).sum()
        print(f"\n    OOS ONLY (Jul 11–Aug 7):")
        print(f"    OOS days:        {len(oarr)}")
        print(f"    OOS positive:    {opos}/{len(oarr)} ({opos/len(oarr)*100:.0f}%)")
        print(f"    OOS avg daily:   {oarr.mean()*100:+.3f}%")
        print(f"    OOS Sharpe:      {oarr.mean()/(oarr.std()+1e-10)*np.sqrt(365):.1f}")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
