#!/usr/bin/env python3
"""
v42k: Full 88-Day Portfolio + New Independent Signal Ideas

EXP AA: Full 88-day portfolio with trailing stop + cross-symbol
  - Best config from v42j: trail act=3 dist=2, P95, window=60s, min_ev=2
  - All 3 symbols with ETH contagion triggers
  - Daily equity curve, weekly breakdown, monthly breakdown
  - Realistic concurrent simulation (max 1 pos per symbol)

EXP BB: OI Buildup Before Cascade
  - Load OI data, check if OI rises significantly in the 1-5 min before a cascade
  - If so, OI buildup could be an EARLY WARNING for cascades
  - Strategy: when OI spikes AND price starts moving → enter cascade MM early

EXP CC: Cascade Momentum — Sequential Cascade Direction
  - When cascade #1 is BUY-dominant, is cascade #2 more likely same direction?
  - If cascades cluster in direction, we can predict next cascade direction
  - Strategy: after first cascade, bias next entry toward same direction

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


def run_trail(cascades, bars, offset=0.15, tp=0.15, sl=0.50, max_hold=30, cooldown=300,
              trail_act=3, trail_dist=2):
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
        best_profit = 0; trailing_active = False; current_sl = sl_p
        for k in range(fi, min(fi+max_hold, len(bars))):
            b = bars.iloc[k]
            if is_long:
                cp = (b['high'] - lim) / lim
                if cp > best_profit: best_profit = cp
                if best_profit >= trail_act/10000 and not trailing_active:
                    trailing_active = True; current_sl = lim*(1+trail_dist/10000)
                if trailing_active:
                    ns = b['high']*(1-trail_dist/10000)
                    if ns > current_sl: current_sl = ns
                if b['low'] <= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; break
                if b['high'] >= tp_p: ep=tp_p; er='tp'; break
            else:
                cp = (lim - b['low']) / lim
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
        trades.append({'net': gross-fee, 'exit': er, 'time': bars.index[fi],
                       'symbol': '', 'direction': 'long' if is_long else 'short',
                       'cascade_end': c['end']})
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


# ============================================================================
# EXP CC: CASCADE MOMENTUM
# ============================================================================

def exp_cc_cascade_momentum(all_cascades):
    print(f"\n{'='*80}")
    print(f"  EXP CC: CASCADE MOMENTUM — SEQUENTIAL DIRECTION")
    print(f"{'='*80}")

    # Sort all cascades by time
    all_c = sorted(all_cascades, key=lambda c: c['end'])

    # Check: does cascade N direction predict cascade N+1 direction?
    same_dir = 0; diff_dir = 0
    for i in range(1, len(all_c)):
        gap = (all_c[i]['end'] - all_c[i-1]['end']).total_seconds()
        if gap > 3600: continue  # only look at cascades within 1 hour
        if all_c[i]['buy_dominant'] == all_c[i-1]['buy_dominant']:
            same_dir += 1
        else:
            diff_dir += 1

    total = same_dir + diff_dir
    if total > 0:
        same_pct = same_dir / total * 100
        print(f"  Cascades within 1hr of previous: {total}")
        print(f"  Same direction: {same_dir} ({same_pct:.1f}%)")
        print(f"  Different direction: {diff_dir} ({100-same_pct:.1f}%)")
        if same_pct > 55:
            print(f"  ✅ MOMENTUM: cascades tend to cluster in same direction")
        elif same_pct < 45:
            print(f"  ✅ REVERSAL: cascades tend to alternate direction")
        else:
            print(f"  ❌ No significant pattern")

    # By gap size
    print(f"\n  BY GAP SIZE:")
    for lo, hi, label in [(0, 60, '<1 min'), (60, 300, '1-5 min'),
                           (300, 900, '5-15 min'), (900, 3600, '15-60 min')]:
        same = diff = 0
        for i in range(1, len(all_c)):
            gap = (all_c[i]['end'] - all_c[i-1]['end']).total_seconds()
            if lo <= gap < hi:
                if all_c[i]['buy_dominant'] == all_c[i-1]['buy_dominant']:
                    same += 1
                else:
                    diff += 1
        tot = same + diff
        if tot >= 10:
            pct = same / tot * 100
            flag = "✅" if abs(pct - 50) > 5 else "  "
            print(f"  {flag} {label:12s}  n={tot:4d}  same_dir={pct:5.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    out_file = 'results/v42k_full_portfolio.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
    all_dates = get_dates('2025-05-12', 88)

    print("="*80)
    print(f"  v42k: FULL 88-DAY PORTFOLIO + NEW IDEAS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load data
    all_liq = {}
    for sym in symbols:
        all_liq[sym] = load_liqs(sym, all_dates)
    gc.collect()

    all_bars = {}
    for sym in symbols:
        all_bars[sym] = load_bars_chunked(sym, all_dates, chunk_days=10)
        gc.collect()

    print(f"\n  [{ram_str()}] all data loaded")

    # ══════════════════════════════════════════════════════════════════════
    # EXP AA: FULL 88-DAY PORTFOLIO WITH TRAILING STOP
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP AA: FULL 88-DAY PORTFOLIO (trail 3/2, cross-symbol)")
    print(f"{'#'*80}")

    all_trades = {}
    for target in symbols:
        if target == 'ETHUSDT':
            combined = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
        else:
            own = detect_cascades(all_liq[target], pct_thresh=95)
            eth = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
            combined = sorted(own + eth, key=lambda c: c['end'])

        trades = run_trail(combined, all_bars[target])
        for t in trades: t['symbol'] = target
        all_trades[target] = trades
        pstats(trades, f"{target} (trail 3/2)")

    # Merge all trades
    merged = []
    for sym in symbols:
        merged.extend(all_trades[sym])
    merged.sort(key=lambda t: t['time'])

    print(f"\n  PORTFOLIO TOTALS:")
    pstats(merged, "ALL SYMBOLS COMBINED")

    # Equity curve
    equity = [1.0]
    for t in merged:
        equity.append(equity[-1] * (1 + t['net']))
    final = equity[-1]
    total_ret = (final - 1) * 100
    max_eq = [equity[0]]
    for e in equity[1:]:
        max_eq.append(max(max_eq[-1], e))
    drawdowns = [(e/m - 1) for e, m in zip(equity, max_eq)]
    max_dd = min(drawdowns) * 100

    print(f"\n  EQUITY CURVE:")
    print(f"    Total return:    {total_ret:+.2f}%")
    print(f"    Max drawdown:    {max_dd:.2f}%")
    print(f"    Final equity:    {final:.4f}")

    # Daily stats
    daily = {}
    for t in merged:
        day = t['time'].date()
        if day not in daily: daily[day] = []
        daily[day].append(t['net'])

    daily_rets = [sum(v) for v in [daily[d] for d in sorted(daily.keys())]]
    darr = np.array(daily_rets)
    pos_days = (darr > 0).sum()
    print(f"\n  DAILY STATS:")
    print(f"    Trading days:    {len(darr)}")
    print(f"    Positive days:   {pos_days}/{len(darr)} ({pos_days/len(darr)*100:.0f}%)")
    print(f"    Avg daily ret:   {darr.mean()*100:+.3f}%")
    print(f"    Daily std:       {darr.std()*100:.3f}%")
    print(f"    Daily Sharpe:    {darr.mean()/(darr.std()+1e-10)*np.sqrt(365):.1f}")
    print(f"    Worst day:       {darr.min()*100:+.3f}%")
    print(f"    Best day:        {darr.max()*100:+.3f}%")

    # Weekly breakdown
    print(f"\n  WEEKLY BREAKDOWN:")
    print(f"  {'Week':15s}  {'Trades':>6s}  {'Return':>8s}  {'WR':>6s}")
    print(f"  {'-'*40}")
    week_trades = {}
    for t in merged:
        wk = t['time'].isocalendar()[:2]
        if wk not in week_trades: week_trades[wk] = []
        week_trades[wk].append(t)
    pos_weeks = 0
    for wk in sorted(week_trades.keys()):
        wt = week_trades[wk]
        arr = np.array([t['net'] for t in wt])
        n = len(arr); tot = arr.sum()*100; wr = (arr>0).mean()*100
        flag = "✅" if tot > 0 else "❌"
        print(f"  {flag} {wk[0]}-W{wk[1]:02d}       {n:>6d}  {tot:>+7.2f}%  {wr:>5.1f}%")
        if tot > 0: pos_weeks += 1
    print(f"\n  Positive weeks: {pos_weeks}/{len(week_trades)} ({pos_weeks/len(week_trades)*100:.0f}%)")

    # Monthly breakdown
    print(f"\n  MONTHLY BREAKDOWN:")
    month_trades = {}
    for t in merged:
        mo = t['time'].strftime('%Y-%m')
        if mo not in month_trades: month_trades[mo] = []
        month_trades[mo].append(t)
    for mo in sorted(month_trades.keys()):
        mt = month_trades[mo]
        arr = np.array([t['net'] for t in mt])
        n = len(arr); tot = arr.sum()*100; wr = (arr>0).mean()*100
        avg = arr.mean()*10000
        flag = "✅" if tot > 0 else "❌"
        print(f"  {flag} {mo}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  tot={tot:+7.2f}%")

    # Realistic concurrent sim
    print(f"\n  REALISTIC CONCURRENT SIMULATION:")
    events = []
    for sym in symbols:
        if sym == 'ETHUSDT':
            comb = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
        else:
            own = detect_cascades(all_liq[sym], pct_thresh=95)
            eth = detect_cascades(all_liq['ETHUSDT'], pct_thresh=95)
            comb = sorted(own + eth, key=lambda c: c['end'])
        for c in comb:
            events.append((c['end'], sym, c))
    events.sort(key=lambda e: e[0])

    position = {sym: None for sym in symbols}
    eq = 1.0; taken = 0; skipped = 0
    last_trade = {sym: None for sym in symbols}
    per_sym = {sym: 0 for sym in symbols}

    for ts, sym, c in events:
        if last_trade[sym] and (ts - last_trade[sym]).total_seconds() < 300: continue
        if position[sym] is not None and ts < position[sym]: skipped += 1; continue

        bars = all_bars[sym]
        idx = bars.index.searchsorted(ts)
        if idx >= len(bars) - 30 or idx < 1: continue
        price = bars.iloc[idx]['close']
        is_long = c['buy_dominant']
        offset = 0.15; tp = 0.15; sl = 0.50
        if is_long:
            lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
        else:
            lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)
        filled = False
        for j in range(idx, min(idx+30, len(bars))):
            b = bars.iloc[j]
            if is_long and b['low'] <= lim: filled=True; fi=j; break
            elif not is_long and b['high'] >= lim: filled=True; fi=j; break
        if not filled: continue

        ep = None; er = 'timeout'; exit_idx = fi+30
        best_profit = 0; trailing_active = False; current_sl = sl_p
        for k in range(fi, min(fi+30, len(bars))):
            b = bars.iloc[k]
            if is_long:
                cp = (b['high']-lim)/lim
                if cp > best_profit: best_profit = cp
                if best_profit >= 3/10000 and not trailing_active:
                    trailing_active = True; current_sl = lim*(1+2/10000)
                if trailing_active:
                    ns = b['high']*(1-2/10000)
                    if ns > current_sl: current_sl = ns
                if b['low'] <= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; exit_idx=k; break
                if b['high'] >= tp_p: ep=tp_p; er='tp'; exit_idx=k; break
            else:
                cp = (lim-b['low'])/lim
                if cp > best_profit: best_profit = cp
                if best_profit >= 3/10000 and not trailing_active:
                    trailing_active = True; current_sl = lim*(1-2/10000)
                if trailing_active:
                    ns = b['low']*(1+2/10000)
                    if ns < current_sl: current_sl = ns
                if b['high'] >= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; exit_idx=k; break
                if b['low'] <= tp_p: ep=tp_p; er='tp'; exit_idx=k; break
        if ep is None:
            exit_idx = min(fi+30, len(bars)-1)
            ep = bars.iloc[exit_idx]['close']
        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
        net = gross - fee
        eq *= (1 + net)
        position[sym] = bars.index[exit_idx]
        last_trade[sym] = ts
        taken += 1; per_sym[sym] += 1

    r_tot = (eq-1)*100
    print(f"    Trades taken:    {taken}")
    print(f"    Trades skipped:  {skipped}")
    print(f"    Total return:    {r_tot:+.2f}%")
    print(f"    Final equity:    {eq:.4f}")
    for sym in symbols:
        print(f"    {sym}: {per_sym[sym]} trades")

    # ══════════════════════════════════════════════════════════════════════
    # EXP CC: CASCADE MOMENTUM
    # ══════════════════════════════════════════════════════════════════════
    all_cascades = []
    for sym in symbols:
        all_cascades.extend(detect_cascades(all_liq[sym], pct_thresh=95))
    exp_cc_cascade_momentum(all_cascades)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
