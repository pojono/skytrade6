#!/usr/bin/env python3
"""
v42g: Portfolio Simulation + Cascade Asymmetry + New Ideas

EXP P: Cascade Direction Asymmetry
  - Do BUY cascades (long liquidations → price drops) revert better than
    SELL cascades (short liquidations → price rises)?
  - If asymmetric, we can size differently or skip one direction.

EXP Q: Multi-Symbol Portfolio Equity Curve
  - Run cascade MM on ETH+SOL+DOGE simultaneously with cross-symbol triggers
  - Measure combined equity curve, max drawdown, daily returns
  - Key question: do symbols diversify or correlate?

EXP R: Cascade Clustering — Back-to-Back Cascades
  - When cascades come in clusters (2+ within 30 min), does the 2nd/3rd
    cascade revert BETTER or WORSE than isolated cascades?

EXP S: Time-Since-Last-Cascade as Signal
  - Does a long gap since last cascade predict better reversion?
  - Hypothesis: market "forgets" → fresh cascade has more impact

EXP T: Cascade Size × Direction Interaction
  - Large BUY cascades vs large SELL cascades — different edge?

60 days, all 3 symbols, RAM-safe (chunked bars).
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
            bars = chunk.set_index('timestamp')['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last').dropna()
            all_bars.append(bars)
            del chunk; gc.collect()
        done = min(start+chunk_days, n)
        el = time.time()-t0; eta = el/done*(n-done) if done > 0 else 0
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
                                     'total_notional': bn+sn, 'buy_dominant': bn > sn,
                                     'n_events': len(cdf)})
                current = [row]
    if len(current) >= min_ev:
        cdf = pd.DataFrame(current)
        bn = cdf[cdf['side']=='Buy']['notional'].sum()
        sn = cdf[cdf['side']=='Sell']['notional'].sum()
        cascades.append({'start': cdf['timestamp'].min(), 'end': cdf['timestamp'].max(),
                         'total_notional': bn+sn, 'buy_dominant': bn > sn,
                         'n_events': len(cdf)})
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
        trades.append({
            'net': gross-fee, 'gross': gross, 'exit': er,
            'time': bars.index[fi], 'hour': bars.index[fi].hour,
            'direction': 'long' if is_long else 'short',
            'notional': c['total_notional'], 'n_events': c['n_events'],
            'cascade_end': c['end'],
        })
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


# ============================================================================
# EXP P: CASCADE DIRECTION ASYMMETRY
# ============================================================================

def exp_p_direction(all_trades, symbol):
    print(f"\n{'='*80}")
    print(f"  EXP P: CASCADE DIRECTION ASYMMETRY — {symbol}")
    print(f"{'='*80}")

    longs = [t for t in all_trades if t['direction'] == 'long']
    shorts = [t for t in all_trades if t['direction'] == 'short']

    pstats(all_trades, "ALL")
    pstats(longs, "LONG only (fade buy-cascade)")
    pstats(shorts, "SHORT only (fade sell-cascade)")

    if longs and shorts:
        l_avg = np.mean([t['net'] for t in longs]) * 10000
        s_avg = np.mean([t['net'] for t in shorts]) * 10000
        diff = l_avg - s_avg
        if abs(diff) > 1:
            better = "LONG" if diff > 0 else "SHORT"
            print(f"\n  ⚡ {better} is better by {abs(diff):.1f} bps")
        else:
            print(f"\n  ≈ No significant asymmetry ({diff:+.1f} bps)")


# ============================================================================
# EXP R: CASCADE CLUSTERING
# ============================================================================

def exp_r_clustering(all_trades):
    print(f"\n{'='*80}")
    print(f"  EXP R: CASCADE CLUSTERING (back-to-back cascades)")
    print(f"{'='*80}")

    # Tag each trade with time since previous trade
    for i, t in enumerate(all_trades):
        if i == 0:
            t['gap_min'] = 999
        else:
            t['gap_min'] = (t['cascade_end'] - all_trades[i-1]['cascade_end']).total_seconds() / 60

    # Isolated vs clustered
    isolated = [t for t in all_trades if t['gap_min'] > 30]
    clustered = [t for t in all_trades if 5 <= t['gap_min'] <= 30]
    rapid = [t for t in all_trades if t['gap_min'] < 5]

    print(f"\n  BY GAP SINCE PREVIOUS CASCADE:")
    pstats(isolated, "Isolated (>30 min gap)")
    pstats(clustered, "Clustered (5-30 min gap)")
    pstats(rapid, "Rapid (<5 min gap)")


# ============================================================================
# EXP S: TIME-SINCE-LAST-CASCADE
# ============================================================================

def exp_s_gap(all_trades):
    print(f"\n{'='*80}")
    print(f"  EXP S: TIME-SINCE-LAST-CASCADE AS SIGNAL")
    print(f"{'='*80}")

    gaps = [t['gap_min'] for t in all_trades if t.get('gap_min', 999) < 999]
    if not gaps:
        print("  No gap data"); return

    p25 = np.percentile(gaps, 25)
    p50 = np.percentile(gaps, 50)
    p75 = np.percentile(gaps, 75)
    print(f"  Gap distribution: P25={p25:.0f}m  P50={p50:.0f}m  P75={p75:.0f}m")

    for lo, hi, label in [(0, 10, '<10 min'), (10, 30, '10-30 min'),
                           (30, 60, '30-60 min'), (60, 120, '1-2 hours'),
                           (120, 999, '>2 hours')]:
        sub = [t for t in all_trades if lo <= t.get('gap_min', 999) < hi]
        pstats(sub, f"Gap {label}")


# ============================================================================
# EXP T: CASCADE SIZE × DIRECTION
# ============================================================================

def exp_t_size_direction(all_trades):
    print(f"\n{'='*80}")
    print(f"  EXP T: CASCADE SIZE × DIRECTION INTERACTION")
    print(f"{'='*80}")

    notionals = [t['notional'] for t in all_trades]
    p50 = np.percentile(notionals, 50)
    p75 = np.percentile(notionals, 75)

    for size_label, size_filt in [("Small (<P50)", lambda t: t['notional'] < p50),
                                   ("Large (>P50)", lambda t: t['notional'] >= p50),
                                   ("XL (>P75)", lambda t: t['notional'] >= p75)]:
        for dir_label, dir_filt in [("LONG", lambda t: t['direction']=='long'),
                                     ("SHORT", lambda t: t['direction']=='short')]:
            sub = [t for t in all_trades if size_filt(t) and dir_filt(t)]
            pstats(sub, f"{size_label} + {dir_label}")


# ============================================================================
# EXP Q: MULTI-SYMBOL PORTFOLIO
# ============================================================================

def exp_q_portfolio(all_symbol_trades):
    print(f"\n{'='*80}")
    print(f"  EXP Q: MULTI-SYMBOL PORTFOLIO SIMULATION")
    print(f"{'='*80}")

    # Merge all trades, sort by time
    all_trades = []
    for sym, trades in all_symbol_trades.items():
        for t in trades:
            t['symbol'] = sym
            all_trades.append(t)
    all_trades.sort(key=lambda t: t['time'])

    print(f"  Total trades across all symbols: {len(all_trades)}")
    for sym in all_symbol_trades:
        print(f"    {sym}: {len(all_symbol_trades[sym])} trades")

    # Equity curve (equal weight per trade)
    equity = [1.0]
    daily_returns = {}
    for t in all_trades:
        equity.append(equity[-1] * (1 + t['net']))
        day = t['time'].date()
        if day not in daily_returns:
            daily_returns[day] = []
        daily_returns[day].append(t['net'])

    final = equity[-1]
    total_ret = (final - 1) * 100
    max_eq = max(equity)
    max_dd = min((e / max(equity[:i+1]) - 1) for i, e in enumerate(equity)) * 100

    print(f"\n  PORTFOLIO EQUITY CURVE:")
    print(f"    Total return:  {total_ret:+.2f}%")
    print(f"    Max drawdown:  {max_dd:.2f}%")
    print(f"    Final equity:  {final:.4f}")
    print(f"    Total trades:  {len(all_trades)}")

    # Daily stats
    daily_rets = []
    for day in sorted(daily_returns.keys()):
        dr = sum(daily_returns[day])
        daily_rets.append(dr)

    daily_arr = np.array(daily_rets)
    pos_days = (daily_arr > 0).sum()
    total_days = len(daily_arr)
    daily_avg = daily_arr.mean() * 100
    daily_std = daily_arr.std() * 100
    daily_sharpe = daily_arr.mean() / (daily_arr.std() + 1e-10) * np.sqrt(365)

    print(f"\n  DAILY STATS:")
    print(f"    Trading days:    {total_days}")
    print(f"    Positive days:   {pos_days}/{total_days} ({pos_days/total_days*100:.0f}%)")
    print(f"    Avg daily ret:   {daily_avg:+.3f}%")
    print(f"    Daily std:       {daily_std:.3f}%")
    print(f"    Daily Sharpe:    {daily_sharpe:.1f}")
    print(f"    Worst day:       {daily_arr.min()*100:+.3f}%")
    print(f"    Best day:        {daily_arr.max()*100:+.3f}%")

    # Correlation between symbols
    print(f"\n  SYMBOL CORRELATION (daily returns):")
    sym_daily = {}
    for sym, trades in all_symbol_trades.items():
        sd = {}
        for t in trades:
            day = t['time'].date()
            sd[day] = sd.get(day, 0) + t['net']
        sym_daily[sym] = sd

    syms = list(all_symbol_trades.keys())
    for i in range(len(syms)):
        for j in range(i+1, len(syms)):
            common_days = set(sym_daily[syms[i]].keys()) & set(sym_daily[syms[j]].keys())
            if len(common_days) < 10: continue
            a = np.array([sym_daily[syms[i]][d] for d in sorted(common_days)])
            b = np.array([sym_daily[syms[j]][d] for d in sorted(common_days)])
            corr = np.corrcoef(a, b)[0, 1]
            print(f"    {syms[i][:3]} vs {syms[j][:3]}: ρ={corr:.3f} ({len(common_days)} days)")

    # Weekly breakdown
    print(f"\n  WEEKLY BREAKDOWN:")
    print(f"  {'Week':15s}  {'Trades':>6s}  {'Return':>8s}  {'WR':>6s}")
    print(f"  {'-'*40}")

    week_trades = {}
    for t in all_trades:
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

    total_weeks = len(week_trades)
    print(f"\n  Positive weeks: {pos_weeks}/{total_weeks} ({pos_weeks/total_weeks*100:.0f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    out_file = 'results/v42g_portfolio_asymmetry.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    n_days = 60
    dates = get_dates('2025-05-12', n_days)
    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']

    print("="*80)
    print(f"  v42g: PORTFOLIO + ASYMMETRY — {n_days} DAYS (RAM-SAFE)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load liqs → cascades (small)
    cascades = {}
    for sym in symbols:
        liq = load_liqs(sym, dates)
        cascades[sym] = detect_cascades(liq, pct_thresh=95)
        print(f"  {sym}: {len(cascades[sym])} cascades")
        del liq
    gc.collect()

    # Load bars (chunked, RAM-safe)
    bars = {}
    for sym in symbols:
        bars[sym] = load_bars_chunked(sym, dates, chunk_days=10)
        gc.collect()

    print(f"\n  [{ram_str()}] all data loaded")

    # Build combined triggers (ETH cascades trigger all symbols)
    all_symbol_trades = {}
    for target in symbols:
        # Combined: own cascades + ETH cascades
        if target == 'ETHUSDT':
            combined = cascades['ETHUSDT']
        else:
            combined = sorted(
                cascades[target] + cascades['ETHUSDT'],
                key=lambda c: c['end']
            )
        trades = run_strat(combined, bars[target])
        all_symbol_trades[target] = trades
        print(f"  {target}: {len(trades)} trades from combined triggers")

    # EXP P: Direction asymmetry (per symbol)
    for sym in symbols:
        exp_p_direction(all_symbol_trades[sym], sym)

    # EXP R: Clustering (all trades merged)
    all_merged = []
    for sym in symbols:
        all_merged.extend(all_symbol_trades[sym])
    all_merged.sort(key=lambda t: t['cascade_end'])
    exp_r_clustering(all_merged)

    # EXP S: Gap signal
    exp_s_gap(all_merged)

    # EXP T: Size × Direction
    print(f"\n  (Using all symbols merged)")
    exp_t_size_direction(all_merged)

    # EXP Q: Portfolio
    exp_q_portfolio(all_symbol_trades)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
