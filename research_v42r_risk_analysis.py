#!/usr/bin/env python3
"""
v42r: Risk Analysis + Portfolio Optimization

EXP SS: Drawdown Analysis — worst-case scenarios
  - Max consecutive losses per symbol
  - Worst hour, worst day, worst week
  - Max drawdown duration
  - Tail risk: worst 5% of trades

EXP TT: Portfolio Allocation
  - Equal weight vs proportional to Sharpe
  - Kelly criterion sizing
  - Correlation between symbols (are losses correlated?)

EXP UU: Time-of-Day Optimization
  - Hourly breakdown of cascade MM performance
  - Are there hours we should skip?
  - Weekend vs weekday

Full 88 days, all 4 symbols, cascade MM with trail + liq accel.
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


def run_trail(cascades, bars, offset=0.15, tp=0.15, sl=0.50, max_hold=30, cooldown=60,
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
        trades.append({'net': gross-fee, 'exit': er, 'time': bars.index[fi],
                       'hour': bars.index[fi].hour,
                       'weekday': bars.index[fi].weekday(),
                       'date': bars.index[fi].date()})
        last_time = c['end']
    return trades


def main():
    out_file = 'results/v42r_risk_analysis.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
    all_dates = get_dates('2025-05-12', 88)

    print("="*80)
    print(f"  v42r: RISK ANALYSIS + PORTFOLIO OPTIMIZATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load data and run strategy per symbol
    all_trades = {}
    for sym in symbols:
        liq = load_liqs(sym, all_dates)
        bars = load_bars_chunked(sym, all_dates, chunk_days=10)
        gc.collect()

        # Also add ETH contagion for non-ETH symbols
        if sym != 'ETHUSDT':
            eth_liq = load_liqs('ETHUSDT', all_dates) if sym == 'SOLUSDT' else None
            if eth_liq is not None:
                eth_cascades = detect_cascades(eth_liq, pct_thresh=95)
                own_cascades = detect_cascades(liq, pct_thresh=95)
                cascades = sorted(own_cascades + eth_cascades, key=lambda c: c['end'])
                del eth_liq; gc.collect()
            else:
                cascades = detect_cascades(liq, pct_thresh=95)
        else:
            cascades = detect_cascades(liq, pct_thresh=95)

        trades = run_trail(cascades, bars)
        all_trades[sym] = trades
        n = len(trades)
        arr = np.array([t['net'] for t in trades])
        wr = (arr>0).mean()*100 if n > 0 else 0
        avg = arr.mean()*10000 if n > 0 else 0
        tot = arr.sum()*100 if n > 0 else 0
        print(f"  {sym}: {n} trades, WR={wr:.1f}%, avg={avg:+.1f}bps, tot={tot:+.1f}%")

        del liq, bars; gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # EXP SS: DRAWDOWN ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP SS: DRAWDOWN ANALYSIS")
    print(f"{'#'*80}")

    for sym in symbols:
        trades = all_trades[sym]
        if not trades: continue
        arr = np.array([t['net'] for t in trades])

        # Max consecutive losses
        losses = arr < 0
        max_consec = 0; current = 0
        for l in losses:
            if l: current += 1; max_consec = max(max_consec, current)
            else: current = 0

        # Equity curve and drawdown
        eq = np.cumprod(1 + arr)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = dd.min() * 100
        max_dd_idx = dd.argmin()

        # Drawdown duration (bars from peak to recovery)
        in_dd = False; dd_start = 0; max_dd_dur = 0; current_dur = 0
        for i in range(len(eq)):
            if eq[i] < peak[i]:
                if not in_dd: dd_start = i; in_dd = True
                current_dur = i - dd_start
            else:
                if in_dd:
                    max_dd_dur = max(max_dd_dur, current_dur)
                    in_dd = False
        if in_dd: max_dd_dur = max(max_dd_dur, current_dur)

        # Tail risk
        worst_5pct = np.percentile(arr, 5) * 10000
        worst_1pct = np.percentile(arr, 1) * 10000
        worst_trade = arr.min() * 10000

        # Win/loss stats
        wins = arr[arr > 0]; losses_arr = arr[arr < 0]
        avg_win = wins.mean() * 10000 if len(wins) > 0 else 0
        avg_loss = losses_arr.mean() * 10000 if len(losses_arr) > 0 else 0

        print(f"\n  === {sym} ({len(trades)} trades) ===")
        print(f"  Max consecutive losses: {max_consec}")
        print(f"  Max drawdown:           {max_dd:+.3f}%")
        print(f"  Max DD duration:        {max_dd_dur} trades")
        print(f"  Worst trade:            {worst_trade:+.1f} bps")
        print(f"  Worst 5% of trades:     <{worst_5pct:+.1f} bps")
        print(f"  Worst 1% of trades:     <{worst_1pct:+.1f} bps")
        print(f"  Avg win:                {avg_win:+.1f} bps ({len(wins)} trades)")
        print(f"  Avg loss:               {avg_loss:+.1f} bps ({len(losses_arr)} trades)")
        print(f"  Win/loss ratio:         {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # EXP TT: CORRELATION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP TT: CROSS-SYMBOL CORRELATION")
    print(f"{'#'*80}")

    # Build daily PnL per symbol
    daily_pnl = {}
    for sym in symbols:
        trades = all_trades[sym]
        if not trades: continue
        dpnl = {}
        for t in trades:
            d = t['date']
            if d not in dpnl: dpnl[d] = 0
            dpnl[d] += t['net']
        daily_pnl[sym] = dpnl

    # Align dates
    all_dates_set = set()
    for sym in daily_pnl:
        all_dates_set.update(daily_pnl[sym].keys())
    all_dates_sorted = sorted(all_dates_set)

    df_daily = pd.DataFrame(index=all_dates_sorted)
    for sym in daily_pnl:
        df_daily[sym] = [daily_pnl[sym].get(d, 0) for d in all_dates_sorted]

    print(f"\n  Daily PnL correlation matrix:")
    corr = df_daily.corr()
    for sym1 in symbols:
        if sym1 not in corr.columns: continue
        row = []
        for sym2 in symbols:
            if sym2 not in corr.columns: row.append("  N/A"); continue
            row.append(f"{corr.loc[sym1, sym2]:+.3f}")
        print(f"  {sym1:10s}  {'  '.join(row)}")

    # Portfolio equity curve
    print(f"\n  PORTFOLIO DAILY STATS:")
    portfolio_daily = df_daily.sum(axis=1)
    pos_days = (portfolio_daily > 0).sum()
    neg_days = (portfolio_daily < 0).sum()
    zero_days = (portfolio_daily == 0).sum()
    print(f"  Total days:     {len(portfolio_daily)}")
    print(f"  Positive days:  {pos_days} ({pos_days/len(portfolio_daily)*100:.0f}%)")
    print(f"  Negative days:  {neg_days}")
    print(f"  Zero days:      {zero_days}")
    print(f"  Avg daily PnL:  {portfolio_daily.mean()*100:+.3f}%")
    print(f"  Worst day:      {portfolio_daily.min()*100:+.3f}%")
    print(f"  Best day:       {portfolio_daily.max()*100:+.3f}%")
    print(f"  Daily Sharpe:   {portfolio_daily.mean()/(portfolio_daily.std()+1e-10)*np.sqrt(365):.1f}")

    # Kelly criterion
    print(f"\n  KELLY CRITERION (per symbol):")
    for sym in symbols:
        trades = all_trades[sym]
        if not trades: continue
        arr = np.array([t['net'] for t in trades])
        wr = (arr > 0).mean()
        avg_win = arr[arr > 0].mean() if (arr > 0).any() else 0
        avg_loss = abs(arr[arr < 0].mean()) if (arr < 0).any() else 1e-10
        kelly = wr - (1 - wr) / (avg_win / avg_loss) if avg_loss > 0 else 0
        print(f"  {sym:10s}  WR={wr:.3f}  W/L={avg_win/avg_loss:.2f}  Kelly={kelly:.3f}  "
              f"→ optimal leverage: {kelly*100:.0f}%")

    # ══════════════════════════════════════════════════════════════════════
    # EXP UU: TIME-OF-DAY ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP UU: TIME-OF-DAY OPTIMIZATION")
    print(f"{'#'*80}")

    # Hourly breakdown per symbol
    for sym in symbols:
        trades = all_trades[sym]
        if not trades: continue
        print(f"\n  === {sym} ===")
        print(f"  {'Hour':>4s}  {'N':>5s}  {'WR':>6s}  {'Avg':>8s}  {'Tot':>8s}")
        print(f"  {'-'*40}")
        hourly = {}
        for t in trades:
            hr = t['hour']
            if hr not in hourly: hourly[hr] = []
            hourly[hr].append(t['net'])
        for hr in range(24):
            if hr not in hourly: continue
            arr = np.array(hourly[hr])
            n = len(arr); wr = (arr>0).mean()*100
            avg = arr.mean()*10000; tot = arr.sum()*100
            flag = "✅" if avg > 0 else "❌"
            print(f"  {flag} {hr:02d}    {n:5d}  {wr:5.1f}%  {avg:+7.1f}bp  {tot:+7.2f}%")

    # Weekday breakdown
    print(f"\n  WEEKDAY BREAKDOWN (all symbols combined):")
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_trades = {i: [] for i in range(7)}
    for sym in symbols:
        for t in all_trades[sym]:
            weekday_trades[t['weekday']].append(t['net'])
    for wd in range(7):
        arr = np.array(weekday_trades[wd]) if weekday_trades[wd] else np.array([0])
        n = len(arr); wr = (arr>0).mean()*100
        avg = arr.mean()*10000; tot = arr.sum()*100
        flag = "✅" if avg > 0 else "❌"
        print(f"  {flag} {weekday_names[wd]:3s}  n={n:5d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  tot={tot:+7.2f}%")

    # Worst hours across all symbols
    print(f"\n  WORST HOURS (all symbols combined):")
    hourly_all = {}
    for sym in symbols:
        for t in all_trades[sym]:
            hr = t['hour']
            if hr not in hourly_all: hourly_all[hr] = []
            hourly_all[hr].append(t['net'])
    hour_stats = []
    for hr in range(24):
        if hr not in hourly_all: continue
        arr = np.array(hourly_all[hr])
        hour_stats.append((hr, arr.mean()*10000, len(arr), (arr>0).mean()*100))
    hour_stats.sort(key=lambda x: x[1])
    print(f"  Worst 5 hours:")
    for hr, avg, n, wr in hour_stats[:5]:
        flag = "✅" if avg > 0 else "❌"
        print(f"    {flag} hr={hr:02d}  avg={avg:+6.1f}bps  n={n}  wr={wr:.0f}%")
    print(f"  Best 5 hours:")
    for hr, avg, n, wr in hour_stats[-5:]:
        print(f"    ✅ hr={hr:02d}  avg={avg:+6.1f}bps  n={n}  wr={wr:.0f}%")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
