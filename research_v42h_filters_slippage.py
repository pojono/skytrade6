#!/usr/bin/env python3
"""
v42h: Combined Filters + Slippage Sensitivity + Realistic Simulation

EXP U: Apply ALL discovered filters simultaneously
  - Direction: prefer LONG (fade buy-cascades) — +2-3 bps edge
  - Gap: 10-30 min gap cascades are best (Sharpe 337)
  - Hours: exclude 08,09,13,16 UTC
  - Cross-symbol: ETH cascades trigger SOL+DOGE
  - Compare: unfiltered vs each filter vs all combined

EXP V: Realistic Concurrent Trade Simulation
  - Max 1 position per symbol at a time
  - Track actual equity curve with position sizing
  - Account for overlapping signals (skip if already in position)

EXP W: Slippage Sensitivity
  - What if limit fill is 0.5, 1, 2, 3 bps worse than theoretical?
  - At what slippage does the strategy break even?
  - Also test: what if SL is hit 1-2 bps worse (slippage on stop)?

60 days, all 3 symbols, RAM-safe.
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


def run_strat_with_slippage(cascades, bars, offset=0.15, tp=0.15, sl=0.50,
                             max_hold=30, cooldown=300,
                             entry_slip_bps=0, exit_slip_bps=0,
                             direction_filter=None, hour_filter=None):
    """
    direction_filter: 'long_only', 'short_only', or None
    hour_filter: set of bad hours to exclude, or None
    entry_slip_bps: additional slippage on entry fill (bps, always adverse)
    exit_slip_bps: additional slippage on SL/timeout exit (bps, always adverse)
    """
    trades = []
    last_time = None
    for c in cascades:
        if last_time and (c['end'] - last_time).total_seconds() < cooldown: continue

        # Hour filter
        if hour_filter and c['end'].hour in hour_filter: continue

        # Direction filter
        is_long = c['buy_dominant']
        if direction_filter == 'long_only' and not is_long: continue
        if direction_filter == 'short_only' and is_long: continue

        idx = bars.index.searchsorted(c['end'])
        if idx >= len(bars) - max_hold or idx < 1: continue
        price = bars.iloc[idx]['close']

        # Apply entry slippage (adverse = worse fill)
        slip = entry_slip_bps / 10000
        if is_long:
            lim = price*(1-offset/100) * (1 + slip)  # fill higher = worse for long
            tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
        else:
            lim = price*(1+offset/100) * (1 - slip)  # fill lower = worse for short
            tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)

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

        # Apply exit slippage on SL/timeout (adverse)
        eslip = exit_slip_bps / 10000
        if er != 'tp':
            if is_long: ep = ep * (1 - eslip)
            else: ep = ep * (1 + eslip)

        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
        trades.append({
            'net': gross-fee, 'exit': er,
            'time': bars.index[fi],
            'direction': 'long' if is_long else 'short',
            'cascade_end': c['end'],
        })
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
# EXP U: COMBINED FILTERS
# ============================================================================

def exp_u_combined_filters(cascades_combined, bars, symbol):
    print(f"\n{'='*80}")
    print(f"  EXP U: COMBINED FILTERS — {symbol}")
    print(f"{'='*80}")

    bad_hours = {8, 9, 13, 16}

    # Baseline
    t0 = run_strat_with_slippage(cascades_combined, bars)
    pstats(t0, "BASELINE (no filters)")

    # Filter 1: direction only
    t1 = run_strat_with_slippage(cascades_combined, bars, direction_filter='long_only')
    pstats(t1, "LONG only")

    # Filter 2: hours only
    t2 = run_strat_with_slippage(cascades_combined, bars, hour_filter=bad_hours)
    pstats(t2, "Exclude bad hours")

    # Filter 3: direction + hours
    t3 = run_strat_with_slippage(cascades_combined, bars,
                                  direction_filter='long_only', hour_filter=bad_hours)
    pstats(t3, "LONG + exclude bad hours")

    # Walk-forward for best filter
    print(f"\n  WALK-FORWARD (train=42d, test=18d):")
    split = bars.index.min() + pd.Timedelta(days=42)
    for label, df, hf in [
        ("Baseline", None, None),
        ("LONG only", 'long_only', None),
        ("LONG + no bad hrs", 'long_only', bad_hours),
    ]:
        train_c = [c for c in cascades_combined if c['end'] < split]
        test_c = [c for c in cascades_combined if c['end'] >= split]
        train_t = run_strat_with_slippage(train_c, bars, direction_filter=df, hour_filter=hf)
        test_t = run_strat_with_slippage(test_c, bars, direction_filter=df, hour_filter=hf)
        print(f"\n  {label}:")
        ts = pstats(train_t, "TRAIN")
        te = pstats(test_t, "TEST")
        if ts and te:
            oos = "✅" if te['tot'] > 0 else "❌"
            print(f"    {oos} train={ts['tot']/42:+.3f}%/d  test={te['tot']/18:+.3f}%/d")


# ============================================================================
# EXP W: SLIPPAGE SENSITIVITY
# ============================================================================

def exp_w_slippage(cascades_combined, bars, symbol):
    print(f"\n{'='*80}")
    print(f"  EXP W: SLIPPAGE SENSITIVITY — {symbol}")
    print(f"{'='*80}")

    print(f"\n  ENTRY SLIPPAGE (exit slippage = 0):")
    for slip in [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        t = run_strat_with_slippage(cascades_combined, bars, entry_slip_bps=slip)
        pstats(t, f"entry_slip={slip:.1f}bps")

    print(f"\n  EXIT SLIPPAGE on SL/timeout (entry slippage = 0):")
    for slip in [0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        t = run_strat_with_slippage(cascades_combined, bars, exit_slip_bps=slip)
        pstats(t, f"exit_slip={slip:.1f}bps")

    print(f"\n  BOTH ENTRY + EXIT SLIPPAGE:")
    for slip in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        t = run_strat_with_slippage(cascades_combined, bars,
                                     entry_slip_bps=slip, exit_slip_bps=slip)
        pstats(t, f"both_slip={slip:.1f}bps")

    # Find breakeven slippage
    print(f"\n  BREAKEVEN SLIPPAGE SEARCH:")
    for slip in np.arange(0, 10.1, 0.5):
        t = run_strat_with_slippage(cascades_combined, bars,
                                     entry_slip_bps=slip, exit_slip_bps=slip)
        if t:
            avg = np.mean([x['net'] for x in t]) * 10000
            if avg <= 0:
                print(f"  ⚠️ Breakeven at ~{slip:.1f} bps both-way slippage")
                break
    else:
        print(f"  ✅ Still profitable at 10 bps both-way slippage!")


# ============================================================================
# EXP V: REALISTIC CONCURRENT SIMULATION
# ============================================================================

def exp_v_realistic(all_cascades, all_bars, symbols):
    print(f"\n{'='*80}")
    print(f"  EXP V: REALISTIC CONCURRENT SIMULATION")
    print(f"{'='*80}")

    # Simulate: max 1 position per symbol, track equity
    # Process all cascades in time order across all symbols

    events = []
    for sym in symbols:
        for c in all_cascades[sym]:
            events.append(('cascade', c['end'], sym, c))
    events.sort(key=lambda e: e[1])

    position = {sym: None for sym in symbols}  # None or {'entry_time', 'exit_time', 'net'}
    equity = 1.0
    equity_curve = [(events[0][1] if events else pd.Timestamp.now(), 1.0)]
    trades_taken = 0
    trades_skipped = 0
    per_symbol_trades = {sym: 0 for sym in symbols}

    offset = 0.15; tp = 0.15; sl = 0.50; max_hold = 30; cooldown_sec = 300
    last_trade_time = {sym: None for sym in symbols}

    for _, ts, sym, c in events:
        bars = all_bars[sym]

        # Cooldown check
        if last_trade_time[sym] and (ts - last_trade_time[sym]).total_seconds() < cooldown_sec:
            continue

        # Position check: skip if already in position
        if position[sym] is not None:
            if ts < position[sym]['exit_time']:
                trades_skipped += 1
                continue
            else:
                position[sym] = None

        idx = bars.index.searchsorted(ts)
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

        ep = None; er = 'timeout'; exit_idx = fi + max_hold
        for k in range(fi, min(fi+max_hold, len(bars))):
            b = bars.iloc[k]
            if is_long:
                if b['low'] <= sl_p: ep=sl_p; er='sl'; exit_idx=k; break
                if b['high'] >= tp_p: ep=tp_p; er='tp'; exit_idx=k; break
            else:
                if b['high'] >= sl_p: ep=sl_p; er='sl'; exit_idx=k; break
                if b['low'] <= tp_p: ep=tp_p; er='tp'; exit_idx=k; break
        if ep is None:
            exit_idx = min(fi+max_hold, len(bars)-1)
            ep = bars.iloc[exit_idx]['close']

        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
        net = gross - fee

        # Update equity (assume 1x position size per trade)
        equity *= (1 + net)
        equity_curve.append((bars.index[exit_idx], equity))

        position[sym] = {'exit_time': bars.index[exit_idx]}
        last_trade_time[sym] = ts
        trades_taken += 1
        per_symbol_trades[sym] += 1

    # Results
    total_ret = (equity - 1) * 100
    max_eq = max(e[1] for e in equity_curve)
    max_dd = min((e[1] / max(ec[1] for ec in equity_curve[:i+1]) - 1)
                 for i, e in enumerate(equity_curve)) * 100

    print(f"\n  REALISTIC SIMULATION RESULTS:")
    print(f"    Trades taken:    {trades_taken}")
    print(f"    Trades skipped:  {trades_skipped} (already in position)")
    print(f"    Total return:    {total_ret:+.2f}%")
    print(f"    Max drawdown:    {max_dd:.2f}%")
    print(f"    Final equity:    {equity:.4f}")

    for sym in symbols:
        print(f"    {sym}: {per_symbol_trades[sym]} trades")

    # Daily breakdown
    daily = {}
    prev_eq = 1.0
    for ts, eq in equity_curve:
        day = ts.date()
        daily[day] = eq
    daily_rets = []
    prev = 1.0
    for day in sorted(daily.keys()):
        dr = daily[day] / prev - 1
        daily_rets.append(dr)
        prev = daily[day]

    darr = np.array(daily_rets)
    pos_days = (darr > 0).sum()
    print(f"\n  DAILY STATS (realistic):")
    print(f"    Trading days:    {len(darr)}")
    print(f"    Positive days:   {pos_days}/{len(darr)} ({pos_days/len(darr)*100:.0f}%)")
    print(f"    Avg daily ret:   {darr.mean()*100:+.3f}%")
    print(f"    Daily Sharpe:    {darr.mean()/(darr.std()+1e-10)*np.sqrt(365):.1f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    out_file = 'results/v42h_filters_slippage.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    n_days = 60
    dates = get_dates('2025-05-12', n_days)
    symbols = ['ETHUSDT', 'SOLUSDT', 'DOGEUSDT']

    print("="*80)
    print(f"  v42h: FILTERS + SLIPPAGE + REALISTIC SIM — {n_days} DAYS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load data (RAM-safe)
    cascades = {}
    for sym in symbols:
        liq = load_liqs(sym, dates)
        cascades[sym] = detect_cascades(liq, pct_thresh=95)
        print(f"  {sym}: {len(cascades[sym])} cascades")
        del liq
    gc.collect()

    bars = {}
    for sym in symbols:
        bars[sym] = load_bars_chunked(sym, dates, chunk_days=10)
        gc.collect()

    print(f"\n  [{ram_str()}] all data loaded")

    # Build combined triggers
    combined = {}
    for sym in symbols:
        if sym == 'ETHUSDT':
            combined[sym] = cascades['ETHUSDT']
        else:
            combined[sym] = sorted(
                cascades[sym] + cascades['ETHUSDT'],
                key=lambda c: c['end']
            )

    # EXP U: Combined filters (per symbol)
    for sym in symbols:
        exp_u_combined_filters(combined[sym], bars[sym], sym)

    # EXP W: Slippage (per symbol)
    for sym in symbols:
        exp_w_slippage(combined[sym], bars[sym], sym)

    # EXP V: Realistic concurrent simulation
    exp_v_realistic(combined, bars, symbols)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
