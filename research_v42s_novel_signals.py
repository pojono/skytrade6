#!/usr/bin/env python3
"""
v42s: Novel Signal Ideas — Completely Independent from Liquidations

EXP VV: Microstructure Mean-Reversion
  - Compute 1-min returns, look for extreme moves (>3 sigma)
  - Fade extreme 1-min moves — classic mean-reversion
  - No liquidation data needed — pure price action

EXP WW: Volatility Breakout
  - When 5-min volatility drops below P10, enter straddle (long vol)
  - When vol expands, close for profit
  - Opposite of cascade MM — profits from vol expansion

EXP XX: Intraday Range Breakout
  - Track rolling high/low over N minutes
  - When price breaks above/below range, enter momentum trade
  - Classic breakout strategy on tick data

EXP YY: Liquidation Clustering Momentum
  - When liquidation count in last 5 min is >P90, enter momentum
  - (opposite of cascade MM which fades — this follows the cascade)
  - Does momentum or mean-reversion work better?

88 days, SOL for initial testing, RAM-safe.
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


def sim_trade_simple(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30):
    """Simple trade sim without trailing stop."""
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
    return {'net': gross-fee, 'exit': er, 'time': bars.index[fi]}


def sim_trade_trail(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30,
                    trail_act=3, trail_dist=2):
    """Trade sim with trailing stop."""
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
    out_file = 'results/v42s_novel_signals.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbol = 'SOLUSDT'
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42s: NOVEL SIGNAL IDEAS — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    liq = load_liqs(symbol, all_dates)
    bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
    gc.collect()
    print(f"\n  [{ram_str()}] data loaded")

    # ══════════════════════════════════════════════════════════════════════
    # EXP VV: MICROSTRUCTURE MEAN-REVERSION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP VV: MICROSTRUCTURE MEAN-REVERSION")
    print(f"{'#'*80}")

    ret_1m = bars['close'].pct_change()
    roll_std = ret_1m.rolling(60, min_periods=30).std()

    for sigma_thresh in [2, 3, 4, 5]:
        # Extreme down moves → go long (mean reversion)
        extreme_down = ret_1m < -sigma_thresh * roll_std
        extreme_up = ret_1m > sigma_thresh * roll_std

        down_signals = bars.index[extreme_down]
        up_signals = bars.index[extreme_up]

        print(f"\n  SIGMA={sigma_thresh}:")
        print(f"    Extreme down: {len(down_signals)}, Extreme up: {len(up_signals)}")

        # Fade extreme moves with trail
        train_trades = []; test_trades = []
        lt = None
        for ts in down_signals:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, True)  # long after extreme down
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        for ts in up_signals:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, False)  # short after extreme up
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        train_trades.sort(key=lambda t: t['time'])
        test_trades.sort(key=lambda t: t['time'])
        ts_r = pstats(train_trades, "TRAIN (fade)")
        te_r = pstats(test_trades, "TEST (fade)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP WW: VOLATILITY BREAKOUT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP WW: VOLATILITY BREAKOUT (long vol after compression)")
    print(f"{'#'*80}")

    # 5-min bars
    bars5 = bars.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    vol5 = bars5['close'].pct_change().abs()
    vol5_roll = vol5.rolling(12, min_periods=6).mean()  # 60-min rolling avg vol

    for pct_thresh in [10, 20, 30]:
        vol_thresh = vol5_roll.quantile(pct_thresh / 100)
        low_vol = vol5_roll < vol_thresh
        signals = bars5.index[low_vol]

        print(f"\n  VOL < P{pct_thresh} ({len(signals)} signals):")

        # After vol compression, enter straddle-like: both long and short
        # Actually: enter in direction of first move after compression
        train_trades = []; test_trades = []
        lt = None
        for ts in signals:
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.searchsorted(ts)
            if idx >= len(bars) - 30 or idx < 5: continue
            # Direction: use last 5-min return
            ret = (bars.iloc[idx]['close'] - bars.iloc[idx-5]['close']) / bars.iloc[idx-5]['close']
            is_long = ret < 0  # fade the recent move (mean reversion after compression)
            t = sim_trade_trail(bars, idx, is_long)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        ts_r = pstats(train_trades, "TRAIN (fade after compression)")
        te_r = pstats(test_trades, "TEST (fade after compression)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP XX: RANGE BREAKOUT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP XX: INTRADAY RANGE BREAKOUT")
    print(f"{'#'*80}")

    for lookback in [30, 60, 120]:
        roll_high = bars['high'].rolling(lookback, min_periods=lookback).max()
        roll_low = bars['low'].rolling(lookback, min_periods=lookback).min()

        # Breakout above range → momentum long
        breakout_up = bars['close'] > roll_high.shift(1)
        breakout_down = bars['close'] < roll_low.shift(1)

        print(f"\n  LOOKBACK={lookback}min:")
        print(f"    Breakout up: {breakout_up.sum()}, Breakout down: {breakout_down.sum()}")

        # Momentum: follow the breakout
        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[breakout_up]:
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, True, offset=0.05, tp=0.15, sl=0.30)  # tighter entry
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        for ts in bars.index[breakout_down]:
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, False, offset=0.05, tp=0.15, sl=0.30)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        train_trades.sort(key=lambda t: t['time'])
        test_trades.sort(key=lambda t: t['time'])
        ts_r = pstats(train_trades, "TRAIN (momentum)")
        te_r = pstats(test_trades, "TEST (momentum)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # Mean-reversion: fade the breakout
        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[breakout_up]:
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, False)  # short after breakout up
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts
        for ts in bars.index[breakout_down]:
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, True)  # long after breakout down
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        train_trades.sort(key=lambda t: t['time'])
        test_trades.sort(key=lambda t: t['time'])
        ts_r = pstats(train_trades, "TRAIN (fade)")
        te_r = pstats(test_trades, "TEST (fade)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP YY: LIQ CLUSTERING MOMENTUM vs MEAN-REVERSION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP YY: LIQ CLUSTERING — MOMENTUM vs MEAN-REVERSION")
    print(f"{'#'*80}")

    liq_ts = liq.set_index('timestamp')
    buy_vol = liq_ts[liq_ts['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
    sell_vol = liq_ts[liq_ts['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
    buy_vol = buy_vol.reindex(bars.index, fill_value=0)
    sell_vol = sell_vol.reindex(bars.index, fill_value=0)
    total_vol = buy_vol + sell_vol

    # Rolling 5-min liq volume
    liq_5m = total_vol.rolling(5, min_periods=1).sum()
    liq_p90 = liq_5m.quantile(0.90)
    liq_p95 = liq_5m.quantile(0.95)

    for thresh, label in [(liq_p90, 'P90'), (liq_p95, 'P95')]:
        high_liq = liq_5m > thresh
        signals = bars.index[high_liq]
        print(f"\n  LIQ 5min > {label} ({len(signals)} signals):")

        # Mean-reversion (fade — our standard approach)
        train_mr = []; test_mr = []
        lt = None
        for ts in signals:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
            if idx >= len(bars) - 30 or idx < 1: continue
            bv = buy_vol.iloc[idx]; sv = sell_vol.iloc[idx]
            is_long = bv > sv  # fade dominant side
            t = sim_trade_trail(bars, idx, is_long)
            if t:
                if ts < split_ts: train_mr.append(t)
                else: test_mr.append(t)
                lt = ts

        ts_r = pstats(train_mr, "TRAIN (mean-reversion/fade)")
        te_r = pstats(test_mr, "TEST (mean-reversion/fade)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

        # Momentum (follow the cascade)
        train_mom = []; test_mom = []
        lt = None
        for ts in signals:
            if lt and (ts - lt).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
            if idx >= len(bars) - 30 or idx < 1: continue
            bv = buy_vol.iloc[idx]; sv = sell_vol.iloc[idx]
            is_long = sv > bv  # follow dominant side (sell liqs = price dropping = go short)
            t = sim_trade_trail(bars, idx, is_long)
            if t:
                if ts < split_ts: train_mom.append(t)
                else: test_mom.append(t)
                lt = ts

        ts_r = pstats(train_mom, "TRAIN (momentum/follow)")
        te_r = pstats(test_mom, "TEST (momentum/follow)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
