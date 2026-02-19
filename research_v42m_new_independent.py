#!/usr/bin/env python3
"""
v42m: Genuinely New Independent Signal Ideas

These are strategies that do NOT depend on liquidation cascades.

EXP GG: Liquidation Volume Acceleration
  - Track rolling liquidation volume (1min, 5min windows)
  - When liq volume accelerates (current >> rolling avg), enter fade trade
  - This catches cascades EARLIER than waiting for P95 threshold
  - Could also work on smaller cascades that don't meet P95

EXP HH: Price-Volume Divergence
  - When price makes new high/low but volume is declining → reversal signal
  - Use 5min bars, compare price extremes with volume
  - Classic technical signal but applied to tick-level data

EXP II: Liquidation Imbalance Ratio
  - Track Buy vs Sell liquidation ratio over rolling windows
  - When ratio is extreme (>80% one side), fade the dominant side
  - Different from cascade detection: uses continuous ratio, not events

EXP JJ: Funding Rate Regime + Cascade Interaction
  - When funding rate is very positive (longs pay), buy-cascades should be
    more violent (forced long liquidations). Does this improve cascade MM?
  - When FR is negative, sell-cascades should be more violent.
  - Load FR data, cross with cascade performance.

88 days, RAM-safe.
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
                df = pd.read_csv(f, usecols=['timestamp', 'price', 'size'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                dfs.append(df)
        if dfs:
            chunk = pd.concat(dfs, ignore_index=True)
            del dfs
            chunk['notional'] = chunk['size'] * chunk['price']
            b = chunk.set_index('timestamp').resample('1min').agg(
                {'price': ['first', 'max', 'min', 'last'], 'notional': 'sum'}).dropna()
            b.columns = ['open', 'high', 'low', 'close', 'volume']
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


def sim_trade(bars, entry_idx, is_long, offset, tp, sl, max_hold, trail_act=3, trail_dist=2):
    """Simulate a single trade with trailing stop."""
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


# ============================================================================
# EXP GG: LIQUIDATION VOLUME ACCELERATION
# ============================================================================

def exp_gg_liq_acceleration(liq_df, bars, symbol):
    print(f"\n{'='*80}")
    print(f"  EXP GG: LIQUIDATION VOLUME ACCELERATION — {symbol}")
    print(f"{'='*80}")

    # Build 1-minute liquidation bars
    liq_df = liq_df.set_index('timestamp')
    buy_vol = liq_df[liq_df['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
    sell_vol = liq_df[liq_df['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
    total_vol = (buy_vol + sell_vol).reindex(bars.index, fill_value=0)
    buy_vol = buy_vol.reindex(bars.index, fill_value=0)
    sell_vol = sell_vol.reindex(bars.index, fill_value=0)

    # Rolling averages
    for window in [5, 15, 30]:
        roll_avg = total_vol.rolling(window*60, min_periods=1).mean()  # per-minute avg
        # Actually use minute-level rolling
        roll_avg = total_vol.rolling(window, min_periods=1).mean()
        ratio = total_vol / (roll_avg + 1)

        print(f"\n  WINDOW={window}min:")
        for thresh in [3, 5, 10, 20]:
            # Find minutes where liq volume > thresh * rolling avg
            signals = ratio[ratio > thresh].index
            if len(signals) < 10:
                print(f"    thresh={thresh}x: {len(signals)} signals (too few)")
                continue

            # For each signal, determine direction from buy/sell imbalance
            trades = []
            last_time = None
            for ts in signals:
                if last_time and (ts - last_time).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
                if idx >= len(bars) - 30 or idx < 1: continue

                bv = buy_vol.iloc[idx] if idx < len(buy_vol) else 0
                sv = sell_vol.iloc[idx] if idx < len(sell_vol) else 0
                is_long = bv > sv  # fade the dominant liquidation side

                t = sim_trade(bars, idx, is_long, 0.15, 0.15, 0.50, 30)
                if t:
                    trades.append(t)
                    last_time = ts

            pstats(trades, f"w={window}m thresh={thresh}x (cd=300s)")

    # Reduced cooldown version
    print(f"\n  BEST CONFIG WITH REDUCED COOLDOWN:")
    roll_avg = total_vol.rolling(15, min_periods=1).mean()
    ratio = total_vol / (roll_avg + 1)
    for thresh in [5, 10]:
        signals = ratio[ratio > thresh].index
        trades = []
        last_time = None
        for ts in signals:
            if last_time and (ts - last_time).total_seconds() < 60: continue
            idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
            if idx >= len(bars) - 30 or idx < 1: continue
            bv = buy_vol.iloc[idx] if idx < len(buy_vol) else 0
            sv = sell_vol.iloc[idx] if idx < len(sell_vol) else 0
            is_long = bv > sv
            t = sim_trade(bars, idx, is_long, 0.15, 0.15, 0.50, 30)
            if t: trades.append(t); last_time = ts
        pstats(trades, f"w=15m thresh={thresh}x (cd=60s)")


# ============================================================================
# EXP II: LIQUIDATION IMBALANCE RATIO
# ============================================================================

def exp_ii_liq_imbalance(liq_df, bars, symbol):
    print(f"\n{'='*80}")
    print(f"  EXP II: LIQUIDATION IMBALANCE RATIO — {symbol}")
    print(f"{'='*80}")

    liq_df = liq_df.set_index('timestamp')
    buy_vol = liq_df[liq_df['side']=='Buy']['notional'].resample('1min').sum().fillna(0)
    sell_vol = liq_df[liq_df['side']=='Sell']['notional'].resample('1min').sum().fillna(0)
    buy_vol = buy_vol.reindex(bars.index, fill_value=0)
    sell_vol = sell_vol.reindex(bars.index, fill_value=0)

    for window in [5, 15, 30, 60]:
        buy_roll = buy_vol.rolling(window, min_periods=1).sum()
        sell_roll = sell_vol.rolling(window, min_periods=1).sum()
        total_roll = buy_roll + sell_roll
        buy_ratio = buy_roll / (total_roll + 1)

        print(f"\n  WINDOW={window}min:")
        for thresh in [0.80, 0.85, 0.90, 0.95]:
            # Buy-dominant → fade (go long, expecting bounce)
            buy_signals = buy_ratio[buy_ratio > thresh].index
            sell_signals = buy_ratio[buy_ratio < (1-thresh)].index

            trades = []
            last_time = None
            for ts in buy_signals:
                if last_time and (ts - last_time).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
                if idx >= len(bars) - 30 or idx < 1: continue
                t = sim_trade(bars, idx, True, 0.15, 0.15, 0.50, 30)  # long (fade buy liqs)
                if t: trades.append(t); last_time = ts

            for ts in sell_signals:
                if last_time and (ts - last_time).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
                if idx >= len(bars) - 30 or idx < 1: continue
                t = sim_trade(bars, idx, False, 0.15, 0.15, 0.50, 30)  # short (fade sell liqs)
                if t: trades.append(t); last_time = ts

            trades.sort(key=lambda t: t['time'])
            n_buy = len(buy_signals); n_sell = len(sell_signals)
            pstats(trades, f"w={window}m imb>{thresh:.0%} (buy={n_buy} sell={n_sell})")


# ============================================================================
# EXP HH: PRICE-VOLUME DIVERGENCE
# ============================================================================

def exp_hh_price_vol_divergence(bars, symbol):
    print(f"\n{'='*80}")
    print(f"  EXP HH: PRICE-VOLUME DIVERGENCE — {symbol}")
    print(f"{'='*80}")

    # Use 5-min bars
    bars5 = bars.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum'
    }).dropna()

    for lookback in [6, 12, 24]:  # 30min, 60min, 120min in 5min bars
        # Price making new high but volume declining
        price_high = bars5['high'].rolling(lookback).max()
        vol_avg = bars5['volume'].rolling(lookback).mean()
        vol_current = bars5['volume']

        # Bearish divergence: price at high, volume below avg
        bear_div = (bars5['high'] >= price_high * 0.999) & (vol_current < vol_avg * 0.5)
        # Bullish divergence: price at low, volume below avg
        price_low = bars5['low'].rolling(lookback).min()
        bull_div = (bars5['low'] <= price_low * 1.001) & (vol_current < vol_avg * 0.5)

        print(f"\n  LOOKBACK={lookback*5}min:")
        print(f"    Bearish divergences: {bear_div.sum()}")
        print(f"    Bullish divergences: {bull_div.sum()}")

        # Trade bearish divergences (short)
        trades = []
        last_time = None
        for ts in bars5[bear_div].index:
            if last_time and (ts - last_time).total_seconds() < 300: continue
            # Find corresponding 1min bar
            idx = bars.index.searchsorted(ts)
            if idx >= len(bars) - 30 or idx < 1: continue
            t = sim_trade(bars, idx, False, 0.15, 0.15, 0.50, 30)
            if t: trades.append(t); last_time = ts

        for ts in bars5[bull_div].index:
            if last_time and (ts - last_time).total_seconds() < 300: continue
            idx = bars.index.searchsorted(ts)
            if idx >= len(bars) - 30 or idx < 1: continue
            t = sim_trade(bars, idx, True, 0.15, 0.15, 0.50, 30)
            if t: trades.append(t); last_time = ts

        trades.sort(key=lambda t: t['time'])
        pstats(trades, f"lookback={lookback*5}min")


# ============================================================================
# MAIN
# ============================================================================

def main():
    out_file = 'results/v42m_new_independent.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    # Use SOL for initial testing (representative)
    symbol = 'SOLUSDT'
    all_dates = get_dates('2025-05-12', 88)

    print("="*80)
    print(f"  v42m: NEW INDEPENDENT SIGNAL IDEAS — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    liq = load_liqs(symbol, all_dates)
    bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
    gc.collect()

    print(f"\n  [{ram_str()}] data loaded")

    # Run experiments
    exp_gg_liq_acceleration(liq.copy(), bars, symbol)
    exp_ii_liq_imbalance(liq.copy(), bars, symbol)
    exp_hh_price_vol_divergence(bars, symbol)

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
