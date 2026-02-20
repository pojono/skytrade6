#!/usr/bin/env python3
"""
Tick-level backtest — NO intra-bar ambiguity possible.

Each tick is processed sequentially. When a limit order is pending,
we check each tick to see if it fills. Once filled, we check each
subsequent tick for TP/SL/trailing stop exit. No lookahead.

We generate signals from 1-min bars (computed from ticks), but
execute trades tick-by-tick.

Tests:
  1. All 5 original signals with tick-level execution
  2. RANDOM direction baseline (same timing, random long/short)
  3. INVERTED signal baseline
  4. Various trade structure parameters
  5. Multiple coins and periods
"""

import sys, time, os, gc, psutil, random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

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

def get_dates(start, n):
    base = datetime.strptime(start, '%Y-%m-%d')
    return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)]


def load_ticks(symbol, date_str, data_dir='data'):
    """Load tick data for a single day. Returns DataFrame with ts, price, side."""
    f = Path(data_dir) / symbol / "bybit" / "futures" / f"{symbol}{date_str}.csv.gz"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, usecols=['timestamp', 'price', 'side'])
    df = df.rename(columns={'timestamp': 'ts'})
    df = df.sort_values('ts').reset_index(drop=True)
    return df


def load_ticks_range(symbol, dates, data_dir='data'):
    """Load tick data for a range of dates."""
    t0 = time.time()
    print(f"  Loading {symbol} ticks...", end='', flush=True)
    all_ticks = []
    for i, d in enumerate(dates):
        df = load_ticks(symbol, d, data_dir)
        if not df.empty:
            all_ticks.append(df)
        if (i+1) % 5 == 0 or i == len(dates)-1:
            print(f" [{i+1}/{len(dates)} {time.time()-t0:.0f}s]", end='', flush=True)
    if not all_ticks:
        print(" NO DATA"); return pd.DataFrame()
    result = pd.concat(all_ticks, ignore_index=True)
    print(f" {len(result):,} ticks ({time.time()-t0:.0f}s) [{ram_str()}]")
    return result


def ticks_to_bars(ticks, freq='1min'):
    """Convert ticks to OHLC bars."""
    ticks_ts = ticks.copy()
    ticks_ts['datetime'] = pd.to_datetime(ticks_ts['ts'], unit='s')
    ticks_ts = ticks_ts.set_index('datetime')
    bars = ticks_ts['price'].resample(freq).agg(
        open='first', high='max', low='min', close='last').dropna()
    return bars


# ============================================================================
# SIGNAL GENERATORS (from 1-min bars)
# ============================================================================

def gen_ret_iqr(bars, window=60, threshold=0.95):
    """Return Distribution Width — fade extreme IQR."""
    ret = bars['close'].pct_change()
    q75 = ret.rolling(window).quantile(0.75)
    q25 = ret.rolling(window).quantile(0.25)
    iqr = q75 - q25
    pct = iqr.rolling(window).rank(pct=True)
    sig = pd.Series(0, index=bars.index)
    sig[pct > threshold] = -1
    sig[pct < (1-threshold)] = 1
    return sig

def gen_wt_mom_div(bars, window=40, threshold=0.90):
    """Weighted Momentum Divergence — fade extreme divergence."""
    ret = bars['close'].pct_change()
    rng = (bars['high'] - bars['low']) / bars['close']
    wt_mom = (ret * rng).rolling(window).sum() / rng.rolling(window).sum()
    simple_mom = ret.rolling(window).mean()
    div = wt_mom - simple_mom
    pct = div.rolling(window).rank(pct=True)
    sig = pd.Series(0, index=bars.index)
    sig[pct > threshold] = -1
    sig[pct < (1-threshold)] = 1
    return sig

def gen_macd_hv(bars, lag=3, threshold=0.90):
    """MACD Histogram Velocity — fade extreme acceleration."""
    ema12 = bars['close'].ewm(span=12).mean()
    ema26 = bars['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    hv = hist - hist.shift(lag)
    pct = hv.rolling(60).rank(pct=True)
    sig = pd.Series(0, index=bars.index)
    sig[pct > threshold] = -1
    sig[pct < (1-threshold)] = 1
    return sig

def gen_stoch_vel(bars, window=30, lag=3, threshold=0.90):
    """Stochastic Velocity — fade extreme rate of change of %K."""
    low_n = bars['low'].rolling(window).min()
    high_n = bars['high'].rolling(window).max()
    k = (bars['close'] - low_n) / (high_n - low_n + 1e-10) * 100
    vel = k - k.shift(lag)
    pct = vel.rolling(window).rank(pct=True)
    sig = pd.Series(0, index=bars.index)
    sig[pct > threshold] = -1
    sig[pct < (1-threshold)] = 1
    return sig

def gen_regime_persist(bars, window=30):
    """Regime Persistence — fade after long same-direction runs."""
    ret = bars['close'].pct_change()
    pos_count = (ret > 0).rolling(window).sum()
    ratio = pos_count / window
    sig = pd.Series(0, index=bars.index)
    sig[ratio > 0.7] = -1   # too many up bars → fade
    sig[ratio < 0.3] = 1    # too many down bars → fade
    return sig


# ============================================================================
# TICK-LEVEL TRADE SIMULATOR
# ============================================================================

class TickTradeSimulator:
    """
    Simulates trades tick-by-tick. No intra-bar ambiguity.
    
    State machine:
      IDLE → signal fires → place limit order → PENDING
      PENDING → tick hits limit → FILLED (record fill price & time)
      FILLED → tick hits TP/SL/trail → CLOSED (record exit)
      PENDING → max_wait ticks elapsed → CANCELLED
      FILLED → max_hold time elapsed → TIMEOUT exit at current price
    """
    
    def __init__(self, offset_pct=0.15, tp_pct=0.15, sl_pct=0.50,
                 trail_act_bps=3, trail_dist_bps=2, max_hold_sec=1800):
        self.offset_pct = offset_pct
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trail_act_bps = trail_act_bps
        self.trail_dist_bps = trail_dist_bps
        self.max_hold_sec = max_hold_sec  # 30 min = 1800 sec
        self.max_wait_sec = max_hold_sec  # max time to wait for fill
    
    def sim_one_trade(self, ticks_arr, start_idx, is_long):
        """
        Simulate a single trade starting from start_idx in ticks_arr.
        ticks_arr: numpy array with columns [ts, price] (pre-extracted)
        
        Returns dict with trade details or None if not filled.
        """
        n = len(ticks_arr)
        if start_idx >= n - 10:
            return None
        
        signal_price = ticks_arr[start_idx, 1]  # price at signal time
        signal_ts = ticks_arr[start_idx, 0]
        
        # Calculate limit price
        if is_long:
            lim = signal_price * (1 - self.offset_pct / 100)
        else:
            lim = signal_price * (1 + self.offset_pct / 100)
        
        # Phase 1: Wait for fill
        fill_idx = None
        for j in range(start_idx + 1, n):  # start from NEXT tick
            ts = ticks_arr[j, 0]
            price = ticks_arr[j, 1]
            
            # Timeout waiting for fill
            if ts - signal_ts > self.max_wait_sec:
                return None
            
            # Check fill
            if is_long and price <= lim:
                fill_idx = j
                fill_price = lim  # limit order fills at limit price
                fill_ts = ts
                break
            elif not is_long and price >= lim:
                fill_idx = j
                fill_price = lim
                fill_ts = ts
                break
        
        if fill_idx is None:
            return None
        
        # Calculate TP and SL prices
        if is_long:
            tp_price = fill_price * (1 + self.tp_pct / 100)
            sl_price = fill_price * (1 - self.sl_pct / 100)
        else:
            tp_price = fill_price * (1 - self.tp_pct / 100)
            sl_price = fill_price * (1 + self.sl_pct / 100)
        
        # Phase 2: Wait for exit (tick by tick, starting from NEXT tick after fill)
        best_profit = 0
        trailing_active = False
        current_sl = sl_price
        exit_price = None
        exit_reason = 'timeout'
        exit_ts = None
        
        for k in range(fill_idx + 1, n):  # start from tick AFTER fill
            ts = ticks_arr[k, 0]
            price = ticks_arr[k, 1]
            
            # Timeout
            if ts - fill_ts > self.max_hold_sec:
                exit_price = price  # market exit at current price
                exit_reason = 'timeout'
                exit_ts = ts
                break
            
            if is_long:
                # Current profit
                cp = (price - fill_price) / fill_price
                if cp > best_profit:
                    best_profit = cp
                
                # Trailing stop activation
                if best_profit >= self.trail_act_bps / 10000 and not trailing_active:
                    trailing_active = True
                    current_sl = fill_price * (1 + self.trail_dist_bps / 10000)
                
                # Update trailing stop
                if trailing_active:
                    new_sl = price * (1 - self.trail_dist_bps / 10000)
                    if new_sl > current_sl:
                        current_sl = new_sl
                
                # Check SL/trail hit
                if price <= current_sl:
                    exit_price = current_sl
                    exit_reason = 'trail' if trailing_active else 'sl'
                    exit_ts = ts
                    break
                
                # Check TP hit
                if price >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    exit_ts = ts
                    break
            else:
                # Short trade
                cp = (fill_price - price) / fill_price
                if cp > best_profit:
                    best_profit = cp
                
                if best_profit >= self.trail_act_bps / 10000 and not trailing_active:
                    trailing_active = True
                    current_sl = fill_price * (1 - self.trail_dist_bps / 10000)
                
                if trailing_active:
                    new_sl = price * (1 + self.trail_dist_bps / 10000)
                    if new_sl < current_sl:
                        current_sl = new_sl
                
                if price >= current_sl:
                    exit_price = current_sl
                    exit_reason = 'trail' if trailing_active else 'sl'
                    exit_ts = ts
                    break
                
                if price <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    exit_ts = ts
                    break
        
        if exit_price is None:
            # Ran out of ticks
            exit_price = ticks_arr[-1, 1]
            exit_reason = 'timeout'
            exit_ts = ticks_arr[-1, 0]
        
        # Calculate P&L
        if is_long:
            gross = (exit_price - fill_price) / fill_price
        else:
            gross = (fill_price - exit_price) / fill_price
        
        # Fees: maker for limit entry, maker for TP (limit), taker for SL/trail/timeout
        fee = MAKER_FEE + (MAKER_FEE if exit_reason == 'tp' else TAKER_FEE)
        net = gross - fee
        
        return {
            'net': net,
            'gross': gross,
            'fee': fee,
            'exit': exit_reason,
            'is_long': is_long,
            'signal_ts': signal_ts,
            'fill_ts': fill_ts,
            'exit_ts': exit_ts,
            'fill_delay': fill_ts - signal_ts,
            'hold_time': exit_ts - fill_ts,
            'fill_price': fill_price,
            'exit_price': exit_price,
        }


def pstats(trades, label):
    if not trades:
        print(f"    {label:50s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "❌"
    print(f"  {flag} {label:50s}  n={n:5d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+8.1f}%  sh={sh:+6.0f}")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot}


def exit_breakdown(trades, label=""):
    """Show exit type distribution."""
    if not trades: return
    exits = defaultdict(list)
    for t in trades:
        exits[t['exit']].append(t['net'] * 10000)
    print(f"    Exit breakdown{' ('+label+')' if label else ''}:")
    for er in ['tp', 'trail', 'sl', 'timeout']:
        if er in exits:
            arr = np.array(exits[er])
            print(f"      {er:8s}: n={len(arr):5d} ({100*len(arr)/len(trades):4.1f}%)  "
                  f"avg={arr.mean():+6.1f}bps  wr={100*(arr>0).mean():5.1f}%")


def run_signal_test(ticks_arr, bar_signals, sim, label, show_exits=False):
    """
    Run a signal through the tick simulator.
    bar_signals: list of (bar_close_ts, direction) tuples
    """
    trades = []
    last_exit_ts = 0  # prevent overlapping trades
    
    for sig_ts, direction in bar_signals:
        # Skip if we're still in a trade
        if sig_ts < last_exit_ts:
            continue
        
        # Find the tick index closest to (but after) the signal timestamp
        idx = np.searchsorted(ticks_arr[:, 0], sig_ts)
        if idx >= len(ticks_arr) - 100:
            continue
        
        t = sim.sim_one_trade(ticks_arr, idx, direction)
        if t:
            trades.append(t)
            last_exit_ts = t['exit_ts']
    
    result = pstats(trades, label)
    if show_exits and trades:
        exit_breakdown(trades)
    return trades


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    t_start = time.time()
    
    print("=" * 100)
    print("  TICK-LEVEL BACKTEST — Zero intra-bar ambiguity")
    print("=" * 100)
    
    # Signal generators
    SIGNALS = {
        'ret_iqr_60_95': lambda bars: gen_ret_iqr(bars, 60, 0.95),
        'wt_mom_40_90': lambda bars: gen_wt_mom_div(bars, 40, 0.90),
        'macd_hv_3_90': lambda bars: gen_macd_hv(bars, 3, 0.90),
        'stoch_vel_30_3_90': lambda bars: gen_stoch_vel(bars, 30, 3, 0.90),
        'regime_persist_30': lambda bars: gen_regime_persist(bars, 30),
    }
    
    # Trade structure variants to test
    STRUCTURES = {
        'default': dict(offset_pct=0.15, tp_pct=0.15, sl_pct=0.50, trail_act_bps=3, trail_dist_bps=2),
        'no_trail': dict(offset_pct=0.15, tp_pct=0.15, sl_pct=0.50, trail_act_bps=99999, trail_dist_bps=2),
        'zero_offset': dict(offset_pct=0.0, tp_pct=0.15, sl_pct=0.50, trail_act_bps=3, trail_dist_bps=2),
        'wide_tp': dict(offset_pct=0.15, tp_pct=0.30, sl_pct=0.50, trail_act_bps=5, trail_dist_bps=3),
        'tight_sl': dict(offset_pct=0.15, tp_pct=0.15, sl_pct=0.20, trail_act_bps=3, trail_dist_bps=2),
    }
    
    # Test period: use shorter period for faster results
    # Load 30 days total, split at day 20 for IS/OOS
    dates_all = get_dates('2025-07-01', 30)
    split_date = '2025-07-21'
    
    for sym in ['ETHUSDT', 'DOGEUSDT']:
        print(f"\n{'#'*100}")
        print(f"  {sym}")
        print(f"{'#'*100}")
        
        # Load ticks for OOS period only (to save memory)
        oos_dates = [d for d in dates_all if d >= split_date]
        ticks = load_ticks_range(sym, oos_dates)
        if ticks.empty:
            continue
        
        # Also need IS period bars for signal warmup
        is_dates = [d for d in dates_all if d < split_date]
        # Load IS ticks just to build bars for signal warmup
        is_ticks = load_ticks_range(sym, is_dates[-10:])  # last 10 days of IS for warmup
        
        # Build 1-min bars from all available ticks
        all_ticks_for_bars = pd.concat([is_ticks, ticks], ignore_index=True)
        del is_ticks; gc.collect()
        
        bars = ticks_to_bars(all_ticks_for_bars)
        del all_ticks_for_bars; gc.collect()
        
        print(f"  Bars: {len(bars):,} (1-min)")
        
        # Prepare tick array for fast simulation (OOS ticks only)
        ticks_arr = ticks[['ts', 'price']].values.astype(np.float64)
        oos_start_ts = pd.Timestamp(split_date).timestamp()
        del ticks; gc.collect()
        
        # Generate signals from bars
        oos_bars = bars[bars.index >= split_date]
        
        # ================================================================
        print(f"\n  === TEST 1: All signals with DEFAULT trade structure ===")
        # ================================================================
        sim = TickTradeSimulator(**STRUCTURES['default'])
        
        for sig_name, sig_fn in SIGNALS.items():
            sig = sig_fn(bars)
            sig_oos = sig[sig.index >= split_date]
            
            # Convert bar signals to (timestamp, direction) list
            bar_signals = []
            for ts, val in sig_oos.items():
                if val != 0:
                    bar_signals.append((ts.timestamp(), val > 0))
            
            trades = run_signal_test(ticks_arr, bar_signals, sim, 
                                     f"{sig_name} ({sym})", show_exits=True)
        
        # ================================================================
        print(f"\n  === TEST 2: RANDOM direction baseline ===")
        # ================================================================
        # Use ret_iqr timing but random direction
        sig = gen_ret_iqr(bars, 60, 0.95)
        sig_oos = sig[sig.index >= split_date]
        signal_times = [(ts.timestamp(), None) for ts, val in sig_oos.items() if val != 0]
        
        for seed in [42, 43, 44]:
            random.seed(seed)
            rand_signals = [(ts, random.choice([True, False])) for ts, _ in signal_times]
            run_signal_test(ticks_arr, rand_signals, sim, 
                           f"RANDOM (seed={seed}) ({sym})")
        
        # ================================================================
        print(f"\n  === TEST 3: INVERTED signal baseline ===")
        # ================================================================
        sig = gen_ret_iqr(bars, 60, 0.95)
        sig_oos = sig[sig.index >= split_date]
        inv_signals = [(ts.timestamp(), val < 0) for ts, val in sig_oos.items() if val != 0]
        run_signal_test(ticks_arr, inv_signals, sim, 
                       f"INVERTED ret_iqr ({sym})", show_exits=True)
        
        # ================================================================
        print(f"\n  === TEST 4: Trade structure variants (ret_iqr signal) ===")
        # ================================================================
        sig = gen_ret_iqr(bars, 60, 0.95)
        sig_oos = sig[sig.index >= split_date]
        real_signals = [(ts.timestamp(), val > 0) for ts, val in sig_oos.items() if val != 0]
        
        for struct_name, params in STRUCTURES.items():
            sim_var = TickTradeSimulator(**params)
            run_signal_test(ticks_arr, real_signals, sim_var,
                           f"struct={struct_name} ({sym})")
        
        # ================================================================
        print(f"\n  === TEST 5: EVERY-30-min entry (no signal) ===")
        # ================================================================
        # Enter every 30 minutes with alternating direction
        every_signals = []
        t = oos_start_ts
        end_ts = ticks_arr[-1, 0]
        i = 0
        while t < end_ts - 3600:
            every_signals.append((t, i % 2 == 0))
            t += 1800  # 30 min
            i += 1
        
        run_signal_test(ticks_arr, every_signals, sim,
                       f"EVERY-30min alternating ({sym})")
        
        # Long only
        every_long = [(ts, True) for ts, _ in every_signals]
        run_signal_test(ticks_arr, every_long, sim,
                       f"EVERY-30min LONG only ({sym})")
        
        # Short only
        every_short = [(ts, False) for ts, _ in every_signals]
        run_signal_test(ticks_arr, every_short, sim,
                       f"EVERY-30min SHORT only ({sym})")
        
        # ================================================================
        print(f"\n  === TEST 6: Fill & hold time analysis ===")
        # ================================================================
        sig = gen_ret_iqr(bars, 60, 0.95)
        sig_oos = sig[sig.index >= split_date]
        real_signals = [(ts.timestamp(), val > 0) for ts, val in sig_oos.items() if val != 0]
        trades = run_signal_test(ticks_arr, real_signals, sim, f"ret_iqr detail ({sym})")
        
        if trades:
            fill_delays = np.array([t['fill_delay'] for t in trades])
            hold_times = np.array([t['hold_time'] for t in trades])
            print(f"    Fill delay: mean={fill_delays.mean():.1f}s  "
                  f"median={np.median(fill_delays):.1f}s  "
                  f"p90={np.percentile(fill_delays, 90):.1f}s")
            print(f"    Hold time:  mean={hold_times.mean():.1f}s  "
                  f"median={np.median(hold_times):.1f}s  "
                  f"p90={np.percentile(hold_times, 90):.1f}s")
            
            # Long vs Short
            longs = [t for t in trades if t['is_long']]
            shorts = [t for t in trades if not t['is_long']]
            if longs:
                l_avg = np.mean([t['net'] for t in longs]) * 10000
                l_wr = 100 * np.mean([t['net'] > 0 for t in longs])
                print(f"    LONG:  n={len(longs):4d}  avg={l_avg:+6.1f}bps  wr={l_wr:5.1f}%")
            if shorts:
                s_avg = np.mean([t['net'] for t in shorts]) * 10000
                s_wr = 100 * np.mean([t['net'] > 0 for t in shorts])
                print(f"    SHORT: n={len(shorts):4d}  avg={s_avg:+6.1f}bps  wr={s_wr:5.1f}%")
        
        del ticks_arr, bars
        gc.collect()
    
    elapsed = time.time() - t_start
    print(f"\n{'#'*100}")
    print(f"  TICK BACKTEST COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*100}")
