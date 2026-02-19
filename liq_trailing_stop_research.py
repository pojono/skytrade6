#!/usr/bin/env python3
"""
TRAILING STOP RESEARCH for liquidation cascade strategy.

Purpose: Investigate whether trailing stops can cut timeout losses while
preserving the high win rate. The current strategy has:
  - TP: 0.12% (fixed take-profit)
  - SL: None (no stop-loss)
  - Timeout: 60 min (exit at close price)
  - Timeout trades: ~4.5% of all trades, avg loss -0.86%
  - Each timeout loss wipes out ~11 TP wins

Research questions:
  1. What is the bar-by-bar PnL path of timeout trades? Do they go profitable
     first then reverse, or go against us immediately?
  2. Can a trailing stop lock in profit on trades that go our way then reverse?
  3. Can a trailing stop cut losses earlier on trades that drift against us?
  4. What trail width is optimal? Too tight = stopped out on noise before TP.
     Too wide = doesn't help with timeouts.
  5. Does trailing stop + TP + timeout outperform current TP + timeout?

Variants to test:
  A. BASELINE: Current strategy (TP=0.12%, no SL, 60min timeout)
  B. TRAILING ONLY: No fixed TP, trailing stop from entry (various widths)
  C. TP + TRAILING: Fixed TP + trailing stop activated after fill
  D. TP + TRAILING (activated after profit): Trailing stop only activates
     once trade is in profit by X bps
  E. TRAILING + TIMEOUT: Trailing stop + reduced timeout (30min, 15min)
  F. FIXED SL + TP: For comparison — simple fixed stop-loss

Trail widths to sweep: 5, 8, 10, 12, 15, 20, 25, 30, 40, 50 bps
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.055
SYMBOLS = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']


# ============================================================================
# DATA LOADING (same as stress test)
# ============================================================================

def load_liquidations(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    print(f"  Loading {len(liq_files)} liq files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 500 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for ev in data['result']['data']:
                            records.append({
                                'timestamp': pd.to_datetime(ev['T'], unit='ms'),
                                'side': ev['S'],
                                'volume': float(ev['v']),
                                'price': float(ev['p']),
                            })
                except Exception:
                    continue
    print(f" done ({len(records):,})")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    return df


def load_ticker(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    
    # Prefer preprocessed WS CSV (~100ms resolution) over REST JSONL (~5s)
    csv_path = symbol_dir / "ticker_prices.csv.gz"
    if csv_path.exists():
        print(f"  Loading ticker CSV (WS ~100ms resolution)...", end='', flush=True)
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df['price'] = df['price'].astype(float)
        df = df[['timestamp', 'price']].sort_values('timestamp').reset_index(drop=True)
        print(f" done ({len(df):,})")
        return df
    
    # Fallback: REST API ticker files (~5s resolution)
    ticker_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    print(f"  Loading {len(ticker_files)} REST ticker files (~5s resolution)...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 500 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    r = data['result']['list'][0]
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'price': float(r['lastPrice']),
                    })
                except Exception:
                    continue
    print(f" done ({len(records):,})")
    return pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)


def build_bars(ticker_df, freq='1min'):
    ticker_df = ticker_df.set_index('timestamp')
    bars = ticker_df['price'].resample(freq).ohlc().dropna()
    print(f"  Building bars... {len(bars):,} bars")
    return bars


# ============================================================================
# CASCADE DETECTION (same as stress test — global P95, min_ev=1)
# ============================================================================

def detect_signals(liq_df, price_bars, pct_thresh=95):
    thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= thresh].copy()
    bar_index = price_bars.index
    bar_close = price_bars['close'].values
    timestamps = large['timestamp'].values
    sides = large['side'].values
    notionals = large['notional'].values
    n = len(large)
    cascades = []
    i = 0
    while i < n:
        cluster = [i]
        j = i + 1
        while j < n:
            dt = (timestamps[j] - timestamps[cluster[-1]]).astype('timedelta64[s]').astype(float)
            if dt <= 60:
                cluster.append(j)
                j += 1
            else:
                break
        c_sides = sides[cluster]
        c_notionals = notionals[cluster]
        c_ts = timestamps[cluster]
        buy_not = c_notionals[c_sides == 'Buy'].sum()
        sell_not = c_notionals[c_sides == 'Sell'].sum()
        buy_dominant = buy_not > sell_not
        end_ts = pd.Timestamp(c_ts[-1])
        end_idx = bar_index.searchsorted(end_ts)
        if end_idx >= len(bar_close) - 120 or end_idx < 10:
            i = cluster[-1] + 1
            continue
        current_price = bar_close[end_idx]
        start_idx = bar_index.searchsorted(pd.Timestamp(c_ts[0]))
        if start_idx > 0:
            pre_price = bar_close[max(0, start_idx - 1)]
            cascade_disp_bps = (current_price - pre_price) / pre_price * 10000
        else:
            cascade_disp_bps = 0
        cascades.append({
            'end': end_ts,
            'n_events': len(cluster),
            'buy_dominant': buy_dominant,
            'end_bar_idx': end_idx,
            'current_price': current_price,
            'cascade_disp_bps': cascade_disp_bps,
        })
        i = cluster[-1] + 1
    return cascades


# ============================================================================
# PART 1: ANALYZE TIMEOUT TRADE PATHS
# Understand what happens bar-by-bar during timeout trades
# ============================================================================

def analyze_trade_paths(cascades, price_bars, tp_pct=0.12, entry_offset_pct=0.15,
                        max_hold_min=60, min_disp_bps=10):
    """Record full bar-by-bar PnL path for every trade, especially timeouts."""
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values
    bar_close = price_bars['close'].values
    
    all_paths = []  # list of dicts with full path info
    last_trade_time = None
    
    for cascade in cascades:
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < 5 * 60:
                continue
        if abs(cascade['cascade_disp_bps']) < min_disp_bps:
            continue
        
        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']
        direction = 'long' if cascade['buy_dominant'] else 'short'
        
        if direction == 'long':
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
        else:
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
        
        # Find fill
        end_bar = min(idx + max_hold_min, len(bar_close) - 1)
        filled = False
        fill_bar = None
        for j in range(idx, end_bar + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True; fill_bar = j; break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True; fill_bar = j; break
        if not filled:
            continue
        
        # Record bar-by-bar path from fill to timeout
        remaining = max_hold_min - (fill_bar - idx)
        exit_end = min(fill_bar + remaining, len(bar_close) - 1)
        
        path_pnl = []  # PnL at each bar (using close)
        path_best = []  # Best PnL seen so far (using high for long, low for short)
        path_worst = []  # Worst PnL seen so far
        
        best_pnl = -999
        worst_pnl = 999
        exit_reason = 'timeout'
        exit_bar = exit_end
        
        for k in range(fill_bar, exit_end + 1):
            # Current PnL at close
            if direction == 'long':
                close_pnl = (bar_close[k] - limit_price) / limit_price * 10000  # in bps
                bar_best = (bar_high[k] - limit_price) / limit_price * 10000
                bar_worst = (bar_low[k] - limit_price) / limit_price * 10000
            else:
                close_pnl = (limit_price - bar_close[k]) / limit_price * 10000
                bar_best = (limit_price - bar_low[k]) / limit_price * 10000
                bar_worst = (limit_price - bar_high[k]) / limit_price * 10000
            
            best_pnl = max(best_pnl, bar_best)
            worst_pnl = min(worst_pnl, bar_worst)
            
            path_pnl.append(close_pnl)
            path_best.append(best_pnl)
            path_worst.append(worst_pnl)
            
            # Check TP hit
            if direction == 'long' and bar_high[k] >= tp_price:
                exit_reason = 'take_profit'
                exit_bar = k
                break
            elif direction == 'short' and bar_low[k] <= tp_price:
                exit_reason = 'take_profit'
                exit_bar = k
                break
        
        # Compute final PnL
        if exit_reason == 'take_profit':
            exit_price = tp_price
        else:
            exit_price = bar_close[exit_end]
        
        if direction == 'long':
            raw_pnl_pct = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl_pct = (limit_price - exit_price) / limit_price * 100
        
        entry_fee = MAKER_FEE_PCT
        exit_fee = MAKER_FEE_PCT if exit_reason == 'take_profit' else TAKER_FEE_PCT
        net_pnl = raw_pnl_pct - entry_fee - exit_fee
        
        all_paths.append({
            'direction': direction,
            'limit_price': limit_price,
            'exit_reason': exit_reason,
            'net_pnl': net_pnl,
            'bars_held': exit_bar - fill_bar,
            'path_pnl_bps': path_pnl,
            'path_best_bps': path_best,
            'path_worst_bps': path_worst,
            'peak_profit_bps': best_pnl,
            'max_drawdown_bps': worst_pnl,
            'time': cascade['end'],
        })
        last_trade_time = cascade['end']
    
    return all_paths


# ============================================================================
# PART 2: STRATEGY WITH TRAILING STOP
# Careful bar-by-bar simulation with trailing stop logic
# ============================================================================

def run_strategy_trailing(cascades, price_bars, 
                          tp_pct=0.12,           # Fixed TP (None = no fixed TP)
                          trail_width_bps=None,  # Trailing stop width in bps (None = no trail)
                          trail_activate_bps=0,  # Only activate trail after this profit (bps)
                          sl_pct=None,           # Fixed stop-loss (None = no SL)
                          entry_offset_pct=0.15,
                          max_hold_min=60,
                          min_disp_bps=10,
                          maker_fee=MAKER_FEE_PCT,
                          taker_fee=TAKER_FEE_PCT):
    """
    Strategy with optional trailing stop.
    
    Trailing stop logic (for long):
      - Track highest price seen since fill (peak_price)
      - trail_level = peak_price * (1 - trail_width_bps / 10000)
      - If bar_low <= trail_level → exit at trail_level
      - Trail only activates once unrealized PnL >= trail_activate_bps
    
    Exit priority per bar:
      1. Fixed SL (if set) — checked first (worst case)
      2. Trailing stop — checked second
      3. Fixed TP — checked third (best case)
      4. Timeout — if none of the above by max_hold_min
    """
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values
    bar_close = price_bars['close'].values
    trades = []
    last_trade_time = None
    
    for cascade in cascades:
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < 5 * 60:
                continue
        if abs(cascade['cascade_disp_bps']) < min_disp_bps:
            continue
        
        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']
        direction = 'long' if cascade['buy_dominant'] else 'short'
        
        if direction == 'long':
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100) if tp_pct else None
            sl_price = limit_price * (1 - sl_pct / 100) if sl_pct else None
        else:
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100) if tp_pct else None
            sl_price = limit_price * (1 + sl_pct / 100) if sl_pct else None
        
        # Find fill
        end_bar = min(idx + max_hold_min, len(bar_close) - 1)
        filled = False
        fill_bar = None
        for j in range(idx, end_bar + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True; fill_bar = j; break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True; fill_bar = j; break
        if not filled:
            continue
        
        # Bar-by-bar exit logic with trailing stop
        remaining = max_hold_min - (fill_bar - idx)
        exit_end = min(fill_bar + remaining, len(bar_close) - 1)
        
        exit_price = None
        exit_reason = 'timeout'
        
        # Trailing stop state
        if direction == 'long':
            peak_price = limit_price  # best price seen (starts at fill)
        else:
            peak_price = limit_price  # worst price for short (starts at fill)
        
        trail_active = False
        
        for k in range(fill_bar, exit_end + 1):
            # --- 1. Check fixed SL ---
            if sl_price is not None:
                if direction == 'long' and bar_low[k] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    break
                elif direction == 'short' and bar_high[k] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    break
            
            # --- 2. Update trailing stop ---
            if trail_width_bps is not None:
                if direction == 'long':
                    peak_price = max(peak_price, bar_high[k])
                    unrealized_bps = (peak_price - limit_price) / limit_price * 10000
                else:
                    peak_price = min(peak_price, bar_low[k])
                    unrealized_bps = (limit_price - peak_price) / limit_price * 10000
                
                # Activate trail once profit threshold reached
                if unrealized_bps >= trail_activate_bps:
                    trail_active = True
                
                if trail_active:
                    if direction == 'long':
                        trail_level = peak_price * (1 - trail_width_bps / 10000)
                        if bar_low[k] <= trail_level:
                            # Stopped out — use trail level (or bar close if gap)
                            exit_price = max(trail_level, bar_low[k])
                            exit_reason = 'trailing_stop'
                            break
                    else:
                        trail_level = peak_price * (1 + trail_width_bps / 10000)
                        if bar_high[k] >= trail_level:
                            exit_price = min(trail_level, bar_high[k])
                            exit_reason = 'trailing_stop'
                            break
            
            # --- 3. Check fixed TP ---
            if tp_price is not None:
                if direction == 'long' and bar_high[k] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    break
                elif direction == 'short' and bar_low[k] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    break
        
        # --- 4. Timeout ---
        if exit_price is None:
            exit_price = bar_close[exit_end]
            exit_reason = 'timeout'
        
        # PnL
        if direction == 'long':
            raw_pnl = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl = (limit_price - exit_price) / limit_price * 100
        
        # Fees: maker for limit entry, maker for TP (limit), taker for everything else
        entry_fee = maker_fee
        if exit_reason == 'take_profit':
            exit_fee = maker_fee
        else:
            exit_fee = taker_fee  # trailing stop and timeout are market exits
        net_pnl = raw_pnl - entry_fee - exit_fee
        
        trades.append({
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'time': cascade['end'],
            'direction': direction,
        })
        last_trade_time = cascade['end']
    
    return trades


# ============================================================================
# ANALYSIS HELPERS
# ============================================================================

def stats(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'total': 0, 'avg': 0, 'sharpe': 0, 'maxdd': 0}
    net = np.array([t['net_pnl'] for t in trades])
    wr = (net > 0).mean() * 100
    total = net.sum()
    avg = net.mean()
    sharpe = avg / net.std() * np.sqrt(252 * 8) if net.std() > 0 else 0
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    maxdd = (peak - cum).max()
    return {'n': len(trades), 'wr': wr, 'total': total, 'avg': avg, 'sharpe': sharpe, 'maxdd': maxdd}


def exit_breakdown(trades):
    """Count trades by exit reason."""
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'count': 0, 'pnl_sum': 0, 'pnl_list': []}
        reasons[r]['count'] += 1
        reasons[r]['pnl_sum'] += t['net_pnl']
        reasons[r]['pnl_list'].append(t['net_pnl'])
    return reasons


def pline(label, s, width=55):
    if s['n'] == 0:
        print(f"    {label:{width}s}  n=    0  (no trades)")
        return
    flag = '✅' if s['total'] > 0 else '❌'
    print(f"  {flag} {label:{width}s}  n={s['n']:>5d}  WR={s['wr']:5.1f}%  "
          f"avg={s['avg']:+.4f}%  tot={s['total']:+7.2f}%  sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    
    print("=" * 100)
    print("  TRAILING STOP RESEARCH — Liquidation Cascade Strategy")
    print("  Baseline: off=0.15%, TP=0.12%, no SL, 60min timeout, disp≥10bps, min_ev=1")
    print("=" * 100)
    
    # Load all data
    all_data = {}
    for symbol in SYMBOLS:
        print(f"\n{'─' * 80}")
        print(f"  Loading {symbol}...")
        print(f"{'─' * 80}")
        liq_df = load_liquidations(symbol)
        ticker_df = load_ticker(symbol)
        bars = build_bars(ticker_df)
        cascades = detect_signals(liq_df, bars)
        print(f"  Signals: {len(cascades)}")
        all_data[symbol] = {'bars': bars, 'cascades': cascades}
    
    # ======================================================================
    # PART 1: TIMEOUT TRADE PATH ANALYSIS
    # ======================================================================
    print("\n" + "#" * 100)
    print("  PART 1: TIMEOUT TRADE PATH ANALYSIS")
    print("  Understanding what happens during timeout trades bar-by-bar")
    print("#" * 100)
    
    all_timeout_paths = []
    all_tp_paths = []
    
    for symbol in SYMBOLS:
        d = all_data[symbol]
        paths = analyze_trade_paths(d['cascades'], d['bars'])
        
        timeouts = [p for p in paths if p['exit_reason'] == 'timeout']
        tps = [p for p in paths if p['exit_reason'] == 'take_profit']
        all_timeout_paths.extend(timeouts)
        all_tp_paths.extend(tps)
        
        print(f"\n  ── {symbol} ──")
        print(f"    Total trades: {len(paths)}")
        print(f"    TP trades: {len(tps)}")
        print(f"    Timeout trades: {len(timeouts)}")
        
        if timeouts:
            peak_profits = [p['peak_profit_bps'] for p in timeouts]
            final_pnls = [p['net_pnl'] for p in timeouts]
            print(f"    Timeout peak profit before reversal:")
            print(f"      Mean:  {np.mean(peak_profits):+.1f} bps")
            print(f"      P25:   {np.percentile(peak_profits, 25):+.1f} bps")
            print(f"      P50:   {np.percentile(peak_profits, 50):+.1f} bps")
            print(f"      P75:   {np.percentile(peak_profits, 75):+.1f} bps")
            print(f"      P90:   {np.percentile(peak_profits, 90):+.1f} bps")
            print(f"    Timeout trades that were profitable at some point:")
            profitable_at_some_point = sum(1 for p in peak_profits if p > 0)
            print(f"      {profitable_at_some_point}/{len(timeouts)} ({profitable_at_some_point/len(timeouts)*100:.0f}%)")
            
            # How many would have been saved by trailing stop at various widths?
            for trail_bps in [5, 8, 10, 12, 15, 20]:
                saved = sum(1 for p in timeouts if p['peak_profit_bps'] >= trail_bps)
                if saved > 0:
                    # Estimate PnL if we had trailed: exit at peak - trail_width
                    est_pnl = [(p['peak_profit_bps'] - trail_bps) / 100 - MAKER_FEE_PCT - TAKER_FEE_PCT 
                               for p in timeouts if p['peak_profit_bps'] >= trail_bps]
                    print(f"      Trail {trail_bps:2d} bps would catch {saved:3d}/{len(timeouts)} "
                          f"(est avg PnL: {np.mean(est_pnl):+.4f}%)")
    
    # Aggregate timeout analysis
    print(f"\n  ── AGGREGATE ({len(all_timeout_paths)} timeout trades) ──")
    if all_timeout_paths:
        peak_profits = [p['peak_profit_bps'] for p in all_timeout_paths]
        print(f"    Peak profit distribution (bps):")
        for pct in [10, 25, 50, 75, 90, 95, 99]:
            print(f"      P{pct:2d}: {np.percentile(peak_profits, pct):+.1f} bps")
        
        # Categorize timeout trades
        never_profitable = sum(1 for p in peak_profits if p <= 0)
        small_profit = sum(1 for p in peak_profits if 0 < p <= 5)
        medium_profit = sum(1 for p in peak_profits if 5 < p <= 12)
        large_profit = sum(1 for p in peak_profits if p > 12)
        print(f"\n    Timeout trade categories:")
        print(f"      Never profitable (peak ≤ 0 bps):  {never_profitable} ({never_profitable/len(all_timeout_paths)*100:.0f}%)")
        print(f"      Small peak (0-5 bps):              {small_profit} ({small_profit/len(all_timeout_paths)*100:.0f}%)")
        print(f"      Medium peak (5-12 bps):            {medium_profit} ({medium_profit/len(all_timeout_paths)*100:.0f}%)")
        print(f"      Large peak (>12 bps = above TP):   {large_profit} ({large_profit/len(all_timeout_paths)*100:.0f}%)")
        
        # Max adverse excursion
        max_adverse = [p['max_drawdown_bps'] for p in all_timeout_paths]
        print(f"\n    Max adverse excursion (worst point during trade, bps):")
        for pct in [10, 25, 50, 75, 90, 95, 99]:
            print(f"      P{pct:2d}: {np.percentile(max_adverse, pct):+.1f} bps")
    
    # ======================================================================
    # PART 2: TRAILING STOP SWEEP
    # ======================================================================
    print("\n" + "#" * 100)
    print("  PART 2: TRAILING STOP WIDTH SWEEP")
    print("  Testing various trail widths with TP=0.12% and 60min timeout")
    print("#" * 100)
    
    # Combine all cascades and bars for aggregate testing
    # But also test per-symbol
    
    trail_widths = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    
    # First: baseline
    print(f"\n  {'Config':<55s}  {'n':>5s}  {'WR':>6s}  {'avg':>8s}  {'total':>8s}  {'Sharpe':>7s}  {'DD':>6s}  {'TO':>4s}  {'TS':>4s}")
    print(f"  {'─'*55}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*4}  {'─'*4}")
    
    all_baseline = []
    for symbol in SYMBOLS:
        d = all_data[symbol]
        trades = run_strategy_trailing(d['cascades'], d['bars'],
                                       tp_pct=0.12, trail_width_bps=None,
                                       max_hold_min=60)
        all_baseline.extend(trades)
    
    s = stats(all_baseline)
    eb = exit_breakdown(all_baseline)
    to_count = eb.get('timeout', {}).get('count', 0)
    ts_count = eb.get('trailing_stop', {}).get('count', 0)
    print(f"  {'BASELINE: TP=12bps, no trail, 60min TO':<55s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to_count:>4d}  {ts_count:>4d}")
    
    # Sweep: TP + trailing stop (activated from entry)
    print(f"\n  --- TP=12bps + Trailing Stop (active from entry) ---")
    for tw in trail_widths:
        all_trades = []
        for symbol in SYMBOLS:
            d = all_data[symbol]
            trades = run_strategy_trailing(d['cascades'], d['bars'],
                                           tp_pct=0.12, trail_width_bps=tw,
                                           trail_activate_bps=0, max_hold_min=60)
            all_trades.extend(trades)
        s = stats(all_trades)
        eb = exit_breakdown(all_trades)
        to_count = eb.get('timeout', {}).get('count', 0)
        ts_count = eb.get('trailing_stop', {}).get('count', 0)
        print(f"  {'TP=12bps + Trail=' + str(tw) + 'bps (from entry)':<55s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to_count:>4d}  {ts_count:>4d}")
    
    # Sweep: TP + trailing stop (activated after 5 bps profit)
    print(f"\n  --- TP=12bps + Trailing Stop (active after 5 bps profit) ---")
    for tw in trail_widths:
        all_trades = []
        for symbol in SYMBOLS:
            d = all_data[symbol]
            trades = run_strategy_trailing(d['cascades'], d['bars'],
                                           tp_pct=0.12, trail_width_bps=tw,
                                           trail_activate_bps=5, max_hold_min=60)
            all_trades.extend(trades)
        s = stats(all_trades)
        eb = exit_breakdown(all_trades)
        to_count = eb.get('timeout', {}).get('count', 0)
        ts_count = eb.get('trailing_stop', {}).get('count', 0)
        print(f"  {'TP=12bps + Trail=' + str(tw) + 'bps (after 5bps)':<55s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to_count:>4d}  {ts_count:>4d}")
    
    # Sweep: TP + trailing stop (activated after 8 bps profit)
    print(f"\n  --- TP=12bps + Trailing Stop (active after 8 bps profit) ---")
    for tw in trail_widths:
        all_trades = []
        for symbol in SYMBOLS:
            d = all_data[symbol]
            trades = run_strategy_trailing(d['cascades'], d['bars'],
                                           tp_pct=0.12, trail_width_bps=tw,
                                           trail_activate_bps=8, max_hold_min=60)
            all_trades.extend(trades)
        s = stats(all_trades)
        eb = exit_breakdown(all_trades)
        to_count = eb.get('timeout', {}).get('count', 0)
        ts_count = eb.get('trailing_stop', {}).get('count', 0)
        print(f"  {'TP=12bps + Trail=' + str(tw) + 'bps (after 8bps)':<55s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to_count:>4d}  {ts_count:>4d}")
    
    # Sweep: Trailing stop ONLY (no fixed TP) — let winners run
    print(f"\n  --- Trail ONLY (no fixed TP), 60min timeout ---")
    for tw in trail_widths:
        all_trades = []
        for symbol in SYMBOLS:
            d = all_data[symbol]
            trades = run_strategy_trailing(d['cascades'], d['bars'],
                                           tp_pct=None, trail_width_bps=tw,
                                           trail_activate_bps=0, max_hold_min=60)
            all_trades.extend(trades)
        s = stats(all_trades)
        eb = exit_breakdown(all_trades)
        to_count = eb.get('timeout', {}).get('count', 0)
        ts_count = eb.get('trailing_stop', {}).get('count', 0)
        print(f"  {'Trail=' + str(tw) + 'bps ONLY (no TP)':<55s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to_count:>4d}  {ts_count:>4d}")
    
    # Sweep: Fixed SL (for comparison)
    print(f"\n  --- Fixed SL + TP=12bps, 60min timeout (comparison) ---")
    for sl in [0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]:
        all_trades = []
        for symbol in SYMBOLS:
            d = all_data[symbol]
            trades = run_strategy_trailing(d['cascades'], d['bars'],
                                           tp_pct=0.12, trail_width_bps=None,
                                           sl_pct=sl, max_hold_min=60)
            all_trades.extend(trades)
        s = stats(all_trades)
        eb = exit_breakdown(all_trades)
        to_count = eb.get('timeout', {}).get('count', 0)
        sl_count = eb.get('stop_loss', {}).get('count', 0)
        print(f"  {'TP=12bps + SL=' + f'{sl:.2f}%':<55s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to_count:>4d}  {sl_count:>4d}")
    
    # Sweep: Trailing stop + shorter timeouts
    print(f"\n  --- Best trail configs + shorter timeouts ---")
    for timeout in [15, 30, 45, 60]:
        for tw in [8, 10, 12, 15, 20]:
            all_trades = []
            for symbol in SYMBOLS:
                d = all_data[symbol]
                trades = run_strategy_trailing(d['cascades'], d['bars'],
                                               tp_pct=0.12, trail_width_bps=tw,
                                               trail_activate_bps=5, max_hold_min=timeout)
                all_trades.extend(trades)
            s = stats(all_trades)
            eb = exit_breakdown(all_trades)
            to_count = eb.get('timeout', {}).get('count', 0)
            ts_count = eb.get('trailing_stop', {}).get('count', 0)
            print(f"  {'TP=12 + Trail=' + str(tw) + 'bps(5act) + TO=' + str(timeout) + 'min':<55s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to_count:>4d}  {ts_count:>4d}")
    
    # ======================================================================
    # PART 3: BEST CONFIG — PER-SYMBOL BREAKDOWN
    # ======================================================================
    print("\n" + "#" * 100)
    print("  PART 3: BEST CONFIGS — PER-SYMBOL BREAKDOWN")
    print("#" * 100)
    
    # We'll pick the top configs from Part 2 and show per-symbol detail
    best_configs = [
        ("BASELINE: TP=12bps, no trail, 60min", dict(tp_pct=0.12, trail_width_bps=None, trail_activate_bps=0, max_hold_min=60)),
        ("TP=12 + Trail=10bps(from entry)", dict(tp_pct=0.12, trail_width_bps=10, trail_activate_bps=0, max_hold_min=60)),
        ("TP=12 + Trail=15bps(from entry)", dict(tp_pct=0.12, trail_width_bps=15, trail_activate_bps=0, max_hold_min=60)),
        ("TP=12 + Trail=10bps(after 5bps)", dict(tp_pct=0.12, trail_width_bps=10, trail_activate_bps=5, max_hold_min=60)),
        ("TP=12 + Trail=15bps(after 5bps)", dict(tp_pct=0.12, trail_width_bps=15, trail_activate_bps=5, max_hold_min=60)),
        ("TP=12 + Trail=12bps(after 8bps)", dict(tp_pct=0.12, trail_width_bps=12, trail_activate_bps=8, max_hold_min=60)),
        ("Trail=12bps ONLY (no TP)", dict(tp_pct=None, trail_width_bps=12, trail_activate_bps=0, max_hold_min=60)),
        ("Trail=15bps ONLY (no TP)", dict(tp_pct=None, trail_width_bps=15, trail_activate_bps=0, max_hold_min=60)),
    ]
    
    for config_name, config_params in best_configs:
        print(f"\n  ── {config_name} ──")
        combined = []
        for symbol in SYMBOLS:
            d = all_data[symbol]
            trades = run_strategy_trailing(d['cascades'], d['bars'], **config_params)
            combined.extend(trades)
            s = stats(trades)
            eb = exit_breakdown(trades)
            to_count = eb.get('timeout', {}).get('count', 0)
            ts_count = eb.get('trailing_stop', {}).get('count', 0)
            tp_count = eb.get('take_profit', {}).get('count', 0)
            sl_count = eb.get('stop_loss', {}).get('count', 0)
            print(f"    {symbol:<12s}  n={s['n']:>5d}  WR={s['wr']:5.1f}%  tot={s['total']:+7.2f}%  sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%  "
                  f"TP={tp_count:>4d}  TS={ts_count:>4d}  TO={to_count:>4d}  SL={sl_count:>4d}")
            
            # Show timeout PnL stats if any
            if to_count > 0:
                to_pnls = eb['timeout']['pnl_list']
                print(f"               Timeout PnL: avg={np.mean(to_pnls):+.4f}%  worst={min(to_pnls):+.4f}%  best={max(to_pnls):+.4f}%")
            if ts_count > 0:
                ts_pnls = eb['trailing_stop']['pnl_list']
                print(f"               Trail  PnL: avg={np.mean(ts_pnls):+.4f}%  worst={min(ts_pnls):+.4f}%  best={max(ts_pnls):+.4f}%")
        
        # Combined
        s = stats(combined)
        eb = exit_breakdown(combined)
        to_count = eb.get('timeout', {}).get('count', 0)
        ts_count = eb.get('trailing_stop', {}).get('count', 0)
        tp_count = eb.get('take_profit', {}).get('count', 0)
        print(f"    {'COMBINED':<12s}  n={s['n']:>5d}  WR={s['wr']:5.1f}%  tot={s['total']:+7.2f}%  sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%  "
              f"TP={tp_count:>4d}  TS={ts_count:>4d}  TO={to_count:>4d}")
    
    # ======================================================================
    # PART 4: WORST-CASE COMPARISON
    # ======================================================================
    print("\n" + "#" * 100)
    print("  PART 4: WORST-CASE COMPARISON — Does trailing stop reduce tail risk?")
    print("#" * 100)
    
    for config_name, config_params in best_configs[:4]:
        all_trades = []
        for symbol in SYMBOLS:
            d = all_data[symbol]
            trades = run_strategy_trailing(d['cascades'], d['bars'], **config_params)
            all_trades.extend(trades)
        
        pnls = np.array([t['net_pnl'] for t in all_trades])
        print(f"\n  ── {config_name} ──")
        print(f"    Worst single trade:     {pnls.min():+.4f}%")
        print(f"    Worst 5 trades avg:     {np.sort(pnls)[:5].mean():+.4f}%")
        print(f"    Worst 10 trades avg:    {np.sort(pnls)[:10].mean():+.4f}%")
        print(f"    P1 PnL:                 {np.percentile(pnls, 1):+.4f}%")
        print(f"    P5 PnL:                 {np.percentile(pnls, 5):+.4f}%")
        
        # Max consecutive losses
        losses = pnls < 0
        max_consec = 0
        current = 0
        for l in losses:
            if l:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        print(f"    Max consecutive losses: {max_consec}")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 100}")
    print(f"  TRAILING STOP RESEARCH COMPLETE — {elapsed:.0f}s")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
