#!/usr/bin/env python3
"""
PARTIAL EXITS RESEARCH for liquidation cascade strategy.

Purpose: Test whether splitting the position into parts with different exit
logic can improve overall performance. All exits use MAKER (limit) orders
except the safety timeout (taker, last resort).

Variants tested:
  BASELINE 1: Current best — 100% trail at 5 bps (taker exit)
  BASELINE 2: 100% fixed TP at 12 bps (maker exit) + 60min timeout

  A. Partial TP (maker) + Trailing Limit (maker) on remainder
     - X% exits at fixed TP limit (maker)
     - (100-X)% runs trailing limit that gets cancel+replaced (maker)
     - Timeout on remainder = taker (safety only)

  B. Two Fixed TP Levels (both maker)
     - X% at TP1 (maker), (100-X)% at TP2 (maker)
     - Timeout on unfilled TP2 = taker

  C. Partial TP (maker) + Tighter Trail (maker) after first exit
     - Phase 1: trail at W bps on full position + TP limit on X%
     - Phase 2 (after TP fills): tighten trail to W2 bps on remainder
     - All exits maker except timeout

  D. Progressive Scale-Out at milestones (all maker)
     - Trail on full position
     - At milestone profit levels, place limit to close portions
     - Limits placed slightly below peak → fill on micro-pullbacks (maker)

  E. Partial Exit on Losers (maker) + Trail on remainder
     - If price goes against by threshold, place limit to close X% (maker)
     - Remainder keeps trailing
     - Timeout = taker (safety)

Fee model:
  - Entry: always maker (0.02%)
  - TP / trailing limit / partial exits: maker (0.02%)
  - Timeout (safety): taker (0.055%)
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE = 0.02    # percent
TAKER_FEE = 0.055   # percent
SYMBOLS = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']


# ============================================================================
# DATA LOADING (reused from trailing stop research)
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
    csv_path = symbol_dir / "ticker_prices.csv.gz"
    if csv_path.exists():
        print(f"  Loading ticker CSV (WS ~100ms)...", end='', flush=True)
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df['price'] = df['price'].astype(float)
        df = df[['timestamp', 'price']].sort_values('timestamp').reset_index(drop=True)
        print(f" done ({len(df):,})")
        return df
    ticker_files = sorted(symbol_dir.glob("ticker_*.jsonl.gz"))
    print(f"  Loading {len(ticker_files)} REST ticker files...", end='', flush=True)
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
# CORE: BAR-BY-BAR SIMULATION WITH PARTIAL EXITS
# ============================================================================

def find_fill(direction, limit_price, bar_high, bar_low, idx, end_bar):
    """Find the bar where the limit order fills."""
    for j in range(idx, end_bar + 1):
        if direction == 'long' and bar_low[j] <= limit_price:
            return j
        elif direction == 'short' and bar_high[j] >= limit_price:
            return j
    return None


def run_partial_exits(cascades, price_bars, variant, params,
                      entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10):
    """
    Unified simulation engine for all partial exit variants.
    
    Each trade tracks multiple "legs" (portions of the position) that can
    exit at different times and prices. The weighted PnL is the final result.
    
    params dict varies by variant — see each variant section below.
    
    All TP and trailing limit exits are MAKER. Only timeout is TAKER.
    """
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values
    bar_close = price_bars['close'].values
    n_bars = len(bar_close)
    
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
        else:
            limit_price = current_price * (1 + entry_offset_pct / 100)
        
        # Find fill
        end_bar = min(idx + max_hold_min, n_bars - 1)
        fill_bar = find_fill(direction, limit_price, bar_high, bar_low, idx, end_bar)
        if fill_bar is None:
            continue
        
        remaining = max_hold_min - (fill_bar - idx)
        exit_end = min(fill_bar + remaining, n_bars - 1)
        
        # ── Dispatch to variant-specific logic ──
        if variant == 'baseline_trail':
            legs = _sim_baseline_trail(direction, limit_price, fill_bar, exit_end,
                                       bar_high, bar_low, bar_close, params)
        elif variant == 'baseline_tp':
            legs = _sim_baseline_tp(direction, limit_price, fill_bar, exit_end,
                                     bar_high, bar_low, bar_close, params)
        elif variant == 'A':
            legs = _sim_variant_A(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        elif variant == 'B':
            legs = _sim_variant_B(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        elif variant == 'C':
            legs = _sim_variant_C(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        elif variant == 'D':
            legs = _sim_variant_D(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        elif variant == 'E':
            legs = _sim_variant_E(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Compute weighted PnL across all legs
        total_net_pnl = 0.0
        exit_reasons = []
        for leg in legs:
            weight = leg['weight']
            ep = leg['exit_price']
            reason = leg['exit_reason']
            
            if direction == 'long':
                raw_pnl = (ep - limit_price) / limit_price * 100
            else:
                raw_pnl = (limit_price - ep) / limit_price * 100
            
            # Fee: maker entry always, maker exit for TP/trail_limit, taker for timeout
            entry_fee = MAKER_FEE
            if reason in ('take_profit', 'trail_limit', 'partial_tp', 'milestone'):
                exit_fee = MAKER_FEE
            else:  # timeout
                exit_fee = TAKER_FEE
            
            net = (raw_pnl - entry_fee - exit_fee) * weight
            total_net_pnl += net
            exit_reasons.append(reason)
        
        # Determine primary exit reason for reporting
        primary_reason = exit_reasons[0] if len(exit_reasons) == 1 else '+'.join(sorted(set(exit_reasons)))
        
        trades.append({
            'net_pnl': total_net_pnl,
            'exit_reason': primary_reason,
            'time': cascade['end'],
            'direction': direction,
            'legs': legs,
        })
        last_trade_time = cascade['end']
    
    return trades


# ============================================================================
# BASELINE: 100% trailing stop (taker exit) — for comparison
# ============================================================================

def _sim_baseline_trail(direction, fill_price, fill_bar, exit_end,
                        bar_high, bar_low, bar_close, params):
    trail_bps = params['trail_bps']
    peak = fill_price
    
    for k in range(fill_bar, exit_end + 1):
        if direction == 'long':
            peak = max(peak, bar_high[k])
            trail_level = peak * (1 - trail_bps / 10000)
            if bar_low[k] <= trail_level:
                return [{'weight': 1.0, 'exit_price': max(trail_level, bar_low[k]),
                         'exit_reason': 'trail_limit'}]
        else:
            peak = min(peak, bar_low[k])
            trail_level = peak * (1 + trail_bps / 10000)
            if bar_high[k] >= trail_level:
                return [{'weight': 1.0, 'exit_price': min(trail_level, bar_high[k]),
                         'exit_reason': 'trail_limit'}]
    
    return [{'weight': 1.0, 'exit_price': bar_close[exit_end], 'exit_reason': 'timeout'}]


# ============================================================================
# BASELINE: 100% fixed TP (maker exit) + timeout
# ============================================================================

def _sim_baseline_tp(direction, fill_price, fill_bar, exit_end,
                     bar_high, bar_low, bar_close, params):
    tp_bps = params['tp_bps']
    
    if direction == 'long':
        tp_price = fill_price * (1 + tp_bps / 10000)
    else:
        tp_price = fill_price * (1 - tp_bps / 10000)
    
    for k in range(fill_bar, exit_end + 1):
        if direction == 'long' and bar_high[k] >= tp_price:
            return [{'weight': 1.0, 'exit_price': tp_price, 'exit_reason': 'take_profit'}]
        elif direction == 'short' and bar_low[k] <= tp_price:
            return [{'weight': 1.0, 'exit_price': tp_price, 'exit_reason': 'take_profit'}]
    
    return [{'weight': 1.0, 'exit_price': bar_close[exit_end], 'exit_reason': 'timeout'}]


# ============================================================================
# VARIANT A: Partial TP (maker) + Trailing Limit (maker) on remainder
# ============================================================================

def _sim_variant_A(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    """
    Leg 1: tp_pct of position exits at fixed TP (maker limit)
    Leg 2: (1-tp_pct) of position runs trailing limit (maker)
    Safety: timeout on any unfilled portion (taker)
    """
    tp_frac = params['tp_frac']         # e.g. 0.5 = 50%
    tp_bps = params['tp_bps']           # fixed TP level in bps
    trail_bps = params['trail_bps']     # trail width for remainder
    
    if direction == 'long':
        tp_price = fill_price * (1 + tp_bps / 10000)
    else:
        tp_price = fill_price * (1 - tp_bps / 10000)
    
    legs = []
    tp_filled = False
    peak = fill_price
    
    for k in range(fill_bar, exit_end + 1):
        # Update peak for trailing
        if direction == 'long':
            peak = max(peak, bar_high[k])
        else:
            peak = min(peak, bar_low[k])
        
        # Check TP on first leg (if not yet filled)
        if not tp_filled:
            if direction == 'long' and bar_high[k] >= tp_price:
                legs.append({'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'})
                tp_filled = True
            elif direction == 'short' and bar_low[k] <= tp_price:
                legs.append({'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'})
                tp_filled = True
        
        # Check trailing limit on second leg
        if direction == 'long':
            trail_level = peak * (1 - trail_bps / 10000)
            if bar_low[k] <= trail_level:
                legs.append({'weight': 1.0 - tp_frac, 'exit_price': max(trail_level, bar_low[k]),
                             'exit_reason': 'trail_limit'})
                # If TP leg hasn't filled yet, it times out
                if not tp_filled:
                    legs.append({'weight': tp_frac, 'exit_price': max(trail_level, bar_low[k]),
                                 'exit_reason': 'trail_limit'})
                return legs
        else:
            trail_level = peak * (1 + trail_bps / 10000)
            if bar_high[k] >= trail_level:
                legs.append({'weight': 1.0 - tp_frac, 'exit_price': min(trail_level, bar_high[k]),
                             'exit_reason': 'trail_limit'})
                if not tp_filled:
                    legs.append({'weight': tp_frac, 'exit_price': min(trail_level, bar_high[k]),
                                 'exit_reason': 'trail_limit'})
                return legs
    
    # Timeout — everything remaining exits at market (taker)
    timeout_price = bar_close[exit_end]
    if not tp_filled:
        legs.append({'weight': tp_frac, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    # Trail leg didn't exit either
    legs.append({'weight': 1.0 - tp_frac, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    return legs


# ============================================================================
# VARIANT B: Two Fixed TP Levels (both maker)
# ============================================================================

def _sim_variant_B(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    """
    Leg 1: frac1 of position at TP1 (maker)
    Leg 2: (1-frac1) of position at TP2 (maker)
    Timeout on unfilled legs (taker)
    """
    frac1 = params['frac1']
    tp1_bps = params['tp1_bps']
    tp2_bps = params['tp2_bps']
    
    if direction == 'long':
        tp1_price = fill_price * (1 + tp1_bps / 10000)
        tp2_price = fill_price * (1 + tp2_bps / 10000)
    else:
        tp1_price = fill_price * (1 - tp1_bps / 10000)
        tp2_price = fill_price * (1 - tp2_bps / 10000)
    
    legs = []
    tp1_filled = False
    tp2_filled = False
    
    for k in range(fill_bar, exit_end + 1):
        if not tp1_filled:
            if direction == 'long' and bar_high[k] >= tp1_price:
                legs.append({'weight': frac1, 'exit_price': tp1_price, 'exit_reason': 'partial_tp'})
                tp1_filled = True
            elif direction == 'short' and bar_low[k] <= tp1_price:
                legs.append({'weight': frac1, 'exit_price': tp1_price, 'exit_reason': 'partial_tp'})
                tp1_filled = True
        
        if not tp2_filled:
            if direction == 'long' and bar_high[k] >= tp2_price:
                legs.append({'weight': 1.0 - frac1, 'exit_price': tp2_price, 'exit_reason': 'partial_tp'})
                tp2_filled = True
            elif direction == 'short' and bar_low[k] <= tp2_price:
                legs.append({'weight': 1.0 - frac1, 'exit_price': tp2_price, 'exit_reason': 'partial_tp'})
                tp2_filled = True
        
        if tp1_filled and tp2_filled:
            return legs
    
    # Timeout unfilled legs
    timeout_price = bar_close[exit_end]
    if not tp1_filled:
        legs.append({'weight': frac1, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    if not tp2_filled:
        legs.append({'weight': 1.0 - frac1, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    return legs


# ============================================================================
# VARIANT C: Partial TP (maker) + Tighter Trail (maker) after first exit
# ============================================================================

def _sim_variant_C(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    """
    Phase 1: Full position has trail at trail_bps + TP limit on tp_frac
    Phase 2 (after TP fills): Tighten trail to trail_tight_bps on remainder
    
    Key: if trail fires before TP, entire position exits at trail (maker).
    If TP fires first, remainder switches to tighter trail.
    """
    tp_frac = params['tp_frac']
    tp_bps = params['tp_bps']
    trail_bps = params['trail_bps']           # Phase 1 trail
    trail_tight_bps = params['trail_tight_bps']  # Phase 2 trail (after TP)
    
    if direction == 'long':
        tp_price = fill_price * (1 + tp_bps / 10000)
    else:
        tp_price = fill_price * (1 - tp_bps / 10000)
    
    peak = fill_price
    tp_filled = False
    current_trail_bps = trail_bps
    
    for k in range(fill_bar, exit_end + 1):
        # Update peak
        if direction == 'long':
            peak = max(peak, bar_high[k])
        else:
            peak = min(peak, bar_low[k])
        
        # Check TP on first leg
        if not tp_filled:
            if direction == 'long' and bar_high[k] >= tp_price:
                tp_filled = True
                current_trail_bps = trail_tight_bps
                # Reset peak to current bar for tighter trail
                if direction == 'long':
                    peak = bar_high[k]
                else:
                    peak = bar_low[k]
                # Don't check trail on same bar as TP — give it a chance
                continue
            elif direction == 'short' and bar_low[k] <= tp_price:
                tp_filled = True
                current_trail_bps = trail_tight_bps
                if direction == 'short':
                    peak = bar_low[k]
                else:
                    peak = bar_high[k]
                continue
        
        # Check trail
        if direction == 'long':
            trail_level = peak * (1 - current_trail_bps / 10000)
            if bar_low[k] <= trail_level:
                trail_exit = max(trail_level, bar_low[k])
                if tp_filled:
                    return [
                        {'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'},
                        {'weight': 1.0 - tp_frac, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'},
                    ]
                else:
                    # Trail fires before TP — entire position exits at trail
                    return [{'weight': 1.0, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'}]
        else:
            trail_level = peak * (1 + current_trail_bps / 10000)
            if bar_high[k] >= trail_level:
                trail_exit = min(trail_level, bar_high[k])
                if tp_filled:
                    return [
                        {'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'},
                        {'weight': 1.0 - tp_frac, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'},
                    ]
                else:
                    return [{'weight': 1.0, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'}]
    
    # Timeout
    timeout_price = bar_close[exit_end]
    legs = []
    if tp_filled:
        legs.append({'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'})
        legs.append({'weight': 1.0 - tp_frac, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    else:
        legs.append({'weight': 1.0, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    return legs


# ============================================================================
# VARIANT D: Progressive Scale-Out at milestones (all maker)
# ============================================================================

def _sim_variant_D(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    """
    Trail on full position. At profit milestones, place limit to close portions.
    Limits placed at (milestone - buffer) bps → fill on micro-pullback (maker).
    
    milestones: list of (profit_bps_trigger, exit_bps, fraction_of_remaining)
    trail_bps: trailing stop on whatever remains
    """
    milestones = params['milestones']  # [(trigger_bps, exit_bps, frac_of_remaining), ...]
    trail_bps = params['trail_bps']
    
    peak = fill_price
    legs = []
    remaining_weight = 1.0
    milestone_idx = 0
    pending_limits = []  # (limit_price, weight)
    
    for k in range(fill_bar, exit_end + 1):
        # Update peak
        if direction == 'long':
            peak = max(peak, bar_high[k])
            profit_bps = (peak - fill_price) / fill_price * 10000
        else:
            peak = min(peak, bar_low[k])
            profit_bps = (fill_price - peak) / fill_price * 10000
        
        # Check if new milestones triggered
        while milestone_idx < len(milestones) and profit_bps >= milestones[milestone_idx][0]:
            trigger, exit_bps, frac = milestones[milestone_idx]
            portion = remaining_weight * frac
            if direction == 'long':
                lim_price = fill_price * (1 + exit_bps / 10000)
            else:
                lim_price = fill_price * (1 - exit_bps / 10000)
            pending_limits.append((lim_price, portion))
            remaining_weight -= portion
            milestone_idx += 1
        
        # Check pending limit fills
        filled_limits = []
        for i, (lim_price, weight) in enumerate(pending_limits):
            if direction == 'long' and bar_low[k] <= lim_price:
                legs.append({'weight': weight, 'exit_price': lim_price, 'exit_reason': 'milestone'})
                filled_limits.append(i)
            elif direction == 'short' and bar_high[k] >= lim_price:
                legs.append({'weight': weight, 'exit_price': lim_price, 'exit_reason': 'milestone'})
                filled_limits.append(i)
        for i in sorted(filled_limits, reverse=True):
            pending_limits.pop(i)
        
        # Check trail on remaining
        if remaining_weight > 0.001:  # avoid float precision issues
            if direction == 'long':
                trail_level = peak * (1 - trail_bps / 10000)
                if bar_low[k] <= trail_level:
                    trail_exit = max(trail_level, bar_low[k])
                    # Trail fires — close all remaining + pending limits at trail
                    for lim_price, weight in pending_limits:
                        legs.append({'weight': weight, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    legs.append({'weight': remaining_weight, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    return legs
            else:
                trail_level = peak * (1 + trail_bps / 10000)
                if bar_high[k] >= trail_level:
                    trail_exit = min(trail_level, bar_high[k])
                    for lim_price, weight in pending_limits:
                        legs.append({'weight': weight, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    legs.append({'weight': remaining_weight, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    return legs
        elif not pending_limits:
            # All milestones filled, no remaining
            return legs
    
    # Timeout — close everything remaining
    timeout_price = bar_close[exit_end]
    for lim_price, weight in pending_limits:
        legs.append({'weight': weight, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    if remaining_weight > 0.001:
        legs.append({'weight': remaining_weight, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    return legs


# ============================================================================
# VARIANT E: Partial Exit on Losers (maker) + Trail on remainder
# ============================================================================

def _sim_variant_E(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    """
    Normal trailing stop on full position.
    If price goes against us by loss_threshold_bps, place limit to close
    cut_frac of position at (loss_threshold - buffer) bps loss (maker).
    Remainder keeps trailing.
    """
    trail_bps = params['trail_bps']
    loss_threshold_bps = params['loss_threshold_bps']  # e.g. 10 bps against
    cut_frac = params['cut_frac']                       # e.g. 0.5
    buffer_bps = params.get('buffer_bps', 2)            # limit placed 2 bps better than worst
    
    peak = fill_price
    loss_cut_triggered = False
    loss_limit_price = None
    loss_limit_filled = False
    remaining_weight = 1.0
    legs = []
    
    for k in range(fill_bar, exit_end + 1):
        # Update peak
        if direction == 'long':
            peak = max(peak, bar_high[k])
            current_loss_bps = (fill_price - bar_low[k]) / fill_price * 10000
        else:
            peak = min(peak, bar_low[k])
            current_loss_bps = (bar_high[k] - fill_price) / fill_price * 10000
        
        # Check if loss threshold breached → place limit to cut
        if not loss_cut_triggered and current_loss_bps >= loss_threshold_bps:
            loss_cut_triggered = True
            # Place limit slightly better than current worst (maker)
            cut_bps = loss_threshold_bps - buffer_bps
            if direction == 'long':
                loss_limit_price = fill_price * (1 - cut_bps / 10000)
            else:
                loss_limit_price = fill_price * (1 + cut_bps / 10000)
        
        # Check if loss limit fills
        if loss_cut_triggered and not loss_limit_filled and loss_limit_price is not None:
            if direction == 'long' and bar_high[k] >= loss_limit_price:
                legs.append({'weight': cut_frac, 'exit_price': loss_limit_price, 'exit_reason': 'partial_tp'})
                loss_limit_filled = True
                remaining_weight -= cut_frac
            elif direction == 'short' and bar_low[k] <= loss_limit_price:
                legs.append({'weight': cut_frac, 'exit_price': loss_limit_price, 'exit_reason': 'partial_tp'})
                loss_limit_filled = True
                remaining_weight -= cut_frac
        
        # Check trail on remaining
        if remaining_weight > 0.001:
            if direction == 'long':
                trail_level = peak * (1 - trail_bps / 10000)
                if bar_low[k] <= trail_level:
                    trail_exit = max(trail_level, bar_low[k])
                    legs.append({'weight': remaining_weight, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    if loss_cut_triggered and not loss_limit_filled:
                        legs.append({'weight': cut_frac, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    return legs
            else:
                trail_level = peak * (1 + trail_bps / 10000)
                if bar_high[k] >= trail_level:
                    trail_exit = min(trail_level, bar_high[k])
                    legs.append({'weight': remaining_weight, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    if loss_cut_triggered and not loss_limit_filled:
                        legs.append({'weight': cut_frac, 'exit_price': trail_exit, 'exit_reason': 'trail_limit'})
                    return legs
    
    # Timeout
    timeout_price = bar_close[exit_end]
    if loss_cut_triggered and not loss_limit_filled:
        legs.append({'weight': cut_frac, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    if remaining_weight > 0.001:
        legs.append({'weight': remaining_weight, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    return legs


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


def pline(label, s, width=60):
    if s['n'] == 0:
        print(f"    {label:{width}s}  n=    0  (no trades)")
        return
    print(f"    {label:{width}s}  n={s['n']:>5d}  WR={s['wr']:5.1f}%  "
          f"avg={s['avg']:+.4f}%  tot={s['total']:+8.2f}%  sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%")


def count_timeouts(trades):
    n_to = sum(1 for t in trades if 'timeout' in t['exit_reason'])
    return n_to


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    
    print("=" * 100)
    print("  PARTIAL EXITS RESEARCH — Liquidation Cascade Strategy")
    print("  All exits are MAKER (limit) except safety timeout (taker)")
    print("  Baseline: off=0.15%, 60min timeout, disp≥10bps, min_ev=1")
    print("=" * 100)
    
    # ── Load data for all symbols ──
    all_data = {}
    for sym in SYMBOLS:
        print(f"\n{'─' * 80}")
        print(f"  Loading {sym}...")
        print(f"{'─' * 80}")
        liq_df = load_liquidations(sym)
        ticker_df = load_ticker(sym)
        bars = build_bars(ticker_df)
        cascades = detect_signals(liq_df, bars)
        print(f"  Signals: {len(cascades)}")
        all_data[sym] = {'bars': bars, 'cascades': cascades}
    
    # ── Helper: run config across all symbols, return combined trades ──
    def run_all(variant, params):
        all_trades = []
        for sym in SYMBOLS:
            trades = run_partial_exits(all_data[sym]['cascades'], all_data[sym]['bars'],
                                       variant, params)
            all_trades.extend(trades)
        return all_trades
    
    def run_per_sym(variant, params):
        result = {}
        for sym in SYMBOLS:
            trades = run_partial_exits(all_data[sym]['cascades'], all_data[sym]['bars'],
                                       variant, params)
            result[sym] = trades
        return result
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 1: BASELINES")
    print(f"{'#' * 100}\n")
    
    # Baseline 1: Pure trail 5 bps (current best — but taker exit)
    b1 = run_all('baseline_trail', {'trail_bps': 5})
    s1 = stats(b1)
    pline("BASELINE 1: Trail=5bps (taker exit)", s1)
    print(f"      Timeouts: {count_timeouts(b1)}")
    
    # Baseline 1b: Pure trail 3 bps
    b1b = run_all('baseline_trail', {'trail_bps': 3})
    s1b = stats(b1b)
    pline("BASELINE 1b: Trail=3bps (taker exit)", s1b)
    
    # Baseline 2: Fixed TP 12 bps (maker exit)
    b2 = run_all('baseline_tp', {'tp_bps': 12})
    s2 = stats(b2)
    pline("BASELINE 2: TP=12bps (maker exit)", s2)
    print(f"      Timeouts: {count_timeouts(b2)}")
    
    # Baseline 2b: Fixed TP 8 bps (maker exit)
    b2b = run_all('baseline_tp', {'tp_bps': 8})
    s2b = stats(b2b)
    pline("BASELINE 2b: TP=8bps (maker exit)", s2b)
    print(f"      Timeouts: {count_timeouts(b2b)}")
    
    # Baseline 2c: Fixed TP 5 bps (maker exit)
    b2c = run_all('baseline_tp', {'tp_bps': 5})
    s2c = stats(b2c)
    pline("BASELINE 2c: TP=5bps (maker exit)", s2c)
    print(f"      Timeouts: {count_timeouts(b2c)}")
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 2: VARIANT A — Partial TP (maker) + Trail (maker) on remainder")
    print(f"{'#' * 100}\n")
    
    print("  Sweep: tp_frac × tp_bps × trail_bps\n")
    print(f"    {'Config':<60s}  {'n':>5s}  {'WR':>6s}  {'avg':>8s}  {'total':>8s}  {'Sharpe':>7s}  {'DD':>6s}  {'TO':>4s}")
    print(f"    {'─'*60}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*4}")
    
    for tp_frac in [0.3, 0.5, 0.7]:
        for tp_bps in [5, 8, 10, 12]:
            for trail_bps in [3, 5, 8]:
                label = f"A: {int(tp_frac*100)}% TP@{tp_bps}bps + {int((1-tp_frac)*100)}% trail@{trail_bps}bps"
                trades = run_all('A', {'tp_frac': tp_frac, 'tp_bps': tp_bps, 'trail_bps': trail_bps})
                s = stats(trades)
                to = count_timeouts(trades)
                print(f"    {label:<60s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+8.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to:>4d}")
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 3: VARIANT B — Two Fixed TPs (both maker)")
    print(f"{'#' * 100}\n")
    
    print(f"    {'Config':<60s}  {'n':>5s}  {'WR':>6s}  {'avg':>8s}  {'total':>8s}  {'Sharpe':>7s}  {'DD':>6s}  {'TO':>4s}")
    print(f"    {'─'*60}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*4}")
    
    for frac1 in [0.5, 0.7]:
        for tp1 in [5, 8]:
            for tp2 in [12, 15, 20, 30]:
                label = f"B: {int(frac1*100)}% TP@{tp1}bps + {int((1-frac1)*100)}% TP@{tp2}bps"
                trades = run_all('B', {'frac1': frac1, 'tp1_bps': tp1, 'tp2_bps': tp2})
                s = stats(trades)
                to = count_timeouts(trades)
                print(f"    {label:<60s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+8.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to:>4d}")
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 4: VARIANT C — TP (maker) + Tighter Trail (maker) after TP")
    print(f"{'#' * 100}\n")
    
    print(f"    {'Config':<60s}  {'n':>5s}  {'WR':>6s}  {'avg':>8s}  {'total':>8s}  {'Sharpe':>7s}  {'DD':>6s}  {'TO':>4s}")
    print(f"    {'─'*60}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*4}")
    
    for tp_frac in [0.5, 0.7]:
        for tp_bps in [5, 8, 10]:
            for trail_bps in [5, 8]:
                for trail_tight in [3, 5]:
                    if trail_tight >= trail_bps:
                        continue  # tight trail must be tighter
                    label = f"C: {int(tp_frac*100)}% TP@{tp_bps}bps, trail {trail_bps}→{trail_tight}bps"
                    trades = run_all('C', {'tp_frac': tp_frac, 'tp_bps': tp_bps,
                                           'trail_bps': trail_bps, 'trail_tight_bps': trail_tight})
                    s = stats(trades)
                    to = count_timeouts(trades)
                    print(f"    {label:<60s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+8.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to:>4d}")
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 5: VARIANT D — Progressive Scale-Out at milestones (maker)")
    print(f"{'#' * 100}\n")
    
    print(f"    {'Config':<60s}  {'n':>5s}  {'WR':>6s}  {'avg':>8s}  {'total':>8s}  {'Sharpe':>7s}  {'DD':>6s}  {'TO':>4s}")
    print(f"    {'─'*60}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*4}")
    
    # (trigger_bps, exit_bps, frac_of_remaining)
    milestone_configs = [
        ("D: 50%@8bps, trail 5",
         [(8, 6, 0.5)], 5),
        ("D: 50%@8bps, trail 8",
         [(8, 6, 0.5)], 8),
        ("D: 50%@10bps, trail 5",
         [(10, 8, 0.5)], 5),
        ("D: 50%@8bps + 25%@15bps, trail 5",
         [(8, 6, 0.5), (15, 13, 0.5)], 5),
        ("D: 50%@8bps + 25%@15bps, trail 8",
         [(8, 6, 0.5), (15, 13, 0.5)], 8),
        ("D: 50%@8bps + 25%@15bps + 12.5%@30bps, trail 5",
         [(8, 6, 0.5), (15, 13, 0.5), (30, 28, 0.5)], 5),
        ("D: 33%@5bps + 33%@10bps + 33%@15bps, trail 5",
         [(5, 3, 0.333), (10, 8, 0.5), (15, 13, 1.0)], 5),
        ("D: 33%@5bps + 33%@10bps, trail 3",
         [(5, 3, 0.333), (10, 8, 0.5)], 3),
        ("D: 70%@8bps, trail 5",
         [(8, 6, 0.7)], 5),
        ("D: 70%@8bps, trail 3",
         [(8, 6, 0.7)], 3),
    ]
    
    for label, milestones, trail in milestone_configs:
        trades = run_all('D', {'milestones': milestones, 'trail_bps': trail})
        s = stats(trades)
        to = count_timeouts(trades)
        print(f"    {label:<60s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+8.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to:>4d}")
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 6: VARIANT E — Partial Exit on Losers (maker) + Trail")
    print(f"{'#' * 100}\n")
    
    print(f"    {'Config':<60s}  {'n':>5s}  {'WR':>6s}  {'avg':>8s}  {'total':>8s}  {'Sharpe':>7s}  {'DD':>6s}  {'TO':>4s}")
    print(f"    {'─'*60}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*4}")
    
    for trail_bps in [5, 8]:
        for loss_thresh in [8, 10, 15, 20]:
            for cut_frac in [0.3, 0.5, 0.7]:
                label = f"E: trail {trail_bps}bps, cut {int(cut_frac*100)}% at -{loss_thresh}bps loss"
                trades = run_all('E', {'trail_bps': trail_bps, 'loss_threshold_bps': loss_thresh,
                                        'cut_frac': cut_frac, 'buffer_bps': 2})
                s = stats(trades)
                to = count_timeouts(trades)
                print(f"    {label:<60s}  {s['n']:>5d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+8.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to:>4d}")
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 7: TOP 3 CONFIGS — Per-Symbol Breakdown")
    print(f"{'#' * 100}\n")
    
    # Find best from each variant (will be filled after seeing results)
    # For now, show a few promising configs per-symbol
    top_configs = [
        ("BASELINE: Trail 5bps (taker)", 'baseline_trail', {'trail_bps': 5}),
        ("BASELINE: TP 12bps (maker)", 'baseline_tp', {'tp_bps': 12}),
        ("A: 50% TP@8bps + 50% trail@5bps", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 5}),
        ("A: 50% TP@8bps + 50% trail@3bps", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 3}),
        ("A: 70% TP@8bps + 30% trail@5bps", 'A', {'tp_frac': 0.7, 'tp_bps': 8, 'trail_bps': 5}),
        ("C: 50% TP@8bps, trail 5→3bps", 'C', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 5, 'trail_tight_bps': 3}),
        ("D: 50%@8bps, trail 5", 'D', {'milestones': [(8, 6, 0.5)], 'trail_bps': 5}),
    ]
    
    for label, variant, params in top_configs:
        print(f"  ── {label} ──")
        sym_trades = run_per_sym(variant, params)
        for sym in SYMBOLS:
            s = stats(sym_trades[sym])
            to = count_timeouts(sym_trades[sym])
            print(f"    {sym:<12s}  n={s['n']:>5d}  WR={s['wr']:5.1f}%  tot={s['total']:+8.2f}%  "
                  f"sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%  TO={to}")
        # Combined
        all_t = []
        for sym in SYMBOLS:
            all_t.extend(sym_trades[sym])
        sc = stats(all_t)
        toc = count_timeouts(all_t)
        print(f"    {'COMBINED':<12s}  n={sc['n']:>5d}  WR={sc['wr']:5.1f}%  tot={sc['total']:+8.2f}%  "
              f"sh={sc['sharpe']:+6.1f}  dd={sc['maxdd']:5.2f}%  TO={toc}")
        print()
    
    # ================================================================
    print(f"\n{'#' * 100}")
    print(f"  PART 8: WORST-CASE COMPARISON")
    print(f"{'#' * 100}\n")
    
    worst_configs = [
        ("BASELINE: Trail 5bps (taker)", 'baseline_trail', {'trail_bps': 5}),
        ("BASELINE: TP 12bps (maker)", 'baseline_tp', {'tp_bps': 12}),
        ("A: 50% TP@8bps + 50% trail@5bps", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 5}),
        ("A: 50% TP@8bps + 50% trail@3bps", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 3}),
        ("C: 50% TP@8bps, trail 5→3bps", 'C', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 5, 'trail_tight_bps': 3}),
    ]
    
    for label, variant, params in worst_configs:
        trades = run_all(variant, params)
        pnls = np.array([t['net_pnl'] for t in trades])
        pnls_sorted = np.sort(pnls)
        print(f"  ── {label} ──")
        print(f"    Worst single trade:     {pnls_sorted[0]:+.4f}%")
        print(f"    Worst 5 trades avg:     {pnls_sorted[:5].mean():+.4f}%")
        print(f"    Worst 10 trades avg:    {pnls_sorted[:10].mean():+.4f}%")
        print(f"    P1 PnL:                 {np.percentile(pnls, 1):+.4f}%")
        print(f"    P5 PnL:                 {np.percentile(pnls, 5):+.4f}%")
        # Max consecutive losses
        losses = (pnls < 0).astype(int)
        max_consec = 0
        current = 0
        for l in losses:
            if l:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        print(f"    Max consecutive losses: {max_consec}")
        print()
    
    elapsed = time.time() - t0
    print("=" * 100)
    print(f"  PARTIAL EXITS RESEARCH COMPLETE — {elapsed:.0f}s")
    print("=" * 100)


if __name__ == '__main__':
    main()
