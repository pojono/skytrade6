#!/usr/bin/env python3
"""
OUT-OF-SAMPLE TEST — Feb 2026 (11 days: Feb 9-19)

Tests the best configs from in-sample (May-Aug 2025) on completely unseen data.
This is the ultimate validation: parameters were chosen on 2025 data,
now we see if they hold on Feb 2026 data.

Configs tested:
  BASELINES:
    1. Trail 3bps (taker exit)
    2. Trail 5bps (taker exit)
    3. TP 12bps (maker exit)

  PARTIAL EXITS (all maker):
    A1. 70% TP@12bps + 30% trail@3bps  (best Sharpe in-sample)
    A2. 50% TP@8bps + 50% trail@3bps   (good balance)
    A3. 50% TP@8bps + 50% trail@5bps   (simplest, most conservative)
    A4. 30% TP@12bps + 70% trail@3bps  (max return in-sample)
    C1. 50% TP@10bps, trail 5→3bps     (highest return in-sample)
    C2. 50% TP@8bps, trail 5→3bps
    D1. 50%@8bps + 25%@15bps + 12.5%@30bps, trail 5  (ultra-conservative)

P95 threshold: computed from 2025 in-sample data (no peeking at OOS).
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE = 0.02
TAKER_FEE = 0.055
SYMBOLS = ['BTCUSDT', 'DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_liquidations(symbol, data_dir='data', date_prefix=None):
    """Load liquidation data, optionally filtering by date prefix."""
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    
    if date_prefix:
        liq_files = [f for f in liq_files if date_prefix in f.name]
    
    print(f"  Loading {len(liq_files)} liq files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 100 == 0:
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
    if len(df) > 0:
        df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_csv(symbol, csv_name, data_dir='data'):
    """Load preprocessed ticker CSV."""
    csv_path = Path(data_dir) / symbol / csv_name
    if not csv_path.exists():
        print(f"  No {csv_name} found!")
        return pd.DataFrame()
    print(f"  Loading {csv_name}...", end='', flush=True)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
    df['price'] = df['price'].astype(float)
    df = df[['timestamp', 'price']].sort_values('timestamp').reset_index(drop=True)
    print(f" done ({len(df):,})")
    return df


def build_bars(ticker_df, freq='1min'):
    ticker_df = ticker_df.set_index('timestamp')
    bars = ticker_df['price'].resample(freq).ohlc().dropna()
    print(f"  Building bars... {len(bars):,} bars")
    return bars


# ============================================================================
# CASCADE DETECTION
# ============================================================================

def detect_signals(liq_df, price_bars, p95_threshold):
    """Detect cascades using a FIXED P95 threshold (from in-sample)."""
    large = liq_df[liq_df['notional'] >= p95_threshold].copy()
    if len(large) == 0:
        return []
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
# SIMULATION ENGINE (copied from partial exits research)
# ============================================================================

def find_fill(direction, limit_price, bar_high, bar_low, idx, end_bar):
    for j in range(idx, end_bar + 1):
        if direction == 'long' and bar_low[j] <= limit_price:
            return j
        elif direction == 'short' and bar_high[j] >= limit_price:
            return j
    return None


def run_partial_exits(cascades, price_bars, variant, params,
                      entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10):
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
        
        end_bar = min(idx + max_hold_min, n_bars - 1)
        fill_bar = find_fill(direction, limit_price, bar_high, bar_low, idx, end_bar)
        if fill_bar is None:
            continue
        
        remaining = max_hold_min - (fill_bar - idx)
        exit_end = min(fill_bar + remaining, n_bars - 1)
        
        if variant == 'baseline_trail':
            legs = _sim_baseline_trail(direction, limit_price, fill_bar, exit_end,
                                       bar_high, bar_low, bar_close, params)
        elif variant == 'baseline_tp':
            legs = _sim_baseline_tp(direction, limit_price, fill_bar, exit_end,
                                     bar_high, bar_low, bar_close, params)
        elif variant == 'A':
            legs = _sim_variant_A(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        elif variant == 'C':
            legs = _sim_variant_C(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        elif variant == 'D':
            legs = _sim_variant_D(direction, limit_price, fill_bar, exit_end,
                                   bar_high, bar_low, bar_close, params)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
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
            entry_fee = MAKER_FEE
            if reason in ('take_profit', 'trail_limit', 'partial_tp', 'milestone'):
                exit_fee = MAKER_FEE
            else:
                exit_fee = TAKER_FEE
            net = (raw_pnl - entry_fee - exit_fee) * weight
            total_net_pnl += net
            exit_reasons.append(reason)
        
        primary_reason = exit_reasons[0] if len(exit_reasons) == 1 else '+'.join(sorted(set(exit_reasons)))
        trades.append({
            'net_pnl': total_net_pnl,
            'exit_reason': primary_reason,
            'time': cascade['end'],
            'direction': direction,
        })
        last_trade_time = cascade['end']
    
    return trades


# ── Variant implementations (same as partial exits research) ──

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


def _sim_variant_A(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    tp_frac = params['tp_frac']
    tp_bps = params['tp_bps']
    trail_bps = params['trail_bps']
    if direction == 'long':
        tp_price = fill_price * (1 + tp_bps / 10000)
    else:
        tp_price = fill_price * (1 - tp_bps / 10000)
    legs = []
    tp_filled = False
    peak = fill_price
    for k in range(fill_bar, exit_end + 1):
        if direction == 'long':
            peak = max(peak, bar_high[k])
        else:
            peak = min(peak, bar_low[k])
        if not tp_filled:
            if direction == 'long' and bar_high[k] >= tp_price:
                legs.append({'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'})
                tp_filled = True
            elif direction == 'short' and bar_low[k] <= tp_price:
                legs.append({'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'})
                tp_filled = True
        if direction == 'long':
            trail_level = peak * (1 - trail_bps / 10000)
            if bar_low[k] <= trail_level:
                legs.append({'weight': 1.0 - tp_frac, 'exit_price': max(trail_level, bar_low[k]),
                             'exit_reason': 'trail_limit'})
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
    timeout_price = bar_close[exit_end]
    if not tp_filled:
        legs.append({'weight': tp_frac, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    legs.append({'weight': 1.0 - tp_frac, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    return legs


def _sim_variant_C(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    tp_frac = params['tp_frac']
    tp_bps = params['tp_bps']
    trail_bps = params['trail_bps']
    trail_tight_bps = params['trail_tight_bps']
    if direction == 'long':
        tp_price = fill_price * (1 + tp_bps / 10000)
    else:
        tp_price = fill_price * (1 - tp_bps / 10000)
    peak = fill_price
    tp_filled = False
    current_trail_bps = trail_bps
    for k in range(fill_bar, exit_end + 1):
        if direction == 'long':
            peak = max(peak, bar_high[k])
        else:
            peak = min(peak, bar_low[k])
        if not tp_filled:
            if direction == 'long' and bar_high[k] >= tp_price:
                tp_filled = True
                current_trail_bps = trail_tight_bps
                peak = bar_high[k] if direction == 'long' else bar_low[k]
                continue
            elif direction == 'short' and bar_low[k] <= tp_price:
                tp_filled = True
                current_trail_bps = trail_tight_bps
                peak = bar_low[k] if direction == 'short' else bar_high[k]
                continue
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
    timeout_price = bar_close[exit_end]
    legs = []
    if tp_filled:
        legs.append({'weight': tp_frac, 'exit_price': tp_price, 'exit_reason': 'partial_tp'})
        legs.append({'weight': 1.0 - tp_frac, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    else:
        legs.append({'weight': 1.0, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
    return legs


def _sim_variant_D(direction, fill_price, fill_bar, exit_end,
                   bar_high, bar_low, bar_close, params):
    milestones = params['milestones']
    trail_bps = params['trail_bps']
    peak = fill_price
    legs = []
    remaining_weight = 1.0
    milestone_idx = 0
    pending_limits = []
    for k in range(fill_bar, exit_end + 1):
        if direction == 'long':
            peak = max(peak, bar_high[k])
            profit_bps = (peak - fill_price) / fill_price * 10000
        else:
            peak = min(peak, bar_low[k])
            profit_bps = (fill_price - peak) / fill_price * 10000
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
        if remaining_weight > 0.001:
            if direction == 'long':
                trail_level = peak * (1 - trail_bps / 10000)
                if bar_low[k] <= trail_level:
                    trail_exit = max(trail_level, bar_low[k])
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
            return legs
    timeout_price = bar_close[exit_end]
    for lim_price, weight in pending_limits:
        legs.append({'weight': weight, 'exit_price': timeout_price, 'exit_reason': 'timeout'})
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


def count_timeouts(trades):
    return sum(1 for t in trades if 'timeout' in t['exit_reason'])


def pline(label, s, width=55):
    if s['n'] == 0:
        print(f"    {label:{width}s}  n=    0  (no trades)")
        return
    print(f"    {label:{width}s}  n={s['n']:>4d}  WR={s['wr']:5.1f}%  "
          f"avg={s['avg']:+.4f}%  tot={s['total']:+7.2f}%  sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    
    print("=" * 100)
    print("  OUT-OF-SAMPLE TEST — Feb 2026 (11 days)")
    print("  Parameters frozen from in-sample (May-Aug 2025)")
    print("  P95 thresholds computed from 2025 data (no peeking)")
    print("=" * 100)
    
    # ── Step 1: Compute P95 thresholds from in-sample 2025 data ──
    print(f"\n{'#' * 100}")
    print(f"  STEP 1: Compute P95 thresholds from IN-SAMPLE (2025) data")
    print(f"{'#' * 100}\n")
    
    p95_thresholds = {}
    for sym in SYMBOLS:
        print(f"  {sym} (in-sample):")
        liq_is = load_liquidations(sym, date_prefix='2025')
        if len(liq_is) == 0:
            print(f"    No 2025 liquidation data!")
            continue
        p95 = liq_is['notional'].quantile(0.95)
        p95_thresholds[sym] = p95
        print(f"    P95 threshold: ${p95:,.0f}")
        print(f"    Liquidations: {len(liq_is):,}")
        print()
    
    # ── Step 2: Load OOS Feb 2026 data ──
    print(f"\n{'#' * 100}")
    print(f"  STEP 2: Load OUT-OF-SAMPLE (Feb 2026) data")
    print(f"{'#' * 100}\n")
    
    oos_data = {}
    for sym in SYMBOLS:
        print(f"  {sym} (OOS):")
        liq_oos = load_liquidations(sym, date_prefix='2026')
        ticker_oos = load_ticker_csv(sym, 'ticker_prices_feb2026.csv.gz')
        
        if len(liq_oos) == 0 or len(ticker_oos) == 0:
            print(f"    Insufficient data, skipping")
            continue
        
        bars = build_bars(ticker_oos)
        
        # Use frozen P95 from in-sample
        p95 = p95_thresholds.get(sym)
        if p95 is None:
            print(f"    No P95 threshold, skipping")
            continue
        
        cascades = detect_signals(liq_oos, bars, p95)
        print(f"    Signals (using IS P95=${p95:,.0f}): {len(cascades)}")
        
        # Also compute OOS P95 for comparison
        oos_p95 = liq_oos['notional'].quantile(0.95)
        print(f"    OOS P95 would be: ${oos_p95:,.0f} (ratio: {oos_p95/p95:.2f}x)")
        
        t_start = ticker_oos['timestamp'].min()
        t_end = ticker_oos['timestamp'].max()
        days = (t_end - t_start).days
        print(f"    Period: {t_start.strftime('%Y-%m-%d')} to {t_end.strftime('%Y-%m-%d')} ({days} days)")
        print()
        
        oos_data[sym] = {'bars': bars, 'cascades': cascades, 'liq_df': liq_oos}
    
    if not oos_data:
        print("  No OOS data available!")
        return
    
    # ── Helper ──
    def run_all(variant, params):
        all_trades = []
        for sym in SYMBOLS:
            if sym not in oos_data:
                continue
            trades = run_partial_exits(oos_data[sym]['cascades'], oos_data[sym]['bars'],
                                       variant, params)
            all_trades.extend(trades)
        return all_trades
    
    def run_per_sym(variant, params):
        result = {}
        for sym in SYMBOLS:
            if sym not in oos_data:
                result[sym] = []
                continue
            trades = run_partial_exits(oos_data[sym]['cascades'], oos_data[sym]['bars'],
                                       variant, params)
            result[sym] = trades
        return result
    
    # ── Step 3: Run all configs ──
    print(f"\n{'#' * 100}")
    print(f"  STEP 3: OOS RESULTS — All Configs")
    print(f"{'#' * 100}\n")
    
    configs = [
        ("BASELINE: Trail 3bps (taker)", 'baseline_trail', {'trail_bps': 3}),
        ("BASELINE: Trail 5bps (taker)", 'baseline_trail', {'trail_bps': 5}),
        ("BASELINE: TP 12bps (maker)", 'baseline_tp', {'tp_bps': 12}),
        ("A1: 70% TP@12bps + 30% trail@3bps", 'A', {'tp_frac': 0.7, 'tp_bps': 12, 'trail_bps': 3}),
        ("A2: 50% TP@8bps + 50% trail@3bps", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 3}),
        ("A3: 50% TP@8bps + 50% trail@5bps", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 5}),
        ("A4: 30% TP@12bps + 70% trail@3bps", 'A', {'tp_frac': 0.3, 'tp_bps': 12, 'trail_bps': 3}),
        ("C1: 50% TP@10bps, trail 5→3bps", 'C', {'tp_frac': 0.5, 'tp_bps': 10, 'trail_bps': 5, 'trail_tight_bps': 3}),
        ("C2: 50% TP@8bps, trail 5→3bps", 'C', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 5, 'trail_tight_bps': 3}),
        ("D1: 50%@8+25%@15+12.5%@30, trail 5", 'D', {'milestones': [(8, 6, 0.5), (15, 13, 0.5), (30, 28, 0.5)], 'trail_bps': 5}),
    ]
    
    print(f"  {'Config':<50s}  {'n':>4s}  {'WR':>6s}  {'avg':>8s}  {'total':>7s}  {'Sharpe':>7s}  {'DD':>6s}  {'TO':>3s}")
    print(f"  {'─'*50}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*3}")
    
    for label, variant, params in configs:
        trades = run_all(variant, params)
        s = stats(trades)
        to = count_timeouts(trades)
        flag = '✅' if s['total'] > 0 else '❌'
        print(f"  {flag} {label:<48s}  {s['n']:>4d}  {s['wr']:5.1f}%  {s['avg']:+.4f}%  {s['total']:+7.2f}%  {s['sharpe']:+6.1f}  {s['maxdd']:5.2f}%  {to:>3d}")
    
    # ── Step 4: Per-symbol breakdown for top configs ──
    print(f"\n{'#' * 100}")
    print(f"  STEP 4: Per-Symbol Breakdown (OOS)")
    print(f"{'#' * 100}\n")
    
    top_configs = [
        ("BASELINE: Trail 5bps (taker)", 'baseline_trail', {'trail_bps': 5}),
        ("A1: 70% TP@12bps + 30% trail@3bps", 'A', {'tp_frac': 0.7, 'tp_bps': 12, 'trail_bps': 3}),
        ("A2: 50% TP@8bps + 50% trail@3bps", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 3}),
        ("C1: 50% TP@10bps, trail 5→3bps", 'C', {'tp_frac': 0.5, 'tp_bps': 10, 'trail_bps': 5, 'trail_tight_bps': 3}),
    ]
    
    for label, variant, params in top_configs:
        print(f"  ── {label} ──")
        sym_trades = run_per_sym(variant, params)
        for sym in SYMBOLS:
            s = stats(sym_trades[sym])
            to = count_timeouts(sym_trades[sym])
            flag = '✅' if s['total'] > 0 else '❌' if s['n'] > 0 else '⬜'
            print(f"    {flag} {sym:<12s}  n={s['n']:>4d}  WR={s['wr']:5.1f}%  tot={s['total']:+7.2f}%  "
                  f"sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%  TO={to}")
        all_t = []
        for sym in SYMBOLS:
            all_t.extend(sym_trades[sym])
        sc = stats(all_t)
        toc = count_timeouts(all_t)
        flag = '✅' if sc['total'] > 0 else '❌'
        print(f"    {flag} {'COMBINED':<12s}  n={sc['n']:>4d}  WR={sc['wr']:5.1f}%  tot={sc['total']:+7.2f}%  "
              f"sh={sc['sharpe']:+6.1f}  dd={sc['maxdd']:5.2f}%  TO={toc}")
        print()
    
    # ── Step 5: Worst-case analysis ──
    print(f"\n{'#' * 100}")
    print(f"  STEP 5: Worst-Case Analysis (OOS)")
    print(f"{'#' * 100}\n")
    
    for label, variant, params in top_configs:
        trades = run_all(variant, params)
        if not trades:
            print(f"  ── {label}: no trades ──\n")
            continue
        pnls = np.array([t['net_pnl'] for t in trades])
        pnls_sorted = np.sort(pnls)
        print(f"  ── {label} ──")
        print(f"    Worst single trade:     {pnls_sorted[0]:+.4f}%")
        if len(pnls_sorted) >= 5:
            print(f"    Worst 5 trades avg:     {pnls_sorted[:5].mean():+.4f}%")
        if len(pnls_sorted) >= 10:
            print(f"    Worst 10 trades avg:    {pnls_sorted[:10].mean():+.4f}%")
        print(f"    Best single trade:      {pnls_sorted[-1]:+.4f}%")
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
    
    # ── Step 6: IS vs OOS comparison ──
    print(f"\n{'#' * 100}")
    print(f"  STEP 6: IN-SAMPLE vs OUT-OF-SAMPLE Comparison")
    print(f"{'#' * 100}\n")
    
    # In-sample numbers (from the partial exits research, per-trade avg)
    is_numbers = {
        "Trail 5bps (taker)":           {'avg': 0.1051, 'wr': 92.3, 'sharpe': 38.4},
        "A1: 70% TP@12 + 30% trail@3": {'avg': 0.0869, 'wr': 95.1, 'sharpe': 83.2},
        "A2: 50% TP@8 + 50% trail@3":  {'avg': 0.0808, 'wr': 95.1, 'sharpe': 56.7},
        "C1: 50% TP@10, trail 5→3":    {'avg': 0.1104, 'wr': 92.3, 'sharpe': 58.2},
    }
    
    oos_configs = [
        ("Trail 5bps (taker)", 'baseline_trail', {'trail_bps': 5}),
        ("A1: 70% TP@12 + 30% trail@3", 'A', {'tp_frac': 0.7, 'tp_bps': 12, 'trail_bps': 3}),
        ("A2: 50% TP@8 + 50% trail@3", 'A', {'tp_frac': 0.5, 'tp_bps': 8, 'trail_bps': 3}),
        ("C1: 50% TP@10, trail 5→3", 'C', {'tp_frac': 0.5, 'tp_bps': 10, 'trail_bps': 5, 'trail_tight_bps': 3}),
    ]
    
    print(f"  {'Config':<35s}  {'IS avg':>8s}  {'OOS avg':>8s}  {'Ratio':>6s}  {'IS WR':>6s}  {'OOS WR':>6s}  {'IS Sh':>6s}  {'OOS Sh':>6s}")
    print(f"  {'─'*35}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
    
    for label, variant, params in oos_configs:
        trades = run_all(variant, params)
        s = stats(trades)
        is_n = is_numbers.get(label, {})
        is_avg = is_n.get('avg', 0)
        is_wr = is_n.get('wr', 0)
        is_sh = is_n.get('sharpe', 0)
        ratio = s['avg'] / is_avg if is_avg > 0 else 0
        print(f"  {label:<35s}  {is_avg:+.4f}%  {s['avg']:+.4f}%  {ratio:5.1f}x  {is_wr:5.1f}%  {s['wr']:5.1f}%  {is_sh:+5.1f}  {s['sharpe']:+5.1f}")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 100}")
    print(f"  OOS TEST COMPLETE — {elapsed:.0f}s")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
