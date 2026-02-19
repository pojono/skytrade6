#!/usr/bin/env python3
"""
COMPREHENSIVE STRESS TEST for liquidation cascade strategy.
Purpose: Attack the strategy from every skeptical angle before going live.

Tests:
  1. LOOK-AHEAD BIAS: Rolling P95 threshold vs global P95
  2. FILL PESSIMISM: Delay fill by 1 bar (can't fill on signal bar)
  3. WALK-FORWARD OOS: Train/test split, rolling windows
  4. FEE SENSITIVITY: At what fee does edge die?
  5. REGIME ANALYSIS: Monthly breakdown, worst periods
  6. RANDOM BASELINE: Random entry at same times — is alpha real?
  7. WORST-CASE: Max consecutive losses, worst drawdown periods
  8. SLIPPAGE: Add slippage to timeout exits
"""

import sys, time, json, gzip, random
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.055
SYMBOLS = ['DOGEUSDT', 'SOLUSDT', 'ETHUSDT', 'XRPUSDT']


# ============================================================================
# DATA LOADING (same as other scripts)
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


def load_ticker_prices(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    ticker_dirs = [symbol_dir / "bybit" / "ticker", symbol_dir]
    ticker_files = []
    for d in ticker_dirs:
        ticker_files.extend(sorted(d.glob("ticker_*.jsonl.gz")))
    ticker_files = sorted(set(ticker_files))
    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
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


def build_price_bars(tick_df, freq='1min'):
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    return bars.dropna()


# ============================================================================
# CASCADE DETECTION
# ============================================================================

def detect_signals_global_p95(liq_df, price_bars, pct_thresh=95):
    """Original: global P95 threshold (potential look-ahead bias)."""
    thresh = liq_df['notional'].quantile(pct_thresh / 100)
    return _detect_from_threshold(liq_df, price_bars, thresh)


def detect_signals_rolling_p95(liq_df, price_bars, pct_thresh=95, lookback=1000):
    """Rolling P95: only use past data to compute threshold (no look-ahead)."""
    bar_index = price_bars.index
    bar_close = price_bars['close'].values
    
    timestamps = liq_df['timestamp'].values
    notionals = liq_df['notional'].values
    sides = liq_df['side'].values
    n = len(liq_df)
    
    cascades = []
    for i in range(lookback, n):
        # Rolling threshold from past lookback events
        window_notionals = notionals[max(0, i - lookback):i]
        thresh = np.percentile(window_notionals, pct_thresh)
        
        if notionals[i] < thresh:
            continue
        
        # This is a P95 event using only past data
        ts = pd.Timestamp(timestamps[i])
        idx = bar_index.searchsorted(ts)
        if idx >= len(bar_close) - 120 or idx < 10:
            continue
        
        current_price = bar_close[idx]
        pre_price = bar_close[max(0, idx - 1)]
        cascade_disp_bps = (current_price - pre_price) / pre_price * 10000
        
        cascades.append({
            'end': ts,
            'n_events': 1,
            'buy_dominant': sides[i] == 'Buy',
            'end_bar_idx': idx,
            'current_price': current_price,
            'cascade_disp_bps': cascade_disp_bps,
        })
    
    return cascades


def _detect_from_threshold(liq_df, price_bars, thresh):
    """Detect signals from a fixed threshold."""
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
# STRATEGY SIMULATION
# ============================================================================

def run_strategy(cascades, price_bars, tp_pct=0.12, sl_pct=None,
                 entry_offset_pct=0.15, max_hold_min=60, min_disp_bps=10,
                 maker_fee=MAKER_FEE_PCT, taker_fee=TAKER_FEE_PCT,
                 fill_delay_bars=0, timeout_slippage_bps=0,
                 random_direction=False):
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
        
        if random_direction:
            direction = 'long' if random.random() < 0.5 else 'short'
        elif cascade['buy_dominant']:
            direction = 'long'
        else:
            direction = 'short'
            
        if direction == 'long':
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100) if sl_pct else None
        else:
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100) if sl_pct else None

        # Fill search starts after delay
        fill_start = idx + fill_delay_bars
        end_bar = min(idx + max_hold_min, len(bar_close) - 1)
        filled = False
        fill_bar = None
        for j in range(fill_start, end_bar + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True; fill_bar = j; break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True; fill_bar = j; break
        if not filled:
            continue

        exit_price = None; exit_reason = 'timeout'
        remaining = max_hold_min - (fill_bar - idx)
        exit_end = min(fill_bar + remaining, len(bar_close) - 1)
        for k in range(fill_bar, exit_end + 1):
            if direction == 'long':
                if sl_price and bar_low[k] <= sl_price:
                    exit_price = sl_price; exit_reason = 'stop_loss'; break
                if bar_high[k] >= tp_price:
                    exit_price = tp_price; exit_reason = 'take_profit'; break
            else:
                if sl_price and bar_high[k] >= sl_price:
                    exit_price = sl_price; exit_reason = 'stop_loss'; break
                if bar_low[k] <= tp_price:
                    exit_price = tp_price; exit_reason = 'take_profit'; break
        if exit_price is None:
            exit_price = bar_close[exit_end]
            # Add slippage on timeout exits
            if timeout_slippage_bps > 0:
                slip = exit_price * timeout_slippage_bps / 10000
                if direction == 'long':
                    exit_price -= slip
                else:
                    exit_price += slip

        if direction == 'long':
            raw_pnl = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl = (limit_price - exit_price) / limit_price * 100

        entry_fee = maker_fee
        exit_fee = maker_fee if exit_reason == 'take_profit' else taker_fee
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


def pline(label, s):
    if s['n'] == 0:
        print(f"    {label:50s}  n=    0  (no trades)")
        return
    flag = '✅' if s['total'] > 0 else '❌'
    print(f"  {flag} {label:50s}  n={s['n']:>5d}  WR={s['wr']:5.1f}%  "
          f"avg={s['avg']:+.4f}%  tot={s['total']:+7.2f}%  sh={s['sharpe']:+6.1f}  dd={s['maxdd']:5.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("  COMPREHENSIVE STRESS TEST — Liquidation Cascade Strategy")
    print("  Config 2 AGGR: off=0.15%, TP=0.12%, no SL, 60min, disp≥10bps, min_ev=1")
    print("=" * 100)

    all_cascades = {}
    all_bars = {}
    all_liq = {}

    for symbol in SYMBOLS:
        print(f"\n{'─'*80}")
        print(f"  Loading {symbol}...")
        print(f"{'─'*80}")
        liq_df = load_liquidations(symbol)
        tick_df = load_ticker_prices(symbol)
        print("  Building bars...", end='', flush=True)
        bars = build_price_bars(tick_df, '1min')
        print(f" {len(bars):,} bars")
        
        cascades = detect_signals_global_p95(liq_df, bars, pct_thresh=95)
        print(f"  Global P95 signals: {len(cascades)}")
        
        all_cascades[symbol] = cascades
        all_bars[symbol] = bars
        all_liq[symbol] = liq_df

    # ========================================================================
    # TEST 1: LOOK-AHEAD BIAS — Rolling vs Global P95
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 1: LOOK-AHEAD BIAS — Rolling P95 vs Global P95")
    print(f"  (Global uses ALL data to compute threshold; Rolling uses only past 1000 events)")
    print(f"{'#'*100}")

    for symbol in SYMBOLS:
        print(f"\n  ── {symbol} ──")
        # Global (what we've been using)
        t_global = run_strategy(all_cascades[symbol], all_bars[symbol])
        s_global = stats(t_global)
        pline("Global P95 (potential bias)", s_global)
        
        # Rolling (no look-ahead)
        rolling_cascades = detect_signals_rolling_p95(all_liq[symbol], all_bars[symbol])
        print(f"    Rolling P95 signals: {len(rolling_cascades)}")
        t_rolling = run_strategy(rolling_cascades, all_bars[symbol])
        s_rolling = stats(t_rolling)
        pline("Rolling P95 (no look-ahead)", s_rolling)
        
        if s_global['n'] > 0 and s_rolling['n'] > 0:
            delta = s_rolling['total'] - s_global['total']
            print(f"    Delta: {delta:+.2f}% ({'rolling better' if delta > 0 else 'global better'})")

    # ========================================================================
    # TEST 2: FILL PESSIMISM — Delay fill by 1 bar
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 2: FILL PESSIMISM — Can't fill on signal bar (1-bar delay)")
    print(f"  (Tests if we're getting unrealistic fills on the same bar as the signal)")
    print(f"{'#'*100}")

    for symbol in SYMBOLS:
        print(f"\n  ── {symbol} ──")
        t_base = run_strategy(all_cascades[symbol], all_bars[symbol], fill_delay_bars=0)
        t_delay = run_strategy(all_cascades[symbol], all_bars[symbol], fill_delay_bars=1)
        pline("Baseline (fill on signal bar)", stats(t_base))
        pline("1-bar delay (fill from bar+1)", stats(t_delay))

    # ========================================================================
    # TEST 3: WALK-FORWARD OOS
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 3: WALK-FORWARD OOS — 60-day train / 30-day test windows")
    print(f"  (Strategy params are fixed, but tests if edge persists across time)")
    print(f"{'#'*100}")

    for symbol in SYMBOLS:
        bars = all_bars[symbol]
        cascades = all_cascades[symbol]
        dates = bars.index
        min_date = dates.min()
        max_date = dates.max()
        
        print(f"\n  ── {symbol} ({min_date.date()} to {max_date.date()}) ──")
        
        # Create rolling windows
        window_start = min_date
        train_days = 60
        test_days = 30
        window_results = []
        
        while True:
            train_end = window_start + pd.Timedelta(days=train_days)
            test_end = train_end + pd.Timedelta(days=test_days)
            if test_end > max_date:
                break
            
            # Filter cascades to test window only
            test_cascades = [c for c in cascades 
                           if train_end <= c['end'] < test_end]
            
            if len(test_cascades) > 0:
                t = run_strategy(test_cascades, bars)
                s = stats(t)
                flag = '✅' if s['total'] > 0 else '❌'
                print(f"    {flag} OOS {train_end.date()} to {test_end.date()}: "
                      f"n={s['n']:>3d}  WR={s['wr']:5.1f}%  tot={s['total']:+6.2f}%")
                window_results.append(s['total'])
            else:
                print(f"    ⚪ OOS {train_end.date()} to {test_end.date()}: no cascades in window")
            
            window_start += pd.Timedelta(days=test_days)
        
        if window_results:
            pos = sum(1 for r in window_results if r > 0)
            print(f"    Summary: {pos}/{len(window_results)} OOS windows positive "
                  f"({pos/len(window_results)*100:.0f}%)")

    # ========================================================================
    # TEST 4: FEE SENSITIVITY
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 4: FEE SENSITIVITY — At what fee does the edge die?")
    print(f"{'#'*100}")

    fee_levels = [
        (0.00, 0.00, "0% maker / 0% taker (ideal)"),
        (0.01, 0.03, "1 bps / 3 bps (VIP)"),
        (0.02, 0.055, "2 bps / 5.5 bps (CURRENT)"),
        (0.03, 0.06, "3 bps / 6 bps (pessimistic)"),
        (0.04, 0.07, "4 bps / 7 bps (very pessimistic)"),
        (0.05, 0.08, "5 bps / 8 bps (worst case)"),
        (0.06, 0.10, "6 bps / 10 bps (extreme)"),
    ]

    for maker, taker, label in fee_levels:
        total = 0
        total_n = 0
        for symbol in SYMBOLS:
            t = run_strategy(all_cascades[symbol], all_bars[symbol],
                           maker_fee=maker, taker_fee=taker)
            s = stats(t)
            total += s['total']
            total_n += s['n']
        flag = '✅' if total > 0 else '❌'
        print(f"  {flag} {label:45s}  n={total_n:>5d}  tot={total:+7.2f}%")

    # ========================================================================
    # TEST 5: RANDOM DIRECTION BASELINE
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 5: RANDOM DIRECTION — Same signals, random long/short")
    print(f"  (Tests if direction signal matters or if it's just mean-reversion at any time)")
    print(f"{'#'*100}")

    n_trials = 20
    random_totals = []
    for trial in range(n_trials):
        random.seed(trial)
        trial_total = 0
        for symbol in SYMBOLS:
            t = run_strategy(all_cascades[symbol], all_bars[symbol], random_direction=True)
            trial_total += sum(tr['net_pnl'] for tr in t)
        random_totals.append(trial_total)
    
    # Real strategy
    real_total = 0
    for symbol in SYMBOLS:
        t = run_strategy(all_cascades[symbol], all_bars[symbol])
        real_total += sum(tr['net_pnl'] for tr in t)
    
    print(f"  Real strategy (correct direction):  {real_total:+7.2f}%")
    print(f"  Random direction (mean of {n_trials} trials): {np.mean(random_totals):+7.2f}%")
    print(f"  Random direction (std):              {np.std(random_totals):7.2f}%")
    print(f"  Random direction (min):             {np.min(random_totals):+7.2f}%")
    print(f"  Random direction (max):             {np.max(random_totals):+7.2f}%")
    print(f"  Z-score of real vs random:           {(real_total - np.mean(random_totals)) / max(np.std(random_totals), 0.01):+6.2f}")
    beats = sum(1 for r in random_totals if real_total > r)
    print(f"  Real beats random:                   {beats}/{n_trials} trials ({beats/n_trials*100:.0f}%)")

    # ========================================================================
    # TEST 6: OPPOSITE DIRECTION — Fade the WRONG way
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 6: OPPOSITE DIRECTION — What if we trade the wrong way?")
    print(f"  (If strategy works in both directions, it's just mean-reversion noise)")
    print(f"{'#'*100}")

    for symbol in SYMBOLS:
        # Flip buy_dominant for opposite direction
        flipped = []
        for c in all_cascades[symbol]:
            fc = dict(c)
            fc['buy_dominant'] = not fc['buy_dominant']
            flipped.append(fc)
        
        t_real = run_strategy(all_cascades[symbol], all_bars[symbol])
        t_flip = run_strategy(flipped, all_bars[symbol])
        print(f"\n  ── {symbol} ──")
        pline("Correct direction", stats(t_real))
        pline("OPPOSITE direction", stats(t_flip))

    # ========================================================================
    # TEST 7: TIMEOUT SLIPPAGE
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 7: TIMEOUT SLIPPAGE — Add 5/10/20 bps slippage on timeout exits")
    print(f"{'#'*100}")

    for slip_bps in [0, 5, 10, 20, 50]:
        total = 0
        total_n = 0
        for symbol in SYMBOLS:
            t = run_strategy(all_cascades[symbol], all_bars[symbol],
                           timeout_slippage_bps=slip_bps)
            s = stats(t)
            total += s['total']
            total_n += s['n']
        flag = '✅' if total > 0 else '❌'
        print(f"  {flag} Slippage={slip_bps:>2d} bps:  n={total_n:>5d}  tot={total:+7.2f}%")

    # ========================================================================
    # TEST 8: WORST-CASE ANALYSIS
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 8: WORST-CASE ANALYSIS — Consecutive losses, worst periods")
    print(f"{'#'*100}")

    for symbol in SYMBOLS:
        trades = run_strategy(all_cascades[symbol], all_bars[symbol])
        if not trades:
            continue
        net = np.array([t['net_pnl'] for t in trades])
        
        # Max consecutive losses
        max_consec_loss = 0
        current_streak = 0
        for pnl in net:
            if pnl < 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0
        
        # Worst N-trade window
        worst_10 = min(np.convolve(net, np.ones(10), mode='valid'))
        worst_20 = min(np.convolve(net, np.ones(20), mode='valid')) if len(net) >= 20 else 0
        worst_50 = min(np.convolve(net, np.ones(50), mode='valid')) if len(net) >= 50 else 0
        
        # Worst single trade
        worst_trade = net.min()
        
        # Time underwater
        cum = np.cumsum(net)
        peak = np.maximum.accumulate(cum)
        underwater = (peak > cum)
        max_underwater = 0
        current_uw = 0
        for uw in underwater:
            if uw:
                current_uw += 1
                max_underwater = max(max_underwater, current_uw)
            else:
                current_uw = 0
        
        # Timeout trade analysis
        timeout_pnls = [t['net_pnl'] for t in trades if t['exit_reason'] == 'timeout']
        
        print(f"\n  ── {symbol} ({len(trades)} trades) ──")
        print(f"    Max consecutive losses:    {max_consec_loss}")
        print(f"    Worst single trade:        {worst_trade:+.4f}%")
        print(f"    Worst 10-trade window:     {worst_10:+.4f}%")
        print(f"    Worst 20-trade window:     {worst_20:+.4f}%")
        print(f"    Worst 50-trade window:     {worst_50:+.4f}%")
        print(f"    Max trades underwater:     {max_underwater}")
        if timeout_pnls:
            print(f"    Timeout trades:            {len(timeout_pnls)} ({len(timeout_pnls)/len(trades)*100:.1f}%)")
            print(f"    Timeout avg PnL:           {np.mean(timeout_pnls):+.4f}%")
            print(f"    Timeout worst:             {min(timeout_pnls):+.4f}%")
            print(f"    Timeout best:              {max(timeout_pnls):+.4f}%")

    # ========================================================================
    # TEST 9: MONTHLY REGIME ANALYSIS
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 9: MONTHLY REGIME — Does it work in all market conditions?")
    print(f"{'#'*100}")

    for symbol in SYMBOLS:
        trades = run_strategy(all_cascades[symbol], all_bars[symbol])
        if not trades:
            continue
        
        print(f"\n  ── {symbol} ──")
        df = pd.DataFrame(trades)
        df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
        
        for month, group in df.groupby('month'):
            n = len(group)
            wr = (group['net_pnl'] > 0).mean() * 100
            total = group['net_pnl'].sum()
            flag = '✅' if total > 0 else '❌'
            print(f"    {flag} {month}: n={n:>3d}  WR={wr:5.1f}%  tot={total:+6.2f}%")

    # ========================================================================
    # TEST 10: DATA GAP CHECK
    # ========================================================================
    print(f"\n{'#'*100}")
    print(f"  TEST 10: DATA COVERAGE — Check for gaps in price data")
    print(f"{'#'*100}")

    for symbol in SYMBOLS:
        bars = all_bars[symbol]
        dates = bars.index
        
        # Check for gaps > 5 minutes
        diffs = pd.Series(dates).diff()
        big_gaps = diffs[diffs > pd.Timedelta(minutes=5)]
        
        total_days = (dates.max() - dates.min()).days
        covered_bars = len(bars)
        expected_bars = total_days * 24 * 60  # 1-min bars
        coverage = covered_bars / expected_bars * 100 if expected_bars > 0 else 0
        
        print(f"\n  ── {symbol} ──")
        print(f"    Date range: {dates.min().date()} to {dates.max().date()} ({total_days} days)")
        print(f"    Bars: {covered_bars:,} / {expected_bars:,} expected ({coverage:.1f}% coverage)")
        print(f"    Gaps > 5min: {len(big_gaps)}")
        if len(big_gaps) > 0:
            biggest = big_gaps.max()
            print(f"    Biggest gap: {biggest}")
            # Show top 5 gaps
            for gap_idx in big_gaps.nlargest(5).index:
                gap_time = dates[gap_idx]
                gap_dur = big_gaps[gap_idx]
                print(f"      {gap_time} — gap of {gap_dur}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*100}")
    print(f"  STRESS TEST COMPLETE — {time.time()-t0:.0f}s")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
