#!/usr/bin/env python3
"""
Liquidation Microstructure Research (v26g)

Deep analysis of liquidation cascade mechanics using tick-level data:

1. CASCADE FORMATION RATE
   - How often does a P95/P90 liquidation lead to a cascade (2+ events in 60s)?
   - By symbol, by side (buy/sell)

2. PRICE MOVEMENT BETWEEN LIQUIDATIONS
   - How many bps does price move from 1st P95 event to 2nd?
   - Distribution of inter-liquidation price moves

3. BOUNCE PROBABILITY CURVES
   - After price drops X bps from a large liquidation, what's P(revert) within 1/5/30 min?
   - Heatmap: X-axis = price displacement (bps), Y-axis = time horizon, color = P(revert)

4. COUNTER-STRATEGY ANALYSIS
   - Ride WITH the cascade (momentum): short when longs liquidated, long when shorts liquidated
   - Compare momentum vs fade at different time horizons

All analysis uses 5-second ticker data for maximum resolution.
"""

import sys
import time
import json
import gzip
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ["DOGEUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT"]
OUT_DIR = Path("results")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_liquidations(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    if not liq_files:
        raise ValueError(f"No liquidation files for {symbol}")
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
    if not ticker_files:
        raise ValueError(f"No ticker files for {symbol}")
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


def build_price_bars(tick_df, freq='5s'):
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    return bars.dropna()


# ============================================================================
# ANALYSIS 1: CASCADE FORMATION RATE
# ============================================================================

def analyze_cascade_formation(liq_df, symbol):
    """How often does a P95/P90 liquidation lead to a cascade?"""
    print(f"\n  ── ANALYSIS 1: CASCADE FORMATION RATE ──")

    results = {}
    for pct in [90, 95, 97, 99]:
        thresh = liq_df['notional'].quantile(pct / 100)
        large = liq_df[liq_df['notional'] >= thresh].copy()

        # For each large liquidation, check if another P-level event follows within 60s
        timestamps = large['timestamp'].values
        sides = large['side'].values
        notionals = large['notional'].values

        n_total = len(large)
        n_cascade = 0  # followed by another within 60s
        n_cascade_3plus = 0  # 3+ events within 60s
        cascade_sizes = []

        i = 0
        while i < n_total:
            # Count how many events cluster from this point
            cluster = [i]
            j = i + 1
            while j < n_total:
                dt = (timestamps[j] - timestamps[cluster[-1]]).astype('timedelta64[s]').astype(float)
                if dt <= 60:
                    cluster.append(j)
                    j += 1
                else:
                    break

            if len(cluster) >= 2:
                n_cascade += 1
                cascade_sizes.append(len(cluster))
            if len(cluster) >= 3:
                n_cascade_3plus += 1

            i = cluster[-1] + 1 if len(cluster) >= 2 else i + 1

        cascade_rate = n_cascade / max(n_total, 1) * 100
        cascade_3_rate = n_cascade_3plus / max(n_total, 1) * 100

        results[pct] = {
            'threshold': thresh,
            'n_large': n_total,
            'n_cascade_2plus': n_cascade,
            'n_cascade_3plus': n_cascade_3plus,
            'cascade_rate_2plus': cascade_rate,
            'cascade_rate_3plus': cascade_3_rate,
            'avg_cascade_size': np.mean(cascade_sizes) if cascade_sizes else 0,
        }

        print(f"    P{pct}: thresh=${thresh:,.0f}  n={n_total:,}  "
              f"→cascade(2+): {n_cascade} ({cascade_rate:.1f}%)  "
              f"→cascade(3+): {n_cascade_3plus} ({cascade_3_rate:.1f}%)  "
              f"avg_size={np.mean(cascade_sizes):.1f}" if cascade_sizes else
              f"    P{pct}: thresh=${thresh:,.0f}  n={n_total:,}  "
              f"→cascade(2+): {n_cascade} ({cascade_rate:.1f}%)  "
              f"→cascade(3+): {n_cascade_3plus} ({cascade_3_rate:.1f}%)")

    # By side
    for side in ['Buy', 'Sell']:
        side_df = liq_df[liq_df['side'] == side]
        thresh95 = liq_df['notional'].quantile(0.95)  # global threshold
        large_side = side_df[side_df['notional'] >= thresh95]
        n_side = len(large_side)
        print(f"    P95 {side}-side: {n_side} events ({n_side/max(len(liq_df[liq_df['notional']>=thresh95]),1)*100:.1f}% of P95)")

    return results


# ============================================================================
# ANALYSIS 2: PRICE MOVEMENT BETWEEN LIQUIDATIONS
# ============================================================================

def analyze_price_between_liquidations(liq_df, tick_df, symbol):
    """How many bps does price move from 1st P95 event to 2nd?"""
    print(f"\n  ── ANALYSIS 2: PRICE MOVEMENT BETWEEN LIQUIDATIONS ──")

    thresh95 = liq_df['notional'].quantile(0.95)
    large = liq_df[liq_df['notional'] >= thresh95].copy()

    # Build a price lookup: for each liquidation, find the closest tick price
    tick_ts = tick_df['timestamp'].values
    tick_px = tick_df['price'].values

    # For each pair of consecutive P95 events within 60s, measure price change
    timestamps = large['timestamp'].values
    sides = large['side'].values
    notionals = large['notional'].values
    prices = large['price'].values

    inter_moves_bps = []
    inter_times_sec = []
    inter_sides = []

    for i in range(len(large) - 1):
        dt_sec = (timestamps[i+1] - timestamps[i]).astype('timedelta64[s]').astype(float)
        if dt_sec <= 60 and dt_sec > 0:
            # Price change from 1st to 2nd liquidation
            p1 = prices[i]
            p2 = prices[i+1]
            move_bps = (p2 - p1) / p1 * 10000
            inter_moves_bps.append(move_bps)
            inter_times_sec.append(dt_sec)
            inter_sides.append(sides[i])

    inter_moves_bps = np.array(inter_moves_bps)
    inter_times_sec = np.array(inter_times_sec)
    inter_sides = np.array(inter_sides)

    print(f"    Pairs within 60s: {len(inter_moves_bps):,}")
    print(f"    Price move (bps): mean={np.mean(inter_moves_bps):+.2f}  "
          f"median={np.median(inter_moves_bps):+.2f}  "
          f"std={np.std(inter_moves_bps):.2f}")
    print(f"    Abs move (bps):   mean={np.mean(np.abs(inter_moves_bps)):.2f}  "
          f"P50={np.percentile(np.abs(inter_moves_bps), 50):.2f}  "
          f"P75={np.percentile(np.abs(inter_moves_bps), 75):.2f}  "
          f"P90={np.percentile(np.abs(inter_moves_bps), 90):.2f}  "
          f"P99={np.percentile(np.abs(inter_moves_bps), 99):.2f}")
    print(f"    Time between (s): mean={np.mean(inter_times_sec):.1f}  "
          f"median={np.median(inter_times_sec):.1f}")

    # By side
    for side in ['Buy', 'Sell']:
        mask = inter_sides == side
        if mask.sum() > 10:
            moves = inter_moves_bps[mask]
            print(f"    {side}-side: n={mask.sum()}  mean_move={np.mean(moves):+.2f} bps  "
                  f"abs_mean={np.mean(np.abs(moves)):.2f} bps")

    return {
        'moves_bps': inter_moves_bps,
        'times_sec': inter_times_sec,
        'sides': inter_sides,
    }


# ============================================================================
# ANALYSIS 3: BOUNCE PROBABILITY CURVES
# ============================================================================

def analyze_bounce_probability(liq_df, price_bars_5s, symbol):
    """
    After a large liquidation, track price path using CLOSE prices at fixed offsets.
    
    Approach:
    - For each P95 liquidation, record the signed displacement (bps) at many time offsets
    - "Adverse" = price moving WITH the cascade direction
    - For buy-side liqs (longs liquidated): adverse = price going DOWN (negative displacement)
    - We normalize so adverse is always negative, favorable is positive
    - Then compute: given max adverse reached X bps, what fraction recovered to 0+ by time T?
    """
    print(f"\n  ── ANALYSIS 3: BOUNCE PROBABILITY CURVES ──")

    thresh95 = liq_df['notional'].quantile(0.95)
    large = liq_df[liq_df['notional'] >= thresh95].copy()

    bar_index = price_bars_5s.index
    bar_close = price_bars_5s['close'].values

    # Time offsets to sample (in 5s bars)
    sample_offsets_sec = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, 1800, 3600]
    sample_offsets_bars = [max(1, int(s / 5)) for s in sample_offsets_sec]
    max_offset_bars = max(sample_offsets_bars)

    # For bounce analysis: check horizons
    horizons_min = [1, 2, 5, 10, 30, 60]
    horizons_bars = [int(h * 60 / 5) for h in horizons_min]

    # Vectorized approach: pre-compute all indices, then slice numpy arrays
    max_horizon_bars = max(horizons_bars)
    timestamps = large['timestamp'].values
    sides = large['side'].values

    # Find bar indices for all events at once
    indices = bar_index.searchsorted(timestamps)
    valid_mask = (indices >= 1) & (indices < len(bar_close) - max_offset_bars - 2)
    valid_indices = indices[valid_mask]
    valid_sides = sides[valid_mask]
    n_valid = len(valid_indices)
    print(f"    Processing {n_valid:,} valid events (vectorized)...", flush=True)

    all_events = []

    # Process in batches for memory efficiency
    batch_size = 500
    for batch_start in range(0, n_valid, batch_size):
        batch_end = min(batch_start + batch_size, n_valid)
        batch_idx = valid_indices[batch_start:batch_end]
        batch_sides = valid_sides[batch_start:batch_end]

        for i in range(len(batch_idx)):
            idx = batch_idx[i]
            side = batch_sides[i]
            entry_price = bar_close[idx]
            if entry_price <= 0:
                continue

            # Sample displacements at fixed offsets (vectorized slice)
            displacements = {}
            for off_sec, off_bars in zip(sample_offsets_sec, sample_offsets_bars):
                future_idx = idx + off_bars
                if future_idx < len(bar_close):
                    future_price = bar_close[future_idx]
                    if side == 'Buy':
                        disp_bps = (future_price - entry_price) / entry_price * 10000
                    else:
                        disp_bps = (entry_price - future_price) / entry_price * 10000
                    displacements[off_sec] = disp_bps

            # Vectorized: get all close prices for the next max_horizon_bars bars
            end_bar = min(idx + max_horizon_bars + 1, len(bar_close))
            future_prices = bar_close[idx + 1 : end_bar]

            if len(future_prices) == 0:
                continue

            # Compute signed displacement array (positive = fade-favorable)
            if side == 'Buy':
                disp_arr = (future_prices - entry_price) / entry_price * 10000
            else:
                disp_arr = (entry_price - future_prices) / entry_price * 10000

            # Max adverse = most negative value in disp_arr
            min_disp = np.min(disp_arr)
            if min_disp < 0:
                max_adverse_bps = -min_disp
                was_adverse = True

                # Find first bounce: first index where disp >= 0 AFTER the min
                min_idx = np.argmin(disp_arr)
                remaining = disp_arr[min_idx:]
                bounce_positions = np.where(remaining >= 0)[0]
                if len(bounce_positions) > 0:
                    first_bounce_bar = min_idx + bounce_positions[0] + 1  # +1 for offset from idx
                    bounce_time_sec = first_bounce_bar * 5
                    bounce_times = {}
                    for h_min in horizons_min:
                        if bounce_time_sec <= h_min * 60:
                            bounce_times[h_min] = bounce_time_sec / 60
                else:
                    bounce_times = {}
            else:
                max_adverse_bps = 0
                was_adverse = False
                bounce_times = {}

            all_events.append({
                'side': side,
                'max_adverse_bps': max_adverse_bps,
                'was_adverse': was_adverse,
                'bounce_times': bounce_times,
                'displacements': displacements,
                'entry_price': entry_price,
            })

        if batch_end % 2000 < batch_size:
            print(f"      [{batch_end}/{n_valid}]", flush=True)

    print(f"    Analyzed {len(all_events):,} P95 liquidation events")

    # ── PRICE PATH SUMMARY ──
    print(f"\n    Average displacement (bps) at each time offset (positive = fade-favorable):")
    print(f"    {'offset':>8s}  {'mean':>7s}  {'median':>7s}  {'P25':>7s}  {'P75':>7s}  {'%>0':>5s}")
    for off_sec in sample_offsets_sec:
        vals = [ev['displacements'].get(off_sec, np.nan) for ev in all_events]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 10:
            continue
        arr = np.array(vals)
        if off_sec < 60:
            label = f"{off_sec}s"
        elif off_sec < 3600:
            label = f"{off_sec//60}m"
        else:
            label = f"{off_sec//3600}h"
        print(f"    {label:>8s}  {np.mean(arr):+6.2f}  {np.median(arr):+6.2f}  "
              f"{np.percentile(arr,25):+6.2f}  {np.percentile(arr,75):+6.2f}  "
              f"{(arr>0).mean()*100:4.1f}%")

    # ── MAX ADVERSE DISTRIBUTION ──
    advs = [ev['max_adverse_bps'] for ev in all_events if ev['was_adverse']]
    all_advs = [ev['max_adverse_bps'] for ev in all_events]
    n_adverse = len(advs)
    print(f"\n    Events with any adverse move: {n_adverse}/{len(all_events)} "
          f"({n_adverse/len(all_events)*100:.1f}%)")
    if advs:
        print(f"    Max adverse (bps, among those that moved adversely): "
              f"mean={np.mean(advs):.1f}  P50={np.percentile(advs,50):.1f}  "
              f"P75={np.percentile(advs,75):.1f}  P90={np.percentile(advs,90):.1f}  "
              f"P95={np.percentile(advs,95):.1f}  P99={np.percentile(advs,99):.1f}")

    # ── BOUNCE PROBABILITY BY DISPLACEMENT ──
    disp_thresholds = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    print(f"\n    Bounce probability (given max adverse >= X bps, P(returns to entry) within T):")
    print(f"    {'≥X bps':>8s}  {'n':>6s}", end='')
    for h in horizons_min:
        print(f"  {h:>4d}m", end='')
    print()

    disp_bins_for_chart = []
    prob_matrix = []
    count_vec = []

    for thresh_bps in disp_thresholds:
        subset = [ev for ev in all_events if ev['max_adverse_bps'] >= thresh_bps]
        n = len(subset)
        if n < 10:
            continue
        disp_bins_for_chart.append(thresh_bps)
        count_vec.append(n)
        row = []
        print(f"    {thresh_bps:>6d}bp  {n:>6d}", end='')
        for h_min in horizons_min:
            bounced = sum(1 for ev in subset if h_min in ev['bounce_times'])
            p = bounced / n
            row.append(p)
            print(f"  {p*100:5.1f}%", end='')
        print()
        prob_matrix.append(row)

    prob_matrix = np.array(prob_matrix) if prob_matrix else np.array([[]])

    # ── OVERALL BOUNCE RATE ──
    print(f"\n    Overall bounce rate (all P95 events):")
    for h_min in horizons_min:
        bounced = sum(1 for ev in all_events if h_min in ev['bounce_times'])
        print(f"    Bounced within {h_min:>2d}m: {bounced}/{len(all_events)} "
              f"({bounced/len(all_events)*100:.1f}%)")

    # ── "POINT OF NO RETURN" ANALYSIS ──
    # At what displacement does bounce probability drop below 50%?
    print(f"\n    Point of no return (displacement where P(bounce within T) < 50%):")
    for h_idx, h_min in enumerate(horizons_min):
        found = False
        for d_idx, thresh_bps in enumerate(disp_bins_for_chart):
            if d_idx < len(prob_matrix) and prob_matrix[d_idx, h_idx] < 0.5:
                print(f"    {h_min:>2d}m: ~{thresh_bps} bps (P={prob_matrix[d_idx, h_idx]*100:.1f}%, n={count_vec[d_idx]})")
                found = True
                break
        if not found:
            print(f"    {h_min:>2d}m: >100 bps (always bounces)")

    return {
        'events': all_events,
        'prob_matrix': prob_matrix,
        'disp_bins': np.array(disp_bins_for_chart),
        'count_vec': count_vec,
        'horizons_min': horizons_min,
        'sample_offsets_sec': sample_offsets_sec,
    }


# ============================================================================
# ANALYSIS 4: COUNTER-STRATEGY (MOMENTUM — RIDE THE CASCADE)
# ============================================================================

def analyze_counter_strategy(liq_df, price_bars_5s, symbol):
    """
    Instead of fading the cascade, ride WITH it:
    - Longs liquidated → SHORT immediately (ride the drop)
    - Shorts liquidated → LONG immediately (ride the spike)

    Compare returns at different exit horizons.
    """
    print(f"\n  ── ANALYSIS 4: COUNTER-STRATEGY (MOMENTUM) ──")

    thresh95 = liq_df['notional'].quantile(0.95)
    large = liq_df[liq_df['notional'] >= thresh95].copy()

    bar_index = price_bars_5s.index
    bar_close = price_bars_5s['close'].values

    # Exit horizons (in 5s bars)
    exit_horizons_sec = [5, 10, 15, 30, 60, 120, 300, 600, 1800]
    exit_horizons_bars = [max(1, int(h / 5)) for h in exit_horizons_sec]

    # Track returns for both strategies
    momentum_returns = {h: [] for h in exit_horizons_sec}
    fade_returns = {h: [] for h in exit_horizons_sec}

    cooldown_sec = 60  # don't enter if another P95 was within 60s before
    last_entry_ts = None

    for _, row in large.iterrows():
        ts = row['timestamp']
        side = row['side']

        # Cooldown
        if last_entry_ts is not None:
            dt = (ts - last_entry_ts).total_seconds()
            if dt < cooldown_sec:
                continue

        idx = bar_index.searchsorted(ts)
        if idx >= len(bar_index) - max(exit_horizons_bars) - 1:
            continue
        if idx < 1:
            continue

        entry_price = bar_close[idx]
        last_entry_ts = ts

        for h_sec, h_bars in zip(exit_horizons_sec, exit_horizons_bars):
            exit_idx = min(idx + h_bars, len(bar_index) - 1)
            exit_price = bar_close[exit_idx]

            if side == 'Buy':
                # Longs liquidated → price dropping
                # Momentum: SHORT (profit if price goes lower)
                momentum_pnl = (entry_price - exit_price) / entry_price * 10000  # bps
                # Fade: LONG (profit if price bounces back up)
                fade_pnl = (exit_price - entry_price) / entry_price * 10000
            else:
                # Shorts liquidated → price spiking
                # Momentum: LONG (profit if price goes higher)
                momentum_pnl = (exit_price - entry_price) / entry_price * 10000
                # Fade: SHORT (profit if price comes back down)
                fade_pnl = (entry_price - exit_price) / entry_price * 10000

            momentum_returns[h_sec].append(momentum_pnl)
            fade_returns[h_sec].append(fade_pnl)

    print(f"    Events analyzed: {len(momentum_returns[exit_horizons_sec[0]]):,}")
    print(f"\n    {'Exit':>8s}  {'n':>5s}  {'Mom avg':>8s} {'Mom wr':>7s} {'Mom P50':>8s}  "
          f"{'Fade avg':>8s} {'Fade wr':>7s} {'Fade P50':>8s}  {'Winner':>8s}")
    print(f"    {'─'*85}")

    for h_sec in exit_horizons_sec:
        mom = np.array(momentum_returns[h_sec])
        fad = np.array(fade_returns[h_sec])
        n = len(mom)
        if n < 10:
            continue

        mom_avg = np.mean(mom)
        mom_wr = (mom > 0).sum() / n * 100
        mom_p50 = np.median(mom)
        fad_avg = np.mean(fad)
        fad_wr = (fad > 0).sum() / n * 100
        fad_p50 = np.median(fad)

        if h_sec < 60:
            label = f"{h_sec}s"
        elif h_sec < 3600:
            label = f"{h_sec//60}m"
        else:
            label = f"{h_sec//3600}h"

        winner = "MOM" if mom_avg > fad_avg else "FADE"
        print(f"    {label:>8s}  {n:>5d}  {mom_avg:+7.2f}bp {mom_wr:5.1f}%  {mom_p50:+7.2f}bp  "
              f"{fad_avg:+7.2f}bp {fad_wr:5.1f}%  {fad_p50:+7.2f}bp  {winner:>8s}")

    return {
        'momentum': momentum_returns,
        'fade': fade_returns,
        'horizons': exit_horizons_sec,
    }


# ============================================================================
# CHARTS
# ============================================================================

def plot_all_charts(symbol, inter_data, bounce_data, counter_data):
    """Generate comprehensive charts."""

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f'{symbol} — Liquidation Microstructure Analysis', fontsize=16, fontweight='bold')
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)

    # ── Chart 1: Price move between consecutive P95 liquidations ──
    ax1 = fig.add_subplot(gs[0, 0])
    moves = inter_data['moves_bps']
    if len(moves) > 0:
        ax1.hist(moves, bins=100, range=(-50, 50), color='steelblue', alpha=0.7, edgecolor='none')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(np.mean(moves), color='orange', linestyle='-', linewidth=2,
                     label=f'mean={np.mean(moves):+.2f} bps')
        ax1.set_xlabel('Price Move (bps)')
        ax1.set_ylabel('Count')
        ax1.set_title('Price Move: 1st → 2nd P95 Liquidation')
        ax1.legend()

    # ── Chart 2: Time between consecutive P95 liquidations ──
    ax2 = fig.add_subplot(gs[0, 1])
    times = inter_data['times_sec']
    if len(times) > 0:
        ax2.hist(times, bins=60, range=(0, 60), color='coral', alpha=0.7, edgecolor='none')
        ax2.axvline(np.median(times), color='blue', linestyle='-', linewidth=2,
                     label=f'median={np.median(times):.1f}s')
        ax2.set_xlabel('Time Between Events (seconds)')
        ax2.set_ylabel('Count')
        ax2.set_title('Time Between Consecutive P95 Liquidations')
        ax2.legend()

    # ── Chart 3: Bounce probability heatmap ──
    ax3 = fig.add_subplot(gs[1, :])
    prob = bounce_data['prob_matrix']
    disp_bins = bounce_data['disp_bins']
    horizons = bounce_data['horizons_min']
    count_vec = bounce_data.get('count_vec', [])

    if len(prob) > 0 and prob.size > 1 and len(disp_bins) > 0:
        valid_labels = [f"≥{d}bp" for d in disp_bins]
        cmap = LinearSegmentedColormap.from_list('bounce',
            ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'])
        im = ax3.imshow(prob.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax3.set_xticks(range(len(valid_labels)))
        ax3.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_yticks(range(len(horizons)))
        ax3.set_yticklabels([f'{h}m' for h in horizons])
        ax3.set_xlabel('Max Adverse Displacement (bps)')
        ax3.set_ylabel('Time Horizon')
        ax3.set_title('Bounce Probability: P(price returns to entry | displaced ≥ X bps)')

        # Add text annotations
        for i in range(len(disp_bins)):
            for j in range(len(horizons)):
                if i < prob.shape[0] and j < prob.shape[1]:
                    val = prob[i, j] if not np.isnan(prob[i, j]) else 0
                    n = count_vec[i] if i < len(count_vec) else 0
                    color = 'white' if val < 0.5 else 'black'
                    ax3.text(i, j, f'{val:.0%}\n(n={n})', ha='center', va='center',
                             fontsize=6, color=color)

        plt.colorbar(im, ax=ax3, label='P(bounce)')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax3.transAxes)

    # ── Chart 4: Max adverse move distribution ──
    ax4 = fig.add_subplot(gs[2, 0])
    advs = [ev['max_adverse_bps'] for ev in bounce_data['events']]
    if advs:
        ax4.hist(advs, bins=100, range=(0, 100), color='purple', alpha=0.7, edgecolor='none')
        ax4.axvline(np.median(advs), color='orange', linewidth=2,
                     label=f'median={np.median(advs):.1f} bps')
        ax4.axvline(np.percentile(advs, 90), color='red', linewidth=2, linestyle='--',
                     label=f'P90={np.percentile(advs, 90):.1f} bps')
        ax4.set_xlabel('Max Adverse Move (bps)')
        ax4.set_ylabel('Count')
        ax4.set_title('Max Adverse Move After P95 Liquidation (60min window)')
        ax4.legend()

    # ── Chart 5: Momentum vs Fade returns by horizon ──
    ax5 = fig.add_subplot(gs[2, 1])
    horizons_sec = counter_data['horizons']
    mom_avgs = [np.mean(counter_data['momentum'][h]) for h in horizons_sec if len(counter_data['momentum'][h]) > 0]
    fad_avgs = [np.mean(counter_data['fade'][h]) for h in horizons_sec if len(counter_data['fade'][h]) > 0]
    valid_h = [h for h in horizons_sec if len(counter_data['momentum'][h]) > 0]

    labels = []
    for h in valid_h:
        if h < 60:
            labels.append(f"{h}s")
        elif h < 3600:
            labels.append(f"{h//60}m")
        else:
            labels.append(f"{h//3600}h")

    x = np.arange(len(labels))
    width = 0.35
    ax5.bar(x - width/2, mom_avgs, width, label='Momentum (ride cascade)', color='#e74c3c', alpha=0.8)
    ax5.bar(x + width/2, fad_avgs, width, label='Fade (counter cascade)', color='#2ecc71', alpha=0.8)
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_xlabel('Exit Horizon')
    ax5.set_ylabel('Average Return (bps)')
    ax5.set_title('Momentum vs Fade: Avg Return by Exit Horizon')
    ax5.legend()

    # ── Chart 6: Momentum vs Fade win rates ──
    ax6 = fig.add_subplot(gs[3, 0])
    mom_wrs = [(np.array(counter_data['momentum'][h]) > 0).mean() * 100
               for h in valid_h]
    fad_wrs = [(np.array(counter_data['fade'][h]) > 0).mean() * 100
               for h in valid_h]

    ax6.plot(labels, mom_wrs, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Momentum WR')
    ax6.plot(labels, fad_wrs, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Fade WR')
    ax6.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Exit Horizon')
    ax6.set_ylabel('Win Rate (%)')
    ax6.set_title('Win Rate: Momentum vs Fade')
    ax6.legend()
    ax6.set_ylim(30, 70)

    # ── Chart 7: Bounce probability by displacement (line chart) ──
    ax7 = fig.add_subplot(gs[3, 1])
    if len(prob) > 0 and prob.size > 1 and len(disp_bins) > 0:
        for h_idx, h_min in enumerate(horizons):
            if h_min in [1, 5, 30, 60] and h_idx < prob.shape[1]:
                probs = []
                x_vals = []
                for d_idx in range(len(disp_bins)):
                    if d_idx < prob.shape[0] and not np.isnan(prob[d_idx, h_idx]):
                        probs.append(prob[d_idx, h_idx] * 100)
                        x_vals.append(disp_bins[d_idx])
                if probs:
                    ax7.plot(x_vals, probs, 'o-', linewidth=2, markersize=5, label=f'{h_min}m')

    ax7.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Max Adverse Displacement (bps)')
    ax7.set_ylabel('Bounce Probability (%)')
    ax7.set_title('P(bounce back to entry) vs Displacement')
    ax7.legend()
    ax7.set_ylim(0, 105)

    chart_path = OUT_DIR / f'liq_microstructure_{symbol}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved: {chart_path}")
    return chart_path


# ============================================================================
# MAIN
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    print(f"\n{'='*90}")
    print(f"  {symbol} — LIQUIDATION MICROSTRUCTURE ANALYSIS")
    print(f"{'='*90}")

    t0 = time.time()

    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    print("  Building 5-second bars...", end='', flush=True)
    price_bars_5s = build_price_bars(tick_df, '5s')
    print(f" {len(price_bars_5s):,} bars")

    days = (price_bars_5s.index.max() - price_bars_5s.index.min()).total_seconds() / 86400
    print(f"  Period: {days:.0f} days, {len(liq_df):,} liquidations, {len(tick_df):,} ticks")

    # Analysis 1: Cascade formation
    cascade_stats = analyze_cascade_formation(liq_df, symbol)

    # Analysis 2: Price between liquidations
    inter_data = analyze_price_between_liquidations(liq_df, tick_df, symbol)

    # Analysis 3: Bounce probability
    bounce_data = analyze_bounce_probability(liq_df, price_bars_5s, symbol)

    # Analysis 4: Counter-strategy
    counter_data = analyze_counter_strategy(liq_df, price_bars_5s, symbol)

    # Charts
    chart_path = plot_all_charts(symbol, inter_data, bounce_data, counter_data)

    elapsed = time.time() - t0
    print(f"\n  {symbol} done in {elapsed:.0f}s")

    return {
        'cascade_stats': cascade_stats,
        'inter_data': inter_data,
        'bounce_data': bounce_data,
        'counter_data': counter_data,
    }


def main():
    t_start = time.time()

    print("=" * 90)
    print("  LIQUIDATION MICROSTRUCTURE RESEARCH (v26g)")
    print("  Tick-level analysis: cascade formation, price dynamics, bounce curves")
    print("=" * 90)

    OUT_DIR.mkdir(exist_ok=True)

    all_results = {}
    for sym in SYMBOLS:
        try:
            all_results[sym] = run_symbol(sym)
        except Exception as e:
            print(f"\n  ✗ {sym} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── CROSS-SYMBOL SUMMARY ──
    print(f"\n\n{'='*90}")
    print(f"  CROSS-SYMBOL SUMMARY")
    print(f"{'='*90}")

    # Cascade formation rates
    print(f"\n  CASCADE FORMATION RATES:")
    print(f"  {'Symbol':>10s}  {'P90→cas':>8s}  {'P95→cas':>8s}  {'P97→cas':>8s}  {'P99→cas':>8s}")
    for sym, res in all_results.items():
        cs = res['cascade_stats']
        print(f"  {sym:>10s}", end='')
        for pct in [90, 95, 97, 99]:
            if pct in cs:
                print(f"  {cs[pct]['cascade_rate_2plus']:7.1f}%", end='')
        print()

    # Counter-strategy crossover point
    print(f"\n  MOMENTUM vs FADE CROSSOVER:")
    for sym, res in all_results.items():
        cd = res['counter_data']
        print(f"  {sym}:")
        for h in cd['horizons']:
            mom = np.array(cd['momentum'][h])
            fad = np.array(cd['fade'][h])
            if len(mom) < 10:
                continue
            if h < 60:
                label = f"{h}s"
            else:
                label = f"{h//60}m"
            winner = "MOM" if np.mean(mom) > np.mean(fad) else "FADE"
            print(f"    {label:>5s}: mom={np.mean(mom):+.2f}bp  fade={np.mean(fad):+.2f}bp  → {winner}")

    elapsed = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
