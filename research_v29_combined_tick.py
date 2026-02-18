#!/usr/bin/env python3
"""
v29: Combined Tick-Level Analysis — Trades + Liquidations + OI + Funding Rate

We have 4 tick-level data streams at second resolution:
  1. TRADES: price, volume, side (from csv.gz, ~2.4M/day)
  2. LIQUIDATIONS: forced exits with side/notional (from jsonl.gz, ~2K/day)
  3. OPEN INTEREST: dollar-denominated OI, updates every ~1.6s (from ticker jsonl.gz)
  4. FUNDING RATE: updates every ~60s (from ticker jsonl.gz)

EXPERIMENT DESIGN:
==================

1. BUILD combined per-second timeseries:
   - Vol (realized vol 60s), trade intensity, liq count/notional (from v28)
   - OI: last value per second, OI delta (change), OI velocity (rate of change)
   - FR: last value per second, FR delta, distance to next funding time

2. PROFILE all 4 signals around regime switches (±30min):
   - Does OI change BEFORE switches? (new positions = conviction)
   - Does FR spike BEFORE switches? (crowding = vulnerability)
   - Combined: OI↑ + FR extreme + liq spike → regime switch?

3. CROSS-CORRELATION at second resolution:
   - OI_delta vs vol, OI_delta vs liq, FR vs vol, FR vs liq
   - Which leads which?

4. COMBINED PREDICTIVE MODEL:
   - Features: vol, liq, OI_delta, FR at multiple timescales
   - Target: regime switch within 1/5/15 min
   - Compare: vol-only vs vol+liq vs vol+liq+OI+FR

5. NOVEL SIGNALS:
   - OI dropping + liq spike = cascade (positions being forced out)
   - OI rising + FR extreme = crowding (vulnerable to reversal)
   - OI flat + vol spike = stop-loss hunting (no new positions)

Data: BTC, May 12-18 2025 (7 days)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import time
import json
import gzip
import warnings
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/BTCUSDT")
SYMBOL = "BTCUSDT"


def mem_gb():
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label=""):
    used, avail = mem_gb()
    print(f"  [RAM] used={used:.1f}GB avail={avail:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_liquidations_all(dates):
    all_liqs = []
    for date_str in dates:
        pattern = f"liquidation_{date_str}_hr*.jsonl.gz"
        files = sorted((DATA_DIR / "bybit" / "liquidations").glob(pattern))
        if not files:
            files = sorted(DATA_DIR.glob(pattern))
        day_count = 0
        for f in files:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for ev in data['result']['data']:
                                all_liqs.append({
                                    'ts_ms': int(ev['T']),
                                    'side': ev['S'],
                                    'volume': float(ev['v']),
                                    'price': float(ev['p']),
                                })
                                day_count += 1
                    except:
                        continue
        print(f"  {date_str}: {day_count:>6,} liquidations", flush=True)
    if not all_liqs:
        return pd.DataFrame()
    liqs = pd.DataFrame(all_liqs)
    liqs['ts_s'] = liqs['ts_ms'] // 1000
    liqs['notional'] = liqs['volume'] * liqs['price']
    liqs['is_buy'] = (liqs['side'] == 'Buy').astype(np.int8)
    print(f"  Total: {len(liqs):,} liquidations")
    return liqs


def load_ticker_data(dates):
    """Load OI and FR from ticker websocket data."""
    oi_records = []
    fr_records = []

    for date_str in dates:
        pattern = f"ticker_{date_str}_hr*.jsonl.gz"
        files = sorted((DATA_DIR / "bybit" / "ticker").glob(pattern))
        day_oi = 0
        day_fr = 0
        for f in files:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        ts_ms = d.get('ts', 0)
                        data = d.get('result', {}).get('data', {})
                        if 'openInterestValue' in data:
                            oi_records.append({
                                'ts_ms': ts_ms,
                                'oi_value': float(data['openInterestValue']),
                            })
                            day_oi += 1
                        if 'fundingRate' in data:
                            fr_records.append({
                                'ts_ms': ts_ms,
                                'funding_rate': float(data['fundingRate']),
                            })
                            day_fr += 1
                    except:
                        continue
        print(f"  {date_str}: {day_oi:>6,} OI updates, {day_fr:>4,} FR updates", flush=True)

    oi_df = pd.DataFrame(oi_records) if oi_records else pd.DataFrame()
    fr_df = pd.DataFrame(fr_records) if fr_records else pd.DataFrame()

    if len(oi_df) > 0:
        oi_df['ts_s'] = oi_df['ts_ms'] // 1000
    if len(fr_df) > 0:
        fr_df['ts_s'] = fr_df['ts_ms'] // 1000

    print(f"  Total: {len(oi_df):,} OI updates, {len(fr_df):,} FR updates")
    return oi_df, fr_df


def aggregate_day_to_seconds(date_str, liqs):
    """Load one day of trades, aggregate to per-second arrays."""
    path = DATA_DIR / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        return None
    t0 = time.time()
    df = pd.read_csv(path, compression='gzip', usecols=['timestamp', 'price', 'size', 'side'])
    n_trades = len(df)
    ts_s = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.float64)
    notionals = prices * sizes
    del df; gc.collect()

    day_start = ts_s.min()
    day_end = ts_s.max()
    n_seconds = day_end - day_start + 1
    offsets = ts_s - day_start

    trade_count = np.bincount(offsets, minlength=n_seconds).astype(np.int32)
    trade_notional = np.bincount(offsets, weights=notionals, minlength=n_seconds)

    price_last = np.full(n_seconds, np.nan, dtype=np.float64)
    _, last_indices = np.unique(offsets[::-1], return_index=True)
    last_indices = len(offsets) - 1 - last_indices
    for ui, li in zip(np.unique(offsets), last_indices):
        price_last[ui] = prices[li]

    del ts_s, prices, sizes, notionals, offsets; gc.collect()

    # Forward-fill prices
    mask = np.isnan(price_last)
    if not mask.all():
        first_valid = np.argmin(mask)
        for i in range(first_valid + 1, n_seconds):
            if mask[i]:
                price_last[i] = price_last[i - 1]

    # Liquidation aggregation
    liq_count = np.zeros(n_seconds, dtype=np.int32)
    liq_notional = np.zeros(n_seconds, dtype=np.float64)
    if len(liqs) > 0:
        liq_day = liqs[(liqs['ts_s'] >= day_start) & (liqs['ts_s'] <= day_end)]
        if len(liq_day) > 0:
            l_offsets = (liq_day['ts_s'].values - day_start).astype(np.int64)
            valid = (l_offsets >= 0) & (l_offsets < n_seconds)
            l_offsets = l_offsets[valid]
            l_not = liq_day['notional'].values[valid]
            liq_count = np.bincount(l_offsets, minlength=n_seconds).astype(np.int32)
            liq_notional = np.bincount(l_offsets, weights=l_not, minlength=n_seconds)

    elapsed = time.time() - t0
    print(f"  {date_str}: {n_trades:,} trades → {n_seconds:,}s ({elapsed:.1f}s)", flush=True)
    return {
        'day_start': day_start, 'n_seconds': n_seconds,
        'trade_count': trade_count, 'trade_notional': trade_notional,
        'price_last': price_last,
        'liq_count': liq_count, 'liq_notional': liq_notional,
    }


def build_combined_series(dates, liqs, oi_df, fr_df):
    """Build full per-second timeseries with all 4 data streams."""
    # Process trades day by day
    day_results = []
    for date_str in dates:
        result = aggregate_day_to_seconds(date_str, liqs)
        if result is not None:
            day_results.append(result)
        gc.collect()

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n_seconds'] - 1
    n = ts_end - ts_start + 1

    # Stitch trade/liq arrays
    trade_count = np.zeros(n, dtype=np.int32)
    trade_notional = np.zeros(n, dtype=np.float64)
    price_last = np.full(n, np.nan, dtype=np.float64)
    liq_count = np.zeros(n, dtype=np.int32)
    liq_notional = np.zeros(n, dtype=np.float64)

    for dr in day_results:
        o = dr['day_start'] - ts_start
        l = dr['n_seconds']
        trade_count[o:o+l] = dr['trade_count']
        trade_notional[o:o+l] = dr['trade_notional']
        price_last[o:o+l] = dr['price_last']
        liq_count[o:o+l] = dr['liq_count']
        liq_notional[o:o+l] = dr['liq_notional']
    del day_results; gc.collect()

    # Forward-fill prices across day boundaries
    for i in range(1, n):
        if np.isnan(price_last[i]):
            price_last[i] = price_last[i - 1]

    # OI: map to per-second array (forward-fill)
    oi_arr = np.full(n, np.nan, dtype=np.float64)
    if len(oi_df) > 0:
        oi_ts = oi_df['ts_s'].values
        oi_val = oi_df['oi_value'].values
        for ts, val in zip(oi_ts, oi_val):
            idx = ts - ts_start
            if 0 <= idx < n:
                oi_arr[idx] = val
        # Forward-fill
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(oi_arr[i]):
                last_valid = oi_arr[i]
            elif not np.isnan(last_valid):
                oi_arr[i] = last_valid
        print(f"  OI: {np.sum(~np.isnan(oi_arr)):,} seconds with data")

    # FR: map to per-second array (forward-fill)
    fr_arr = np.full(n, np.nan, dtype=np.float64)
    if len(fr_df) > 0:
        fr_ts = fr_df['ts_s'].values
        fr_val = fr_df['funding_rate'].values
        for ts, val in zip(fr_ts, fr_val):
            idx = ts - ts_start
            if 0 <= idx < n:
                fr_arr[idx] = val
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(fr_arr[i]):
                last_valid = fr_arr[i]
            elif not np.isnan(last_valid):
                fr_arr[i] = last_valid
        print(f"  FR: {np.sum(~np.isnan(fr_arr)):,} seconds with data")

    print_mem("computing features")

    # Log returns
    log_ret = np.zeros(n, dtype=np.float64)
    valid = (price_last[1:] > 0) & (price_last[:-1] > 0)
    log_ret[1:][valid] = np.log(price_last[1:][valid] / price_last[:-1][valid])

    def fast_rolling_std(arr, window):
        cs = np.cumsum(arr); cs2 = np.cumsum(arr ** 2)
        result = np.full(len(arr), np.nan, dtype=np.float64)
        s = cs[window:] - cs[:-window]
        s2 = cs2[window:] - cs2[:-window]
        var = s2 / window - (s / window) ** 2
        np.clip(var, 0, None, out=var)
        result[window:] = np.sqrt(var)
        return result

    def fast_rolling_sum(arr, window):
        cs = np.cumsum(arr.astype(np.float64))
        result = np.zeros(len(arr), dtype=np.float64)
        result[window:] = cs[window:] - cs[:-window]
        return result

    def fast_rolling_mean(arr, window):
        cs = np.cumsum(np.nan_to_num(arr, nan=0.0).astype(np.float64))
        count = np.cumsum((~np.isnan(arr)).astype(np.float64))
        result = np.full(len(arr), np.nan, dtype=np.float64)
        s = cs[window:] - cs[:-window]
        c = count[window:] - count[:-window]
        valid = c > 0
        result[window:][valid] = s[valid] / c[valid]
        return result

    vol_60s = fast_rolling_std(log_ret, 60)
    liq_60s = fast_rolling_sum(liq_count, 60)
    liq_300s = fast_rolling_sum(liq_count, 300)
    liq_not_60s = fast_rolling_sum(liq_notional, 60)
    trade_10s = fast_rolling_sum(trade_count, 10)

    # OI features
    oi_delta_10s = np.zeros(n, dtype=np.float64)
    oi_delta_60s = np.zeros(n, dtype=np.float64)
    oi_delta_300s = np.zeros(n, dtype=np.float64)
    if not np.all(np.isnan(oi_arr)):
        oi_delta_10s[10:] = oi_arr[10:] - oi_arr[:-10]
        oi_delta_60s[60:] = oi_arr[60:] - oi_arr[:-60]
        oi_delta_300s[300:] = oi_arr[300:] - oi_arr[:-300]

    # FR features
    fr_60s = fast_rolling_mean(fr_arr, 60)
    fr_300s = fast_rolling_mean(fr_arr, 300)

    # FR distance to funding time (8h cycle: 00:00, 08:00, 16:00 UTC)
    seconds = np.arange(ts_start, ts_end + 1)
    time_in_day = seconds % 86400  # seconds since midnight UTC
    funding_times = np.array([0, 28800, 57600, 86400])  # 00:00, 08:00, 16:00, 24:00
    fr_time_to_funding = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tod = time_in_day[i]
        next_funding = funding_times[funding_times > tod]
        if len(next_funding) > 0:
            fr_time_to_funding[i] = next_funding[0] - tod
        else:
            fr_time_to_funding[i] = 86400 - tod

    data = {
        'ts_s': seconds, 'price': price_last, 'log_ret': log_ret,
        'vol_60s': vol_60s,
        'trade_count': trade_count, 'trade_10s': trade_10s,
        'liq_count': liq_count, 'liq_60s': liq_60s, 'liq_300s': liq_300s,
        'liq_notional': liq_notional, 'liq_not_60s': liq_not_60s,
        'oi': oi_arr, 'oi_delta_10s': oi_delta_10s,
        'oi_delta_60s': oi_delta_60s, 'oi_delta_300s': oi_delta_300s,
        'fr': fr_arr, 'fr_60s': fr_60s, 'fr_300s': fr_300s,
        'fr_time_to_funding': fr_time_to_funding,
    }

    sec_df = pd.DataFrame(data)
    del data; gc.collect()

    print(f"  Built {len(sec_df):,} seconds with all 4 streams")
    print_mem("done")
    return sec_df, ts_start


def detect_regime_switches(sec_df, vol_col='vol_60s', median_window=3600,
                           threshold=2.0, min_gap=1800):
    vol = sec_df[vol_col].values
    n = len(vol)
    vol_median = np.full(n, np.nan)
    for i in range(median_window, n):
        window = vol[i-median_window:i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > median_window // 2:
            vol_median[i] = np.median(valid_w)

    switches = []
    in_volatile = False
    for i in range(median_window + 1, n):
        if np.isnan(vol[i]) or np.isnan(vol_median[i]):
            continue
        if not in_volatile and vol[i] > vol_median[i] * threshold:
            if not switches or (sec_df['ts_s'].iloc[i] - switches[-1]) > min_gap:
                switches.append(sec_df['ts_s'].iloc[i])
            in_volatile = True
        elif in_volatile and vol[i] < vol_median[i] * 1.2:
            in_volatile = False
    return switches


# ============================================================================
# ANALYSIS 1: Combined profiles around switches
# ============================================================================

def analysis_combined_profiles(sec_df, switches):
    """Profile all 4 signals ±30min around regime switches."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS 1: Combined 4-Stream Profiles Around {len(switches)} Switches (±30min)")
    print(f"{'='*80}")

    window = 1800  # ±30min
    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    cols = ['vol_60s', 'liq_60s', 'liq_not_60s', 'trade_10s',
            'oi_delta_60s', 'oi_delta_300s', 'fr', 'fr_time_to_funding']
    profiles = {c: [] for c in cols}

    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < window or idx + window >= len(ts_arr):
            continue
        for c in cols:
            profiles[c].append(sec_df[c].values[idx-window:idx+window].copy())

    n_valid = len(profiles['vol_60s'])
    print(f"  Valid switches: {n_valid}")

    avgs = {c: np.nanmean(profiles[c], axis=0) for c in cols}

    # Print at 30-second resolution
    step = 30
    print(f"\n  {'Offset':>8s}  {'Vol_60s':>10s}  {'Liq_60s':>8s}  {'LiqNot$':>10s}  "
          f"{'Trades/10s':>10s}  {'OI_Δ60s$':>12s}  {'OI_Δ5m$':>12s}  "
          f"{'FR':>10s}  {'ToFunding':>10s}")
    print(f"  {'-'*105}")

    for i in range(0, len(avgs['vol_60s']), step):
        offset = i - window
        marker = " ← SWITCH" if abs(offset) < step // 2 else ""
        vol = avgs['vol_60s'][i]
        liq = avgs['liq_60s'][i]
        liq_not = avgs['liq_not_60s'][i]
        trd = avgs['trade_10s'][i]
        oi60 = avgs['oi_delta_60s'][i]
        oi300 = avgs['oi_delta_300s'][i]
        fr = avgs['fr'][i]
        ttf = avgs['fr_time_to_funding'][i]

        vol_s = f"{vol:.6f}" if not np.isnan(vol) else "       nan"
        fr_s = f"{fr:.8f}" if not np.isnan(fr) else "       nan"

        print(f"  {offset:>+7d}s  {vol_s:>10s}  {liq:>8.1f}  ${liq_not:>9,.0f}  "
              f"{trd:>10.0f}  ${oi60:>11,.0f}  ${oi300:>11,.0f}  "
              f"{fr_s:>10s}  {ttf/60:>8.0f}min{marker}")

    # Key metrics summary
    print(f"\n  --- Key Metrics at Switch vs Baseline ---")
    baseline_slice = slice(0, window // 2)  # -30min to -15min
    switch_slice = slice(window - 5, window + 5)  # ±5s around switch

    for name, col in [('Vol_60s', 'vol_60s'), ('Liq_60s', 'liq_60s'),
                       ('LiqNot_60s', 'liq_not_60s'), ('OI_Δ60s', 'oi_delta_60s'),
                       ('OI_Δ5min', 'oi_delta_300s'), ('FR', 'fr')]:
        baseline = np.nanmean(avgs[col][baseline_slice])
        at_switch = np.nanmean(avgs[col][switch_slice])
        if baseline != 0 and not np.isnan(baseline):
            ratio = at_switch / baseline
            print(f"    {name:>12s}: baseline={baseline:>12.4f}  switch={at_switch:>12.4f}  ratio={ratio:>6.2f}x")
        else:
            print(f"    {name:>12s}: baseline={baseline:>12.4f}  switch={at_switch:>12.4f}")


# ============================================================================
# ANALYSIS 2: OI behavior around switches
# ============================================================================

def analysis_oi_around_switches(sec_df, switches):
    """Deep dive into OI dynamics around regime switches."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS 2: Open Interest Dynamics Around Switches")
    print(f"{'='*80}")

    window = 1800
    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    oi = sec_df['oi'].values
    oi_d60 = sec_df['oi_delta_60s'].values
    oi_d300 = sec_df['oi_delta_300s'].values

    # Classify OI behavior at each switch
    oi_rising = 0; oi_falling = 0; oi_flat = 0
    oi_deltas_at_switch = []
    oi_deltas_before = []

    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < window or idx + window >= len(ts_arr):
            continue

        # OI change in 5 minutes BEFORE switch
        d300_before = oi_d300[idx]
        oi_deltas_before.append(d300_before)

        # OI change in 60s around switch
        d60_at = oi_d60[idx]
        oi_deltas_at_switch.append(d60_at)

        if d300_before > 1e6:
            oi_rising += 1
        elif d300_before < -1e6:
            oi_falling += 1
        else:
            oi_flat += 1

    total = oi_rising + oi_falling + oi_flat
    print(f"\n  OI behavior in 5min before switch ({total} switches):")
    print(f"    OI rising  (>$1M):  {oi_rising:>4d} ({oi_rising/total*100:.1f}%)")
    print(f"    OI falling (<-$1M): {oi_falling:>4d} ({oi_falling/total*100:.1f}%)")
    print(f"    OI flat    (±$1M):  {oi_flat:>4d} ({oi_flat/total*100:.1f}%)")

    oi_before = np.array(oi_deltas_before)
    oi_at = np.array(oi_deltas_at_switch)
    print(f"\n  OI_Δ5min before switch: mean=${np.mean(oi_before):+,.0f}, median=${np.median(oi_before):+,.0f}")
    print(f"  OI_Δ60s at switch:     mean=${np.mean(oi_at):+,.0f}, median=${np.median(oi_at):+,.0f}")

    # Compare to random
    excl_mask = np.zeros(len(ts_arr), dtype=bool)
    valid_indices = []
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is not None and idx >= window and idx + window < len(ts_arr):
            valid_indices.append(idx)
            excl_mask[max(0, idx-1800):min(len(ts_arr), idx+1801)] = True

    np.random.seed(42)
    pool = np.arange(window, len(ts_arr) - window)
    pool = pool[~excl_mask[pool]]
    sample = np.random.choice(pool, size=min(len(pool), len(valid_indices) * 5), replace=False)

    random_d300 = oi_d300[sample]
    random_d60 = oi_d60[sample]

    valid_sw = ~np.isnan(oi_before) & np.isfinite(oi_before)
    valid_rnd = ~np.isnan(random_d300) & np.isfinite(random_d300)

    if valid_sw.sum() > 20 and valid_rnd.sum() > 20:
        # Test if OI change is different at switches
        stat, pval = mannwhitneyu(np.abs(oi_before[valid_sw]),
                                   np.abs(random_d300[valid_rnd]),
                                   alternative='greater')
        print(f"\n  |OI_Δ5min| at switch vs random: p={pval:.6f} "
              f"{'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")
        print(f"    Switch |OI_Δ5min| mean: ${np.mean(np.abs(oi_before[valid_sw])):,.0f}")
        print(f"    Random |OI_Δ5min| mean: ${np.mean(np.abs(random_d300[valid_rnd])):,.0f}")

    # OI direction vs price direction at switch
    print(f"\n  OI direction vs price move at switch:")
    price_up = 0; price_down = 0
    oi_up_price_up = 0; oi_up_price_down = 0
    oi_down_price_up = 0; oi_down_price_down = 0

    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < 300 or idx + 60 >= len(ts_arr):
            continue
        price_change = sec_df['price'].iloc[idx + 30] - sec_df['price'].iloc[idx - 30]
        oi_change = oi_d300[idx]
        if np.isnan(oi_change):
            continue

        if price_change > 0:
            price_up += 1
            if oi_change > 0: oi_up_price_up += 1
            else: oi_down_price_up += 1
        else:
            price_down += 1
            if oi_change > 0: oi_up_price_down += 1
            else: oi_down_price_down += 1

    total_dir = price_up + price_down
    if total_dir > 0:
        print(f"    Price UP   switches: {price_up:>4d} — OI rising: {oi_up_price_up} ({oi_up_price_up/max(price_up,1)*100:.0f}%), OI falling: {oi_down_price_up} ({oi_down_price_up/max(price_up,1)*100:.0f}%)")
        print(f"    Price DOWN switches: {price_down:>4d} — OI rising: {oi_up_price_down} ({oi_up_price_down/max(price_down,1)*100:.0f}%), OI falling: {oi_down_price_down} ({oi_down_price_down/max(price_down,1)*100:.0f}%)")


# ============================================================================
# ANALYSIS 3: FR behavior around switches
# ============================================================================

def analysis_fr_around_switches(sec_df, switches):
    """Funding rate dynamics around regime switches."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS 3: Funding Rate Dynamics Around Switches")
    print(f"{'='*80}")

    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}
    fr = sec_df['fr'].values
    fr_300 = sec_df['fr_300s'].values
    ttf = sec_df['fr_time_to_funding'].values

    # FR at switch vs random
    switch_fr = []
    switch_ttf = []
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is not None and not np.isnan(fr[idx]):
            switch_fr.append(fr[idx])
            switch_ttf.append(ttf[idx])

    # Random sample
    window = 1800
    excl_mask = np.zeros(len(ts_arr), dtype=bool)
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is not None:
            excl_mask[max(0, idx-1800):min(len(ts_arr), idx+1801)] = True

    np.random.seed(42)
    pool = np.arange(window, len(ts_arr) - window)
    pool = pool[~excl_mask[pool]]
    sample = np.random.choice(pool, size=min(len(pool), len(switches) * 5), replace=False)

    random_fr = fr[sample]
    random_ttf = ttf[sample]

    switch_fr = np.array(switch_fr)
    switch_ttf = np.array(switch_ttf)
    random_fr = random_fr[~np.isnan(random_fr)]

    print(f"\n  FR at switch:  mean={np.mean(switch_fr):.8f}, median={np.median(switch_fr):.8f}, "
          f"std={np.std(switch_fr):.8f}")
    print(f"  FR at random:  mean={np.mean(random_fr):.8f}, median={np.median(random_fr):.8f}, "
          f"std={np.std(random_fr):.8f}")

    # Test if |FR| is higher at switches (more extreme = more crowded)
    stat, pval = mannwhitneyu(np.abs(switch_fr), np.abs(random_fr), alternative='greater')
    print(f"  |FR| at switch vs random: p={pval:.6f} "
          f"{'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")

    # Time to funding
    print(f"\n  Time to next funding at switch:")
    print(f"    Switch: mean={np.mean(switch_ttf)/60:.0f}min, median={np.median(switch_ttf)/60:.0f}min")
    print(f"    Random: mean={np.mean(random_ttf)/60:.0f}min, median={np.median(random_ttf)/60:.0f}min")

    # Do switches cluster near funding times?
    bins = [0, 1800, 3600, 7200, 14400, 28800]
    bin_labels = ['0-30m', '30m-1h', '1h-2h', '2h-4h', '4h-8h']
    sw_hist = np.histogram(switch_ttf, bins=bins)[0]
    rnd_hist = np.histogram(random_ttf, bins=bins)[0]

    print(f"\n  {'Time to funding':>15s}  {'Switches':>10s}  {'Random':>10s}  {'Lift':>8s}")
    print(f"  {'-'*50}")
    for i, label in enumerate(bin_labels):
        sw_pct = sw_hist[i] / len(switch_ttf) * 100
        rnd_pct = rnd_hist[i] / len(random_ttf) * 100
        lift = sw_pct / max(rnd_pct, 0.01)
        print(f"  {label:>15s}  {sw_pct:>9.1f}%  {rnd_pct:>9.1f}%  {lift:>7.2f}x")


# ============================================================================
# ANALYSIS 4: Cross-correlations between all pairs
# ============================================================================

def analysis_cross_correlations(sec_df):
    """Cross-correlation between all signal pairs at second resolution."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS 4: Cross-Correlations Between All Signal Pairs")
    print(f"{'='*80}")

    # Use 10-second sampled data to make xcorr tractable
    step = 10
    n = len(sec_df)
    indices = np.arange(0, n, step)

    signals = {
        'vol_60s': sec_df['vol_60s'].values[indices],
        'liq_60s': sec_df['liq_60s'].values[indices],
        'oi_Δ60s': sec_df['oi_delta_60s'].values[indices],
        'fr': sec_df['fr'].values[indices],
        'trades_10s': sec_df['trade_10s'].values[indices],
    }

    # Remove nans for correlation
    valid = np.ones(len(indices), dtype=bool)
    for s in signals.values():
        valid &= ~np.isnan(s) & np.isfinite(s)

    print(f"  Valid samples: {valid.sum():,} (at {step}s resolution)")

    # Spearman correlations (robust to outliers)
    print(f"\n  Spearman Rank Correlations (contemporaneous):")
    print(f"  {'':>12s}  {'vol_60s':>10s}  {'liq_60s':>10s}  {'oi_Δ60s':>10s}  {'fr':>10s}  {'trades':>10s}")
    print(f"  {'-'*65}")

    names = list(signals.keys())
    for i, n1 in enumerate(names):
        row = f"  {n1:>12s}"
        for j, n2 in enumerate(names):
            if i == j:
                row += f"  {'1.000':>10s}"
            else:
                v1 = signals[n1][valid]
                v2 = signals[n2][valid]
                rho, pval = spearmanr(v1, v2)
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                row += f"  {rho:>+8.4f}{sig:>2s}"
            pass
        print(row)

    # Lead-lag: cross-correlation at different lags
    print(f"\n  Lead-Lag Cross-Correlations (key pairs):")
    pairs = [
        ('oi_Δ60s', 'vol_60s', 'OI_delta → Vol'),
        ('oi_Δ60s', 'liq_60s', 'OI_delta → Liq'),
        ('fr', 'vol_60s', 'FR → Vol'),
        ('fr', 'liq_60s', 'FR → Liq'),
        ('liq_60s', 'vol_60s', 'Liq → Vol'),
    ]

    lags_to_test = [-300, -180, -120, -60, -30, -10, 0, 10, 30, 60, 120, 180, 300]

    for sig1_name, sig2_name, label in pairs:
        s1 = signals[sig1_name][valid]
        s2 = signals[sig2_name][valid]

        print(f"\n  {label}:")
        print(f"  {'Lag':>8s}  {'Spearman':>10s}  {'Bar':>20s}")
        print(f"  {'-'*42}")

        best_lag = 0; best_rho = 0
        for lag in lags_to_test:
            lag_samples = lag // step
            if lag_samples >= 0:
                v1 = s1[:len(s1)-lag_samples] if lag_samples > 0 else s1
                v2 = s2[lag_samples:] if lag_samples > 0 else s2
            else:
                v1 = s1[-lag_samples:]
                v2 = s2[:len(s2)+lag_samples]

            if len(v1) > 100:
                rho, _ = spearmanr(v1, v2)
                if abs(rho) > abs(best_rho):
                    best_rho = rho; best_lag = lag
                bar_len = int(abs(rho) * 40)
                bar = '█' * bar_len
                marker = " ← PEAK" if lag == 0 else ""
                print(f"  {lag*step:>+7d}s  {rho:>+10.4f}  {bar}{marker}")

        if best_lag != 0:
            print(f"  → Peak at lag={best_lag*step}s ({sig1_name} {'leads' if best_lag > 0 else 'lags'} {sig2_name})")


# ============================================================================
# ANALYSIS 5: Novel combined signals
# ============================================================================

def analysis_novel_signals(sec_df, switches):
    """Test novel combined signals from all 4 streams."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS 5: Novel Combined Signals")
    print(f"{'='*80}")

    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    vol = sec_df['vol_60s'].values
    liq = sec_df['liq_60s'].values
    oi_d = sec_df['oi_delta_300s'].values
    fr = sec_df['fr'].values

    # Compute percentiles for thresholds
    vol_p75 = np.nanpercentile(vol, 75)
    vol_p90 = np.nanpercentile(vol, 90)
    liq_p90 = np.nanpercentile(liq[liq > 0], 90) if np.any(liq > 0) else 1
    liq_p95 = np.nanpercentile(liq[liq > 0], 95) if np.any(liq > 0) else 1
    oi_d_p10 = np.nanpercentile(oi_d[np.isfinite(oi_d)], 10)
    oi_d_p90 = np.nanpercentile(oi_d[np.isfinite(oi_d)], 90)
    fr_p90 = np.nanpercentile(np.abs(fr[~np.isnan(fr)]), 90)

    print(f"\n  Thresholds:")
    print(f"    vol P75={vol_p75:.6f}, P90={vol_p90:.6f}")
    print(f"    liq (>0) P90={liq_p90:.0f}, P95={liq_p95:.0f}")
    print(f"    OI_Δ5m P10=${oi_d_p10:+,.0f}, P90=${oi_d_p90:+,.0f}")
    print(f"    |FR| P90={fr_p90:.8f}")

    # Build switch lookup (is there a switch within next T seconds?)
    switch_set = set(switches)

    def has_switch_within(idx, horizon):
        for h in range(0, horizon):
            if idx + h < len(ts_arr) and ts_arr[idx + h] in switch_set:
                return True
        return False

    # Define novel signal conditions
    signals = {
        'OI_drop + Liq_spike': lambda i: (oi_d[i] < oi_d_p10) & (liq[i] >= liq_p90),
        'OI_rise + FR_extreme': lambda i: (oi_d[i] > oi_d_p90) & (np.abs(fr[i]) > fr_p90),
        'Liq_spike + FR_extreme': lambda i: (liq[i] >= liq_p90) & (np.abs(fr[i]) > fr_p90),
        'OI_drop + Liq + FR_ext': lambda i: (oi_d[i] < oi_d_p10) & (liq[i] >= liq_p90) & (np.abs(fr[i]) > fr_p90),
        'Liq_spike_only': lambda i: liq[i] >= liq_p95,
        'OI_drop_only': lambda i: oi_d[i] < oi_d_p10,
        'FR_extreme_only': lambda i: np.abs(fr[i]) > fr_p90,
    }

    # Test each signal: when it fires, P(switch within 60s/300s/900s)?
    # Use last 30% as test set
    split = int(len(ts_arr) * 0.7)
    test_indices = np.arange(split, len(ts_arr) - 900)

    # Precompute: for each second in test set, is there a switch within horizon?
    print(f"\n  Precomputing switch proximity for test set ({len(test_indices):,} seconds)...")
    switch_within = {}
    for horizon in [60, 300, 900]:
        sw_arr = np.zeros(len(ts_arr), dtype=bool)
        for sw in switches:
            idx = ts_to_idx.get(sw)
            if idx is not None:
                lo = max(0, idx - horizon)
                sw_arr[lo:idx+1] = True
        switch_within[horizon] = sw_arr

    base_rates = {h: switch_within[h][test_indices].mean() for h in [60, 300, 900]}
    print(f"  Base rates: 60s={base_rates[60]:.4f}, 300s={base_rates[300]:.4f}, 900s={base_rates[900]:.4f}")

    print(f"\n  {'Signal':>25s}  {'Fires':>8s}  {'P(sw 1m)':>10s}  {'Lift':>6s}  "
          f"{'P(sw 5m)':>10s}  {'Lift':>6s}  {'P(sw 15m)':>10s}  {'Lift':>6s}")
    print(f"  {'-'*95}")

    for name, condition in signals.items():
        # Find where signal fires in test set
        fires = []
        for i in test_indices:
            try:
                if condition(i) and not np.isnan(vol[i]):
                    fires.append(i)
            except:
                continue

        if len(fires) < 10:
            print(f"  {name:>25s}  {len(fires):>8d}  (too few)")
            continue

        fires = np.array(fires)
        results = {}
        for horizon in [60, 300, 900]:
            hit_rate = switch_within[horizon][fires].mean()
            lift = hit_rate / max(base_rates[horizon], 1e-6)
            results[horizon] = (hit_rate, lift)

        print(f"  {name:>25s}  {len(fires):>8d}  "
              f"{results[60][0]:>10.4f}  {results[60][1]:>5.1f}x  "
              f"{results[300][0]:>10.4f}  {results[300][1]:>5.1f}x  "
              f"{results[900][0]:>10.4f}  {results[900][1]:>5.1f}x")


# ============================================================================
# ANALYSIS 6: Combined predictive model
# ============================================================================

def analysis_predictive_model(sec_df, switches):
    """Compare prediction with different feature sets."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS 6: Combined Predictive Model — Feature Set Comparison")
    print(f"{'='*80}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler

    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    # Target: switch within 300s (5min)
    horizon = 300
    target = np.zeros(len(ts_arr), dtype=np.int8)
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is not None:
            lo = max(0, idx - horizon)
            target[lo:idx+1] = 1

    # Feature sets
    feature_sets = {
        'vol_only': ['vol_60s'],
        'vol+liq': ['vol_60s', 'liq_60s', 'liq_300s'],
        'vol+liq+OI': ['vol_60s', 'liq_60s', 'liq_300s', 'oi_delta_60s', 'oi_delta_300s'],
        'vol+liq+OI+FR': ['vol_60s', 'liq_60s', 'liq_300s', 'oi_delta_60s', 'oi_delta_300s', 'fr', 'fr_300s'],
        'all_features': ['vol_60s', 'liq_60s', 'liq_300s', 'liq_not_60s',
                         'oi_delta_60s', 'oi_delta_300s', 'fr', 'fr_300s',
                         'fr_time_to_funding', 'trade_10s'],
    }

    # Sample at 10-second intervals to reduce data size
    step = 10
    indices = np.arange(3600, len(ts_arr) - horizon, step)  # skip first hour (warmup)

    split = int(len(indices) * 0.7)
    train_idx = indices[:split]
    test_idx = indices[split:]

    print(f"  Target: switch within {horizon}s")
    print(f"  Train: {len(train_idx):,} samples, Test: {len(test_idx):,} samples")
    print(f"  Base rate (test): {target[test_idx].mean():.4f}")

    print(f"\n  {'Feature Set':>20s}  {'AUC':>8s}  {'Prec@P90':>10s}  {'Recall':>8s}  {'Features':>10s}")
    print(f"  {'-'*65}")

    for fs_name, fs_cols in feature_sets.items():
        # Build feature matrix
        X_train = np.column_stack([sec_df[c].values[train_idx] for c in fs_cols])
        X_test = np.column_stack([sec_df[c].values[test_idx] for c in fs_cols])
        y_train = target[train_idx]
        y_test = target[test_idx]

        # Remove rows with nan/inf
        valid_train = np.all(np.isfinite(X_train), axis=1)
        valid_test = np.all(np.isfinite(X_test), axis=1)
        X_train = X_train[valid_train]; y_train = y_train[valid_train]
        X_test = X_test[valid_test]; y_test = y_test[valid_test]

        if len(X_train) < 100 or len(X_test) < 100 or y_train.sum() < 10:
            print(f"  {fs_name:>20s}  insufficient data")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Gradient Boosting (handles nonlinear interactions)
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        # Precision at P90 threshold (top 10% of predictions)
        threshold = np.percentile(y_prob, 90)
        y_pred_p90 = (y_prob >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_p90, zero_division=0)
        rec = recall_score(y_test, y_pred_p90, zero_division=0)

        print(f"  {fs_name:>20s}  {auc:>8.4f}  {prec:>10.4f}  {rec:>8.4f}  {len(fs_cols):>10d}")

        # Feature importance for the full model
        if fs_name == 'all_features':
            print(f"\n  Feature Importance (all_features):")
            importances = model.feature_importances_
            for fname, imp in sorted(zip(fs_cols, importances), key=lambda x: -x[1]):
                bar = '█' * int(imp * 50)
                print(f"    {fname:>20s}  {imp:.4f}  {bar}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()
    dates = [f"2025-05-{d:02d}" for d in range(12, 19)]

    print(f"\n{'='*80}")
    print(f"  v29: COMBINED TICK-LEVEL ANALYSIS — {SYMBOL}")
    print(f"  Streams: Trades + Liquidations + Open Interest + Funding Rate")
    print(f"  Dates: {dates[0]} to {dates[-1]} (7 days)")
    print(f"{'='*80}")
    print_mem("start")

    # Load all data sources
    print(f"\n  --- Loading Liquidations ---")
    liqs = load_liquidations_all(dates)

    print(f"\n  --- Loading Ticker (OI + FR) ---")
    oi_df, fr_df = load_ticker_data(dates)
    print_mem("after data load")

    # Build combined timeseries
    print(f"\n  --- Building Combined Second-Level Timeseries ---")
    t0 = time.time()
    sec_df, ts_start_val = build_combined_series(dates, liqs, oi_df, fr_df)
    del liqs, oi_df, fr_df; gc.collect()
    print(f"  Completed in {time.time()-t0:.1f}s")

    # Quick data summary
    print(f"\n  --- Data Summary ---")
    for col in ['vol_60s', 'liq_60s', 'oi', 'oi_delta_60s', 'fr']:
        vals = sec_df[col].values
        valid = ~np.isnan(vals) & np.isfinite(vals)
        if valid.sum() > 0:
            print(f"    {col:>15s}: mean={np.mean(vals[valid]):>12.4f}, "
                  f"std={np.std(vals[valid]):>12.4f}, "
                  f"valid={valid.sum():,}/{len(vals):,}")

    # Detect regime switches
    print(f"\n  --- Detecting Regime Switches ---")
    switches = detect_regime_switches(sec_df)
    print(f"  Found {len(switches)} regime switches (min_gap=30min)")

    if len(switches) < 10:
        switches = detect_regime_switches(sec_df, threshold=1.5)
        print(f"  Retried with threshold=1.5: {len(switches)} switches")

    print_mem("before analyses")

    # Run all analyses
    analysis_combined_profiles(sec_df, switches)
    analysis_oi_around_switches(sec_df, switches)
    analysis_fr_around_switches(sec_df, switches)
    analysis_cross_correlations(sec_df)
    analysis_novel_signals(sec_df, switches)
    analysis_predictive_model(sec_df, switches)

    print_mem("all done")
    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"  Complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
