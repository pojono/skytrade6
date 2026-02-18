#!/usr/bin/env python3
"""
v40: Regime-Switch Causal Chain Validation & Trading Strategy

GOALS:
======
1. VALIDATE the causal chain from v29 on ALL 5 symbols × 64 days:
   - FR cycle → positioning pressure (2-4h before funding)
   - OI rises slightly (minutes before)
   - Liquidations fire (-30s)
   - Vol threshold crossed → REGIME SWITCH (0s)
   - Cascade peak (+30s): liq spike, OI dropping fast
   - Unwind (+30s to +150s): OI drops total
   - Stabilization (+5 to +8min)

2. BUILD a profitable trading strategy based on regime switches:
   - Entry: detect regime switch via liq spike + vol threshold
   - Direction: use liq side imbalance (sell-liq dominant → price dropping → short bounce)
   - Confirmation: OI dropping fast = real cascade (not fake-out)
   - Exit: fixed time or vol decay

Data: 5 symbols × ~64 days (May 12 – Aug 7, 2025)
     BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT
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
from collections import defaultdict

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("data")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]

# Common date range across all symbols (liq + ticker available)
START_DATE = "2025-05-12"
END_DATE = "2025-08-07"

# Regime detection params (from v29)
VOL_WINDOW = 60        # 60s realized vol
MEDIAN_WINDOW = 3600   # 1h rolling median for baseline
SWITCH_THRESHOLD = 2.0 # vol > 2x median = regime switch
MIN_GAP = 1800         # 30min minimum between switches

# Strategy params
STRATEGY_FEE_BPS = 4.0                 # 2bps maker each way = 4bps round trip


def mem_gb():
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label=""):
    used, avail = mem_gb()
    print(f"  [RAM] used={used:.1f}GB avail={avail:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING (per-symbol, day-by-day to manage memory)
# ============================================================================

def get_dates_list():
    """Generate list of date strings in range."""
    dates = pd.date_range(START_DATE, END_DATE)
    return [d.strftime("%Y-%m-%d") for d in dates]


def load_liquidations_day(symbol, date_str):
    """Load liquidation events for one day."""
    sym_dir = DATA_DIR / symbol
    # Try bybit subdirectory first, then root
    patterns = [
        sym_dir / "bybit" / "liquidations" / f"liquidation_{date_str}_hr*.jsonl.gz",
    ]
    files = []
    for pat in patterns:
        files = sorted(pat.parent.glob(pat.name))
        if files:
            break
    if not files:
        # Try root level
        files = sorted(sym_dir.glob(f"liquidation_{date_str}_hr*.jsonl.gz"))

    records = []
    for f in files:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for ev in data['result']['data']:
                            records.append({
                                'ts_ms': int(ev['T']),
                                'side': ev['S'],
                                'volume': float(ev['v']),
                                'price': float(ev['p']),
                            })
                except:
                    continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df['ts_s'] = df['ts_ms'] // 1000
    df['notional'] = df['volume'] * df['price']
    df['is_sell'] = (df['side'] == 'Sell').astype(np.int8)
    return df


def load_ticker_day(symbol, date_str):
    """Load OI and FR from ticker data for one day."""
    sym_dir = DATA_DIR / symbol
    files = sorted((sym_dir / "bybit" / "ticker").glob(f"ticker_{date_str}_hr*.jsonl.gz"))
    if not files:
        files = sorted(sym_dir.glob(f"ticker_{date_str}_hr*.jsonl.gz"))

    oi_records = []
    fr_records = []
    for f in files:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    ts_ms = d.get('ts', 0)
                    data = d.get('result', {}).get('data', {})
                    if 'openInterestValue' in data:
                        oi_records.append((ts_ms, float(data['openInterestValue'])))
                    if 'fundingRate' in data:
                        fr_records.append((ts_ms, float(data['fundingRate'])))
                except:
                    continue

    oi_df = pd.DataFrame(oi_records, columns=['ts_ms', 'oi_value']) if oi_records else pd.DataFrame()
    fr_df = pd.DataFrame(fr_records, columns=['ts_ms', 'funding_rate']) if fr_records else pd.DataFrame()
    if len(oi_df) > 0:
        oi_df['ts_s'] = oi_df['ts_ms'] // 1000
    if len(fr_df) > 0:
        fr_df['ts_s'] = fr_df['ts_ms'] // 1000
    return oi_df, fr_df


def load_trades_day(symbol, date_str):
    """Load trade data for one day from csv.gz."""
    sym_dir = DATA_DIR / symbol / "bybit" / "futures"
    path = sym_dir / f"{symbol}{date_str}.csv.gz"
    if not path.exists():
        return None, 0
    df = pd.read_csv(path, compression='gzip', usecols=['timestamp', 'price', 'size', 'side'])
    n = len(df)
    return df, n


def build_second_series_day(symbol, date_str, liqs_day, oi_day, fr_day):
    """Build per-second arrays for one day. Returns dict of numpy arrays."""
    trades_df, n_trades = load_trades_day(symbol, date_str)
    if trades_df is None or n_trades == 0:
        return None

    ts_s = trades_df['timestamp'].values.astype(np.int64)
    prices = trades_df['price'].values.astype(np.float64)
    sizes = trades_df['size'].values.astype(np.float64)
    notionals = prices * sizes
    del trades_df; gc.collect()

    day_start = int(ts_s.min())
    day_end = int(ts_s.max())
    n_seconds = day_end - day_start + 1
    if n_seconds <= 0 or n_seconds > 100000:
        return None

    offsets = (ts_s - day_start).astype(np.int64)

    trade_count = np.bincount(offsets, minlength=n_seconds).astype(np.int32)
    trade_notional = np.bincount(offsets, weights=notionals, minlength=n_seconds)

    # Last price per second
    price_last = np.full(n_seconds, np.nan, dtype=np.float64)
    _, last_indices = np.unique(offsets[::-1], return_index=True)
    last_indices = len(offsets) - 1 - last_indices
    unique_offsets = np.unique(offsets)
    for ui, li in zip(unique_offsets, last_indices):
        if 0 <= ui < n_seconds:
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
    liq_sell_count = np.zeros(n_seconds, dtype=np.int32)
    liq_buy_count = np.zeros(n_seconds, dtype=np.int32)
    if len(liqs_day) > 0:
        ld = liqs_day[(liqs_day['ts_s'] >= day_start) & (liqs_day['ts_s'] <= day_end)]
        if len(ld) > 0:
            l_off = (ld['ts_s'].values - day_start).astype(np.int64)
            valid = (l_off >= 0) & (l_off < n_seconds)
            l_off = l_off[valid]
            l_not = ld['notional'].values[valid]
            l_sell = ld['is_sell'].values[valid]
            liq_count = np.bincount(l_off, minlength=n_seconds).astype(np.int32)
            liq_notional = np.bincount(l_off, weights=l_not, minlength=n_seconds)
            liq_sell_count = np.bincount(l_off, weights=l_sell.astype(np.float64), minlength=n_seconds).astype(np.int32)
            liq_buy_count = liq_count - liq_sell_count

    # OI: map to per-second (forward-fill)
    oi_arr = np.full(n_seconds, np.nan, dtype=np.float64)
    if len(oi_day) > 0:
        oi_ts = oi_day['ts_s'].values
        oi_val = oi_day['oi_value'].values
        for ts, val in zip(oi_ts, oi_val):
            idx = ts - day_start
            if 0 <= idx < n_seconds:
                oi_arr[idx] = val
        last_valid = np.nan
        for i in range(n_seconds):
            if not np.isnan(oi_arr[i]):
                last_valid = oi_arr[i]
            elif not np.isnan(last_valid):
                oi_arr[i] = last_valid

    # FR: map to per-second (forward-fill)
    fr_arr = np.full(n_seconds, np.nan, dtype=np.float64)
    if len(fr_day) > 0:
        fr_ts = fr_day['ts_s'].values
        fr_val = fr_day['funding_rate'].values
        for ts, val in zip(fr_ts, fr_val):
            idx = ts - day_start
            if 0 <= idx < n_seconds:
                fr_arr[idx] = val
        last_valid = np.nan
        for i in range(n_seconds):
            if not np.isnan(fr_arr[i]):
                last_valid = fr_arr[i]
            elif not np.isnan(last_valid):
                fr_arr[i] = last_valid

    return {
        'day_start': day_start,
        'n_seconds': n_seconds,
        'n_trades': n_trades,
        'trade_count': trade_count,
        'trade_notional': trade_notional,
        'price_last': price_last,
        'liq_count': liq_count,
        'liq_notional': liq_notional,
        'liq_sell_count': liq_sell_count,
        'liq_buy_count': liq_buy_count,
        'oi': oi_arr,
        'fr': fr_arr,
    }


def fast_rolling_std(arr, window):
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
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


def stitch_days_and_compute_features(day_results):
    """Stitch day-level arrays into continuous series and compute features."""
    if not day_results:
        return None

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n_seconds'] - 1
    n = ts_end - ts_start + 1

    trade_count = np.zeros(n, dtype=np.int32)
    trade_notional = np.zeros(n, dtype=np.float64)
    price_last = np.full(n, np.nan, dtype=np.float64)
    liq_count = np.zeros(n, dtype=np.int32)
    liq_notional = np.zeros(n, dtype=np.float64)
    liq_sell_count = np.zeros(n, dtype=np.int32)
    liq_buy_count = np.zeros(n, dtype=np.int32)
    oi_arr = np.full(n, np.nan, dtype=np.float64)
    fr_arr = np.full(n, np.nan, dtype=np.float64)

    total_trades = 0
    for dr in day_results:
        o = dr['day_start'] - ts_start
        l = dr['n_seconds']
        trade_count[o:o+l] = dr['trade_count']
        trade_notional[o:o+l] = dr['trade_notional']
        price_last[o:o+l] = dr['price_last']
        liq_count[o:o+l] = dr['liq_count']
        liq_notional[o:o+l] = dr['liq_notional']
        liq_sell_count[o:o+l] = dr['liq_sell_count']
        liq_buy_count[o:o+l] = dr['liq_buy_count']
        oi_arr[o:o+l] = dr['oi']
        fr_arr[o:o+l] = dr['fr']
        total_trades += dr['n_trades']
    del day_results; gc.collect()

    # Forward-fill across day boundaries
    for i in range(1, n):
        if np.isnan(price_last[i]) and not np.isnan(price_last[i-1]):
            price_last[i] = price_last[i-1]
        if np.isnan(oi_arr[i]) and not np.isnan(oi_arr[i-1]):
            oi_arr[i] = oi_arr[i-1]
        if np.isnan(fr_arr[i]) and not np.isnan(fr_arr[i-1]):
            fr_arr[i] = fr_arr[i-1]

    # Log returns
    log_ret = np.zeros(n, dtype=np.float64)
    valid = (price_last[1:] > 0) & (price_last[:-1] > 0)
    log_ret[1:][valid] = np.log(price_last[1:][valid] / price_last[:-1][valid])

    # Rolling features
    vol_60s = fast_rolling_std(log_ret, 60)
    liq_60s = fast_rolling_sum(liq_count, 60)
    liq_300s = fast_rolling_sum(liq_count, 300)
    liq_not_60s = fast_rolling_sum(liq_notional, 60)
    liq_sell_60s = fast_rolling_sum(liq_sell_count, 60)
    liq_buy_60s = fast_rolling_sum(liq_buy_count, 60)

    # OI deltas
    oi_delta_60s = np.zeros(n, dtype=np.float64)
    oi_delta_300s = np.zeros(n, dtype=np.float64)
    if not np.all(np.isnan(oi_arr)):
        oi_delta_60s[60:] = oi_arr[60:] - oi_arr[:-60]
        oi_delta_300s[300:] = oi_arr[300:] - oi_arr[:-300]

    # FR time to funding (8h cycle: 00:00, 08:00, 16:00 UTC)
    seconds = np.arange(ts_start, ts_end + 1)
    time_in_day = seconds % 86400
    funding_times = np.array([0, 28800, 57600, 86400])
    fr_time_to_funding = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tod = time_in_day[i]
        nf = funding_times[funding_times > tod]
        fr_time_to_funding[i] = nf[0] - tod if len(nf) > 0 else 86400 - tod

    data = {
        'ts_s': seconds,
        'price': price_last,
        'log_ret': log_ret,
        'vol_60s': vol_60s,
        'trade_count': trade_count,
        'liq_count': liq_count,
        'liq_60s': liq_60s,
        'liq_300s': liq_300s,
        'liq_not_60s': liq_not_60s,
        'liq_sell_60s': liq_sell_60s,
        'liq_buy_60s': liq_buy_60s,
        'liq_notional': liq_notional,
        'oi': oi_arr,
        'oi_delta_60s': oi_delta_60s,
        'oi_delta_300s': oi_delta_300s,
        'fr': fr_arr,
        'fr_time_to_funding': fr_time_to_funding,
    }

    return data, ts_start, n, total_trades


def detect_regime_switches(data, n, threshold=None):
    """Detect regime switches using vol threshold crossing."""
    if threshold is None:
        threshold = SWITCH_THRESHOLD
    vol = data['vol_60s']
    ts_s = data['ts_s']

    # Compute rolling median of vol (1h window) — chunked for speed
    vol_median = np.full(n, np.nan)
    chunk = MEDIAN_WINDOW
    # Compute median every 300s (5min) for speed, forward-fill between
    step = 300
    for i in range(chunk, n, step):
        window = vol[max(0, i-chunk):i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > chunk // 4:
            vol_median[i] = np.median(valid_w)

    # Fill forward
    for i in range(1, n):
        if np.isnan(vol_median[i]) and not np.isnan(vol_median[i-1]):
            vol_median[i] = vol_median[i-1]

    switches = []
    in_volatile = False
    for i in range(chunk + 1, n):
        if np.isnan(vol[i]) or np.isnan(vol_median[i]):
            continue
        if not in_volatile and vol[i] > vol_median[i] * threshold:
            if not switches or (ts_s[i] - switches[-1]) > MIN_GAP:
                switches.append(int(ts_s[i]))
            in_volatile = True
        elif in_volatile and vol[i] < vol_median[i] * 1.2:
            in_volatile = False

    return switches


# ============================================================================
# PART 1: CAUSAL CHAIN VALIDATION
# ============================================================================

def validate_causal_chain(data, switches, n, symbol):
    """Validate the complete causal chain around regime switches."""
    print(f"\n{'='*80}")
    print(f"  CAUSAL CHAIN VALIDATION — {symbol} ({len(switches)} switches)")
    print(f"{'='*80}")

    if len(switches) < 5:
        print(f"  Too few switches ({len(switches)}), skipping")
        return {}

    window = 1800  # ±30min
    ts_arr = data['ts_s']
    ts_to_idx = {}
    for i in range(n):
        ts_to_idx[int(ts_arr[i])] = i

    cols = ['vol_60s', 'liq_60s', 'liq_not_60s', 'oi_delta_60s', 'oi_delta_300s',
            'fr', 'fr_time_to_funding', 'liq_sell_60s', 'liq_buy_60s']
    profiles = {c: [] for c in cols}
    price_profiles = []

    valid_count = 0
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < window or idx + window >= n:
            continue
        valid_count += 1
        for c in cols:
            profiles[c].append(data[c][idx-window:idx+window].copy())
        # Price change relative to switch
        p_slice = data['price'][idx-window:idx+window].copy()
        if not np.isnan(p_slice[window]):
            p_slice = (p_slice / p_slice[window] - 1) * 100  # % change from switch
            price_profiles.append(p_slice)

    if valid_count < 5:
        print(f"  Only {valid_count} valid switches, skipping")
        return {}

    avgs = {c: np.nanmean(profiles[c], axis=0) for c in cols}
    if price_profiles:
        avg_price = np.nanmean(price_profiles, axis=0)

    # === Validate each element of the causal chain ===
    results = {'symbol': symbol, 'n_switches': valid_count}

    # 1. Liq spike at -30s (before vol threshold)
    baseline_liq = np.nanmean(avgs['liq_60s'][:window//2])
    liq_at_minus30 = avgs['liq_60s'][window - 30]
    liq_at_switch = avgs['liq_60s'][window]
    liq_at_plus30 = avgs['liq_60s'][window + 30]
    results['liq_baseline'] = baseline_liq
    results['liq_minus30'] = liq_at_minus30
    results['liq_at_switch'] = liq_at_switch
    results['liq_plus30'] = liq_at_plus30
    results['liq_lead_ratio'] = liq_at_minus30 / max(baseline_liq, 0.01)
    results['liq_peak_ratio'] = liq_at_plus30 / max(baseline_liq, 0.01)

    # 2. Vol spike at switch
    baseline_vol = np.nanmean(avgs['vol_60s'][:window//2])
    vol_at_switch = avgs['vol_60s'][window]
    results['vol_baseline'] = baseline_vol
    results['vol_at_switch'] = vol_at_switch
    results['vol_ratio'] = vol_at_switch / max(baseline_vol, 1e-10)

    # 3. OI drop after switch (cascade unwind)
    baseline_oi_d60 = np.nanmean(avgs['oi_delta_60s'][:window//2])
    oi_d60_at_switch = avgs['oi_delta_60s'][window]
    oi_d60_at_plus30 = avgs['oi_delta_60s'][window + 30]
    oi_d300_at_plus150 = avgs['oi_delta_300s'][window + 150]
    results['oi_d60_baseline'] = baseline_oi_d60
    results['oi_d60_at_switch'] = oi_d60_at_switch
    results['oi_d60_at_plus30'] = oi_d60_at_plus30
    results['oi_d300_at_plus150'] = oi_d300_at_plus150
    results['oi_drop_ratio'] = oi_d60_at_switch / max(abs(baseline_oi_d60), 1)

    # 4. FR timing — cluster 2-4h before funding
    switch_ttf = []
    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is not None:
            switch_ttf.append(data['fr_time_to_funding'][idx])

    if switch_ttf:
        switch_ttf = np.array(switch_ttf)
        # Count in 2-4h bucket vs expected uniform
        in_2_4h = np.sum((switch_ttf >= 7200) & (switch_ttf < 14400))
        total = len(switch_ttf)
        expected_pct = (14400 - 7200) / 28800  # 25% of 8h cycle
        actual_pct = in_2_4h / total
        results['fr_2_4h_pct'] = actual_pct
        results['fr_2_4h_lift'] = actual_pct / expected_pct

    # 5. OI rising before switch
    oi_before = avgs['oi_delta_300s'][window - 300]  # 5min before
    results['oi_before_switch'] = oi_before

    # 6. Stabilization time (vol back to < 1.5x baseline)
    stab_time = None
    for t in range(window, min(window + 600, len(avgs['vol_60s']))):
        if avgs['vol_60s'][t] < baseline_vol * 1.5:
            stab_time = t - window
            break
    results['stabilization_s'] = stab_time

    # Print summary
    print(f"\n  Causal Chain Elements:")
    print(f"    1. Liq lead (-30s):     {results['liq_lead_ratio']:.2f}x baseline "
          f"({'✅ CONFIRMED' if results['liq_lead_ratio'] > 1.5 else '❌ NOT CONFIRMED'})")
    print(f"    2. Vol spike (0s):      {results['vol_ratio']:.2f}x baseline "
          f"({'✅ CONFIRMED' if results['vol_ratio'] > 1.5 else '❌ NOT CONFIRMED'})")
    print(f"    3. Liq peak (+30s):     {results['liq_peak_ratio']:.2f}x baseline "
          f"({'✅ CONFIRMED' if results['liq_peak_ratio'] > 3.0 else '❌ NOT CONFIRMED'})")
    print(f"    4. OI drop at switch:   ${results['oi_d60_at_switch']:+,.0f} "
          f"({'✅ CONFIRMED' if results['oi_d60_at_switch'] < -10000 else '❌ NOT CONFIRMED'})")
    print(f"    5. OI unwind (+150s):   ${results['oi_d300_at_plus150']:+,.0f} "
          f"({'✅ CONFIRMED' if results['oi_d300_at_plus150'] < -50000 else '❌ NOT CONFIRMED'})")
    print(f"    6. FR 2-4h clustering:  {results.get('fr_2_4h_pct', 0)*100:.1f}% "
          f"(lift={results.get('fr_2_4h_lift', 0):.2f}x) "
          f"({'✅ CONFIRMED' if results.get('fr_2_4h_lift', 0) > 1.2 else '❌ NOT CONFIRMED'})")
    print(f"    7. OI before switch:    ${results['oi_before_switch']:+,.0f} "
          f"({'✅ rising' if results['oi_before_switch'] > 0 else '⚠️ falling'})")
    print(f"    8. Stabilization:       {results.get('stabilization_s', 'N/A')}s "
          f"({'✅' if results.get('stabilization_s') and results['stabilization_s'] < 600 else '⚠️'})")

    # Print profile at key timepoints
    print(f"\n  Profile at Key Timepoints:")
    print(f"  {'Offset':>8s}  {'Vol_60s':>10s}  {'Liq_60s':>8s}  {'LiqNot$':>10s}  "
          f"{'OI_Δ60s$':>12s}  {'OI_Δ5m$':>12s}  {'LiqSell':>8s}  {'LiqBuy':>8s}")
    print(f"  {'-'*90}")

    timepoints = [-720, -420, -300, -120, -60, -30, 0, 30, 60, 90, 150, 300, 480]
    for offset in timepoints:
        i = window + offset
        if 0 <= i < len(avgs['vol_60s']):
            marker = " ← SWITCH" if offset == 0 else ""
            print(f"  {offset:>+7d}s  {avgs['vol_60s'][i]:.6f}  {avgs['liq_60s'][i]:>8.1f}  "
                  f"${avgs['liq_not_60s'][i]:>9,.0f}  ${avgs['oi_delta_60s'][i]:>11,.0f}  "
                  f"${avgs['oi_delta_300s'][i]:>11,.0f}  {avgs['liq_sell_60s'][i]:>8.1f}  "
                  f"{avgs['liq_buy_60s'][i]:>8.1f}{marker}")

    return results


# ============================================================================
# PART 2: REGIME-SWITCH TRADING STRATEGY
# ============================================================================

def compute_liq_p90_fast(liq_60, n):
    """Compute rolling P90 of liq_60s efficiently (every 300s, forward-fill)."""
    liq_p90 = np.full(n, np.nan)
    chunk = 3600
    step = 300  # recompute every 5min
    for i in range(chunk, n, step):
        window = liq_60[max(0, i-chunk):i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > 100:
            liq_p90[i] = np.percentile(valid_w, 90)
    # Forward-fill
    for i in range(1, n):
        if np.isnan(liq_p90[i]) and not np.isnan(liq_p90[i-1]):
            liq_p90[i] = liq_p90[i-1]
    return liq_p90


def simulate_trades(data, switches, n, params, liq_p90):
    """
    Simulate trades for a single strategy variant.
    Returns list of trade dicts.

    params keys:
      entry_delay: seconds after switch to enter
      hold: max hold time in seconds
      stop_mult: stop loss = entry_price * vol * stop_mult
      tp_mult: take profit = entry_price * vol * tp_mult (0 = no TP)
      need_oi_confirm: require OI dropping at entry
      need_liq_confirm: require liq > P90 at switch
      direction_mode: 'momentum' | 'reversion' | 'price_momentum'
        momentum: same direction as liq cascade (sell-liq → SHORT)
        reversion: opposite to liq cascade (sell-liq → LONG)
        price_momentum: use price change from -60s to switch
    """
    ts_arr = data['ts_s']
    ts_to_idx = {}
    for i in range(n):
        ts_to_idx[int(ts_arr[i])] = i

    price = data['price']
    vol = data['vol_60s']
    liq_60 = data['liq_60s']
    liq_sell = data['liq_sell_60s']
    liq_buy = data['liq_buy_60s']
    oi_d60 = data['oi_delta_60s']

    entry_delay = params['entry_delay']
    hold_time = params['hold']
    stop_mult = params['stop_mult']
    tp_mult = params.get('tp_mult', 0)
    need_oi = params['need_oi_confirm']
    need_liq = params['need_liq_confirm']
    dir_mode = params.get('direction_mode', 'momentum')

    trades_list = []

    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < 3600 or idx + entry_delay + hold_time + 60 >= n:
            continue

        entry_idx = idx + entry_delay
        if np.isnan(price[entry_idx]) or price[entry_idx] <= 0:
            continue
        if np.isnan(price[idx]) or price[idx] <= 0:
            continue

        # Liq confirmation
        if need_liq:
            if np.isnan(liq_p90[idx]) or liq_60[idx] < liq_p90[idx]:
                continue

        # OI confirmation
        if need_oi:
            if np.isnan(oi_d60[entry_idx]) or oi_d60[entry_idx] >= 0:
                continue

        # Direction
        sell_liqs = liq_sell[idx]  # use liq at switch time, not entry
        buy_liqs = liq_buy[idx]

        if dir_mode == 'momentum':
            # Ride the cascade: sell-liqs = shorts getting liquidated = price dropping = SHORT
            # Actually: sell-liqs = forced SELLS = price pushed DOWN → go SHORT
            if sell_liqs > buy_liqs:
                direction = -1  # sell-liqs dominate → price dropping → SHORT
            elif buy_liqs > sell_liqs:
                direction = 1   # buy-liqs dominate → price rising → LONG
            else:
                # Use price momentum as tiebreaker
                if price[idx] < price[max(0, idx - 60)]:
                    direction = -1
                else:
                    direction = 1
        elif dir_mode == 'reversion':
            # Fade the cascade after it peaks
            if sell_liqs > buy_liqs:
                direction = 1   # sell-liqs pushed price down → LONG reversion
            elif buy_liqs > sell_liqs:
                direction = -1  # buy-liqs pushed price up → SHORT reversion
            else:
                direction = 1
        elif dir_mode == 'price_momentum':
            # Use actual price change to determine direction
            lookback = min(60, idx)
            price_change = price[idx] - price[idx - lookback]
            direction = -1 if price_change < 0 else 1
        else:
            direction = 1

        entry_price = price[entry_idx]
        recent_vol = vol[idx] if not np.isnan(vol[idx]) else 0.001
        if np.isnan(recent_vol) or recent_vol <= 0:
            recent_vol = 0.001
        stop_distance = entry_price * recent_vol * stop_mult
        tp_distance = entry_price * recent_vol * tp_mult if tp_mult > 0 else float('inf')

        exit_price = None
        exit_reason = None
        exit_idx = None

        for t in range(1, hold_time + 60):
            check_idx = entry_idx + t
            if check_idx >= n or np.isnan(price[check_idx]):
                continue

            current_price = price[check_idx]
            pnl_raw = (current_price - entry_price) * direction

            # Stop loss
            if pnl_raw < -stop_distance:
                exit_price = current_price
                exit_reason = 'stop'
                exit_idx = check_idx
                break

            # Take profit
            if tp_mult > 0 and pnl_raw > tp_distance:
                exit_price = current_price
                exit_reason = 'tp'
                exit_idx = check_idx
                break

            # Time exit
            if t >= hold_time:
                exit_price = current_price
                exit_reason = 'time'
                exit_idx = check_idx
                break

        if exit_price is None or np.isnan(exit_price) or exit_price <= 0:
            continue

        ret_raw = (exit_price / entry_price - 1) * direction
        ret_net = ret_raw - STRATEGY_FEE_BPS / 10000

        trades_list.append({
            'switch_ts': sw,
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'ret_raw': ret_raw,
            'ret_net': ret_net,
            'exit_reason': exit_reason,
            'hold_time': exit_idx - entry_idx,
        })

    return trades_list


def compute_trade_stats(trades_list):
    """Compute strategy statistics from a list of trade dicts."""
    if not trades_list:
        return {'n_trades': 0}

    df = pd.DataFrame(trades_list)
    n_trades = len(df)
    win_rate = (df['ret_net'] > 0).mean()
    avg_ret = df['ret_net'].mean()
    total_ret = df['ret_net'].sum()
    sharpe = avg_ret / max(df['ret_net'].std(), 1e-8) * np.sqrt(252 * 24)

    cum = np.cumsum(df['ret_net'].values)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 0

    avg_win = df.loc[df['ret_net'] > 0, 'ret_net'].mean() if (df['ret_net'] > 0).any() else 0
    avg_loss = df.loc[df['ret_net'] <= 0, 'ret_net'].mean() if (df['ret_net'] <= 0).any() else 0
    n_win = (df['ret_net'] > 0).sum()
    n_loss = (df['ret_net'] <= 0).sum()
    profit_factor = abs(avg_win * n_win) / max(abs(avg_loss * n_loss), 1e-10)

    stops = (df['exit_reason'] == 'stop').sum()
    tps = (df['exit_reason'] == 'tp').sum() if 'tp' in df['exit_reason'].values else 0
    longs = (df['direction'] == 1).sum()
    shorts = (df['direction'] == -1).sum()

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_ret_bps': avg_ret * 10000,
        'total_ret_pct': total_ret * 100,
        'sharpe': sharpe,
        'max_dd_pct': max_dd * 100,
        'profit_factor': profit_factor,
        'stops_pct': stops / n_trades * 100,
        'tp_pct': tps / n_trades * 100,
        'longs': longs,
        'shorts': shorts,
        'avg_hold_s': df['hold_time'].mean(),
        'trades_list': trades_list,
    }


def run_strategy(data, switches, n, symbol):
    """
    Test multiple strategy approaches on regime switches:

    A) MOMENTUM: ride the cascade direction
       - Sell-liqs dominate → price dropping → SHORT
       - Enter quickly, exit when cascade fades

    B) REVERSION: fade the cascade after it peaks
       - Sell-liqs dominate → price overshot down → LONG
       - Enter late (after peak), exit on bounce

    C) PRICE MOMENTUM: use actual price direction, not liq side

    Each with multiple timing/confirmation variants.
    """
    print(f"\n{'='*80}")
    print(f"  TRADING STRATEGY — {symbol} ({len(switches)} switches)")
    print(f"{'='*80}")

    if len(switches) < 10:
        print(f"  Too few switches, skipping strategy")
        return {}

    print(f"  Computing rolling liq P90...", flush=True)
    liq_p90 = compute_liq_p90_fast(data['liq_60s'], n)
    print(f"  Done.", flush=True)

    # ===== Strategy Variants =====
    variants = {
        # --- MOMENTUM: ride the cascade ---
        'mom_5s_1min':       {'entry_delay': 5,  'hold': 60,  'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'momentum'},
        'mom_5s_2min':       {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'momentum'},
        'mom_liq_5s_1min':   {'entry_delay': 5,  'hold': 60,  'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_liq_5s_2min':   {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_oi_5s_2min':    {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': False, 'direction_mode': 'momentum'},
        'mom_full_5s_2min':  {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_full_5s_3min':  {'entry_delay': 5,  'hold': 180, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_tp_5s_2min':    {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 2.0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'momentum'},
        # --- REVERSION: fade after cascade peak ---
        'rev_90s_3min':      {'entry_delay': 90,  'hold': 180, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_120s_5min':     {'entry_delay': 120, 'hold': 300, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_oi_90s_3min':   {'entry_delay': 90,  'hold': 180, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_180s_5min':     {'entry_delay': 180, 'hold': 300, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_300s_8min':     {'entry_delay': 300, 'hold': 480, 'stop_mult': 4.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        # --- PRICE MOMENTUM ---
        'pmom_5s_1min':      {'entry_delay': 5,  'hold': 60,  'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'price_momentum'},
        'pmom_5s_2min':      {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'price_momentum'},
        'pmom_liq_5s_2min':  {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'price_momentum'},
        'pmom_full_5s_2min': {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'price_momentum'},
    }

    all_results = {}

    for vname, vparams in variants.items():
        trades_list = simulate_trades(data, switches, n, vparams, liq_p90)
        all_results[vname] = compute_trade_stats(trades_list)

    # Print results table
    print(f"\n  {'Variant':>24s}  {'Trades':>7s}  {'WinR%':>6s}  {'AvgBps':>7s}  {'TotRet%':>8s}  "
          f"{'Sharpe':>7s}  {'MaxDD%':>7s}  {'PF':>6s}  {'Stop%':>6s}  {'L/S':>7s}")
    print(f"  {'-'*100}")

    for vname, r in sorted(all_results.items(), key=lambda x: x[1].get('sharpe', -999), reverse=True):
        if r['n_trades'] == 0:
            print(f"  {vname:>24s}  {0:>7d}  {'N/A':>6s}")
            continue
        print(f"  {vname:>24s}  {r['n_trades']:>7d}  {r['win_rate']*100:>5.1f}%  "
              f"{r['avg_ret_bps']:>+7.1f}  {r['total_ret_pct']:>+7.2f}%  "
              f"{r['sharpe']:>7.2f}  {r['max_dd_pct']:>6.2f}%  "
              f"{r['profit_factor']:>6.2f}  {r['stops_pct']:>5.1f}%  "
              f"{r['longs']:>3d}/{r['shorts']:>3d}")

    return all_results


# ============================================================================
# PART 3: WALK-FORWARD OUT-OF-SAMPLE TEST
# ============================================================================

def walk_forward_strategy(data, n, symbol, all_results):
    """
    Walk-forward OOS test: use first 70% to pick best variant,
    test on last 30% with that variant.
    """
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD OOS TEST — {symbol}")
    print(f"{'='*80}")

    split_idx = int(n * 0.7)
    ts_arr = data['ts_s']

    # Detect switches on full data
    all_switches = detect_regime_switches(data, n)
    oos_switches = [s for s in all_switches if ts_arr[split_idx] <= s <= ts_arr[-1]]
    is_switches = [s for s in all_switches if ts_arr[0] <= s < ts_arr[split_idx]]

    print(f"  IS period:  {split_idx:,} seconds, {len(is_switches)} switches")
    print(f"  OOS period: {n - split_idx:,} seconds, {len(oos_switches)} switches")

    if len(oos_switches) < 5:
        print(f"  Too few OOS switches, skipping")
        return {}

    # Find top 3 IS variants
    ranked = sorted(
        [(k, v) for k, v in all_results.items() if v.get('n_trades', 0) >= 10],
        key=lambda x: x[1].get('sharpe', -999), reverse=True
    )[:3]

    if not ranked:
        print(f"  No valid IS variants")
        return {}

    # Variant params lookup
    variant_params_map = {
        'mom_5s_1min':       {'entry_delay': 5,  'hold': 60,  'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'momentum'},
        'mom_5s_2min':       {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'momentum'},
        'mom_liq_5s_1min':   {'entry_delay': 5,  'hold': 60,  'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_liq_5s_2min':   {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_oi_5s_2min':    {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': False, 'direction_mode': 'momentum'},
        'mom_full_5s_2min':  {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_full_5s_3min':  {'entry_delay': 5,  'hold': 180, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'mom_tp_5s_2min':    {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 2.0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'momentum'},
        'rev_90s_3min':      {'entry_delay': 90,  'hold': 180, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_120s_5min':     {'entry_delay': 120, 'hold': 300, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_oi_90s_3min':   {'entry_delay': 90,  'hold': 180, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_180s_5min':     {'entry_delay': 180, 'hold': 300, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'rev_300s_8min':     {'entry_delay': 300, 'hold': 480, 'stop_mult': 4.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'reversion'},
        'pmom_5s_1min':      {'entry_delay': 5,  'hold': 60,  'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'price_momentum'},
        'pmom_5s_2min':      {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': False, 'direction_mode': 'price_momentum'},
        'pmom_liq_5s_2min':  {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': False, 'need_liq_confirm': True,  'direction_mode': 'price_momentum'},
        'pmom_full_5s_2min': {'entry_delay': 5,  'hold': 120, 'stop_mult': 3.0, 'tp_mult': 0, 'need_oi_confirm': True,  'need_liq_confirm': True,  'direction_mode': 'price_momentum'},
    }

    liq_p90 = compute_liq_p90_fast(data['liq_60s'], n)

    print(f"\n  Top 3 IS variants tested OOS:")
    print(f"  {'Variant':>24s}  {'IS_Sharpe':>10s}  {'OOS_Trades':>10s}  {'OOS_WinR%':>10s}  "
          f"{'OOS_AvgBps':>10s}  {'OOS_TotRet%':>12s}  {'OOS_Sharpe':>10s}")
    print(f"  {'-'*95}")

    oos_results = {}
    for vname, is_stats in ranked:
        params = variant_params_map.get(vname)
        if params is None:
            continue
        trades_list = simulate_trades(data, oos_switches, n, params, liq_p90)
        stats = compute_trade_stats(trades_list)
        oos_results[vname] = stats

        if stats['n_trades'] == 0:
            print(f"  {vname:>24s}  {is_stats['sharpe']:>10.2f}  {0:>10d}  N/A")
        else:
            print(f"  {vname:>24s}  {is_stats['sharpe']:>10.2f}  {stats['n_trades']:>10d}  "
                  f"{stats['win_rate']*100:>9.1f}%  {stats['avg_ret_bps']:>+10.1f}  "
                  f"{stats['total_ret_pct']:>+11.2f}%  {stats['sharpe']:>10.2f}")

    return oos_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_global = time.time()
    dates = get_dates_list()

    print(f"\n{'#'*80}")
    print(f"  v40: REGIME-SWITCH CAUSAL CHAIN VALIDATION & TRADING STRATEGY")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Period:  {START_DATE} to {END_DATE} ({len(dates)} days)")
    print(f"{'#'*80}")
    print_mem("start")

    all_validation = []
    all_strategy = {}
    all_oos = {}

    for sym_idx, symbol in enumerate(SYMBOLS):
        t_sym = time.time()
        print(f"\n\n{'#'*80}")
        print(f"  PROCESSING {symbol} ({sym_idx+1}/{len(SYMBOLS)})")
        print(f"{'#'*80}")

        # Load data day by day
        day_results = []
        total_liqs = 0
        total_oi = 0
        total_fr = 0

        for di, date_str in enumerate(dates):
            # Load liquidations for this day
            liqs_day = load_liquidations_day(symbol, date_str)
            oi_day, fr_day = load_ticker_day(symbol, date_str)

            result = build_second_series_day(symbol, date_str, liqs_day, oi_day, fr_day)
            if result is not None:
                day_results.append(result)
                total_liqs += len(liqs_day)
                total_oi += len(oi_day)
                total_fr += len(fr_day)

            del liqs_day, oi_day, fr_day
            gc.collect()

            if (di + 1) % 10 == 0 or di == len(dates) - 1:
                elapsed = time.time() - t_sym
                print(f"  [{di+1}/{len(dates)}] {date_str} | {elapsed:.0f}s | "
                      f"{len(day_results)} days loaded | "
                      f"liqs={total_liqs:,} oi={total_oi:,} fr={total_fr:,}", flush=True)

        if len(day_results) < 10:
            print(f"  WARNING: Only {len(day_results)} days loaded for {symbol}, skipping")
            continue

        # Stitch and compute features
        print(f"\n  Stitching {len(day_results)} days...")
        t0 = time.time()
        result = stitch_days_and_compute_features(day_results)
        if result is None:
            print(f"  ERROR: Failed to stitch data for {symbol}")
            continue
        data, ts_start, n_total, total_trades = result
        print(f"  Stitched: {n_total:,} seconds, {total_trades:,} trades ({time.time()-t0:.1f}s)")
        print_mem(f"after stitch {symbol}")

        # Detect regime switches
        print(f"\n  Detecting regime switches...")
        switches = detect_regime_switches(data, n_total)
        print(f"  Found {len(switches)} regime switches")

        if len(switches) < 10:
            print(f"  Retrying with threshold=1.5...")
            switches_15 = detect_regime_switches(data, n_total, threshold=1.5)
            if len(switches_15) > len(switches):
                switches = switches_15
                print(f"  With threshold=1.5: {len(switches)} switches")

        # PART 1: Validate causal chain
        validation = validate_causal_chain(data, switches, n_total, symbol)
        all_validation.append(validation)

        # PART 2: Run strategy variants
        strategy_results = run_strategy(data, switches, n_total, symbol)
        all_strategy[symbol] = strategy_results

        # PART 3: Walk-forward OOS test with top IS variants
        oos = walk_forward_strategy(data, n_total, symbol, strategy_results)
        all_oos[symbol] = oos

        # Cleanup
        del data
        gc.collect()
        print(f"\n  {symbol} complete in {time.time()-t_sym:.0f}s")
        print_mem(f"after {symbol}")

    # ============================================================================
    # CROSS-SYMBOL SUMMARY
    # ============================================================================

    print(f"\n\n{'#'*80}")
    print(f"  CROSS-SYMBOL SUMMARY")
    print(f"{'#'*80}")

    # Validation summary
    print(f"\n  === CAUSAL CHAIN VALIDATION ===")
    print(f"  {'Symbol':>10s}  {'Switches':>8s}  {'LiqLead':>8s}  {'VolSpike':>9s}  "
          f"{'LiqPeak':>8s}  {'OI_Drop':>12s}  {'OI_Unwind':>12s}  {'FR_Lift':>8s}  {'Stab_s':>7s}")
    print(f"  {'-'*100}")

    confirmed_count = 0
    for v in all_validation:
        if not v:
            continue
        sym = v['symbol']
        # Count confirmations
        confirms = 0
        if v.get('liq_lead_ratio', 0) > 1.5: confirms += 1
        if v.get('vol_ratio', 0) > 1.5: confirms += 1
        if v.get('liq_peak_ratio', 0) > 3.0: confirms += 1
        if v.get('oi_d60_at_switch', 0) < -10000: confirms += 1
        if v.get('oi_d300_at_plus150', 0) < -50000: confirms += 1
        if v.get('fr_2_4h_lift', 0) > 1.2: confirms += 1

        print(f"  {sym:>10s}  {v['n_switches']:>8d}  "
              f"{v.get('liq_lead_ratio', 0):>7.2f}x  "
              f"{v.get('vol_ratio', 0):>8.2f}x  "
              f"{v.get('liq_peak_ratio', 0):>7.2f}x  "
              f"${v.get('oi_d60_at_switch', 0):>+11,.0f}  "
              f"${v.get('oi_d300_at_plus150', 0):>+11,.0f}  "
              f"{v.get('fr_2_4h_lift', 0):>7.2f}x  "
              f"{str(v.get('stabilization_s', 'N/A')):>7s}  "
              f"[{confirms}/6 confirmed]")
        if confirms >= 4:
            confirmed_count += 1

    print(f"\n  Causal chain confirmed on {confirmed_count}/{len(all_validation)} symbols")

    # Strategy summary
    print(f"\n  === BEST STRATEGY PER SYMBOL (In-Sample) ===")
    print(f"  {'Symbol':>10s}  {'Variant':>28s}  {'Trades':>7s}  {'WinR%':>6s}  "
          f"{'AvgBps':>7s}  {'TotRet%':>8s}  {'Sharpe':>7s}  {'MaxDD%':>7s}  {'PF':>6s}")
    print(f"  {'-'*100}")

    for sym, results in all_strategy.items():
        best_v = None
        best_s = -999
        for vname, r in results.items():
            if r.get('n_trades', 0) >= 10 and r.get('sharpe', -999) > best_s:
                best_s = r['sharpe']
                best_v = vname
        if best_v:
            r = results[best_v]
            print(f"  {sym:>10s}  {best_v:>28s}  {r['n_trades']:>7d}  {r['win_rate']*100:>5.1f}%  "
                  f"{r['avg_ret_bps']:>+7.1f}  {r['total_ret_pct']:>+7.2f}%  "
                  f"{r['sharpe']:>7.2f}  {r['max_dd_pct']:>6.2f}%  {r['profit_factor']:>6.2f}")

    # OOS summary
    print(f"\n  === OUT-OF-SAMPLE RESULTS ===")
    print(f"  {'Symbol':>10s}  {'Trades':>7s}  {'WinR%':>6s}  {'AvgBps':>7s}  "
          f"{'TotRet%':>8s}  {'Sharpe':>7s}  {'MaxDD%':>7s}")
    print(f"  {'-'*60}")

    for sym, oos_dict in all_oos.items():
        if not oos_dict:
            print(f"  {sym:>10s}  {0:>7d}  N/A")
            continue
        # oos_dict is {variant_name: stats_dict}, pick best
        best_oos = None
        best_oos_sharpe = -999
        for vname, r in oos_dict.items():
            if r.get('n_trades', 0) >= 5 and r.get('sharpe', -999) > best_oos_sharpe:
                best_oos_sharpe = r['sharpe']
                best_oos = r
        if best_oos is None or best_oos.get('n_trades', 0) == 0:
            print(f"  {sym:>10s}  {0:>7d}  N/A")
            continue
        r = best_oos
        print(f"  {sym:>10s}  {r['n_trades']:>7d}  {r['win_rate']*100:>5.1f}%  "
              f"{r['avg_ret_bps']:>+7.1f}  {r['total_ret_pct']:>+7.2f}%  "
              f"{r['sharpe']:>7.2f}  {r['max_dd_pct']:>6.2f}%")

    # Portfolio-level OOS
    if all_oos:
        all_rets = []
        for sym, oos_dict in all_oos.items():
            if not oos_dict:
                continue
            for vname, r in oos_dict.items():
                if r.get('n_trades', 0) > 0 and 'trades_list' in r:
                    for t in r['trades_list']:
                        all_rets.append(t['ret_net'])
        if all_rets:
            all_rets = np.array(all_rets)
            print(f"\n  PORTFOLIO OOS: {len(all_rets)} trades, "
                  f"WinR={np.mean(all_rets > 0)*100:.1f}%, "
                  f"AvgBps={np.mean(all_rets)*10000:+.1f}, "
                  f"TotRet={np.sum(all_rets)*100:+.2f}%, "
                  f"Sharpe={np.mean(all_rets)/max(np.std(all_rets), 1e-8)*np.sqrt(252*24):.2f}")

    elapsed = time.time() - t_global
    print(f"\n{'#'*80}")
    print(f"  COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
