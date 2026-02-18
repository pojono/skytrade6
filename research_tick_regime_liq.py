#!/usr/bin/env python3
"""
Tick-Level Regime Switch Analysis (v27c)

Previous experiments used 5-min bars — way too coarse for crypto.
This experiment works at SECOND resolution using raw tick trades + liquidations.

EXPERIMENT DESIGN:
==================

1. DEFINE REGIME SWITCHES at second resolution:
   - Compute rolling realized volatility in 10-second windows
   - A "regime switch" = vol crosses from below to above 2× its 1-hour rolling median
   - This gives us precise switch timestamps (to the second)

2. BUILD SECOND-LEVEL TIMESERIES around each switch (±30 min):
   - Volatility: realized vol in trailing 10s, 30s, 60s, 300s windows
   - Trade intensity: trades per second, notional per second
   - Liquidation intensity: liquidations per second, notional per second
   - Liquidation imbalance: buy vs sell liq notional

3. MEASURE LEAD/LAG at second resolution:
   - When does liq intensity first spike relative to vol spike?
   - Cross-correlation at 1-second resolution
   - First-crossing analysis: which signal crosses threshold first?

4. PREDICTIVE TEST:
   - Given a liq spike, how often does a vol regime switch follow within 1/5/15/60 min?
   - Precision, recall, lead time distribution

Data: BTC tick trades + liquidations, 7 days (May 12-18, 2025)
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
# DATA LOADING — chunked, one day at a time
# ============================================================================

def load_liquidations_all(dates):
    """Load all liquidation events (tiny — ~14K rows for 7 days)."""
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


def aggregate_day_to_seconds(date_str, liqs):
    """Load one day of trades, aggregate to per-second arrays, free trades immediately.
    Returns dict of numpy arrays indexed by second offset from day start."""
    path = DATA_DIR / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        print(f"  SKIP {date_str} (no file)")
        return None

    t0 = time.time()
    # Load only needed columns to save RAM
    df = pd.read_csv(path, compression='gzip', usecols=['timestamp', 'price', 'size', 'side'])
    n_trades = len(df)

    ts_s = df['timestamp'].values.astype(np.int64)  # floor to second
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.float64)
    notionals = prices * sizes
    is_buy = (df['side'].values == 'Buy')
    del df
    gc.collect()

    day_start = ts_s.min()
    day_end = ts_s.max()
    n_seconds = day_end - day_start + 1

    # Offset indices
    offsets = ts_s - day_start

    # Vectorized aggregation using np.bincount
    trade_count = np.bincount(offsets, minlength=n_seconds).astype(np.int32)
    trade_notional = np.bincount(offsets, weights=notionals, minlength=n_seconds)

    # Last price per second: iterate backward through unique seconds
    price_last = np.full(n_seconds, np.nan, dtype=np.float64)
    # Use pandas for last-price (efficient with groupby on sorted data)
    unique_secs, last_idx = np.unique(offsets, return_index=False), None
    # Faster: since data is sorted by time, last occurrence per second
    _, last_indices = np.unique(offsets[::-1], return_index=True)
    last_indices = len(offsets) - 1 - last_indices
    for ui, li in zip(np.unique(offsets), last_indices):
        price_last[ui] = prices[li]

    del ts_s, prices, sizes, notionals, is_buy, offsets
    gc.collect()

    # Forward-fill prices
    mask = np.isnan(price_last)
    idx_arr = np.arange(n_seconds)
    if not mask.all():
        # Find first valid
        first_valid = np.argmin(mask)
        for i in range(first_valid + 1, n_seconds):
            if mask[i]:
                price_last[i] = price_last[i - 1]

    # Liquidation aggregation for this day
    liq_count = np.zeros(n_seconds, dtype=np.int32)
    liq_notional = np.zeros(n_seconds, dtype=np.float64)
    liq_buy_notional = np.zeros(n_seconds, dtype=np.float64)
    liq_sell_notional = np.zeros(n_seconds, dtype=np.float64)

    if len(liqs) > 0:
        liq_day = liqs[(liqs['ts_s'] >= day_start) & (liqs['ts_s'] <= day_end)]
        if len(liq_day) > 0:
            l_offsets = (liq_day['ts_s'].values - day_start).astype(np.int64)
            valid = (l_offsets >= 0) & (l_offsets < n_seconds)
            l_offsets = l_offsets[valid]
            l_not = liq_day['notional'].values[valid]
            l_buy = liq_day['is_buy'].values[valid]

            liq_count = np.bincount(l_offsets, minlength=n_seconds).astype(np.int32)
            liq_notional = np.bincount(l_offsets, weights=l_not, minlength=n_seconds)
            liq_buy_notional = np.bincount(l_offsets[l_buy == 1],
                                            weights=l_not[l_buy == 1],
                                            minlength=n_seconds)
            liq_sell_notional = np.bincount(l_offsets[l_buy == 0],
                                             weights=l_not[l_buy == 0],
                                             minlength=n_seconds)

    elapsed = time.time() - t0
    print_mem(f"{date_str}: {n_trades:,} trades → {n_seconds:,}s ({elapsed:.1f}s)")

    return {
        'day_start': day_start,
        'n_seconds': n_seconds,
        'trade_count': trade_count,
        'trade_notional': trade_notional,
        'price_last': price_last,
        'liq_count': liq_count,
        'liq_notional': liq_notional,
        'liq_buy_notional': liq_buy_notional,
        'liq_sell_notional': liq_sell_notional,
    }


def build_second_series(dates, liqs):
    """Build full per-second timeseries by processing one day at a time."""
    # First pass: get time range
    day_results = []
    for date_str in dates:
        result = aggregate_day_to_seconds(date_str, liqs)
        if result is not None:
            day_results.append(result)
        gc.collect()

    if not day_results:
        raise ValueError("No data loaded")

    # Stitch together into continuous arrays
    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n_seconds'] - 1
    n = ts_end - ts_start + 1
    print(f"\n  Stitching {len(day_results)} days → {n:,} seconds ({n/3600:.1f}h)")
    print_mem("before stitch")

    trade_count = np.zeros(n, dtype=np.int32)
    trade_notional = np.zeros(n, dtype=np.float64)
    price_last = np.full(n, np.nan, dtype=np.float64)
    liq_count = np.zeros(n, dtype=np.int32)
    liq_notional = np.zeros(n, dtype=np.float64)
    liq_buy_notional = np.zeros(n, dtype=np.float64)
    liq_sell_notional = np.zeros(n, dtype=np.float64)

    for dr in day_results:
        offset = dr['day_start'] - ts_start
        length = dr['n_seconds']
        trade_count[offset:offset+length] = dr['trade_count']
        trade_notional[offset:offset+length] = dr['trade_notional']
        price_last[offset:offset+length] = dr['price_last']
        liq_count[offset:offset+length] = dr['liq_count']
        liq_notional[offset:offset+length] = dr['liq_notional']
        liq_buy_notional[offset:offset+length] = dr['liq_buy_notional']
        liq_sell_notional[offset:offset+length] = dr['liq_sell_notional']

    del day_results
    gc.collect()

    # Forward-fill prices across day boundaries
    for i in range(1, n):
        if np.isnan(price_last[i]):
            price_last[i] = price_last[i - 1]

    # Log returns per second
    log_ret = np.zeros(n, dtype=np.float64)
    valid = (price_last[1:] > 0) & (price_last[:-1] > 0)
    log_ret[1:][valid] = np.log(price_last[1:][valid] / price_last[:-1][valid])

    # Vectorized rolling std (realized vol)
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

    # Vectorized rolling sum
    def fast_rolling_sum(arr, window):
        cs = np.cumsum(arr.astype(np.float64))
        result = np.zeros(len(arr), dtype=np.float64)
        result[window:] = cs[window:] - cs[:-window]
        return result

    print_mem("computing rolling features")

    vol_10s = fast_rolling_std(log_ret, 10)
    vol_30s = fast_rolling_std(log_ret, 30)
    vol_60s = fast_rolling_std(log_ret, 60)
    vol_300s = fast_rolling_std(log_ret, 300)

    liq_count_30s = fast_rolling_sum(liq_count, 30)
    liq_count_60s = fast_rolling_sum(liq_count, 60)
    liq_count_300s = fast_rolling_sum(liq_count, 300)
    liq_notional_60s = fast_rolling_sum(liq_notional, 60)
    liq_notional_300s = fast_rolling_sum(liq_notional, 300)
    trade_count_10s = fast_rolling_sum(trade_count, 10)
    trade_notional_60s = fast_rolling_sum(trade_notional, 60)

    seconds = np.arange(ts_start, ts_end + 1)

    print_mem("building dataframe")

    sec_df = pd.DataFrame({
        'ts_s': seconds,
        'price': price_last,
        'log_ret': log_ret,
        'vol_10s': vol_10s,
        'vol_30s': vol_30s,
        'vol_60s': vol_60s,
        'vol_300s': vol_300s,
        'trade_count': trade_count,
        'trade_notional': trade_notional,
        'trade_count_10s': trade_count_10s,
        'trade_notional_60s': trade_notional_60s,
        'liq_count': liq_count,
        'liq_notional': liq_notional,
        'liq_buy_notional': liq_buy_notional,
        'liq_sell_notional': liq_sell_notional,
        'liq_count_30s': liq_count_30s,
        'liq_count_60s': liq_count_60s,
        'liq_count_300s': liq_count_300s,
        'liq_notional_60s': liq_notional_60s,
        'liq_notional_300s': liq_notional_300s,
    })

    # Free intermediate arrays
    del (trade_count, trade_notional, price_last, log_ret,
         vol_10s, vol_30s, vol_60s, vol_300s,
         liq_count, liq_notional, liq_buy_notional, liq_sell_notional,
         liq_count_30s, liq_count_60s, liq_count_300s,
         liq_notional_60s, liq_notional_300s,
         trade_count_10s, trade_notional_60s, seconds)
    gc.collect()

    print(f"  Built {len(sec_df):,} seconds")
    print_mem("done")
    return sec_df, ts_start


# ============================================================================
# DETECT REGIME SWITCHES at second resolution
# ============================================================================

def detect_regime_switches(sec_df, vol_col='vol_60s', median_window=3600, threshold=2.0,
                           min_gap=300):
    """
    Detect regime switches: vol crosses from below to above threshold × rolling median.

    Returns list of switch timestamps (seconds).
    """
    vol = sec_df[vol_col].values
    n = len(vol)

    # Rolling median of vol (1-hour window)
    # Use a simple rolling approach
    vol_median = np.full(n, np.nan)
    for i in range(median_window, n):
        window = vol[i-median_window:i]
        valid = window[~np.isnan(window)]
        if len(valid) > median_window // 2:
            vol_median[i] = np.median(valid)

    # Detect crossings: vol goes from below to above threshold * median
    switches = []
    in_volatile = False
    for i in range(median_window + 1, n):
        if np.isnan(vol[i]) or np.isnan(vol_median[i]):
            continue
        thresh = vol_median[i] * threshold
        if not in_volatile and vol[i] > thresh:
            # Check we're not too close to last switch
            if not switches or (sec_df['ts_s'].iloc[i] - switches[-1]) > min_gap:
                switches.append(sec_df['ts_s'].iloc[i])
            in_volatile = True
        elif in_volatile and vol[i] < vol_median[i] * 1.2:
            in_volatile = False

    return switches, vol_median


# ============================================================================
# ANALYSIS 1: Second-level profiles around switches
# ============================================================================

def analysis_profiles(sec_df, switches, window=1800):
    """Build average profiles of vol and liq around regime switches (±30min)."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: Second-Level Profiles Around {len(switches)} Switches (±{window}s)")
    print(f"{'='*70}")

    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    vol_profiles = []
    liq_count_profiles = []
    liq_notional_profiles = []
    trade_count_profiles = []

    for sw in switches:
        idx = ts_to_idx.get(sw)
        if idx is None or idx < window or idx + window >= len(ts_arr):
            continue
        vol_profiles.append(sec_df['vol_60s'].values[idx-window:idx+window])
        liq_count_profiles.append(sec_df['liq_count_60s'].values[idx-window:idx+window])
        liq_notional_profiles.append(sec_df['liq_notional_60s'].values[idx-window:idx+window])
        trade_count_profiles.append(sec_df['trade_count_10s'].values[idx-window:idx+window])

    if not vol_profiles:
        print("  No valid profiles")
        return

    avg_vol = np.nanmean(vol_profiles, axis=0)
    avg_liq_count = np.mean(liq_count_profiles, axis=0)
    avg_liq_not = np.mean(liq_notional_profiles, axis=0)
    avg_trade = np.mean(trade_count_profiles, axis=0)

    # Normalize for comparison
    def norm(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        return (arr - mn) / (mx - mn + 1e-15)

    vol_n = norm(avg_vol)
    liq_n = norm(avg_liq_count)
    trade_n = norm(avg_trade)

    # Print at 30-second resolution
    print(f"\n  {'Offset':>8s}  {'Vol_60s':>10s}  {'vol_norm':>9s}  {'Liq_60s':>8s}  {'liq_norm':>9s}  "
          f"{'Trades_10s':>10s}  {'trd_norm':>9s}  {'LiqNot_60s':>12s}")
    print(f"  {'-'*90}")

    step = 30  # print every 30 seconds
    for i in range(0, len(avg_vol), step):
        offset = i - window
        marker = " ← SWITCH" if abs(offset) < step//2 else ""
        if abs(offset) <= 600 or offset % 300 == 0:  # detailed near switch, sparse far
            print(f"  {offset:>+7d}s  {avg_vol[i]:>10.6f}  {vol_n[i]:>9.3f}  "
                  f"{avg_liq_count[i]:>8.1f}  {liq_n[i]:>9.3f}  "
                  f"{avg_trade[i]:>10.1f}  {trade_n[i]:>9.3f}  "
                  f"${avg_liq_not[i]:>11,.0f}{marker}")

    # Find when each signal first reaches 50% of its peak
    vol_half = np.nanmin(avg_vol) + 0.5 * (np.nanmax(avg_vol) - np.nanmin(avg_vol))
    liq_half = np.min(avg_liq_count) + 0.5 * (np.max(avg_liq_count) - np.min(avg_liq_count))

    vol_first_50 = None
    liq_first_50 = None
    for i in range(window):  # only look before switch
        if avg_vol[i] >= vol_half and vol_first_50 is None:
            vol_first_50 = i - window
        if avg_liq_count[i] >= liq_half and liq_first_50 is None:
            liq_first_50 = i - window

    print(f"\n  First time signal reaches 50% of peak (before switch):")
    print(f"    Volatility:   {vol_first_50}s" if vol_first_50 else "    Volatility:   never before switch")
    print(f"    Liq count:    {liq_first_50}s" if liq_first_50 else "    Liq count:    never before switch")
    if vol_first_50 and liq_first_50:
        lead = vol_first_50 - liq_first_50
        if lead > 0:
            print(f"    → Liquidations reach 50% {lead}s BEFORE volatility")
        elif lead < 0:
            print(f"    → Volatility reaches 50% {-lead}s BEFORE liquidations")
        else:
            print(f"    → Simultaneous")


# ============================================================================
# ANALYSIS 2: Cross-correlation at 1-second resolution
# ============================================================================

def analysis_xcorr(sec_df):
    """Cross-correlation between liq_count_60s and vol_60s at 1-second lags."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: Cross-Correlation at 1-Second Resolution")
    print(f"{'='*70}")

    vol = sec_df['vol_60s'].fillna(0).values
    liq = sec_df['liq_count_60s'].values

    # Standardize
    vol_z = (vol - np.mean(vol)) / (np.std(vol) + 1e-10)
    liq_z = (liq - np.mean(liq)) / (np.std(liq) + 1e-10)

    n = len(vol_z)
    # Test lags from -300s to +300s (±5min)
    lags = list(range(-300, 301, 1))
    xcorr = []
    for lag in lags:
        if lag >= 0:
            c = np.mean(liq_z[:n-lag] * vol_z[lag:]) if lag < n else 0
        else:
            c = np.mean(liq_z[-lag:] * vol_z[:n+lag]) if -lag < n else 0
        xcorr.append(c)

    xcorr = np.array(xcorr)
    peak_idx = np.argmax(xcorr)
    peak_lag = lags[peak_idx]
    peak_val = xcorr[peak_idx]

    print(f"\n  Peak cross-correlation: {peak_val:.4f} at lag={peak_lag}s")
    if peak_lag > 0:
        print(f"  → Liquidations LEAD volatility by {peak_lag}s")
    elif peak_lag < 0:
        print(f"  → Volatility LEADS liquidations by {-peak_lag}s")
    else:
        print(f"  → Simultaneous")

    # Print around peak
    print(f"\n  {'Lag(s)':>8s}  {'XCorr':>8s}  {'Bar':>40s}")
    print(f"  {'-'*60}")
    max_xc = max(xcorr)
    for lag, xc in zip(lags, xcorr):
        if lag % 10 == 0 or lag == peak_lag:
            bar_len = int(xc / max_xc * 35) if max_xc > 0 else 0
            bar = '█' * max(bar_len, 0)
            marker = " ← PEAK" if lag == peak_lag else ""
            print(f"  {lag:>+7d}s  {xc:>+8.4f}  {bar}{marker}")

    # Also check: is the xcorr function asymmetric?
    # Average xcorr for negative lags (vol leads) vs positive (liq leads)
    neg_avg = np.mean(xcorr[:300])  # lags -300 to -1
    pos_avg = np.mean(xcorr[301:])  # lags +1 to +300
    print(f"\n  Average xcorr for lags -300..-1 (vol leads): {neg_avg:.4f}")
    print(f"  Average xcorr for lags +1..+300 (liq leads):  {pos_avg:.4f}")
    print(f"  Asymmetry (pos - neg): {pos_avg - neg_avg:+.4f}")
    if pos_avg > neg_avg:
        print(f"  → Slight evidence that liq leads vol")
    else:
        print(f"  → Slight evidence that vol leads liq")

    return peak_lag, peak_val


# ============================================================================
# ANALYSIS 3: First-crossing at individual switch events
# ============================================================================

def analysis_first_crossing(sec_df, switches):
    """For each switch, find when vol and liq first spike — which comes first?"""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 3: First-Crossing — Which Spikes First at Each Switch?")
    print(f"{'='*70}")

    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    vol = sec_df['vol_60s'].values
    liq = sec_df['liq_count_60s'].values

    # Thresholds: 75th percentile of each signal
    vol_p75 = np.nanpercentile(vol[~np.isnan(vol)], 75)
    liq_p75 = np.percentile(liq, 75)
    vol_p90 = np.nanpercentile(vol[~np.isnan(vol)], 90)
    liq_p90 = np.percentile(liq, 90)

    print(f"  vol_60s P75={vol_p75:.6f}, P90={vol_p90:.6f}")
    print(f"  liq_count_60s P75={liq_p75:.1f}, P90={liq_p90:.1f}")

    for pct_label, vol_thr, liq_thr in [("P75", vol_p75, liq_p75), ("P90", vol_p90, liq_p90)]:
        vol_leads = 0
        liq_leads = 0
        simultaneous = 0
        vol_lead_times = []
        liq_lead_times = []

        lookback = 600  # 10 min before switch

        for sw in switches:
            idx = ts_to_idx.get(sw)
            if idx is None or idx < lookback:
                continue

            # Scan backward from switch to find first crossing
            vol_first = None
            liq_first = None

            for j in range(idx, idx - lookback - 1, -1):
                if j < 0:
                    break
                if not np.isnan(vol[j]) and vol[j] >= vol_thr:
                    vol_first = j
                else:
                    break

            for j in range(idx, idx - lookback - 1, -1):
                if j < 0:
                    break
                if liq[j] >= liq_thr:
                    liq_first = j
                else:
                    break

            if vol_first is not None:
                vol_first = idx - vol_first  # seconds before switch
            if liq_first is not None:
                liq_first = idx - liq_first

            if vol_first is not None and liq_first is not None:
                if liq_first > vol_first:
                    liq_leads += 1
                    liq_lead_times.append(liq_first - vol_first)
                elif vol_first > liq_first:
                    vol_leads += 1
                    vol_lead_times.append(vol_first - liq_first)
                else:
                    simultaneous += 1
            elif liq_first is not None:
                liq_leads += 1
            elif vol_first is not None:
                vol_leads += 1

        total = vol_leads + liq_leads + simultaneous
        print(f"\n  Threshold: {pct_label} ({total} switches analyzed)")
        print(f"    Liq spikes first:  {liq_leads:>4d} ({liq_leads/max(total,1)*100:.1f}%)")
        print(f"    Vol spikes first:  {vol_leads:>4d} ({vol_leads/max(total,1)*100:.1f}%)")
        print(f"    Simultaneous:      {simultaneous:>4d} ({simultaneous/max(total,1)*100:.1f}%)")
        if liq_lead_times:
            lt = np.array(liq_lead_times)
            print(f"    Liq lead time: median={np.median(lt):.0f}s, mean={np.mean(lt):.0f}s, "
                  f"p25={np.percentile(lt,25):.0f}s, p75={np.percentile(lt,75):.0f}s")
        if vol_lead_times:
            vt = np.array(vol_lead_times)
            print(f"    Vol lead time: median={np.median(vt):.0f}s, mean={np.mean(vt):.0f}s, "
                  f"p25={np.percentile(vt,25):.0f}s, p75={np.percentile(vt,75):.0f}s")


# ============================================================================
# ANALYSIS 4: Liq spike → future vol increase (predictive test)
# ============================================================================

def analysis_predictive(sec_df, switches):
    """If we see a liq spike, how often does a vol regime switch follow?"""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 4: Liq Spike → Future Vol Switch (Predictive Test)")
    print(f"{'='*70}")

    liq = sec_df['liq_count_60s'].values
    vol = sec_df['vol_60s'].values
    ts_arr = sec_df['ts_s'].values

    switch_set = set(switches)

    for pct in [90, 95, 99]:
        liq_thr = np.percentile(liq, pct)
        print(f"\n  Liq threshold: P{pct} = {liq_thr:.1f} liquidations in 60s")

        # Find all seconds where liq crosses threshold (with 60s cooldown)
        liq_spikes = []
        last_spike = -999
        for i in range(len(liq)):
            if liq[i] >= liq_thr and (i - last_spike) > 60:
                liq_spikes.append(i)
                last_spike = i

        print(f"  Liq spikes: {len(liq_spikes)}")

        for horizon_s in [60, 300, 900]:
            # For each liq spike, check if a regime switch happens within horizon
            hits = 0
            for spike_idx in liq_spikes:
                spike_ts = ts_arr[spike_idx]
                for sw in switches:
                    if 0 < (sw - spike_ts) <= horizon_s:
                        hits += 1
                        break

            # Also: for each switch, was there a liq spike in the preceding horizon?
            switch_with_liq = 0
            for sw in switches:
                for spike_idx in liq_spikes:
                    spike_ts = ts_arr[spike_idx]
                    if 0 < (sw - spike_ts) <= horizon_s:
                        switch_with_liq += 1
                        break

            prec = hits / max(len(liq_spikes), 1)
            rec = switch_with_liq / max(len(switches), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)

            print(f"    Horizon {horizon_s:>4d}s ({horizon_s//60}min): "
                  f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} "
                  f"(hits={hits}/{len(liq_spikes)}, switches_with_liq={switch_with_liq}/{len(switches)})")


# ============================================================================
# ANALYSIS 5: Detailed case studies — zoom into individual switches
# ============================================================================

def analysis_case_studies(sec_df, switches, liqs, n_cases=5):
    """Print detailed second-by-second view of individual regime switches."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 5: Case Studies — Individual Switches (Second-by-Second)")
    print(f"{'='*70}")

    ts_arr = sec_df['ts_s'].values
    ts_to_idx = {ts: i for i, ts in enumerate(ts_arr)}

    for case_num, sw in enumerate(switches[:n_cases]):
        idx = ts_to_idx.get(sw)
        if idx is None or idx < 120 or idx + 60 >= len(ts_arr):
            continue

        sw_dt = pd.to_datetime(sw, unit='s', utc=True)
        print(f"\n  --- Case {case_num+1}: Switch at {sw_dt} ---")

        # Show ±2min at 5-second resolution
        print(f"  {'Offset':>8s}  {'Price':>10s}  {'Vol_60s':>10s}  {'Trades/s':>9s}  "
              f"{'Liq_60s':>8s}  {'LiqNot_60s':>12s}  {'Events':>30s}")
        print(f"  {'-'*95}")

        for offset in range(-120, 61, 5):
            i = idx + offset
            if i < 0 or i >= len(ts_arr):
                continue

            price = sec_df['price'].iloc[i]
            v60 = sec_df['vol_60s'].iloc[i]
            tc = sec_df['trade_count'].iloc[i]
            lc60 = sec_df['liq_count_60s'].iloc[i]
            ln60 = sec_df['liq_notional_60s'].iloc[i]

            # Check for individual liq events in this 5-second window
            events = []
            ts_start = ts_arr[i]
            ts_end = ts_arr[min(i + 5, len(ts_arr) - 1)]
            liq_window = liqs[(liqs['ts_s'] >= ts_start) & (liqs['ts_s'] < ts_end)]
            for _, lev in liq_window.iterrows():
                events.append(f"{lev['side'][0]}${lev['notional']:,.0f}")

            event_str = " ".join(events[:3])
            if len(events) > 3:
                event_str += f" +{len(events)-3}more"

            marker = " ← SWITCH" if offset == 0 else ""
            v60_str = f"{v60:.6f}" if not np.isnan(v60) else "     nan"
            print(f"  {offset:>+7d}s  ${price:>9,.1f}  {v60_str:>10s}  {tc:>8d}  "
                  f"{lc60:>8.0f}  ${ln60:>11,.0f}  {event_str}{marker}")


# ============================================================================
# ANALYSIS 6: Does liq predict vol INCREASE (not just level)?
# ============================================================================

def analysis_vol_change_prediction(sec_df):
    """Can current liq intensity predict future vol INCREASE at second resolution?"""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 6: Liq Intensity → Future Vol Change (Second Resolution)")
    print(f"{'='*70}")

    liq = sec_df['liq_count_60s'].values
    vol = sec_df['vol_60s'].values

    for horizon in [10, 30, 60, 120, 300]:
        # Future vol change
        future_vol = np.roll(vol, -horizon)
        future_vol[-horizon:] = np.nan
        vol_change = (future_vol - vol) / np.clip(vol, 1e-10, None)

        # Remove nans
        valid = ~(np.isnan(vol_change) | np.isnan(vol) | np.isnan(liq))
        liq_v = liq[valid]
        vc_v = vol_change[valid]

        if len(liq_v) < 1000:
            continue

        # Quintiles of liq intensity
        try:
            quintiles = pd.qcut(liq_v, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            continue

        df_tmp = pd.DataFrame({'liq': liq_v, 'vol_change': vc_v, 'q': quintiles})
        result = df_tmp.groupby('q')['vol_change'].agg(['mean', 'median', 'count'])

        print(f"\n  Horizon: {horizon}s — Future vol change by current liq_count_60s quintile:")
        print(f"  {'Quintile':>10s}  {'Mean Δvol':>10s}  {'Median Δvol':>12s}  {'Count':>8s}")
        print(f"  {'-'*45}")
        for q, row in result.iterrows():
            print(f"  {str(q):>10s}  {row['mean']:>+10.4f}  {row['median']:>+12.4f}  {int(row['count']):>8d}")

        # Q5 vs Q1 difference
        q1 = df_tmp[df_tmp['q'] == 'Q1']['vol_change'].values
        q5 = df_tmp[df_tmp['q'] == 'Q5']['vol_change'].values
        if len(q1) > 50 and len(q5) > 50:
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(q5, q1, alternative='greater')
            print(f"  Q5 > Q1 test: p={pval:.6f} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    # Use 7 days of overlapping data
    dates = [f"2025-05-{d:02d}" for d in range(12, 19)]

    print(f"\n{'='*70}")
    print(f"  TICK-LEVEL REGIME + LIQUIDATION ANALYSIS — {SYMBOL}")
    print(f"  Dates: {dates[0]} to {dates[-1]} (7 days)")
    print(f"{'='*70}")
    print_mem("start")

    # Load liquidations first (tiny — ~14K rows)
    print(f"\n  --- Loading Liquidations ---")
    liqs = load_liquidations_all(dates)
    print_mem("after liqs")

    # Build second-level timeseries (processes trades one day at a time)
    print(f"\n  --- Building Second-Level Timeseries (chunked by day) ---")
    t0 = time.time()
    sec_df, ts_start_val = build_second_series(dates, liqs)
    print(f"  Completed in {time.time()-t0:.1f}s")

    # Time range
    ts_end_val = sec_df['ts_s'].iloc[-1]
    duration_h = (ts_end_val - ts_start_val) / 3600
    print(f"  Time range: {pd.to_datetime(ts_start_val, unit='s', utc=True)} to "
          f"{pd.to_datetime(ts_end_val, unit='s', utc=True)} ({duration_h:.1f}h)")

    # Detect regime switches
    print(f"\n  --- Detecting Regime Switches ---")
    switches, vol_median = detect_regime_switches(sec_df)
    print(f"  Found {len(switches)} regime switches")
    if switches:
        for i, sw in enumerate(switches[:10]):
            dt = pd.to_datetime(sw, unit='s', utc=True)
            print(f"    {i+1}. {dt}")
        if len(switches) > 10:
            print(f"    ... and {len(switches)-10} more")

    if len(switches) < 5:
        print("  Not enough switches for analysis. Trying lower threshold...")
        switches, vol_median = detect_regime_switches(sec_df, threshold=1.5)
        print(f"  Found {len(switches)} regime switches (threshold=1.5)")

    if len(switches) < 5:
        print("  Still not enough. Aborting.")
        return

    print_mem("before analyses")

    # Run analyses
    analysis_profiles(sec_df, switches)
    analysis_xcorr(sec_df)
    analysis_first_crossing(sec_df, switches)
    analysis_predictive(sec_df, switches)
    analysis_case_studies(sec_df, switches, liqs=liqs)
    analysis_vol_change_prediction(sec_df)

    print_mem("all done")
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
