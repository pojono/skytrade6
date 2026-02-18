#!/usr/bin/env python3
"""
v31: Tick-Level Microstructure Regime Classification

Classify market into 4 states using tick-level data (trades, ticker, liquidations):
  1. COMPRESSION — low vol, narrowing range, declining volume, OI building (coiling spring)
  2. BREAKOUT    — sudden vol expansion, high trade intensity, directional move
  3. SQUEEZE     — liquidation cascade, OI dropping, one-sided flow, fast price move
  4. EXHAUSTION  — high vol but fading momentum, volume declining from peak, reversal signals

APPROACH:
  Phase 1: Rule-based classification using microstructure features
  Phase 2: Unsupervised clustering (KMeans/GMM) to find natural regimes
  Phase 3: Compare rule-based vs unsupervised, measure transition probabilities
  Phase 4: Test if regimes are predictive of future returns/vol

DATA: BTCUSDT, 3 days first (May 12-14 2025), then expand.
RAM: ~3.6GB available — use float32, process day-by-day, gc aggressively.
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
import argparse

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")

# Defaults — overridden by CLI args
SYMBOL = "BTCUSDT"
DATES = [f"2025-05-{d:02d}" for d in range(12, 15)]

# Classification window: 5 minutes (300s) — main regime window
# Feature horizons for context
HORIZONS = [60, 300, 900]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# UTILITIES
# ============================================================================

def mem_gb():
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label=""):
    u, a = mem_gb()
    print(f"  [RAM] used={u:.1f}GB avail={a:.1f}GB {label}", flush=True)

def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds/60:.1f}m"


# ============================================================================
# DATA LOADING — trades (per-second aggregation, day by day)
# ============================================================================

def load_trades_day(date_str):
    """Load one day of trades, return per-second aggregates as float32."""
    path = DATA_DIR / SYMBOL / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        print(f"  SKIP {date_str} (no trades file)")
        return None

    df = pd.read_csv(path, compression='gzip',
                     usecols=['timestamp', 'price', 'size', 'side', 'tickDirection'])
    n_raw = len(df)

    ts_s = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.float64)
    notionals = prices * sizes
    is_buy = (df['side'] == 'Buy').values.astype(np.float32)
    is_plus_tick = df['tickDirection'].isin(['PlusTick', 'ZeroPlusTick']).values.astype(np.float32)
    del df; gc.collect()

    day_start = int(ts_s.min())
    day_end = int(ts_s.max())
    n = day_end - day_start + 1
    off = (ts_s - day_start).astype(np.int32)
    del ts_s

    trade_count = np.bincount(off, minlength=n).astype(np.float32)
    trade_notional = np.bincount(off, weights=notionals, minlength=n).astype(np.float32)
    buy_notional = np.bincount(off, weights=notionals * is_buy, minlength=n).astype(np.float32)
    buy_count = np.bincount(off, weights=is_buy, minlength=n).astype(np.float32)
    plus_ticks = np.bincount(off, weights=is_plus_tick, minlength=n).astype(np.float32)
    size_sum = np.bincount(off, weights=sizes, minlength=n).astype(np.float32)

    # Large trades (> P90 notional)
    p90_not = np.percentile(notionals, 90)
    large_count = np.bincount(off, weights=(notionals > p90_not).astype(np.float32),
                              minlength=n).astype(np.float32)

    # VWAP components
    vwap_num = np.bincount(off, weights=prices * sizes, minlength=n).astype(np.float32)
    vwap_den = np.bincount(off, weights=sizes, minlength=n).astype(np.float32)

    # Last price per second
    price_last = np.full(n, np.nan, dtype=np.float32)
    _, last_idx = np.unique(off[::-1], return_index=True)
    last_idx = len(off) - 1 - last_idx
    for uo, li in zip(np.unique(off), last_idx):
        price_last[uo] = prices[li]

    # High/low per second (for range calculation)
    price_high = np.full(n, -np.inf, dtype=np.float32)
    price_low = np.full(n, np.inf, dtype=np.float32)
    np.maximum.at(price_high, off, prices.astype(np.float32))
    np.minimum.at(price_low, off, prices.astype(np.float32))
    # Fix seconds with no trades
    no_trade = trade_count == 0
    price_high[no_trade] = np.nan
    price_low[no_trade] = np.nan

    del prices, sizes, notionals, is_buy, is_plus_tick, off; gc.collect()

    return {
        'day_start': day_start, 'n': n, 'n_raw': n_raw,
        'trade_count': trade_count, 'trade_notional': trade_notional,
        'buy_notional': buy_notional, 'buy_count': buy_count,
        'plus_ticks': plus_ticks, 'large_count': large_count,
        'size_sum': size_sum,
        'vwap_num': vwap_num, 'vwap_den': vwap_den,
        'price_last': price_last,
        'price_high': price_high, 'price_low': price_low,
    }


# ============================================================================
# DATA LOADING — liquidations
# ============================================================================

def load_liquidations(dates):
    """Load liquidation events, return arrays."""
    records = []
    for date_str in dates:
        day_count = 0
        for f in sorted((DATA_DIR / SYMBOL / "bybit" / "liquidations").glob(
                f"liquidation_{date_str}_hr*.jsonl.gz")):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        if 'result' in d and 'data' in d['result']:
                            for ev in d['result']['data']:
                                records.append((
                                    int(ev['T']) // 1000,
                                    1.0 if ev['S'] == 'Buy' else 0.0,
                                    float(ev['v']) * float(ev['p']),
                                ))
                                day_count += 1
                    except:
                        continue
        print(f"  {date_str}: {day_count:>6,} liquidations", flush=True)

    if not records:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(records)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.float32), arr[:, 2].astype(np.float32)


# ============================================================================
# DATA LOADING — ticker (OI, funding, bid/ask)
# ============================================================================

def load_ticker_into_arrays(dates, ts_start, n):
    """Load ticker data directly into per-second arrays (RAM-safe)."""
    field_map = {
        'openInterestValue': 'oi', 'fundingRate': 'fr',
        'bid1Price': 'bid', 'bid1Size': 'bid_sz',
        'ask1Price': 'ask', 'ask1Size': 'ask_sz',
        'markPrice': 'mark', 'indexPrice': 'index',
    }
    arrays = {v: np.full(n, np.nan, dtype=np.float32) for v in field_map.values()}
    count = 0

    for date_str in dates:
        files = sorted((DATA_DIR / SYMBOL / "bybit" / "ticker").glob(
            f"ticker_{date_str}_hr*.jsonl.gz"))
        for f in files:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        ts_s = d.get('ts', 0) // 1000
                        idx = ts_s - ts_start
                        if idx < 0 or idx >= n:
                            continue
                        data = d.get('result', {}).get('data', {})
                        for src, dst in field_map.items():
                            if src in data:
                                arrays[dst][idx] = float(data[src])
                        count += 1
                    except:
                        continue
        print(f"    {date_str}: loaded ticker", flush=True)

    # Forward-fill all
    for k in arrays:
        arr = arrays[k]
        mask = np.isnan(arr)
        if not mask.all():
            first_valid = np.argmin(mask)
            for i in range(first_valid + 1, n):
                if mask[i]:
                    arr[i] = arr[i - 1]
        arrays[k] = arr

    print(f"  Ticker: {count:,} updates → {n:,}s arrays", flush=True)
    return arrays


# ============================================================================
# BUILD COMBINED PER-SECOND ARRAYS
# ============================================================================

def build_arrays(dates):
    """Build per-second raw arrays from all sources."""
    t0 = time.time()
    print_mem("start")

    # 1. Load liquidations (small)
    print(f"\n[1/3] Loading liquidations...", flush=True)
    liq_ts, liq_is_buy, liq_not = load_liquidations(dates)
    print(f"  Total: {len(liq_ts):,} liquidation events", flush=True)

    # 2. Load trades day by day
    print(f"\n[2/3] Loading trades...", flush=True)
    day_results = []
    for i, date_str in enumerate(dates):
        dr = load_trades_day(date_str)
        if dr:
            day_results.append(dr)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f"    {date_str}: {dr['n_raw']:,} trades → {dr['n']:,}s "
                  f"({elapsed:.1f}s elapsed, ETA {fmt_time(eta)})", flush=True)
        gc.collect()

    if not day_results:
        print("ERROR: No trade data loaded!")
        return None

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n'] - 1
    n = ts_end - ts_start + 1

    # Allocate combined arrays
    raw_keys = ['trade_count', 'trade_notional', 'buy_notional', 'buy_count',
                'plus_ticks', 'large_count', 'size_sum',
                'vwap_num', 'vwap_den']
    raw = {k: np.zeros(n, dtype=np.float32) for k in raw_keys}
    raw['price_last'] = np.full(n, np.nan, dtype=np.float32)
    raw['price_high'] = np.full(n, np.nan, dtype=np.float32)
    raw['price_low'] = np.full(n, np.nan, dtype=np.float32)

    for dr in day_results:
        o = dr['day_start'] - ts_start
        l = dr['n']
        for k in raw_keys:
            raw[k][o:o+l] = dr[k]
        raw['price_last'][o:o+l] = dr['price_last']
        raw['price_high'][o:o+l] = dr['price_high']
        raw['price_low'][o:o+l] = dr['price_low']
    del day_results; gc.collect()

    # Forward-fill price
    p = raw['price_last']
    mask = np.isnan(p)
    if not mask.all():
        fv = np.argmin(mask)
        for i in range(fv + 1, n):
            if mask[i]:
                p[i] = p[i - 1]

    # Liquidation per-second arrays
    raw['liq_count'] = np.zeros(n, dtype=np.float32)
    raw['liq_notional'] = np.zeros(n, dtype=np.float32)
    raw['liq_buy_count'] = np.zeros(n, dtype=np.float32)
    raw['liq_sell_count'] = np.zeros(n, dtype=np.float32)
    if len(liq_ts) > 0:
        l_off = (liq_ts - ts_start).astype(np.int64)
        valid = (l_off >= 0) & (l_off < n)
        l_off_v = l_off[valid]
        raw['liq_count'] = np.bincount(l_off_v, minlength=n).astype(np.float32)
        raw['liq_notional'] = np.bincount(l_off_v, weights=liq_not[valid], minlength=n).astype(np.float32)
        raw['liq_buy_count'] = np.bincount(l_off_v, weights=liq_is_buy[valid], minlength=n).astype(np.float32)
        raw['liq_sell_count'] = raw['liq_count'] - raw['liq_buy_count']
    del liq_ts, liq_is_buy, liq_not; gc.collect()

    print_mem("before ticker")

    # 3. Ticker
    print(f"\n[3/3] Loading ticker...", flush=True)
    ticker = load_ticker_into_arrays(dates, ts_start, n)
    raw.update(ticker)
    del ticker; gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Built {n:,} seconds ({n/86400:.1f} days) of raw arrays in {elapsed:.1f}s", flush=True)
    print_mem("raw arrays done")
    return raw, ts_start, n


# ============================================================================
# FEATURE ENGINEERING — regime-specific features
# ============================================================================

def compute_regime_features(raw, ts_start, n):
    """Compute features specifically designed for regime classification."""
    t0 = time.time()

    def rsum(arr, w):
        cs = np.cumsum(arr.astype(np.float64))
        r = np.zeros(len(arr), dtype=np.float64)
        r[w:] = cs[w:] - cs[:-w]
        return r.astype(np.float32)

    def rmean(arr, w):
        cs = np.cumsum(arr.astype(np.float64))
        r = np.full(len(arr), np.nan, dtype=np.float64)
        r[w:] = (cs[w:] - cs[:-w]) / w
        return r.astype(np.float32)

    def rstd(arr, w):
        a64 = arr.astype(np.float64)
        cs = np.cumsum(a64)
        cs2 = np.cumsum(a64**2)
        r = np.full(len(arr), np.nan, dtype=np.float64)
        s = cs[w:] - cs[:-w]
        s2 = cs2[w:] - cs2[:-w]
        var = s2/w - (s/w)**2
        np.clip(var, 0, None, out=var)
        r[w:] = np.sqrt(var)
        return r.astype(np.float32)

    def rmax(arr, w):
        """Rolling max using stride tricks — memory efficient."""
        r = np.full(len(arr), np.nan, dtype=np.float32)
        # Use cumulative approach for large arrays
        a = arr.astype(np.float32)
        for i in range(w, len(arr)):
            r[i] = np.nanmax(a[i-w:i])
            if i % 100000 == 0 and i > 0:
                pass  # progress tracked at higher level
        return r

    def rmin(arr, w):
        r = np.full(len(arr), np.nan, dtype=np.float32)
        a = arr.astype(np.float32)
        for i in range(w, len(arr)):
            r[i] = np.nanmin(a[i-w:i])
        return r

    def delta(arr, w):
        r = np.full(len(arr), np.nan, dtype=np.float32)
        r[w:] = arr[w:] - arr[:-w]
        return r

    def delta_pct(arr, w):
        r = np.full(len(arr), np.nan, dtype=np.float32)
        denom = arr[:-w].astype(np.float64).copy()
        denom[denom == 0] = np.nan
        r[w:] = ((arr[w:].astype(np.float64) - arr[:-w].astype(np.float64)) / denom).astype(np.float32)
        return r

    # Log returns (per second)
    p = raw['price_last'].astype(np.float64)
    log_ret = np.zeros(n, dtype=np.float32)
    v = (p[1:] > 0) & (p[:-1] > 0)
    log_ret[1:][v] = (np.log(p[1:][v] / p[:-1][v])).astype(np.float32)

    features = {}
    total_features = 0

    print(f"  Computing features at {len(HORIZONS)} horizons...", flush=True)

    for hi, h in enumerate(HORIZONS):
        hs = f"_{h}s"
        ht0 = time.time()

        # === VOLATILITY / RANGE ===
        features[f'vol{hs}'] = rstd(log_ret, h)
        features[f'ret{hs}'] = rsum(log_ret, h)  # cumulative return over window
        features[f'abs_ret{hs}'] = np.abs(features[f'ret{hs}'])

        # Range: (high - low) / mid over window
        # Use rolling max/min of price_last (cheaper than price_high/low)
        # For 60s/300s this is manageable
        if h <= 300:
            p32 = raw['price_last']
            rmax_p = rmax(p32, h)
            rmin_p = rmin(p32, h)
            mid_p = (rmax_p + rmin_p) / 2
            features[f'range_bps{hs}'] = np.where(mid_p > 0,
                (rmax_p - rmin_p) / mid_p * 10000, np.nan).astype(np.float32)
            del rmax_p, rmin_p, mid_p
        else:
            # For 900s, use vol as proxy for range
            features[f'range_bps{hs}'] = features[f'vol{hs}'] * np.sqrt(h).astype(np.float32) * 10000

        # Vol ratio: current vol vs longer-term vol (compression detector)
        if h == 60:
            features['vol_ratio_60_300'] = np.where(
                features.get('vol_300s', np.ones(n)) > 0,
                features[f'vol{hs}'] / np.maximum(features.get('vol_300s', np.ones(n, dtype=np.float32)), 1e-10),
                1.0).astype(np.float32)

        # === TRADE INTENSITY ===
        features[f'trade_count{hs}'] = rsum(raw['trade_count'], h)
        features[f'trade_notional{hs}'] = rsum(raw['trade_notional'], h)
        features[f'large_count{hs}'] = rsum(raw['large_count'], h)

        # Trade intensity change (acceleration)
        features[f'trade_count_delta{hs}'] = delta(features[f'trade_count{hs}'], h)

        # Buy ratio (order flow imbalance)
        buy_sum = rsum(raw['buy_count'], h)
        tc_sum = features[f'trade_count{hs}']
        features[f'buy_ratio{hs}'] = np.where(tc_sum > 0, buy_sum / tc_sum, 0.5).astype(np.float32)

        # Net buy $ flow
        buy_not_sum = rsum(raw['buy_notional'], h)
        tn_sum = features[f'trade_notional{hs}']
        features[f'net_buy_pct{hs}'] = np.where(tn_sum > 0,
            (2 * buy_not_sum - tn_sum) / tn_sum, 0).astype(np.float32)

        # Tick imbalance
        plus_sum = rsum(raw['plus_ticks'], h)
        tc_sum_f = tc_sum.astype(np.float64)
        features[f'tick_imbalance{hs}'] = np.where(tc_sum_f > 0,
            (2 * plus_sum.astype(np.float64) - tc_sum_f) / tc_sum_f, 0).astype(np.float32)

        # VWAP deviation
        vwap_n = rsum(raw['vwap_num'], h)
        vwap_d = rsum(raw['vwap_den'], h)
        vwap = np.where(vwap_d > 0, vwap_n / vwap_d, raw['price_last'])
        features[f'vwap_dev_bps{hs}'] = np.where(vwap > 0,
            (raw['price_last'].astype(np.float64) - vwap) / vwap * 10000, 0).astype(np.float32)

        # === LIQUIDATIONS ===
        features[f'liq_count{hs}'] = rsum(raw['liq_count'], h)
        features[f'liq_notional{hs}'] = rsum(raw['liq_notional'], h)
        liq_buy_sum = rsum(raw['liq_buy_count'], h)
        liq_sell_sum = rsum(raw['liq_sell_count'], h)
        liq_total = features[f'liq_count{hs}']
        features[f'liq_imbalance{hs}'] = np.where(liq_total > 0,
            (liq_buy_sum - liq_sell_sum) / liq_total, 0).astype(np.float32)

        # Liq intensity relative to trade volume
        features[f'liq_to_trade{hs}'] = np.where(tn_sum > 0,
            features[f'liq_notional{hs}'] / tn_sum, 0).astype(np.float32)

        # === OI ===
        features[f'oi_delta{hs}'] = delta(raw['oi'], h)
        features[f'oi_delta_pct{hs}'] = delta_pct(raw['oi'], h)

        # === SPREAD / BOOK ===
        spread = raw['ask'].astype(np.float64) - raw['bid'].astype(np.float64)
        mid = (raw['ask'].astype(np.float64) + raw['bid'].astype(np.float64)) / 2
        features[f'spread_bps{hs}'] = rmean(
            np.where(mid > 0, spread / mid * 10000, np.nan).astype(np.float32), h)

        bid_sz = raw['bid_sz'].astype(np.float64)
        ask_sz = raw['ask_sz'].astype(np.float64)
        total_sz = bid_sz + ask_sz
        book_imb = np.where(total_sz > 0, (bid_sz - ask_sz) / total_sz, 0).astype(np.float32)
        features[f'book_imbalance{hs}'] = rmean(book_imb, h)

        # === FUNDING RATE ===
        features[f'fr{hs}'] = rmean(raw['fr'], h)

        # === BASIS ===
        basis = raw['mark'].astype(np.float64) - raw['index'].astype(np.float64)
        features[f'basis_bps{hs}'] = rmean(
            np.where(raw['index'].astype(np.float64) > 0,
                     basis / raw['index'].astype(np.float64) * 10000, np.nan).astype(np.float32), h)

        n_feats = sum(1 for k in features if k.endswith(hs))
        total_features += n_feats
        elapsed = time.time() - ht0
        print(f"    {h}s: {n_feats} features ({elapsed:.1f}s)", flush=True)
        gc.collect()

    # Vol ratio (needs both 60s and 300s computed)
    if 'vol_60s' in features and 'vol_300s' in features:
        features['vol_ratio_60_300'] = np.where(
            features['vol_300s'] > 1e-10,
            features['vol_60s'] / features['vol_300s'],
            1.0).astype(np.float32)
        total_features += 1

    if 'vol_300s' in features and 'vol_900s' in features:
        features['vol_ratio_300_900'] = np.where(
            features['vol_900s'] > 1e-10,
            features['vol_300s'] / features['vol_900s'],
            1.0).astype(np.float32)
        total_features += 1

    elapsed = time.time() - t0
    print(f"\n  Total: {total_features} features computed in {elapsed:.1f}s", flush=True)
    print_mem("features done")
    return features


# ============================================================================
# PHASE 1: RULE-BASED REGIME CLASSIFICATION
# ============================================================================

def classify_regimes_rules(features, n):
    """
    Rule-based classification into 4 regimes at 5-minute (300s) resolution.
    Downsample to 300s bars first, then classify each bar.
    """
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"PHASE 1: Rule-Based Regime Classification")
    print(f"{'='*70}")

    # Downsample to 300s bars (non-overlapping)
    bar_len = 300
    n_bars = n // bar_len
    print(f"  {n:,} seconds → {n_bars:,} bars of {bar_len}s", flush=True)

    # Extract key features at 300s horizon, sampled every 300s
    bar_idx = np.arange(n_bars) * bar_len + bar_len  # end of each bar
    bar_idx = bar_idx[bar_idx < n]
    n_bars = len(bar_idx)

    def get(name):
        if name in features:
            return features[name][bar_idx]
        return np.zeros(n_bars, dtype=np.float32)

    vol = get('vol_300s')
    ret = get('ret_300s')
    abs_ret = get('abs_ret_300s')
    range_bps = get('range_bps_300s')
    trade_count = get('trade_count_300s')
    trade_notional = get('trade_notional_300s')
    trade_delta = get('trade_count_delta_300s')
    liq_count = get('liq_count_300s')
    liq_notional = get('liq_notional_300s')
    liq_imbalance = get('liq_imbalance_300s')
    oi_delta_pct = get('oi_delta_pct_300s')
    net_buy_pct = get('net_buy_pct_300s')
    tick_imbalance = get('tick_imbalance_300s')
    vol_ratio = get('vol_ratio_60_300') if 'vol_ratio_60_300' in features else np.ones(n_bars)

    # Compute percentiles for adaptive thresholds
    vol_p25 = np.nanpercentile(vol, 25)
    vol_p50 = np.nanpercentile(vol, 50)
    vol_p75 = np.nanpercentile(vol, 75)
    vol_p90 = np.nanpercentile(vol, 90)

    range_p25 = np.nanpercentile(range_bps, 25)
    range_p50 = np.nanpercentile(range_bps, 50)
    range_p75 = np.nanpercentile(range_bps, 75)

    trade_p50 = np.nanpercentile(trade_count, 50)
    trade_p75 = np.nanpercentile(trade_count, 75)

    liq_p75 = np.nanpercentile(liq_count[liq_count > 0], 75) if (liq_count > 0).any() else 1
    liq_not_p75 = np.nanpercentile(liq_notional[liq_notional > 0], 75) if (liq_notional > 0).any() else 1

    print(f"\n  Adaptive thresholds:")
    print(f"    Vol P25/P50/P75/P90: {vol_p25:.6f} / {vol_p50:.6f} / {vol_p75:.6f} / {vol_p90:.6f}")
    print(f"    Range P25/P50/P75:   {range_p25:.1f} / {range_p50:.1f} / {range_p75:.1f} bps")
    print(f"    Trade count P50/P75: {trade_p50:.0f} / {trade_p75:.0f}")
    print(f"    Liq count P75 (>0):  {liq_p75:.0f}")

    # Classification rules (priority order: squeeze > breakout > exhaustion > compression)
    regimes = np.full(n_bars, -1, dtype=np.int8)  # -1 = unclassified

    # SQUEEZE: High liquidations + OI dropping + directional
    is_squeeze = (
        (liq_count > liq_p75) &
        (liq_notional > liq_not_p75) &
        (np.nan_to_num(oi_delta_pct) < -0.001) &  # OI declining
        (np.abs(np.nan_to_num(liq_imbalance)) > 0.3)  # one-sided liquidations
    )

    # BREAKOUT: High vol + high range + directional + volume surge
    is_breakout = (
        (vol > vol_p75) &
        (range_bps > range_p75) &
        (np.abs(np.nan_to_num(ret)) > np.nanpercentile(np.abs(np.nan_to_num(ret)), 75)) &
        (trade_count > trade_p50)
    )

    # EXHAUSTION: High vol but momentum fading + volume declining
    # Detect: vol still high, but return is small relative to vol (choppy)
    # Or: large range but small net return (reversal)
    ret_to_range = np.where(range_bps > 0,
        np.abs(np.nan_to_num(ret)) * 10000 / range_bps, 0)
    is_exhaustion = (
        (vol > vol_p50) &
        (ret_to_range < 0.3) &  # Low efficiency: big range, small net move
        (trade_delta < 0) &  # Volume declining
        (~is_squeeze) & (~is_breakout)
    )

    # COMPRESSION: Low vol + low range + low volume
    is_compression = (
        (vol <= vol_p50) &
        (range_bps <= range_p50) &
        (~is_squeeze) & (~is_breakout) & (~is_exhaustion)
    )

    # Assign (priority order)
    regimes[is_compression] = 0  # COMPRESSION
    regimes[is_exhaustion] = 3   # EXHAUSTION
    regimes[is_breakout] = 1     # BREAKOUT
    regimes[is_squeeze] = 2      # SQUEEZE

    # Remaining unclassified → assign to nearest by vol level
    unclassified = regimes == -1
    regimes[unclassified & (vol <= vol_p50)] = 0  # low vol → compression
    regimes[unclassified & (vol > vol_p50)] = 3   # high vol unclassified → exhaustion

    regime_names = {0: 'COMPRESSION', 1: 'BREAKOUT', 2: 'SQUEEZE', 3: 'EXHAUSTION'}
    regime_counts = {regime_names[i]: (regimes == i).sum() for i in range(4)}

    print(f"\n  Regime distribution ({n_bars} bars):")
    for name, count in regime_counts.items():
        pct = count / n_bars * 100
        print(f"    {name:15s}: {count:5d} ({pct:5.1f}%)")

    elapsed = time.time() - t0
    print(f"\n  Classification done in {elapsed:.1f}s", flush=True)

    return regimes, bar_idx, regime_names


# ============================================================================
# PHASE 2: UNSUPERVISED CLUSTERING
# ============================================================================

def cluster_regimes(features, n):
    """Use KMeans and GMM to find natural regime clusters."""
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"PHASE 2: Unsupervised Clustering")
    print(f"{'='*70}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    # Downsample to 300s bars
    bar_len = 300
    n_bars = n // bar_len
    bar_idx = np.arange(n_bars) * bar_len + bar_len
    bar_idx = bar_idx[bar_idx < n]
    n_bars = len(bar_idx)

    # Select key features for clustering
    cluster_features = [
        'vol_300s', 'ret_300s', 'abs_ret_300s', 'range_bps_300s',
        'trade_count_300s', 'trade_notional_300s',
        'liq_count_300s', 'liq_notional_300s', 'liq_imbalance_300s',
        'oi_delta_pct_300s', 'net_buy_pct_300s', 'tick_imbalance_300s',
        'book_imbalance_300s', 'spread_bps_300s', 'basis_bps_300s',
    ]

    # Add vol ratios if available
    for f in ['vol_ratio_60_300', 'vol_ratio_300_900']:
        if f in features:
            cluster_features.append(f)

    # Build feature matrix
    X_list = []
    used_features = []
    for f in cluster_features:
        if f in features:
            X_list.append(features[f][bar_idx])
            used_features.append(f)

    X = np.column_stack(X_list).astype(np.float32)
    print(f"  Feature matrix: {X.shape[0]} bars × {X.shape[1]} features", flush=True)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans with 4 clusters
    print(f"  Running KMeans (k=4)...", flush=True)
    km = KMeans(n_clusters=4, n_init=10, random_state=42, max_iter=300)
    km_labels = km.fit_predict(X_scaled)

    # GMM with 4 components (add regularization for numerical stability)
    print(f"  Running GMM (k=4)...", flush=True)
    X_scaled_64 = X_scaled.astype(np.float64)
    try:
        gmm = GaussianMixture(n_components=4, covariance_type='full', n_init=3,
                              random_state=42, reg_covar=1e-4)
        gmm_labels = gmm.fit_predict(X_scaled_64)
    except ValueError:
        print(f"    Full covariance failed, trying 'tied'...", flush=True)
        gmm = GaussianMixture(n_components=4, covariance_type='tied', n_init=3,
                              random_state=42, reg_covar=1e-3)
        gmm_labels = gmm.fit_predict(X_scaled_64)

    # Also try k=3 and k=5 for comparison
    print(f"  Running KMeans (k=3, k=5) for comparison...", flush=True)
    km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
    km5 = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_scaled)

    # Analyze clusters
    print(f"\n  KMeans (k=4) cluster profiles:")
    for c in range(4):
        mask = km_labels == c
        count = mask.sum()
        pct = count / n_bars * 100
        print(f"\n    Cluster {c} ({count} bars, {pct:.1f}%):")
        for fi, fname in enumerate(used_features[:8]):  # Show top 8 features
            vals = X[mask, fi]
            print(f"      {fname:25s}: mean={np.mean(vals):10.4f}  std={np.std(vals):10.4f}")

    print(f"\n  GMM (k=4) cluster profiles:")
    for c in range(4):
        mask = gmm_labels == c
        count = mask.sum()
        pct = count / n_bars * 100
        print(f"\n    Cluster {c} ({count} bars, {pct:.1f}%):")
        for fi, fname in enumerate(used_features[:8]):
            vals = X[mask, fi]
            print(f"      {fname:25s}: mean={np.mean(vals):10.4f}  std={np.std(vals):10.4f}")

    # Inertia / BIC for model selection
    print(f"\n  Model selection metrics:")
    print(f"    KMeans k=3 inertia: {km3.inertia_:.1f}")
    print(f"    KMeans k=4 inertia: {km.inertia_:.1f}")
    print(f"    KMeans k=5 inertia: {km5.inertia_:.1f}")
    print(f"    GMM k=4 BIC: {gmm.bic(X_scaled):.1f}")

    elapsed = time.time() - t0
    print(f"\n  Clustering done in {elapsed:.1f}s", flush=True)

    return km_labels, gmm_labels, bar_idx, used_features, X, X_scaled


# ============================================================================
# PHASE 3: COMPARE & ANALYZE TRANSITIONS
# ============================================================================

def analyze_regimes(rule_regimes, km_labels, gmm_labels, features, bar_idx, regime_names, n):
    """Compare rule-based vs unsupervised, analyze transitions and predictiveness."""
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"PHASE 3: Regime Analysis & Transitions")
    print(f"{'='*70}")

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Agreement between methods
    ari_km = adjusted_rand_score(rule_regimes, km_labels)
    ari_gmm = adjusted_rand_score(rule_regimes, gmm_labels)
    nmi_km = normalized_mutual_info_score(rule_regimes, km_labels)
    nmi_gmm = normalized_mutual_info_score(rule_regimes, gmm_labels)
    ari_km_gmm = adjusted_rand_score(km_labels, gmm_labels)

    print(f"\n  Agreement scores:")
    print(f"    Rules vs KMeans:  ARI={ari_km:.3f}  NMI={nmi_km:.3f}")
    print(f"    Rules vs GMM:     ARI={ari_gmm:.3f}  NMI={nmi_gmm:.3f}")
    print(f"    KMeans vs GMM:    ARI={ari_km_gmm:.3f}")

    # Transition matrix for rule-based regimes
    print(f"\n  Rule-based transition matrix (row=from, col=to):")
    n_regimes = 4
    trans = np.zeros((n_regimes, n_regimes), dtype=np.int32)
    for i in range(len(rule_regimes) - 1):
        trans[rule_regimes[i], rule_regimes[i+1]] += 1

    # Normalize to probabilities
    trans_pct = trans.astype(np.float64)
    row_sums = trans_pct.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_pct = trans_pct / row_sums * 100

    header = "           " + "".join(f"{regime_names[j]:>14s}" for j in range(n_regimes))
    print(f"    {header}")
    for i in range(n_regimes):
        row = f"    {regime_names[i]:11s}"
        for j in range(n_regimes):
            row += f"{trans_pct[i,j]:13.1f}%"
        print(row)

    # Average regime duration (consecutive same-regime bars)
    print(f"\n  Average regime duration (consecutive bars × 5min):")
    for r in range(n_regimes):
        # Find runs
        is_r = (rule_regimes == r).astype(np.int8)
        diffs = np.diff(is_r)
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if is_r[0] == 1:
            starts = np.concatenate([[0], starts])
        if is_r[-1] == 1:
            ends = np.concatenate([ends, [len(is_r)]])
        if len(starts) > 0 and len(ends) > 0:
            durations = ends[:len(starts)] - starts[:len(ends)]
            durations = durations[durations > 0]
            if len(durations) > 0:
                avg_dur = np.mean(durations)
                max_dur = np.max(durations)
                print(f"    {regime_names[r]:15s}: avg={avg_dur:.1f} bars ({avg_dur*5:.0f}min), "
                      f"max={max_dur} bars ({max_dur*5:.0f}min), episodes={len(durations)}")

    # Predictiveness: what happens AFTER each regime?
    print(f"\n  Forward returns by regime (next 5min bar):")
    ret_300s = features.get('ret_300s', np.zeros(n, dtype=np.float32))
    vol_300s = features.get('vol_300s', np.zeros(n, dtype=np.float32))

    for r in range(n_regimes):
        mask = rule_regimes[:-1] == r  # Current bar regime
        if mask.sum() == 0:
            continue
        # Next bar's return and vol
        next_ret = ret_300s[bar_idx[1:len(rule_regimes)]][mask] * 10000  # bps
        next_vol = vol_300s[bar_idx[1:len(rule_regimes)]][mask]

        print(f"    After {regime_names[r]:15s} (n={mask.sum():4d}):")
        print(f"      Next ret:  mean={np.nanmean(next_ret):+6.2f}bps  "
              f"std={np.nanstd(next_ret):6.2f}bps  "
              f"|ret|={np.nanmean(np.abs(next_ret)):6.2f}bps")
        print(f"      Next vol:  mean={np.nanmean(next_vol):.6f}  "
              f"std={np.nanstd(next_vol):.6f}")

    elapsed = time.time() - t0
    print(f"\n  Analysis done in {elapsed:.1f}s", flush=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global SYMBOL, DATES

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=SYMBOL)
    parser.add_argument('--start', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', help='End date YYYY-MM-DD')
    parser.add_argument('--days', type=int, help='Number of days from start')
    args = parser.parse_args()

    SYMBOL = args.symbol
    if args.start:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(args.start, '%Y-%m-%d')
        if args.end:
            end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        elif args.days:
            end_dt = start_dt + timedelta(days=args.days - 1)
        else:
            end_dt = start_dt + timedelta(days=2)  # default 3 days
        DATES = []
        dt = start_dt
        while dt <= end_dt:
            DATES.append(dt.strftime('%Y-%m-%d'))
            dt += timedelta(days=1)

    t_start = time.time()
    print(f"{'='*70}")
    print(f"v31: Tick-Level Microstructure Regime Classification")
    print(f"{'='*70}")
    print(f"Symbol: {SYMBOL}")
    print(f"Dates:  {DATES[0]} to {DATES[-1]} ({len(DATES)} days)")
    print(f"Horizons: {HORIZONS}")
    print_mem("start")

    # Load and build arrays
    result = build_arrays(DATES)
    if result is None:
        return
    raw, ts_start, n = result

    # Compute features
    features = compute_regime_features(raw, ts_start, n)
    del raw; gc.collect()
    print_mem("after feature computation")

    # Phase 1: Rule-based classification
    rule_regimes, bar_idx, regime_names = classify_regimes_rules(features, n)

    # Phase 2: Unsupervised clustering
    km_labels, gmm_labels, bar_idx2, used_features, X, X_scaled = cluster_regimes(features, n)

    # Phase 3: Compare and analyze
    analyze_regimes(rule_regimes, km_labels, gmm_labels, features, bar_idx, regime_names, n)

    # Save results
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {fmt_time(total_time)}")
    print_mem("final")

    # Write results to file
    date_tag = f"{DATES[0]}_to_{DATES[-1]}"
    output_file = RESULTS_DIR / f"v31_regimes_{SYMBOL}_{date_tag}.txt"
    print(f"\nResults saved to {output_file}")

    # Redirect stdout to capture summary
    import io
    buf = io.StringIO()

    def write_summary():
        buf.write(f"v31: Tick-Level Microstructure Regime Classification\n")
        buf.write(f"{'='*70}\n")
        buf.write(f"Symbol: {SYMBOL}\n")
        buf.write(f"Dates:  {DATES[0]} to {DATES[-1]} ({len(DATES)} days)\n")
        buf.write(f"Total time: {fmt_time(total_time)}\n\n")

        buf.write(f"RULE-BASED REGIME DISTRIBUTION:\n")
        for i in range(4):
            count = (rule_regimes == i).sum()
            pct = count / len(rule_regimes) * 100
            buf.write(f"  {regime_names[i]:15s}: {count:5d} ({pct:5.1f}%)\n")

        buf.write(f"\nKMEANS CLUSTER DISTRIBUTION:\n")
        for c in range(4):
            count = (km_labels == c).sum()
            pct = count / len(km_labels) * 100
            buf.write(f"  Cluster {c}: {count:5d} ({pct:5.1f}%)\n")

        buf.write(f"\nTRANSITION MATRIX (rule-based, %):\n")
        n_regimes = 4
        trans = np.zeros((n_regimes, n_regimes), dtype=np.int32)
        for i in range(len(rule_regimes) - 1):
            trans[rule_regimes[i], rule_regimes[i+1]] += 1
        trans_pct = trans.astype(np.float64)
        row_sums = trans_pct.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_pct = trans_pct / row_sums * 100

        header = "           " + "".join(f"{regime_names[j]:>14s}" for j in range(n_regimes))
        buf.write(f"  {header}\n")
        for i in range(n_regimes):
            row = f"  {regime_names[i]:11s}"
            for j in range(n_regimes):
                row += f"{trans_pct[i,j]:13.1f}%"
            buf.write(row + "\n")

        # Forward returns
        buf.write(f"\nFORWARD RETURNS BY REGIME (next 5min):\n")
        ret_300s = features.get('ret_300s', np.zeros(n, dtype=np.float32))
        for r in range(4):
            mask = rule_regimes[:-1] == r
            if mask.sum() == 0:
                continue
            next_ret = ret_300s[bar_idx[1:len(rule_regimes)]][mask] * 10000
            buf.write(f"  After {regime_names[r]:15s}: "
                      f"mean={np.nanmean(next_ret):+6.2f}bps  "
                      f"|ret|={np.nanmean(np.abs(next_ret)):6.2f}bps  "
                      f"n={mask.sum()}\n")

    write_summary()
    with open(output_file, 'w') as f:
        f.write(buf.getvalue())
    print(buf.getvalue())


if __name__ == "__main__":
    main()
