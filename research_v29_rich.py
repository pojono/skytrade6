#!/usr/bin/env python3
"""
v29-rich: Exhaustive Feature Extraction at Multiple Horizons

Goal: Extract EVERY possible feature from all 4 data streams at 6 horizons,
then use feature importance to find which actually matter for regime prediction.

RAW DATA AVAILABLE:
  Trades:  timestamp, side, size, price, tickDirection
  Liqs:    T (timestamp), S (side), v (volume), p (price)
  Ticker:  bid1Price, bid1Size, ask1Price, ask1Size, fundingRate,
           indexPrice, markPrice, openInterestValue, prevPrice1h,
           prevPrice24h, price24hPcnt, turnover24h, volume24h

HORIZONS: 10s, 30s, 60s, 300s (5m), 900s (15m), 3600s (1h)

FEATURES PER HORIZON (where applicable):
  From Trades:
    - vol (realized volatility = std of log returns)
    - trade_count
    - trade_notional ($ volume)
    - buy_ratio (fraction of buy-side trades)
    - avg_trade_size (mean notional per trade)
    - large_trade_count (trades > P90 size)
    - vwap_deviation (price vs VWAP)
    - tick_imbalance (PlusTick - MinusTick count)

  From Liquidations:
    - liq_count
    - liq_notional ($)
    - liq_buy_ratio (fraction of buy-side liqs)
    - liq_avg_size (mean notional per liq)

  From Ticker (OI):
    - oi_delta (change in OI $)
    - oi_delta_pct (% change in OI)
    - oi_acceleration (delta of delta)

  From Ticker (FR):
    - fr (funding rate level)
    - fr_abs (|FR|)
    - fr_delta (change in FR)

  From Ticker (Spread/Book):
    - spread (ask - bid)
    - spread_bps (spread / mid in bps)
    - book_imbalance (bid_size - ask_size) / (bid_size + ask_size)
    - mid_price

  From Ticker (Basis):
    - basis (mark - index price)
    - basis_bps (basis / index in bps)

  Clock features (no horizon):
    - fr_time_to_funding (seconds to next 8h funding)
    - hour_of_day (0-23)
    - day_of_week (0-6)

EXPERIMENT:
  1. Build all features on 2 weeks of BTC (May 11-24)
  2. Per-chunk ML (week 1 train, week 2 test)
  3. Feature importance ranking
  4. Ablation: top-10, top-20, top-30 features vs all
  5. Horizon importance: which horizons matter most?
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

DATA_DIR = Path("data")
SYMBOL = "BTCUSDT"

# 2 weeks: May 11-24
DATES = [f"2025-05-{d:02d}" for d in range(11, 25)]
HORIZONS = [10, 30, 60, 300, 900, 3600]


def mem_gb():
    m = psutil.virtual_memory()
    return m.used / 1e9, m.available / 1e9

def print_mem(label=""):
    used, avail = mem_gb()
    print(f"  [RAM] used={used:.1f}GB avail={avail:.1f}GB {label}", flush=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_trades_day(date_str):
    """Load one day of trades, return per-second aggregates."""
    path = DATA_DIR / SYMBOL / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, compression='gzip',
                     usecols=['timestamp', 'price', 'size', 'side', 'tickDirection'])
    ts_s = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.float64)
    notionals = prices * sizes
    is_buy = (df['side'] == 'Buy').values.astype(np.float64)
    is_plus_tick = df['tickDirection'].isin(['PlusTick', 'ZeroPlusTick']).values.astype(np.float64)
    is_minus_tick = df['tickDirection'].isin(['MinusTick', 'ZeroMinusTick']).values.astype(np.float64)

    day_start = ts_s.min()
    day_end = ts_s.max()
    n = day_end - day_start + 1
    off = ts_s - day_start

    # Per-second aggregates
    trade_count = np.bincount(off, minlength=n).astype(np.float64)
    trade_notional = np.bincount(off, weights=notionals, minlength=n)
    buy_notional = np.bincount(off, weights=notionals * is_buy, minlength=n)
    buy_count = np.bincount(off, weights=is_buy, minlength=n)
    plus_ticks = np.bincount(off, weights=is_plus_tick, minlength=n)
    minus_ticks = np.bincount(off, weights=is_minus_tick, minlength=n)
    size_sum = np.bincount(off, weights=sizes, minlength=n)

    # Large trades (> P90 of notional)
    p90_not = np.percentile(notionals, 90)
    is_large = (notionals > p90_not).astype(np.float64)
    large_count = np.bincount(off, weights=is_large, minlength=n)

    # VWAP per second
    vwap_num = np.bincount(off, weights=prices * sizes, minlength=n)
    vwap_den = np.bincount(off, weights=sizes, minlength=n)

    # Last price per second
    price_last = np.full(n, np.nan, dtype=np.float64)
    _, last_idx = np.unique(off[::-1], return_index=True)
    last_idx = len(off) - 1 - last_idx
    unique_offs = np.unique(off)
    for uo, li in zip(unique_offs, last_idx):
        price_last[uo] = prices[li]

    del df, ts_s, prices, sizes, notionals, is_buy, is_plus_tick, is_minus_tick, off
    gc.collect()

    return {
        'day_start': day_start, 'n': n,
        'trade_count': trade_count, 'trade_notional': trade_notional,
        'buy_notional': buy_notional, 'buy_count': buy_count,
        'plus_ticks': plus_ticks, 'minus_ticks': minus_ticks,
        'large_count': large_count, 'size_sum': size_sum,
        'vwap_num': vwap_num, 'vwap_den': vwap_den,
        'price_last': price_last,
    }


def load_liquidations(dates):
    """Load all liquidations, return arrays."""
    records = []
    for date_str in dates:
        files = sorted((DATA_DIR / SYMBOL / "bybit" / "liquidations").glob(
            f"liquidation_{date_str}_hr*.jsonl.gz"))
        for f in files:
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
                    except:
                        continue
    if not records:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(records)
    return arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2]


def load_ticker_into_arrays(dates, ts_start, n):
    """Load ticker data directly into per-second arrays (RAM-safe, no dict accumulation)."""
    field_map = {
        'openInterestValue': 'oi', 'fundingRate': 'fr',
        'bid1Price': 'bid', 'bid1Size': 'bid_sz',
        'ask1Price': 'ask', 'ask1Size': 'ask_sz',
        'markPrice': 'mark', 'indexPrice': 'index',
    }
    arrays = {v: np.full(n, np.nan, dtype=np.float64) for v in field_map.values()}
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
        arrays[k] = pd.Series(arrays[k]).ffill().values

    print(f"  Ticker: {count:,} updates → {n:,}s arrays", flush=True)
    return arrays


# ============================================================================
# BUILD COMBINED PER-SECOND ARRAYS
# ============================================================================

def build_arrays(dates, liq_ts, liq_is_buy, liq_not):
    """Build per-second raw arrays from all sources. Ticker loaded directly into arrays."""
    t0 = time.time()

    # Load trades day by day
    day_results = []
    for date_str in dates:
        dr = load_trades_day(date_str)
        if dr:
            day_results.append(dr)
            print(f"    {date_str}: {int(dr['trade_count'].sum()):,} trades ({time.time()-t0:.1f}s)", flush=True)
        gc.collect()

    if not day_results:
        return None

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n'] - 1
    n = ts_end - ts_start + 1

    # Allocate arrays
    raw_keys = ['trade_count', 'trade_notional', 'buy_notional', 'buy_count',
                'plus_ticks', 'minus_ticks', 'large_count', 'size_sum',
                'vwap_num', 'vwap_den']
    raw = {k: np.zeros(n, dtype=np.float64) for k in raw_keys}
    raw['price_last'] = np.full(n, np.nan, dtype=np.float64)

    for dr in day_results:
        o = dr['day_start'] - ts_start
        l = dr['n']
        for k in raw_keys:
            raw[k][o:o+l] = dr[k]
        raw['price_last'][o:o+l] = dr['price_last']
    del day_results; gc.collect()

    # Forward-fill price
    raw['price_last'] = pd.Series(raw['price_last']).ffill().values

    # Liquidation per-second arrays
    raw['liq_count'] = np.zeros(n, dtype=np.float64)
    raw['liq_notional'] = np.zeros(n, dtype=np.float64)
    raw['liq_buy_count'] = np.zeros(n, dtype=np.float64)
    if len(liq_ts) > 0:
        l_off = (liq_ts - ts_start).astype(np.int64)
        valid = (l_off >= 0) & (l_off < n)
        l_off_v = l_off[valid]
        raw['liq_count'] = np.bincount(l_off_v, minlength=n).astype(np.float64)
        raw['liq_notional'] = np.bincount(l_off_v, weights=liq_not[valid], minlength=n)
        raw['liq_buy_count'] = np.bincount(l_off_v, weights=liq_is_buy[valid], minlength=n)

    print_mem("before ticker")

    # Ticker: load directly into per-second arrays (RAM-safe)
    print(f"  Loading ticker directly into arrays...", flush=True)
    ticker_arrays = load_ticker_into_arrays(dates, ts_start, n)
    raw.update(ticker_arrays)
    del ticker_arrays; gc.collect()

    print(f"  Built {n:,} seconds of raw arrays ({time.time()-t0:.1f}s)", flush=True)
    print_mem("raw arrays")
    return raw, ts_start, n


# ============================================================================
# FEATURE ENGINEERING — exhaustive, multi-horizon
# ============================================================================

def compute_features(raw, ts_start, n):
    """Compute ALL features at ALL horizons. Returns DataFrame."""
    t0 = time.time()

    def rsum(arr, w):
        cs = np.cumsum(arr)
        r = np.zeros(len(arr))
        r[w:] = cs[w:] - cs[:-w]
        return r

    def rstd(arr, w):
        cs = np.cumsum(arr); cs2 = np.cumsum(arr**2)
        r = np.full(len(arr), np.nan)
        s = cs[w:] - cs[:-w]; s2 = cs2[w:] - cs2[:-w]
        var = s2/w - (s/w)**2; np.clip(var, 0, None, out=var)
        r[w:] = np.sqrt(var)
        return r

    def rmean(arr, w):
        cs = np.cumsum(arr)
        r = np.full(len(arr), np.nan)
        r[w:] = (cs[w:] - cs[:-w]) / w
        return r

    def delta(arr, w):
        r = np.full(len(arr), np.nan)
        r[w:] = arr[w:] - arr[:-w]
        return r

    def delta_pct(arr, w):
        r = np.full(len(arr), np.nan)
        denom = arr[:-w].copy()
        denom[denom == 0] = np.nan
        r[w:] = (arr[w:] - arr[:-w]) / denom
        return r

    # Log returns
    p = raw['price_last']
    log_ret = np.zeros(n)
    v = (p[1:] > 0) & (p[:-1] > 0)
    log_ret[1:][v] = np.log(p[1:][v] / p[:-1][v])

    features = {}

    print(f"  Computing features at {len(HORIZONS)} horizons...", flush=True)

    for h in HORIZONS:
        hs = f"_{h}s"
        print(f"    Horizon {h}s...", flush=True)

        # --- TRADES ---
        features[f'vol{hs}'] = rstd(log_ret, h)
        features[f'trade_count{hs}'] = rsum(raw['trade_count'], h)
        features[f'trade_notional{hs}'] = rsum(raw['trade_notional'], h)
        features[f'large_count{hs}'] = rsum(raw['large_count'], h)

        # Buy ratio
        buy_sum = rsum(raw['buy_count'], h)
        tc_sum = rsum(raw['trade_count'], h)
        features[f'buy_ratio{hs}'] = np.where(tc_sum > 0, buy_sum / tc_sum, 0.5)

        # Buy $ ratio
        buy_not_sum = rsum(raw['buy_notional'], h)
        tn_sum = rsum(raw['trade_notional'], h)
        features[f'buy_not_ratio{hs}'] = np.where(tn_sum > 0, buy_not_sum / tn_sum, 0.5)

        # Avg trade size
        features[f'avg_trade_size{hs}'] = np.where(tc_sum > 0, tn_sum / tc_sum, 0)

        # Tick imbalance
        plus_sum = rsum(raw['plus_ticks'], h)
        minus_sum = rsum(raw['minus_ticks'], h)
        tick_total = plus_sum + minus_sum
        features[f'tick_imbalance{hs}'] = np.where(tick_total > 0,
            (plus_sum - minus_sum) / tick_total, 0)

        # VWAP deviation
        vwap_n = rsum(raw['vwap_num'], h)
        vwap_d = rsum(raw['vwap_den'], h)
        vwap = np.where(vwap_d > 0, vwap_n / vwap_d, p)
        features[f'vwap_dev{hs}'] = np.where(vwap > 0, (p - vwap) / vwap * 10000, 0)  # bps

        # --- LIQUIDATIONS ---
        features[f'liq_count{hs}'] = rsum(raw['liq_count'], h)
        features[f'liq_notional{hs}'] = rsum(raw['liq_notional'], h)

        liq_buy_sum = rsum(raw['liq_buy_count'], h)
        liq_c_sum = features[f'liq_count{hs}']
        features[f'liq_buy_ratio{hs}'] = np.where(liq_c_sum > 0, liq_buy_sum / liq_c_sum, 0.5)

        features[f'liq_avg_size{hs}'] = np.where(liq_c_sum > 0,
            features[f'liq_notional{hs}'] / liq_c_sum, 0)

        # --- OI ---
        features[f'oi_delta{hs}'] = delta(raw['oi'], h)
        features[f'oi_delta_pct{hs}'] = delta_pct(raw['oi'], h)

        # OI acceleration (delta of delta)
        if h >= 30:
            features[f'oi_accel{hs}'] = delta(features[f'oi_delta{hs}'], h)

        # --- FR ---
        features[f'fr{hs}'] = rmean(raw['fr'], h)
        features[f'fr_abs{hs}'] = rmean(np.abs(np.nan_to_num(raw['fr'])), h)
        features[f'fr_delta{hs}'] = delta(raw['fr'], h)

        # --- SPREAD / BOOK ---
        spread = raw['ask'] - raw['bid']
        mid = (raw['ask'] + raw['bid']) / 2
        features[f'spread_bps{hs}'] = rmean(
            np.where(mid > 0, spread / mid * 10000, np.nan), h)

        bid_sz = raw['bid_sz']; ask_sz = raw['ask_sz']
        total_sz = bid_sz + ask_sz
        book_imb = np.where(total_sz > 0, (bid_sz - ask_sz) / total_sz, 0)
        features[f'book_imbalance{hs}'] = rmean(book_imb, h)

        # --- BASIS ---
        basis = raw['mark'] - raw['index']
        features[f'basis_bps{hs}'] = rmean(
            np.where(raw['index'] > 0, basis / raw['index'] * 10000, np.nan), h)

        gc.collect()

    # --- CLOCK FEATURES (no horizon) ---
    seconds = np.arange(ts_start, ts_start + n)
    tod = seconds % 86400
    features['hour_of_day'] = (tod / 3600).astype(np.float64)
    features['day_of_week'] = ((seconds // 86400) % 7).astype(np.float64)
    features['fr_time_to_funding'] = np.where(
        tod < 28800, 28800 - tod,
        np.where(tod < 57600, 57600 - tod, 86400 - tod)).astype(np.float64)

    # --- CROSS-STREAM RATIOS ---
    for h in [60, 300, 900]:
        hs = f"_{h}s"
        # Liq intensity relative to trade volume
        features[f'liq_to_trade_ratio{hs}'] = np.where(
            features[f'trade_notional{hs}'] > 0,
            features[f'liq_notional{hs}'] / features[f'trade_notional{hs}'],
            0)
        # OI change relative to trade volume
        features[f'oi_delta_to_trade{hs}'] = np.where(
            features[f'trade_notional{hs}'] > 0,
            np.abs(np.nan_to_num(features[f'oi_delta{hs}'])) / features[f'trade_notional{hs}'],
            0)

    elapsed = time.time() - t0
    feat_df = pd.DataFrame(features)
    print(f"  Computed {len(feat_df.columns)} features in {elapsed:.1f}s", flush=True)
    print_mem("features done")
    return feat_df


# ============================================================================
# REGIME SWITCH DETECTION
# ============================================================================

def detect_switches(vol, ts_start, median_window=3600, threshold=2.0, min_gap=1800):
    vol_median = pd.Series(vol).rolling(median_window, min_periods=median_window//4).median().values
    valid = ~np.isnan(vol) & ~np.isnan(vol_median)
    above = valid & (vol > vol_median * threshold)
    below = valid & (vol < vol_median * 1.2)

    switches = []
    in_volatile = False
    last_sw = -999999
    for i in range(median_window + 1, len(vol)):
        if not valid[i]: continue
        if not in_volatile and above[i]:
            if (ts_start + i) - last_sw > min_gap:
                switches.append(i)
                last_sw = ts_start + i
            in_volatile = True
        elif in_volatile and below[i]:
            in_volatile = False
    return switches


# ============================================================================
# ML EXPERIMENT
# ============================================================================

def run_experiment(feat_df, switch_indices, n):
    """3-phase pipeline: (1) univariate screening, (2) redundancy removal, (3) ML on survivors."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr

    # Target: switch within 300s
    horizon = 300
    target = np.zeros(n, dtype=np.int8)
    for idx in switch_indices:
        lo = max(0, idx - horizon)
        target[lo:idx+1] = 1

    # Sample at 10s
    sample_idx = np.arange(3600, n - horizon, 10)
    split = n // 2
    train_idx = sample_idx[sample_idx < split]
    test_idx = sample_idx[sample_idx >= split]

    all_cols = list(feat_df.columns)
    X_all = feat_df.values
    y_all = target

    # Drop cols with >50% nan in train
    valid_cols = []
    for i, c in enumerate(all_cols):
        nan_frac = np.isnan(X_all[train_idx, i]).mean()
        if nan_frac < 0.5:
            valid_cols.append((i, c))
    col_indices = [i for i, _ in valid_cols]
    col_names = [c for _, c in valid_cols]
    print(f"\n  Valid features: {len(valid_cols)}/{len(all_cols)} "
          f"(dropped {len(all_cols)-len(valid_cols)} with >50% NaN)")

    X_train_raw = np.nan_to_num(X_all[train_idx][:, col_indices], nan=0, posinf=0, neginf=0)
    X_test_raw = np.nan_to_num(X_all[test_idx][:, col_indices], nan=0, posinf=0, neginf=0)
    y_train = target[train_idx]
    y_test = target[test_idx]

    print(f"  Train: {len(X_train_raw):,} ({y_train.sum():,} pos, {y_train.mean():.4f})")
    print(f"  Test:  {len(X_test_raw):,} ({y_test.sum():,} pos, {y_test.mean():.4f})")

    # =====================================================================
    # PHASE 1: Univariate Screening (fast — no ML needed)
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 1: Univariate Screening ({len(col_names)} features)")
    print(f"{'='*80}")

    uni_aucs = []
    for i, name in enumerate(col_names):
        # Univariate AUC = how well this single feature separates pos/neg
        vals = X_train_raw[:, i]
        if vals.std() < 1e-12:
            uni_aucs.append((name, i, 0.5))
            continue
        # Fast: use rank-biserial = Spearman with binary target
        rho, _ = spearmanr(vals, y_train)
        # Convert Spearman to approximate AUC: AUC ≈ 0.5 + rho/2
        approx_auc = 0.5 + abs(rho) / 2
        uni_aucs.append((name, i, approx_auc))

    uni_aucs.sort(key=lambda x: -x[2])

    print(f"\n  --- All Features Ranked by Univariate AUC ---")
    print(f"  {'Rank':>4s}  {'Feature':>35s}  {'Uni AUC':>8s}  {'Stream':>12s}  {'Horizon':>8s}  {'Bar':>20s}")
    print(f"  {'-'*95}")

    stream_map = {
        'vol': 'trades', 'trade_': 'trades', 'buy_': 'trades', 'avg_trade': 'trades',
        'large_': 'trades', 'tick_': 'trades', 'vwap_': 'trades',
        'liq_': 'liq', 'oi_': 'OI', 'fr': 'FR', 'basis': 'FR',
        'spread': 'book', 'book_': 'book',
        'hour': 'clock', 'day_': 'clock',
    }

    def get_stream(name):
        for prefix, stream in stream_map.items():
            if name.startswith(prefix):
                return stream
        return 'other'

    def get_horizon(name):
        for h in HORIZONS:
            if f"_{h}s" in name:
                return f"{h}s"
        return '-'

    for rank, (name, idx, auc) in enumerate(uni_aucs):
        bar_len = int((auc - 0.5) * 80)
        bar = '█' * max(bar_len, 0)
        sig = '***' if auc > 0.55 else '**' if auc > 0.53 else '*' if auc > 0.51 else ''
        print(f"  {rank+1:>4d}  {name:>35s}  {auc:>8.4f}  {get_stream(name):>12s}  "
              f"{get_horizon(name):>8s}  {bar} {sig}")

    # Summary by stream
    print(f"\n  --- Univariate AUC by Data Stream (mean of top feature per stream) ---")
    stream_best = defaultdict(list)
    for name, idx, auc in uni_aucs:
        stream_best[get_stream(name)].append(auc)
    print(f"  {'Stream':>12s}  {'Best AUC':>10s}  {'Mean AUC':>10s}  {'#Feat':>6s}")
    print(f"  {'-'*45}")
    for stream in ['trades', 'liq', 'OI', 'FR', 'book', 'clock']:
        aucs = stream_best.get(stream, [])
        if aucs:
            print(f"  {stream:>12s}  {max(aucs):>10.4f}  {np.mean(aucs):>10.4f}  {len(aucs):>6d}")

    # Summary by horizon
    print(f"\n  --- Univariate AUC by Horizon (mean across all streams) ---")
    horizon_best = defaultdict(list)
    for name, idx, auc in uni_aucs:
        horizon_best[get_horizon(name)].append(auc)
    print(f"  {'Horizon':>10s}  {'Best AUC':>10s}  {'Mean AUC':>10s}  {'#Feat':>6s}")
    print(f"  {'-'*40}")
    for h in [f"{x}s" for x in HORIZONS] + ['-']:
        aucs = horizon_best.get(h, [])
        if aucs:
            print(f"  {h:>10s}  {max(aucs):>10.4f}  {np.mean(aucs):>10.4f}  {len(aucs):>6d}")

    # =====================================================================
    # PHASE 2: Redundancy Removal (correlation clustering)
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 2: Redundancy Removal (|corr| > 0.90 = redundant)")
    print(f"{'='*80}")

    # Only compute corr on features with AUC > 0.505 (some signal)
    signal_features = [(name, idx, auc) for name, idx, auc in uni_aucs if auc > 0.505]
    print(f"  Features with AUC > 0.505: {len(signal_features)}/{len(uni_aucs)}")

    if len(signal_features) > 1:
        sig_indices = [idx for _, idx, _ in signal_features]
        sig_names = [name for name, _, _ in signal_features]
        sig_aucs = {name: auc for name, _, auc in signal_features}

        # Compute correlation matrix on train data (subsample for speed)
        subsample = np.random.RandomState(42).choice(len(X_train_raw),
                                                      size=min(10000, len(X_train_raw)),
                                                      replace=False)
        X_sub = X_train_raw[subsample][:, sig_indices]
        corr_matrix = np.corrcoef(X_sub.T)

        # Greedy redundancy removal: keep feature with highest AUC in each cluster
        kept = set(range(len(sig_names)))
        removed = {}  # name -> (removed_by, correlation)
        CORR_THRESH = 0.90

        # Sort by AUC descending — keep the best, remove its duplicates
        order = sorted(range(len(sig_names)), key=lambda i: -sig_aucs[sig_names[i]])
        for i in order:
            if i not in kept:
                continue
            for j in order:
                if j <= i or j not in kept:
                    continue
                if abs(corr_matrix[i, j]) > CORR_THRESH:
                    kept.discard(j)
                    removed[sig_names[j]] = (sig_names[i], corr_matrix[i, j])

        kept_features = [(sig_names[i], sig_indices[i], sig_aucs[sig_names[i]]) for i in sorted(kept)]

        print(f"  Removed {len(removed)} redundant features (|corr| > {CORR_THRESH}):")
        # Show removed features grouped by what they're redundant with
        by_parent = defaultdict(list)
        for child, (parent, corr) in removed.items():
            by_parent[parent].append((child, corr))
        for parent, children in sorted(by_parent.items(), key=lambda x: -len(x[1]))[:15]:
            child_str = ", ".join(f"{c} ({r:.2f})" for c, r in children[:4])
            if len(children) > 4:
                child_str += f", +{len(children)-4} more"
            print(f"    {parent:>35s} ← {child_str}")

        print(f"\n  Surviving features: {len(kept_features)}")
    else:
        kept_features = signal_features

    # =====================================================================
    # PHASE 3: ML on survivors only
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  PHASE 3: ML on {len(kept_features)} Non-Redundant Signal Features")
    print(f"{'='*80}")

    from sklearn.ensemble import GradientBoostingClassifier

    survivor_indices = [idx for _, idx, _ in kept_features]
    survivor_names = [name for name, _, _ in kept_features]

    X_tr = X_train_raw[:, survivor_indices]
    X_te = X_test_raw[:, survivor_indices]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Full model on survivors
    print(f"\n  Training GBM on {len(survivor_names)} features...", flush=True)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42, min_samples_leaf=50
    )
    model.fit(X_tr_s, y_train)
    y_prob = model.predict_proba(X_te_s)[:, 1]
    auc_surv = roc_auc_score(y_test, y_prob)
    print(f"  AUC ({len(survivor_names)} survivors): {auc_surv:.4f}")

    # GBM feature importance
    importances = model.feature_importances_
    imp_order = np.argsort(importances)[::-1]

    print(f"\n  --- Top 30 by GBM Importance (survivors only) ---")
    print(f"  {'Rank':>4s}  {'Feature':>35s}  {'GBM Imp':>8s}  {'Uni AUC':>8s}  {'Stream':>8s}  {'Hz':>6s}")
    print(f"  {'-'*80}")
    for rank, idx in enumerate(imp_order[:30]):
        name = survivor_names[idx]
        uni = sig_aucs.get(name, 0.5)
        print(f"  {rank+1:>4d}  {name:>35s}  {importances[idx]:>8.4f}  {uni:>8.4f}  "
              f"{get_stream(name):>8s}  {get_horizon(name):>6s}")

    # Importance by stream
    print(f"\n  --- GBM Importance by Data Stream ---")
    stream_imp = defaultdict(float)
    for i, name in enumerate(survivor_names):
        stream_imp[get_stream(name)] += importances[i]
    for stream in ['trades', 'liq', 'OI', 'FR', 'book', 'clock']:
        imp = stream_imp.get(stream, 0)
        bar = '█' * int(imp * 60)
        print(f"  {stream:>12s}  {imp:>8.4f}  {bar}")

    # Importance by horizon
    print(f"\n  --- GBM Importance by Horizon ---")
    horizon_imp = defaultdict(float)
    for i, name in enumerate(survivor_names):
        horizon_imp[get_horizon(name)] += importances[i]
    for h in [f"{x}s" for x in HORIZONS] + ['-']:
        imp = horizon_imp.get(h, 0)
        bar = '█' * int(imp * 60)
        print(f"  {h:>10s}  {imp:>8.4f}  {bar}")

    # Ablation: top-N survivors
    print(f"\n  --- Ablation: Top-N Survivors ---")
    print(f"  {'N':>6s}  {'AUC':>8s}  {'Features':>50s}")
    print(f"  {'-'*70}")
    for top_n in [3, 5, 10, 15, 20, len(survivor_names)]:
        top_n = min(top_n, len(survivor_names))
        top_idx = imp_order[:top_n]
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, min_samples_leaf=50
        )
        m.fit(X_tr_s[:, top_idx], y_train)
        yp = m.predict_proba(X_te_s[:, top_idx])[:, 1]
        auc_n = roc_auc_score(y_test, yp)
        feat_str = ", ".join(survivor_names[i] for i in top_idx[:5])
        if top_n > 5:
            feat_str += f", +{top_n-5} more"
        marker = " ← ALL" if top_n == len(survivor_names) else ""
        print(f"  {top_n:>6d}  {auc_n:>8.4f}  {feat_str}{marker}")

    # Feature set comparison
    print(f"\n  --- Feature Set Comparison (survivors only) ---")
    feature_sets = {
        'trades_only': [i for i, n in enumerate(survivor_names) if get_stream(n) == 'trades'],
        'trades+liq': [i for i, n in enumerate(survivor_names) if get_stream(n) in ('trades', 'liq')],
        'trades+liq+OI': [i for i, n in enumerate(survivor_names) if get_stream(n) in ('trades', 'liq', 'OI')],
        '+FR+book+clock': list(range(len(survivor_names))),
    }
    print(f"  {'Set':>20s}  {'#Feat':>6s}  {'AUC':>8s}")
    print(f"  {'-'*40}")
    for fs_name, fs_idx in feature_sets.items():
        if not fs_idx:
            continue
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, min_samples_leaf=50
        )
        m.fit(X_tr_s[:, fs_idx], y_train)
        yp = m.predict_proba(X_te_s[:, fs_idx])[:, 1]
        auc_fs = roc_auc_score(y_test, yp)
        print(f"  {fs_name:>20s}  {len(fs_idx):>6d}  {auc_fs:>8.4f}")

    # Per-horizon ablation
    print(f"\n  --- Per-Horizon Ablation (survivors only) ---")
    print(f"  {'Horizon':>10s}  {'#Feat':>6s}  {'AUC':>8s}")
    print(f"  {'-'*30}")
    for h in [f"{x}s" for x in HORIZONS] + ['-']:
        h_idx = [i for i, n in enumerate(survivor_names) if get_horizon(n) == h]
        if len(h_idx) < 2:
            continue
        m = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42, min_samples_leaf=50
        )
        m.fit(X_tr_s[:, h_idx], y_train)
        yp = m.predict_proba(X_te_s[:, h_idx])[:, 1]
        auc_h = roc_auc_score(y_test, yp)
        print(f"  {h:>10s}  {len(h_idx):>6d}  {auc_h:>8.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    print(f"\n{'#'*80}")
    print(f"  v29-RICH: EXHAUSTIVE FEATURE EXTRACTION")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Dates: {DATES[0]} to {DATES[-1]} ({len(DATES)} days)")
    print(f"  Horizons: {HORIZONS}")
    print(f"{'#'*80}")
    print_mem("start")

    # Load raw data
    print(f"\n  --- Loading Data ---")
    liq_ts, liq_is_buy, liq_not = load_liquidations(DATES)
    print(f"  Liquidations: {len(liq_ts):,}")

    # Build per-second arrays (ticker loaded directly into arrays, RAM-safe)
    print(f"\n  --- Building Per-Second Arrays ---")
    result = build_arrays(DATES, liq_ts, liq_is_buy, liq_not)
    del liq_ts, liq_is_buy, liq_not; gc.collect()

    if result is None:
        print("ERROR: No data!")
        return

    raw, ts_start, n = result

    # Compute features
    print(f"\n  --- Feature Engineering ---")
    feat_df = compute_features(raw, ts_start, n)
    del raw; gc.collect()

    # Detect switches
    print(f"\n  --- Detecting Regime Switches ---")
    vol_col = 'vol_60s'
    vol = feat_df[vol_col].values
    switch_indices = detect_switches(vol, ts_start)
    print(f"  Found {len(switch_indices)} switches")

    # Run experiment
    run_experiment(feat_df, switch_indices, n)

    elapsed = time.time() - t_start
    print(f"\n{'#'*80}")
    print(f"  COMPLETE: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
