#!/usr/bin/env python3
"""
v33: Temporal Patterns in Volatility, OI, Funding Rate & Liquidations

QUESTION: Are there systematic patterns in market microstructure by:
  1. Hour of day (UTC) — 0-23
  2. Day of week — Mon-Sun
  3. Trading session — Asia (00-08), Europe (08-16), US (16-24 UTC)
  4. Funding cycle — 8h windows around funding times (00:00, 08:00, 16:00 UTC)

METRICS:
  - Realized volatility (5-min bars)
  - OI change rate
  - Funding rate level
  - Liquidation count & notional
  - Trade volume & intensity
  - Buy/sell imbalance

START: BTC 14 days (2 full weeks) for day-of-week significance
EXPAND: ETH + longer period if patterns found
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import time
import json
import gzip
import warnings
import argparse
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
SYMBOL = "BTCUSDT"
DATES = [f"2025-05-{d:02d}" for d in range(11, 25)]  # 14 days, 2 full weeks

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


# ============================================================================
# DATA LOADING
# ============================================================================

def load_trades_day(date_str):
    path = DATA_DIR / SYMBOL / "bybit" / "futures" / f"{SYMBOL}{date_str}.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, compression='gzip',
                     usecols=['timestamp', 'price', 'size', 'side'])
    ts_s = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.float64)
    notionals = prices * sizes
    is_buy = (df['side'] == 'Buy').values.astype(np.float32)
    n_raw = len(df)
    del df; gc.collect()

    day_start = int(ts_s.min()); day_end = int(ts_s.max())
    n = day_end - day_start + 1
    off = (ts_s - day_start).astype(np.int32)

    trade_count = np.bincount(off, minlength=n).astype(np.float32)
    trade_notional = np.bincount(off, weights=notionals, minlength=n).astype(np.float32)
    buy_notional = np.bincount(off, weights=notionals * is_buy, minlength=n).astype(np.float32)

    price_last = np.full(n, np.nan, dtype=np.float32)
    _, last_idx = np.unique(off[::-1], return_index=True)
    last_idx = len(off) - 1 - last_idx
    for uo, li in zip(np.unique(off), last_idx):
        price_last[uo] = prices[li]

    del ts_s, prices, sizes, notionals, is_buy, off; gc.collect()
    return {'day_start': day_start, 'n': n, 'n_raw': n_raw,
            'trade_count': trade_count, 'trade_notional': trade_notional,
            'buy_notional': buy_notional, 'price_last': price_last}


def load_ticker_oi(dates, ts_start, n):
    arrays = {k: np.full(n, np.nan, dtype=np.float32)
              for k in ['oi', 'fr', 'bid', 'ask']}
    count = 0
    for date_str in dates:
        for f in sorted((DATA_DIR / SYMBOL / "bybit" / "ticker").glob(
                f"ticker_{date_str}_hr*.jsonl.gz")):
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        ts_s = d.get('ts', 0) // 1000
                        idx = ts_s - ts_start
                        if idx < 0 or idx >= n:
                            continue
                        data = d.get('result', {}).get('data', {})
                        if 'openInterestValue' in data:
                            arrays['oi'][idx] = float(data['openInterestValue'])
                        if 'fundingRate' in data:
                            arrays['fr'][idx] = float(data['fundingRate'])
                        if 'bid1Price' in data:
                            arrays['bid'][idx] = float(data['bid1Price'])
                        if 'ask1Price' in data:
                            arrays['ask'][idx] = float(data['ask1Price'])
                        count += 1
                    except:
                        continue
        print(f"    {date_str}: loaded ticker", flush=True)
    for k in arrays:
        arr = arrays[k]; mask = np.isnan(arr)
        if not mask.all():
            fv = np.argmin(mask)
            for i in range(fv + 1, n):
                if mask[i]: arr[i] = arr[i - 1]
    print(f"  Ticker: {count:,} updates → {n:,}s arrays", flush=True)
    return arrays


def load_liquidations_into_arrays(dates, ts_start, n):
    liq_count = np.zeros(n, dtype=np.float32)
    liq_notional = np.zeros(n, dtype=np.float32)
    liq_buy_count = np.zeros(n, dtype=np.float32)
    total = 0
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
                                ts_s = int(ev['T']) // 1000
                                idx = ts_s - ts_start
                                if idx < 0 or idx >= n:
                                    continue
                                vol = float(ev['v'])
                                price = float(ev['p'])
                                liq_count[idx] += 1
                                liq_notional[idx] += vol * price
                                if ev['S'] == 'Buy':
                                    liq_buy_count[idx] += 1
                                day_count += 1
                    except:
                        continue
        total += day_count
        print(f"    {date_str}: {day_count:>6,} liquidations", flush=True)
    print(f"  Liquidations: {total:,} total events", flush=True)
    return liq_count, liq_notional, liq_buy_count


def build_arrays(dates):
    t0 = time.time(); print_mem("start")

    print(f"\n[1/3] Loading trades...", flush=True)
    day_results = []
    for i, date_str in enumerate(dates):
        dr = load_trades_day(date_str)
        if dr:
            day_results.append(dr)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f"    {date_str}: {dr['n_raw']:,} trades ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
        gc.collect()
    if not day_results:
        return None

    ts_start = day_results[0]['day_start']
    ts_end = day_results[-1]['day_start'] + day_results[-1]['n'] - 1
    n = ts_end - ts_start + 1

    raw = {k: np.zeros(n, dtype=np.float32) for k in
           ['trade_count', 'trade_notional', 'buy_notional']}
    raw['price_last'] = np.full(n, np.nan, dtype=np.float32)
    for dr in day_results:
        o = dr['day_start'] - ts_start; l = dr['n']
        for k in ['trade_count', 'trade_notional', 'buy_notional']:
            raw[k][o:o+l] = dr[k]
        raw['price_last'][o:o+l] = dr['price_last']
    del day_results; gc.collect()

    p = raw['price_last']; mask = np.isnan(p)
    if not mask.all():
        fv = np.argmin(mask)
        for i in range(fv + 1, n):
            if mask[i]: p[i] = p[i - 1]

    print(f"\n[2/3] Loading liquidations...", flush=True)
    liq_c, liq_n, liq_b = load_liquidations_into_arrays(dates, ts_start, n)
    raw['liq_count'] = liq_c
    raw['liq_notional'] = liq_n
    raw['liq_buy_count'] = liq_b
    del liq_c, liq_n, liq_b; gc.collect()

    print_mem("before ticker")
    print(f"\n[3/3] Loading ticker...", flush=True)
    ticker = load_ticker_oi(dates, ts_start, n)
    raw.update(ticker); del ticker; gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Built {n:,} seconds ({n/86400:.1f} days) in {elapsed:.0f}s", flush=True)
    print_mem("arrays done")
    return raw, ts_start, n


# ============================================================================
# AGGREGATE INTO 5-MINUTE BARS
# ============================================================================

def build_5min_bars(raw, ts_start, n):
    """Aggregate per-second data into 5-minute bars with temporal labels."""
    t0 = time.time()
    bar_size = 300  # 5 minutes
    n_bars = n // bar_size

    p = raw['price_last'].astype(np.float64)

    bars = []
    for i in range(n_bars):
        s = i * bar_size
        e = s + bar_size

        bar_ts = ts_start + s
        bar_p = p[s:e]

        # Skip if no valid price
        if np.isnan(bar_p).all() or bar_p[0] <= 0 or bar_p[-1] <= 0:
            continue

        # Realized vol: std of 1-second log returns within bar
        valid = (bar_p[1:] > 0) & (bar_p[:-1] > 0)
        if valid.sum() > 10:
            log_rets = np.log(bar_p[1:][valid] / bar_p[:-1][valid])
            vol = np.std(log_rets) * 10000  # in bps per second
        else:
            vol = 0.0

        # Price range
        p_valid = bar_p[bar_p > 0]
        if len(p_valid) > 0:
            bar_range = (p_valid.max() - p_valid.min()) / p_valid.mean() * 10000
        else:
            bar_range = 0.0

        # Return
        ret = (bar_p[-1] - bar_p[0]) / bar_p[0] * 10000 if bar_p[0] > 0 else 0.0

        # Trade metrics
        tc = raw['trade_count'][s:e].sum()
        tn = raw['trade_notional'][s:e].sum()
        bn = raw['buy_notional'][s:e].sum()
        buy_ratio = bn / tn if tn > 0 else 0.5

        # Liquidations
        lc = raw['liq_count'][s:e].sum()
        ln = raw['liq_notional'][s:e].sum()
        lb = raw['liq_buy_count'][s:e].sum()
        liq_buy_ratio = lb / lc if lc > 0 else 0.5

        # OI change
        oi_start = raw['oi'][s]
        oi_end = raw['oi'][e - 1] if e - 1 < n else raw['oi'][-1]
        oi_chg = (oi_end - oi_start) / oi_start * 10000 if oi_start > 0 and not np.isnan(oi_start) else 0.0

        # Funding rate (snapshot at bar start)
        fr = raw['fr'][s] if not np.isnan(raw['fr'][s]) else 0.0

        # Spread
        bid = raw['bid'][s]; ask = raw['ask'][s]
        spread = (ask - bid) / ((ask + bid) / 2) * 10000 if bid > 0 and ask > 0 and not (np.isnan(bid) or np.isnan(ask)) else 0.0

        # Temporal labels
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(bar_ts, tz=timezone.utc)
        hour = dt.hour
        minute = dt.minute
        weekday = dt.weekday()  # 0=Mon, 6=Sun
        weekday_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday]

        # Trading session
        if 0 <= hour < 8:
            session = 'Asia'
        elif 8 <= hour < 16:
            session = 'Europe'
        else:
            session = 'US'

        # Funding cycle: minutes until next funding (00:00, 08:00, 16:00 UTC)
        minutes_in_day = hour * 60 + minute
        funding_times = [0, 480, 960, 1440]  # 00:00, 08:00, 16:00, 24:00
        min_to_funding = min(ft - minutes_in_day for ft in funding_times if ft > minutes_in_day) if any(ft > minutes_in_day for ft in funding_times) else 1440 - minutes_in_day
        hours_to_funding = min_to_funding / 60

        # Funding window label
        if min_to_funding <= 30:
            funding_phase = 'pre_funding_30m'
        elif min_to_funding <= 60:
            funding_phase = 'pre_funding_1h'
        elif hours_to_funding >= 7:
            funding_phase = 'post_funding_1h'
        elif hours_to_funding >= 6.5:
            funding_phase = 'post_funding_1.5h'
        else:
            funding_phase = 'mid_cycle'

        bars.append({
            'ts': bar_ts, 'hour': hour, 'minute': minute,
            'weekday': weekday, 'weekday_name': weekday_name,
            'session': session,
            'hours_to_funding': hours_to_funding,
            'funding_phase': funding_phase,
            'vol_bps': vol,
            'range_bps': bar_range,
            'ret_bps': ret,
            'abs_ret_bps': abs(ret),
            'trade_count': tc,
            'trade_notional': tn,
            'buy_ratio': buy_ratio,
            'liq_count': lc,
            'liq_notional': ln,
            'liq_buy_ratio': liq_buy_ratio,
            'oi_chg_bps': oi_chg,
            'fr': fr,
            'spread_bps': spread,
        })

    df = pd.DataFrame(bars)
    elapsed = time.time() - t0
    print(f"  Built {len(df):,} 5-min bars in {elapsed:.1f}s", flush=True)
    return df


# ============================================================================
# ANALYSIS: HOUR OF DAY
# ============================================================================

def analyze_hour_of_day(df):
    print(f"\n{'='*70}")
    print(f"HOUR OF DAY ANALYSIS (UTC)")
    print(f"{'='*70}")

    metrics = ['vol_bps', 'range_bps', 'abs_ret_bps', 'trade_notional',
               'liq_count', 'liq_notional', 'oi_chg_bps', 'spread_bps']
    labels = ['Vol (bps/s)', 'Range (bps)', '|Ret| (bps)', 'Volume ($)',
              'Liq Count', 'Liq Not ($)', 'OI Chg (bps)', 'Spread (bps)']

    hourly = df.groupby('hour')

    # Print header
    print(f"\n  {'Hour':>4s}", end='')
    for l in labels:
        print(f"  {l:>12s}", end='')
    print(f"  {'N':>5s}")
    print(f"  {'-'*120}")

    hour_stats = {}
    for h in range(24):
        sub = df[df['hour'] == h]
        if len(sub) < 5:
            continue
        row = {'n': len(sub)}
        print(f"  {h:>4d}", end='')
        for m, l in zip(metrics, labels):
            val = sub[m].mean()
            row[m] = val
            if 'notional' in m.lower() or 'volume' in m.lower():
                print(f"  {val:>12,.0f}", end='')
            else:
                print(f"  {val:>12.3f}", end='')
        print(f"  {len(sub):>5d}")
        hour_stats[h] = row

    # Statistical test: Kruskal-Wallis for each metric
    print(f"\n  Kruskal-Wallis test (H0: no difference across hours):")
    print(f"  {'Metric':>20s}  {'H-stat':>8s}  {'p-value':>10s}  {'Significant':>12s}")
    print(f"  {'-'*55}")

    for m, l in zip(metrics, labels):
        groups = [df[df['hour'] == h][m].values for h in range(24) if len(df[df['hour'] == h]) > 5]
        if len(groups) < 3:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "no"
        print(f"  {l:>20s}  {h_stat:>8.1f}  {p_val:>10.6f}  {sig:>12s}")

    # Find peak and trough hours for key metrics
    print(f"\n  Peak/Trough hours:")
    for m, l in zip(['vol_bps', 'liq_count', 'trade_notional', 'oi_chg_bps'], 
                    ['Volatility', 'Liquidations', 'Volume', 'OI Change']):
        means = df.groupby('hour')[m].mean()
        peak_h = means.idxmax()
        trough_h = means.idxmin()
        ratio = means.max() / max(means.min(), 1e-10)
        print(f"  {l:>15s}: peak={peak_h:02d}:00 ({means.max():.3f}), "
              f"trough={trough_h:02d}:00 ({means.min():.3f}), ratio={ratio:.2f}x")

    return hour_stats


# ============================================================================
# ANALYSIS: DAY OF WEEK
# ============================================================================

def analyze_day_of_week(df):
    print(f"\n{'='*70}")
    print(f"DAY OF WEEK ANALYSIS")
    print(f"{'='*70}")

    metrics = ['vol_bps', 'range_bps', 'abs_ret_bps', 'trade_notional',
               'liq_count', 'liq_notional', 'oi_chg_bps', 'spread_bps']
    labels = ['Vol (bps/s)', 'Range (bps)', '|Ret| (bps)', 'Volume ($)',
              'Liq Count', 'Liq Not ($)', 'OI Chg (bps)', 'Spread (bps)']

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    print(f"\n  {'Day':>4s}", end='')
    for l in labels:
        print(f"  {l:>12s}", end='')
    print(f"  {'N':>5s}")
    print(f"  {'-'*120}")

    for wd in range(7):
        sub = df[df['weekday'] == wd]
        if len(sub) < 5:
            continue
        print(f"  {day_names[wd]:>4s}", end='')
        for m, l in zip(metrics, labels):
            val = sub[m].mean()
            if 'notional' in m.lower() or 'volume' in m.lower():
                print(f"  {val:>12,.0f}", end='')
            else:
                print(f"  {val:>12.3f}", end='')
        print(f"  {len(sub):>5d}")

    # Weekday vs Weekend
    print(f"\n  Weekday vs Weekend:")
    wd_mask = df['weekday'] < 5
    we_mask = df['weekday'] >= 5

    print(f"  {'':>20s}  {'Weekday':>12s}  {'Weekend':>12s}  {'Ratio':>8s}  {'t-stat':>8s}  {'p-value':>10s}")
    print(f"  {'-'*75}")

    for m, l in zip(metrics, labels):
        wd_vals = df[wd_mask][m].values
        we_vals = df[we_mask][m].values
        wd_mean = np.mean(wd_vals)
        we_mean = np.mean(we_vals)
        ratio = wd_mean / max(we_mean, 1e-10)
        t_stat, p_val = stats.ttest_ind(wd_vals, we_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        if 'notional' in m.lower() or 'volume' in m.lower():
            print(f"  {l:>20s}  {wd_mean:>12,.0f}  {we_mean:>12,.0f}  {ratio:>7.2f}x  {t_stat:>8.2f}  {p_val:>9.4f} {sig}")
        else:
            print(f"  {l:>20s}  {wd_mean:>12.3f}  {we_mean:>12.3f}  {ratio:>7.2f}x  {t_stat:>8.2f}  {p_val:>9.4f} {sig}")

    # Kruskal-Wallis across all 7 days
    print(f"\n  Kruskal-Wallis across 7 days:")
    print(f"  {'Metric':>20s}  {'H-stat':>8s}  {'p-value':>10s}  {'Significant':>12s}")
    print(f"  {'-'*55}")
    for m, l in zip(metrics, labels):
        groups = [df[df['weekday'] == wd][m].values for wd in range(7) if len(df[df['weekday'] == wd]) > 5]
        if len(groups) < 3:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "no"
        print(f"  {l:>20s}  {h_stat:>8.1f}  {p_val:>10.6f}  {sig:>12s}")


# ============================================================================
# ANALYSIS: TRADING SESSION
# ============================================================================

def analyze_sessions(df):
    print(f"\n{'='*70}")
    print(f"TRADING SESSION ANALYSIS (Asia 00-08, Europe 08-16, US 16-24 UTC)")
    print(f"{'='*70}")

    metrics = ['vol_bps', 'range_bps', 'abs_ret_bps', 'trade_notional',
               'liq_count', 'liq_notional', 'oi_chg_bps', 'spread_bps', 'buy_ratio']
    labels = ['Vol (bps/s)', 'Range (bps)', '|Ret| (bps)', 'Volume ($)',
              'Liq Count', 'Liq Not ($)', 'OI Chg (bps)', 'Spread (bps)', 'Buy Ratio']

    sessions = ['Asia', 'Europe', 'US']

    print(f"\n  {'Session':>8s}", end='')
    for l in labels:
        print(f"  {l:>12s}", end='')
    print(f"  {'N':>5s}")
    print(f"  {'-'*130}")

    for sess in sessions:
        sub = df[df['session'] == sess]
        if len(sub) < 5:
            continue
        print(f"  {sess:>8s}", end='')
        for m, l in zip(metrics, labels):
            val = sub[m].mean()
            if 'notional' in m.lower() or 'volume' in m.lower():
                print(f"  {val:>12,.0f}", end='')
            else:
                print(f"  {val:>12.3f}", end='')
        print(f"  {len(sub):>5d}")

    # Pairwise comparisons
    print(f"\n  Pairwise t-tests (vol_bps):")
    for i, s1 in enumerate(sessions):
        for s2 in sessions[i+1:]:
            v1 = df[df['session'] == s1]['vol_bps'].values
            v2 = df[df['session'] == s2]['vol_bps'].values
            t_stat, p_val = stats.ttest_ind(v1, v2)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    {s1} vs {s2}: t={t_stat:+.2f}, p={p_val:.4f} {sig}")

    # Session × Day of Week interaction
    print(f"\n  Session × Day of Week (vol_bps mean):")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"  {'':>8s}", end='')
    for d in day_names:
        print(f"  {d:>6s}", end='')
    print()
    print(f"  {'-'*58}")
    for sess in sessions:
        print(f"  {sess:>8s}", end='')
        for wd in range(7):
            sub = df[(df['session'] == sess) & (df['weekday'] == wd)]
            if len(sub) > 0:
                print(f"  {sub['vol_bps'].mean():>6.3f}", end='')
            else:
                print(f"  {'n/a':>6s}", end='')
        print()


# ============================================================================
# ANALYSIS: FUNDING CYCLE
# ============================================================================

def analyze_funding_cycle(df):
    print(f"\n{'='*70}")
    print(f"FUNDING CYCLE ANALYSIS")
    print(f"{'='*70}")

    # Bin by hours-to-funding in 30-min increments
    df = df.copy()
    df['htf_bin'] = (df['hours_to_funding'] * 2).astype(int) / 2  # Round to nearest 0.5h

    metrics = ['vol_bps', 'liq_count', 'trade_notional', 'oi_chg_bps', 'spread_bps', 'abs_ret_bps']
    labels = ['Vol (bps/s)', 'Liq Count', 'Volume ($)', 'OI Chg (bps)', 'Spread (bps)', '|Ret| (bps)']

    print(f"\n  Hours-to-funding profile:")
    print(f"  {'HTF':>5s}", end='')
    for l in labels:
        print(f"  {l:>12s}", end='')
    print(f"  {'N':>5s}")
    print(f"  {'-'*90}")

    for htf in sorted(df['htf_bin'].unique()):
        sub = df[df['htf_bin'] == htf]
        if len(sub) < 5:
            continue
        print(f"  {htf:>5.1f}", end='')
        for m in metrics:
            val = sub[m].mean()
            if 'notional' in m.lower() or 'volume' in m.lower():
                print(f"  {val:>12,.0f}", end='')
            else:
                print(f"  {val:>12.3f}", end='')
        print(f"  {len(sub):>5d}")

    # Funding phase comparison
    print(f"\n  Funding phase comparison:")
    phases = ['post_funding_1h', 'post_funding_1.5h', 'mid_cycle', 'pre_funding_1h', 'pre_funding_30m']
    phase_labels = ['Post 0-1h', 'Post 1-1.5h', 'Mid cycle', 'Pre 1h', 'Pre 30m']

    print(f"  {'Phase':>15s}", end='')
    for l in labels:
        print(f"  {l:>12s}", end='')
    print(f"  {'N':>5s}")
    print(f"  {'-'*95}")

    for phase, plabel in zip(phases, phase_labels):
        sub = df[df['funding_phase'] == phase]
        if len(sub) < 5:
            continue
        print(f"  {plabel:>15s}", end='')
        for m in metrics:
            val = sub[m].mean()
            if 'notional' in m.lower() or 'volume' in m.lower():
                print(f"  {val:>12,.0f}", end='')
            else:
                print(f"  {val:>12.3f}", end='')
        print(f"  {len(sub):>5d}")

    # Kruskal-Wallis for funding phase
    print(f"\n  Kruskal-Wallis across funding phases:")
    print(f"  {'Metric':>20s}  {'H-stat':>8s}  {'p-value':>10s}  {'Significant':>12s}")
    print(f"  {'-'*55}")
    for m, l in zip(metrics, labels):
        groups = [df[df['funding_phase'] == p][m].values for p in phases if len(df[df['funding_phase'] == p]) > 5]
        if len(groups) < 3:
            continue
        h_stat, p_val = stats.kruskal(*groups)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "no"
        print(f"  {l:>20s}  {h_stat:>8.1f}  {p_val:>10.6f}  {sig:>12s}")


# ============================================================================
# ANALYSIS: OI DYNAMICS BY TIME
# ============================================================================

def analyze_oi_dynamics(df):
    print(f"\n{'='*70}")
    print(f"OI DYNAMICS BY TIME PERIOD")
    print(f"{'='*70}")

    # OI change distribution by session
    print(f"\n  OI change distribution by session:")
    sessions = ['Asia', 'Europe', 'US']
    print(f"  {'Session':>8s}  {'Mean':>8s}  {'Std':>8s}  {'P10':>8s}  {'P25':>8s}  {'P50':>8s}  {'P75':>8s}  {'P90':>8s}  {'%Pos':>6s}")
    print(f"  {'-'*80}")
    for sess in sessions:
        sub = df[df['session'] == sess]['oi_chg_bps']
        if len(sub) < 5:
            continue
        pcts = np.percentile(sub, [10, 25, 50, 75, 90])
        pct_pos = (sub > 0).mean() * 100
        print(f"  {sess:>8s}  {sub.mean():>+7.2f}  {sub.std():>7.2f}  "
              f"{pcts[0]:>+7.2f}  {pcts[1]:>+7.2f}  {pcts[2]:>+7.2f}  {pcts[3]:>+7.2f}  {pcts[4]:>+7.2f}  {pct_pos:>5.1f}%")

    # OI change by hour
    print(f"\n  OI change by hour (mean bps per 5-min bar):")
    print(f"  {'Hour':>4s}  {'OI Chg':>8s}  {'%Pos':>6s}  {'Vol':>8s}  {'Liqs':>6s}")
    print(f"  {'-'*40}")
    for h in range(24):
        sub = df[df['hour'] == h]
        if len(sub) < 5:
            continue
        oi_mean = sub['oi_chg_bps'].mean()
        pct_pos = (sub['oi_chg_bps'] > 0).mean() * 100
        vol_mean = sub['vol_bps'].mean()
        liq_mean = sub['liq_count'].mean()
        print(f"  {h:>4d}  {oi_mean:>+7.2f}  {pct_pos:>5.1f}%  {vol_mean:>7.3f}  {liq_mean:>5.1f}")

    # Correlation: OI change vs vol, liqs
    print(f"\n  Correlations (all bars):")
    pairs = [
        ('oi_chg_bps', 'vol_bps', 'OI change vs Vol'),
        ('oi_chg_bps', 'liq_count', 'OI change vs Liqs'),
        ('oi_chg_bps', 'abs_ret_bps', 'OI change vs |Ret|'),
        ('liq_count', 'vol_bps', 'Liqs vs Vol'),
        ('liq_count', 'abs_ret_bps', 'Liqs vs |Ret|'),
        ('fr', 'oi_chg_bps', 'FR vs OI change'),
        ('fr', 'vol_bps', 'FR vs Vol'),
        ('spread_bps', 'vol_bps', 'Spread vs Vol'),
    ]
    print(f"  {'Pair':>25s}  {'Spearman':>9s}  {'p-value':>10s}  {'Sig':>5s}")
    print(f"  {'-'*55}")
    for c1, c2, label in pairs:
        v1 = df[c1].values; v2 = df[c2].values
        valid = ~(np.isnan(v1) | np.isnan(v2))
        if valid.sum() < 20:
            continue
        rho, p_val = stats.spearmanr(v1[valid], v2[valid])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "no"
        print(f"  {label:>25s}  {rho:>+8.3f}  {p_val:>10.6f}  {sig:>5s}")


# ============================================================================
# ANALYSIS: LIQUIDATION PATTERNS
# ============================================================================

def analyze_liq_patterns(df):
    print(f"\n{'='*70}")
    print(f"LIQUIDATION TEMPORAL PATTERNS")
    print(f"{'='*70}")

    # Only bars with liquidations
    liq_bars = df[df['liq_count'] > 0]
    print(f"  Bars with liquidations: {len(liq_bars):,} / {len(df):,} ({len(liq_bars)/len(df)*100:.1f}%)")

    # Liq probability by hour
    print(f"\n  Liquidation probability by hour:")
    print(f"  {'Hour':>4s}  {'P(liq)':>7s}  {'Avg Count':>10s}  {'Avg Not':>12s}  {'Buy%':>6s}")
    print(f"  {'-'*45}")
    for h in range(24):
        sub = df[df['hour'] == h]
        if len(sub) < 5:
            continue
        p_liq = (sub['liq_count'] > 0).mean()
        avg_count = sub['liq_count'].mean()
        avg_not = sub['liq_notional'].mean()
        liq_sub = sub[sub['liq_count'] > 0]
        buy_pct = liq_sub['liq_buy_ratio'].mean() * 100 if len(liq_sub) > 0 else 50
        print(f"  {h:>4d}  {p_liq:>6.1%}  {avg_count:>9.1f}  {avg_not:>11,.0f}  {buy_pct:>5.1f}%")

    # Liq probability by day of week
    print(f"\n  Liquidation probability by day:")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"  {'Day':>4s}  {'P(liq)':>7s}  {'Avg Count':>10s}  {'Avg Not':>12s}")
    print(f"  {'-'*40}")
    for wd in range(7):
        sub = df[df['weekday'] == wd]
        if len(sub) < 5:
            continue
        p_liq = (sub['liq_count'] > 0).mean()
        avg_count = sub['liq_count'].mean()
        avg_not = sub['liq_notional'].mean()
        print(f"  {day_names[wd]:>4s}  {p_liq:>6.1%}  {avg_count:>9.1f}  {avg_not:>11,.0f}")

    # Large liquidation events (P90+) by time
    if len(liq_bars) > 20:
        p90 = liq_bars['liq_count'].quantile(0.9)
        large_liqs = liq_bars[liq_bars['liq_count'] >= p90]
        print(f"\n  Large liquidation events (≥{p90:.0f} liqs per 5min, n={len(large_liqs)}):")
        print(f"    Hour distribution:")
        hour_dist = large_liqs['hour'].value_counts().sort_index()
        for h, cnt in hour_dist.items():
            pct = cnt / len(large_liqs) * 100
            bar = '█' * int(pct / 2)
            print(f"      {h:>2d}:00  {cnt:>3d} ({pct:>4.1f}%) {bar}")


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
        end_dt = (datetime.strptime(args.end, '%Y-%m-%d') if args.end
                  else start_dt + timedelta(days=(args.days or 14) - 1))
        DATES = []
        dt = start_dt
        while dt <= end_dt:
            DATES.append(dt.strftime('%Y-%m-%d'))
            dt += timedelta(days=1)

    t_start = time.time()
    print(f"{'='*70}")
    print(f"v33: Temporal Patterns in Market Microstructure")
    print(f"{'='*70}")
    print(f"Symbol: {SYMBOL}")
    print(f"Dates:  {DATES[0]} to {DATES[-1]} ({len(DATES)} days)")
    print_mem("start")

    # Build raw arrays
    result = build_arrays(DATES)
    if result is None:
        print("ERROR: No data loaded")
        return
    raw, ts_start, n = result

    # Build 5-min bars
    print(f"\n  Building 5-minute bars...", flush=True)
    df = build_5min_bars(raw, ts_start, n)
    del raw; gc.collect()

    print(f"  Date range in bars: {df['weekday_name'].iloc[0]} to {df['weekday_name'].iloc[-1]}")
    print(f"  Days covered: {df['weekday_name'].value_counts().to_dict()}")

    # Run all analyses
    analyze_hour_of_day(df)
    analyze_day_of_week(df)
    analyze_sessions(df)
    analyze_funding_cycle(df)
    analyze_oi_dynamics(df)
    analyze_liq_patterns(df)

    # Save results
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_time:.0f}s ({total_time/60:.1f}min)")
    print_mem("final")

    date_tag = f"{DATES[0]}_to_{DATES[-1]}"
    output_file = RESULTS_DIR / f"v33_temporal_{SYMBOL}_{date_tag}.txt"
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
