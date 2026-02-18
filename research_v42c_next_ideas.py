#!/usr/bin/env python3
"""
v42c: Next round of signal research

EXP F: Cascade MM + Seasonality + Size combined filter
  - Only trade P97 cascades during good hours (15,04,11,21 UTC)
  - Compare vs baseline (all hours, P95)

EXP G: Spot-Futures Volume Imbalance
  - When spot volume surges but futures doesn't (or vice versa) → signal?
  - Spot leading futures = informed flow in spot market
  - Futures leading spot = leveraged speculation

EXP H: Cascade Hour-of-Day Interaction
  - Do cascades at certain hours revert MORE reliably?
  - Cross cascade detection with hour → find best cascade hours

EXP I: Vol Compression → Breakout
  - When 1h realized vol drops below P20, does a breakout follow?
  - If so, can we profit with a straddle-like entry (long+short at offset)?

EXP J: Trade Flow Toxicity (VPIN-like)
  - Compute volume-synchronized probability of informed trading
  - Use as filter: only enter cascade MM when toxicity is LOW (less adverse selection)

SOLUSDT, 30 days. Expand winners.
"""

import sys, time, json, gzip, os, gc, psutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

MAKER_FEE = 0.0002
TAKER_FEE = 0.00055


def ram_str():
    p = psutil.Process().memory_info().rss / 1024**3
    a = psutil.virtual_memory().available / 1024**3
    return f"RAM={p:.1f}GB, avail={a:.1f}GB"


class Tee:
    def __init__(self, fp):
        self.file = open(fp, 'w', buffering=1)
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
    def flush(self):
        self.stdout.flush()
        self.file.flush()


def get_dates(start, n):
    base = datetime.strptime(start, '%Y-%m-%d')
    return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)]


def load_futures_trades(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time()
    n = len(dates)
    print(f"  Loading futures {n} days...", end='', flush=True)
    dfs = []
    for i, d in enumerate(dates):
        f = base / f"{symbol}{d}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, usecols=['timestamp', 'side', 'size', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            dfs.append(df)
        if (i+1) % 10 == 0:
            el = time.time() - t0; eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not dfs:
        print(" NO DATA"); return pd.DataFrame()
    r = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f" {len(r):,} trades ({time.time()-t0:.0f}s) [{ram_str()}]")
    return r


def load_spot_trades(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "spot"
    t0 = time.time()
    n = len(dates)
    print(f"  Loading spot {n} days...", end='', flush=True)
    dfs = []
    for i, d in enumerate(dates):
        f = base / f"{symbol}_{d}.csv.gz"
        if f.exists():
            df = pd.read_csv(f)
            if 'timestamp' in df.columns and 'price' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df[['timestamp', 'price', 'volume', 'side']].copy()
                df.rename(columns={'volume': 'size'}, inplace=True)
                dfs.append(df)
        if (i+1) % 10 == 0:
            el = time.time() - t0; eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not dfs:
        print(" NO DATA"); return pd.DataFrame()
    r = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f" {len(r):,} trades ({time.time()-t0:.0f}s) [{ram_str()}]")
    return r


def load_liquidations_dates(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "liquidations"
    t0 = time.time()
    n = len(dates)
    print(f"  Loading liqs {n} days...", end='', flush=True)
    recs = []
    for i, d in enumerate(dates):
        for hr in range(24):
            f = base / f"liquidation_{d}_hr{hr:02d}.jsonl.gz"
            if not f.exists(): continue
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    try:
                        data = json.loads(line)
                        if 'result' in data and 'data' in data['result']:
                            for ev in data['result']['data']:
                                recs.append({
                                    'timestamp': pd.to_datetime(ev['T'], unit='ms'),
                                    'side': ev['S'], 'volume': float(ev['v']),
                                    'price': float(ev['p']),
                                })
                    except: continue
        if (i+1) % 10 == 0:
            el = time.time() - t0; eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not recs:
        print(" NO DATA"); return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    print(f" {len(df):,} ({time.time()-t0:.0f}s) [{ram_str()}]")
    return df


def detect_cascades(liq_df, pct_thresh=95, window=60, min_ev=2):
    vol_thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= vol_thresh]
    cascades = []
    current = []
    for _, row in large.iterrows():
        if not current:
            current = [row]
        else:
            dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
            if dt <= window:
                current.append(row)
            else:
                if len(current) >= min_ev:
                    cdf = pd.DataFrame(current)
                    bn = cdf[cdf['side']=='Buy']['notional'].sum()
                    sn = cdf[cdf['side']=='Sell']['notional'].sum()
                    cascades.append({
                        'end': cdf['timestamp'].max(),
                        'total_notional': bn+sn, 'buy_dominant': bn > sn,
                        'n_events': len(cdf),
                    })
                current = [row]
    if len(current) >= min_ev:
        cdf = pd.DataFrame(current)
        bn = cdf[cdf['side']=='Buy']['notional'].sum()
        sn = cdf[cdf['side']=='Sell']['notional'].sum()
        cascades.append({
            'end': cdf['timestamp'].max(),
            'total_notional': bn+sn, 'buy_dominant': bn > sn,
            'n_events': len(cdf),
        })
    return cascades


def run_cascade_strat(cascades, bars, offset=0.20, tp=0.20, sl=0.50,
                      max_hold=30, cooldown=300):
    trades = []
    last_time = None
    for c in cascades:
        if last_time and (c['end'] - last_time).total_seconds() < cooldown: continue
        idx = bars.index.searchsorted(c['end'])
        if idx >= len(bars) - max_hold or idx < 1: continue
        price = bars.iloc[idx]['close']
        is_long = c['buy_dominant']
        if is_long:
            lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
        else:
            lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)
        filled = False
        for j in range(idx, min(idx+max_hold, len(bars))):
            b = bars.iloc[j]
            if is_long and b['low'] <= lim: filled=True; fi=j; break
            elif not is_long and b['high'] >= lim: filled=True; fi=j; break
        if not filled: continue
        ep = None; er = 'timeout'
        for k in range(fi, min(fi+max_hold, len(bars))):
            b = bars.iloc[k]
            if is_long:
                if b['low'] <= sl_p: ep=sl_p; er='sl'; break
                if b['high'] >= tp_p: ep=tp_p; er='tp'; break
            else:
                if b['high'] >= sl_p: ep=sl_p; er='sl'; break
                if b['low'] <= tp_p: ep=tp_p; er='tp'; break
        if ep is None: ep = bars.iloc[min(fi+max_hold, len(bars)-1)]['close']
        if is_long: gross = (ep-lim)/lim
        else: gross = (lim-ep)/lim
        fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
        trades.append({'net': gross-fee, 'gross': gross, 'exit': er,
                       'time': bars.index[fi], 'hour': bars.index[fi].hour,
                       'notional': c['total_notional']})
        last_time = c['end']
    return trades


def pstats(trades, label):
    if not trades:
        print(f"    {label:40s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "  "
    print(f"  {flag} {label:40s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.2f}%  sharpe={sh:+8.1f}")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh}


# ============================================================================
# EXP F: COMBINED CASCADE SIZE + HOUR FILTER
# ============================================================================

def exp_f_combined(cascades_p95, cascades_p97, bars):
    print(f"\n{'='*80}")
    print(f"  EXP F: COMBINED CASCADE SIZE + HOUR FILTER")
    print(f"{'='*80}")

    good_hours = {4, 11, 15, 21, 23}
    bad_hours = {1, 8, 14, 18}

    # Baseline: P95, all hours
    print(f"\n  BASELINE vs FILTERED:")
    t_base = run_cascade_strat(cascades_p95, bars)
    pstats(t_base, "P95 all hours (baseline)")

    # P97, all hours
    t_p97 = run_cascade_strat(cascades_p97, bars)
    pstats(t_p97, "P97 all hours")

    # P95, good hours only
    c_good = [c for c in cascades_p95 if c['end'].hour in good_hours]
    t_good = run_cascade_strat(c_good, bars)
    pstats(t_good, "P95 good hours only")

    # P95, exclude bad hours
    c_nobad = [c for c in cascades_p95 if c['end'].hour not in bad_hours]
    t_nobad = run_cascade_strat(c_nobad, bars)
    pstats(t_nobad, "P95 exclude bad hours")

    # P97, good hours only
    c_p97_good = [c for c in cascades_p97 if c['end'].hour in good_hours]
    t_p97_good = run_cascade_strat(c_p97_good, bars)
    pstats(t_p97_good, "P97 good hours only")

    # P97, exclude bad hours
    c_p97_nobad = [c for c in cascades_p97 if c['end'].hour not in bad_hours]
    t_p97_nobad = run_cascade_strat(c_p97_nobad, bars)
    pstats(t_p97_nobad, "P97 exclude bad hours")

    # Walk-forward: train 20d, test 10d
    print(f"\n  WALK-FORWARD (train=20d, test=10d):")
    split = bars.index.min() + pd.Timedelta(days=20)
    for label, cascades in [("P95 no-bad", c_nobad), ("P97 no-bad", c_p97_nobad),
                             ("P97 good-only", c_p97_good)]:
        train = [c for c in cascades if c['end'] < split]
        test = [c for c in cascades if c['end'] >= split]
        tt = run_cascade_strat(train, bars)
        te = run_cascade_strat(test, bars)
        print(f"\n  {label}:")
        pstats(tt, "TRAIN")
        pstats(te, "TEST")


# ============================================================================
# EXP G: SPOT-FUTURES VOLUME IMBALANCE
# ============================================================================

def exp_g_volume_imbalance(fut_df, spot_df, bars):
    print(f"\n{'='*80}")
    print(f"  EXP G: SPOT-FUTURES VOLUME IMBALANCE")
    print(f"{'='*80}")

    # Compute 1-min volume for both
    print("  Computing 1-min volumes...", end='', flush=True)
    fut_df['notional'] = fut_df['price'] * fut_df['size']
    fut_vol = fut_df.set_index('timestamp')['notional'].resample('1min').sum().fillna(0)

    spot_df['notional'] = spot_df['price'] * spot_df['size']
    spot_vol = spot_df.set_index('timestamp')['notional'].resample('1min').sum().fillna(0)

    df = pd.DataFrame({'fut_vol': fut_vol, 'spot_vol': spot_vol}).dropna()
    df = df[(df['fut_vol'] > 0) & (df['spot_vol'] > 0)]
    print(f" {len(df):,} bars")

    # Compute ratio and z-score
    df['ratio'] = df['fut_vol'] / df['spot_vol']
    df['ratio_ma'] = df['ratio'].rolling(60, min_periods=30).mean()
    df['ratio_std'] = df['ratio'].rolling(60, min_periods=30).std()
    df['ratio_z'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std'].replace(0, np.nan)

    # Also compute spot volume surge
    df['spot_vol_ma'] = df['spot_vol'].rolling(60, min_periods=30).mean()
    df['spot_surge'] = df['spot_vol'] / df['spot_vol_ma'].replace(0, np.nan)

    # Forward returns
    close = bars['close'].reindex(df.index).ffill()
    for h in [5, 15, 60]:
        df[f'fwd_{h}m'] = close.shift(-h) / close - 1

    df = df.dropna()

    # Test: does spot surge predict futures returns?
    print(f"\n  SPOT VOLUME SURGE → FUTURES FORWARD RETURN:")
    print(f"  {'Bucket':15s}  {'Count':>7s}  {'fwd_5m':>10s}  {'fwd_15m':>10s}  {'fwd_60m':>10s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(0, 1, 'spot quiet'), (1, 2, 'spot normal'),
                           (2, 5, 'spot 2-5x'), (5, 999, 'spot >5x')]:
        mask = (df['spot_surge'] >= lo) & (df['spot_surge'] < hi)
        sub = df[mask]
        if len(sub) < 20: continue
        f5 = sub['fwd_5m'].mean()*10000
        f15 = sub['fwd_15m'].mean()*10000
        f60 = sub['fwd_60m'].mean()*10000
        print(f"  {label:15s}  {len(sub):>7,d}  {f5:>+9.2f}  {f15:>+9.2f}  {f60:>+9.2f}")

    # Test: does futures/spot ratio z-score predict returns?
    print(f"\n  FUT/SPOT RATIO Z-SCORE → FUTURES FORWARD RETURN:")
    print(f"  {'Z bucket':15s}  {'Count':>7s}  {'fwd_5m':>10s}  {'fwd_15m':>10s}  {'fwd_60m':>10s}")
    print(f"  {'-'*60}")

    for lo, hi, label in [(-999, -2, 'z<-2 (spot dom)'), (-2, -1, '-2<z<-1'),
                           (-1, 1, '-1<z<1'), (1, 2, '1<z<2'),
                           (2, 999, 'z>2 (fut dom)')]:
        mask = (df['ratio_z'] >= lo) & (df['ratio_z'] < hi)
        sub = df[mask]
        if len(sub) < 20: continue
        f5 = sub['fwd_5m'].mean()*10000
        f15 = sub['fwd_15m'].mean()*10000
        f60 = sub['fwd_60m'].mean()*10000
        print(f"  {label:15s}  {len(sub):>7,d}  {f5:>+9.2f}  {f15:>+9.2f}  {f60:>+9.2f}")

    # Test: does spot surge DIRECTION predict futures direction?
    print(f"\n  SPOT SURGE + DIRECTION → FUTURES RETURN:")
    spot_buy = fut_df.set_index('timestamp').loc[fut_df['side'].values == 'Buy', 'notional'].resample('1min').sum().fillna(0)
    spot_sell = fut_df.set_index('timestamp').loc[fut_df['side'].values == 'Sell', 'notional'].resample('1min').sum().fillna(0)
    df['buy_pct'] = spot_buy.reindex(df.index).fillna(0) / (df['fut_vol'] + 1e-10)

    for lo, hi, label in [(0, 0.4, 'sell dominant'), (0.4, 0.6, 'balanced'),
                           (0.6, 1.01, 'buy dominant')]:
        mask = (df['buy_pct'] >= lo) & (df['buy_pct'] < hi) & (df['spot_surge'] > 2)
        sub = df[mask]
        if len(sub) < 20: continue
        f5 = sub['fwd_5m'].mean()*10000
        f15 = sub['fwd_15m'].mean()*10000
        print(f"  {label:15s} + surge  n={len(sub):>5,d}  fwd_5m={f5:>+6.2f}  fwd_15m={f15:>+6.2f}")


# ============================================================================
# EXP H: CASCADE HOUR-OF-DAY INTERACTION
# ============================================================================

def exp_h_cascade_hour(cascades, bars):
    print(f"\n{'='*80}")
    print(f"  EXP H: CASCADE HOUR-OF-DAY INTERACTION")
    print(f"{'='*80}")

    # Run strategy and tag each trade with hour
    trades = run_cascade_strat(cascades, bars)
    if not trades:
        print("  No trades"); return

    # Group by hour
    print(f"\n  CASCADE MM PERFORMANCE BY HOUR:")
    print(f"  {'Hour':>4s}  {'N':>4s}  {'WR%':>6s}  {'AvgNet':>8s}  {'TotNet':>8s}  {'Sharpe':>8s}")
    print(f"  {'-'*50}")

    hour_stats = []
    for hr in range(24):
        hr_trades = [t for t in trades if t['hour'] == hr]
        if len(hr_trades) < 3: continue
        arr = np.array([t['net'] for t in hr_trades])
        n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
        tot = arr.sum()*100; std = arr.std()
        sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
        flag = "✅" if arr.mean() > 0 else "❌"
        print(f"  {flag} {hr:02d}:00  {n:>4d}  {wr:>5.1f}%  {avg:>+7.1f}  {tot:>+7.2f}%  {sh:>+7.1f}")
        hour_stats.append({'hr': hr, 'n': n, 'wr': wr, 'avg': avg, 'sharpe': sh})

    if hour_stats:
        best = max(hour_stats, key=lambda x: x['avg'])
        worst = min(hour_stats, key=lambda x: x['avg'])
        print(f"\n  Best cascade hour:  {best['hr']:02d}:00 (avg={best['avg']:+.1f}bps, wr={best['wr']:.0f}%)")
        print(f"  Worst cascade hour: {worst['hr']:02d}:00 (avg={worst['avg']:+.1f}bps, wr={worst['wr']:.0f}%)")


# ============================================================================
# EXP I: VOL COMPRESSION → BREAKOUT
# ============================================================================

def exp_i_vol_compression(bars):
    print(f"\n{'='*80}")
    print(f"  EXP I: VOL COMPRESSION → BREAKOUT")
    print(f"{'='*80}")

    close = bars['close']
    ret_1m = close.pct_change()

    # 60-min realized vol
    vol_60m = ret_1m.rolling(60).std() * np.sqrt(60) * 10000  # in bps
    vol_60m_pct = vol_60m.rank(pct=True)

    # Forward absolute return (proxy for breakout magnitude)
    fwd_abs_60m = close.shift(-60) / close - 1
    fwd_abs_60m = fwd_abs_60m.abs() * 10000

    fwd_abs_120m = close.shift(-120) / close - 1
    fwd_abs_120m = fwd_abs_120m.abs() * 10000

    df = pd.DataFrame({
        'vol_60m': vol_60m, 'vol_pct': vol_60m_pct,
        'fwd_abs_60m': fwd_abs_60m, 'fwd_abs_120m': fwd_abs_120m,
    }).dropna()

    print(f"\n  VOL PERCENTILE → FORWARD |RETURN|:")
    print(f"  {'Vol bucket':15s}  {'Count':>7s}  {'|fwd_60m|':>10s}  {'|fwd_120m|':>10s}  {'Ratio':>7s}")
    print(f"  {'-'*55}")

    baseline_60 = df['fwd_abs_60m'].mean()
    for lo, hi, label in [(0, 0.10, 'P0-10 (quiet)'), (0.10, 0.25, 'P10-25'),
                           (0.25, 0.50, 'P25-50'), (0.50, 0.75, 'P50-75'),
                           (0.75, 0.90, 'P75-90'), (0.90, 1.01, 'P90-100 (wild)')]:
        mask = (df['vol_pct'] >= lo) & (df['vol_pct'] < hi)
        sub = df[mask]
        if len(sub) < 20: continue
        f60 = sub['fwd_abs_60m'].mean()
        f120 = sub['fwd_abs_120m'].mean()
        ratio = f60 / baseline_60
        flag = "✅" if ratio > 1.2 else "  "
        print(f"  {flag} {label:15s}  {len(sub):>7,d}  {f60:>9.1f}  {f120:>9.1f}  {ratio:>6.2f}x")

    # Strategy: when vol is in P0-10 (compressed), enter straddle
    # Buy at +offset, sell at -offset, TP at 2x offset, SL at offset
    print(f"\n  STRADDLE STRATEGY (enter when vol < P10):")
    compressed = df[df['vol_pct'] < 0.10].index

    for offset in [0.10, 0.15, 0.20]:
        wins = 0; losses = 0; total_ret = 0
        n_trades = 0
        last_entry = None

        for ts in compressed:
            if last_entry and (ts - last_entry).total_seconds() < 3600: continue
            idx = bars.index.get_loc(ts) if ts in bars.index else bars.index.searchsorted(ts)
            if idx >= len(bars) - 120: continue

            entry_price = bars.iloc[idx]['close']
            tp_dist = offset * 2 / 100
            sl_dist = offset / 100

            # Check if price moves ±tp_dist within 120 bars
            hit_up = False; hit_down = False
            for k in range(idx+1, min(idx+120, len(bars))):
                move = (bars.iloc[k]['high'] - entry_price) / entry_price
                move_down = (entry_price - bars.iloc[k]['low']) / entry_price
                if move >= tp_dist: hit_up = True
                if move_down >= tp_dist: hit_down = True
                if hit_up or hit_down: break

            if hit_up or hit_down:
                # Win: captured the breakout
                gross = tp_dist - MAKER_FEE - TAKER_FEE
                wins += 1
            else:
                # Timeout: small loss from fees
                gross = -MAKER_FEE - TAKER_FEE
                losses += 1

            total_ret += gross
            n_trades += 1
            last_entry = ts

        if n_trades >= 5:
            wr = wins / n_trades * 100
            avg = total_ret / n_trades * 10000
            flag = "✅" if total_ret > 0 else "  "
            print(f"  {flag} offset={offset:.2f}%  n={n_trades:4d}  wr={wr:5.1f}%  "
                  f"avg={avg:+6.1f}bps  total={total_ret*100:+6.2f}%")


# ============================================================================
# EXP J: TRADE FLOW TOXICITY (VPIN-like)
# ============================================================================

def exp_j_toxicity(fut_df, cascades, bars):
    print(f"\n{'='*80}")
    print(f"  EXP J: TRADE FLOW TOXICITY (VPIN-like)")
    print(f"{'='*80}")

    # Compute 1-min buy/sell volume imbalance
    print("  Computing trade flow metrics...", end='', flush=True)
    buy_vol = fut_df[fut_df['side']=='Buy'].set_index('timestamp')['size'].resample('1min').sum().fillna(0)
    sell_vol = fut_df[fut_df['side']=='Sell'].set_index('timestamp')['size'].resample('1min').sum().fillna(0)
    total_vol = buy_vol + sell_vol

    # VPIN proxy: |buy - sell| / total over rolling window
    imbalance = (buy_vol - sell_vol).abs()
    vpin_10m = imbalance.rolling(10).sum() / total_vol.rolling(10).sum().replace(0, np.nan)
    vpin_30m = imbalance.rolling(30).sum() / total_vol.rolling(30).sum().replace(0, np.nan)
    print(f" done [{ram_str()}]")

    # For each cascade, look up VPIN at cascade time
    print(f"\n  CASCADE MM PERFORMANCE BY TOXICITY LEVEL:")
    trades = run_cascade_strat(cascades, bars)
    if not trades:
        print("  No trades"); return

    # Tag each trade with VPIN
    for t in trades:
        ts = t['time']
        idx = vpin_30m.index.searchsorted(ts)
        if idx > 0 and idx < len(vpin_30m):
            t['vpin'] = vpin_30m.iloc[idx-1]
        else:
            t['vpin'] = np.nan

    valid = [t for t in trades if not np.isnan(t.get('vpin', np.nan))]
    if len(valid) < 20:
        print("  Not enough trades with VPIN data"); return

    vpins = [t['vpin'] for t in valid]
    p33 = np.percentile(vpins, 33)
    p66 = np.percentile(vpins, 66)

    print(f"  VPIN range: {min(vpins):.3f} to {max(vpins):.3f} (P33={p33:.3f}, P66={p66:.3f})")
    print(f"\n  {'Toxicity':15s}  {'N':>4s}  {'WR%':>6s}  {'AvgNet':>8s}  {'TotNet':>8s}")
    print(f"  {'-'*50}")

    for label, filt in [("LOW (safe)", lambda t: t['vpin'] < p33),
                         ("MEDIUM", lambda t: p33 <= t['vpin'] < p66),
                         ("HIGH (toxic)", lambda t: t['vpin'] >= p66)]:
        sub = [t for t in valid if filt(t)]
        if len(sub) < 5: continue
        arr = np.array([t['net'] for t in sub])
        n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000; tot = arr.sum()*100
        flag = "✅" if avg > 0 else "  "
        print(f"  {flag} {label:15s}  {n:>4d}  {wr:>5.1f}%  {avg:>+7.1f}  {tot:>+7.2f}%")

    # Key question: is LOW toxicity better for cascade MM?
    low = [t for t in valid if t['vpin'] < p33]
    high = [t for t in valid if t['vpin'] >= p66]
    if len(low) >= 5 and len(high) >= 5:
        low_avg = np.mean([t['net'] for t in low]) * 10000
        high_avg = np.mean([t['net'] for t in high]) * 10000
        edge = low_avg - high_avg
        if edge > 1:
            print(f"\n  ✅ LOW TOXICITY OUTPERFORMS HIGH BY {edge:.1f} bps — use as filter!")
        elif edge < -1:
            print(f"\n  ⚠️ HIGH TOXICITY OUTPERFORMS LOW BY {-edge:.1f} bps — cascades in toxic flow revert MORE")
        else:
            print(f"\n  ❌ No significant difference ({edge:+.1f} bps)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'SOLUSDT'
    out_file = f'results/v42c_next_ideas_{symbol}.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    print("="*80)
    print(f"  v42c: NEXT IDEAS — {symbol} — 30 DAYS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    dates = get_dates('2025-05-12', 30)

    # Load all data
    liq_df = load_liquidations_dates(symbol, dates)
    fut_df = load_futures_trades(symbol, dates)
    spot_df = load_spot_trades(symbol, dates)

    print("  Building 1-min bars...", end='', flush=True)
    bars = fut_df.set_index('timestamp')['price'].resample('1min').agg(
        open='first', high='max', low='min', close='last').dropna()
    print(f" {len(bars):,} bars")

    days = (bars.index.max() - bars.index.min()).total_seconds() / 86400
    print(f"  Period: {bars.index.min()} to {bars.index.max()} ({days:.0f} days)")
    print(f"  [{ram_str()}]")

    # Detect cascades
    cascades_p95 = detect_cascades(liq_df, pct_thresh=95)
    cascades_p97 = detect_cascades(liq_df, pct_thresh=97)
    print(f"  Cascades: P95={len(cascades_p95)}, P97={len(cascades_p97)}")

    # Run experiments
    exp_f_combined(cascades_p95, cascades_p97, bars)
    print(f"\n  [{ram_str()}] after EXP F")

    exp_g_volume_imbalance(fut_df, spot_df, bars)
    del spot_df; gc.collect()
    print(f"\n  [{ram_str()}] after EXP G")

    exp_h_cascade_hour(cascades_p95, bars)
    print(f"\n  [{ram_str()}] after EXP H")

    exp_i_vol_compression(bars)
    print(f"\n  [{ram_str()}] after EXP I")

    exp_j_toxicity(fut_df, cascades_p95, bars)
    print(f"\n  [{ram_str()}] after EXP J")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
