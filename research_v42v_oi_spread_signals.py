#!/usr/bin/env python3
"""
v42v: OI + Spread + Funding Rate Signals

EXP BBB: OI Velocity Signal
  - Track 5-second OI changes, compute rolling velocity
  - When OI drops rapidly (positions closing), fade the move
  - When OI rises rapidly (new positions), follow momentum

EXP CCC: Bid-Ask Spread Signal
  - Wide spread = stressed market = mean-reversion opportunity
  - Narrow spread = calm market = skip
  - Combine with cascade/liq signals for quality filter

EXP DDD: Funding Rate Extreme Signal
  - When funding rate is extreme (>P95 or <P5), fade it
  - High positive funding = crowded longs = short opportunity
  - High negative funding = crowded shorts = long opportunity

SOLUSDT for initial testing, walk-forward.
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


def load_bars_chunked(symbol, dates, data_dir='data', chunk_days=10):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} bars...", end='', flush=True)
    all_bars = []
    for start in range(0, n, chunk_days):
        chunk_dates = dates[start:start+chunk_days]
        dfs = []
        for d in chunk_dates:
            f = base / f"{symbol}{d}.csv.gz"
            if f.exists():
                df = pd.read_csv(f, usecols=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                dfs.append(df)
        if dfs:
            chunk = pd.concat(dfs, ignore_index=True)
            del dfs
            b = chunk.set_index('timestamp')['price'].resample('1min').agg(
                open='first', high='max', low='min', close='last').dropna()
            all_bars.append(b)
            del chunk; gc.collect()
        done = min(start+chunk_days, n)
        el = time.time()-t0
        print(f" [{done}/{n} {el:.0f}s]", end='', flush=True)
    if not all_bars: print(" NO DATA"); return pd.DataFrame()
    result = pd.concat(all_bars).sort_index()
    result = result[~result.index.duplicated(keep='first')]
    print(f" {len(result):,} bars ({time.time()-t0:.0f}s) [{ram_str()}]")
    return result


def load_ticker_chunked(symbol, dates, data_dir='data'):
    """Load ticker data (5s snapshots) and resample to 1-min."""
    base = Path(data_dir) / symbol
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} ticker...", end='', flush=True)
    all_recs = []
    loaded = 0
    for i, d in enumerate(dates):
        for hr in range(24):
            f = base / f"ticker_{d}_hr{hr:02d}.jsonl.gz"
            if not f.exists(): continue
            try:
                with gzip.open(f, 'rt') as fh:
                    for line in fh:
                        try:
                            data = json.loads(line)
                            item = data['result']['list'][0]
                            all_recs.append({
                                'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                                'oi': float(item['openInterest']),
                                'oi_value': float(item['openInterestValue']),
                                'funding_rate': float(item['fundingRate']),
                                'bid1': float(item['bid1Price']),
                                'ask1': float(item['ask1Price']),
                                'last_price': float(item['lastPrice']),
                            })
                        except: continue
            except: continue
            loaded += 1
        if (i+1) % 15 == 0:
            el = time.time()-t0
            print(f" [{i+1}/{n} {el:.0f}s]", end='', flush=True)

    if not all_recs: print(" NO DATA"); return pd.DataFrame()
    df = pd.DataFrame(all_recs).sort_values('timestamp').reset_index(drop=True)
    df['spread_bps'] = (df['ask1'] - df['bid1']) / df['last_price'] * 10000
    df['oi_pct_change'] = df['oi'].pct_change() * 100

    # Resample to 1-min
    df = df.set_index('timestamp')
    ticker_1m = df.resample('1min').agg({
        'oi': 'last',
        'oi_value': 'last',
        'funding_rate': 'last',
        'spread_bps': 'mean',
        'oi_pct_change': 'sum',
        'last_price': 'last',
    }).dropna(subset=['oi'])

    print(f" {len(ticker_1m):,} 1m bars from {loaded} files ({time.time()-t0:.0f}s) [{ram_str()}]")
    return ticker_1m


def sim_trade_trail(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30,
                    trail_act=3, trail_dist=2):
    if entry_idx >= len(bars) - max_hold or entry_idx < 1: return None
    price = bars.iloc[entry_idx]['close']
    if is_long:
        lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
    else:
        lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)
    filled = False
    for j in range(entry_idx, min(entry_idx+max_hold, len(bars))):
        b = bars.iloc[j]
        if is_long and b['low'] <= lim: filled=True; fi=j; break
        elif not is_long and b['high'] >= lim: filled=True; fi=j; break
    if not filled: return None
    ep = None; er = 'timeout'
    best_profit = 0; trailing_active = False; current_sl = sl_p
    for k in range(fi, min(fi+max_hold, len(bars))):
        b = bars.iloc[k]
        if is_long:
            cp = (b['high']-lim)/lim
            if cp > best_profit: best_profit = cp
            if best_profit >= trail_act/10000 and not trailing_active:
                trailing_active = True; current_sl = lim*(1+trail_dist/10000)
            if trailing_active:
                ns = b['high']*(1-trail_dist/10000)
                if ns > current_sl: current_sl = ns
            if b['low'] <= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; break
            if b['high'] >= tp_p: ep=tp_p; er='tp'; break
        else:
            cp = (lim-b['low'])/lim
            if cp > best_profit: best_profit = cp
            if best_profit >= trail_act/10000 and not trailing_active:
                trailing_active = True; current_sl = lim*(1-trail_dist/10000)
            if trailing_active:
                ns = b['low']*(1+trail_dist/10000)
                if ns < current_sl: current_sl = ns
            if b['high'] >= current_sl: ep=current_sl; er='trail' if trailing_active else 'sl'; break
            if b['low'] <= tp_p: ep=tp_p; er='tp'; break
    if ep is None: ep = bars.iloc[min(fi+max_hold, len(bars)-1)]['close']
    if is_long: gross = (ep-lim)/lim
    else: gross = (lim-ep)/lim
    fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
    return {'net': gross-fee, 'exit': er, 'time': bars.index[fi]}


def pstats(trades, label):
    if not trades:
        print(f"    {label:55s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "  "
    print(f"  {flag} {label:55s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.2f}%  sh={sh:+8.1f}")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot, 'sharpe': sh}


def main():
    out_file = 'results/v42v_oi_spread_signals.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    symbol = 'SOLUSDT'
    all_dates = get_dates('2025-05-12', 88)
    split_ts = pd.Timestamp('2025-07-11')

    print("="*80)
    print(f"  v42v: OI + SPREAD + FUNDING RATE SIGNALS — {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    bars = load_bars_chunked(symbol, all_dates, chunk_days=10)
    ticker = load_ticker_chunked(symbol, all_dates)
    gc.collect()

    # Align ticker to bars index
    common_idx = bars.index.intersection(ticker.index)
    print(f"  Common 1-min bars: {len(common_idx):,} / {len(bars):,}")

    if len(common_idx) < 1000:
        print("  ⚠️ Not enough ticker data, aborting")
        return

    # ══════════════════════════════════════════════════════════════════════
    # EXP BBB: OI VELOCITY SIGNAL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP BBB: OI VELOCITY SIGNAL")
    print(f"{'#'*80}")

    # OI change over rolling windows
    oi = ticker['oi'].reindex(bars.index, method='ffill')
    oi_pct_5m = oi.pct_change(5) * 100
    oi_pct_15m = oi.pct_change(15) * 100
    oi_pct_60m = oi.pct_change(60) * 100

    print(f"  OI stats: mean={oi.mean():.0f}, std_5m={oi_pct_5m.std():.4f}%")

    # Strategy: when OI drops rapidly (positions closing = squeeze), fade the price move
    for window, oi_series, label in [(5, oi_pct_5m, '5m'), (15, oi_pct_15m, '15m'), (60, oi_pct_60m, '60m')]:
        for pct_thresh in [90, 95]:
            # OI dropping = positions closing
            drop_thresh = oi_series.quantile((100-pct_thresh)/100)
            rise_thresh = oi_series.quantile(pct_thresh/100)

            # OI drop → price likely to reverse (squeeze unwind)
            oi_drop = oi_series < drop_thresh
            oi_rise = oi_series > rise_thresh

            # Determine direction from recent price move
            price_ret = bars['close'].pct_change(window)

            train_trades = []; test_trades = []
            lt = None

            # OI dropping + price falling → go long (squeeze reversal)
            for ts in bars.index[oi_drop & (price_ret < 0)]:
                if ts not in common_idx: continue
                if lt and (ts - lt).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade_trail(bars, idx, True)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            # OI dropping + price rising → go short (squeeze reversal)
            for ts in bars.index[oi_drop & (price_ret > 0)]:
                if ts not in common_idx: continue
                if lt and (ts - lt).total_seconds() < 300: continue
                idx = bars.index.get_loc(ts)
                t = sim_trade_trail(bars, idx, False)
                if t:
                    if ts < split_ts: train_trades.append(t)
                    else: test_trades.append(t)
                    lt = ts

            train_trades.sort(key=lambda t: t['time'])
            test_trades.sort(key=lambda t: t['time'])

            print(f"\n  OI drop P{100-pct_thresh} ({label}):")
            ts_r = pstats(train_trades, "TRAIN (fade squeeze)")
            te_r = pstats(test_trades, "TEST (fade squeeze)")
            if ts_r and te_r:
                oos = "✅" if te_r['tot'] > 0 else "❌"
                print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP CCC: BID-ASK SPREAD SIGNAL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP CCC: BID-ASK SPREAD SIGNAL")
    print(f"{'#'*80}")

    spread = ticker['spread_bps'].reindex(bars.index, method='ffill')
    spread_roll = spread.rolling(60, min_periods=30).mean()
    spread_std = spread.rolling(60, min_periods=30).std()

    print(f"  Spread stats: mean={spread.mean():.2f}bps, median={spread.median():.2f}bps, "
          f"P95={spread.quantile(0.95):.2f}bps")

    # Wide spread = stressed market → fade recent move
    for z_thresh in [1, 2, 3]:
        wide_spread = spread > (spread_roll + z_thresh * spread_std)
        price_ret = bars['close'].pct_change(5)

        train_trades = []; test_trades = []
        lt = None
        for ts in bars.index[wide_spread]:
            if ts not in common_idx: continue
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.get_loc(ts)
            ret = price_ret.iloc[idx] if idx < len(price_ret) else 0
            is_long = ret < 0  # fade recent move
            t = sim_trade_trail(bars, idx, is_long)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        print(f"\n  Wide spread (z>{z_thresh}):")
        ts_r = pstats(train_trades, "TRAIN (fade)")
        te_r = pstats(test_trades, "TEST (fade)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP DDD: FUNDING RATE EXTREME SIGNAL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP DDD: FUNDING RATE EXTREME SIGNAL")
    print(f"{'#'*80}")

    fr = ticker['funding_rate'].reindex(bars.index, method='ffill')
    fr_roll = fr.rolling(60*8, min_periods=60).mean()  # 8-hour rolling mean
    fr_std = fr.rolling(60*8, min_periods=60).std()

    print(f"  Funding rate stats: mean={fr.mean()*100:.4f}%, median={fr.median()*100:.4f}%, "
          f"P5={fr.quantile(0.05)*100:.4f}%, P95={fr.quantile(0.95)*100:.4f}%")

    for z_thresh in [1, 2, 3]:
        high_fr = fr > (fr_roll + z_thresh * fr_std)  # crowded longs
        low_fr = fr < (fr_roll - z_thresh * fr_std)   # crowded shorts

        train_trades = []; test_trades = []
        lt = None

        # High funding → short (fade crowded longs)
        for ts in bars.index[high_fr]:
            if ts not in common_idx: continue
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, False)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        # Low funding → long (fade crowded shorts)
        for ts in bars.index[low_fr]:
            if ts not in common_idx: continue
            if lt and (ts - lt).total_seconds() < 300: continue
            idx = bars.index.get_loc(ts)
            t = sim_trade_trail(bars, idx, True)
            if t:
                if ts < split_ts: train_trades.append(t)
                else: test_trades.append(t)
                lt = ts

        train_trades.sort(key=lambda t: t['time'])
        test_trades.sort(key=lambda t: t['time'])

        print(f"\n  Funding extreme (z>{z_thresh}):")
        ts_r = pstats(train_trades, "TRAIN (fade crowded)")
        te_r = pstats(test_trades, "TEST (fade crowded)")
        if ts_r and te_r:
            oos = "✅" if te_r['tot'] > 0 else "❌"
            print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    # ══════════════════════════════════════════════════════════════════════
    # EXP EEE: COMBINED OI + SPREAD + FUNDING
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  EXP EEE: COMBINED OI + SPREAD + FUNDING")
    print(f"{'#'*80}")

    # Best individual signals combined
    oi_drop_5m = oi_pct_5m < oi_pct_5m.quantile(0.05)
    wide_spread_2z = spread > (spread_roll + 2 * spread_std)
    price_ret = bars['close'].pct_change(5)

    # OI drop + wide spread → strongest mean-reversion signal
    combined = oi_drop_5m & wide_spread_2z

    train_trades = []; test_trades = []
    lt = None
    for ts in bars.index[combined]:
        if ts not in common_idx: continue
        if lt and (ts - lt).total_seconds() < 300: continue
        idx = bars.index.get_loc(ts)
        ret = price_ret.iloc[idx] if idx < len(price_ret) else 0
        is_long = ret < 0
        t = sim_trade_trail(bars, idx, is_long)
        if t:
            if ts < split_ts: train_trades.append(t)
            else: test_trades.append(t)
            lt = ts

    print(f"\n  OI drop + wide spread:")
    ts_r = pstats(train_trades, "TRAIN")
    te_r = pstats(test_trades, "TEST")
    if ts_r and te_r:
        oos = "✅" if te_r['tot'] > 0 else "❌"
        print(f"    {oos} train={ts_r['tot']/60:+.3f}%/d  test={te_r['tot']/28:+.3f}%/d")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
