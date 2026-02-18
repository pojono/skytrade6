#!/usr/bin/env python3
"""
v42e: More signal research — round 3

EXP K: Trade Imbalance Momentum
  - Compute rolling buy/sell volume ratio (1m, 5m, 15m)
  - When buy_ratio > P90 → momentum long? Or mean-reversion short?
  - Test both directions at multiple horizons

EXP L: Cross-Symbol Cascade Contagion
  - Load ETH + SOL liquidations simultaneously
  - When ETH has a cascade, does SOL cascade within 60s?
  - If so, can we front-run SOL cascade by detecting ETH cascade first?

EXP M: Post-Cascade Vol Expansion
  - After a cascade, vol spikes. During high-vol, TP/SL structure is favorable.
  - Enter AFTER cascade settles (5-10 min), trade the elevated vol with tight TP/SL.
  - Different from cascade MM: this trades the vol regime, not the cascade itself.

EXP N: Whale Trade Detection
  - Find P99 individual trades (by notional)
  - Does a whale buy predict short-term up? Or does it get faded?

EXP O: Large Trade Mean-Reversion
  - After a P99 single trade, does price revert within 1-5 minutes?
  - If so, fade the whale.

30 days, SOLUSDT first. Uses futures trades + liquidations.
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


def load_futures(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} futures {n}d...", end='', flush=True)
    dfs = []
    for i, d in enumerate(dates):
        f = base / f"{symbol}{d}.csv.gz"
        if f.exists():
            df = pd.read_csv(f, usecols=['timestamp', 'side', 'size', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            dfs.append(df)
        if (i+1) % 10 == 0:
            el = time.time()-t0; eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not dfs: print(" NO DATA"); return pd.DataFrame()
    r = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f" {len(r):,} ({time.time()-t0:.0f}s) [{ram_str()}]")
    return r


def load_liqs(symbol, dates, data_dir='data'):
    base = Path(data_dir) / symbol / "bybit" / "liquidations"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol} liqs {n}d...", end='', flush=True)
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
            el = time.time()-t0; eta = el/(i+1)*(n-i-1)
            print(f" [{i+1}/{n} {el:.0f}s ETA {eta:.0f}s]", end='', flush=True)
    if not recs: print(" NO DATA"); return pd.DataFrame()
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
        if not current: current = [row]
        else:
            dt = (row['timestamp'] - current[-1]['timestamp']).total_seconds()
            if dt <= window: current.append(row)
            else:
                if len(current) >= min_ev:
                    cdf = pd.DataFrame(current)
                    bn = cdf[cdf['side']=='Buy']['notional'].sum()
                    sn = cdf[cdf['side']=='Sell']['notional'].sum()
                    cascades.append({'start': cdf['timestamp'].min(), 'end': cdf['timestamp'].max(),
                                     'total_notional': bn+sn, 'buy_dominant': bn > sn, 'n_events': len(cdf)})
                current = [row]
    if len(current) >= min_ev:
        cdf = pd.DataFrame(current)
        bn = cdf[cdf['side']=='Buy']['notional'].sum()
        sn = cdf[cdf['side']=='Sell']['notional'].sum()
        cascades.append({'start': cdf['timestamp'].min(), 'end': cdf['timestamp'].max(),
                         'total_notional': bn+sn, 'buy_dominant': bn > sn, 'n_events': len(cdf)})
    return cascades


# ============================================================================
# EXP K: TRADE IMBALANCE MOMENTUM
# ============================================================================

def exp_k_imbalance(fut_df, bars):
    print(f"\n{'='*80}")
    print(f"  EXP K: TRADE IMBALANCE MOMENTUM")
    print(f"{'='*80}")

    # 1-min buy/sell volume
    print("  Computing buy/sell volumes...", end='', flush=True)
    buy_vol = fut_df[fut_df['side']=='Buy'].set_index('timestamp').resample('1min')['size'].sum().fillna(0)
    sell_vol = fut_df[fut_df['side']=='Sell'].set_index('timestamp').resample('1min')['size'].sum().fillna(0)
    total = buy_vol + sell_vol
    buy_ratio = buy_vol / total.replace(0, np.nan)
    print(" done")

    # Rolling imbalance at multiple windows
    close = bars['close']
    df = pd.DataFrame({'close': close}).dropna()

    for w in [5, 15, 60]:
        col = f'imb_{w}m'
        rb = buy_vol.rolling(w).sum()
        rs = sell_vol.rolling(w).sum()
        df[col] = (rb / (rb + rs).replace(0, np.nan)).reindex(df.index)

    # Forward returns
    for h in [5, 15, 60]:
        df[f'fwd_{h}m'] = df['close'].shift(-h) / df['close'] - 1

    df = df.dropna()

    # Test: does high buy imbalance predict UP or DOWN?
    for w in [5, 15, 60]:
        col = f'imb_{w}m'
        print(f"\n  {w}-MIN BUY IMBALANCE → FORWARD RETURN:")
        print(f"  {'Bucket':15s}  {'Count':>7s}  {'fwd_5m':>10s}  {'fwd_15m':>10s}  {'fwd_60m':>10s}")
        print(f"  {'-'*60}")

        for lo, hi, label in [(0, 0.35, 'sell dom'), (0.35, 0.45, 'slight sell'),
                               (0.45, 0.55, 'balanced'), (0.55, 0.65, 'slight buy'),
                               (0.65, 1.01, 'buy dom')]:
            mask = (df[col] >= lo) & (df[col] < hi)
            sub = df[mask]
            if len(sub) < 50: continue
            f5 = sub['fwd_5m'].mean()*10000
            f15 = sub['fwd_15m'].mean()*10000
            f60 = sub['fwd_60m'].mean()*10000
            print(f"  {label:15s}  {len(sub):>7,d}  {f5:>+9.2f}  {f15:>+9.2f}  {f60:>+9.2f}")

    # Strategy test: momentum (follow imbalance) vs reversion (fade imbalance)
    print(f"\n  STRATEGY TEST (15m imbalance, 15m hold):")
    imb = df['imb_15m']
    fwd = df['fwd_15m']

    for thresh in [0.60, 0.65, 0.70]:
        # Momentum: buy when buy_dom, sell when sell_dom
        mom_long = fwd[imb > thresh].dropna()
        mom_short = -fwd[imb < (1-thresh)].dropna()
        mom_all = pd.concat([mom_long, mom_short])
        # Subsample for cooldown
        mom_sampled = mom_all.iloc[::15]
        if len(mom_sampled) >= 10:
            avg = mom_sampled.mean()*10000
            wr = (mom_sampled > 0).mean()*100
            net = avg - (MAKER_FEE + TAKER_FEE)*10000
            flag = "✅" if net > 0 else "  "
            print(f"  {flag} MOMENTUM imb>{thresh:.2f}  n={len(mom_sampled):5d}  wr={wr:5.1f}%  "
                  f"gross={avg:+6.2f}bps  net={net:+6.2f}bps")

        # Reversion: sell when buy_dom, buy when sell_dom
        rev_long = fwd[imb < (1-thresh)].dropna()
        rev_short = -fwd[imb > thresh].dropna()
        rev_all = pd.concat([rev_long, rev_short])
        rev_sampled = rev_all.iloc[::15]
        if len(rev_sampled) >= 10:
            avg = rev_sampled.mean()*10000
            wr = (rev_sampled > 0).mean()*100
            net = avg - (MAKER_FEE + TAKER_FEE)*10000
            flag = "✅" if net > 0 else "  "
            print(f"  {flag} REVERSION imb>{thresh:.2f}  n={len(rev_sampled):5d}  wr={wr:5.1f}%  "
                  f"gross={avg:+6.2f}bps  net={net:+6.2f}bps")


# ============================================================================
# EXP L: CROSS-SYMBOL CASCADE CONTAGION
# ============================================================================

def exp_l_contagion(dates):
    print(f"\n{'='*80}")
    print(f"  EXP L: CROSS-SYMBOL CASCADE CONTAGION")
    print(f"{'='*80}")

    # Load liquidations for ETH and SOL
    eth_liq = load_liqs('ETHUSDT', dates)
    sol_liq = load_liqs('SOLUSDT', dates)

    if eth_liq.empty or sol_liq.empty:
        print("  Missing data"); return

    eth_cascades = detect_cascades(eth_liq, pct_thresh=95)
    sol_cascades = detect_cascades(sol_liq, pct_thresh=95)
    print(f"  ETH cascades: {len(eth_cascades)}, SOL cascades: {len(sol_cascades)}")

    # For each ETH cascade, check if SOL cascades within various windows
    print(f"\n  ETH CASCADE → SOL CASCADE CONTAGION:")
    print(f"  {'Window':10s}  {'ETH→SOL':>8s}  {'Rate':>6s}  {'SOL→ETH':>8s}  {'Rate':>6s}")
    print(f"  {'-'*50}")

    for window_sec in [30, 60, 120, 300]:
        eth_to_sol = 0
        for ec in eth_cascades:
            for sc in sol_cascades:
                dt = (sc['start'] - ec['end']).total_seconds()
                if 0 < dt <= window_sec:
                    eth_to_sol += 1
                    break

        sol_to_eth = 0
        for sc in sol_cascades:
            for ec in eth_cascades:
                dt = (ec['start'] - sc['end']).total_seconds()
                if 0 < dt <= window_sec:
                    sol_to_eth += 1
                    break

        eth_rate = eth_to_sol / len(eth_cascades) * 100 if eth_cascades else 0
        sol_rate = sol_to_eth / len(sol_cascades) * 100 if sol_cascades else 0
        print(f"  {window_sec:>4d}s       {eth_to_sol:>8d}  {eth_rate:>5.1f}%  {sol_to_eth:>8d}  {sol_rate:>5.1f}%")

    # Strategy: when ETH cascades, immediately enter SOL cascade MM
    # (front-run the SOL cascade)
    print(f"\n  FRONT-RUN STRATEGY: ETH cascade → enter SOL MM immediately")

    # Load SOL futures for bars
    sol_fut = load_futures('SOLUSDT', dates)
    sol_bars = sol_fut.set_index('timestamp')['price'].resample('1min').agg(
        open='first', high='max', low='min', close='last').dropna()
    del sol_fut; gc.collect()

    for offset, tp, sl in [(0.15, 0.15, 0.50), (0.20, 0.20, 0.50), (0.10, 0.10, 0.25)]:
        trades = []
        last_time = None
        for ec in eth_cascades:
            if last_time and (ec['end'] - last_time).total_seconds() < 300: continue
            idx = sol_bars.index.searchsorted(ec['end'])
            if idx >= len(sol_bars) - 30 or idx < 1: continue
            price = sol_bars.iloc[idx]['close']
            # Use ETH cascade direction for SOL (contagion = same direction)
            is_long = ec['buy_dominant']
            if is_long:
                lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
            else:
                lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)
            filled = False
            for j in range(idx, min(idx+30, len(sol_bars))):
                b = sol_bars.iloc[j]
                if is_long and b['low'] <= lim: filled=True; fi=j; break
                elif not is_long and b['high'] >= lim: filled=True; fi=j; break
            if not filled: continue
            ep = None; er = 'timeout'
            for k in range(fi, min(fi+30, len(sol_bars))):
                b = sol_bars.iloc[k]
                if is_long:
                    if b['low'] <= sl_p: ep=sl_p; er='sl'; break
                    if b['high'] >= tp_p: ep=tp_p; er='tp'; break
                else:
                    if b['high'] >= sl_p: ep=sl_p; er='sl'; break
                    if b['low'] <= tp_p: ep=tp_p; er='tp'; break
            if ep is None: ep = sol_bars.iloc[min(fi+30, len(sol_bars)-1)]['close']
            if is_long: gross = (ep-lim)/lim
            else: gross = (lim-ep)/lim
            fee = MAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
            trades.append(gross - fee)
            last_time = ec['end']

        if len(trades) >= 5:
            arr = np.array(trades)
            n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000; tot = arr.sum()*100
            flag = "✅" if avg > 0 else "  "
            print(f"  {flag} off={offset} tp={tp} sl={sl}  n={n:4d}  wr={wr:5.1f}%  "
                  f"avg={avg:+6.1f}bps  tot={tot:+6.2f}%")

    del eth_liq, sol_liq; gc.collect()


# ============================================================================
# EXP N+O: WHALE TRADES & LARGE TRADE MEAN-REVERSION
# ============================================================================

def exp_no_whale(fut_df, bars):
    print(f"\n{'='*80}")
    print(f"  EXP N+O: WHALE TRADES & LARGE TRADE MEAN-REVERSION")
    print(f"{'='*80}")

    fut_df['notional'] = fut_df['price'] * fut_df['size']

    # Find P95, P99 trades
    p95 = fut_df['notional'].quantile(0.95)
    p99 = fut_df['notional'].quantile(0.99)
    p999 = fut_df['notional'].quantile(0.999)
    print(f"  Trade notional: P95=${p95:,.0f}  P99=${p99:,.0f}  P99.9=${p999:,.0f}")

    close = bars['close']

    for thresh_label, thresh in [('P95', p95), ('P99', p99), ('P99.9', p999)]:
        whales = fut_df[fut_df['notional'] >= thresh].copy()
        print(f"\n  {thresh_label} WHALE TRADES (>{thresh:,.0f}): {len(whales):,}")

        # Aggregate to 1-second: count whale buys vs sells
        whale_buys = whales[whales['side']=='Buy'].set_index('timestamp').resample('1s')['notional'].sum().fillna(0)
        whale_sells = whales[whales['side']=='Sell'].set_index('timestamp').resample('1s')['notional'].sum().fillna(0)

        # Find seconds with whale activity
        whale_seconds = pd.DataFrame({
            'buy_not': whale_buys, 'sell_not': whale_sells
        }).fillna(0)
        whale_seconds = whale_seconds[(whale_seconds['buy_not'] > 0) | (whale_seconds['sell_not'] > 0)]
        whale_seconds['net_buy'] = whale_seconds['buy_not'] - whale_seconds['sell_not']
        whale_seconds['is_buy'] = whale_seconds['net_buy'] > 0

        # Forward returns from 1-min bars
        for h in [1, 5, 15]:
            whale_seconds[f'fwd_{h}m'] = close.reindex(whale_seconds.index, method='ffill').shift(-h) / \
                                          close.reindex(whale_seconds.index, method='ffill') - 1

        ws = whale_seconds.dropna()
        if len(ws) < 20:
            print(f"    Not enough data ({len(ws)} events)")
            continue

        # Momentum: does whale buy predict UP?
        buys = ws[ws['is_buy']]
        sells = ws[~ws['is_buy']]

        print(f"    Whale buys: {len(buys):,}  Whale sells: {len(sells):,}")
        print(f"    {'Direction':12s}  {'fwd_1m':>10s}  {'fwd_5m':>10s}  {'fwd_15m':>10s}")
        print(f"    {'-'*50}")

        if len(buys) >= 10:
            f1 = buys['fwd_1m'].mean()*10000
            f5 = buys['fwd_5m'].mean()*10000
            f15 = buys['fwd_15m'].mean()*10000
            print(f"    {'Whale BUY':12s}  {f1:>+9.2f}  {f5:>+9.2f}  {f15:>+9.2f}")

        if len(sells) >= 10:
            f1 = sells['fwd_1m'].mean()*10000
            f5 = sells['fwd_5m'].mean()*10000
            f15 = sells['fwd_15m'].mean()*10000
            print(f"    {'Whale SELL':12s}  {f1:>+9.2f}  {f5:>+9.2f}  {f15:>+9.2f}")

        # Momentum signal: buy after whale buy, sell after whale sell
        if len(buys) >= 10 and len(sells) >= 10:
            mom_ret = pd.concat([buys['fwd_5m'], -sells['fwd_5m']])
            mom_sampled = mom_ret.iloc[::5]  # 5-second cooldown
            avg = mom_sampled.mean()*10000
            wr = (mom_sampled > 0).mean()*100
            net = avg - (MAKER_FEE + TAKER_FEE)*10000
            flag = "✅" if net > 0 else "  "
            print(f"    {flag} MOMENTUM (follow whale) 5m:  n={len(mom_sampled):5d}  wr={wr:5.1f}%  "
                  f"gross={avg:+6.2f}bps  net={net:+6.2f}bps")

            # Reversion: fade the whale
            rev_ret = pd.concat([-buys['fwd_5m'], sells['fwd_5m']])
            rev_sampled = rev_ret.iloc[::5]
            avg = rev_sampled.mean()*10000
            wr = (rev_sampled > 0).mean()*100
            net = avg - (MAKER_FEE + TAKER_FEE)*10000
            flag = "✅" if net > 0 else "  "
            print(f"    {flag} REVERSION (fade whale) 5m:   n={len(rev_sampled):5d}  wr={wr:5.1f}%  "
                  f"gross={avg:+6.2f}bps  net={net:+6.2f}bps")


# ============================================================================
# EXP M: POST-CASCADE VOL EXPANSION TRADE
# ============================================================================

def exp_m_post_cascade(cascades, bars):
    print(f"\n{'='*80}")
    print(f"  EXP M: POST-CASCADE VOL EXPANSION TRADE")
    print(f"{'='*80}")

    # After cascade ends, wait 5-10 min, then enter tight TP/SL trade
    # Thesis: vol is elevated, so TP/SL hits faster → more trades per unit time
    print(f"\n  POST-CASCADE ENTRY (wait after cascade, enter tight TP/SL):")

    for wait_min in [3, 5, 10, 15]:
        for tp_bps, sl_bps in [(10, 5), (15, 8), (20, 10), (8, 4)]:
            trades = []
            last_time = None
            for c in cascades:
                entry_time = c['end'] + pd.Timedelta(minutes=wait_min)
                if last_time and (entry_time - last_time).total_seconds() < 600: continue
                idx = bars.index.searchsorted(entry_time)
                if idx >= len(bars) - 30 or idx < 1: continue

                price = bars.iloc[idx]['close']
                # Direction: fade the cascade (same as cascade MM)
                is_long = c['buy_dominant']
                if is_long:
                    tp_price = price * (1 + tp_bps/10000)
                    sl_price = price * (1 - sl_bps/10000)
                else:
                    tp_price = price * (1 - tp_bps/10000)
                    sl_price = price * (1 + sl_bps/10000)

                # Check TP/SL within 30 min
                ep = None; er = 'timeout'
                for k in range(idx, min(idx+30, len(bars))):
                    b = bars.iloc[k]
                    if is_long:
                        if b['low'] <= sl_price: ep=sl_price; er='sl'; break
                        if b['high'] >= tp_price: ep=tp_price; er='tp'; break
                    else:
                        if b['high'] >= sl_price: ep=sl_price; er='sl'; break
                        if b['low'] <= tp_price: ep=tp_price; er='tp'; break
                if ep is None: ep = bars.iloc[min(idx+30, len(bars)-1)]['close']

                if is_long: gross = (ep - price) / price
                else: gross = (price - ep) / price
                fee = TAKER_FEE + (MAKER_FEE if er=='tp' else TAKER_FEE)
                trades.append(gross - fee)
                last_time = entry_time

            if len(trades) >= 10:
                arr = np.array(trades)
                n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000; tot = arr.sum()*100
                flag = "✅" if avg > 0 else "  "
                print(f"  {flag} wait={wait_min:2d}m TP={tp_bps}bps SL={sl_bps}bps  "
                      f"n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  tot={tot:+6.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'SOLUSDT'
    out_file = f'results/v42e_more_ideas_{symbol}.txt'
    os.makedirs('results', exist_ok=True)
    tee = Tee(out_file)
    sys.stdout = tee
    t0 = time.time()

    n_days = 30
    dates = get_dates('2025-05-12', n_days)

    print("="*80)
    print(f"  v42e: MORE IDEAS — {symbol} — {n_days} DAYS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{ram_str()}]")
    print("="*80)

    # Load data
    fut_df = load_futures(symbol, dates)
    liq_df = load_liqs(symbol, dates)

    print("  Building 1-min bars...", end='', flush=True)
    bars = fut_df.set_index('timestamp')['price'].resample('1min').agg(
        open='first', high='max', low='min', close='last').dropna()
    print(f" {len(bars):,} bars [{ram_str()}]")

    cascades = detect_cascades(liq_df, pct_thresh=95)
    print(f"  Cascades P95: {len(cascades)}")

    # Run experiments
    exp_k_imbalance(fut_df, bars)
    print(f"\n  [{ram_str()}] after EXP K")

    exp_l_contagion(dates)
    gc.collect()
    print(f"\n  [{ram_str()}] after EXP L")

    exp_m_post_cascade(cascades, bars)
    print(f"\n  [{ram_str()}] after EXP M")

    exp_no_whale(fut_df, bars)
    print(f"\n  [{ram_str()}] after EXP N+O")

    elapsed = time.time() - t0
    print(f"\n{'#'*80}")
    print(f"  COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*80}")


if __name__ == '__main__':
    main()
