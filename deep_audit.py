#!/usr/bin/env python3
"""
Deep self-audit: The numbers are too good. Let's find out why.

KEY TESTS:
1. RANDOM signal test — same trade structure, random entry direction
2. INVERTED signal test — flip long↔short, should lose money if signal matters
3. EVERY-BAR test — enter every single bar, no signal filter
4. SAME-BAR fill+exit bug check — can we fill AND exit on the same bar?
5. Trailing stop intra-bar lookahead — does trail use high to set trail then low to exit?
6. Long-only vs Short-only breakdown — is one side carrying all the profit?
7. Exit type distribution — what % are TP vs trail vs SL vs timeout?
8. Average trade P&L by exit type — are trail exits actually profitable?
9. Fill rate analysis — what % of signals actually get filled?
10. Consecutive trade overlap — are we counting overlapping trades?
"""

import sys, time, os, gc, psutil, random
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

def get_dates(start, n):
    base = datetime.strptime(start, '%Y-%m-%d')
    return [(base + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)]

def load_bars_chunked(symbol, dates, data_dir='data', chunk_days=10):
    base = Path(data_dir) / symbol / "bybit" / "futures"
    t0 = time.time(); n = len(dates)
    print(f"  Loading {symbol}...", end='', flush=True)
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


# ============================================================================
# ORIGINAL sim_trade (exact copy from research scripts)
# ============================================================================
def sim_trade_original(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30,
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
    return {'net': gross-fee, 'exit': er, 'time': bars.index[fi], 'fill_bar': fi,
            'entry_bar': entry_idx, 'gross': gross, 'fee': fee, 'is_long': is_long,
            'lim': lim, 'ep': ep}


# ============================================================================
# FIXED sim_trade — fixes the same-bar fill+exit bug
# Exit loop must start from fi+1 (bar AFTER fill), not fi
# ============================================================================
def sim_trade_fixed(bars, entry_idx, is_long, offset=0.15, tp=0.15, sl=0.50, max_hold=30,
              trail_act=3, trail_dist=2):
    if entry_idx >= len(bars) - max_hold or entry_idx < 1: return None
    price = bars.iloc[entry_idx]['close']
    if is_long:
        lim = price*(1-offset/100); tp_p = lim*(1+tp/100); sl_p = lim*(1-sl/100)
    else:
        lim = price*(1+offset/100); tp_p = lim*(1-tp/100); sl_p = lim*(1+sl/100)
    # Fill detection — start from entry_idx+1 (next bar after signal)
    filled = False
    for j in range(entry_idx+1, min(entry_idx+max_hold, len(bars))):
        b = bars.iloc[j]
        if is_long and b['low'] <= lim: filled=True; fi=j; break
        elif not is_long and b['high'] >= lim: filled=True; fi=j; break
    if not filled: return None
    # Exit detection — start from fi+1 (bar AFTER fill bar)
    ep = None; er = 'timeout'
    best_profit = 0; trailing_active = False; current_sl = sl_p
    for k in range(fi+1, min(fi+max_hold, len(bars))):
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
    return {'net': gross-fee, 'exit': er, 'time': bars.index[fi], 'fill_bar': fi,
            'entry_bar': entry_idx, 'gross': gross, 'fee': fee, 'is_long': is_long,
            'lim': lim, 'ep': ep}


def pstats(trades, label):
    if not trades:
        print(f"    {label:55s}  NO TRADES"); return None
    arr = np.array([t['net'] for t in trades])
    n = len(arr); wr = (arr>0).mean()*100; avg = arr.mean()*10000
    tot = arr.sum()*100; std = arr.std()
    sh = arr.mean()/(std+1e-10)*np.sqrt(252*24*60)
    flag = "✅" if arr.mean() > 0 else "❌"
    print(f"  {flag} {label:55s}  n={n:4d}  wr={wr:5.1f}%  avg={avg:+6.1f}bps  "
          f"tot={tot:+7.1f}%  sh={sh:+8.0f}")
    return {'n': n, 'wr': wr, 'avg': avg, 'tot': tot}


def gen_ret_iqr(bars, window=60, threshold=0.95):
    ret = bars['close'].pct_change()
    q75 = ret.rolling(window).quantile(0.75)
    q25 = ret.rolling(window).quantile(0.25)
    iqr = q75 - q25
    pct = iqr.rolling(window).rank(pct=True)
    sig = pd.Series(0, index=bars.index)
    sig[pct > threshold] = -1   # high IQR → fade → short
    sig[pct < (1-threshold)] = 1  # low IQR → fade → long
    return sig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    t_start = time.time()
    
    print("=" * 100)
    print("  DEEP SELF-AUDIT: Why are the numbers too good?")
    print("=" * 100)
    
    # Use the original test period
    dates = get_dates('2025-05-12', 88)
    split = '2025-07-11'
    
    # Test on ETH first (our "best" coin)
    for sym in ['ETHUSDT', 'DOGEUSDT']:
        bars = load_bars_chunked(sym, dates)
        if bars.empty: continue
        
        oos = bars[bars.index >= split]
        print(f"\n  OOS bars: {len(oos):,} ({oos.index[0]} to {oos.index[-1]})")
        
        # Generate the real signal
        sig = gen_ret_iqr(bars, 60, 0.95)
        sig_oos = sig[sig.index >= split]
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 1: SAME-BAR FILL+EXIT BUG CHECK ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # The original code starts the exit loop from fi (fill bar).
        # This means on the SAME bar the limit order fills, it can also
        # hit TP or trail. In reality, if your limit fills at the low of a bar,
        # you can't also exit at the high of that same bar — you don't know
        # the intra-bar sequence (did low happen before high?).
        
        print("\n  Testing original sim (exit starts on fill bar):")
        trades_orig = []
        for i in range(len(oos)):
            idx = bars.index.get_loc(oos.index[i])
            s = sig_oos.iloc[i]
            if s == 0: continue
            t = sim_trade_original(bars, idx, s > 0)
            if t: trades_orig.append(t)
        pstats(trades_orig, f"ORIGINAL ret_iqr {sym}")
        
        # Count same-bar fill+exit
        same_bar = sum(1 for t in trades_orig if t['fill_bar'] == t.get('fill_bar') and t['exit'] != 'timeout')
        same_bar_exits = {}
        for t in trades_orig:
            # Check if exit happened on fill bar by looking at exit reason
            # We need to trace this more carefully
            pass
        
        print(f"\n  Testing FIXED sim (exit starts on bar AFTER fill):")
        trades_fixed = []
        for i in range(len(oos)):
            idx = bars.index.get_loc(oos.index[i])
            s = sig_oos.iloc[i]
            if s == 0: continue
            t = sim_trade_fixed(bars, idx, s > 0)
            if t: trades_fixed.append(t)
        pstats(trades_fixed, f"FIXED ret_iqr {sym}")
        
        if trades_orig and trades_fixed:
            orig_avg = np.mean([t['net'] for t in trades_orig]) * 10000
            fixed_avg = np.mean([t['net'] for t in trades_fixed]) * 10000
            print(f"\n  ⚠️  IMPACT: original={orig_avg:+.1f}bps → fixed={fixed_avg:+.1f}bps  Δ={fixed_avg-orig_avg:+.1f}bps")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 2: TRAILING STOP INTRA-BAR LOOKAHEAD ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # On a single bar, the code:
        #   1. Uses b['high'] to compute best_profit and update trailing stop
        #   2. Then uses b['low'] to check if trailing stop was hit
        # This is LOOKAHEAD within the bar — it assumes price went to high FIRST
        # then came back to low. In reality, low might happen before high.
        # This systematically biases trail exits to be MORE profitable.
        
        print("\n  Analyzing trail exit profitability:")
        exit_stats = {}
        for t in trades_orig:
            er = t['exit']
            if er not in exit_stats: exit_stats[er] = []
            exit_stats[er].append(t['net'] * 10000)
        
        for er in ['tp', 'trail', 'sl', 'timeout']:
            if er in exit_stats:
                arr = np.array(exit_stats[er])
                print(f"    {er:8s}: n={len(arr):4d}  avg={arr.mean():+6.1f}bps  "
                      f"wr={100*(arr>0).mean():5.1f}%  "
                      f"min={arr.min():+6.1f}  max={arr.max():+6.1f}")
        
        # Check: how many trail exits happen on the same bar as trail activation?
        # This would be the worst case of intra-bar lookahead
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 3: RANDOM SIGNAL TEST ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # If random signals are also profitable, the edge is in the trade
        # structure, not the signal.
        
        print("\n  Testing RANDOM direction with same trade structure:")
        random.seed(42)
        for trial in range(3):
            trades_rand = []
            signal_indices = [i for i in range(len(oos)) if sig_oos.iloc[i] != 0]
            for i in signal_indices:
                idx = bars.index.get_loc(oos.index[i])
                direction = random.choice([True, False])  # random long/short
                t = sim_trade_original(bars, idx, direction)
                if t: trades_rand.append(t)
            pstats(trades_rand, f"RANDOM direction (seed={42+trial}) {sym}")
            random.seed(42 + trial + 1)
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 4: INVERTED SIGNAL TEST ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # Flip long↔short. If inverted signals are also profitable,
        # the signal direction doesn't matter.
        
        print("\n  Testing INVERTED signals (flip long↔short):")
        trades_inv = []
        for i in range(len(oos)):
            idx = bars.index.get_loc(oos.index[i])
            s = sig_oos.iloc[i]
            if s == 0: continue
            t = sim_trade_original(bars, idx, s < 0)  # INVERTED
            if t: trades_inv.append(t)
        pstats(trades_inv, f"INVERTED ret_iqr {sym}")
        
        if trades_orig and trades_inv:
            orig_avg = np.mean([t['net'] for t in trades_orig]) * 10000
            inv_avg = np.mean([t['net'] for t in trades_inv]) * 10000
            print(f"\n  Signal vs Inverted: original={orig_avg:+.1f}bps  inverted={inv_avg:+.1f}bps")
            if inv_avg > 0:
                print(f"  🔴 CRITICAL: Inverted signal is ALSO profitable! Signal direction may not matter!")
            else:
                print(f"  ✅ Inverted signal is negative — signal direction matters")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 5: EVERY-BAR ENTRY TEST ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # Enter on EVERY bar (no signal filter) with alternating long/short.
        # If this is also profitable, the trade structure alone generates edge.
        
        print("\n  Testing EVERY bar entry (no signal), alternating long/short:")
        trades_every_alt = []
        for i in range(0, len(oos), 30):  # every 30 bars to avoid overlap
            idx = bars.index.get_loc(oos.index[i])
            t = sim_trade_original(bars, idx, i % 2 == 0)  # alternate
            if t: trades_every_alt.append(t)
        pstats(trades_every_alt, f"EVERY-30-bars alternating {sym}")
        
        print("\n  Testing EVERY bar entry, LONG only:")
        trades_every_long = []
        for i in range(0, len(oos), 30):
            idx = bars.index.get_loc(oos.index[i])
            t = sim_trade_original(bars, idx, True)
            if t: trades_every_long.append(t)
        pstats(trades_every_long, f"EVERY-30-bars LONG only {sym}")
        
        print("\n  Testing EVERY bar entry, SHORT only:")
        trades_every_short = []
        for i in range(0, len(oos), 30):
            idx = bars.index.get_loc(oos.index[i])
            t = sim_trade_original(bars, idx, False)
            if t: trades_every_short.append(t)
        pstats(trades_every_short, f"EVERY-30-bars SHORT only {sym}")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 6: LONG vs SHORT BREAKDOWN ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        
        trades_long = [t for t in trades_orig if t['is_long']]
        trades_short = [t for t in trades_orig if not t['is_long']]
        print(f"\n  Signal-based trades breakdown:")
        pstats(trades_long, f"LONG trades only {sym}")
        pstats(trades_short, f"SHORT trades only {sym}")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 7: FILL RATE ANALYSIS ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        
        total_signals = sum(1 for i in range(len(oos)) if sig_oos.iloc[i] != 0)
        filled = len(trades_orig)
        print(f"\n  Total signals fired: {total_signals}")
        print(f"  Trades filled: {filled}")
        print(f"  Fill rate: {100*filled/total_signals:.1f}%")
        print(f"  Unfilled: {total_signals - filled} ({100*(total_signals-filled)/total_signals:.1f}%)")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 8: LIMIT OFFSET ANALYSIS ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # The 15bps limit offset creates a natural edge: you buy 15bps below
        # close, so you start with a 15bps advantage. But you also pay ~7.5bps
        # in fees. Net starting advantage: ~7.5bps.
        # The question is: does the price tend to revert after touching the limit?
        
        print("\n  Testing different offsets to see if edge scales with offset:")
        for offset in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
            trades_off = []
            for i in range(len(oos)):
                idx = bars.index.get_loc(oos.index[i])
                s = sig_oos.iloc[i]
                if s == 0: continue
                t = sim_trade_original(bars, idx, s > 0, offset=offset)
                if t: trades_off.append(t)
            if trades_off:
                arr = np.array([t['net'] for t in trades_off])
                n = len(arr); avg = arr.mean()*10000; wr = (arr>0).mean()*100
                fill_rate = 100*n/total_signals
                print(f"    offset={offset:.2f}%: n={n:4d} ({fill_rate:4.0f}% fill)  "
                      f"avg={avg:+6.1f}bps  wr={wr:5.1f}%")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 9: WHAT IF WE REMOVE TRAILING STOP? ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # Test with no trailing stop — just TP and SL
        
        print("\n  Testing WITHOUT trailing stop (trail_act=99999):")
        trades_notrail = []
        for i in range(len(oos)):
            idx = bars.index.get_loc(oos.index[i])
            s = sig_oos.iloc[i]
            if s == 0: continue
            t = sim_trade_original(bars, idx, s > 0, trail_act=99999)
            if t: trades_notrail.append(t)
        pstats(trades_notrail, f"NO trailing stop {sym}")
        
        # Exit type distribution without trail
        exit_stats_nt = {}
        for t in trades_notrail:
            er = t['exit']
            if er not in exit_stats_nt: exit_stats_nt[er] = []
            exit_stats_nt[er].append(t['net'] * 10000)
        for er in ['tp', 'sl', 'timeout']:
            if er in exit_stats_nt:
                arr = np.array(exit_stats_nt[er])
                print(f"    {er:8s}: n={len(arr):4d}  avg={arr.mean():+6.1f}bps")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 10: MARKET CLOSE-TO-CLOSE BASELINE ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # Simple test: if you just buy at close and sell at next close,
        # what's the average return? This tells us if the market was trending.
        
        oos_ret = oos['close'].pct_change().dropna()
        print(f"\n  OOS close-to-close returns:")
        print(f"    Mean: {oos_ret.mean()*10000:+.2f} bps/bar")
        print(f"    Std:  {oos_ret.std()*10000:.2f} bps/bar")
        print(f"    Positive bars: {100*(oos_ret>0).mean():.1f}%")
        print(f"    Cumulative: {oos_ret.sum()*100:+.1f}%")
        
        # 30-bar forward returns (matching our max_hold)
        fwd30 = (oos['close'].shift(-30) / oos['close'] - 1).dropna()
        print(f"\n  30-bar forward returns:")
        print(f"    Mean: {fwd30.mean()*10000:+.2f} bps")
        print(f"    Positive: {100*(fwd30>0).mean():.1f}%")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 11: ORIGINAL vs FIXED COMPARISON WITH ALL FIXES ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        
        print("\n  FIXED sim (next-bar fill, next-bar exit):")
        trades_fixed2 = []
        for i in range(len(oos)):
            idx = bars.index.get_loc(oos.index[i])
            s = sig_oos.iloc[i]
            if s == 0: continue
            t = sim_trade_fixed(bars, idx, s > 0)
            if t: trades_fixed2.append(t)
        pstats(trades_fixed2, f"FULLY FIXED ret_iqr {sym}")
        
        # Exit breakdown for fixed
        exit_stats_fixed = {}
        for t in trades_fixed2:
            er = t['exit']
            if er not in exit_stats_fixed: exit_stats_fixed[er] = []
            exit_stats_fixed[er].append(t['net'] * 10000)
        for er in ['tp', 'trail', 'sl', 'timeout']:
            if er in exit_stats_fixed:
                arr = np.array(exit_stats_fixed[er])
                print(f"    {er:8s}: n={len(arr):4d}  avg={arr.mean():+6.1f}bps  wr={100*(arr>0).mean():5.1f}%")
        
        # Also test fixed + random
        print(f"\n  FIXED sim + RANDOM direction:")
        random.seed(42)
        trades_fixed_rand = []
        signal_indices = [i for i in range(len(oos)) if sig_oos.iloc[i] != 0]
        for i in signal_indices:
            idx = bars.index.get_loc(oos.index[i])
            direction = random.choice([True, False])
            t = sim_trade_fixed(bars, idx, direction)
            if t: trades_fixed_rand.append(t)
        pstats(trades_fixed_rand, f"FIXED + RANDOM direction {sym}")
        
        # Also test fixed + inverted
        print(f"\n  FIXED sim + INVERTED signal:")
        trades_fixed_inv = []
        for i in range(len(oos)):
            idx = bars.index.get_loc(oos.index[i])
            s = sig_oos.iloc[i]
            if s == 0: continue
            t = sim_trade_fixed(bars, idx, s < 0)  # INVERTED
            if t: trades_fixed_inv.append(t)
        pstats(trades_fixed_inv, f"FIXED + INVERTED {sym}")
        
        # ====================================================================
        print(f"\n{'#'*100}")
        print(f"  AUDIT 12: ZERO-OFFSET TEST ({sym})")
        print(f"{'#'*100}")
        # ====================================================================
        # If offset=0, we enter at market (close price). This removes the
        # "buy below market" advantage. If still profitable, the signal+exits
        # provide genuine edge.
        
        print("\n  Testing with offset=0 (market entry) + FIXED sim:")
        trades_zero = []
        for i in range(len(oos)):
            idx = bars.index.get_loc(oos.index[i])
            s = sig_oos.iloc[i]
            if s == 0: continue
            t = sim_trade_fixed(bars, idx, s > 0, offset=0.0)
            if t: trades_zero.append(t)
        pstats(trades_zero, f"FIXED + offset=0 {sym}")
        
        print("\n  Testing with offset=0 + RANDOM direction + FIXED sim:")
        random.seed(42)
        trades_zero_rand = []
        for i in signal_indices:
            idx = bars.index.get_loc(oos.index[i])
            direction = random.choice([True, False])
            t = sim_trade_fixed(bars, idx, direction, offset=0.0)
            if t: trades_zero_rand.append(t)
        pstats(trades_zero_rand, f"FIXED + offset=0 + RANDOM {sym}")
        
        del bars, oos, sig, sig_oos
        gc.collect()
    
    elapsed = time.time() - t_start
    print(f"\n{'#'*100}")
    print(f"  DEEP AUDIT COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min) [{ram_str()}]")
    print(f"{'#'*100}")
