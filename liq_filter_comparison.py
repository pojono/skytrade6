#!/usr/bin/env python3
"""
Filter Comparison — Test each research filter individually and combined
on all 3 actionable configs across 4 symbols.

Configs:
  1. SAFE:       off=0.15%, TP=0.15%, SL=0.50%, hold=60min
  2. AGGRESSIVE:  off=0.15%, TP=0.12%, SL=none,  hold=60min
  3. QUALITY:     off=0.20%, TP=0.15%, SL=0.50%, hold=30min (DOGE US-hours)

Filters tested (one-by-one, then combined):
  A. BASELINE — no filters
  B. + Bad hours (exclude 08,09,13,16 UTC)
  C. + Long-only (fade buy-side liquidations only)
  D. + Displacement ≥10 bps
  E. + Weekday only (Mon-Fri)
  F. ALL FILTERS combined
  G. ALL FILTERS + US-hours only (13-18 UTC)
"""

import sys, time, json, gzip
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

sys.stdout.reconfigure(line_buffering=True)

SYMBOLS = ["DOGEUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT"]
OUT_DIR = Path("results")

MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.055

BAD_HOURS = {8, 9, 13, 16}
US_HOURS = set(range(13, 19))  # 13-18 UTC


# ============================================================================
# DATA LOADING (reuse from liq_integrated_strategy.py)
# ============================================================================

def load_liquidations(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    liq_dirs = [symbol_dir / "bybit" / "liquidations", symbol_dir]
    liq_files = []
    for d in liq_dirs:
        liq_files.extend(sorted(d.glob("liquidation_*.jsonl.gz")))
    liq_files = sorted(set(liq_files))
    if not liq_files:
        raise ValueError(f"No liquidation files for {symbol}")
    print(f"  Loading {len(liq_files)} liq files...", end='', flush=True)
    records = []
    for i, file in enumerate(liq_files, 1):
        if i % 500 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data and 'data' in data['result']:
                        for ev in data['result']['data']:
                            records.append({
                                'timestamp': pd.to_datetime(ev['T'], unit='ms'),
                                'side': ev['S'],
                                'volume': float(ev['v']),
                                'price': float(ev['p']),
                            })
                except Exception:
                    continue
    print(f" done ({len(records):,})")
    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    df['notional'] = df['volume'] * df['price']
    return df


def load_ticker_prices(symbol, data_dir='data'):
    symbol_dir = Path(data_dir) / symbol
    ticker_dirs = [symbol_dir / "bybit" / "ticker", symbol_dir]
    ticker_files = []
    for d in ticker_dirs:
        ticker_files.extend(sorted(d.glob("ticker_*.jsonl.gz")))
    ticker_files = sorted(set(ticker_files))
    if not ticker_files:
        raise ValueError(f"No ticker files for {symbol}")
    print(f"  Loading {len(ticker_files)} ticker files...", end='', flush=True)
    records = []
    for i, file in enumerate(ticker_files, 1):
        if i % 500 == 0:
            print(f" {i}", end='', flush=True)
        with gzip.open(file, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    r = data['result']['list'][0]
                    records.append({
                        'timestamp': pd.to_datetime(data['ts'], unit='ms'),
                        'price': float(r['lastPrice']),
                    })
                except Exception:
                    continue
    print(f" done ({len(records):,})")
    return pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)


def build_price_bars(tick_df, freq='1min'):
    df = tick_df.set_index('timestamp')
    bars = df['price'].resample(freq).agg(['first', 'max', 'min', 'last'])
    bars.columns = ['open', 'high', 'low', 'close']
    return bars.dropna()


# ============================================================================
# CASCADE DETECTION
# ============================================================================

def detect_cascades(liq_df, price_bars, pct_thresh=95):
    thresh = liq_df['notional'].quantile(pct_thresh / 100)
    large = liq_df[liq_df['notional'] >= thresh].copy()

    bar_index = price_bars.index
    bar_close = price_bars['close'].values

    timestamps = large['timestamp'].values
    sides = large['side'].values
    notionals = large['notional'].values
    n = len(large)

    cascades = []
    i = 0
    while i < n:
        cluster = [i]
        j = i + 1
        while j < n:
            dt = (timestamps[j] - timestamps[cluster[-1]]).astype('timedelta64[s]').astype(float)
            if dt <= 60:
                cluster.append(j)
                j += 1
            else:
                break

        if len(cluster) >= 2:
            c_sides = sides[cluster]
            c_notionals = notionals[cluster]
            c_ts = timestamps[cluster]

            buy_not = c_notionals[c_sides == 'Buy'].sum()
            sell_not = c_notionals[c_sides == 'Sell'].sum()
            buy_dominant = buy_not > sell_not

            end_ts = pd.Timestamp(c_ts[-1])
            end_idx = bar_index.searchsorted(end_ts)
            if end_idx >= len(bar_close) - 120 or end_idx < 10:
                i = cluster[-1] + 1
                continue

            current_price = bar_close[end_idx]

            # Displacement
            start_idx = bar_index.searchsorted(pd.Timestamp(c_ts[0]))
            if start_idx > 0:
                pre_price = bar_close[max(0, start_idx - 1)]
                cascade_disp_bps = (current_price - pre_price) / pre_price * 10000
            else:
                cascade_disp_bps = 0

            cascades.append({
                'end': end_ts,
                'n_events': len(cluster),
                'buy_dominant': buy_dominant,
                'end_bar_idx': end_idx,
                'current_price': current_price,
                'cascade_disp_bps': cascade_disp_bps,
                'hour_utc': end_ts.hour,
                'day_of_week': end_ts.dayofweek,
            })

        i = cluster[-1] + 1 if len(cluster) >= 2 else i + 1

    return cascades


# ============================================================================
# STRATEGY
# ============================================================================

def run_strategy(cascades, price_bars,
                 entry_offset_pct=0.15, tp_pct=0.15, sl_pct=0.50,
                 max_hold_min=60, cooldown_min=5,
                 # Filters
                 exclude_bad_hours=False,
                 weekday_only=False,
                 long_only=False,
                 min_cascade_disp_bps=0,
                 us_hours_only=False,
                 ):
    bar_high = price_bars['high'].values
    bar_low = price_bars['low'].values
    bar_close = price_bars['close'].values

    trades = []
    last_trade_time = None

    for cascade in cascades:
        if last_trade_time is not None:
            dt = (cascade['end'] - last_trade_time).total_seconds()
            if dt < cooldown_min * 60:
                continue

        # FILTERS
        if exclude_bad_hours and cascade['hour_utc'] in BAD_HOURS:
            continue
        if weekday_only and cascade['day_of_week'] >= 5:
            continue
        if long_only and not cascade['buy_dominant']:
            continue
        if abs(cascade['cascade_disp_bps']) < min_cascade_disp_bps:
            continue
        if us_hours_only and cascade['hour_utc'] not in US_HOURS:
            continue

        idx = cascade['end_bar_idx']
        current_price = cascade['current_price']

        # Direction
        if cascade['buy_dominant']:
            direction = 'long'
            limit_price = current_price * (1 - entry_offset_pct / 100)
            tp_price = limit_price * (1 + tp_pct / 100)
            sl_price = limit_price * (1 - sl_pct / 100) if sl_pct else None
        else:
            direction = 'short'
            limit_price = current_price * (1 + entry_offset_pct / 100)
            tp_price = limit_price * (1 - tp_pct / 100)
            sl_price = limit_price * (1 + sl_pct / 100) if sl_pct else None

        # Fill simulation
        filled = False
        fill_bar_idx = None
        end_bar_idx = min(idx + max_hold_min, len(bar_close) - 1)

        for j in range(idx, end_bar_idx + 1):
            if direction == 'long' and bar_low[j] <= limit_price:
                filled = True
                fill_bar_idx = j
                break
            elif direction == 'short' and bar_high[j] >= limit_price:
                filled = True
                fill_bar_idx = j
                break

        if not filled:
            continue

        # TP/SL simulation
        exit_price = None
        exit_reason = 'timeout'
        remaining = max_hold_min - (fill_bar_idx - idx)
        exit_end = min(fill_bar_idx + remaining, len(bar_close) - 1)

        for k in range(fill_bar_idx, exit_end + 1):
            if direction == 'long':
                if sl_price and bar_low[k] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    break
                if bar_high[k] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    break
            else:
                if sl_price and bar_high[k] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'stop_loss'
                    break
                if bar_low[k] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'take_profit'
                    break

        if exit_price is None:
            exit_price = bar_close[exit_end]

        # PnL with fees
        if direction == 'long':
            raw_pnl_pct = (exit_price - limit_price) / limit_price * 100
        else:
            raw_pnl_pct = (limit_price - exit_price) / limit_price * 100

        entry_fee = MAKER_FEE_PCT
        exit_fee = MAKER_FEE_PCT if exit_reason == 'take_profit' else TAKER_FEE_PCT
        net_pnl_pct = raw_pnl_pct - entry_fee - exit_fee

        trades.append({
            'net_pnl_pct': net_pnl_pct,
            'exit_reason': exit_reason,
            'entry_time': price_bars.index[fill_bar_idx] if fill_bar_idx < len(price_bars.index) else None,
        })
        last_trade_time = cascade['end']

    return trades


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze(trades):
    if len(trades) < 3:
        return None
    net = np.array([t['net_pnl_pct'] for t in trades])
    exits = [t['exit_reason'] for t in trades]
    n = len(trades)

    n_tp = sum(1 for e in exits if e == 'take_profit')
    n_sl = sum(1 for e in exits if e == 'stop_loss')
    n_to = sum(1 for e in exits if e == 'timeout')

    total = np.sum(net)
    avg = np.mean(net)
    wr = (net > 0).mean() * 100

    # Sharpe
    if np.std(net) > 0 and n > 10:
        days = 282
        tpd = n / days
        daily_ret = avg * tpd
        daily_std = np.std(net) * np.sqrt(tpd)
        sharpe = daily_ret / daily_std * np.sqrt(365) if daily_std > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd) if len(dd) > 0 else 0

    # Positive months
    times = [t['entry_time'] for t in trades if t.get('entry_time') is not None]
    if times:
        months = pd.Series(net[:len(times)], index=pd.DatetimeIndex(times)).resample('M').sum()
        n_pos = (months > 0).sum()
        n_months = len(months)
    else:
        n_pos = 0
        n_months = 0

    return {
        'fills': n, 'wr': wr, 'total': total, 'avg': avg,
        'tp_pct': n_tp / n * 100, 'sl_pct': n_sl / n * 100, 'to_pct': n_to / n * 100,
        'sharpe': sharpe, 'max_dd': max_dd,
        'pos_months': n_pos, 'n_months': n_months,
    }


def fmt_row(label, r):
    if r is None:
        return f"  {label:<45s}  {'too few trades':>10s}"
    mo = f"{r['pos_months']}/{r['n_months']}" if r['n_months'] > 0 else "—"
    return (f"  {label:<45s}  {r['fills']:>5d}  {r['wr']:>5.1f}%  "
            f"{r['avg']:>+8.4f}%  {r['total']:>+8.2f}%  "
            f"{r['sharpe']:>+7.1f}  {r['max_dd']:>5.2f}%  "
            f"TP={r['tp_pct']:>4.0f}% SL={r['sl_pct']:>4.0f}% TO={r['to_pct']:>4.0f}%  "
            f"mo={mo}")


HDR = (f"  {'filter':<45s}  {'fills':>5s}  {'WR':>5s}  "
       f"{'avg_net':>8s}  {'total':>8s}  "
       f"{'Sharpe':>7s}  {'maxDD':>5s}  "
       f"{'exit breakdown':>25s}  {'months':>6s}")


# ============================================================================
# FILTER SETS
# ============================================================================

FILTER_SETS = OrderedDict([
    ("A: BASELINE (no filters)",
     dict(exclude_bad_hours=False, weekday_only=False, long_only=False,
          min_cascade_disp_bps=0, us_hours_only=False)),

    ("B: + Bad hours (skip 08,09,13,16)",
     dict(exclude_bad_hours=True, weekday_only=False, long_only=False,
          min_cascade_disp_bps=0, us_hours_only=False)),

    ("C: + Long-only (fade buy-side)",
     dict(exclude_bad_hours=False, weekday_only=False, long_only=True,
          min_cascade_disp_bps=0, us_hours_only=False)),

    ("D: + Displacement ≥10 bps",
     dict(exclude_bad_hours=False, weekday_only=False, long_only=False,
          min_cascade_disp_bps=10, us_hours_only=False)),

    ("E: + Weekday only (Mon-Fri)",
     dict(exclude_bad_hours=False, weekday_only=True, long_only=False,
          min_cascade_disp_bps=0, us_hours_only=False)),

    ("F: ALL FILTERS combined",
     dict(exclude_bad_hours=True, weekday_only=True, long_only=True,
          min_cascade_disp_bps=10, us_hours_only=False)),

    ("G: ALL + US-hours only (13-18 UTC)",
     dict(exclude_bad_hours=True, weekday_only=True, long_only=True,
          min_cascade_disp_bps=10, us_hours_only=True)),
])

# 3 configs from ACTIONABLE guide
CONFIGS = OrderedDict([
    ("Config1 SAFE (off=0.15 TP=0.15 SL=0.50 60m)",
     dict(entry_offset_pct=0.15, tp_pct=0.15, sl_pct=0.50, max_hold_min=60)),

    ("Config2 AGGR (off=0.15 TP=0.12 SL=none 60m)",
     dict(entry_offset_pct=0.15, tp_pct=0.12, sl_pct=None, max_hold_min=60)),

    ("Config3 QUAL (off=0.20 TP=0.15 SL=0.50 30m)",
     dict(entry_offset_pct=0.20, tp_pct=0.15, sl_pct=0.50, max_hold_min=30)),
])


# ============================================================================
# MAIN
# ============================================================================

def run_symbol(symbol, data_dir='data'):
    print(f"\n{'='*110}")
    print(f"  {symbol} — FILTER COMPARISON")
    print(f"{'='*110}")

    t0 = time.time()

    liq_df = load_liquidations(symbol, data_dir)
    tick_df = load_ticker_prices(symbol, data_dir)

    print("  Building 1-min bars...", end='', flush=True)
    bars = build_price_bars(tick_df, '1min')
    print(f" {len(bars):,} bars")

    days = (bars.index.max() - bars.index.min()).total_seconds() / 86400
    print(f"  Period: {days:.0f} days, {len(liq_df):,} liquidations")

    print("  Detecting cascades...", end='', flush=True)
    cascades = detect_cascades(liq_df, bars, pct_thresh=95)
    print(f" {len(cascades):,} cascades")

    results = {}  # results[config_name][filter_name] = analysis dict

    for cfg_name, cfg_params in CONFIGS.items():
        print(f"\n  ── {cfg_name} ──")
        print(HDR)

        results[cfg_name] = {}

        for filt_name, filt_params in FILTER_SETS.items():
            trades = run_strategy(cascades, bars,
                                  cooldown_min=5,
                                  **cfg_params, **filt_params)
            r = analyze(trades)
            results[cfg_name][filt_name] = r
            print(fmt_row(filt_name, r))

    # Monthly breakdown for ALL FILTERS combined on each config
    print(f"\n  ── MONTHLY BREAKDOWN (ALL FILTERS) ──")
    all_filt = FILTER_SETS["F: ALL FILTERS combined"]
    for cfg_name, cfg_params in CONFIGS.items():
        trades = run_strategy(cascades, bars, cooldown_min=5,
                              **cfg_params, **all_filt)
        if len(trades) < 3:
            print(f"\n  {cfg_name}: too few trades")
            continue
        net = np.array([t['net_pnl_pct'] for t in trades])
        times = [t['entry_time'] for t in trades if t.get('entry_time') is not None]
        if not times:
            continue
        s = pd.Series(net[:len(times)], index=pd.DatetimeIndex(times))
        monthly = s.resample('M').agg(['sum', 'count', lambda x: (x > 0).mean() * 100])
        monthly.columns = ['total', 'trades', 'wr']
        print(f"\n  {cfg_name}:")
        n_pos = 0
        for idx_m, row in monthly.iterrows():
            flag = "✅" if row['total'] > 0 else "  "
            if row['total'] > 0:
                n_pos += 1
            print(f"    {flag} {idx_m.strftime('%Y-%m')}: n={int(row['trades']):>4d}  "
                  f"WR={row['wr']:.0f}%  total={row['total']:>+.2f}%")
        print(f"    Positive months: {n_pos}/{len(monthly)}")

    elapsed = time.time() - t0
    print(f"\n  {symbol} done in {elapsed:.0f}s")
    return results


def main():
    t_start = time.time()

    print("=" * 110)
    print("  FILTER COMPARISON — Research Filters One-by-One and Combined")
    print("  3 Configs × 7 Filter Sets × 4 Symbols")
    print(f"  Fees: maker={MAKER_FEE_PCT:.3f}%, taker={TAKER_FEE_PCT:.3f}%")
    print("=" * 110)

    OUT_DIR.mkdir(exist_ok=True)

    # Tee output
    out_path = OUT_DIR / "liq_filter_comparison.txt"
    import io

    class Tee:
        def __init__(self, *fps):
            self.fps = fps
        def write(self, data):
            for fp in self.fps:
                fp.write(data)
        def flush(self):
            for fp in self.fps:
                fp.flush()

    log_file = open(out_path, 'w', buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)

    all_results = {}
    for sym in SYMBOLS:
        try:
            all_results[sym] = run_symbol(sym)
        except Exception as e:
            print(f"\n  ✗ {sym} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── CROSS-SYMBOL SUMMARY ──
    print(f"\n{'='*110}")
    print(f"  CROSS-SYMBOL SUMMARY")
    print(f"{'='*110}")

    for cfg_name in CONFIGS:
        print(f"\n  ── {cfg_name} ──")
        print(f"  {'filter':<45s}  ", end='')
        for sym in SYMBOLS:
            print(f"  {sym[:4]:>8s}", end='')
        print(f"  {'COMBINED':>10s}  {'avg_Sharpe':>10s}")

        for filt_name in FILTER_SETS:
            print(f"  {filt_name:<45s}  ", end='')
            totals = []
            sharpes = []
            for sym in SYMBOLS:
                r = all_results.get(sym, {}).get(cfg_name, {}).get(filt_name)
                if r:
                    print(f"  {r['total']:>+7.1f}%", end='')
                    totals.append(r['total'])
                    sharpes.append(r['sharpe'])
                else:
                    print(f"  {'—':>8s}", end='')
            combined = sum(totals) if totals else 0
            avg_sh = np.mean(sharpes) if sharpes else 0
            print(f"  {combined:>+9.1f}%  {avg_sh:>+9.1f}")

    # Filter improvement summary
    print(f"\n{'='*110}")
    print(f"  FILTER IMPACT SUMMARY (ALL FILTERS vs BASELINE)")
    print(f"{'='*110}")
    print(f"  {'config':<45s}  {'symbol':>8s}  {'base_total':>10s}  {'filt_total':>10s}  "
          f"{'delta':>8s}  {'base_mo':>8s}  {'filt_mo':>8s}")

    for cfg_name in CONFIGS:
        for sym in SYMBOLS:
            base = all_results.get(sym, {}).get(cfg_name, {}).get("A: BASELINE (no filters)")
            filt = all_results.get(sym, {}).get(cfg_name, {}).get("F: ALL FILTERS combined")
            if base and filt:
                delta = filt['total'] - base['total']
                base_mo = f"{base['pos_months']}/{base['n_months']}"
                filt_mo = f"{filt['pos_months']}/{filt['n_months']}"
                print(f"  {cfg_name:<45s}  {sym[:4]:>8s}  {base['total']:>+9.1f}%  "
                      f"{filt['total']:>+9.1f}%  {delta:>+7.1f}%  {base_mo:>8s}  {filt_mo:>8s}")

    elapsed = time.time() - t_start
    print(f"\n{'='*110}")
    print(f"  DONE — {elapsed:.0f}s total")
    print(f"{'='*110}")

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
