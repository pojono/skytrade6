#!/usr/bin/env python3
"""
v43h: Volume Imbalance Momentum — 3-Year Validation on Pre-Aggregated OHLCV

Uses pre-aggregated 1h OHLCV with buy/sell volume (1143 days, Jan 2023 – Feb 2026).
Much faster than tick-level processing.

Signal: buy/sell volume imbalance z-score on 1h bars.
When z > threshold → long (buy pressure momentum).
When z < -threshold → short (sell pressure momentum).

Holding: 4h. Entry: limit order. Exit: TP limit or timeout market.
Fees: maker 0.02% + taker 0.055% = 7.5 bps RT (timeout)
      maker 0.02% + maker 0.02% = 4 bps RT (TP hit)

Walk-forward: monthly results over 3 years.
Cross-symbol: SOL, ETH, BTC.
"""

import sys, time, gc
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_BPS = 2.0
TAKER_FEE_BPS = 5.5
PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


# ============================================================================
# DATA
# ============================================================================

def load_all_1h(symbol, exchange='bybit_futures'):
    """Load ALL 1h OHLCV bars for a symbol. Very fast (~190KB per day)."""
    ohlcv_dir = PARQUET_DIR / symbol / 'ohlcv' / '1h' / exchange
    if not ohlcv_dir.exists():
        print(f"  {ohlcv_dir} not found!")
        return pd.DataFrame()

    files = sorted(ohlcv_dir.glob('*.parquet'))
    print(f"  Loading {len(files)} daily files...", end='', flush=True)
    t0 = time.time()

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result['timestamp'] = pd.to_datetime(result['timestamp_us'], unit='us')
    result = result.set_index('timestamp').sort_index()
    result = result[~result.index.duplicated(keep='first')]

    elapsed = time.time() - t0
    print(f" {len(result):,} bars in {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")
    return result


def add_features(bars, window=24):
    """Add volume imbalance and related features. All backward-looking."""
    bv = bars['buy_volume'].values.astype(np.float64)
    sv = bars['sell_volume'].values.astype(np.float64)
    tv = bv + sv

    # Volume imbalance: (buy - sell) / total
    imb = np.where(tv > 0, (bv - sv) / tv, 0)
    bars['vol_imbalance'] = imb

    # Z-score of imbalance
    imb_s = pd.Series(imb, index=bars.index)
    roll_mean = imb_s.rolling(window, min_periods=window//2).mean()
    roll_std = imb_s.rolling(window, min_periods=window//2).std()
    bars['vol_imb_z'] = ((imb_s - roll_mean) / roll_std.clip(lower=1e-8)).values

    # Cumulative imbalance (4h)
    bars['cum_imb_4h'] = imb_s.rolling(4, min_periods=2).sum().values

    # Volume ratio (current vs rolling avg)
    tv_s = pd.Series(tv, index=bars.index)
    bars['vol_ratio'] = (tv_s / tv_s.rolling(window, min_periods=window//2).mean().clip(lower=1)).values

    # Buy ratio (fraction of volume that is buy)
    bars['buy_ratio'] = np.where(tv > 0, bv / tv, 0.5)
    br = pd.Series(bars['buy_ratio'].values, index=bars.index)
    br_mean = br.rolling(window, min_periods=window//2).mean()
    br_std = br.rolling(window, min_periods=window//2).std()
    bars['buy_ratio_z'] = ((br - br_mean) / br_std.clip(lower=1e-8)).values

    # Returns
    close = bars['close'].values.astype(np.float64)
    bars['ret_1h_bps'] = np.concatenate([[0], (close[1:] - close[:-1]) / close[:-1] * 10000])

    return bars


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(bars, signal_col='vol_imb_z', z_threshold=2.0,
                 direction='momentum', hold_bars=4,
                 entry_offset_bps=5, tp_bps=None, sl_bps=None,
                 cooldown_bars=2):
    """Run backtest on 1h bars."""
    z = bars[signal_col].values
    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values
    n = len(bars)
    times = bars.index

    trades = []
    last_exit = -cooldown_bars - 1

    for i in range(24, n - hold_bars - 3):
        if i <= last_exit + cooldown_bars:
            continue
        if np.isnan(z[i]) or abs(z[i]) < z_threshold:
            continue

        if direction == 'momentum':
            d = 'long' if z[i] > 0 else 'short'
        else:
            d = 'short' if z[i] > 0 else 'long'

        price = close[i]
        if d == 'long':
            ep = price * (1 - entry_offset_bps / 10000)
        else:
            ep = price * (1 + entry_offset_bps / 10000)

        # Fill within 2 bars
        fb = None
        for j in range(i + 1, min(i + 3, n)):
            if d == 'long' and low[j] <= ep:
                fb = j; break
            elif d == 'short' and high[j] >= ep:
                fb = j; break
        if fb is None:
            continue

        # TP/SL
        tp_p = None; sl_p = None
        if tp_bps:
            tp_p = ep * (1 + tp_bps / 10000) if d == 'long' else ep * (1 - tp_bps / 10000)
        if sl_bps:
            sl_p = ep * (1 - sl_bps / 10000) if d == 'long' else ep * (1 + sl_bps / 10000)

        # Exit
        ee = min(fb + hold_bars, n - 1)
        xp = None; xr = None; xb = ee

        for k in range(fb + 1, ee + 1):
            if sl_p:
                if (d == 'long' and low[k] <= sl_p) or (d == 'short' and high[k] >= sl_p):
                    xp = sl_p; xr = 'sl'; xb = k; break
            if tp_p:
                if (d == 'long' and high[k] >= tp_p) or (d == 'short' and low[k] <= tp_p):
                    xp = tp_p; xr = 'tp'; xb = k; break

        if xp is None:
            xp = close[ee]; xr = 'to'

        # PnL
        raw = (xp - ep) / ep * 10000 if d == 'long' else (ep - xp) / ep * 10000
        ef = MAKER_FEE_BPS
        xf = MAKER_FEE_BPS if xr == 'tp' else TAKER_FEE_BPS
        net = raw - ef - xf

        trades.append({
            'time': times[i], 'dir': d, 'xr': xr,
            'raw': raw, 'net': net, 'hold': xb - fb, 'z': z[i],
        })
        last_exit = xb

    return trades


def summarize(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'avg': 0, 'total': 0, 'sharpe': 0}
    net = np.array([t['net'] for t in trades])
    n = len(net)
    wr = (net > 0).sum() / n * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0
    return {'n': n, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("v43h: Volume Imbalance Momentum — 3-Year OHLCV Validation")
    print("=" * 80)

    configs = [
        # Signal, direction, z, tp, sl, label
        ('vol_imb_z', 'momentum',   2.0, None, None, 'mom z>2.0 noTP'),
        ('vol_imb_z', 'momentum',   1.5, None, None, 'mom z>1.5 noTP'),
        ('vol_imb_z', 'momentum',   2.5, None, None, 'mom z>2.5 noTP'),
        ('vol_imb_z', 'momentum',   2.0, 100,  None, 'mom z>2.0 TP=100'),
        ('vol_imb_z', 'momentum',   2.0, 50,   None, 'mom z>2.0 TP=50'),
        ('vol_imb_z', 'momentum',   2.0, 50,   100,  'mom z>2.0 TP50 SL100'),
        ('vol_imb_z', 'contrarian', 2.0, None, None, 'contr z>2.0 noTP'),
        ('vol_imb_z', 'contrarian', 1.5, None, None, 'contr z>1.5 noTP'),
        ('buy_ratio_z', 'momentum', 2.0, None, None, 'buyR mom z>2.0 noTP'),
        ('buy_ratio_z', 'momentum', 1.5, None, None, 'buyR mom z>1.5 noTP'),
        ('cum_imb_4h', 'momentum',  0.1, None, None, 'cumImb>0.1 noTP'),
        ('cum_imb_4h', 'momentum',  0.2, None, None, 'cumImb>0.2 noTP'),
    ]

    for symbol in ['SOLUSDT', 'ETHUSDT', 'BTCUSDT']:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        bars = load_all_1h(symbol)
        if bars.empty:
            continue

        # Check if buy_volume exists
        if 'buy_volume' not in bars.columns:
            print(f"  No buy_volume column! Columns: {list(bars.columns)}")
            continue

        bars = add_features(bars)
        print(f"  Features added. {len(bars):,} bars, "
              f"{bars.index[0]} to {bars.index[-1]}")

        # Group by month
        bars['month'] = bars.index.to_period('M')
        month_keys = sorted(bars['month'].unique())
        print(f"  {len(month_keys)} months")

        # Run each config on full data, then analyze monthly
        for sig_col, dir_mode, z_thresh, tp, sl, label in configs:
            if sig_col not in bars.columns:
                continue

            trades = run_backtest(bars, signal_col=sig_col, z_threshold=z_thresh,
                                  direction=dir_mode, tp_bps=tp, sl_bps=sl)
            s = summarize(trades)

            if s['n'] < 10:
                print(f"  {label:30s} | n={s['n']:5d} — too few trades")
                continue

            # Monthly breakdown
            monthly = {}
            for t in trades:
                m = t['time'].to_period('M')
                monthly.setdefault(str(m), []).append(t['net'])

            pos_months = sum(1 for v in monthly.values() if sum(v) > 0)
            total_months = len(monthly)

            # Direction breakdown
            long_trades = [t for t in trades if t['dir'] == 'long']
            short_trades = [t for t in trades if t['dir'] == 'short']
            l_avg = np.mean([t['net'] for t in long_trades]) if long_trades else 0
            s_avg = np.mean([t['net'] for t in short_trades]) if short_trades else 0

            # Exit breakdown
            exits = {}
            for t in trades:
                exits[t['xr']] = exits.get(t['xr'], 0) + 1

            print(f"  {label:30s} | n={s['n']:5d} WR={s['wr']:5.1f}% "
                  f"avg={s['avg']:+7.1f}bps total={s['total']:+8.2f}% "
                  f"Sh={s['sharpe']:+6.1f} | "
                  f"pos_mo={pos_months}/{total_months} "
                  f"L={l_avg:+.1f} S={s_avg:+.1f} exits={exits}")

        # Detailed monthly for best config (mom z>2.0 noTP)
        print(f"\n  Monthly detail: mom z>2.0 noTP")
        trades = run_backtest(bars, signal_col='vol_imb_z', z_threshold=2.0,
                               direction='momentum')
        if trades:
            monthly = {}
            for t in trades:
                m = str(t['time'].to_period('M'))
                monthly.setdefault(m, []).append(t['net'])

            for m in sorted(monthly.keys()):
                nets = monthly[m]
                avg = np.mean(nets)
                total = sum(nets) / 100
                wr = sum(1 for x in nets if x > 0) / len(nets) * 100
                print(f"    {m}: n={len(nets):3d} WR={wr:5.1f}% "
                      f"avg={avg:+7.1f}bps total={total:+6.2f}%")

        del bars
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
