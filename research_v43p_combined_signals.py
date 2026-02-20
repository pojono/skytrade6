#!/usr/bin/env python3
"""
v43p: Combined Weak Signals — Can Multiple Weak Edges Stack?

Individual signals tested in v43 are each too weak to overcome fees alone:
  - Funding rate: IC=-0.12 on SOL (contrarian, 4h horizon)
  - Basis (futures-spot): MR with AC=-0.38 at lag-1h
  - Volume imbalance: IC≈0 alone but might add conditional value
  - Price momentum: 1d autocorrelation varies by regime
  - Volatility regime: low vol → momentum, high vol → MR (weak)

Hypothesis: If these signals are UNCORRELATED, combining them could produce
an edge that exceeds fees. Even 3 signals at IC=0.05 each, if uncorrelated,
give combined IC ≈ 0.05 * sqrt(3) ≈ 0.087.

Strategy: Score each hour with a composite signal. Only trade when
multiple signals agree (high conviction). Use limit orders, hold 4-8h.

Data: 1h OHLCV + ticker (funding, OI) for 76 days with full data coverage.
Validation: Walk-forward monthly, IS/OOS split, random baseline.
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
np.random.seed(42)

MAKER_FEE_BPS = 2.0
RT_FEE_BPS = MAKER_FEE_BPS * 2  # 4 bps maker+maker
PARQUET_DIR = Path('parquet')


def get_ram_mb():
    try:
        import psutil
        return psutil.virtual_memory().used / 1024**2
    except ImportError:
        return 0


def load_1h_ohlcv(symbol, exchange='bybit_futures'):
    d = PARQUET_DIR / symbol / 'ohlcv' / '1h' / exchange
    if not d.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    return raw[~raw.index.duplicated(keep='first')]


def load_ticker_hourly(symbol):
    """Load ticker data and resample to 1h (funding rate, OI, spread)."""
    ticker_dir = PARQUET_DIR / symbol / 'ticker'
    if not ticker_dir.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(ticker_dir.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    raw = raw[~raw.index.duplicated(keep='first')]

    # Resample to 1h
    hourly = pd.DataFrame()
    hourly['funding_rate'] = raw['funding_rate'].resample('1h').last()
    hourly['oi'] = raw['open_interest'].resample('1h').last()
    hourly['mark_price'] = raw['mark_price'].resample('1h').last()
    hourly['index_price'] = raw['index_price'].resample('1h').last()
    hourly['bid1'] = raw['bid1_price'].resample('1h').last()
    hourly['ask1'] = raw['ask1_price'].resample('1h').last()
    return hourly.dropna(subset=['funding_rate'])


def load_spot_1h(symbol):
    d = PARQUET_DIR / symbol / 'ohlcv' / '1h' / 'bybit_spot'
    if not d.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    return raw[~raw.index.duplicated(keep='first')]


def build_features(futures, ticker, spot):
    """Build all feature signals on aligned 1h bars."""
    # Align all data
    common = futures.index.intersection(ticker.index)
    if not spot.empty:
        common = common.intersection(spot.index)

    df = pd.DataFrame(index=common)
    df['close'] = futures.loc[common, 'close']
    df['volume'] = futures.loc[common, 'volume']
    if 'buy_volume' in futures.columns:
        df['buy_volume'] = futures.loc[common, 'buy_volume']
        df['sell_volume'] = futures.loc[common, 'sell_volume']

    df['funding_rate'] = ticker.loc[common, 'funding_rate']
    df['oi'] = ticker.loc[common, 'oi']
    df['mark_price'] = ticker.loc[common, 'mark_price']
    df['index_price'] = ticker.loc[common, 'index_price']

    if not spot.empty:
        df['spot_close'] = spot.loc[common, 'close']

    c = df['close'].values.astype(np.float64)
    n = len(c)

    # ============================================================
    # SIGNAL 1: Funding Rate (contrarian — negative IC confirmed)
    # ============================================================
    fr = df['funding_rate'].values * 10000  # in bps
    fr_s = pd.Series(fr)
    df['sig_funding_z'] = ((fr_s - fr_s.rolling(72, min_periods=24).mean()) /
                            fr_s.rolling(72, min_periods=24).std().clip(lower=1e-8)).values
    # Contrarian: high funding → short signal (negative)
    df['sig_funding'] = -df['sig_funding_z']

    # ============================================================
    # SIGNAL 2: Mark-Index Spread (contrarian)
    # ============================================================
    if 'mark_price' in df.columns and 'index_price' in df.columns:
        mis = (df['mark_price'].values - df['index_price'].values) / df['index_price'].values * 10000
        mis_s = pd.Series(mis)
        df['sig_mis_z'] = ((mis_s - mis_s.rolling(72, min_periods=24).mean()) /
                            mis_s.rolling(72, min_periods=24).std().clip(lower=1e-8)).values
        df['sig_mis'] = -df['sig_mis_z']  # contrarian

    # ============================================================
    # SIGNAL 3: OI Change (contrarian — high OI increase = crowded)
    # ============================================================
    oi = df['oi'].values.astype(np.float64)
    oi_pct = np.zeros(n)
    oi_pct[1:] = (oi[1:] - oi[:-1]) / np.maximum(oi[:-1], 1) * 10000
    oi_s = pd.Series(oi_pct)
    df['sig_oi_z'] = ((oi_s - oi_s.rolling(72, min_periods=24).mean()) /
                       oi_s.rolling(72, min_periods=24).std().clip(lower=1e-8)).values
    df['sig_oi'] = -df['sig_oi_z']  # contrarian

    # ============================================================
    # SIGNAL 4: Basis (futures-spot, contrarian/MR)
    # ============================================================
    if 'spot_close' in df.columns:
        basis = (c - df['spot_close'].values) / df['spot_close'].values * 10000
        basis_s = pd.Series(basis)
        df['sig_basis_z'] = ((basis_s - basis_s.rolling(72, min_periods=24).mean()) /
                              basis_s.rolling(72, min_periods=24).std().clip(lower=1e-8)).values
        df['sig_basis'] = -df['sig_basis_z']  # MR: high basis → short

    # ============================================================
    # SIGNAL 5: Price momentum (short-term MR, medium-term momentum)
    # ============================================================
    ret_1h = np.zeros(n)
    ret_1h[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret_1h)

    # 4h MR
    df['sig_mr_4h'] = -((ret_s.rolling(4).sum() - ret_s.rolling(48, min_periods=12).mean() * 4) /
                          (ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2)).values

    # 24h momentum
    ret_24h = np.zeros(n)
    ret_24h[24:] = (c[24:] - c[:-24]) / c[:-24] * 10000
    ret24_s = pd.Series(ret_24h)
    df['sig_mom_24h'] = ((ret24_s - ret24_s.rolling(168, min_periods=48).mean()) /
                          ret24_s.rolling(168, min_periods=48).std().clip(lower=1e-8)).values

    # ============================================================
    # SIGNAL 6: Volume imbalance (if available)
    # ============================================================
    if 'buy_volume' in df.columns:
        bv = df['buy_volume'].values
        sv = df['sell_volume'].values
        tv = bv + sv
        imb = np.where(tv > 0, (bv - sv) / tv, 0)
        imb_s = pd.Series(imb)
        df['sig_vol_imb'] = ((imb_s - imb_s.rolling(24, min_periods=8).mean()) /
                              imb_s.rolling(24, min_periods=8).std().clip(lower=1e-8)).values

    # ============================================================
    # SIGNAL 7: Volatility regime
    # ============================================================
    rvol = ret_s.rolling(24, min_periods=8).std().values
    rvol_s = pd.Series(rvol)
    df['sig_rvol_z'] = ((rvol_s - rvol_s.rolling(168, min_periods=48).mean()) /
                          rvol_s.rolling(168, min_periods=48).std().clip(lower=1e-8)).values

    # Forward returns (for evaluation only — NOT used in signals)
    df['fwd_1h'] = 0.0
    df['fwd_4h'] = 0.0
    df['fwd_8h'] = 0.0
    df.iloc[:-1, df.columns.get_loc('fwd_1h')] = (c[1:] - c[:-1]) / c[:-1] * 10000
    if n > 4:
        df.iloc[:-4, df.columns.get_loc('fwd_4h')] = (c[4:] - c[:-4]) / c[:-4] * 10000
    if n > 8:
        df.iloc[:-8, df.columns.get_loc('fwd_8h')] = (c[8:] - c[:-8]) / c[:-8] * 10000

    return df


def evaluate_signals(df, signal_cols, fwd_col='fwd_4h'):
    """Evaluate individual signal ICs and correlations."""
    valid = df.dropna(subset=signal_cols + [fwd_col])
    valid = valid[valid[fwd_col] != 0]

    if len(valid) < 50:
        return

    print(f"\n  Individual Signal ICs (vs {fwd_col}):")
    ics = {}
    for col in signal_cols:
        if col in valid.columns:
            ic = np.corrcoef(valid[col].values, valid[fwd_col].values)[0, 1]
            ics[col] = ic
            print(f"    {col:20s}: IC={ic:+.4f}")

    # Signal correlations
    print(f"\n  Signal Correlations:")
    sig_data = valid[signal_cols].values
    corr = np.corrcoef(sig_data.T)
    for i in range(len(signal_cols)):
        for j in range(i + 1, len(signal_cols)):
            print(f"    {signal_cols[i]:15s} vs {signal_cols[j]:15s}: {corr[i,j]:+.3f}")

    return ics


def simulate_combined(df, signal_cols, weights=None, threshold=1.5,
                       hold_bars=4, fee_bps=4, label=''):
    """
    Simulate combined signal strategy.
    Composite = weighted sum of z-scored signals.
    Trade when |composite| > threshold.
    """
    if weights is None:
        weights = {col: 1.0 / len(signal_cols) for col in signal_cols}

    # Compute composite signal
    composite = np.zeros(len(df))
    for col in signal_cols:
        if col in df.columns:
            vals = df[col].values
            vals = np.nan_to_num(vals, 0)
            composite += vals * weights.get(col, 0)

    # Forward return
    c = df['close'].values.astype(np.float64)
    n = len(c)

    trades = []
    last_exit = 0

    for i in range(72, n - hold_bars):  # warmup 72 bars
        if i < last_exit + 4:  # cooldown
            continue

        sig = composite[i]
        if abs(sig) < threshold:
            continue

        trade_dir = 'long' if sig > 0 else 'short'

        # Entry at close of bar i, exit at close of bar i+hold_bars
        entry_price = c[i]
        exit_price = c[i + hold_bars]

        if trade_dir == 'long':
            raw_bps = (exit_price - entry_price) / entry_price * 10000
        else:
            raw_bps = (entry_price - exit_price) / entry_price * 10000

        net_bps = raw_bps - fee_bps

        trades.append({
            'time': df.index[i],
            'dir': trade_dir,
            'sig': sig,
            'net_bps': net_bps,
        })
        last_exit = i + hold_bars

    if not trades:
        print(f"  {label}: NO TRADES"); return None

    net = np.array([t['net_bps'] for t in trades])
    n_t = len(net)
    wr = (net > 0).sum() / n_t * 100
    avg = net.mean()
    total = net.sum() / 100
    std = net.std() if n_t > 1 else 1
    sharpe = avg / std * np.sqrt(252 * 6) if std > 0 else 0

    # Direction breakdown
    long_t = [t for t in trades if t['dir'] == 'long']
    short_t = [t for t in trades if t['dir'] == 'short']

    print(f"  {label}")
    print(f"    n={n_t:4d} WR={wr:5.1f}% avg={avg:+7.1f}bps total={total:+7.2f}% Sharpe={sharpe:+5.2f}")
    if long_t:
        ln = np.array([t['net_bps'] for t in long_t])
        print(f"    LONG:  n={len(ln)} WR={(ln>0).sum()/len(ln)*100:.1f}% avg={ln.mean():+.1f}bps")
    if short_t:
        sn = np.array([t['net_bps'] for t in short_t])
        print(f"    SHORT: n={len(sn)} WR={(sn>0).sum()/len(sn)*100:.1f}% avg={sn.mean():+.1f}bps")

    return {'n': n_t, 'wr': wr, 'avg': avg, 'total': total, 'sharpe': sharpe}


def main():
    t0 = time.time()
    print("=" * 80)
    print("v43p: Combined Weak Signals — Multi-Signal Strategy")
    print("=" * 80)

    for symbol in ['SOLUSDT', 'ETHUSDT', 'BTCUSDT']:
        print(f"\n{'='*80}")
        print(f"  {symbol}")
        print(f"{'='*80}")

        futures = load_1h_ohlcv(symbol)
        ticker = load_ticker_hourly(symbol)
        spot = load_spot_1h(symbol)

        if futures.empty or ticker.empty:
            print(f"  Missing data"); continue

        print(f"  Futures: {len(futures):,} bars")
        print(f"  Ticker: {len(ticker):,} bars")
        print(f"  Spot: {len(spot):,} bars" if not spot.empty else "  Spot: N/A")

        df = build_features(futures, ticker, spot)
        print(f"  Combined: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

        # Available signals
        signal_cols = [c for c in df.columns if c.startswith('sig_')]
        print(f"  Signals: {signal_cols}")

        # ============================================================
        # STEP 1: Individual signal ICs
        # ============================================================
        for fwd in ['fwd_1h', 'fwd_4h', 'fwd_8h']:
            evaluate_signals(df, signal_cols, fwd)

        # ============================================================
        # STEP 2: IS/OOS split
        # ============================================================
        split = int(len(df) * 0.65)
        df_is = df.iloc[:split]
        df_oos = df.iloc[split:]
        print(f"\n  IS: {len(df_is)} bars | OOS: {len(df_oos)} bars")

        # ============================================================
        # STEP 3: Combined signal strategies
        # ============================================================
        # Equal weight
        for thresh in [1.0, 1.5, 2.0, 2.5]:
            for hold in [4, 8]:
                label = f"Equal-weight thresh={thresh} hold={hold}h"
                print(f"\n  --- {label} ---")
                simulate_combined(df_is, signal_cols, threshold=thresh,
                                   hold_bars=hold, label=f"IS {label}")
                simulate_combined(df_oos, signal_cols, threshold=thresh,
                                   hold_bars=hold, label=f"OOS {label}")

        # IC-weighted (use IS ICs as weights)
        valid_is = df_is.dropna(subset=signal_cols + ['fwd_4h'])
        valid_is = valid_is[valid_is['fwd_4h'] != 0]
        if len(valid_is) > 50:
            ic_weights = {}
            for col in signal_cols:
                if col in valid_is.columns:
                    ic = np.corrcoef(valid_is[col].values, valid_is['fwd_4h'].values)[0, 1]
                    ic_weights[col] = ic
            # Normalize
            total_abs = sum(abs(v) for v in ic_weights.values())
            if total_abs > 0:
                ic_weights = {k: v / total_abs for k, v in ic_weights.items()}
                print(f"\n  IC-weighted (from IS):")
                for k, v in ic_weights.items():
                    print(f"    {k:20s}: {v:+.3f}")

                for thresh in [1.0, 1.5, 2.0]:
                    label = f"IC-weighted thresh={thresh} hold=4h"
                    print(f"\n  --- {label} ---")
                    simulate_combined(df_is, signal_cols, weights=ic_weights,
                                       threshold=thresh, hold_bars=4, label=f"IS {label}")
                    simulate_combined(df_oos, signal_cols, weights=ic_weights,
                                       threshold=thresh, hold_bars=4, label=f"OOS {label}")

        # ============================================================
        # STEP 4: Best individual signals only
        # ============================================================
        # Test top-2 signals by IC
        valid_all = df.dropna(subset=signal_cols + ['fwd_4h'])
        valid_all = valid_all[valid_all['fwd_4h'] != 0]
        if len(valid_all) > 50:
            all_ics = {}
            for col in signal_cols:
                ic = np.corrcoef(valid_all[col].values, valid_all['fwd_4h'].values)[0, 1]
                all_ics[col] = ic

            sorted_ics = sorted(all_ics.items(), key=lambda x: abs(x[1]), reverse=True)
            top2 = [s[0] for s in sorted_ics[:2]]
            top3 = [s[0] for s in sorted_ics[:3]]

            print(f"\n  Top-2 signals: {top2}")
            for thresh in [1.0, 1.5, 2.0]:
                label = f"Top-2 thresh={thresh} hold=4h"
                simulate_combined(df_is, top2, threshold=thresh,
                                   hold_bars=4, label=f"IS {label}")
                simulate_combined(df_oos, top2, threshold=thresh,
                                   hold_bars=4, label=f"OOS {label}")

            print(f"\n  Top-3 signals: {top3}")
            for thresh in [1.0, 1.5, 2.0]:
                label = f"Top-3 thresh={thresh} hold=4h"
                simulate_combined(df_is, top3, threshold=thresh,
                                   hold_bars=4, label=f"IS {label}")
                simulate_combined(df_oos, top3, threshold=thresh,
                                   hold_bars=4, label=f"OOS {label}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s, RAM={get_ram_mb():.0f}MB")


if __name__ == '__main__':
    main()
