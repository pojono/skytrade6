#!/usr/bin/env python3
"""
Walk-forward validation of vol dip-buying on all top-50 Bybit symbols.
Identifies new Tier A candidates to add to the portfolio.
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path('parquet')
OUT_DIR = Path('strategy_vol_dip_buying')

RT_FEE_BPS = 4.0
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4
MIN_MONTHS = 12  # need at least 12 months of trading data (after 6mo warmup)

# All symbols we have data for (existing + newly downloaded)
ALL_SYMBOLS = [
    # Original 20
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
    # New from top 50
    "ENSOUSDT", "HYPEUSDT", "1000PEPEUSDT", "XAUTUSDT",
    "AXSUSDT", "BCHUSDT", "AAVEUSDT", "FARTCOINUSDT",
    "ZECUSDT", "TAOUSDT", "INJUSDT", "ENAUSDT",
    "TRUMPUSDT", "VIRTUALUSDT", "SNXUSDT", "WIFUSDT",
    "CRVUSDT", "PENGUUSDT", "ZROUSDT", "BERAUSDT",
    "PAXGUSDT", "ASTERUSDT", "WLFIUSDT",
    # Also check PEPEUSDT (we had it already)
    "PEPEUSDT",
]


def load_1h(symbol):
    d = PARQUET_DIR / symbol / 'ohlcv' / '1h' / 'bybit_futures'
    if not d.exists():
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in sorted(d.glob('*.parquet'))]
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    raw['timestamp'] = pd.to_datetime(raw['timestamp_us'], unit='us')
    raw = raw.set_index('timestamp').sort_index()
    return raw[~raw.index.duplicated(keep='first')]


def compute_signals(df):
    c = df['close'].values.astype(np.float64)
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 10000
    ret_s = pd.Series(ret, index=df.index)

    rvol = ret_s.rolling(24, min_periods=8).std()
    rvol_mean = rvol.rolling(168, min_periods=48).mean()
    rvol_std = rvol.rolling(168, min_periods=48).std().clip(lower=1e-8)
    df['rvol_z'] = ((rvol - rvol_mean) / rvol_std).values

    r4 = ret_s.rolling(4).sum()
    r4_mean = ret_s.rolling(48, min_periods=12).mean() * 4
    r4_std = ret_s.rolling(48, min_periods=12).std().clip(lower=1e-8) * 2
    df['mr_4h'] = -((r4 - r4_mean) / r4_std).values

    df['combined'] = (df['rvol_z'].values + df['mr_4h'].values) / 2
    return df


def walkforward_monthly(df):
    """Walk-forward monthly simulation, 6-month warmup, no lookahead."""
    sig = df['combined'].values
    c = df['close'].values.astype(np.float64)
    n = len(c)
    df['month'] = df.index.to_period('M')
    months = sorted(df['month'].unique())

    if len(months) < 7:
        return None

    warmup_end = months[5]
    trade_months = [m for m in months if m > warmup_end]

    if len(trade_months) < MIN_MONTHS:
        return None

    monthly_results = []
    all_trades = []
    last_exit = 0

    for m in trade_months:
        mask = df['month'] == m
        idxs = np.where(mask.values)[0]
        if len(idxs) == 0:
            continue

        month_pnl = 0.0
        month_trades = 0
        month_wins = 0

        for i in idxs:
            if i < last_exit + COOLDOWN_BARS:
                continue
            if i + HOLD_BARS >= n:
                continue
            if np.isnan(sig[i]) or abs(sig[i]) < THRESHOLD:
                continue

            trade_dir = 'long' if sig[i] > 0 else 'short'
            entry = c[i]
            exit_p = c[i + HOLD_BARS]
            raw_bps = ((exit_p - entry) / entry * 10000) if trade_dir == 'long' else \
                      ((entry - exit_p) / entry * 10000)
            net_bps = raw_bps - RT_FEE_BPS

            month_pnl += net_bps / 100  # convert to %
            month_trades += 1
            if net_bps > 0:
                month_wins += 1
            last_exit = i + HOLD_BARS

            all_trades.append({
                'net_bps': net_bps,
                'dir': trade_dir,
            })

        monthly_results.append({
            'month': m,
            'pnl_pct': month_pnl,
            'trades': month_trades,
            'wins': month_wins,
        })

    if not all_trades:
        return None

    mdf = pd.DataFrame(monthly_results)
    monthly_pnl = mdf['pnl_pct']
    total_trades = mdf['trades'].sum()
    total_wins = mdf['wins'].sum()

    cum = monthly_pnl.cumsum()
    peak = cum.cummax()
    maxdd = (peak - cum).max()

    n_months = len(mdf)
    pos_months = (monthly_pnl > 0).sum()
    total_pnl = monthly_pnl.sum()
    ann = total_pnl / max(n_months / 12, 0.5)
    sharpe = monthly_pnl.mean() / max(monthly_pnl.std(), 0.001) * np.sqrt(12)
    calmar = ann / max(maxdd, 0.1)
    avg_bps = np.mean([t['net_bps'] for t in all_trades])
    long_pct = sum(1 for t in all_trades if t['dir'] == 'long') / len(all_trades) * 100

    return {
        'n_months': n_months,
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'ann': ann,
        'sharpe': sharpe,
        'maxdd': maxdd,
        'calmar': calmar,
        'pos_months': pos_months,
        'pos_pct': pos_months / n_months * 100,
        'win_rate': total_wins / max(total_trades, 1) * 100,
        'avg_bps': avg_bps,
        'long_pct': long_pct,
        'monthly_pnl': monthly_pnl,
    }


def main():
    t0 = time.time()
    print("=" * 110)
    print("TOP 50 SYMBOL VALIDATION — Vol Dip-Buying Walk-Forward")
    print("=" * 110)

    results = []

    for si, symbol in enumerate(ALL_SYMBOLS, 1):
        df = load_1h(symbol)
        if df.empty:
            print(f"[{si:2d}/{len(ALL_SYMBOLS)}] {symbol:16s} — NO DATA")
            continue

        n_bars = len(df)
        date_range = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"

        if n_bars < 2000:
            print(f"[{si:2d}/{len(ALL_SYMBOLS)}] {symbol:16s} — TOO SHORT ({n_bars} bars, {date_range})")
            continue

        df = compute_signals(df)
        r = walkforward_monthly(df)

        if r is None:
            print(f"[{si:2d}/{len(ALL_SYMBOLS)}] {symbol:16s} — INSUFFICIENT MONTHS ({n_bars} bars, {date_range})")
            continue

        # Determine tier
        if r['sharpe'] >= 1.0:
            tier = 'A'
        elif r['sharpe'] >= 0.5:
            tier = 'B'
        else:
            tier = 'C'

        r['symbol'] = symbol
        r['tier'] = tier
        r['n_bars'] = n_bars
        r['date_range'] = date_range
        results.append(r)

        print(f"[{si:2d}/{len(ALL_SYMBOLS)}] {symbol:16s} Tier {tier} | "
              f"{r['total_trades']:4d} trades | {r['n_months']:2d} mo | "
              f"total: {r['total_pnl']:+6.1f}% | mSharpe: {r['sharpe']:+5.2f} | "
              f"avg: {r['avg_bps']:+5.1f} bps | DD: {r['maxdd']:.1f}%")

    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    # ================================================================
    # FULL RESULTS TABLE
    # ================================================================
    print(f"\n{'='*110}")
    print("  FULL RESULTS — Sorted by Monthly Sharpe")
    print(f"{'='*110}")

    print(f"\n  {'#':>3s}  {'Symbol':16s} {'Tier':>4s}  {'Months':>6s}  {'Trades':>7s}  "
          f"{'Total':>7s}  {'Ann':>7s}  {'mSharpe':>8s}  {'MaxDD':>6s}  {'Calmar':>7s}  "
          f"{'WR':>5s}  {'Avg bps':>8s}  {'Long%':>6s}  {'Pos%':>5s}")
    print("  " + "-" * 108)

    tier_a_new = []
    tier_b_new = []

    for i, r in enumerate(results, 1):
        is_original = r['symbol'] in [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
            "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
            "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
        ]
        marker = '' if is_original else ' ★'

        print(f"  {i:3d}  {r['symbol']:16s} {r['tier']:>4s}  {r['n_months']:>5d}  "
              f"{r['total_trades']:>6d}  {r['total_pnl']:+6.1f}%  {r['ann']:+6.1f}%  "
              f"{r['sharpe']:+7.2f}  {r['maxdd']:5.1f}%  {r['calmar']:+6.2f}  "
              f"{r['win_rate']:4.0f}%  {r['avg_bps']:+7.1f}  {r['long_pct']:5.0f}%  "
              f"{r['pos_pct']:4.0f}%{marker}")

        if not is_original and r['tier'] == 'A':
            tier_a_new.append(r)
        if not is_original and r['tier'] == 'B':
            tier_b_new.append(r)

    # ================================================================
    # NEW TIER A CANDIDATES
    # ================================================================
    print(f"\n{'='*110}")
    print("  NEW TIER A CANDIDATES (★ = not in original 20)")
    print(f"{'='*110}")

    if tier_a_new:
        for r in tier_a_new:
            print(f"  ★ {r['symbol']:16s} mSharpe: {r['sharpe']:+.2f}, "
                  f"CAGR: {r['ann']:+.1f}%, DD: {r['maxdd']:.1f}%, "
                  f"avg: {r['avg_bps']:+.1f} bps/trade, {r['total_trades']} trades, "
                  f"{r['n_months']} months, {r['date_range']}")
    else:
        print("  None found.")

    print(f"\n  NEW TIER B CANDIDATES:")
    if tier_b_new:
        for r in tier_b_new:
            print(f"  ★ {r['symbol']:16s} mSharpe: {r['sharpe']:+.2f}, "
                  f"CAGR: {r['ann']:+.1f}%, DD: {r['maxdd']:.1f}%, "
                  f"avg: {r['avg_bps']:+.1f} bps/trade, {r['total_trades']} trades, "
                  f"{r['n_months']} months")
    else:
        print("  None found.")

    # ================================================================
    # SUMMARY STATS
    # ================================================================
    n_total = len(results)
    n_a = sum(1 for r in results if r['tier'] == 'A')
    n_b = sum(1 for r in results if r['tier'] == 'B')
    n_c = sum(1 for r in results if r['tier'] == 'C')
    n_positive = sum(1 for r in results if r['total_pnl'] > 0)

    print(f"\n{'='*110}")
    print("  SUMMARY")
    print(f"{'='*110}")
    print(f"  Total symbols tested:  {n_total}")
    print(f"  Tier A (Sharpe > 1.0): {n_a}")
    print(f"  Tier B (Sharpe 0.5-1): {n_b}")
    print(f"  Tier C (Sharpe < 0.5): {n_c}")
    print(f"  Net positive:          {n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")
    print(f"  New Tier A found:      {len(tier_a_new)}")
    print(f"  New Tier B found:      {len(tier_b_new)}")

    # ================================================================
    # RECOMMENDATION
    # ================================================================
    all_tier_a = [r for r in results if r['tier'] == 'A']
    print(f"\n{'='*110}")
    print("  RECOMMENDED TIER A PORTFOLIO")
    print(f"{'='*110}")

    # Filter: need at least 18 months of data for robustness
    robust_a = [r for r in all_tier_a if r['n_months'] >= 18]
    marginal_a = [r for r in all_tier_a if r['n_months'] < 18]

    print(f"\n  Robust (≥18 months):")
    for r in robust_a:
        is_new = '★' if r['symbol'] not in [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
            "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
            "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
            "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
        ] else ' '
        print(f"  {is_new} {r['symbol']:16s} Sharpe: {r['sharpe']:+.2f}, "
              f"CAGR: {r['ann']:+.1f}%, DD: {r['maxdd']:.1f}%, "
              f"{r['n_months']} months")

    if marginal_a:
        print(f"\n  Marginal (<18 months — monitor but don't include yet):")
        for r in marginal_a:
            is_new = '★' if r['symbol'] not in [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
                "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
                "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
                "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
            ] else ' '
            print(f"  {is_new} {r['symbol']:16s} Sharpe: {r['sharpe']:+.2f}, "
                  f"CAGR: {r['ann']:+.1f}%, DD: {r['maxdd']:.1f}%, "
                  f"{r['n_months']} months")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
