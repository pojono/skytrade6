#!/usr/bin/env python3
"""
Leverage Analysis for Vol Dip-Buying Strategy.

Simulates Tier A portfolio at 1x–10x leverage.
Shows: returns, drawdowns, risk of ruin, margin calls, Sharpe scaling.
"""

import sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path('parquet')
OUT_DIR = Path('strategy_vol_dip_buying')
CHART_DIR = OUT_DIR / 'charts'

RT_FEE_BPS = 4.0
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4

TIER_A = ["SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "UNIUSDT", "SUIUSDT"]
# Excluding POL (too short history)

LEVERAGES = [1, 2, 3, 5, 7, 10]


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


def get_all_trades(df):
    sig = df['combined'].values
    c = df['close'].values.astype(np.float64)
    n = len(c)
    trades = []
    last_exit = 0

    for i in range(0, n - HOLD_BARS):
        if i < last_exit + COOLDOWN_BARS:
            continue
        if np.isnan(sig[i]) or abs(sig[i]) < THRESHOLD:
            continue

        trade_dir = 'long' if sig[i] > 0 else 'short'
        entry = c[i]
        exit_p = c[i + HOLD_BARS]
        raw_pct = ((exit_p - entry) / entry * 100) if trade_dir == 'long' else \
                  ((entry - exit_p) / entry * 100)

        trades.append({
            'entry_time': df.index[i],
            'raw_pct': raw_pct,  # before fees, before leverage
        })
        last_exit = i + HOLD_BARS

    return trades


def simulate_leveraged_portfolio(all_symbol_trades, leverage, fee_pct=0.04):
    """
    Simulate equal-weight portfolio with leverage.
    Each trade uses (1/n_symbols) of capital * leverage.
    Fee scales with leverage (you pay fees on notional).
    
    Returns monthly returns series.
    """
    # Merge all trades, compute leveraged PnL per trade
    all_trades = []
    n_symbols = len(all_symbol_trades)
    weight = 1.0 / n_symbols  # equal weight

    for symbol, trades in all_symbol_trades.items():
        for t in trades:
            # Leveraged return on allocated capital
            # raw_pct is the unleveraged price move
            # Fee is on notional = leverage * allocated capital
            leveraged_pct = weight * (t['raw_pct'] * leverage - fee_pct * leverage)
            all_trades.append({
                'time': t['entry_time'],
                'month': t['entry_time'].to_period('M'),
                'pct': leveraged_pct,
                'raw_pct': t['raw_pct'],
                'symbol': symbol,
            })

    # Aggregate to monthly
    monthly = defaultdict(float)
    trade_details = defaultdict(list)
    for t in all_trades:
        monthly[t['month']] += t['pct']
        trade_details[t['month']].append(t)

    months = sorted(monthly.keys())
    monthly_ret = pd.Series({m: monthly[m] for m in months})

    # Compute per-trade worst loss (for liquidation analysis)
    worst_single_trade = min(t['pct'] for t in all_trades) if all_trades else 0
    worst_raw = min(t['raw_pct'] for t in all_trades) if all_trades else 0

    # Simulate equity curve with compounding
    equity = [100.0]
    for m in months:
        new_eq = equity[-1] * (1 + monthly[m] / 100)
        equity.append(max(new_eq, 0))  # can't go below 0

    equity_series = pd.Series(equity[1:], index=months)

    return monthly_ret, equity_series, worst_single_trade, worst_raw, len(all_trades)


def main():
    t0 = time.time()
    print("=" * 90)
    print("LEVERAGE ANALYSIS — Vol Dip-Buying, Tier A Portfolio (6 coins)")
    print("=" * 90)

    # Load and compute trades for all Tier A symbols
    all_symbol_trades = {}
    for si, symbol in enumerate(TIER_A, 1):
        df = load_1h(symbol)
        if df.empty:
            continue
        df = compute_signals(df)
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())
        warmup_end = months[5]
        trade_df = df[df['month'] > warmup_end]
        trades = get_all_trades(trade_df)
        all_symbol_trades[symbol] = trades
        print(f"  [{si}/6] {symbol}: {len(trades)} trades")

    # Run for each leverage level
    results = {}
    print(f"\n{'Lev':>4s}  {'Total':>8s}  {'Ann':>7s}  {'mSharpe':>8s}  {'MaxDD':>7s}  "
          f"{'Calmar':>7s}  {'Pos%':>5s}  {'Worst Mo':>9s}  {'Worst Tr':>9s}  "
          f"{'Liq Risk':>9s}")
    print("-" * 90)

    for lev in LEVERAGES:
        monthly_ret, equity, worst_trade, worst_raw, n_trades = \
            simulate_leveraged_portfolio(all_symbol_trades, lev)

        cum = monthly_ret.cumsum()
        total = cum.iloc[-1]
        n_months = len(monthly_ret)
        ann = total / max(n_months / 12, 0.5)
        sharpe = monthly_ret.mean() / max(monthly_ret.std(), 0.001) * np.sqrt(12)
        peak = cum.cummax()
        maxdd = (peak - cum).max()
        calmar = ann / max(maxdd, 0.1)
        pos = (monthly_ret > 0).sum()
        worst_month = monthly_ret.min()

        # Liquidation risk: at what leverage does worst single trade wipe out position?
        # Liquidation happens when loss > (1/leverage) * 100% of margin
        # With isolated margin: loss > 100/leverage %
        # worst_raw is the worst unleveraged trade return
        liq_threshold = 100.0 / lev  # % move that causes liquidation
        # Count trades where |raw loss| > liq_threshold
        all_raw = []
        for sym, trades in all_symbol_trades.items():
            for t in trades:
                if t['raw_pct'] < 0:
                    all_raw.append(abs(t['raw_pct']))
        n_dangerous = sum(1 for r in all_raw if r > liq_threshold)
        liq_risk_pct = n_dangerous / max(len(all_raw), 1) * 100

        results[lev] = {
            'monthly_ret': monthly_ret, 'equity': equity,
            'total': total, 'ann': ann, 'sharpe': sharpe,
            'maxdd': maxdd, 'calmar': calmar, 'worst_month': worst_month,
            'worst_trade': worst_trade, 'liq_risk': liq_risk_pct,
            'pos_pct': pos / n_months * 100,
        }

        print(f"  {lev:2d}x  {total:+7.1f}%  {ann:+6.1f}%  {sharpe:+7.2f}  "
              f"{maxdd:6.1f}%  {calmar:+6.2f}  {pos/n_months*100:4.0f}%  "
              f"{worst_month:+8.2f}%  {worst_trade:+8.2f}%  "
              f"{liq_risk_pct:7.2f}%")

    # ================================================================
    # Chart 1: Equity curves at different leverages
    # ================================================================
    print("\nGenerating charts...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Tier A Portfolio — Equity Curves at Different Leverage Levels',
                 fontsize=14, fontweight='bold')

    colors = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']

    for ax, lev, color in zip(axes.flat, LEVERAGES, colors):
        r = results[lev]
        cum = r['monthly_ret'].cumsum()
        dates = [p.to_timestamp() for p in cum.index]

        ax.fill_between(dates, 0, cum.values, alpha=0.3, color=color)
        ax.plot(dates, cum.values, color=color, linewidth=2)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

        ax.set_title(f"{lev}x Leverage\n"
                     f"Total: {r['total']:+.0f}%, Ann: {r['ann']:+.0f}%/yr, "
                     f"Sharpe: {r['sharpe']:+.2f}, DD: {r['maxdd']:.0f}%",
                     fontsize=10)
        ax.set_ylabel('Cumulative %')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'leverage_equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'leverage_equity_curves.png'}")

    # ================================================================
    # Chart 2: Leverage vs metrics (4 subplots)
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of Leverage on Key Metrics', fontsize=14, fontweight='bold')

    levs = LEVERAGES
    totals = [results[l]['total'] for l in levs]
    sharpes = [results[l]['sharpe'] for l in levs]
    maxdds = [results[l]['maxdd'] for l in levs]
    calmars = [results[l]['calmar'] for l in levs]
    worst_months = [results[l]['worst_month'] for l in levs]
    liq_risks = [results[l]['liq_risk'] for l in levs]

    # Total return
    ax = axes[0][0]
    ax.bar(range(len(levs)), totals, color=['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b'])
    ax.set_xticks(range(len(levs)))
    ax.set_xticklabels([f'{l}x' for l in levs])
    ax.set_ylabel('Total Return %')
    ax.set_title('Total Return vs Leverage')
    for i, v in enumerate(totals):
        ax.text(i, v + 5, f'{v:+.0f}%', ha='center', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Max Drawdown
    ax = axes[0][1]
    ax.bar(range(len(levs)), maxdds, color=['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b'])
    ax.set_xticks(range(len(levs)))
    ax.set_xticklabels([f'{l}x' for l in levs])
    ax.set_ylabel('Max Drawdown %')
    ax.set_title('Max Drawdown vs Leverage')
    for i, v in enumerate(maxdds):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Sharpe
    ax = axes[1][0]
    ax.plot(levs, sharpes, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Monthly Sharpe')
    ax.set_title('Sharpe Ratio vs Leverage')
    for l, s in zip(levs, sharpes):
        ax.annotate(f'{s:+.2f}', (l, s), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Calmar
    ax = axes[1][1]
    ax.plot(levs, calmars, 'o-', color='#9b59b6', linewidth=2, markersize=8)
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Calmar Ratio')
    ax.set_title('Calmar Ratio (Ann Return / Max DD) vs Leverage')
    for l, c in zip(levs, calmars):
        ax.annotate(f'{c:.2f}', (l, c), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'leverage_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'leverage_metrics.png'}")

    # ================================================================
    # Chart 3: Drawdown comparison
    # ================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    for lev, color in zip([1, 2, 3, 5], ['#27ae60', '#3498db', '#f39c12', '#e74c3c']):
        r = results[lev]
        cum = r['monthly_ret'].cumsum()
        peak = cum.cummax()
        dd = -(peak - cum)
        dates = [p.to_timestamp() for p in dd.index]
        ax.plot(dates, dd.values, color=color, linewidth=1.5, label=f'{lev}x (DD: {r["maxdd"]:.1f}%)')

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.legend(fontsize=11)
    ax.set_title('Drawdown Comparison Across Leverage Levels', fontsize=13, fontweight='bold')
    ax.set_ylabel('Drawdown %')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'leverage_drawdown_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'leverage_drawdown_comparison.png'}")

    # ================================================================
    # Projected returns at different leverage + capital
    # ================================================================
    print(f"\n{'='*90}")
    print("  PROJECTED ANNUAL RETURNS ($) BY LEVERAGE AND CAPITAL")
    print(f"{'='*90}")

    capitals = [5000, 10000, 25000, 50000, 100000]
    print(f"\n  {'':>10s}", end='')
    for cap in capitals:
        print(f"  ${cap:>8,}", end='')
    print()
    print(f"  {'':>10s}", end='')
    for _ in capitals:
        print(f"  {'--------':>9s}", end='')
    print()

    for lev in LEVERAGES:
        r = results[lev]
        ann = r['ann']
        print(f"  {lev:2d}x ({ann:+.0f}%)", end='')
        for cap in capitals:
            yearly = cap * ann / 100
            print(f"  ${yearly:>8,.0f}", end='')
        print(f"   (DD: {r['maxdd']:.0f}%)")

    # ================================================================
    # Risk summary
    # ================================================================
    print(f"\n{'='*90}")
    print("  RISK ANALYSIS BY LEVERAGE")
    print(f"{'='*90}")

    print(f"\n  {'Lev':>4s}  {'MaxDD':>7s}  {'Worst Mo':>9s}  {'Calmar':>7s}  "
          f"{'Liq Risk':>9s}  {'Recommendation':>30s}")
    print("  " + "-" * 80)

    for lev in LEVERAGES:
        r = results[lev]
        if lev <= 2:
            rec = "✅ Safe — recommended"
        elif lev <= 3:
            rec = "⚠️  Moderate — acceptable"
        elif lev <= 5:
            rec = "⚠️  Aggressive — experienced only"
        else:
            rec = "❌ Dangerous — not recommended"

        print(f"  {lev:2d}x   {r['maxdd']:6.1f}%  {r['worst_month']:+8.2f}%  "
              f"{r['calmar']:+6.2f}  {r['liq_risk']:7.2f}%   {rec}")

    # ================================================================
    # Optimal leverage (Kelly criterion approximation)
    # ================================================================
    print(f"\n{'='*90}")
    print("  KELLY CRITERION — OPTIMAL LEVERAGE")
    print(f"{'='*90}")

    # For the 1x portfolio
    r1 = results[1]
    mu = r1['monthly_ret'].mean() / 100  # monthly mean return (decimal)
    sigma = r1['monthly_ret'].std() / 100  # monthly std (decimal)
    kelly = mu / (sigma ** 2) if sigma > 0 else 0
    half_kelly = kelly / 2

    print(f"\n  Monthly mean return (1x): {mu*100:+.3f}%")
    print(f"  Monthly std (1x):         {sigma*100:.3f}%")
    print(f"  Full Kelly leverage:      {kelly:.1f}x")
    print(f"  Half Kelly (recommended): {half_kelly:.1f}x")
    print(f"\n  Note: Half Kelly is standard practice to account for")
    print(f"  estimation error and non-normal returns.")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
