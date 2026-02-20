#!/usr/bin/env python3
"""
Final Portfolio Analysis: Top 100 scan complete.
Compare portfolio sizes from 6 to 14 robust Tier A coins.
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

# All robust Tier A (≥18 months), sorted by Sharpe
ALL_TIER_A_ROBUST = [
    "ONDOUSDT",     # +2.08, 20mo
    "TAOUSDT",      # +2.00, 20mo
    "SOLUSDT",      # +1.67, 32mo
    "HBARUSDT",     # +1.58, 32mo
    "SEIUSDT",      # +1.54, 25mo
    "ADAUSDT",      # +1.44, 32mo
    "BNBUSDT",      # +1.37, 32mo
    "XRPUSDT",      # +1.37, 32mo
    "AAVEUSDT",     # +1.32, 32mo
    "UNIUSDT",      # +1.31, 32mo
    "XLMUSDT",      # +1.27, 32mo
    "CRVUSDT",      # +1.26, 32mo
    "SUIUSDT",      # +1.08, 28mo
    "1000BONKUSDT", # +1.05, 32mo
]

# Original 6 for comparison
ORIGINAL_6 = ["SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "UNIUSDT", "SUIUSDT"]


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


def get_trades(df):
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
        raw_bps = ((exit_p - entry) / entry * 10000) if trade_dir == 'long' else \
                  ((entry - exit_p) / entry * 10000)
        trades.append({
            'entry_time': df.index[i],
            'exit_time': df.index[min(i + HOLD_BARS, n - 1)],
            'net_bps': raw_bps - RT_FEE_BPS,
            'net_pct': (raw_bps - RT_FEE_BPS) / 100,
        })
        last_exit = i + HOLD_BARS
    return trades


def simulate_portfolio(symbol_trades, n_symbols, leverage=1):
    all_events = []
    weight = 1.0 / n_symbols
    for sym, trades in symbol_trades.items():
        for t in trades:
            portfolio_impact_pct = weight * leverage * t['net_pct']
            all_events.append({
                'time': t['entry_time'],
                'exit_time': t['exit_time'],
                'pct': portfolio_impact_pct,
                'symbol': sym,
            })
    all_events.sort(key=lambda x: x['time'])
    if not all_events:
        return None

    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)
    for ev in all_events:
        m = ev['time'].to_period('M')
        monthly_pnl[m] += ev['pct']
        monthly_trades[m] += 1

    months = sorted(monthly_pnl.keys())
    monthly_series = pd.Series({m: monthly_pnl[m] for m in months})

    equity = 100.0
    for m in months:
        equity *= (1 + monthly_pnl[m] / 100)

    n_months = len(months)
    n_years = n_months / 12
    total_return = (equity / 100 - 1) * 100
    cagr = ((equity / 100) ** (1 / max(n_years, 0.5)) - 1) * 100
    sharpe = monthly_series.mean() / max(monthly_series.std(), 0.001) * np.sqrt(12)
    cum = monthly_series.cumsum()
    peak = cum.cummax()
    maxdd = (peak - cum).max()
    calmar = cagr / max(maxdd, 0.1)
    pos_months = (monthly_series > 0).sum()
    total_trades_count = sum(monthly_trades[m] for m in months)
    trades_per_month = total_trades_count / n_months

    return {
        'equity': equity, 'total_return': total_return, 'cagr': cagr,
        'sharpe': sharpe, 'maxdd': maxdd, 'calmar': calmar,
        'n_months': n_months, 'pos_months': pos_months,
        'pos_pct': pos_months / n_months * 100,
        'total_trades': total_trades_count,
        'trades_per_month': trades_per_month,
        'monthly_series': monthly_series,
        'worst_month': monthly_series.min(),
        'best_month': monthly_series.max(),
    }


def main():
    t0 = time.time()
    print("=" * 110)
    print("FINAL PORTFOLIO ANALYSIS — Top 100 Bybit Scan Complete")
    print("=" * 110)

    # Load all trades
    all_trades = {}
    for symbol in ALL_TIER_A_ROBUST:
        df = load_1h(symbol)
        if df.empty:
            continue
        df = compute_signals(df)
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())
        warmup_end = months[5] if len(months) > 5 else months[-1]
        trade_df = df[df['month'] > warmup_end]
        trades = get_trades(trade_df)
        all_trades[symbol] = trades
        avg_bps = np.mean([t['net_bps'] for t in trades]) if trades else 0
        print(f"  {symbol:16s}: {len(trades)} trades, avg {avg_bps:+.1f} bps")

    # ================================================================
    # Incrementally add symbols by Sharpe rank
    # ================================================================
    print(f"\n{'='*110}")
    print("  INCREMENTAL PORTFOLIO BUILD — Adding by Sharpe Rank (1x)")
    print(f"{'='*110}")

    print(f"\n  {'N':>3s}  {'Added':16s}  {'CAGR':>7s}  {'Total':>8s}  {'Sharpe':>7s}  "
          f"{'MaxDD':>6s}  {'Calmar':>7s}  {'Pos%':>5s}  {'Tr/mo':>6s}  {'Worst':>7s}")
    print("  " + "-" * 95)

    inc_results = []
    available = [s for s in ALL_TIER_A_ROBUST if s in all_trades]

    for n in range(1, len(available) + 1):
        subset = {s: all_trades[s] for s in available[:n]}
        r = simulate_portfolio(subset, n, leverage=1)
        if r is None:
            continue
        inc_results.append(r)
        added = available[n-1].replace('USDT', '')
        print(f"  {n:3d}  {added:16s}  {r['cagr']:+6.1f}%  {r['total_return']:+7.1f}%  "
              f"{r['sharpe']:+6.2f}  {r['maxdd']:5.1f}%  {r['calmar']:+6.2f}  "
              f"{r['pos_pct']:4.0f}%  {r['trades_per_month']:5.1f}  {r['worst_month']:+6.2f}%")

    # ================================================================
    # Key portfolio comparisons
    # ================================================================
    print(f"\n{'='*110}")
    print("  KEY PORTFOLIO COMPARISONS (1x and 2x leverage)")
    print(f"{'='*110}")

    configs = {
        "Original 6": [s for s in ORIGINAL_6 if s in all_trades],
        "Top 6 by Sharpe": available[:6],
        "Top 9": available[:9],
        "Top 12": available[:12],
        "All 14 Tier A": available[:14],
    }

    print(f"\n  {'Config':20s}  {'Coins':50s}")
    print("  " + "-" * 75)
    for name, syms in configs.items():
        sym_str = ', '.join([s.replace('USDT', '') for s in syms])
        print(f"  {name:20s}  {sym_str}")

    print(f"\n  --- 1x Leverage ---")
    print(f"  {'Config':20s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>6s}  {'Calmar':>7s}  "
          f"{'Pos%':>5s}  {'Tr/mo':>6s}  {'Worst':>7s}  {'Best':>7s}")
    print("  " + "-" * 85)

    portfolio_results_1x = {}
    for name, syms in configs.items():
        subset = {s: all_trades[s] for s in syms}
        r = simulate_portfolio(subset, len(syms), leverage=1)
        if r is None:
            continue
        portfolio_results_1x[name] = r
        print(f"  {name:20s}  {r['cagr']:+6.1f}%  {r['sharpe']:+6.2f}  {r['maxdd']:5.1f}%  "
              f"{r['calmar']:+6.2f}  {r['pos_pct']:4.0f}%  {r['trades_per_month']:5.1f}  "
              f"{r['worst_month']:+6.2f}%  {r['best_month']:+6.2f}%")

    print(f"\n  --- 2x Leverage ---")
    print(f"  {'Config':20s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>6s}  {'Calmar':>7s}")
    print("  " + "-" * 55)

    portfolio_results_2x = {}
    for name, syms in configs.items():
        subset = {s: all_trades[s] for s in syms}
        r = simulate_portfolio(subset, len(syms), leverage=2)
        if r is None:
            continue
        portfolio_results_2x[name] = r
        print(f"  {name:20s}  {r['cagr']:+6.1f}%  {r['sharpe']:+6.2f}  {r['maxdd']:5.1f}%  "
              f"{r['calmar']:+6.2f}")

    # ================================================================
    # Correlation matrix for all 14 Tier A
    # ================================================================
    print(f"\n{'='*110}")
    print("  PAIRWISE CORRELATION — All 14 Tier A")
    print(f"{'='*110}")

    sym_monthly = {}
    for symbol in available:
        trades = all_trades[symbol]
        monthly = defaultdict(float)
        for t in trades:
            m = t['entry_time'].to_period('M')
            monthly[m] += t['net_pct']
        sym_monthly[symbol] = pd.Series(monthly)

    all_months = set()
    for s in sym_monthly.values():
        all_months.update(s.index)
    all_months = sorted(all_months)

    df_corr = pd.DataFrame(index=all_months)
    for sym, series in sym_monthly.items():
        df_corr[sym.replace('USDT', '')] = series.reindex(all_months).fillna(0)

    corr = df_corr.corr()

    # Avg pairwise for different portfolio sizes
    for n_label, n_count in [("Top 6", 6), ("Top 9", 9), ("Top 12", 12), ("All 14", 14)]:
        syms = [s.replace('USDT', '') for s in available[:n_count]]
        pairs = []
        for i, s1 in enumerate(syms):
            for s2 in syms[i+1:]:
                if s1 in corr.columns and s2 in corr.columns:
                    pairs.append(corr.loc[s1, s2])
        avg = np.mean(pairs) if pairs else 0
        print(f"  Avg pairwise correlation ({n_label:>6s}): {avg:+.3f}")

    # ================================================================
    # Charts
    # ================================================================
    print("\nGenerating charts...")

    # Chart 1: Portfolio equity curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Final Portfolio Comparison — Top 100 Bybit Scan', fontsize=14, fontweight='bold')

    colors = {
        "Original 6": '#95a5a6',
        "Top 6 by Sharpe": '#27ae60',
        "Top 9": '#3498db',
        "Top 12": '#9b59b6',
        "All 14 Tier A": '#e67e22',
    }

    for ax, results, title in [(axes[0], portfolio_results_1x, '1x Leverage'),
                                (axes[1], portfolio_results_2x, '2x Leverage')]:
        for name, r in results.items():
            cum = r['monthly_series'].cumsum()
            dates = [p.to_timestamp() for p in cum.index]
            ax.plot(dates, cum.values, color=colors.get(name, 'gray'), linewidth=2,
                    label=f"{name}\nCAGR:{r['cagr']:+.0f}% Sh:{r['sharpe']:+.2f} DD:{r['maxdd']:.0f}%")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Cumulative Return %')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'final_portfolio_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'final_portfolio_comparison.png'}")

    # Chart 2: Incremental build — CAGR, Sharpe, DD vs N
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Incremental Portfolio Build — Adding Tier A Coins by Sharpe', fontsize=13, fontweight='bold')

    ns = list(range(1, len(inc_results) + 1))
    cagrs = [r['cagr'] for r in inc_results]
    sharpes = [r['sharpe'] for r in inc_results]
    maxdds = [r['maxdd'] for r in inc_results]

    labels = [s.replace('USDT', '') for s in available]

    ax = axes[0]
    ax.bar(ns, cagrs, color=['#27ae60' if c > 25 else '#f39c12' if c > 15 else '#e74c3c' for c in cagrs])
    ax.set_xticks(ns)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('CAGR %')
    ax.set_title('CAGR vs Portfolio Size')
    for i, v in enumerate(cagrs):
        ax.text(i+1, v + 0.5, f'{v:+.0f}%', ha='center', fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ns, sharpes, 'o-', color='#9b59b6', linewidth=2, markersize=6)
    ax.set_xticks(ns)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Monthly Sharpe')
    ax.set_title('Sharpe vs Portfolio Size')
    for i, v in enumerate(sharpes):
        ax.text(i+1, v + 0.05, f'{v:.2f}', ha='center', fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(ns, maxdds, color=['#27ae60' if d < 5 else '#f39c12' if d < 10 else '#e74c3c' for d in maxdds])
    ax.set_xticks(ns)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Max Drawdown %')
    ax.set_title('Max Drawdown vs Portfolio Size')
    for i, v in enumerate(maxdds):
        ax.text(i+1, v + 0.3, f'{v:.1f}%', ha='center', fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'final_incremental_build.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'final_incremental_build.png'}")

    # Chart 3: Correlation matrix for all 14
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, label='Correlation')
    ax.set_title('Monthly Return Correlation — All 14 Tier A Symbols', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'final_correlation_14.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'final_correlation_14.png'}")

    # ================================================================
    # PROJECTED RETURNS
    # ================================================================
    print(f"\n{'='*110}")
    print("  PROJECTED RETURNS — Recommended Portfolios")
    print(f"{'='*110}")

    for name in ["Top 9", "Top 12"]:
        r1 = portfolio_results_1x.get(name)
        r2 = portfolio_results_2x.get(name)
        if not r1 or not r2:
            continue

        print(f"\n  --- {name} ---")
        print(f"  {'Capital':>10s}  {'1x Monthly':>11s}  {'1x Yearly':>10s}  {'1x MaxDD$':>10s}  "
              f"{'2x Monthly':>11s}  {'2x Yearly':>10s}  {'2x MaxDD$':>10s}")
        for cap in [5000, 10000, 25000, 50000, 100000]:
            m1 = cap * r1['cagr'] / 100 / 12
            y1 = cap * r1['cagr'] / 100
            d1 = cap * r1['maxdd'] / 100
            m2 = cap * r2['cagr'] / 100 / 12
            y2 = cap * r2['cagr'] / 100
            d2 = cap * r2['maxdd'] / 100
            print(f"  ${cap:>9,}  ${m1:>10,.0f}  ${y1:>9,.0f}  ${d1:>9,.0f}  "
                  f"${m2:>10,.0f}  ${y2:>9,.0f}  ${d2:>9,.0f}")

    # ================================================================
    # FINAL RECOMMENDATION
    # ================================================================
    print(f"\n{'='*110}")
    print("  FINAL RECOMMENDATION")
    print(f"{'='*110}")

    # Find best config
    best_sharpe_name = max(portfolio_results_1x, key=lambda k: portfolio_results_1x[k]['sharpe'])
    best_cagr_name = max(portfolio_results_1x, key=lambda k: portfolio_results_1x[k]['cagr'])
    best_calmar_name = max(portfolio_results_1x, key=lambda k: portfolio_results_1x[k]['calmar'])

    print(f"\n  Best Sharpe:  {best_sharpe_name} ({portfolio_results_1x[best_sharpe_name]['sharpe']:+.2f})")
    print(f"  Best CAGR:    {best_cagr_name} ({portfolio_results_1x[best_cagr_name]['cagr']:+.1f}%)")
    print(f"  Best Calmar:  {best_calmar_name} ({portfolio_results_1x[best_calmar_name]['calmar']:+.2f})")

    rec = portfolio_results_1x.get("Top 9") or portfolio_results_1x.get("Top 12")
    rec_name = "Top 9" if "Top 9" in portfolio_results_1x else "Top 12"
    if rec:
        print(f"\n  ★ RECOMMENDED: {rec_name}")
        print(f"    Symbols: {', '.join([s.replace('USDT','') for s in configs[rec_name]])}")
        print(f"    CAGR: {rec['cagr']:+.1f}%, Sharpe: {rec['sharpe']:+.2f}, "
              f"MaxDD: {rec['maxdd']:.1f}%, Calmar: {rec['calmar']:+.2f}")
        print(f"    Trades/month: {rec['trades_per_month']:.0f}, "
              f"Positive months: {rec['pos_pct']:.0f}%")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
