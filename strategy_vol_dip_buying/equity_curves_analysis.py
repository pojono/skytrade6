#!/usr/bin/env python3
"""
Comprehensive Equity Curves, Drawdown, and Projected Returns Analysis
for the Vol Dip-Buying Strategy across 20 coins.

Outputs:
  - Per-symbol equity curves (PNG)
  - Portfolio equity curve (equal-weight, Tier-A only, all)
  - Monthly returns heatmap per symbol
  - Drawdown chart
  - Monthly returns table (CSV)
  - Projected returns at various capital levels
  - Correlation matrix between symbols
"""

import sys, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

PARQUET_DIR = Path('parquet')
OUT_DIR = Path('strategy_vol_dip_buying')
CHART_DIR = OUT_DIR / 'charts'
CHART_DIR.mkdir(parents=True, exist_ok=True)

RT_FEE_BPS = 4.0
THRESHOLD = 2.0
HOLD_BARS = 4
COOLDOWN_BARS = 4

ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
]

TIER_A = ["SOLUSDT", "POLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "UNIUSDT", "SUIUSDT"]
TIER_B = ["ARBUSDT", "LINKUSDT", "LTCUSDT", "ETHUSDT", "NEARUSDT", "OPUSDT"]


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


def simulate_all_trades(df):
    """Return list of all trades with timestamps and PnL."""
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
        net_bps = raw_bps - RT_FEE_BPS

        trades.append({
            'entry_time': df.index[i],
            'exit_time': df.index[i + HOLD_BARS],
            'dir': trade_dir,
            'net_bps': net_bps,
            'net_pct': net_bps / 100,
        })
        last_exit = i + HOLD_BARS

    return trades


def trades_to_monthly(trades, all_months):
    """Convert trade list to monthly returns Series."""
    monthly = defaultdict(float)
    for t in trades:
        m = t['entry_time'].to_period('M')
        monthly[m] += t['net_pct']

    return pd.Series({m: monthly.get(m, 0.0) for m in all_months})


def main():
    t0 = time.time()
    print("=" * 80)
    print("EQUITY CURVES & COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # ================================================================
    # 1. Run walk-forward for all symbols, collect monthly returns
    # ================================================================
    symbol_monthly = {}  # symbol -> pd.Series of monthly returns (%)
    symbol_trades = {}   # symbol -> list of trades
    symbol_stats = {}    # symbol -> dict of stats

    for si, symbol in enumerate(ALL_SYMBOLS, 1):
        df = load_1h(symbol)
        if df.empty or len(df) < 2000:
            print(f"[{si:2d}/20] {symbol:12s} — SKIP")
            continue

        df = compute_signals(df)
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())

        # Get all trades from month 6 onward (warmup)
        warmup_end = months[5] if len(months) > 5 else months[-1]
        trade_df = df[df['month'] > warmup_end]
        trades = simulate_all_trades(trade_df)

        active_months = months[6:]
        monthly_ret = trades_to_monthly(trades, active_months)

        symbol_monthly[symbol] = monthly_ret
        symbol_trades[symbol] = trades

        cum = monthly_ret.cumsum()
        n_trades = len(trades)
        pos_months = (monthly_ret > 0).sum()
        total_months = len(monthly_ret)

        # Sharpe
        m_sharpe = monthly_ret.mean() / max(monthly_ret.std(), 0.001) * np.sqrt(12)

        # Max DD
        peak = cum.cummax()
        dd = peak - cum
        maxdd = dd.max()

        # Longest losing streak
        streak = 0; max_streak = 0
        for v in monthly_ret.values:
            if v < 0:
                streak += 1; max_streak = max(max_streak, streak)
            else:
                streak = 0

        symbol_stats[symbol] = {
            'n_trades': n_trades, 'total': cum.iloc[-1] if len(cum) > 0 else 0,
            'pos_months': pos_months, 'total_months': total_months,
            'm_sharpe': m_sharpe, 'maxdd': maxdd, 'max_streak': max_streak,
            'ann_ret': cum.iloc[-1] / max(total_months / 12, 0.5) if len(cum) > 0 else 0,
        }

        print(f"[{si:2d}/20] {symbol:12s} {n_trades:4d} trades | "
              f"{pos_months}/{total_months} pos | total: {cum.iloc[-1]:+.1f}% | "
              f"mSharpe: {m_sharpe:+.2f}")

    # ================================================================
    # 2. Per-symbol equity curves
    # ================================================================
    print("\nGenerating per-symbol equity curves...")

    fig, axes = plt.subplots(5, 4, figsize=(24, 20))
    fig.suptitle('Vol Dip-Buying: Per-Symbol Equity Curves (Walk-Forward, No Lookahead)',
                 fontsize=16, fontweight='bold')

    sorted_symbols = sorted(symbol_monthly.keys(),
                           key=lambda s: symbol_stats[s]['m_sharpe'], reverse=True)

    for idx, symbol in enumerate(sorted_symbols):
        ax = axes[idx // 4][idx % 4]
        monthly = symbol_monthly[symbol]
        cum = monthly.cumsum()

        dates = [p.to_timestamp() for p in cum.index]
        vals = cum.values

        tier = 'A' if symbol in TIER_A else ('B' if symbol in TIER_B else 'C')
        color = '#2ecc71' if tier == 'A' else ('#3498db' if tier == 'B' else '#e74c3c')

        ax.fill_between(dates, 0, vals, alpha=0.3, color=color)
        ax.plot(dates, vals, color=color, linewidth=1.5)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

        stats = symbol_stats[symbol]
        short_sym = symbol.replace('USDT', '')
        ax.set_title(f"{short_sym} (Tier {tier}) — {stats['total']:+.1f}%, "
                     f"Sharpe {stats['m_sharpe']:+.2f}", fontsize=10)
        ax.set_ylabel('Cumulative %')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'equity_curves_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'equity_curves_all.png'}")

    # ================================================================
    # 3. Portfolio equity curves (equal-weight)
    # ================================================================
    print("Generating portfolio equity curves...")

    # Align all monthly returns to common index
    all_months_set = set()
    for s in symbol_monthly:
        all_months_set.update(symbol_monthly[s].index)
    common_months = sorted(all_months_set)

    # Build DataFrame of monthly returns
    monthly_df = pd.DataFrame(index=common_months)
    for s in sorted_symbols:
        monthly_df[s] = symbol_monthly[s].reindex(common_months).fillna(0)

    # Portfolio variants
    tier_a_cols = [s for s in sorted_symbols if s in TIER_A]
    tier_ab_cols = [s for s in sorted_symbols if s in TIER_A or s in TIER_B]
    all_cols = sorted_symbols

    port_tiera = monthly_df[tier_a_cols].mean(axis=1)
    port_tierab = monthly_df[tier_ab_cols].mean(axis=1)
    port_all = monthly_df[all_cols].mean(axis=1)

    # Top 5 by Sharpe
    top5 = sorted_symbols[:5]
    port_top5 = monthly_df[top5].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Portfolio Equity Curves — Equal-Weight Monthly Rebalance',
                 fontsize=14, fontweight='bold')

    portfolios = [
        ('Tier A (7 coins)', port_tiera, '#2ecc71'),
        ('Tier A+B (13 coins)', port_tierab, '#3498db'),
        ('All 20 coins', port_all, '#9b59b6'),
        ('Top 5 by Sharpe', port_top5, '#e67e22'),
    ]

    for ax, (name, port, color) in zip(axes.flat, portfolios):
        cum = port.cumsum()
        dates = [p.to_timestamp() for p in cum.index]

        ax.fill_between(dates, 0, cum.values, alpha=0.3, color=color)
        ax.plot(dates, cum.values, color=color, linewidth=2)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

        # Stats
        sharpe = port.mean() / max(port.std(), 0.001) * np.sqrt(12)
        peak = cum.cummax()
        maxdd = (peak - cum).max()
        ann = cum.iloc[-1] / max(len(cum) / 12, 0.5)

        ax.set_title(f"{name}\nTotal: {cum.iloc[-1]:+.1f}%, Ann: {ann:+.1f}%/yr, "
                     f"Sharpe: {sharpe:+.2f}, MaxDD: {maxdd:.1f}%", fontsize=10)
        ax.set_ylabel('Cumulative %')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'portfolio_equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'portfolio_equity_curves.png'}")

    # ================================================================
    # 4. Drawdown chart
    # ================================================================
    print("Generating drawdown charts...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Portfolio Drawdown Analysis', fontsize=14, fontweight='bold')

    for ax, (name, port, color) in zip(axes.flat, portfolios):
        cum = port.cumsum()
        peak = cum.cummax()
        dd = -(peak - cum)
        dates = [p.to_timestamp() for p in dd.index]

        ax.fill_between(dates, 0, dd.values, alpha=0.4, color='#e74c3c')
        ax.plot(dates, dd.values, color='#c0392b', linewidth=1.5)
        ax.axhline(y=0, color='gray', linewidth=0.5)

        maxdd = dd.min()
        ax.set_title(f"{name} — Max Drawdown: {maxdd:.1f}%", fontsize=10)
        ax.set_ylabel('Drawdown %')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'drawdown_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'drawdown_analysis.png'}")

    # ================================================================
    # 5. Monthly returns heatmap
    # ================================================================
    print("Generating monthly returns heatmap...")

    # Build year-month matrix for Tier A portfolio
    port_df = port_tiera.to_frame('ret')
    port_df['year'] = [p.year for p in port_df.index]
    port_df['month_num'] = [p.month for p in port_df.index]

    years = sorted(port_df['year'].unique())
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    heatmap_data = pd.DataFrame(index=years, columns=range(1, 13), dtype=float)
    for _, row in port_df.iterrows():
        heatmap_data.loc[row['year'], row['month_num']] = row['ret']

    fig, ax = plt.subplots(figsize=(14, 5))
    data = heatmap_data.values.astype(float)
    mask = np.isnan(data)

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto',
                   vmin=-5, vmax=5)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 3 else 'black'
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Monthly Return %')
    ax.set_title('Tier A Portfolio — Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'monthly_heatmap_tierA.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'monthly_heatmap_tierA.png'}")

    # ================================================================
    # 6. Per-symbol monthly heatmap (all symbols)
    # ================================================================
    print("Generating per-symbol returns heatmap...")

    # Average monthly return per symbol
    sym_month_avg = monthly_df[sorted_symbols].copy()
    sym_month_avg.index = [f"{p.year}-{p.month:02d}" for p in sym_month_avg.index]

    # Transpose for heatmap: symbols as rows, months as columns
    # Too many months, so aggregate by quarter
    quarterly = {}
    for s in sorted_symbols:
        m = symbol_monthly[s]
        m_df = m.to_frame('ret')
        m_df['quarter'] = [f"{p.year}Q{(p.month-1)//3+1}" for p in m_df.index]
        quarterly[s] = m_df.groupby('quarter')['ret'].sum()

    q_df = pd.DataFrame(quarterly)
    q_df = q_df.T  # symbols as rows

    fig, ax = plt.subplots(figsize=(20, 10))
    data = q_df.values.astype(float)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)

    ax.set_xticks(range(len(q_df.columns)))
    ax.set_xticklabels(q_df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(q_df.index)))
    short_names = [s.replace('USDT', '') for s in q_df.index]
    ax.set_yticklabels(short_names, fontsize=9)

    for i in range(len(q_df.index)):
        for j in range(len(q_df.columns)):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 10 else 'black'
                ax.text(j, i, f'{val:+.0f}', ha='center', va='center',
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Quarterly Return %')
    ax.set_title('All Symbols — Quarterly Returns Heatmap (%)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'quarterly_heatmap_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'quarterly_heatmap_all.png'}")

    # ================================================================
    # 7. Correlation matrix
    # ================================================================
    print("Generating correlation matrix...")

    corr = monthly_df[sorted_symbols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    short = [s.replace('USDT', '') for s in sorted_symbols]
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short, fontsize=9)

    for i in range(len(short)):
        for j in range(len(short)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.35 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Monthly Return Correlation Between Symbols', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'correlation_matrix.png'}")

    # ================================================================
    # 8. Save monthly returns CSV
    # ================================================================
    print("Saving monthly returns CSV...")

    csv_df = monthly_df[sorted_symbols].copy()
    csv_df.index = [str(p) for p in csv_df.index]
    csv_df['Tier_A_avg'] = port_tiera.values
    csv_df['Tier_AB_avg'] = port_tierab.values
    csv_df['All_avg'] = port_all.values
    csv_df['Top5_avg'] = port_top5.values
    csv_df.to_csv(OUT_DIR / 'monthly_returns_all.csv')
    print(f"  Saved {OUT_DIR / 'monthly_returns_all.csv'}")

    # ================================================================
    # 9. Print comprehensive summary
    # ================================================================
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE PORTFOLIO ANALYSIS")
    print("=" * 80)

    for name, port in [('Tier A (7)', port_tiera), ('Tier A+B (13)', port_tierab),
                        ('All 20', port_all), ('Top 5', port_top5)]:
        cum = port.cumsum()
        sharpe = port.mean() / max(port.std(), 0.001) * np.sqrt(12)
        peak = cum.cummax()
        maxdd = (peak - cum).max()
        n_months = len(port)
        n_years = n_months / 12
        ann = cum.iloc[-1] / max(n_years, 0.5)
        pos = (port > 0).sum()
        avg_monthly = port.mean()
        std_monthly = port.std()

        # Worst month, best month
        worst = port.min()
        best = port.max()
        worst_m = str(port.idxmin())
        best_m = str(port.idxmax())

        # Calmar ratio
        calmar = ann / max(maxdd, 0.1)

        # Win months in a row (max)
        max_win_streak = 0; ws = 0
        for v in port.values:
            if v > 0:
                ws += 1; max_win_streak = max(max_win_streak, ws)
            else:
                ws = 0

        print(f"\n  --- {name} ---")
        print(f"  Total return:      {cum.iloc[-1]:+.1f}%")
        print(f"  Annualized:        {ann:+.1f}%/yr")
        print(f"  Monthly Sharpe:    {sharpe:+.2f}")
        print(f"  Calmar ratio:      {calmar:.2f}")
        print(f"  Max drawdown:      {maxdd:.1f}%")
        print(f"  Positive months:   {pos}/{n_months} ({pos/n_months*100:.0f}%)")
        print(f"  Avg monthly:       {avg_monthly:+.2f}%")
        print(f"  Std monthly:       {std_monthly:.2f}%")
        print(f"  Best month:        {best:+.2f}% ({best_m})")
        print(f"  Worst month:       {worst:+.2f}% ({worst_m})")
        print(f"  Max win streak:    {max_win_streak} months")

    # ================================================================
    # 10. Projected returns at various capital levels
    # ================================================================
    print("\n" + "=" * 80)
    print("  PROJECTED RETURNS (Tier A Portfolio, Equal-Weight)")
    print("=" * 80)

    cum_tiera = port_tiera.cumsum()
    ann_ret = cum_tiera.iloc[-1] / max(len(port_tiera) / 12, 0.5)
    maxdd_pct = (cum_tiera.cummax() - cum_tiera).max()

    capitals = [1000, 5000, 10000, 25000, 50000, 100000]
    print(f"\n  Based on: {ann_ret:+.1f}%/yr annualized, {maxdd_pct:.1f}% max drawdown")
    print(f"  {'Capital':>10s}  {'Monthly':>10s}  {'Yearly':>10s}  {'Max DD$':>10s}  {'3yr Proj':>10s}")
    for cap in capitals:
        monthly_dollar = cap * (ann_ret / 100) / 12
        yearly_dollar = cap * (ann_ret / 100)
        maxdd_dollar = cap * (maxdd_pct / 100)
        proj_3yr = cap * (1 + ann_ret / 100) ** 3 - cap
        print(f"  ${cap:>9,}  ${monthly_dollar:>9,.0f}  ${yearly_dollar:>9,.0f}  "
              f"${maxdd_dollar:>9,.0f}  ${proj_3yr:>9,.0f}")

    # ================================================================
    # 11. Yearly breakdown table
    # ================================================================
    print("\n" + "=" * 80)
    print("  YEARLY BREAKDOWN — Tier A Portfolio")
    print("=" * 80)

    port_df2 = port_tiera.to_frame('ret')
    port_df2['year'] = [p.year for p in port_df2.index]

    print(f"\n  {'Year':>6s}  {'Return':>8s}  {'Months':>7s}  {'Pos':>4s}  "
          f"{'Avg/mo':>7s}  {'Best':>7s}  {'Worst':>7s}  {'Sharpe':>7s}")
    for year in sorted(port_df2['year'].unique()):
        yr = port_df2[port_df2['year'] == year]['ret']
        total = yr.sum()
        n = len(yr)
        pos = (yr > 0).sum()
        avg = yr.mean()
        best = yr.max()
        worst = yr.min()
        sharpe = yr.mean() / max(yr.std(), 0.001) * np.sqrt(12)
        print(f"  {year:>6d}  {total:+7.1f}%  {n:>5d}mo  {pos:>3d}+  "
              f"{avg:+6.2f}%  {best:+6.2f}%  {worst:+6.2f}%  {sharpe:+6.2f}")

    # ================================================================
    # 12. Risk metrics
    # ================================================================
    print("\n" + "=" * 80)
    print("  RISK METRICS — Tier A Portfolio")
    print("=" * 80)

    rets = port_tiera.values
    print(f"\n  Skewness:          {pd.Series(rets).skew():+.2f}")
    print(f"  Kurtosis:          {pd.Series(rets).kurtosis():+.2f}")
    print(f"  VaR 5% (monthly):  {np.percentile(rets, 5):+.2f}%")
    print(f"  CVaR 5% (monthly): {rets[rets <= np.percentile(rets, 5)].mean():+.2f}%")
    print(f"  % months > +2%:    {(rets > 2).sum()}/{len(rets)} ({(rets > 2).sum()/len(rets)*100:.0f}%)")
    print(f"  % months < -2%:    {(rets < -2).sum()}/{len(rets)} ({(rets < -2).sum()/len(rets)*100:.0f}%)")

    # Average correlation between Tier A symbols
    tier_a_corr = monthly_df[tier_a_cols].corr()
    mask = np.triu(np.ones_like(tier_a_corr, dtype=bool), k=1)
    avg_corr = tier_a_corr.values[mask].mean()
    print(f"  Avg pairwise corr: {avg_corr:+.2f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Charts saved to: {CHART_DIR.resolve()}")
    print(f"CSV saved to: {(OUT_DIR / 'monthly_returns_all.csv').resolve()}")


if __name__ == '__main__':
    main()
