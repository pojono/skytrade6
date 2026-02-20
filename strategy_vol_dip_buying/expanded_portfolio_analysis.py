#!/usr/bin/env python3
"""
Expanded Portfolio Analysis: Compare original 6 Tier A vs expanded 9 Tier A.

New Tier A candidates: TAOUSDT, AAVEUSDT, CRVUSDT
Question: Does adding these 3 improve compounded returns?
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

# Original Tier A
ORIGINAL_A = ["SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "UNIUSDT", "SUIUSDT"]

# New Tier A candidates
NEW_A = ["TAOUSDT", "AAVEUSDT", "CRVUSDT"]

# Expanded Tier A
EXPANDED_A = ORIGINAL_A + NEW_A

# Also test adding best Tier B
BEST_B = ["WIFUSDT", "1000PEPEUSDT"]
FULL_EXPANDED = EXPANDED_A + BEST_B


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
    """Simulate compounded equal-weight portfolio."""
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

    # Monthly aggregation
    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)
    for ev in all_events:
        m = ev['time'].to_period('M')
        monthly_pnl[m] += ev['pct']
        monthly_trades[m] += 1

    months = sorted(monthly_pnl.keys())
    monthly_series = pd.Series({m: monthly_pnl[m] for m in months})

    # Compounded equity
    equity = 100.0
    equity_points = [(months[0].to_timestamp(), 100.0)]
    for m in months:
        equity *= (1 + monthly_pnl[m] / 100)
        equity_points.append((m.to_timestamp() + pd.offsets.MonthEnd(0), equity))

    # Stats
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
        'equity': equity,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'maxdd': maxdd,
        'calmar': calmar,
        'n_months': n_months,
        'pos_months': pos_months,
        'pos_pct': pos_months / n_months * 100,
        'total_trades': total_trades_count,
        'trades_per_month': trades_per_month,
        'monthly_series': monthly_series,
        'equity_points': equity_points,
        'worst_month': monthly_series.min(),
        'best_month': monthly_series.max(),
    }


def main():
    t0 = time.time()
    print("=" * 100)
    print("EXPANDED PORTFOLIO ANALYSIS — Original 6 vs Expanded 9 Tier A")
    print("=" * 100)

    # Load all trades
    all_trades = {}
    for symbol in FULL_EXPANDED:
        df = load_1h(symbol)
        if df.empty:
            print(f"  {symbol}: NO DATA")
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

    # Define portfolio configs to test
    configs = [
        ("Original 6 (Tier A)", {s: all_trades[s] for s in ORIGINAL_A if s in all_trades}),
        ("+ TAO (7)", {s: all_trades[s] for s in ORIGINAL_A + ["TAOUSDT"] if s in all_trades}),
        ("+ TAO + AAVE (8)", {s: all_trades[s] for s in ORIGINAL_A + ["TAOUSDT", "AAVEUSDT"] if s in all_trades}),
        ("+ TAO + AAVE + CRV (9)", {s: all_trades[s] for s in EXPANDED_A if s in all_trades}),
        ("+ WIF + PEPE (11)", {s: all_trades[s] for s in FULL_EXPANDED if s in all_trades}),
    ]

    # Run simulations at 1x and 2x
    print(f"\n{'='*100}")
    print("  PORTFOLIO COMPARISON — 1x Leverage")
    print(f"{'='*100}")

    print(f"\n  {'Config':30s}  {'CAGR':>7s}  {'Total':>8s}  {'Sharpe':>7s}  {'MaxDD':>6s}  "
          f"{'Calmar':>7s}  {'Pos%':>5s}  {'Tr/mo':>6s}  {'Worst':>7s}  {'Best':>7s}")
    print("  " + "-" * 100)

    results_1x = {}
    for name, trades in configs:
        n = len(trades)
        r = simulate_portfolio(trades, n, leverage=1)
        if r is None:
            continue
        results_1x[name] = r
        print(f"  {name:30s}  {r['cagr']:+6.1f}%  {r['total_return']:+7.1f}%  "
              f"{r['sharpe']:+6.2f}  {r['maxdd']:5.1f}%  {r['calmar']:+6.2f}  "
              f"{r['pos_pct']:4.0f}%  {r['trades_per_month']:5.1f}  "
              f"{r['worst_month']:+6.2f}%  {r['best_month']:+6.2f}%")

    print(f"\n{'='*100}")
    print("  PORTFOLIO COMPARISON — 2x Leverage")
    print(f"{'='*100}")

    print(f"\n  {'Config':30s}  {'CAGR':>7s}  {'Total':>8s}  {'Sharpe':>7s}  {'MaxDD':>6s}  "
          f"{'Calmar':>7s}")
    print("  " + "-" * 70)

    results_2x = {}
    for name, trades in configs:
        n = len(trades)
        r = simulate_portfolio(trades, n, leverage=2)
        if r is None:
            continue
        results_2x[name] = r
        print(f"  {name:30s}  {r['cagr']:+6.1f}%  {r['total_return']:+7.1f}%  "
              f"{r['sharpe']:+6.2f}  {r['maxdd']:5.1f}%  {r['calmar']:+6.2f}")

    # ================================================================
    # Chart: Side-by-side equity curves
    # ================================================================
    print("\nGenerating charts...")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Portfolio Comparison: Original 6 vs Expanded Tier A', fontsize=14, fontweight='bold')

    colors = {
        "Original 6 (Tier A)": '#27ae60',
        "+ TAO (7)": '#2ecc71',
        "+ TAO + AAVE (8)": '#3498db',
        "+ TAO + AAVE + CRV (9)": '#9b59b6',
        "+ WIF + PEPE (11)": '#e67e22',
    }

    # 1x leverage
    ax = axes[0]
    for name, r in results_1x.items():
        cum = r['monthly_series'].cumsum()
        dates = [p.to_timestamp() for p in cum.index]
        ax.plot(dates, cum.values, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{name}\nCAGR:{r['cagr']:+.0f}% Sh:{r['sharpe']:+.2f} DD:{r['maxdd']:.0f}%")

    ax.set_title('1x Leverage — Cumulative Returns', fontsize=12)
    ax.set_ylabel('Cumulative Return %')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # 2x leverage
    ax = axes[1]
    for name, r in results_2x.items():
        cum = r['monthly_series'].cumsum()
        dates = [p.to_timestamp() for p in cum.index]
        ax.plot(dates, cum.values, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{name}\nCAGR:{r['cagr']:+.0f}% Sh:{r['sharpe']:+.2f} DD:{r['maxdd']:.0f}%")

    ax.set_title('2x Leverage — Cumulative Returns', fontsize=12)
    ax.set_ylabel('Cumulative Return %')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'expanded_portfolio_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'expanded_portfolio_comparison.png'}")

    # ================================================================
    # Drawdown comparison
    # ================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    for name, r in results_1x.items():
        cum = r['monthly_series'].cumsum()
        peak = cum.cummax()
        dd = -(peak - cum)
        dates = [p.to_timestamp() for p in dd.index]
        ax.plot(dates, dd.values, color=colors.get(name, 'gray'), linewidth=1.5,
                label=f"{name} (DD: {r['maxdd']:.1f}%)")

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.set_title('Drawdown Comparison — Original vs Expanded Tier A (1x)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Drawdown %')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'expanded_drawdown_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'expanded_drawdown_comparison.png'}")

    # ================================================================
    # Correlation of new symbols with existing
    # ================================================================
    print(f"\n{'='*100}")
    print("  CORRELATION OF NEW SYMBOLS WITH EXISTING TIER A")
    print(f"{'='*100}")

    # Build monthly returns for each symbol
    sym_monthly = {}
    for symbol in EXPANDED_A:
        if symbol not in all_trades:
            continue
        trades = all_trades[symbol]
        monthly = defaultdict(float)
        for t in trades:
            m = t['entry_time'].to_period('M')
            monthly[m] += t['net_pct']
        sym_monthly[symbol] = pd.Series(monthly)

    # Align to common months
    all_months = set()
    for s in sym_monthly.values():
        all_months.update(s.index)
    all_months = sorted(all_months)

    df_corr = pd.DataFrame(index=all_months)
    for sym, series in sym_monthly.items():
        df_corr[sym.replace('USDT', '')] = series.reindex(all_months).fillna(0)

    corr = df_corr.corr()

    # Show correlation of new symbols with existing
    new_syms = [s.replace('USDT', '') for s in NEW_A]
    orig_syms = [s.replace('USDT', '') for s in ORIGINAL_A]

    print(f"\n  {'New Symbol':>12s}", end='')
    for o in orig_syms:
        print(f"  {o:>6s}", end='')
    print(f"  {'Avg':>6s}")
    print("  " + "-" * (12 + 7 * (len(orig_syms) + 1)))

    for n in new_syms:
        if n not in corr.columns:
            continue
        print(f"  {n:>12s}", end='')
        corrs = []
        for o in orig_syms:
            if o in corr.columns:
                c = corr.loc[n, o]
                corrs.append(c)
                print(f"  {c:+5.2f}", end='')
            else:
                print(f"  {'N/A':>6s}", end='')
        avg_corr = np.mean(corrs) if corrs else 0
        print(f"  {avg_corr:+5.2f}")

    # Avg pairwise correlation for original vs expanded
    orig_pairs = []
    for i, s1 in enumerate(orig_syms):
        for s2 in orig_syms[i+1:]:
            if s1 in corr.columns and s2 in corr.columns:
                orig_pairs.append(corr.loc[s1, s2])

    expanded_syms = orig_syms + new_syms
    expanded_pairs = []
    for i, s1 in enumerate(expanded_syms):
        for s2 in expanded_syms[i+1:]:
            if s1 in corr.columns and s2 in corr.columns:
                expanded_pairs.append(corr.loc[s1, s2])

    print(f"\n  Avg pairwise correlation (original 6):  {np.mean(orig_pairs):+.3f}")
    print(f"  Avg pairwise correlation (expanded 9):  {np.mean(expanded_pairs):+.3f}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'='*100}")
    print("  FINAL VERDICT")
    print(f"{'='*100}")

    r_orig = results_1x.get("Original 6 (Tier A)")
    r_9 = results_1x.get("+ TAO + AAVE + CRV (9)")

    if r_orig and r_9:
        print(f"\n  {'Metric':20s}  {'Original 6':>12s}  {'Expanded 9':>12s}  {'Delta':>10s}")
        print("  " + "-" * 60)

        metrics = [
            ('CAGR', r_orig['cagr'], r_9['cagr'], '%'),
            ('Sharpe', r_orig['sharpe'], r_9['sharpe'], ''),
            ('Max DD', r_orig['maxdd'], r_9['maxdd'], '%'),
            ('Calmar', r_orig['calmar'], r_9['calmar'], ''),
            ('Trades/month', r_orig['trades_per_month'], r_9['trades_per_month'], ''),
            ('Pos months %', r_orig['pos_pct'], r_9['pos_pct'], '%'),
            ('Worst month', r_orig['worst_month'], r_9['worst_month'], '%'),
        ]

        for name, v1, v2, unit in metrics:
            delta = v2 - v1
            better = '✅' if (delta > 0 and name not in ['Max DD', 'Worst month']) or \
                            (delta < 0 and name in ['Max DD']) or \
                            (delta > 0 and name == 'Worst month') else '❌'
            print(f"  {name:20s}  {v1:>11.1f}{unit}  {v2:>11.1f}{unit}  "
                  f"{delta:>+9.1f}{unit} {better}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
