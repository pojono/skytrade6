#!/usr/bin/env python3
"""
Compounding Analysis: Does more symbols = better compounding?

Key question: The strategy catches rare events (~4 trades/month/symbol).
More symbols means more events, more time capital is deployed, more compounding.
But weaker symbols dilute per-trade edge.

This script simulates COMPOUNDED returns (not simple sum) for various portfolio sizes,
and also tests: what if we only add Tier A-quality symbols?
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

ALL_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "POLUSDT", "LTCUSDT", "UNIUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "NEARUSDT", "FILUSDT", "ATOMUSDT", "SUIUSDT",
]

# Sorted by Sharpe from our analysis
TIER_A = ["SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "UNIUSDT", "SUIUSDT"]
TIER_B = ["ARBUSDT", "LINKUSDT", "LTCUSDT", "ETHUSDT", "NEARUSDT", "OPUSDT"]
TIER_C = ["ATOMUSDT", "BTCUSDT", "FILUSDT", "DOGEUSDT", "AVAXUSDT", "APTUSDT", "DOTUSDT"]

# Ordered by quality
SYMBOLS_BY_QUALITY = TIER_A + TIER_B + TIER_C


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


def simulate_compounded(symbol_trades, n_symbols, leverage=1):
    """
    Simulate compounded equity curve.
    
    Capital is split equally across n_symbols.
    Each trade uses its allocated slice * leverage.
    Trades compound: profit from one trade increases capital for next.
    
    Key insight: with more symbols, capital is deployed more often,
    so compounding happens faster.
    """
    # Collect all trades with their symbol allocation weight
    all_events = []
    weight = 1.0 / n_symbols

    for sym, trades in symbol_trades.items():
        for t in trades:
            # Return on total portfolio from this trade
            # = weight * leverage * trade_return_pct
            portfolio_impact_pct = weight * leverage * t['net_pct']
            all_events.append({
                'time': t['entry_time'],
                'exit_time': t['exit_time'],
                'pct': portfolio_impact_pct,
                'symbol': sym,
            })

    # Sort by entry time
    all_events.sort(key=lambda x: x['time'])

    if not all_events:
        return None

    # Compound equity
    equity = 100.0
    equity_curve = [(all_events[0]['time'], equity)]
    monthly_equity = {}

    for ev in all_events:
        equity *= (1 + ev['pct'] / 100)
        equity_curve.append((ev['exit_time'], equity))
        month = ev['time'].to_period('M')
        monthly_equity[month] = equity

    # Monthly returns from compounded equity
    months = sorted(monthly_equity.keys())
    monthly_ret = []
    prev_eq = 100.0
    for m in months:
        ret = (monthly_equity[m] - prev_eq) / prev_eq * 100
        monthly_ret.append((m, ret))
        prev_eq = monthly_equity[m]

    monthly_series = pd.Series({m: r for m, r in monthly_ret})

    # Stats
    total_return = (equity / 100 - 1) * 100
    n_months = len(months)
    n_years = n_months / 12
    # CAGR
    cagr = ((equity / 100) ** (1 / max(n_years, 0.5)) - 1) * 100
    sharpe = monthly_series.mean() / max(monthly_series.std(), 0.001) * np.sqrt(12)
    cum = monthly_series.cumsum()
    peak = cum.cummax()
    maxdd = (peak - cum).max()
    calmar = cagr / max(maxdd, 0.1)

    # Time in market
    total_hours = (all_events[-1]['exit_time'] - all_events[0]['time']).total_seconds() / 3600
    trade_hours = len(all_events) * HOLD_BARS  # each trade is 4h
    time_in_market = trade_hours / max(total_hours, 1) * 100

    return {
        'equity': equity,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'maxdd': maxdd,
        'calmar': calmar,
        'n_trades': len(all_events),
        'n_months': n_months,
        'time_in_market': time_in_market,
        'monthly_series': monthly_series,
        'equity_curve': equity_curve,
    }


def main():
    t0 = time.time()
    print("=" * 100)
    print("COMPOUNDING ANALYSIS — Does More Symbols = Better Compounding?")
    print("=" * 100)

    # Load all trades
    all_symbol_trades = {}
    for si, symbol in enumerate(SYMBOLS_BY_QUALITY, 1):
        df = load_1h(symbol)
        if df.empty or len(df) < 2000:
            continue
        df = compute_signals(df)
        df['month'] = df.index.to_period('M')
        months = sorted(df['month'].unique())
        warmup_end = months[5] if len(months) > 5 else months[-1]
        trade_df = df[df['month'] > warmup_end]
        trades = get_trades(trade_df)
        all_symbol_trades[symbol] = trades
        tier = 'A' if symbol in TIER_A else ('B' if symbol in TIER_B else 'C')
        print(f"  [{si:2d}/19] {symbol:12s} (Tier {tier}) — {len(trades)} trades, "
              f"avg {np.mean([t['net_bps'] for t in trades]):+.1f} bps/trade")

    # ================================================================
    # Test: incrementally add symbols by quality
    # ================================================================
    print(f"\n{'='*100}")
    print("  COMPOUNDED RETURNS — Adding Symbols by Quality (1x leverage)")
    print(f"{'='*100}")

    print(f"\n  {'N':>3s}  {'Symbols':>40s}  {'CAGR':>7s}  {'Total':>8s}  {'Sharpe':>7s}  "
          f"{'MaxDD':>6s}  {'Calmar':>7s}  {'Trades':>7s}  {'Tr/mo':>6s}  {'Time%':>6s}")
    print("  " + "-" * 105)

    incremental_results = []
    available = [s for s in SYMBOLS_BY_QUALITY if s in all_symbol_trades]

    for n in range(1, len(available) + 1):
        subset = {s: all_symbol_trades[s] for s in available[:n]}
        r = simulate_compounded(subset, n, leverage=1)
        if r is None:
            continue

        syms_str = '+'.join([s.replace('USDT', '') for s in available[:n]])
        if len(syms_str) > 40:
            syms_str = syms_str[:37] + '...'

        trades_per_month = r['n_trades'] / max(r['n_months'], 1)

        incremental_results.append({
            'n': n, 'cagr': r['cagr'], 'total': r['total_return'],
            'sharpe': r['sharpe'], 'maxdd': r['maxdd'], 'calmar': r['calmar'],
            'trades': r['n_trades'], 'tr_per_mo': trades_per_month,
            'time_in_market': r['time_in_market'],
            'monthly_series': r['monthly_series'],
            'equity_curve': r['equity_curve'],
        })

        print(f"  {n:3d}  {syms_str:>40s}  {r['cagr']:+6.1f}%  {r['total_return']:+7.1f}%  "
              f"{r['sharpe']:+6.2f}  {r['maxdd']:5.1f}%  {r['calmar']:+6.2f}  "
              f"{r['n_trades']:>6d}  {trades_per_month:5.1f}  {r['time_in_market']:5.1f}%")

    # ================================================================
    # Same analysis at 2x leverage
    # ================================================================
    print(f"\n{'='*100}")
    print("  COMPOUNDED RETURNS — Adding Symbols by Quality (2x leverage)")
    print(f"{'='*100}")

    print(f"\n  {'N':>3s}  {'CAGR':>7s}  {'Total':>8s}  {'Sharpe':>7s}  {'MaxDD':>6s}  {'Calmar':>7s}")
    print("  " + "-" * 50)

    lev2_results = []
    for n in range(1, len(available) + 1):
        subset = {s: all_symbol_trades[s] for s in available[:n]}
        r = simulate_compounded(subset, n, leverage=2)
        if r is None:
            continue
        lev2_results.append({
            'n': n, 'cagr': r['cagr'], 'total': r['total_return'],
            'sharpe': r['sharpe'], 'maxdd': r['maxdd'], 'calmar': r['calmar'],
        })
        print(f"  {n:3d}  {r['cagr']:+6.1f}%  {r['total_return']:+7.1f}%  "
              f"{r['sharpe']:+6.2f}  {r['maxdd']:5.1f}%  {r['calmar']:+6.2f}")

    # ================================================================
    # Key comparison table
    # ================================================================
    print(f"\n{'='*100}")
    print("  KEY COMPARISON — Compounded vs Simple Returns")
    print(f"{'='*100}")

    key_ns = [1, 3, 6, 10, 13, 19]
    print(f"\n  {'N coins':>8s}  {'CAGR 1x':>8s}  {'CAGR 2x':>8s}  {'Trades/mo':>10s}  "
          f"{'Time in mkt':>12s}  {'MaxDD 1x':>9s}  {'Sharpe 1x':>10s}")
    print("  " + "-" * 75)

    for n in key_ns:
        if n > len(incremental_results):
            continue
        r1 = incremental_results[n - 1]
        r2 = lev2_results[n - 1] if n <= len(lev2_results) else None
        cagr2 = r2['cagr'] if r2 else 0
        print(f"  {n:>7d}  {r1['cagr']:+7.1f}%  {cagr2:+7.1f}%  {r1['tr_per_mo']:>9.1f}  "
              f"{r1['time_in_market']:>10.1f}%  {r1['maxdd']:>8.1f}%  {r1['sharpe']:>9.2f}")

    # ================================================================
    # Chart: CAGR and Sharpe vs number of symbols
    # ================================================================
    print("\nGenerating charts...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impact of Adding More Symbols (Ordered by Quality)',
                 fontsize=14, fontweight='bold')

    ns = [r['n'] for r in incremental_results]
    cagrs = [r['cagr'] for r in incremental_results]
    sharpes = [r['sharpe'] for r in incremental_results]
    maxdds = [r['maxdd'] for r in incremental_results]
    calmars = [r['calmar'] for r in incremental_results]
    trades_mo = [r['tr_per_mo'] for r in incremental_results]
    time_pcts = [r['time_in_market'] for r in incremental_results]

    # Tier boundaries
    tier_a_end = len(TIER_A)
    tier_b_end = len(TIER_A) + len(TIER_B)

    # CAGR
    ax = axes[0][0]
    ax.plot(ns, cagrs, 'o-', color='#2ecc71', linewidth=2, markersize=6)
    if len(lev2_results) > 0:
        ax.plot([r['n'] for r in lev2_results], [r['cagr'] for r in lev2_results],
                's--', color='#3498db', linewidth=2, markersize=5, label='2x leverage')
        ax.legend()
    ax.axvspan(0.5, tier_a_end + 0.5, alpha=0.1, color='green', label='Tier A')
    ax.axvspan(tier_a_end + 0.5, tier_b_end + 0.5, alpha=0.1, color='blue')
    ax.axvspan(tier_b_end + 0.5, 20.5, alpha=0.1, color='red')
    ax.set_xlabel('Number of Symbols')
    ax.set_ylabel('CAGR %')
    ax.set_title('Compounded Annual Growth Rate')
    ax.grid(True, alpha=0.3)
    # Mark peak
    peak_idx = np.argmax(cagrs)
    ax.annotate(f'Peak: {cagrs[peak_idx]:+.1f}% @ {ns[peak_idx]} coins',
               (ns[peak_idx], cagrs[peak_idx]),
               textcoords="offset points", xytext=(10, 10), fontsize=10,
               fontweight='bold', color='#27ae60',
               arrowprops=dict(arrowstyle='->', color='#27ae60'))

    # Sharpe
    ax = axes[0][1]
    ax.plot(ns, sharpes, 'o-', color='#9b59b6', linewidth=2, markersize=6)
    ax.axvspan(0.5, tier_a_end + 0.5, alpha=0.1, color='green')
    ax.axvspan(tier_a_end + 0.5, tier_b_end + 0.5, alpha=0.1, color='blue')
    ax.axvspan(tier_b_end + 0.5, 20.5, alpha=0.1, color='red')
    ax.set_xlabel('Number of Symbols')
    ax.set_ylabel('Monthly Sharpe')
    ax.set_title('Sharpe Ratio')
    ax.grid(True, alpha=0.3)

    # Trades/month and time in market
    ax = axes[1][0]
    ax2 = ax.twinx()
    l1 = ax.bar(ns, trades_mo, color='#3498db', alpha=0.6, label='Trades/month')
    l2 = ax2.plot(ns, time_pcts, 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Time in market %')
    ax.set_xlabel('Number of Symbols')
    ax.set_ylabel('Trades per Month', color='#3498db')
    ax2.set_ylabel('Time in Market %', color='#e74c3c')
    ax.set_title('Trade Frequency & Capital Utilization')
    lines = [l1] + l2
    labels = ['Trades/month', 'Time in market %']
    ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Max DD
    ax = axes[1][1]
    ax.plot(ns, maxdds, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax.axvspan(0.5, tier_a_end + 0.5, alpha=0.1, color='green')
    ax.axvspan(tier_a_end + 0.5, tier_b_end + 0.5, alpha=0.1, color='blue')
    ax.axvspan(tier_b_end + 0.5, 20.5, alpha=0.1, color='red')
    ax.set_xlabel('Number of Symbols')
    ax.set_ylabel('Max Drawdown %')
    ax.set_title('Max Drawdown (higher = worse)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'compounding_vs_symbols.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'compounding_vs_symbols.png'}")

    # ================================================================
    # Chart 2: Equity curves for key portfolio sizes
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Compounded Equity Curves — Different Portfolio Sizes (1x)',
                 fontsize=14, fontweight='bold')

    key_configs = [1, 3, 6, 10, 13, 19]
    colors = ['#27ae60', '#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

    for ax, n, color in zip(axes.flat, key_configs, colors):
        if n > len(incremental_results):
            continue
        r = incremental_results[n - 1]
        ec = r['equity_curve']
        times = [t for t, _ in ec]
        vals = [v for _, v in ec]

        ax.fill_between(times, 100, vals, alpha=0.3, color=color)
        ax.plot(times, vals, color=color, linewidth=1.5)
        ax.axhline(y=100, color='gray', linewidth=0.5, linestyle='--')

        syms = available[:n]
        tier_label = "Tier A" if n <= 6 else f"A+B" if n <= 12 else "A+B+C"
        ax.set_title(f"{n} coins ({tier_label})\n"
                     f"CAGR: {r['cagr']:+.1f}%, DD: {r['maxdd']:.1f}%, "
                     f"Sharpe: {r['sharpe']:+.2f}\n"
                     f"{r['tr_per_mo']:.0f} trades/mo, {r['time_in_market']:.1f}% time in mkt",
                     fontsize=9)
        ax.set_ylabel('Equity')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'compounding_equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {CHART_DIR / 'compounding_equity_curves.png'}")

    # ================================================================
    # Final verdict
    # ================================================================
    print(f"\n{'='*100}")
    print("  VERDICT")
    print(f"{'='*100}")

    # Find optimal N by CAGR
    best_cagr_idx = np.argmax(cagrs)
    best_cagr_n = ns[best_cagr_idx]

    # Find optimal N by Calmar
    best_calmar_idx = np.argmax(calmars)
    best_calmar_n = ns[best_calmar_idx]

    # Find optimal N by Sharpe
    best_sharpe_idx = np.argmax(sharpes)
    best_sharpe_n = ns[best_sharpe_idx]

    print(f"\n  Best CAGR:   {cagrs[best_cagr_idx]:+.1f}% at {best_cagr_n} coins")
    print(f"  Best Sharpe: {sharpes[best_sharpe_idx]:+.2f} at {best_sharpe_n} coins")
    print(f"  Best Calmar: {calmars[best_calmar_idx]:+.2f} at {best_calmar_n} coins")

    print(f"\n  At 6 coins (Tier A):  CAGR {cagrs[5]:+.1f}%, Sharpe {sharpes[5]:+.2f}, "
          f"DD {maxdds[5]:.1f}%, {trades_mo[5]:.0f} tr/mo")
    if len(cagrs) >= 12:
        print(f"  At 12 coins (A+B):    CAGR {cagrs[11]:+.1f}%, Sharpe {sharpes[11]:+.2f}, "
              f"DD {maxdds[11]:.1f}%, {trades_mo[11]:.0f} tr/mo")
    if len(cagrs) >= 19:
        print(f"  At 19 coins (all):    CAGR {cagrs[18]:+.1f}%, Sharpe {sharpes[18]:+.2f}, "
              f"DD {maxdds[18]:.1f}%, {trades_mo[18]:.0f} tr/mo")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
