"""Markdown report generator for the settlement trading pipeline."""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline.config import (
    REPORT_FILE, TAKER_FEE_BPS, MAKER_FEE_BPS,
    LIMIT_FILL_RATE, CAP_PCT, LONG_ENTRY_MAX_T_S,
    LONG_EXIT_ML_THRESHOLD, LONG_HOLD_FIXED_MS,
    GROSS_PNL_BPS,
)


def generate_report(strategies, short_exit_results=None, long_exit_results=None,
                    output_path=None):
    """Generate comprehensive markdown report from backtest results.

    Args:
        strategies: dict from compare_strategies()
        short_exit_results: short exit ML training metrics
        long_exit_results: long exit ML training metrics
        output_path: where to write the report
    """
    path = output_path or REPORT_FILE
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Settlement Trading Pipeline — Report")
    lines.append(f"")
    lines.append(f"**Generated:** {now}")
    lines.append(f"")

    # Get the best strategy for headline numbers
    best_name = max(strategies.keys(),
                    key=lambda k: strategies[k][1]['combined_per_day'])
    best_results, best_summary = strategies[best_name]
    n_days = best_summary['n_days']

    lines.append(f"## Executive Summary")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Best strategy** | {best_name} |")
    lines.append(f"| **Daily revenue** | **${best_summary['combined_per_day']:.1f}/day** |")
    lines.append(f"| Short leg | ${best_summary['short_per_day']:.1f}/day ({best_summary['short_wr']:.0f}% WR) |")
    lines.append(f"| Long leg | ${best_summary['long_per_day']:.1f}/day ({best_summary['long_wr']:.0f}% WR) |")
    lines.append(f"| Settlements traded | {best_summary['n_traded']} / {best_summary['n_settlements']} |")
    lines.append(f"| Long trades taken | {best_summary['n_long_taken']} / {best_summary['n_traded']} |")
    lines.append(f"| Data period | {n_days} days |")
    lines.append(f"")

    # Strategy comparison
    lines.append(f"## Strategy Comparison")
    lines.append(f"")
    lines.append(f"| Strategy | Short $/day | Long $/day | **Total $/day** | Long WR |")
    lines.append(f"|----------|------------|-----------|----------------|---------|")
    for name, (_, s) in strategies.items():
        marker = " **← best**" if name == best_name else ""
        lines.append(f"| {name} | ${s['short_per_day']:.1f} | ${s['long_per_day']:.1f} | "
                     f"**${s['combined_per_day']:.1f}** | {s['long_wr']:.0f}% |{marker}")
    lines.append(f"")

    # Configuration
    lines.append(f"## Configuration")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Taker fee | {TAKER_FEE_BPS} bps/leg |")
    lines.append(f"| Maker fee | {MAKER_FEE_BPS} bps/leg |")
    lines.append(f"| Limit fill rate | {LIMIT_FILL_RATE:.0%} |")
    lines.append(f"| Position cap | {CAP_PCT:.0%} of depth_20 |")
    lines.append(f"| Short gross edge | {GROSS_PNL_BPS} bps (ML LOSO) |")
    lines.append(f"| Long entry rule | bottom T ≤ {LONG_ENTRY_MAX_T_S:.0f}s |")
    lines.append(f"| Long exit ML threshold | p ≥ {LONG_EXIT_ML_THRESHOLD} |")
    lines.append(f"| Long fixed hold | +{LONG_HOLD_FIXED_MS//1000}s |")
    lines.append(f"")

    # ML model performance
    if short_exit_results:
        lines.append(f"## Short Exit ML")
        lines.append(f"")
        lines.append(f"| Model | Train AUC | Test AUC | Gap |")
        lines.append(f"|-------|-----------|----------|-----|")
        lines.append(f"| LogReg | {short_exit_results.get('lr_auc_train', 0):.4f} | "
                     f"{short_exit_results.get('lr_auc_test', 0):.4f} | "
                     f"{short_exit_results.get('lr_auc_train', 0) - short_exit_results.get('lr_auc_test', 0):.3f} |")
        lines.append(f"| HGBC | {short_exit_results.get('hgbc_auc_train', 0):.4f} | "
                     f"{short_exit_results.get('hgbc_auc_test', 0):.4f} | "
                     f"{short_exit_results.get('hgbc_auc_train', 0) - short_exit_results.get('hgbc_auc_test', 0):.3f} |")
        if short_exit_results.get('loso_auc'):
            lines.append(f"| LOSO (HGBC) | — | {short_exit_results['loso_auc']:.4f} | — |")
        lines.append(f"")
        lines.append(f"Target: `near_bottom_10` — {short_exit_results.get('n_ticks', 0):,} ticks, "
                     f"{short_exit_results.get('n_settle', 0)} settlements")
        lines.append(f"")

    if long_exit_results:
        lines.append(f"## Long Exit ML")
        lines.append(f"")
        lines.append(f"| Model | Train AUC | Test AUC | Gap |")
        lines.append(f"|-------|-----------|----------|-----|")
        lines.append(f"| LogReg | {long_exit_results.get('lr_auc_train', 0):.4f} | "
                     f"{long_exit_results.get('lr_auc_test', 0):.4f} | "
                     f"{long_exit_results.get('lr_auc_train', 0) - long_exit_results.get('lr_auc_test', 0):.3f} |")
        lines.append(f"| HGBC | {long_exit_results.get('hgbc_auc_train', 0):.4f} | "
                     f"{long_exit_results.get('hgbc_auc_test', 0):.4f} | "
                     f"{long_exit_results.get('hgbc_auc_train', 0) - long_exit_results.get('hgbc_auc_test', 0):.3f} |")
        if long_exit_results.get('loso_auc'):
            lines.append(f"| LOSO (HGBC) | — | {long_exit_results['loso_auc']:.4f} | — |")
        lines.append(f"")
        lines.append(f"Target: `near_peak_10` — {long_exit_results.get('n_ticks', 0):,} ticks, "
                     f"{long_exit_results.get('n_settle', 0)} settlements")
        lines.append(f"")

    # Outcome distribution
    lines.append(f"## Outcome Distribution ({best_name})")
    lines.append(f"")
    traded = [r for r in best_results if r.passes_filters]
    if traded:
        lines.append(f"| Outcome | Count | % | Avg $ |")
        lines.append(f"|---------|-------|---|-------|")
        outcomes = {}
        for r in traded:
            if r.outcome not in outcomes:
                outcomes[r.outcome] = []
            outcomes[r.outcome].append(r.combined_dollar)
        for outcome in sorted(outcomes.keys()):
            vals = outcomes[outcome]
            n = len(vals)
            pct = n / len(traded) * 100
            avg = np.mean(vals)
            lines.append(f"| {outcome} | {n} | {pct:.0f}% | ${avg:+.2f} |")
        lines.append(f"")

    # Per-symbol breakdown
    lines.append(f"## Per-Symbol Performance ({best_name})")
    lines.append(f"")
    from collections import defaultdict
    sym_data = defaultdict(list)
    for r in traded:
        sym_data[r.symbol].append(r)

    sym_stats = []
    for sym, recs in sym_data.items():
        n = len(recs)
        s_wr = sum(1 for r in recs if r.short_win) / n * 100
        long_recs = [r for r in recs if r.long_taken]
        l_wr = sum(1 for r in long_recs if r.long_win) / len(long_recs) * 100 if long_recs else 0
        avg_c = np.mean([r.combined_dollar for r in recs])
        sym_stats.append((sym, n, s_wr, l_wr, avg_c))

    sym_stats.sort(key=lambda x: x[4], reverse=True)
    lines.append(f"| Symbol | N | Short WR | Long WR | Avg $/trade |")
    lines.append(f"|--------|---|----------|---------|-------------|")
    for sym, n, s_wr, l_wr, avg_c in sym_stats[:20]:
        lines.append(f"| {sym} | {n} | {s_wr:.0f}% | {l_wr:.0f}% | ${avg_c:+.2f} |")
    lines.append(f"")

    # Worst trades
    lines.append(f"## Worst Combined Trades ({best_name})")
    lines.append(f"")
    worst = sorted(traded, key=lambda r: r.combined_dollar)[:10]
    lines.append(f"| File | Symbol | Short $ | Long $ | Combined $ | Drop | Exit |")
    lines.append(f"|------|--------|---------|--------|-----------|------|------|")
    for r in worst:
        lines.append(f"| {r.file_name} | {r.symbol} | ${r.short_dollar:+.2f} | "
                     f"${r.long_dollar:+.2f} | ${r.combined_dollar:+.2f} | "
                     f"{r.drop_bps:+.0f}bps | {r.long_exit_method} |")
    lines.append(f"")

    # Production rules
    lines.append(f"## Production Rules")
    lines.append(f"")
    lines.append(f"```python")
    lines.append(f"# 1. FILTERS (skip if fails)")
    lines.append(f"if depth_20 < 2000 or spread_bps > 8:")
    lines.append(f"    skip()")
    lines.append(f"")
    lines.append(f"# 2. SHORT LEG (always)")
    lines.append(f"notional = adaptive_size(depth_20, cap=0.15)")
    lines.append(f"short_entry = market_sell(notional)  # taker")
    lines.append(f"short_exit = limit_buy(notional)     # maker if fills, taker rescue at 1s")
    lines.append(f"")
    lines.append(f"# 3. LONG ENTRY DECISION (at short exit moment)")
    lines.append(f"if ml_exit_time <= 15.0:  # seconds since settlement")
    lines.append(f"    buy_qty = 2 * notional  # 1x close short + 1x open long")
    lines.append(f"else:")
    lines.append(f"    buy_qty = 1 * notional  # just close short")
    lines.append(f"")
    lines.append(f"# 4. LONG EXIT (if long taken)")
    lines.append(f"# Poll recovery ticks every 100ms")
    lines.append(f"# LogReg predicts p(near_peak_10)")
    lines.append(f"if pred_prob >= 0.6:")
    lines.append(f"    limit_sell(long_notional)  # recovery peaking")
    lines.append(f"elif time_since_bottom >= 30s:")
    lines.append(f"    limit_sell(long_notional)  # forced timeout")
    lines.append(f"```")
    lines.append(f"")

    # Write
    text = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Report written: {path} ({len(lines)} lines)")
    return path
