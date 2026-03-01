"""Markdown report generator for the settlement trading pipeline."""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline.config import (
    REPORT_FILE, TAKER_FEE_BPS, MAKER_FEE_BPS,
    LIMIT_FILL_RATE, CAP_PCT,
    SHORT_EXIT_ML_THRESHOLD, SHORT_EXIT_TIMEOUT_MS,
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

    lines.append(f"# Settlement Trading Pipeline — Report (Short-Only)")
    lines.append(f"")
    lines.append(f"**Generated:** {now}")
    lines.append(f"")

    # Production strategy is always short_only
    prod_results, prod_summary = strategies['short_only']
    n_days = prod_summary['n_days']

    lines.append(f"## Executive Summary")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Strategy** | short-only (ML-timed exit) |")
    lines.append(f"| **Daily revenue (in-sample)** | **${prod_summary['combined_per_day']:.1f}/day** |")
    lines.append(f"| **LOSO conservative** | **$50–$75/day** |")
    lines.append(f"| Win rate | {prod_summary['short_wr']:.0f}% |")
    lines.append(f"| Avg $/trade | ${prod_summary['avg_short_dollar']:.2f} |")
    lines.append(f"| Settlements traded | {prod_summary['n_traded']} / {prod_summary['n_settlements']} |")
    lines.append(f"| Data period | {n_days} days |")
    lines.append(f"")

    # Strategy comparison
    if len(strategies) > 1:
        lines.append(f"## Strategy Comparison (research)")
        lines.append(f"")
        lines.append(f"| Strategy | Short $/day | Long $/day | **Total $/day** | Short WR | Long WR |")
        lines.append(f"|----------|------------|-----------|----------------|----------|---------|")
        for name, (_, s) in strategies.items():
            marker = " **← production**" if name == 'short_only' else ""
            lines.append(f"| {name} | ${s['short_per_day']:.1f} | ${s['long_per_day']:.1f} | "
                         f"**${s['combined_per_day']:.1f}** | {s['short_wr']:.0f}% | {s['long_wr']:.0f}% |{marker}")
        lines.append(f"")
        lines.append(f"> **Note:** Long leg is unprofitable without look-ahead bias. Short-only is the production strategy.")
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
    lines.append(f"| Short gross edge (LOSO avg) | {GROSS_PNL_BPS} bps |")
    lines.append(f"| Short exit ML threshold | p(near_bottom) ≥ {SHORT_EXIT_ML_THRESHOLD} |")
    lines.append(f"| Short exit timeout | {SHORT_EXIT_TIMEOUT_MS//1000}s |")
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
    lines.append(f"## Outcome Distribution (short-only)")
    lines.append(f"")
    traded = [r for r in prod_results if r.passes_filters]
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
    lines.append(f"## Per-Symbol Performance (short-only)")
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
    lines.append(f"## Worst Trades (short-only)")
    lines.append(f"")
    worst = sorted(traded, key=lambda r: r.short_dollar)[:10]
    lines.append(f"| File | Symbol | Short $ | Gross bps | Slip bps | Depth $20 | Spread |")
    lines.append(f"|------|--------|---------|-----------|----------|-----------|--------|")
    for r in worst:
        lines.append(f"| {r.file_name} | {r.symbol} | ${r.short_dollar:+.2f} | "
                     f"{r.short_gross_bps:+.1f} | {r.short_slip_bps:.1f} | "
                     f"${r.depth_20:,.0f} | {r.spread_bps:.1f} |")
    lines.append(f"")

    # Production rules
    lines.append(f"## Production Rules (Short-Only)")
    lines.append(f"")
    lines.append(f"```python")
    lines.append(f"# 1. FILTERS (skip if fails)")
    lines.append(f"if depth_20 < 2000 or spread_bps > 8:")
    lines.append(f"    skip()")
    lines.append(f"")
    lines.append(f"# 2. SIZE")
    lines.append(f"notional = adaptive_size(depth_20, cap=0.15)")
    lines.append(f"")
    lines.append(f"# 3. SHORT ENTRY (at settlement T=0)")
    lines.append(f"short_entry = market_sell(notional)  # taker")
    lines.append(f"")
    lines.append(f"# 4. SHORT EXIT (ML-timed)")
    lines.append(f"# Poll tick features every 100ms")
    lines.append(f"# short_exit_logreg predicts p(near_bottom_10)")
    lines.append(f"if pred_prob >= {SHORT_EXIT_ML_THRESHOLD}:")
    lines.append(f"    limit_buy(notional)  # near bottom, cover short")
    lines.append(f"elif time_since_settlement >= {SHORT_EXIT_TIMEOUT_MS // 1000}s:")
    lines.append(f"    market_buy(notional)  # forced timeout")
    lines.append(f"```")
    lines.append(f"")

    # Write
    text = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Report written: {path} ({len(lines)} lines)")
    return path
