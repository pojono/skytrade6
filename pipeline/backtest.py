"""Combined both-legs backtest.

Simulates the full strategy per settlement:
  1. Short entry at settlement (taker)
  2. Short exit at ML-detected bottom (limit + rescue)
  3. Long entry decision (rule: bottom T ≤ 15s)
  4. Long exit at ML-detected peak (limit + rescue) OR fixed hold
  5. Position sizing (adaptive based on depth)
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from pipeline.config import (
    TAKER_FEE_BPS, MAKER_FEE_BPS, GROSS_PNL_BPS,
    LIMIT_FILL_RATE, CAP_PCT,
    LONG_HOLD_FIXED_MS, LONG_EXIT_ML_THRESHOLD,
    compute_notional, compute_long_notional, mixed_fee_bps,
)
from pipeline.features import find_bottom, build_long_exit_ticks
from pipeline.models import should_go_long, predict_long_exit

from research_position_sizing import compute_slippage_bps


# ═══════════════════════════════════════════════════════════════════════
# TRADE RESULT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    """Result of one settlement trade (both legs)."""
    settle_id: str
    symbol: str
    file_name: str

    # Filters
    passes_filters: bool
    depth_20: float
    spread_bps: float
    fr_abs_bps: float

    # Position sizing
    short_notional: float
    long_notional: float

    # Short leg
    short_gross_bps: float
    short_slip_bps: float
    short_fee_bps: float
    short_net_bps: float
    short_dollar: float
    short_win: bool

    # Bottom detection
    bottom_bps: Optional[float]
    bottom_t_s: Optional[float]
    drop_bps: float

    # Long leg decision
    long_eligible: bool      # passes T<=15s rule
    long_taken: bool         # actually traded

    # Long leg (if taken)
    long_recovery_bps: float
    long_fee_bps: float
    long_slip_bps: float
    long_net_bps: float
    long_dollar: float
    long_win: bool
    long_exit_method: str    # 'ml', 'fixed', 'none'
    long_exit_t_ms: Optional[float]

    # Combined
    combined_dollar: float

    # Outcome category
    outcome: str             # 'both_win', 'short_win_long_lose', etc.


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _get_spread_at(sd, t_ms):
    """Look up actual L1 spread at a given time from ob1 data.
    Falls back to T-0 spread if no post-settlement ob1 data available.
    """
    if t_ms is None or len(sd.ob1_times) == 0:
        return sd.spread_bps

    mask = sd.ob1_times <= t_ms
    if mask.sum() == 0:
        return sd.spread_bps

    idx = mask.sum() - 1
    _, bp, bq, ap, aq = sd.ob1_bids[idx]  # ob1_bids stores full (t, bp, bq, ap, aq) tuples
    if bp <= 0 or ap <= 0:
        return sd.spread_bps
    mid = (bp + ap) / 2
    return (ap - bp) / mid * 10000


# ═══════════════════════════════════════════════════════════════════════
# PER-SETTLEMENT SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def simulate_settlement(sd, long_exit_model=None, long_exit_features=None,
                        use_ml_exit=True, use_fixed_exit=True, no_long=False):
    """Simulate full strategy on one settlement.

    Args:
        sd: SettlementData object
        long_exit_model: trained LogReg model for long exit (optional)
        long_exit_features: feature column names for long exit model
        use_ml_exit: if True, use ML for long exit timing
        use_fixed_exit: if True and ML unavailable, use fixed hold
        no_long: if True, never take the long leg

    Returns:
        TradeResult
    """
    mid = sd.mid_price
    depth_20 = sd.depth_20

    # Position sizing (independent for each leg)
    short_notional = compute_notional(depth_20)
    long_notional = compute_long_notional(depth_20)

    # Slippage
    entry_s = compute_slippage_bps(sd.asks, short_notional, 'buy', mid_price=mid)
    exit_s = compute_slippage_bps(sd.bids, short_notional, 'sell', mid_price=mid)
    rt_slip = entry_s['slippage_bps'] + exit_s['slippage_bps']

    # Short leg PnL
    fee_save = LIMIT_FILL_RATE * (TAKER_FEE_BPS - MAKER_FEE_BPS)
    short_exit_fee = mixed_fee_bps()
    short_fee = TAKER_FEE_BPS + short_exit_fee
    short_net = GROSS_PNL_BPS - rt_slip + fee_save
    short_dollar = short_net * short_notional / 10000

    # Find bottom
    bottom_bps, bottom_t = find_bottom(sd)
    drop_bps = -bottom_bps if bottom_bps is not None else 0

    # Long leg decision
    long_eligible = bottom_t is not None and should_go_long(bottom_t)
    long_taken = sd.passes_filters and long_eligible and not no_long

    # Long leg simulation
    long_recovery_bps = 0.0
    long_fee = 0.0
    long_slip = 0.0
    long_net = 0.0
    long_dollar = 0.0
    long_win = False
    long_exit_method = 'none'
    long_exit_t = None

    if long_taken and bottom_bps is not None:
        # Fees (limit both sides)
        long_fee = 2 * mixed_fee_bps()

        # ML exit
        if use_ml_exit and long_exit_model is not None and long_exit_features is not None:
            recovery_ticks = build_long_exit_ticks(sd, bottom_bps, bottom_t)
            if recovery_ticks:
                ml_result = predict_long_exit(long_exit_model, long_exit_features, recovery_ticks)
                if ml_result is not None:
                    long_recovery_bps = ml_result['exit_recovery_bps']
                    long_exit_t = ml_result['exit_t_ms']
                    long_exit_method = 'ml'

        # Fallback: fixed hold
        if long_exit_method == 'none' and use_fixed_exit:
            target_t = bottom_t + LONG_HOLD_FIXED_MS
            for t_ms in sd.price_bins:
                if target_t - 500 <= t_ms <= target_t + 500:
                    long_recovery_bps = sd.price_bins[t_ms] - bottom_bps
                    long_exit_t = t_ms
                    long_exit_method = 'fixed'

        # ── Honest slippage: actual L1 spread at entry/exit + depth-walking ──
        # 1) Depth-walking cost from T-0 full OB (only source for depth info)
        #    = total_slippage_vs_mid - half_spread_at_T0
        t0_half_spread = sd.spread_bps / 2
        long_buy_s = compute_slippage_bps(sd.asks, long_notional, 'buy', mid_price=mid)
        long_sell_s = compute_slippage_bps(sd.bids, long_notional, 'sell', mid_price=mid)
        depth_walk_buy = max(0, long_buy_s['slippage_bps'] - t0_half_spread)
        depth_walk_sell = max(0, long_sell_s['slippage_bps'] - t0_half_spread)

        # 2) Actual L1 spread at bottom time (long entry) and exit time
        entry_spread_bps = _get_spread_at(sd, bottom_t)
        exit_spread_bps = _get_spread_at(sd, long_exit_t) if long_exit_t else entry_spread_bps

        # 3) Total long slippage = actual half-spread at each moment + depth-walking
        long_entry_slip = entry_spread_bps / 2 + depth_walk_buy
        long_exit_slip = exit_spread_bps / 2 + depth_walk_sell
        long_slip = long_entry_slip + long_exit_slip

        long_net = long_recovery_bps - long_fee - long_slip
        long_dollar = long_net * long_notional / 10000
        long_win = long_dollar > 0

    # Combined
    combined = short_dollar
    if long_taken:
        combined += long_dollar

    # Outcome
    if not long_taken:
        outcome = 'short_only_win' if short_dollar > 0 else 'short_only_lose'
    elif short_dollar > 0 and long_win:
        outcome = 'both_win'
    elif short_dollar > 0 and not long_win:
        outcome = 'short_win_long_lose'
    elif short_dollar <= 0 and long_win:
        outcome = 'short_lose_long_win'
    else:
        outcome = 'both_lose'

    return TradeResult(
        settle_id=sd.settle_id,
        symbol=sd.symbol,
        file_name=sd.file_path.name,
        passes_filters=sd.passes_filters,
        depth_20=depth_20,
        spread_bps=sd.spread_bps,
        fr_abs_bps=sd.fr_abs_bps,
        short_notional=short_notional,
        long_notional=long_notional if long_taken else 0,
        short_gross_bps=GROSS_PNL_BPS,
        short_slip_bps=rt_slip,
        short_fee_bps=short_fee,
        short_net_bps=short_net,
        short_dollar=short_dollar if sd.passes_filters else 0,
        short_win=short_dollar > 0,
        bottom_bps=bottom_bps,
        bottom_t_s=bottom_t / 1000 if bottom_t else None,
        drop_bps=drop_bps,
        long_eligible=long_eligible,
        long_taken=long_taken,
        long_recovery_bps=long_recovery_bps,
        long_fee_bps=long_fee,
        long_slip_bps=long_slip,
        long_net_bps=long_net,
        long_dollar=long_dollar,
        long_win=long_win,
        long_exit_method=long_exit_method,
        long_exit_t_ms=long_exit_t,
        combined_dollar=combined if sd.passes_filters else 0,
        outcome=outcome,
    )


# ═══════════════════════════════════════════════════════════════════════
# FULL BACKTEST — all settlements
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(settlements, long_exit_model=None, long_exit_features=None,
                 use_ml_exit=True, no_long=False, label="backtest"):
    """Run combined backtest on all settlements.

    Returns (list of TradeResult, summary dict).
    """
    print(f"\n  Running {label} on {len(settlements)} settlements...")
    t0 = time.time()

    results = []
    for i, sd in enumerate(settlements):
        tr = simulate_settlement(
            sd,
            long_exit_model=long_exit_model,
            long_exit_features=long_exit_features,
            use_ml_exit=use_ml_exit,
            no_long=no_long,
        )
        results.append(tr)

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(settlements)}] {time.time()-t0:.1f}s")

    # Summary stats
    traded = [r for r in results if r.passes_filters]
    long_trades = [r for r in traded if r.long_taken]

    # Count unique days
    dates = set()
    for r in results:
        parts = r.file_name.split('_')
        if len(parts) >= 2:
            dates.add(parts[1])
    n_days = max(1, len(dates))

    total_short = sum(r.short_dollar for r in traded)
    total_long = sum(r.long_dollar for r in long_trades)
    total_combined = sum(r.combined_dollar for r in traded)

    short_wr = sum(1 for r in traded if r.short_win) / len(traded) * 100 if traded else 0
    long_wr = sum(1 for r in long_trades if r.long_win) / len(long_trades) * 100 if long_trades else 0

    summary = {
        'n_settlements': len(results),
        'n_traded': len(traded),
        'n_skipped': len(results) - len(traded),
        'n_long_taken': len(long_trades),
        'n_long_skipped': len(traded) - len(long_trades),
        'n_days': n_days,
        'short_wr': short_wr,
        'long_wr': long_wr,
        'short_total': total_short,
        'long_total': total_long,
        'combined_total': total_combined,
        'short_per_day': total_short / n_days,
        'long_per_day': total_long / n_days,
        'combined_per_day': total_combined / n_days,
        'avg_short_dollar': np.mean([r.short_dollar for r in traded]) if traded else 0,
        'avg_long_dollar': np.mean([r.long_dollar for r in long_trades]) if long_trades else 0,
        'avg_combined_dollar': np.mean([r.combined_dollar for r in traded]) if traded else 0,
    }

    # Outcome distribution
    outcomes = {}
    for r in traded:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
    summary['outcomes'] = outcomes

    # Long exit method distribution
    methods = {}
    for r in long_trades:
        methods[r.long_exit_method] = methods.get(r.long_exit_method, 0) + 1
    summary['long_exit_methods'] = methods

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  {label.upper()} RESULTS ({n_days} days)")
    print(f"  {'='*60}")
    print(f"  Settlements: {len(results)} total, {len(traded)} traded, {len(results)-len(traded)} skipped")
    print(f"  Short leg:  {len(traded)} trades, WR={short_wr:.0f}%, ${total_short/n_days:.1f}/day")
    print(f"  Long leg:   {len(long_trades)} trades, WR={long_wr:.0f}%, ${total_long/n_days:.1f}/day")
    print(f"  Combined:   ${total_combined/n_days:.1f}/day")
    print(f"  Avg $/trade: short=${summary['avg_short_dollar']:.2f}  "
          f"long=${summary['avg_long_dollar']:.2f}  "
          f"combined=${summary['avg_combined_dollar']:.2f}")

    if outcomes:
        print(f"\n  Outcomes:")
        for outcome, count in sorted(outcomes.items()):
            print(f"    {outcome:25s}: {count:3d} ({count/len(traded)*100:.0f}%)")

    if methods:
        print(f"\n  Long exit methods:")
        for method, count in sorted(methods.items()):
            print(f"    {method:10s}: {count:3d}")

    print(f"  [{time.time()-t0:.1f}s]")

    return results, summary


def compare_strategies(settlements, long_exit_model=None, long_exit_features=None):
    """Run multiple backtest variants for comparison.

    Returns dict of {strategy_name: (results, summary)}.
    """
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")

    strategies = {}

    # 1. Short-only (no long leg)
    print("\n  [1/3] Short-only...")
    results_short, summary_short = run_backtest(
        settlements, use_ml_exit=False, no_long=True, label="short-only"
    )
    strategies['short_only'] = (results_short, summary_short)

    # 2. Short + Long (fixed exit)
    print("\n  [2/3] Short + Long (fixed +20s exit)...")
    results_fixed, summary_fixed = run_backtest(
        settlements, use_ml_exit=False, label="short+long (fixed)"
    )
    strategies['fixed_exit'] = (results_fixed, summary_fixed)

    # 3. Short + Long (ML exit)
    if long_exit_model is not None:
        print("\n  [3/3] Short + Long (ML exit)...")
        results_ml, summary_ml = run_backtest(
            settlements,
            long_exit_model=long_exit_model,
            long_exit_features=long_exit_features,
            use_ml_exit=True,
            label="short+long (ML exit)"
        )
        strategies['ml_exit'] = (results_ml, summary_ml)

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    n_days = summary_short['n_days']
    print(f"\n  {'Strategy':30s}  {'Short $/d':>10s}  {'Long $/d':>10s}  {'Total $/d':>10s}  {'Long WR':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for name, (_, s) in strategies.items():
        print(f"  {name:30s}  ${s['short_per_day']:9.1f}  ${s['long_per_day']:9.1f}  "
              f"${s['combined_per_day']:9.1f}  {s['long_wr']:7.0f}%")

    return strategies
