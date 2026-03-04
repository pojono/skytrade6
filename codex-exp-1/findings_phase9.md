# Findings Phase 9: Pre-Trade Meta Filter

This phase adds a lightweight pre-trade classifier layer on top of the frozen 3-symbol strategy.

## Objective

The goal was to reject weaker signals before the paper-trading engine sees them, using only features already known at entry time.

Implemented tools:

- `analyze_trade_outcomes.py`: compares winner vs loser feature behavior on exported candidate trades.
- `train_simple_meta_filter.py`: fits a simple threshold-based meta-filter on train months and evaluates it on the last 2 test months.
- `paper_trade_filtered.py`: applies the selected rule, exports filtered trades, and replays the existing conservative live-style simulation.

## Outcome Analysis

Source: `out/trade_outcome_analysis.md`

- Trades analyzed: `35,928`
- Winner rate on raw candidate trades: `54.09%`
- Mean net edge on raw candidate trades: `3.6648 bps`

By symbol:

- `CRVUSDT`: `54.86%` winners, `4.4967 bps`
- `GALAUSDT`: `54.99%` winners, `2.7094 bps`
- `SEIUSDT`: `43.82%` winners, `-0.3899 bps`

Important pattern:

- `SEIUSDT` remains the weakest leg even before execution replay.
- Winners tend to have larger absolute spread, but also higher spread velocity.
- The simple score alone is not sufficient; the best filter came from combining higher score with stricter spread size.

## Selected Meta Filter

Source: `out/meta_filter_report.md`

Train months: `2025-08` through `2026-01`
Test months: `2026-02`, `2026-03`

Selected threshold rule:

```json
{
  "max_velocity": 12.0,
  "min_carry": 2.0,
  "min_ls": 0.15,
  "min_oi": 5.0,
  "min_score": 8.0,
  "min_spread_abs": 12.0,
  "sei_score_extra": 10.0
}
```

Raw candidate-trade performance after filtering:

- Train trades: `6,734`
- Train avg net: `2.5983 bps`
- Train win rate: `52.24%`
- Test trades: `2,914`
- Test avg net: `6.0858 bps`
- Test win rate: `50.24%`

The raw candidate-trade win rate is not the same as the final paper-trading fill win rate because the live-style engine still applies caps, selector ordering, and symbol conflicts.

## Filtered Paper Replay

Source: `out/paper_report_v3_meta.md`

The same conservative live-style replay was used:

- `10%` allocation
- `1` open position
- `1` open position per symbol
- selector mode `spread`
- daily cap `3`
- daily loss stop `1%`
- monthly loss stop `3%`
- base fee `6 bps`
- extra slippage `1 bps`
- spread slippage coeff `0.10`
- velocity slippage coeff `0.05`

Before meta-filter (baseline `out/paper_report_v3_live.md`):

- Filled trades: `1,421`
- Final capital: `$105,358.62`
- Total PnL: `$5,358.62`
- Avg net edge per fill: `3.6742 bps`
- Win rate: `58.69%`

After meta-filter (`out/paper_report_v3_meta.md`):

- Filled trades: `956`
- Final capital: `$104,741.42`
- Total PnL: `$4,741.42`
- Avg net edge per fill: `4.8466 bps`
- Win rate: `59.73%`

Net effect:

- Trade count down `32.72%`
- Total dollars down `11.52%`
- Avg edge per fill up `31.91%`
- Win rate up `1.04` percentage points

Interpretation:

- The meta-filter is improving trade quality.
- At the current low-capital deployment, reducing turnover also reduces total account PnL.
- This is a quality-over-quantity filter, not a pure PnL maximizer at `10%` allocation.

## Symbol-Level Effect

Filtered paper replay contribution:

- `CRVUSDT`: `560` filled trades, `$3,388.17`
- `GALAUSDT`: `324` filled trades, `$1,357.92`
- `SEIUSDT`: `72` filled trades, `-$4.67`

The filter sharply reduced `SEIUSDT` activity but did not make it clearly additive. That keeps `SEIUSDT` as the weak point.

## Practical Conclusion

The pre-trade classifier works, but it solves trade quality more than account-level return:

- Better average edge per fill
- Slightly better fill win rate
- Less total turnover
- Slightly lower absolute PnL at the current conservative allocation

Current best interpretation:

- Use the meta-filter when execution quality matters most.
- Keep the unfiltered 3-symbol live variant when maximizing absolute dollars at `10%` allocation.
- The next refinement should likely be symbol-specific: either a stricter `SEIUSDT` rule or removing `SEIUSDT` from the filtered sleeve.
