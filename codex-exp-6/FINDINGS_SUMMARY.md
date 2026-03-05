# Codex Exp 6 Findings Summary

This document consolidates the main research outcomes for `codex-exp-6` up to the current state.

## 1) Market Structure Baseline

Source: `out/market_structure_report.md`

- Universe (`2026-01-02` to `2026-03-02`, Binance): 60 symbols
- 1m pairwise return correlation: ~`0.3766`
- Strong synchronized moves are frequent across horizons
- Best raw strategy family:
  - `breadth_trend` at `240m`
  - gross `13.01` bps
  - maker net `+5.01` bps
  - taker net `-6.99` bps

Conclusion:

- There is real market-wide structure.
- The only consistent candidate was slow-horizon breadth trend under maker assumptions.

## 2) Strict Metrics Filters (OI + Positioning)

Source: `out/market_structure_report_strict_oi_1pct_40.md`, `out/market_structure_report_strict_oi_2pct_40.md`

- On the strict filtered runs (40 symbols):
  - `breadth_trend 240m` remained the best practical candidate.
  - Filtered cross-sectional variants had tiny sample counts or unstable extremes.
- Walk-forward for `breadth_trend 240m` stayed unstable:
  - one strongly losing split
  - one strongly winning split
  - one near-flat split

Conclusion:

- Metrics filters did not convert this into a stable always-on strategy.

## 3) Execution-Realism Stress for the 240m Breadth Candidate

Source: `out/breadth_trend_execution_report_full_exec_100.md`

- Window: `2026-01-02` to `2026-03-02`
- Universe: 100 symbols
- Assumptions:
  - maker fee: 4 bps/side
  - queue miss: 35%
  - partial fill: 75%
  - adverse selection: 1 bps
- Aggregate:
  - gross edge `15.76` bps
  - expected net edge `+3.30` bps
- Walk-forward:
  - average expected net `-0.61` bps
  - worst split `-17.85` bps

Conclusion:

- Aggregate edge exists under this maker model.
- Stability across time remains the blocking issue.

## 4) Rule-Based Regime Filters (Trade/No-Trade)

Source: `out/regime_filter_report_full_regime_100.md`

- Best stabilizer found:
  - low short-term volatility (`rv_60 <= q50`)
  - worst split improved from `-18.86` bps (all signals) to `-16.88` bps
  - but still negative
- Some filters increased average expectancy but worsened worst-split behavior.

Conclusion:

- Rule-based gating helps slightly.
- It does not fix regime fragility.

## 5) ML Trade Gate (Shorter vs Longer History)

### 5a) Shorter Window (encouraging)

Source: `out/ml_trade_gate_report_full_ml_100.md`

- Window: `2026-01-02` to `2026-03-02`
- Universe: 99 symbols
- ML gate improved out-of-sample metrics on available folds:
  - avg uplift `+1.01` bps
  - worst traded split `+1.84` bps

### 5b) Longer Window (fragility exposed)

Source: `out/ml_trade_gate_report_full_ml_220d_80_s6.md`

- Window: `2025-07-26` to `2026-03-02`
- Universe: 80 symbols
- With `prob_threshold=0.68`:
  - avg base expected net `-3.18` bps
  - avg traded expected net `-47.40` bps
  - avg uplift `-53.63` bps
  - worst traded split `-116.41` bps
  - trade share only `3.43%`

Conclusion:

- ML gate looked promising on shorter data.
- Longer history showed severe overfit/regime mismatch at the chosen threshold.

## 6) Lead-Lag Pivot Scan

Source: `out/lead_lag_report_smoke_ll.md`

- Initial 14-day smoke:
  - average absolute best lag correlation: `0.0293`
  - mixed leadership between Binance and Bybit

Conclusion:

- Measurable but weak in current form.
- Not yet stronger than breadth-based direction.

## Current Status

- Closest candidate: `240m` breadth trend with maker-oriented execution.
- Not yet production-ready due to split/regime instability.
- Taker-surviving edge not found.
- Conditional strategy framing is necessary (not always-on).

## Suggested Next Step

Prioritize robustness over headline edge:

1. Re-run ML gate and rule-gate selection with strict constraints:
   - minimum trades per fold
   - require all test folds represented
   - optimize by worst-fold first, then average
2. Promote only regimes with:
   - positive worst-fold expected net
   - adequate trade count and coverage
   - stability under small threshold perturbations
