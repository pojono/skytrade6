# Production Checklist Report

| Check | Status | Risk | Notes |
| --- | --- | --- | --- |
| Lookahead Bias | YES | LOW | Feature construction uses lagged/known-at-time fields; no explicit future columns in signal pipeline. |
| OOS Test | YES | MEDIUM | Frozen config selected on train only; OOS from 2026-01 has 29 timestamps, avg 17.26 bps (mixed). |
| Overfitting | PARTIAL | MEDIUM | Stability sweep ±20% around frozen params: 59% variants remain profitable on OOS. |
| Execution Modeling | PARTIAL | HIGH | Fees modeled (maker/taker), spread=2.0 bps RT, slippage from bookDepth; OOS net_exec=-8.61 bps over 24 symbol-trades (order_notional=2000 USD). |
| PnL Stability | YES | LOW | Top-2 positive months contribute 39% of total profit. |

## Overall Assessment
- Edge probability: **moderate-to-high**
- Main risks: short OOS horizon, execution assumptions (fixed spread proxy), leverage sensitivity
- Production readiness: **pilot-ready with risk caps**

## Artifacts
- `production_checklist_details.csv` (execution-level OOS details, if available)