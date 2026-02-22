# Strategy Research Plan — ML-Based Crypto Futures Trading

**Date:** 2026-02-21
**Goal:** Develop profitable crypto futures strategies for Bybit using ML on 15m+ candles
**Data:** 5 coins × 1143 days (2023-01-01 to 2026-02-16) tick-level trades → 763 features + 120 targets

---

## Constraints

| Constraint | Detail |
|---|---|
| **Fees** | Maker 0.02%, Taker 0.055% → prefer limit orders |
| **No HFT** | 15m candles minimum, no subsecond reaction |
| **No trailing stops** | Requires tick simulation, leads to inflated expectations |
| **No lookahead bias** | Strict WFO, targets use .shift(-N) only |
| **Overfitting prevention** | Walk-Forward Optimization (WFO), purged CV, feature selection |
| **CPU only** | LightGBM, XGBoost, linear models — no GPU |
| **RAM conscious** | Process day-by-day, monitor memory, show progress |

## Critical Lessons from Prior Research

1. **ALL previous bar-based signals failed** tick-level validation (TICK_LEVEL_FINAL_REPORT.md)
   - 201 signals tested, zero predictive power
   - Root cause: simulation bugs in bar-based backtester
   - **Implication:** Must use extremely conservative backtesting with realistic fills
2. **Fees kill short-hold strategies** — 15m/30m holds consistently negative
   - Need 4h+ holding periods OR very high win-rate with limit orders
3. **Cascade MM works** (+1,175% in 88d) but requires tick-level reaction — excluded here
4. **Overfitting is the #1 risk** — Vol Compression Straddle: 99% WR on 30d → 0/7 on 60d

## Approach: Conservative ML with Realistic Backtesting

### Key Design Decisions

1. **Target selection:** Use `tgt_profitable_long_N` / `tgt_profitable_short_N` (fee-aware binary targets)
   - These already incorporate 4 bps round-trip fees
   - Binary classification is more robust than regression
2. **Model:** LightGBM classifier (fast, handles missing values, built-in feature importance)
3. **Feature selection:** Start with top-50 by mutual information, then let model select
4. **WFO protocol:**
   - Train: 60 days, Gap: 5 days (purge), Test: 20 days
   - Roll forward by 20 days
   - Minimum 5 WFO folds before declaring anything
5. **Position sizing:** Fixed fractional (1% risk per trade) or Kelly-based
6. **Entry:** Limit orders at close price (maker fee 0.02%)
7. **Exit:** Limit orders at target (maker) or market stop (taker 0.055%)
8. **Holding periods:** 3, 5, 10 candles (45min to 2.5h at 15m timeframe)

---

## Phase 1: Infrastructure (Day 1)

### 1.1 Generate Development Dataset
- BTCUSDT, 2024-01-01 to 2024-06-30 (6 months, 15m candles)
- ~17,280 candles — enough for 3+ WFO folds
- Save as single parquet for fast iteration

### 1.2 Build WFO Backtesting Framework
- `strategy_ml_wfo.py` — core framework
- Walk-forward engine with purged train/test splits
- Realistic fee model (maker entry, taker stop-loss, maker TP)
- Performance metrics: Sharpe, max DD, win rate, profit factor, avg trade bps
- No trailing stops, no intra-bar assumptions
- Position: enter at next candle open (conservative), exit at target candle close

### 1.3 Feature Pipeline
- Load parquet → drop target cols → handle inf/nan → standardize
- Feature importance via mutual information with target
- Correlation filter (drop features with |corr| > 0.95 to each other)

---

## Phase 2: Strategy Candidates (Days 2-3)

### Strategy A: Direction Classifier (Long/Short/Flat)
- **Target:** `tgt_optimal_action_5` (long/short/flat after fees, 5-bar horizon)
- **Model:** LightGBM multiclass
- **Entry:** When model predicts long or short with probability > threshold
- **Exit:** Fixed 5-bar hold (75 min at 15m)
- **Hypothesis:** With 763 features, ML can find non-linear combinations that predict direction

### Strategy B: Profitable Trade Classifier
- **Target:** `tgt_profitable_long_5` (binary: will long be profitable after fees?)
- **Model:** LightGBM binary classifier, separate models for long and short
- **Entry:** When P(profitable) > calibrated threshold (e.g., 0.6)
- **Exit:** Fixed 5-bar hold
- **Hypothesis:** Easier binary task, model only needs to identify high-probability setups

### Strategy C: Regime-Filtered Trend Following
- **Target:** `tgt_ret_sign_5` (will next 5 bars be up or down?)
- **Filter:** Only trade when `tgt_vol_regime` prediction = high vol
- **Model:** LightGBM binary + vol regime classifier
- **Entry:** Follow predicted direction in high-vol regimes only
- **Exit:** Fixed hold or early exit if regime changes
- **Hypothesis:** Trends are more persistent in high-vol regimes

### Strategy D: Mean-Reversion with Limit Orders
- **Target:** `tgt_mid_reversion_3` (does price return to current level within 3 bars?)
- **Model:** LightGBM binary
- **Entry:** Limit order at current close when P(reversion) > threshold
- **Exit:** TP at reversion level (maker), SL at 2x range (taker)
- **Hypothesis:** Mean-reversion works in low-vol regimes, limit orders minimize fees

### Strategy E: Volatility-Adaptive Sizing
- **Target:** `tgt_pnl_long_bps_5` (regression: expected P&L in bps)
- **Model:** LightGBM regressor
- **Entry:** Direction from sign of prediction, size from magnitude
- **Exit:** Fixed 5-bar hold
- **Hypothesis:** Regression captures both direction and magnitude for optimal sizing

---

## Phase 3: Evaluation & Selection (Day 4)

For each strategy candidate:
1. Run full WFO on BTCUSDT 6-month dev set
2. Compute OOS metrics across all folds
3. Check for stability: are most folds profitable?
4. Check for regime dependence: does it work in both trending and ranging markets?

**Selection criteria (ALL must be met):**
- OOS Sharpe > 1.0 (annualized)
- OOS avg trade > 2 bps (after fees)
- OOS max drawdown < 15%
- At least 60% of WFO folds profitable
- Profit factor > 1.3

---

## Phase 4: Validation (Day 5)

Best 1-2 strategies:
1. Extend to BTCUSDT full 2024 (12 months)
2. Test on ETHUSDT (same period)
3. If still profitable → test on SOL, DOGE, XRP
4. Final: full 2023-2025 backtest on all coins

---

## Phase 5: Robustness (Day 6+)

- Parameter sensitivity analysis
- Feature ablation (remove top features, does it still work?)
- Transaction cost sensitivity (what if fees are 2x?)
- Regime analysis (performance by vol regime, trend regime)
- Monte Carlo: shuffle trade order, bootstrap confidence intervals

---

## File Structure

```
strategy_ml_wfo.py          — WFO backtesting framework
strategy_research_01.py     — Strategy A: Direction classifier
strategy_research_02.py     — Strategy B: Profitable trade classifier
strategy_research_03.py     — Strategy C: Regime-filtered trend
strategy_research_04.py     — Strategy D: Mean-reversion limit orders
strategy_research_05.py     — Strategy E: Vol-adaptive sizing
STRATEGY_RESEARCH.md        — This plan (kept updated)
STRATEGY_FINDINGS_*.md      — Results for each strategy
```

---

## Current Status

- [x] 763 features + 120 targets implemented
- [x] Tick data available: 5 coins × 1143 days
- [ ] Generate dev dataset (BTCUSDT 6mo, 15m)
- [ ] Build WFO framework
- [ ] Test Strategy A-E
- [ ] Select best candidates
- [ ] Multi-coin validation
- [ ] Full period backtest
