# PLAN v43: Fresh OOS Validation & New Strategy Development

**Date:** 2026-02-20
**Goal:** Develop a profitable trading strategy for Bybit crypto futures
**Fees:** maker 0.02%, taker 0.055% (prefer limit orders)
**Constraint:** No HFT (no sub-second reaction required)

---

## Situation Assessment

### What We Have
- **Data:** 276 GB raw + 359 GB parquet
  - Ticker + Liquidations: 5 symbols × May 2025 – Feb 2026 (~9.5 months)
  - Bybit futures trades: 5 symbols × Jan 2023 – Feb 2026
  - Binance klines/metrics: 3 symbols
- **Previous research:** 56+ experiments, 201 signal types discovered
- **Best strategies (in-sample May-Aug 2025):**
  1. Cascade MM with trailing stop: 92% WR, +10.4 bps/trade, Sharpe 627
  2. Microstructure MR (price-only): 89% WR, +14.7 bps DOGE, Sharpe 476
  3. Liq Acceleration: 86% WR, +10.3 bps DOGE, Sharpe 517
  4. VWAP Deviation (price-only): 85% WR DOGE, massive volume
  5. Vol Spike Fade (price-only): 88% WR DOGE, +14.7 bps

### Critical Gap
ALL previous OOS tests used Jul-Aug 2025 data (28 days). We now have **6+ months of completely unseen data** (Aug 2025 – Feb 2026). This is the ultimate test.

### Key Risks
- **Overfitting:** 201 signals tested = massive multiple testing problem
- **Regime change:** May-Aug 2025 was a specific market regime
- **Lookahead bias:** Must use ONLY parameters from in-sample period

---

## Plan

### ITERATION 1: Cascade MM True OOS (Aug 2025 – Feb 2026)
- **Why first:** Strongest theoretical basis (liquidation cascades = forced flow = structural edge)
- **Test:** Use EXACT params from v42j (trail 3/2, offset 0.15%, TP 0.15%, SL 0.50%)
- **P95 threshold:** Computed from May-Jul 2025 ONLY (no peeking)
- **Symbols:** ETH + SOL first, then DOGE + XRP
- **Period:** Aug 8 – Feb 19 (6+ months, completely unseen)
- **Success criteria:** Positive total return, >70% WR, positive Sharpe

### ITERATION 2: Microstructure MR True OOS
- **Why:** Price-only = deployable anywhere, no liquidation data dependency
- **Test:** sigma=2, cooldown=60s, trail 3/2 (from v42t)
- **Same period:** Aug 8 – Feb 19

### ITERATION 3: Funding Cycle + Vol Spike Combo (NEW IDEA)
- **Hypothesis:** Post-funding (00:00, 08:00, 16:00 UTC) vol spike is 14-21% higher.
  Vol spikes predict 6x next-bar volatility. Combine: enter MR trades in the
  30-min post-funding window when vol spike detected.
- **Why novel:** Exploits two proven structural effects simultaneously
- **Advantage:** Predictable timing (every 8h) + structural trigger = no HFT needed

### ITERATION 4: LS Ratio Extended OOS
- **Why:** IC=0.20 is extraordinary, but only tested on 31 days
- **Test:** Download extended Binance metrics, test on 6+ months

### ITERATION 5: Best candidates → broader validation
- Take winners from iterations 1-4
- Test on all 5 symbols, full date range
- Walk-forward with rolling windows

---

## Anti-Overfitting Protocol
1. **Fix all parameters BEFORE looking at OOS data**
2. **No re-optimization on OOS** — if it fails, it fails
3. **Report ALL results** including failures
4. **Multiple testing correction:** Bonferroni for significance
5. **Rolling window stability:** Must be positive in >60% of 30-day windows
6. **Cross-symbol consistency:** Must work on ≥3 of 5 symbols

## Anti-Lookahead Protocol
1. P95 thresholds computed from training period only
2. Rolling features use ONLY past data (no future bars)
3. Entry at cascade END (not start or before)
4. Fill simulation: limit order must touch bar's high/low
5. No same-bar entry+exit

---
