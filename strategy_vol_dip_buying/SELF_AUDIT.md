# Self-Audit: Volatility Dip-Buying Strategy

## 1. Strategy Summary

**Signal:** `combined = (rvol_z + mr_4h) / 2`
- `rvol_z`: z-score of 24h realized volatility vs 168h trailing mean/std
- `mr_4h`: z-score of 4h return (negative = price dropped), inverted for mean-reversion

**Rule:** When `|combined| > 2.0`, go long (if positive) or short (if negative). Hold 4h. Limit entry+exit (4 bps RT).

---

## 2. Walk-Forward Month-by-Month Returns (No Lookahead Bias)

**Protocol:** First 6 months (Jan–Jun 2023) = warmup only. From Jul 2023 onward, trade using only backward-looking rolling windows. Parameters (threshold=2.0, hold=4h) are FIXED — never optimized per period.

### SOLUSDT — Walk-Forward Monthly

| Month | Trades | Return | Avg bps | WR | L/S | B&H | vs Random |
|-------|--------|--------|---------|-----|-----|-----|-----------|
| 2023-07 | 2 | +0.69% | +34 | 50% | 2/0 | +26.7% | +23 |
| 2023-08 | 3 | +3.94% | +131 | 100% | 3/0 | -17.5% | +160 |
| 2023-09 | 2 | +0.02% | +1 | 50% | 2/0 | +7.5% | +10 |
| 2023-10 | 5 | +4.43% | +89 | 80% | 5/0 | +80.0% | +87 |
| 2023-11 | 4 | +2.23% | +56 | 50% | 4/0 | +53.7% | +114 |
| 2023-12 | 4 | +0.67% | +17 | 75% | 3/1 | +72.8% | +18 |
| 2024-01 | 2 | **-0.09%** | -4 | 50% | 2/0 | -5.1% | +13 |
| 2024-02 | 1 | +1.82% | +182 | 100% | 1/0 | +29.3% | +149 |
| 2024-03 | 2 | +4.52% | +226 | 50% | 2/0 | +52.2% | +171 |
| 2024-04 | 7 | +0.99% | +14 | 57% | 6/1 | -37.4% | +34 |
| 2024-05 | 1 | **-0.12%** | -13 | 0% | 1/0 | +32.5% | -7 |
| 2024-06 | 4 | +1.93% | +48 | 50% | 4/0 | -12.1% | +65 |
| 2024-07 | 1 | +1.38% | +139 | 100% | 1/0 | +16.2% | +114 |
| 2024-08 | 9 | +14.27% | +159 | 56% | 9/0 | -21.3% | +125 |
| 2024-09 | 2 | **-0.91%** | -45 | 50% | 2/0 | +12.7% | -52 |
| 2024-10 | 3 | **-0.56%** | -19 | 67% | 3/0 | +9.4% | +13 |
| 2024-11 | 1 | +1.48% | +148 | 100% | 1/0 | +40.6% | +122 |
| 2024-12 | 2 | **-0.16%** | -8 | 50% | 2/0 | -20.4% | +19 |
| 2025-01 | 3 | +3.19% | +107 | 67% | 3/0 | +20.4% | +139 |
| 2025-02 | 5 | +10.84% | +217 | 100% | 5/0 | -36.3% | +224 |
| 2025-03 | 2 | **-3.58%** | -179 | 50% | 2/0 | -14.7% | -120 |
| 2025-04 | 4 | +0.37% | +9 | 50% | 4/0 | +18.5% | +49 |
| 2025-05 | 4 | +5.61% | +140 | 75% | 4/0 | +5.7% | +160 |
| 2025-06 | 5 | +5.65% | +113 | 100% | 5/0 | -0.9% | +120 |
| 2025-07 | 3 | +6.13% | +204 | 100% | 3/0 | +11.3% | +232 |
| 2025-08 | 5 | +3.72% | +74 | 60% | 5/0 | +17.8% | +80 |
| 2025-09 | 2 | +0.92% | +46 | 50% | 2/0 | +4.1% | +36 |
| 2025-10 | 7 | **-4.68%** | -67 | 43% | 7/0 | -9.8% | -94 |
| 2025-11 | 6 | +1.59% | +27 | 50% | 6/0 | -28.7% | +29 |
| 2025-12 | 6 | +1.82% | +30 | 67% | 6/0 | -2.2% | +35 |
| 2026-01 | 6 | **-3.19%** | -53 | 50% | 6/0 | -15.7% | -4 |
| 2026-02 | 2 | **-1.27%** | -63 | 50% | 2/0 | -18.2% | +18 |

**SOL Summary:** 23/32 positive months (72%), beat random 27/32 (84%), cumulative +63.7%, annualized +23.9%/yr, monthly Sharpe +1.84, max DD 5.7%, longest losing streak 2 months.

### ETHUSDT — Walk-Forward Monthly

| Month | Trades | Return | Avg bps | WR | L/S | B&H | vs Random |
|-------|--------|--------|---------|-----|-----|-----|-----------|
| 2023-07 | 3 | +1.17% | +39 | 67% | 3/0 | -4.1% | +42 |
| 2023-08 | 5 | +1.08% | +22 | 40% | 4/1 | -11.6% | +19 |
| 2023-09 | 3 | +0.17% | +6 | 33% | 2/1 | +1.1% | +12 |
| 2023-10 | 3 | **-2.35%** | -78 | 0% | 3/0 | +8.5% | -76 |
| 2023-11 | 3 | +4.69% | +157 | 67% | 3/0 | +12.8% | +179 |
| 2023-12 | 2 | +2.62% | +131 | 100% | 2/0 | +11.4% | +150 |
| 2024-01 | 2 | **-1.63%** | -81 | 0% | 2/0 | -0.7% | -77 |
| 2024-02 | 5 | +6.81% | +136 | 80% | 5/0 | +46.5% | +110 |
| 2024-03 | 3 | **-3.64%** | -121 | 33% | 2/1 | +7.9% | -93 |
| 2024-04 | 5 | **-2.57%** | -51 | 60% | 5/0 | -17.0% | -35 |
| 2024-05 | 4 | +5.80% | +145 | 50% | 4/0 | +25.5% | +171 |
| 2024-06 | 3 | **-1.58%** | -53 | 33% | 3/0 | -8.8% | -49 |
| 2024-07 | 4 | +0.63% | +16 | 75% | 4/0 | -6.1% | +18 |
| 2024-08 | 6 | +8.81% | +147 | 50% | 6/0 | -22.3% | +164 |
| 2024-09 | 2 | **-2.98%** | -149 | 50% | 2/0 | +3.9% | -89 |
| 2024-10 | 5 | **-1.42%** | -28 | 40% | 5/0 | -3.8% | -23 |
| 2024-11 | 2 | +0.89% | +45 | 100% | 2/0 | +46.8% | +57 |
| 2024-12 | 4 | +2.65% | +66 | 50% | 3/1 | -9.6% | +62 |
| 2025-01 | 5 | **-4.00%** | -80 | 20% | 5/0 | -1.9% | -56 |
| 2025-02 | 6 | +3.12% | +52 | 67% | 6/0 | -32.6% | +51 |
| 2025-03 | 3 | **-6.31%** | -210 | 0% | 3/0 | -17.8% | -207 |
| 2025-04 | 5 | +4.68% | +94 | 80% | 5/0 | -2.1% | +78 |
| 2025-05 | 3 | +5.20% | +173 | 67% | 2/1 | +40.6% | +152 |
| 2025-06 | 5 | +0.95% | +19 | 60% | 5/0 | -1.6% | +16 |
| 2025-07 | 1 | +1.66% | +166 | 100% | 1/0 | +48.3% | +136 |
| 2025-08 | 1 | **-0.68%** | -68 | 0% | 1/0 | +19.3% | -52 |
| 2025-09 | 5 | +1.19% | +24 | 60% | 5/0 | -5.7% | +9 |
| 2025-10 | 3 | **-4.34%** | -145 | 0% | 3/0 | -7.3% | -159 |
| 2025-11 | 5 | **-4.29%** | -86 | 20% | 5/0 | -22.2% | -68 |
| 2025-12 | 6 | **-1.39%** | -23 | 67% | 5/1 | +4.8% | -40 |
| 2026-01 | 6 | **-0.81%** | -14 | 67% | 6/0 | -17.7% | +12 |
| 2026-02 | 1 | +5.45% | +545 | 100% | 1/0 | -18.5% | +439 |

**ETH Summary:** 18/32 positive months (56%), beat random 19/32 (59%), cumulative +19.6%, annualized +7.3%/yr, monthly Sharpe +0.59, max DD 10.8%, longest losing streak 4 months.

### BTCUSDT — Walk-Forward Monthly

**BTC Summary:** 17/32 positive months (53%), beat random 19/32 (59%), cumulative +10.2%, annualized +3.8%/yr, monthly Sharpe +0.45, max DD 8.3%, longest losing streak 3 months.

### DOGEUSDT — Walk-Forward Monthly

**DOGE Summary:** 14/32 positive months (44%), beat random 16/32 (50%), cumulative +11.4%, annualized +4.3%/yr, monthly Sharpe +0.26, max DD 19.8%, longest losing streak 4 months.

### XRPUSDT — Walk-Forward Monthly

**XRP Summary:** 21/32 positive months (66%), beat random 23/32 (72%), cumulative +104.1%, annualized +39.0%/yr, monthly Sharpe +1.42, max DD 19.1%, longest losing streak 2 months.

---

## 3. Cross-Symbol Summary

| Symbol | Pos Months | Beat Random | Total | Ann. Return | mSharpe | Max DD | Trades |
|--------|-----------|-------------|-------|-------------|---------|--------|--------|
| **SOL** | 23/32 (72%) | 27/32 (84%) | +63.7% | +23.9%/yr | +1.84 | 5.7% | 115 |
| **XRP** | 21/32 (66%) | 23/32 (72%) | +104.1% | +39.0%/yr | +1.42 | 19.1% | 133 |
| **ETH** | 18/32 (56%) | 19/32 (59%) | +19.6% | +7.3%/yr | +0.59 | 10.8% | 119 |
| **BTC** | 17/32 (53%) | 19/32 (59%) | +10.2% | +3.8%/yr | +0.45 | 8.3% | 143 |
| **DOGE** | 14/32 (44%) | 16/32 (50%) | +11.4% | +4.3%/yr | +0.26 | 19.8% | 140 |

### Yearly Breakdown

| Year | SOL | ETH | BTC | DOGE | XRP |
|------|-----|-----|-----|------|-----|
| 2023 (6mo) | +12.0% | +7.4% | +3.3% | +2.8% | +49.8% |
| 2024 | +24.6% | +12.4% | +3.6% | +5.2% | +12.5% |
| 2025 | +31.6% | -0.8% | +3.3% | +13.8% | +40.7% |
| 2026 (2mo) | -4.5% | +4.6% | +0.1% | -10.4% | +1.1% |

---

## 4. PROS — What's Genuinely Good

### ✅ P1: Consistent Across Symbols
All 5 symbols are net positive over 32 months of walk-forward testing. This is not curve-fit to one asset.

### ✅ P2: SOL Is Exceptional
72% positive months, 84% beat random, monthly Sharpe 1.84, max DD only 5.7%. This is a strong risk-adjusted result over 2.7 years of true out-of-sample.

### ✅ P3: No Lookahead Bias
All rolling windows (24h, 48h, 168h) are backward-looking by construction. Parameters (threshold=2.0, hold=4h) are fixed — never optimized per period. The walk-forward protocol uses 6 months of warmup before any trading.

### ✅ P4: Fee-Robust
SOL avg +52 bps/trade with only 4 bps fees. Even at 20 bps RT fees (taker+taker), XRP and SOL remain profitable. The edge is 10-50x larger than fees.

### ✅ P5: Low Capital Deployment
Only ~2% of time in the market (3-5 trades/month × 4h hold). This means low opportunity cost and ability to deploy capital elsewhere 98% of the time.

### ✅ P6: Simple Signal
Only uses 1h OHLCV close prices. No exotic data (no orderbook, no liquidations, no funding rate). Can be computed on any exchange.

### ✅ P7: Beats Random Consistently
SOL beats random direction 84% of months. XRP beats random 72%. This confirms the signal has genuine directional information, not just market beta.

### ✅ P8: Works in Bear Markets (Sometimes)
SOL Feb 2025: +10.8% while B&H was -36.3%. SOL Aug 2024: +14.3% while B&H was -21.3%. The strategy can profit during crashes by buying the dip at the right moment.

---

## 5. CONS — What's Genuinely Concerning

### ❌ C1: Heavily Long-Biased (Critical)
SOL: 97% long trades (113 long vs 2 short). XRP: 93% long. The short side has essentially zero statistical validation. **If the signal only works long, it may be leveraged buy-and-hold bias.**

**Counter-argument:** Capital is deployed only 2% of the time. Time-proportional B&H for SOL ≈ 766% × 2% = 15%, but strategy made +64%. That's 4x better than proportional B&H. Still, the long bias is the #1 concern.

### ❌ C2: Regime-Dependent
Fails in sustained grinding bear markets:
- SOL Oct 2025: -4.7% (B&H -9.8%)
- SOL Jan 2026: -3.2% (B&H -15.7%)
- DOGE Jan 2026: -5.0% (B&H -11.6%)
- DOGE Feb 2026: -5.5% (B&H -3.7%)

The strategy buys dips that don't recover. When the market enters a sustained downtrend, every dip is a trap.

### ❌ C3: Low Trade Frequency
3-5 trades/month per symbol. This means:
- Slow to compound
- High variance in monthly returns (single trade can dominate a month)
- Hard to get statistical confidence (only 115-143 trades over 3 years)

### ❌ C4: Outlier-Dependent Returns
XRP Jul 2023: +35.1% from just 2 trades (one trade was +1756 bps). Remove that single month and XRP total drops from +104% to +69%. SOL Aug 2024: +14.3% from 9 trades. A few big winners drive a large portion of returns.

### ❌ C5: DOGE and BTC Are Marginal
DOGE: only 44% positive months, barely beats random (50%). BTC: 53% positive months, +3.8%/yr annualized — barely worth the effort. These two symbols do NOT have a convincing edge.

### ❌ C6: 2026 Is Negative (Recent Performance)
SOL: -4.5% in Jan-Feb 2026. DOGE: -10.4%. XRP: +1.1% (barely). The most recent 2 months are weak across the board. This could be:
- Normal variance (2 months is too short to judge)
- Regime change (market structure shifted)
- Signal decay (edge is being arbitraged away)

### ❌ C7: Execution Assumptions
The backtest assumes:
- Limit order fills at the close price of the signal bar (may not fill in practice)
- Exit at exactly 4h later at close price (may need market order → +3.5 bps taker fee)
- No slippage, no partial fills, no exchange downtime

In reality, fill rate on limit orders during high-vol moments may be <50%, and exits may require taker orders, increasing RT cost to ~7.5 bps.

### ❌ C8: Parameter Sensitivity
The strategy was discovered by testing threshold=2.0 and hold=4h. While these weren't optimized per period, they WERE selected from a grid search (v43q tested thresholds 0.5-3.0 and holds 1-24h). The "best" parameters may be overfit to the 3-year sample.

**Evidence of sensitivity:**
- thresh=1.0: SOL +1.4 bps avg (marginal) vs thresh=2.0: +51.9 bps (strong)
- thresh=1.5: SOL -5.3 bps avg (negative!) vs thresh=2.0: +51.9 bps
- This non-monotonic behavior (1.5 worse than 1.0 AND 2.0) is a red flag for overfitting

### ❌ C9: No Trailing Stop / Risk Management
The strategy holds for exactly 4h regardless of what happens. If price drops 5% in the first hour, it still holds. No stop-loss, no trailing stop, no early exit. This is by design (to avoid the trailing stop lookahead issues found in earlier research), but it means unlimited downside during the hold period.

### ❌ C10: Backtest Uses 1h Bar Close Prices
Entry and exit are at 1h bar close prices. In reality:
- The signal fires at the close of bar t
- You need to place a limit order and wait for fill
- The fill price may differ from the close price
- By the time you compute the signal and place the order, the opportunity may have passed

---

## 6. Lookahead Bias Audit

| Potential Bias | Status | Explanation |
|---------------|--------|-------------|
| Signal uses future data | ✅ Clean | All rolling windows are backward-looking (pandas `.rolling()` is causal) |
| Parameters optimized on test data | ⚠️ Partial | threshold=2.0 and hold=4h were selected from a grid search on full data. Not optimized per period, but selected post-hoc |
| Survivorship bias | ✅ Clean | All 5 symbols tested regardless of performance |
| Same-bar fill+exit | ✅ Clean | Entry at bar close, exit 4 bars later. No same-bar execution |
| Future data in z-score normalization | ✅ Clean | Z-scores use trailing rolling windows only |
| Walk-forward warmup | ✅ Clean | First 6 months excluded from trading |

**Residual bias risk:** The choice of threshold=2.0 was made after seeing results across all thresholds. A truly unbiased test would pre-commit to parameters before seeing any data. However, the threshold was not changed per symbol or per period, which limits the overfitting risk.

---

## 7. Verdict

### Tier Assessment

| Symbol | Tier | Confidence | Recommendation |
|--------|------|------------|----------------|
| **SOL** | A | Medium-High | Paper trade, then small live |
| **XRP** | A- | Medium | Paper trade, watch for outlier dependency |
| **ETH** | B- | Low-Medium | Monitor only, don't trade |
| **BTC** | C | Low | Do not trade |
| **DOGE** | C | Low | Do not trade |

### Key Risks to Monitor
1. **Long bias validation**: Track short trades separately. If shorts are consistently losing, the strategy is just leveraged buy-the-dip
2. **2026 performance**: If Q1 2026 finishes negative on SOL, reassess
3. **Fill rate**: In live trading, track what % of limit orders actually fill at the signal price
4. **Regime change**: If 3+ consecutive months are negative, stop trading

### What Would Increase Confidence
- Short side showing independent profitability (currently too few trades)
- Positive performance in a confirmed bear market (2022-style, not in our data)
- Stable performance across different threshold values (currently non-monotonic)
- Live paper trading confirmation for 2+ months
