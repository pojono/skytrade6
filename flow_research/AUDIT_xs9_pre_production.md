# Pre-Production Audit: XS-9 Combined Strategy

**Date:** 2026-03-07  
**Strategy:** Vol Dip-Buying + Fragility Overlay (XS-9)  
**Auditor:** Automated code + data audit  
**Scripts audited:** `xs9_combined_backtest.py`, `top50_validation.py`, `walkforward_audit.py`, `xs8c_deep_dive.py`

---

## Summary Table

| Check              | Status    | Risk       | Notes                                                              |
|--------------------|-----------|------------|--------------------------------------------------------------------|
| Lookahead Bias     | PARTIAL   | **HIGH**   | VDB signal clean, but fragility coefficients trained on overlapping data |
| OOS Test           | NO        | **HIGH**   | No true OOS — all 8 months used for both fitting and evaluation    |
| Overfitting        | PARTIAL   | **HIGH**   | VDB params non-monotonic; overlay sizing fit on in-sample quintile PnL |
| Execution Modeling | PARTIAL   | **MEDIUM** | 4bps maker fees modeled; slippage, partial fills, spread not modeled |
| PnL Stability      | NO        | **CRITICAL** | Top-5 trades = 106% of PnL; remove them → strategy is negative    |

---

## 1. Lookahead Bias

**Status: PARTIAL | Risk: HIGH**

### VDB Signal (clean ✅)

The vol dip-buying signal itself has **no lookahead bias**:

- `rvol_z`: rolling(24) std → rolling(168) z-score. All backward-looking.
- `mr_4h`: rolling(4) sum → rolling(48) z-score. All backward-looking.
- `combined = (rvol_z + mr_4h) / 2`. Pure function of past prices.
- Parameters (threshold=2.0, hold=4h) are fixed, not adapted per period.
- Entry at bar[i] close, exit at bar[i+4] close. No same-bar execution.

### Fragility Overlay (leakage ⚠️)

Three sources of data leakage in the overlay:

**a) Coefficient leakage.** The fragility score formula `frag_score = -4.62 * crowd_oi - 0.95 * pca_var1` uses coefficients from XS-8 LogReg trained on **the first 60% of Jul 2025–Feb 2026 data** (≈Jul–Oct 2025). The XS-9 backtest runs on **all** Jul 2025–Feb 2026. The Jul–Oct period is used for both coefficient fitting and strategy evaluation.

**b) Symbol selection bias.** The 9 Tier-A symbols were selected by `top50_validation.py` based on full-period walk-forward performance (Jan 2023–Feb 2026). The XS-9 8-month window is a subset of this selection period. Symbols were chosen *because* they were good, then re-tested on overlapping data.

**c) Sizing rule leakage.** The multipliers (Q5→50%, Q4→75%) were designed *after* observing that Q5 baseline trades average -75bp. This is explicit in-sample optimization of the overlay parameters.

### Mitigating factors

- VDB signal uses only backward-looking rolling windows (no fitting)
- Fragility quintiles use expanding percentiles (no future data in rank computation)
- LogReg coefficients are simple (2 features, linear) — limited capacity to overfit

---

## 2. Honest Out-of-Sample Test

**Status: NO | Risk: HIGH**

### VDB (original strategy — has OOS ✅)

The original VDB validation (`walkforward_audit.py`, `top50_validation.py`) covers **32 months** (Jul 2023–Feb 2026) of walk-forward testing with 6-month warmup. Parameters were fixed before the walk-forward period. This is a credible OOS test for the base strategy.

SOL: 23/32 positive months (72%), Sharpe 1.84, +63.7% total.

### XS-9 (combined — NO OOS ❌)

The XS-9 combined backtest covers only **8 months** (Jul 2025–Feb 2026), which is the *same* period used to:

1. Train the XS-8 fragility coefficients (first 60%)
2. Validate the XS-8 fragility model (last 40%)
3. Observe the quintile PnL breakdown that motivated the sizing rules
4. Run the combined backtest

**There is no data that was not used in some step of model development.** The Sharpe improvement (+0.74 → +1.03) has zero honest OOS validation.

### What would be needed

A true OOS test requires:
- **New time period** (e.g., Mar–Jun 2026) where neither VDB params nor fragility coefficients were fit
- **Or** a strict temporal split: fit fragility on Jul–Oct 2025, then freeze everything and test on Nov 2025–Feb 2026 only (4 months, ~146 trades)

---

## 3. Overfitting / Parameter Stability

**Status: PARTIAL | Risk: HIGH**

### VDB parameters (unstable ⚠️)

Parameter sensitivity on the 8-month XS-9 window (9 symbols):

**Threshold sweep** (hold=4h fixed):

| Threshold | N trades | Avg bps | Total % |
|-----------|----------|---------|---------|
| 1.5       | 838      | +7.0    | +59.0%  |
| 1.7       | 540      | +8.1    | +43.9%  |
| **1.8**   | **439**  | **+13.3** | **+58.3%** |
| **2.0**   | **292**  | **+9.3**  | **+27.3%** |
| 2.2       | 190      | +1.1    | +2.1%   |
| 2.5       | 109      | -12.9   | -14.1%  |
| 3.0       | 43       | -213.8  | -91.9%  |

- Edge **exists across 1.5–2.0** (positive avg bps), which is encouraging.
- But **threshold=1.8 is better than 2.0** in this window (+13.3 vs +9.3 bps, +58% vs +27% total). The "official" parameter is not optimal on the test window.
- Drops to near-zero at 2.2 and strongly negative at 2.5+. The cliff is steep.
- **Non-monotonic**: 2.0 is worse than 1.8 but better than 2.2. This pattern was also seen in the original SELF_AUDIT.

**Hold sweep** (threshold=2.0 fixed):

| Hold (h) | Avg bps | Total % |
|----------|---------|---------|
| 2        | -15.8   | -54.8%  |
| **3**    | **+27.5** | **+87.0%** |
| **4**    | +9.3    | +27.3%  |
| **5**    | +11.1   | +30.8%  |
| 6        | -31.9   | -85.8%  |
| 8        | -30.2   | -74.2%  |

- Hold=3h is **3× better** than hold=4h on this window. Hold=5h is similar to 4h.
- Hold=2h and hold≥6h are strongly negative.
- Edge exists in a **narrow band** (3–5h). Outside this band, the strategy dies.

### Parameter count

| Component | Parameters | How chosen |
|-----------|-----------|------------|
| VDB signal | threshold=2.0, hold=4h, cooldown=4h, rvol_window=24h, z_window=168h, mr_window=48h | Grid search on 3-year data |
| Fragility | coef_oi=-4.62, coef_pca=-0.95, Q5_mult=0.50, Q4_mult=0.75, warmup=2000 | LogReg fit + in-sample design |
| **Total** | **~11 parameters** | Mixed: some structural, some fitted |

11 parameters for 292 trades = **26 trades per parameter**. This is dangerously low. Minimum recommended ratio is 50–100 trades per parameter.

### Assessment

The base VDB edge (positive avg bps across threshold 1.5–2.0 and hold 3–5h) is probably **real but weaker than reported**. The exact parameter combination (2.0, 4h) is not the best on this window, which suggests it was not overfit to this specific period — but the edge magnitude is sensitive to exact params.

The fragility overlay has **zero OOS validation** of its sizing rules.

---

## 4. Realistic Execution (Fees + Slippage)

**Status: PARTIAL | Risk: MEDIUM**

### What's modeled

| Component | Modeled? | Value | Notes |
|-----------|----------|-------|-------|
| Maker fees (entry) | ✅ | 2 bps | 0.02% Bybit maker |
| Maker fees (exit) | ✅ | 2 bps | 0.02% Bybit maker |
| **RT total** | ✅ | **4 bps** | maker+maker |
| Bid/ask spread | ❌ | 0 bps | Not modeled |
| Slippage | ❌ | 0 bps | Not modeled |
| Partial fills | ❌ | N/A | Assumed 100% fill |
| Market orders (fallback) | ❌ | +7 bps | Not modeled if limit fails |

### Fee sensitivity

| Fee scenario | Avg bps | Total % | Status |
|-------------|---------|---------|--------|
| 4 bps (maker+maker) | +9.3 | +27.3% | Profitable |
| 8 bps (maker+taker) | +5.3 | +15.6% | Marginal |
| 12 bps (taker+maker+slip) | +1.3 | +3.9% | Breakeven |
| 20 bps (taker+taker) | -6.7 | -19.4% | **Dead** |

**The edge survives maker+maker (4bps) and marginal at maker+taker (8bps). At realistic slippage (12bps) the strategy is at breakeven.**

### Real-world execution concerns

1. **Entry timing.** Signal fires at 1h candle close. Bot needs ~5-10s to compute + place order. Price may move.
2. **Limit fill rate during vol spikes.** The strategy enters during high-volatility moments (that's the signal). Limit orders may not fill because the market is moving fast. Empirical fill rate during vol spikes: likely **50-70%**, not 100%.
3. **Exit execution.** After 4h hold, exit via limit order. If not filled within 30s, market order needed → +5.5bps taker. Realistic: ~50% of exits may need market orders.
4. **Realistic RT cost:** 2bp maker entry + ~4bp blended exit (50% maker, 50% taker) + 1bp spread + 1bp slippage = **~8bps RT**. At 8bps, avg trade drops to +5.3bps — still positive but thin.

---

## 5. PnL Stability

**Status: NO | Risk: CRITICAL**

### Monthly distribution (XS-9 baseline, 8 months)

| Month | PnL | Trades | Rank |
|-------|-----|--------|------|
| 2025-07 | +26.8% | 33 | Best |
| 2026-01 | +17.8% | 51 | 2nd |
| 2025-11 | +14.8% | 42 | 3rd |
| 2025-08 | +8.5% | 27 | 4th |
| 2026-02 | -2.7% | 33 | 5th |
| 2025-09 | -8.7% | 25 | 6th |
| 2025-12 | -9.9% | 39 | 7th |
| **2025-10** | **-19.3%** | **42** | **Worst** |

- 4/8 months positive (50%). **Not good** — coin flip would give same result.
- **One month (July +27%) accounts for ~100% of total PnL (+27%).** Remove July → near breakeven.
- Two worst months (Oct, Dec) total -29%. If you started in Oct, you'd see -29% before any recovery.

### Trade concentration (CRITICAL ⚠️)

| Metric | Value |
|--------|-------|
| Total PnL | +2,727 bps (+27.3%) |
| Top 1 trade | +824 bps = **30% of total** |
| Top 3 trades | +1,919 bps = **70% of total** |
| **Top 5 trades** | **+2,884 bps = 106% of total** |
| Without top 5 | **-157 bps (-1.6%) — NEGATIVE** |
| Bottom 5 trades | -3,082 bps |

**Without the top 5 trades (out of 292), the strategy is net negative.** This is the most damning finding in the audit. The "edge" is entirely driven by ~5 outlier trades. The remaining 287 trades collectively lose money.

### Statistical significance

- Avg trade: +9.3 bps, Std: 203.7 bps
- t-statistic: 9.3 / (203.7 / √292) = **0.78**
- **p-value ≈ 0.22 (one-sided)**
- **The average trade return is NOT statistically different from zero** at any conventional significance level.

---

## Overall Assessment

### Probability that edge is real

| Component | Probability | Reasoning |
|-----------|------------|-----------|
| VDB mean-reversion effect | **60-70%** | Structural argument is sound (vol spike → overreaction → reversion). Positive across 1.5-2.0 threshold and 3-5h hold. 32-month walk-forward on SOL is strong. But non-monotonic params and outlier dependency weaken confidence. |
| Fragility overlay improvement | **20-30%** | No OOS test. Sizing rules fit on in-sample data. Only 39 Q5 trades. Coefficients from overlapping period. The Sharpe improvement could easily be noise. |
| Combined XS-9 as reported | **15-25%** | The +52%/yr, Sharpe 1.03 figure is almost certainly overstated. Realistic expected performance is substantially lower. |

### Key risks

1. **🔴 PnL concentration.** Top 5 of 292 trades = 106% of PnL. Strategy is a leveraged tail bet, not a consistent edge. One or two missing outliers → underwater.

2. **🔴 No OOS for overlay.** The Sharpe improvement from fragility sizing (+0.29) has zero independent validation. Could be pure noise on 39 Q5 trades.

3. **🟠 Parameter sensitivity.** The edge exists in a narrow threshold band (1.5–2.0) and hold band (3–5h). Small execution delays could push effective parameters outside the profitable zone.

4. **🟠 Execution gap.** At realistic fees+slippage (8–12bps), the edge shrinks from +9.3 to +1.3–5.3 bps/trade. Fill rate during vol spikes is uncertain.

5. **🟡 8-month window.** The XS-9 test covers only 8 months. The original VDB has 32 months — but on different symbols and data source.

6. **🟡 Long bias.** 96% long trades (279/292). Short side is unvalidated (13 trades). In a sustained bear market, the strategy is buy-the-dip into falling knives.

### Production readiness

| Component | Readiness | Recommendation |
|-----------|-----------|----------------|
| VDB base strategy | ⚠️ **Conditional GO** | Deploy only with strict risk limits. Edge is real but thin and tail-dependent. Use 1x leverage, small capital. |
| Fragility overlay | ❌ **NO-GO** | Do not deploy until validated on truly independent OOS period (Mar–Jun 2026). |
| Combined XS-9 | ❌ **NO-GO** | The reported Sharpe 1.03 is not trustworthy. Deploy VDB alone first. |

### Recommended path to production

1. **Paper trade VDB alone** for 4-8 weeks (no overlay)
2. **Track fill rates, slippage, execution latency** — compute realized RT cost
3. **Accumulate new data** (Mar–Jun 2026) for honest OOS test of fragility overlay
4. **If VDB paper trade confirms >0 avg bps after real costs** → go live with 1x leverage, small capital
5. **After 3+ months of live VDB data** → test fragility overlay on the new OOS data
6. **Only then combine** if overlay shows genuine improvement on new data
