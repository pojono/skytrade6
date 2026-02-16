# Plan v15 — ML-Powered Grid Bot Improvements

**Objective:** Incrementally add ML findings from v9-v12 to the grid bot, measuring each improvement against the **Fix 1.00% (24h)** baseline.

**Baseline (Fix 1.00% 24h, 2 bps maker fee, 13 months):**

| Symbol | PnL | Sharpe | MaxDD |
|--------|-----|--------|-------|
| BTC | +$789 | 0.83 | -$837 |
| ETH | +$901 | 0.45 | -$2,002 |
| SOL | +$2,151 | 0.85 | -$2,424 |
| XRP | -$1,997 | -0.87 | -$4,728 |
| DOGE | -$1,131 | -0.41 | -$5,968 |

**Method:** Add one ML feature at a time. Test on all 5 symbols. Keep if it improves PnL or Sharpe on ≥3/5 symbols without worsening the others significantly.

---

## Step 1: Direct Range Prediction

**Source:** v11 Ridge range model (R²=0.33 at 1h)
**Current:** spacing = predicted_vol × 5.6 / divisor (indirect, via vol×k constant)
**Change:** Train a Ridge model to predict `fwd_range` (high-low)/close directly, use that for spacing.
**Why it might help:** Range model captures intra-bar extremes that return-based vol misses. Range and vol are 89% correlated but not identical — the 11% difference could matter.
**Risk:** v11 showed range prediction ≈ vol-derived range (R²=0.326 vs 0.324). May be a wash.

---

## Step 2: P90 Quantile Range for Grid Width

**Source:** v11 Linear Quantile Regression (P90 calibration: 90.5% actual, near-perfect)
**Current:** spacing = predicted_range / divisor (uses point estimate)
**Change:** Train a P90 quantile model. Set grid width so that 90% of 1h price action stays within the grid.
**Why it might help:** P90 gives a "safety margin" — the grid rarely gets overrun, reducing inventory blowup during unexpected moves. v11 showed P90 coverage = 87% (close to target 90%).
**Risk:** Wider grid = fewer fills = less grid profit. Need to check if reduced inventory losses offset reduced fills.

---

## Step 3: Adaptive Rebalance Interval

**Source:** v9 vol prediction + regime detection
**Current:** Fixed 24h rebalance for all periods.
**Change:** Rebalance more frequently during high-vol periods (e.g., 8h when predicted_vol > 2× median), less frequently during calm (e.g., 48h when predicted_vol < 0.5× median).
**Why it might help:** v14 showed 24h > 8h rebalance overall, but during extreme trends, 24h lets inventory grow dangerously. Adaptive rebalance could get the best of both worlds.
**Risk:** More rebalances = more fees. Need the inventory savings to exceed the extra fee cost.

---

## Step 4: Breakout Detection → Widen Grid

**Source:** v12 Logistic Regression breakout model (AUC=0.654 at 1h, 0.687 at 4h)
**Current:** Grid spacing doesn't react to breakout probability.
**Change:** When breakout probability > threshold (e.g., 0.4), multiply spacing by 1.5-2×. This widens the grid before large moves.
**Key features:** atr_1h, range_compression_1h, bb_pctile_2h, bb_width_2h, bar_eff_1h
**Why it might help:** Breakouts are when grids get destroyed — inventory accumulates rapidly in one direction. Widening before a breakout reduces the number of levels that get filled against you.
**Risk:** AUC=0.65 means lots of false positives. Widening unnecessarily reduces fills during calm periods.

---

## Step 5: Consolidation Detection → Tighten Grid

**Source:** v12 feature `consolidation_2h_vs_24h` (corr=0.297 with S/R break)
**Current:** Grid doesn't know about consolidation.
**Change:** When consolidation ratio is high (tight range relative to recent history), tighten grid spacing to capture more oscillations within the narrow range.
**Why it might help:** Consolidation = mean-reverting price action = ideal for grid bots. Tighter grid during consolidation = more fills, each profitable.
**Risk:** Consolidation often precedes breakouts. Tightening right before a breakout is the worst possible timing. Need to combine with Step 4.

---

## Step 6: 4h Range for Rebalance Timing

**Source:** v11 4h range prediction (R²=0.19)
**Current:** Rebalance on fixed schedule regardless of expected future range.
**Change:** Use predicted 4h range to decide rebalance timing. If predicted 4h range is small (calm ahead), delay rebalance. If large (volatile ahead), rebalance sooner to limit inventory.
**Why it might help:** Aligns rebalance timing with expected market conditions.
**Risk:** 4h prediction is weaker (R²=0.19 vs 0.33 at 1h). May not be reliable enough.

---

## Step 7: Asymmetric Grid

**Source:** v11 upside/downside prediction (R²=0.13 for upside, 0.13 for downside)
**Current:** Symmetric grid (equal levels above and below center).
**Change:** If predicted upside > downside, shift grid center slightly upward (more sell levels, fewer buy levels). Vice versa.
**Why it might help:** Could reduce inventory accumulation by having more levels in the expected direction.
**Risk:** v11 showed asymmetry prediction R²≈0 and directional accuracy=50.2%. This is essentially random. **Lowest priority — likely won't work.**

---

## Execution Order & Rationale

| Priority | Step | Expected Impact | Confidence |
|----------|------|----------------|------------|
| 1 | **Step 1: Direct range** | Low (wash with vol×k) | Medium |
| 2 | **Step 2: P90 quantile** | Medium (safety margin) | High |
| 3 | **Step 4: Breakout widen** | Medium-High (prevent blowup) | Medium |
| 4 | **Step 5: Consolidation tighten** | Medium (more fills in ideal conditions) | Medium |
| 5 | **Step 3: Adaptive rebalance** | Medium (best of both worlds) | Medium |
| 6 | **Step 6: 4h rebalance timing** | Low-Medium | Low |
| 7 | **Step 7: Asymmetric grid** | Very Low (direction unpredictable) | Very Low |

**Stop early if:** After 3 steps with no improvement, the remaining steps are unlikely to help.

---

## Success Criteria

For each step, measure on all 5 symbols:
- **Primary:** Total PnL improvement vs baseline
- **Secondary:** Sharpe ratio improvement
- **Guard:** MaxDD should not worsen by >50%

A step is "kept" if it improves PnL on ≥3/5 symbols without catastrophic MaxDD increase on any.

---

## Files

| File | Description |
|------|-------------|
| `grid_bot_v15.py` | New script with incremental ML improvements |
| `FINDINGS_v15_ml_grid.md` | Results of each step |
| `results/grid_v15_*.txt` | Raw output per step per symbol |
