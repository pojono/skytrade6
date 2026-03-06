# Phase 11: New Signal IC Analysis

**Date:** 2026-03-06
**Script:** `research_cross_section/phase11_new_signals.py`
**Script:** `research_cross_section/phase11b_funding_combo.py`

---

## Signals Tested

| Signal | Formula | ICIR | t-stat | Verdict |
|--------|---------|------|--------|---------|
| funding (baseline) | most-recent settlement rate | +0.203 | +7.28 | Strong positive |
| funding_trend | `funding - funding.shift(3)` (24h change) | +0.106 | +3.79 | Positive, significant |
| oi_growth | `oi.pct_change(3)` (24h OI change) | -0.084 | -3.02 | Negative — rejected |
| vol_anomaly | `volume / rolling_90bar_mean - 1` | -0.117 | -4.14 | Negative — rejected |
| btc_relative | `coin_8h_ret - BTC_8h_ret` | -0.184 | -6.57 | Negative — rejected |
| mom_24h (baseline) | 24h price return | -0.161 | -5.75 | Negative IC (but superadditive) |
| composite (baseline) | (funding + mom_24h) / 2 | -0.061 | -2.19 | Negative IC but portfolio Sharpe +2.99 |

**Key: IC is computed cross-sectionally at each 8h bar, measuring rank correlation between signal and 8h forward return.**

---

## Key Findings

### 1. funding_trend has real positive IC
`funding_trend` = change in funding rate over last 24h (3 × 8h bars) has ICIR +0.106, t-stat +3.79. This is statistically significant. Intuition: coins where funding is *rising* are gaining directional conviction — longs are increasing their cost to stay in the position.

### 2. IC ≠ portfolio Sharpe (the momentum paradox)
The composite signal (funding + mom_24h) has negative IC (-0.061) but generates Sharpe 3.5+ in the portfolio. This is the well-documented "conditioning effect": funding selects directional coins, and within that subset, momentum amplifies the carry. The signals are not additive at the IC level but are superadditive at the portfolio level.

This means the IC analysis alone cannot predict which combos will work — portfolio-level backtests are needed.

### 3. All "exotic" signals rejected
oi_growth, vol_anomaly, and btc_relative all have negative IC. They should not be added to the composite.

---

## Portfolio-Level Tests (Phase 11b, correct `.first()` resample, Jan 2025 – Mar 2026)

| Combo | Sharpe | Ann Ret | MaxDD | $1k→ |
|-------|--------|---------|-------|------|
| A: Funding only | 2.895* | — | -39.5% | $11,585* |
| B: Funding + mom_24h (baseline) | 2.177* | — | -55.1%* | $6,657* |
| C: Funding + funding_trend | 2.186* | — | -44.9% | $5,070* |
| D: 2×Funding + funding_trend | 2.343* | — | -45.7% | $6,415* |
| E: Funding + mom + trend | 2.408* | — | -58.3%* | $8,596* |
| F: 2×Funding + mom + trend | 2.875* | — | -54.4%* | $13,069* |

*Note: Phase 11b had a `.last()` resample bug (7h time shift on forward returns). Phase 16 corrected this and showed different magnitudes — see Phase 16 findings for authoritative numbers.*

**Phase 16 (correct alignment) result for "Funding + mom + f_trend":**
- Sharpe: **3.470** vs baseline **3.452**
- MaxDD: **-32.7%** vs baseline **-45.4%** — significant improvement

---

## Conclusion

**Recommendation:** Add `funding_trend` as a 3rd signal component.

New composite (Phase 16 variant 4):
```python
z_funding_trend = cs_zscore(funding - funding.shift(3))  # or diff(24) at 1h then resample
composite = (2*z_funding + z_mom24h + z_funding_trend) / 4
```

Effect: MaxDD -45% → -33% with negligible Sharpe change (+0.018). The funding_trend signal helps the strategy avoid the worst months (May 2025: -14.8% → +6.2%).
