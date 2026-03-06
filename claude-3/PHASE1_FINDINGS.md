# Phase 1 Findings: Signal IC Analysis

**Date:** 2026-03-06

---

## Summary

Phase 1 computed cross-sectional Spearman IC (Information Coefficient) between each signal and forward returns across all symbols at each timestamp. Two runs were performed: **dirty data** (initial) and **clean data** (after fixing 2025-01-01 corruption). Results differ substantially.

---

## Data Quality Issue

Raw kline CSVs had a corruption on 2025-01-01: close prices for many symbols were zeroed out or near-zero (~0.001). This caused `pct_change()` to produce `inf` and `-1.0` forward return values. About 22% of rows had |fwd_8h| > 100%.

**Fix applied in `phase1_build_signals.py`:**
```python
price_median = df["close"].median()
bad_price = (df["close"] <= 0) | (df["close"] < price_median * 0.01)
df.loc[bad_price, "close"] = np.nan
df["close"] = df["close"].interpolate(method="time", limit=12)
```

All 131 signal parquets were rebuilt after this fix. **Always use clean-data results.**

---

## Run 1: Dirty Data (1h bars, Phase 1 IC)

| Signal | ICIR @fwd_8h | Notes |
|--------|-------------|-------|
| prem_z | **+0.936** | Dominant but spurious |
| mom_24h | +0.421 | Appears positive |
| mom_8h | +0.399 | Appears positive |
| funding | +0.506 | Positive |
| oi_div | -0.016 | Near zero |
| ls_z | -0.094 | Weak negative |

**Critical finding:** prem_z ICIR of 0.936 was an artifact of the corruption. The zeroed prices on 2025-01-01 created artificial basis spikes that correlated with subsequent "recovery" returns. This was not a real signal.

**Why dirty momentum looked positive:** The large negative returns on 2025-01-01 (due to corrupt prices) made coins look like past losers, which then "bounced" — generating apparent momentum continuation that was really a price recovery artifact.

---

## Run 2: Clean Data (8h bars, Phase 2b IC — use this)

IC computed at 8h rebalancing frequency (non-overlapping bars) after rebuilding all signal parquets.

| Signal | IC @fwd_8h | ICIR @fwd_8h | IC @fwd_24h | ICIR @fwd_24h |
|--------|-----------|-------------|------------|--------------|
| funding | +0.0195 | **+0.151** | +0.0281 | **+0.220** |
| prem_z | -0.0006 | -0.005 | +0.0023 | +0.019 |
| mom_8h | -0.0321 | -0.172 | -0.0251 | -0.134 |
| mom_24h | -0.0352 | **-0.186** | -0.0255 | -0.137 |
| mom_48h | -0.0274 | -0.145 | -0.0312 | -0.166 |
| ls_z | 0.0000 | 0.000 | -0.0081 | -0.068 |
| oi_div | -0.0123 | -0.105 | -0.0133 | -0.115 |

### Key Findings

**Funding carry is the only real signal.** ICIR = +0.151 at 8h, +0.220 at 24h. Both statistically significant (t-stat = 5.4 and 7.9 respectively). Interpretation: coins with positive funding (longs pay shorts) tend to outperform — the funding is predictive of directional demand imbalance.

**Momentum is strongly mean-reverting at 8h scale.** All momentum lookbacks show negative IC at 8h horizon. IC = -0.035, ICIR = -0.186 for mom_24h. This is the opposite of what dirty data showed.

**prem_z is dead on clean data.** ICIR ≈ 0 at 8h. The dirty-data IC of +0.936 was entirely spurious.

**ls_z (L/S ratio) is flat.** IC ≈ 0 at all horizons. No usable signal.

**oi_div (OI-price divergence) is weakly negative.** Consistent but small negative IC. Not usable alone.

### Implication: Momentum Paradox

Despite momentum having negative IC at 8h, the combo `funding + mom_24h` produces better backtests than `funding` alone (Sharpe 3.27 vs 1.25).

**Explanation:** When combined with funding, momentum acts as a *conditioner*, not a standalone predictor. Funding selects coins with net long demand; high recent returns within that subset signal self-reinforcing uptrends. The global negative IC reflects mean-reversion across all coins; conditional on high funding, momentum continuation dominates.

---

## Signal Autocorrelation (Turnover Proxy)

Measured as 1-period lagged correlation of each signal:

| Signal | 1-lag autocorr | ~Turnover/rebal |
|--------|---------------|-----------------|
| funding | ~0.98 | Very low (~5%) |
| prem_z | ~0.85 | Low (~15%) |
| mom_24h | ~0.70 | Medium (~30%) |
| mom_8h | ~0.50 | High (~50%) |
| ls_z | ~0.75 | Medium (~25%) |

Funding is slow-moving (8h autocorr ≈ 0.98), which minimizes transaction costs for the primary alpha source.

---

## Scripts

- `research_cross_section/phase1_build_signals.py` — builds signal parquets
- `research_cross_section/phase1_compute_ic.py` — IC computation (dirty data; run 1)
- `research_cross_section/phase2b_ic_and_combos.py` — IC computation (clean data, 8h; run 2)

## Output Files

- `results/ic_summary.csv` — Phase 1 IC (dirty data, 1h bars)
- `results/phase2b_ic_8h.csv` — Phase 2b IC (clean data, 8h bars) — **authoritative**
