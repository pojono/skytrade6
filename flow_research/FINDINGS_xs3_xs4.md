# FINDINGS — XS-3 State Transition + XS-4 Funding Carry

**Date:** 2026-03-03  
**Period:** 2026-01-01 → 2026-02-28  
**Universe:** 52 altcoins (Bybit perps)

---

## VERDICT: BOTH NO-GO ❌

Neither model survives production-grade testing.

| Model | Best Config | Mean daily (bp) | OOS consistency | Sharpe |
|-------|-------------|---:|---|---:|
| **XS-3** State Transition | K=5, slip=0 | **-115** | Both halves negative | -7.46 |
| **XS-4** Funding Carry A | Q20, 8h, slip=0 | **-34** | Both halves negative | -3.47 |
| **XS-4** Funding Carry B | \|z\|≥2, 8h, slip=0 | **+67** | Feb +144 / Jan -15 | 3.02 |
| **XS-4** Funding Carry B | \|z\|≥2, 24h, slip=0 | **+54** | Feb -149 / Jan +235 | 1.35 |

No configuration passes all GO criteria at slip=2bp.

---

# XS-3 — State Transition (Dispersion Regime Onset)

## Design

Instead of trading dispersion as a continuous signal (XS-2's failure), we trade **regime onset events**:
- **Enter:** dispersion_60m percentile crosses P80 upward
- **Exit:** dispersion_60m percentile crosses P50 downward
- One position per regime. No vintages. No rebalancing.

## Results

### Regime Statistics
- **373 regimes** detected over 59 days (~6.3/day)
- **Average duration: 79 minutes** (median similar)
- **Exposure: ~35%** of calendar time

### Performance (K=5, all slippage levels)

| Slip | RT cost | Mean/regime (bp) | Daily (bp) | Sharpe | HR | Max DD |
|---:|---:|---:|---:|---:|---:|---:|
| 0bp | 20bp | -17.90 | -115.14 | -7.46 | 34% | -6785 |
| 2bp | 24bp | -21.90 | -140.86 | -9.12 | 32% | -8202 |
| 5bp | 30bp | -27.90 | -179.45 | -11.57 | 29% | -10350 |

### Walk-Forward

| Period | Daily (bp) | Sharpe |
|--------|---:|---:|
| OOS Feb | -136 | -7.54 |
| OOS Jan | -95 | -7.51 |

Both halves deeply negative. K=3 and K=10 equally bad.

### Root Cause

The dispersion regime onset **has no directional content**. When dispersion spikes:
- High vol_z coins go both up AND down (that's what dispersion means)
- Longing the highest vol_z and shorting the lowest is betting on a **continuation of spread widening**
- But dispersion is mean-reverting, so spread widening reverses → the position loses

The fundamental flaw: **dispersion is a volatility phenomenon, not a directional one.** You can't build a directional L/S portfolio from a non-directional signal without additional directional information.

---

# XS-4 — Funding Carry Cross-Section

## Design

Trade funding rate extremes as a structural factor:
- **Version A:** Long bottom 20% funding_z, Short top 20% (8h hold)
- **Version B:** Long |z|≥2 negative, Short |z|≥2 positive (8h or 24h hold)
- Rebalance every 8h aligned with funding settlements
- PnL = price_pnl + funding_received

## Funding Rate Statistics

- Mean FR: ~0.5bp per 8h (~5% annualized)
- Std FR: ~1.2bp per 8h
- |z| ≥ 2 frequency: varies by coin, ~5-10% of observations

## Critical Finding: Funding Component is Negligible

**Funding income = 1-2% of total PnL.**

Over an 8h hold period:
- Funding received: ~1-2bp (one settlement)
- Price movement std: ~100-200bp
- Signal-to-noise: funding/price ≈ 1:100

This means XS-4 is **NOT a carry trade** — it's a price speculation trade that happens to use funding_z as a sorting variable.

## Results

### Version A (Q20, 8h)

| Slip | N | Mean/rebal (bp) | Daily (bp) | Sharpe | HR |
|---:|---:|---:|---:|---:|---:|
| 0bp | 171 | -11.5 | -33.9 | -3.47 | 45% |
| 2bp | 171 | -15.5 | -45.7 | -4.68 | 43% |

Both OOS halves negative. Classic non-signal.

### Version B (|z|≥2, 8h)

| Slip | N | Mean/rebal (bp) | Daily (bp) | Sharpe | HR |
|---:|---:|---:|---:|---:|---:|
| 0bp | 60 | +45.5 | +66.6 | 3.02 | 47% |
| 2bp | 60 | +41.5 | +60.7 | 2.76 | 47% |

**But OOS is split:** Feb +137bp/day (Sharpe 6.0) vs Jan -20bp/day (Sharpe -0.9).

This is N=60 trades over 59 days — too few for statistical significance. The Feb "edge" is likely 1-2 large outlier trades in a small sample.

### Version B (|z|≥2, 24h)

Similar issue but reversed: Jan +235bp/day vs Feb -156bp/day.

The **opposite month** wins in 8h vs 24h — a classic sign of noise, not signal. If there were a real carry edge, both holding periods should profit in the same month.

## Walk-Forward Verdict

No version passes all criteria:
- Version A: both halves negative
- Version B 8h: Jan negative
- Version B 24h: Feb negative

## Root Cause

1. **Funding is too small relative to price noise.** At 8h frequency on altcoins, a 1bp funding edge is drowned by 150bp price volatility. Funding carry works on majors (BTC/ETH) with lower vol and larger notional, or on 1h settlement coins with higher funding frequency.

2. **N=60 trades is insufficient.** Version B's threshold (|z|≥2) triggers rarely, giving us only 1 trade/day. With 150bp per-trade volatility, you need hundreds of trades for statistical significance.

3. **Cross-sectional funding sort ≠ individual coin carry trade.** Our [existing FR research](FINDINGS_xs_research.md) shows that the **single-coin HOLD strategy** (enter when FR > 20bp, exit when < 8bp, hold until reversion) works at +$1,443/day on 1h coins. But the cross-sectional version (sort all coins by z-score, long bottom / short top) fails because:
   - The sort doesn't select coins with actionable FR levels
   - Bottom 20% by z-score might all have near-zero FR
   - The cross-sectional hedge destroys the carry component

---

# Lessons Learned (XS-2 + XS-3 + XS-4)

## What Doesn't Work in Cross-Section

| Approach | Why it fails |
|----------|-------------|
| **Dispersion as continuous signal** (XS-2) | 74% overlap, fees destroy edge |
| **Dispersion as onset event** (XS-3) | Non-directional signal → L/S has no edge |
| **Funding carry cross-section** (XS-4) | Funding too small vs price noise at 8h freq |

## Structural Issues

1. **Altcoin L/S is hard.** With 150-300bp intraday vol per coin, you need very strong signals (>50bp edge per trade) to overcome fees + noise. None of our signals reach this threshold.

2. **Cross-sectional hedging is expensive.** Going long AND short equal-weight baskets means you're paying fees on both legs. If the signal only works on one leg, you're paying double for single-leg alpha.

3. **Frequency matters.** At 5m (XS-2) → fee death. At 8h (XS-4) → too few trades. The sweet spot for crypto structural factors may be daily or weekly rebalancing with concentrated positions — but that needs more data (months, not 59 days).

## What Might Work Instead

Based on our prior research:

1. **Single-coin FR HOLD** (already validated): +$1,443/day on 1h Bybit coins. Entry ≥20bp FR, exit <8bp. This is NOT a cross-sectional strategy — it's a targeted carry trade on specific coins at specific times.

2. **Longer horizon cross-section** (untested): weekly rebalance, larger universe, fundamentals-based sort (market cap, volume rank). Would need 6+ months of data.

3. **Event-driven** (partially validated in flow_shock_research): specific microstructure events (liquidation cascades, OI squeezes) that have directional content.

---

## Files

| File | Description |
|------|-------------|
| `xs3_state_transition.py` | XS-3 production script |
| `xs4_funding_carry.py` | XS-4 production script |
| `output/xs3/xs3_results.csv` | XS-3 results by K and slippage |
| `output/xs3/xs3_regimes_K5.csv` | XS-3 regime-level detail |
| `output/xs4/xs4_results.csv` | XS-4 results by version and slippage |
| `output/xs4/xs4_rebalances_A_8h.csv` | XS-4 Version A trade detail |
