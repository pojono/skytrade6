# XS-6 — Extreme Move Probability Model (Big Move Uplift)

**Date:** 2026-03-03  
**Data:** 52 Bybit perps, 2026-01-01 → 2026-02-28 (59 days)  
**Grid:** 5-minute signal intervals on unified 1m mark-price grid  
**Train/Test:** Jan / Feb with ±24h purge around boundary  
**Protection:** volatility-matched baseline, 2000 permutation tests, BH FDR correction  
**Script:** `xs6_bigmove_uplift.py` → `output/xs6/xs6_uplift.csv`

---

## TL;DR

**43 / 2,592 combos pass all hard filters.** The strongest signal is **volatility compression + OI build (S07)** — when realized vol is in P20 but open interest is surging (z ≥ 1.5), probability of a big move in next 12-24h is **1.7-5.4x baseline** even after vol-matching. This holds across multiple coins and weeks.

Funding extremes (S01/S02) also show uplift but are less stable week-to-week. Intersection states (S04/S05) are too rare for reliable estimates.

**Key caveat:** These uplift factors tell you *when* big moves are more likely, not *which direction*. A convex payoff structure (bracket orders, wide TP) is needed to monetize — that's XS-7.

---

## 1. Setup

### Targets (6 definitions per signal)
- **Def A (ATR-normalized):** `|ret| >= k × ATR_1h`, k ∈ {3.0, 4.0}
- **Def B (raw bp):** `|ret| >= 300bp (12h)` or `|ret| >= 500bp (24h)`

### 10 States Tested
| State | Condition | Mean Frequency |
|-------|-----------|---------------|
| S01 fund_high | funding_z ≥ +2 | 2.9% |
| S02 fund_low | funding_z ≤ -2 | 5.1% |
| S03 oi_surge | OI change z ≥ +2 | 3.0% |
| S04 fund_hi+oi_hi | fund_z ≥ +2 AND oi_z ≥ +2 | 0.3% |
| S05 fund_lo+oi_hi | fund_z ≤ -2 AND oi_z ≥ +2 | 0.5% |
| S06 compress_vol | rv_2h ≤ P20 AND volume_2h ≥ P60 | 0.4% |
| S07 compress_oi | rv_6h ≤ P20 AND oi_z ≥ 1.5 | 0.4% |
| S08 stall_oi | trend_2h ≤ P20 AND oi_z ≥ +2 | 0.4% |
| S09 stall_fund | trend_2h ≤ P20 AND |fund_z| ≥ 2 | 1.1% |
| S10 thin_move | volume_1h ≤ P20 AND |ret_30m| ≥ P80 | 0.9% |

### PASS Criteria (hard)
- State frequency ≥ 0.3% (`rateS`)
- nS in test ≥ 30
- Vol-matched uplift in test ≥ 1.5x
- Absolute delta in test ≥ +1% probability
- FDR q-value < 0.10

---

## 2. Universe-Level Results (pooled across 52 coins)

Best states by universe uplift (ATR-normalized, k=3):

| State | Horizon | pS | p0 | Uplift | OOS Uplift |
|-------|---------|----|----|--------|------------|
| S07 compress_oi | 12h | 0.59% | 0.17% | **3.4x** | **5.9x** |
| S07 compress_oi | 24h | 1.40% | 0.52% | **2.7x** | **4.8x** |
| S01 fund_high | 12h | 0.52% | 0.17% | **3.0x** | **4.2x** |
| S01 fund_high | 24h | 1.35% | 0.52% | **2.6x** | **5.0x** |
| S04 fund_hi+oi_hi | 24h | 2.51% | 0.52% | **4.8x** | **7.2x** |

S04/S05 have the highest raw uplift but only 0.3-0.5% frequency — fragile estimates.

---

## 3. Top Symbol-Level Candidates (honest assessment)

### Tier 1: Strong evidence (weekly stable, reasonable uplift)

| Symbol | State | H | Def | nS_test | pS | p0 | Uplift_m | Δ | Weekly | q_fdr |
|--------|-------|---|-----|---------|----|----|----------|---|--------|-------|
| CCUSDT | S02_fund_low | 24h | B | 816 | 22.5% | 17.0% | **2.0x** | +5.5% | 6/8 | 0.013 |
| CCUSDT | S02_fund_low | 12h | B | 864 | 31.8% | 24.1% | **1.7x** | +7.7% | 5/8 | 0.013 |
| ALCHUSDT | S03_oi_surge | 24h | B | 194 | 52.6% | 21.0% | **1.7x** | +31.6% | 6/9 | 0.013 |
| 1000RATSUSDT | S01_fund_high | 24h | B | 336 | 91.4% | 40.6% | **1.9x** | +50.8% | 5/6 | 0.013 |
| ARBUSDT | S03_oi_surge | 24h | B | 226 | 60.6% | 33.0% | **1.8x** | +27.6% | 5/9 | 0.051 |
| IPUSDT | S03_oi_surge | 24h | B | 221 | 44.8% | 25.6% | **1.6x** | +19.2% | 6/8 | 0.013 |
| GUNUSDT | S03_oi_surge | 24h | B | 232 | 61.6% | 36.6% | **1.5x** | +25.0% | 6/9 | 0.058 |
| COAIUSDT | S01_fund_high | 24h | B | 658 | 49.5% | 38.5% | **1.6x** | +11.0% | 6/9 | 0.013 |
| GRASSUSDT | S09_stall_fund | 24h | B | 107 | 53.3% | 40.0% | **1.5x** | +13.2% | 4/8 | 0.013 |

These have: large nS_test (>100), reasonable uplift (1.5-2.0x), positive in ≥50% of weeks.

### Tier 2: High uplift but episodic (driven by 1-2 big events)

| Symbol | State | H | Def | nS_test | Uplift_m | Weekly |
|--------|-------|---|-----|---------|----------|--------|
| BTRUSDT | S01_fund_high | 24h | A_k3 | 192 | **5.5x** | 1/5 |
| COAIUSDT | S04_fund_hi+oi_hi | 24h | A_k3 | 45 | **10.0x** | 1/4 |
| ATHUSDT | S07_compress_oi | 24h | A_k3 | 60 | **16.7x** | 1/4 |
| FARTCOINUSDT | S07_compress_oi | 24h | B | 38 | **2.1x** | 4/6 |
| ATOMUSDT | S01_fund_high | 24h | B | 96 | **6.2x** | 1/2 |

High uplift but poor weekly stability → likely driven by 1-2 massive moves in Feb.

### Tier 3: Compression states (best conceptual fit for convex strategies)

| Symbol | State | H | Def | nS_test | pS | p0 | Uplift_m | Weekly |
|--------|-------|---|-----|---------|----|----|----------|--------|
| EIGENUSDT | S07_compress_oi | 24h | B | 44 | 75.0% | 30.2% | **2.5x** | 3/3 |
| 1000PEPEUSDT | S07_compress_oi | 12h | B | 95 | 75.8% | 34.9% | **2.4x** | 2/6 |
| 1000PEPEUSDT | S07_compress_oi | 24h | B | 95 | 46.3% | 23.8% | **2.4x** | 3/6 |
| CYBERUSDT | S06_compress_vol | 24h | B | 135 | 64.4% | 43.3% | **2.3x** | 3/3 |
| ATOMUSDT | S07_compress_oi | 12h | B | 51 | 41.2% | 27.6% | **2.1x** | 4/5 |
| APEXUSDT | S06_compress_vol | 12h | B | 38 | 89.5% | 44.9% | **2.1x** | 3/5 |
| APEXUSDT | S06_compress_vol | 24h | B | 38 | 84.2% | 37.8% | **2.0x** | 4/5 |
| ANIMEUSDT | S09_stall_fund | 24h | B | 82 | 39.0% | 20.8% | **2.1x** | 4/5 |

S06/S07 = volatility compression. When vol is low but OI is building, pressure accumulates → explosion. This is the best conceptual foundation for a convex payoff strategy.

---

## 4. State Rankings (by number of PASS candidates)

| State | # PASS | Avg OOS Uplift | Best Coin | Best Uplift |
|-------|--------|---------------|-----------|-------------|
| S07 compress_oi | 8 | 1.77x | EIGENUSDT | 2.52x |
| S01 fund_high | 7 | 1.35x | 1000RATSUSDT | 1.92x |
| S02 fund_low | 6 | — | CCUSDT | 1.97x |
| S03 oi_surge | 6 | — | ARBUSDT | 1.75x |
| S09 stall_fund | 5 | — | ANIMEUSDT | 2.15x |
| S05 fund_lo+oi_hi | 4 | — | AIXBTUSDT | 2.04x |
| S06 compress_vol | 3 | — | CYBERUSDT | 2.33x |
| S10 thin_move | 2 | — | JELLYJELLYUSDT | 7.14x |
| S04 fund_hi+oi_hi | 1 | — | COAIUSDT | 10.00x |
| S08 stall_oi | 1 | — | INITUSDT | 5.00x |

---

## 5. Interpretation & Caveats

### What works
1. **Volatility compression + positioning (S06/S07):** When vol collapses but OI keeps building or volume stays elevated → coiled spring. 2-2.5x uplift, multiple coins, decent weekly stability. Best foundation for a convex strategy.

2. **Funding extremes (S01/S02):** Extreme crowding predicts big moves. The S01 funding_high universe uplift is 3-5x. However, the big move direction is ambiguous — it could be continuation (squeeze) or reversal.

3. **OI surge (S03):** Rising OI accelerates upcoming volatility. Consistent 1.5-1.8x uplift across 6+ coins. High frequency (3%) means many opportunities.

### What doesn't work (or is unreliable)
1. **Intersection states S04/S05:** Too rare (0.3-0.5%) for stable estimates. The very high uplifts (5-10x) are driven by handful of events.

2. **S08 stall_oi:** Only 1 PASS candidate — trend stall + OI is not consistently predictive on its own.

3. **Def A (ATR-normalized) targets:** Much rarer events (0.1-1% base rate), so noisier estimates. Def B (raw bp) provides more reliable results due to higher base rates.

### Caveats
- **2 months of data.** Jan/Feb 2026 includes a major crypto drawdown — results could be regime-dependent.
- **No direction prediction.** These states flag *when* to expect big moves, not *which way*.
- **Selection bias protection is imperfect.** Vol-matched baseline uses ATR quintile + hour-of-day, but doesn't match on all confounders (e.g., day-of-week, market regime).
- **Some coins have structural vol.** 1000RATSUSDT with 91% pS_test for B-24h just means this coin moves >5% in most 24h windows regardless of state.

---

## 6. Next Steps (XS-7 direction)

The most actionable finding: **S07 (compress_oi) provides 2-2.5x uplift for big moves with decent stability.** To monetize:

1. **Direction skew analysis:** Within S07 events, check `P(ret > +k) vs P(ret < -k)`. If directionally skewed → directional entry. If symmetric → straddle-like bracket orders.

2. **Entry timing:** S07 fires when rv_6h is in P20 AND oi_z ≥ 1.5. How soon after the signal does the move come? If clustered in first 4-8h → tighter time stop.

3. **Convex payoff construction:**
   - Bracket orders: simultaneous buy-stop at +X bps, sell-stop at -X bps
   - Wide TP (e.g., 3-4x ATR) with time stop (12-24h)
   - Kelly-based sizing using the uplift probability

4. **Multi-coin portfolio:** S07 fires on different coins at different times → can diversify across 5-10 positions simultaneously.

---

## Files

- **Script:** `flow_research/xs6_bigmove_uplift.py`
- **Full results:** `flow_research/output/xs6/xs6_uplift.csv` (2,592 rows)
- **Top states:** `flow_research/output/xs6/xs6_top_states.md`
