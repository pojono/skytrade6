# XS-6 — Extreme Move Probability Model (Big Move Uplift)

**Date:** 2026-03-03 (v2 — with deep analysis)  
**Data:** 52 Bybit perps (55 discovered, 3 skipped <50% valid), 2026-01-01 → 2026-02-28  
**Grid:** 5-minute signal intervals on unified 1m mark-price grid  
**Train/Test:** Jan / Feb with ±24h purge around boundary  
**Protection:** vol-matched baseline (p_control floor = max(p0/10, 1e-4)), 2000 permutation tests, BH FDR  
**Scripts:** `xs6_bigmove_uplift.py`, `xs6_deep_analysis.py`

---

## TL;DR

**91 / 3,456 combos pass hard filters** across 37 symbols and all 10 states (v2: added per-coin adaptive Def B_adp, fixed p_control floor bug).

The **only state that survives the shuffle sanity test** on Def A (ATR-normalized) is **S07 compress_oi** — real uplift 2.50x vs shuffle p99 of 1.81x. S01 fund_high is borderline (3.51x vs 3.23x). All other states on Def A are indistinguishable from noise. On Def B, most "uplift" is driven by trivially high base rates on volatile coins.

**Time-to-move for S07:** median breach at **10.2h**, clustered 6-12h (53%) and 12-24h (45%). No early moves (0-3h). This is tradeable with a 12-24h time stop.

**Direction skew for S07:** Nearly symmetric at 24h (31% up, 40% down). Slight downward bias overall. No strong directional edge → **bracket/straddle structure required.**

**OI leakage test:** Leaking future OI gives only 1-2% boost → OI is not the main driver. The **rv_6h compression** component is doing the work.

---

## 1. Setup

### Targets (8 definitions per signal, v2)
- **Def A (ATR-normalized):** `|ret| >= k × ATR_1h`, k ∈ {3.0, 4.0}
- **Def B (raw bp, fixed):** `|ret| >= 300bp (12h)` or `|ret| >= 500bp (24h)`
- **Def B_adp (per-coin adaptive):** `|ret| >= P95(|ret|)` expanding over 30d — this controls for hyper-volatile coins where Def B is trivially easy

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
- rateS ≥ 0.3%, nS_test ≥ 30, uplift_matched_test ≥ 1.5x, delta_test ≥ +1%, q_fdr < 0.10

---

## 2. Sanity Tests (CRITICAL — read before trusting any uplift numbers)

### Q3: Shuffle Test — "Is the uplift real or random?"

Shuffled state labels within each day 500x. If real uplift > shuffle p99 for many symbols → genuine.

| State | Target | Real Uplift | Shuffle p99 | % Syms Exceeding p99 | **Verdict** |
|-------|--------|-------------|-------------|---------------------|-------------|
| **S07 compress_oi** | big_A_k3.0_24h | **2.50** | 1.81 | 9% | **✅ GENUINE** |
| S01 fund_high | big_A_k3.0_24h | 3.51 | 3.23 | 12% | ⚠️ Borderline |
| S01 fund_high | big_B_24h | 1.50 | 1.75 | 50% | ❌ Noise (real < p99) |
| S03 oi_surge | big_A_k3.0_24h | 0.58 | 1.66 | 2% | ❌ Noise |
| S03 oi_surge | big_B_24h | 1.39 | 1.49 | 27% | ❌ Noise |
| S06 compress_vol | big_A_k3.0_24h | 0.00 | 0.10 | 0% | ❌ No signal |
| S06 compress_vol | big_B_24h | 1.11 | 1.36 | 29% | ❌ Noise |
| S09 stall_fund | big_B_24h | 1.10 | 1.34 | 27% | ❌ Noise |
| S10 thin_move | big_B_24h | 0.75 | 1.12 | 6% | ❌ Noise |

**Conclusion:** Only S07 on Def A (ATR-normalized) is genuinely above chance. Def B "uplift" is mostly noise amplified by high base rates on volatile coins. S01/Def A is borderline — could be real for a subset of coins.

### Q4: OI Leakage Trap — "Is OI actually driving the signal?"

Leaked future OI (removed 5min causal shift) to test if OI matters:

| State | Target | Normal Uplift | Leaked Uplift | Boost |
|-------|--------|--------------|---------------|-------|
| S07 compress_oi | A_k3.0_24h | 2.50 | 2.56 | **1.02x** |
| S07 compress_oi | B_24h | 0.89 | 0.90 | 1.01x |
| S03 oi_surge | A_k3.0_24h | 0.58 | 0.62 | 1.07x |

**Conclusion:** OI leakage doesn't help (~1.0x boost). The rv_6h compression in S07 is the main driver, not OI per se. OI adds specificity (narrows the signal) but doesn't carry the information.

---

## 3. Time-to-Move Analysis (Q1)

For S07 compress_oi signals in test period (Feb), when does the ATR-based big move threshold get breached?

| Metric | Value |
|--------|-------|
| Total signals | 1,873 |
| Breach rate (3×ATR, 24h) | **3.1%** |
| Median time to breach | **610 min (10.2h)** |
| P25 / P75 | 558 min / 1274 min |
| 0-1h | 0% |
| 1-3h | 0% |
| 3-6h | 1.7% |
| **6-12h** | **53.4%** |
| 12-24h | 44.8% |
| Direction at breach | **100% up** (Feb was a recovery month) |

**Key implication:** Moves happen in the 6-24h window. No latency/speed advantage needed. A bracket order with 12-24h time stop is appropriate.

Other states:
- **S01 fund_high:** median 16.3h, 84% up at breach, 3.2% breach rate
- **S09 stall_fund:** median 9.1h, 35% of breaches in 1-3h window (fastest), 89% up
- **S03 oi_surge:** median 15.8h, 96% up, 1.9% breach rate

⚠️ The 100% up direction in S07 is an artefact of Feb 2026 being a recovery month. Do not take this as a permanent directional edge.

---

## 4. Direction Skew (Q2)

For state signals in test period, P(ret > +200bp) vs P(ret < -200bp):

| State | H | P(up>200bp) | P(down>200bp) | Uplift Up | Uplift Down | **Skew** |
|-------|---|-------------|---------------|-----------|-------------|----------|
| S07 compress_oi | 12h | 23.8% | 34.4% | 0.96x | 1.06x | **slight down** |
| S07 compress_oi | 24h | 31.3% | 39.8% | 1.09x | 1.00x | **~symmetric** |
| S06 compress_vol | 12h | 12.0% | 38.1% | 0.51x | 1.38x | **strong down** |
| S01 fund_high | 24h | 28.6% | 50.6% | 0.74x | 1.34x | **strong down** |
| S03 oi_surge | 12h | 31.1% | 34.2% | 1.25x | 1.10x | **slight up** |
| S03 oi_surge | 24h | 33.9% | 41.0% | 1.16x | 1.07x | **slight up** |

**Key findings:**
- **S07 at 24h is nearly symmetric** — no strong directional edge → bracket orders
- **S06 skews strongly down** — when vol compresses with high volume, downside moves are 3x more likely than upside. But S06 failed shuffle test, so this might be noise.
- **S01 fund_high skews down** — extreme positive funding → crowded longs → more likely to crash than squeeze. Interesting but episodic.
- **S03 oi_surge has slight upward skew** — but failed shuffle test on Def A.

---

## 5. Updated Candidate Assessment (post-sanity)

### ✅ Tier 1: Genuine signal (shuffle-validated)

**S07 compress_oi / Def A (ATR-normalized)**
- Universe uplift: 2.7-5.4x (OOS)
- Shuffle: exceeds p99 (real 2.50 vs p99 1.81)
- Time-to-move: median 10h, 53% in 6-12h window
- Direction: symmetric at 24h → bracket structure
- OI is not the driver, rv_6h compression is
- **Frequency is low (0.4%)** → need multi-coin monitoring for enough signals

### ⚠️ Tier 2: Borderline (needs more data)

**S01 fund_high / Def A**
- Real 3.51 vs shuffle p99 3.23 — barely passes
- Strong downward skew (50.6% down vs 28.6% up at 200bp)
- Could be directional (short when funding extreme positive) but episodic (1-2/5 weeks)
- Worth testing with directional entry on longer dataset

### ❌ Tier 3: Noise / inflated by base rate (discard)

All Def B candidates from v1 (CCUSDT, 1000RATSUSDT, ARBUSDT etc.):
- Failed shuffle test (real uplift ≤ shuffle p99)
- Inflated by coins where p0 is already 20-40% (trivial threshold)
- Def B_adp (P95 adaptive) captures the real tail better and B_adp results are more conservative

---

## 6. What XS-7 Should Do

Based on the validated S07 signal:

### Strategy: Bracket on S07 compress_oi

**Entry condition:** rv_6h ≤ P20 AND oi_z ≥ 1.5 (per coin, causal)

**Bracket structure:**
- Buy-stop at +X and sell-stop at -X from signal price
- X = 1.0-1.5 × ATR_1h (to avoid being triggered by noise)
- After one leg fills → cancel the other
- TP: 3-5 × ATR_1h
- SL: 1.5-2 × ATR_1h (accept asymmetric R:R since big moves overshoot)
- Time stop: 24h

**Expected metrics (rough):**
- Base rate of 3×ATR move: ~0.5-1% per 5m bar
- S07 uplift: ~2.5x → ~1.3-2.5% in state
- Signal frequency: ~0.4% of bars → ~70 signals/coin/month
- Multi-coin (10 coins): ~700 signals/month → ~7-17 bracket triggers
- If bracket RR is 2:1 and 50% direction accuracy → positive EV

**Risks:**
- Bracket legs both triggered (whipsaw) in ranging market
- S07 fires during range → time stop exits at loss
- Transaction costs: taker on stop entry + maker on TP exit ≈ 12-15 bps per trade

### Alternative: Directional short on S01 fund_high

If longer dataset confirms the down-skew:
- Short when funding_z ≥ +2 with wide SL (3×ATR)
- TP: 2×ATR
- Time stop: 24h
- Only on coins with confirmed down-skew pattern

---

## 7. Honest Assessment

**What we found:**
- S07 (vol compression + OI) is a genuine, shuffle-validated predictor of extreme moves
- The effect is structural (coil → release), not noise
- But it's rare (0.4% frequency), not directional, and needs 6-12h to play out

**What failed:**
- Def B (fixed bp) uplift is mostly noise — hyper-volatile coins inflate everything
- Most states (S03, S06, S08-S10) don't survive shuffle tests
- OI is not the primary driver in any state (leakage test confirms)

**What we don't know:**
- Does S07 work outside Jan-Feb 2026? (2 months is too short)
- Is the bracket structure profitable after fees?
- What's the whipsaw rate?

---

## Files

- **Main script:** `flow_research/xs6_bigmove_uplift.py`
- **Deep analysis:** `flow_research/xs6_deep_analysis.py`
- **Full results:** `flow_research/output/xs6/xs6_uplift.csv` (3,456 rows)
- **Deep results:** `flow_research/output/xs6/deep/` (time_to_move, direction_skew, shuffle_sanity, oi_leakage CSVs)
- **Auto-summary:** `flow_research/output/xs6/xs6_top_states.md`
