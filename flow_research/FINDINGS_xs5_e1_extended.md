# XS-5 E1 Extended: Crowded Long → SHORT — 5-Month Validation

**Date:** 2026-03-03
**Script:** `flow_research/xs5_e1_extended.py`
**Data:** 58 Bybit perps, 2025-10-01 → 2026-02-28 (152 days, 50 symbols with Oct data)

---

## VERDICT: NO-GO ❌

E1 has directional content (+49bp mean, 66% WR at 12h) but fails on statistical significance
(p=0.34) and has catastrophic tail risk (P5 = -763bp). Not tradeable as-is.

---

## 1) Event Summary

- **32 E1 events** in 5 months across 58 symbols (~6.4/month, ~0.2/day)
- 14 unique symbols triggered E1
- Top: ARCUSDT(6), COAIUSDT(4), HANAUSDT(4), HUSDT(3), BTRUSDT(3), 1000RATSUSDT(3)

### Monthly Distribution

| Month | Events | Mean Net (12h) | WR |
|-------|-------:|---------------:|---:|
| 2025-10 | 5 | +40bp | 60% |
| 2025-11 | 12 | -44bp | 50% |
| 2025-12 | 4 | -102bp | 75% |
| 2026-01 | 1 | +5bp | 100% |
| 2026-02 | 10 | +231bp | 80% |

**Feb 2026 dominates.** 10 of 32 events are in Feb, and Feb alone contributes +2,310bp total.
Without Feb: 22 trades, mean = -34bp. The earlier xs5 result (+211bp) was a Feb artifact.

---

## 2) Best Config Results (slip=5bp, no SL/TP)

| Hold | N | Mean Net | Median | WR | PF | MFE | MAE | p-value | CI 95% |
|------|---|---------|--------|-----|-----|-----|-----|---------|--------|
| 4h | 32 | -54bp | +3bp | 53% | 0.66 | +425bp | -294bp | 0.79 | [-197, +100] |
| 12h | 32 | **+49bp** | +26bp | **66%** | **1.40** | +481bp | -312bp | 0.34 | [-104, +209] |
| 24h | 32 | +70bp | +26bp | 66% | 1.57 | +553bp | -312bp | 0.35 | [-101, +248] |
| 48h | 32 | +70bp | +26bp | 66% | 1.57 | +553bp | -312bp | 0.32 | [-98, +249] |

- 12h and 24h identical after 12h because **all trades exit via unwind** (31/32) or time_stop (1/32)
- Mean hold time: ~220 minutes (~3.7h) — funding_z normalizes fast
- 4h is worse because some trades haven't unwound yet

---

## 3) The Catastrophic Trades

Three trades destroy the edge:

| Symbol | Date | Net | MAE | What happened |
|--------|------|----:|----:|---------------|
| HUSDT | 2025-11-05 | **-911bp** | -1198bp | fz=+9.2, oiz=7.3 — massive pump despite extreme long crowding |
| CCUSDT | 2025-11-17 | **-876bp** | -897bp | fz=+2.2, oiz=11.2 — OI explosion continued, squeeze UP |
| HANAUSDT | 2025-12-29 | **-671bp** | -818bp | fz=+4.3, oiz=2.3 — gap up, funding_z unwound via price rising |

These 3 trades account for -2,458bp. The other 29 trades average **+135bp** and have 72% WR.

**Root cause:** When OI acceleration is extreme (oi_z > 7) AND funding is extreme, the crowd
is sometimes right — price continues in their direction via short squeeze / momentum cascade.

---

## 4) cat8 SL Results

Catastrophe SL at 8×ATR doesn't help — it makes things worse:

| Hold | Config | Mean Net | WR | PF |
|------|--------|---------|-----|-----|
| 12h | none | +49bp | 66% | 1.40 |
| 12h | cat8 | -36bp | 59% | 0.77 |

The SL clips the big winners but doesn't save from the catastrophic losses
(which happen fast, within 1h, and blow past 8×ATR).

---

## 5) Why the Earlier xs5 Result Was Misleading

The original xs5 found E1: N=11, mean=+211bp, WR=82%. This was because:
1. Only Jan-Feb data — missed the Nov-Dec disasters entirely
2. 10 of 11 events were in Feb 2026, which was the best month
3. Jan had only 1 event (trivially +5bp)

With 5 months: N=32, mean=+49bp, WR=66%, p=0.34. **Not significant.**

---

## 6) Potential Filters (NOT tested, for future work)

To cut the catastrophic tail:
- **Cap oi_z at 5** — the worst losses all had oi_z > 7 (HUSDT 7.3, CCUSDT 11.2)
- **Require negative 4h return** — add `ret_4h < 0` to confirm reversal has started
- **Turnover filter** — require turnover declining (exhaustion, not momentum)
- **Cross-symbol regime** — don't short when BTC/ETH are pumping (correlation drag)

But adding filters to N=32 is pure overfitting territory. Need 6+ months more data.

---

## 7) Conclusion

E1 (Crowded Long → SHORT) shows a **real but weak** directional signal:
- Mean +49bp at 12h hold, 66% WR, PF 1.40
- But p=0.34, CI includes zero, and 3 catastrophic losses dominate the risk
- The signal is **regime-dependent**: works in risk-off months (Feb 2026), fails in momentum months (Nov 2025)
- Not tradeable without either (a) 100+ events for statistical power, or (b) a reliable catastrophe filter

**Status: DEAD for now. Revisit with 12+ months of data.**
