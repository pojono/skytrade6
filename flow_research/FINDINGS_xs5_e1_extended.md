# XS-5 E1 Extended: Crowded Long → SHORT — 8-Month Validation

**Date:** 2026-03-03
**Script:** `flow_research/xs5_e1_extended.py`
**Data:** 58 Bybit perps, 2025-07-01 → 2026-02-28 (8 months, 24 symbols with E1 events)

---

## VERDICT: NO-GO ❌ (DEFINITIVE)

With 60 E1 events over 8 months, the signal is **dead**.
12h hold: mean = -1bp, PF = 0.99, p = 0.50. Only 2 of 8 months positive.

---

## 1) Event Summary

- **60 E1 events** in 8 months across 24 symbols (~7.5/month, ~0.25/day)
- Top: ARCUSDT(9), 1000RATSUSDT(6), FARTCOINUSDT(6), COAIUSDT(5), HUSDT(5), HANAUSDT(4)

### Monthly Distribution (12h hold, slip=5bp)

| Month | Events | Mean Net |
|-------|-------:|---------:|
| 2025-07 | 13 | -16bp |
| 2025-08 | 8 | +95bp |
| 2025-09 | 7 | -64bp |
| 2025-10 | 5 | -308bp |
| 2025-11 | 12 | -44bp |
| 2025-12 | 4 | -102bp |
| 2026-01 | 1 | +5bp |
| 2026-02 | 10 | +231bp |

Only 2 of 8 months clearly positive (Aug, Feb). Oct is catastrophic (-308bp).
The earlier results (+49bp at 5mo, +211bp at 2mo) were **sampling artifacts**.

---

## 2) Best Config Results (slip=5bp, no SL/TP)

| Hold | N | Mean Net | WR | PF | p-value |
|------|---|---------|-----|-----|--------|
| 4h | 60 | **-49bp** | 47% | 0.69 | 0.85 |
| 12h | 60 | **-1bp** | 53% | 0.99 | 0.50 |
| 24h | 60 | +11bp | 53% | 1.07 | 0.46 |
| 48h | 60 | +11bp | 53% | 1.07 | 0.44 |

No hold window produces a meaningful edge. All p-values > 0.4.

---

## 3) Progressive Sample Size Degradation

As we added more data, the "edge" disappeared:

| Period | N | Mean Net (12h) | WR | PF | p |
|--------|---|---------------|-----|-----|---|
| Jan-Feb 2026 (original xs5) | 11 | +211bp | 82% | 4.30 | — |
| Oct 2025 – Feb 2026 (5mo) | 32 | +49bp | 66% | 1.40 | 0.34 |
| **Jul 2025 – Feb 2026 (8mo)** | **60** | **-1bp** | **53%** | **0.99** | **0.50** |

Classic overfitting mirage: small sample showed strong signal, each data extension diluted it.

---

## 4) Why E1 Doesn't Work

1. **Crowded longs are often RIGHT** — high funding + rising OI often means genuine demand, not just degen leverage. Price continues up in >50% of cases.
2. **Catastrophic losses are unavoidable** — JELLYJELLYUSDT -1690bp, HUSDT -911bp, CCUSDT -876bp. These are short squeezes where being short against the crowd is exactly wrong.
3. **The "stall" filter (trend_2h ≤ 0.3) is too weak** — it fires during brief consolidation within an uptrend, not at actual exhaustion.
4. **Regime dependence** — only works in risk-off months (Feb 2026 market-wide correction). Useless as a standalone signal.

---

## 5) Conclusion

**E1 is definitively dead.** 60 events over 8 months, mean = -1bp, PF = 0.99, p = 0.50.
The original +211bp result was a Feb 2026 sampling artifact.

No amount of filter tweaking can save N=60 with zero base edge.
The thesis ("crowded longs + stalling price → short") does not hold on altcoin perps.

**Status: DEAD. No further work needed.**
