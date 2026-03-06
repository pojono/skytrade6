# Phase 12: Dynamic Leverage

**Date:** 2026-03-06
**Script:** `research_cross_section/phase12_dynamic_leverage.py`

---

## Objective

Test whether scaling leverage dynamically reduces drawdowns without hurting Sharpe:
1. **Vol-targeting**: scale gross exposure to target 40% annualized portfolio vol
2. **Drawdown scaling**: cut leverage to 50% at -10% DD, 25% at -20% DD
3. **Combined**: both applied together

---

## Results (full dataset 2024–2026, No-Majors universe)

| Variant | Sharpe | Ann Ret | MaxDD | $1k→ |
|---------|--------|---------|-------|------|
| Baseline (1x) | 2.177 | +405% | -55.1% | $6,657 |
| Vol-target (40% ann) | 1.674 | +99% | -30.5% | $2,239 |
| DD-scaling | 1.882 | +166% | -33.0% | $3,139 |
| Combined (VT + DD) | 1.443 | +63% | -29.2% | $1,770 |

---

## Key Finding: Dynamic Leverage is Counterproductive

All three leverage variants **reduce Sharpe** despite reducing MaxDD.

**Why?** This strategy's return distribution is highly positively skewed:
- ~70% of all profit comes from 3–4 monster months (Sep/Oct 2025, Jan 2026)
- These monster months are also high-volatility months
- Vol-targeting scales DOWN exactly during those months (high recent vol → low leverage)
- You lose the bulk of alpha by reducing leverage when it's most profitable

This is the opposite of what vol-targeting is designed for. It works for strategies with symmetric, persistent alpha. It fails for strategies with burst-return profiles.

**Drawdown scaling** has a similar problem: a -10% drawdown often precedes a strong recovery, and cutting to 50% at that point means you capture less of the rebound.

---

## Conclusion

**Reject dynamic leverage for this strategy.**

The correct drawdown management is:
- Structural regime filter (gates on signal quality, not recent P&L)
- Hard stop at -20% DD → cut gross 50% (per risk management rules)
- These are separate from dynamic leverage; they're one-time emergency responses

Do not implement rolling vol-targeting or continuous DD-scaling.
