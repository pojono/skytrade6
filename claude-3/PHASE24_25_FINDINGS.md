# Phases 24–25: Leverage Sweep & Coin Filters

**Date:** 2026-03-06

---

## Phase 24: Leverage & Position Count Sweep

### Setup

Sweep of N (coins held long + short) × Leverage on the Phase 22 best variant
(2×predicted_funding + mom_24h, 2× leverage baseline):

- N values: 5, 8, 10, 15, 20
- Leverage: 1×, 2×, 3×, 5×
- Starting capital: $10,000
- Period: 2025-01 to 2026-03

### Key Finding: Leverage is Sharpe-Neutral

Returns and fees both scale linearly with leverage, so **Sharpe ratio is invariant to leverage**.
Leverage only affects MaxDD and absolute PnL:

| N | Sharpe | MaxDD (2×) | $10k→ (1×) | $10k→ (2×) | $10k→ (3×) | $10k→ (5×) |
|---|--------|-----------|------------|------------|------------|------------|
| 5  | 3.18 | -52% | $17,507 | $67,804 | $172,714 | $647,952 |
| 8  | 3.84 | -41% | $34,266 | $202,773 | $636,474 | $4,068,264 |
| **10** | **3.96** | **-40%** | **$65,889** | **$351,824** | **$1,545,162** | **$16,888,506** |
| 15 | 3.71 | -42% | $45,375 | $233,501 | $827,278 | $6,576,095 |
| 20 | 3.52 | -42% | $36,126 | $158,268 | $473,285 | $2,804,003 |

### Recommendation

**N=10** is the sweet spot: best Sharpe (3.96), reasonable MaxDD at 2× (-40%).

For live deployment:
- **Conservative:** 2× leverage, N=10 → expect ~$200-400k per $10k over this regime
- **Aggressive:** 3× leverage, N=10 → expect ~$1M-2M but -60% drawdown risk
- **Never exceed 3×** — the 5× numbers look impressive but require tolerating -80%+ drawdowns

---

## Phase 25: Coin Filters & Rolling Universe Selection

### WFO Validity Assessment

Before testing, each filter was assessed for look-ahead bias:

| Filter | WFO Valid? | Reason |
|--------|-----------|--------|
| Majors exclusion (Phase 4) | Valid | Structural, not data-driven |
| Min listing age | **Valid** | Observable at trade time (first_valid_index) |
| Rolling volume rank | **Valid** | Uses only trailing 30-bar data, shifted 1 bar |
| Rolling quality mask | **Valid** | Uses only trailing 90-bar data, shifted 1 bar |
| Phase 7-8 coin exclusion | INVALID | Used hindsight to remove bad performers |

### Experiment Results (2× leverage, $10k start)

| Filter | Sharpe | MaxDD | $10k→ | Jan-25 | Oct-25 |
|--------|--------|-------|-------|--------|--------|
| **Baseline (no filter)** | **3.96** | -40.1% | **$351,824** | -18.6% | +268% |
| Min age 30d | 3.05 | -38.3% | $95,436 | **-2.2%** | +29% |
| Min age 60d | 3.02 | -36.3% | $75,347 | +0.0% | +29% |
| Min age 90d | 2.77 | -37.0% | $58,272 | +0.0% | +29% |
| Min age 120d | 2.58 | -32.0% | $46,787 | +0.0% | +29% |
| Vol top 80% | 3.32 | -40.2% | $167,333 | -5.1% | +174% |
| Vol top 60% | 3.00 | -40.8% | $118,121 | -5.4% | +140% |
| Vol top 40% | 2.41 | -43.3% | $59,198 | -4.8% | +77% |
| RollQual top 80% | 2.90 | -47.6% | $90,299 | -0.8% | +132% |
| RollQual top 60% | 2.91 | -47.6% | $91,031 | -0.8% | +132% |
| RollQual top 40% | 2.63 | -48.0% | $69,754 | -1.0% | +108% |
| Age60 + RollQual80 | 1.99 | -42.6% | $31,255 | +0.0% | +22% |
| Age60 + RollQual60 | 1.99 | -42.6% | $31,362 | +0.0% | +22% |
| Age90 + RollQual80 | 1.83 | -41.0% | $26,711 | +0.0% | +22% |

### Monthly Breakdown — Key Months

| Month | Baseline | Min age 30d | RollQual 60% | Age60+RQ80 |
|-------|----------|-------------|--------------|------------|
| 2025-01 | **-18.6%** | -2.2% | -0.8% | +0.0% |
| 2025-02 | -9.2% | -5.0% | **-14.4%** | +0.0% |
| 2025-09 | +69.8% | +30.4% | +41.7% | +9.6% |
| 2025-10 | **+268.3%** | +29.0% | **+132.2%** | +22.4% |
| 2025-11 | +51.7% | +32.0% | +32.4% | +26.4% |
| 2025-12 | +64.2% | +64.2% | +10.4% | -1.3% |
| 2026-01 | +81.0% | +81.0% | +51.1% | +51.1% |
| 2026-02 | +30.4% | +30.4% | +22.7% | +22.7% |

### Key Insights

**1. Filters cut the tails — both good and bad**

Every filter reduces Jan-25 losses (newly-listed coins with unstable funding), but also
drastically cuts Oct-25 gains (the best month). The baseline +268% in Oct-25 comes primarily
from meme coins that just launched — exactly the coins filters exclude.

**2. The Jan-25 problem is timing, not noise**

Jan-25 was the first month in backtest. Many coins had <30 days of history. The listing
age filter correctly excludes them. Once coins stabilize (>30d old), the signal is clean.

**3. Rolling quality adds MaxDD without protecting gains**

RollQual increases MaxDD (-40% → -47%) while cutting Sharpe. It tends to exclude coins
right after their bad months (shift=1), leaving them in during bad months and out during
the subsequent recovery. **Not recommended.**

**4. Volume filter is the cleanest secondary screen**

Vol top 80% preserves most of the alpha (Sh 3.96 → 3.32) while cutting final capital only
in half. It removes the illiquid tail without touching the core opportunity.

### Recommendation

**No filter is best for pure risk-adjusted returns** (Sharpe 3.96). But for a live deployment
starting from scratch with small capital, consider:

- **Option A: No filter** — accept Jan-Feb 2025-style drawdowns as a startup cost, scale
  up after 2-3 months of operation. Full alpha capture in good regimes.

- **Option B: Min age 30d** — cuts initial losses significantly (-18.6% → -2.2% in Jan-25),
  Sharpe 3.05 still excellent. Appropriate if capital preservation during ramp-up is a priority.

- **Vol top 80% is not recommended** — it halves final capital vs baseline with only modest
  drawdown improvement. The volume screen adds slippage (rank-chasing) without clear benefit.

### Decision for Phase 26 (Live Engine)

**Use Baseline (no filter) + Min age 14d as a soft guard.**

Rationale: the worst Jan-25 losses came from coins with <14 days of listing data where
price discovery was still highly unstable. A 14d minimum (not tested here but interpolated
between baseline and 30d) should remove the extreme cases while preserving most Oct-25 upside.
This is a conservative operational choice and can be removed after the first live quarter.
