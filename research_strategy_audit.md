# Funding Rate Arbitrage Strategy — Full Strict Audit

Date: 2026-02-24
Data: Bybit 1h coins, 86 symbols, 106 days (2025-11-10 to 2026-02-24)
Backtest: HOLD strategy, $10,000 notional per position

---

## FINDING 1: CRITICAL — FR Direction (Not a Simple Cash-and-Carry)

### The problem

99.2% of high-|FR| (≥20 bps) events are **negative** funding rates, not positive.

| |FR| threshold | Positive FR | Negative FR |
|---|---|---|
| ≥ 5 bps | 5.1% | 94.9% |
| ≥ 10 bps | 2.2% | 97.8% |
| ≥ 20 bps | 0.8% | **99.2%** |
| ≥ 50 bps | 0.7% | **99.3%** |

### What this means

- **Negative FR**: shorts pay longs → to collect, we need **LONG futures + SHORT spot**
- Short spot = margin borrow the coin, sell it, buy back later
- This is NOT a simple "buy spot, short futures" cash-and-carry
- **Positive-FR-only** (standard cash-and-carry): **$-8/day** — strategy is dead

### Impact

The strategy **requires margin borrowing** for almost every trade. This introduces:
- Borrow interest (~10-36% APR)
- Borrow availability risk (not all coins, not all sizes)
- Borrow recall risk (exchange can force-close your short)

### Backtest validity

The signed-FR backtest correctly handles direction: long-futures trades earn $1,463/day, while the 16 short-futures trades (positive FR) lost $-6/day. The strategy works, but only if you can short spot.

---

## FINDING 2: Fee Calculation Was WRONG — True Cost is ~50 bps, Not 31 or 39

### Correct exchange fees

| Component | Per side | Round trip |
|---|---|---|
| Spot taker fee | 0.100% | 0.200% = **20 bps** |
| Futures taker fee | 0.055% | 0.110% = **11 bps** |
| **Exchange fees total** | | **31 bps** |

### Hidden cost: bid-ask spread (slippage)

Measured from 2 days of Bybit tick data at settlement times (53 coins):

| Liquidity tier | Coins | Futures spread | Spot spread | Total spread |
|---|---|---|---|---|
| Liquid (< 10bp) | 12 | 1.4 bp | 6.2 bp | 7.6 bp |
| Medium (10-20bp) | 21 | 3.1 bp | 11.4 bp | 14.5 bp |
| Illiquid (20-30bp) | 14 | 5.7 bp | 18.2 bp | 23.9 bp |
| Very illiquid (>30bp) | 6 | 12.2 bp | 24.2 bp | 36.5 bp |
| **All coins (mean)** | **53** | **4.6 bp** | **13.7 bp** | **18.3 bp** |
| **All coins (median)** | | | | **16.1 bp** |

The spot spread alone (13.7 bps mean) is larger than the entire futures fee (11 bps). These are altcoins with thin order books.

### Hidden cost: margin borrow interest

For short-spot trades (99.2% of trades):
- 20% APR → 0.055%/day → ~1.5 bps per 6.5h hold
- Negligible individually, but adds up

### Full realistic cost model

| Cost component | bps |
|---|---|
| Exchange fees (spot + futures) | 31.0 |
| Bid-ask spread (median) | 15.0 |
| Margin borrow (6.5h at 20% APR) | 1.5 |
| **Total realistic RT cost** | **~47-50 bps** |

Previous assumptions: 39 bps (overcharged on exchange fees, but missed spread entirely).

---

## FINDING 3: Strategy Still Profitable at Correct Costs

| Cost model | RT bps | Daily P&L | Win rate | Trades |
|---|---|---|---|---|
| Exchange fees only (31 bps) | 31 | **$1,457** | 62% | 712 |
| + median spread (46 bps) | 46 | **$1,357** | 56% | 712 |
| + mean spread + borrow (50 bps) | 50 | **$1,330** | 54% | 712 |
| Conservative (55 bps) | 55 | **$1,297** | 52% | 712 |
| Very conservative (60 bps) | 60 | **$1,263** | 49% | 712 |

At the realistic ~50 bps RT, the strategy earns **$1,330/day** — 5% less than the old $1,404 figure (which used 39 bps), and 9% less than the naive 31 bps figure.

The strategy is **robust to cost assumptions** because the average FR collected per trade (~368 bps) dwarfs even a 60 bps RT cost.

---

## FINDING 4: No Lookahead Bias

### Verified

1. **Entry signal**: uses FR_H (the rate just settled at hour H). This is backward-looking — it's a known, published rate. ✓
2. **FR collected**: starts at H+1 (next settlement). We don't use future FR for entry. ✓
3. **Exit signal**: uses the just-settled FR to decide exit. Known information. ✓
4. **Direction**: FR sign is known at settlement time. No lookahead. ✓

### Potential subtle bias: using |FR| implicitly assumes we know we can take the profitable side

Taking abs(FR) assumes we can always position to collect. In practice this means margin-borrowing for short spot, which may not always be possible for every coin/size. This is an **execution assumption**, not a lookahead bias.

---

## FINDING 5: Low Overfitting Risk — Parameters Are Not Fragile

### Entry threshold sensitivity (exit=8, 3 pos, 50 bps RT)

| Entry threshold | Daily P&L | Win rate |
|---|---|---|
| 10 bps | $1,162 | 33% |
| 15 bps | $1,285 | 46% |
| **20 bps** | **$1,330** | **54%** |
| 25 bps | $1,323 | 58% |
| 30 bps | $1,305 | 60% |
| 50 bps | $1,205 | 73% |
| 100 bps | $912 | 85% |

The P&L curve is **flat between 15-30 bps**. No sharp peak = not overfit.

### Exit threshold sensitivity (entry=20, 3 pos, 50 bps RT)

| Exit threshold | Daily P&L |
|---|---|
| 2 bps | $1,405 |
| 5 bps | $1,377 |
| **8 bps** | **$1,330** |
| 10 bps | $1,278 |
| 15 bps | $1,139 |

Again flat between 2-10 bps. Not fragile.

### In-sample vs out-of-sample (50 bps RT)

| Config | IS (53 days) | OOS (53 days) | Degradation |
|---|---|---|---|
| e=20 x=5 | $1,383 | $1,367 | **-1%** |
| e=20 x=8 | $1,371 | $1,286 | -6% |
| e=30 x=5 | $1,385 | $1,307 | -6% |
| e=30 x=8 | $1,376 | $1,234 | -10% |
| e=50 x=8 | $1,315 | $1,094 | -17% |

Lower entry thresholds show minimal IS/OOS degradation. Higher thresholds (50+) show more degradation because they're fit to specific high-FR regimes. **e=20, x=5 is the most robust config (-1% OOS degradation).**

### Monthly consistency (entry=20, exit=8, 50 bps RT)

All 4 months profitable:
- 2025-11: $310/day
- 2025-12: $405/day
- 2026-01: $421/day
- 2026-02: $314/day

---

## FINDING 6: Futures Premium (Basis) Risk is Negligible

Basis at settlement times (2 days, all 1h coins):
- Mean: -11.9 bps (futures at discount during this bearish sample period)
- Std: 17.0 bps
- Basis drift over 6.5h hold: std ≈ 2 bps = $2 on $10k
- Basis risk is **0.5% of average FR income per trade** ($368)

The basis is mean-reverting (that's what FR does — push futures back toward spot). Not a concern.

---

## FINDING 7: Delta-Neutral Risks Are Real but Manageable

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **Borrow recall** | HIGH — naked long futures | Low for liquid coins | Use liquid coins only, diversify |
| **Margin call on short spot** | HIGH — forced close | Medium on volatile coins | Keep 2× maintenance margin |
| FR sign flip | Medium — paying instead of collecting | 5.7% of trades | Exit threshold catches this (backtest models it) |
| Partial fills | Low — brief directional exposure | Every trade | Submit both legs within <1s |
| Exchange delisting | HIGH — both legs close differently | Very rare | Avoid brand-new listings |

### The biggest real risk: borrow recall

When you short spot via margin, the exchange can recall the borrow at any time. If recalled:
- Your short spot is forcibly closed (you buy back the coin)
- Your long futures position is now **naked** — fully directional
- If price then drops, you lose money

Mitigation: trade only coins with deep borrow pools, and keep alerts to close futures if spot position is recalled.

---

## FINDING 8: Slippage Is the Biggest Hidden Cost

The spot market spread on these altcoins is **13.7 bps mean** — this is the single largest cost component after exchange fees.

Impact on strategy:
- At 31 bps (fees only): $1,457/day
- At 50 bps (fees + realistic spread + borrow): $1,330/day
- Difference: **-$127/day (-8.7%)**

### Can we reduce spread cost?

1. **Use limit orders instead of market orders**: could recover 50-70% of spread
   - Risk: non-fill, partial fill, timing slippage
   - Both legs must fill within the same settlement window
2. **Filter for liquid coins only**: 25 coins with <15bp spread
   - But this limits the opportunity set
3. **Use maker fees instead of taker**: Bybit spot maker = 0.1%, futures maker = 0.02%
   - Spot maker = same as taker on Bybit VIP0 (both 0.1%)
   - Futures maker = 0.02% vs 0.055% taker → saves 7 bps RT on futures

---

## CORRECTED STRATEGY NUMBERS

### Realistic scenario (50 bps RT, entry ≥ 20, exit < 8, 3 positions)

| Metric | Old (39 bps) | Corrected (50 bps) | Change |
|---|---|---|---|
| Daily P&L | $1,404 | **$1,330** | -5.3% |
| Win rate | 59% | **54%** | -5 pp |
| Avg winner | $378 | **$401** | +6% |
| Avg loser | -$28 | **-$36** | -29% |
| Winner/loser ratio | 13.5× | **11.1×** | -18% |

### Most robust config: entry ≥ 20, exit < 5

At 50 bps RT: **$1,377/day, 58% WR, -1% OOS degradation**

This is the recommended config — slightly tighter exit, minimal IS/OOS gap.

---

## FINDING 9: CRITICAL — Most Backtest Coins Are NOT Tradeable

### The problem

The backtest trades 86 coins on Bybit. But checking against the LIVE Bybit API:

| Filter | Coins | % of 86 |
|---|---|---|
| Have futures (in backtest) | 86 | 100% |
| Have spot pair on Bybit | 48 | 56% |
| **Have margin trading (can short spot)** | **28** | **33%** |

Since 99.2% of entries require **short spot** (negative FR), we need margin borrowing. Only 28 of 86 coins support this.

### Impact on P&L

| Coin filter | Trades | Daily P&L | % of backtest |
|---|---|---|---|
| All coins (unrestricted backtest) | 712 | $1,330 | 100% |
| Only coins with spot pair | 515 | $617 | 46% |
| **Only margin-enabled (actually tradeable)** | **292** | **$320** | **24%** |

**The backtest overstates P&L by 77%.** 55% of the reported P&L came from coins with NO spot pair at all.

### P&L attribution by tradeability

| Category | Total P&L (106 days) | % of total |
|---|---|---|
| Margin-enabled (tradeable) | $31,947 | 22.6% |
| Spot but no margin (untradeable for neg FR) | $31,938 | 22.6% |
| No spot pair at all (impossible) | $77,662 | **54.9%** |

### Top contributing coins — mostly untradeable

| Coin | Total P&L | Spot? | Margin? |
|---|---|---|---|
| RIVERUSDT | +$22,197 | ✗ | ✗ |
| PIPPINUSDT | +$11,355 | ✗ | ✗ |
| BEATUSDT | +$7,514 | ✗ | ✗ |
| SENTUSDT | +$7,401 | ✓ | ✗ |
| LAUSDT | +$5,393 | ✓ | ✓ ← actually tradeable |
| AXSUSDT | +$4,748 | ✓ | ✓ ← actually tradeable |

### Why this happens

New coin listings often launch with **futures only** (no spot or margin). These coins tend to have extreme funding rates because:
- High speculative demand on futures
- No spot market to arbitrage against → FR stays extreme longer
- Our backtest captures these "impossible" trades

### The true strategy performance

**With only margin-enabled coins: $320/day on $60k capital = 195% annual ROI.**

Still profitable, but dramatically less than the $1,330/day headline.

---

## SUMMARY: Audit Verdict

| Question | Verdict | Details |
|---|---|---|
| Lookahead bias? | ✓ **None** | Uses settled FR only; direction known at entry |
| Overfitting? | ✓ **Low risk** | Flat parameter surface; 1% IS/OOS gap at best config |
| Futures premium risk? | ✓ **Negligible** | 2 bps std vs 368 bps avg income |
| Slippage? | ⚠ **Significant** | 15-18 bps hidden cost from spread (esp. spot) |
| Delta-neutral risk? | ⚠ **Moderate** | Borrow recall is the main risk; FR flips handled |
| Fee calculation? | ✗ **Was wrong** | True RT = ~50 bps (31 fees + 15 spread + 1.5 borrow) |
| Tradeability? | ✗ **CRITICAL** | Only 28/86 coins have margin; backtest overstated by 77% |
| **Strategy still profitable?** | **⚠ YES, but** | **$320/day realistic (not $1,330)** |

### Key corrections from this audit

1. **77% of backtest P&L is untradeable** — most high-FR coins lack spot/margin pairs on Bybit
2. **FR is overwhelmingly negative** (99.2% of entries) → requires margin borrowing for short spot
3. **True RT cost is ~50 bps**, not 31 or 39 — spot spread is the biggest hidden cost
4. **Realistic daily P&L: ~$320** with 28 margin-enabled 1h coins (not $1,330)
5. **Borrow recall risk** is the most dangerous operational risk
6. **Best config shifts to tighter exit** (exit < 5 instead of < 8) at higher costs
7. **Multi-exchange expansion** may help recover some P&L if Binance/OKX have more margin pairs
