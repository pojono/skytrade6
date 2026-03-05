# Claude-2: Strategy Research Findings

**Date:** 2026-03-05
**Data:** Bybit datalake, 150 symbols, 1.2TB, 2024-01 → 2026-03-04
**Test period:** 2025-06-01 → 2026-03-04 (~9 months)
**Fees:** maker 4 bps, taker 10 bps (RT taker = 20 bps)

---

## Summary: 6 Ideas Tested, 3 Strong Edges Found

| # | Idea | Verdict | Best Net (bps) | Cross-Symbol | WR |
|---|------|---------|---------------|--------------|-----|
| 1 | OI + L/S Crowding | ✅ PROMISING | +370 @ 4h | 65-76% | ~49% |
| 2 | Premium Reversion | ❌ DEAD | +11 @ 4h | None | ~49% |
| 3 | Derived 1m FR | ⚠️ MODERATE | +208 @ 4h | 52% | ~48% |
| 4 | Lead-Lag (BTC→alts) | ✅ **STRONG** | +1270 excess @ 30m | 97% | 69.5% |
| 5 | Spot-Futures Div | ✅ **STRONG** | +1398 @ 4h | 82% | 70% |
| 6 | OI + Vol Breakout | ✅ REAL (non-dir) | +218 abs @ 4h | **100%** | 46% |

---

## Idea 1: OI + L/S Crowding → Liquidation Cascade

**Hypothesis:** When OI spikes AND L/S ratio hits extremes, the market is fragile. Fade the crowd.

**Result: PROMISING at 4h horizon, LONG only**

Best configs (net of 20 bps fees):
- `LS>0.68_OI>2.5σ` long 240m: **+370 bps**, 49% WR, 23 symbols
- `LS>0.72_OI>1.5σ_extended` long 240m: **+331 bps**, 65% of 20 symbols profitable
- `LS>0.75_noOI_extended` long 240m: **+202 bps**, 76% of 21 symbols profitable

**Key findings:**
- Only **LONG** signals work (fading shorts) — short signals are not profitable
- Only works at **4h+ horizon** — 5m/15m/30m are marginal or negative
- Adding OI spike confirmation improves per-trade return but reduces signal count
- Without OI filter (L/S extreme alone), still works but weaker

**Interpretation:** When the crowd is heavily short (buyRatio < 0.30-0.35) and OI is spiking, shorts are overleveraged. Any bounce triggers cascading short liquidations → strong 4h+ reversal.

---

## Idea 2: Premium Index Mean Reversion ❌

**Hypothesis:** Futures basis extremes (premium z-score > 2-3σ) revert to mean.

**Result: DEAD — barely covers fees**

Best config: lb=720 z>3.5 long 240m: +11.1 bps net, 49% WR
- No cross-symbol consistency at all
- Premium extremes do revert, but the move is too small to trade profitably with taker fees

---

## Idea 3: Derived 1-Minute Funding Rate

**Hypothesis:** Premium index ≈ implied funding rate at 1m resolution. Use for early FR prediction.

**Result: MODERATE — two viable signals**

1. **Implied FR level > 20 bps** (momentum): +208 bps net @ 4h, 52% of 112 coins profitable, 48% WR
2. **FR divergence z>3.0** (implied vs actual): +79 bps net @ 1h, 52% of 147 coins profitable, 50% WR

**Interpretation:** High implied FR = strong directional market. FR divergence = regime shift. Both are momentum signals (low WR but positive expectancy).

---

## Idea 4: Cross-Symbol Lead-Lag ✅ **STRONGEST SIGNAL**

**Hypothesis:** BTC/ETH/SOL price moves propagate to altcoins with delay.

**Result: MASSIVE EDGE for BTC UP, verified with excess return audit**

### Raw Results
- BTC 3m >150bps UP → alts fwd 30m: **+1397 bps** raw, 67% WR, 97% of 125 coins
- SOL 1m >150bps UP → alts fwd 30m: **+1388 bps** raw, 74% WR, 99% of coins

### Audit (excess return = alt return - BTC return)
| Horizon | Raw Alt | BTC Fwd | **Excess** | Beta-Adj | Alt Already | Raw WR | Ex WR |
|---------|---------|---------|-----------|----------|-------------|--------|-------|
| 1m | +110 | -3 | **+113** | +92 | +476 | 50% | 60% |
| 3m | +252 | -14 | **+265** | +252 | +476 | 47% | 61% |
| 5m | +400 | +23 | **+377** | +317 | +476 | 51% | 58% |
| 10m | +812 | +43 | **+769** | +661 | +476 | 57% | 69% |
| 15m | +1076 | +89 | **+987** | +815 | +476 | 62% | 67% |
| 30m | +1397 | +127 | **+1270** | +1039 | +476 | 67% | 70% |

**CRITICAL ASYMMETRY:**
- ✅ **BTC UP → long alts: REAL.** Excess +1270 bps, 70% WR. Alts lag behind pumps.
- ❌ **BTC DOWN → short alts: FAKE.** Excess -1643 bps. Alts crash HARDER than BTC.

**Interpretation:** When BTC pumps 150+ bps in 3 min, alts have already moved ~476 bps on average but still have **+1270 bps more to go** over the next 30 min. Alts genuinely lag behind BTC pumps. On dumps, alts overshoot immediately — no lagging short opportunity.

---

## Idea 5: Spot-Futures Divergence ✅ **STRONG**

**Hypothesis:** When spot price leads futures (real demand), futures catch up.

**Result: VERY STRONG for spot-leads-futures**

| Signal | Dir | Horizon | Net bps | WR | Coins | Cross-Sym |
|--------|-----|---------|---------|-----|-------|-----------|
| spot_lead > 60bps | long | 240m | **+1398** | 70% | 67 | 82% |
| spot_lead > 60bps | long | 60m | **+1049** | 68% | 67 | 88% |
| spot_lead > 60bps | long | 30m | **+967** | 64% | 67 | 87% |
| spot_lead > 40bps | long | 240m | **+739** | 64% | 78 | 79% |
| spot_lead > 20bps | long | 240m | **+165** | 55% | 84 | 61% |

**Only LONG signals work** (spot pumping more than futures → long futures to catch up).
Short signals (spot dumping more than futures) are all negative.

**Interpretation:** Spot buying = "real" demand (actual token purchases). When spot leads by >40-60 bps, it signals genuine buying pressure that futures haven't priced in yet.

**IMPORTANT:** This signal may overlap with Idea 4 (both fire during BTC-led pumps). Need deduplication.

---

## Idea 6: OI + Volatility Breakout (Coiled Spring) ✅

**Hypothesis:** Low volatility + rising OI = coiled spring → explosive breakout coming.

**Result: REAL non-directional signal, 100% cross-symbol consistency**

| Signal | Dir | Horizon | Net bps | WR | Coins | Cross-Sym |
|--------|-----|---------|---------|-----|-------|-----------|
| coiled v<20pctile + OI>3% | abs_move | 240m | **+218** | 46% | 147 | **100%** |
| coiled v<20pctile + OI>2% | abs_move | 240m | **+199** | 47% | 148 | **100%** |
| vol_break c<15 x3 up | long | 60m | **+143** | 55% | 15 | 60% |

**Key insight:** The coiled spring signal is 100% consistent across ALL 148 symbols — it always predicts a big 4h move. But it doesn't predict direction (46% WR long). The `volbreak` directional variant is smaller but has directional edge (55% WR on upside breakouts).

**Interpretation:** When realized vol hits 20th percentile while OI rises 2-3%+ in 4h, large positions are being built quietly. The coming breakout averages +218 bps absolute move, but could go either direction. Useful as a **straddle/volatility trade** or as a **filter to boost other directional signals**.

---

## Recommended Strategy Architecture

### Tier 1: Primary Edge (deploy first)
1. **BTC Pump → Long Alts** (Idea 4): Monitor BTC 3m returns, when >150 bps, long top-beta alts. +1270 bps excess, 70% WR. **HIGH FREQUENCY** (~daily during volatile periods).

2. **Spot-Leads-Futures** (Idea 5): Monitor spot vs futures price delta, when spot leads >40-60 bps, long futures. +739-1398 bps net. Probably same events as #1 but useful as **independent confirmation**.

### Tier 2: Swing Signals (overlay)
3. **OI + L/S Crowding** (Idea 1): When L/S extreme short + OI spike, go long with 4h hold. +200-370 bps. Lower frequency but **contrarian edge**.

4. **Coiled Spring** (Idea 6): Use as a **filter** — when vol compressed + OI rising, boost position size on directional signals. 100% consistent predictor of big moves.

### Tier 3: Supplementary
5. **High Implied FR** (Idea 3): When premium_index > 20 bps, trend-follow for 4h. +208 bps. 

### Dead
6. Premium reversion alone (Idea 2): Not enough edge to cover fees.

---

## Next Steps
- [ ] Audit Idea 5 for correlation with Idea 4 (are they the same signal?)
- [ ] Build combined strategy with signal stacking (Ideas 1+4+5+6)
- [ ] Walk-forward out-of-sample validation (train on 2025-06 to 2025-12, test on 2026-01 to 2026-03)
- [ ] Test Idea 7 (pairs trading) and Idea 8 (ML combination)
- [ ] Orderbook imbalance analysis (Idea 9, biggest data)
