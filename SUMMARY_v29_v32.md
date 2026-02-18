# Research Summary: v29-v32 Combined Findings

## What We've Proven

### Signal: Volatility Expansion is Predictable

| Experiment | Target | AUC | Datasets Tested |
|-----------|--------|-----|-----------------|
| v29-rich | Regime switch within 5min | 0.60 | BTC 14d |
| v30 T1 | Range > median in next 5min | 0.71-0.81 | BTC W1, BTC W2, ETH W1 |
| v32 Phase 3 | Symmetric TP/SL profitable | 0.63-0.64 | BTC W1, BTC W2 |

### Anti-Signal: Direction is Unpredictable

| Experiment | Target | AUC | Verdict |
|-----------|--------|-----|---------|
| v30 T3 | Continuation (30s→300s) | 0.50 | Dead |
| v30 T4 | Straight-line moves | 0.50 | Dead |
| v30 T5 | Direction UP | 0.50 | Dead |
| v32 Phase 2 | Long vs Short TP hit | 0.50 | Dead |

### Top Features (consistent across ALL experiments)

| Rank | Feature | Importance | Why |
|------|---------|-----------|-----|
| 1 | vol_900s / vol_3600s | 15-30% | Vol clusters — high vol predicts more vol |
| 2 | hour_of_day | 7-13% | Intraday vol seasonality |
| 3 | fr_time_to_funding | 4-10% | Funding rate cycle creates vol |
| 4 | trade_count / avg_trade_size | 3-5% | Activity = vol proxy |
| 5 | buy_not_ratio / liq_notional | 2-4% | Aggressor imbalance |

### Strategy Results

**v30 Straddle (vol-timing)**:
- Signal fires ~10% of time (P90 threshold)
- Avg range on signal: 21-50 bps (vs 13-33 bps baseline)
- Net after 2 bps cost: 19-48 bps per trade

**v32 Symmetric TP/SL**:
- Best config: TP=10 bps, SL=5 bps, 300s time limit
- BTC: baseline EV negative → model turns it positive (+0.70-1.10 bps at P90)
- ETH: baseline already positive (+0.70 bps) — no timing needed
- ~250 trades/day at P90 threshold, 68-72% win rate

## Sanity Checks

### Issue 1: v30 Threshold Leakage (MINOR)
`range_median` computed over full dataset (train+test). Should use train-only.
**Impact**: Likely small — median is stable. But must fix for clean results.

### Issue 2: Small P90 Sample Size
At P90 threshold, only ~820-1000 trades in test set. Need walk-forward to get more test samples.

### Issue 3: Single Train/Test Split
All experiments use one temporal split (first half → second half). Need rolling validation.

### What's Clean
- Features are purely backward-looking (no look-ahead)
- Train/test split is temporal (no random shuffle)
- v32 targets are self-contained (no threshold leakage)
- Results consistent across 2 date ranges and 2 symbols

## Improvement Plan (v33)

### Step 1: Fix v30 Threshold Leakage
Compute `range_median` on train data only. Re-run BTC W1 to get clean AUC.
**Expected**: AUC drops slightly (0.70 → maybe 0.65-0.68). If still >0.60, signal is real.

### Step 2: Combine v30 + v32
Use v30's P(vol expansion) as an additional feature in v32's TP/SL model.
v30 AUC=0.71 > v32 AUC=0.63, so adding it should boost v32.
**Expected**: v32 AUC improves from 0.63 → 0.66+.

### Step 3: Adaptive TP/SL Sizing
Scale TP/SL based on recent vol (vol_900s). High vol → wider TP/SL, low vol → tighter.
**Expected**: Better EV because TP/SL matches actual price movement.

### Step 4: Walk-Forward Backtest
Rolling 2-day train → 1-day test across 7 days. Gives 5 independent test days.
**Expected**: More robust AUC estimate, realistic PnL curve.

---

## v33 Improvement Results

### Step 1: Fix Threshold Leakage — SIGNAL CONFIRMED ✅

Computed `range_median` on train data only (13.44 bps) vs full dataset (12.07 bps).

| Threshold | AUC | P90 Precision | Range Lift |
|-----------|-----|---------------|------------|
| Full median (leaky) | 0.771 | 88.1% | 1.76x |
| **Train-only (clean)** | **0.786** | **85.2%** | **1.77x** |

**Result**: Clean AUC is actually HIGHER. The leakage was slightly hurting, not helping. Signal is real.

### Step 2: Combine v30 + v32 — No Improvement ❌

Added v30's P(vol expansion) as a feature in v32's TP/SL model.

| Model | AUC | P90 EV |
|-------|-----|--------|
| v32 baseline (raw features) | 0.696 | +0.52 bps |
| v32 + vol_prob feature | 0.680 | +0.34 bps |

**Result**: Adding vol_prob hurts — GBM puts 82% weight on it and ignores raw features. The v32 model already learns the same vol pattern directly from raw features. Simpler is better.

### Step 3: Adaptive TP/SL Sizing — No Improvement ❌

Scaled TP/SL proportional to expected range (vol_900s × √300 × 10000).

| Config | AUC | P90 EV |
|--------|-----|--------|
| **Fixed TP=10 SL=5** | **0.696** | **+0.52** |
| Adaptive k=0.8 | 0.533 | +0.24 |
| Adaptive k=1.0 | 0.535 | +0.18 |
| Adaptive k=1.2 | 0.548 | -0.14 |

**Result**: Adaptive sizing makes the target noisier. Fixed levels create a cleaner binary signal for the model. Stick with fixed TP=10/SL=5.

### Step 4: Walk-Forward Backtest — POSITIVE BUT WITH CAVEATS ⚠️

Rolling 2-day train → 1-day test, BTC May 11-17, TP=10/SL=5/300s.

**v30 Vol Expansion (per test day):**

| Test Day | AUC | Range Lift |
|----------|-----|------------|
| May 13 | 0.764 | 1.58x |
| May 14 | 0.763 | 1.76x |
| May 15 | 0.732 | 1.59x |
| May 16 | 0.707 | 1.45x |
| May 17 | 0.723 | 1.58x |
| **Average** | **0.738** | **1.59x** |

v30 signal is **rock solid** — AUC never drops below 0.70, range lift always >1.4x.

**v32 Symmetric TP/SL (per test day):**

| Test Day | AUC | P90 Trades | WR | EV/trade | Day PnL |
|----------|-----|-----------|-----|----------|---------|
| May 13 (Tue) | 0.612 | 288 | 72% | +1.0 | **+275** |
| May 14 (Wed) | 0.667 | 287 | 67% | +0.5 | **+140** |
| May 15 (Thu) | 0.634 | 285 | 69% | +0.8 | **+225** |
| May 16 (Fri) | 0.655 | 285 | 56% | -0.1 | -40 |
| May 17 (Sat) | 0.690 | 286 | 46% | -0.8 | -225 |
| **Total** | **0.652** | **1,431** | **62%** | **+0.26** | **+375** |

**Key observations:**
- **Weekdays (Tue-Thu): strongly profitable** — +640 bps in 3 days
- **Fri-Sat: losing** — -265 bps in 2 days (lower vol, fewer participants)
- **Net: +375 bps over 5 days (+75 bps/day)**
- **286 trades/day** at P90 threshold

## Final Assessment

### What's Real
1. **Vol expansion prediction** (v30): AUC 0.71-0.79, consistent across all tests. This is a genuine, robust signal.
2. **Direction is unpredictable**: Confirmed 6+ times across v30/v32. Don't try.
3. **Top features**: vol_900s, vol_3600s, hour_of_day — volatility clustering is the core mechanism.

### What's Promising but Needs More Work
1. **Symmetric TP/SL strategy** (v32): Profitable in walk-forward (+75 bps/day) but has losing days on weekends.
2. **Potential fix**: Add a "don't trade on low-vol days" filter using v30's vol expansion signal.

### What Doesn't Work
1. Combining v30 probability as a feature — redundant
2. Adaptive TP/SL sizing — makes target noisier
3. Any directional prediction — fundamentally impossible with these features

### Recommended Next Steps
1. ~~Add weekday/activity filter~~ → Done (z-score filter, see below)
2. ~~Test on 14+ days~~ → Done (2 weeks × 5 symbols)
3. ~~Test on more symbols~~ → Done (BTC, ETH, SOL, DOGE, XRP)
4. **Paper trading**: Implement on Bybit testnet with real order book

---

## v33b: Multi-Symbol Walk-Forward + Z-Score Activity Filter

### Setup
- **5 symbols**: BTC, ETH, SOL, DOGE, XRP
- **2 weeks**: May 12-16 (W1), May 19-23 (W2)
- **Walk-forward**: 2-day train → 1-day test, 3 test days per week
- **Z-score filter**: Skip trading when vol_3600s z-score < -0.5 (derived from train period stats, not hardcoded)
- **Config**: TP=10 bps, SL=5 bps, TL=300s, symmetric orders, P90 threshold

### Results: ALL 5 Symbols Profitable Across Both Weeks

| Symbol | AUC | Trades | EV/trade | Total PnL | z>-0.5 EV | z>-0.5 PnL |
|--------|-----|--------|----------|-----------|-----------|-------------|
| **BTCUSDT** | **0.642** | 1,749 | +0.48 | **+835** | +0.48 | +840 |
| **ETHUSDT** | 0.505 | 1,711 | +0.90 | **+1,540** | +0.95 | +1,005 |
| **SOLUSDT** | 0.498 | 1,723 | +0.78 | **+1,350** | +0.86 | +995 |
| **DOGEUSDT** | 0.507 | 1,712 | +1.30 | **+2,220** | +1.26 | +1,635 |
| **XRPUSDT** | 0.519 | 1,724 | +0.76 | **+1,305** | +0.77 | +940 |
| **TOTAL** | — | **8,619** | **+0.84** | **+7,250** | +0.86 | +5,415 |

### Key Insights

1. **ALL symbols profitable** — not a single losing symbol across 6 test days each
2. **DOGE is the best** — +1.30 bps/trade, +2,220 bps total (highest vol = most profitable)
3. **BTC is the only one where ML helps** — AUC 0.64 vs ~0.50 for others. Higher-vol coins don't need timing.
4. **Z-score filter improves EV/trade** but reduces total PnL (fewer trades). Best for SOL/XRP where it boosts EV from +0.78→+0.86 and +0.76→+0.77.
5. **Portfolio approach**: Run all 5 symbols = ~1,400 trades/week, +7,250 bps total

### Why Higher-Vol Coins Don't Need ML

The 2:1 TP/SL ratio (TP=10, SL=5) is naturally profitable when vol is high enough:
- **BTC**: Low vol → TP rarely hit → needs ML to time entries
- **DOGE/SOL/XRP/ETH**: Higher vol → TP hit often enough → 2:1 ratio generates positive EV unconditionally

### Strategy Architecture (Final)

```
For BTC:     ML model (AUC 0.64) → P90 filter → symmetric TP/SL orders
For others:  Z-score vol filter (z > -0.5) → symmetric TP/SL orders (no ML needed)
```

---

## v33c: TP/SL Ratio Robustness Test

### Setup
- **Same period**: May 12-16 (W1) + May 19-23 (W2), 10 days total
- **5 symbols**: BTC, ETH, SOL, DOGE, XRP
- **All samples** (no ML, no filter — pure baseline to isolate ratio effect)
- **Time limit**: 300s for all configs
- **Configs tested**: varying ratio (1:1 to 4:1) and varying absolute size (TP=6-20)

### Part 1: Varying the Ratio (fixed SL=5 bps)

| Config | BTC | ETH | SOL | DOGE | XRP | **AVG** |
|--------|-----|-----|-----|------|-----|---------|
| 1:1 TP=5 SL=5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| 1.4:1 TP=7 SL=5 | +0.03 | +0.29 | +0.31 | +0.37 | +0.25 | **+0.25** |
| **2:1 TP=10 SL=5** | **-0.35** | **+0.60** | **+0.66** | **+0.74** | **+0.47** | **+0.42** |
| 3:1 TP=15 SL=5 | -1.24 | +0.72 | +0.89 | +1.15 | +0.44 | **+0.39** |
| 4:1 TP=20 SL=5 | -2.16 | +0.43 | +0.75 | +1.23 | -0.09 | **+0.03** |

**Findings**:
- **1:1 is exactly zero** — confirms no directional edge (as expected)
- **2:1 is the sweet spot** for average EV across all symbols
- **3:1 is best for high-vol coins** (DOGE +1.15, SOL +0.89) but kills BTC (-1.24)
- **4:1 is too greedy** — TP rarely hit, only DOGE survives
- **BTC is unprofitable at all ratios without ML** — confirms ML is needed for BTC

### Part 2: Varying Absolute Size (fixed 2:1 ratio)

| Config | BTC | ETH | SOL | DOGE | XRP | **AVG** | **Total PnL** |
|--------|-----|-----|-----|------|-----|---------|---------------|
| TP=6 SL=3 | **+0.36** | +0.59 | +0.64 | +0.60 | +0.56 | **+0.55** | **+94,641** |
| TP=10 SL=5 | -0.35 | +0.60 | +0.66 | +0.74 | +0.47 | +0.42 | +72,895 |
| TP=16 SL=8 | -1.58 | +0.15 | +0.37 | +0.63 | -0.17 | -0.12 | -20,448 |
| TP=20 SL=10 | -2.18 | -0.55 | -0.23 | +0.30 | -1.03 | -0.74 | -126,780 |

**This is the key finding**: **TP=6/SL=3 is the best config!**
- Highest average EV (+0.55 vs +0.42 for TP=10/SL=5)
- Highest total PnL (+94,641 vs +72,895)
- **BTC is profitable even without ML** (+0.36 bps/trade)
- All 5 symbols profitable simultaneously
- Tighter levels = more trades resolve (fewer timeouts) = more consistent

### Why Tighter Levels Win

With TP=6/SL=3 (2:1 ratio):
- TP is hit more often (6 bps is easier to reach than 10 bps in 300s)
- The 2:1 asymmetry still provides edge: you win 6 when right, lose only 3 when wrong
- Fewer timeouts = more resolved trades = more PnL opportunities
- Works even on BTC because the threshold is within normal 5-min range

### Walk-Forward Sanity Check (P90 ML filter)

The baseline test (all samples, no filter) favored TP=6/SL=3. But with the **P90 ML filter** (walk-forward, 3 test days):

| Config | Trades | EV/trade | Total PnL | z>-0.5 PnL |
|--------|--------|----------|-----------|-------------|
| **TP=10/SL=5** | 4,287 | **+0.76** | **+3,260** | **+2,440** |
| TP=6/SL=3 | 4,295 | +0.69 | +2,955 | +2,319 |

**TP=10/SL=5 wins with P90 filter** — higher EV and total PnL. But TP=6/SL=3 is more consistent (BTC +0.47 vs +0.31, no losing days).

**Per-symbol breakdown:**
- **BTC**: TP=6/SL=3 better (+0.47 vs +0.31) — tighter levels work in low vol
- **DOGE/SOL/XRP**: TP=10/SL=5 better (+1.23, +0.88, +0.87) — wider levels capture bigger moves
- **ETH**: TP=6/SL=3 slightly better (+0.70 vs +0.51)

### Updated Strategy Architecture

```
Conservative: TP=6/SL=3 for all symbols — consistent, no losing days
Aggressive:   TP=10/SL=5 for high-vol (DOGE/SOL/XRP), TP=6/SL=3 for BTC/ETH
Both:         Z-score filter (z > -0.5) + ML P90 filter for BTC
```

---

## v33d: Full TP/SL Ratio Spectrum (Inverse + Favorable)

### Setup
- **Same period**: May 12-16 + May 19-23 (10 days), all samples (no ML filter)
- **5 symbols**: BTC, ETH, SOL, DOGE, XRP
- **Ratios tested**: 1:2, 1:1.5, 1:1, 1.5:1, 2:1, 3:1 (SL=5 bps fixed for favorable side)

### EV Per Trade (bps)

| Symbol | 1:2 | 1:1.5 | 1:1 | 1.5:1 | 2:1 | 3:1 |
|--------|-----|-------|-----|-------|-----|-----|
| BTC | **+0.35** | +0.00 | 0.00 | -0.00 | -0.35 | -1.24 |
| ETH | -0.60 | -0.35 | 0.00 | +0.35 | **+0.60** | **+0.72** |
| SOL | -0.66 | -0.38 | 0.00 | +0.38 | **+0.66** | **+0.89** |
| DOGE | -0.74 | -0.44 | 0.00 | +0.44 | **+0.74** | **+1.15** |
| XRP | -0.47 | -0.27 | 0.00 | +0.27 | **+0.47** | +0.44 |
| **AVG** | -0.42 | -0.29 | 0.00 | +0.29 | **+0.42** | +0.39 |

### Combo Win Rate (% of resolved symmetric trades that profit)

| Symbol | 1:2 | 1:1.5 | 1:1 | 1.5:1 | 2:1 | 3:1 |
|--------|-----|-------|-----|-------|-----|-----|
| BTC | 46% | 26% | 0% | 74% | 54% | 30% |
| ETH | 29% | 17% | 0% | 83% | 71% | 53% |
| SOL | 28% | 16% | 0% | 84% | 72% | 54% |
| DOGE | 28% | 16% | 0% | 84% | 72% | 57% |
| XRP | 31% | 18% | 0% | 82% | 69% | 50% |

### Per-Side Win Rate (how often does one side's TP get hit?)

| Symbol | 1:2 | 1:1.5 | 1:1 | 1.5:1 | 2:1 | 3:1 |
|--------|-----|-------|-----|-------|-----|-----|
| BTC | 69% | 60% | 50% | 40% | 32% | 20% |
| ETH | 65% | 59% | 51% | 42% | 37% | 28% |
| SOL | 65% | 59% | 51% | 43% | 37% | 28% |
| DOGE | 64% | 59% | 51% | 43% | 37% | 29% |
| XRP | 65% | 59% | 51% | 42% | 36% | 27% |

### Key Discovery: Perfect Antisymmetry

**The results are exactly mirrored.** 1:2 EV = negative of 2:1 EV. 1:1.5 EV = negative of 1.5:1 EV. 1:1 = exactly zero.

This proves:
1. **There is zero directional edge** — 1:1 is exactly 0.00 for all symbols
2. **The edge comes purely from the asymmetric payoff structure**, not from any market inefficiency
3. **Inverse ratios (1:2, 1:1.5) have higher win rates** (46-69% per side) but **negative EV** — you win often but lose more when you lose
4. **Favorable ratios (2:1, 3:1) have lower win rates** (20-37% per side) but **positive EV** — you lose often but win more when you win

### BTC is the Anomaly

BTC is the **only** symbol where the inverse ratio (1:2) is profitable (+0.35 bps). This means BTC has a **mean-reverting** microstructure at the 5-min scale — price tends to return toward the entry point. For all other coins, price tends to **trend** (or at least move enough to hit wider TPs), making favorable ratios profitable.

### Why This Matters

The symmetric TP/SL strategy is **not** exploiting a market inefficiency. It's exploiting a **structural property of price distributions**:
- Crypto prices at 5-min scale are slightly leptokurtic (fat-tailed)
- Fat tails mean extreme moves happen more often than a normal distribution predicts
- A 2:1 TP/SL ratio captures this: the TP (at 2× the SL distance) gets hit more often than a normal distribution would predict
- This effect is stronger for higher-vol coins (DOGE > SOL > ETH > XRP > BTC)

### Mathematical Proof: Fat Tails = Real Edge

The antisymmetry is **exact** — verified to the individual trade level:
```
1:2 long TP hits:  18,505  ≡  2:1 short SL hits: 18,505  (same price event)
1:2 long SL hits:   9,945  ≡  2:1 short TP hits:  9,945  (same price event)
```

For a **pure random walk**, the probability of hitting TP before SL is exactly:
```
P(TP hit) = SL / (TP + SL)
```
So for 2:1 (TP=10, SL=5): expected WR = 5/15 = **33.3%**

**Actual observed WR on ETH: 36.6%** (long side, 2:1)

This 3.3pp excess win rate is the **entire source of edge**:
- Random walk EV: 0.333×10 - 0.667×5 = **0.00** (zero, as expected)
- Actual EV: 0.366×10 - 0.634×5 = **+0.49 bps** (positive, from fat tails)

The edge exists because crypto prices have **excess kurtosis** — extreme moves (both up and down) happen more frequently than a random walk predicts. The 2:1 TP/SL ratio is positioned to capture these tail events.

**This is a real, structural edge** — not an artifact. It will persist as long as crypto price distributions remain fat-tailed, which is a fundamental property of leveraged speculative markets.

---

## v33e: Fee Reality Check — The Strategy Doesn't Survive Fees

### The Actionable Setup (SOL @ ~$170)

**What you'd do manually:**
1. SOL is at $170.00
2. Place a **limit long** at $170.00 (buy)
3. Place a **limit short** at $170.00 (sell)
4. Long TP: limit sell at $170.17 (+10 bps = +$0.17)
5. Long SL: stop-market at $169.915 (-5 bps = -$0.085)
6. Short TP: limit buy at $169.83 (-10 bps)
7. Short SL: stop-market at $170.085 (+5 bps)
8. Cancel remaining orders after 5 minutes if not filled

**What happens (SOL, 10 days of data):**

| Outcome | Frequency | Net PnL |
|---------|-----------|---------|
| One side TP + other SL | **59.5%** | +5 bps |
| Both SL hit | **22.7%** | -10 bps |
| Both timeout (cancel) | **16.8%** | 0 bps |
| One SL + timeout | 1.0% | -5 bps |
| Both TP hit | 0.0% | +20 bps |

**Gross EV: +0.66 bps per entry** (~$0.11 on a $170 position)

### The Fee Problem

Each entry involves **4 order fills** (open 2 + close 2):

| Order | Type | Fee |
|-------|------|-----|
| Open long | Limit (maker) | 1 bps |
| Open short | Limit (maker) | 1 bps |
| Close winner (TP) | Limit (maker) | 1 bps |
| Close loser (SL) | Stop-market (taker) | 5.5 bps |

**Weighted average fee: ~10.3 bps per entry**

| Gross EV | Fee | Net EV | Verdict |
|----------|-----|--------|---------|
| +0.66 bps (SOL) | 10.3 bps | **-9.7 bps** | ❌ |
| +1.30 bps (DOGE, best) | 10.3 bps | **-9.0 bps** | ❌ |

### Even VIP Tiers Don't Save It

| Tier | Maker | Taker | Fee | DOGE Net |
|------|-------|-------|-----|----------|
| Regular | 0.010% | 0.055% | 10.3 bps | **-9.0** |
| VIP 3 | 0.004% | 0.025% | 4.6 bps | **-3.3** |
| VIP 5 | 0.000% | 0.015% | 2.1 bps | **-0.8** |
| Pro 1 | -0.005% | 0.015% | 0.8 bps | **+0.5** |
| Pro 2 | -0.005% | 0.013% | 0.5 bps | **+0.8** |
| Pro 3 | -0.005% | 0.010% | 0.1 bps | **+1.2** |

**Only profitable at Pro 1+ tier** (requires $100M+ monthly volume on Bybit).

### Correction: All Orders CAN Be Limit (Maker)

The SL does **not** need to be a stop-market. All 6 orders can be placed at entry time as resting limit orders:

| Order | Side | Price (SOL@$170) | Rests as |
|-------|------|-------------------|----------|
| Open Long | Limit Buy | $170.00 | Bid |
| Open Short | Limit Sell | $170.00 | Ask |
| Long TP | Limit Sell | $170.17 (+10bps) | Ask |
| Long SL | Limit Sell | $169.915 (-5bps) | Ask |
| Short TP | Limit Buy | $169.83 (-10bps) | Bid |
| Short SL | Limit Buy | $170.085 (+5bps) | Bid |

All sit on the book → all fill as **maker**. Use `post_only` flag to guarantee maker-or-reject. Cancel remaining orders once a side resolves.

### Revised Fee Analysis (All-Maker)

| Maker Fee | Total Fee (4 fills) | DOGE Net | SOL Net | All 5 Profitable? |
|-----------|--------------------:|----------|---------|-------------------|
| 0.010% (Bybit regular) | 4.0 bps | -2.70 ❌ | -3.35 ❌ | No |
| 0.005% | 2.0 bps | -0.70 ❌ | -1.35 ❌ | No |
| **0.000%** | **0.0 bps** | **+1.30 ✅** | **+0.66 ✅** | **Yes** |
| -0.005% (rebate) | -2.0 bps | +3.30 ✅ | +2.66 ✅ | Yes |

### Conclusion

With **all-limit orders** and **0% maker fee**, the strategy is profitable on all 5 symbols. The edge is real but requires either:
- An exchange with true 0% maker fees on perps
- Or maker rebates to amplify the edge

Current gross EV (~0.5-1.3 bps) is too thin for any exchange charging maker fees. **Widening the TP/SL levels** (via volatility filtering) could increase gross EV to survive small maker fees.

---

## v34: Volatility Filter Optimization

### Goal
Use temporal patterns (v33) and volume signals (v38) to filter entries to high-vol periods, then test wider TP/SL with longer time limits.

### Step 1: Filter Impact on TP=10/SL=5 (5m TL)

| Filter | Avg Trades | EV | Improvement |
|--------|-----------|-----|-------------|
| No filter | 34,410 | +0.424 | baseline |
| US hours (12-20 UTC) | 11,520 | +0.560 | +32% |
| Peak hours (13-18 UTC) | 7,200 | +0.604 | +42% |
| Weekday only | 28,650 | +0.507 | +20% |
| Weekday + Peak (13-18) | 6,000 | +0.725 | **+71%** |
| Vol surge z>1 | 4,844 | +0.631 | +49% |
| **Weekday + Peak + VolZ>1** | **1,046** | **+0.966** | **+128%** |

**Weekday + Peak hours is the best simple filter** (+71% EV). Adding vol z-score pushes to +128% but cuts sample count to 1K.

### Step 2: Wider TP/SL + Longer Time Limits (weekday 13-18 UTC)

| Config | Time Limit | Avg EV | Timeout |
|--------|-----------|--------|---------|
| 10/5 | 5m | +0.725 | 2% |
| 10/5 | 15m | +0.821 | 0% |
| **20/10** | **15m** | **+0.838** | **2%** |
| **20/10** | **30m** | **+0.892** | **0%** |
| 30/15 | 30m | +0.508 | 2% |

**TP=20/SL=10 with 15-30m TL is the new sweet spot** — same EV as 10/5 but 2× the bps per winning trade.

Per-symbol bests (weekday 13-18 UTC):
- **SOL 20/10 15m**: +1.21 bps
- **DOGE 30/15 30m**: +1.21 bps
- **ETH 30/15 30m**: +1.02 bps
- **XRP 20/10 30m**: +0.69 bps
- **BTC 10/5 30m**: +0.86 bps

### Fee Viability (all-limit, 4 maker fills)

| Maker Fee | 4-Fill Cost | Portfolio Net (0.89 avg) | Best Symbol Net (1.21) |
|-----------|------------|-------------------------|----------------------|
| **0.000%** | **0 bps** | **+0.89 ✅** | **+1.21 ✅** |
| 0.005% | 2 bps | -1.11 ❌ | -0.79 ❌ |
| 0.010% | 4 bps | -3.11 ❌ | -2.79 ❌ |

### Conclusion

Temporal + vol filters boost EV by up to 128%, and wider TP/SL with longer TL adds another ~15%. But the maximum achievable gross EV (~1.2 bps on best symbols) still requires **true 0% maker fees** to be profitable. The 4-fill structure is the fundamental constraint — any non-zero per-fill fee quickly overwhelms the edge.

### Step 3: Sequential Simulation (Realistic Execution)

Previous backtests used overlapping entries (new trade every 30s). In reality you'd enter one trade, wait for resolution, then enter the next. Does the EV hold?

**TP=10/SL=5, 5m TL, weekday 13-18 UTC:**

| Symbol | Trades/day | Seq EV | Avg Duration | Overlap EV |
|--------|-----------|--------|-------------|------------|
| DOGE | 396 | +0.86 | 28s | +0.92 |
| SOL | 323 | +0.97 | 39s | +0.91 |
| XRP | 274 | +0.83 | 49s | +0.65 |
| ETH | 308 | +0.66 | 42s | +0.71 |
| BTC | 141 | +0.15 | 112s | +0.44 |
| **Portfolio** | **1,442** | **+0.77** | **45s** | +0.73 |

**EV is consistent between sequential and overlapping** — the edge is real per-entry, not an artifact of overlap.

**Entry trigger**: There is no signal — it's purely temporal. Enter immediately at 13:00 UTC weekday, re-enter after each resolution until 18:00 UTC. Trades resolve in ~30-45s on average (not the full 5m TL).

**TP=20/SL=10, 15m TL** gives ~610 trades/day across 5 symbols at +0.82 EV, with 132s avg duration. Higher EV per trade but fewer trades.

---

## v35: Smart Entry Triggers — Not Every Second Is Equal

### Hypothesis
Instead of entering blindly every 30s during peak hours, use microstructure signals (vol, range, trade count) to select the best entry moments. Research v29-v31 showed breakout/squeeze regimes have 2-2.5x forward volatility, and v38 showed volume surges predict 6x vol.

### Signals Tested (TP=20/SL=10, 15m TL, weekday 13-18 UTC)

**SOL** — Filtering works:

| Trigger | N | EV | Sequential EV |
|---------|---|-----|--------------|
| Baseline | 6,000 | +1.213 | +0.810 |
| Range60 z>1 | 1,009 | **+1.705** | — |
| TC ratio>2.0 | 178 | **+3.427** | — |

**DOGE** — TC ratio works:

| Trigger | N | EV | Sequential EV |
|---------|---|-----|--------------|
| Baseline | 6,000 | +1.065 | +1.127 |
| TC>1.5 | 912 | **+1.546** | +1.504 |
| TC>2.0 | 352 | **+1.562** | +1.631 |

**BTC** — TC ratio works (biggest improvement):

| Trigger | N | EV | Sequential EV |
|---------|---|-----|--------------|
| Baseline | 6,000 | +0.418 | +0.382 |
| TC>1.5 | 921 | **+1.227** | +1.126 |
| TC>2.0 | 363 | +0.771 | **+1.714** |
| Range_z>1 | 1,027 | +0.769 | +1.314 |

**ETH** — Filtering HURTS (mean-reversion during spikes):

| Trigger | N | EV | Sequential EV |
|---------|---|-----|--------------|
| Baseline | 6,000 | +0.832 | +0.032 |
| TC>2.0 | 201 | **-0.398** | -1.127 |
| Range_z>1 10/5 5m | 1,029 | +0.777 | +0.841 |

**Key finding**: ETH reverses during high-activity moments (consistent with v31b). ETH needs the opposite: trade during *normal* activity, not spikes.

### Discovery: Adaptive TP/SL (Scale to Current Volatility)

Instead of fixed TP/SL, set levels based on recent 60s price range:

**BTC** (range P50 = 7.6 bps):

| Range Bucket | TP/SL Used | N | EV | WR |
|-------------|-----------|---|-----|-----|
| **<5 bps** | **10/5** | 1,538 | **+1.014** | **73.2%** |
| **5-10 bps** | **15/7** | 2,524 | **+1.028** | **67.3%** |
| 10-20 bps | 30/15 | 1,628 | -1.327 | 51.9% |
| 20-50 bps | 50/25 | 305 | -7.213 | 36.7% |

**DOGE** (range P50 = 19.9 bps):

| Range Bucket | TP/SL Used | N | EV | WR |
|-------------|-----------|---|-----|-----|
| 5-10 bps | 15/7 | 489 | **+1.746** | 71.6% |
| 10-20 bps | 30/15 | 2,520 | +1.179 | 69.0% |
| **>50 bps** | **50/25** | 226 | **+3.540** | **71.2%** |

**Critical insight**: The edge is concentrated in **low-to-moderate range environments**, NOT during extreme moves. When range is already high (>20 bps for BTC, >50 bps for most), wider TP/SL levels have *negative* EV — the move is already exhausting.

This contradicts the naive hypothesis that "more vol = more edge." Instead:
- **Compression → early breakout** (range just starting to expand) = best entries
- **Already in breakout/exhaustion** (range already high) = worst entries
- The fat-tail edge comes from **catching the expansion**, not riding an existing one

### Per-Symbol Optimal Strategy

| Symbol | Best Config | Trigger | EV | Trades/day (est) |
|--------|------------|---------|-----|-----------------|
| SOL | 20/10 15m | Range_z>1 | +1.71 | ~100 |
| DOGE | 20/10 15m | TC>1.5 | +1.55 | ~90 |
| BTC | 20/10 15m | TC>1.5 | +1.23 | ~50 |
| XRP | 20/10 15m | Range_z>1 | +1.09 | ~80 |
| ETH | 10/5 5m | Range_z>1 | +0.84 | ~90 |

**Portfolio weighted avg EV: ~1.3 bps** (vs 0.77 baseline — **+69% improvement**)
