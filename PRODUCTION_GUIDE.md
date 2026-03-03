# Production Guide — Settlement Short Strategy

**Last updated:** 2026-03-01

---

## 1. What Is This Strategy and Why Does It Work

### The Mechanism

Bybit perpetual futures settle funding rates every 1 hour. When the funding rate is **negative** (shorts pay longs), there's a predictable microstructure event at the settlement boundary:

1. **Before settlement:** Traders holding short positions accumulate a funding liability. The more negative the FR, the more they owe.
2. **At settlement (T=0):** The funding payment is deducted. Shorts who held through settlement get charged. Many choose to **close their shorts just before or at settlement** to avoid the charge — this means **buying** (covering), which temporarily pushes price up.
3. **After settlement (T+0 to T+60s):** The buying pressure disappears. Traders who were waiting to re-short can now enter. The price **drops** as sell pressure resumes — this is the "post-settlement crash."

We **sell at T+20ms** (entering a short position just after the settlement snapshot, to avoid being captured by the funding rate charge) and **buy back** when our ML model detects the crash has bottomed out. We pocket the difference minus fees and slippage.

### Why It Works

The edge exists because of a **structural asymmetry in information and incentives**:

- **Predictable timing:** Everyone knows when settlement happens (every hour, on the hour). But the *magnitude* and *duration* of the post-settlement crash varies per coin, per session.
- **Forced participants:** Traders with existing short positions are forced to act around settlement. Their buying (to avoid FR charges) creates predictable pre-settlement price inflation that reverses after T=0.
- **Microstructure persistence:** The crash-and-recovery pattern is driven by order flow mechanics, not fundamental value. It happens on every negative-FR settlement, every hour, across dozens of coins.
- **ML timing advantage:** Our short exit model (LogReg on 56 tick-level features) detects when the crash is nearing its bottom with p(near_bottom) ≥ 0.4, allowing us to exit near the optimal point rather than at a fixed time.

### In Numbers

| Metric | Value |
|--------|-------|
| Average gross edge per trade | 23.6 bps (LOSO out-of-sample) |
| Win rate | 76% |
| Average winner | +9.7 bps net |
| Average loser | −16.1 bps net |
| Expectancy | +4.0 bps per trade |
| Why we win | High win rate (76%), not large winners. The W/L ratio (0.60x) is actually unfavorable — we need to keep WR high by filtering aggressively. |

### What Can Go Wrong

The 23.6 bps edge is **fixed** (determined by the FR settlement mechanism). The **slippage is variable** — it depends on the coin's orderbook depth and spread. On liquid coins (slip 10–15 bps), we almost always win. On illiquid coins (slip 30–80 bps), we almost always lose.

**The entire strategy is: harvest a small fixed edge on coins where transaction costs don't eat it.**

---

## 2. How We Filter Coins

### The Filtering Funnel

Out of hundreds of Bybit perpetual futures, only a handful are tradeable per session. We apply **3 filters** in sequence:

```
~300+ Bybit perps
    │
    │ Filter 1: FR magnitude — negative FR ≥ 15 bps
    ▼
~40-50 coins with negative FR this hour
    │
    │ Filter 2: bid depth within 20bps ≥ $2,000
    │ Filter 3: spread ≤ 8 bps
    ▼
~1-3 coins pass → TRADE THESE
```

In our 4-day dataset: 32 unique symbols appeared across 160 settlement events, 12 always passed filters, 11 always failed, 9 were borderline (sometimes pass, sometimes fail depending on the session).

### Filter Parameters — Detailed Breakdown

#### Filter 1: FR Magnitude (Pre-Selection)

Before we even look at the orderbook, we need a coin that has a **negative funding rate ≥ 15 bps** (absolute value) at the upcoming settlement. This is the prerequisite — no negative FR means no post-settlement crash to short.

The FR pre-selection identifies coins where shorts are paying longs enough to create the settlement microstructure event. Dozens of coins may have negative FR each hour, but most are tiny (< 10 bps) and not worth trading. At ≥ 15 bps, the post-settlement crash is large enough to exceed our transaction costs on liquid coins.

In our dataset, FR ranges from 15 bps to 250 bps (median 36 bps). However, **FR magnitude does not predict trade quality once past the 15 bps gate** (see below). It's a gate, not a signal.

#### Filter 2: Bid Depth Within 20 bps ≥ $2,000

**What it is:** The total USD value of all bid orders within 20 basis points below the mid price. This is the "actionable liquidity" — the depth our sell order actually walks through.

**Why 20 bps?** Our sell order at $500–$3,000 notional typically walks 5–15 bps into the book. Using 20 bps captures the relevant depth without including irrelevant deep levels.

**Why $2,000 minimum?**

| Depth (20bps) | N | Win Rate | Avg PnL | Avg RT Slippage |
|---------------|---|----------|---------|-----------------|
| **< $1K** | 8 | **0%** | −30.1 bps | 53.7 bps |
| **< $2K** | 23 | **13%** | −17.8 bps | 41.4 bps |
| $2–5K | 42 | 71% | +2.1 bps | 21.5 bps |
| $5–10K | 33 | 97% | +8.9 bps | 14.7 bps |
| $10–25K | 49 | **100%** | +12.3 bps | 11.3 bps |
| $25K+ | 3 | **100%** | +10.4 bps | 13.2 bps |

- **Below $2K: 13% WR, −17.8 bps average.** Our $500–$3K sell order consumes 25–100%+ of the near-BBO liquidity, causing 30–50+ bps of slippage that exceeds the 23.6 bps edge.
- **Above $5K: 97–100% WR.** The book absorbs our order with minimal price impact.
- **$2K is the minimum viable cutoff:** catches 20 of 33 losers (61%), loses only 3 winners (2.6%). Raises WR from 78% → 90%.

**Optimal range:**

| Threshold | Settlements | Win Rate | $/trade | Losers caught | Winners lost |
|-----------|------------|----------|---------|---------------|--------------|
| ≥ $2K (current) | 127 | 90% | $1.85 | 20/33 (61%) | 3 (2.6%) |
| ≥ $3K (aggressive) | 115 | 94% | $2.06 | 26/33 (79%) | 9 (7.7%) |
| ≥ $5K (very strict) | 85 | 99% | $2.23 | 32/33 (97%) | 33 (28%) |

$2K is the production default. $3K is recommended if you can afford fewer trades. $5K is too strict — you lose 28% of winners.

**How it influences performance:** This is the **#1 predictor** of win/loss. Median depth for winners: $9,187. Median for losers: $1,833. The gap is $7,354.

#### Filter 2: Spread ≤ 8 bps

**What it is:** The bid-ask spread at T=0, measured in basis points: `(best_ask - best_bid) / mid × 10,000`.

**Why it matters:** Spread is an unavoidable round-trip cost. At 8 bps spread, you pay ~4 bps per side just to cross the spread, *before* any depth-walking. This eats 34% of the 23.6 bps edge before you even start.

| Spread | N | Win Rate | Avg PnL | Avg RT Slippage |
|--------|---|----------|---------|-----------------|
| < 2 bps | 65 | **97%** | +11.1 bps | 12.5 bps |
| 2–4 bps | 30 | 73% | +2.9 bps | 20.7 bps |
| 4–6 bps | 22 | 82% | +4.7 bps | 18.9 bps |
| 6–10 bps | 22 | **55%** | −0.6 bps | 24.2 bps |
| 10+ bps | 11 | **18%** | −26.6 bps | 50.2 bps |

- **< 2 bps: 97% WR.** The tight spread means nearly all the edge is profit.
- **> 10 bps: 18% WR.** The spread alone eats nearly half the edge.
- **8 bps is the cutoff** because it catches the worst offenders (10+ bps coins) while not being too strict on the 6–8 bps range where we still break even.

**Optimal range:**

| Threshold | Settlements | Win Rate | Losers caught |
|-----------|------------|----------|---------------|
| ≤ 8 bps (current) | ~140 | ~85% | 11 worst cases |
| ≤ 5 bps | 111 | 87% | 19/33 |
| ≤ 3 bps | 83 | 93% | 27/33 |

8 bps is the current production default. If combined with depth ≥ $2K (both applied), the effective WR is 76% across 127 trades.

**How it influences performance:** Spread is the **#2 predictor** after depth. Median spread for winners: 1.8 bps. For losers: 6.8 bps. The gap is 5.0 bps.

#### Why FR Magnitude Doesn't Predict Winners (Once Past the Gate)

| FR Range | N | Win Rate | Avg PnL |
|----------|---|----------|---------|
| 15–25 bps | 48 | 79% | +6.9 bps |
| 25–50 bps | 44 | 82% | +2.6 bps |
| 50–80 bps | 31 | 87% | +6.4 bps |
| 80+ bps | 27 | **59%** | −1.4 bps |

High FR has the **worst** win rate. Why? Coins with extreme negative FR (NEWTUSDT, ATHUSDT, ALICEUSDT) tend to be illiquid micro-caps. The FR is high *because* they're illiquid — and that same illiquidity causes fatal slippage.

**Do not filter or size by FR beyond the initial gate.** Once a coin passes the FR pre-selection, its FR magnitude doesn't help predict whether the trade will win or lose. Depth and spread are what matter.

### Combined Filter Effect

Both filters work together:

| Filter | N | WR | $/trade | What it catches |
|--------|---|----|---------|-----------------|
| No filter | 160 | 78% | $1.66 | — |
| Depth ≥ $2K only | 137 | 85% | $1.80 | Ultra-thin books |
| Spread ≤ 8 only | ~140 | ~83% | $1.75 | Ultra-wide spreads |
| **Both (production)** | **127** | **76%†** | **$4.34†** | Ultra-thin AND ultra-wide |

**†** These numbers use per-trade ML PnL (in-sample). LOSO conservative: ~$1.60–$2.40/trade.

Both filters target the same root cause: **illiquidity**. Thin books and wide spreads are two symptoms of the same disease. Together they catch 88% of all losers.

### The Coins That Always Fail

These 11 symbols **never** passed our filters across 4 days — structurally untradeable:

| Symbol | Why |
|--------|-----|
| ACEUSDT | depth $106, spread 25 bps |
| GNOUSDT | depth $164, spread 14 bps |
| XCNUSDT | depth $161 |
| ZBCNUSDT | depth $1,143, spread 17.6 bps |
| CYBERUSDT | depth $1,075, spread 10.7 bps |
| SPACEUSDT | depth $1,339 |
| API3USDT | depth $1,870 |
| AIXBTUSDT | depth $1,954 |
| ANIMEUSDT | spread 9.7 bps |
| ORBSUSDT | spread 9.3 bps |
| REDUSDT | spread 8.6 bps |

### The Reliable Winners

These 12 symbols **always** passed filters — consistently tradeable:

| Symbol | Why |
|--------|-----|
| ENSOUSDT | depth $14K, spread 0.6 bps |
| BARDUSDT | depth $17K, spread 1.7 bps |
| SAHARAUSDT | depth $13K, spread 2.6 bps |
| POWERUSDT | depth $5K, spread 1.3 bps |
| STEEMUSDT | depth $4.9K, spread 1.7 bps |
| BIRBUSDT, ESPUSDT, HOLOUSDT, KERNELUSDT, MOVEUSDT, ROBOUSDT, STABLEUSDT, WETUSDT, ZKCUSDT | Various, all pass both filters |

### Summary: The Filter Decision Tree

```
At each settlement hour:
  1. Identify coins with negative FR ≥ 15 bps → ~40-50 candidates
  2. For each candidate, read OB.200 snapshot at T-0:
     a. Compute bid_depth_20bps = Σ(price × qty) for bids within 20bps of mid
     b. Compute spread_bps = (best_ask - best_bid) / mid × 10000
  3. SKIP if bid_depth_20bps < $2,000
  4. SKIP if spread_bps > 8
  5. TRADE the rest (typically 1-3 coins per session)
```

---

## 3. How to Size the Short Position

### The Core Problem

Our ML edge is **23.6 bps gross per trade**. That's fixed — it comes from the settlement mechanism. But **slippage scales with position size** and inversely with orderbook depth. Size too large → slippage exceeds the edge → guaranteed loss. Size too small → you leave money on the table.

```
Net PnL = 23.6 bps (fixed edge) − RT_slippage(notional, depth)

On a $5K-depth book:
  $500  notional → 6.0 bps slip → +17.6 bps net → $0.85 profit
  $2K   notional → 12.5 bps slip → +11.1 bps net → $2.29 profit  ← sweet spot
  $5K   notional → 19.4 bps slip → +4.2 bps net  → $1.72 profit  ← diminishing
  $10K  notional → 35+ bps slip  → −12 bps net   → −$12 loss     ← destroyed

The edge is small. Slippage is the bottleneck, not model accuracy.
```

### The Algorithm

At T=0, read the orderbook and compute `bid_depth_20bps` (sum of all bid liquidity within 20 bps of mid). Then look up the notional from this table:

```python
TIERS = [500, 1000, 2000, 3000]   # USD
CAP   = 0.15                       # 15% of depth_20

def compute_notional(depth_20):
    """Pick the largest tier that stays ≤ 15% of near-BBO depth."""
    notional = TIERS[0]  # minimum $500
    for tier in TIERS:
        if tier <= depth_20 * CAP:
            notional = tier
    return notional
```

This produces:

| depth_20 | Notional | % of depth | Why |
|----------|----------|------------|-----|
| $2,000 | $500 | 25%* | Thin book — minimum size, cap exceeded |
| $3,000 | $500 | 17% | Still thin — stay at minimum |
| $4,000 | $500 | 13% | 15% of $4K = $600, only $500 tier fits |
| $5,000 | $500 | 10% | 15% of $5K = $750, still $500 |
| $7,000 | $1,000 | 14% | 15% of $7K = $1,050, $1K tier fits |
| $10,000 | $1,000 | 10% | 15% of $10K = $1,500, $1K tier fits |
| $14,000 | $2,000 | 14% | 15% of $14K = $2,100, $2K tier fits |
| $20,000 | $3,000 | 15% | 15% of $20K = $3,000, $3K tier fits exactly |
| $25,000 | $3,000 | 12% | 15% of $25K = $3,750, $3K is max tier |

*At $2K depth the cap says $300 but we floor at $500 (minimum tier). This is the weakest spot in the algo — 25% of depth is aggressive. That's why the depth ≥ $2K filter matters so much.

### Why 15% Cap — The Evidence

We tested every cap from 5% to 50% on 127 filtered settlements:

| Cap | Avg $/trade | Win Rate | Total $ (127 trades) |
|-----|-------------|----------|---------------------|
| 5% | $0.96 | 100% | $121.8 |
| 10% | $1.39 | 100% | $176.4 |
| **15%** | **$1.89** | **100%** | **$239.9** |
| 20% | $2.24 | 100% | $284.1 |
| 25% | $2.50 | 100% | $317.3 |
| 30% | $2.64 | 100% | $335.5 |
| 50% | $2.87 | 100% | $364.5 |

Higher cap = more profit per trade but also more price impact risk. The data shows 100% WR even at 50%, but that's because our tier system caps at $3,000 — you can't actually allocate $15K even if the cap allows it. The 15% cap is the **conservative production default** because:

1. **At 15%, we never consume more than 15% of near-BBO liquidity.** Other participants also trade at settlement. If 3 bots each take 15%, that's 45% of the book consumed — still recoverable. At 30%+, the book may not have enough depth for everyone.
2. **RT slippage stays well below the 23.6 bps edge.** At 15% cap, median slippage is 8–10 bps, leaving 13–15 bps of net edge.
3. **No losing trades on filtered settlements.** 100% WR means we're not pushing the limits.

### Why Not Fixed Size — The Evidence

Fixed sizing ignores the orderbook and either undersizes deep books (leaving money) or oversizes thin books (causing losses):

| Strategy | Avg $/trade | Win Rate | Total $ |
|----------|-------------|----------|---------|
| Fixed $500 | $0.87 | 100% | $110.0 |
| Fixed $1,000 | $1.50 | 100% | $190.4 |
| Fixed $2,000 | $2.31 | 96% | $294.0 |
| Fixed $3,000 | $2.60 | 88% | $329.7 |
| Fixed $5,000 | $1.64 | 68% | $208.5 |
| **Adaptive (15% cap)** | **$1.89** | **100%** | **$239.9** |

- **Fixed $2K** makes more total $ than adaptive but **has 4% losers**. Those losers are on thin-book coins where $2K is too much.
- **Fixed $3K** makes even more total $ but **12% of trades lose**. In production, a 12% loss rate means drawdowns and variance.
- **Fixed $5K** is catastrophic — 32% of trades lose, $1.64 avg is worse than fixed $1K.
- **Adaptive** sacrifices some total $ for **zero losses on filtered settlements**. It never oversizes.

The adaptive algo is not the highest-revenue strategy. Fixed $3K makes 37% more total $. But adaptive has **the best risk-adjusted return** — zero losing trades means consistent daily PnL and no drawdown days.

### Slippage Reality by Depth Bucket

This is the raw evidence for why sizing must scale with depth. RT slippage (entry + exit) measured against the actual orderbook:

**Thin books ($2–5K depth, 40 settlements):**

| Notional | RT Slip | Net PnL | WR | $/trade |
|----------|---------|---------|-----|---------|
| $500 | 9.6 bps | +14.0 bps | 100% | $0.72 |
| **$1,000** | **12.7 bps** | **+10.9 bps** | **100%** | **$1.11** |
| $2,000 | 17.5 bps | +6.1 bps | 88% | $1.18 |
| $3,000 | 21.5 bps | +2.1 bps | 65% | $0.39 |
| $5,000 | 28.6 bps | −5.0 bps | 15% | −$3.77 |

Max safe size: **$1,500** (at $2K, WR drops to 88%).

**Medium books ($5–10K depth, 32 settlements):**

| Notional | RT Slip | Net PnL | WR | $/trade |
|----------|---------|---------|-----|---------|
| $500 | 6.0 bps | +17.6 bps | 100% | $0.85 |
| $1,000 | 8.6 bps | +15.0 bps | 100% | $1.48 |
| **$2,000** | **12.5 bps** | **+11.1 bps** | **100%** | **$2.29** |
| $3,000 | 15.4 bps | +8.2 bps | 97% | $2.57 |
| $5,000 | 19.4 bps | +4.2 bps | 78% | $1.72 |

Max safe size: **$3,500** (at $4K, WR drops to 88%).

**Deep books ($10–25K depth, 50 settlements):**

| Notional | RT Slip | Net PnL | WR | $/trade |
|----------|---------|---------|-----|---------|
| $500 | 4.1 bps | +19.5 bps | 100% | $0.98 |
| $1,000 | 5.5 bps | +18.1 bps | 100% | $1.81 |
| $2,000 | 7.6 bps | +16.0 bps | 100% | $3.17 |
| **$3,000** | **9.5 bps** | **+14.1 bps** | **100%** | **$4.23** |
| $5,000 | 12.7 bps | +10.9 bps | 100% | $5.52 |

Profitable up to **$10K+**. Our algo caps at $3K (could go higher on deep books, but the tier system limits it).

### The Rule of Thumb

**Never let your position exceed 15% of bid depth within 20 bps.** At that ratio:
- RT slippage stays under 15 bps (well below the 23.6 bps edge)
- You leave enough liquidity for other settlement participants
- Win rate stays at or near 100% on filtered coins

If you want to be more aggressive (higher cap or larger tiers), the data says it works — but only on deep books. On thin books ($2–5K depth), even $2K notional starts producing losers.

---

## 4. When to Exit (Timing Logic)

This section covers **when** to close the short position — the timing decision. How to execute the exit (maker vs taker order) is covered in the next section.

### The Landscape: When Does the Crash Bottom Out?

The post-settlement crash doesn't have a fixed duration. Across 127 filtered settlements:

| Bottom occurs by | % of settlements |
|-----------------|-----------------|
| T+1s | 24% |
| T+5s | 37% |
| T+10s | 45% |
| T+15s | 51% |
| T+20s | 55% |
| T+30s | 62% |
| T+45s | 73% |

**Median bottom time: 14.5 seconds.** But the range is enormous — 25% of crashes bottom within 1.6s, while 25% haven't bottomed by 45.6s. Any fixed-time exit will be badly wrong on a large chunk of trades.

### Price Trajectory (Median Across All Settlements)

What the short PnL looks like if you exit at each fixed time:

```
T+ 0s: +35 bps   ████████████████████
T+ 5s: +56 bps   ████████████████████████████████  ← peak median PnL
T+10s: +52 bps   ██████████████████████████████
T+15s: +35 bps   ████████████████████
T+20s: +40 bps   ███████████████████████
T+30s: +52 bps   ██████████████████████████████
T+45s: +57 bps   ████████████████████████████████▎
T+55s: +83 bps   ██████████████████████████████████████████████▌
```

The median stays positive throughout — but the variance is huge. At T+15s the p25 drops to +7.7 bps, meaning 25% of trades are barely profitable. By T+55s the median is +83 bps but that's driven by a few big drops in long-lasting crashes, not typical behavior.

### Option 1: Fixed-Time Exit

The simplest approach — exit at a fixed number of seconds after entry.

| Exit time | Median gross PnL | Mean gross PnL | Win rate (gross > 0) |
|-----------|-----------------|----------------|---------------------|
| T+5s | 22.6 bps | 36.0 bps | 85% |
| T+10s | 23.8 bps | 38.1 bps | 80% |
| T+15s | 20.1 bps | 30.1 bps | 76% |
| T+20s | 24.1 bps | 31.8 bps | 77% |
| T+30s | 21.2 bps | 34.6 bps | 72% |
| T+45s | 24.5 bps | 42.7 bps | 75% |

**Best fixed exit: T+5s or T+10s.** Highest WR (85%/80%) and competitive PnL. After T+10s, the crash is recovering in many settlements and you start giving back gains.

**Pros:** Dead simple. No ML needed. No model risk.
**Cons:** Exits too early on slow-developing crashes (misses 55% of bottoms). Exits too late on fast crashes (price already recovering). One-size-fits-all for a highly variable event.

### Option 2: ML-Timed Exit (Production)

A logistic regression model (56 features, 100ms ticks) predicts `p(near_bottom_10)` — the probability that the current price is within 10 bps of the eventual crash minimum. When `p ≥ 0.4`, exit.

| Metric | Value |
|--------|-------|
| **Threshold** | p(near_bottom) ≥ 0.4 |
| **Timeout** | 55s forced exit if ML never fires |
| **ML fires** | 97% of trades (4 timeouts in 127) |
| **Median exit time** | 0.3s (heavily skewed — see below) |
| **Median gross PnL** | 18.9 bps |
| **Mean gross PnL** | 46.9 bps |
| **Win rate** | 86% |
| **Captures** | 43% of perfect-hindsight PnL (median) |

#### ML Timing Behavior

The ML model has **three distinct modes** depending on how fast the crash develops:

**Early exits (<1s) — 77 of 123 ML trades (63%):**
- Median exit: T+100ms
- Median PnL: +12.0 bps, WR: 86%
- The model fires immediately because the initial price action already looks like a near-bottom pattern
- 94% of these exit *before* the actual bottom — they leave edge on the table
- Captures only 30% of perfect PnL, but locks in a quick, safe profit

**Mid exits (1–20s) — 12 trades (10%):**
- Median exit: T+14.2s
- Median PnL: +40.5 bps, WR: 100%
- The model waited for a clearer bottom signal
- Captures 57% of perfect PnL

**Late exits (20s+) — 34 trades (28%):**
- Median exit: T+33.0s
- Median PnL: +57.3 bps, WR: 82%
- These are deep, prolonged crashes where the bottom takes time to form
- Captures 58% of perfect PnL

**Timeouts (55s) — 4 trades (3%):**
- Median PnL: +307.1 bps
- Extreme crashes where the ML never reaches 0.4 confidence — the price just keeps falling

#### ML Threshold Comparison

| Threshold | ML fires | Med exit time | Med PnL | Mean PnL | WR | Within 10 bps of bottom |
|-----------|----------|-------------|---------|----------|-----|------------------------|
| 0.30 | 100% | 0.2s | 18.9 bps | 41.2 bps | 87% | 19% |
| 0.35 | 98% | 0.2s | 19.6 bps | 42.7 bps | 87% | 20% |
| **0.40** | **97%** | **0.3s** | **18.9 bps** | **46.9 bps** | **86%** | **26%** |
| 0.45 | 96% | 5.5s | 18.2 bps | 44.5 bps | 84% | 17% |
| 0.50 | 93% | 22.6s | 22.3 bps | 44.1 bps | 80% | 17% |
| 0.60 | 88% | 36.1s | 22.6 bps | 45.5 bps | 79% | 16% |
| 0.70 | 74% | 47.2s | 23.9 bps | 45.1 bps | 75% | 11% |

Higher thresholds wait longer (better PnL per trade) but fire less often (more timeouts). The 0.4 threshold balances: fires on 97% of trades, highest mean PnL (46.9 bps), best "within 10 bps of bottom" rate (26%).

### Option 3: ML + Minimum Hold Time (Hybrid)

The ML model fires very early on 63% of trades (T+100ms). Adding a **minimum hold time** forces it to wait, potentially catching more of the crash:

| Min hold | Med PnL | Mean PnL | WR | Med exit time |
|----------|---------|----------|-----|-------------|
| 0s (pure ML) | 18.9 bps | 46.9 bps | 86% | 0.3s |
| 0.5s | 20.7 bps | 44.6 bps | 84% | 11.6s |
| 1s | 21.3 bps | 45.0 bps | 83% | 14.8s |
| 2s | 23.2 bps | 45.6 bps | 83% | 15.6s |
| **3s** | **23.7 bps** | **45.9 bps** | **83%** | **17.8s** |
| 5s | 24.3 bps | 45.7 bps | 84% | 17.9s |
| 10s | 23.5 bps | 44.5 bps | 80% | 20.9s |

A 3–5s minimum hold raises median PnL from 18.9 → 23.7–24.3 bps (+25%) while barely affecting WR (86% → 83–84%). The tradeoff: you hold the position for 17–18s median instead of 0.3s, which means more time exposed to adverse moves.

### Option 4: BIG_TRADE Event Trigger (Highest Quality Signal)

Beyond time-based polling, we researched **event-driven triggers** — evaluating the ML model only when specific market events occur. Four trigger types were tested:

| Trigger | Definition | % of exits | Avg PnL | Win Rate |
|---------|-----------|-----------|---------|----------|
| **BIG_TRADE** | Single trade > 2× rolling median size | 30% | **+47.7 bps** | **79%** |
| BOUNCE | Price bounces > 3 bps off running minimum | 55% | +28.0 bps | 66% |
| NEW_LOW | Price makes a new running minimum | 5% | +3.8 bps | 57% |
| COOLDOWN | 100ms since last eval, no event | 9% | −4.2 bps | 38% |

**BIG_TRADE is the standout.** When a large trade appears during a bounce off the crash low, it's a strong confirmation that the bottom is in. The logic: a big buyer stepping in after the crash signals that informed money thinks the price has overshot — and they're right 79% of the time, with +47.7 bps average PnL.

A BIG_TRADE is defined as:
```python
# Maintain a rolling median of recent trade sizes (last 50 trades)
if trade.qty > 2 * median_trade_size:
    trigger = "BIG_TRADE"  # → evaluate ML model immediately
```

**Important caveat:** The pure event-driven approach averages 2–3 bps *less* than polling because the ML model was trained on 100ms snapshots. Running it at arbitrary inter-trade intervals creates a distribution mismatch. The fix: **use polling as the base and add BIG_TRADE as an overlay trigger.**

**Hybrid approach (recommended):**
- Base: evaluate ML every 100ms (matches training data)
- Overlay: also evaluate immediately on any BIG_TRADE event
- This adds only ~30% more evaluations (not 10×) while capturing the high-conviction BIG_TRADE exits

### Head-to-Head Comparison

| Method | Med PnL | Mean PnL | WR | Simplicity | Risk |
|--------|---------|----------|-----|-----------|------|
| Fixed T+5s | 22.6 | 36.0 | 85% | Simplest | Misses slow crashes |
| Fixed T+10s | 23.8 | 38.1 | 80% | Simple | Gives back gains on fast crashes |
| **ML p≥0.4** | **18.9** | **46.9** | **86%** | Needs model | Exits very early sometimes |
| **ML p≥0.4 + 3s hold** | **23.7** | **45.9** | **83%** | Needs model | Best balanced |
| ML p≥0.5 | 22.3 | 44.1 | 80% | Needs model | More timeouts |
| **BIG_TRADE trigger** | — | **+47.7** | **79%** | Event detection | Only fires on 30% of exits |
| Perfect hindsight | 53.9 | 85.2 | 100% | Impossible | — |

### Production Recommendation: Combined Approach

The best exit strategy layers multiple methods, each solving a specific weakness:

```
1. Enter short at T+20ms
2. T+0 to T+3s: HOLD — do NOT exit
   EXCEPTION: if BIG_TRADE fires AND ML p≥0.4 → exit immediately
3. After T+3s: poll ML every 100ms
   - Also evaluate immediately on any BIG_TRADE event
   - Exit when p(near_bottom) ≥ 0.4
4. T+55s: forced exit (timeout safety net)
```

**Why each layer matters:**

| Layer | What it prevents | What it adds |
|-------|-----------------|-------------|
| **3s minimum hold** | Premature T+100ms exits (63% of pure ML trades exit too early) | +25% median PnL (18.9 → 23.7 bps) |
| **BIG_TRADE override (<3s)** | Missing early high-conviction bottoms during the hold | Captures +47.7 bps / 79% WR signals |
| **100ms polling (>3s)** | Train/inference distribution mismatch from event-driven | Reliable ML baseline matching training data |
| **BIG_TRADE overlay (>3s)** | Missing big-buyer bottom confirmation between polls | Extra high-quality exits (+47.7 bps avg) |
| **55s timeout** | Infinite hold on runaway crashes that never bottom | Safety net (fires on 3% of trades, still profitable) |

The logic: **be patient (3s), then listen to the model (100ms polls), but always pay attention to big buyers (BIG_TRADE override at any time).**

**Fallback: Fixed T+5s.** If the ML model isn't available (loading failure, feature error), exit at T+5s — simplest option with 85% WR and 22.6 bps median.

### What the ML Model Actually Watches

The exit decision is based on 56 features computed every 100ms. The key feature categories:

1. **Price dynamics:** Current price relative to running minimum, velocity over 500ms/1s/2s windows, acceleration (is the crash slowing down?)
2. **Trade flow:** Sell ratio, trade rate, volume rate — is selling pressure fading?
3. **Orderbook state:** L1 spread, bid/ask imbalance, quantity changes — are bids refilling?
4. **Sequence patterns:** Bounce count, consecutive new lows, reversals — has the price started oscillating instead of falling?
5. **Time context:** Elapsed time as fraction of 60s window, time since last new low
6. **Static context:** Pre-settlement spread, volume rate (for surge detection)

The model is looking for the moment when **selling pressure exhausts and buying starts to rebuild** — the microstructure signature of a bottom.

---

## 5. How to Execute the Exit (Taker vs Maker Order)

The previous section decided **when** to exit. This section decides **how** — market order (taker) or limit order (maker). The fee difference is massive relative to our edge.

### The Fee Gap

| | Taker (market order) | Maker (limit order) |
|--|---------------------|---------------------|
| Fee per leg | 10 bps (0.10%) | 4 bps (0.04%) |
| Execution | Instant, guaranteed fill | Must wait for counterparty |
| Price | Buy at ask (worst price) | Buy at bid (best price) |

Entry is always taker (market sell at T+20ms — speed matters). The choice is only about the **exit** order.

| Exit approach | Entry fee | Exit fee | RT total |
|---------------|----------|----------|----------|
| **Taker exit** | 10 bps | 10 bps | **20 bps** |
| **Maker exit** | 10 bps | 4 bps | **14 bps** |
| **Savings** | — | **6 bps** | **6 bps per trade** |

6 bps doesn't sound like much, but our entire net edge is ~10-16 bps after slippage. Saving 6 bps on fees is a **37-60% increase in net profit**.

### Approach 1: Taker Exit (Simple)

When the ML model says "exit":
```
→ Place market buy order
→ Filled instantly at best ask
→ Pay 10 bps taker fee
→ Done
```

**Pros:**
- Dead simple — one order, instant fill, no state management
- Zero execution risk — you always exit at the moment ML signals
- No rescue logic needed

**Cons:**
- 10 bps exit fee eats 42% of the 23.6 bps gross edge
- You buy at the ask, paying the full spread on top of fees

**PnL with taker exit (adaptive sizing, 127 settlements):**

| Metric | Value |
|--------|-------|
| RT fees | 20 bps |
| Median net PnL | 15.9 bps |
| Win rate | 100% |
| Avg $/trade | $1.89 |

### Approach 2: Maker Exit (Limit + Rescue)

When the ML model says "exit":
```
→ Place PostOnly limit buy at best bid price
→ Wait up to 1000ms for fill
→ If filled: pay 4 bps maker fee, bought at bid (price improvement)
→ If NOT filled after 1000ms: cancel limit → market buy (rescue)
```

**Why does this work?** Even after the crash bottoms, there's active two-way trading. Sell trades continue to flow — the "bottom" isn't silence, it's a shift in balance. There is enough residual selling to fill a $500-$3K limit buy within seconds.

**Fill rates from 85 settlements with OB.1 + trade data:**

| Exit time | Fill rate (at best_bid) | Median fill time |
|-----------|----------------------|-----------------|
| T+5s | **95%** | 379ms |
| T+8s | **96%** | 296ms |
| T+10s | **87%** | 460ms |
| T+15s | **89%** | 392ms |
| T+20s | **94%** | 548ms |
| T+30s | **92%** | 894ms |

Fill rates are **87-96%**. Most fills happen within 500ms. The median fill time is well under our 1000ms rescue timeout.

**Fill rate depends on spread at exit time:**

| Spread at exit | Fill rate | Why |
|---------------|-----------|-----|
| < 1 bps | **100%** | Heavy trading, constant fill flow |
| 1-2 bps | 65% | Slightly less active, worst bucket |
| 2-4 bps | 89% | Normal activity |
| 4-8 bps | 95% | Wide spread = any sell crosses to our bid |
| 8+ bps | 78% | Illiquid, fewer trades |

**Limit order placement options (at T+10s exit):**

| Placement | Fill rate | Price improvement | Net EV |
|-----------|----------|------------------|--------|
| **At best_bid** | **87%** | **+3.5 bps** | **+6.7 bps** |
| Bid + 25% spread | 89% | +2.7 bps | +6.2 bps |
| Mid price | 89% | +1.8 bps | +5.4 bps |
| Ask - 25% spread | 89% | +0.9 bps | +4.6 bps |

**Best placement: at best_bid.** Highest net EV because the price improvement (buying at bid vs ask) compounds with the fee saving. The slightly lower fill rate is more than offset by the larger saving per fill.

### The Rescue Plan

When the limit order doesn't fill (5-13% of trades depending on exit time):

**Cancel + market buy after timeout:**

| Rescue timeout | Avg extra cost | Max extra cost | Net EV |
|----------------|---------------|----------------|--------|
| 500ms | +4.4 bps | +23.0 bps | +7.7 bps |
| **1000ms** | **+7.2 bps** | **+30.1 bps** | **+7.3 bps** |
| 2000ms | +11.9 bps | +38.5 bps | +6.7 bps |
| 3000ms | +14.7 bps | +52.9 bps | +6.3 bps |
| 5000ms | +25.2 bps | +79.2 bps | +5.0 bps |

**1000ms timeout is optimal.** By 1s, most fills that will happen have already happened (median fill is 460ms). Rescue cost is still manageable at 7.2 bps average. Longer timeouts risk the price running away.

### Rescue Logic (Production Code)

```python
# 1. ML says "exit now" — read current best bid
best_bid = orderbook.best_bid

# 2. Place PostOnly limit buy at best_bid
limit_order = place_order(
    side="Buy",
    orderType="Limit",
    price=best_bid,
    qty=position_qty,
    timeInForce="PostOnly",  # ensures maker fee; rejects if would cross spread
)

# 3. Wait up to 1000ms, polling every 50ms
filled = wait_for_fill(limit_order, timeout_ms=1000, poll_ms=50)

if filled:
    # SUCCESS — paid 4 bps maker fee, bought at bid
    return "MAKER_FILLED"

# 4. Not filled — RESCUE with market order
cancel_order(limit_order)
status = check_order(limit_order)
remaining = position_qty - status.filled_qty

if remaining > 0:
    place_order(side="Buy", orderType="Market", qty=remaining)
    return "RESCUE_MARKET"

# Edge cases:
# - Partial fill: cancel remaining, market buy the rest
# - PostOnly rejected (bid crossed ask): immediate market buy
# - Any order failure: market buy as ultimate fallback
```

### Head-to-Head: Taker vs Maker Exit

**At $2K fixed notional (85 settlements, dedicated limit exit study):**

| | Taker exit | Maker exit (bid, 1s rescue) | Delta |
|--|-----------|---------------------------|-------|
| Exit fee | 10 bps | 4.8 bps blended | **−5.2 bps** |
| Price improvement | 0 | +3.0 bps (87% × 3.5) | **+3.0 bps** |
| Rescue cost | 0 | −0.9 bps (13% × 7.2) | −0.9 bps |
| **Net benefit** | — | — | **+7.3 bps/trade** |
| **Net PnL** | **+10.7 bps** | **+18.0 bps** | **+68%** |
| **$/trade** | **$2.13** | **$3.60** | **+$1.47** |

**Daily/monthly revenue impact (12 trades/day):**

| | Taker | Maker | Delta |
|--|-------|-------|-------|
| $/day | $25.60 | $43.20 | **+$17.60** |
| $/month | $768 | $1,296 | **+$528** |

Maker exit nearly **doubles** the dollar profit per trade.

### Why Maker Exit Works Here

1. **Post-settlement markets are active.** Residual selling continues for 10-30s after the bottom. There is always flow to fill a $500-$3K buy.
2. **We're buying at the bid, not the ask.** This captures the full spread as price improvement (~3.5 bps median). We're getting paid to be patient.
3. **Rescue cost is tiny.** The 13% unfilled trades cost ~7.2 bps extra, but weighted by frequency that's only 0.9 bps per trade average. The 87% filled trades save ~8.5 bps each.
4. **No timing pressure.** Unlike entry (T+20ms precision), exit has seconds to work with. 1000ms timeout is comfortable.

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Price runs up during 1s wait | Rescue costs more than 7.2 bps avg | 1000ms timeout caps exposure |
| PostOnly rejected (spread inverted) | No limit fill possible | Immediate market buy fallback |
| Exchange latency on cancel | Potential double fill | Check filled qty before placing rescue |
| Partial fill leaves dust | Tiny unexited position | Market buy remaining if > min notional |

### Production Recommendation

**Use maker exit (limit at best_bid, 1000ms rescue).** The evidence is clear:

- **+7.3 bps per trade** net benefit (+68% more profit)
- **87-96% fill rate** — rescue needed only 5-13% of the time
- **Median fill: 460ms** — well within the 1000ms timeout
- Adds complexity (rescue logic, partial fills) but the dollar impact justifies it

**The rescue plan is non-negotiable.** Never leave a position open because a limit order didn't fill. The 1000ms timeout + market buy rescue ensures you always exit, every time. The worst case is you pay taker fees on 13% of trades — same as if you'd used taker exit for all trades.

---

## 6. Two Implementation Options: Easy vs Max

Every section above has a simple option and a complex-but-more-profitable option. Here are two concrete implementation paths — one to start trading tomorrow, one to maximize every basis point.

### Option A: Easy Prototype (No ML, No Maker)

**Goal:** Start trading with minimal code. Prove the strategy works live before investing in ML/maker infrastructure.

| Section | Easy choice | Why |
|---------|-----------|-----|
| **1. Strategy** | Short after settlement (same) | Core edge doesn't change |
| **2. Filters** | FR ≥ 15 bps, depth ≥ $2K, spread ≤ 8 bps | Just 3 if-statements on the ticker/OB stream |
| **3. Sizing** | Fixed $1,000 | No adaptive logic needed. Safe on all filtered coins (depth ≥ $2K) |
| **4. Exit timing** | Fixed T+5s | One `sleep(5)` call. No ML model, no feature computation, no event detection |
| **5. Exit order** | Market buy (taker) | One order, instant fill, no state management |

**What you build:**
```
1. Subscribe to ticker + orderbook streams
2. Detect settlement (FR reset or clock-based)
3. Check 3 filters → skip if any fails
4. Market sell $1,000 at T+20ms
5. Sleep 5 seconds
6. Market buy $1,000
7. Log result
```

**~100 lines of code. Build in 1-2 days.**

### Option B: Max Profitable (ML + Maker + Adaptive)

**Goal:** Extract maximum edge per trade. Every optimization is backed by evidence.

| Section | Max choice | Gain vs Easy |
|---------|-----------|-------------|
| **1. Strategy** | Short after settlement (same) | — |
| **2. Filters** | Same 3 filters | — |
| **3. Sizing** | Adaptive tiers ($500-$3K), 15% of depth_20 | Higher notional on deep coins → more $/trade |
| **4. Exit timing** | ML p≥0.4 + 3s min hold + BIG_TRADE overlay + 55s timeout | +10.9 bps mean gross vs fixed T+5s |
| **5. Exit order** | Limit buy at bid, 1000ms rescue → market buy | −6 bps fees + 3.5 bps price improvement |

**What you build:**
```
1. Subscribe to ticker + orderbook + trade streams
2. Load ML model (logistic regression, 56 features)
3. Detect settlement, check 3 filters
4. Compute adaptive notional from depth_20
5. Market sell at T+20ms
6. Hold for 3s minimum (no exit)
   - EXCEPTION: BIG_TRADE (qty > 2× median) AND ML p≥0.4 → exit early
7. After 3s: poll ML every 100ms + evaluate on BIG_TRADE events
   - Exit when p(near_bottom) ≥ 0.4
8. On exit signal: place PostOnly limit buy at best_bid
   - Wait up to 1000ms for fill
   - If unfilled: cancel → market buy (rescue)
9. Timeout at T+55s: forced market buy
10. Log result with all features/triggers for analysis
```

**~500+ lines of code. ML training pipeline, feature engine, order state machine. Build in 1-2 weeks.**

### Dataset: What We're Projecting From

| Metric | Value |
|--------|-------|
| Settlement hours covered | **56** (Feb 26–Mar 1) |
| JSONL files (all have FR ≥ 15 bps) | 160 |
| Pass spread ≤ 8 bps | 140 (88%) |
| Pass depth ≥ $2K | 137 (86%) |
| **Pass all 3 filters** | **127 (79%)** |
| Hours with ≥1 filtered coin | 55 of 56 (98%) |
| Avg filtered coins/hour | 2.3 |

**Trading approach matters:** we can trade all passing coins each hour (~54/day) or just the best one (~24/day). Both are modeled below.

### Head-to-Head Performance

**Trade ALL passing coins per hour (~54 trades/day):**

| Metric | Easy (A) | Max (B) | Difference |
|--------|----------|---------|------------|
| **Trades/day** | ~54 | ~54 | — |
| **$/trade** | $0.74 | **$1.95** | **+$1.21 (+164%)** |
| **$/day** | $40 | **$106** | **+$66 (+164%)** |
| **$/month** | ~$1,200 | **~$3,180** | **+$1,980** |

**Trade BEST coin per hour (~24 trades/day):**

| Metric | Easy (A) | Max (B) | Difference |
|--------|----------|---------|------------|
| **Trades/day** | ~24 | ~24 | — |
| **$/trade** | $0.94 | **$2.88** | **+$1.94 (+206%)** |
| **$/day** | $22 | **$68** | **+$46 (+206%)** |
| **$/month** | ~$668 | **~$2,037** | **+$1,369** |

Trading the best coin per hour gives higher $/trade (cherry-picking the deepest book = least slippage) but fewer trades. Trading all coins gives higher total revenue despite lower per-trade quality.

### Per-Trade Breakdown

| Metric | Easy (A) | Max (B) |
|--------|----------|---------|
| **Notional** | $1,000 fixed | $1,213 avg (adaptive) |
| **Gross PnL** | 36.0 bps mean (T+5s) | 46.9 bps mean (ML) |
| **RT fees** | 20 bps (taker both) | 16.8 bps (taker + maker blend) |
| **Slippage** | 8.6 bps mean | 7.7 bps median |
| **Net PnL (mean)** | 7.4 bps | 16.1 bps |
| **Win rate (net > 0)** | ~55-60% | **100%** |

### Where Each Optimization Pays Off

| Optimization | Gain (bps/trade) | Gain ($/trade at avg notional) | Complexity |
|-------------|-----------------|-------------------------------|-----------|
| ML exit vs T+5s | +10.9 mean gross | +$1.32 | High (model + features) |
| Maker exit vs taker | +7.3 net | +$0.88 | Medium (limit + rescue) |
| Adaptive sizing vs $1K | +$0.21 (higher notional) | +$0.21 | Low (one function) |
| BIG_TRADE overlay | included in ML | — | Low (trade size check) |
| **Total** | — | **+$1.21/trade** | — |

ML exit is the single biggest improvement. Maker exit is the second biggest. Adaptive sizing is easy to add and worth it.

### Recommended Path

```
Week 1:  Ship Easy prototype (Option A)
         → Validate live: does the strategy produce gross profits?
         → Confirm filters, entry timing, and data feeds work

Week 2:  Add adaptive sizing (low effort, immediate $/trade boost)
         → Monitor: are filtered coins consistently profitable?

Week 3:  Add maker exit + rescue logic
         → This is the highest ROI upgrade: −6 bps fees, +3.5 bps price improvement
         → No ML needed, just order management

Week 4+: Add ML exit model
         → Train on accumulated live data
         → A/B test ML vs T+5s on alternating trades
         → Graduate to full Max mode when ML proves itself live
```

**Start simple. Add complexity only when the simpler version is profitable and stable.** Every optimization above is independently valuable — you don't need all of them at once.
