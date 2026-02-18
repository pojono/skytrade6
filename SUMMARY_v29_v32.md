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
1. **Add weekday/activity filter**: Skip trading when vol_3600s is below a threshold
2. **Test on 14+ days**: Need more walk-forward days to confirm weekend effect
3. **Test on more symbols**: ETH (already profitable unconditionally), SOL, DOGE
4. **Paper trading**: Implement on Bybit testnet with real order book
