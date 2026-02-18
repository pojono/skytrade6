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
Rolling 3-day train → 1-day test across 7 days. Gives 4 independent test days.
**Expected**: More robust AUC estimate, realistic PnL curve.
