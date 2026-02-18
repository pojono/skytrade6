# v31: Tick-Level Microstructure Regime Classification

## Goal

Classify market into 4 states using tick-level data (trades, ticker, liquidations):
1. **COMPRESSION** — low vol, narrowing range, declining volume, OI building (coiling spring)
2. **BREAKOUT** — sudden vol expansion, high trade intensity, directional move
3. **SQUEEZE** — liquidation cascade, OI dropping, one-sided flow, fast price move
4. **EXHAUSTION** — high vol but fading momentum, volume declining from peak, reversal signals

## Data

- **Sources**: Bybit futures trades, ticker (OI/FR/bid-ask), liquidations
- **Resolution**: Per-second aggregation → 5-minute (300s) classification bars
- **Feature horizons**: 60s, 300s, 900s

## Method

### Phase 1: Rule-Based Classification
Adaptive percentile thresholds on:
- **Volatility** (realized vol of log returns)
- **Range** (high-low / mid in bps)
- **Trade intensity** (count, notional, acceleration)
- **Liquidation intensity** (count, notional, imbalance)
- **OI change** (delta %)
- **Return efficiency** (|net return| / range — low = choppy/exhaustion)

Priority: Squeeze > Breakout > Exhaustion > Compression

### Phase 2: Unsupervised Clustering
KMeans (k=3,4,5) and GMM (k=4) on 17 standardized features.

### Phase 3: Cross-Validation
Tested across 3 configurations:
- BTC May 12-14 (3 days)
- BTC May 12-18 (7 days)
- ETH May 12-14 (3 days)
- BTC Aug 4-6 (3 days, different market period)

## Results

### Regime Distribution (remarkably stable across all tests)

| Test | Compression | Breakout | Squeeze | Exhaustion |
|------|------------|----------|---------|------------|
| BTC May 12-14 (3d) | 49.9% | 9.6% | 4.6% | 35.8% |
| BTC May 12-18 (7d) | 49.7% | 10.3% | 4.0% | 36.0% |
| ETH May 12-14 (3d) | 49.7% | 7.9% | 5.4% | 37.0% |
| BTC Aug 4-6 (3d) | 49.9% | 9.3% | 2.5% | 38.2% |

**Key**: Market spends ~50% of time in compression, ~36% in exhaustion, ~10% in breakout, ~4% in squeeze. This is consistent across assets and time periods.

### Transition Matrix (BTC 7-day, representative)

|  | → COMPRESSION | → BREAKOUT | → SQUEEZE | → EXHAUSTION |
|--|--------------|-----------|----------|-------------|
| **COMPRESSION** | **78.6%** | 1.9% | 1.3% | 18.2% |
| **BREAKOUT** | 5.8% | **36.7%** | 8.2% | **49.3%** |
| **SQUEEZE** | 8.8% | **27.5%** | 23.8% | **40.0%** |
| **EXHAUSTION** | **26.9%** | 12.6% | 4.3% | **56.3%** |

**Key transitions**:
- Compression is very sticky (75-79% self-transition)
- Breakout → Exhaustion is the dominant path (49-67%)
- Squeeze → Breakout or Exhaustion (not back to compression)
- Exhaustion → Compression (27-33%) — the "cool down" path
- **Natural cycle**: Compression → Breakout → Exhaustion → Compression

### Regime Duration

| Regime | Avg Duration | Max Duration | Episodes/week |
|--------|-------------|-------------|---------------|
| Compression | 23 min | 250 min (4h) | ~30/day |
| Breakout | 8 min | 30 min | ~19/day |
| Squeeze | 7 min | 20 min | ~9/day |
| Exhaustion | 11 min | 70 min | ~45/day |

### Forward Returns by Regime

#### BTC (7-day, May 12-18)
| After Regime | Mean Return | |Return| | Std | n |
|-------------|------------|---------|-----|---|
| Compression | +0.27 bps | 5.65 bps | 8.35 bps | 1002 |
| Breakout | -0.09 bps | **13.62 bps** | 17.44 bps | 207 |
| Squeeze | **-1.65 bps** | **14.73 bps** | 18.44 bps | 80 |
| Exhaustion | +0.13 bps | 8.76 bps | 11.87 bps | 725 |

#### ETH (3-day, May 12-14)
| After Regime | Mean Return | |Return| | Std | n |
|-------------|------------|---------|-----|---|
| Compression | +1.36 bps | 16.03 bps | 21.72 bps | 428 |
| Breakout | **-7.77 bps** | **32.63 bps** | 43.31 bps | 68 |
| Squeeze | +2.88 bps | **33.59 bps** | 40.94 bps | 47 |
| Exhaustion | +0.53 bps | 21.71 bps | 28.47 bps | 319 |

#### BTC (3-day, Aug 4-6, out-of-sample period)
| After Regime | Mean Return | |Return| | Std | n |
|-------------|------------|---------|-----|---|
| Compression | -0.16 bps | 4.91 bps | 6.45 bps | 430 |
| Breakout | +1.49 bps | **8.00 bps** | 12.10 bps | 80 |
| Squeeze | +0.44 bps | **10.83 bps** | 14.34 bps | 22 |
| Exhaustion | +0.02 bps | 6.63 bps | 9.10 bps | 330 |

### Unsupervised vs Rule-Based

- **Low agreement**: ARI = 0.11-0.17 between rule-based and KMeans/GMM
- **KMeans vs GMM agreement**: ARI = 0.38-0.42 (moderate)
- KMeans naturally finds ~3 clusters: one large "quiet" cluster (~47%), one "active" cluster (~41%), and one "extreme" cluster (~9%)
- The unsupervised clusters don't map cleanly to the 4 rule-based regimes — they primarily separate by volume/volatility level, not by the qualitative regime characteristics

## Key Findings

### 1. Regime Distribution is Universal (~50/36/10/4)
The compression/exhaustion/breakout/squeeze split is remarkably stable across BTC, ETH, and different time periods. This suggests these are fundamental market microstructure states.

### 2. Clear Regime Lifecycle
**Compression → Breakout → Exhaustion → Compression** is the dominant cycle. Squeeze is a rare but important variant of breakout (with liquidation cascade).

### 3. Regimes Predict Forward Volatility (2.5x spread)
- After breakout/squeeze: |ret| is 2-2.5x higher than after compression
- This is actionable for volatility-based strategies (straddles, gamma scalping)

### 4. Squeeze Has Negative Forward Return (BTC)
- BTC squeeze → mean -1.65 bps next 5min (potential mean-reversion signal)
- ETH breakout → mean -7.77 bps (even stronger mean-reversion)
- Caution: small sample sizes (22-80 events)

### 5. Compression is Sticky but Breakout is Not
- Compression self-transition: 75-79% (can last hours)
- Breakout self-transition: 20-37% (typically 1-2 bars = 5-10 min)
- This means breakout detection must be fast — by the time you detect it, it may already be transitioning to exhaustion

### 6. Unsupervised Clustering Disagrees with Rules
- The natural data clusters are primarily volume/volatility-based (quiet vs active)
- The rule-based regimes capture more nuanced state (e.g., distinguishing breakout from squeeze requires liquidation data)
- Suggests the 4-state model adds information beyond simple vol regimes

## Limitations & Next Steps

1. **Small sample for squeeze** — need longer date ranges to get statistical significance
2. **Rule thresholds are percentile-based** — may not generalize to regime changes (e.g., 2024 vs 2025 market structure)
3. **No directional prediction yet** — regimes predict vol but not direction
4. **Rolling window features are lagging** — by definition, 300s features describe the past 5 min
5. **Could try ML classifier** — train on rule-based labels, use as features for a more nuanced model
6. **Test on SOL/DOGE** — higher-vol assets may have different regime dynamics
7. **Regime-conditional strategies** — e.g., grid bot in compression, momentum in breakout, mean-reversion in squeeze
