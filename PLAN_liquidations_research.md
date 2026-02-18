# Liquidations Data Research Plan

**Created**: February 18, 2026  
**Data Period**: Feb 9-17, 2026 (9 days)  
**Symbols**: BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT  

---

## Data Summary

### Downloaded Files
- **BTCUSDT**: 197 files (~300 KB compressed)
- **ETHUSDT**: 199 files (~217 KB compressed)
- **SOLUSDT**: 197 files (~181 KB compressed)
- **DOGEUSDT**: 178 files (~94 KB compressed)
- **XRPUSDT**: 192 files (~145 KB compressed)

### Data Structure
```json
{
    "ts": 1770665589949,           // Websocket timestamp (ms)
    "exchange": "bybit",
    "source": "ws",
    "stream": "liquidation",
    "symbol": "BTCUSDT",
    "result": {
        "ts": 1770665589887,       // Event timestamp (ms)
        "data": [
            {
                "T": 1770665589426,  // Liquidation timestamp (ms)
                "s": "BTCUSDT",      // Symbol
                "S": "Sell",         // Side (Buy/Sell)
                "v": "0.017",        // Volume (contracts)
                "p": "71182.70"      // Price
            }
        ]
    }
}
```

### Key Fields
- **T**: Exact liquidation timestamp
- **S**: Side - "Buy" (long liquidated) or "Sell" (short liquidated)
- **v**: Volume in contracts
- **p**: Liquidation price
- **Resolution**: Real-time (sub-second granularity)

---

## Research Questions

### 1. Liquidation Cascades & Price Impact
**Hypothesis**: Large liquidation events trigger price moves and subsequent cascades

**Analysis**:
- Detect liquidation clusters (multiple liquidations within 1-5 seconds)
- Measure price impact in next 10s, 30s, 1min, 5min
- Identify cascade patterns (initial liquidation → price move → more liquidations)
- Compare cascade severity across volatility regimes

**Metrics**:
- Cascade size (total volume liquidated)
- Cascade duration (time from first to last liquidation)
- Price impact (max drawdown/rally after cascade starts)
- Recovery time (time to return to pre-cascade price)

**Expected Findings**:
- Large cascades (>$1M notional) should predict short-term continuation
- Cascades in low liquidity periods have larger price impact
- Buy liquidations (longs getting stopped) → price drops further
- Sell liquidations (shorts getting stopped) → price rallies further

---

### 2. Liquidation Imbalance as Directional Signal
**Hypothesis**: Imbalance between long vs short liquidations predicts reversals

**Analysis**:
- Calculate rolling imbalance: (sell_liq_volume - buy_liq_volume) / total_liq_volume
- Aggregate over 5min, 15min, 1h windows
- Test predictive power for forward returns (5min, 15min, 1h, 4h)
- Compare to OI changes and funding rate

**Metrics**:
- Liquidation imbalance ratio
- Cumulative liquidation volume by side
- Information coefficient vs forward returns
- Sharpe ratio of long/short strategy based on imbalance

**Expected Findings**:
- Extreme long liquidations (imbalance < -0.5) → mean reversion (buy signal)
- Extreme short liquidations (imbalance > +0.5) → mean reversion (sell signal)
- Moderate imbalances may predict continuation
- Works best at regime transitions (quiet → volatile)

---

### 3. Liquidation Volume vs OI Velocity
**Hypothesis**: Liquidations explain OI changes better than organic position changes

**Analysis**:
- Load 5-second ticker data (OI snapshots)
- Load liquidations data (real-time events)
- Calculate OI velocity (rate of change)
- Decompose OI changes into: liquidations vs organic flow
- Measure correlation and lead/lag relationships

**Metrics**:
- OI change explained by liquidations (R²)
- Residual OI change (organic flow)
- Lead/lag correlation (does OI lead liquidations or vice versa?)
- Regime-dependent analysis (quiet vs volatile)

**Expected Findings**:
- Liquidations explain 30-60% of OI changes in volatile regimes
- Organic flow dominates in quiet regimes
- OI spikes often precede liquidation cascades (leverage buildup)
- Liquidations lag price moves by 0-5 seconds

---

### 4. Liquidation Heatmaps & Support/Resistance
**Hypothesis**: Price levels with historical liquidation clusters act as magnets

**Analysis**:
- Build liquidation density heatmap by price level
- Identify "liquidation zones" (price ranges with high historical liquidations)
- Test if price gravitates toward these zones
- Measure probability of reversal at liquidation zones

**Metrics**:
- Liquidation density by price level (volume per $100 range)
- Time spent near liquidation zones vs random levels
- Reversal probability at liquidation zones
- Breakout probability (price breaks through zone)

**Expected Findings**:
- Price tends to visit liquidation-rich zones (stop hunting)
- Reversals more likely at major liquidation zones
- Breakouts through zones are explosive (cascade effect)
- Zones shift over time as leverage rebuilds

---

### 5. Liquidations as Volatility Predictor
**Hypothesis**: Liquidation rate predicts upcoming volatility spikes

**Analysis**:
- Calculate liquidation rate (events per minute)
- Measure realized volatility in next 5min, 15min, 1h
- Build volatility prediction model using liquidation features
- Compare to traditional volatility predictors (ATR, Bollinger Bands)

**Metrics**:
- Liquidation rate (events/min, volume/min)
- Forward realized volatility
- Prediction R² and IC
- Regime transition detection accuracy

**Expected Findings**:
- Rising liquidation rate predicts volatility spike in 5-15 minutes
- Liquidation rate is a leading indicator (vs lagging indicators like ATR)
- Works best for detecting quiet → volatile transitions
- Can be used for dynamic position sizing

---

### 6. Liquidation-Based Market Microstructure
**Hypothesis**: Liquidations reveal hidden order flow and market maker behavior

**Analysis**:
- Analyze liquidation timing relative to price ticks
- Identify "sweep" patterns (rapid liquidations across price levels)
- Detect market maker absorption (large liquidations with minimal price impact)
- Study liquidation clustering by time-of-day and day-of-week

**Metrics**:
- Liquidation-to-price-tick delay
- Sweep detection (consecutive liquidations across price levels)
- Price impact per $1M liquidated (efficiency)
- Temporal patterns (hourly, daily)

**Expected Findings**:
- Liquidations cluster during low liquidity hours (Asia session)
- Market makers absorb liquidations during high liquidity (US/EU session)
- Sweeps predict continuation (aggressive liquidation hunting)
- Weekend liquidations have larger price impact

---

### 7. Cross-Asset Liquidation Contagion
**Hypothesis**: Liquidations in one asset trigger liquidations in correlated assets

**Analysis**:
- Detect liquidation events in BTC
- Measure liquidation response in ETH, SOL, DOGE, XRP
- Calculate cross-asset liquidation correlation
- Identify lead/lag relationships (which asset liquidates first?)

**Metrics**:
- Cross-asset liquidation correlation (5s, 30s, 1min windows)
- Lead/lag analysis (Granger causality)
- Contagion probability (P(ETH liquidation | BTC liquidation))
- Contagion severity (volume ratio)

**Expected Findings**:
- BTC liquidations lead altcoin liquidations by 5-30 seconds
- Correlation is stronger in volatile regimes
- SOL/DOGE have highest contagion (most speculative)
- ETH is more independent (institutional positioning)

---

### 8. Liquidation-Enhanced Trading Strategies

#### Strategy 1: Cascade Fade
- **Entry**: After large liquidation cascade (>$5M notional)
- **Direction**: Fade the move (buy after sell liquidations, sell after buy liquidations)
- **Exit**: 5-15 minute mean reversion target
- **Stop**: Cascade continues (more liquidations in same direction)

#### Strategy 2: Imbalance Reversal
- **Entry**: Extreme liquidation imbalance (>70% one-sided)
- **Direction**: Opposite to liquidation side
- **Exit**: Imbalance normalizes or 1-hour target
- **Stop**: Imbalance worsens

#### Strategy 3: Liquidation Zone Bounce
- **Entry**: Price reaches historical liquidation zone
- **Direction**: Reversal (buy at support zone, sell at resistance zone)
- **Exit**: Price moves away from zone (>0.5% move)
- **Stop**: Zone breaks (cascade through zone)

#### Strategy 4: Volatility Breakout
- **Entry**: Liquidation rate spikes (>3σ above mean)
- **Direction**: Trend following (direction of price move)
- **Exit**: Liquidation rate normalizes
- **Stop**: Price reverses before volatility spike

---

## Implementation Plan

### Phase 1: Data Processing (Week 1)
1. **Build liquidations parser**
   - Load JSONL files
   - Extract T, S, v, p fields
   - Convert to pandas DataFrame
   - Handle multiple liquidations per message

2. **Aggregate liquidations**
   - 1-second bars (count, volume, buy/sell split)
   - 5-second bars (align with ticker data)
   - 1-minute bars (for strategy backtesting)
   - Hourly bars (for regime analysis)

3. **Merge with existing data**
   - Join with 5-second ticker (OI, funding, price)
   - Join with 1-hour long/short ratio
   - Create unified dataset

### Phase 2: Exploratory Analysis (Week 1-2)
1. **Liquidation statistics**
   - Total volume by symbol
   - Buy vs sell distribution
   - Hourly/daily patterns
   - Size distribution (histogram)

2. **Cascade detection**
   - Define cascade criteria (volume threshold, time window)
   - Label cascade events
   - Measure cascade characteristics

3. **Visualization**
   - Liquidation heatmap by price level
   - Time series of liquidation rate
   - Cascade examples with price overlay

### Phase 3: Feature Engineering (Week 2)
1. **Liquidation features**
   - `liq_volume_1s/5s/1m`: Total liquidation volume
   - `liq_buy_volume/liq_sell_volume`: Volume by side
   - `liq_imbalance`: (sell - buy) / total
   - `liq_rate`: Events per minute
   - `liq_cascade_flag`: Binary cascade indicator
   - `liq_cascade_size`: Cumulative cascade volume

2. **Rolling statistics**
   - `liq_volume_mean/std/max`: Rolling 5min, 15min, 1h
   - `liq_imbalance_zscore`: Z-score of imbalance
   - `liq_rate_zscore`: Z-score of liquidation rate

3. **Cross-asset features**
   - `liq_btc_lead`: BTC liquidation volume 30s ago
   - `liq_correlation_5m`: Cross-asset liquidation correlation

### Phase 4: Predictive Analysis (Week 2-3)
1. **IC analysis**
   - Calculate IC for all liquidation features vs forward returns
   - Test across multiple horizons (5min, 15min, 1h, 4h)
   - Compare to OI/funding features from v25

2. **Walk-forward validation**
   - Train/test split by time
   - Ridge regression baseline
   - Gradient boosting (XGBoost)
   - Feature importance analysis

3. **Regime-dependent analysis**
   - Separate quiet vs volatile regimes
   - Test if liquidations work better in certain regimes
   - Combine with HMM regime detection from v21

### Phase 5: Strategy Backtesting (Week 3-4)
1. **Implement 4 strategies** (listed above)
2. **Backtest on 9-day dataset**
   - Calculate returns, Sharpe, win rate
   - Measure drawdowns
   - Transaction cost analysis

3. **Optimization**
   - Parameter tuning (thresholds, windows)
   - Risk management (position sizing, stops)
   - Portfolio construction (combine strategies)

4. **Out-of-sample validation**
   - Download additional data (Feb 18-28)
   - Test strategies on new data
   - Measure performance degradation

---

## Expected Deliverables

### Code
- `load_liquidations.py`: Data loading and parsing
- `liquidations_features.py`: Feature engineering
- `liquidations_analysis.py`: Exploratory analysis
- `liquidations_strategies.py`: Strategy backtesting
- `liquidations_cascade_detector.py`: Real-time cascade detection

### Results
- `FINDINGS_v26_liquidations.md`: Main findings document
- `results/liquidations_stats_*.txt`: Summary statistics
- `results/liquidations_ic_*.txt`: IC analysis results
- `results/liquidations_backtest_*.txt`: Strategy performance
- `results/liquidations_cascade_examples.png`: Visualization

### Insights
- Quantify liquidation cascade impact
- Identify best liquidation-based signals
- Compare liquidations vs OI/funding for prediction
- Validate or reject liquidation-based strategies

---

## Success Criteria

### Minimum Viable Results
- ✅ Successfully parse and aggregate liquidations data
- ✅ Detect at least 50 cascade events across all symbols
- ✅ Calculate liquidation imbalance and test IC (even if negative)
- ✅ Backtest at least 2 strategies with full metrics

### Good Results
- ✅ Liquidation features show IC > 0.05 for some horizons
- ✅ At least 1 strategy has Sharpe > 1.0 in-sample
- ✅ Cascade detection accuracy > 70%
- ✅ Liquidations explain >30% of OI changes

### Excellent Results
- ✅ Liquidation features outperform OI/funding features
- ✅ Multiple strategies with Sharpe > 1.5 in-sample and > 1.0 out-of-sample
- ✅ Cross-asset contagion is statistically significant
- ✅ Liquidation zones predict reversals with >60% accuracy

---

## Risk Factors & Challenges

### Data Quality
- **Issue**: Liquidations may be incomplete (only public liquidations)
- **Mitigation**: Compare total liquidation volume to OI changes for validation

### Look-Ahead Bias
- **Issue**: Websocket timestamps may not reflect actual execution time
- **Mitigation**: Use conservative delays (5-10s) in strategy backtesting

### Overfitting
- **Issue**: Only 9 days of data for backtesting
- **Mitigation**: Simple strategies, minimal parameters, out-of-sample validation

### Market Regime
- **Issue**: Feb 9-17 may not be representative (specific volatility regime)
- **Mitigation**: Download additional periods, test across different regimes

### Execution Challenges
- **Issue**: Liquidation-based strategies may require sub-second execution
- **Mitigation**: Focus on 1-5 minute horizons, realistic slippage assumptions

---

## Next Steps

1. **Immediate** (Today):
   - Create `load_liquidations.py` script
   - Parse and validate data for all symbols
   - Generate summary statistics

2. **Short-term** (This week):
   - Implement cascade detection
   - Build liquidation features
   - Run IC analysis vs forward returns

3. **Medium-term** (Next week):
   - Merge with ticker/OI data
   - Test liquidation vs OI decomposition
   - Backtest cascade fade strategy

4. **Long-term** (Next 2 weeks):
   - Complete all 4 strategy backtests
   - Download additional data for validation
   - Write comprehensive findings document

---

## Comparison to Previous Research

### v25 (High-Res OI/Funding)
- **Data**: 5-second ticker snapshots
- **Result**: Negative IC, features don't work
- **Issue**: OI changes are noisy, hard to interpret

### v26 (Liquidations)
- **Data**: Real-time liquidation events
- **Advantage**: Direct measure of forced position closures
- **Hypothesis**: Cleaner signal than OI snapshots

**Key Question**: Do liquidations explain why v25 failed?
- If liquidations = most OI changes → OI velocity is just liquidation noise
- If liquidations ≠ OI changes → organic flow is the real signal

---

## References & Related Work

### Academic
- "Liquidation Cascades in Cryptocurrency Markets" (2021)
- "The Impact of Forced Selling on Asset Prices" (2020)
- "Market Microstructure in Crypto Derivatives" (2022)

### Industry
- Bybit liquidation documentation
- Binance liquidation analysis tools
- Coinglass liquidation heatmaps

### Internal
- `FINDINGS_v25_hires_oi_funding.md`: OI velocity research
- `FINDINGS_v21_hmm_regimes.md`: Regime detection
- `FINDINGS_v20_regime_classification.md`: Volatility regimes

---

**Status**: Ready to implement  
**Priority**: High (liquidations may be the missing piece from v25)  
**Timeline**: 2-4 weeks for complete analysis  
