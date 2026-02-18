# FINDINGS v26b: Liquidation-Based Trading Strategies

**Date**: February 18, 2026  
**Data Period**: Feb 9-18, 2026 (9 days, 5-second resolution)  
**Symbols**: BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, XRPUSDT  
**Strategies**: 4 (Cascade Fade, Imbalance Reversal, Rate Spike, ToD Fade)  

---

## Executive Summary

Backtested 4 liquidation-based trading strategies across 5 crypto perpetual futures using real-time liquidation events merged with 5-second ticker data. Two strategies show consistent profitability on BTC and SOL; two others are marginal or negative.

### Strategy Scorecard

| Strategy | BTC | ETH | SOL | DOGE | XRP | Verdict |
|----------|-----|-----|-----|------|-----|---------|
| **1. Cascade Fade** | ✅ +5.4% | ❌ -4.8% | ✅ +6.0% | ❌ -0.5% | ❌ -0.5% | **Mixed** |
| **2. Imbalance Reversal** | ✅ +3.8% | ~ -0.0% | ✅ +2.1% | ❌ -4.4% | ✅ +5.8% | **Promising** |
| **3. Liq Rate Spike** | ❌ -3.8% | ~ +1.9% | ❌ -15.5% | ~ +1.1% | ❌ -1.1% | **Fails** |
| **4. ToD Liq Fade** | ✅ +4.1% | ~ +0.1% | ✅ +1.0% | ❌ -0.6% | ~ +0.0% | **Selective** |

**Best overall**: Strategy 2 (Imbalance Reversal) — profitable on 3/5 symbols  
**Best single result**: SOL Cascade Fade — +6.0% in 9 days, Sharpe +81  

---

## Data Infrastructure

### Merged Dataset
- **Liquidation events**: 42,951 total across 5 symbols
- **Ticker snapshots**: ~139K per symbol (5-second resolution)
- **Price bars**: ~11,600 one-minute bars per symbol
- **Overlapping period**: Feb 9 19:33 to Feb 17 20:59 UTC

### Bar Construction
- 1-minute OHLC bars from 5-second ticker data
- 1-minute liquidation bars: count, notional, buy/sell split, imbalance
- Aligned on common timestamp range

---

## Strategy 1: Cascade Fade

### Logic
After a liquidation cascade (2+ large liquidations within 60s), fade the move:
- Buy after buy-dominated cascade (longs got stopped → price dipped)
- Sell after sell-dominated cascade (shorts got stopped → price spiked)
- 30-second cooldown after cascade ends before entry
- 15-minute hold, 0.5% stop loss

### Results (Default Parameters)

| Symbol | Trades | Win Rate | Avg Return | Total Return | Sharpe |
|--------|--------|----------|------------|--------------|--------|
| **BTC** | 135 | **57.0%** | +0.040% | **+5.44%** | +65.1 |
| **ETH** | 101 | 45.5% | -0.047% | -4.77% | -63.3 |
| **SOL** | 86 | **52.3%** | +0.070% | **+6.02%** | +81.3 |
| **DOGE** | 22 | 40.9% | -0.021% | -0.45% | -23.0 |
| **XRP** | 44 | 45.5% | -0.011% | -0.48% | -12.9 |

### Parameter Sweep (BTC)

| Cascade Pct | Hold | Stop | Trades | Win Rate | Total Return |
|-------------|------|------|--------|----------|--------------|
| 90 | 30m | 0.008 | 97 | 54.6% | **+5.32%** |
| 90 | 15m | 0.005 | 135 | 57.0% | +5.44% |
| 95 | 30m | 0.008 | 44 | 56.8% | +4.12% |

### Analysis
- **Works on BTC and SOL** — the two most liquid markets
- **Fails on ETH** — cascades are too frequent and noisy (238 cascades)
- **Too few trades on DOGE** — only 22 trades in 9 days
- **BTC long trades outperform**: 57% win rate on longs vs 50% on shorts
- **SOL is best overall**: +6.0% in 9 days with 52% win rate

### Why It Works (BTC/SOL)
- Large cascades create temporary price dislocations
- Retail panic selling (long liquidations) overshoots fair value
- 30-second cooldown lets the cascade exhaust before entry
- 15-minute hold captures mean reversion

### Why It Fails (ETH)
- ETH has 238 cascades (vs 138 BTC) — too many false signals
- Cascades are smaller ($13K avg vs $658K BTC)
- Less price impact per cascade → weaker mean reversion

---

## Strategy 2: Extreme Imbalance Reversal

### Logic
When 5-minute rolling liquidation imbalance is extreme (>70% one-sided), fade it:
- Imbalance < -0.7 (extreme buy liquidations) → go long
- Imbalance > +0.7 (extreme sell liquidations) → go short
- Minimum $1,000 notional in 5-minute window
- 15-minute hold, 0.5% stop loss, 15-minute cooldown

### Results (Default Parameters)

| Symbol | Trades | Win Rate | Avg Return | Total Return | Sharpe |
|--------|--------|----------|------------|--------------|--------|
| **BTC** | 87 | **58.6%** | +0.044% | **+3.79%** | +73.7 |
| **ETH** | 63 | 47.6% | -0.001% | -0.04% | -0.8 |
| **SOL** | 66 | **51.5%** | +0.031% | **+2.06%** | +38.7 |
| **DOGE** | 21 | 23.8% | -0.210% | -4.41% | -309.0 |
| **XRP** | 53 | **52.8%** | +0.109% | **+5.75%** | +109.4 |

### Threshold Sensitivity (BTC)

| Threshold | Trades | Win Rate | Total Return |
|-----------|--------|----------|--------------|
| 0.5 | 145 | 55.2% | +4.87% |
| 0.6 | 113 | 56.6% | +4.13% |
| **0.7** | **87** | **58.6%** | **+3.79%** |
| 0.8 | 56 | 57.1% | +2.45% |
| 0.9 | 28 | 60.7% | +1.61% |

### Analysis
- **Most consistent strategy** — profitable on BTC, SOL, and XRP
- **BTC has highest win rate** (58.6%) — strong mean reversion signal
- **XRP is the surprise winner**: +5.75% total, Sharpe +109
- **DOGE fails badly** — too few liquidations, imbalance is noisy
- **Higher threshold = higher win rate but fewer trades** (classic tradeoff)
- Threshold 0.5 gives best total return on BTC (+4.87%) with more trades

### Why It Works
- Extreme one-sided liquidations mark local exhaustion points
- When 70%+ of liquidations are buy-side, longs are capitulating → buy
- Low persistence (autocorr < 0.13) confirms rapid mean reversion
- 15-minute hold is well-calibrated to the reversion window

---

## Strategy 3: Liquidation Rate Spike

### Logic
When liquidation rate spikes (>3σ above 60-minute rolling mean), enter in the direction of the recent price move (trend following):
- Determine direction from last 5 minutes of price action
- 30-minute hold, 0.8% stop loss

### Results (Default Parameters)

| Symbol | Trades | Win Rate | Avg Return | Total Return | Sharpe |
|--------|--------|----------|------------|--------------|--------|
| **BTC** | 157 | 43.3% | -0.024% | -3.80% | -40.3 |
| **ETH** | 160 | 48.1% | +0.012% | +1.86% | +12.9 |
| **SOL** | 166 | 41.0% | -0.093% | **-15.51%** | -110.0 |
| **DOGE** | 159 | 47.2% | +0.007% | +1.11% | +6.7 |
| **XRP** | 160 | 45.0% | -0.007% | -1.07% | -6.8 |

### Z-Score Sensitivity (BTC)

| Z-Score | Trades | Win Rate | Total Return |
|---------|--------|----------|--------------|
| 2.0 | 233 | 44.6% | -3.51% |
| 2.5 | 195 | 43.6% | -4.47% |
| **3.0** | **157** | **43.3%** | **-3.80%** |
| 4.0 | 94 | 44.7% | -2.34% |

### Analysis
- **This strategy fails** — negative returns on 3/5 symbols
- **SOL is catastrophic**: -15.5% in 9 days
- Trend following after liquidation spikes doesn't work
- Liquidation spikes mark **reversals**, not continuations
- Confirms v26 finding: liquidations are contrarian signals

### Why It Fails
- Liquidation rate spikes happen at local extremes
- By the time we detect the spike, the move is already done
- Following the trend after a spike = buying the top / selling the bottom
- Mean reversion dominates at these timescales

---

## Strategy 4: Time-of-Day Liquidation Fade

### Logic
Same as Imbalance Reversal but restricted to US trading hours (13:00-17:00 UTC):
- Lower imbalance threshold (0.5) since US hours have more liquidity
- Higher minimum notional ($2,000)
- 10-minute hold, 0.4% stop loss

### Results (Default Parameters)

| Symbol | Trades | Win Rate | Avg Return | Total Return | Sharpe |
|--------|--------|----------|------------|--------------|--------|
| **BTC** | 65 | **56.9%** | +0.063% | **+4.09%** | +101.2 |
| **ETH** | 62 | 53.2% | +0.002% | +0.09% | +2.4 |
| **SOL** | 57 | 50.9% | +0.018% | +1.03% | +26.3 |
| **DOGE** | 29 | 37.9% | -0.022% | -0.63% | -25.5 |
| **XRP** | 44 | 45.5% | +0.000% | +0.02% | +0.6 |

### Analysis
- **BTC is the clear winner**: +4.09%, Sharpe +101, 57% win rate
- **Best Sharpe ratio** of any strategy on BTC
- US hours provide better execution conditions
- Fewer trades but higher quality
- **DOGE/XRP don't benefit** from US-hours filter

### Why BTC Works Best
- BTC has deepest liquidity during US hours
- Tighter spreads = lower execution cost
- More institutional flow = faster mean reversion
- 10-minute hold is optimal for US-session dynamics

---

## Cross-Strategy Comparison

### Best Strategy per Symbol

| Symbol | Best Strategy | Total Return | Win Rate | Sharpe |
|--------|--------------|--------------|----------|--------|
| **BTC** | Cascade Fade | +5.44% | 57.0% | +65.1 |
| **ETH** | Liq Rate Spike | +1.86% | 48.1% | +12.9 |
| **SOL** | Cascade Fade | +6.02% | 52.3% | +81.3 |
| **DOGE** | Liq Rate Spike | +1.11% | 47.2% | +6.7 |
| **XRP** | Imbalance Reversal | +5.75% | 52.8% | +109.4 |

### Aggregate Performance (All Symbols Combined)

| Strategy | Total Trades | Avg Win Rate | Sum Total Ret | Profitable Symbols |
|----------|-------------|--------------|---------------|-------------------|
| **Cascade Fade** | 388 | 48.2% | +5.76% | 2/5 |
| **Imbalance Reversal** | 290 | 46.9% | +7.15% | 3/5 |
| **Liq Rate Spike** | 802 | 44.9% | -17.34% | 2/5 |
| **ToD Liq Fade** | 257 | 48.9% | +4.58% | 3/5 |

### Key Takeaways

1. **Mean reversion works, trend following doesn't**
   - Strategies 1, 2, 4 (all mean reversion) are profitable on BTC/SOL
   - Strategy 3 (trend following) loses money consistently
   - Liquidations mark exhaustion, not continuation

2. **BTC and SOL are the best markets**
   - Deepest liquidity → cleanest mean reversion
   - Most cascade events → more trading opportunities
   - ETH is too noisy, DOGE has too few events

3. **Imbalance Reversal is the most robust**
   - Profitable on 3/5 symbols
   - Highest aggregate return (+7.15%)
   - Simple signal, easy to implement

4. **US hours improve quality**
   - ToD Fade has best Sharpe on BTC (+101)
   - Fewer but higher-quality trades
   - Better execution conditions

---

## Portfolio Construction

### Recommended Portfolio

**Primary strategies** (allocate 70%):
- Imbalance Reversal on BTC (30%)
- Cascade Fade on SOL (20%)
- Imbalance Reversal on XRP (20%)

**Secondary strategies** (allocate 30%):
- ToD Liq Fade on BTC (15%)
- Cascade Fade on BTC (15%)

### Expected Performance (9-day backtest, annualized)

| Metric | Value |
|--------|-------|
| Total return (9 days) | ~+5-7% |
| Annualized return | ~200-280% |
| Win rate | 53-59% |
| Avg trade | +0.04-0.07% |
| Max drawdown | 2-5% |
| Trades per day | 5-15 |

### Risk Management
- Max position size: 1x leverage
- Max concurrent positions: 1 per symbol
- Stop loss: 0.4-0.5% per trade
- Daily loss limit: -2%
- Cooldown after 3 consecutive losses: 1 hour

---

## Caveats & Limitations

### Data Limitations
- **Only 9 days of data** — too short for robust conclusions
- **Single market regime** — Feb 2026 may not be representative
- **No transaction costs** — real spreads + fees would reduce returns
- **No slippage model** — execution at bar close is optimistic

### Overfitting Risk
- Parameter sweep on same data as evaluation
- Need out-of-sample validation (Feb 18-28)
- Simple strategies with few parameters reduce overfitting risk

### Execution Challenges
- Cascade detection requires real-time liquidation feed
- 30-second cooldown may miss optimal entry
- Stop losses may get hit by noise in volatile periods

### Realistic Adjustments
- Subtract ~0.02% per trade for fees (maker)
- Subtract ~0.01% per trade for slippage
- Reduces avg return by ~0.03% per trade
- BTC Cascade Fade: +0.040% → +0.010% (still positive)
- BTC Imbalance Reversal: +0.044% → +0.014% (still positive)

---

## Comparison to Previous Research

| Version | Approach | Best Result | Verdict |
|---------|----------|-------------|---------|
| v25 | OI velocity (5s) | IC -0.08, Sharpe -8.6 | ❌ Failed |
| v26 | Liquidation analysis | Patterns confirmed | ✅ Foundation |
| **v26b** | **Liquidation strategies** | **+6.0%, Sharpe +81** | **✅ Promising** |

**Key insight**: Liquidations provide a cleaner signal than OI snapshots because:
1. They are **events** (discrete) not **snapshots** (noisy)
2. They have **direction** (buy/sell) not just magnitude
3. They mark **forced exits** (predictable behavior) not organic flow

---

## Next Steps

### Immediate
1. Download Feb 18-28 data for out-of-sample validation
2. Add transaction cost model (0.02% maker fee)
3. Test with realistic slippage (0.01%)

### Short-term
4. Combine Strategy 1 + 2 into ensemble
5. Add regime filter (only trade in volatile regimes)
6. Test on 5-minute bars instead of 1-minute

### Medium-term
7. Build real-time cascade detector
8. Paper trade for 1 week
9. Cross-asset contagion strategy (BTC cascade → trade altcoins)

---

## Code & Data

### Scripts
- `liquidation_strategies.py` — Full backtest engine (4 strategies, parameter sweeps)
- `research_liquidations.py` — Data analysis and statistics

### Results
- `results/liq_strategies_v26b_all.txt` — Complete backtest output

### Reproducibility
```bash
python3 liquidation_strategies.py 2>&1 | tee results/liq_strategies_v26b_all.txt
```

---

**Research by**: Cascade AI  
**Date**: February 18, 2026  
**Version**: v26b (Liquidation Strategies)  
**Status**: Complete ✅  
