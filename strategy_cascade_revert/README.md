# Cascade Revert — Liquidation Cascade Mean-Reversion Strategy

**Strategy Name:** Cascade Revert  
**Asset Class:** Crypto Perpetual Futures (Bybit)  
**Type:** Event-driven mean-reversion with limit-order entry  
**Edge Source:** Structural microstructure dislocation from forced liquidations  
**Status:** OOS-validated on 2 independent time periods, ready for paper trading

---

## Table of Contents

1. [Strategy Overview](#1-strategy-overview)
2. [How It Works](#2-how-it-works)
3. [Research Journey — What We Tried](#3-research-journey--what-we-tried)
4. [What Worked](#4-what-worked)
5. [What Didn't Work](#5-what-didnt-work)
6. [Final Strategy Parameters](#6-final-strategy-parameters)
7. [Performance Summary](#7-performance-summary)
8. [Key Documents Index](#8-key-documents-index)
9. [Key Scripts Index](#9-key-scripts-index)
10. [Key Results Files](#10-key-results-files)
11. [Lessons Learned](#11-lessons-learned)
12. [Risk Factors & Caveats](#12-risk-factors--caveats)
13. [Next Steps](#13-next-steps)

---

## 1. Strategy Overview

**The idea in one sentence:** When large forced liquidations push price too far, place a limit order to catch the overshoot, then exit with a trailing limit when price snaps back.

### Why It Works

Leveraged traders get liquidated when price moves against them. Large liquidations create a temporary supply/demand imbalance — forced selling pushes price below fair value (or forced buying pushes it above). This dislocation mean-reverts within minutes. By placing a limit order at the dislocation level, we capture the bounce.

### Edge Decomposition

| Source | Contribution | Evidence |
|--------|-------------|----------|
| **Structural timing** (being there when dislocation happens) | ~72% of returns | Random direction still profitable |
| **Direction signal** (fading the liquidation side) | ~28% of returns | Statistically significant (Z=3.32) |

The edge is primarily **structural**, not directional. This makes it robust — even trading the wrong direction is profitable because the limit order captures the bounce regardless.

---

## 2. How It Works

### Signal Detection
1. Subscribe to Bybit WebSocket liquidation stream
2. Track rolling P95 of liquidation notional values
3. When a liquidation exceeds P95 threshold → potential cascade
4. Check if price displaced ≥10 bps from pre-cascade level
5. Determine direction: buy-side liquidations (longs stopped) → go long (fade the forced selling)

### Entry
- Place **limit order** at `current_price × (1 - 0.15%)` for longs (or `× (1 + 0.15%)` for shorts)
- Entry is always a **maker** order (0.02% fee)
- Max wait for fill: 60 minutes
- Cooldown: 5 minutes between trades per symbol

### Exit — Partial Exit Strategy (Variant A1, recommended)
- **70% of position:** Fixed take-profit limit at +12 bps from fill (maker, 0.02% fee)
- **30% of position:** Trailing limit at 3 bps from peak (maker, 0.02% fee)
- **Safety timeout:** 60 minutes → market close remainder (taker, 0.055% fee — very rare, <1% of trades)

### Fee Structure
| Exit Type | Entry Fee | Exit Fee | Round-Trip |
|-----------|-----------|----------|------------|
| TP or Trail (maker) | 0.02% | 0.02% | **0.04%** |
| Timeout (taker) | 0.02% | 0.055% | **0.075%** |

---

## 3. Research Journey — What We Tried

The strategy emerged from a broad research program spanning 42+ research versions. Here's the chronological path:

### Phase 1: Microstructure Exploration (v1–v9)
- Started with BTC tick-level microstructure signals (trade imbalance, VWAP deviation)
- Found contrarian signals (IC ~ -0.03) but too weak for standalone trading
- Explored regime detection (HMM, ML classifiers) — marginal improvement
- Grid bots tested — profitable but not exceptional
- **Verdict:** Pure microstructure signals too noisy for direct trading

### Phase 2: Alternative Data Sources (v10–v25)
- Multi-horizon prediction, range prediction, breakout prediction — all marginal
- Order book features (v23) — some signal but decays fast
- OI + funding rate signals (v24) — worked in-sample, failed OOS
- High-resolution OI velocity (v25) — failed completely
- **Verdict:** Most "alternative" signals are noise or already priced in

### Phase 3: Liquidation Discovery (v26)
- **v26:** First analysis of raw liquidation data — discovered cascades are frequent, clustered, and mean-reverting
- **v26b/c:** Market-order fade strategies — profitable on 1-min bars but couldn't overcome fees at tick level
- **v26d:** **Breakthrough** — limit-order entry into cascades. Sharpe 90-170, all symbols profitable
- **v26e:** Fee-aware testing confirmed viability with realistic 2 bps maker fees
- **v26f:** Risk-reward optimization → TP=0.15%, SL=0.50% is optimal
- **v26g:** Microstructure analysis — displacement ≥10 bps filter improves quality
- **v26i/j:** Integrated strategy with research filters (bad hours, long-only, displacement)
- **v26k:** Filter comparison — each filter adds ~1-2 bps, combined +3-5 bps

### Phase 4: Validation & Robustness (v27–v42)
- **v27:** Regime-conditioned liquidation analysis
- **v28:** Liquidation buildup patterns
- **v29–v32:** Combined tick features, target optimization, TP/SL tuning
- **v33:** Temporal patterns — hours 08, 09, 13, 16 UTC are worst for reversion
- **v34–v42:** Extensive feature engineering, tick-level triggers, regime switches
- **v41:** Walk-forward OOS validation — 15/15 configs positive, 12/12 rolling windows positive
- **v42:** New signal variants, Sortino optimization, Copenhagen analysis

### Phase 5: Exit Optimization (Trailing Stop + Partial Exits)
- **Trailing stop research:** Discovered all 46 timeout trades were profitable at some point (peak +8 bps median). 3 bps trail: +42% more return, 9x less drawdown than baseline
- **Partial exits research:** Tested 5 variants (A–E) across 200+ configs. Variant A (partial TP + trail) gives best Sharpe; Variant C (phase-shifted trail) gives highest return
- **Key insight:** Maker-only exits save ~40% of profit vs taker exits

### Phase 6: Out-of-Sample Validation (Feb 2026)
- Downloaded 11 days of completely unseen data (Feb 9-19, 2026)
- P95 thresholds frozen from 2025 in-sample data (no peeking)
- **Result:** Every config performs 1.1x BETTER OOS than in-sample
- 366 trades, 97.8% WR, +33.65% return (A1 config), Sharpe +118.5
- DOGE and XRP had **zero losing trades** in 11 days

---

## 4. What Worked

### Critical Discoveries (in order of importance)

| Discovery | Impact | Source |
|-----------|--------|--------|
| **Limit orders into cascades** (not market orders after) | Flipped strategy from -100% to +60% | v26d |
| **Trailing stop replaces fixed TP** | +42% more return, 9x less drawdown | Trailing stop research |
| **Partial exits (maker-only)** | Saves ~40% of fee drag, Sharpe 2x higher | Partial exits research |
| **P95 cascade size filter** | Focuses on meaningful events only | v26d, v41 |
| **Displacement ≥10 bps filter** | Removes noise cascades | v26g, v26j |
| **Wide SL (0.50%) beats tight SL** | Fewer stop-outs → fewer taker fees | v26e, v26f |
| **WS ticker data (~100ms)** | Much better than REST (~5s) — baseline improved from +100% to +160% | Trailing stop research |
| **5-minute cooldown between trades** | Prevents overtrading during extended cascades | v26d |

### Filters That Improved Performance

| Filter | Effect | Source |
|--------|--------|--------|
| Bad hours (08, 09, 13, 16 UTC) | Removes worst reversion hours | v33, v42H |
| Long-only (fade buy-side liqs) | Buy-side reverts 2-3 bps better | v42g |
| Displacement ≥10 bps | Only trade real dislocations | v26g |
| P95 cascade size | Focus on large forced liquidations | v26d |

---

## 5. What Didn't Work

### Strategies That Failed

| Approach | Why It Failed | Source |
|----------|--------------|--------|
| **Market-order fade** | Can't overcome fees (raw edge +1-4 bps, fees 4-7.5 bps) | v26b/c, v26i |
| **OI + funding rate signals** | Worked in-sample, failed OOS completely | v24, v24b |
| **High-res OI velocity** | No predictive power at any timescale | v25 |
| **Order book features** | Signal decays too fast to trade | v23 |
| **ML regime classifiers** | Marginal improvement, added complexity | v9, v20, v21 |
| **Grid bots** | Profitable but inferior to cascade strategy | v14–v17 |
| **Fixed stop-loss** | Hurts performance — cuts winners short | Trailing stop research |
| **Variant B (two fixed TPs)** | Reintroduces timeout problem (up to 520 timeouts) | Partial exits research |
| **Variant E (partial loss cutting)** | Hurts performance — trail already handles this | Partial exits research |
| **Trail activation threshold** | Delaying trail activation slightly hurts returns | Trailing stop research |
| **Tight SL (0.25%)** | Too many stop-outs → excessive taker fees | v26e |
| **Wide trail (8-15 bps)** | Too loose — gives back too much profit | Trailing stop research |

### Signals That Were Noise

| Signal | IC | Verdict | Source |
|--------|-----|---------|--------|
| OI velocity (any timeframe) | ~0 | No signal | v25 |
| Funding rate | ~0 OOS | Overfits | v24b |
| LS ratio | ~0 OOS | Overfits | v24c |
| Breakout prediction | ~0.01 | Too weak | v12 |
| Range prediction | ~0.01 | Too weak | v11 |

---

## 6. Final Strategy Parameters

### Recommended Config: A1 (Best Risk-Adjusted)

```
SIGNAL:
  Cascade detection:  P95 notional threshold (rolling from last ~1000 events)
  Clustering:         Events within 60s of each other
  Displacement:       ≥10 bps price move during cascade
  Direction:          Fade the dominant liquidation side
  Cooldown:           5 minutes between trades per symbol

ENTRY:
  Type:               Limit order (maker)
  Offset:             0.15% from current price
  Max wait:           60 minutes for fill

EXIT (Variant A1):
  70% of position:    Fixed TP at +12 bps (limit/maker)
  30% of position:    Trailing limit at 3 bps from peak (maker)
  Safety timeout:     60 minutes → market close (taker, <1% of trades)

SYMBOLS:
  Primary:            BTCUSDT, DOGEUSDT, SOLUSDT, ETHUSDT, XRPUSDT
  
FEES:
  Maker:              0.02%
  Taker:              0.055%
```

### Alternative Configs

| Config | Description | Best For |
|--------|-------------|----------|
| **A1:** 70% TP@12 + 30% trail@3 | Highest Sharpe (+83 IS, +119 OOS) | Risk-adjusted returns |
| **C1:** 50% TP@10, trail 5→3 | Highest return (+296% IS, +43% OOS/11d) | Maximum return |
| **A2:** 50% TP@8 + 50% trail@3 | Good balance, simpler | Balanced approach |
| **D1:** Progressive scale-out | Smoothest equity curve (Sharpe +68) | Ultra-conservative |

---

## 7. Performance Summary

### In-Sample (May–Aug 2025, 87 days, 5 symbols)

| Config | Trades | WR | Return | Sharpe | Max DD | Timeouts |
|--------|--------|-----|--------|--------|--------|----------|
| A1: 70% TP@12 + 30% trail@3 | 2,618 | 95.1% | +228.67% | +83.2 | 0.11% | 0 |
| C1: 50% TP@10, trail 5→3 | 2,618 | 92.3% | +296.07% | +58.2 | 0.19% | 0 |
| Trail 5bps pure (taker) | 2,618 | 92.3% | +276.41% | +38.4 | 0.19% | 1 |

### Out-of-Sample (Feb 9–19, 2026, 11 days, 5 symbols)

| Config | Trades | WR | Return | Sharpe | Max DD | Timeouts |
|--------|--------|-----|--------|--------|--------|----------|
| A1: 70% TP@12 + 30% trail@3 | 366 | 97.8% | +33.65% | +118.5 | 0.05% | 0 |
| C1: 50% TP@10, trail 5→3 | 366 | 94.5% | +42.61% | +81.4 | 0.08% | 0 |
| Trail 5bps pure (taker) | 366 | 94.5% | +41.81% | +60.2 | 0.08% | 0 |

### IS vs OOS Comparison (the money table)

| Config | IS avg/trade | OOS avg/trade | Ratio | IS WR | OOS WR |
|--------|-------------|---------------|-------|-------|--------|
| A1 | +0.087% | +0.092% | **1.1x** | 95.1% | 97.8% |
| C1 | +0.110% | +0.116% | **1.1x** | 92.3% | 94.5% |
| Trail 5bps | +0.105% | +0.114% | **1.1x** | 92.3% | 94.5% |

**Every config performs BETTER out-of-sample.** This is the opposite of overfitting.

### Per-Symbol OOS (A1 config, Feb 2026)

| Symbol | Trades | WR | Return | Sharpe |
|--------|--------|-----|--------|--------|
| BTC | 81 | 93.8% | +6.72% | +89.5 |
| DOGE | 47 | **100%** | +4.46% | +183.1 |
| SOL | 79 | 98.7% | +7.37% | +137.9 |
| ETH | 95 | 97.9% | +8.67% | +104.5 |
| XRP | 64 | **100%** | +6.43% | +165.1 |

### Trade Frequency

| Period | Avg/day | Min/day | Max/day | Avg/week | Avg/month |
|--------|---------|---------|---------|----------|-----------|
| IS (87d, combined) | 40.9 | 3 | 124 | 238 | ~900 |
| OOS (11d, combined) | 33.3 | 4 | 66 | 183 | ~1,000 |

### Trade Duration (Signal → Exit)

| Metric | IS | OOS |
|--------|-----|-----|
| Average | 4.6 min | 4.3 min |
| Median | 1.0 min | 1.2 min |
| P90 | 13.1 min | 12.3 min |
| Max | 60.8 min | 58.4 min |

Position held time (fill → exit) is **< 1 minute** in 100% of trades — the 3 bps trail triggers within the same 1-min bar as the fill.

### Worst-Case (OOS)

| Config | Worst Trade | Max Consec Losses |
|--------|------------|-------------------|
| A1 | -0.049% | 1 |
| C1 | -0.069% | 2 |
| Trail 5bps | -0.069% | 2 |

---

## 8. Key Documents Index

### Core Strategy Documents

| Document | Description |
|----------|-------------|
| [`FINDINGS_v26_liquidations.md`](../FINDINGS_v26_liquidations.md) | Initial liquidation data analysis — cascade frequency, clustering, imbalances |
| [`FINDINGS_v26d_cascade_mm.md`](../FINDINGS_v26d_cascade_mm.md) | **Breakthrough:** Limit-order cascade MM, Sharpe 90-170 |
| [`FINDINGS_v26e_cascade_mm_fees.md`](../FINDINGS_v26e_cascade_mm_fees.md) | Fee-aware validation — strategy survives real fees |
| [`FINDINGS_v26f_cascade_mm_rr.md`](../FINDINGS_v26f_cascade_mm_rr.md) | Risk-reward optimization → TP=0.15%, SL=0.50% |
| [`FINDINGS_v26g_liq_microstructure.md`](../FINDINGS_v26g_liq_microstructure.md) | Displacement filter, cascade microstructure |
| [`FINDINGS_v26j_integrated_strategy.md`](../FINDINGS_v26j_integrated_strategy.md) | All research filters combined — 85% positive months |
| [`FINDINGS_v26k_filter_comparison.md`](../FINDINGS_v26k_filter_comparison.md) | Individual filter contribution analysis |

### Exit Optimization

| Document | Description |
|----------|-------------|
| [`FINDINGS_trailing_stop.md`](../FINDINGS_trailing_stop.md) | Trailing stop research — 3bps trail: +42% more return, 9x less DD |
| [`FINDINGS_partial_exits.md`](../FINDINGS_partial_exits.md) | Partial exit variants A–E — maker-only exits, Sharpe 2x higher |
| [`TRAILING_STOP_EXPLAINED.md`](../TRAILING_STOP_EXPLAINED.md) | Detailed explanation of trailing stop mechanics |

### Validation & Confidence

| Document | Description |
|----------|-------------|
| [`FINDINGS_v41_cascade_mm_oos.md`](../FINDINGS_v41_cascade_mm_oos.md) | Walk-forward OOS: 15/15 positive, 12/12 rolling windows positive |
| [`CONFIDENCE_ASSESSMENT.md`](../CONFIDENCE_ASSESSMENT.md) | Honest self-audit — confidence 7/10, thin margins, proceed with caution |
| [`FINDINGS_latency_analysis.md`](../FINDINGS_latency_analysis.md) | Millisecond latency analysis — 250ms median WS delivery |
| [`VALIDATION_REPORT.md`](../VALIDATION_REPORT.md) | Cross-symbol validation on 7 new OOS symbols |

### Trading Guides

| Document | Description |
|----------|-------------|
| [`TRADING_GUIDE_v3.md`](../TRADING_GUIDE_v3.md) | Step-by-step execution guide (latest, includes trailing stops) |
| [`TRADING_GUIDE_v2.md`](../TRADING_GUIDE_v2.md) | Earlier execution guide |
| [`TRADING_GUIDE_cascade_mm.md`](../TRADING_GUIDE_cascade_mm.md) | Original cascade MM trading guide |
| [`TRADING_RULES.md`](../TRADING_RULES.md) | Risk management rules |

### Background Research (what led to the strategy)

| Document | Description |
|----------|-------------|
| [`FINDINGS.md`](../FINDINGS.md) | Original microstructure research (BTC tick data) |
| [`FINDINGS_v26b_liquidation_strategies.md`](../FINDINGS_v26b_liquidation_strategies.md) | Early liquidation strategies (market-order, failed) |
| [`FINDINGS_v26c_liquidation_strategies_full.md`](../FINDINGS_v26c_liquidation_strategies_full.md) | Full liquidation strategy sweep |
| [`FINDINGS_v33_temporal_patterns.md`](../FINDINGS_v33_temporal_patterns.md) | Hour-of-day patterns — bad hours discovery |
| [`FINDINGS_v39_exhaustive_features.md`](../FINDINGS_v39_exhaustive_features.md) | Exhaustive feature search |

### Dead Ends (documented for completeness)

| Document | Description |
|----------|-------------|
| [`FINDINGS_v24_oi_funding.md`](../FINDINGS_v24_oi_funding.md) | OI + funding signals — failed OOS |
| [`FINDINGS_v24b_oi_funding_oos.md`](../FINDINGS_v24b_oi_funding_oos.md) | OI funding OOS failure |
| [`FINDINGS_v25_hires_oi_funding.md`](../FINDINGS_v25_hires_oi_funding.md) | High-res OI velocity — no signal |
| [`FINDINGS_v23_orderbook.md`](../FINDINGS_v23_orderbook.md) | Order book features — signal decays too fast |
| [`FINDINGS_v14_grid_bot.md`](../FINDINGS_v14_grid_bot.md) | Grid bot research — profitable but inferior |

---

## 9. Key Scripts Index

### Core Strategy Scripts

| Script | Description |
|--------|-------------|
| [`liq_partial_exits_research.py`](../liq_partial_exits_research.py) | Partial exits research — all 5 variants, 200+ configs |
| [`liq_trailing_stop_research.py`](../liq_trailing_stop_research.py) | Trailing stop sweep — 3-15 bps, activation thresholds |
| [`liq_oos_feb2026.py`](../liq_oos_feb2026.py) | Feb 2026 OOS test — frozen P95, 10 configs |
| [`liq_trade_stats_and_equity.py`](../liq_trade_stats_and_equity.py) | Trade frequency, duration stats, equity curve PNGs |
| [`liq_integrated_strategy.py`](../liq_integrated_strategy.py) | Integrated strategy with all research filters |
| [`liq_cascade_mm_fees.py`](../liq_cascade_mm_fees.py) | Fee-aware cascade MM sweep |
| [`liq_stress_test.py`](../liq_stress_test.py) | 10-test stress test suite |
| [`liq_cross_symbol_validation.py`](../liq_cross_symbol_validation.py) | Cross-symbol OOS validation (7 new symbols) |
| [`liq_latency_analysis.py`](../liq_latency_analysis.py) | Millisecond latency analysis |

### Data Pipeline Scripts

| Script | Description |
|--------|-------------|
| [`download_dataminer.sh`](../download_dataminer.sh) | Download liquidation + ticker data from dataminer server |
| [`preprocess_ws_ticker_all.py`](../preprocess_ws_ticker_all.py) | Preprocess WS ticker JSONL → CSV for all symbols |
| [`download_binance_data.py`](../download_binance_data.py) | Download Binance kline data |

---

## 10. Key Results Files

### Text Results

| File | Description |
|------|-------------|
| [`results/liq_oos_feb2026.txt`](../results/liq_oos_feb2026.txt) | Feb 2026 OOS test output |
| [`results/liq_partial_exits_research.txt`](../results/liq_partial_exits_research.txt) | Partial exits full sweep results |
| [`results/liq_trailing_stop_research.txt`](../results/liq_trailing_stop_research.txt) | Trailing stop sweep results |
| [`results/liq_trade_stats_and_equity.txt`](../results/liq_trade_stats_and_equity.txt) | Trade frequency/duration stats |
| [`results/liq_stress_test.txt`](../results/liq_stress_test.txt) | Stress test results |
| [`results/liq_cross_symbol_validation.txt`](../results/liq_cross_symbol_validation.txt) | Cross-symbol validation |

### Equity Curve PNGs

| File | Description |
|------|-------------|
| [`results/equity_combined_a1.png`](../results/equity_combined_a1.png) | Combined 5-symbol equity — A1 config (IS + OOS) |
| [`results/equity_combined_c1.png`](../results/equity_combined_c1.png) | Combined 5-symbol equity — C1 config (IS + OOS) |
| [`results/equity_oos_a1.png`](../results/equity_oos_a1.png) | OOS-only zoomed — A1 config |
| [`results/equity_oos_c1.png`](../results/equity_oos_c1.png) | OOS-only zoomed — C1 config |
| [`results/equity_btc_a1.png`](../results/equity_btc_a1.png) | BTC equity — A1 |
| [`results/equity_eth_a1.png`](../results/equity_eth_a1.png) | ETH equity — A1 |
| [`results/equity_sol_a1.png`](../results/equity_sol_a1.png) | SOL equity — A1 |
| [`results/equity_doge_a1.png`](../results/equity_doge_a1.png) | DOGE equity — A1 |
| [`results/equity_xrp_a1.png`](../results/equity_xrp_a1.png) | XRP equity — A1 |

---

## 11. Lessons Learned

### On Strategy Development

1. **Limit orders change everything.** Market-order strategies lost -65% to -141%. The same signal with limit orders earned +20% to +62%. The entry mechanism matters more than the signal.

2. **Fees are the #1 enemy.** Gross edge is ~8-10 bps per trade. Maker round-trip is 4 bps, taker is 7.5 bps. The difference between maker and taker exits is ~40% of profit. Optimize for maker fills.

3. **Trailing stops beat fixed TP.** Every timeout trade was profitable at some point. A trailing stop captures whatever the bounce gives instead of waiting for a fixed target that may never be reached.

4. **Wide stop-loss beats tight.** Counter-intuitive: 0.50% SL beats 0.25% SL because fewer stop-outs mean fewer taker fees and more trades that eventually revert to TP.

5. **Simple filters compound.** Each filter (bad hours, displacement, long-only) adds only 1-2 bps individually. Combined, they add 3-5 bps and turn 50% positive months into 85%.

### On Data & Methodology

6. **WS data (~100ms) is dramatically better than REST (~5s).** Baseline improved from +100% to +160% just from better data resolution. The REST-based backtest was conservative.

7. **OOS validation is non-negotiable.** Many signals (OI, funding, LS ratio) looked great in-sample and failed completely OOS. The cascade strategy is the only one that survived multiple OOS tests.

8. **Frozen parameters are the real test.** The Feb 2026 OOS test used P95 thresholds computed from 2025 data with zero peeking. The strategy performed 1.1x better OOS — the opposite of overfitting.

### On What Doesn't Work

9. **Most "alternative data" signals are noise.** OI velocity, funding rate, LS ratio, order book depth — all failed OOS. The only reliable signal is the liquidation event itself.

10. **ML adds complexity without proportional benefit.** Regime classifiers, HMM models, gradient boosting — marginal improvements that don't justify the complexity. Simple rules outperform.

11. **Don't cut losers early.** Variant E (partial loss cutting) hurt performance. The trailing stop already handles risk management. Adding more exit logic just adds fees.

---

## 12. Risk Factors & Caveats

### Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **3 bps trail ≈ bid-ask spread** | High | Use 5 bps trail for live; monitor actual slippage |
| **Fees consume 51% of gross edge** | High | Maker-only exits; pursue VIP fee tier |
| **No bear market data** | Medium | Strategy is structural (not directional) — should be regime-agnostic |
| **183-day data gap** (Aug 2025 – Feb 2026) | Medium | OOS Feb 2026 validates post-gap |
| **Fill rate unknown** | High | Paper trade first; kill if fill rate < 50% |
| **Execution latency** | Medium | 250ms median WS delivery; strategy tolerant to 1-2s delay |
| **Exchange risk** | Low | Bybit-specific; may need adaptation for other exchanges |

### What Could Kill the Strategy

1. **Bybit changes liquidation mechanism** — if liquidations become less visible or less impactful
2. **Competition** — if many bots start fading cascades, the edge will compress
3. **Spread widening** — if spreads widen beyond 3-5 bps during cascades, the trail becomes ineffective
4. **Fee increase** — any increase in maker fees directly erodes the thin margin

### Realistic Expectations for Live Trading

| Metric | Backtest | Realistic Live Estimate |
|--------|----------|------------------------|
| Return | +228% (IS, 87d) | +90-160% (50-70% of backtest) |
| Win rate | 95% | 85-92% |
| Max drawdown | 0.11% | 1-3% |
| Avg PnL/trade | +0.087% | +0.04-0.06% |
| Fill rate | 100% (assumed) | 60-90% (unknown) |

---

## 13. Next Steps

### Phase 1: Paper Trading (2-4 weeks)
1. Implement WebSocket listener for liquidation + ticker streams
2. Run paper orders on Bybit testnet
3. **Measure:** fill rate, actual latency, slippage, WR, avg PnL
4. **Kill criteria:** fill rate < 50% OR WR < 80% OR avg PnL < +0.02%

### Phase 2: Minimum Live (2-4 weeks)
1. Start with 1 symbol (XRP — most robust)
2. Minimum position size ($100-500 per trade)
3. Compare live metrics to backtest
4. **Scale up only if:** fill rate > 60%, WR > 85%, avg PnL > +0.02%

### Phase 3: Scale
1. Add symbols one at a time
2. Target VIP fee tier to improve margins
3. Consider mid-tier coins (ADA, LTC, NEAR) — higher per-trade quality, less competition

---

*Strategy: Cascade Revert*  
*Last updated: Feb 20, 2026*  
*Research program: skytrade6 (42+ versions, 5 months)*
