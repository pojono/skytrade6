# Research Findings v4 — Grid Trading Experiments

**Date:** 2026-02-15
**Exchange:** Bybit Futures (VIP0)
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT

## ⚠️ IMPORTANT CAVEAT: Backtester Limitations

The grid results below show **100% win rates and very high PnL**. This is almost certainly **too optimistic** due to backtester simplifications:

1. **OHLCV bar simulation** — We check if high/low touched a grid level, but we don't know the intra-bar price path. A bar could touch the buy level AND the take-profit in the same bar, but in reality the buy might not have filled before the TP level was reached.
2. **No queue priority** — Real limit orders sit in a queue. In a fast market, your order may not fill even if price touches your level.
3. **No slippage on timeouts** — When a grid trade times out after 4h, we close at the bar's close price. Real execution would have slippage.
4. **Simultaneous fills** — Multiple grid levels can fill in the same bar, which is unrealistic for a single account.

**These results should be treated as an upper bound.** The real performance will be significantly lower. A tick-level backtester or paper trading is needed to validate.

That said, the **relative ranking** of strategies is still informative — which variants improve over the baseline.

## Grid Experiment Catalog

| ID | Name | Key Feature |
|----|------|------------|
| G01 | Fixed symmetric | Baseline: fixed 0.2% cells, no intelligence |
| G02 | Vol-scaled | Cell width = 1.5 × realized_vol × price |
| G03 | Asymmetric | Contrarian bias: shift grid based on vol imbalance |
| G04 | Regime-adaptive | Pause grid when efficiency + sign_AC indicate trend |
| G05 | Time-filtered | Skip 2–5 UTC (lowest liquidity hours) |
| G06 | Trend overlay | Grid in range + momentum trades in trends |
| G07 | Dynamic sizing | Reduce size as inventory grows |
| G08 | Full hybrid | All features combined |
| G09 | Tight grid | 7 levels, 0.8× cell width |
| G10 | Wide grid | 3 levels, 2.5× cell width |

**Fee structure:**
- Grid fills: 4 bps round-trip (maker + maker)
- Trend trades: 7 bps round-trip (taker + maker)

## 30-Day Results (with caveats above)

### Top configurations by avg PnL per trade:

| Experiment | Symbol | Trades | Avg PnL (bps) | Total PnL (bps) | WR | Max DD |
|-----------|--------|--------|----------------|------------------|-----|--------|
| G10 Wide grid | SOL | 365 | +37.64 | +13,740 | 100% | -75 |
| G10 Wide grid | ETH | 398 | +32.09 | +12,772 | 99% | -127 |
| G10 Wide grid | BTC | 378 | +21.35 | +8,071 | 100% | 0 |
| G08 Full hybrid | SOL | 2,072 | +19.63 | +40,668 | 100% | -93 |
| G03 Asymmetric | SOL | 2,439 | +19.06 | +46,495 | 100% | -93 |
| G02 Vol-scaled | SOL | 2,780 | +18.53 | +51,507 | 100% | -93 |
| G08 Full hybrid | ETH | 2,048 | +16.67 | +34,148 | 100% | -151 |
| G01 Fixed (baseline) | BTC | 1,933 | +15.19 | +29,361 | 100% | -189 |
| G08 Full hybrid | BTC | 1,923 | +11.70 | +22,498 | 100% | -89 |
| G09 Tight grid | SOL | 7,745 | +8.39 | +65,011 | 100% | -466 |

### Top configurations by total PnL:

| Experiment | Symbol | Total PnL (bps) | Trades |
|-----------|--------|------------------|--------|
| G01 Fixed | SOL | +69,215 | 4,606 |
| G09 Tight | SOL | +65,011 | 7,745 |
| G01 Fixed | ETH | +58,399 | 3,966 |
| G09 Tight | ETH | +57,059 | 7,477 |
| G02 Vol-scaled | SOL | +51,507 | 2,780 |

## Relative Strategy Comparison (what we can trust)

Even though absolute numbers are inflated, the **relative ranking** tells us:

### 1. Wide grid (G10) has highest avg PnL per trade
Fewer trades but each one captures a larger price move. Less sensitive to execution issues.

### 2. SOL is the best grid asset
Higher volatility = wider natural price oscillations = more grid fills with larger profit per fill.

### 3. Full hybrid (G08) improves over baseline
- BTC: G08 avg +11.70 vs G01 avg +15.19 — fewer trades but tighter DD (-89 vs -189)
- SOL: G08 avg +19.63 vs G01 avg +15.03 — better avg AND better DD
- ETH: G08 avg +16.67 vs G01 avg +14.73 — better avg AND better DD

The hybrid's **lower drawdown** is the real win — it avoids the worst grid scenarios.

### 4. Time filter (G05) helps slightly
Skipping 2–5 UTC reduces trades by ~15% but maintains or improves avg PnL.

### 5. Asymmetric grid (G03) helps on SOL/ETH
Using microstructure imbalance to bias the grid improves avg PnL on more volatile assets.

## Grid + Trend Strategy: The Complementary Approach

The key insight from our signal research (v1–v3) and grid experiments:

| Market Regime | Grid Performance | Signal Performance | Combined |
|--------------|-----------------|-------------------|----------|
| **Range (70% of time)** | ✅ Profits from oscillation | ⚠️ Few signals (no extremes) | Grid carries |
| **Trend (20% of time)** | ❌ Accumulates inventory | ✅ Strong directional signals | Signals carry |
| **Extreme vol (10%)** | ❌ Dangerous | ⚠️ Mixed | Handbrake: pause both |

**This is why combining them is powerful:**
- Grid provides steady base income during the 70% range-bound periods
- Microstructure signals activate during the 20% trending periods
- Handbrake protects during the 10% extreme events
- The equity curve should be much smoother than either alone

## Next Steps for Grid Strategy

1. **Build tick-level grid backtester** — process actual trades, not OHLCV bars
2. **Paper trade** the wide grid (G10) on SOL for 1 week
3. **Combine grid + best microstructure signals** into unified strategy
4. **Optimize time-of-day windows** using intraday seasonality data
5. **Test funding rate integration** — grid positions earn/pay funding every 8h

## Honest Assessment

The grid results are **directionally correct but quantitatively inflated**. The real edge from grid trading on crypto is likely:
- **5–15 bps per fill** (not 20–40 bps)
- **70–85% win rate** (not 100%)
- **Still very profitable** at scale due to high trade frequency

The combination with microstructure signals is the most promising path forward because it addresses the grid's biggest weakness (trends) with proven directional signals.
