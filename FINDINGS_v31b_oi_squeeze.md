# FINDINGS v31b/c: OI Divergence as Squeeze Continuation Signal

## Hypothesis

When price moves directionally AND OI drops simultaneously, this signals forced liquidation (not voluntary positioning). The positive feedback loop creates predictable continuation:

```
price up → shorts liquidated → forced market buys → price up → more shorts liquidated
```

## Method

### v31b: Per-Second OI Divergence
- Compute `price_momentum` and `oi_velocity` at multiple lookback windows (30s, 60s, 120s, 300s)
- Signal: `|price_mom| > threshold AND oi_vel < -threshold`
- Compare: OI-divergence vs pure momentum vs anti-signal (OI rising)
- Measure forward directional return at 30s, 60s, 120s, 300s, 600s

### v31c: Episode-Based Detection
- Detect discrete squeeze **onset** events (non-overlapping, with cooldown)
- Multiple strength levels: moderate (P75/P50), strong (P75/P75), extreme (P90/P75)
- Measure forward continuation from episode onset
- Analyze predictors: what at onset predicts stronger continuation?

## Cross-Validation Matrix

| Test | Symbol | Dates | Days |
|------|--------|-------|------|
| 1 | BTCUSDT | May 12-18 2025 | 7 |
| 2 | ETHUSDT | May 12-18 2025 | 7 |
| 3 | BTCUSDT | Aug 4-10 2025 | 7 |

## Results

### v31b: Per-Second Signal (BTC May, 7 days)

**Overall: very weak at per-second resolution.** Best edge is +0.5% continuation over momentum baseline.

BUT the **asymmetry analysis** revealed real signal:

| Direction | Cont% at 300s | Mean Dir Ret | Avg Liqs in Window |
|-----------|--------------|-------------|-------------------|
| **Long squeeze** (shorts liq'd) | **55.3%** | **+2.22 bps** | 26.5 |
| Short squeeze (longs liq'd) | 48.9% | -0.32 bps | 10.0 |

**Long squeezes (shorts being liquidated) have real continuation signal. Short squeezes do not.**

### v31c: Episode-Based Results

#### Long Squeeze Episodes (BTC May, 7d)

| Strength | Fwd 30s | Fwd 60s | Fwd 120s | Fwd 300s | Fwd 600s | N |
|----------|---------|---------|----------|----------|----------|---|
| Moderate | 53.1% +0.55bps | 53.0% +0.73bps | 52.9% +0.73bps | **54.1% +1.14bps** | 53.3% +0.87bps | 1114 |
| Strong | 53.3% +0.48bps | 52.9% +0.74bps | 53.2% +0.83bps | 51.3% +0.69bps | 52.7% +0.59bps | 818 |
| Extreme | 53.4% +0.53bps | 53.2% +0.79bps | 53.0% +0.80bps | **53.2% +1.21bps** | 53.8% +0.78bps | 526 |

#### Short Squeeze Episodes — No Signal

Short squeeze (longs liquidated) shows ~48-50% continuation across all configs. No tradeable signal.

### Critical Finding: Liquidation Count is the Real Predictor

**Continuation by liquidation count in lookback window (all episodes combined):**

#### BTC May (7d)
| Liq Count | Cont% | Dir Ret | N |
|-----------|-------|---------|---|
| 0 liqs | 50.9% | +0.51 bps | 5114 |
| 1-5 liqs | **45.1%** | **-1.14 bps** | 1648 |
| 6-20 liqs | 51.8% | +0.32 bps | 785 |
| **20+ liqs** | **52.6%** | **+1.74 bps** | 589 |

#### BTC Aug (7d, out-of-sample)
| Liq Count | Cont% | Dir Ret | N |
|-----------|-------|---------|---|
| 0 liqs | 50.2% | -0.24 bps | 4157 |
| 1-5 liqs | 47.5% | -0.40 bps | 773 |
| 6-20 liqs | 43.8% | -1.09 bps | 242 |
| **20+ liqs** | **62.8%** | **+5.36 bps** | 164 |

#### ETH May (7d) — REVERSAL!
| Liq Count | Cont% | Dir Ret | N |
|-----------|-------|---------|---|
| 0 liqs | 49.2% | +0.96 bps | 4877 |
| 1-5 liqs | 48.1% | +0.17 bps | 2196 |
| 6-20 liqs | **44.6%** | **-4.03 bps** | 1130 |
| **20+ liqs** | **45.7%** | **-2.98 bps** | 884 |

### Predictor Correlations (with 300s forward directional return)

| Predictor | BTC May | BTC Aug | ETH May |
|-----------|---------|---------|---------|
| `price_mom` | +0.083*** | +0.069*** | +0.054*** |
| `oi_vel` | +0.017 | +0.019 | +0.067*** |
| `liq_count_lb` | -0.013 | **+0.050***| **-0.077***|
| `buy_ratio_lb` | +0.042*** | +0.023 | +0.017 |
| `liq_buy_ratio` | -0.043*** | +0.020 | -0.046*** |

## Key Findings

### 1. OI Divergence Alone is Weak
Per-second OI velocity divergence from price produces only +0.5% edge over momentum baseline. The signal is real but too noisy at tick resolution.

### 2. Long Squeeze Has Signal, Short Squeeze Does Not
Shorts being liquidated (price up + OI down) → 55% continuation at 300s, +2.2bps.
Longs being liquidated (price down + OI down) → 49% continuation, no signal.
This asymmetry is consistent with crypto market structure (more leveraged longs, more crowded shorts at key levels).

### 3. Liquidation Count > OI Velocity as Predictor
The number of actual liquidation events in the lookback window is a stronger predictor than OI velocity. The mechanism: OI can drop for many reasons (voluntary closing), but liquidation count directly measures forced closing.

### 4. The "1-5 Liqs" Trap
A small number of liquidations (1-5) during a move is actually a **negative** signal (-1.14 bps in BTC, -0.40 in Aug). This may represent isolated stop-outs that don't cascade — the move is running out of fuel.

### 5. 20+ Liqs is Strong in BTC, Reverses in ETH
- **BTC**: 20+ liqs → 52.6-62.8% continuation, +1.7 to +5.4 bps — **consistent across time periods**
- **ETH**: 20+ liqs → 45.7% continuation, -3.0 bps — **mean reversion**

This is the most important finding. The hypothesis works for BTC but **reverses** for ETH. Possible explanations:
- ETH has more retail leverage → liquidation cascades exhaust faster
- ETH market makers are more aggressive at fading liquidation moves
- ETH OI is more fragmented → cascades don't self-feed as strongly

### 6. Price Momentum is the Most Stable Predictor
`price_mom` correlation with forward continuation is +0.05 to +0.08 (***) across all tests. This is the only predictor that is consistently positive across all assets and time periods.

## Trading Implications

### What Works
- **BTC long squeeze with 20+ liquidations**: 53-63% continuation probability, +1.7 to +5.4 bps per episode
- **Best lookback**: 120s (balance of signal/noise)
- **Best forward horizon**: 300s (5 min)

### What Doesn't Work
- OI divergence alone (too noisy)
- Short squeeze in any asset
- Any squeeze signal in ETH (reverses)
- Small liquidation counts (1-5 liqs = negative signal)

### Caveats
- Sample sizes for 20+ liq episodes are small (164-589 per 7-day period)
- BTC Aug result (+5.36 bps, 62.8%) may be period-specific
- Need to test execution feasibility (can you actually enter during a cascade?)
- Slippage during high-liq periods would eat into the edge

## Next Steps
1. Test on more BTC date ranges to confirm 20+ liq signal stability
2. Test on SOL (higher leverage, more liquidations)
3. Build a proper backtest with realistic entry/exit and slippage
4. Investigate WHY ETH reverses — is it market maker behavior?
5. Combine with v31 regime classification (squeeze regime + high liq count)
