# Cross-Exchange Futures Premium Divergence — Research Findings

**Date:** Feb 25, 2026  
**Data:** 52 hours of 1-min tick data (Feb 22-24), Binance + Bybit, 487 common symbols  
**Scripts:** `research_basis_premium_arb.py`, `research_extreme_basis_arb.py`

## The Idea

When the futures premium (markPrice - indexPrice) diverges between exchanges on the same symbol, short the overpriced futures on one exchange, long the cheaper futures on the other. Delta-neutral. Premiums must converge → profit.

Focus on **extreme** divergences (>100 bps) where convergence is most certain.

## Results — Unfiltered (All Symbols)

The raw numbers look great:

| Entry threshold | Trades (52h) | Gross bps | WR | $/day @8bps maker |
|---|---|---|---|---|
| ≥100 bps | 247 | 48.4 | 73% | $4,603 |
| ≥200 bps | 54 | 141.9 | 83% | $3,337 |
| ≥300 bps | 21 | 229.2 | 81% | $2,144 |
| ≥500 bps | 8 | 337.4 | 88% | $1,216 |

Bigger divergences → higher WR, bigger P&L per trade. As expected.

## Audit — 5 Showstoppers

### 1. Illiquid coins dominate

The extreme events happen almost exclusively on coins you **cannot trade at size**:

| Symbol | BBO depth | BA spread | % of ≥300bps events |
|---|---|---|---|
| POWERUSDT | $98 | 2.0 bps | **62%** |
| DGBUSDT | $71 | 12.8 bps | 6% |
| IDEXUSDT | $31 | 27.9 bps | — |
| SXPUSDT | $176 | 10.5 bps | 7% |

You can't place $10K on $31-98 of BBO depth.

### 2. Concentration risk

- ≥100 bps: top 5 symbols = **66%** of all events
- ≥200 bps: top 5 symbols = **82%** (POWERUSDT alone = 38%)
- ≥300 bps: top 5 symbols = **89%** (POWERUSDT alone = 62%)

### 3. markPrice ≠ tradeable price

markPrice is exchange-computed fair value (EMA of last price + basis). It's NOT the price you can trade at.

| Symbol | markPrice vs lastPrice gap |
|---|---|
| POWERUSDT | **36.6 bps** avg |
| DGBUSDT | 18.6 bps avg |
| All symbols | 5.9 bps avg, P99 = 38 bps |

A "500 bps mark price divergence" might only be a 400 bps tradeable divergence — and the slippage to trade it is 20+ bps.

### 4. Massive drawdown before convergence

Manual check of DGBUSDT (our "best" trade, 650 bps gross):
- Entry at -500 bps spread
- Spread **widened to -1170 bps** (670 bps adverse) before reverting
- Net P&L = +650 bps only because it held 120m and eventually converged

This is 2:1 drawdown-to-profit ratio. With leverage, this blows up the position.

### 5. Persistent basis offsets

AZTECUSDT has a **permanent** -108 bps offset (BN always 108 bps below BB). Seeing -150 bps is only 42 bps of actual divergence. These coins inflate apparent event counts.

## Results — Filtered to Tradeable Coins Only

Filtering to coins with **BBO ≥ $200 and BA spread ≤ 5 bps** (115 / 487 symbols):

| Entry threshold | Events (52h) | Trades (no overlap) |
|---|---|---|
| ≥50 bps | 2,878 min-bars | 143 |
| ≥100 bps | 273 min-bars | 30 |
| ≥200 bps | 4 min-bars | — |
| ≥500 bps | **0** | **0** |

**The extreme events almost completely vanish on liquid coins.**

### P&L on liquid coins

| Entry | Trades | Gross | WR | Net (fees+slip) | $/day |
|---|---|---|---|---|---|
| ≥30 bps | 395 | 4.5 bps | 63% | **-11.7 bps** | **-$2,127** |
| ≥50 bps | 143 | 8.2 bps | 59% | **-5.8 bps** | **-$386** |
| ≥100 bps | 30 | 21.3 bps | 57% | **+8.1 bps** | **+$113** |

Only ≥100 bps entries are net positive ($113/day), but with only 30 trades in 52 hours across 15 symbols, this is noise.

## Why Extreme Divergences Happen on Illiquid Coins

The divergence IS real, but it exists **because** the coin is illiquid:
- Thin orderbook → one trade moves markPrice significantly on one exchange
- Low volume → mark price EMA is sticky/stale on the other
- The "convergence" is just the stale mark price catching up

This is not arbitrageable because:
1. You can't get filled at the mark price (it's a theoretical value)
2. The orderbook is too thin to trade at meaningful size
3. Slippage on entry/exit eats the entire edge

## Verdict

**The extreme basis divergence strategy does NOT work.**

- On liquid coins (where you can actually trade): divergences are small (<50 bps) and don't converge fast enough to beat fees + slippage
- On illiquid coins (where divergences are huge): you can't trade them, the mark price isn't tradeable, and drawdowns are massive
- The strategy that looked like $4,600/day is actually $113/day on tradeable coins — likely pure noise

## Comparison to FR HOLD Strategy

| | Basis premium arb | FR HOLD (validated) |
|---|---|---|
| Edge source | Mark price convergence | FR autocorrelation |
| Validated? | ❌ Noise on liquid coins | ✅ 200 days, all months positive |
| $/day | ~$113 (if real) | $1,443 (Bybit 1h) |
| Mechanism | Theoretical (mark prices) | Mechanical (FR settlement) |
| Execution | 4 futures legs, cross-exchange | 1 futures leg, single exchange |

**The FR HOLD strategy remains the best validated approach.**
