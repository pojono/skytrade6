# Top 3 Actionable Trading Configs

**Based on all liquidation cascade research (v26–v42, 282 days, 4 symbols)**  
**Fees:** Bybit maker=0.02%, taker=0.055%

---

## Config 1: SAFEST — Conservative with Stop Loss

**Best for:** First live deployment, risk-averse, proving the edge is real

```
WHAT TO TRADE:    DOGE, SOL, ETH, XRP (all 4 simultaneously)
DIRECTION:        LONG ONLY (fade buy-side liquidation cascades)
                  → When longs get liquidated and price drops, BUY

CASCADE TRIGGER:  P95 liquidation event, min 2 events within 60 seconds
                  Price must have moved ≥10 bps during cascade

ENTRY:            Limit BUY at 0.15% BELOW current price (fade the drop)
                  → e.g., price is $100 → place limit buy at $99.85

TAKE PROFIT:      0.15% above entry (limit order, maker fee)
                  → e.g., entered at $99.85 → TP at $100.00

STOP LOSS:        0.50% below entry (market order, taker fee)
                  → e.g., entered at $99.85 → SL at $99.35

MAX HOLD:         60 minutes (exit at market if neither TP nor SL hit)

HOURS:            SKIP hours 08, 09, 13, 16 UTC
                  Trade all other hours

COOLDOWN:         5 minutes between trades per symbol
```

### Expected Performance (backtest, 282 days)

| Symbol | Trades | Win Rate | Total Return | Pos Months | Avg Hold |
|--------|--------|----------|-------------|------------|----------|
| DOGE | 287 | 88% | +8.7% | 4/5 | ~2 min |
| SOL | 350 | 89% | +12.4% | **5/5** | ~3 min |
| ETH | 409 | 88% | +11.7% | 4/5 | ~3 min |
| XRP | 268 | 89% | +9.4% | 4/5 | ~4 min |
| **TOTAL** | **1,314** | **88.5%** | **+42.2%** | **17/20** | — |

### Fee Breakdown
- **87-89% of trades** exit at TP → pay maker+maker = **0.04% round-trip**
- **11-12% of trades** exit at SL → pay maker+taker = **0.075% round-trip**
- **Average fee per trade: ~0.045%**

### Why This Config
- SL=0.50% protects against tail risk
- 88.5% WR means steady equity curve
- 17/20 positive months = very consistent
- Survived OOS walk-forward (v41): all 15 tests positive

---

## Config 2: BEST RETURN — Aggressive No Stop Loss

**Best for:** Maximizing returns once Config 1 is proven live

```
WHAT TO TRADE:    DOGE, SOL, ETH, XRP (all 4 simultaneously)
DIRECTION:        LONG ONLY (fade buy-side liquidation cascades)

CASCADE TRIGGER:  P95 liquidation event, min 2 events within 60 seconds
                  Price must have moved ≥10 bps during cascade

ENTRY:            Limit BUY at 0.15% BELOW current price

TAKE PROFIT:      0.12% above entry (limit order, maker fee)

STOP LOSS:        NONE

MAX HOLD:         60 minutes (exit at market if TP not hit)

HOURS:            SKIP hours 08, 09, 13, 16 UTC

COOLDOWN:         5 minutes between trades per symbol
```

### Expected Performance (backtest, 282 days)

| Symbol | Trades | Win Rate | Total Return | Pos Months | Sharpe |
|--------|--------|----------|-------------|------------|--------|
| SOL | 1,305 | 92.2% | **+25.8%** | **5/5** | +44 |
| DOGE | 926 | 94.4% | **+20.9%** | 4/5 | +42 |
| ETH | 1,438 | 91.2% | +10.7% | — | +13 |
| XRP | 910 | 91.3% | +9.2% | — | +13 |
| **TOTAL** | **4,579** | **92%** | **+66.7%** | — | — |

### Fee Breakdown
- **91-94% of trades** exit at TP → pay maker+maker = **0.04% round-trip**
- **0% of trades** hit SL (no SL)
- **6-9% of trades** timeout → pay maker+taker = **0.075% round-trip**
- **Average fee per trade: ~0.042%** (cheapest of all configs)

### Why This Config
- No SL = no expensive taker exits on losers
- 92% WR, avg fee 0.042% (vs 0.045% with SL)
- +66.7% combined = ~86% annualized
- **Risk:** No downside protection per trade. Max hold 60min limits damage, but flash crashes are unprotected. Use small position size.

---

## Config 3: HIGHEST QUALITY — DOGE US-Hours Only

**Best for:** Highest Sharpe, lowest drawdown, part-time trading

```
WHAT TO TRADE:    DOGEUSDT ONLY
DIRECTION:        LONG ONLY (fade buy-side liquidation cascades)

CASCADE TRIGGER:  P95 liquidation event, min 2 events within 60 seconds

ENTRY:            Limit BUY at 0.20% BELOW current price

TAKE PROFIT:      0.15% above entry (limit order, maker fee)

STOP LOSS:        0.50% below entry (market order, taker fee)

MAX HOLD:         30 minutes

HOURS:            13:00–18:00 UTC ONLY (US session)
                  = 8am–1pm ET (summer) / 9am–2pm ET (winter)

COOLDOWN:         5 minutes between trades
```

### Expected Performance (backtest, 282 days)

| Metric | Value |
|--------|-------|
| Trades | 247 |
| Win Rate | **87.0%** |
| Net/trade | +0.028% |
| Total Return | **+6.9%** |
| **Sharpe** | **+77** |
| **Max Drawdown** | **2.1%** |
| Positive Months | 4/5 |
| Avg Hold | 1.8 min |

### Why This Config
- **Sharpe +77** is exceptional — best risk-adjusted return of any config
- **Max DD 2.1%** — almost no drawdown
- Only trades during US hours when liquidation cascades are most frequent and revert best
- 0.20% offset = wider spread capture, fewer but higher-quality fills
- Fewer trades (247 vs 1,314) but each one is higher conviction
- **Perfect for:** running alongside other strategies, or if you can only monitor during US hours

---

## Quick Comparison

| | Config 1: Safe | Config 2: Aggressive | Config 3: Quality |
|---|---|---|---|
| **Symbols** | 4 | 4 | DOGE only |
| **Stop Loss** | 0.50% | None | 0.50% |
| **Trades** | 1,314 | 4,579 | 247 |
| **Win Rate** | 88.5% | 92% | 87% |
| **Total Return** | +42.2% | +66.7% | +6.9% |
| **Sharpe** | ~3-5 | ~13-44 | **+77** |
| **Max DD** | ~5-10% | ~6-11% | **2.1%** |
| **Pos Months** | 17/20 | — | 4/5 |
| **Avg Fee** | 0.045% | 0.042% | 0.045% |
| **Risk Level** | Low | Medium | Very Low |

---

## Implementation Checklist

### Before Going Live
- [ ] Set up Bybit API with limit order capability
- [ ] Implement real-time liquidation WebSocket feed (`wss://stream.bybit.com/v5/public/linear`)
- [ ] Implement cascade detection (P95 threshold, 2+ events within 60s)
- [ ] Implement bad-hours filter (skip 08, 09, 13, 16 UTC)
- [ ] Implement long-only filter (only trade when buy-side liquidations dominate)
- [ ] Implement displacement check (cascade moved price ≥10 bps)
- [ ] Paper trade for 2-4 weeks, measure actual fill rate vs backtest
- [ ] Compare live fill rate to backtest — if <70% of backtest fills, edge may not survive

### Position Sizing
- **Config 1/3:** Risk 1-2% of account per trade (SL=0.50% → position = 200-400% of risk capital)
- **Config 2:** Risk 0.5-1% of account per trade (no SL → be conservative)
- **Never risk more than 5% of account on concurrent open positions**

### Kill Switches
- **Daily loss limit:** Stop trading if daily loss exceeds 2%
- **Weekly loss limit:** Stop trading if weekly loss exceeds 5%
- **Fill rate monitor:** If fill rate drops below 50% for 3 consecutive days, pause and investigate
- **Win rate monitor:** If WR drops below 70% over 50+ trades, pause and investigate

---

## Source References

| Finding | Source File | Script |
|---------|-----------|--------|
| Cascade detection | `FINDINGS_v26d_cascade_mm.md` | `liq_cascade_mm.py` |
| Fee model | `FINDINGS_v26e_cascade_mm_fees.md` | `liq_cascade_mm_fees.py` |
| No-SL optimal | `FINDINGS_v26f_cascade_mm_rr.md` | `liq_cascade_mm_rr.py` |
| Microstructure | `FINDINGS_v26g_liq_microstructure.md` | `liq_microstructure.py` |
| Research filters | `FINDINGS_v26j_integrated_strategy.md` | `liq_integrated_strategy.py` |
| OOS validation | `FINDINGS_v41_cascade_mm_oos.md` | — |
| Bad hours | `FINDINGS_v33_temporal_patterns.md` | — |
| Long-only, contagion | `FINDINGS_v42_v42f_summary.md` | — |
| Full research index | `RESEARCH_INDEX_liquidation_strategy.md` | — |
