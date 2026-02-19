# Top 3 Actionable Trading Configs

**Based on all liquidation cascade research (v26–v42, 282 days, 4 symbols)**  
**Fees:** Bybit maker=0.02%, taker=0.055%

---

> **⚠️ UPDATE (v26k): Displacement ≥10 bps is the #1 filter.**  
> Filter comparison (see `FINDINGS_v26k_filter_comparison.md`) showed that the single  
> displacement filter outperforms ALL filters combined. Over-filtering kills trade count.  
> **Revised numbers with displacement filter applied to all configs below.**  
> Old numbers without displacement are in parentheses for reference.

---

## Config 1: SAFEST — Conservative with Stop Loss

**Best for:** First live deployment, risk-averse, proving the edge is real

```
WHAT TO TRADE:    DOGE, SOL, ETH, XRP (all 4 simultaneously)
DIRECTION:        BOTH (fade cascade direction — buy or sell)

CASCADE TRIGGER:  P95 liquidation event, min 2 events within 60 seconds
FILTER:           Cascade displacement ≥10 bps (THE key filter)

ENTRY:            Limit order at 0.15% offset from market (fade the cascade)
                  → Buy-side cascade: limit BUY below market
                  → Sell-side cascade: limit SELL above market

TAKE PROFIT:      0.15% from entry (limit order, maker fee)

STOP LOSS:        0.50% from entry (market order, taker fee)

MAX HOLD:         60 minutes (exit at market if neither TP nor SL hit)

HOURS:            ALL hours (no hour filter needed)

COOLDOWN:         5 minutes between trades per symbol
```

### Expected Performance (backtest, 282 days)

| Symbol | Trades | Win Rate | Total Return | Sharpe | Max DD | Pos Months |
|--------|--------|----------|-------------|--------|--------|------------|
| DOGE | 506 | 88.5% | **+17.3%** | +4.1 | 2.2% | 5/10 |
| SOL | 633 | 87.7% | **+18.8%** | +3.9 | 2.5% | 5/10 |
| ETH | 808 | 87.7% | **+23.6%** | +4.3 | 4.7% | 5/10 |
| XRP | 447 | 87.2% | **+12.4%** | +3.0 | 5.2% | 4/10 |
| **TOTAL** | **2,394** | **87.8%** | **+72.1%** | **+3.8** | — | **19/40** |

*(Old multi-filter version: +42.2% combined, 1,314 trades — displacement alone is +72% better)*

### Fee Breakdown
- **88% of trades** exit at TP → pay maker+maker = **0.04% round-trip**
- **11% of trades** exit at SL → pay maker+taker = **0.075% round-trip**
- **1% of trades** timeout → pay maker+taker = **0.075% round-trip**

### Why This Config
- SL=0.50% protects against tail risk
- 87.8% WR means steady equity curve
- 19/40 positive months across all symbols
- Displacement filter alone outperforms all-filters-combined by 91%
- Survived OOS walk-forward (v41): all 15 tests positive

---

## Config 2: BEST RETURN — Aggressive No Stop Loss

**Best for:** Maximizing returns once Config 1 is proven live

```
WHAT TO TRADE:    DOGE, SOL, ETH, XRP (all 4 simultaneously)
DIRECTION:        BOTH (fade cascade direction — buy or sell)

CASCADE TRIGGER:  P95 liquidation event, min 2 events within 60 seconds
FILTER:           Cascade displacement ≥10 bps (THE key filter)

ENTRY:            Limit order at 0.15% offset from market (fade the cascade)

TAKE PROFIT:      0.12% from entry (limit order, maker fee)

STOP LOSS:        NONE

MAX HOLD:         60 minutes (exit at market if TP not hit)

HOURS:            ALL hours (no hour filter needed)

COOLDOWN:         5 minutes between trades per symbol
```

### Expected Performance (backtest, 282 days)

| Symbol | Trades | Win Rate | Total Return | Sharpe | Max DD | Pos Months |
|--------|--------|----------|-------------|--------|--------|------------|
| DOGE | 506 | 96.8% | **+19.1%** | +2.9 | 5.4% | 5/10 |
| SOL | 633 | 94.8% | **+24.5%** | +5.2 | 5.8% | 5/10 |
| ETH | 808 | 95.4% | **+28.3%** | +4.5 | 3.6% | 5/10 |
| XRP | 447 | 95.3% | **+19.5%** | +5.3 | 2.5% | 5/10 |
| **TOTAL** | **2,394** | **95.6%** | **+91.4%** | **+4.5** | — | **20/40** |

*(Old no-filter version: +66.5% combined — displacement adds +37%)*

### Fee Breakdown
- **95-97% of trades** exit at TP → pay maker+maker = **0.04% round-trip**
- **0% of trades** hit SL (no SL)
- **3-5% of trades** timeout → pay maker+taker = **0.075% round-trip**
- **Average fee per trade: ~0.041%** (cheapest of all configs)

### Why This Config
- No SL = no expensive taker exits on losers
- 95.6% WR with displacement filter (was 92% without)
- **+91.4% combined = ~118% annualized** — best total return
- 20/40 positive months (all symbols have 5/10)
- **Risk:** No downside protection per trade. Max hold 60min limits damage, but flash crashes are unprotected. Use small position size.

---

## Config 3: HIGHEST QUALITY — Best Risk-Adjusted

**Best for:** Best Sharpe, lowest drawdown, highest per-trade quality

```
WHAT TO TRADE:    DOGE, SOL, ETH, XRP (all 4 simultaneously)
DIRECTION:        BOTH (fade cascade direction — buy or sell)

CASCADE TRIGGER:  P95 liquidation event, min 2 events within 60 seconds
FILTER:           Cascade displacement ≥10 bps (THE key filter)

ENTRY:            Limit order at 0.20% offset from market (wider = higher quality)

TAKE PROFIT:      0.15% from entry (limit order, maker fee)

STOP LOSS:        0.50% from entry (market order, taker fee)

MAX HOLD:         30 minutes

HOURS:            ALL hours (no hour filter needed)

COOLDOWN:         5 minutes between trades per symbol
```

### Expected Performance (backtest, 282 days)

| Symbol | Trades | Win Rate | Total Return | Sharpe | Max DD | Pos Months |
|--------|--------|----------|-------------|--------|--------|------------|
| DOGE | 460 | 88.5% | **+18.2%** | +4.8 | 2.3% | 4/10 |
| SOL | 540 | 85.7% | **+15.5%** | +3.6 | 5.2% | 4/10 |
| ETH | 694 | 86.0% | **+20.0%** | +4.1 | 3.6% | 5/10 |
| XRP | 390 | 85.6% | **+11.3%** | +3.1 | 2.2% | 5/10 |
| **TOTAL** | **2,084** | **86.5%** | **+65.0%** | **+3.9** | — | **18/40** |

*(Old no-filter version: +35.5% combined — displacement adds +83%)*

### Why This Config
- 0.20% offset = wider spread capture, fewer but higher-quality fills
- Best avg Sharpe (+3.9) across all symbols
- Max DD consistently low (2.2-5.2% per symbol)
- 18/40 positive months across all symbols
- Good balance of trade count and quality

---

## Quick Comparison (all with displacement ≥10 bps filter)

| | Config 1: Safe | Config 2: Aggressive | Config 3: Quality |
|---|---|---|---|
| **Symbols** | 4 | 4 | 4 |
| **Offset** | 0.15% | 0.15% | 0.20% |
| **TP** | 0.15% | 0.12% | 0.15% |
| **Stop Loss** | 0.50% | None | 0.50% |
| **Hold** | 60min | 60min | 30min |
| **Trades** | 2,394 | 2,394 | 2,084 |
| **Win Rate** | 87.8% | **95.6%** | 86.5% |
| **Total Return** | +72.1% | **+91.4%** | +65.0% |
| **Avg Sharpe** | +3.8 | **+4.5** | +3.9 |
| **Pos Months** | 19/40 | **20/40** | 18/40 |
| **Risk Level** | Low | Medium | Low |

---

## Implementation Checklist

### Before Going Live
- [ ] Set up Bybit API with limit order capability
- [ ] Implement real-time liquidation WebSocket feed (`wss://stream.bybit.com/v5/public/linear`)
- [ ] Implement cascade detection (P95 threshold, 2+ events within 60s)
- [ ] Implement displacement check (cascade moved price ≥10 bps) — **THE key filter**
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
| **Filter comparison** | **`FINDINGS_v26k_filter_comparison.md`** | **`liq_filter_comparison.py`** |
| OOS validation | `FINDINGS_v41_cascade_mm_oos.md` | — |
| Bad hours | `FINDINGS_v33_temporal_patterns.md` | — |
| Long-only, contagion | `FINDINGS_v42_v42f_summary.md` | — |
| Full research index | `RESEARCH_INDEX_liquidation_strategy.md` | — |
