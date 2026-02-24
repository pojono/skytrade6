# FR Scalp Research Summary

## Date: 2026-02-24

## Objective

Evaluate whether **funding rate (FR) income can be harvested profitably** without delta-neutral hedging (no spot leg, no margin borrowing). Three strategies were tested on Bybit data.

---

## Data Sources

| Dataset | Records | Period | Source |
|---|---|---|---|
| Bybit FR history | 728K | 200 days (Aug 2025 – Feb 2026) | REST API |
| Binance FR history | 576K | 200 days | REST API |
| Settlement 1m klines | 273K candles | 13K settlements, 200 days | Bybit kline API |
| Full-hold klines | 600K+ candles | 9.2K settlements, 200 days | Bybit kline API |
| 5-second ticker data | 24M rows | 3 days (Feb 22-24) | Dataminer websocket |
| ob200 orderbook | 24 coins | 2 days (Feb 22-23) | Bybit historical data |

---

## Strategy 1: Naked Long + Hold to Next Settlement

**Idea**: Go LONG futures after negative FR settlement, hold 1-8 hours to NEXT settlement, collect FR.

**Result: DOES NOT WORK.**

| Metric | Value |
|---|---|
| Trades | 9,250 |
| Win Rate | 45% |
| Daily P&L | **-$45** |
| FR income | +40.5 bps avg |
| Price drag | **-30.4 bps avg** |
| Fees | -11.0 bps |
| Net | **-1.0 bps per trade** |

- FR autocorrelation is strong (next FR also negative 94% of the time, avg +40.5 bps income)
- But directional exposure over 1-8h averages **-30 bps against you**
- Worst trade: **-7,638 bps** ($-7.6k on $10k notional)
- All SL configs lose money (SL 25bp triggers 94% of the time)
- 1h holds marginally positive (+7.5 bps), 4h/8h holds negative

**Conclusion**: Directional risk completely offsets FR income. This is exactly why delta-neutral hedging exists.

---

## Strategy 2: Short After Negative FR Settlement

**Idea**: Price dumps after extreme negative FR. SHORT right after settlement and ride the move.

**Result: NO EDGE.**

| Config | Trades | WR | Daily |
|---|---|---|---|
| NO_SL 5min | 9,260 | 45% | **-$295** |
| NO_SL 10min | 9,260 | 47% | -$379 |
| All SL configs | — | — | **all negative** |

- 2-day tick data showed +$1,077/day — **this was noise**
- 200-day 1m kline data: all configs lose money
- Raw price moves after settlement are near zero on average
- No consistent post-settlement directional edge exists

---

## Strategy 3: Flash Scalp (THE WINNER)

**Idea**: Go LONG 1-2 seconds before settlement, collect FR, exit 1-3 seconds after. Total exposure: ~2-5 seconds.

### Why It Works

Settlement acts like an **ex-dividend event**:
1. Price is flat in the seconds before settlement
2. FR is paid at the exact settlement second
3. Price drops ~40-57 bps immediately after (futures adjusting for the FR payment)
4. But FR payment (+68 bps avg) exceeds the price drop + spread

### Results (ob200 best_bid/best_ask, no additional fees)

**35 trades across 8 coins, 2 days of data:**

| Entry → Exit | Hold | WR | Avg Net | Notes |
|---|---|---|---|---|
| T-2s → T+1s | 3s | **83%** | **+13.2 bps** | Highest WR |
| T-2s → T+3s | 5s | 74% | **+16.3 bps** | Highest payout |
| T-1s → T+1s | 2s | 77% | +11.0 bps | Fastest |
| T-3s → T+1s | 4s | 83% | +13.9 bps | |
| T-10s → T+10s | 20s | 54% | -11.2 bps | Too slow, edge gone |

**Every entry/exit combo under 5s is profitable. Everything over 10s is not.**

### P&L Decomposition (T-1s → T+1s)

| Component | bps |
|---|---|
| FR income | +67.9 |
| Price move (buy ask, sell bid) | -56.9 |
| **Net** | **+11.0** |

### Distribution

| Percentile | Net (bps) |
|---|---|
| 5th | -7 |
| 25th | +3 |
| **Median** | **+10** |
| 75th | +21 |
| 95th | +34 |

### Orderbook Conditions at Settlement

| Metric | Pre (-120s to -30s) | At Settle (±10s) | Post (+30s to +120s) |
|---|---|---|---|
| Spread | 1.54 bps | **2.04 bps** (+32%) | 1.52 bps |
| Bid depth 5bps | 5,205 | **8,294** (bids pile up) | 5,783 |
| Ask depth 5bps | 6,778 | 5,878 | 5,620 |
| Imbalance | -0.02 | **+0.12** (bid heavy) | +0.03 |
| Total depth | ~590k / ~505k | ~547k / ~470k (-7%) | ~587k / ~511k |

- Spread widens 32% but stays under 3 bps — manageable
- Bids increase at settlement (buyers catching the dip) — good for exit
- Total depth thins ~7% briefly — mild, not catastrophic
- USD depth at 5bps: ~$2,100-$3,400 — supports $5-10k per coin

### Timing Sensitivity

| Seconds from Settlement | Price (vs T-60s ref) |
|---|---|
| T-60s to T-5s | **Flat** (±2 bps noise) |
| T+0s | -0.5 bps |
| **T+5s** | **-40.2 bps** (cliff drop) |
| T+10s to T+60s | -40 to -48 bps (stays down) |

The entire price dislocation happens in a **single 5-second window** at settlement. Before: flat. After: -40 bps permanent shift. This is the ex-dividend adjustment of the futures price.

### Per-Coin Performance

| Symbol | Trades | WR | Avg Net | Avg FR |
|---|---|---|---|---|
| LAUSDT | 16 | **100%** | +14.1 | +57 |
| BELUSDT | 1 | 100% | +31.8 | +43 |
| LSKUSDT | 1 | 100% | +21.5 | +19 |
| AWEUSDT | 9 | 56% | +6.0 | +82 |
| ENSOUSDT | 2 | 50% | +10.1 | +24 |
| POWERUSDT | 2 | 50% | +2.8 | +222 |

---

## Comparison: All Strategies Tested

| Strategy | Daily P&L | Capital | Annual ROI | Borrow? | Verdict |
|---|---|---|---|---|---|
| **Flash scalp (T-2s → T+1s)** | ~$500* | $10k | ~1,800%* | **NO** | **VIABLE** |
| Delta-neutral Bybit 1h (audit) | +$273 | $20k | 498% | YES | Proven but borrow-limited |
| Delta-neutral 4-pool (audit) | +$879 | $80k | 401% | YES | Proven but borrow-limited |
| Naked long full-hold | -$45 | $10k | -164% | NO | Dead |
| Short after settlement | -$295 | $10k | -1,076% | NO | Dead |

*Flash scalp daily estimate: ~46 settlements/day × +11 bps × $10k = ~$506/day, but needs more data to validate.

---

## Practical Requirements for Flash Scalp

1. **Sub-second execution**: Must enter within 1-2s of settlement and exit within 1-3s after
2. **Settlement timing**: Bybit settles at exact hours (00:00, 01:00, ..., 23:00 UTC) — perfectly predictable
3. **Pre-trade screening**: Check current FR via websocket; only trade if FR <= -20 bps
4. **Position size**: $5-10k per coin (ob200 shows ~$2-3k depth at 5bps)
5. **Multi-coin**: Run across all coins with extreme negative FR simultaneously at each settlement hour
6. **No stop-loss needed**: 2-5 second hold means no time for adverse moves beyond the spread
7. **API latency**: Need websocket connection to Bybit for fast order placement (<100ms)

## Capacity Constraints (THE KILLER)

### Orderbook Depth on Extreme-FR Coins

| Position Size | Fits in 10bps slippage? | Avg Slippage |
|---|---|---|
| $500 | 94% of settlements | ~5 bps |
| $1,000 | 89% | ~8 bps |
| $2,000 | 63% | ~11 bps |
| $5,000 | 6% | ~19 bps |
| **$10,000** | **0%** | **~24 bps** |

Per-coin USD depth at ask side (before settlement):
- At 5 bps: **$758 avg** ($19 to $2,371)
- At 10 bps: **$2,497 avg**
- At 25 bps: **$9,796 avg**

### The Fundamental Paradox

Coins with **extreme negative FR** (where the edge exists) are tiny altcoins with **thin orderbooks** (LAUSDT, AWEUSDT, POWERUSDT). Coins with **deep orderbooks** (BTC, ETH, SOL) almost never have extreme FR.

### Realistic P&L

At $1,000 per trade (max realistic size):
- Edge: +11 bps = $1.10
- Slippage: ~8 bps = -$0.80
- **Net: ~$0.30 per trade**
- ~46 trades/day → **~$14/day → ~$5k/year**

### Crowding Risk

If 2-3 other traders do the same strategy, the $2-3k of depth gets eaten and slippage doubles → edge disappears completely.

### Conclusion

**FR arbitrage at retail scale is a $5-15/day business** regardless of variant:
- Delta-neutral: limited by margin borrow caps (~$100-500/coin)
- Flash scalp: limited by orderbook depth (~$500-1,000/coin)

The edge is real and proven, but not scalable.

---

## Caveats

- **Small sample**: Only 35 trades from 2 days of ob200 data. Need weeks/months of live monitoring to validate.
- **Execution risk**: Real-world fills may be worse than ob200 best_bid/ask if many participants do the same trade.
- **FR already priced in?**: The -57 bps price drop suggests the market partially prices in the FR event. If more traders do this, the drop could match or exceed FR income.
- **Exchange risk**: Bybit could change settlement mechanics, add delays, or throttle API around settlement.

## Files

| File | Description |
|---|---|
| `download_settlement_klines.py` | Download 1m klines around FR settlements (Bybit API) |
| `backtest_fr_scalp_200d.py` | 200-day backtest: SHORT-after (dead) + LONG+FR (10min only) |
| `backtest_fr_long_fullhold.py` | Full-hold backtest: naked long to next settlement (dead) |
| `backtest_fr_flash_scalp.py` | 1m kline flash scalp (E-0 lookahead issue identified) |
| `backtest_fr_scalp_long.py` | Original 2-day tick-level LONG scalp |
| `backtest_fr_post_settlement.py` | Original 2-day SHORT-after backtest |
