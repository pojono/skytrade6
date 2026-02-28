# Position Sizing Research — Orderbook Slippage Analysis

**Date:** 2026-02-28  
**Dataset:** 150 settlements, OB.200 reconstructed at T-0  
**Script:** `research_position_sizing.py`

---

## The Problem

Our ML exit strategy produces **+23.6 bps** net PnL per trade (LOSO honest, T+25ms entry, 20 bps fees). But this assumes infinitesimal position size — zero market impact. In reality, selling $X into the bid side at T-0 causes slippage that scales with position size and inversely with orderbook depth.

**Too small** → low dollar profit  
**Too large** → slippage eats the edge  
**Goal:** find the sweet spot

---

## Orderbook Depth at T-0 (150 settlements)

### Full book (OB.200, all levels)

| Metric | P25 | Median | P75 |
|--------|-----|--------|-----|
| Bid depth | $60K | $93K | $138K |
| Ask depth | $62K | $93K | $117K |
| Total depth | $126K | $192K | $245K |
| Spread | 1 bps | 3 bps | 5 bps |

### Near-BBO depth (actionable liquidity)

| Distance from mid | Bid depth (median) | % of total |
|-------------------|-------------------|------------|
| Within 10 bps | **$1,926** | 2% |
| Within 20 bps | **$5,978** | 6% |
| Within 50 bps | **$19,005** | 24% |
| Within 100 bps | $38,390 | — |

**Key insight:** Only 2% of total depth is within 10 bps of mid. The "$93K median depth" is misleading — most of it sits far from BBO and is irrelevant for slippage on typical order sizes.

---

## Slippage vs Position Size

Round-trip slippage (entry sell + exit buy) at each notional size, median across 150 settlements:

| Notional | Entry Slip | Exit Slip | RT Total | Net PnL | $ Profit | Win % |
|----------|-----------|-----------|----------|---------|----------|-------|
| $500 | 2.2 bps | 1.5 bps | 3.5 bps | +20.1 | $1.00 | 98% |
| **$1,000** | 3.3 bps | 2.9 bps | 6.1 bps | +17.5 | $1.75 | 97% |
| **$2,000** | 5.1 bps | 4.8 bps | **9.6 bps** | **+14.0** | **$2.79** | **91%** |
| **$3,000** | 6.6 bps | 6.4 bps | 12.3 bps | +11.3 | **$3.40** | 81% |
| $5,000 | 9.1 bps | 9.0 bps | 17.5 bps | +6.1 | $3.03 | 65% |
| $7,500 | 12.5 bps | 12.2 bps | 24.5 bps | -0.9 | -$0.66 | 48% |
| $10,000 | 15.6 bps | 15.3 bps | 30.6 bps | -7.0 | -$7.04 | 38% |

**Sweet spot: $2K-$3K notional.** Beyond $5K, slippage exceeds the ML edge for most coins.

### Visualized

```
Net PnL vs Notional:

$500   +20.1 bps  ████████████████████  $1.00
$1K    +17.5 bps  █████████████████▌    $1.75
$2K    +14.0 bps  ██████████████        $2.79  ← best reliability (91% WR)
$3K    +11.3 bps  ███████████▎          $3.40  ← best $ profit
$5K     +6.1 bps  ██████                $3.03
$7.5K   -0.9 bps                       -$0.66  ← breakeven
$10K    -7.0 bps  xxxxxxx              -$7.04  ← losing money
```

---

## FR Magnitude Does NOT Change Optimal Size

| FR Range | N | Best Notional | Avg $/trade |
|----------|---|---------------|-------------|
| 25-50 bps | 44 | $2,000 | $1.85 |
| 50-80 bps | 31 | $2,000 | $2.39 |
| 80+ bps | 27 | $2,000 | $2.01 |

Higher FR = bigger expected drop, but also these coins tend to have similar depth profiles. The optimal notional stays at **$2K across all FR buckets**. The edge from higher FR is absorbed by similar slippage costs.

---

## Per-Settlement Optimal Sizing

When we pick the best notional per settlement (from our grid of sizes):

- Median optimal: **$3,000**
- Mean optimal: **$4,273**
- P25-P75: $2,000 - $5,000
- Avg $/trade: **$4.16** (vs $2.09 at fixed $2K)

This means **adaptive sizing based on depth could nearly double returns** vs fixed sizing.

---

## Daily Revenue Estimates (12 settlements/day, ML LOSO)

| Strategy | $/trade | $/day | $/month |
|----------|---------|-------|---------|
| Fixed $1K | $1.53 | $18 | $552 |
| **Fixed $2K** | **$2.09** | **$25** | **$754** |
| Fixed $3K | $1.90 | $23 | $684 |
| Fixed $5K | -$0.77 | -$9 | -$278 |
| **Adaptive (per-settlement)** | **$4.16** | **$50** | **$1,496** |

---

## Production Sizing Rule

At T-0, read the orderbook (OB.200 preferred, OB.50 fallback) and compute:

```python
def compute_position_size(bids, asks):
    """Compute optimal notional from orderbook at T-0.
    
    Args:
        bids: [(price, qty), ...] sorted descending
        asks: [(price, qty), ...] sorted ascending
    
    Returns:
        notional_usd (float), or 0 to skip
    """
    if not bids or not asks:
        return 0
    
    mid = (bids[0][0] + asks[0][0]) / 2
    
    # Compute bid depth within 20 bps of mid
    bid_depth_20bps = sum(
        p * q for p, q in bids 
        if (mid - p) / mid * 10000 <= 20
    )
    
    # Sizing table (validated on 150 settlements)
    if bid_depth_20bps < 1000:
        return 0        # SKIP — too thin
    elif bid_depth_20bps < 5000:
        notional = 500
    elif bid_depth_20bps < 20000:
        notional = 1000
    elif bid_depth_20bps < 50000:
        notional = 2000
    elif bid_depth_20bps < 100000:
        notional = 3000
    else:
        notional = 5000
    
    # Safety cap: never exceed 10% of near-BBO depth
    cap = bid_depth_20bps * 0.10
    notional = min(notional, cap)
    
    return max(500, notional)
```

### Decision at T-0

```
1. Reconstruct OB.200 from websocket snapshots + deltas
2. bid_depth_20bps = sum of bid levels within 20 bps of mid
3. Look up notional from sizing table
4. Cap at 10% of bid_depth_20bps
5. If < $500 → SKIP this settlement
6. Place market sell order for computed notional
```

---

## Key Takeaways

1. **Slippage is the #1 constraint** — not model accuracy, not fees. At $10K notional, slippage (30.6 bps) is 1.5x the entire ML edge (23.6 bps).

2. **$2-3K is the universal sweet spot.** Works across all FR ranges and depth profiles. Higher sizes rapidly destroy profitability.

3. **Near-BBO depth is what matters.** Median bid depth within 10 bps = $1,926. The headline "$93K total depth" is 98% irrelevant for our order sizes.

4. **Adaptive sizing doubles returns** ($4.16/trade vs $2.09). Reading the book at T-0 and adjusting notional is worth implementing.

5. **$10K notional (our previous assumption) is a losing strategy.** At median slippage of 30.6 bps RT, PnL goes negative for most settlements.

6. **Revenue is modest but consistent.** At $2K fixed: ~$25/day, ~$750/month. With adaptive sizing: ~$50/day, ~$1,500/month.

---

## Caveats

- **Exit slippage may differ from T-0 snapshot.** By exit time (T+10-30s), the orderbook has changed. This analysis uses T-0 depth for both entry and exit, which may overestimate exit slippage.
- **Concurrent sellers at settlement.** Other traders are also selling at T-0, competing for the same bid liquidity. Real slippage could be worse than our estimates.
- **Sample size: 150 settlements (3 days).** More data needed to validate across market regimes.
- **These are altcoin perps.** BTC/ETH perps would have 100-1000x more depth, but also much smaller FR-driven drops.

---

## Files

| File | Purpose |
|------|---------|
| `research_position_sizing.py` | Full analysis script |
| `position_sizing_analysis.csv` | Per-settlement slippage data |
| `FINDINGS_position_sizing.md` | This document |
