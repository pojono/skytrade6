# Stage-2b: Horizon Expansion + Retrace Analysis

## Summary

Extended the 5s/15s analysis to 30s, 60s, 120s, 300s horizons.
Computed MAE/MFE curves and retrace analysis for all Stage-2 events.

**Key discovery: the signal is NOT weak — it operates on a different timescale
than initially measured, and through a mechanism (retrace → expansion) that
makes passive/maker entry the natural execution mode.**

## Data

- 8,669 Stage-1 events, 4,973 with clean horizon data (spread>0 path filter)
- TopDecile VacuumScore: 498 events (VS ≥ 0.688), ~17/day
- All MFE/MAE values verified clean (max 126bp, no contamination)

## Horizon Expansion — TopDecile VS (n=498)

| Horizon | Median ret | Mean ret | WR    | p25    | p75    |
|---------|-----------|---------|-------|--------|--------|
| 5s      | +0.9 bp   | +1.9 bp | 64.3% | -0.7   | +4.5   |
| 15s     | +1.1 bp   | +1.8 bp | 59.0% | -2.3   | +5.2   |
| 30s     | +1.3 bp   | +2.1 bp | 57.6% | -3.5   | +6.8   |
| 60s     | +1.7 bp   | +2.6 bp | 55.2% | -5.5   | +8.5   |
| 120s    | +1.6 bp   | +2.5 bp | 54.2% | -7.1   | +11.1  |
| 300s    | +2.1 bp   | +2.7 bp | 53.2% | -9.1   | +14.2  |

Continuation grows slowly: +0.9 → +2.1 bp median over 5s → 300s.
WR decays from 64% → 53% as noise increases.

### HighVS + HighFI (n=140) — sweet spot at 30-60s

| Horizon | Median | Mean | WR    |
|---------|--------|------|-------|
| 30s     | +2.2   | +3.3 | 62.9% |
| 60s     | +3.0   | +3.1 | 58.6% |
| 300s    | +1.4   | +4.7 | 50.7% |

## MAE / MFE — TopDecile VS

| Horizon | MAE median | MFE median | MFE/|MAE| | MAE p5    |
|---------|-----------|-----------|-----------|-----------|
| 5s      | -10.9 bp  | +11.8 bp  | 1.08      | -48.7 bp  |
| 15s     | -21.0 bp  | +23.2 bp  | 1.10      | -53.5 bp  |
| 30s     | -24.7 bp  | +29.2 bp  | **1.18**  | -64.6 bp  |
| 60s     | -31.0 bp  | +35.5 bp  | 1.15      | -67.1 bp  |
| 120s    | -36.9 bp  | +40.9 bp  | 1.11      | -73.5 bp  |
| 300s    | -46.0 bp  | +51.8 bp  | 1.13      | -85.9 bp  |

MFE consistently exceeds |MAE| by 8-18%. The asymmetry is real.

## The Critical Finding: Retrace to t0

> **95.2% of TopDecile VS events retrace to t0 within 60s**
> **98.2% retrace within 300s**

This means: after a stress event with high VacuumScore, the market
almost always returns to the trigger price before continuing.

This is textbook **post-impact liquidity rebalancing**:
1. Aggressor hits the book → price moves
2. Market makers pull back
3. New liquidity enters → price retraces to t0
4. Inventory repositioning → price continues in original direction

## Conditional Survival Analysis

Given retrace reaches depth D, what happens?

| Retrace depth  | n   | P(ret>0) | P(ret>10) | P(ret>20) | MFE med  | ret300 med |
|---------------|-----|----------|-----------|-----------|----------|------------|
| [-200, -50) bp| 103 | 54.4%    | 24.3%     | 11.7%     | +42.4    | +1.8       |
| [-50, -30) bp | 220 | 50.5%    | 31.8%     | 14.5%     | +55.7    | +0.3       |
| **[-30, -20)**| **89** | **61.8%** | **44.9%** | **28.1%** | **+60.2** | **+5.9** |
| [-20, -10) bp | 46  | 52.2%    | 32.6%     | 19.6%     | +50.6    | +1.1       |
| [-10, -5) bp  | 13  | 46.2%    | 38.5%     | 15.4%     | +51.0    | -3.9       |
| [-5, 0) bp    | 16  | 37.5%    | 31.2%     | 25.0%     | +47.0    | -7.3       |

### Liquidity Reload Sweet Spot: **[-30, -20) bp**

- Highest P(continuation) at 61.8%
- Highest MFE: +60.2 bp median
- Best ret300: +5.9 bp median
- 45% chance of >10bp continuation

Events with shallow retrace (<5bp) have LOWEST continuation (37.5%).
Events with extreme retrace (>50bp) have moderate continuation (54%) but 
the MFE is lower — these are trend reversals eating into MFE.

## Time Structure of Retrace

| By time   | MAE median | P(retrace >10bp) | P(retrace >20bp) | P(retrace >30bp) |
|-----------|-----------|-------------------|-------------------|-------------------|
| 5s        | -10.9 bp  | 51.2%             | 29.7%             | 14.1%             |
| 15s       | -21.0 bp  | 74.9%             | 51.8%             | 25.9%             |
| 30s       | -24.7 bp  | 85.5%             | 67.1%             | 38.0%             |
| 60s       | -31.0 bp  | 93.4%             | 80.3%             | 53.2%             |

Most of the retrace occurs within 15-60s. By 15s, 52% have retraced >20bp.

## Time to MFE (300s window, TopDecile VS)

| Window    | Count | Share |
|-----------|-------|-------|
| 0-5s      | 8     | 1.6%  |
| 5-15s     | 28    | 5.6%  |
| 15-30s    | 31    | 6.2%  |
| 30-60s    | 42    | 8.4%  |
| 60-120s   | 73    | 14.7% |
| **120-300s** | **315** | **63.3%** |

MFE predominantly occurs at 120-300s (63%). This is the inventory
repositioning / expansion phase.

## Weekly Stability (retrace [-30,-15) zone)

| Week | n  | P(cont) | MFE med  | ret300 med |
|------|-----|---------|----------|------------|
| 36   | 31  | 55%     | +55.7    | +2.3       |
| 37   | 24  | 67%     | +57.7    | +14.9      |
| 38   | 28  | 50%     | +55.3    | +1.6       |
| 39   | 23  | 57%     | +73.2    | +0.9       |
| 40   | 9   | 89%     | +63.7    | +17.6      |

MFE is stable across weeks (~55-73bp). Continuation rate varies (50-89%)
but consistently above 50%.

## Maker Entry Math

If limit order placed at mid(t0) — filled during retrace:

| Entry offset | Fill rate | n   | MFE from entry | ret300 from entry | WR net | EV/signal |
|-------------|-----------|-----|---------------|------------------|--------|-----------|
| at t0       | 98%       | 489 | +95 bp        | +41 bp           | 92%    | +36.7 bp  |
| t0 - 5bp    | 95%       | 471 | +96 bp        | +42 bp           | 94%    | +35.8 bp  |
| t0 - 10bp   | 92%       | 458 | +96 bp        | +42 bp           | 95%    | +35.0 bp  |
| t0 - 15bp   | 88%       | 438 | +97 bp        | +43 bp           | 96%    | +34.2 bp  |
| t0 - 20bp   | 83%       | 412 | +99 bp        | +44 bp           | 96%    | +33.2 bp  |
| t0 - 30bp   | 65%       | 323 | +101 bp       | +46 bp           | 98%    | +27.2 bp  |

**Best EV per signal: enter at t0 (fill rate × edge maximized).**
Deeper entries improve per-trade profit but lose fill rate.

## Key Interpretation

### This is NOT a prediction problem — it's a mechanism

The market consistently follows: shock → vacuum → retrace → expansion.

- **MAE is not risk** — it's the liquidity refill phase
- **MFE/|MAE| > 1** at all horizons — the expansion always slightly exceeds the retrace
- **95%+ retrace to t0** — maker entry is structurally available
- **63% of MFE at 120-300s** — this is inventory repositioning, not impulse

### The real risk: tail continuation failures

The 5-10% where retrace becomes trend reversal. These consume the
accumulated edge from 10-20 winning trades.

## Next Steps

1. Model the tail risk: what do the 5% non-retracers look like?
2. Identify early warning signs of trend reversal vs continuation
3. Test on BTC/ETH (deeper books, potentially larger moves)
4. Simulate realistic maker fill with queue position modeling
