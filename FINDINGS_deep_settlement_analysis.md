# Deep Settlement Analysis — Full 60s Trajectory

**Date:** 2026-02-28  
**Dataset:** 131 settlements, 30 symbols, 3 dates (Feb 26-28)  
**Script:** `analyse_settlement_deep.py`

---

## Critical Finding: We Were Only Measuring the First 5 Seconds!

Our previous analysis used a **5-second window**. The real picture is dramatically different:

| Metric | First 5s only | Full 60s |
|--------|--------------|----------|
| Avg worst drop | -74.6 bps | **-104.5 bps** |
| Median worst drop | -53.8 bps | **-78.5 bps** |
| Bottoms captured | 40% | **100%** |

**60% of bottoms happen AFTER T+5s.** The median bottom is at **T+11.4s**, not T+1-2s as we assumed.

### Price trajectory (avg last trade price at each horizon):

| Exit Time | Avg Price | Median Price | Profit (avg) | Win Rate |
|-----------|-----------|-------------|-------------|----------|
| T+0.5s | -48.0 bps | -36.5 bps | +28.0 bps | 68% |
| T+1s | -47.2 bps | -36.6 bps | +27.2 bps | 67% |
| T+5s | -50.8 bps | -35.5 bps | +30.8 bps | 67% |
| T+10s | **-53.2 bps** | -36.9 bps | **+33.2 bps** | **70%** |
| T+30s | -51.5 bps | -30.8 bps | +31.5 bps | 63% |
| T+60s | -57.3 bps | -33.5 bps | +37.3 bps | 60% |

**Key insight:** Price drops fast in first 500ms, then keeps drifting down slowly. T+10s is the best risk/reward exit (highest win rate, good avg PnL).

---

## Recovery / Bounce After Initial Drop

### Recovery speed from bottom:

| Time after bottom | Avg recovery | % of drop recovered |
|-------------------|-------------|-------------------|
| +100ms | +11.5 bps | 20% |
| +500ms | +16.5 bps | 24% |
| +1s | +20.7 bps | 32% |
| +5s | +28.7 bps | 46% |
| +10s | +33.8 bps | 56% |
| +30s | +46.0 bps | 106% |

**Only 27% of settlements fully recover** to the pre-settlement price. The average max recovery is 166% of the drop (meaning some overshoot), but median is much less.

### Recovery by FR magnitude:

| FR Range | N | Avg Drop | Bottom @ | Rec@1s | Rec@5s | Rec@30s | Rec% | Full Recovery |
|----------|---|----------|----------|--------|--------|---------|------|---------------|
| \|FR\| 15-30 | 48 | -46.9 | 22.3s | +11.1 | +16.7 | +30.1 | 266% | 40% |
| \|FR\| 30-60 | 43 | -80.8 | 16.5s | +21.7 | +28.0 | +59.6 | 157% | 30% |
| \|FR\| 60-100 | 22 | -150.3 | 18.3s | +24.3 | +36.6 | +51.3 | 70% | 14% |
| \|FR\| >100 | 17 | -273.9 | 28.2s | +36.0 | +51.7 | +51.9 | 41% | 0% |

**Key insight:**
- **Small FR (15-30 bps):** Drops are small, but price often fully recovers (40%). These are the "noise" settlements.
- **Large FR (>100 bps):** Drops are massive (-274 bps avg), recovery is slow and incomplete (41%), and price **never** returns to pre-settlement level. These are structural shifts.

---

## Volume Patterns

### Volume surge at settlement:
- **31x normal rate** in the first second
- Volume stays elevated: 12x at 5s, 11x at 10s

### Can we predict volume?
- **|FR| → post volume: r = +0.76** (strong!) — bigger FR = more post-settlement volume
- **pre_vol → post volume: r = +0.75** (strong!) — actively traded coins stay active

### Sell pressure timeline:
| Window | Sell Ratio | Interpretation |
|--------|-----------|----------------|
| T+0.5s | **68.6%** | Heavy selling wave |
| T+1s | 65.5% | Still dominated by sells |
| T+2s | 62.9% | Selling fading |
| T+5s | 59.5% | Moderate sell bias |
| T+10s | 60.8% | Second sell wave? |
| T+30s | 55.4% | Near neutral |
| T+60s | 53.8% | Almost balanced |

**Key insight:** Sell pressure is strongest in first 500ms, decays to ~60% by T+5s, then slowly normalizes. There's a slight second sell wave around T+10s.

---

## Open Interest — YES, It Matters!

### Pre-settlement OI change:
- OI rising before settlement → bigger drops: **r = -0.44**
- This is the **2nd strongest predictor** after FR itself
- Interpretation: rising OI = new positions opening = more FR pressure at settlement

### Post-settlement OI trajectory:
| Time | Avg OI Change | Interpretation |
|------|--------------|----------------|
| T+5s | **+1.17%** | Positions opening (arbitrageurs entering?) |
| T+10s | +1.16% | Still elevated |
| T+30s | +0.18% | Most closed |
| T+60s | +0.02% | Back to baseline |

### By FR magnitude:
| FR Range | OI@5s | OI@30s | OI@60s |
|----------|-------|--------|--------|
| \|FR\|<40 | +0.23% | -0.01% | -0.08% |
| \|FR\| 40-80 | +1.63% | +0.33% | +0.02% |
| \|FR\|>80 | **+3.16%** | +0.52% | +0.29% |

**Key insight:** High-FR settlements see a massive OI spike (+3.2%) at T+5s — people are opening positions to capture the drop. By T+60s most have closed. **For |FR|>80, OI stays elevated (+0.29%) even at T+60s** — some positions are being held longer.

---

## Optimal Exit Timing by FR Magnitude

| FR Range | N | Exit T+1s | Exit T+5s | Exit T+10s | Exit T+30s |
|----------|---|-----------|-----------|------------|------------|
| \|FR\| 15-30 | 48 | **-9 bps** | **-4 bps** | **-2 bps** | **-4 bps** |
| \|FR\| 30-60 | 43 | +21 bps | +15 bps | +16 bps | -9 bps |
| \|FR\| 60-100 | 22 | +51 bps | +53 bps | **+63 bps** | **+75 bps** |
| \|FR\| >100 | 17 | +113 bps | +140 bps | +141 bps | **+181 bps** |

**Critical trading implications:**

1. **|FR| < 30 bps: DON'T TRADE.** Net loss at every exit time (fees eat the drop).
2. **|FR| 30-60 bps: Exit by T+10s.** Recovery kills profits if you hold too long (PnL goes negative by T+30s).
3. **|FR| 60-100 bps: Hold longer is better.** T+30s gives +75 bps (vs +51 at T+1s). Price keeps drifting down.
4. **|FR| > 100 bps: Hold even longer.** T+30s gives +181 bps! The sell wave is massive and sustained.

### Current strategy review:
- **Current exit: T+5.5s** — this is suboptimal for high-FR trades!
- For |FR|>60 bps, holding to T+10-30s captures 20-40 bps more profit
- For |FR|<30 bps, we should skip entirely

---

## Actionable Recommendations

### 1. Dynamic exit timing based on FR magnitude
```
if |FR| < 25 bps:  SKIP (don't trade)
if |FR| 25-50 bps: exit at T+5s  (quick scalp)
if |FR| 50-80 bps: exit at T+10s (let it drift)
if |FR| > 80 bps:  exit at T+20-30s (sustained sell wave)
```

### 2. Add OI change as a feature
- Pre-settlement OI change (r=-0.44) is a strong predictor
- Already in our V2 feature set but wasn't being utilized effectively

### 3. Volume-based position sizing
- |FR| predicts volume: r=0.76
- Pre-settlement volume predicts post volume: r=0.75
- Higher volume = better fills, less slippage → can size larger

### 4. Monitor sell ratio for exit signal
- When sell ratio drops below 55% → sell wave exhausted → time to exit
- Could implement as a real-time exit condition
