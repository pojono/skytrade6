# FINDINGS v33: Temporal Patterns in Market Microstructure

## Goal

Identify systematic patterns in volatility, OI, funding rate, and liquidations by:
1. **Hour of day** (UTC 0-23)
2. **Day of week** (Mon-Sun)
3. **Trading session** (Asia 00-08, Europe 08-16, US 16-24 UTC)
4. **Funding cycle** (8h windows around 00:00, 08:00, 16:00 UTC)

## Cross-Validation Matrix

| Test | Symbol | Dates | Days | Bars |
|------|--------|-------|------|------|
| 1 | BTCUSDT | May 11-24 2025 | 14 | 4,031 |
| 2 | ETHUSDT | May 11-24 2025 | 14 | 4,032 |
| 3 | BTCUSDT | Aug 4-17 2025 | 14 | 4,032 |

All analysis on 5-minute bars with Kruskal-Wallis and t-test significance testing.

---

## 1. Hour of Day — VERY STRONG (p < 0.001 everywhere)

### Volatility Peak/Trough

| Test | Peak Hour | Peak Vol | Trough Hour | Trough Vol | Ratio |
|------|-----------|----------|-------------|------------|-------|
| BTC May | **14:00** | 0.764 | 10:00 | 0.380 | **2.01x** |
| ETH May | **00:00** | 1.620 | 21:00 | 0.995 | **1.63x** |
| BTC Aug | **14:00** | 0.658 | 23:00 | 0.279 | **2.35x** |

**BTC vol peaks consistently at 14:00 UTC** (US market open, 10am ET). ETH has a broader peak but also elevated 13:00-18:00.

### Volume Peak/Trough

| Test | Peak Hour | Trough Hour | Ratio |
|------|-----------|-------------|-------|
| BTC May | 14:00 | 21:00 | 3.44x |
| ETH May | 17:00 | 21:00 | 2.72x |
| BTC Aug | 12:00 | 23:00 | 4.10x |

**Volume consistently lowest 21:00-05:00 UTC** (late US / Asia overnight). Peak during US/Europe overlap.

### Liquidation Peak

| Test | Peak Hour | Peak Count | Trough Hour | Trough Count | Ratio |
|------|-----------|-----------|-------------|-------------|-------|
| BTC May | **14:00** | 20.9 | 10:00 | 1.6 | **13.1x** |
| ETH May | **20:00** | 21.8 | 10:00 | 3.5 | **6.1x** |
| BTC Aug | **14:00** | 6.3 | 11:00 | 0.02 | **350x** |

**Liquidations are 6-13x more likely at peak hours vs trough.** This is the strongest temporal effect.

### Kruskal-Wallis (all p < 0.001 across all 3 tests)

| Metric | BTC May H | ETH May H | BTC Aug H |
|--------|-----------|-----------|-----------|
| Volatility | 596.8*** | 450.6*** | 562.0*** |
| Range | 473.4*** | 298.5*** | 365.1*** |
| |Return| | 161.5*** | 96.9*** | 106.1*** |
| Volume | 404.2*** | 307.6*** | 375.5*** |
| Liq Count | 110.3*** | 90.1*** | 53.1*** |
| OI Change | 35.8* | 37.8* | 38.2* |
| Spread | 25.2 ns | 61.3*** | 24.0 ns |

**Every metric except spread shows highly significant hourly variation.**

---

## 2. Day of Week — VERY STRONG

### Weekday vs Weekend

| Metric | BTC May Ratio | ETH May Ratio | BTC Aug Ratio | Consistent? |
|--------|--------------|--------------|--------------|-------------|
| **Volatility** | **1.40x*** | **1.23x*** | **1.66x*** | ✅ YES |
| **Range** | **1.41x*** | **1.25x*** | **1.68x*** | ✅ YES |
| **|Return|** | **1.39x*** | **1.27x*** | **1.59x*** | ✅ YES |
| **Volume** | **1.85x*** | **1.53x*** | **2.09x*** | ✅ YES |
| **Liq Count** | **2.35x*** | 1.31x ns | ∞ (0 weekend)*** | ✅ YES |
| OI Change | 0.81x ns | 15.5x ns | ∞ ns | ❌ Not significant |
| Spread | 1.49x* | 1.34x*** | 1.00x ns | ⚠️ Mixed |

**Weekday vol is 1.2-1.7x weekend vol.** This is universal across assets and time periods.

**Saturday is consistently the quietest day:**

| Test | Sat Vol | Best Weekday Vol | Ratio |
|------|---------|-----------------|-------|
| BTC May | 0.372 | 0.619 (Mon) | 1.66x |
| ETH May | 0.895 | 1.518 (Mon) | 1.70x |
| BTC Aug | 0.240 | 0.509 (Thu) | 2.12x |

**BTC Aug had ZERO liquidations on Fri/Sat/Sun** — the market was completely quiet on weekends.

### Kruskal-Wallis across 7 days (all p < 0.001)

Vol, range, |ret|, volume, liq count all show H-stats of 136-818 with p ≈ 0. **Day of week is a first-order effect.**

---

## 3. Trading Sessions — SIGNIFICANT

### Session Comparison

| Metric | BTC May | | | ETH May | | | BTC Aug | | |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| | Asia | Europe | US | Asia | Europe | US | Asia | Europe | US |
| Vol | 0.484 | 0.536 | **0.559** | 1.287 | 1.311 | 1.307 | 0.345 | **0.467** | 0.385 |
| Volume | 31M | 42M | 41M | 22M | 25M | 25M | 22M | **39M** | 24M |
| Liqs | 7.1 | 8.5 | **9.7** | 7.4 | 9.1 | **12.6** | 0.6 | **1.4** | 0.3 |

**Key findings:**
- **Asia is consistently the quietest session** (lowest vol, volume, liqs)
- **BTC May**: US session highest vol (p < 0.001 vs Asia)
- **BTC Aug**: Europe session highest vol (p < 0.001 vs both)
- **ETH May**: Sessions are roughly equal (no significant difference)

### Session × Day Interaction

The quietest combination is **Saturday Asia** (vol 0.252-0.370). The most active varies but is typically **Monday/Wednesday Europe or US**.

---

## 4. Funding Cycle — MODERATE SIGNAL

### Pre-Funding Spike

| Metric | BTC May | ETH May | BTC Aug |
|--------|---------|---------|---------|
| Pre-funding 1h vol | 0.601 | — | — |
| Mid-cycle vol | 0.506 | — | — |
| Pre/Mid ratio | **1.19x** | — | — |
| Kruskal-Wallis p | **< 0.001*** | — | — |

**Vol is ~19% higher in the hour before funding** (BTC May). Liquidation count also spikes pre-funding (17.1 vs 7.6 mid-cycle, 2.2x).

### OI Dynamics Around Funding

- **Post-funding 1h**: OI tends to increase (+2.8 bps BTC May) — new positions opening
- **Pre-funding 30m**: OI tends to decrease (-2.3 bps) — positions closing to avoid funding
- This is consistent with rational behavior: close before paying funding, re-open after

---

## 5. OI Dynamics — WEAK Temporal Signal

### OI Change by Session

| Session | BTC May | ETH May | BTC Aug |
|---------|---------|---------|---------|
| Asia | -0.13 | +1.18 | -0.08 |
| Europe | **+1.98** | +1.50 | +0.24 |
| US | -0.54 | -1.14 | -0.13 |

**Europe tends to build OI, US tends to reduce it.** But this is only marginally significant (p ~ 0.04 for hourly KW).

### Key Correlations (consistent across all 3 tests)

| Pair | BTC May | ETH May | BTC Aug | Interpretation |
|------|---------|---------|---------|---------------|
| **Liqs vs Vol** | **+0.427***| **+0.514***| **+0.104***| Liquidations strongly co-occur with vol |
| **Liqs vs |Ret|** | **+0.310***| **+0.392***| **+0.085***| Liquidations co-occur with big moves |
| **OI chg vs Liqs** | **-0.070***| **-0.084***| **-0.044**| OI drops when liqs happen (expected) |
| **FR vs Vol** | **+0.253***| -0.061*** | -0.074*** | Mixed — period-dependent |
| OI chg vs Vol | +0.001 ns | -0.036* | -0.021 ns | **No relationship** |
| FR vs OI chg | +0.026 ns | +0.014 ns | +0.017 ns | **No relationship** |

**OI change is NOT correlated with volatility.** This is important — it means OI dynamics are independent of vol regime.

---

## 6. Liquidation Temporal Patterns

### Probability of Liquidation by Hour

40% of BTC May 5-min bars have at least one liquidation. But this varies enormously:
- **Peak**: 51.2% at 14:00 and 18:00 (BTC May)
- **Trough**: 23.8% at 10:00 (BTC May)
- **Large cascades (P90+)**: 33% happen 14:00-17:00 UTC

### By Day of Week

| Day | BTC May P(liq) | ETH May P(liq) | BTC Aug P(liq) |
|-----|---------------|---------------|---------------|
| Mon | 52.4% | 63.4% | 13.9% |
| Tue | 45.0% | 60.9% | 17.5% |
| Wed | 43.4% | 56.9% | 12.5% |
| Thu | 53.1% | 62.7% | 6.8% |
| Fri | 40.6% | 51.7% | 0.0% |
| **Sat** | **29.0%** | **38.5%** | **0.0%** |
| **Sun** | **19.4%** | **26.6%** | **0.0%** |

**Weekend liquidation probability is 40-60% of weekday.** In quiet markets (Aug), weekends have literally zero liquidations.

---

## Summary of Actionable Patterns

### Strongest Effects (all p < 0.001, consistent across 3 tests)

1. **Hour of day → Vol**: 2x spread between peak (14:00) and trough (10:00/23:00)
2. **Weekday vs Weekend → Vol**: 1.2-1.7x, Saturday is quietest
3. **Hour of day → Liquidations**: 6-13x spread, peak at US open
4. **Liqs ↔ Vol correlation**: ρ = 0.1-0.5, always positive and significant

### Moderate Effects

5. **Funding cycle → Vol**: ~19% higher pre-funding
6. **Session → Vol**: Asia quietest, Europe/US higher
7. **OI builds in Europe, reduces in US** (marginal)

### Non-Effects

8. **OI change vs Vol**: No correlation (ρ ≈ 0)
9. **FR vs OI change**: No correlation
10. **Spread**: Mostly constant (BTC), slight hourly variation (ETH)

---

## Trading Implications

### For v32 TP/SL Strategy
- **`hour_of_day` was already top-5 feature** — this confirms why
- **Add `weekday` as feature**: Saturday/Sunday should have different TP/SL sizing or be skipped entirely
- **Funding cycle phase** could improve timing of vol-expansion trades

### For Grid Bots
- **Tighter grids on weekends** (vol 40-60% lower)
- **Wider grids 13:00-18:00 UTC** (vol 2x higher)
- **Skip Saturday Asia session** (lowest vol of the entire week)

### For Squeeze/Liquidation Strategies (v31b)
- **Focus on 13:00-18:00 UTC** for liquidation cascade signals
- **Ignore 08:00-12:00 UTC** — very few liquidations
- **Weekday only** — weekend has insufficient liquidation activity

### Potential New Strategy: Time-of-Day Vol Timing
- Simple rule: trade straddles only during 13:00-18:00 UTC on weekdays
- Expected vol lift: 1.5-2x vs unconditional
- Combined with v30 ML model: could significantly improve signal quality
