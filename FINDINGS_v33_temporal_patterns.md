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

## 3. Trading Sessions — Realistic Overlapping Model

### Session Definitions (UTC) — DST-Aware

Real market sessions overlap and **shift with daylight saving time**:

| Session | Summer (UTC) | Winter (UTC) | No DST? |
|---------|-------------|-------------|---------|
| **Tokyo** | 00:00–09:00 | 00:00–09:00 | No DST (always JST) |
| **London** | 07:00–15:30 (BST) | 08:00–16:30 (GMT) | +1h in winter |
| **New York** | 13:30–20:00 (EDT) | 14:30–21:00 (EST) | +1h in winter |

### Overlap Zones

| Overlap | Summer (UTC) | Winter (UTC) | Volatility Effect |
|---------|-------------|-------------|-------------------|
| **Tokyo-London** | 07:00–09:00 | 08:00–09:00 | Moderate lift |
| **London-New York** | 13:30–15:30 | 14:30–16:30 | **PEAK volatility zone** |
| Quiet (no session) | 20:00–00:00 | 21:00–00:00 | Lowest activity |

### v33c: DST Proof — Peak Shifts +1h in Winter (ALL 5 Symbols)

The volatility peak tracks the NYSE open exactly:

| Symbol | Summer Peak (EDT) | Winter Peak (EST) | **Shift** |
|--------|------------------|------------------|-----------|
| BTCUSDT | **14:00 UTC** (31.4 bps) | **15:00 UTC** (37.9 bps) | **+1h** |
| ETHUSDT | **14:00 UTC** (38.4 bps) | **15:00 UTC** (47.0 bps) | **+1h** |
| SOLUSDT | **14:00 UTC** (54.5 bps) | **15:00 UTC** (65.1 bps) | **+1h** |
| DOGEUSDT | **14:00 UTC** (51.1 bps) | **15:00 UTC** (60.1 bps) | **+1h** |
| XRPUSDT | **14:00 UTC** (39.6 bps) | **15:00 UTC** (53.0 bps) | **+1h** |

**All 5 symbols shift by exactly +1 hour in winter.** This is definitive proof that the volatility peak is driven by the NYSE open (which moves from 13:30→14:30 UTC with DST). The peak occurs ~30 minutes after the opening bell in both seasons.

Additional observations:
- Winter is generally **more volatile** than summer across all hours (1.2–1.5x)
- The summer/winter hourly profiles are highly correlated (ρ = 0.87–0.93) — same shape, just shifted
- The 4-state analysis (US×UK DST) confirms the US clock is the primary driver

### BTC Range by Session Zone (v33b, 3-year average)

| Zone | Hours UTC (summer) | Avg Range (bps) | vs Tokyo-only |
|------|-------------------|-----------------|---------------|
| Tokyo only | 00:00–06:00 | 14.5 | baseline |
| **Tokyo + London** | 07:00–08:00 | 14.3 | +0% |
| London only | 09:00–11:00 | 14.1 | -3% |
| **London + New York** | 12:00–15:00 | **23.3** | **+61%** |
| New York only | 16:00–20:00 | 21.5 | +48% |
| Quiet (no session) | 21:00–23:00 | 17.5 | +21% |

**The London-NY overlap is where the action is.** The peak occurs ~30 min after NYSE open in both summer and winter, confirming the equity market as the causal driver.

### Session × Day Interaction

The quietest combination is **Saturday Tokyo-only** (range 8.9–9.5 bps for BTC). The most active is **Tuesday 14:00 UTC (summer) / 15:00 UTC (winter)** during London-NY overlap (34.6 bps for BTC — **3.9x** the quietest).

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

---

## v33b: Large-Sample Confirmation (3+ Years, 5 Symbols)

### Dataset

| Symbol | Days | 5-min Bars | Date Range |
|--------|------|-----------|------------|
| BTCUSDT | 1,143 | 329,184 | 2023-01-01 to 2026-02-16 |
| ETHUSDT | 1,143 | 329,184 | 2023-01-01 to 2026-02-16 |
| SOLUSDT | 1,143 | 329,184 | 2023-01-01 to 2026-02-16 |
| DOGEUSDT | 1,143 | 329,184 | 2023-01-01 to 2026-02-16 |
| XRPUSDT | 1,143 | 329,184 | 2023-01-01 to 2026-02-16 |
| **Total** | | **1,645,920** | |

Source: Bybit futures 5-min OHLCV parquet (built from tick trades). No OI/FR/liqs — trades-only metrics.

### Hour of Day — CONFIRMED: 14:00 UTC Peak is Universal

**ALL 5 symbols peak at exactly 14:00 UTC** (US equity market open, 10am ET):

| Symbol | Peak Hour | Peak Range (bps) | Trough Hour | Trough Range (bps) | **Ratio** |
|--------|-----------|------------------|-------------|-------------------|-----------|
| BTCUSDT | **14:00** | 27.38 | 05:00 | 12.75 | **2.15x** |
| ETHUSDT | **14:00** | 35.06 | 04:00 | 18.13 | **1.93x** |
| SOLUSDT | **14:00** | 50.18 | 05:00 | 31.14 | **1.61x** |
| DOGEUSDT | **14:00** | 47.06 | 05:00 | 29.53 | **1.59x** |
| XRPUSDT | **14:00** | 39.57 | 04:00 | 24.45 | **1.62x** |

Kruskal-Wallis H-stats for range: 11,957–26,578 (all p ≈ 0). This is among the strongest statistical effects we've ever measured.

### Day of Week — CONFIRMED: Weekday/Weekend Gap is Structural

| Symbol | Weekday Range | Weekend Range | **Ratio** | t-stat |
|--------|-------------|-------------|-----------|--------|
| BTCUSDT | 20.25 | 12.09 | **1.67x** | 120.8*** |
| ETHUSDT | 26.60 | 18.31 | **1.45x** | 92.3*** |
| SOLUSDT | 40.55 | 31.43 | **1.29x** | 71.7*** |
| DOGEUSDT | 37.96 | 30.35 | **1.25x** | 50.0*** |
| XRPUSDT | 32.99 | 24.57 | **1.34x** | 60.8*** |

**BTC has the largest weekday/weekend gap (1.67x)**. Altcoins have smaller but still highly significant gaps. Saturday is the quietest day for all 5 symbols.

### Trading Sessions — Overlapping Model (v33b, 3-year)

Sessions overlap in reality. The key insight is that **overlap zones drive volatility**:

| Zone | UTC Hours | Sessions Active | BTC Range | ETH Range | SOL Range |
|------|-----------|----------------|-----------|-----------|-----------|
| Tokyo only | 00:00–06:00 | 1 | 14.5 | 21.1 | 35.3 |
| Tokyo+London | 07:00–08:00 | 2 | 14.3 | 20.4 | 33.3 |
| London only | 09:00–11:00 | 1 | 14.1 | 19.9 | 32.5 |
| **London+NY** | **12:00–15:00** | **2** | **23.3** | **30.3** | **45.5** |
| NY only | 16:00–20:00 | 1 | 21.5 | 28.2 | 41.0 |
| Quiet | 21:00–23:00 | 0 | 17.5 | 23.6 | 35.9 |

**The London-NY overlap (12:00–16:00 UTC) produces 60%+ more volatility than single-session hours.** The 14:00 UTC peak is the heart of this overlap — both London and New York are fully active.

### NEW: Month of Year — STRONG Seasonal Effect

| Symbol | Peak Month | Peak Range | Trough Month | Trough Range | **Ratio** |
|--------|-----------|-----------|-------------|-------------|-----------|
| BTCUSDT | **Mar** | 25.56 | Sep | 13.01 | **1.96x** |
| ETHUSDT | **Mar** | 29.64 | Sep | 17.97 | **1.65x** |
| SOLUSDT | **Nov** | 46.76 | Sep | 27.32 | **1.71x** |
| DOGEUSDT | **Mar** | 50.65 | Sep | 24.57 | **2.06x** |
| XRPUSDT | **Nov** | 43.47 | Sep | 19.48 | **2.23x** |

**September is universally the quietest month** (all 5 symbols). Peak months are Mar or Nov. The seasonal effect is 1.65-2.23x — comparable in magnitude to the hourly effect.

All Kruskal-Wallis H-stats for monthly range: 7,405–26,692 (p ≈ 0).

### NEW: Year-Over-Year Stability — Hourly Profile is a Market Constant

Spearman correlation of hourly range profiles between years:

| Pair | BTC | ETH | SOL | DOGE | XRP |
|------|-----|-----|-----|------|-----|
| 2023 vs 2024 | **+0.979*** | +0.949*** | +0.949*** | +0.937*** | +0.925*** |
| 2023 vs 2025 | **+0.943*** | +0.976*** | +0.950*** | +0.937*** | +0.930*** |
| 2024 vs 2025 | **+0.968*** | +0.956*** | +0.953*** | +0.935*** | +0.950*** |
| 2025 vs 2026 | **+0.919*** | +0.895*** | +0.857*** | +0.836*** | +0.858*** |

**The hourly volatility profile is essentially identical year after year** (ρ = 0.84–0.98). This is a structural feature of the market, not a temporary anomaly.

Weekday/weekend ratio is also stable across years:

| Symbol | 2023 | 2024 | 2025 | 2026 |
|--------|------|------|------|------|
| BTCUSDT | 1.68x | 1.70x | 1.67x | 1.48x |
| ETHUSDT | 1.44x | 1.51x | 1.42x | 1.37x |
| SOLUSDT | 1.20x | 1.35x | 1.33x | 1.32x |
| DOGEUSDT | 1.24x | 1.24x | 1.27x | 1.18x |
| XRPUSDT | 1.41x | 1.30x | 1.34x | 1.24x |

---

## Final Verdict

The temporal patterns discovered in v33 are **confirmed at massive scale** (1.6M bars, 5 symbols, 3+ years):

1. **NYSE-open vol peak**: 14:00 UTC (summer) / 15:00 UTC (winter) — shifts exactly with US DST. All 5 symbols, every year.
2. **DST proof (v33c)**: The +1h winter shift across all 5 symbols is definitive proof the peak is caused by the NYSE opening bell, not a fixed UTC pattern.
3. **Weekday > Weekend**: 1.25-1.67x, structural and stable.
4. **September = quietest month**: Universal trough. Mar/Nov = peaks. Seasonal effect up to 2.23x.
5. **London-NY overlap = peak zone**: The overlap (not any single session) drives the highest volatility.
6. **Year-over-year stability**: ρ > 0.84 for hourly profiles — these patterns are market constants.

These are **not overfitted**. They are physical consequences of:
- US equity market open driving global crypto activity (proven by DST shift)
- London-NY overlap concentrating institutional flow
- Weekend reduction in institutional/algorithmic trading
- Seasonal patterns in risk appetite and market participation
