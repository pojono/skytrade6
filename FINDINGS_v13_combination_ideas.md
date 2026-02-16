# Research Findings v13 — Combining v3 Directional Signals with v9-12 ML Predictions

**Date:** 2026-02-16
**Symbols:** BTCUSDT (primary), ETHUSDT, SOLUSDT (validation)
**Period:** 2025-12-01 → 2025-12-30 (30 days)
**Method:** Walk-forward Ridge vol prediction + v3 signal backtesting
**Runtime:** ~1 min per symbol

---

## Hypothesis

v3 has **direction** (which way to trade), v9-12 has **magnitude** (how much vol/range to expect). Combining them should produce better risk-adjusted returns than either alone.

## Ideas Tested

| Idea | Concept | Result |
|------|---------|--------|
| **#1 Vol-adaptive sizing** | Scale position by target_vol / predicted_vol | Mixed — helps momentum, hurts contrarian |
| **#2 Vol-regime filter** | Only trade in specific vol regimes | One BTC-specific win, not robust |
| **#3 Dynamic TP from range** | Exit early at predicted range P50/P75 | **Complete failure** — all worse |
| **#4 Breakout filter** | Only trade when breakout likely/unlikely | Marginal — rescued one losing signal |
| **#6 Contrarian + compression** | Fade retail during vol compression | Doesn't work |

---

## Idea #1: Vol-Adaptive Position Sizing

### BTCUSDT Results (30 days)

| Signal | Direction | Fixed Avg | VolAdapt Avg | Δ PnL | Δ Sharpe |
|--------|-----------|----------|-------------|-------|----------|
| E01 (contrarian) t=1.0 | Contrarian | **+13.68** | +9.73 | **-3.94** | -0.057 |
| E03 (vol breakout) t=1.5 | Momentum | +9.91 | **+12.61** | **+2.70** | +0.047 |
| E09 (cum imbalance) t=1.0 | Momentum | +9.39 | **+10.91** | **+1.52** | -0.013 |
| E06 (vol surge) t=1.0 | Momentum | +7.68 | **+15.32** | **+7.64** | +0.064 |

### Interpretation

Vol-adaptive sizing **helps momentum signals** (+1.5 to +7.6 bps improvement) but **hurts the contrarian signal** (-3.9 bps). This makes sense:

- **Momentum signals** benefit from sizing up during high vol because high vol = the trend is strong
- **Contrarian signals** get hurt because high vol = you're fading a strong move, and sizing up amplifies the loss

**Verdict: Use vol-sizing ONLY for momentum signals (E03, E09, E06). Never for contrarian (E01).**

---

## Idea #2: Vol-Regime Filtered Signals

### E09 (Cum Imbalance Momentum) — Cross-Symbol

| Symbol | Baseline | Low-vol | High-vol | Rising-vol |
|--------|---------|---------|----------|-----------|
| **BTCUSDT** | +9.39 | +2.86 | -12.86 | **+14.64** |
| **ETHUSDT** | +2.76 | -12.90 | — | -1.60 |
| **SOLUSDT** | +19.95 | +8.50 | — | +16.76 |

### E01 (Contrarian) — BTCUSDT

| Filter | Trades | Avg PnL | WR |
|--------|--------|---------|-----|
| All | 161 | +13.68 | 51% |
| **Low-vol** | 93 | **+11.28** | **62%** |
| High-vol | 92 | -11.85 | 48% |

### Interpretation

- **E09 + Rising-vol** looked great on BTC (+55% improvement) but **failed on ETH and SOL**. Not robust — likely BTC-specific noise in Dec 2025.
- **E01 contrarian works best in low-vol** (WR jumps from 51% to 62%) — consistent with the theory that retail flow is "dumb money" in calm markets. But the avg PnL actually drops slightly, so the higher WR is offset by smaller moves.
- **No vol filter consistently improves signals across all symbols.**

**Verdict: Vol-regime filtering does not reliably improve signals. The baseline signals are already capturing the right regime mix.**

---

## Idea #3: Dynamic Take-Profit — COMPLETE FAILURE

### BTCUSDT Results

| Signal | Fixed 4h | TP@P50 | TP@P75 | TP@P100 |
|--------|---------|--------|--------|---------|
| E01 | **+13.68** | -20.44 | -23.43 | -24.67 |
| E03 | **+9.91** | -28.74 | -26.28 | -19.16 |
| E09 | **+9.39** | -23.79 | -20.77 | -13.53 |

**Every single TP variant made things dramatically worse** (20-40 bps worse).

### Why It Failed

1. **TP exits lock in small gains but miss big winners.** The v3 signals are profitable because of a few large winning trades that more than offset many small losses. TP cuts off the tail.
2. **TP creates more trades** (exits free up capital for re-entry), and the additional trades are worse quality.
3. **The predicted range is a median/average — but profitable trades need the price to move BEYOND average.** Setting TP at the average range guarantees you exit before the signal fully plays out.

**Verdict: Fixed 4h holding period is already optimal. Do NOT use dynamic TP with these signals.**

---

## Idea #4: Breakout Confirmation Filter

### BTCUSDT Results

| Signal | Baseline | BO Likely | BO Unlikely |
|--------|---------|-----------|-------------|
| E01 (contrarian) | **+13.68** | -5.74 | +1.82 |
| E03 (momentum) | **+9.91** | +5.72 | -17.47 |
| E09 (momentum) | **+9.39** | +1.81 | +1.86 |
| E06 (vol surge) | -21.06 | **+0.98** | -56.69 |

### Interpretation

- The breakout filter **rescued E06** from -21 bps to +1 bps — but +1 bps is barely profitable after fees.
- For already-winning signals (E01, E03, E09), the filter **made things worse** by reducing trade count and removing some good trades.
- The filter is too blunt — vol_ratio as a breakout proxy doesn't capture the nuance needed.

**Verdict: Breakout filter adds no value to winning signals. Skip.**

---

## Idea #6: Contrarian + Vol Compression

### BTCUSDT Results

| Filter | Trades | Avg PnL | Δ vs Baseline |
|--------|--------|---------|--------------|
| Baseline | 161 | +13.68 | — |
| Strong compress (<0.6) | 101 | -7.58 | -21.26 |
| Mild compress (<0.8) | 130 | +5.89 | -7.79 |
| Expanding (>1.2) | 68 | -4.57 | -18.25 |

**All compression filters made E01 worse.** The hypothesis that "compressed range + retail imbalance = snap back" doesn't hold in the data.

**Verdict: Doesn't work. The contrarian signal works across all vol regimes, not specifically during compression.**

---

## Summary: What Actually Works

### Worth Keeping

| Enhancement | Signal | Improvement | Robust? |
|------------|--------|------------|---------|
| **Vol-adaptive sizing for momentum** | E03, E09, E06 | +1.5 to +7.6 bps | Needs multi-symbol validation |

### Doesn't Work

| Enhancement | Why |
|------------|-----|
| Dynamic TP from predicted range | Cuts off winning tail, creates worse re-entries |
| Vol-regime filtering | Not robust across symbols |
| Breakout confirmation filter | Too blunt, removes good trades |
| Contrarian + compression | Hypothesis doesn't hold |
| Vol-adaptive sizing for contrarian | Amplifies losses when fading strong moves |

---

## Key Lesson

**The v3 directional signals and v9-12 vol predictions are largely independent systems.** They capture different aspects of the market:

- v3 signals detect **microstructure flow** (who is buying/selling, how aggressively)
- v9-12 models detect **volatility regime** (how much the market is moving)

Combining them is harder than expected because:

1. **The signals don't need vol information to work.** E09 already implicitly captures vol through cumulative imbalance — high vol periods naturally produce larger imbalances.
2. **Vol prediction helps with sizing but not with timing.** Knowing vol is high doesn't tell you WHEN to enter — the v3 signal already handles that.
3. **Dynamic exits destroy the signal.** The 4h fixed hold was found through optimization in v3 — it's already the right answer.

**The one genuine synergy is vol-adaptive sizing for momentum signals** — but even this needs validation on more data before trusting it.

---

## Skipped Ideas

- **#5 Full Stack Trade Chain:** Skipped because individual components (#2, #3, #4) didn't add value. Stacking non-improvements won't help.
- **#7 Grid Bot + Directional Bias:** Still worth exploring separately as it's a fundamentally different architecture (continuous grid vs discrete signals).

---

## Files

| File | Description |
|------|-------------|
| `combo_ideas.py` | Implementation of ideas #1-#4, #6 |
| `results/combo_ideas_BTC_30d.txt` | BTCUSDT full results |
| `results/combo_ideas_ETH_30d.txt` | ETHUSDT idea #2 validation |
| `results/combo_ideas_SOL_30d.txt` | SOLUSDT idea #2 validation |
| `IDEAS_v3_plus_ml.md` | Original idea descriptions |
