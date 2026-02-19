# Liquidation Strategy Research Index

**Master reference for all liquidation cascade strategy research.**  
**Period:** Feb 2026 | **Data:** 282 days (2025-05-11 to 2026-02-17) | **Symbols:** DOGE, SOL, ETH, XRP

---

## Research Timeline & Evolution

```
v26  → Initial liquidation data exploration
v26b → Basic strategy ideas (momentum, fade, cascade detection)
v26c → Full strategy sweep (multiple approaches)
v26d → Cascade MM with limit orders — FIRST PROFITABLE STRATEGY (0% fees)
v26e → Fee-aware sweep — strategy survives real fees (thin edge)
v26f → R:R optimization — no-SL is optimal, +66.7% combined
v26g → Microstructure analysis — bounce curves, displacement, cascade anatomy
v26h → Big-move market-order fade — FAILED (fees eat entire edge)
v26i → Big-move limit-order + filters — +58.9% combined, 95.8% WR
v26j → Integrated strategy with ALL research filters — 85% positive months

Supporting research (from other experiment streams):
v27  → Regime detection + liquidation as leading indicator
v33  → Temporal patterns (hours, days, sessions, funding cycles)
v41  → Walk-forward OOS validation (all 3 symbols survive)
v42  → New signals (cascade size, seasonality, contagion, direction asymmetry)
```

---

## Findings Files

### Core Strategy Development (v26 series)

| File | Version | What It Covers | Key Result |
|------|---------|---------------|------------|
| `FINDINGS_v26_liquidations.md` | v26 | Initial liquidation data exploration | Liquidation patterns identified |
| `FINDINGS_v26b_liquidation_strategies.md` | v26b | Basic strategy ideas | Fade > momentum |
| `FINDINGS_v26c_liquidation_strategies_full.md` | v26c | Full strategy sweep | Cascade detection works |
| `FINDINGS_v26d_cascade_mm.md` | v26d | Cascade MM with limit orders (0% fees) | **+35-57% per symbol, 85-92% WR** |
| `FINDINGS_v26e_cascade_mm_fees.md` | v26e | Fee-aware sweep (maker+taker) | **All 4 symbols net positive; DOGE +15.9%** |
| `FINDINGS_v26f_cascade_mm_rr.md` | v26f | R:R optimization, no-SL discovery | **No-SL optimal; +66.7% combined** |
| `FINDINGS_v26g_liq_microstructure.md` | v26g | Microstructure: bounce curves, displacement | 65% bounce within 60min; displacement filter |
| `FINDINGS_v26i_bigmove_limit.md` | v26i | Limit-order big-move + microstructure filters | **+58.9% combined, 95.8% WR** |
| `FINDINGS_v26j_integrated_strategy.md` | v26j | ALL research filters combined | **+42.2% combined, 17/20 months positive** |

### Supporting Research

| File | Version | What It Covers | Key Result |
|------|---------|---------------|------------|
| `FINDINGS_v27_regime_liquidations.md` | v27 | Liquidations as regime leading indicator | Liqs lead by 10-600s at tick level |
| `FINDINGS_v33_temporal_patterns.md` | v33 | Hour/day/session/funding patterns | **Bad hours: 08,09,13,16 UTC; weekday > weekend** |
| `FINDINGS_v41_cascade_mm_oos.md` | v41 | Walk-forward OOS validation | **All 15 OOS tests positive, 12/12 rolling windows** |
| `FINDINGS_v42_v42f_summary.md` | v42 | New signals: cascade size, contagion, direction | **P97 > P95; LONG +2-3bps; cross-symbol contagion** |

### Trading Guide

| File | What It Covers |
|------|---------------|
| `TRADING_GUIDE_cascade_mm.md` | Actionable trading guide with exact parameters |

---

## Scripts & Results

| Script | Results File | Purpose |
|--------|-------------|---------|
| `liq_cascade_mm.py` | `results/liq_cascade_mm.txt` | Original cascade MM (0% fees) |
| `liq_cascade_mm_fees.py` | `results/liq_cascade_mm_fees.txt` | Fee-aware sweep (v26e) |
| `liq_cascade_mm_rr.py` | `results/liq_cascade_mm_rr.txt` | R:R optimization (v26f) |
| `liq_microstructure.py` | `results/liq_microstructure.txt` + `.png` charts | Microstructure analysis (v26g) |
| `liq_bigmove_strategy.py` | `results/liq_bigmove_strategy.txt` | Market-order big-move (v26h, FAILED) |
| `liq_bigmove_limit.py` | `results/liq_bigmove_limit.txt` | Limit-order big-move (v26i) |
| `liq_integrated_strategy.py` | `results/liq_integrated_strategy.txt` | Integrated strategy with all filters (v26j) |

---

## Key Discoveries (Chronological)

### 1. Cascade MM Works (v26d)
- Fade liquidation cascades with limit orders placed at offset from market price
- The entry offset IS the alpha — you only fill if cascade pushes price to your limit
- 85-92% WR at 0% fees, +35-57% per symbol over 282 days

### 2. Fees Are the Main Enemy (v26e)
- 0.02% maker fee destroys 65-87% of gross returns
- Best config: off=0.20%, TP=0.15%, SL=0.50%
- All 4 symbols still net positive but edge is thin (0.5-2.0 bps/trade)
- DOGE is best: +15.9%, Sharpe +53

### 3. No-SL Is Optimal (v26f)
- **The trick isn't wider TP — it's wider SL (or no SL)**
- Wider TP → fewer wins → more SL exits → higher avg fee → WORSE
- No-SL: 94% TP rate, 0% SL rate, 6% timeout rate
- Avg fee drops from 0.047% to 0.042% — saves 0.5 bps/trade
- Combined: +66.7% across 4 symbols (~86% annualized)

### 4. Microstructure Confirms Mean-Reversion (v26g)
- 94% of cascade events see adverse price movement
- Median adverse: 62-78 bps
- Only 6-7% bounce within 1 min, but ~65% bounce within 60 min
- Deeper displacement = lower bounce probability (point of no return)

### 5. Market Orders Can't Overcome Fees (v26h)
- Raw fade edge: +1-4 bps
- Round-trip fees: 4-7.5 bps
- **Every single market-order config is net negative**

### 6. Limit Orders + Filters Work (v26i)
- Limit order offset captures spread as alpha
- Combined filters (cascade 3+, P97 1+, disp 10+): +58.9% combined, 95.8% WR
- Universal config: off=0.15%, TP=0.15%, no SL, 60min hold

### 7. Research Filters Fix Consistency (v26j)
- **Bad hours filter (v33/v42H):** exclude 08,09,13,16 UTC
- **Long-only filter (v42g):** fade buy-side liquidations only (+2-3 bps better)
- **Displacement filter (v26g):** cascade must move price ≥10 bps
- Result: baseline 10/20 positive months → **17/20 positive months (85%)**
- Filters turned 7 losing months into winners

### 8. Strategy Survives OOS (v41)
- Walk-forward: all 15 OOS tests positive across 3 symbols
- Rolling windows: 12/12 positive (100%)
- Direction signal adds real value on ETH (+7.1%) and DOGE (+4.9%)

### 9. Cross-Symbol Contagion (v42f)
- ETH cascade → enter on SOL/DOGE: 91% WR, +5.3 bps
- Combined triggers increase return by 70%+
- 28/28 rolling 15-day windows positive

---

## Recommended Production Configs

### Conservative (with SL)
```
Entry:      Limit order at 0.15% offset, fade cascade direction
TP:         0.15% (maker fee exit)
SL:         0.50% (taker fee exit)
Max hold:   60 minutes
Filters:    Exclude hours 08,09,13,16 UTC; long-only; displacement ≥10 bps
Cooldown:   5 minutes
Symbols:    DOGE, SOL, ETH, XRP simultaneously
Expected:   +42.2% combined / 282 days, 88.5% WR, 17/20 months positive
```

### Aggressive (no SL)
```
Entry:      Limit order at 0.15% offset, fade cascade direction
TP:         0.12% (maker fee exit)
SL:         none
Max hold:   60 minutes (taker fee on timeout)
Filters:    Exclude hours 08,09,13,16 UTC; long-only; displacement ≥10 bps
Cooldown:   5 minutes
Symbols:    DOGE, SOL, ETH, XRP simultaneously
Expected:   +66.7% combined / 282 days, 91-94% WR
```

### Highest Quality (US hours only, DOGE)
```
Entry:      Limit order at 0.20% offset, fade cascade direction
TP:         0.15% (maker fee exit)
SL:         0.50% (taker fee exit)
Hours:      13-18 UTC only (US session)
Expected:   Sharpe +77, max DD 2.1%, 87% WR
```

---

## How to Reproduce

1. **Data required:** Bybit liquidation + ticker data in `data/<SYMBOL>/bybit/liquidations/` and `data/<SYMBOL>/bybit/ticker/`
2. **Run any script:** `python3 <script>.py` — results go to `results/` directory
3. **Each script is self-contained** — loads data, runs sweeps, outputs results with progress logging
4. **All results files are committed** — can compare without re-running

---

## Dead Ends (Don't Repeat)

| Approach | Why It Failed | Reference |
|----------|--------------|-----------|
| Market-order fade | Raw edge (+1-4 bps) < fees (4-7.5 bps) | v26h |
| Wider TP (0.25-0.50%) | Fewer TP hits → more SL exits → higher avg fee | v26e, v26f |
| Tight SL (0.15-0.25%) | 37-45% SL rate → massive taker fee drag | v26e, v26f |
| OI divergence | No predictive power | v42 EXP C |
| Funding rate pre-settlement | No edge | v42 EXP D |
| Whale trade detection | Zero signal at any threshold | v42 EXP N |
| Post-cascade vol expansion | All configs negative | v42 EXP M |
| Vol compression straddle | 99% WR on 30d → 0/7 on 60d (overfitting) | v42 EXP I |
