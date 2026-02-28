# Event-Driven vs Polling Exit ML

**Date:** 2026-02-28  
**Data:** 139 settlements, 141 JSONL recordings  
**Models:** LogReg (55 features), HGBC (55 features), trained on 100ms ticks  

---

## Architecture

### Polling (current)
Every 100ms, compute features from all data up to now, run model.

### Event-Driven
Maintain streaming state incrementally (O(1) per event). Evaluate model on **triggers**:
- **BOUNCE** — price bounced >3 bps off running minimum
- **BIG_TRADE** — single trade >2x rolling median size
- **NEW_LOW** — price made new running minimum
- **COOLDOWN** — 100ms since last evaluation (fallback)

---

## Results: LogReg

| Mode | N | Avg PnL | Med PnL | WR | Avg Exit | Evals/settle |
|------|---|---------|---------|-----|----------|-------------|
| **Polling 100ms** | 139 | **+35.0** | +13.0 | 63% | 15.6s | 68 |
| **Event-driven** | 139 | +33.0 | **+14.9** | **67%** | **9.9s** | 624 |
| Every trade | 139 | +31.1 | +14.1 | 67% | 9.4s | 609 |

## Results: HGBC

| Mode | N | Avg PnL | WR | Avg Exit | Evals/settle |
|------|---|---------|-----|----------|-------------|
| **Polling 100ms** | 139 | **+62.2** | **81%** | 21.4s | 87 |
| **Event-driven** | 139 | +59.7 | 77% | 16.5s | 1,278 |

---

## Key Findings

### 1. Polling wins on average PnL
- LogReg: polling +35.0 vs event-driven +33.0 (**-2.0 bps**)
- HGBC: polling +62.2 vs event-driven +59.7 (**-2.5 bps**)
- The model was TRAINED on 100ms snapshots — running it between ticks creates distribution mismatch

### 2. Event-driven exits much earlier
- LogReg: 9.9s vs 15.6s (5.7s earlier)
- HGBC: 16.5s vs 21.4s (4.9s earlier)
- 45% of settlements exit EARLIER with event-driven, only 1% later, 55% same
- Earlier exit = less capital at risk, but catches false bounces

### 3. Event-driven has higher win rate but lower average
- LogReg WR: 67% (event) vs 63% (poll)
- More frequent triggers catch small bounces correctly (win) but miss bigger moves (lower avg)

### 4. Trigger quality varies enormously

| Trigger | Exits | Avg PnL | Win Rate |
|---------|-------|---------|----------|
| **BIG_TRADE** | 30% | **+47.7** | **79%** |
| BOUNCE | 55% | +28.0 | 66% |
| NEW_LOW | 5% | +3.8 | 57% |
| COOLDOWN | 9% | **-4.2** | **38%** |

**BIG_TRADE is the killer trigger:** When a large trade hits during a bounce, it's a strong confirmation signal (+47.7 bps, 79% WR).

COOLDOWN exits are net negative — the model is unreliable when nothing interesting is happening.

### 5. 10x more model evaluations
- Event-driven: 624 evals/settlement (LogReg) / 1,278 (HGBC)
- Polling: 68 evals (LogReg) / 87 (HGBC)
- LogReg handles this fine (<0.01ms per eval)
- HGBC is too slow for event-driven (30min simulation vs 2min polling)

---

## Recommendation

### For production: Hybrid approach

1. **Use polling 100ms as the base** — matches training data, higher PnL
2. **Add BIG_TRADE trigger** — only evaluate model on large trades (>2x median)
   - This catches the +47.7 bps opportunity without the noise of BOUNCE/COOLDOWN
   - Only ~30% more evaluations (not 10x)
3. **Do NOT use pure event-driven** — the train/inference distribution mismatch costs 2-3 bps

### To unlock full event-driven potential
- Retrain the model ON event-driven features (variable intervals)
- Add time-since-last-eval as a feature
- Need 500+ settlements to avoid overfitting on the richer feature space

---

## Technical Notes

- `StreamingState` class maintains all state in O(1) per event using deques
- Rolling windows (500ms, 1s, 2s, 5s) maintained incrementally
- Trigger detection is zero-cost (checked on each trade, no extra computation)
- Full simulation: LogReg 2min, HGBC 30min (HGBC too slow for event-driven production)
