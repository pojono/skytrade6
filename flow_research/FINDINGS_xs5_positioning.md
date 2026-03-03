# XS-5: Positioning Extremes Event Model — FINDINGS

**Date:** 2026-03-03
**Script:** `flow_research/xs5_positioning_events.py`
**Data:** 52 Bybit perps, 2026-01-01 → 2026-02-28 (59 days)
**Outputs:** `output/xs5/xs5_report.csv`, `xs5_events.parquet`, `xs5_trades.parquet`

---

## VERDICT: NO-GO ❌

No event type passes all GO criteria under realistic costs (20bp RT + 10bp slippage).
E1 shows promise but has only 11 trades — insufficient sample.

---

## 1) Event Detection Summary

| Event | Count | Description |
|-------|------:|-------------|
| E1 (Crowded Long → SHORT) | 11 | funding_z ≥ +2, oi_z ≥ +2, trend ≤ 0.3, ret_2h ≥ 0 |
| E2 (Crowded Short → LONG) | 94 | funding_z ≤ -2, oi_z ≥ +2, trend ≤ 0.3, ret_2h ≤ 0 |
| E4 (FR Extreme Reversion) | 572 | |funding_z| ≥ 3, oi_z ≤ 1 |
| **Total** | **677** | Across 52 symbols, 12h cooldown per sym×event |

- E1 is extremely rare (11 events in 59 days across 52 coins)
- E4 dominates (84% of all events) — extreme funding z-scores happen often
- All 52 symbols triggered at least one event
- Top symbols: FFUSDT(26), ENSOUSDT(25), BERAUSDT(24), BIOUSDT(24)

---

## 2) Results by Event Type (slip=5bp, no SL/TP — pure time stop + unwind)

### E1: Crowded Long → SHORT (N=11)
| Hold | Mean Net | Med Net | WR | PF | MFE | MAE | OOS Feb | OOS Jan |
|------|--------:|--------:|---:|---:|----:|----:|--------:|--------:|
| 4h | +44bp | +23bp | 64% | 1.42 | +580bp | -221bp | +48bp (N=10) | +5bp (N=1) |
| 12h | +211bp | +56bp | 82% | 4.30 | +725bp | -239bp | +231bp (N=10) | +5bp (N=1) |
| 24h | +270bp | +56bp | 82% | 5.22 | +934bp | -239bp | +296bp (N=10) | +5bp (N=1) |

**Very promising** — high mean, high PF, consistent OOS. But only 11 trades total (10 in Feb, 1 in Jan).
Not enough data to trust. The result is consistent with E1 being a real signal, but also consistent with lucky small sample.

### E2: Crowded Short → LONG (N=94)
| Hold | Mean Net | Med Net | WR | PF | MFE | MAE | OOS Feb | OOS Jan |
|------|--------:|--------:|---:|---:|----:|----:|--------:|--------:|
| 4h | -14bp | -30bp | 43% | 0.90 | +276bp | -232bp | -70bp (N=44) | +36bp (N=50) |
| 12h | +4bp | -19bp | 46% | 1.03 | +419bp | -320bp | -63bp (N=44) | +64bp (N=50) |
| 24h | -21bp | -24bp | 45% | 0.89 | +424bp | -350bp | -70bp (N=44) | +23bp (N=50) |

**No edge.** WR < 50%, PF ~1, huge OOS split (Jan positive, Feb negative).
The asymmetry (crowded short → long) doesn't work as expected.

### E4: FR Extreme Reversion (N=572)
| Hold | Mean Net | Med Net | WR | PF | MFE | MAE | OOS Feb | OOS Jan |
|------|--------:|--------:|---:|---:|----:|----:|--------:|--------:|
| 4h | +35bp | -28bp | 42% | 1.34 | +304bp | -203bp | +82bp (N=275) | -9bp (N=297) |
| 12h | +48bp | -32bp | 41% | 1.40 | +375bp | -251bp | +84bp (N=275) | +14bp (N=297) |
| 24h | +42bp | -31bp | 41% | 1.33 | +389bp | -265bp | +79bp (N=275) | +9bp (N=297) |

**Weak positive mean** dragged up by fat right tail (mean >> median).
WR is 41% — you lose most trades but winners are larger. OOS inconsistent (Jan near zero).
p_perm ≈ 0.06 — borderline, not significant after multiple testing.

---

## 3) SL/TP Impact

| Config | E4 4h Mean | E4 4h WR | E4 4h PF |
|--------|----------:|------:|------:|
| none (pure time+unwind) | +35bp | 42% | 1.34 |
| cat6 (SL=6×ATR) | +29bp | 35% | 1.29 |
| wide (SL=4×ATR, TP=6×ATR) | +29bp | 35% | 1.29 |
| orig (SL=2.5×ATR, TP=4×ATR) | **-30bp** | 35% | 0.59 |

- Tight SL (2.5×ATR) destroys the strategy — gets stopped out before the move materializes
- No SL/TP is best, followed by catastrophe SL (6×ATR)
- These are high-vol altcoins; 2.5 ATR SL is way too tight

---

## 4) Exit Reason Analysis (no SL/TP, slip=5bp)

For E4 (the main event type, N=572):
- **Unwind exits** (funding_z normalized): 66-98% of trades depending on hold window
- **Time stops**: 2-34% of trades
- Most trades exit via funding normalization, not time stop

This means the "unwind" condition (funding_z returns to [-1, +1]) fires very quickly.
The move hasn't had time to play out yet.

---

## 5) Key Observations

### Why E1 looks good but can't be trusted:
- Only 11 events in 59 days × 52 coins
- 10 of 11 are in February — zero statistical independence
- Could be a single market regime (Feb 2026 BTC correction drove altcoin longs underwater)

### Why E2 fails:
- "Crowded short" coins (negative funding + rising OI) often continue dropping
- The "stall" filter (trend ≤ 0.3) catches some exhaustion but not enough
- Short squeezes happen but are unpredictable in timing

### Why E4 is marginal:
- Extreme funding z-scores (≥3σ) are common on altcoins — not rare enough
- 572 events in 59 days = ~10/day = not the "rare fat signal" we wanted
- Mean is positive but median is negative — a few big winners mask many small losers
- OOS inconsistency (Feb profitable, Jan near zero) suggests regime dependence

### MFE/MAE tells the story:
- Mean MFE: +300-400bp (the move IS there)
- Mean MAE: -200-270bp (but drawdown is enormous before it arrives)
- The signal identifies the right *direction* but the *timing* is terrible
- You need to survive -200bp drawdown to capture +300bp move, with only 41% probability

---

## 6) What Would Need to Change

To make this viable:
1. **E1 needs more data** — extend to 6+ months, add more coins. If the 82% WR and +211bp persist with N≥40, it's a real signal.
2. **E4 needs a timing filter** — current trigger is too early. Adding a momentum confirmation (e.g., first 1h close against the extreme) might improve timing.
3. **Wider unwind threshold** — the [-1, +1] funding_z unwind exits too aggressively. Consider [-0.5, +0.5] or removing it for time-stop only.
4. **Consider using E4 as a regime indicator** rather than a trade signal — mark periods of extreme positioning and layer a tactical entry on top.

---

## 7) Integrity Checklist

- ✅ Uniform 1m calendar grid (84,960 points)
- ✅ OI shifted +5min for causal alignment
- ✅ FR shifted +1min for causal alignment
- ✅ Entry = next 1m close after signal (no peek)
- ✅ 12h cooldown per symbol per event type
- ✅ Z-scores computed backward-only (rolling windows)
- ✅ Walk-forward: Jan vs Feb split
- ✅ Bootstrap CI + sign-flip permutation test
- ✅ Multiple SL/TP configs tested (no cherry-picking)
- ✅ No threshold optimization — all thresholds fixed a priori from spec
