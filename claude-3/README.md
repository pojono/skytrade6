# Claude-3: Cross-Sectional Price Prediction Research

**Date:** 2026-03-06
**Goal:** Exploit the cross-exchange multi-universe dataset (131–154 symbols, 2 years) for price prediction using cross-sectional signals.

---

## Key Insight

Previous research (v1–v42) tried to predict a single coin's direction from its own microstructure — confirmed dead (AUC ≈ 0.50). This research takes a fundamentally different approach: **cross-sectional ranking** — predicting which coins will outperform others over the next N hours. Common market noise cancels out; signal-to-noise is much higher for relative returns.

---

## Documents

| File | Contents |
|------|----------|
| `RESEARCH_PLAN.md` | Original 6-signal plan with fee math |
| `PHASE1_FINDINGS.md` | IC analysis — dirty data run, then clean-data rerun |
| `PHASE2_FINDINGS.md` | Portfolio backtests, walk-forward OOS, combo comparison |
| `PHASE3_FINDINGS.md` | Execution realism: rebal freq, universe size, capacity |
| `STRATEGY_SPEC.md` | Final recommended configuration |

## Scripts

All scripts live in `research_cross_section/`:

| Script | Description |
|--------|-------------|
| `universe.txt` | 131 symbols with 400+ kline days |
| `phase1_build_signals.py` | Build per-symbol 1h signal parquets (with data quality fix) |
| `phase1_compute_ic.py` | Cross-sectional IC at 1h frequency (initial exploration) |
| `phase2_portfolio_backtest.py` | Full portfolio sim: all signals, all fee scenarios |
| `phase2b_ic_and_combos.py` | IC at correct 8h frequency + combo comparison |
| `phase3_execution.py` | Rebal frequency, universe size, vol sizing, capacity |
| `signals/*.parquet` | 131 pre-built signal files (172 MB total) |
| `results/*.csv` | All output tables |

## Result Files

| File | Contents |
|------|----------|
| `results/ic_summary.csv` | Phase 1 IC (1h, dirty data) |
| `results/phase2b_ic_8h.csv` | IC at 8h frequency (clean data) — **use this** |
| `results/phase2b_combos.csv` | 8-combo portfolio comparison |
| `results/phase2b_walkforward.csv` | Per-window OOS stats per combo |
| `results/phase3_rebal_freq.csv` | 8h vs 16h vs 24h comparison |
| `results/phase3_universe_size.csv` | N=5/10/15/20/30 per leg |
| `results/phase3_capacity.csv` | AUM vs market impact |
