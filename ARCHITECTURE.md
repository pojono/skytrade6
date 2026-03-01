# Architecture — Settlement Trading Pipeline v4

**Last updated:** 2026-03-01

---

## Overview

A modular Python pipeline that exploits funding-rate settlement microstructure on Bybit perpetual futures. The system detects the post-settlement price crash, shorts into it, then optionally goes long to catch the recovery bounce.

```
┌──────────────────────────────────────────────────────────────┐
│                     pipeline/run.py                          │
│              CLI orchestrator (6 steps)                       │
├──────────┬───────────┬───────────┬───────────┬───────────────┤
│ data.py  │features.py│ models.py │backtest.py│  report.py    │
│ parse +  │ tick-level│ train +   │ simulate  │  markdown     │
│ download │ features  │ persist   │ per-settle│  generation   │
├──────────┴───────────┴───────────┴───────────┴───────────────┤
│                      config.py                                │
│          constants, sizing functions, paths                    │
└──────────────────────────────────────────────────────────────┘
        ▲                                         ▲
        │                                         │
  research_position_sizing.py              models/*.pkl
  (compute_slippage_bps)                   (saved ML models)
```

---

## Directory Structure

```
skytrade6/
├── pipeline/                    # Core pipeline package
│   ├── __init__.py
│   ├── config.py                # All constants, thresholds, sizing
│   ├── data.py                  # JSONL parser, SettlementData, OB reconstruction
│   ├── features.py              # Short exit + long exit tick features
│   ├── models.py                # ML training, persistence, inference
│   ├── backtest.py              # Per-settlement simulation, strategy comparison
│   ├── report.py                # Markdown report generator
│   └── run.py                   # CLI entry point
├── models/                      # Persisted ML models
│   ├── short_exit_logreg.pkl    # Short exit LogReg (trained, NOT used in backtest)
│   ├── short_exit_hgbc.pkl      # Short exit HGBC (trained, NOT used in backtest)
│   └── long_exit_logreg.pkl     # Long exit LogReg (ACTIVE — drives exit timing)
├── charts_settlement/           # Raw JSONL data (one file per settlement)
├── research_position_sizing.py  # Slippage computation (external dependency)
├── AUDIT_ML_SYSTEM.md           # Full audit of ML system (2026-03-01)
├── REPORT_ml_settlement.md      # Auto-generated backtest report
└── ARCHITECTURE.md              # This file
```

---

## Pipeline Steps

Run: `python3 -m pipeline.run [--skip-download] [--skip-training] [--backtest-only]`

| Step | Module | What it does |
|------|--------|-------------|
| 1. Download | `data.py` | SCP new JSONL files from remote server |
| 2. Parse | `data.py` | Parse all JSONL into `SettlementData` objects (single pass) |
| 3. Train short exit | `models.py` | Train LogReg + HGBC on `target_near_bottom_10` |
| 4. Train long exit | `models.py` | Train LogReg on `target_near_peak_10` |
| 5. Backtest | `backtest.py` | Run 3 strategies: short-only, fixed exit, ML exit |
| 6. Report | `report.py` | Generate `REPORT_ml_settlement.md` |

---

## Data Layer (`data.py`)

### SettlementData

Each JSONL file is parsed once into a `SettlementData` dataclass containing:

| Field Group | Fields | Source |
|-------------|--------|--------|
| **Pre-settlement** | `ref_price`, `fr_bps`, `pre_vol_rate`, `pre_trade_rate`, `pre_spread_bps` | Last pre-T0 trades/tickers/ob1 |
| **T-0 orderbook** | `bids`, `asks`, `mid_price`, `spread_bps`, `depth_20` | Last OB.200 snapshot before T=0 |
| **Post trades** | `post_times`, `post_prices`, `post_prices_bps`, `post_sides`, `post_notionals`, `post_sizes` | publicTrade messages, t≥0 |
| **Post OB.1** | `ob1_times`, `ob1_bids`, `ob1_asks` | orderbook.1 L1 updates, t≥0 |
| **Post OB.200** | `ob200_deltas` | orderbook.200 delta updates (~100ms, ~500+/file) |
| **Tickers** | `tickers` | Funding rate snapshots |
| **Derived** | `passes_filters`, `price_bins` | Computed from above |

### OB Reconstruction (`reconstruct_ob_at`)

Rebuilds the full orderbook at any post-settlement time by applying OB.200 delta updates to the T-0 snapshot:

```
T-0 snapshot → apply deltas[0..t] → full book at time t
```

Returns `bids`, `asks`, `mid_price`, `spread_bps`, `depth_20` at the requested time. Used for long leg sizing and slippage.

### Filters

A settlement is skipped if:
- `depth_20 < $2,000` (not enough liquidity)
- `spread_bps > 8` (too expensive to trade)

---

## ML Models (`models.py`, `features.py`)

### Model 1: Short Exit (trained but not used in backtest)

| Property | Value |
|----------|-------|
| **Target** | `target_near_bottom_10` — is price within 10 bps of eventual crash minimum? |
| **Features** | 56 tick-level features (100ms resolution) |
| **Categories** | Price velocity/acceleration, trade flow, OB state, sequence features, static context |
| **Models** | LogReg (C=0.1), HGBC (max_depth=6, 300 iters) |
| **Validation** | 70/30 split (alphabetical by symbol) + LOSO |
| **Status** | ⚠️ Trained and saved but not used in backtest — the 23.6 bps LOSO average is used as a constant |

### Model 2: Long Entry Decision (rule-based)

| Property | Value |
|----------|-------|
| **Type** | Simple rule, not ML |
| **Rule** | `bottom_t ≤ 15 seconds` |
| **Rationale** | Early bottoms (≤15s) have 73% WR vs 41% for late |
| **Status** | ⚠️ Uses `find_bottom()` which has look-ahead bias (see audit) |

### Model 3: Long Exit (`long_exit_logreg`) — ACTIVE

| Property | Value |
|----------|-------|
| **Target** | `target_near_peak_10` — is price within 10 bps of future recovery max? |
| **Features** | 28 recovery tick features (100ms resolution) |
| **Categories** | Recovery dynamics, velocity, buy/sell pressure, OB state, static context |
| **Model** | LogReg (C=0.1) via Pipeline (Imputer→Scaler→LR) |
| **Threshold** | p(near_peak) ≥ 0.6 → exit signal |
| **Precision** | 75.4% at threshold 0.6 |
| **Fallback** | Fixed +20s hold if ML never fires |
| **Status** | ✅ Active — loaded at backtest time, drives long exit timing |

### Persistence

Models are saved as pickle files in `models/`:
```python
{'model': <sklearn Pipeline or classifier>, 'feature_cols': [...]}
```

`--backtest-only` and `--skip-training` modes load from disk.

---

## Backtest Engine (`backtest.py`)

### Per-Settlement Simulation

```
simulate_settlement(sd, long_exit_model, ...)
  │
  ├── SHORT LEG (always, if passes filters)
  │   ├── Sizing: compute_notional(depth_20) from T-0 OB
  │   ├── Entry: taker sell at settlement
  │   ├── Exit: limit buy + taker rescue (blended fee)
  │   ├── PnL: 23.6 bps gross - RT slippage + fee savings
  │   └── Slippage: walk T-0 OB (asks for entry, bids for exit)
  │
  ├── BOTTOM DETECTION
  │   └── find_bottom(): scan price_bins 1s–30s for minimum
  │
  ├── LONG ENTRY DECISION
  │   └── should_go_long(): bottom_t ≤ 15s?
  │
  └── LONG LEG (conditional)
      ├── OB Reconstruction: reconstruct_ob_at(sd, bottom_t)
      ├── Sizing: compute_long_notional(entry_ob['depth_20'])
      ├── Entry: buy at bottom price (using entry-time OB for slippage)
      ├── Exit: ML threshold 0.6, or fixed +20s fallback
      ├── Exit slippage: reconstruct_ob_at(sd, exit_t)
      └── PnL: recovery_bps - fees - entry_slip - exit_slip
```

### Strategy Comparison

Three variants are compared:
1. **short_only** — no long leg
2. **fixed_exit** — long leg with +20s fixed hold
3. **ml_exit** — long leg with LogReg exit signal

### Position Sizing

```python
# Short leg
tiers = [500, 1000, 2000, 3000]
cap = 15% of depth_20 (at T-0)

# Long leg (independent)
tiers = [250, 500, 750, 1000, 1500]
cap = 15% of depth_20 (at bottom_t, from reconstructed OB)
```

### Cost Model

| Component | Short Leg | Long Leg |
|-----------|-----------|----------|
| **Entry fee** | 10 bps (taker) | Blended: 54% maker + 46% taker |
| **Exit fee** | Blended (54% maker) | Blended (54% maker) |
| **Entry slippage** | Walk T-0 asks | Walk entry-time asks |
| **Exit slippage** | Walk T-0 bids | Walk exit-time bids |
| **Spread** | Implicit in slippage vs mid | Implicit in slippage vs mid |

---

## Data Flow

```
Remote server (skytrade7)
    │
    │ SCP download
    ▼
charts_settlement/*.jsonl        ← Raw JSONL (162 files)
    │
    │ parse_settlement()
    ▼
List[SettlementData]             ← Parsed once, shared everywhere
    │
    ├──► build_short_exit_ticks()  → train_short_exit()  → models/short_exit_*.pkl
    │
    ├──► build_long_exit_ticks()   → train_long_exit()   → models/long_exit_logreg.pkl
    │
    ├──► simulate_settlement()     → TradeResult per settlement
    │       uses: reconstruct_ob_at(), compute_slippage_bps(),
    │             predict_long_exit(), should_go_long()
    │
    └──► compare_strategies()      → generate_report()   → REPORT_ml_settlement.md
```

---

## External Dependencies

| Dependency | Used for |
|-----------|----------|
| `research_position_sizing.py` | `compute_slippage_bps()` — OB depth-walking |
| `research_position_sizing.py` | `parse_last_ob_before_settlement()` — T-0 OB snapshot |
| `sklearn` | LogisticRegression, HGBC, Pipeline, LOSO |
| `numpy`, `pandas` | Data manipulation |

---

## Configuration (`config.py`)

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TAKER_FEE_BPS` | 10 | Per-leg taker fee |
| `MAKER_FEE_BPS` | 4 | Per-leg maker fee |
| `LIMIT_FILL_RATE` | 0.54 | Probability limit order fills |
| `GROSS_PNL_BPS` | 23.6 | Short leg average edge (LOSO) |
| `MIN_DEPTH_20` | $2,000 | Minimum bid depth for trading |
| `MAX_SPREAD_BPS` | 8 | Maximum spread for trading |
| `LONG_ENTRY_MAX_T_S` | 15s | Bottom time cutoff for long entry |
| `LONG_EXIT_ML_THRESHOLD` | 0.6 | LogReg probability threshold |
| `LONG_HOLD_FIXED_MS` | 20,000 | Fixed hold fallback |
| `CAP_PCT` | 0.15 | Position cap as % of depth_20 |
| `TICK_MS` | 100 | Feature tick resolution |

---

## Current Performance (4 days, 160 settlements)

| Strategy | Short $/day | Long $/day | **Total $/day** | Long WR |
|----------|------------|-----------|----------------|---------|
| short_only | $72.5 | — | $72.5 | — |
| fixed_exit | $72.5 | $42.1 | $114.5 | 55% |
| **ml_exit** | **$72.5** | **$53.1** | **$125.6** | **60%** |

---

## Known Issues (see AUDIT_ML_SYSTEM.md)

| Issue | Severity | Description |
|-------|----------|-------------|
| Look-ahead bottom detection | 🔴 Critical | `find_bottom()` uses future prices |
| Short exit model unused | 🔴 Critical | Trained but never called in backtest |
| 4-day sample size | 🔴 Critical | Insufficient for reliable estimates |
| Alphabetical split | 🟡 Misleading | Not a temporal split |
| Constant short PnL | 🟡 Simplification | No per-trade variance |
| Last-trade price bins | 🟡 Optimistic | Not VWAP |
