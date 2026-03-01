# Architecture — Settlement Trading Pipeline v4 (Short-Only)

**Last updated:** 2026-03-01  
**Strategy:** Short-only (long leg excluded — unprofitable without look-ahead)

---

## Overview

A modular Python pipeline that exploits funding-rate settlement microstructure on Bybit perpetual futures. The system detects the post-settlement price crash and shorts into it, using ML to time the exit near the crash bottom.

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
│   ├── short_exit_logreg.pkl    # Short exit LogReg — ACTIVE (drives exit timing + PnL)
│   ├── short_exit_hgbc.pkl      # Short exit HGBC (backup, not used in production)
│   └── long_exit_logreg.pkl     # Long exit LogReg (DEPRECATED — long leg excluded)
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
| 4. Train long exit | `models.py` | Train LogReg on `target_near_peak_10` (research only) |
| 5. Backtest | `backtest.py` | Run short-only strategy (+ long variants for research) |
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

Returns `bids`, `asks`, `mid_price`, `spread_bps`, `depth_20` at the requested time. Used for research (long leg sizing/slippage) and future strategies.

### Filters

A settlement is skipped if:
- `depth_20 < $2,000` (not enough liquidity)
- `spread_bps > 8` (too expensive to trade)

---

## ML Models (`models.py`, `features.py`)

### Production Model: Short Exit (`short_exit_logreg`)

| Property | Value |
|----------|-------|
| **Target** | `target_near_bottom_10` — is price within 10 bps of eventual crash minimum? |
| **Features** | 56 tick-level features (100ms resolution) |
| **Categories** | Price velocity/acceleration, trade flow, OB state, sequence features, static context |
| **Models** | LogReg (C=0.1), HGBC (max_depth=6, 300 iters) |
| **Threshold** | p(near_bottom) ≥ 0.4 → exit signal |
| **Timeout** | 55s forced exit if ML never fires |
| **Validation** | 70/30 split (alphabetical by symbol) + LOSO |
| **Status** | ✅ Active — drives per-trade short PnL |

### Deprecated: Long Leg Models

> **Why excluded:** The long leg was showing $53/day (60% WR) in backtest, but this relied
> on `find_bottom()` — a function that scanned all future prices for the perfect minimum
> (look-ahead bias). Once replaced with the ML-detected exit time (honest, no future data),
> the long leg collapsed to **−$17 to −$22/day (19–23% WR)**. It is unprofitable and excluded
> from production.

| Model | Status | Notes |
|-------|--------|-------|
| Long entry rule (`short_exit_t ≤ 15s`) | ❌ Excluded | Still in code for research, not used in production |
| Long exit ML (`long_exit_logreg`) | ❌ Excluded | 75.4% precision but recovery too small to cover costs |

### Persistence

Models are saved as pickle files in `models/`:
```python
{'model': <sklearn Pipeline or classifier>, 'feature_cols': [...]}
```

`--backtest-only` and `--skip-training` modes load from disk.

---

## Backtest Engine (`backtest.py`)

### Per-Settlement Simulation (Production: Short-Only)

```
simulate_settlement(sd, short_exit_model, ...)
  │
  └── SHORT LEG (if passes filters)
      ├── Sizing: compute_notional(depth_20) from T-0 OB
      ├── Entry: taker sell at settlement (T=0)
      ├── ML Exit: first tick where p(near_bottom) ≥ 0.4
      ├── Timeout: forced exit at 55s if ML never fires
      ├── PnL: (entry_bps - exit_bps) - slippage + fee savings
      └── Slippage: walk T-0 OB (asks for entry, bids for exit)
```

### Position Sizing

```python
# Short leg
tiers = [500, 1000, 2000, 3000]   # notional USD
cap = 15% of depth_20 (at T-0)
```

### Cost Model

| Component | Value |
|-----------|-------|
| **Entry fee** | 10 bps (taker) |
| **Exit fee** | Blended: 54% × 4 bps (maker) + 46% × 10 bps (taker) |
| **Entry slippage** | Walk T-0 asks |
| **Exit slippage** | Walk T-0 bids |
| **Spread** | Implicit in slippage vs mid |

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
    ├──► build_short_exit_ticks()  → train_short_exit()  → models/short_exit_logreg.pkl
    │
    ├──► simulate_settlement()     → TradeResult per settlement
    │       uses: short_exit_model, compute_slippage_bps()
    │
    └──► compare_strategies()      → generate_report()   → REPORT_ml_settlement.md
```

---

## External Dependencies

| Dependency | Used for |
|-----------|----------|
| `research_position_sizing.py` | `compute_slippage_bps()` — OB depth-walking |
| `research_position_sizing.py` | `parse_last_ob_before_settlement()` — T-0 OB snapshot |
| `sklearn` | LogisticRegression, HGBC, LOSO |
| `numpy`, `pandas` | Data manipulation |

---

## Configuration (`config.py`)

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TAKER_FEE_BPS` | 10 | Per-leg taker fee |
| `MAKER_FEE_BPS` | 4 | Per-leg maker fee |
| `LIMIT_FILL_RATE` | 0.54 | Probability limit order fills |
| `GROSS_PNL_BPS` | 23.6 | Short leg LOSO average (fallback if no ML model) |
| `SHORT_EXIT_ML_THRESHOLD` | 0.4 | p(near_bottom) threshold for short exit |
| `SHORT_EXIT_TIMEOUT_MS` | 55,000 | Forced exit if ML never fires |
| `MIN_DEPTH_20` | $2,000 | Minimum bid depth for trading |
| `MAX_SPREAD_BPS` | 8 | Maximum spread for trading |
| `CAP_PCT` | 0.15 | Position cap as % of depth_20 |
| `TICK_MS` | 100 | Feature tick resolution |

---

## Current Performance (4 days, 160 settlements)

### Production Strategy: Short-Only

| Metric | In-Sample | LOSO Conservative |
|--------|-----------|-------------------|
| **Daily revenue** | $137.7† | **$50–$75** |
| **Win rate** | 76% | ~76% |
| **Trades/day** | ~32 | ~32 |
| **Avg $/trade** | $4.34 | ~$1.60–$2.40 |

**†** In-sample: production model evaluated on its own training data. LOSO gave 23.6 bps average gross edge.

### Why Not the Long Leg?

| Metric | With look-ahead (old) | Without look-ahead (honest) |
|--------|----------------------|-----------------------------|
| Long $/day | +$53.1 | **−$17 to −$22** |
| Long WR | 60% | **19–23%** |
| Entry method | `find_bottom()` — perfect hindsight | ML exit time — no future data |

The long leg's apparent profitability was entirely due to look-ahead bias. See `AUDIT_ML_SYSTEM.md` for full analysis.

---

## Known Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Short PnL in-sample | 🟡 Inflated | Production model tested on training data |
| 4-day sample size | 🔴 Critical | Insufficient for reliable estimates |
| Alphabetical split | 🟡 Misleading | Not a temporal split |
| Last-trade price bins | 🟡 Optimistic | Not VWAP |
| No latency simulation | 🟡 Missing | Real execution adds 50–200ms delay |
