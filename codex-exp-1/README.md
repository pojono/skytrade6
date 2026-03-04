# Codex Exp 1

This workspace is a focused cross-exchange research sandbox built on the local `datalake` dataset.

Objective: find repeatable, cross-exchange patterns across 90+ shared USDT perpetual symbols that can survive realistic costs and become a robust trading strategy.

Constraint: no claims of guaranteed profitability. The bar is positive expectancy after fees, slippage, missing data, and out-of-sample validation.

## Scope

- Data source: local `datalake/binance/*` and `datalake/bybit/*`
- Universe: symbols present on both exchanges with at least 90 overlapping daily files
- Primary horizon: 1-minute bars
- First hypothesis: short-horizon spread dislocations between Binance and Bybit mean revert

## Files

- `research_plan.md`: experiment phases, hypotheses, and validation rules
- `universe_scan.py`: discovers shared symbols and overlap quality
- `cross_exchange_edge_scan.py`: tests a first spread-reversion edge
- `out/`: output directory for generated CSV and markdown summaries

## Quick Start

Build the eligible universe:

```bash
python3 codex-exp-1/universe_scan.py --min-overlap-days 90
```

Run the first spread-edge scan:

```bash
python3 codex-exp-1/cross_exchange_edge_scan.py \
  --min-overlap-days 90 \
  --min-signal-bps 8 \
  --max-symbols 120 \
  --test-days 30 \
  --fee-bps-roundtrip 6
```

Test net performance under a fee assumption:

```bash
python3 codex-exp-1/cross_exchange_edge_scan.py \
  --min-overlap-days 90 \
  --min-signal-bps 8 \
  --fee-bps-roundtrip 6
```

Run a fast recent-history pilot sweep before scaling:

```bash
python3 codex-exp-1/pilot_iterate.py \
  --pilot-symbols 8 \
  --recent-days 60 \
  --signal-grid 4,8,12 \
  --fee-grid 0,2,4,6
```

## Output

The scripts write into `codex-exp-1/out/`:

- `universe_summary.csv`
- `eligible_symbols.txt`
- `cross_exchange_edge_summary.csv`
- `cross_exchange_edge_report.md`

## If More Data Is Needed

The local `datalake` already covers 116 common symbols. If you want to extend coverage, use:

- `datalake/download_binance_data.py`
- `datalake/download_bybit_data.py`

Refer to `datalake/README.md` for supported data types and date ranges.
