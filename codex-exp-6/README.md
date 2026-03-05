# Codex Exp 6

This workspace studies market-wide crypto structure across the local `datalake` and turns that into a fee-aware strategy research loop.

Primary objective: determine whether broad market breadth, synchronization, and cross-sectional continuation produce repeatable edge after explicit maker and taker costs.

## Scope

- Data source: local `datalake/binance/*` and `datalake/bybit/*`
- Core input: 1-minute kline closes
- First focus: shared Binance/Bybit perpetual symbols with stable overlap
- Strategy families:
  - market breadth trend
  - cross-sectional momentum
  - breadth-gated cross-sectional momentum

## Files

- `research_plan.md`: hypotheses, validation bar, and next phases
- `universe_scan.py`: shared-universe coverage scan for Binance and Bybit
- `analyze_market_structure.py`: breadth/sync analysis plus fee-aware strategy evaluation
- `breadth_trend_execution.py`: focused execution realism test for the 240m breadth setup
- `regime_filter_scan.py`: finds practical trade/no-trade filters for the 240m breadth setup
- `ml_trade_gate.py`: ML classifier for trade/no-trade gating with walk-forward validation
- `lead_lag_scan.py`: scans Binance/Bybit short-horizon lead-lag across shared symbols
- `out/`: generated CSV and markdown reports

## Quick Start

Build the eligible shared universe:

```bash
python3 codex-exp-6/universe_scan.py --min-overlap-days 90
```

Run the market-structure study on recent shared symbols:

```bash
python3 codex-exp-6/analyze_market_structure.py \
  --exchange binance \
  --min-overlap-days 90 \
  --lookback-days 30 \
  --max-symbols 60 \
  --horizons 1,5,15,60,240
```

Stress the same study under stricter cost assumptions:

```bash
python3 codex-exp-6/analyze_market_structure.py \
  --exchange binance \
  --min-overlap-days 90 \
  --lookback-days 45 \
  --maker-fee-bps 4 \
  --taker-fee-bps 10
```

Run the same study with Binance metrics filters:

```bash
python3 codex-exp-6/analyze_market_structure.py \
  --exchange binance \
  --min-overlap-days 90 \
  --lookback-days 60 \
  --max-symbols 60 \
  --use-metrics-filters \
  --metrics-lookback-bars 3 \
  --min-oi-change 0.0 \
  --taker-ratio-threshold 1.05 \
  --top-trader-ratio-threshold 1.02 \
  --account-ratio-threshold 1.02 \
  --walkforward-splits 3 \
  --output-tag metrics_strict
```

## Output

The scripts write into `codex-exp-6/out/`:

- `universe_summary.csv`
- `eligible_symbols.txt`
- `market_structure_summary.csv`
- `strategy_summary.csv`
- `market_structure_report.md`
- `walkforward_summary*.csv` when walk-forward is enabled
- `cache/` stores merged per-symbol pickle caches to speed repeated runs

## If More Data Is Needed

The local `datalake` already has broad symbol coverage. If you need more history or symbols, use:

- `datalake/download_binance_data.py`
- `datalake/download_bybit_data.py`

Refer to `datalake/README.md` for supported types and usage.

If you want a cold run without cached symbol merges, add `--no-cache`.
