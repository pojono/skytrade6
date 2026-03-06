# Current Findings (2026-03-06) — Data Inventory Refresh

## Source of truth used
- `datalake/DATA_INVENTORY.md` (updated 2026-03-06)
- `datalake/binance/*` raw daily files only (no external parquet assumptions)

## What changed after new downloads
- Binance now has broad execution data coverage:
  - `*_trades.csv.gz`: 22,854 files
  - `*_bookDepth.csv.gz`: 22,646 files
  - `*_metrics.csv`: 44,626 files
- For `codex-exp-2` signal windows (`delayed_confirmation_features.csv`), required symbol-day pairs: `213`.
  - Covered by both Binance trades + bookDepth: `195/213` pairs.
  - Missing only `18/213` pairs, concentrated in:
    - `1000PEPEUSDT` (7)
    - `KITEUSDT` (4)
    - `XMRUSDT` (4)
    - `PAXGUSDT` (3)

## Fresh strategy rerun (datalake-native)
- Script: `codex-exp-2/research_fee_aware_ls_momentum.py`
- Outputs refreshed:
  - `codex-exp-2/samples_4h.csv`
  - `codex-exp-2/grid_results.csv`
  - `codex-exp-2/FINDINGS_codex_exp_2.md`

Top config (ranked by consistency and positive train/test):
- `ls_threshold=2.0`, `taker_threshold=0.5`, `min_oi=20M`, `top_n=3`, `breadth=0.60`, `median_ls=0.0`
- Train: `+25.91 bps` (all-taker)
- Test: `+11.64 bps` (all-taker), `+17.64 bps` (taker/maker), `+23.64 bps` (all-maker), `+7.64 bps` (24 bps stress)
- Train trades: `126`, Test trades: `29`

## Practical interpretation
- On the datalake-only momentum framework, the strategy still clears your fee hurdle (`maker 0.04%`, `taker 0.1%` round-trip `14 bps`) in multiple configs.
- Execution-orderbook validation is now mostly unblocked, but not fully complete because 18 signal pairs still lack Binance trades/bookDepth.

## Next immediate step
- Complete those 18 missing symbol-days, then rerun the orderbook-execution simulation (`30s entry delay + 60m hold`) so final PnL uses realistic fill impact everywhere, not partially.
