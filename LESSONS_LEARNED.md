# Lessons Learned

**Date:** 2026-02-15
**Project:** skytrade6 — Crypto microstructure research

## 1. Data Pipeline

- **Daily-partitioned parquet** is the right structure for this scale of data. One file per day per source per data type allows incremental builds and bounded memory.
- **OKX data is not UTC-aligned** — their daily files start/end at UTC+8. The `build_parquet.py` script handles this with a sliding-window re-partitioning approach.
- **Incremental builds** (skip existing files) save hours on re-runs. Always check if output exists before processing.
- **~30 GB** of parquet for BTCUSDT alone (92 days, 6 sources). Plan storage accordingly.

## 2. Notebooks vs Scripts

- **Notebooks are bad for heavy computation.** They hide progress, buffer stdout, and crash the machine when they OOM with no warning.
- **Python scripts with real-time progress** are far better for iterative research:
  - Print progress every N files/rows
  - Show RAM usage (`psutil.virtual_memory().used`)
  - Show elapsed time and ETA
  - Process data day-by-day, never load everything at once
- **Notebooks are fine for** visualization and final presentation of results, but not for the compute step.
- `jupyter nbconvert --execute` captures all stdout until the cell finishes — useless for monitoring long-running cells.

## 3. Memory Management

- **Never load all trades into memory at once.** BTCUSDT has 360M+ trades per exchange over 92 days. That's 30+ GB in a DataFrame.
- **Process one day at a time**, compute features, `del trades` immediately.
- **Feature DataFrames are tiny** (~288 rows per day at 5m intervals) and can safely be accumulated.
- **RAM stayed under 8 GB** when processing day-by-day, vs OOM crash when loading everything.

## 4. Fee-Aware Research

- **Always include fees from the start.** A signal that looks great gross can be worthless net of fees.
- **Bybit VIP0 fees:** maker 2 bps, taker 5 bps → 7 bps round-trip.
- Individual microstructure features have decile spreads of ~1-2 bps at 15m — **below the fee threshold**.
- The composite signal at 4h holding achieves ~13.7 bps avg — **above fees**.
- **Holding period is critical.** Short holds (15m, 30m) are consistently negative after fees. The signal needs time to play out.

## 5. Signal Research

- **Start simple, test fast.** The 7-day test ran in <60 seconds and gave a quick read on viability.
- **Then extend carefully.** The 7-day result (+4.59 bps at 2h) didn't hold at 30 days. The 4h config did.
- **Parameter sensitivity matters.** The difference between profitable and unprofitable was threshold (1.0 vs 1.5) and holding period (4h vs 2h).
- **Cross-exchange consistency** is a good sanity check — features that work on all 3 exchanges are more likely real.
- **Regime dependence** is real — signals are 2x stronger in low-vol. This is useful for position sizing.

## 6. What Didn't Work

- **Sub-minute lead-lag** — exists but edge is ~2-3 bps, completely eaten by VIP0 fees.
- **Cross-exchange spread arb** — spreads are 0.2 bps mean, 0.85 bps std. Way below fee threshold.
- **Binance metrics** (OI, L/S ratios) — correlations < 0.01 with 5m forward returns. Not directly predictive at this frequency.
- **Short holding periods** (15m, 30m) — signal is too weak relative to fees.
- **Loading all data into notebooks** — caused OOM and machine reboot.

## 7. Process Improvements

- **Always show progress to console.** No silent long-running processes.
- **Test on small data first** (7 days), then scale up.
- **Keep scripts focused** — one script per experiment, not one giant notebook.
- **Save intermediate results to parquet** so you don't recompute features every time.
- **Monitor RAM** during processing — add `psutil` checks.
