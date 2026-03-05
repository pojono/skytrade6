# Per-Trade Classifier

This is a second-stage take/skip classifier on top of the current base strategy.

## Setup

- Base input is the current best signal instances only (the base top-3 strategy rows).
- Target = whether a trade is positive after `20 bps` round-trip cost.
- Train period: before `2026-01-01`.
- Holdout test period: `2026-01-01` onward.
- Model selection uses expanding time-order validation on the train period.
- Features: `ls_z, taker_z, mom_4h, score_abs, breadth_mom, median_ls_z, median_taker_z, log_oi_med_3d, symbol_train_avg_bps, symbol_train_hit_rate, symbol_train_trades, symbol_seen_train`

## Class Balance

- Train positive rate: 47.0%
- Test positive rate: 47.6%

## Best CV Configuration

- Model: `logistic`
- Probability threshold: `0.70`
- Max positions per timestamp: `2`
- CV decisions: 21
- CV avg: 141.48 bps
- CV hit rate: 81.0%

## Holdout Comparison

- Base holdout decisions: 27
- Base holdout avg: 26.95 bps
- Base holdout hit rate: 55.6%
- Classifier holdout decisions: 8
- Classifier holdout trade rows: 8
- Classifier holdout avg: -31.35 bps
- Classifier holdout hit rate: 50.0%

## Holdout Diagnostic Across Top CV Candidates

- Best test result among top CV candidates: `hgb` threshold `0.60` max-pos `1`
- That candidate holdout avg: 2.61 bps
- That candidate holdout decisions: 12

## Interpretation

- If holdout avg improves while retaining enough decisions, the classifier is useful.
- If it only improves by collapsing to almost no trades, it is too sparse to trust.
- If the best holdout result among strong CV candidates is still below the base strategy, the classifier is not adding value yet.
- This still uses only explicit fees; execution realism is not added here.

## Top Holdout Trades Kept

| Timestamp | Symbol | Prob | Net Bps |
|---|---|---:|---:|
| 2026-01-30 20:05:00+00:00 | XPLUSDT | 0.971 | -132.18 |
| 2026-01-01 08:05:00+00:00 | WIFUSDT | 0.964 | 53.72 |
| 2026-01-08 16:05:00+00:00 | WIFUSDT | 0.960 | -399.71 |
| 2026-01-01 12:05:00+00:00 | WIFUSDT | 0.959 | 210.52 |
| 2026-02-16 00:05:00+00:00 | WIFUSDT | 0.940 | 49.23 |
| 2026-01-09 00:05:00+00:00 | WIFUSDT | 0.914 | 64.14 |
| 2026-01-23 04:05:00+00:00 | ETHUSDT | 0.813 | -67.64 |
| 2026-02-04 04:05:00+00:00 | AVAXUSDT | 0.729 | -28.90 |
