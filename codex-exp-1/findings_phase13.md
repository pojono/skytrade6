# Findings Phase 13: Baseline vs Filtered 25% Downside Comparison

This phase runs the same downside analysis on the baseline `25%` variant and compares it directly to the frozen replay-optimized `25%` variant.

Compared variants:

1. Baseline `25%`
   - source replay: `out/paper_report_v3_base_sized25.md`
   - downside: `out/downside_report_v3_base_sized25.md`
2. Replay-optimized filter `25%`
   - source replay: `out/paper_report_v3_replayopt_sized25.md`
   - downside: `out/downside_report_v3_replayopt_sized25.md`

## PnL Comparison

Baseline `25%`:

- `1,421` fills
- final capital: `$105,183.40`
- total PnL: `$5,183.40`
- win rate: `50.32%`
- average net edge: `1.4242 bps`

Filtered `25%`:

- `897` fills
- final capital: `$107,638.99`
- total PnL: `$7,638.99`
- win rate: `53.29%`
- average net edge: `3.2849 bps`

Filtered advantage:

- `+$2,455.59` more total PnL
- `+2.97` percentage points win rate
- `+1.8607 bps` average net edge

## Drawdown Comparison

Baseline `25%`:

- max realized drawdown: `$1,399.34`
- max realized drawdown: `1.40%`
- drawdown window:
  - start: `2025-08-12 00:13:00 UTC`
  - end: `2025-10-25 00:13:00 UTC`

Filtered `25%`:

- max realized drawdown: `$270.08`
- max realized drawdown: `0.25%`
- drawdown window:
  - start: `2026-02-21 02:08:00 UTC`
  - end: `2026-02-23 00:38:00 UTC`

Filtered improvement:

- drawdown dollars reduced by `80.70%`
- drawdown percent reduced from `1.40%` to `0.25%`

This is a large structural difference, not just noise.

## Weekly Stability Comparison

Baseline `25%`:

- weeks observed: `31`
- positive weeks: `18`
- negative weeks: `13`
- worst week: `2025-W41` at `-$480.71`

Filtered `25%`:

- weeks observed: `31`
- positive weeks: `26`
- negative weeks: `5`
- worst week: `2025-W34` at `-$102.94`

Filtered improvement:

- `8` fewer negative weeks
- worst week loss reduced by `78.59%`

## Monthly Stability Comparison

Baseline `25%`:

- months observed: `8`
- positive months: `5`
- negative months: `3`
- negative months:
  - `2025-08`: `-$191.55`
  - `2025-09`: `-$402.99`
  - `2025-10`: `-$594.51`

Filtered `25%`:

- months observed: `8`
- positive months: `7`
- negative months: `1`
- negative month:
  - `2025-08`: `-$181.75`

Filtered improvement:

- only one losing month instead of three
- the early drawdown cluster is mostly removed

## Practical Conclusion

This is the strongest comparison so far because both variants use:

- the same market sample
- the same live-style replay engine
- the same `25%` sizing
- the same size-aware slippage model

Under those matched assumptions, the replay-optimized filtered variant is better on both:

- return
- downside

That makes it the current best research candidate in the repo.

## Current Best Candidate

The best current candidate is now:

- replay-optimized filter
- `25%` allocation
- one open position
- selector mode `spread`
- daily cap `3`
- size-aware slippage enabled

Why it is the best current candidate:

- better total PnL than baseline at the same larger size
- materially better realized drawdown
- materially fewer negative weeks and months
- better win rate and better average edge

The main remaining caveat is unchanged:

- this is still a modeled replay, not live execution
- funding, order book depth, and adverse path risk are still simplified
