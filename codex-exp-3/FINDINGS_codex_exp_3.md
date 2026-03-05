# Codex Exp 3: Useful Repo Finding, Revalidated Cross-Venue

This note treats prior repo PnL as untrusted and re-checks the strongest plausible signal under a simple but stricter test.

## Rejected Path

- A fresh `codex-exp-3` experiment requiring Binance and Bybit positioning to simultaneously confirm a long setup did not produce any configuration that stayed positive in both train and test after `8 bps` all-maker fees.
- That means the broad "two-venue crowding confirmation" idea is weaker than the repo's optimistic strategy reports suggest.

## Surviving Path

- The most credible prior result was the rare-event long-only Binance LS momentum strategy from `codex-exp-2`.
- That strategy was originally selected under a much harsher `20 bps` all-taker hurdle using raw `datalake/binance` data.
- In this version, the Binance sample is rebuilt directly inside `codex-exp-3` from raw `datalake/binance` files, with no dependency on `codex-exp-2` outputs.
- Those exact entry timestamps are then repriced using Bybit futures closes and averaged with Binance returns.

## Strategy

- Signal source: Binance positioning + taker-flow metrics.
- Entry cadence: every 4 hours at `HH:05 UTC`.
- Hold: fixed 4 hours.
- Long-only rule:
  - `ls_z >= 2.0`
  - `taker_z >= 0.5`
  - `mom_4h > 0`
  - `oi_med_3d >= 20,000,000`
  - market breadth `>= 0.60`
  - median LS z-score `>= 0.0`
- Selection: top `3` names by the existing `score_abs` ranking.
- Fee hurdle: `8 bps` round-trip all-maker (`0.04%` per side).

## Coverage

- Rebuilt Binance sample rows: 146,268
- Selected symbol-level entries: 178
- Selected portfolio timestamps: 114
- Unique traded symbols: 63
- Bybit-repriced symbol entries retained: 178
- Train timestamps: 87
- Test timestamps: 27

## Cross-Venue Results

- Train avg/trade after 8 bps on average(Binance, Bybit): 27.52 bps
- Test avg/trade after 8 bps on average(Binance, Bybit): 40.02 bps
- Test avg/trade after 12 bps on average(Binance, Bybit): 36.02 bps
- Test avg/trade after 16 bps on average(Binance, Bybit): 32.02 bps
- Test win rate after 8 bps: 59.3%

## Execution-Aware Soft Penalty

- Ranking penalty = (avg positive 60s VWAP drag + fill shortfall penalty) / 4.0
- Fill shortfall penalty = 8.0 bps scaled by (1 - Bybit fill rate)
- Soft-penalized test timestamps: 27
- Soft-penalized test avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): 39.91 bps
- Soft-penalized test avg/trade after 12 bps on average(Binance, Bybit net of Bybit funding): 35.91 bps
- Soft-penalized test avg/trade after 16 bps on average(Binance, Bybit net of Bybit funding): 31.91 bps
- Soft-penalized test win rate after 8 bps: 59.3%

## Hard-Filter Comparison

- Hard-filter threshold: average positive 60s VWAP drag >= 8.0 bps, or Bybit maker-fill proxy < 100%
- Hard-filtered symbols: AVAXUSDT, BARDUSDT, ENAUSDT, LINKUSDT, PAXGUSDT, XLMUSDT, XRPUSDT
- Filtered test timestamps: 23
- Filtered test avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): 37.81 bps
- Filtered test avg/trade after 12 bps on average(Binance, Bybit net of Bybit funding): 33.81 bps
- Filtered test avg/trade after 16 bps on average(Binance, Bybit net of Bybit funding): 29.81 bps
- Filtered test win rate after 8 bps: 60.9%

## Partial Funding Adjustment (Bybit Leg Only)

- Train avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): 27.97 bps
- Test avg/trade after 8 bps on average(Binance, Bybit net of Bybit funding): 39.91 bps
- Test avg/trade after 12 bps on average(Binance, Bybit net of Bybit funding): 35.91 bps
- Test avg/trade after 16 bps on average(Binance, Bybit net of Bybit funding): 31.91 bps
- Test win rate after 8 bps with Bybit funding applied: 59.3%
- Mean Bybit funding impact per 4h trade in test: 0.223 bps

## Venue Comparison

- Train avg/trade after 8 bps on Binance-only pricing: 27.18 bps
- Train avg/trade after 8 bps on Bybit-only pricing: 27.87 bps
- Test avg/trade after 8 bps on Binance-only pricing: 39.53 bps
- Test avg/trade after 8 bps on Bybit-only pricing: 40.51 bps
- Test avg/trade after 8 bps on Bybit-only pricing net of Bybit funding: 40.29 bps

## Interpretation

- The useful signal is not a generic cross-exchange spread trade.
- The useful signal is a selective continuation trade: when Binance positioning becomes extremely bullish and taker flow confirms it, the same coin tends to keep rising over the next 4 hours on both Binance and Bybit.
- Because the Bybit repricing stays positive, this looks more like a real underlying asset effect than a Binance-only artifact.
- The funding-aware numbers are only a partial adjustment because Binance funding is not joined here; they are still a more conservative check than pure price returns.
- This still does not fully model queue priority or partial fills, but the drag sweeps below show how much extra friction the edge can absorb.

## Execution-Drag Sweep

| Extra Drag | Total Cost | Test Avg | Test Win Rate |
|---|---:|---:|---:|
| 0 bps | 8 bps | 40.02 bps | 59.3% |
| 2 bps | 10 bps | 38.02 bps | 59.3% |
| 4 bps | 12 bps | 36.02 bps | 55.6% |
| 6 bps | 14 bps | 34.02 bps | 55.6% |
| 8 bps | 16 bps | 32.02 bps | 55.6% |
| 12 bps | 20 bps | 28.02 bps | 55.6% |

## Funding-Aware Drag Sweep

| Extra Drag | Total Cost | Test Avg (Funding Adj) | Test Win Rate |
|---|---:|---:|---:|
| 0 bps | 8 bps | 39.91 bps | 59.3% |
| 2 bps | 10 bps | 37.91 bps | 59.3% |
| 4 bps | 12 bps | 35.91 bps | 55.6% |
| 6 bps | 14 bps | 33.91 bps | 55.6% |
| 8 bps | 16 bps | 31.91 bps | 55.6% |
| 12 bps | 20 bps | 27.91 bps | 55.6% |

## Monthly Test Breakdown

| Month | Trades | Avg 8bps | Avg 12bps | Avg 16bps | Win Rate |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 21 | 35.47 | 31.47 | 27.47 | 61.9% |
| 2026-02 | 6 | 55.95 | 51.95 | 47.95 | 50.0% |
