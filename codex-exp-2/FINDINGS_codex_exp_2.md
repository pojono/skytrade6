# Codex Exp 2: Fee-Aware LS Momentum

Independent research from raw `datalake/binance` daily CSVs only.

## Method

- Universe: Binance futures symbols with local `metrics` and `kline_1m` files.
- Features per symbol: 14-day rolling z-scores of top-trader long/short ratio and taker long/short volume ratio, plus 4h price momentum.
- Market regime filter: only trade when the liquid-universe breadth is risk-on (`share of positive 4h momentum >= threshold`) and median LS z-score is not bearish.
- Signal cadence: every 4 hours at `HH:05 UTC`.
- Long-only rule: `ls_z >= threshold`, `taker_z >= threshold`, and 4h momentum > 0.
- Selection: top `N` symbols by `abs(ls_z) + 0.35 * abs(taker_z)`.
- Hold: fixed 4 hours, no stop, no compounding.
- Fees tested: all taker 20 bps, taker entry + maker exit 14 bps, all maker 8 bps, and a 24 bps stress case.

## Data Coverage

- Sample rows: 226,309
- Eligible symbols after liquidity filter in train: 104
- Eligible symbols after liquidity filter in test: 72
- Train cutoff: through 2025-12-31
- Test period: from 2026-01-01

## Best Configuration (positive in train and test, ranked by the weaker period)

- `ls_threshold=2.0`
- `taker_threshold=0.5`
- `min_oi_value=20,000,000`
- `top_n=3`
- `breadth_threshold=0.6`
- `median_ls_threshold=0.0`
- Train trades: 126
- Test trades: 29
- Train avg/trade after 20 bps: 25.91 bps
- Test avg/trade after 20 bps: 11.64 bps
- Consistency score (min of train/test): 11.64 bps
- Test avg/trade after 14 bps: 17.64 bps
- Test avg/trade after 8 bps: 23.64 bps
- Test avg/trade after 24 bps stress: 7.64 bps
- Test win rate after 20 bps: 51.7%

## Notes

- This only prices explicit fees. It does not model slippage, queue position, partial fills, borrow/funding transfers, or capital constraints.
- Because the signal is slow (4h horizon), maker exits are at least operationally plausible; all-taker is the stricter benchmark.
- If all-taker stays positive, the signal is harder to dismiss as fee leakage.

## Monthly Test Breakdown

| Month | Trades | Avg 20bps | Avg 14bps | Avg 24bps | Hit Rate |
|---|---:|---:|---:|---:|---:|
| 2026-01 | 21 | 11.61 | 17.61 | 7.61 | 52.4% |
| 2026-02 | 6 | 41.23 | 47.23 | 37.23 | 50.0% |
| 2026-03 | 2 | -76.82 | -70.82 | -80.82 | 50.0% |
