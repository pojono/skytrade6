# Robust Walk-Forward Strategy

## Setup

- Universe source: trade-flow covered symbols (min overlap days=5)
- Walk-forward warmup days: 10
- Validation tail days per step: 4
- Min symbols per config: 2
- Min positive symbols in validation: 2
- Max top-symbol share in validation: 0.80
- Max trades per symbol per day: 1

## Outcome

- Days traded: 20
- Total trades: 28
- Avg net after taker fees: 5.7083 bps
- Win rate: 57.14%

## Symbol

| Symbol | Trades | Avg Net bps |
|---|---:|---:|
| CRVUSDT | 13 | 17.7585 |
| GALAUSDT | 10 | 1.9120 |
| GUNUSDT | 4 | -15.2565 |
| INITUSDT | 1 | -29.1204 |

## Monthly

| Month | Trades | Avg Net bps |
|---|---:|---:|
| 2026-02 | 23 | 6.8768 |
| 2026-03 | 5 | 0.3333 |

## Notes

- Gate selection is strictly walk-forward: only prior days are used to choose the next-day config.
- This is still bounded by local data coverage; if almost all symbols have zero triggers, diversification remains limited.
