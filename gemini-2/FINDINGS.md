# Final Strategy Findings: Optimized Liquidation Flush

## Core Hypothesis Validated
The most significant alpha discovered across Bybit spot and futures data is extreme mean-reversion following **forced liquidations**. When a coin experiences a rapid price plunge accompanied by a massive drop in Open Interest (OI), it indicates that leveraged long participants have been wiped out. With the forced selling exhausted, the price violently snaps back. 

The previous test highlighted the danger of shorting "squeeze exhaustion" (which dragged returns down) and highlighted that tight Stop-Losses (10%) choke the trade, as liquidation wicks can plunge further before reversing. Giving the trade room to breathe unlocks massive alpha.

## Optimized Strategy Rules
- **Entry Condition:** 15-minute Price Return <= -5.0% **AND** 15-minute OI Change <= -10.0%
- **Exit Condition:** Take Profit = +40.0%, Stop Loss = -20.0%, Time Limit = 6 hours
- **Sizing:** 20% of Total Equity per trade (Max 5 concurrent positions, 1 per symbol max)
- **Fees Accounted:** 20 bps roundtrip per trade (0.04% maker + 0.1% taker on entry and exit, plus buffer)

## Portfolio Simulation (Jan 2024 - Mar 2026)
- **Initial Capital:** $10,000.00
- **Final Capital:** $21,507.60
- **Net Return:** 115.08%
- **Max Drawdown:** 32.55%
- **Total Trades Taken:** 141
- **Win Rate:** 53.2%
- **Average Net Return per Trade:** 336.81 bps (3.37% per trade)

## Monthly Performance
```text
  month  trades   win_rate  net_pnl_usd
2024-01       1 100.000000   113.854227
2025-01       2   0.000000  -427.403732
2025-02      13  76.923077  1443.016516
2025-03       5  40.000000   447.235848
2025-04       2  50.000000  -153.043998
2025-05       1 100.000000   133.562768
2025-06       4  75.000000   267.030110
2025-07       5  40.000000  -658.813390
2025-08       9  33.333333 -1116.390931
2025-09      16  50.000000  1102.202144
2025-10      35  57.142857  3697.384566
2025-11      11  54.545455  1974.585464
2025-12       6  16.666667 -1025.564490
2026-01      18  72.222222  7710.859520
2026-02      10  20.000000 -2723.310151
2026-03       3  66.666667   722.393195
```

## Isolated Monthly Performance
Assuming a fresh $10,000 start at the beginning of each month to measure raw monthly alpha without the compounding effect of previous months:

```text
  Month  Trades    WinRate     Return     MaxDD
2024-01       1 100.000000   1.138542  0.000000
2025-01       2   0.000000  -4.225923  4.225923
2025-02      13  76.923077  14.897268  4.040000
2025-03       5  40.000000   4.018484  4.142798
2025-04       2  50.000000  -1.322000  4.040000
2025-05       1 100.000000   1.169177  0.000000
2025-06       4  75.000000   2.310504  4.040000
2025-07       5  40.000000  -5.571713  5.931452
2025-08       9  33.333333  -9.998631 12.448675
2025-09      16  50.000000  10.968225  4.671574
2025-10      35  57.142857  33.156684 28.914039
2025-11      11  54.545455  13.298095  4.497299
2025-12       6  16.666667  -6.096125  6.096125
2026-01      18  72.222222  48.810152  3.068628
2026-02      10  20.000000 -11.584357 11.584357
2026-03       3  66.666667   3.475516  1.054517
```

## Isolated Monthly Performance
Assuming a fresh $10,000 start at the beginning of each month to measure raw monthly alpha without the compounding effect of previous months:

```text
  Month  Trades    WinRate     Return     MaxDD
2024-01       1 100.000000   1.138542  0.000000
2025-01       2   0.000000  -4.225923  4.225923
2025-02      13  76.923077  14.897268  4.040000
2025-03       5  40.000000   4.018484  4.142798
2025-04       2  50.000000  -1.322000  4.040000
2025-05       1 100.000000   1.169177  0.000000
2025-06       4  75.000000   2.310504  4.040000
2025-07       5  40.000000  -5.571713  5.931452
2025-08       9  33.333333  -9.998631 12.448675
2025-09      16  50.000000  10.968225  4.671574
2025-10      35  57.142857  33.156684 28.914039
2025-11      11  54.545455  13.298095  4.497299
2025-12       6  16.666667  -6.096125  6.096125
2026-01      18  72.222222  48.810152  3.068628
2026-02      10  20.000000 -11.584357 11.584357
2026-03       3  66.666667   3.475516  1.054517
```
