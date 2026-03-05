# Trade Entry Findings: Top 3 Test Symbols

This is a trade-level sanity check for the three highest-frequency symbols in the 2026 test sample:

- `WIFUSDT`
- `LINKUSDT`
- `LTCUSDT`

Data used:

- Binance `*_trades.csv`
- Bybit `*_trades.csv`
- Only the actual strategy signal timestamps from the 2026 test period
- First 60 seconds after each signal

Method:

- Reference price = the last 1-minute kline close before the `HH:05 UTC` signal timestamp
- Maker-fill proxy = whether at least one trade in the next 60 seconds printed at or below that reference price
- Market-impact proxy = first-trade bps and 60-second trade VWAP bps versus that reference price

## Aggregate

- Signals checked: `13`
- Binance maker-fill proxy within 60s: `100%`
- Bybit maker-fill proxy within 60s: `100%`
- Binance first-trade drift: `+0.03 bps`
- Bybit first-trade drift: `-0.17 bps`
- Binance 60s VWAP drift: `+2.68 bps`
- Bybit 60s VWAP drift: `+3.15 bps`

## By Symbol

| Symbol | Signals | Binance Fill | Bybit Fill | Binance First | Bybit First | Binance VWAP | Bybit VWAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| WIFUSDT | 5 | 100% | 100% | -0.50 bps | -0.35 bps | +1.07 bps | -1.08 bps |
| LINKUSDT | 4 | 100% | 100% | +0.27 bps | +0.26 bps | +10.64 bps | +14.98 bps |
| LTCUSDT | 4 | 100% | 100% | +0.45 bps | -0.37 bps | -3.26 bps | -3.38 bps |

## Interpretation

- The passive-entry proxy looks favorable on this small sample: all 13 checks traded back through the signal reference within one minute on both venues.
- Immediate market-entry drift is tiny on average, which is consistent with the earlier flat drag assumptions not being absurd.
- `LINKUSDT` is the main caution flag. Its first-minute VWAP moves materially above the signal reference, which means chasing after the signal could cost around `10-15 bps` on that name.
- `WIFUSDT` and `LTCUSDT` look materially cleaner in this small sample.

Files:

- Detail rows: `trade_entry_feasibility_top3.csv`
- Symbol summary: `trade_entry_feasibility_summary_top3.csv`
