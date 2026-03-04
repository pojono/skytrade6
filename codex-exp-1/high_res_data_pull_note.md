# High-Resolution Data Pull Note

To support sub-minute / microstructure testing for the current strategy symbols, a bounded recent bulk-data pull was completed.

## Symbols

- `CRVUSDT`
- `GALAUSDT`
- `SEIUSDT`

## Date Window

- `2026-02-24` through `2026-03-02`

## Downloaded Data

### Bybit

Downloaded with:

```bash
python3 datalake/download_bybit_data.py CRVUSDT,GALAUSDT,SEIUSDT 2026-02-24 2026-03-02 -t orderbook,trades -c 2
```

Result:

- `7` days of `_orderbook.jsonl` per symbol
- `7` days of `_trades.csv` per symbol
- all requested files found for all three symbols

### Binance

Downloaded with:

```bash
python3 datalake/download_binance_data.py CRVUSDT,GALAUSDT,SEIUSDT 2026-02-24 2026-03-02 -t bookDepth,bookTicker,trades -c 2
```

Result:

- `7` days of `_bookDepth.csv` per symbol
- `7` days of `_trades.csv` per symbol
- `_bookTicker.csv` archives returned `404` for these dates / symbols and were not available

## Why This Matters

The research up to now was based on 1-minute aggregates.

This new data makes it possible to test:

- intraminute spread behavior
- order book imbalance around entry signals
- microstructure-aware slippage assumptions
- whether the one-minute edge survives when evaluated with finer execution realism

## Immediate Next Use

The next logical research step is:

- reconstruct sub-minute snapshots around each historical candidate trade
- estimate entry quality and adverse excursion using order book and trades
- replace part of the current modeled slippage with measured microstructure-based assumptions
