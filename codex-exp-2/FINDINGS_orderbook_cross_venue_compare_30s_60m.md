# Cross-Venue Orderbook Comparison: Binance vs Bybit

## Setup
- Same signal timestamps and same Binance agg-trade reference prices for entry/exit.
- Only execution depth source is changed: Binance `book_depth` vs Bybit `orderbook/bybit_futures`.
- Fees: 20 bps round-trip.
- Gate fixed from current Binance walk-forward: `ret_30s_bps >= 1.471836` and `buy_share_30s >= 0.700624`.
- Overlap symbols in this run: `1000PEPEUSDT,BNBUSDT,BTCUSDT,ETHUSDT,LINKUSDT,ONDOUSDT,PENGUUSDT,SOLUSDT,SUIUSDT,TAOUSDT,TRUMPUSDT,WIFUSDT,XRPUSDT`

## Results (Test, overlap subset only)

 order_notional_usd  test_rows_overlap  test_gated_rows_overlap  binance_test_avg_bps  bybit_test_avg_bps  test_diff_bybit_minus_binance_bps  binance_test_gated_avg_bps  bybit_test_gated_avg_bps  test_gated_diff_bybit_minus_binance_bps                                                                     symbols_overlap_test
            10000.0                 34                        8            -26.578784          -29.224056                          -2.645272                   -9.633279                -13.259438                                -3.626160 1000PEPEUSDT,BNBUSDT,ETHUSDT,LINKUSDT,ONDOUSDT,SUIUSDT,TAOUSDT,TRUMPUSDT,WIFUSDT,XRPUSDT
            50000.0                 22                        4            -21.678593          -24.754036                          -3.075443                  -27.288663                -29.906607                                -2.617944                                    1000PEPEUSDT,BNBUSDT,ETHUSDT,LINKUSDT,SUIUSDT,XRPUSDT
           100000.0                 13                        3            -30.575267          -33.530225                          -2.954958                  -43.713143                -47.015159                                -3.302016                                                                  BNBUSDT,ETHUSDT,XRPUSDT

## Notes
- This is an overlap-only comparison, not a full-universe backtest.
- Current Bybit orderbook coverage is sparse vs Binance, so results are directionally useful but not final.