# Market Structure Report

- Exchange: `binance`
- Date range: `2026-01-02` to `2026-03-02`
- Universe size: `60`
- Maker fee assumption: `4.00` bps per side
- Taker fee assumption: `10.00` bps per side

## Horizon Summary

- 1m: avg corr `0.3766`, strong sync `0.552`, very sync `0.322`, top breadth fwd `2.22` bps, bottom breadth fwd `-2.07` bps
- 5m: avg corr `0.3766`, strong sync `0.647`, very sync `0.435`, top breadth fwd `2.51` bps, bottom breadth fwd `-1.82` bps
- 15m: avg corr `0.3766`, strong sync `0.683`, very sync `0.483`, top breadth fwd `-0.89` bps, bottom breadth fwd `-1.13` bps
- 60m: avg corr `0.3766`, strong sync `0.696`, very sync `0.492`, top breadth fwd `-7.94` bps, bottom breadth fwd `3.32` bps
- 240m: avg corr `0.3766`, strong sync `0.717`, very sync `0.520`, top breadth fwd `19.70` bps, bottom breadth fwd `-6.37` bps

## Best Maker Candidates

- breadth_trend 240m: gross `13.01` bps, maker net `5.01` bps, samples `19474`
- filtered_cross_sectional_momentum 240m: gross `10.98` bps, maker net `-5.02` bps, samples `79720`
- cross_sectional_momentum 240m: gross `10.97` bps, maker net `-5.03` bps, samples `85920`
- breadth_trend 5m: gross `2.17` bps, maker net `-5.83` bps, samples `18343`
- breadth_trend 1m: gross `2.15` bps, maker net `-5.85` bps, samples `19769`

## Best Taker Candidates

- breadth_trend 240m: gross `13.01` bps, taker net `-6.99` bps, samples `19474`
- breadth_trend 5m: gross `2.17` bps, taker net `-17.83` bps, samples `18343`
- breadth_trend 1m: gross `2.15` bps, taker net `-17.85` bps, samples `19769`
- breadth_trend 15m: gross `0.23` bps, taker net `-19.77` bps, samples `19529`
- breadth_trend 60m: gross `-5.35` bps, taker net `-25.35` bps, samples `19738`
