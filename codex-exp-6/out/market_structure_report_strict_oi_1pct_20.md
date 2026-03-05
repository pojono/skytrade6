# Market Structure Report

- Exchange: `binance`
- Date range: `2026-01-02` to `2026-03-02`
- Universe size: `20`
- Maker fee assumption: `4.00` bps per side
- Taker fee assumption: `10.00` bps per side

## Horizon Summary

- 1m: avg corr `0.4191`, strong sync `0.641`, very sync `0.419`, top breadth fwd `1.85` bps, bottom breadth fwd `-1.94` bps
- 5m: avg corr `0.4191`, strong sync `0.705`, very sync `0.512`, top breadth fwd `1.98` bps, bottom breadth fwd `-1.90` bps
- 15m: avg corr `0.4191`, strong sync `0.731`, very sync `0.553`, top breadth fwd `-0.84` bps, bottom breadth fwd `-1.43` bps
- 60m: avg corr `0.4191`, strong sync `0.741`, very sync `0.560`, top breadth fwd `-5.10` bps, bottom breadth fwd `0.67` bps
- 240m: avg corr `0.4191`, strong sync `0.759`, very sync `0.585`, top breadth fwd `20.90` bps, bottom breadth fwd `-13.79` bps

## Best Maker Candidates

- breadth_trend 240m: gross `17.83` bps, maker net `9.83` bps, samples `21331`
- breadth_gated_cross_sectional 240m: gross `20.17` bps, maker net `4.17` bps, samples `65203`
- cross_sectional_momentum 240m: gross `19.81` bps, maker net `3.81` bps, samples `85920`
- filtered_cross_sectional_momentum 1m: gross `11.39` bps, maker net `-4.61` bps, samples `1`
- breadth_gated_filtered_cross_sectional 1m: gross `11.39` bps, maker net `-4.61` bps, samples `1`

## Best Taker Candidates

- breadth_trend 240m: gross `17.83` bps, taker net `-2.17` bps, samples `21331`
- breadth_trend 5m: gross `1.94` bps, taker net `-18.06` bps, samples `24180`
- breadth_trend 1m: gross `1.89` bps, taker net `-18.11` bps, samples `21222`
- breadth_trend 15m: gross `0.38` bps, taker net `-19.62` bps, samples `27420`
- breadth_gated_cross_sectional 240m: gross `20.17` bps, taker net `-19.83` bps, samples `65203`

## Walk-Forward Stability

- breadth_trend 240m: splits `3`, avg maker `6.37` bps, worst maker `-20.80` bps, avg taker `-5.63` bps
- breadth_gated_cross_sectional 240m: splits `3`, avg maker `4.85` bps, worst maker `-11.35` bps, avg taker `-19.15` bps
- cross_sectional_momentum 240m: splits `3`, avg maker `4.56` bps, worst maker `-8.16` bps, avg taker `-19.44` bps
- breadth_gated_filtered_cross_sectional 1m: splits `1`, avg maker `-4.61` bps, worst maker `-4.61` bps, avg taker `-28.61` bps
- filtered_cross_sectional_momentum 1m: splits `1`, avg maker `-4.61` bps, worst maker `-4.61` bps, avg taker `-28.61` bps
