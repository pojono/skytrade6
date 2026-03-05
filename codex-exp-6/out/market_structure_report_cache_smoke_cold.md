# Market Structure Report

- Exchange: `binance`
- Date range: `2026-02-28` to `2026-03-02`
- Universe size: `5`
- Maker fee assumption: `4.00` bps per side
- Taker fee assumption: `10.00` bps per side

## Horizon Summary

- 1m: avg corr `0.7375`, strong sync `0.790`, very sync `0.790`, top breadth fwd `1.95` bps, bottom breadth fwd `-1.50` bps
- 5m: avg corr `0.7375`, strong sync `0.806`, very sync `0.806`, top breadth fwd `-0.03` bps, bottom breadth fwd `-0.82` bps
- 15m: avg corr `0.7375`, strong sync `0.821`, very sync `0.821`, top breadth fwd `1.57` bps, bottom breadth fwd `3.52` bps
- 60m: avg corr `0.7375`, strong sync `0.832`, very sync `0.832`, top breadth fwd `11.17` bps, bottom breadth fwd `5.72` bps

## Best Maker Candidates

- cross_sectional_momentum 60m: gross `10.93` bps, maker net `-5.07` bps, samples `4200`
- breadth_trend 60m: gross `2.65` bps, maker net `-5.35` bps, samples `2343`
- breadth_gated_cross_sectional 60m: gross `10.60` bps, maker net `-5.40` bps, samples `3495`
- breadth_trend 1m: gross `1.73` bps, maker net `-6.27` bps, samples `2157`
- breadth_trend 5m: gross `0.39` bps, maker net `-7.61` bps, samples `2216`

## Best Taker Candidates

- breadth_trend 60m: gross `2.65` bps, taker net `-17.35` bps, samples `2343`
- breadth_trend 1m: gross `1.73` bps, taker net `-18.27` bps, samples `2157`
- breadth_trend 5m: gross `0.39` bps, taker net `-19.61` bps, samples `2216`
- breadth_trend 15m: gross `-1.01` bps, taker net `-21.01` bps, samples `2249`
- cross_sectional_momentum 60m: gross `10.93` bps, taker net `-29.07` bps, samples `4200`

## Walk-Forward Stability

- cross_sectional_momentum 60m: splits `2`, avg maker `-3.99` bps, worst maker `-14.08` bps, avg taker `-27.99` bps
- breadth_trend 60m: splits `2`, avg maker `-4.68` bps, worst maker `-11.70` bps, avg taker `-16.68` bps
- breadth_gated_cross_sectional 60m: splits `2`, avg maker `-4.68` bps, worst maker `-14.99` bps, avg taker `-28.68` bps
- breadth_trend 1m: splits `2`, avg maker `-6.25` bps, worst maker `-6.70` bps, avg taker `-18.25` bps
- breadth_trend 5m: splits `2`, avg maker `-7.61` bps, worst maker `-7.76` bps, avg taker `-19.61` bps
