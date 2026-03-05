# Market Structure Report

- Exchange: `binance`
- Date range: `2026-01-02` to `2026-03-02`
- Universe size: `40`
- Maker fee assumption: `4.00` bps per side
- Taker fee assumption: `10.00` bps per side

## Horizon Summary

- 1m: avg corr `0.4080`, strong sync `0.590`, very sync `0.368`, top breadth fwd `2.00` bps, bottom breadth fwd `-1.94` bps
- 5m: avg corr `0.4080`, strong sync `0.672`, very sync `0.472`, top breadth fwd `1.90` bps, bottom breadth fwd `-1.61` bps
- 15m: avg corr `0.4080`, strong sync `0.702`, very sync `0.514`, top breadth fwd `-1.44` bps, bottom breadth fwd `-0.62` bps
- 60m: avg corr `0.4080`, strong sync `0.710`, very sync `0.520`, top breadth fwd `-7.20` bps, bottom breadth fwd `5.28` bps
- 240m: avg corr `0.4080`, strong sync `0.738`, very sync `0.546`, top breadth fwd `17.50` bps, bottom breadth fwd `-11.69` bps

## Best Maker Candidates

- breadth_trend 240m: gross `14.43` bps, maker net `6.43` bps, samples `19906`
- cross_sectional_momentum 240m: gross `14.65` bps, maker net `-1.35` bps, samples `85920`
- breadth_gated_cross_sectional 240m: gross `11.24` bps, maker net `-4.76` bps, samples `63418`
- breadth_trend 1m: gross `1.97` bps, maker net `-6.03` bps, samples `19720`
- breadth_trend 5m: gross `1.74` bps, maker net `-6.26` bps, samples `20177`

## Best Taker Candidates

- breadth_trend 240m: gross `14.43` bps, taker net `-5.57` bps, samples `19906`
- breadth_trend 1m: gross `1.97` bps, taker net `-18.03` bps, samples `19720`
- breadth_trend 5m: gross `1.74` bps, taker net `-18.26` bps, samples `20177`
- breadth_trend 15m: gross `-0.46` bps, taker net `-20.46` bps, samples `19947`
- cross_sectional_momentum 240m: gross `14.65` bps, taker net `-25.35` bps, samples `85920`

## Walk-Forward Stability

- breadth_trend 240m: splits `3`, avg maker `1.25` bps, worst maker `-32.12` bps, avg taker `-10.75` bps
- cross_sectional_momentum 240m: splits `3`, avg maker `-0.77` bps, worst maker `-15.77` bps, avg taker `-24.77` bps
- breadth_gated_cross_sectional 240m: splits `3`, avg maker `-4.24` bps, worst maker `-21.13` bps, avg taker `-28.24` bps
- breadth_trend 1m: splits `3`, avg maker `-6.00` bps, worst maker `-6.44` bps, avg taker `-18.00` bps
- breadth_trend 5m: splits `3`, avg maker `-6.38` bps, worst maker `-7.00` bps, avg taker `-18.38` bps
