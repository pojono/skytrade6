# State-Conditioned Reversion Report

- Input: `/home/ubuntu/Projects/skytrade6/codex-8/out/dislocation_panel.csv.gz`
- Hold horizon: `15` minutes
- Fee assumption: `8.0` bps round trip
- Candidate filter: `bybit_discount_bps >= 6.0`
- Train/Test cutoff: `2025-12-27T04:48:00+00:00`

## Dataset

- Raw rows: `2,130,190`
- Candidate rows: `245,635`
- Train candidates: `172,121`
- Test candidates: `73,514`
- Symbols: `6`
- Train positive-after-fee rate: `34.38%`
- Test positive-after-fee rate: `31.60%`

## Model

- Train ROC AUC: `0.5754`
- Test ROC AUC: `0.5756`
- State sleeve threshold: `0.5954`
- Model sleeve threshold: `0.5262`

## Test Summary

- `baseline`: trades=73,514, net/trade=-8.42 bps, win_rate=48.4%, after_fee_hit=31.6%, gap_close=+0.18 bps, total_net=-619295.1 bps
- `state_top_quantile`: trades=15,915, net/trade=-8.13 bps, win_rate=50.0%, after_fee_hit=33.3%, gap_close=+1.05 bps, total_net=-129397.6 bps
- `model_top_quantile`: trades=13,781, net/trade=-8.46 bps, win_rate=49.8%, after_fee_hit=40.7%, gap_close=+0.93 bps, total_net=-116534.7 bps

## Top Positive Feature Weights

- `realized_vol_15m_bps`: `+0.5976`
- `crowding_align_z_240`: `+0.0733`
- `rel_ret_5m_bps`: `+0.0591`
- `oi_gap_30m_z_240`: `+0.0514`
- `discount_z_240`: `+0.0343`
- `rel_ret_15m_bps`: `+0.0316`
- `bn_taker_imbalance`: `+0.0134`
- `premium_gap_z_240`: `+0.0027`

## Top Negative Feature Weights

- `premium_gap_bps`: `-0.0594`
- `oi_gap_30m`: `-0.0532`
- `crowding_gap_z_240`: `-0.0329`
- `bybit_discount_bps`: `-0.0263`
- `oi_gap_5m`: `-0.0214`
- `discount_z_60`: `-0.0159`
- `crowding_gap`: `+0.0016`
- `premium_gap_z_240`: `+0.0027`
