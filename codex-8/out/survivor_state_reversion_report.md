# State-Conditioned Reversion Report

- Input: `/home/ubuntu/Projects/skytrade6/codex-8/out/dislocation_panel_survivors.csv.gz`
- Hold horizon: `15` minutes
- Fee assumption: `8.0` bps round trip
- Candidate filter: `bybit_discount_bps >= 6.0`
- Train/Test cutoff: `2026-01-03T16:34:00+00:00`

## Dataset

- Raw rows: `1,066,102`
- Candidate rows: `229,807`
- Train candidates: `156,312`
- Test candidates: `73,495`
- Symbols: `3`
- Train positive-after-fee rate: `41.32%`
- Test positive-after-fee rate: `39.02%`

## Model

- Train ROC AUC: `0.5392`
- Test ROC AUC: `0.5235`
- State sleeve threshold: `0.9508`
- Model sleeve threshold: `0.5184`

## Test Summary

- `baseline`: trades=73,495, net/trade=-8.71 bps, win_rate=47.2%, after_fee_hit=39.0%, gap_close=+12.34 bps, total_net=-639996.2 bps
- `state_top_quantile`: trades=15,762, net/trade=-9.37 bps, win_rate=47.3%, after_fee_hit=38.7%, gap_close=+18.02 bps, total_net=-147713.5 bps
- `model_top_quantile`: trades=4,371, net/trade=-9.29 bps, win_rate=49.7%, after_fee_hit=44.4%, gap_close=+10.84 bps, total_net=-40601.5 bps

## Top Positive Feature Weights

- `realized_vol_15m_bps`: `+0.2222`
- `premium_gap_bps`: `+0.0376`
- `discount_z_60`: `+0.0370`
- `oi_gap_5m`: `+0.0304`
- `crowding_gap`: `+0.0262`
- `discount_z_240`: `+0.0245`
- `oi_gap_30m`: `+0.0168`
- `crowding_align_z_240`: `+0.0111`

## Top Negative Feature Weights

- `bybit_discount_bps`: `-0.0379`
- `premium_gap_z_240`: `-0.0245`
- `rel_ret_15m_bps`: `-0.0227`
- `bn_taker_imbalance`: `-0.0226`
- `crowding_gap_z_240`: `-0.0225`
- `rel_ret_5m_bps`: `-0.0136`
- `oi_gap_30m_z_240`: `-0.0124`
- `crowding_align_z_240`: `+0.0111`
