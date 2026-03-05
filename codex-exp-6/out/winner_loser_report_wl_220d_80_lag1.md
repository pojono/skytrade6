# Winner/Loser Regime Report

- Date range: `2025-07-26` to `2026-03-02`
- Universe size: `80`
- Daily rows (train/test): `146` / `63`
- Probability threshold: `0.55`

## Classifier Quality (Test)

- Accuracy: `0.556`
- Precision: `0.375`
- Recall: `0.115`
- F1: `0.176`
- AUC: `0.494`

## Trade/No-Trade Outcome (Test)

- Trade days: `8` of `63` (12.70%)
- Avg expected net on trade days: `-13.73` bps
- Avg expected net on no-trade days: `-7.76` bps

## Top Winner-vs-Loser Differences (Train)

- lag1_signal_count: winner `308.710`, loser `356.405`, delta `-47.695`
- lag1_disp_240_mean: winner `200.274`, loser `174.908`, delta `25.366`
- lag1_signal_align_240_mean: winner `253.432`, loser `251.066`, delta `2.366`
- lag1_signal_align_60_mean: winner `65.944`, loser `67.746`, delta `-1.802`
- lag1_rv_60_mean: winner `9.314`, loser `10.543`, delta `-1.229`
- lag1_rv_240_mean: winner `10.325`, loser `10.839`, delta `-0.513`
- lag1_top_trader_ratio_mean: winner `1.983`, loser `1.926`, delta `0.056`
- lag1_account_ratio_mean: winner `1.785`, loser `1.763`, delta `0.022`
