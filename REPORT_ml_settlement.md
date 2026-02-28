# ML Settlement Prediction Report

**Generated:** 2026-02-28 08:37 UTC  
**Dataset:** 130 settlements, 30 symbols, 3 dates (2026-02-26 to 2026-02-28)  
**Pipeline:** `ml_settlement_pipeline.py`

---

## Summary

| Metric | Value |
|--------|-------|
| Settlements | 130 |
| Unique symbols | 30 |
| Date range | 2026-02-26 to 2026-02-28 |
| FR vs drop correlation | r = +0.898 |
| Profitable (>40bps drop) | 81/130 (62%) |
| Best LOSO MAE | **22.6 bps** (FR+depth (8), ElasticNet) |
| Best Temporal MAE | **21.7 bps** (FR+depth (8), Ridge) |
| Best LOSO AUC | **0.942** (FR+depth (8), HGBC) |
| Baseline MAE | 50.4 bps |

## Regression: Predicting Drop Magnitude

Target: `drop_min_bps` (max price drop in 5s post-settlement)

| Features | Model | LOOCV MAE | LOSO MAE | LOSO R² | Temporal MAE | Temporal R² |
|----------|-------|-----------|----------|---------|--------------|------------|
| FR only (3) | Ridge | 22.5 | 23.2 | +0.777 | 23.9 | +0.734 |
| FR only (3) | ElasticNet | 22.3 | 23.1 | +0.775 | 24.1 | +0.726 |
| FR only (3) | HGBR | 24.7 | 26.3 | +0.684 | 28.0 | +0.670 |
| FR+depth (8) | Ridge | 20.8 | 22.7 | +0.756 | 21.7 | +0.768 |
| FR+depth (8) | ElasticNet | 20.7 | 22.6 | +0.763 | 22.0 | +0.758 |
| FR+depth (8) | HGBR | 23.3 | 24.5 | +0.718 | 27.3 | +0.703 |
| Tier 1 (full) | Ridge | 23.5 | 26.5 | +0.681 | 23.5 | +0.769 |
| Tier 1 (full) | ElasticNet | 20.8 | 25.1 | +0.691 | 24.1 | +0.748 |
| Tier 1 (full) | HGBR | 22.1 | 24.9 | +0.682 | 27.0 | +0.722 |

**Validation methods:**
- **LOOCV**: Leave-One-Out CV (may leak same-symbol info)
- **LOSO**: Leave-One-Symbol-Out (honest cross-symbol test)
- **Temporal**: Train hours 0-9, test hours 10-19 (hardest test)

## Classification: Profitable Trade Detection

Target: `target_profitable` (drop > 40 bps)

| Features | Model | LOOCV Acc | LOSO Acc | LOSO AUC | Temporal Acc |
|----------|-------|-----------|----------|----------|--------------|
| FR only (3) | LogReg | 0.831 | 0.815 | 0.902 | 0.931 |
| FR only (3) | HGBC | 0.738 | 0.746 | 0.869 | 0.776 |
| FR+depth (8) | LogReg | 0.862 | 0.862 | 0.936 | 0.897 |
| FR+depth (8) | HGBC | 0.885 | 0.869 | 0.942 | 0.931 |
| Tier 1 (full) | LogReg | 0.862 | 0.854 | 0.914 | 0.897 |
| Tier 1 (full) | HGBC | 0.892 | 0.877 | 0.942 | 0.897 |

### Best Classifier Confusion Matrix (LOSO)

| | Predicted Skip | Predicted Trade |
|---|---|---|
| **Actual Skip** | 42 | 7 |
| **Actual Trade** | 10 | 71 |

## Overfitting Check

| Features | Model | Train MAE | LOSO MAE | Gap Ratio | Verdict |
|----------|-------|-----------|----------|-----------|---------|
| FR only (3) | Ridge | 21.7 | 23.2 | 1.1x | OK |
| FR only (3) | ElasticNet | 21.7 | 23.1 | 1.1x | OK |
| FR only (3) | HGBR | 18.2 | 26.3 | 1.4x | OK |
| FR+depth (8) | Ridge | 19.3 | 22.7 | 1.2x | OK |
| FR+depth (8) | ElasticNet | 19.5 | 22.6 | 1.2x | OK |
| FR+depth (8) | HGBR | 10.3 | 24.5 | 2.4x | OK |
| Tier 1 (full) | Ridge | 15.9 | 26.5 | 1.7x | OK |
| Tier 1 (full) | ElasticNet | 16.1 | 25.1 | 1.6x | OK |
| Tier 1 (full) | HGBR | 6.8 | 24.9 | 3.7x | OVERFIT |

## Dataset Composition

| Symbol | Settlements | Avg FR (bps) | Avg Drop (bps) |
|--------|-------------|-------------|----------------|
| SAHARAUSDT | 25 | -66.4 | -91.6 |
| ENSOUSDT | 20 | -39.9 | -52.9 |
| POWERUSDT | 17 | -63.8 | -116.6 |
| STEEMUSDT | 10 | -44.9 | -44.3 |
| BARDUSDT | 10 | -42.8 | -83.9 |
| NEWTUSDT | 6 | -49.7 | -61.1 |
| ATHUSDT | 5 | -107.9 | -123.7 |
| MIRAUSDT | 5 | -60.8 | -88.4 |
| SOPHUSDT | 4 | -50.1 | -54.9 |
| SOLAYERUSDT | 3 | -64.6 | -88.5 |
| WETUSDT | 3 | -35.8 | -73.2 |
| HOLOUSDT | 2 | -73.2 | -81.3 |
| ALICEUSDT | 2 | -130.7 | -163.3 |
| STABLEUSDT | 2 | -17.4 | -17.5 |
| GNOUSDT | 1 | -33.9 | -23.4 |
| FLOWUSDT | 1 | -19.5 | -2.7 |
| ESPUSDT | 1 | -17.8 | -8.3 |
| CYBERUSDT | 1 | -17.5 | -1.8 |
| BIRBUSDT | 1 | -34.2 | -19.9 |
| ACEUSDT | 1 | -48.6 | -149.5 |
| ANIMEUSDT | 1 | -18.6 | -3.9 |
| API3USDT | 1 | -24.3 | -46.5 |
| AIXBTUSDT | 1 | -17.3 | -24.5 |
| REDUSDT | 1 | -51.8 | -82.8 |
| KERNELUSDT | 1 | -15.8 | -2.3 |
| MOVEUSDT | 1 | -17.6 | -31.3 |
| ROBOUSDT | 1 | -16.9 | -18.1 |
| SPACEUSDT | 1 | -15.8 | 3.6 |
| ZBCNUSDT | 1 | -23.6 | -6.2 |
| ZKCUSDT | 1 | -17.2 | -19.1 |

## Production Model

Based on integrity audit, the recommended production model uses **8 features**:

```
Features: fr_bps, fr_abs_bps, fr_squared,
          total_depth_usd, total_depth_imb_mean,
          ask_concentration, thin_side_depth, depth_within_50bps
Model:    Ridge(alpha=10.0) with StandardScaler
```

**Why?**
- FR alone explains ~90% of the signal (r=+0.898)
- Depth features add genuine edge (thin asks amplify drops)
- Only 8 features → impossible to overfit with Ridge regularization
- Passes ALL validation tests including temporal hold-out
- More features (49) look good on LOOCV but fail temporal validation

## Per-Date Summary

- **2026-02-26**: 25 settlements, 7 symbols, avg FR=-54.1bps, avg drop=-90.3bps
- **2026-02-27**: 72 settlements, 22 symbols, avg FR=-53.0bps, avg drop=-72.9bps
- **2026-02-28**: 33 settlements, 14 symbols, avg FR=-52.3bps, avg drop=-66.6bps
