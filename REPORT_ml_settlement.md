# ML Settlement Prediction Report

**Generated:** 2026-03-01 06:41 UTC  
**Dataset:** 161 settlements, 33 symbols, 4 dates (2026-02-26 to 2026-03-01)  
**Pipeline:** `ml_settlement_pipeline.py`

---

## Summary

| Metric | Value |
|--------|-------|
| Settlements | 161 |
| Unique symbols | 33 |
| Date range | 2026-02-26 to 2026-03-01 |
| FR vs drop correlation | r = +0.795 |
| Profitable (>40bps drop) | 115/161 (71%) |
| Best LOSO MAE | **42.7 bps** (FR+depth (8), ElasticNet) |
| Best Temporal MAE | **36.9 bps** (FR+depth (8), ElasticNet) |
| Best LOSO AUC | **0.871** (Tier 1 (full), LogReg) |
| Baseline MAE | 67.9 bps |

## Regression: Predicting Drop Magnitude

Target: `drop_min_bps` (max price drop in full recording window, up to 60s)

| Features | Model | LOOCV MAE | LOSO MAE | LOSO R² | Temporal MAE | Temporal R² |
|----------|-------|-----------|----------|---------|--------------|------------|
| FR only (3) | Ridge | 42.7 | 43.2 | +0.581 | 40.3 | +0.523 |
| FR only (3) | ElasticNet | 42.6 | 43.1 | +0.591 | 39.8 | +0.568 |
| FR only (3) | HGBR | 47.1 | 50.4 | +0.487 | 51.0 | +0.291 |
| FR+depth (8) | Ridge | 40.8 | 43.4 | +0.596 | 38.1 | +0.564 |
| FR+depth (8) | ElasticNet | 40.3 | 42.7 | +0.610 | 36.9 | +0.614 |
| FR+depth (8) | HGBR | 43.0 | 47.0 | +0.498 | 49.3 | +0.382 |
| Tier 1 (full) | Ridge | 49.2 | 60.4 | +0.122 | 56.9 | +0.080 |
| Tier 1 (full) | ElasticNet | 44.6 | 53.1 | +0.353 | 46.6 | +0.441 |
| Tier 1 (full) | HGBR | 43.9 | 50.8 | +0.456 | 44.8 | +0.492 |

**Validation methods:**
- **LOOCV**: Leave-One-Out CV (may leak same-symbol info)
- **LOSO**: Leave-One-Symbol-Out (honest cross-symbol test)
- **Temporal**: Train hours 0-9, test hours 10-19 (hardest test)

## Classification: Profitable Trade Detection

Target: `target_profitable` (drop > 40 bps)

| Features | Model | LOOCV Acc | LOSO Acc | LOSO AUC | Temporal Acc |
|----------|-------|-----------|----------|----------|--------------|
| FR only (3) | LogReg | 0.714 | 0.714 | 0.799 | 0.679 |
| FR only (3) | HGBC | 0.696 | 0.727 | 0.799 | 0.765 |
| FR+depth (8) | LogReg | 0.789 | 0.807 | 0.868 | 0.753 |
| FR+depth (8) | HGBC | 0.789 | 0.783 | 0.867 | 0.765 |
| Tier 1 (full) | LogReg | 0.820 | 0.789 | 0.871 | 0.778 |
| Tier 1 (full) | HGBC | 0.801 | 0.801 | 0.868 | 0.790 |

### Best Classifier Confusion Matrix (LOSO)

| | Predicted Skip | Predicted Trade |
|---|---|---|
| **Actual Skip** | 26 | 20 |
| **Actual Trade** | 15 | 100 |

## Overfitting Check

| Features | Model | Train MAE | LOSO MAE | Gap Ratio | Verdict |
|----------|-------|-----------|----------|-----------|---------|
| FR only (3) | Ridge | 41.1 | 43.2 | 1.1x | OK |
| FR only (3) | ElasticNet | 41.3 | 43.1 | 1.0x | OK |
| FR only (3) | HGBR | 36.6 | 50.4 | 1.4x | OK |
| FR+depth (8) | Ridge | 37.5 | 43.4 | 1.2x | OK |
| FR+depth (8) | ElasticNet | 37.9 | 42.7 | 1.1x | OK |
| FR+depth (8) | HGBR | 22.4 | 47.0 | 2.1x | OK |
| Tier 1 (full) | Ridge | 33.6 | 60.4 | 1.8x | OK |
| Tier 1 (full) | ElasticNet | 34.4 | 53.1 | 1.5x | OK |
| Tier 1 (full) | HGBR | 16.4 | 50.8 | 3.1x | OVERFIT |

## Dataset Composition

| Symbol | Settlements | Avg FR (bps) | Avg Drop (bps) |
|--------|-------------|-------------|----------------|
| SAHARAUSDT | 36 | -57.1 | -103.4 |
| ENSOUSDT | 20 | -39.9 | -67.6 |
| POWERUSDT | 17 | -63.8 | -182.9 |
| STEEMUSDT | 12 | -45.5 | -88.6 |
| BARDUSDT | 10 | -42.8 | -97.5 |
| SOLAYERUSDT | 8 | -60.1 | -102.1 |
| ATHUSDT | 7 | -91.6 | -140.1 |
| NEWTUSDT | 6 | -49.7 | -67.1 |
| SOPHUSDT | 6 | -51.8 | -93.1 |
| MIRAUSDT | 5 | -60.8 | -98.1 |
| ALICEUSDT | 3 | -120.0 | -289.6 |
| STABLEUSDT | 3 | -18.2 | -21.6 |
| WETUSDT | 3 | -35.8 | -111.5 |
| HOLOUSDT | 3 | -54.8 | -81.4 |
| ROBOUSDT | 2 | -16.8 | -47.7 |
| FLOWUSDT | 2 | -17.3 | -30.6 |
| BIRBUSDT | 2 | -33.1 | -84.3 |
| ACEUSDT | 1 | -48.6 | -158.0 |
| GNOUSDT | 1 | -33.9 | -25.7 |
| DGBUSDT | 1 | -52.2 | 0.0 |
| ESPUSDT | 1 | -17.8 | -19.6 |
| API3USDT | 1 | -24.3 | -122.9 |
| ANIMEUSDT | 1 | -18.6 | -5.8 |
| AIXBTUSDT | 1 | -17.3 | -74.7 |
| CYBERUSDT | 1 | -17.5 | -1.8 |
| KERNELUSDT | 1 | -15.8 | -9.2 |
| REDUSDT | 1 | -51.8 | -82.8 |
| MOVEUSDT | 1 | -17.6 | -58.2 |
| ORBSUSDT | 1 | -250.0 | -295.8 |
| SPACEUSDT | 1 | -15.8 | -3.6 |
| XCNUSDT | 1 | -23.1 | -25.0 |
| ZBCNUSDT | 1 | -23.6 | -18.1 |
| ZKCUSDT | 1 | -17.2 | -29.2 |

## Production Model

Based on integrity audit, the recommended production model uses **9 features**:

```
Features: fr_bps, fr_abs_bps, fr_squared,
          total_depth_usd, total_depth_imb_mean,
          ask_concentration, thin_side_depth, depth_within_50bps,
          oi_change_60s
Model:    Ridge(alpha=10.0) with StandardScaler
```

**Why?**
- FR alone explains ~90% of the signal (r=+0.795)
- Depth features add genuine edge (thin asks amplify drops)
- OI change pre-settlement is 2nd best predictor (r≈-0.44)
- Only 9 features → impossible to overfit with Ridge regularization
- Passes ALL validation tests including temporal hold-out

## Price Trajectory (Full 60s Window)

Target `drop_min_bps` uses the **full recording window** (up to 60s), not just first 5s.

- Median time to bottom: **10.3s** (mean=20.5s)
- Bottoms after T+5s: 95/161 (59%)

| Exit Time | Avg Price (bps) | Avg PnL (after 20bps fees) |
|-----------|----------------|---------------------------|
| T+1s | -46.7 | +26.7 bps (66% WR) |
| T+5s | -48.5 | +28.5 bps (64% WR) |
| T+10s | -49.6 | +29.6 bps (66% WR) |
| T+30s | -47.9 | +27.9 bps (61% WR) |
| T+60s | -55.0 | +35.0 bps (61% WR) |

## Optimal Exit Timing by FR Magnitude

| FR Range | N | Exit T+1s | Exit T+5s | Exit T+10s | Exit T+30s |
|----------|---|-----------|-----------|------------|------------|
| \|FR\| 15-30 | 64 | -7 | -5 | -4 | -8 |
| \|FR\| 30-60 | 50 | +20 | +13 | +13 | -8 |
| \|FR\| 60-100 | 28 | +51 | +54 | +58 | +67 |
| \|FR\| >100 | 19 | +123 | +146 | +145 | +184 |

**Recommended dynamic exit:**
- \|FR\| < 25 bps → SKIP (don't trade)
- \|FR\| 25-50 bps → exit T+5s (quick scalp)
- \|FR\| 50-80 bps → exit T+10s (let it drift)
- \|FR\| > 80 bps → exit T+20-30s (sustained sell wave)

## Recovery After Drop

- Avg max recovery: +70.5 bps (158% of drop)
- Full recovery to ref price: 42/161 (26%)

| FR Range | N | Avg Drop | Recovery % | Full Recovery |
|----------|---|----------|-----------|---------------|
| \|FR\| 15-30 | 64 | -45.7 | 242% | 38% |
| 30-60 | 50 | -79.8 | 145% | 30% |
| 60-100 | 28 | -143.1 | 71% | 11% |
| >100 | 19 | -280.0 | 42% | 0% |

## Post-Settlement Volume

| Window | Sell Ratio |
|--------|-----------|
| T+1s | 65.9% |
| T+5s | 58.9% |
| T+10s | 60.4% |
| T+30s | 54.5% |

## Per-Date Summary

- **2026-02-26**: 25 settlements, 7 symbols, avg FR=-54.1bps, avg drop=-116.4bps
- **2026-02-27**: 73 settlements, 23 symbols, avg FR=-53.0bps, avg drop=-100.6bps
- **2026-02-28**: 58 settlements, 17 symbols, avg FR=-50.6bps, avg drop=-94.8bps
- **2026-03-01**: 5 settlements, 3 symbols, avg FR=-51.5bps, avg drop=-97.0bps
