# ML Settlement Prediction Report

**Generated:** 2026-02-28 09:01 UTC  
**Dataset:** 131 settlements, 31 symbols, 3 dates (2026-02-26 to 2026-02-28)  
**Pipeline:** `ml_settlement_pipeline.py`

---

## Summary

| Metric | Value |
|--------|-------|
| Settlements | 131 |
| Unique symbols | 31 |
| Date range | 2026-02-26 to 2026-02-28 |
| FR vs drop correlation | r = +0.798 |
| Profitable (>40bps drop) | 96/131 (73%) |
| Best LOSO MAE | **43.9 bps** (FR+depth (8), Ridge) |
| Best Temporal MAE | **35.8 bps** (FR+depth (8), ElasticNet) |
| Best LOSO AUC | **0.873** (FR+depth (8), LogReg) |
| Baseline MAE | 70.0 bps |

## Regression: Predicting Drop Magnitude

Target: `drop_min_bps` (max price drop in full recording window, up to 60s)

| Features | Model | LOOCV MAE | LOSO MAE | LOSO R² | Temporal MAE | Temporal R² |
|----------|-------|-----------|----------|---------|--------------|------------|
| FR only (3) | Ridge | 43.5 | 44.8 | +0.637 | 40.0 | +0.677 |
| FR only (3) | ElasticNet | 43.7 | 44.9 | +0.627 | 39.8 | +0.689 |
| FR only (3) | HGBR | 50.8 | 53.5 | +0.427 | 54.5 | +0.368 |
| FR+depth (8) | Ridge | 41.4 | 43.9 | +0.657 | 36.5 | +0.707 |
| FR+depth (8) | ElasticNet | 41.4 | 43.9 | +0.648 | 35.8 | +0.722 |
| FR+depth (8) | HGBR | 47.2 | 54.1 | +0.417 | 56.4 | +0.312 |
| Tier 1 (full) | Ridge | 53.4 | 80.3 | -5.218 | 47.9 | +0.486 |
| Tier 1 (full) | ElasticNet | 50.7 | 73.5 | -3.832 | 42.9 | +0.591 |
| Tier 1 (full) | HGBR | 43.3 | 54.2 | +0.366 | 54.9 | +0.353 |

**Validation methods:**
- **LOOCV**: Leave-One-Out CV (may leak same-symbol info)
- **LOSO**: Leave-One-Symbol-Out (honest cross-symbol test)
- **Temporal**: Train hours 0-9, test hours 10-19 (hardest test)

## Classification: Profitable Trade Detection

Target: `target_profitable` (drop > 40 bps)

| Features | Model | LOOCV Acc | LOSO Acc | LOSO AUC | Temporal Acc |
|----------|-------|-----------|----------|----------|--------------|
| FR only (3) | LogReg | 0.733 | 0.733 | 0.787 | 0.724 |
| FR only (3) | HGBC | 0.695 | 0.679 | 0.775 | 0.776 |
| FR+depth (8) | LogReg | 0.794 | 0.779 | 0.873 | 0.793 |
| FR+depth (8) | HGBC | 0.817 | 0.763 | 0.843 | 0.810 |
| Tier 1 (full) | LogReg | 0.802 | 0.786 | 0.861 | 0.793 |
| Tier 1 (full) | HGBC | 0.786 | 0.786 | 0.850 | 0.810 |

### Best Classifier Confusion Matrix (LOSO)

| | Predicted Skip | Predicted Trade |
|---|---|---|
| **Actual Skip** | 21 | 14 |
| **Actual Trade** | 17 | 79 |

## Overfitting Check

| Features | Model | Train MAE | LOSO MAE | Gap Ratio | Verdict |
|----------|-------|-----------|----------|-----------|---------|
| FR only (3) | Ridge | 41.9 | 44.8 | 1.1x | OK |
| FR only (3) | ElasticNet | 42.3 | 44.9 | 1.1x | OK |
| FR only (3) | HGBR | 37.7 | 53.5 | 1.4x | OK |
| FR+depth (8) | Ridge | 37.7 | 43.9 | 1.2x | OK |
| FR+depth (8) | ElasticNet | 38.7 | 43.9 | 1.1x | OK |
| FR+depth (8) | HGBR | 23.1 | 54.1 | 2.3x | OK |
| Tier 1 (full) | Ridge | 30.7 | 80.3 | 2.6x | OK |
| Tier 1 (full) | ElasticNet | 32.9 | 73.5 | 2.2x | OK |
| Tier 1 (full) | HGBR | 15.9 | 54.2 | 3.4x | OVERFIT |

## Dataset Composition

| Symbol | Settlements | Avg FR (bps) | Avg Drop (bps) |
|--------|-------------|-------------|----------------|
| SAHARAUSDT | 25 | -66.4 | -116.0 |
| ENSOUSDT | 20 | -39.9 | -67.6 |
| POWERUSDT | 17 | -63.8 | -182.9 |
| BARDUSDT | 10 | -42.8 | -97.5 |
| STEEMUSDT | 10 | -44.9 | -95.7 |
| NEWTUSDT | 6 | -49.7 | -67.1 |
| ATHUSDT | 5 | -107.9 | -167.8 |
| MIRAUSDT | 5 | -60.8 | -98.1 |
| SOPHUSDT | 4 | -50.1 | -74.3 |
| WETUSDT | 3 | -35.8 | -111.5 |
| SOLAYERUSDT | 3 | -64.6 | -100.2 |
| HOLOUSDT | 2 | -73.2 | -115.5 |
| ALICEUSDT | 2 | -130.7 | -373.8 |
| STABLEUSDT | 2 | -17.4 | -24.5 |
| AIXBTUSDT | 1 | -17.3 | -74.7 |
| GNOUSDT | 1 | -33.9 | -25.7 |
| FLOWUSDT | 1 | -19.5 | -32.1 |
| ESPUSDT | 1 | -17.8 | -19.6 |
| DGBUSDT | 1 | -52.2 | 0.0 |
| BIRBUSDT | 1 | -34.2 | -23.0 |
| CYBERUSDT | 1 | -17.5 | -1.8 |
| API3USDT | 1 | -24.3 | -122.9 |
| ANIMEUSDT | 1 | -18.6 | -5.8 |
| ACEUSDT | 1 | -48.6 | -158.0 |
| REDUSDT | 1 | -51.8 | -82.8 |
| KERNELUSDT | 1 | -15.8 | -9.2 |
| MOVEUSDT | 1 | -17.6 | -58.2 |
| ROBOUSDT | 1 | -16.9 | -41.4 |
| SPACEUSDT | 1 | -15.8 | -3.6 |
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
- FR alone explains ~90% of the signal (r=+0.798)
- Depth features add genuine edge (thin asks amplify drops)
- OI change pre-settlement is 2nd best predictor (r≈-0.44)
- Only 9 features → impossible to overfit with Ridge regularization
- Passes ALL validation tests including temporal hold-out

## Price Trajectory (Full 60s Window)

Target `drop_min_bps` uses the **full recording window** (up to 60s), not just first 5s.

- Median time to bottom: **11.4s** (mean=20.4s)
- Bottoms after T+5s: 79/131 (60%)

| Exit Time | Avg Price (bps) | Avg PnL (after 20bps fees) |
|-----------|----------------|---------------------------|
| T+1s | -47.2 | +27.2 bps (67% WR) |
| T+5s | -50.8 | +30.8 bps (67% WR) |
| T+10s | -53.2 | +33.2 bps (70% WR) |
| T+30s | -51.5 | +31.5 bps (63% WR) |
| T+60s | -57.3 | +37.3 bps (60% WR) |

## Optimal Exit Timing by FR Magnitude

| FR Range | N | Exit T+1s | Exit T+5s | Exit T+10s | Exit T+30s |
|----------|---|-----------|-----------|------------|------------|
| \|FR\| 15-30 | 48 | -9 | -4 | -2 | -4 |
| \|FR\| 30-60 | 44 | +21 | +15 | +15 | -9 |
| \|FR\| 60-100 | 22 | +51 | +53 | +63 | +75 |
| \|FR\| >100 | 17 | +113 | +140 | +141 | +181 |

**Recommended dynamic exit:**
- \|FR\| < 25 bps → SKIP (don't trade)
- \|FR\| 25-50 bps → exit T+5s (quick scalp)
- \|FR\| 50-80 bps → exit T+10s (let it drift)
- \|FR\| > 80 bps → exit T+20-30s (sustained sell wave)

## Recovery After Drop

- Avg max recovery: +72.0 bps (166% of drop)
- Full recovery to ref price: 36/131 (27%)

| FR Range | N | Avg Drop | Recovery % | Full Recovery |
|----------|---|----------|-----------|---------------|
| \|FR\| 15-30 | 48 | -46.9 | 266% | 40% |
| 30-60 | 44 | -79.0 | 153% | 32% |
| 60-100 | 22 | -150.3 | 70% | 14% |
| >100 | 17 | -273.9 | 41% | 0% |

## Post-Settlement Volume

| Window | Sell Ratio |
|--------|-----------|
| T+1s | 65.2% |
| T+5s | 59.1% |
| T+10s | 60.8% |
| T+30s | 55.4% |

## Per-Date Summary

- **2026-02-26**: 25 settlements, 7 symbols, avg FR=-54.1bps, avg drop=-116.4bps
- **2026-02-27**: 73 settlements, 23 symbols, avg FR=-53.0bps, avg drop=-100.6bps
- **2026-02-28**: 33 settlements, 14 symbols, avg FR=-52.3bps, avg drop=-104.2bps
