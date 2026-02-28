# Microstructure Exit ML Research — Findings

**Date:** 2026-02-28  
**Dataset:** 74,898 ticks (100ms intervals) from 130 settlements, 30 symbols  
**Script:** `research_exit_ml.py`

---

## Executive Summary

**ML exit signal works.** The model predicts "will price drop ≥5 bps more in next 1s?" with **AUC=0.74 on unseen symbols** (Leave-One-Symbol-Out). When used as an exit trigger, it captures **+5,115 bps total** vs +4,031 for fixed T+5s exit — a **+27% improvement**.

But there's a catch: the model is somewhat overfit (train AUC=0.87 vs test AUC=0.74), and the improvement over a simple fixed T+10s exit (+4,411) is more modest (+16%).

---

## ML Model Results

| Model | Train AUC | Test AUC | LOSO (symbol) AUC | Overfit Gap |
|-------|-----------|----------|-------------------|-------------|
| LogReg | 0.768 | **0.732** | — | 0.037 |
| HGBC_light | 0.874 | **0.735** | **0.743** | 0.139 |
| HGBC_deep | 0.926 | 0.729 | — | 0.197 |

- **LogReg has lowest overfit** (0.037 gap) and nearly matches HGBC on test
- **HGBC_light is best on LOSO** (honest cross-symbol): AUC=0.743
- Deeper model overfits more without improving test AUC

### AUC by Time Phase (Test Set, HGBC_light)

| Phase | AUC | N | Positive Rate |
|-------|-----|---|---------------|
| 0-1s | 0.714 | 339 | 44.5% |
| 1-5s | 0.728 | 1,560 | 41.5% |
| 5-10s | 0.758 | 1,950 | 27.5% |
| 10-30s | 0.720 | 7,800 | 25.1% |
| 30-60s | 0.738 | 10,963 | 24.6% |

**Best prediction at T+5-10s** (AUC=0.758) — right when the sell wave is transitioning. Early phase (0-1s) is noisier.

---

## Top Features (Permutation Importance)

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| **distance_from_low_bps** | +0.048 | How far above running minimum — key signal |
| **running_min_bps** | +0.028 | Depth of drop so far — deeper = more likely to continue |
| **price_velocity_1s** | +0.019 | Speed of price change — momentum |
| **trade_rate_2s** | +0.018 | Trading intensity — high rate = more pressure |
| **time_since_new_low_ms** | +0.018 | Time since last new low — exhaustion signal |
| **ob1_imbalance** | +0.011 | Bid/ask imbalance — demand signal |
| **trade_rate_1s** | +0.009 | Short-term trade rate |
| **avg_size_500ms** | +0.009 | Average trade size — institutional flow |
| **avg_size_1s** | +0.009 | Average trade size 1s |
| **t_seconds** | +0.008 | Time since settlement |

**Key insight:** The model mainly uses price-based features (distance from low, velocity) and trade flow features (rate, size). The orderbook imbalance adds genuine signal. FR magnitude (static) is surprisingly NOT in the top 10 — the dynamic microstructure features dominate.

---

## Backtest: Exit Strategy Comparison

| Strategy | Trades | Avg PnL | Median PnL | Win Rate | Total PnL |
|----------|--------|---------|------------|----------|-----------|
| **oracle** (perfect) | 130 | +81.5 | +55.1 | 88% | **+10,589** |
| **ml_exit P<0.30** | 130 | **+39.3** | +13.7 | 65% | **+5,115** |
| ml_exit P<0.40 | 130 | +38.1 | +16.1 | 67% | +4,956 |
| ml_exit P<0.50 | 130 | +37.5 | +17.3 | 66% | +4,877 |
| fixed_10s | 130 | +33.9 | +17.4 | **72%** | +4,411 |
| fixed_30s | 130 | +32.0 | +11.5 | 63% | +4,155 |
| fixed_5s (current) | 130 | +31.0 | +15.5 | 67% | +4,031 |
| ml_plus_trail | 130 | +26.3 | +15.0 | 69% | +3,420 |
| trailing_15bps | 130 | +23.2 | +13.1 | 66% | +3,013 |

### Key Takeaways

1. **Oracle shows +10,589 bps** — there's a massive ceiling to reach (2.6x our current strategy)
2. **ML exit (P<0.30) is the best strategy: +5,115 bps** (+27% over fixed T+5s)
3. **Fixed T+10s is surprisingly good**: +4,411 (nearly as good as ML with zero complexity)
4. **Trailing stop is the WORST**: +3,013 (noise triggers early exit before the real bottom)
5. **ML + trailing stop combo is worse than ML alone**: the trailing stop hurts more than helps

### Why ML exit beats fixed timing:
- On high-FR settlements, ML holds longer (correctly predicting more drop ahead)
- On low-FR settlements, ML exits early (correctly predicting exhaustion)
- Net effect: ~+8 bps/trade over fixed T+5s, ~+5 bps/trade over fixed T+10s

### Why trailing stop is bad:
- Post-settlement price is noisy — 15 bps bounces are common even before the real bottom
- Trailing stop exits at the first bounce, missing the continued decline

---

## Honest Assessment

**Is the ML exit worth deploying?**

| Metric | Fixed T+10s | ML Exit (P<0.30) | Difference |
|--------|------------|------------------|------------|
| Avg PnL/trade | +33.9 bps | +39.3 bps | **+5.4 bps** |
| Total PnL | +4,411 | +5,115 | **+704 bps** |
| Win Rate | 72% | 65% | -7% |
| Complexity | Zero | High | — |

The ML adds **+5.4 bps/trade** on average. At $1,000 notional, that's ~$0.54/trade. The win rate is actually lower (65% vs 72%) because ML holds longer on some trades and they reverse.

**Recommendation:**
1. **Quick win: Change fixed exit from T+5.5s to T+10s** — this alone adds +2.9 bps/trade for zero effort
2. **ML exit: Promising but needs more data** — AUC=0.74 is decent but overfit gap is 0.14
3. **Trailing stops: Do NOT use** — they hurt performance

---

## Production Implementation Notes

If deploying ML exit:
- Model: HGBC_light (200 trees, depth 5, min_leaf 50)
- Features: 39 features computed every 100ms
- Decision: exit when P(further_drop) < 0.30
- Minimum hold: 500ms (don't exit in first 500ms regardless)
- Latency budget: ~5-15ms per cycle (plenty fast for 100ms ticks)
- Fallback: hard timeout at T+60s
