# Walk-Forward ML Backtest: First Period Analysis

## Design

**Zero lookahead.** Train on 12 months of data (Jan-Dec 2024), 30-day purge gap, predict on completely unseen 30-day trade window (Jan 25 - Feb 23, 2025).

The full pipeline runs from scratch at each rebalance:
1. **Tier 2** — Stability scan on 360d selection window
2. **Tier 4** — Hierarchical clustering → ~321 representative features
3. **Tier 5** — Single-feature expanding-window WFO (120d min train, 45d test)
4. **Tier 6** — Multi-feature Ridge/LogReg (top 20 features per target)
5. **Predict** — Apply trained models to 30-day trade window

No information from the trade window is used at any stage.

## Period 1 Results: SOLUSDT 4h

| Metric | Value |
|--------|-------|
| **Selection** | 2024-01-01 → 2024-12-25 (2160 candles, 360d) |
| **Trade window** | 2025-01-25 → 2025-02-23 (180 candles, 30d) |
| **Tier 2 survivors** | 1025 features, 35005 (feat,tgt) pairs |
| **Tier 4 cluster reps** | 321 |
| **Tier 5 survivors** | 5600+ (feat,tgt) pairs |
| **Targets predicted** | 51 |
| **Positive on trade window** | 39/51 (76%) |
| **Tier A (score > 0.03)** | 37 targets |
| **Mean positive score** | +0.20 |
| **Pipeline time** | 39 minutes |

## Cross-Validation Against Research Pipeline

The WFO Period 1 scores were compared against the 3-split research pipeline (which used the full 2-year dataset with different methodology):

- **Spearman rank correlation: 0.81** — target rankings are nearly identical
- **Pearson correlation: 0.74** — absolute scores are well-correlated

The same targets that were Tier A in the research pipeline are Tier A in the WFO, despite using completely different data splits and methodology. This confirms the signal is real, not an artifact of any particular split design.

## 30 Cross-Validated Predictable Targets

These targets are positive in ALL three checks:
1. WFO Period 1 (train 2024, predict Jan-Feb 2025) — score > 0.03
2. SOL research pipeline (3-split, all splits positive) — mean holdout > 0.03
3. XRP research pipeline (3-split, all splits positive) — mean holdout > 0.03

See `predictable_targets.json` for the full list with feature assignments.

### Tier 1: Structural (AUC_dev > 0.25)

| Target | WFO P1 | SOL Research | XRP Research |
|--------|--------|-------------|-------------|
| tgt_breakout_any_5 | +0.483 | +0.343 | +0.397 |
| tgt_breakout_any_10 | +0.478 | +0.354 | +0.465 |
| tgt_consolidation_5 | +0.377 | +0.367 | +0.375 |
| tgt_breakout_down_10 | +0.343 | +0.282 | +0.271 |
| tgt_breakout_up_5 | +0.320 | +0.315 | +0.329 |
| tgt_consolidation_3 | +0.318 | +0.356 | +0.345 |
| tgt_vol_expansion_10 | +0.317 | +0.327 | +0.317 |
| tgt_breakout_up_3 | +0.317 | +0.335 | +0.344 |
| tgt_crash_10 | +0.291 | +0.250 | +0.205 |
| tgt_breakout_down_3 | +0.288 | +0.315 | +0.320 |
| tgt_breakout_up_10 | +0.286 | +0.295 | +0.307 |
| tgt_breakout_down_5 | +0.284 | +0.305 | +0.298 |
| tgt_crash_5 | +0.264 | +0.196 | +0.132 |

### Tier 2: Events/Risk (AUC_dev 0.10–0.25)

| Target | WFO P1 | SOL Research | XRP Research |
|--------|--------|-------------|-------------|
| tgt_tail_event_3 | +0.241 | +0.206 | +0.177 |
| tgt_crash_3 | +0.237 | +0.194 | +0.084 |
| tgt_vol_expansion_5 | +0.234 | +0.249 | +0.245 |
| tgt_tail_event_5 | +0.234 | +0.229 | +0.212 |
| tgt_tail_event_1 | +0.207 | +0.174 | +0.089 |
| tgt_breakout_any_3 | +0.202 | +0.308 | +0.368 |
| tgt_profitable_long_1 | +0.192 | +0.191 | +0.216 |
| tgt_liquidation_cascade_5 | +0.188 | +0.203 | +0.114 |
| tgt_profitable_long_3 | +0.181 | +0.125 | +0.096 |
| tgt_profitable_short_1 | +0.175 | +0.194 | +0.209 |
| tgt_profitable_long_5 | +0.171 | +0.092 | +0.098 |
| tgt_profitable_short_5 | +0.140 | +0.096 | +0.097 |
| tgt_profitable_short_3 | +0.118 | +0.128 | +0.100 |
| tgt_alpha_1 | +0.112 | +0.144 | +0.106 |
| tgt_relative_ret_1 | +0.112 | +0.144 | +0.106 |

### Tier 3: Weak but Consistent (score 0.03–0.10)

| Target | WFO P1 | SOL Research | XRP Research |
|--------|--------|-------------|-------------|
| tgt_adverse_selection_1 | +0.085 | +0.080 | +0.102 |
| tgt_profitable_long_10 | +0.083 | +0.073 | +0.070 |
| tgt_profitable_short_10 | +0.075 | +0.075 | +0.063 |

## Feature Analysis

### Core Features (appear in ≥10 of 30 predictable targets)

| Feature | Targets | Signal Type |
|---------|---------|-------------|
| max_drawdown | 19/30 | Volatility/Activity |
| trade_rate_std | 15/30 | Volatility/Activity |
| wave_wavelength_z | 14/30 | Volatility/Activity |
| volume_per_second_std | 14/30 | Volatility/Activity |
| realized_vol | 13/30 | Volatility/Activity |
| price_path_length | 13/30 | Volatility/Activity |
| total_energy | 13/30 | Volatility/Activity |
| burstiness | 12/30 | Volatility/Activity |
| vol_price_feedback_z | 12/30 | Volatility/Activity |
| max_trades_per_second_z | 12/30 | Volatility/Activity |
| roll_spread | 12/30 | Volatility/Activity |
| is_swing_low | 12/30 | Price Structure |
| max_up_sweep_z | 11/30 | Volatility/Activity |
| trade_time_uniformity | 11/30 | Volatility/Activity |
| kinetic_energy | 11/30 | Volatility/Activity |
| eigen_ratio_z | 11/30 | Volatility/Activity |
| max_monotonic_run | 11/30 | Volatility/Activity |
| dist_to_resistance_5_bps | 11/30 | Price Structure |
| is_swing_high | 11/30 | Price Structure |
| atr_change_pct_14 | 10/30 | Volatility/Activity |

### Three Signal Families

**1. Volatility/Activity** (drives breakout, consolidation, crash, vol_expansion, tail events)
- `realized_vol`, `trade_rate_std`, `volume_per_second_std`, `price_path_length`
- `roll_spread`, `eigen_ratio_z`, `max_trades_per_second_z`
- `wave_wavelength_z`, `max_monotonic_run`, `total_energy`, `kinetic_energy`
- `burstiness`, `max_drawdown`, `max_drawup`, `atr_change_pct_14`

**2. Price Structure** (drives breakout direction, profitability)
- `close_to_vwap`, `dist_to_r1_bps`, `dist_to_support/resistance_*_bps`
- `is_swing_high`, `is_swing_low`, `cvd_close_vs_range`
- `time_to_high_pct`, `taylor_price_a0`

**3. Microstructure State** (drives profitability, alpha)
- `hilbert_std_amplitude`, `predator_follows_prey_z`, `trade_size_iqr`
- `support_touches_20`, `resistance_touches_20`, `breakout_net_10`

### Unpredictable Targets (DO NOT USE)

| Target | WFO P1 | Issue |
|--------|--------|-------|
| tgt_fill_prob_long_1 | -0.338 | Strongly anti-predictable |
| tgt_fill_prob_short_3/5 | -0.299 | Strongly anti-predictable |
| tgt_mean_reversion_5 | -0.215 | Anti-predictable |
| tgt_fill_prob_short_1 | -0.087 | Anti-predictable |
| tgt_mid_reversion_1 | -0.079 | Anti-predictable |
| tgt_ret_sign_3 | -0.037 | Not predictable |
| tgt_adverse_selection_5 | -0.019 | Not predictable |
| tgt_liquidation_cascade_3 | -0.019 | Period-dependent |
| tgt_ret_sign_10 | -0.015 | Not predictable |
| tgt_autocorr_break_5 | -0.011 | Not predictable |
| tgt_adverse_selection_3 | -0.007 | Not predictable |

## Key Takeaways

1. **30 targets are cross-validated predictable** across WFO Period 1, SOL research (3-split), and XRP research (3-split). These should be the target universe for production.

2. **121 unique features** are used across the 30 targets, but only **~20 core features** appear in ≥10 targets. The signal is concentrated.

3. **Two independent prediction systems** exist:
   - "What kind of move?" (volatility features → breakout/consolidation/crash)
   - "Which direction?" (price structure features → up/down breakout, profitability)

4. **Mean reversion is a trap** — looks good in some research splits but fails or reverses on individual periods.

5. **Return sign is not predictable** — microstructure features predict *what kind of move*, not *which direction* at the return level.
