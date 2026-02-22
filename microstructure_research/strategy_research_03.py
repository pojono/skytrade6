#!/usr/bin/env python3
"""
Strategy Research 03: Smarter approaches after initial screening failure.

Key changes from Research 02:
  1. Curated feature groups (20-50 features) instead of 700+
  2. Linear models (Ridge) alongside LightGBM — less overfitting
  3. Regime-conditional trading — only trade in high-vol or trending regimes
  4. 1h candles — stronger signal, less noise
  5. Longer training windows (60d)
  6. Expanding window (not rolling) — more data for training
"""

import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from strategy_ml_wfo import (
    load_features, split_features_targets, prepare_features,
    generate_wfo_splits, backtest_signals, print_metrics,
    print_wfo_summary, BacktestResult,
    train_lgbm_classifier, train_lgbm_regressor,
    MAKER_FEE, TAKER_FEE,
)


def mem_gb():
    return psutil.virtual_memory().used / 1024**3


# ---------------------------------------------------------------------------
# Curated Feature Groups
# ---------------------------------------------------------------------------

# Group 1: Core microstructure (most informative, least noisy)
FEATURES_CORE = [
    "return", "range", "realized_vol", "total_volume", "total_notional",
    "trade_count", "buy_ratio", "effective_spread_bps",
    "vwap_close_bps", "twap_close_bps",
    "mean_trade_size", "median_trade_size", "trade_size_skew",
    "buy_volume_ratio", "sell_volume_ratio",
    "cvd_normalized", "ofi_normalized",
    "high_low_ratio", "close_position_in_range",
]

# Group 2: Momentum / trend
FEATURES_MOMENTUM = [
    "return_reversal", "consecutive_same_direction",
    "return_z", "range_z", "realized_vol_z",
    "volume_surprise", "range_surprise",
    "cvd_normalized_z", "ofi_normalized_z",
    "buy_ratio_z", "effective_spread_bps_z",
]

# Group 3: Volatility / regime
FEATURES_VOLATILITY = [
    "realized_vol", "realized_vol_z",
    "range", "range_z",
    "gk_volatility", "parkinson_volatility",
    "effective_spread_bps", "effective_spread_bps_z",
    "trade_count", "trade_count_z",
    "high_low_ratio",
]

# Group 4: Order flow
FEATURES_ORDERFLOW = [
    "buy_ratio", "buy_volume_ratio", "sell_volume_ratio",
    "cvd_normalized", "ofi_normalized",
    "trade_size_skew", "trade_size_kurtosis",
    "large_trade_ratio", "small_trade_ratio",
    "buy_ratio_z", "cvd_normalized_z", "ofi_normalized_z",
    "vpin_estimate",
]

# Group 5: Price structure
FEATURES_PRICE = [
    "close_position_in_range", "high_before_low",
    "vwap_close_bps", "twap_close_bps",
    "poc_position_in_range",
    "close_vs_fair_value_bps",
    "return", "return_z",
]

# Combined curated set
FEATURES_CURATED = list(set(
    FEATURES_CORE + FEATURES_MOMENTUM + FEATURES_VOLATILITY +
    FEATURES_ORDERFLOW + FEATURES_PRICE
))


def get_available_features(X, feature_list):
    """Return only features that exist in X."""
    return [f for f in feature_list if f in X.columns]


# ---------------------------------------------------------------------------
# Strategy F: Ridge Regression on curated features
# ---------------------------------------------------------------------------
def run_strategy_f(df, X, targets, hold_bars=5, timeframe="15m",
                   min_pred_bps=3.0, use_expanding=True):
    """Ridge regression on curated features. Linear model = less overfitting."""
    tgt_name = f"tgt_pnl_long_bps_{hold_bars}"
    y = targets[tgt_name]

    feats = get_available_features(X, FEATURES_CURATED)
    print(f"\n{'='*60}")
    print(f"  Strategy F: Ridge Regression ({timeframe})")
    print(f"  Hold: {hold_bars} bars, min_pred: {min_pred_bps} bps")
    print(f"  Features: {len(feats)} curated")
    print(f"  Window: {'expanding' if use_expanding else 'rolling'}")
    print(f"{'='*60}")

    if use_expanding:
        splits = generate_wfo_splits(df.index, train_days=30, test_days=15,
                                      gap_days=3, step_days=15, min_train_rows=200)
    else:
        splits = generate_wfo_splits(df.index, train_days=45, test_days=15,
                                      gap_days=3, step_days=15, min_train_rows=200)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        if use_expanding:
            # Expanding: train from very start to train_end
            train_mask = df.index < split.train_end
        else:
            train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask, feats]
        y_train = y.reindex(X_train.index)
        X_test = X.loc[test_mask, feats]
        df_test = df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Standardize
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        # Ridge regression
        model = Ridge(alpha=10.0)
        model.fit(X_tr_scaled, y_train)

        pred = model.predict(X_te_scaled)

        # Generate signals
        signals = pd.Series(0, index=X_test.index, dtype=int)
        signals[pred > min_pred_bps] = 1
        signals[pred < -min_pred_bps] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        n_l = (signals == 1).sum()
        n_s = (signals == -1).sum()
        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades} (L={n_l},S={n_s}), "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%} [{elapsed:.0f}s]")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ---------------------------------------------------------------------------
# Strategy G: LightGBM on curated features only
# ---------------------------------------------------------------------------
def run_strategy_g(df, X, targets, hold_bars=5, timeframe="15m",
                   percentile_threshold=75):
    """LightGBM but ONLY on curated features (not 700+)."""
    tgt_name = f"tgt_pnl_long_bps_{hold_bars}"
    y = targets[tgt_name]

    feats = get_available_features(X, FEATURES_CURATED)
    print(f"\n{'='*60}")
    print(f"  Strategy G: LightGBM Curated ({timeframe})")
    print(f"  Hold: {hold_bars} bars, pct: {percentile_threshold}")
    print(f"  Features: {len(feats)} curated")
    print(f"{'='*60}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=15,
                                  gap_days=3, step_days=15, min_train_rows=200)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = df.index < split.train_end  # expanding window
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask, feats]
        y_train = y.reindex(X_train.index)
        X_test = X.loc[test_mask, feats]
        df_test = df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Train/val split
        sp = int(len(X_train) * 0.85)
        model = train_lgbm_regressor(
            X_train.iloc[:sp], y_train.iloc[:sp],
            X_train.iloc[sp:], y_train.iloc[sp:],
            params={"num_leaves": 15, "max_depth": 4, "min_child_samples": 100,
                    "reg_alpha": 1.0, "reg_lambda": 5.0, "learning_rate": 0.03},
        )

        pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        thresh = max(np.percentile(train_pred, percentile_threshold), 0)

        signals = pd.Series(0, index=X_test.index, dtype=int)
        signals[pred > thresh] = 1

        # Also predict short side
        tgt_short = f"tgt_pnl_short_bps_{hold_bars}"
        ys = targets[tgt_short].reindex(X_train.index)
        if ys.notna().sum() > 100:
            model_s = train_lgbm_regressor(
                X_train.iloc[:sp], ys.iloc[:sp],
                X_train.iloc[sp:], ys.iloc[sp:],
                params={"num_leaves": 15, "max_depth": 4, "min_child_samples": 100,
                        "reg_alpha": 1.0, "reg_lambda": 5.0, "learning_rate": 0.03},
            )
            pred_s = model_s.predict(X_test)
            train_pred_s = model_s.predict(X_train)
            thresh_s = max(np.percentile(train_pred_s, percentile_threshold), 0)

            # Short where short model is confident AND long model is not
            for i in range(len(X_test)):
                if pred_s[i] > thresh_s and pred[i] < thresh:
                    signals.iloc[i] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        n_l = (signals == 1).sum()
        n_s = (signals == -1).sum()
        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades} (L={n_l},S={n_s}), "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}, "
              f"thresh={thresh:.1f} [{elapsed:.0f}s]")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ---------------------------------------------------------------------------
# Strategy H: Regime-conditional (only trade in high-vol)
# ---------------------------------------------------------------------------
def run_strategy_h(df, X, targets, hold_bars=5, timeframe="15m",
                   vol_quantile=0.6):
    """Only trade when current volatility is above median.
    Hypothesis: signals are stronger in volatile markets.
    """
    tgt_name = f"tgt_pnl_long_bps_{hold_bars}"
    y = targets[tgt_name]

    feats = get_available_features(X, FEATURES_CURATED)
    print(f"\n{'='*60}")
    print(f"  Strategy H: Regime-Conditional ({timeframe})")
    print(f"  Hold: {hold_bars} bars, vol_quantile: {vol_quantile}")
    print(f"  Features: {len(feats)} curated")
    print(f"{'='*60}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=15,
                                  gap_days=3, step_days=15, min_train_rows=200)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = df.index < split.train_end
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask, feats]
        y_train = y.reindex(X_train.index)
        X_test = X.loc[test_mask, feats]
        df_test = df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Compute vol regime filter on test data (using PAST vol only — no lookahead)
        vol_col = "realized_vol" if "realized_vol" in df.columns else "range"
        vol_series = df[vol_col]
        # Rolling quantile up to current bar (no lookahead)
        vol_rolling_q = vol_series.expanding(min_periods=20).quantile(vol_quantile)
        is_high_vol = vol_series > vol_rolling_q

        # Only keep test rows where vol is high
        test_high_vol = is_high_vol.reindex(X_test.index).fillna(False)

        # Train model on ALL training data (not just high-vol)
        sp = int(len(X_train) * 0.85)
        model = train_lgbm_regressor(
            X_train.iloc[:sp], y_train.iloc[:sp],
            X_train.iloc[sp:], y_train.iloc[sp:],
            params={"num_leaves": 15, "max_depth": 4, "min_child_samples": 100,
                    "reg_alpha": 1.0, "reg_lambda": 5.0, "learning_rate": 0.03},
        )

        pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        thresh = max(np.percentile(train_pred, 70), 0)

        signals = pd.Series(0, index=X_test.index, dtype=int)
        for i in range(len(X_test)):
            if not test_high_vol.iloc[i]:
                continue  # skip low-vol periods
            if pred[i] > thresh:
                signals.iloc[i] = 1
            elif pred[i] < -thresh:
                signals.iloc[i] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        n_l = (signals == 1).sum()
        n_s = (signals == -1).sum()
        n_filtered = (~test_high_vol).sum()
        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades} (L={n_l},S={n_s}), "
              f"filtered={n_filtered}, "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%} [{elapsed:.0f}s]")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ---------------------------------------------------------------------------
# Strategy I: Logistic Regression on binary target
# ---------------------------------------------------------------------------
def run_strategy_i(df, X, targets, hold_bars=10, timeframe="15m",
                   min_prob=0.58):
    """Logistic regression on profitable_long target. Simplest possible model."""
    tgt_name = f"tgt_profitable_long_{hold_bars}"
    y = targets[tgt_name]

    feats = get_available_features(X, FEATURES_CURATED)
    print(f"\n{'='*60}")
    print(f"  Strategy I: Logistic Regression ({timeframe})")
    print(f"  Hold: {hold_bars} bars, min_prob: {min_prob}")
    print(f"  Features: {len(feats)} curated, Target: {tgt_name}")
    print(f"  Base rate: {y.dropna().mean():.3f}")
    print(f"{'='*60}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=15,
                                  gap_days=3, step_days=15, min_train_rows=200)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = df.index < split.train_end
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask, feats]
        y_train = y.reindex(X_train.index)
        X_test = X.loc[test_mask, feats]
        df_test = df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        model.fit(X_tr_s, y_train)

        probs = model.predict_proba(X_te_s)[:, 1]

        signals = pd.Series(0, index=X_test.index, dtype=int)
        signals[probs > min_prob] = 1

        # Also check short side
        tgt_short = f"tgt_profitable_short_{hold_bars}"
        ys = targets[tgt_short].reindex(X_train.index)
        if ys.notna().sum() > 100:
            model_s = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
            model_s.fit(X_tr_s, ys)
            probs_s = model_s.predict_proba(X_te_s)[:, 1]
            for i in range(len(X_test)):
                if probs_s[i] > min_prob and probs[i] < 0.5:
                    signals.iloc[i] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        n_l = (signals == 1).sum()
        n_s = (signals == -1).sum()
        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades} (L={n_l},S={n_s}), "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%} [{elapsed:.0f}s]")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ---------------------------------------------------------------------------
# Strategy J: Combined Ridge + LightGBM (stacking)
# ---------------------------------------------------------------------------
def run_strategy_j(df, X, targets, hold_bars=5, timeframe="15m",
                   min_agreement_bps=3.0):
    """Stack Ridge and LightGBM: only trade when both agree on direction
    and predicted P&L > threshold.
    """
    tgt_name = f"tgt_pnl_long_bps_{hold_bars}"
    y = targets[tgt_name]

    feats = get_available_features(X, FEATURES_CURATED)
    print(f"\n{'='*60}")
    print(f"  Strategy J: Ridge+LightGBM Stack ({timeframe})")
    print(f"  Hold: {hold_bars} bars, min_agreement: {min_agreement_bps} bps")
    print(f"  Features: {len(feats)} curated")
    print(f"{'='*60}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=15,
                                  gap_days=3, step_days=15, min_train_rows=200)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = df.index < split.train_end
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask, feats]
        y_train = y.reindex(X_train.index)
        X_test = X.loc[test_mask, feats]
        df_test = df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Ridge
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_tr_s, y_train)
        pred_ridge = ridge.predict(X_te_s)

        # LightGBM
        sp = int(len(X_train) * 0.85)
        lgbm = train_lgbm_regressor(
            X_train.iloc[:sp], y_train.iloc[:sp],
            X_train.iloc[sp:], y_train.iloc[sp:],
            params={"num_leaves": 15, "max_depth": 4, "min_child_samples": 100,
                    "reg_alpha": 1.0, "reg_lambda": 5.0, "learning_rate": 0.03},
        )
        pred_lgbm = lgbm.predict(X_test)

        # Average predictions
        pred_avg = (pred_ridge + pred_lgbm) / 2.0

        signals = pd.Series(0, index=X_test.index, dtype=int)
        for i in range(len(X_test)):
            # Both must agree on direction AND average > threshold
            if (pred_ridge[i] > 0 and pred_lgbm[i] > 0 and
                    pred_avg[i] > min_agreement_bps):
                signals.iloc[i] = 1
            elif (pred_ridge[i] < 0 and pred_lgbm[i] < 0 and
                  pred_avg[i] < -min_agreement_bps):
                signals.iloc[i] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        n_l = (signals == 1).sum()
        n_s = (signals == -1).sum()
        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades} (L={n_l},S={n_s}), "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%} [{elapsed:.0f}s]")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  STRATEGY RESEARCH 03: Smarter Approaches")
    print(f"  RAM: {mem_gb():.1f} GB")
    print(f"{'='*70}")

    all_results = {}

    # --- Test on 15m candles first ---
    for tf, tf_label in [("15m", "15m"), ("1h", "1h")]:
        try:
            df = load_features("BTCUSDT", "2024-01-01", "2024-03-31", tf)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n  Skipping {tf}: {e}")
            continue

        X, targets = split_features_targets(df)
        X = prepare_features(X)
        print(f"  {tf}: {X.shape[0]} candles, {X.shape[1]} features")
        print(f"  RAM: {mem_gb():.1f} GB")

        # Determine hold bars based on timeframe
        # 15m: 5 bars = 75min, 10 bars = 2.5h
        # 1h: 3 bars = 3h, 5 bars = 5h
        if tf == "15m":
            holds = [5, 10]
        else:
            holds = [3, 5]

        for hold in holds:
            # F: Ridge regression
            for min_bps in [2.0, 5.0]:
                label = f"F_ridge_{tf}_h{hold}_min{min_bps}"
                r = run_strategy_f(df, X, targets, hold_bars=hold, timeframe=tf,
                                   min_pred_bps=min_bps)
                all_results[label] = r

            # G: LightGBM curated
            for pct in [70, 80]:
                label = f"G_lgbm_cur_{tf}_h{hold}_p{pct}"
                r = run_strategy_g(df, X, targets, hold_bars=hold, timeframe=tf,
                                   percentile_threshold=pct)
                all_results[label] = r

            # H: Regime-conditional
            label = f"H_regime_{tf}_h{hold}"
            r = run_strategy_h(df, X, targets, hold_bars=hold, timeframe=tf)
            all_results[label] = r

            # I: Logistic regression
            label = f"I_logistic_{tf}_h{hold}"
            r = run_strategy_i(df, X, targets, hold_bars=hold, timeframe=tf)
            all_results[label] = r

            # J: Stacked Ridge+LightGBM
            for min_agr in [2.0, 5.0]:
                label = f"J_stack_{tf}_h{hold}_min{min_agr}"
                r = run_strategy_j(df, X, targets, hold_bars=hold, timeframe=tf,
                                   min_agreement_bps=min_agr)
                all_results[label] = r

        del df, X, targets
        gc.collect()

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY: Research 03 — Smarter Approaches")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<35} {'Folds':>5} {'Trades':>7} {'Avg bps':>8} "
          f"{'WR':>6} {'PF':>6} {'MaxDD':>8} {'Sharpe':>7} {'Prof%':>6}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*6}")

    for label, fold_results in sorted(all_results.items()):
        if not fold_results:
            print(f"  {label:<35} {'N/A':>5}")
            continue

        all_trades = []
        prof_folds = 0
        for split, result in fold_results:
            all_trades.extend(result.trades)
            if result.total_pnl_bps > 0:
                prof_folds += 1

        n_folds = len(fold_results)
        if all_trades:
            combined = BacktestResult(trades=all_trades)
            pf_str = f"{prof_folds}/{n_folds}"
            print(f"  {label:<35} {n_folds:>5} {combined.n_trades:>7} "
                  f"{combined.avg_pnl_bps:>+7.2f} {combined.win_rate:>5.1%} "
                  f"{combined.profit_factor:>5.2f} "
                  f"{combined.max_drawdown_bps:>7.0f} "
                  f"{combined.sharpe:>6.2f} {pf_str:>6}")
        else:
            print(f"  {label:<35} {n_folds:>5} {'0':>7}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s, RAM: {mem_gb():.1f} GB")


if __name__ == "__main__":
    main()
