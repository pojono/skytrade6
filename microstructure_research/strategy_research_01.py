#!/usr/bin/env python3
"""
Strategy Research 01: Quick screening of ML strategy candidates on 14-day dev set.
Tests multiple approaches to identify which show any signal before scaling up.
"""

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import lightgbm as lgb

warnings.filterwarnings("ignore")

from strategy_ml_wfo import (
    load_features, split_features_targets, prepare_features,
    generate_wfo_splits, backtest_signals, print_metrics,
    print_wfo_summary, BacktestResult, Trade,
    drop_correlated_features,
    train_lgbm_classifier, train_lgbm_regressor,
    MAKER_FEE, TAKER_FEE,
)


def mem_gb():
    return psutil.virtual_memory().used / 1024**3


def select_top_features(X, y, k=50):
    """Quick feature selection via F-test."""
    from sklearn.feature_selection import f_classif, f_regression
    mask = y.notna()
    X_c, y_c = X.loc[mask], y.loc[mask]
    if len(X_c) < 50:
        return X.columns.tolist()[:k]
    try:
        if y_c.nunique() <= 10:
            scores, _ = f_classif(X_c, y_c)
        else:
            scores, _ = f_regression(X_c, y_c)
    except Exception:
        return X.columns.tolist()[:k]
    scores = np.nan_to_num(scores, nan=0.0)
    ranked = sorted(zip(X.columns, scores), key=lambda x: -x[1])
    return [n for n, s in ranked[:k]]


# ===================================================================
# STRATEGY A: Binary profitable-long classifier
# ===================================================================
def run_strategy_a(df, X, targets, hold_bars=5, threshold=0.55):
    """Predict if a long trade will be profitable after fees (5-bar hold).
    Enter long when P(profitable) > threshold.
    """
    target_name = f"tgt_profitable_long_{hold_bars}"
    y = targets[target_name]
    print(f"\n--- Strategy A: Profitable Long Classifier ---")
    print(f"  Target: {target_name}, threshold={threshold}")
    print(f"  Base rate: {y.dropna().mean():.3f}")

    splits = generate_wfo_splits(df.index, train_days=8, test_days=3,
                                  gap_days=1, step_days=3, min_train_rows=200)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Feature selection
        selected = select_top_features(X_train, y_train, k=50)
        X_tr, X_te = X_train[selected], X_test[selected]

        # Train/val split
        sp = int(len(X_tr) * 0.8)
        model = train_lgbm_classifier(
            X_tr.iloc[:sp], y_train.iloc[:sp],
            X_tr.iloc[sp:], y_train.iloc[sp:],
        )

        # Predict
        probs = model.predict(X_te)
        signals = pd.Series(0, index=X_te.index, dtype=int)
        signals[probs > threshold] = 1  # long only

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        print(f"  Fold {split.fold_id}: trades={result.n_trades}, "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# STRATEGY B: Binary profitable-short classifier
# ===================================================================
def run_strategy_b(df, X, targets, hold_bars=5, threshold=0.55):
    """Same as A but for short trades."""
    target_name = f"tgt_profitable_short_{hold_bars}"
    y = targets[target_name]
    print(f"\n--- Strategy B: Profitable Short Classifier ---")
    print(f"  Target: {target_name}, threshold={threshold}")
    print(f"  Base rate: {y.dropna().mean():.3f}")

    splits = generate_wfo_splits(df.index, train_days=8, test_days=3,
                                  gap_days=1, step_days=3, min_train_rows=200)

    fold_results = []
    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        selected = select_top_features(X_train, y_train, k=50)
        X_tr, X_te = X_train[selected], X_test[selected]

        sp = int(len(X_tr) * 0.8)
        model = train_lgbm_classifier(
            X_tr.iloc[:sp], y_train.iloc[:sp],
            X_tr.iloc[sp:], y_train.iloc[sp:],
        )

        probs = model.predict(X_te)
        signals = pd.Series(0, index=X_te.index, dtype=int)
        signals[probs > threshold] = -1  # short only

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        print(f"  Fold {split.fold_id}: trades={result.n_trades}, "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# STRATEGY C: Combined long+short with separate models
# ===================================================================
def run_strategy_c(df, X, targets, hold_bars=5, threshold=0.55):
    """Two models: one for long, one for short. Trade whichever is more confident."""
    y_long = targets[f"tgt_profitable_long_{hold_bars}"]
    y_short = targets[f"tgt_profitable_short_{hold_bars}"]
    print(f"\n--- Strategy C: Combined Long/Short Classifier ---")
    print(f"  Targets: profitable_long_{hold_bars} + profitable_short_{hold_bars}")
    print(f"  Threshold: {threshold}")

    splits = generate_wfo_splits(df.index, train_days=8, test_days=3,
                                  gap_days=1, step_days=3, min_train_rows=200)

    fold_results = []
    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        yl_tr = y_long.reindex(X_train.index)
        ys_tr = y_short.reindex(X_train.index)

        valid_l = yl_tr.notna()
        valid_s = ys_tr.notna()

        if valid_l.sum() < 100 or valid_s.sum() < 100 or len(X_test) < 10:
            continue

        # Feature selection (use long target)
        selected = select_top_features(X_train.loc[valid_l], yl_tr.loc[valid_l], k=50)
        X_tr_l = X_train.loc[valid_l][selected]
        X_tr_s = X_train.loc[valid_s][selected]
        X_te = X_test[selected]

        # Train long model
        sp = int(len(X_tr_l) * 0.8)
        model_l = train_lgbm_classifier(
            X_tr_l.iloc[:sp], yl_tr.loc[valid_l].iloc[:sp],
            X_tr_l.iloc[sp:], yl_tr.loc[valid_l].iloc[sp:],
        )

        # Train short model
        sp_s = int(len(X_tr_s) * 0.8)
        model_s = train_lgbm_classifier(
            X_tr_s.iloc[:sp_s], ys_tr.loc[valid_s].iloc[:sp_s],
            X_tr_s.iloc[sp_s:], ys_tr.loc[valid_s].iloc[sp_s:],
        )

        # Predict
        prob_l = model_l.predict(X_te)
        prob_s = model_s.predict(X_te)

        signals = pd.Series(0, index=X_te.index, dtype=int)
        for i in range(len(X_te)):
            pl, ps = prob_l[i], prob_s[i]
            if pl > threshold and pl > ps:
                signals.iloc[i] = 1
            elif ps > threshold and ps > pl:
                signals.iloc[i] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        print(f"  Fold {split.fold_id}: trades={result.n_trades} "
              f"(L={n_long},S={n_short}), "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# STRATEGY D: Regression-based (predict P&L, trade when expected > threshold)
# ===================================================================
def run_strategy_d(df, X, targets, hold_bars=5, min_expected_bps=3.0):
    """Predict expected P&L in bps, trade when expected > min_expected_bps."""
    target_name = f"tgt_pnl_long_bps_{hold_bars}"
    y = targets[target_name]
    print(f"\n--- Strategy D: P&L Regression ---")
    print(f"  Target: {target_name}, min_expected={min_expected_bps} bps")

    splits = generate_wfo_splits(df.index, train_days=8, test_days=3,
                                  gap_days=1, step_days=3, min_train_rows=200)

    fold_results = []
    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        selected = select_top_features(X_train, y_train, k=50)
        X_tr, X_te = X_train[selected], X_test[selected]

        sp = int(len(X_tr) * 0.8)
        model = train_lgbm_regressor(
            X_tr.iloc[:sp], y_train.iloc[:sp],
            X_tr.iloc[sp:], y_train.iloc[sp:],
        )

        pred = model.predict(X_te)
        signals = pd.Series(0, index=X_te.index, dtype=int)
        signals[pred > min_expected_bps] = 1   # long when expected P&L > threshold
        signals[pred < -min_expected_bps] = -1  # short when expected loss > threshold

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        print(f"  Fold {split.fold_id}: trades={result.n_trades}, "
              f"avg={result.avg_pnl_bps:+.2f} bps, "
              f"pred_mean={pred.mean():+.2f}, pred_std={pred.std():.2f}")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# STRATEGY E: Return sign with TP/SL
# ===================================================================
def run_strategy_e(df, X, targets, hold_bars=5, threshold=0.55, tp_bps=30, sl_bps=50):
    """Predict return sign, use TP/SL for exits."""
    target_name = f"tgt_ret_sign_{hold_bars}"
    y = targets[target_name]
    # Map: -1->0, 1->1 for binary classification
    y_binary = ((y + 1) / 2).fillna(0.5)  # -1->0, 1->1
    print(f"\n--- Strategy E: Return Sign + TP/SL ---")
    print(f"  Target: {target_name}, TP={tp_bps}bps, SL={sl_bps}bps")
    print(f"  Base rate (up): {(y_binary.dropna() > 0.5).mean():.3f}")

    splits = generate_wfo_splits(df.index, train_days=8, test_days=3,
                                  gap_days=1, step_days=3, min_train_rows=200)

    fold_results = []
    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train, y_train = X.loc[train_mask], y_binary.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        valid = y_train.notna() & (y_train != 0.5)
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        selected = select_top_features(X_train, y_train, k=50)
        X_tr, X_te = X_train[selected], X_test[selected]

        sp = int(len(X_tr) * 0.8)
        model = train_lgbm_classifier(
            X_tr.iloc[:sp], y_train.iloc[:sp],
            X_tr.iloc[sp:], y_train.iloc[sp:],
        )

        probs = model.predict(X_te)  # P(up)
        signals = pd.Series(0, index=X_te.index, dtype=int)
        signals[probs > threshold] = 1
        signals[probs < (1 - threshold)] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars,
                                   tp_bps=tp_bps, sl_bps=sl_bps)
        fold_results.append((split, result))

        print(f"  Fold {split.fold_id}: trades={result.n_trades}, "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  STRATEGY RESEARCH 01: Quick Screening")
    print(f"  BTCUSDT 15m, 2024-01-01 to 2024-01-14 (14 days)")
    print(f"  RAM: {mem_gb():.1f} GB")
    print(f"{'='*70}")

    # Load data
    df = load_features("BTCUSDT", "2024-01-01", "2024-01-14", "15m")
    X, targets = split_features_targets(df)
    X = prepare_features(X)
    X = drop_correlated_features(X, threshold=0.95)
    print(f"  Features after cleaning: {X.shape}")
    print(f"  RAM: {mem_gb():.1f} GB")

    # Run all strategies
    results = {}

    # Strategy A: Profitable Long
    for hold in [3, 5, 10]:
        for thresh in [0.50, 0.55, 0.60]:
            label = f"A_long_h{hold}_t{thresh}"
            r = run_strategy_a(df, X, targets, hold_bars=hold, threshold=thresh)
            results[label] = r

    # Strategy B: Profitable Short
    for hold in [3, 5]:
        r = run_strategy_b(df, X, targets, hold_bars=hold, threshold=0.55)
        results[f"B_short_h{hold}"] = r

    # Strategy C: Combined Long/Short
    for hold in [3, 5]:
        r = run_strategy_c(df, X, targets, hold_bars=hold, threshold=0.55)
        results[f"C_combined_h{hold}"] = r

    # Strategy D: Regression
    for hold in [3, 5]:
        for min_bps in [2.0, 5.0]:
            r = run_strategy_d(df, X, targets, hold_bars=hold, min_expected_bps=min_bps)
            results[f"D_reg_h{hold}_min{min_bps}"] = r

    # Strategy E: Return sign + TP/SL
    for hold in [5, 10]:
        r = run_strategy_e(df, X, targets, hold_bars=hold, threshold=0.55,
                           tp_bps=30, sl_bps=50)
        results[f"E_sign_h{hold}"] = r

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: All Strategy Candidates")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<30} {'Folds':>5} {'Trades':>7} {'Avg bps':>8} "
          f"{'WR':>6} {'PF':>6} {'Prof Folds':>10}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*10}")

    for label, fold_results in sorted(results.items()):
        if not fold_results:
            print(f"  {label:<30} {'N/A':>5}")
            continue

        all_trades = []
        prof_folds = 0
        for split, result in fold_results:
            all_trades.extend(result.trades)
            if result.total_pnl_bps > 0:
                prof_folds += 1

        if all_trades:
            combined = BacktestResult(trades=all_trades)
            n_folds = len(fold_results)
            print(f"  {label:<30} {n_folds:>5} {combined.n_trades:>7} "
                  f"{combined.avg_pnl_bps:>+7.2f} {combined.win_rate:>5.1%} "
                  f"{combined.profit_factor:>5.2f} "
                  f"{prof_folds}/{n_folds}")
        else:
            print(f"  {label:<30} {len(fold_results):>5} {'0':>7}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s, RAM: {mem_gb():.1f} GB")


if __name__ == "__main__":
    main()
