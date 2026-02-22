#!/usr/bin/env python3
"""
Strategy Research 02: Proper WFO on 90-day dataset.

Refined based on 14-day screening findings:
  - Regression (predict P&L magnitude) > classification
  - 5-bar hold (75min) better than 3-bar
  - Need selectivity: fewer, higher-quality trades
  - Feature selection critical with 700+ features

Tests refined strategies with proper WFO (5+ folds):
  A: P&L regression with adaptive threshold
  B: Dual-model long/short with confidence gating
  C: Ensemble of multiple targets (ret_sign + profitable + pnl)
  D: Longer hold (10-bar, 2.5h) with higher threshold
  E: 1h candle approach (if data available)
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


def get_feature_importance(model, feature_names, top_k=20):
    """Get top feature importances from LightGBM model."""
    imp = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    return ranked[:top_k]


# ===================================================================
# STRATEGY A: P&L Regression with adaptive threshold
# ===================================================================
def run_strategy_a(df, X, targets, hold_bars=5, percentile_threshold=75):
    """Predict P&L in bps. Only trade when predicted P&L is in top percentile
    of training predictions (adaptive threshold per fold).
    """
    tgt_long = f"tgt_pnl_long_bps_{hold_bars}"
    tgt_short = f"tgt_pnl_short_bps_{hold_bars}"
    y_long = targets[tgt_long]
    y_short = targets[tgt_short]

    print(f"\n{'='*60}")
    print(f"  Strategy A: P&L Regression (adaptive threshold)")
    print(f"  Hold: {hold_bars} bars, Percentile: {percentile_threshold}")
    print(f"{'='*60}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=10,
                                  gap_days=3, step_days=10, min_train_rows=500)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    all_importances = []
    t0 = time.time()

    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        # Train long model
        yl_tr = y_long.reindex(X_train.index)
        valid_l = yl_tr.notna()
        if valid_l.sum() < 200 or len(X_test) < 20:
            continue

        selected = select_top_features(X_train.loc[valid_l], yl_tr.loc[valid_l], k=80)
        X_tr_sel = X_train.loc[valid_l][selected]
        X_te_sel = X_test[selected]

        sp = int(len(X_tr_sel) * 0.8)
        model_l = train_lgbm_regressor(
            X_tr_sel.iloc[:sp], yl_tr.loc[valid_l].iloc[:sp],
            X_tr_sel.iloc[sp:], yl_tr.loc[valid_l].iloc[sp:],
        )

        # Train short model
        ys_tr = y_short.reindex(X_train.index)
        valid_s = ys_tr.notna()
        X_tr_s = X_train.loc[valid_s][selected]
        sp_s = int(len(X_tr_s) * 0.8)
        model_s = train_lgbm_regressor(
            X_tr_s.iloc[:sp_s], ys_tr.loc[valid_s].iloc[:sp_s],
            X_tr_s.iloc[sp_s:], ys_tr.loc[valid_s].iloc[sp_s:],
        )

        # Predict on test
        pred_long = model_l.predict(X_te_sel)
        pred_short = model_s.predict(X_te_sel)

        # Adaptive threshold: use percentile of TRAINING predictions
        train_pred_l = model_l.predict(X_tr_sel)
        train_pred_s = model_s.predict(X_tr_s)
        thresh_l = np.percentile(train_pred_l, percentile_threshold)
        thresh_s = np.percentile(train_pred_s, percentile_threshold)

        # Generate signals
        signals = pd.Series(0, index=X_te_sel.index, dtype=int)
        for i in range(len(X_te_sel)):
            pl, ps = pred_long[i], pred_short[i]
            if pl > thresh_l and pl > ps and pl > 0:
                signals.iloc[i] = 1
            elif ps > thresh_s and ps > pl and ps > 0:
                signals.iloc[i] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        # Track feature importance
        imp = get_feature_importance(model_l, selected, top_k=10)
        all_importances.extend(imp)

        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades}, "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}, "
              f"thresh_l={thresh_l:.1f}, thresh_s={thresh_s:.1f} [{elapsed:.0f}s]")

    if fold_results:
        all_trades = print_wfo_summary(fold_results)

        # Aggregate feature importance
        if all_importances:
            imp_agg = {}
            for name, score in all_importances:
                imp_agg[name] = imp_agg.get(name, 0) + score
            top_feats = sorted(imp_agg.items(), key=lambda x: -x[1])[:15]
            print(f"\n  Top features (aggregate gain):")
            for name, score in top_feats:
                print(f"    {name:<40s} {score:.0f}")

    return fold_results


# ===================================================================
# STRATEGY B: Dual binary classifier with confidence gating
# ===================================================================
def run_strategy_b(df, X, targets, hold_bars=5, min_prob=0.60):
    """Separate binary models for long/short profitability.
    Only trade when P(profitable) > min_prob AND the other side < 0.5.
    """
    tgt_long = f"tgt_profitable_long_{hold_bars}"
    tgt_short = f"tgt_profitable_short_{hold_bars}"
    y_long = targets[tgt_long]
    y_short = targets[tgt_short]

    print(f"\n{'='*60}")
    print(f"  Strategy B: Dual Classifier (min_prob={min_prob})")
    print(f"  Hold: {hold_bars} bars")
    print(f"{'='*60}")
    print(f"  Long base rate: {y_long.dropna().mean():.3f}")
    print(f"  Short base rate: {y_short.dropna().mean():.3f}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=10,
                                  gap_days=3, step_days=10, min_train_rows=500)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        yl_tr = y_long.reindex(X_train.index)
        ys_tr = y_short.reindex(X_train.index)
        valid_l = yl_tr.notna()
        valid_s = ys_tr.notna()

        if valid_l.sum() < 200 or valid_s.sum() < 200 or len(X_test) < 20:
            continue

        selected = select_top_features(X_train.loc[valid_l], yl_tr.loc[valid_l], k=80)
        X_tr_l = X_train.loc[valid_l][selected]
        X_tr_s = X_train.loc[valid_s][selected]
        X_te = X_test[selected]

        # Long model
        sp = int(len(X_tr_l) * 0.8)
        model_l = train_lgbm_classifier(
            X_tr_l.iloc[:sp], yl_tr.loc[valid_l].iloc[:sp],
            X_tr_l.iloc[sp:], yl_tr.loc[valid_l].iloc[sp:],
        )

        # Short model
        sp_s = int(len(X_tr_s) * 0.8)
        model_s = train_lgbm_classifier(
            X_tr_s.iloc[:sp_s], ys_tr.loc[valid_s].iloc[:sp_s],
            X_tr_s.iloc[sp_s:], ys_tr.loc[valid_s].iloc[sp_s:],
        )

        prob_l = model_l.predict(X_te)
        prob_s = model_s.predict(X_te)

        signals = pd.Series(0, index=X_te.index, dtype=int)
        for i in range(len(X_te)):
            pl, ps = prob_l[i], prob_s[i]
            # Only trade when one side is confident AND the other is not
            if pl > min_prob and ps < 0.5 and pl > ps:
                signals.iloc[i] = 1
            elif ps > min_prob and pl < 0.5 and ps > pl:
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
# STRATEGY C: Ensemble of multiple targets
# ===================================================================
def run_strategy_c(df, X, targets, hold_bars=5, min_votes=3):
    """Train models on multiple targets, trade when majority agree.
    Targets: ret_sign, profitable_long, profitable_short, pnl_regression.
    """
    print(f"\n{'='*60}")
    print(f"  Strategy C: Multi-Target Ensemble (min_votes={min_votes})")
    print(f"  Hold: {hold_bars} bars")
    print(f"{'='*60}")

    target_configs = [
        (f"tgt_ret_sign_{hold_bars}", "sign"),
        (f"tgt_profitable_long_{hold_bars}", "long"),
        (f"tgt_profitable_short_{hold_bars}", "short"),
        (f"tgt_pnl_long_bps_{hold_bars}", "pnl"),
    ]

    splits = generate_wfo_splits(df.index, train_days=30, test_days=10,
                                  gap_days=3, step_days=10, min_train_rows=500)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        if len(X_train) < 200 or len(X_test) < 20:
            continue

        # Select features once (using ret_sign target)
        y_sign = targets[f"tgt_ret_sign_{hold_bars}"].reindex(X_train.index)
        valid_sign = y_sign.notna()
        if valid_sign.sum() < 200:
            continue
        selected = select_top_features(X_train.loc[valid_sign], y_sign.loc[valid_sign], k=80)

        # Train all models
        votes_long = np.zeros(len(X_test))
        votes_short = np.zeros(len(X_test))

        for tgt_name, tgt_type in target_configs:
            y_tr = targets[tgt_name].reindex(X_train.index)
            valid = y_tr.notna()
            if valid.sum() < 100:
                continue

            X_tr = X_train.loc[valid][selected]
            X_te = X_test[selected]
            sp = int(len(X_tr) * 0.8)

            if tgt_type == "sign":
                # Binary: up (1) vs down (-1) -> map to 0/1
                y_binary = ((y_tr.loc[valid] + 1) / 2).clip(0, 1)
                model = train_lgbm_classifier(
                    X_tr.iloc[:sp], y_binary.iloc[:sp],
                    X_tr.iloc[sp:], y_binary.iloc[sp:],
                )
                probs = model.predict(X_te)
                votes_long += (probs > 0.55).astype(float)
                votes_short += (probs < 0.45).astype(float)

            elif tgt_type == "long":
                model = train_lgbm_classifier(
                    X_tr.iloc[:sp], y_tr.loc[valid].iloc[:sp],
                    X_tr.iloc[sp:], y_tr.loc[valid].iloc[sp:],
                )
                probs = model.predict(X_te)
                votes_long += (probs > 0.55).astype(float)

            elif tgt_type == "short":
                model = train_lgbm_classifier(
                    X_tr.iloc[:sp], y_tr.loc[valid].iloc[:sp],
                    X_tr.iloc[sp:], y_tr.loc[valid].iloc[sp:],
                )
                probs = model.predict(X_te)
                votes_short += (probs > 0.55).astype(float)

            elif tgt_type == "pnl":
                model = train_lgbm_regressor(
                    X_tr.iloc[:sp], y_tr.loc[valid].iloc[:sp],
                    X_tr.iloc[sp:], y_tr.loc[valid].iloc[sp:],
                )
                pred = model.predict(X_te)
                votes_long += (pred > 3.0).astype(float)
                votes_short += (pred < -3.0).astype(float)

        # Generate signals from votes
        signals = pd.Series(0, index=X_test.index, dtype=int)
        for i in range(len(X_test)):
            vl, vs = votes_long[i], votes_short[i]
            if vl >= min_votes and vl > vs:
                signals.iloc[i] = 1
            elif vs >= min_votes and vs > vl:
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
# STRATEGY D: Longer hold with higher selectivity
# ===================================================================
def run_strategy_d(df, X, targets, hold_bars=10, percentile_threshold=85):
    """Same as Strategy A but with longer hold and higher selectivity.
    Hypothesis: longer holds capture more of the signal, higher threshold = fewer but better trades.
    """
    tgt_long = f"tgt_pnl_long_bps_{hold_bars}"
    y = targets[tgt_long]

    print(f"\n{'='*60}")
    print(f"  Strategy D: Long-Hold Regression (h={hold_bars}, pct={percentile_threshold})")
    print(f"{'='*60}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=10,
                                  gap_days=3, step_days=10, min_train_rows=500)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        y_tr = y.reindex(X_train.index)
        valid = y_tr.notna()
        if valid.sum() < 200 or len(X_test) < 20:
            continue

        selected = select_top_features(X_train.loc[valid], y_tr.loc[valid], k=80)
        X_tr = X_train.loc[valid][selected]
        X_te = X_test[selected]

        sp = int(len(X_tr) * 0.8)
        model = train_lgbm_regressor(
            X_tr.iloc[:sp], y_tr.loc[valid].iloc[:sp],
            X_tr.iloc[sp:], y_tr.loc[valid].iloc[sp:],
        )

        pred = model.predict(X_te)
        train_pred = model.predict(X_tr)
        thresh = np.percentile(train_pred, percentile_threshold)

        signals = pd.Series(0, index=X_te.index, dtype=int)
        signals[pred > max(thresh, 0)] = 1  # long only when predicted P&L > threshold AND > 0

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades}, "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}, "
              f"thresh={thresh:.1f} [{elapsed:.0f}s]")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# STRATEGY E: Cumulative return regression (multi-horizon)
# ===================================================================
def run_strategy_e(df, X, targets, horizons=[3, 5, 10], min_expected_bps=5.0):
    """Train regression on cumulative return. Trade the horizon with best
    expected return, but only if it exceeds min_expected_bps.
    """
    print(f"\n{'='*60}")
    print(f"  Strategy E: Multi-Horizon Cum Return (min={min_expected_bps} bps)")
    print(f"  Horizons: {horizons}")
    print(f"{'='*60}")

    splits = generate_wfo_splits(df.index, train_days=30, test_days=10,
                                  gap_days=3, step_days=10, min_train_rows=500)
    print(f"  WFO folds: {len(splits)}")

    fold_results = []
    t0 = time.time()

    for split in splits:
        train_mask = (df.index >= split.train_start) & (df.index < split.train_end)
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask]
        X_test, df_test = X.loc[test_mask], df.loc[test_mask]

        if len(X_train) < 200 or len(X_test) < 20:
            continue

        # Train a model per horizon
        models = {}
        selected = None
        for h in horizons:
            tgt_name = f"tgt_cum_ret_{h}"
            y_tr = targets[tgt_name].reindex(X_train.index)
            valid = y_tr.notna()
            if valid.sum() < 200:
                continue

            if selected is None:
                selected = select_top_features(X_train.loc[valid], y_tr.loc[valid], k=80)

            X_tr = X_train.loc[valid][selected]
            sp = int(len(X_tr) * 0.8)
            model = train_lgbm_regressor(
                X_tr.iloc[:sp], y_tr.loc[valid].iloc[:sp],
                X_tr.iloc[sp:], y_tr.loc[valid].iloc[sp:],
            )
            models[h] = model

        if not models or selected is None:
            continue

        X_te = X_test[selected]

        # For each test row, pick the best horizon
        signals = pd.Series(0, index=X_te.index, dtype=int)
        hold_per_signal = pd.Series(5, index=X_te.index, dtype=int)  # default

        for i in range(len(X_te)):
            row = X_te.iloc[i:i+1]
            best_ret = 0
            best_h = None
            best_dir = 0
            for h, model in models.items():
                pred = model.predict(row)[0]
                # Convert cum_ret to bps, subtract fees
                pred_bps = pred * 10000 - (MAKER_FEE + MAKER_FEE) * 10000
                if abs(pred_bps) > abs(best_ret):
                    best_ret = pred_bps
                    best_h = h
                    best_dir = 1 if pred_bps > 0 else -1

            if abs(best_ret) > min_expected_bps and best_h is not None:
                signals.iloc[i] = best_dir
                hold_per_signal.iloc[i] = best_h

        # Backtest with the most common hold period
        avg_hold = int(hold_per_signal[signals != 0].mean()) if (signals != 0).any() else 5
        result = backtest_signals(df_test, signals, hold_bars=avg_hold)
        fold_results.append((split, result))

        n_l = (signals == 1).sum()
        n_s = (signals == -1).sum()
        elapsed = time.time() - t0
        print(f"  Fold {split.fold_id}: trades={result.n_trades} (L={n_l},S={n_s}), "
              f"avg={result.avg_pnl_bps:+.2f} bps, WR={result.win_rate:.1%}, "
              f"avg_hold={avg_hold} [{elapsed:.0f}s]")

    if fold_results:
        print_wfo_summary(fold_results)
    return fold_results


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  STRATEGY RESEARCH 02: Proper WFO on 90-day dataset")
    print(f"  BTCUSDT 15m, 2024-01-01 to 2024-03-31")
    print(f"  RAM: {mem_gb():.1f} GB")
    print(f"{'='*70}")

    # Load data
    df = load_features("BTCUSDT", "2024-01-01", "2024-03-31", "15m")
    X, targets = split_features_targets(df)
    X = prepare_features(X)
    X = drop_correlated_features(X, threshold=0.95)
    print(f"  Features after cleaning: {X.shape}")
    print(f"  RAM: {mem_gb():.1f} GB")

    # Run all strategies
    all_results = {}

    # Strategy A: P&L Regression (adaptive threshold)
    for pct in [70, 75, 80]:
        label = f"A_pnl_reg_h5_p{pct}"
        r = run_strategy_a(df, X, targets, hold_bars=5, percentile_threshold=pct)
        all_results[label] = r

    # Strategy B: Dual classifier
    for prob in [0.55, 0.60, 0.65]:
        label = f"B_dual_h5_p{prob}"
        r = run_strategy_b(df, X, targets, hold_bars=5, min_prob=prob)
        all_results[label] = r

    # Strategy C: Ensemble
    for votes in [2, 3]:
        label = f"C_ensemble_h5_v{votes}"
        r = run_strategy_c(df, X, targets, hold_bars=5, min_votes=votes)
        all_results[label] = r

    # Strategy D: Longer hold
    for hold in [10]:
        for pct in [75, 85]:
            label = f"D_longhold_h{hold}_p{pct}"
            r = run_strategy_d(df, X, targets, hold_bars=hold, percentile_threshold=pct)
            all_results[label] = r

    # Strategy E: Multi-horizon
    for min_bps in [3.0, 5.0]:
        label = f"E_multihorizon_min{min_bps}"
        r = run_strategy_e(df, X, targets, min_expected_bps=min_bps)
        all_results[label] = r

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY: All Strategy Candidates (90-day WFO)")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<30} {'Folds':>5} {'Trades':>7} {'Avg bps':>8} "
          f"{'WR':>6} {'PF':>6} {'MaxDD':>8} {'Sharpe':>7} {'Prof%':>6}")
    print(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*6}")

    for label, fold_results in sorted(all_results.items()):
        if not fold_results:
            print(f"  {label:<30} {'N/A':>5}")
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
            print(f"  {label:<30} {n_folds:>5} {combined.n_trades:>7} "
                  f"{combined.avg_pnl_bps:>+7.2f} {combined.win_rate:>5.1%} "
                  f"{combined.profit_factor:>5.2f} "
                  f"{combined.max_drawdown_bps:>7.0f} "
                  f"{combined.sharpe:>6.2f} {pf_str:>6}")
        else:
            print(f"  {label:<30} {n_folds:>5} {'0':>7}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s, RAM: {mem_gb():.1f} GB")
    print(f"\n  NOTE: Positive avg bps + majority profitable folds = candidate for validation")


if __name__ == "__main__":
    main()
