#!/usr/bin/env python3
"""
Strategy Research 04: Deep validation of top candidates from Research 03.

Top candidates to validate:
  I: Logistic Regression on curated features (1h h5: +48 bps, 1h h3: +34 bps)
  F: Ridge Regression on curated features (1h h5: +27 bps)

Validation approach:
  1. Multiple WFO split configurations (different train/test/step)
  2. Cross-symbol: BTCUSDT + ETHUSDT
  3. Parameter sensitivity: vary threshold, hold period
  4. Feature importance analysis
  5. Fold-level consistency check
"""

import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from strategy_ml_wfo import (
    load_features, split_features_targets, prepare_features,
    generate_wfo_splits, backtest_signals, print_metrics,
    print_wfo_summary, BacktestResult,
    train_lgbm_regressor,
    MAKER_FEE, TAKER_FEE,
)


def mem_gb():
    return psutil.virtual_memory().used / 1024**3


# ---------------------------------------------------------------------------
# Curated features (same as Research 03)
# ---------------------------------------------------------------------------
FEATURES_CURATED = list(set([
    "return", "range", "realized_vol", "total_volume", "total_notional",
    "trade_count", "buy_ratio", "effective_spread_bps",
    "vwap_close_bps", "twap_close_bps",
    "mean_trade_size", "median_trade_size", "trade_size_skew",
    "buy_volume_ratio", "sell_volume_ratio",
    "cvd_normalized", "ofi_normalized",
    "high_low_ratio", "close_position_in_range",
    "return_reversal", "consecutive_same_direction",
    "return_z", "range_z", "realized_vol_z",
    "volume_surprise", "range_surprise",
    "cvd_normalized_z", "ofi_normalized_z",
    "buy_ratio_z", "effective_spread_bps_z",
    "gk_volatility", "parkinson_volatility",
    "trade_count_z",
    "trade_size_kurtosis",
    "large_trade_ratio", "small_trade_ratio",
    "vpin_estimate",
    "high_before_low",
    "poc_position_in_range",
    "close_vs_fair_value_bps",
]))


def get_available_features(X, feature_list):
    return [f for f in feature_list if f in X.columns]


# ---------------------------------------------------------------------------
# Core strategy runners (simplified from Research 03)
# ---------------------------------------------------------------------------

def run_logistic(df, X, targets, feats, hold_bars, min_prob,
                 train_days, test_days, gap_days, step_days, label=""):
    """Logistic regression on profitable_long target."""
    tgt_long = f"tgt_profitable_long_{hold_bars}"
    tgt_short = f"tgt_profitable_short_{hold_bars}"
    y_long = targets[tgt_long]

    splits = generate_wfo_splits(df.index, train_days=train_days,
                                  test_days=test_days, gap_days=gap_days,
                                  step_days=step_days, min_train_rows=200)
    if not splits:
        return []

    fold_results = []
    for split in splits:
        train_mask = df.index < split.train_end  # expanding
        test_mask = (df.index >= split.test_start) & (df.index < split.test_end)

        X_train = X.loc[train_mask, feats]
        y_train = y_long.reindex(X_train.index)
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

        # Short side
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

    return fold_results


def run_ridge(df, X, targets, feats, hold_bars, min_pred_bps,
              train_days, test_days, gap_days, step_days, label=""):
    """Ridge regression on P&L target."""
    tgt_name = f"tgt_pnl_long_bps_{hold_bars}"
    y = targets[tgt_name]

    splits = generate_wfo_splits(df.index, train_days=train_days,
                                  test_days=test_days, gap_days=gap_days,
                                  step_days=step_days, min_train_rows=200)
    if not splits:
        return []

    fold_results = []
    for split in splits:
        train_mask = df.index < split.train_end  # expanding
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

        model = Ridge(alpha=10.0)
        model.fit(X_tr_s, y_train)
        pred = model.predict(X_te_s)

        signals = pd.Series(0, index=X_test.index, dtype=int)
        signals[pred > min_pred_bps] = 1
        signals[pred < -min_pred_bps] = -1

        result = backtest_signals(df_test, signals, hold_bars=hold_bars)
        fold_results.append((split, result))

    return fold_results


def summarize_results(label, fold_results):
    """Print one-line summary for a strategy run."""
    if not fold_results:
        print(f"  {label:<45} {'N/A':>5}")
        return None

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
        print(f"  {label:<45} {n_folds:>3} {combined.n_trades:>5} "
              f"{combined.avg_pnl_bps:>+7.2f} {combined.win_rate:>5.1%} "
              f"{combined.profit_factor:>5.2f} "
              f"{combined.max_drawdown_bps:>7.0f} "
              f"{combined.sharpe:>6.2f} {pf_str:>5}")
        return combined
    else:
        print(f"  {label:<45} {n_folds:>3} {'0':>5}")
        return None


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    print(f"{'='*70}")
    print(f"  STRATEGY RESEARCH 04: Deep Validation")
    print(f"  RAM: {mem_gb():.1f} GB")
    print(f"{'='*70}")

    # WFO split configurations to test
    split_configs = [
        {"train_days": 30, "test_days": 10, "gap_days": 2, "step_days": 10, "name": "30t/10s"},
        {"train_days": 30, "test_days": 15, "gap_days": 3, "step_days": 15, "name": "30t/15s"},
        {"train_days": 45, "test_days": 10, "gap_days": 3, "step_days": 10, "name": "45t/10s"},
        {"train_days": 20, "test_days": 10, "gap_days": 2, "step_days": 10, "name": "20t/10s"},
    ]

    symbols = ["BTCUSDT", "ETHUSDT"]
    all_results = {}

    for symbol in symbols:
        print(f"\n{'#'*70}")
        print(f"  {symbol}")
        print(f"{'#'*70}")

        for tf in ["1h", "15m"]:
            try:
                df = load_features(symbol, "2024-01-01", "2024-03-31", tf)
            except (FileNotFoundError, ValueError) as e:
                print(f"  Skipping {symbol} {tf}: {e}")
                continue

            X, targets = split_features_targets(df)
            X = prepare_features(X)
            feats = get_available_features(X, FEATURES_CURATED)
            print(f"\n  {symbol} {tf}: {X.shape[0]} candles, {len(feats)} curated features")

            if tf == "1h":
                holds = [3, 5, 8]
            else:
                holds = [5, 10, 20]

            # ---------------------------------------------------------------
            # Test 1: Logistic regression — parameter sweep
            # ---------------------------------------------------------------
            print(f"\n  --- Logistic Regression ({tf}) ---")
            print(f"  {'Label':<45} {'F':>3} {'Tr':>5} {'Avg':>7} {'WR':>5} "
                  f"{'PF':>5} {'MaxDD':>7} {'Sh':>6} {'P%':>5}")
            print(f"  {'-'*45} {'-'*3} {'-'*5} {'-'*7} {'-'*5} "
                  f"{'-'*5} {'-'*7} {'-'*6} {'-'*5}")

            for hold in holds:
                for min_prob in [0.55, 0.58, 0.60, 0.65]:
                    for sc in split_configs:
                        label = f"I_{symbol[:3]}_{tf}_h{hold}_p{min_prob}_{sc['name']}"
                        r = run_logistic(df, X, targets, feats, hold,
                                         min_prob, sc["train_days"],
                                         sc["test_days"], sc["gap_days"],
                                         sc["step_days"])
                        combined = summarize_results(label, r)
                        all_results[label] = (r, combined)

            # ---------------------------------------------------------------
            # Test 2: Ridge regression — parameter sweep
            # ---------------------------------------------------------------
            print(f"\n  --- Ridge Regression ({tf}) ---")
            print(f"  {'Label':<45} {'F':>3} {'Tr':>5} {'Avg':>7} {'WR':>5} "
                  f"{'PF':>5} {'MaxDD':>7} {'Sh':>6} {'P%':>5}")
            print(f"  {'-'*45} {'-'*3} {'-'*5} {'-'*7} {'-'*5} "
                  f"{'-'*5} {'-'*7} {'-'*6} {'-'*5}")

            for hold in holds:
                for min_bps in [2.0, 5.0, 10.0]:
                    for sc in split_configs[:2]:  # fewer configs for Ridge
                        label = f"F_{symbol[:3]}_{tf}_h{hold}_min{min_bps}_{sc['name']}"
                        r = run_ridge(df, X, targets, feats, hold,
                                      min_bps, sc["train_days"],
                                      sc["test_days"], sc["gap_days"],
                                      sc["step_days"])
                        combined = summarize_results(label, r)
                        all_results[label] = (r, combined)

            del df, X, targets
            gc.collect()

    # ===================================================================
    # CROSS-SYMBOL CONSISTENCY
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  CROSS-SYMBOL CONSISTENCY CHECK")
    print(f"{'='*70}")

    # Find configs that are positive on BOTH symbols
    btc_keys = {k for k in all_results if "_BTC_" in k}
    eth_keys = {k for k in all_results if "_ETH_" in k}

    # Match by removing symbol prefix
    def strip_symbol(k):
        return k.replace("_BTC_", "_XXX_").replace("_ETH_", "_XXX_")

    btc_map = {strip_symbol(k): k for k in btc_keys}
    eth_map = {strip_symbol(k): k for k in eth_keys}

    common = set(btc_map.keys()) & set(eth_map.keys())
    both_positive = []

    for key in sorted(common):
        bk, ek = btc_map[key], eth_map[key]
        b_r, b_c = all_results[bk]
        e_r, e_c = all_results[ek]
        if b_c and e_c and b_c.avg_pnl_bps > 0 and e_c.avg_pnl_bps > 0:
            both_positive.append((key, bk, ek, b_c, e_c))

    if both_positive:
        print(f"\n  Configs positive on BOTH BTC and ETH: {len(both_positive)}")
        print(f"\n  {'Config':<40} {'BTC avg':>8} {'BTC WR':>7} {'ETH avg':>8} {'ETH WR':>7} {'Combined':>9}")
        print(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*9}")
        for key, bk, ek, bc, ec in sorted(both_positive,
                key=lambda x: (x[3].avg_pnl_bps + x[4].avg_pnl_bps)/2, reverse=True)[:20]:
            avg = (bc.avg_pnl_bps + ec.avg_pnl_bps) / 2
            print(f"  {key:<40} {bc.avg_pnl_bps:>+7.1f} {bc.win_rate:>6.1%} "
                  f"{ec.avg_pnl_bps:>+7.1f} {ec.win_rate:>6.1%} {avg:>+8.1f}")
    else:
        print("\n  No configs positive on both symbols.")

    # ===================================================================
    # FEATURE IMPORTANCE (for best Logistic model)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  FEATURE IMPORTANCE (Logistic 1h h5)")
    print(f"{'='*70}")

    for symbol in symbols:
        try:
            df = load_features(symbol, "2024-01-01", "2024-03-31", "1h")
        except:
            continue
        X, targets = split_features_targets(df)
        X = prepare_features(X)
        feats = get_available_features(X, FEATURES_CURATED)

        tgt = targets["tgt_profitable_long_5"]
        valid = tgt.notna()
        Xv = X.loc[valid, feats]
        yv = tgt.loc[valid]

        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xv)
        model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        model.fit(Xs, yv)

        importance = pd.Series(np.abs(model.coef_[0]), index=feats).sort_values(ascending=False)
        print(f"\n  {symbol} — Top 15 features by |coefficient|:")
        for feat, imp in importance.head(15).items():
            sign = "+" if model.coef_[0][feats.index(feat)] > 0 else "-"
            print(f"    {sign} {feat:<35} {imp:.4f}")

        del df, X, targets
        gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s, RAM: {mem_gb():.1f} GB")


if __name__ == "__main__":
    main()
