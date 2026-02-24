#!/usr/bin/env python3
"""
Deep dive into 1m predictability:
1. Feature importance — what drives each target?
2. Walk-forward validation — are scores stable across time?
3. Prediction quality — calibration, precision at extremes
4. Simple P&L test — can we actually make money?

Uses cached 1m bars from test_1m_predictability.py.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb


# ============================================================
# CONFIG
# ============================================================
PARQUET_DIR = Path("./parquet")
CACHE_DIR = Path("./parquet/SOLUSDT/1m_cache")
SYMBOL = "SOLUSDT"
START_DATE = "2025-07-01"
END_DATE = "2025-12-31"

FEE_BPS = 4.0
FEE_FRAC = FEE_BPS / 10000.0

# Targets to deep-dive
TARGETS_OF_INTEREST = [
    # Directional (actionable)
    "tgt_profitable_long_3",
    "tgt_profitable_short_3",
    "tgt_profitable_long_5",
    "tgt_profitable_short_5",
    # Breakout
    "tgt_breakout_up_3",
    "tgt_breakout_down_3",
    # Mean reversion (new, interesting)
    "tgt_mean_revert_5",
    "tgt_mean_revert_15",
    "tgt_mean_revert_30",
    # Vol expansion (gate)
    "tgt_vol_expansion_5",
    "tgt_vol_expansion_15",
    "tgt_vol_expansion_30",
    # Magnitude (sizing)
    "tgt_ret_mag_1",
    "tgt_ret_mag_3",
    "tgt_ret_mag_5",
]


# ============================================================
# DATA LOADING (reuse cached 1m bars)
# ============================================================
def load_1m_bars():
    dates = pd.date_range(START_DATE, END_DATE)
    all_bars = []
    t0 = time.time()

    print(f"  Loading cached 1m bars...", flush=True)
    for i, date in enumerate(dates, 1):
        ds = date.strftime("%Y-%m-%d")
        cache_path = CACHE_DIR / f"{ds}.parquet"
        if cache_path.exists():
            all_bars.append(pd.read_parquet(cache_path))

        if i % 50 == 0 or i == len(dates):
            print(f"    [{i}/{len(dates)}] elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.concat(all_bars, ignore_index=True).sort_values("timestamp_us").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df.set_index("datetime", inplace=True)
    df["returns"] = df["close"].pct_change()
    print(f"  Loaded {len(df):,} bars", flush=True)
    return df


def add_derived_features(df):
    bph = 60
    df["rvol_5m"] = df["returns"].rolling(5).std()
    df["rvol_15m"] = df["returns"].rolling(15).std()
    df["rvol_1h"] = df["returns"].rolling(bph).std()
    df["rvol_4h"] = df["returns"].rolling(4 * bph).std()
    df["vol_ratio_5m_1h"] = df["rvol_5m"] / df["rvol_1h"].clip(lower=1e-10)

    w1h = bph
    df["vol_zscore_1h"] = (df["volume"] - df["volume"].rolling(w1h).mean()) / \
                          df["volume"].rolling(w1h).std().clip(lower=1e-10)
    df["rate_zscore_1h"] = (df["arrival_rate"] - df["arrival_rate"].rolling(w1h).mean()) / \
                           df["arrival_rate"].rolling(w1h).std().clip(lower=1e-10)

    df["mom_5m"] = df["close"].pct_change(5)
    df["mom_15m"] = df["close"].pct_change(15)
    df["mom_1h"] = df["close"].pct_change(bph)

    df["price_zscore_1h"] = (df["close"] - df["close"].rolling(bph).mean()) / \
                            df["close"].rolling(bph).std().clip(lower=1e-10)
    df["price_zscore_4h"] = (df["close"] - df["close"].rolling(4 * bph).mean()) / \
                            df["close"].rolling(4 * bph).std().clip(lower=1e-10)

    df["range_zscore_1h"] = (df["price_range"] - df["price_range"].rolling(w1h).mean()) / \
                            df["price_range"].rolling(w1h).std().clip(lower=1e-10)

    df["cum_imbalance_5m"] = df["vol_imbalance"].rolling(5).sum()
    df["cum_imbalance_15m"] = df["vol_imbalance"].rolling(15).sum()
    df["cum_imbalance_1h"] = df["vol_imbalance"].rolling(bph).sum()

    df["vwap_zscore_1h"] = (df["close_vs_vwap"] - df["close_vs_vwap"].rolling(w1h).mean()) / \
                           df["close_vs_vwap"].rolling(w1h).std().clip(lower=1e-10)

    log_hl = np.log(df["high"] / df["low"].clip(lower=1e-10))
    df["parkvol_1h"] = (log_hl**2).rolling(bph).mean().apply(np.sqrt) / (4 * np.log(2))**0.5

    for h, label in [(5, "5m"), (15, "15m"), (bph, "1h")]:
        net_move = (df["close"] - df["close"].shift(h)).abs()
        sum_moves = df["returns"].abs().rolling(h).sum() * df["close"]
        df[f"efficiency_{label}"] = net_move / sum_moves.clip(lower=1e-10)

    df["kyle_lambda_5m"] = df["kyle_lambda"].rolling(5).mean()
    df["kyle_lambda_15m"] = df["kyle_lambda"].rolling(15).mean()
    df["large_imb_5m"] = df["large_imbalance"].rolling(5).mean()
    df["large_imb_15m"] = df["large_imbalance"].rolling(15).mean()

    return df


def add_targets(df):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    ret = df["returns"].values
    n = len(df)

    for p in [1, 3, 5, 10, 15, 30, 60]:
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n-p] = c[p:] / c[:n-p] - 1.0
        df[f"tgt_ret_{p}"] = fwd_ret
        df[f"tgt_ret_mag_{p}"] = np.abs(fwd_ret)

    for p in [5, 15, 30, 60]:
        fwd_vol = np.full(n, np.nan)
        for i in range(n - p):
            fwd_vol[i] = ret[i+1:i+1+p].std()
        df[f"tgt_vol_{p}"] = fwd_vol

    for p in [3, 5, 10, 15]:
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n-p] = c[p:] / c[:n-p] - 1.0
        df[f"tgt_profitable_long_{p}"] = (fwd_ret > FEE_FRAC).astype(float)
        df.loc[np.isnan(fwd_ret), f"tgt_profitable_long_{p}"] = np.nan
        df[f"tgt_profitable_short_{p}"] = (fwd_ret < -FEE_FRAC).astype(float)
        df.loc[np.isnan(fwd_ret), f"tgt_profitable_short_{p}"] = np.nan

    for p in [3, 5, 10]:
        bu = np.full(n, np.nan)
        bd = np.full(n, np.nan)
        for i in range(n - p):
            bu[i] = 1.0 if (h[i+1:i+1+p] > h[i]).any() else 0.0
            bd[i] = 1.0 if (l[i+1:i+1+p] < l[i]).any() else 0.0
        df[f"tgt_breakout_up_{p}"] = bu
        df[f"tgt_breakout_down_{p}"] = bd

    for p in [5, 15, 30]:
        fwd_range = np.full(n, np.nan)
        for i in range(n - p):
            fwd_h = h[i+1:i+1+p].max()
            fwd_l = l[i+1:i+1+p].min()
            fwd_range[i] = (fwd_h - fwd_l) / c[i] * 10000
        df[f"tgt_range_bps_{p}"] = fwd_range

    vol = df["rvol_1h"].values
    for p in [5, 15, 30]:
        fwd_vol = np.full(n, np.nan)
        for i in range(n - p):
            fwd_vol[i] = ret[i+1:i+1+p].std()
        df[f"tgt_vol_expansion_{p}"] = (fwd_vol > vol).astype(float)
        df.loc[np.isnan(fwd_vol) | np.isnan(vol), f"tgt_vol_expansion_{p}"] = np.nan

    ma_1h = df["close"].rolling(60).mean().values
    for p in [5, 15, 30]:
        revert = np.full(n, np.nan)
        for i in range(n - p):
            if np.isnan(ma_1h[i]) or c[i] == 0:
                continue
            dist_now = abs(c[i] - ma_1h[i]) / c[i]
            min_dist = min(abs(c[j] - ma_1h[i]) / c[i] for j in range(i+1, i+1+p))
            revert[i] = 1.0 if min_dist < dist_now * 0.5 else 0.0
        df[f"tgt_mean_revert_{p}"] = revert

    return df


# ============================================================
# ANALYSIS 1: Feature importance per target
# ============================================================
def analyze_feature_importance(df, target, feat_cols):
    """Train model, return top features and their importance."""
    y = df[target].values
    valid = np.isfinite(y)

    unique_vals = np.unique(y[valid])
    is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0})

    # Feature selection
    corrs = []
    for f in feat_cols:
        x = df[f].values
        mask = valid & np.isfinite(x)
        if mask.sum() < 500:
            corrs.append(0.0)
            continue
        try:
            c_val, _ = spearmanr(x[mask], y[mask])
            corrs.append(abs(c_val) if np.isfinite(c_val) else 0.0)
        except:
            corrs.append(0.0)

    top_idx = np.argsort(corrs)[::-1][:30]
    selected = [feat_cols[j] for j in top_idx if corrs[j] > 0.005]
    if len(selected) < 5:
        return None, None

    X = df[selected].values[valid]
    y_clean = y[valid]
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    if is_binary:
        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", verbosity=-1,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=20, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        )
        model.fit(X, y_clean.astype(int))
    else:
        model = lgb.LGBMRegressor(
            objective="regression", metric="rmse", verbosity=-1,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=20, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        )
        model.fit(X, y_clean)

    imp = model.feature_importances_
    imp_pairs = sorted(zip(selected, imp), key=lambda x: -x[1])
    return imp_pairs, selected


# ============================================================
# ANALYSIS 2: Walk-forward validation (6 monthly folds)
# ============================================================
def walk_forward_validation(df, target, feat_cols, n_folds=6):
    """Run expanding-window WFO with monthly trade periods."""
    y = df[target].values
    unique_vals = np.unique(y[np.isfinite(y)])
    is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0})

    n = len(df)
    fold_size = n // (n_folds + 1)  # reserve 1 fold for initial training
    min_train = fold_size  # at least 1 month of training

    fold_scores = []
    fold_details = []

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        df_train = df.iloc[:train_end]
        df_test = df.iloc[test_start:test_end]

        y_tr = df_train[target].values
        y_te = df_test[target].values
        valid_tr = np.isfinite(y_tr)
        valid_te = np.isfinite(y_te)

        if valid_tr.sum() < 1000 or valid_te.sum() < 500:
            continue

        # Select features on training data
        corrs = []
        for f in feat_cols:
            x = df_train[f].values
            mask = valid_tr & np.isfinite(x)
            if mask.sum() < 500:
                corrs.append(0.0)
                continue
            try:
                c_val, _ = spearmanr(x[mask], y_tr[mask])
                corrs.append(abs(c_val) if np.isfinite(c_val) else 0.0)
            except:
                corrs.append(0.0)

        top_idx = np.argsort(corrs)[::-1][:30]
        selected = [feat_cols[j] for j in top_idx if corrs[j] > 0.005]
        if len(selected) < 5:
            continue

        X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
        y_tr_c = y_tr[valid_tr]
        y_te_c = y_te[valid_te]

        try:
            if is_binary:
                model = lgb.LGBMClassifier(
                    objective="binary", metric="auc", verbosity=-1,
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    num_leaves=15, min_child_samples=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                )
                model.fit(X_tr, y_tr_c.astype(int))
                pred = model.predict_proba(X_te)[:, 1]
                score = roc_auc_score(y_te_c.astype(int), pred) - 0.5

                # Precision at top/bottom decile
                top_10 = pred >= np.percentile(pred, 90)
                bot_10 = pred <= np.percentile(pred, 10)
                prec_top = y_te_c[top_10].mean() if top_10.sum() > 10 else np.nan
                prec_bot = 1 - y_te_c[bot_10].mean() if bot_10.sum() > 10 else np.nan

                fold_details.append({
                    "fold": fold + 1,
                    "train_size": len(X_tr),
                    "test_size": len(X_te),
                    "score": score,
                    "prec_top10": prec_top,
                    "prec_bot10": prec_bot,
                    "base_rate": y_te_c.mean(),
                })
            else:
                model = lgb.LGBMRegressor(
                    objective="regression", metric="rmse", verbosity=-1,
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    num_leaves=15, min_child_samples=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                )
                model.fit(X_tr, y_tr_c)
                pred = model.predict(X_te)
                score_val, _ = spearmanr(pred, y_te_c)
                score = score_val if np.isfinite(score_val) else 0.0

                # Top/bottom decile actual values
                top_10 = pred >= np.percentile(pred, 90)
                bot_10 = pred <= np.percentile(pred, 10)
                mean_top = y_te_c[top_10].mean() if top_10.sum() > 10 else np.nan
                mean_bot = y_te_c[bot_10].mean() if bot_10.sum() > 10 else np.nan

                fold_details.append({
                    "fold": fold + 1,
                    "train_size": len(X_tr),
                    "test_size": len(X_te),
                    "score": score,
                    "mean_top10": mean_top,
                    "mean_bot10": mean_bot,
                })

            fold_scores.append(score)
        except Exception as e:
            pass

    return fold_scores, fold_details


# ============================================================
# ANALYSIS 3: Simple P&L test for directional targets
# ============================================================
def pnl_test(df, feat_cols):
    """
    Simple P&L test: train model to predict profitable_long/short_3,
    trade when confident, measure actual returns.
    Uses last 2 months as OOS test.
    """
    n = len(df)
    # Last 2 months (~87k bars) as test
    test_bars = 60 * 24 * 60  # 60 days
    train_end = n - test_bars
    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:]

    print(f"\n  P&L Test: train={len(df_train):,}, test={len(df_test):,} bars")

    # Train models for long and short
    for direction, tgt_name in [("LONG", "tgt_profitable_long_3"),
                                 ("SHORT", "tgt_profitable_short_3")]:
        y_tr = df_train[tgt_name].values
        y_te = df_test[tgt_name].values
        valid_tr = np.isfinite(y_tr)
        valid_te = np.isfinite(y_te)

        # Feature selection
        corrs = []
        for f in feat_cols:
            x = df_train[f].values
            mask = valid_tr & np.isfinite(x)
            if mask.sum() < 500:
                corrs.append(0.0)
                continue
            try:
                c_val, _ = spearmanr(x[mask], y_tr[mask])
                corrs.append(abs(c_val) if np.isfinite(c_val) else 0.0)
            except:
                corrs.append(0.0)

        top_idx = np.argsort(corrs)[::-1][:30]
        selected = [feat_cols[j] for j in top_idx if corrs[j] > 0.005]

        X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
        y_tr_c = y_tr[valid_tr].astype(int)
        y_te_c = y_te[valid_te].astype(int)

        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", verbosity=-1,
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=20, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        )
        model.fit(X_tr, y_tr_c)
        pred = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te_c, pred)

        # Get actual forward returns for test period
        fwd_ret_col = "tgt_ret_3"
        fwd_ret = df_test[fwd_ret_col].values[valid_te]

        print(f"\n  === {direction} (predict {tgt_name}) ===")
        print(f"  AUC: {auc:.4f} (edge: {auc-0.5:+.4f})")
        print(f"  Base rate: {y_te_c.mean():.2%}")

        # Test at different confidence thresholds
        print(f"\n  {'Threshold':>10} {'Trades':>8} {'WinRate':>8} {'AvgRet':>10} {'TotalRet':>10} {'Sharpe':>8}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = pred >= thresh
            if mask.sum() < 20:
                continue

            # Actual returns when we trade
            if direction == "LONG":
                trade_rets = fwd_ret[mask] - FEE_FRAC
            else:
                trade_rets = -fwd_ret[mask] - FEE_FRAC

            trade_rets = trade_rets[np.isfinite(trade_rets)]
            if len(trade_rets) < 10:
                continue

            n_trades = len(trade_rets)
            win_rate = (trade_rets > 0).mean()
            avg_ret = trade_rets.mean()
            total_ret = trade_rets.sum()
            sharpe = avg_ret / trade_rets.std() * np.sqrt(252 * 24 * 60 / 3) if trade_rets.std() > 0 else 0

            print(f"  {thresh:>10.2f} {n_trades:>8} {win_rate:>8.1%} "
                  f"{avg_ret*10000:>+10.2f}bp {total_ret*100:>+10.2f}% {sharpe:>8.2f}")

    # Also test mean reversion
    print(f"\n  === MEAN REVERSION (predict tgt_mean_revert_5) ===")
    tgt_mr = "tgt_mean_revert_5"
    y_tr = df_train[tgt_mr].values
    y_te = df_test[tgt_mr].values
    valid_tr = np.isfinite(y_tr)
    valid_te = np.isfinite(y_te)

    corrs = []
    for f in feat_cols:
        x = df_train[f].values
        mask = valid_tr & np.isfinite(x)
        if mask.sum() < 500:
            corrs.append(0.0)
            continue
        try:
            c_val, _ = spearmanr(x[mask], y_tr[mask])
            corrs.append(abs(c_val) if np.isfinite(c_val) else 0.0)
        except:
            corrs.append(0.0)

    top_idx = np.argsort(corrs)[::-1][:30]
    selected = [feat_cols[j] for j in top_idx if corrs[j] > 0.005]

    X_tr = np.nan_to_num(df_train[selected].values[valid_tr], nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(df_test[selected].values[valid_te], nan=0, posinf=0, neginf=0)
    y_tr_c = y_tr[valid_tr].astype(int)
    y_te_c = y_te[valid_te].astype(int)

    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", verbosity=-1,
        n_estimators=200, max_depth=5, learning_rate=0.05,
        num_leaves=20, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42,
    )
    model.fit(X_tr, y_tr_c)
    pred = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te_c, pred)

    print(f"  AUC: {auc:.4f} (edge: {auc-0.5:+.4f})")
    print(f"  Base rate: {y_te_c.mean():.2%}")

    # For mean reversion: when model says "revert likely", we trade contrarian
    # If price > MA, go short (expect revert down). If price < MA, go long.
    ma_1h = df_test["close"].rolling(60).mean().values
    close_test = df_test["close"].values[valid_te]
    fwd_ret = df_test["tgt_ret_5"].values[valid_te]

    # Price position relative to MA
    above_ma = close_test > ma_1h[valid_te]

    print(f"\n  {'Threshold':>10} {'Trades':>8} {'WinRate':>8} {'AvgRet':>10} {'TotalRet':>10} {'Sharpe':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = pred >= thresh
        if mask.sum() < 20:
            continue

        # Contrarian: short when above MA, long when below
        trade_rets = np.where(above_ma[mask], -fwd_ret[mask], fwd_ret[mask]) - FEE_FRAC
        trade_rets = trade_rets[np.isfinite(trade_rets)]
        if len(trade_rets) < 10:
            continue

        n_trades = len(trade_rets)
        win_rate = (trade_rets > 0).mean()
        avg_ret = trade_rets.mean()
        total_ret = trade_rets.sum()
        sharpe = avg_ret / trade_rets.std() * np.sqrt(252 * 24 * 60 / 5) if trade_rets.std() > 0 else 0

        print(f"  {thresh:>10.2f} {n_trades:>8} {win_rate:>8.1%} "
              f"{avg_ret*10000:>+10.2f}bp {total_ret*100:>+10.2f}% {sharpe:>8.2f}")


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  1-MINUTE DEEP DIVE — Feature Importance, WFO, P&L")
    print(f"  {SYMBOL}, {START_DATE} to {END_DATE}")
    print("=" * 80)

    df = load_1m_bars()
    print("\n  Adding features...", flush=True)
    df = add_derived_features(df)
    print("  Adding targets...", flush=True)
    df = add_targets(df)
    df = df.iloc[240:].copy()
    print(f"  Ready: {len(df):,} bars, {len(df.columns)} columns\n")

    feat_cols = [c for c in df.columns
                 if not c.startswith("tgt_")
                 and c not in ("open", "high", "low", "close", "volume",
                               "timestamp_us", "returns")]

    # ============================================================
    # PART 1: Feature importance for each target
    # ============================================================
    print("=" * 80)
    print("  PART 1: FEATURE IMPORTANCE")
    print("=" * 80)

    for tgt in TARGETS_OF_INTEREST:
        if tgt not in df.columns:
            continue
        imp_pairs, selected = analyze_feature_importance(df, tgt, feat_cols)
        if imp_pairs is None:
            continue

        print(f"\n  {tgt}:")
        for fname, imp in imp_pairs[:8]:
            print(f"    {imp:>6.0f}  {fname}")

    # ============================================================
    # PART 2: Walk-forward validation
    # ============================================================
    print(f"\n{'=' * 80}")
    print("  PART 2: WALK-FORWARD VALIDATION (6 folds)")
    print("=" * 80)

    print(f"\n  {'Target':<35} {'Folds':>6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Stable?':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for tgt in TARGETS_OF_INTEREST:
        if tgt not in df.columns:
            continue
        scores, details = walk_forward_validation(df, tgt, feat_cols)
        if not scores:
            continue

        mean_s = np.mean(scores)
        std_s = np.std(scores)
        min_s = min(scores)
        max_s = max(scores)
        stable = "YES" if min_s > 0 and std_s < mean_s else "no"

        print(f"  {tgt:<35} {len(scores):>6} {mean_s:>+8.4f} {std_s:>8.4f} "
              f"{min_s:>+8.4f} {max_s:>+8.4f} {stable:>8}")

    # Print fold details for key targets
    for tgt in ["tgt_profitable_long_3", "tgt_mean_revert_5", "tgt_ret_mag_3"]:
        if tgt not in df.columns:
            continue
        scores, details = walk_forward_validation(df, tgt, feat_cols)
        if not details:
            continue

        print(f"\n  Fold details for {tgt}:")
        for d in details:
            if "prec_top10" in d:
                print(f"    Fold {d['fold']}: score={d['score']:+.4f}, "
                      f"prec_top10={d['prec_top10']:.2%}, "
                      f"prec_bot10={d['prec_bot10']:.2%}, "
                      f"base={d['base_rate']:.2%}")
            else:
                print(f"    Fold {d['fold']}: score={d['score']:+.4f}, "
                      f"top10_mean={d['mean_top10']:.6f}, "
                      f"bot10_mean={d['mean_bot10']:.6f}")

    # ============================================================
    # PART 3: P&L Test
    # ============================================================
    print(f"\n{'=' * 80}")
    print("  PART 3: P&L TEST (last 2 months OOS)")
    print("=" * 80)

    pnl_test(df, feat_cols)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  Total time: {elapsed:.0f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
